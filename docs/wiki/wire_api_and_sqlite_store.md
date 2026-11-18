# 5. HTTP/SSE Wire API, Disconnect Resilience & SQLite WAL

`HeylookServer` (`Sources/HeylookServer/`) and `HeylookCore`'s `Store/` and `Wire/` layers share one orchestration contract across local GUI and HTTP/SSE clients: **`WorkbenchBackend`** (`WorkbenchService`), **`ConversationOrchestrator`**, **`PatternRouter`**, **`HTTPConnectionHandler`**, **`SSEConnectionStreamer` / `SSEWirePipeline`**, and **`SQLiteStore` (`WAL`)**.

---

## 5.1 Canonical Saga: `WorkbenchService` & `ConversationOrchestrator`

- **`WorkbenchBackend` & `WorkbenchService`** (`Sources/HeylookServer/Orchestration/`): Single actor-isolated service unifying model listing, load/reload/unload/fit/image-plan, conversation/notebook/preset CRUD, voice-to-turn transcription, embeddings, operational settings, and performance profile queries for both HTTP handlers and `HeylookUI`.
- **`ConversationOrchestrator`** (`ConversationOrchestrator.swift`): Owns stateful conversation generation (`POST /v1/conversations/:id/generate` & `/voice-generate`) and live prompt preview (`POST /v1/conversations/:id/prompt`):
  1. **Existence & Lock Guards**: Validates conversation existence (`404` across all modes including `append`) and acquires a per-conversation lock (`RequestRegistry.tryAcquireConversationLock`). Any concurrent `generate`, `voice-generate`, conversation `PUT`/`PATCH`/`DELETE`, message `POST`/`PUT`/`PATCH`/`DELETE`, or `/prompt` call returns HTTP **`409 generation_in_progress`**.
  2. **Detachable `SubscriberHub`**: Engine streaming feeds an in-memory broadcaster. If the HTTP socket drops mid-stream, generation continues cleanly on Apple Silicon until completion or explicit `DELETE /v1/requests/:id`, persisting the result and emitting `heylook_saved` on active streams.
  3. **Atomic Non-Destructive Tail Commits**: History is **never** truncated prior to generation. Only once non-empty assistant output (or an aborted turn with partial output) settles does `SQLiteStore` execute a single transaction:
     - **`replaceTailWithMessage`**: Atomically deletes `position >= fromPosition`, inserts the assistant message, and writes `message_stats` (`append` / `regenerate` / edit-and-regenerate).
     - **`replaceTailWithUpdate`**: Atomically deletes `position > anchor.position` and merges `continue` output onto `anchorMessageID`, preserving the anchor's original `model_id` and existing `thinking` (when `updatedThinking == nil`).
  4. **Full `user` & `assistant` `continue` + Open-Thinking Resume**: Supports continuing either a partial assistant turn (including `startsInThinkingOverride: true` when trailing thinking lacks a closing tag) or an existing user message, stripping stale special tokens from replayed turns and reporting omitted historical media in `dropped_media`.

---

## 5.2 Transport, Routing & Wire Validation

- **`PatternRouter`** (`Sources/HeylookServer/Routing/PatternRouter.swift`): Supports parameter captures (`:id`) and greedy wildcard routes (`*id`, `*rest`) so unencoded multi-segment model IDs (`org/model`) and `%2F`-encoded IDs resolve identically.
- **`HTTPConnectionHandler`** (`Sources/HeylookServer/Transport/HTTPConnectionHandler.swift`): True HTTP/1.1 Keep-Alive connection reuse loop supporting both `Content-Length` and `Transfer-Encoding: chunked` request framing plus cooperative peer-drop `StreamCancellationSignal` propagation.
- **Security & Request Hygiene (`HeylookHTTPServer` / `ErrorMapping`)**:
  - **Host DNS-Rebinding Guard**: Rejects disallowed `Host` headers against configured `allowedHosts` / bound listener host.
  - **Admin Token Guard**: Constant-time verification of `X-Heylook-Admin-Token` returning **`401`** on `/v1/admin/*`, `/v1/cache/clear`, and `/v1/data/clear` when configured.
  - **`X-Request-ID` Validation**: Invalid client request IDs are silently replaced with server-generated IDs (`RequestRegistry.isValidClientRequestID`).
  - **Backpressure & Rate Headers**: Saturated `GenerationGate` maps to **`503 model_overloaded`** with `Retry-After: 1` and `X-RateLimit-*` telemetry headers.
  - **Strict Validation (`400` / `422`)**: Missing `model` on `/v1/messages` returns `400`; `StrictSamplerParams` (`WireDTOs.swift`) and retired request keys (`logprobs`, `top_logprobs`, `show_special_tokens: true`, `enable_thinking`, `max_new_tokens`, `system_prompt`, `chat_template_kwargs`, `tools`, `tool_choice`, `preset`, `sampler`, `vision_tokens`) fail loudly with HTTP **`422`**.
- **`SSEConnectionStreamer` & `SSEWirePipeline`** (`Sources/HeylookServer/Transport/SSEStreamWriter.swift`):
  - Writes chunked SSE frames directly with **`5.0s` `event: ping`** (`{"type":"ping"}`) keepalives during long prefill and decode pauses.
  - Emits `message_start`, `heylook_progress` (`PrefillProgressBridge`), `content_block_start`, `content_block_delta` (`thinking_delta` carrying **both** `thinking` and `text` keys; `text_delta`), `content_block_stop`, `message_delta`, `message_stop` (embedding `MessagesResponse.PerformanceExtension` with `nil` for unmeasured TPS fields and `generationDurationMs` deducting `queueWaitMs`), and terminal `heylook_saved`.

---

## 5.3 HTTP Route Surface

| Route Group | Endpoints | Source File |
| :--- | :--- | :--- |
| **Stateless Inference & Voice** | `POST /v1/messages`, `POST /v1/audio/transcriptions`, `POST /v1/conversations/:id/voice-generate` | `Routes/MessagesAndAudioRoutes.swift` |
| **Conversations & Media** | `GET/POST /v1/conversations`, `GET/PUT/PATCH/DELETE /v1/conversations/:id`, `POST /clone`, `POST /prompt`, `POST /generate`, `GET/POST /messages`, `PUT/PATCH/DELETE /messages/:msg_id`, `DELETE /v1/conversations/:id/messages?after=`, `GET /v1/conversations/:id/media/:media_id` | `Routes/ConversationsRoutes.swift` |
| **Notebooks & Presets** | `GET/POST /v1/notebooks`, `GET/PUT/PATCH/DELETE /v1/notebooks/:id`, `GET/POST /v1/presets`, `GET/PUT/PATCH/DELETE /v1/presets/:id` | `Routes/NotebooksAndPresetsRoutes.swift` |
| **Public Models & Admin** | `GET /v1/models`, `POST /v1/models/*id/image-plan`, `POST /v1/models/*id/load`, `GET /v1/admin/model-options`, `GET/POST/PATCH/DELETE /v1/admin/models*`, `/scan-config`, `/validate`, `/*id/status`, `/*id/reload`, `/*id/unload`, `/*id/fit`, `/*id/chat-template`, `GET/PUT /v1/admin/config`, `GET /v1/admin/config/logs`, `DELETE /v1/admin/config/:key`, `POST /v1/admin/reload` | `Routes/ModelsAndAdminRoutes.swift` |
| **Embeddings** | `POST /v1/embeddings` | `Routes/EmbeddingsRoutes.swift` |
| **Monitoring & Controls** | `GET /v1/system/metrics`, `GET /v1/performance/profile/:range`, `GET /v1/capabilities`, `POST /v1/cache/clear`, `POST /v1/data/clear`, `POST /v1/telemetry/events`, `DELETE /v1/requests/:id`, `POST /v1/requests/:id/reasoning-end` | `Routes/MonitoringAndRequestsRoutes.swift` |
| **Optional Static Assets** | `GET/HEAD /`, `/index.html`, `/icon.svg`, `/favicon.ico`, `/manifest.json`, `/apple-touch-icon.png`, `/js/*rest`, `/css/*rest` (only active when `HEYLOOK_FRONTEND_DIR` exists; zero SPA fallback) | `Routes/StaticFrontendRoutes.swift` |

---

## 5.4 SQLite `WAL` Store & Disk Observability (`SQLiteStore`, `ObservabilityDiskLog`)

- **`SQLiteStore`** (`Sources/HeylookCore/Store/SQLiteStore.swift`):
  - **Tables**: `conversations`, `messages`, `media_blobs` (composite primary key `(conversation_id, media_id)`), `message_stats`, `notebooks`, `presets` (`COLLATE NOCASE` unique `name`, **unseeded** out of the box). Emits `mutationStream()` notifications to refresh SwiftUI surfaces reactively without polling.
  - **Conversation-Scoped Media Blobs**: Images and audio embedded in messages are deduplicated by 64-hex SHA-256 `(conversation_id, media_id)`, served at `GET /v1/conversations/:id/media/:media_id` with sanitized `Content-Type`, `Cache-Control: private, max-age=31536000, immutable`, and `X-Content-Type-Options: nosniff`, externalized on `type: "url"` blocks, copied atomically across `cloneConversationWithMessages`, and garbage-collected per conversation (`gcOrphanMediaBlobsUnlocked`) after truncations/deletes.
  - **Non-Compacting Deletes**: `deleteMessage` removes only the targeted message row without renumbering higher `position` values (`gaps are valid`). `DELETE /v1/conversations/:id/messages?after=<pos>` deletes strictly `position > after` (missing/invalid `after` returns `422`).
  - **Exact DTO Keys (`WireDTOs.swift`)**: `NotebookDetailDTO` uses `"content"` (matching the wire spec), `PresetDTO` uses `"params"`, and `ConversationSummaryDTO` includes live `"generating"`.
- **`ObservabilityDiskLog`** (`Sources/HeylookCore/Store/ObservabilityDiskLog.swift`):
  - Controlled by `[settings]` in `heylook.toml` (`observability_level`: `off | minimal | standard | debug`, `observability_retention_days`, `mlx_cache_limit_gb`).
  - Writes structured JSONL to `events.jsonl` and `metrics.jsonl` with 50 MB size rotation, hourly retention sweeps, and newest-first `readTail(limit:)` (`1...500`) served by `GET /v1/admin/config/logs`.
