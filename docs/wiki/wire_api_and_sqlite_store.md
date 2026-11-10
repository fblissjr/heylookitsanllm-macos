# HTTP/SSE Wire API, Disconnect-Resilient Generation & SQLite Store

Covers `Sources/HeylookCore/Wire/MessagesWire.swift`, `Sources/HeylookCore/Store/SQLiteStore.swift`, and `Sources/HeylookServer/HeylookHTTPServer.swift`.

---

## 1. Security, Request ID & Admin Middleware (`HeylookRouter`)

Every request handled by `HeylookRouter` passes through four cross-cutting guards before route dispatch:

1. **Strict DNS-Rebinding `Host` Validation (`isAllowedHostHeader`)**:
   - Enforced on **both** loopback (`127.0.0.1`) and wildcard (`0.0.0.0` / `::`) binds. Loopback/LAN literal IPs and `.local` mDNS peers pass when appropriate, whereas arbitrary DNS hostnames (e.g. `evil.example`) are rejected with HTTP `403` unless explicitly listed in `allowed_hosts`.
2. **Request Correlation (`X-Request-ID`)**:
   - Client-supplied `X-Request-ID` headers are validated against `^[A-Za-z0-9._:-]{1,128}$` (answering HTTP `422` when malformed). When omitted, `HeylookRouter` / `HeylookHTTPServer` generates `req_<hex>` and echoes `X-Request-ID` on both JSON and chunked SSE HTTP responses.
3. **Admin Bearer / Header Auth (`X-Heylook-Admin-Token`)**:
   - When `HEYLOOK_ADMIN_TOKEN` or `heylook.toml` `admin_token` is configured, all `/v1/admin/*` and `/v1/data/clear` routes require `X-Heylook-Admin-Token` (or `Authorization: Bearer ...`), returning HTTP `401` when missing and HTTP `403` when mismatched.
4. **Read-Only Config Lock (`HEYLOOK_READONLY_MODEL_CONFIG=1`)**:
   - Blocks mutating admin config, sidecar, and template override routes with HTTP `409 read_only_config`.

---

## 2. Authoritative `/v1/*` Route Table (`HeylookRouter`)

| Method & Route | Purpose & Wire Contract |
| :--- | :--- |
| `POST /v1/messages` | Anthropic Messages-compatible chat & VLM generation (`stream: true` incremental SSE or `stream: false` JSON). Resolves the 5-layer `SamplerCascade`, enforces `stop_sequences` (`stop_reason = "stop_sequence"`), forwards `response_format` (`json_schema`), validates `thinking_depth` (`400`), and enforces HTTP `422` capability gating on unsupported media blocks. |
| `GET / POST /v1/conversations` | Lists conversations (`updated_ms DESC`) or creates a conversation (`201`). |
| `GET / PATCH / DELETE /v1/conversations/{id}` | Hydrates conversation + ordered messages (with `media_blobs` restored into `content_blocks`), updates `title` / `system_prompt` / `default_model_id` / `params` / `applied_preset_id`, or deletes the conversation and GCs orphan blobs. |
| `POST /v1/conversations/{id}/clone` | Clones a conversation (optionally truncated at `up_to_position` for turn-level branching) into a new independent `ConversationRecord` (`201`). |
| `PATCH / DELETE /v1/conversations/{id}/messages/{msg_id}` | Updates message `content` / `thinking` (synchronizing text inside multimodal `content_blocks` while preserving media attachments; optional `truncate_after: true`), or deletes a single turn and compacts subsequent `position` indices. |
| `POST /v1/conversations/{id}/generate` | **Disconnect-resilient server-owned generation** supporting `mode = "append" \| "regenerate" \| "continue"`. Acquires `RequestRegistry.shared.tryAcquireConversationLock` (answering HTTP `409 generation_in_progress` on contention), strips historical media when switching to a text-only model (`dropped_media`), runs in a server-owned `Task`, records `message_stats`, and emits `event: heylook_saved`. |
| `POST /v1/conversations/{id}/generate/reasoning-end` | Signals mid-stream `reasoning-end` for the active request locked to conversation `{id}`. |
| `POST /v1/conversations/{id}/voice-generate` | Runs ANE audio front-end + speech transcription (`CoreAIProvider.transcribeAudio`, answering truthful `runtime_unavailable` until an ASR tower is linked) and streams server-owned generation (`heylook_voice_transcribed` + `heylook_saved`). |
| `GET / POST /v1/conversations/{id}/prompt` | Renders the exact formatted prompt via `HeylookJinjaRenderer` (`PromptRenderResult`: `prompt`, `source`, `markers`, `continuation`, `dropped_media`, `unrendered_media`). |
| `GET / POST / PATCH / DELETE /v1/notebooks[/{id}]` & `POST /v1/notebooks/{id}/generate` | Full scratchpad CRUD plus streaming continuation generation (`NotebookGenerateRequest`), persisting updated `body`, `params`, and `applied_preset_id`. |
| `GET / POST / PATCH / DELETE /v1/presets[/{id}]` | Preset CRUD with `COLLATE NOCASE` unique-name enforcement (`409` on duplicate name). |
| `GET /v1/models`, `POST /v1/models/reload`, `GET /v1/model-options` | Lists `EngineDescription` provenance, re-scans `heylook.toml` folders (`added`, `updated`, `removed`), and publishes provider/compute/quantization schema options. |
| `POST /v1/models/{id}/image-plan` | Returns HTTP `409` when unloaded, or HTTP `200` with `tokens_per_image` and resized dimensions derived from `processor_config.json` / `config.json` (`image_max_tokens`). |
| `GET / PATCH /v1/admin/models/{id}`, `/load`, `/unload`, `/unload-all`, `/cache/clear`, `/fit`, `POST /v1/admin/models/validate` | Sidecar inspection/editing, explicit load/unload, KV/Metal cache clear without weight unload, `GGUFSpawnConfig.assessRAMFit` memory-headroom check, and path preflight validation. |
| `GET / PUT / DELETE /v1/admin/models/{id}/chat-template` | Reads `ChatTemplateInspection`, validates Jinja2 syntax via `HeylookJinjaRenderer.validateSyntax` (`422` on invalid syntax) before writing `chat_template.heylook.jinja`, or deletes the operator override. |
| `GET / PATCH / DELETE /v1/admin/config[/{key}]` & `GET / PATCH / DELETE /v1/admin/config/logs` | Live `heylook.toml` configuration management and structured `HeylookObservabilityLog` query, `level`/`retention_limit` tuning, and ring clear. |
| `GET /v1/performance/profile/{range}`, `POST /v1/telemetry/events`, `POST /v1/data/clear` | Aggregates `message_stats` across `1h` / `6h` / `24h` / `7d` / `30d` (overall + `by_model`), records structured client/UI telemetry events, or clears user data while preserving presets. |
| `POST /v1/embeddings` & `GET /v1/capabilities` | Multimodal `EmbeddingGemma 2` endpoint (`text`, `image`, `audio`, `video` windows, Matryoshka `768`/`512`/`256`/`128`, `EmbeddingRequestLimits` `422` guard) and server capability/status discovery (`/v1/status`, `/v1/system/metrics`). |
| `POST /v1/requests/{id}/reasoning-end` & `DELETE /v1/requests/{id}` | Mid-stream reasoning budget close and immediate task cancellation (`RequestRegistry.abort`). |

---

## 3. Canonical Anthropic Block SSE Sequence (`consumeStreamAndFormatSSE`)

Both `POST /v1/messages` (`stream: true`) and `POST /v1/conversations/{id}/generate` emit the full Anthropic block lifecycle flushed incrementally over `NWConnection` via `SSEConnectionStreamer`:

1. `event: message_start` (`id`, `model`, `request_id`)
2. `event: content_block_start` (`index`, `thinking` or `text`)
3. `event: content_block_delta` (`thinking_delta` or `text_delta`)
4. `event: content_block_stop` (`index`)
5. `event: message_delta` (`stop_reason`, `stop_sequence`, `usage`, `performance.cache`, `performance.speculative`, `performance.generation_tps`, `dropped_media`)
6. `event: heylook_saved` (for conversation routes after `SQLiteStore` commit)
7. `event: message_stop`

---

## 4. Zero-Dependency `SQLiteStore` & Shared UI/HTTP State (`SQLiteStore.swift`)

- Uses macOS/iOS system `libsqlite3` in `WAL` (`journal_mode=WAL`) + `NORMAL` synchronous mode. Outdated table schemas are detected at startup via `PRAGMA table_info(...)` and cleanly recreated (`No Migration Code` rule), while default presets are seeded once via `store_meta` (`presets_seeded = '1'`).
- **SHA-256 Deduplicated `media_blobs`**: Base64 multimodal attachments in `MessageRecord.contentBlocks` are externalized into `media_blobs(sha256, mime_type, data, byte_size)` as `blob://<sha256>` references on write, hydrated transparently on read, and garbage-collected when messages or conversations are deleted.
- **Zero Split-Brain Between SwiftUI & HTTP**: `HeylookWorkbenchViewModel` and `HeylookRouter` share the same `SQLiteStore` (`store.mutationStream()` refreshes UI state whenever external `/v1/*` clients mutate conversations, messages, notebooks, or presets) and the same `RequestRegistry.shared` conversation lock table (`409 generation_in_progress`).
