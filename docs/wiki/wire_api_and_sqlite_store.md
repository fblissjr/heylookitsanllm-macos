# HTTP/SSE Wire API, Disconnect-Resilient Generation & SQLite Store

Covers `Sources/HeylookCore/Wire/MessagesWire.swift`, `Sources/HeylookCore/Store/SQLiteStore.swift`, and `Sources/HeylookServer/HeylookHTTPServer.swift`.

---

## 1. Authoritative `/v1/*` Route Table (`HeylookRouter`)

All routes enforce loopback/Host-header validation (`allowedHosts`) to prevent DNS-rebinding attacks when bound to `127.0.0.1`.

| Method & Route | Purpose & Wire Contract |
| :--- | :--- |
| `POST /v1/messages` | Anthropic Messages-compatible chat & VLM generation (`stream: true` incremental SSE or `stream: false` JSON). Resolves the 5-layer `SamplerCascade`, enforces HTTP `400` validation on `thinking_depth` / `reasoning_effort`, and enforces HTTP `422` capability gating when image/audio blocks are sent to a model where `supportsVision` / `supportsAudio` is `false`. |
| `POST /v1/conversations/{id}/generate` | **Disconnect-resilient server-owned generation**: appends the user turn to `SQLiteStore` immediately, runs generation in a server-owned `Task` that survives client SSE socket drops (e.g., iOS lock screen), persists the completed or cancelled assistant turn to `SQLiteStore`, and emits `event: heylook_saved`. |
| `POST /v1/conversations/{id}/voice-generate` | Transcribes 16 kHz mono WAV audio on the ANE lane (`CoreAIProvider.transcribeAudio`), emits `event: heylook_voice_transcribed`, and streams the disconnect-resilient assistant response (`event: heylook_saved`). |
| `POST /v1/conversations/{id}/prompt` | Renders the exact formatted prompt (`renderPromptPreview`) and active `template_source`, `thinking_enabled`, and `thinking_depth` for a conversation without running generation. |
| `POST /v1/models/{id}/image-plan` | Returns HTTP `409 Conflict` when the model is unloaded, or HTTP `200` with `tokens_per_image` derived from `processor_config.json` / `config.json` (`vision_config`) when resident. |
| `POST /v1/embeddings` | Multimodal `EmbeddingGemma 2` endpoint supporting flat string arrays or structured `content` block arrays (`text`, `image`, `audio`, `video` with `windows: {"sampling": "fps" \| "content"}`), `prompt_name` / `task`, and Matryoshka `dimensions` (`768`, `512`, `256`, `128`). Enforces `EmbeddingRequestLimits` with HTTP `422`. |
| `GET /v1/capabilities` | Publishes server version, supported Matryoshka dimensions, and `embedding_limits` (`max_items = 16`, `max_blocks_per_item = 8`, `max_request_media_bytes = 201,326,592`). |
| `GET /v1/models` | Lists all registered models with full `EngineDescription` provenance (`ProvenanceField`s, `reasoningFormat`, `thinkingControls`, residency, and memory footprint). |
| `GET / PATCH /v1/admin/models/{id}` | Inspects or updates a model's `model.heylook.toml` sidecar configuration in place (preserving `reasoningFormat` and `thinkingControls`). |
| `POST /v1/admin/models/{id}/load` & `/unload` | Explicitly activates a model in `ComputeScheduler` or unloads it (`provider.unload()`). |
| `POST /v1/admin/models/unload-all` | Unloads all GPU and ANE models, clears Metal/CoreML caches, and terminates all `llama-server` subprocesses. |
| `GET / PUT / DELETE /v1/admin/models/{id}/chat-template` | Reads, writes, or reverts the Rung-2 operator template override (`chat_template.heylook.jinja`) without mutating vendor `chat_template.jinja`. |
| `POST /v1/requests/{id}/reasoning-end` & `DELETE /v1/requests/{id}` | Mid-stream control endpoints: forces `ReasoningParser` out of thinking mode (`</think>`) on the next token, or aborts generation immediately. |

---

## 2. Unified SSE Framing (`HeylookRouter.consumeStreamAndFormatSSE`)

Both `POST /v1/messages` (`stream: true`) and `POST /v1/conversations/{id}/generate` share a single SSE framing pipeline (`HeylookRouter.consumeStreamAndFormatSSE`) flushed incrementally over `NWConnection` via `SSEConnectionStreamer`:

1. `event: message_start` (with `request_id` and `model`)
2. `event: content_block_delta` (`thinking_delta` for reasoning trace tokens, `text_delta` for visible answer tokens)
3. `event: message_delta` (`stop_reason`, `usage`, `performance.cache`, `performance.speculative`, `performance.generation_tps`)
4. `event: heylook_saved` (for conversation routes after `SQLiteStore` commit)
5. `event: message_stop`

---

## 3. Zero-Dependency `SQLiteStore` (`SQLiteStore.swift`)

- Uses macOS/iOS system `libsqlite3` in `WAL` (`journal_mode=WAL`) + `NORMAL` synchronous mode for lock-free concurrent reads during SSE streaming.
- Stores `conversations`, `messages` (including `thinking`, `stop_reason`, `cache_report_json`, `speculative_report_json`, `generation_tps`, and `compute_placement`), `notebooks`, and `presets`.
- **Save & Continue Branching (`deleteMessages(conversationID:fromPosition:)`)**: Editing any prior message updates that turn at `position` and deletes all subsequent turns (`position + 1...`), pairing with `MLXSwiftProvider`'s KV-cache prefix truncation (`sliceTruncations`).
