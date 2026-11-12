# Engine Contract, Provenance, Sampler Cascade & Template Analysis

Covers `Sources/HeylookCore/Engine/`, `Sources/HeylookCore/Sampling/`, `Sources/HeylookCore/Parsing/`, and `Sources/HeylookCore/Config/`.

---

## 1. `EngineDescription` & `ProvenanceField<T>` (`EngineDescription.swift`)

Every registered or resident model publishes an `EngineDescription` struct. Rather than returning bare scalars that hide whether a setting came from a model checkpoint, an operator sidecar, or a live runtime query, each configurable or observable property is wrapped in `ProvenanceField<T>`:

- **`value`**: The effective value in force (`String`, `Int`, `Bool`, `ComputePlacement`, etc.).
- **`provenance` (`ProvenanceSource`)**:
  - `.derived`: Inferred from checkpoint files on disk (`config.json`, `generation_config.json`, `tokenizer.json`, `chat_template.jinja`, `.gguf` header metadata).
  - `.configured`: Explicitly set by the operator in `heylook.toml`, `model.heylook.toml`, a preset, or a request override.
  - `.observed`: Confirmed at runtime by the active execution engine (e.g., `CoreAIBundleManifest` `metadata.json` or `llama-server` `GET /props`).
  - `.observedCached`: Restored from a previous runtime observation.
- **`effect` (`EffectClass`)**:
  - `.perRequest`: Takes effect immediately on the next generation call (`temperature`, `top_p`, `thinking`, `thinking_depth`, `thinking_budget`).
  - `.reload`: Requires unloading and reloading the model (`ctx_size`, `flash_attn`, `compute_unit`, `chat_template.heylook.jinja`).
  - `.restart`: Requires restarting the server listener (`server_port`, `server_host`).

> **Rule**: `ComputeScheduler.applySidecarUpdate(_:for:)` must preserve `reasoningFormat` and `thinkingControls` (and update `thinkingControls.supportsThinking` and `templateSource` when edited) rather than resetting them to `EngineDescription` defaults.

---

## 2. 5-Layer Deterministic `SamplerCascade` (`SamplerCascade.swift`)

Sampling and thinking parameters resolve through a strict 5-layer cascade (`HeylookConfig.resolveEffectiveSampling`):

$$\text{builtin} \longrightarrow \text{vendor} \longrightarrow \text{model.heylook.toml} \longrightarrow \text{preset} \longrightarrow \text{request}$$

| Layer | Source | Provenance Recorded |
| :--- | :--- | :--- |
| **1. `builtin`** | `SamplerSettings.builtinDefaults` (`temp=0.7`, `top_p=0.95`, `top_k=40`, `min_p=0.05`, `max_tokens=2048`) | `.derived` |
| **2. `vendor`** | `generation_config.json` in MLX/CoreAI model directory or `general.sampling.*` KV metadata in `.gguf` binary header | `.derived` |
| **3. `modelSidecar`** | `model.heylook.toml` in the model directory or `[[models]]` entry in `heylook.toml` | `.configured` |
| **4. `preset`** | Selected `PresetRecord` in `SQLiteStore` | `.configured` |
| **5. `request`** | Per-request fields on `MessagesRequest` (`POST /v1/messages`) or `userOverriddenSamplerKeys` in `HeylookWorkbenchViewModel` | `.configured` (`request` pill in UI) |

---

## 3. Zero-Hardcoding `ModelTemplateInfo` & `ThinkingControls` (`ModelTemplateInfo.swift`)

Following `VISION.md` (*"Derive, don't copy"*), `ModelTemplateInfo.inspect(modelDirectory:)` inspects `chat_template.jinja`, `chat_template.heylook.jinja`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, and `.gguf` `tokenizer.chat_template` without any hardcoded model-name branches:

1. **`ReasoningFormat`**:
   - `.tagPair(open:close:prefillsThinking:)`: `<think>...</think>`, `<|thought|>...<|/thought|>`, `<thought>...</thought>`. Automatically sets `prefillsThinking = true` when the `add_generation_prompt` tail of the template opens `<think>` without closing it.
   - `.gemmaChannel(openPrefix:closeTag:)`: `<|channel>thought\n...<channel|>` (Gemma 4).
   - `.harmonyChannel`: `<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>` (GPT-OSS).
   - `.recipientChannel`: `to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>...<|eot|>` (Muse-Glimmer / Muse-Spark).
   - `.none`: Templates with no reasoning markers (`Llama 3.2`, `Mistral`, etc.).
2. **`ThinkingControls`**:
   - `supportsThinking`: `true` iff the template has a reasoning format or thinking toggle variable.
   - `switchVariable`: e.g., `"enable_thinking"`.
   - `depthVariable` & `depthValues`: Extracted directly from Jinja `in [...]` checks for `"reasoning_effort"`, `"reasoning_strength"`, or `"thinking_mode"` (e.g., `["low", "medium", "high"]` or `["none", "low", "medium", "high", "max"]`).
3. **Dynamic `StripSpecials` & EOS Token Union**:
   - Extracts `"special": true` tokens from `tokenizer.json` (`added_tokens`) and `tokenizer_config.json` (`added_tokens_decoder`), minus `reasoningFormat.structuralDelimiters`, so turn-end tokens (`<turn|>`, `<|eot|>`, `<|im_end|>`) are scrubbed from visible output without eating reasoning channel delimiters before `ReasoningParser` processes them.
   - Collects `eos_token_id` arrays from `config.json` and `generation_config.json` so `MLXSwiftProvider` configures `extraEOSTokens` on `ModelConfiguration`.

---

## 4. 4-Rung Chat Template Ladder, `HeylookJinjaRenderer` & `ChatTemplateInspection` (`HeylookConfig.swift`, `HeylookJinjaRenderer.swift`)

When resolving a model's chat template (for `llama-server --chat-template-file`, `MLXSwiftProvider`, or `GET`/`POST /v1/conversations/{id}/prompt`), `HeylookConfig` evaluates the template ladder in order:

1. **Rung 1 (`chat_template_path` / `chat_template` in `model.heylook.toml`)**: Explicit path or inline Jinja string in sidecar (`.configured`, `origin = "sidecar"`). When `chat_template_path` is set, `ChatTemplateInspection.inertReason` warns that `chat_template.heylook.jinja` is shadowed.
2. **Rung 2 (`chat_template.heylook.jinja`)**: Operator override file written by `PUT /v1/admin/models/{id}/chat-template` or the Template Override sheet (`.configured`, `origin = "operator_override"`). Deleting the override via `DELETE /v1/admin/models/{id}/chat-template` cleanly reverts to Rung 3 or 4 without mutating vendor files.
3. **Rung 3 (`chat_template.jinja`)**: Vendor-shipped Jinja file in the model directory (`.derived`, `origin = "vendor_file"`), skipped if `use_sidecar_chat_template = false` or if `--mmproj` vision/audio is active and the sidecar template lacks media placeholders while the `.gguf` header carries a multimodal template.
4. **Rung 4 (`tokenizer_config.json` / `gguf_embedded`)**: `chat_template` string embedded in `tokenizer_config.json` or `tokenizer.chat_template` in the `.gguf` binary header (`.derived`, `origin = "gguf_header"`).

### Real Jinja2 AST Evaluation & Integrity Checks
- **`HeylookJinjaRenderer`**: Provides a full recursive-descent Jinja2 lexer, AST parser, syntax validator (`validateSyntax`), and evaluator supporting `for`/`else` (`loop.index0`, `loop.first`, `loop.last`), `if`/`elif`/`else`, `set` & `namespace(...)` mutation across loop iterations, `macro` definitions, array slicing (`messages[::-1]`), filters (`tojson`, `trim`, `length`, `join`, `replace`, `selectattr`), `raise_exception(...)`, and `huggingface/swift-jinja` execution when linked under `HEYLOOK_ENABLE_NATIVE_MLX=1`.
- **Prefix-Stability Lint (`checkPrefixStability`)**: Renders a 2-turn conversation and a 3-turn extension and verifies that the 2-turn output (minus generation prompt tail) is an exact byte prefix of the 3-turn render, flagging `strftime_now()` or history-mutating templates (`prefixStable = false`).
- **Post-Download Vendor Modification Detection (`checkModifiedSinceDownload`)**: Computes git-blob `SHA-1` (`blob <len>\0...`) or Git-LFS `SHA-256` against `.cache/huggingface/download/<filename>.metadata` ETags so operators can see when a vendor `chat_template.jinja` or `tokenizer_config.json` was edited on disk (`modifiedSinceDownload == true`).
- **Offline Subword Tokenization (`CheckpointPieceTokenizer`)**: Parses both BPE (`vocab` + merge ranks) and SentencePiece Unigram (`vocab` `[piece, score]` log-probabilities via Viterbi dynamic programming) from `tokenizer.json` for accurate offline prompt/budget token counting.
- **Non-Destructive Config Helpers**: `HeylookConfig.addScanFolder(_:configURL:)`, `removeScanFolder(_:configURL:)`, and `save(to:)` update `heylook.toml` atomically while preserving `[[models]]`, `admin_token`, and `allowed_hosts`.
