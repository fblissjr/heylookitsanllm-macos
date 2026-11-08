# Engine Contract, Provenance, Sampler Cascade & Template Analysis

Covers `Sources/HeylookCore/Domain/` and `Sources/HeylookCore/Config/HeylookConfig.swift`.

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

## 4. 4-Rung Chat Template Ladder (`HeylookConfig.swift`)

When resolving a model's chat template (for `llama-server --chat-template-file`, `MLXSwiftProvider`, or `POST /v1/conversations/{id}/prompt`), `HeylookConfig.resolveGGUFTemplate` evaluates 4 rungs in order:

1. **Rung 1 (`chat_template_path` in `model.heylook.toml`)**: Explicit path in sidecar (`.configured`).
2. **Rung 2 (`chat_template.heylook.jinja`)**: Operator override file written by `PUT /v1/admin/models/{id}/chat-template` or the Sidecar Editor sheet (`.configured`). Deleting the override via `DELETE /v1/admin/models/{id}/chat-template` cleanly reverts to Rung 3 or 4 without mutating vendor files.
3. **Rung 3 (`chat_template.jinja`)**: Vendor-shipped Jinja file in the model directory or parent repo folder (`.derived`).
4. **Rung 4 (`tokenizer_config.json` / `gguf_embedded`)**: `chat_template` string embedded in `tokenizer_config.json` or `tokenizer.chat_template` in the `.gguf` binary header (`.derived`).
