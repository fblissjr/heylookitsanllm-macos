# 2. Engine Contract, Provenance, Samplers & Templates

Every configuration field, capability flag, template property, and sampler parameter in `heylookitsanllm-macos` separates **truthful wire introspection** (`EngineContract`), **internal runtime metadata** (`EngineDescription`), **declarative field schemas** (`ProviderConfigSchema`), **5-layer sampler precedence** (`SamplerCascade`), and **zero-heuristic template probing** (`ModelTemplateInfo` + `HeylookJinjaRenderer`).

---

## 2.1 Truthful Wire Contract (`EngineContract`, `Fact<Value>`, `SettingFact`)

Defined in `Sources/HeylookCore/Engine/EngineContract.swift` and `Sources/HeylookCore/Engine/EngineDescription.swift`:

- **Internal Provenance (`EngineDescription` & `ProvenanceField<T>`)**: Tracks runtime identity, capabilities, context sizes, and reasoning metadata across six provenance states (`derived`, `configured`, `observed`, `observed_cached`, `unknown`, `not_applicable`), three effect classes (`reload`, `per_request`, `read_only`), and explicit `ComputePlacement` (`mlx_gpu`, `ane`, `hybrid_ane_gpu`, `cpu_and_gpu`, `cpu_only`, `gguf_subprocess`, `remote`).
- **Truthful Fact Serialization (`Fact<Value>`)**: Every field under `EngineContract` wraps its payload in `Fact<Value>(value:provenance:source:)`. When serializing to JSON, `.unknown` and `.notApplicable` **force `"value": null`** rather than leaking stale estimates or zeros. Both `value` (when `String`) and `source` pass through `EngineSanitization.publicValue` / `EngineSanitization.publicPathString` so local filesystem paths are never leaked to public API clients.
- **`SettingFact`**: Each setting in `EngineContract.settings` encodes `{value, configured, auto, reason, provenance, effect}` across six wire `ConfigFieldEffect` values (`per_request`, `applies_live`, `descriptive`, `requires_reload`, `load_time_only`, `identity`).
- **`EngineContract` Sub-Blocks**:
  - `context`: `length` (`Fact<Int>`) vs `running` (`Fact<Int>` — `not_applicable` on dynamic MLX/CoreAI KV growth; `observed` from `llama-server /props` on loaded GGUF).
  - `template`: `EngineTemplateFacts` (`origin`, sanitized `path`, `sha256`, `running_sha256`, `prefix_stable`, `caps`).
  - `thinking`: `EngineThinkingFacts` (`supported`, `source`, `control`, `depth: EngineThinkingDepthFacts?`, `budget: EngineThinkingBudgetFacts?`, `default_enabled`, `format`).
  - `speculative`: `EngineSpeculativeFacts` (`supported`, `in_force`, `draft_model`, `spec_type`).
  - `decoding`: `EngineDecodingFacts` (`method`, `reason`, `draft_model`, `mtp_tokens`).
  - `embeddings`: `EngineEmbeddingFacts?` (`modalities`, `dimensions`, `native_dimensions`, `max_input_tokens`, `prompts`, `normalization`, `video_audio`, `video_windows`, `limits`, `provenance`, `load_profile`, `compute_placement`).
- **Public vs Admin Model Rows**:
  - `PublicModelWireRow` (`GET /v1/models`, `GET /v1/models/*id`) strictly **omits `modelPath`** (`EngineSanitization`) and surfaces `id`, `object`, `owned_by`, `provider`, `modalities`, `capabilities`, `thinking_default`, `sampler_defaults`, `sampler_sources`, and `engine: EngineContract`.
  - `AdminModelWireRow` (`GET /v1/admin/models`) additionally surfaces `model_path`, `config` (explicit keys only via `ModelSidecarConfig.toExplicitJSONDictionary()`), `source` (`.config` vs `.discovered`), `stale_reload_fields`, and the live `EngineContract`.

---

## 2.2 Declarative Schema & Comment-Preserving TOML (`ProviderConfigSchema`, `TOMLCodec`)

- **`ProviderConfigSchema`** (`Sources/HeylookCore/Config/ProviderConfigSchema.swift`) is the single canonical source of truth for per-provider configuration across `mlx`, `gguf`, and `coreai`. It directly derives all **13** sampler descriptors from `SamplerCascade.SamplerFieldKey.allCases` without duplicating key lists.
  - Each `ConfigFieldDescriptor` specifies `type`, bounds (`minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`), `enum`, `effect`, CLI `arg` (for `gguf`), `ui` (`normal | advanced | hidden`), `shape`, `reason`, `file_only`, and `load_setting`.
  - Drives four consumers uniformly:
    1. `modelOptionsJSON(...)` for `GET /v1/admin/model-options`.
    2. `validatePatch(...)` for `PATCH /v1/admin/models/*id`, rejecting unknown keys, invalid bounds/types, and `isFileOnly` fields (e.g. `server_binary`, `model_path`) with HTTP **`422`**.
    3. `parseReloadSettings(...)` for `POST /v1/admin/models/*id/reload`, rejecting non-`load_setting` keys with HTTP **`400`**.
    4. `staleReloadFields(...)` and `buildEngineSettings(...)` comparing the snapshot captured at load time against the current on-disk/saved config (plus template staleness).
- **`TOMLCodec`** (`Sources/HeylookCore/Config/TOMLCodec.swift`) provides a recursive-descent parser and comment-preserving mutator for `heylook.toml` (`updateScanConfig`, `updateSettings`) and per-model `model.heylook.toml` (`updateModelSidecar`, `applyModelSidecarPatch`). When all explicit overrides are unset (`null` / removed), `updateModelSidecar` returns `nil` so an empty sidecar file is cleanly deleted from disk rather than leaving stale overrides behind.

---

## 2.3 Five-Layer `SamplerCascade`

Defined in `Sources/HeylookCore/Sampling/SamplerCascade.swift`:

1. **Layer 1 — `builtin`**:
   - `fallbackTemperature = 1.0`, `fallbackTopP = 0.95`, `defaultMaxTokens = 16384`, `defaultRepetitionContextSize = 64`.
   - Inactive optional knobs default to engine off-sentinels (`top_k = 0`, `min_p = 0.0`, `repetition_penalty = 1.0`, `presence_penalty = 0.0`).
2. **Layer 2 — `vendor`**: Reads `generation_config.json` from the checkpoint directory, importing vendor baseline keys (`VENDOR_SAMPLING_KEYS = ["temperature", "top_p", "top_k"]`).
3. **Layer 3 — `model.heylook.toml`**: Model sidecar or `[[models]]` explicit overrides (skipping any keys listed in `unset`).
4. **Layer 4 — `preset`**: Active preset snapshot stamped onto a conversation or notebook.
5. **Layer 5 — `request`**: Explicit per-turn overrides in `/v1/messages` or `/v1/conversations/*/generate`.

`SamplerFieldKey` (`allCases`, 13 canonical keys) ensures identical resolution across `HeylookCore`, `ProviderConfigSchema`, `PresetsSurface`, `DocumentSettingsController`, and `MessagesWire` (`StrictSamplerParams` validates API sampler payloads with `422` on unknown keys or out-of-range values). Note `.thinking` maps to wire `"enable_thinking"` in `ProviderConfigSchema` and sidecar explicit JSON exports while accepting `/v1/messages` `thinking: Bool | {type, budget_tokens}`.

---

## 2.4 Render-Probe `ThinkingControls`, Template Ladder & 4-Protocol `ReasoningParser`

### Zero-Heuristic Thinking Probe (`ModelTemplateInfo.swift` & `HeylookJinjaRenderer.swift`)

Instead of switching on model ID substrings, `ModelTemplateInfo.detectThinkingControls` runs real AST + template renders against `HeylookJinjaRenderer`:
- Probes variable references (`enable_thinking`, `thinking`, `reasoning_effort`, `reasoning_strength`, etc.) and evaluates outputs with candidate toggles and a sentinel string `garbageDepthSentinel = "zq-not-a-level"`.
- Classifies depth-word semantics from observed rendered text:
  - `unknown`: `"ignored"` | `"verbatim"` | `"fallback"` | `"raises"` (e.g., whether an unsupported `thinking_depth` word is ignored, echoed verbatim, triggers fallback, or raises a Jinja error), plus `changes_prefix`, `off`, `aliases`, and `budgetEnforced`.
- `HeylookJinjaRenderer.checkPrefixStability(template:)` detects non-prefix-stable templates caused by `strftime_now`/`now()` or turn-index logic (such as templates that strip `reasoning_content` from older assistant turns prior to `last_query_index`).

### Chat Template Resolution Ladder (`HeylookConfig.inspectChatTemplate`)

Templates are discovered and inspected in strict precedence order:
1. Inline or referenced `chat_template` / `chat_template_path` in `model.heylook.toml`.
2. Operator override file `chat_template.heylook.jinja` in the model directory (validated via `HeylookJinjaRenderer.validateSyntax` before writing).
3. Vendor sidecar `chat_template.jinja` (unless `use_sidecar_chat_template = false`).
4. `tokenizer_config.json` -> `chat_template.json` -> embedded GGUF `tokenizer.chat_template` (`GGUFHeaderReader`).

`inspectChatTemplate` surfaces every copy found (`sources` ladder), Hugging Face download provenance (`modified_since_download`), `inert_reason` when an override is shadowed, truthful `observedStale: Bool?` (`nil` when unloaded), and `refused_shapes`.

### 4-Protocol Streaming `ReasoningParser` (`Sources/HeylookCore/Parsing/ReasoningParser.swift`)

Wait-free character state machines split reasoning vs visible content across four structural reasoning protocols derived by `ModelTemplateInfo`:
1. `.tagPair` (`<think>...</think>`, `<|thought|>...<|/thought|>`, `<thought>...</thought>`, including open-thinking resume when history ends inside an open think block).
2. `.gemmaChannel` (`<|channel>thought\n...<channel|>`).
3. `.harmonyChannel` (`<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>`).
4. `.recipientChannel` (`to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>...<|eot|>`).

`forceEndReasoning()` closes active reasoning cleanly when mid-generation stop/think-end is triggered (`POST /v1/requests/{id}/reasoning-end`), while `StripSpecials` filters trailing special tokens without hardcoding model IDs.
