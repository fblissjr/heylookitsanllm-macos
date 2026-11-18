# 6. Model Family Onboarding, Quantization & Export Playbooks

`heylookitsanllm-macos` targets Apple Silicon (`macOS 27.2+`, `iOS 27.2+`) using **Swift 6.4 (`v6` language mode)** with remote SwiftPM packages (`mlx-swift`, `mlx-swift-lm`, `swift-transformers`, optional `apple/coreai-models`). New model architectures are onboarded via **checkpoint signatures** and **declarative templates**, never brittle model-ID substring switches.

---

## 6.1 Signature-Based Onboarding (`CheckpointSignature` & `ModelSpecialization`)

When a model checkpoint directory or `.gguf` file is scanned or loaded (`Sources/HeylookEngines/Specialization/ModelSpecialization.swift`), `CheckpointSignature` inspects:
- `config.json` (model type, hidden sizes, sliding-window / linear-attention / hybrid layer configs, vision & preprocessor settings)
- `preprocessor_config.json` (image processor patch multiples & constraints)
- GGUF binary header metadata (`GGUFHeaderReader`, up to 64 MB header slice)
- Paired speculative drafter `config.json` (`gemma4_assistant` `backbone_hidden_size`, vocabulary/hidden alignment)

`ModelSpecializationRegistry` maps the parsed signature to one of the concrete `ModelSpecialization` strategies:

| Family / Pattern | Signature Criteria | Automatic Specialization Behavior |
| :--- | :--- | :--- |
| **Gemma 4 / Multimodal SWA** (`Gemma4Specialization`) | `gemma4` architecture / Gemma 4 vision projector | Aligns images to **`48px`** (`16 * 3`); validates `gemma4_assistant` `backbone_hidden_size == target.hidden_size`; enables Leviathan MTP speculative decoding after multimodal prefill; sets GGUF `--ctx-checkpoints 32 --checkpoint-min-step 256` and suppresses `--cache-reuse`. |
| **Hybrid Recurrent / SWA / SSM** (`HybridRecurrentSWASpecialization`) | Sliding-window attention, linear attention, Mamba/SSM/hybrid recurrent checkpoints (e.g., `qwen3_5`, `lfm2`, `granite`) | Computes accurate SWA/SSM KV footprints in `RAMFitCalculator`; omits incompatible GGUF `--cache-reuse`; configures `--ctx-checkpoints 32 --checkpoint-min-step 256`. |
| **Standard Full Attention** (`StandardAttentionSpecialization`) | Standard dense / MoE transformers without SWA/SSM state | Allows full prefix KV shifting and `--cache-reuse` alongside `APCSnapshotCache` prefix reuse. |

### Multimodal Embeddings (`EmbeddingGemma 2` & GGUF Embeddings)

Defined in `Sources/HeylookEmbeddings/Engine/EmbeddingModelFacts.swift` and `NativeMLXEmbeddingGemma2.swift`:
- **Dynamic Prompt Discovery**: `PromptCatalog` parses `config_sentence_transformers.json` (`prompts` and `default_prompt_name`) from the checkpoint rather than relying on hardcoded model tables.
- **MLX vs `hybrid_ane` vs `all_ane`**:
  - MLX GPU runs straight from safetensors weights.
  - `hybrid_ane` runs Vision and/or Audio `.mlmodelc` towers on Apple Neural Engine (`ANECoreMLRuntime`) and passes resulting `softTokens` directly into `MLXVLM.EmbeddingGemma2.embed(inputIds:attentionMask:softTokens:)` without needing `TextANE.mlmodelc`.
  - `all_ane` requires compiled `TextANE.mlmodelc` in addition to Vision/Audio towers.
- **GGUF Embeddings (`GGUFEmbeddingEngine`)**: Automatically sizes `ctx_size <= n_ubatch <= n_batch` via `EmbeddingModelFacts.ggufEmbeddingProfile` and rejects video `windows` requests with HTTP `400`.

---

## 6.2 Template, Thinking & Special-Token Verification

1. **Zero-Heuristic Render Probes (`ModelTemplateInfo`)**: `detectThinkingControls` probes `HeylookJinjaRenderer` with AST variable inspection + `garbageDepthSentinel = "zq-not-a-level"` to derive `unknown` depth handling (`ignored` / `verbatim` / `fallback` / `raises`), `changes_prefix`, `off`, `aliases`, and `budgetEnforced`.
2. **4-Protocol Streaming (`ReasoningParser`)**: Supports `.tagPair` (`<think>...</think>`, `<|thought|>...<|/thought|>`, `<thought>...</thought>`), `.gemmaChannel` (`<|channel>thought\n...<channel|>`), `.harmonyChannel`, and `.recipientChannel` out of the box.
3. **Override Non-Invasiveness (`chat_template.heylook.jinja`)**: Never modify upstream vendor `tokenizer_config.json` or `chat_template.jinja` files when tuning templates. Place edits in `chat_template.heylook.jinja` (or use the Models config sheet / `PUT /v1/admin/models/*id/chat-template`). Inspect active ladder resolution, prefix stability (`checkPrefixStability`), and refused conversation shapes via the UI or API.
4. **Upstream Retirement Checklist**: When upgrading `mlx-swift-lm` or `llama.cpp`, verify whether runtime shims (`PreRenderedVLMTokenizerProxy`, `LeviathanMTPTokenIterator`, GGUF `--ctx-checkpoints` / `--cache-reuse` exclusions) can be simplified or retired against live upstream behavior.

---

## 6.3 Quantization, Core AI & CoreML ANE Export Playbooks

### Pure-Swift MLX Quantization (`convert-mlx`)

`MLXModelConverter` (`Sources/HeylookEngines/Providers/MLXSwiftProvider.swift`) runs `MLXLMCommon.convert` natively in Swift with `affine`, `mxfp4`, `mxfp8`, `nvfp4`, and `q4_zero` modes:

```bash
./scripts/build_native.sh cli convert-mlx <src-dir> <dst-dir> --bits 4 --group-size 64 --mode affine
```

Also accessible directly from the UI (`Models` -> `Convert MLX…`) or dry-run validated via `--dry-run`.

### Core AI `.aimodel` Export (`scripts/export_coreai.sh`)

```bash
./scripts/export_coreai.sh list
./scripts/export_coreai.sh [llm|gemma4|whisper|parakeet|embedding|compile] ...
```

Exports compiled Core AI `.aimodel` bundles (`apple/coreai-models` + `xcrun coreai-build`). Build Swift targets with `HEYLOOK_ENABLE_COREAI=1` when exercising `CoreAIProvider` and `CoreAISpeech`.

### 3-Tower CoreML ANE Export (`EmbeddingGemma 2`)

```bash
uv run --python 3.12 --with coremltools --with safetensors --with "torch<2.8" --with "numpy<2" \
  python ./scripts/export_ane_coreml.py --compile
```

Outputs compiled `.mlmodelc` encoders inspectable via:

```bash
swift run heylook-cli inspect-ane <checkpoint-dir>
swift run heylook-cli inspect-checkpoint <checkpoint-dir>
```
