# Model Family Support & Porting Playbook (`heylookitsanllm-macos`)

`heylookitsanllm-macos` is a **100% native Swift 6.2** personal inference workbench and server targeting `macOS 27.2+` and `iOS 27.2+`. Model discovery, chat template analysis, reasoning stream parsing, special-token scrubbing, KV cache reuse, Metal GPU execution, and Neural Engine (ANE) execution run natively in Swift without any Python runtime dependency.

---

## 1. Supported Model Families Across the 3 Execution Tiers

| Tier | Runtime Framework | Provider Actor | Supported Model Families |
| :--- | :--- | :--- | :--- |
| **Tier 1: In-Process MLX Swift (Metal GPU)** | `ml-explore/mlx-swift` + `ml-explore/mlx-swift-lm` (`MLXLLM`, `MLXVLM`, `MLXLMCommon`) | `MLXSwiftProvider` | **61 LLM architectures** (`LLMTypeRegistry`): `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_5` (`_moe`/`_text`), `qwen3_8` (`_moe`/`_text`), `gemma`, `gemma2`, `gemma3` (`_text`), `gemma3n`, `gemma4` (`_unified`/`_text`), `deepseek_v2`, `deepseek_v3`, `gpt_oss`, `llama`, `mistral`, `mistral3`, `mixtral`, `glm4` (`_moe`/`_moe_lite`), `minimax`, `mimo` (`_v2_flash`), `lfm2` (`_moe`), `phi`/`phi3`/`phimoe`, `olmo2`/`olmo3`/`olmoe`, `nemotron_h`, `nemotron_labs_diffusion`, `jamba`, `mamba2`, `falcon_h1`, `ernie4_5`, `exaone4`, `bailing_moe`, `baichuan_m1`, `smollm3`, `granite` (`moehybrid`), `helium`, `hunyuan_v1_dense`, `apertus`, `afmoe`, `bitnet`, `cohere`, `starcoder2`, `internlm2`, `minicpm`, `nanbeige`, `nanochat`, `lille-130m`, `mellum`, `openelm`<br>**20 VLM architectures** (`VLMTypeRegistry`): `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` (`_moe`), `qwen3_5` (`_moe`), `qwen3_8` (`_moe`), `gemma3`, `gemma4` (`_unified`), `muse_glimmer`, `pixtral`, `mistral3`, `lfm2_vl`, `glm_ocr`, `paligemma`, `idefics3`, `smolvlm`, `fastvlm`<br>**Speculative Drafters** (`MTPDrafterTypeRegistry`): `gemma4_assistant`, `qwen3_5_mtp`, `qwen3_8_mtp` |
| **Tier 2: Apple Neural Engine (ANE) & CoreAI** | `apple/coreai-models` (`CoreAILanguageModels`, `CoreAISpeech`, `CoreAIVision`), `argmaxinc/argmax-oss-swift` (`WhisperKit`), `CoreML` | `CoreAIProvider` & `HeylookEmbeddings` (`ANECoreMLRuntime`) | **`.aimodel` bundles** (`Schema 0.2`): `gemma3`, `gemma3n`, `gemma4`, `gpt_oss`, `mistral`, `mixtral`, `muse_glimmer`, `olmo2`, `phi`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_moe`, `smollm2`, `vlm`<br>**Multimodal Embeddings**: `EmbeddingGemma 2` (`embeddinggemma-2-MLX-BF16` + Vision/Audio/Text ANE `.mlmodelc` towers)<br>**Speech**: `whisper`, `parakeet`, `wav2vec2` |
| **Tier 3: GGUF (`llama-server`)** | `llama.cpp` subprocess + pure-Swift `GGUFHeaderReader` | `GGUFSubprocessProvider` | Any GGUF architecture supported by `llama.cpp` (`qwen35`/`qwen35moe`, `qwen38`, `gemma4`, `llama4`, `deepseek2`/`deepseek3`, `gpt-oss`, `mamba2`, `jamba`, `command-r`, etc.) with automatic `mmproj-*.gguf` vision/audio pairing, `--cache-ram 8192` hybrid-recurrent SSM sizing, and `--spec-type draft-mtp` |

---

## 2. Zero-Hardcoding Onboarding (`ModelTemplateInfo` & `ReasoningParser`)

Following `VISION.md` Principle 2 (*"Derive, don't copy"*), `HeylookCore` derives model capabilities and streaming conventions directly from checkpoint files (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `.aimodel/metadata.json`, and `.gguf` headers) via `ModelTemplateInfo`:

1. **4 Structural Reasoning Formats (`ReasoningFormat`)**:
   - `.tagPair(open:close:prefillsThinking:)`: `<think>...</think>`, `<|thought|>...<|/thought|>`, `<thought>...</thought>`, with automatic `prefillsThinking` detection when the generation-prompt tail of `chat_template` opens `<think>` without closing it.
   - `.gemmaChannel(openPrefix:closeTag:)`: `<|channel>thought\n...<channel|>` (Gemma 4).
   - `.harmonyChannel`: `<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>` (GPT-OSS).
   - `.recipientChannel`: `to=self<|message|>...<|eom|><|start|>assistant to=user<|message|>...<|eot|>` (Muse-Glimmer / Muse-Spark).
2. **Template-Derived Thinking Controls (`ThinkingControls`)**:
   - Detects `enable_thinking` (`switchVariable`) and depth controls (`reasoning_effort`, `reasoning_strength`, `thinking_mode` + valid `depthValues` such as `["low", "medium", "high"]` or `["none", "low", "medium", "high", "max"]`).
   - Marks non-reasoning templates (`Llama 3.2`, `Gemma 2`, `Mistral`) as `supportsThinking = false` so the UI and `/v1/models` never advertise dead thinking controls.
3. **Dynamic Special & Turn-End Token Scrubbing (`StripSpecials`)**:
   - Unions `"special": true` tokens from `tokenizer.json` (`added_tokens`) and `tokenizer_config.json` (`added_tokens_decoder`), stripping checkpoint-specific turn markers (`<turn|>`, `<|eot|>`, `<|eom|>`) while keeping `reasoningFormat.structuralDelimiters` intact until `ReasoningParser` consumes them.
4. **File-Derived Vision Inspection & Runtime VLM-to-LLM Fallback (`ModelTemplateInfo.hasVisionConfiguration`)**:
   - Inspects `config.json` (`vision_config`), `processor_config.json`, `preprocessor_config.json`, and `.aimodel/metadata.json` (`vision.backbone`) directly on disk without maintaining static `model_type` string sets.
   - At load time, `MLXSwiftProvider.ensureContainerLoaded()` attempts `VLMModelFactory.shared` first when `supportsVision` is true and automatically falls back to `LLMModelFactory.shared` (updating `supportsVision = false` with `.observed` provenance) if a checkpoint ships a `vision_config` stub for an LLM-only `model_type`.

---

## 3. Playbook: Porting a New Model Family to Tier 1 (`mlx-swift-lm`)

Porting a new LLM or VLM architecture to **Tier 1 (`mlx-swift-lm`)** takes **~2–6 hours (~300–500 LOC of Swift)** for an LLM and **~1–2 days (~700–1,200 LOC)** for a VLM:

1. **Create the Model File in `mlx-swift-lm`**:
   - Add `Libraries/MLXLLM/Models/<Family>.swift` (or `Libraries/MLXVLM/Models/<Family>.swift`).
   - Define the `Codable` configuration struct matching `config.json` (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `rope_theta`, etc.).
   - Compose the decoder layers using `MLXNN` and `MLXLMCommon` primitives:
     - `Linear` / `SwitchLinear` (for MoE routing)
     - `RMSNorm` / `LayerNorm`
     - `RoPE` / `YarnRoPE`
     - `MLXFast.scaledDotProductAttention` with `KVCacheSimple`, `RotatingKVCache`, or `MambaCache` (for hybrid SSM/recurrent layers)
   - Implement `sanitize(weights: [String: MLXArray]) -> [String: MLXArray]` to strip `model.` prefixes or drop unused tower weights when loading a unified checkpoint as text-only.
   - Conform to `ChatConventionsProviding` if the family has an intrinsic `ReasoningConfig` or `ToolCallFormat`.
2. **Register `model_type` (1 Line)**:
   - Add the `model_type` entry to `LLMTypeRegistry` (`LLMModelFactory.swift`) and/or `VLMTypeRegistry` (`VLMModelFactory.swift`). No static `model_type` table updates are needed in `HeylookCore`.
3. **Verify in `heylookitsanllm-macos`**:
   - Drop the checkpoint into the watched models directory and run `swift run heylook-cli models` and `swift run heylook-cli eval-lab`.

---

## 4. Playbook: Porting a New Model Family to Tier 2 (`coreai-models` / ANE)

Tier 2 has a **generic Swift runtime** (`CoreAIRunner` + `CoreAIProvider` + `ANECoreMLRuntime`), so no new Swift model classes are needed for standard `.aimodel` bundles. All porting work happens at export time:

### Tier 2A: CoreAI GPU `.aimodel` Bundle (~1 Day)
1. Author the model graph in `coreai-models` using standard `(B, S, D)` layout, `nn.Linear` / `SwitchLinear`, and explicit stateful buffers (`keyCache`, `valueCache`, plus `convStates` / `recurrentStates` for hybrid models).
2. Add a quantization YAML config (4-bit per-block asymmetric `uint4` with `block_size: 32`, leaving MoE router gates unquantized).
3. Export the `.aimodel` bundle (`schema_version: "0.2"` `metadata.json` + `main.mlmodelc` + optional `prefill.mlmodelc`, `vision.mlmodelc`, `dflash.mlmodelc`).
4. Place the `.aimodel` bundle in `heylook`'s models folder; `CoreAIBundleManifest.inspect(at:)` automatically reads `language.max_context_length`, `vision.backbone`, and `speculative.drafter_kind`.

### Tier 2B: 100% Apple Neural Engine (ANE) Residency (~2–4 Days)
When targeting 100% ANE residency (so the model runs on the ANE lane concurrently with a 100B+ GPU model without acquiring `GenerationGate`):
1. **Use `BC1S` 4D Tensor Layout**: Reshape activations from `(B, S, D)` to `(B, D, 1, S)` and replace `nn.Linear` with `nn.Conv2d(in_channels, out_channels, kernel_size=1)`.
2. **Static Shapes & Per-Head Sequential Attention**: Compile with static sequence/window shapes and per-head sequential `K @ Q` attention so the CoreML ANE compiler places 99–100% of ops on the Neural Engine.
3. **Prevent FP16 Overflow (`<= 65,504`)**: The ANE executes strictly in FP16. Models with large residual streams or RMSNorm scales (such as Gemma 2/3/4 and EmbeddingGemma 2) require pre-scaling before squaring (`ANESafeRMSNorm`) and externalized embeddings (`(V, 1, D)` or memory-mapped `embed_tokens_fp16.bin`).
4. **Verify PSNR & ANE Residency**:
   - Check layer-by-layer PSNR (`>= 70 dB` layout parity in PyTorch, `>= 40 dB` compiled FP16 vs. reference, `>= 35 dB` after 4-bit palettization).
   - Run `swift run heylook-cli inspect-ane <path>` to confirm `MLComputePlan` reports `99–100%` Neural Engine residency with zero unexpected CPU/GPU fallbacks.
