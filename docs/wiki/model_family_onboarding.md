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

---

## 5. Model Conversion & Quantization Pipelines (Native Swift vs. Native Core AI)

### 5.1 Converting Models to Native MLX Swift (`MLXModelConverter` & `heylook-cli convert-mlx`)

`mlx-swift-lm` executes `.safetensors` weights directly on Metal GPU via `MLXSwiftProvider`, giving you three ways to onboard or quantize a model:

1. **Zero-Conversion Drop-In (`.safetensors` `BF16` / `FP16` / Pre-Quantized / ParoQuant)**:
   - Unquantized Hugging Face `.safetensors` checkpoints, `mlx-community` checkpoints, and **ParoQuant** (`quantization_config.quant_method == "paroquant"`, Givens rotations + channel scales) load directly with zero conversion step. Every model class in `mlx-swift-lm` runs `sanitize(weights:)` in Swift on load.
2. **100% Pure-Swift In-Process Quantization & Sharded Export (`MLXModelConverter` / `MLXLMCommon.convert`)**:
   - `HeylookEngines.MLXModelConverter` wraps `LLMModelFactory.shared.convert` and `MLXLMCommon.convert` (`Libraries/MLXLMCommon/ModelConversion.swift` in `mlx-swift-lm`) to quantize, shard (`<= 5 GB` per shard + `model.safetensors.index.json`), copy sidecars (`*.json`, `*.jinja`, `*.tiktoken`), and write `"quantization"` into `config.json` entirely in Swift:
     ```bash
     # Inspect conversion plan (offline / dry-run)
     swift run heylook-cli convert-mlx ./raw-model ./quantized-model --bits 4 --group-size 64 --mode affine --dry-run

     # Execute full pure-Swift Metal quantization & sharded safetensors export
     ./scripts/build_native.sh cli convert-mlx ./raw-model ./quantized-model --bits 4 --group-size 64 --mode affine
     ./scripts/build_native.sh cli convert-mlx ./raw-model ./quantized-q4-0 --calibration q4_zero
     ./scripts/build_native.sh cli convert-mlx ./raw-model ./quantized-mxfp8 --mode mxfp8
     ```
   - Supported pure-Swift quantization modes (`MLXQuantizationMode`):
     - `affine` (`4-bit` or `8-bit`, default `group_size = 64`, or `calibration = q4_zero` for exact symmetric 32-element `Q4_0` lattice alignment)
     - `mxfp4` (`4-bit`, default `group_size = 32`)
     - `mxfp8` (`8-bit`, default `group_size = 32`)
     - `nvfp4` (`4-bit`, default `group_size = 32`)
3. **Offline AWQ / DWQ Quantization (`mlx_lm.convert` / `mlx_vlm.convert`)**:
   - When calibration-dataset AWQ or Distilled Weight Quantization (DWQ) is desired, run `mlx_lm.convert` or `mlx_vlm.convert` to produce standard MLX `.safetensors` shards.

---

### 5.2 Converting Models to Native Core AI (`scripts/export_coreai.sh` & `scripts/export_ane_coreml.py`)

Core AI models are exported as `.aimodel` resource bundles (`metadata.json` Schema `0.2` + `main.aimodel` + optional `prefill.aimodel`, `vision.aimodel`, `dflash.aimodel`) or 3-tower CoreML `.mlmodelc` bundles:

1. **Catalog & Experimental `.aimodel` Export (`./scripts/export_coreai.sh`)**:
   ```bash
   # List supported LLM, diffusion, and utility presets in coreai-models
   ./scripts/export_coreai.sh list --type llm

   # Export macOS 27.2+ GPU .aimodel (defaults to 4-bit block-32 + prefill function)
   ./scripts/export_coreai.sh llm Qwen/Qwen3.5-9B

   # Export macOS 27.2+ with FP4 weights + FP8 activations + FP8 KV cache
   ./scripts/export_coreai.sh llm Qwen/Qwen3.5-9B --compression fp4_weights_fp8_activations_fp8_kv_cache

   # Export iOS 27.2+ / ANE .aimodel (defaults to 4-bit group-32 LUT palettization + INT8 embeddings)
   ./scripts/export_coreai.sh llm Qwen/Qwen3-1.7B --platform iOS

   # Export Gemma 4 with static context buckets, INT8 PLE sidecar, and multi-bucket vision encoder
   ./scripts/export_coreai.sh gemma4 google/gemma-4-E4B-it --include-vision

   # Export EmbeddingGemma 2 .aimodel (macOS GPU or iOS ANE)
   ./scripts/export_coreai.sh embedding embeddinggemma-2 --platform iOS

   # Optional Ahead-of-Time (AOT) compilation via xcrun coreai-build
   ./scripts/export_coreai.sh compile ~/.heylook/models/Qwen3.5-9B/main.aimodel --platform macOS
   ```

2. **`coreai-opt` Compression Presets (`macOS 27.2+` vs. `iOS 27.2+` / ANE)**:

   | Target | `--compression` Preset | Hardware Representation |
   | :--- | :--- | :--- |
   | `macOS` | `4bit` *(default)* | INT4 weight-only, block size 32 |
   | `macOS` | `4bit_weights_8bit_kv_cache` | INT4 per-block weights + INT8 symmetric per-tensor KV cache |
   | `macOS 27.2+` | `fp4_weights_fp8_activations_fp8_kv_cache` | Per-block **FP4 (`e2m1`)** weights + **FP8 (`e4m3`)** activations & KV cache |
   | `macOS 27.2+` | `fp8_weights_fp8_activations_fp8_kv_cache` | Per-tensor **FP8 (`e4m3`)** weights, activations, and KV cache |
   | `iOS / ANE` | `4bit_weight_palettized_group32` *(default)* | 4-bit LUT palettization (channel group size 32) + INT8 embedding table |
   | `iOS / ANE` | `4bit_weight_palettized_group8` | Higher-accuracy 4-bit LUT palettization (channel group size 8) |

3. **3-Tower `EmbeddingGemma 2` CoreML ANE Exporter (`./scripts/export_ane_coreml.py`)**:
   Compiles `EmbeddingGemma2VisionANE.mlmodelc` (100% ANE), `EmbeddingGemma2AudioANE.mlmodelc` (99% ANE), `EmbeddingGemma2TextANE.mlmodelc` (100% ANE), and `embed_tokens_fp16.bin` with `ANESafeRMSNorm` FP16 overflow pre-scaling (`1/16`):
   ```bash
   uv run --python 3.12 --with coremltools --with safetensors --with "torch<2.8" --with "numpy<2" \
     python ./scripts/export_ane_coreml.py --compile
   ```

---

## 6. Hybrid Compute Placement (ANE + GPU + CPU) & When to Choose Core AI vs. Native MLX Swift

### 6.1 Running Across ANE + GPU + CPU
Core AI supports heterogeneous execution at two levels:
- **Multi-Asset / Multi-Tower Bundles (Recommended)**: A single `.aimodel` directory assigns each sub-model to the compute unit that matches its memory/compute profile—for example, `token_table` / `ple` lookup tables on **CPU/GPU**, `vision` / `audio` / `embedding` / `dflash` drafter towers on **ANE** (`BC1S` `(B, D, 1, S)` with `1x1 Conv2d`), and the `main` / `prefill` autoregressive decoder on **GPU** (`(B, S, D)` with `nn.Linear` and fused SDPA).
- **Single-Graph Partitioning (Avoid Mid-Layer Transfers)**: While `CoreAI` automatically falls back to GPU/CPU for ops unsupported on ANE (such as `fp32` intermediates or rank > 5 tensors), switching `ANE <-> GPU <-> CPU` *inside* a decoder layer segments the compiled graph and inserts expensive `BC1S <-> BSD` layout transposes. Keep each compiled function 99–100% resident on its target accelerator.

### 6.2 Why Not Use Core AI Everywhere Instead of Native MLX Swift?

| Dimension | Native MLX Swift (`MLXSwiftProvider`) | Native Core AI (`CoreAIProvider` & `ANECoreMLRuntime`) |
| :--- | :--- | :--- |
| **Target Accelerator on macOS** | Metal GPU + Accelerate CPU over unified memory | `macOS` exports run on Metal GPU/CPU; `iOS` / `BC1S` exports run on ANE (suited for compact low-wattage models) |
| **Onboarding & Load Time** | **Zero export step**: `mmap`s Hugging Face `.safetensors` directly in seconds | Requires offline `torch.export` + `coreai-opt` + `TorchConverter` export and device specialization |
| **Multi-Turn KV Cache Slicing** | **Dynamic `[KVCache]` in Swift**: grows incrementally and trims in place (`cache.trim(n)`) on message edits/branches | Stateful `NDArray` buffers bound to compiled `max_context_length` or static `(context_bucket, query_length)` functions |
| **Architecture Breadth** | **61 LLMs + 20 VLMs** in pure Swift (`mlx-swift-lm`) | ~15 LLM families + utility/speech/diffusion recipes in `coreai-models` |
| **Best Workload Fit** | Primary Mac Studio interactive LLMs & VLMs (`0.8B` to `122B-A10B`), multi-turn branching chat, dynamic vision resolution | **ANE Co-Resident Lane** (always-on `EmbeddingGemma 2`, `Whisper`/`Parakeet` voice transcription, vision pre-encoding), **`iOS 27.2+` palettized ANE models**, **`DFlash` block-diffusion**, and **`macOS 27.2+` `FP4`/`FP8` KV-cache bundles** |
