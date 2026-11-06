# Vision

`heylookitsanllm-macos` (`heylook`) is a native Swift 6.2 personal inference workbench and server for Apple Silicon (`macOS 27.2+` and `iOS 27.2+`). It runs the models the owner wants to use—across **Metal GPU (`MLX Swift`)**, **Apple Neural Engine (`CoreAI` / `CoreML` / `Argmax`)**, and **`GGUF` (`llama-server`)**—and makes them seamless to drive from a Mac Studio desktop app, a native iPhone/iPad app, a browser, or local developer tools. It is a daily driver and a workbench at the same time.

## What it is for

- Running current open models locally on Apple Silicon with **vision, audio, multimodal embeddings, and thinking as the normal case**, not the special one.
- Exploiting **heterogeneous Apple Silicon compute (192 GB Metal GPU + Apple Neural Engine + CPU)**: keeping a large primary LLM/VLM resident in Metal wired memory while embeddings (`EmbeddingGemma 2` ANE towers, `CoreAI` `.aimodel` bundles), speech (`CoreAISpeech` / `WhisperKit` / `TTSKit`), and auxiliary tasks run concurrently on the Neural Engine without GPU command-queue contention or LRU eviction.
- Being the single place the owner's devices and tools call for inference, over one well-defined Anthropic Messages-conformant `/v1/*` wire.
- Trying things: a new model, a new quant, an ANE/GPU tower split, a speculative drafter, or a retrieval strategy, with a short path from download to answer to measurement.

## What it is not

- A multi-user or hosted service.
- A model zoo. `heylook` does not re-implement architectures for coverage. Model and runtime code it owns is code it chose to specialize and tune.
- A compatibility layer for every client protocol. One wire (`/v1/messages`, `/v1/conversations/{id}/generate`, `/v1/embeddings`), done well.
- A place to preserve old behaviour or old data for its own sake.

## Principles

1. **Nothing happens where it cannot be seen.** Every setting in force can be explained: its value, where it came from (`derived`, `configured`, `observed`, `observed_cached`, `unknown`, `not_applicable`), its effect class (`per_request`, `reload`, `restart`), and its active compute unit (`mlx_gpu`, `ane`, `cpu_and_gpu`, `cpu_only`, `gguf_subprocess`, `remote`). Every request reports what it cost and what it reused. A silent fallback is a bug even when the answer looks right.
2. **Derive, don't copy.** Facts come from where they live: the model's own files (`config.json`, `.aimodel` `metadata.json`, `.mlmodelc` compute plan, GGUF header), its template, and the engine's own report. A hand-kept list of anything derivable will drift.
3. **Evidence over belief.** Claims about speed, memory, retrieval recall, or behaviour are measured through the real path, with their conditions written down. A measurement without its conditions is an anecdote.
4. **Enforce, don't remind.** A rule that matters is a Swift 6 compile-time check (`Sendable`, actor isolation) or a test that runs. Anything else is a hope.
5. **All engines are first class.** `heylook` speaks one API across `MLXSwift`, `CoreAI`/`CoreML`, and `GGUF`. A feature either works across engines or is refused loudly (`400 invalid_request_error`) where an engine cannot support it. Neither engine is held back to match the slowest.
6. **Build on upstream, own what you specialize.** Use `mlx-swift`, `mlx-swift-lm`, `coreai-models`, `argmax-oss-swift`, and `llama.cpp` as far as they go; replace a layer only to specialize it for the owner's models and hardware. A custom path must beat the upstream reference before it replaces it, and is kept only while it earns its place.
7. **Simple where it can be, with judgment.** Remove what is dead or duplicated. Keep what earns its place, even when removing it would be tidier.
8. **The interface tells the truth.** It discloses cost, cache reuse (`cache X/Y`), thinking budgets (`End thinking`), and ANE/GPU memory splits rather than hiding them behind spinners, and interrupts only before something would be lost. What works on the Mac Studio desktop works on the phone.
9. **Private by default.** Conversations, notebooks, and indexed media vectors live on the owner's devices in SQLite (`WAL` mode). Nothing is logged or sent anywhere unless the owner turns it on.

## Changing this document

It follows the owner's use as that use evolves. When a change would conflict with something written here, raise the conflict with the owner before building it, name the principle it collides with, and let the owner decide: change the work, or change this document. The vision moves by an explicit edit, never by drift.
