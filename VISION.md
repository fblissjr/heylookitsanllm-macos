# Vision (`heylookitsanllm-macos`)

> A timeless statement of purpose, enduring principles, tradeoff rules, and long-term direction for **`heylook`** (`heylookitsanllm-macos`). Tactical milestones, subsystem status tables, and concrete runtime benchmarks belong in `docs/project/` and the repository wiki; this document captures what should still hold true even if every runtime dependency, concurrency primitive, model architecture, Apple Silicon generation, and OS release is replaced five years from now.

---

## 1. Purpose — Why `heylook` Exists

**`heylook`** (`heylookitsanllm-macos`) is a personal native macOS & iOS inference studio, multimodal media/embedding workbench, and embedded `/v1/*` HTTP/SSE server built for a single owner on Apple Silicon. It runs the open models the owner wants to use—with **vision, audio, multimodal embeddings, and thinking as the normal case**, not the exception—and makes them seamless to drive from a Mac workstation at a desk, an iPhone or iPad in hand, a browser, or the owner's own local tools.

Its non-negotiable baseline floor is **complete behavioral, wire, engine-contract, and 6-surface frontend parity with `heylookitsanllm`**, expressed on a timeless, sovereign, native Apple Silicon foundation. It is a daily driver and an engineering workbench at the same time:

- **Your files and weights stay yours, untouched in place** — model directories are strictly observed, never mutated behind your back; per-model overrides live in explicit sidecars (`model.heylook.toml`, `chat_template.heylook.jinja`), and conversations, notebooks, presets, and vector indexes reside in local SQLite (`WAL` mode) under your custody.
- **Every answer and setting tells the truth** — you see what parameter value is in force, where it came from, what prompt and template were actually rendered, what KV prefix was reused or missed, and what speculative drafting achieved.
- **A short path from checkpoint to measurement to daily craft** — trying a new model, a quantization recipe, an ANE/GPU tower split, or a speculative drafter moves straight through the real execution path with honest telemetry.

---

## 2. Core Beliefs & Guiding Principles

These seven commitments sit beneath concrete libraries and OS versions and are meant to endure across generations of hardware and runtimes.

### 1. Commodity Upstream, Surgical Specialization
Open upstream projects—`mlx-swift`, `mlx-swift-lm`, `coreai-models`, `llama.cpp`, `swift-transformers`, and emerging on-device runtimes such as `LiteRT`—are the default home for model coverage, operators, and kernels. `heylook` does not re-implement model architectures merely for breadth.

Where upstream leaves meaningful performance, correctness, or hardware utilization on the table for models in daily use, `heylook` owns **clean, composable specialization hooks**—such as `EmbeddingGemma 2` hybrid ANE + GPU tower splitting, `Gemma 4` visual padding trimming and single-row attention mask fast paths, lossless Leviathan MTP rejection sampling, and tailored KV cache capture policies. Every owned specialization must earn its keep with reproducible measurements against the upstream baseline and carry an explicit upstream-retirement check so it can be deleted cleanly once upstream absorbs the fix.

### 2. Runtime-Agnostic Core, First-Class Engines
`heylook` presents one unified API, scheduler, and telemetry contract across heterogeneous execution engines: **Metal GPU (`MLX Swift`)**, **Apple Neural Engine (`Core AI` / `CoreML`)**, **process-isolated `GGUF` (`llama-server`)**, **remote LAN/Tailscale peers (Bonjour `_heylook._tcp`)**, and future runtime adapters such as **`LiteRT`**.

First class never means lowest common denominator. Each runtime has distinct strengths—unified-memory prefix reuse, static ANE compilation, or `llama.cpp` slot checkpoints—and the core never flattens those differences away. Controls or modalities an engine cannot honor fail loudly and explicitly (`400 invalid_request_error` or `422`), never silently coerced or dropped.

### 3. Truthful Runtime — Zero Silent Fallbacks, Zero Fakes
Nothing happens where it cannot be seen, and nothing pretends to be what it is not. Every capability, context ceiling, and sampler value carries explicit provenance (`derived`, `configured`, `observed`, `observed_cached`, `unknown`, `not_applicable`), its effect class, and its compute placement.

Production providers, schedulers, embedding engines, and evaluation suites must never silently fabricate output, masquerade hash vectors as semantic embeddings (`allowSyntheticFallback` defaults to `false`), inject demo fixtures as real checkpoints, or ship hardcoded geometry approximations (such as a fixed `28 px` patch stride) when checkpoint files exist to derive them. A silent fallback is a bug even when the output looks plausible.

### 4. Derive, Don't Copy — One Orchestration Spine
Truth comes from where it lives: checkpoint configurations (`config.json`, `processor_config.json`, `generation_config.json`, `config_sentence_transformers.json`), template files (`chat_template.jinja`, `chat_template.heylook.jinja`), `.aimodel` / `.mlmodelc` metadata, `GGUF` headers, and declarative Swift descriptors (`ProviderConfigSchema`, `SamplerCascade`). Thinking toggles, depth levels, and unknown-level refusals derive from AST inspection and actual render probes—never brittle model-ID substring lists.

Equally critical: SwiftUI workbench surfaces and `/v1/*` HTTP/SSE callers drive the **exact same orchestration, validation, and SQLite persistence spine** (canonical conversation generation, `RequestRegistry` locking, and atomic tail commit). Shadow execution paths that bypass server invariants are architectural defects.

### 5. Heterogeneous Silicon Harmony
Apple Silicon unifies high-bandwidth GPU memory, the Apple Neural Engine (ANE), and CPU cores on one die. `heylook` treats that asymmetry as a first-class scheduling problem rather than letting one workload starve another: large primary LLMs and VLMs remain pinned in Metal wired memory behind a serialized GPU generation gate, while multimodal embedding towers (`EmbeddingGemma 2`), on-device speech recognition (`SpeechTranscriber` / `CoreAISpeech`), and background utilities execute concurrently on the ANE and CPU without GPU queue contention or LRU eviction.

### 6. Intimate in the Hand, Effortless at the Desk
Checking a run or dictating a thought on an iPhone or iPad deserves the same craft as dissecting prompt markers, cosine matrices, or kernel telemetry on a Mac workstation. `heylook` targets `macOS 27.2+` and `iOS 27.2+` today (supporting `27.2.1` hosts and guarding `27.3+` SDK features cleanly with `#available`), adapting fluidly between touch-first voice handoffs, compact live readouts, and dense multi-pane inspection tools.

That commitment extends forward cleanly across OS generations: clean `#available(macOS 27.3, iOS 27.3, *)` seams for Siri `@AppIntent(.tool)` (`ToolIntent`) and Spotlight `IndexedEntity` integration, paired with durable signing and TCC / Full Disk Access protected-folder custody so user grants survive rebuilds.

### 7. Sovereign by Construction & Elegant Simplicity
Privacy is structural, not a policy banner. Conversations, notebooks, presets, SHA-256 deduplicated `media_blobs`, observability JSONL logs, and vector stores live locally in private SQLite (`WAL` mode) with zero external telemetry, strict DNS-rebinding `Host` verification, and opt-in admin authentication (`X-Heylook-Admin-Token`). Model directories remain untouched except for explicit operator-authored sidecars.

Simplicity is defended actively: prefer holistic refactoring over local bandaids, decompose monolithic view models and routers along clear ownership boundaries, delete dead scaffolding and unused compatibility shims, and enforce invariants with Swift 6.4 strict concurrency (`Sendable`, actor isolation) and contract tests rather than comments.

---

## 3. The Tradeoff Compass

When two valid engineering choices collide, `heylook` breaks ties using five ordering rules:

1. **Truthfulness and durability over apparent smoothness.** Never hide unloaded weights, unsupported engine controls, missing runtimes, or cache misses behind synthetic fallbacks, silent coercions, or optimistic approximations.
2. **Holistic refactoring over local bandaids.** When a shared invariant breaks because UI and HTTP diverged or a monolith grew entangled, unify the underlying spine rather than patching symptoms in two places.
3. **Complete behavioral and frontend parity with `heylookitsanllm` before net-new surface expansion.** Close every wire, schema, KV-cache, engine, and 6-surface parity gap against `heylookitsanllm` (`v2.0.251+`) as the foundation before adding native-only capabilities on top.
4. **Upstream leverage for breadth, measured specialization for depth.** Rely on `mlx-swift-lm`, `coreai-models`, and `llama.cpp` by default; own custom specialization code only when measured benchmarks prove a clear win, and retire it when upstream lands the capability.
5. **Derived schemas and compile-time invariants over manual synchronization.** Let checkpoint files, declarative descriptor tables, and Swift 6 type/actor safety enforce consistency instead of relying on hand-copied lists or tribal memory.

---

## 4. Explicit Non-Goals

Knowing what `heylook` refuses to become keeps the codebase lean, truthful, and calm:

- **Not a cloud or multi-tenant service:** No multi-user identity, public-internet exposure, billing, or SaaS fleet machinery.
- **Not a hand-maintained model zoo:** No hardcoded per-model-ID routing branches (`"deepseek-r1"`, `"qwq"`, `"embed"`) or custom re-implementations of architectures solely for catalog breadth.
- **Not a multi-dialect protocol shim:** One clean Anthropic Messages-conformant `/v1/*` wire and unified conversation/embedding/admin contract—never sprawling adapters for unrelated wire dialects.
- **No phantom dependencies or speculative docs:** Only real dependencies declared in `Package.swift` belong in active architecture (never phantom references such as `hummingbird` or `argmax-oss-swift`).
- **No legacy schema migration cruft:** Solo deployment favors clean idempotent `CREATE TABLE IF NOT EXISTS` definitions and clean resets over multi-version migration ladders.

---

## 5. The Living Horizon

`heylook` moves along three disciplined horizons—never skipping ahead of its foundation:

1. **Horizon 1 — Complete Parity Baseline (`heylookitsanllm` Floor):** Close all core schema/provenance, `SamplerCascade`, `TOMLCodec`, MLX VLM cache & lossless MTP, GGUF SWA/cache/stop, canonical `/v1/conversations/{id}/generate` saga, and 6-surface SwiftUI workbench parity gaps, backed by pure-invariant contract tests and live hardware verification.
2. **Horizon 2 — Composable Specialization & Multi-Runtime Foundation:** Cleanly separate runtime execution adapters (`RuntimeAdapter`) so emerging engines like `LiteRT` plug in alongside `MLX Swift`, `Core AI`, and `GGUF`, while surgical model specializations (`Gemma 4`, `EmbeddingGemma 2`) live behind measurable, retirement-tested hooks.
3. **Horizon 3 — Deep Native OS Integration (`27.3+` & Beyond):** Bring sovereign local intelligence to the OS surface via `#available(macOS 27.3, iOS 27.3, *)` Siri `@AppIntent(.tool)` (`ToolIntent`), Spotlight `IndexedEntity` indexing across conversations and multimodal artifacts, physical iOS device deployment with `.aimodel` + Bonjour failover, and durable Apple-signed TCC / Full Disk Access custody across macOS rebuilds.

---

## Changing This Document

This vision evolves only with the owner's actual use. Whenever a proposed feature, dependency, or shortcut conflicts with a principle or tradeoff rule above, surface the conflict explicitly by principle name before building and let the owner decide: adjust the work, or amend this document. The vision moves by deliberate edit, never by drift.
