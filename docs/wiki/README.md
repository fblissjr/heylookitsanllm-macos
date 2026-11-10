# Heylook LLM Knowledge Wiki (`docs/wiki/`)

Welcome to the **LLM Knowledge Wiki** for `heylookitsanllm-macos` (`heylook`) — a 100% native Swift 6.2 multiplatform (`macOS 27.2+` and `iOS 27.2+`) personal inference workbench, multimodal media/embedding studio, and embedded HTTP/SSE server for Apple Silicon.

`AGENTS.md` at the repository root acts as the **Index Router** into this wiki. When working on a specific subsystem or file glob, consult the corresponding wiki page below before making architectural changes.

---

## Domain Map & Routing Table

| Domain / Subsystem | Wiki Page | Primary Source Targets & Files |
| :--- | :--- | :--- |
| **1. System Topology & Core Invariants** | [architecture_overview.md](./architecture_overview.md) | `Package.swift`, `Sources/HeylookApp/`, `Sources/HeylookCLI/`, `Sources/HeylookUI/` |
| **2. Engine Contract, Provenance, Samplers & Templates** | [engine_contract_and_provenance.md](./engine_contract_and_provenance.md) | `Sources/HeylookCore/Domain/EngineDescription.swift`, `SamplerCascade.swift`, `ReasoningParser.swift`, `ModelTemplateInfo.swift`, `Config/HeylookConfig.swift` |
| **3. Providers, Streaming Pipeline & Dual-Lane Compute** | [providers_and_heterogeneous_compute.md](./providers_and_heterogeneous_compute.md) | `Sources/HeylookEngines/ComputeScheduler.swift`, `Providers/StreamingPipeline.swift`, `MLXSwiftProvider.swift`, `CoreAIProvider.swift`, `GGUFSubprocessProvider.swift`, `RemoteHeylookProvider.swift`, `SubprocessSupport.swift` |
| **4. Multimodal Embeddings, ANE Towers & Matryoshka Store** | [multimodal_embeddings_and_ane.md](./multimodal_embeddings_and_ane.md) | `Sources/HeylookEmbeddings/Domain/Artifact.swift`, `Engine/EmbeddingEngine.swift`, `NativeMLXEmbeddingGemma2.swift`, `ModelComputeOptions.swift`, `Pipeline/ArtifactProcessor.swift`, `Store/VectorStore.swift`, `IndexCoordinator.swift` |
| **5. HTTP/SSE Wire API, Disconnect Resilience & SQLite WAL** | [wire_api_and_sqlite_store.md](./wire_api_and_sqlite_store.md) | `Sources/HeylookCore/Wire/MessagesWire.swift`, `Store/SQLiteStore.swift`, `Sources/HeylookServer/HeylookHTTPServer.swift` |
| **6. Model Family Onboarding, Quantization & Export Playbooks** | [model_family_onboarding.md](./model_family_onboarding.md) | `Sources/HeylookCore/Domain/ModelTemplateInfo.swift`, `Sources/HeylookEngines/Providers/MLXSwiftProvider.swift` (`MLXModelConverter`), `scripts/export_coreai.sh`, `scripts/export_ane_coreml.py` |
| **7. SwiftUI Architecture & macOS 27.2+ Design System** | [ui_architecture_and_macos27.md](./ui_architecture_and_macos27.md) | `Sources/HeylookUI/`, `Sources/HeylookApp/`, `scripts/build_native.sh` |
| **8. Sharp Edges & Incident Rationales** | [sharp_edges.md](./sharp_edges.md) | Cross-cutting invariants across `HeylookCore`, `HeylookEmbeddings`, `HeylookEngines`, `HeylookServer`, and `HeylookUI` |

---

## Project Status, User Guide & Vision

- **[VISION.md](../../VISION.md)**: Core design principles (*Visibility*, *Derive, Don't Copy*, *Truthful Runtime*, *Heterogeneous Apple Silicon Co-Residency*). Breaks ties when a design choice is unclear.
- **[docs/USER_GUIDE.md](../USER_GUIDE.md)**: Complete operator guide for the 6-surface SwiftUI Workbench, `heylook.toml` configuration, `heylook-cli`, and `/v1/*` HTTP/SSE endpoints.
- **[docs/project/CURRENT.md](../project/CURRENT.md)**: Live readiness matrix of every target and subsystem.
- **[docs/project/TODO.md](../project/TODO.md)**: Completed roadmap phases (Phases 0–14) and next hardware/mobile frontiers.
