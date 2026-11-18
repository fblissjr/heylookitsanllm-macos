import Foundation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines

/// Single actor-isolated workbench orchestration service unifying inference, model lifecycle,
/// conversations, notebooks, presets, config, telemetry, and embeddings for BOTH the HTTP/SSE API
/// and native `HeylookUI` surfaces.
public actor WorkbenchService: WorkbenchBackend {
    public let store: SQLiteStore
    public let scheduler: ComputeScheduler
    public let registry: RequestRegistry
    public let orchestrator: ConversationOrchestrator
    public let isReadOnlyConfig: Bool

    private var defaultEmbeddingEngine: any EmbeddingEngine
    private var hasCustomEmbeddingEngine: Bool
    private var cachedEmbeddingEnginesByModel: [String: any EmbeddingEngine] = [:]
    private var loadingModelIDs: Set<String> = []
    private var loadedTemplateTexts: [String: String] = [:]

    private var scanIntervalSeconds: Int = 900
    private var memoryOnlySettings: [String: JSONValue] = [:]
    private var memoryOnlyKeys: Set<String> = []
    private var cachedOperationalSettings: OperationalSettings

    private let startedAtUnixMs: Int64

    public init(
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        registry: RequestRegistry = .shared,
        embeddingEngine: (any EmbeddingEngine)? = nil,
        isReadOnlyConfig: Bool = HeylookConfig.isReadOnlyModelConfig
    ) {
        self.store = store
        self.scheduler = scheduler
        self.registry = registry
        self.orchestrator = ConversationOrchestrator(
            store: store,
            scheduler: scheduler,
            registry: registry
        )
        self.isReadOnlyConfig = isReadOnlyConfig
        self.hasCustomEmbeddingEngine = embeddingEngine != nil
        self.defaultEmbeddingEngine = embeddingEngine ?? NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
            computeOptions: .hybridANETowers
        )
        self.startedAtUnixMs = SQLiteStore.nowMillis()
        self.cachedOperationalSettings = Self.readStoredOperationalSettings().effective
        ObservabilityDiskLog.shared.apply(settings: self.cachedOperationalSettings)
    }

    public func setEmbeddingEngine(_ engine: any EmbeddingEngine) {
        self.defaultEmbeddingEngine = engine
        self.hasCustomEmbeddingEngine = true
        self.cachedEmbeddingEnginesByModel[engine.modelID] = engine
    }

    public func currentEmbeddingEngine() -> any EmbeddingEngine {
        defaultEmbeddingEngine
    }

    public func unloadAllForShutdown() async {
        await scheduler.unloadAllModels()
        await defaultEmbeddingEngine.unload()
        for (_, engine) in cachedEmbeddingEnginesByModel {
            await engine.unload()
        }
        cachedEmbeddingEnginesByModel.removeAll()
        loadedTemplateTexts.removeAll()
    }

    // MARK: - Public & Admin Models

    public func listPublicModels() async -> [PublicModelWireRow] {
        let models = await scheduler.listModels()
        var rows: [PublicModelWireRow] = []
        rows.reserveCapacity(models.count)
        for desc in models {
            let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
            let sidecar = HeylookConfig.loadSidecar(in: dirURL)
            let base = HeylookConfig.buildPublicWireRow(for: desc, sidecar: sidecar)
            let enrichedContract = await enrichEngineContract(for: desc, base: base.engine, dirURL: dirURL, sidecar: sidecar)
            rows.append(
                PublicModelWireRow(
                    id: base.id,
                    object: base.object,
                    ownedBy: base.ownedBy,
                    provider: base.provider,
                    modalities: base.modalities,
                    capabilities: base.capabilities,
                    thinkingDefault: base.thinkingDefault,
                    samplerDefaults: base.samplerDefaults,
                    samplerSources: base.samplerSources,
                    engine: enrichedContract
                )
            )
        }
        return rows
    }

    public func listAdminModels() async -> [AdminModelWireRow] {
        let models = await scheduler.listModels()
        let config = HeylookConfig.loadDefault()
        let writtenIDs = Set(config.explicitModels.map(\.id))
        var rows: [AdminModelWireRow] = []
        rows.reserveCapacity(models.count)
        for desc in models {
            let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
            let sidecar = HeylookConfig.loadSidecar(in: dirURL)
            let isFromConfig = writtenIDs.contains(desc.modelID) || sidecar != nil
            let loadedSnap = desc.isLoaded ? await scheduler.loadedConfigSnapshot(for: desc.modelID) : nil
            let base = HeylookConfig.buildAdminWireRow(
                for: desc,
                sidecar: sidecar,
                isFromConfig: isFromConfig,
                loadedSnapshot: loadedSnap,
                loadedTemplateText: desc.isLoaded ? loadedTemplateTexts[desc.modelID] : nil
            )
            let enrichedContract = await enrichEngineContract(for: desc, base: base.engine, dirURL: dirURL, sidecar: sidecar)
            rows.append(
                AdminModelWireRow(
                    id: base.id,
                    provider: base.provider,
                    description: base.description,
                    tags: base.tags,
                    capabilities: base.capabilities,
                    config: base.config,
                    loaded: base.loaded,
                    source: base.source,
                    staleReloadFields: base.staleReloadFields,
                    engine: enrichedContract,
                    thinkingDefault: base.thinkingDefault,
                    samplerDefaults: base.samplerDefaults,
                    samplerSources: base.samplerSources
                )
            )
        }
        return rows
    }

    public func adminModel(id: String) async throws -> AdminModelWireRow {
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.notFound("Model '\(id)' not found")
        }
        let config = HeylookConfig.loadDefault()
        let writtenIDs = Set(config.explicitModels.map(\.id))
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        let sidecar = HeylookConfig.loadSidecar(in: dirURL)
        let loadedSnap = desc.isLoaded ? await scheduler.loadedConfigSnapshot(for: id) : nil
        let base = HeylookConfig.buildAdminWireRow(
            for: desc,
            sidecar: sidecar,
            isFromConfig: writtenIDs.contains(id) || sidecar != nil,
            loadedSnapshot: loadedSnap,
            loadedTemplateText: desc.isLoaded ? loadedTemplateTexts[id] : nil
        )
        let enrichedContract = await enrichEngineContract(for: desc, base: base.engine, dirURL: dirURL, sidecar: sidecar)
        return AdminModelWireRow(
            id: base.id,
            provider: base.provider,
            description: base.description,
            tags: base.tags,
            capabilities: base.capabilities,
            config: base.config,
            loaded: base.loaded,
            source: base.source,
            staleReloadFields: base.staleReloadFields,
            engine: enrichedContract,
            thinkingDefault: base.thinkingDefault,
            samplerDefaults: base.samplerDefaults,
            samplerSources: base.samplerSources
        )
    }

    private func enrichEngineContract(
        for desc: EngineDescription,
        base: EngineContract,
        dirURL: URL,
        sidecar: ModelSidecarConfig?
    ) async -> EngineContract {
        var context = base.context
        var speculative = base.speculative
        if desc.isLoaded, let prov = try? await scheduler.provider(for: desc.modelID) {
            let observed = await prov.observedContractState()
            if let runningCtx = observed.runningContextLength {
                context = EngineContextFacts(
                    length: base.context.length,
                    running: Fact<Int>(value: runningCtx, provenance: .observed, source: "running_provider")
                )
            }
            if let specInForce = observed.speculativeInForce {
                speculative = EngineSpeculativeFacts(
                    drafter: base.speculative.drafter,
                    type: base.speculative.type,
                    inForce: Fact<Bool>(value: specInForce, provenance: .observed, source: "running_provider")
                )
            }
        }

        let embFacts: EngineEmbeddingFacts?
        if desc.supportsEmbeddings.value {
            let mmproj = sidecar?.extraConfig["mmproj_path"]?.stringValue.map { URL(fileURLWithPath: $0) }
            embFacts = EmbeddingModelFacts.deriveFacts(
                provider: desc.provider,
                modelPath: URL(fileURLWithPath: desc.modelPath),
                mmprojPath: mmproj,
                computePlacement: desc.computePlacement.value.rawValue,
                ctxSize: sidecar?.ctxSize
            ) ?? base.embedding
        } else {
            embFacts = base.embedding
        }

        return EngineContract(
            runtime: base.runtime,
            context: context,
            template: base.template,
            settings: base.settings,
            cache: base.cache,
            thinking: base.thinking,
            speculative: speculative,
            decoding: base.decoding,
            embedding: embFacts,
            image: base.image,
            steering: base.steering
        )
    }

    public func loadModel(id: String, warm: Bool = false) async throws -> ModelLoadResult {
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.badRequest("Model '\(id)' not found or not enabled")
        }
        loadingModelIDs.insert(id)
        defer { loadingModelIDs.remove(id) }

        do {
            _ = try await scheduler.activateModel(id, warm: true)
        } catch let schedErr as ComputeSchedulerError {
            switch schedErr {
            case .modelNotFound:
                throw WorkbenchError.badRequest("Model '\(id)' not found or not enabled")
            case .queueOverloaded(_, let maxDepth):
                throw WorkbenchError.modelOverloaded(
                    message: RequestMiddleware.queueFullMessage,
                    queueCapacity: maxDepth
                )
            case .modelBusy:
                throw WorkbenchError.conflict(
                    code: "generation_in_progress",
                    message: schedErr.localizedDescription
                )
            }
        } catch {
            throw error
        }

        // Capture exact template text snapshot at load time (`ComputeScheduler` stores config snapshot)
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        if let prov = try? await scheduler.provider(for: id) {
            let observed = await prov.observedContractState()
            if let tpl = observed.loadedTemplateText {
                loadedTemplateTexts[id] = tpl
            } else {
                let tplState = HeylookConfig.inspectChatTemplate(in: dirURL)
                loadedTemplateTexts[id] = tplState.template
            }
        }

        guard warm else {
            return ModelLoadResult(status: "loaded", modelID: id)
        }

        let warmStartedAt = Date()
        do {
            if desc.supportsEmbeddings.value {
                let resolved = try resolveEmbeddingEngine(for: desc)
                if resolved.usesGPU {
                    _ = try await scheduler.withGPUEmbeddingLease {
                        try await resolved.engine.embedResult([.text("hi", promptName: nil)], dimensions: nil)
                    }
                } else {
                    _ = try await resolved.engine.embedResult([.text("hi", promptName: nil)], dimensions: nil)
                }
            } else {
                try await scheduler.withComputeLease(for: id) {
                    let provider = try await scheduler.provider(for: id)
                    try await provider.warmUp1Token()
                }
            }
            let elapsedMs = Int(max(0, Date().timeIntervalSince(warmStartedAt) * 1000.0))
            return ModelLoadResult(
                status: "loaded",
                modelID: id,
                warmed: true,
                warmMs: elapsedMs,
                warmError: nil
            )
        } catch {
            let detail = String(error.localizedDescription.prefix(500))
            return ModelLoadResult(
                status: "loaded",
                modelID: id,
                warmed: false,
                warmMs: nil,
                warmError: detail
            )
        }
    }

    public func reloadModel(
        id: String,
        loadSettings: [String: JSONValue]?,
        warm: Bool = false
    ) async throws -> ModelLoadResult {
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.badRequest("Model '\(id)' not found or not enabled")
        }

        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        if let loadSettings, !loadSettings.isEmpty {
            let parsed = ProviderConfigSchema.parseReloadSettings(
                provider: desc.provider,
                modelID: id,
                settings: loadSettings
            )
            let canonicalUpdates: [String: JSONValue]
            switch parsed {
            case .success(let dict):
                canonicalUpdates = dict
            case .failure(let schemaErr):
                throw WorkbenchError.badRequest(schemaErr.message)
            }

            let currentExplicit = HeylookConfig.loadSidecar(in: dirURL)?.toExplicitJSONDictionary() ?? [:]
            var diffUpdates: [String: JSONValue] = [:]
            for (k, v) in canonicalUpdates {
                let existingVal = currentExplicit[k]
                if v == .null {
                    if existingVal != nil {
                        diffUpdates[k] = .null
                    }
                } else if existingVal != v {
                    diffUpdates[k] = v
                }
            }

            if !diffUpdates.isEmpty {
                if isReadOnlyConfig {
                    throw WorkbenchError.conflict(
                        code: "readonly_config",
                        message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
                    )
                }
                _ = try HeylookConfig.applySidecarPatch(diffUpdates, in: dirURL)
                let updatedSidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig()
                _ = try await scheduler.applySidecarUpdate(updatedSidecar, for: id)
            } else if desc.isLoaded {
                let sidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig()
                let stale = await scheduler.staleReloadFields(for: id, updatedSidecar: sidecar)
                if stale.isEmpty && !loadingModelIDs.contains(id) {
                    return try await loadModel(id: id, warm: warm)
                }
            }
        }

        if loadingModelIDs.contains(id) {
            throw WorkbenchError.conflict(
                code: "load_in_progress",
                message: "A load of '\(id)' is already in flight; reload after it completes."
            )
        }

        if await isModelGenerating(id) {
            throw WorkbenchError.conflict(
                code: "generation_in_progress",
                message: "Cannot reload model '\(id)' while generation is in progress; stop or wait for completion."
            )
        }

        loadedTemplateTexts.removeValue(forKey: id)
        try await scheduler.tryUnloadModel(id)

        // Rebuild provider instance with fresh disk/sidecar config if updated
        if let latestDesc = await scheduler.modelDescription(for: id) {
            let freshSidecar = HeylookConfig.loadSidecar(in: dirURL)
            let updatedDesc = freshSidecar.map { awaitSafeSidecarDescription(latestDesc, sidecar: $0) } ?? latestDesc
            let freshProvider = ProviderFactory.makeProvider(for: updatedDesc)
            await scheduler.register(provider: freshProvider, description: updatedDesc)
        }

        return try await loadModel(id: id, warm: warm)
    }

    private func awaitSafeSidecarDescription(_ existing: EngineDescription, sidecar: ModelSidecarConfig) -> EngineDescription {
        EngineDescription(
            modelID: existing.modelID,
            provider: existing.provider,
            modelPath: existing.modelPath,
            architecture: existing.architecture,
            quantization: existing.quantization,
            computePlacement: sidecar.computePlacement.map {
                ProvenanceField(value: $0, provenance: .configured, effect: .reload, computePlacement: $0)
            } ?? existing.computePlacement,
            isPinned: sidecar.pinned ?? existing.isPinned,
            isLoaded: existing.isLoaded,
            contextSize: sidecar.ctxSize.map {
                ProvenanceField(value: $0, provenance: .configured, effect: .reload)
            } ?? existing.contextSize,
            flashAttention: sidecar.flashAttn.map {
                ProvenanceField(value: $0, provenance: .configured, effect: .reload)
            } ?? existing.flashAttention,
            supportsVision: sidecar.supportsVision.map {
                ProvenanceField(value: $0, provenance: .configured, effect: .reload)
            } ?? existing.supportsVision,
            supportsAudio: existing.supportsAudio,
            supportsEmbeddings: existing.supportsEmbeddings,
            supportsThinking: sidecar.supportsThinking.map {
                ProvenanceField(value: $0, provenance: .configured, effect: .perRequest)
            } ?? existing.supportsThinking,
            thinkingBudget: existing.thinkingBudget,
            reasoningFormat: existing.reasoningFormat,
            thinkingControls: existing.thinkingControls,
            templateSource: existing.templateSource,
            speculativeDrafter: existing.speculativeDrafter,
            estimatedMemoryMB: existing.estimatedMemoryMB,
            tags: sidecar.tags.isEmpty ? existing.tags : sidecar.tags
        )
    }

    public func unloadModel(id: String) async throws -> ModelUnloadResult {
        if await isModelGenerating(id) {
            throw WorkbenchError.conflict(
                code: "generation_in_progress",
                message: "Cannot unload model '\(id)' while generation is in progress; abort generation first."
            )
        }
        guard let desc = await scheduler.modelDescription(for: id) else {
            return ModelUnloadResult(status: "not_loaded", modelID: id)
        }
        let wasLoaded = desc.isLoaded
        loadedTemplateTexts.removeValue(forKey: id)
        try await scheduler.tryUnloadModel(id)
        return ModelUnloadResult(
            status: wasLoaded ? "unloaded" : "not_loaded",
            modelID: id
        )
    }

    public func modelStatus(id: String) async -> [String: JSONValue] {
        guard let desc = await scheduler.modelDescription(for: id), desc.isLoaded else {
            return ["loaded": .bool(false)]
        }

        let metrics = await scheduler.metrics()
        let activeCount = registry.activeRequests().filter { $0.modelID == id }.count
        let snap = HardwareMemoryMonitor.snapshot()

        var out: [String: JSONValue] = [
            "loaded": .bool(true),
            "requests_active": .int(activeCount),
            "requests_queued": .int(metrics.gpuQueueDepth),
        ]

        var contextUsed: Int? = nil
        var contextCapacity: Int? = desc.contextSize.value
        if let provider = try? await scheduler.provider(for: id) {
            let observed = await provider.observedContractState()
            if let cap = observed.runningContextLength {
                contextCapacity = cap
            }
            if let used = observed.contextUsed {
                contextUsed = used
            }
        }
        if let contextUsed {
            out["context_used"] = .int(contextUsed)
        }
        if let contextCapacity {
            out["context_capacity"] = .int(contextCapacity)
        }

        if desc.provider == .mlx {
            out["memory_mb"] = .double(Double(snap.metalCurrentAllocatedMB))
            out["memory_source"] = .string("mlx_metal_active")
        } else if desc.estimatedMemoryMB > 0 {
            out["memory_mb"] = .double(Double(desc.estimatedMemoryMB))
            out["memory_source"] = .string(desc.provider == .gguf ? "gguf_weights_estimate" : "coreai_estimate")
        }

        return out
    }

    public func assessModelFit(
        id: String,
        configOverrides: [String: JSONValue],
        headroomGB: Double
    ) async throws -> RAMFitEvaluator.FitReport {
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.notFound("Model '\(id)' not found")
        }
        return try RAMFitEvaluator.evaluate(
            description: desc,
            configOverrides: configOverrides,
            headroomGB: headroomGB
        )
    }

    public func planImages(id: String, sizes: [[Int]]) async throws -> ImagePlanWireResponse {
        guard let desc = await scheduler.modelDescription(for: id), desc.isLoaded else {
            throw WorkbenchError.conflict(
                code: "model_not_loaded",
                message: "\(id) is not loaded -- load it to plan image sizes; planning never loads a model"
            )
        }
        try ModelSpecializationRegistry.validateImagePlanSizes(sizes)
        guard desc.supportsVision.value else {
            throw WorkbenchError.badRequest("\(id) is served as text; it takes no images")
        }

        let provider = try await scheduler.provider(for: id)
        do {
            let result = try await provider.planImages(sizes: sizes)
            return ImagePlanWireResponse(
                modelID: id,
                engine: result.engine,
                images: result.images,
                source: result.source
            )
        } catch let err as ProviderExecutionError {
            switch err {
            case .invalidRequest:
                throw WorkbenchError.badRequest(err.localizedDescription)
            case .runtimeUnavailable:
                throw WorkbenchError.serviceUnavailable(err.localizedDescription)
            default:
                throw err
            }
        }
    }

    // MARK: - Chat Templates & Model Config Admin

    public func getChatTemplate(id: String) async throws -> HeylookConfig.ChatTemplateState {
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.notFound("Model '\(id)' not found")
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        let fallbackSource = desc.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json"
        let loadedTpl = desc.isLoaded ? loadedTemplateTexts[id] : nil
        return HeylookConfig.inspectChatTemplate(
            in: dirURL,
            fallbackSource: fallbackSource,
            loadedTemplateText: loadedTpl
        )
    }

    public func setChatTemplate(id: String, template: String) async throws -> HeylookConfig.ChatTemplateState {
        if isReadOnlyConfig {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
            )
        }
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.notFound("Model '\(id)' not found")
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        do {
            let state = try HeylookConfig.writeChatTemplateOverride(template, in: dirURL)
            let updatedSidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig()
            _ = try? await scheduler.applySidecarUpdate(updatedSidecar, for: id)
            let loadedTpl = desc.isLoaded ? loadedTemplateTexts[id] : nil
            return HeylookConfig.inspectChatTemplate(
                in: dirURL,
                fallbackSource: desc.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json",
                loadedTemplateText: loadedTpl,
                refusedShapes: state.refusedShapes
            )
        } catch let jinjaErr as JinjaTemplateError {
            throw WorkbenchError.badRequest(jinjaErr.localizedDescription)
        }
    }

    public func deleteChatTemplate(id: String) async throws -> HeylookConfig.ChatTemplateState {
        if isReadOnlyConfig {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
            )
        }
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.notFound("Model '\(id)' not found")
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        guard HeylookConfig.hasChatTemplateOverride(in: dirURL) else {
            throw WorkbenchError.notFound("Model '\(id)' has no chat template override to remove")
        }
        let fallbackSource = desc.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json"
        _ = HeylookConfig.deleteChatTemplateOverride(in: dirURL, fallbackSource: fallbackSource)
        let updatedSidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig()
        _ = try? await scheduler.applySidecarUpdate(updatedSidecar, for: id)
        let loadedTpl = desc.isLoaded ? loadedTemplateTexts[id] : nil
        return HeylookConfig.inspectChatTemplate(
            in: dirURL,
            fallbackSource: fallbackSource,
            loadedTemplateText: loadedTpl
        )
    }

    public func patchAdminModel(id: String, updates: ModelConfigPatchPayload) async throws -> AdminModelPatchResult {
        guard let desc = await scheduler.modelDescription(for: id) else {
            throw WorkbenchError.notFound("Model '\(id)' not found")
        }
        if isReadOnlyConfig {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
            )
        }

        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        var reloadFields: [String] = []
        var patchMap = updates.config ?? [:]
        if let tags = updates.tags {
            patchMap["tags"] = .array(tags.map { .string($0) })
        }

        if !patchMap.isEmpty {
            let validation = ProviderConfigSchema.validatePatch(provider: desc.provider, patch: patchMap)
            switch validation {
            case .failure(let schemaErr):
                if schemaErr.statusCode == 422 {
                    throw WorkbenchError.unprocessable(message: schemaErr.message, field: schemaErr.field)
                } else {
                    throw WorkbenchError.badRequest(schemaErr.message)
                }
            case .success(let validResult):
                reloadFields = validResult.reloadRequiredFields
                _ = try HeylookConfig.applySidecarPatch(validResult.canonicalPatch, in: dirURL)
                let updatedSidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig()
                _ = try await scheduler.applySidecarUpdate(updatedSidecar, for: id)
            }
        }

        let updatedRow = try await adminModel(id: id)
        return AdminModelPatchResult(model: updatedRow, reloadRequiredFields: reloadFields, warning: nil)
    }

    public func addAdminModel(_ payload: [String: JSONValue]) async throws -> AdminModelWireRow {
        if isReadOnlyConfig {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
            )
        }
        guard case .string(let rawID)? = payload["id"],
              !rawID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkbenchError.badRequest("Model 'id' is required")
        }
        let modelID = rawID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard case .string(let rawProv)? = payload["provider"],
              let prov = EngineProviderKind(rawValue: rawProv.lowercased()) else {
            throw WorkbenchError.badRequest("Valid 'provider' (mlx, gguf, coreai) is required")
        }
        guard case .object(let cfgDict)? = payload["config"],
              case .string(let modelPath)? = cfgDict["model_path"],
              !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkbenchError.badRequest("'config.model_path' is required")
        }

        let targetURL = HeylookConfig.activeConfigURL()
        let fm = FileManager.default
        var text = (try? String(contentsOf: targetURL, encoding: .utf8)) ?? ""
        if !text.isEmpty && !text.hasSuffix("\n") { text += "\n" }
        text += "\n[[models]]\n"
        text += "id = \(TOMLCodec.formatValue(.string(modelID)))\n"
        text += "provider = \(TOMLCodec.formatValue(.string(prov.rawValue)))\n"
        text += "model_path = \(TOMLCodec.formatValue(.string(modelPath)))\n"
        try fm.createDirectory(at: targetURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: targetURL, atomically: true, encoding: .utf8)

        let discovered = HeylookConfig.discoverConfiguredModels()
        await scheduler.reloadRegistry(discovered)
        return try await adminModel(id: modelID)
    }

    public func deleteAdminModel(id: String) async throws -> [String: JSONValue] {
        if isReadOnlyConfig {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
            )
        }
        if await isModelGenerating(id) {
            throw WorkbenchError.conflict(
                code: "generation_in_progress",
                message: "Cannot remove model '\(id)' while generation is in progress"
            )
        }

        _ = try? await unloadModel(id: id)

        let targetURL = HeylookConfig.activeConfigURL()
        var removedFromTOML = false
        if let rawText = try? String(contentsOf: targetURL, encoding: .utf8) {
            let (newText, removed) = Self.removeExplicitModelBlock(id: id, fromTOML: rawText)
            if removed {
                try newText.write(to: targetURL, atomically: true, encoding: .utf8)
                removedFromTOML = true
            }
        }

        if let desc = await scheduler.modelDescription(for: id) {
            let sidecarURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
                .appendingPathComponent("model.heylook.toml")
            if FileManager.default.fileExists(atPath: sidecarURL.path) {
                try? FileManager.default.removeItem(at: sidecarURL)
                removedFromTOML = true
            }
        }

        guard removedFromTOML else {
            throw WorkbenchError.notFound("Model '\(id)' has no removable configuration entry")
        }

        let discovered = HeylookConfig.discoverConfiguredModels()
        await scheduler.reloadRegistry(discovered)
        return [
            "status": .string("removed"),
            "model_id": .string(id),
        ]
    }

    public func validateModelConfig(_ payload: [String: JSONValue]) async -> [String: JSONValue] {
        var errors: [String] = []
        var warnings: [String] = []

        let id = payload["id"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let provRaw = payload["provider"]?.stringValue?.lowercased() ?? ""
        if id.isEmpty {
            errors.append("Missing non-empty 'id'")
        }
        guard let prov = EngineProviderKind(rawValue: provRaw) else {
            errors.append("Invalid 'provider': '\(provRaw)'")
            return [
                "valid": .bool(false),
                "errors": .array(errors.map { .string($0) }),
                "warnings": .array([]),
            ]
        }
        guard case .object(let cfg)? = payload["config"] else {
            errors.append("Missing object 'config'")
            return [
                "valid": .bool(false),
                "errors": .array(errors.map { .string($0) }),
                "warnings": .array([]),
            ]
        }

        if let pathVal = cfg["model_path"]?.stringValue, !pathVal.isEmpty {
            let expanded = (pathVal as NSString).expandingTildeInPath
            if !FileManager.default.fileExists(atPath: expanded) {
                warnings.append("model_path does not exist yet: \(pathVal)")
            }
        } else {
            errors.append("config.model_path is required")
        }

        let schema = ProviderConfigSchema.schema(for: prov)
        for (k, v) in cfg {
            if let desc = schema.descriptor(named: k) {
                if case .failure(let err) = ProviderConfigSchema.validatePatch(provider: prov, patch: [desc.name: v]) {
                    if !desc.isFileOnly {
                        errors.append(err.message)
                    }
                }
            }
        }

        return [
            "valid": .bool(errors.isEmpty),
            "errors": .array(errors.map { .string($0) }),
            "warnings": .array(warnings.map { .string($0) }),
        ]
    }

    public func getScanConfig() async -> ScanConfigPayload {
        let cfg = HeylookConfig.loadDefault()
        let served = await scheduler.listModels().count
        return ScanConfigPayload(
            folders: cfg.scanFolders,
            scanIntervalSeconds: scanIntervalSeconds,
            modelsServed: served
        )
    }

    public func updateScanConfig(folders: [String]?, scanIntervalSeconds: Int?) async throws -> ScanConfigPayload {
        if isReadOnlyConfig {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG)"
            )
        }
        if let interval = scanIntervalSeconds {
            guard interval >= 0 else {
                throw WorkbenchError.badRequest("scan_interval_seconds must be >= 0")
            }
            self.scanIntervalSeconds = interval
        }

        let targetURL = HeylookConfig.activeConfigURL()
        var text = (try? String(contentsOf: targetURL, encoding: .utf8)) ?? "# heylook.toml\n"
        var dedupedFolders: [String]? = nil
        if let folders {
            var seen = Set<String>()
            dedupedFolders = folders.compactMap { raw in
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, seen.insert(trimmed).inserted else { return nil }
                return trimmed
            }
        }
        text = TOMLCodec.updateScanConfig(
            in: text,
            folders: dedupedFolders,
            autoDiscover: scanIntervalSeconds.map { $0 > 0 }
        )
        if let interval = scanIntervalSeconds {
            text = TOMLCodec.updateKey(in: text, section: "scan", key: "scan_interval_seconds", value: .int(interval))
        }
        try FileManager.default.createDirectory(at: targetURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: targetURL, atomically: true, encoding: .utf8)

        let discovered = HeylookConfig.discoverConfiguredModels()
        await scheduler.reloadRegistry(discovered)
        let cfg = HeylookConfig.loadDefault()
        let served = await scheduler.listModels().count
        return ScanConfigPayload(
            folders: cfg.scanFolders,
            scanIntervalSeconds: self.scanIntervalSeconds,
            modelsServed: served
        )
    }

    public func reloadAllModels() async throws -> [String: JSONValue] {
        loadedTemplateTexts.removeAll()
        await scheduler.unloadAllModels()
        let discovered = HeylookConfig.discoverConfiguredModels()
        await scheduler.reloadRegistry(discovered)
        let available = await scheduler.listModels().map(\.modelID)
        return [
            "status": .string("success"),
            "message": .string("Model configuration reloaded"),
            "cache_cleared": .bool(true),
            "models_available": .array(available.map { .string($0) }),
        ]
    }

    // MARK: - Stateless Messages API

    public func createMessageResponse(
        request: MessagesRequest,
        requestID: String,
        cancellation: StreamCancellationSignal,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> MessagesResponse? {
        let modelID = try request.requireModel()
        guard let desc = await scheduler.modelDescription(for: modelID) else {
            throw WorkbenchError.badRequest("Model '\(modelID)' not found or not enabled")
        }

        // Strict stateless media capability gate (422)
        if Self.requestContainsMedia(request, kind: "image") && !desc.supportsVision.value {
            throw WorkbenchError.unprocessable(
                message: "Model '\(modelID)' does not support image inputs (supports_vision == false)"
            )
        }
        if Self.requestContainsMedia(request, kind: "audio") && !desc.supportsAudio.value {
            throw WorkbenchError.unprocessable(
                message: "Model '\(modelID)' does not support audio inputs (supports_audio == false)"
            )
        }

        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        let tplInfo = ModelTemplateInfo.inspect(modelDirectory: dirURL, modelID: desc.modelID)
        if let depth = request.thinkingDepth, !depth.isEmpty {
            if let controls = desc.thinkingControls {
                if let problem = controls.checkDepth(depth) {
                    throw WorkbenchError.badRequest(problem)
                }
            } else if let problem = tplInfo.thinkingControls.checkDepth(depth) {
                throw WorkbenchError.badRequest(problem)
            }
        }

        let requestSampler = SamplerSettings(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            repetitionPenalty: request.repetitionPenalty,
            presencePenalty: request.presencePenalty,
            maxTokens: request.maxTokens,
            seed: request.seed,
            thinking: request.thinking,
            thinkingDepth: request.thinkingDepth,
            thinkingBudget: request.thinkingBudget,
            stopSequences: request.stopSequences
        )
        let resolvedSampling = HeylookConfig.resolveEffectiveSampling(
            modelPath: desc.modelPath,
            preset: nil,
            request: requestSampler,
            thinkingCapable: desc.supportsThinking.value
        ).effective

        let reqStartedAt = Date()
        let execID = registry.register(requestID: requestID, conversationID: nil, modelID: modelID)
        defer { registry.finish(requestID: requestID, executionID: execID) }

        cancellation.onCancel { [registry] in
            registry.abort(requestID: requestID)
        }

        let streamMode = request.stream ?? false
        do {
            return try await scheduler.withComputeLease(for: modelID) { [registry, scheduler] in
                if cancellation.isCancelled || registry.state(for: requestID).isAborted {
                    throw CancellationError()
                }
                let provider = try await scheduler.provider(for: modelID)

                let stream = await PrefillProgressBridge.$current.withValue({ processed, total in
                    if streamMode, let onSSEFrame {
                        await onSSEFrame(HeylookProgressSSEEvent(processed: processed, total: total).sseFrame())
                    }
                }) {
                    await provider.generate(
                        requestID: requestID,
                        request: request,
                        effectiveSampling: resolvedSampling,
                        registry: registry
                    )
                }

                let outcome = await SSEWirePipeline.consumeStreamAndFormatSSE(
                    stream: stream,
                    modelID: modelID,
                    requestID: requestID,
                    reasoningFormat: tplInfo.reasoningFormat,
                    startsInThinking: false,
                    thinkingEnabled: resolvedSampling.thinking ?? desc.supportsThinking.value,
                    stopSequences: resolvedSampling.stopSequences ?? [],
                    requestStartedAt: reqStartedAt,
                    registry: registry,
                    onEvent: streamMode ? onSSEFrame : nil
                )

                if streamMode {
                    return nil
                }

                if let err = outcome.error {
                    throw WorkbenchError.internalError(err.message)
                }

                var blocks: [MessagesResponse.ContentBlock] = []
                if !outcome.thinking.isEmpty {
                    blocks.append(.thinking(outcome.thinking))
                }
                if !outcome.content.isEmpty || blocks.isEmpty {
                    blocks.append(.text(outcome.content))
                }
                let approxOut = max(
                    outcome.telemetry.completionTokens,
                    max(0, (outcome.thinking.count + outcome.content.count) / 4)
                )
                let usage = outcome.telemetry.usageCounts(fallbackOutputTokens: approxOut)
                let perf = outcome.telemetry.buildPerformance(
                    requestDurationMs: outcome.requestDurationMs,
                    generationDurationMs: outcome.generationDurationMs,
                    thinkingDurationMs: outcome.thinkingDurationMs,
                    contentDurationMs: outcome.contentDurationMs
                )
                return MessagesResponse(
                    id: outcome.messageID,
                    model: modelID,
                    content: blocks,
                    stopReason: outcome.stopReason,
                    stopSequence: outcome.stopSequence,
                    usage: usage,
                    performance: perf
                )
            }
        } catch let schedErr as ComputeSchedulerError {
            switch schedErr {
            case .queueOverloaded(_, let maxDepth):
                throw WorkbenchError.modelOverloaded(
                    message: RequestMiddleware.queueFullMessage,
                    queueCapacity: maxDepth
                )
            case .modelNotFound:
                throw WorkbenchError.badRequest("Model '\(modelID)' not found or not enabled")
            case .modelBusy:
                throw WorkbenchError.conflict(
                    code: "generation_in_progress",
                    message: schedErr.localizedDescription
                )
            }
        }
    }

    // MARK: - Conversation Generation & Voice

    public func previewPrompt(
        conversationID: String,
        request: PromptPreviewRequestPayload
    ) async throws -> PromptPreviewWireResponse {
        try await orchestrator.previewPrompt(conversationID: conversationID, request: request)
    }

    public func generateInConversation(
        conversationID: String,
        request: ConversationGenerateRequest,
        requestID: String,
        detachOnSubscriberExit: Bool = true,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> HeylookSavedEvent {
        try await orchestrator.generate(
            conversationID: conversationID,
            request: request,
            requestID: requestID,
            detachOnSubscriberExit: detachOnSubscriberExit,
            onSSEFrame: onSSEFrame
        )
    }

    public func generateInConversation(
        conversationID: String,
        request: ConversationGenerateRequest,
        requestID: String,
        cancellationSignal: StreamCancellationSignal,
        detachOnSubscriberExit: Bool = true,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> HeylookSavedEvent {
        try await orchestrator.generate(
            conversationID: conversationID,
            request: request,
            requestID: requestID,
            cancellationSignal: cancellationSignal,
            detachOnSubscriberExit: detachOnSubscriberExit,
            onSSEFrame: onSSEFrame
        )
    }

    private static func mapASRError(_ error: Error) -> Error {
        if let schedErr = error as? ComputeSchedulerError {
            switch schedErr {
            case .modelNotFound(let id):
                return WorkbenchError.notFound("ASR model '\(id)' not found")
            case .queueOverloaded(_, let maxDepth):
                return WorkbenchError.modelOverloaded(message: RequestMiddleware.queueFullMessage, queueCapacity: maxDepth)
            case .modelBusy(let id):
                return WorkbenchError.conflict(code: "generation_in_progress", message: "ASR model '\(id)' is busy")
            }
        }
        if let execErr = error as? ProviderExecutionError {
            switch execErr {
            case .invalidRequest:
                return WorkbenchError.badRequest(execErr.localizedDescription)
            case .checkpointNotFound, .binaryNotFound, .runtimeUnavailable:
                return WorkbenchError.serviceUnavailable(execErr.localizedDescription)
            }
        }
        return error
    }

    public func transcribeAudio(_ request: AudioTranscriptionRequest, requestID _: String) async throws -> [String: JSONValue] {
        guard let wavData = Data(base64Encoded: request.audioBase64), !wavData.isEmpty else {
            throw WorkbenchError.badRequest("audio_base64 must be non-empty valid base64 WAV audio")
        }
        do {
            let asr = try await scheduler.transcribeVoice(
                wavData: wavData,
                languageHint: request.languageHint ?? "en",
                preferredASRModelID: request.model
            )
            return [
                "text": .string(asr.text),
                "model": .string(asr.modelID),
                "backend": .string(asr.backend),
                "compute_placement": .string(asr.computePlacement.rawValue),
                "duration_seconds": .double(asr.durationSeconds),
            ]
        } catch {
            throw Self.mapASRError(error)
        }
    }

    public func voiceGenerate(
        conversationID: String,
        request: ConversationVoiceGenerateRequest,
        requestID: String,
        cancellationSignal: StreamCancellationSignal,
        onSSEFrame: (@Sendable (String) async -> Void)?
    ) async throws -> [String: JSONValue] {
        try enforceConversationNotGenerating(conversationID)
        guard let wavData = Data(base64Encoded: request.audioBase64), !wavData.isEmpty else {
            throw WorkbenchError.badRequest("audio_base64 must be non-empty valid base64 WAV audio")
        }
        guard (try? store.getConversationDetail(id: conversationID)) != nil else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        let asr: ANESpeechTranscriptionResult
        do {
            asr = try await scheduler.transcribeVoice(
                wavData: wavData,
                languageHint: request.languageHint ?? "en",
                preferredASRModelID: request.asrModel
            )
        } catch {
            throw Self.mapASRError(error)
        }

        let transcript = asr.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else {
            throw WorkbenchError.badRequest("Speech transcription returned empty transcript")
        }

        let genReq = ConversationGenerateRequest(
            model: request.model,
            content: transcript,
            mode: .append,
            sampling: request.sampling
        )
        let saved = try await orchestrator.generate(
            conversationID: conversationID,
            request: genReq,
            requestID: requestID,
            cancellationSignal: cancellationSignal,
            detachOnSubscriberExit: true,
            onSSEFrame: onSSEFrame
        )
        let encoder = JSONEncoder()
        let savedData = try encoder.encode(saved)
        let savedJSON = (try JSONSerialization.jsonObject(with: savedData) as? [String: Any]) ?? [:]
        var dict: [String: JSONValue] = [:]
        for (k, v) in savedJSON {
            if let jv = JSONValue.fromAny(v) {
                dict[k] = jv
            }
        }
        dict["transcript"] = .string(transcript)
        dict["asr_model"] = .string(asr.modelID)
        dict["asr_placement"] = .string(asr.computePlacement.rawValue)
        return dict
    }

    // MARK: - Embeddings

    public func embed(request: EmbeddingWireRequest, requestID: String) async throws -> EmbeddingWireResponse {
        let modelID = request.model.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelID.isEmpty else {
            throw WorkbenchError.badRequest("model is required")
        }
        guard !request.items.isEmpty && request.items.count <= EmbeddingRequestLimits.maxItems else {
            throw WorkbenchError.unprocessable(
                message: "input must contain 1...\(EmbeddingRequestLimits.maxItems) items"
            )
        }
        if let dims = request.dimensions, dims <= 0 {
            throw WorkbenchError.unprocessable(message: "dimensions must be > 0")
        }

        let resolved = try await resolveEmbeddingEngine(forModel: modelID)
        let engine = resolved.engine
        var embeddingInputs: [EmbeddingInput] = []
        embeddingInputs.reserveCapacity(request.items.count)

        var totalRequestMediaBytes = 0
        for (idx, item) in request.items.enumerated() {
            if let badMimeMsg = item.hasInvalidMediaMIME {
                throw WorkbenchError.unprocessable(message: badMimeMsg)
            }
            guard !item.content.isEmpty && item.content.count <= EmbeddingRequestLimits.maxBlocksPerItem else {
                throw WorkbenchError.unprocessable(
                    message: "input[\(idx)].content must contain 1...\(EmbeddingRequestLimits.maxBlocksPerItem) blocks"
                )
            }

            var parts: [EmbeddingPart] = []
            var itemWindowOpts: VideoWindowOptions? = nil
            for (bIdx, block) in item.content.enumerated() {
                switch block {
                case .text(let s):
                    guard s.count <= EmbeddingRequestLimits.maxTextChars else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(idx)] text exceeds \(EmbeddingRequestLimits.maxTextChars) characters"
                        )
                    }
                    parts.append(.text(s))
                case .thinking:
                    throw WorkbenchError.unprocessable(
                        message: "input[\(idx)]: thinking blocks are not valid in /v1/embeddings"
                    )
                case .media(let m):
                    guard let kind = ArtifactKind(rawValue: m.kind), kind.isMedia else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(idx)]: unsupported media type '\(m.kind)'"
                        )
                    }
                    let expectedPrefix = "\(m.kind)/"
                    guard m.mediaType.lowercased().hasPrefix(expectedPrefix) else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(idx)]: media_type '\(m.mediaType)' must match content block type '\(m.kind)'"
                        )
                    }
                    if let explicitWin = item.videoWindowsByBlockIndex[bIdx] {
                        guard m.kind == "video" else {
                            throw WorkbenchError.unprocessable(
                                message: "input[\(idx)]: windows applies to video blocks only"
                            )
                        }
                        guard item.content.count == 1 else {
                            throw WorkbenchError.unprocessable(
                                message: "input[\(idx)]: an item with video windows must contain only that video block"
                            )
                        }
                        itemWindowOpts = explicitWin
                    }
                    let rawBytes: Data
                    if !m.dataBase64.isEmpty {
                        guard let decoded = Data(base64Encoded: m.dataBase64) else {
                            throw WorkbenchError.badRequest("input[\(idx)]: invalid base64 media payload")
                        }
                        rawBytes = decoded
                    } else if let urlStr = m.url, let url = URL(string: urlStr),
                              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" {
                        let (fetched, resp) = try await URLSession.shared.data(from: url)
                        if let httpResp = resp as? HTTPURLResponse, !(200...299).contains(httpResp.statusCode) {
                            throw WorkbenchError.badRequest("input[\(idx)]: failed to fetch media URL (HTTP \(httpResp.statusCode))")
                        }
                        rawBytes = fetched
                    } else {
                        throw WorkbenchError.badRequest("input[\(idx)]: media source must provide base64 data or http(s) url")
                    }

                    let capBytes = (m.kind == "video")
                        ? EmbeddingRequestLimits.maxVideoBytes
                        : EmbeddingRequestLimits.maxImageOrAudioBytes
                    guard rawBytes.count <= capBytes else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(idx)]: \(m.kind) payload exceeds \(capBytes / (1024 * 1024)) MiB limit"
                        )
                    }
                    totalRequestMediaBytes += rawBytes.count
                    guard totalRequestMediaBytes <= EmbeddingRequestLimits.maxRequestMediaBytes else {
                        throw WorkbenchError.unprocessable(
                            message: "Total request media payload exceeds \(EmbeddingRequestLimits.maxRequestMediaBytes / (1024 * 1024)) MiB"
                        )
                    }
                    let sampling = itemWindowOpts?.sampling ?? .off
                    parts.append(.media(kind: kind, mimeType: m.mediaType, data: rawBytes, windowSampling: sampling))
                }
            }

            let embInput: EmbeddingInput
            if item.hasExplicitPromptName {
                embInput = EmbeddingInput(
                    parts: parts,
                    promptName: item.promptName,
                    title: nil,
                    videoWindows: itemWindowOpts
                )
            } else {
                embInput = EmbeddingInput(
                    parts: parts,
                    promptName: nil,
                    title: nil,
                    videoWindows: itemWindowOpts
                )
            }
            _ = try embInput.resolvedVideoWindows(itemIndex: idx)
            embeddingInputs.append(embInput)
        }

        let execID = registry.register(requestID: requestID, conversationID: nil, modelID: modelID)
        let startedAt = Date()
        let schedulerRef = self.scheduler
        let usesGPU = resolved.usesGPU
        let dims = request.dimensions
        let inputsCopy = embeddingInputs
        let runnerTask = Task<EmbeddingResult, Error> {
            try Task.checkCancellation()
            let res: EmbeddingResult
            if usesGPU {
                res = try await schedulerRef.withGPUEmbeddingLease {
                    try await engine.embedResult(inputsCopy, dimensions: dims)
                }
            } else {
                res = try await engine.embedResult(inputsCopy, dimensions: dims)
            }
            try Task.checkCancellation()
            return res
        }
        registry.addCancelHandler(for: requestID, executionID: execID) {
            runnerTask.cancel()
        }
        defer { registry.finish(requestID: requestID, executionID: execID) }

        let result: EmbeddingResult
        do {
            result = try await runnerTask.value
        } catch is CancellationError {
            throw WorkbenchError.conflict(code: "cancelled", message: "Embedding request '\(requestID)' was cancelled")
        }

        if registry.state(for: requestID).isAborted {
            throw WorkbenchError.conflict(code: "cancelled", message: "Embedding request '\(requestID)' was cancelled")
        }

        let elapsedMs = result.elapsedMS > 0
            ? result.elapsedMS
            : (Date().timeIntervalSince(startedAt) * 1000.0)

        let rows: [EmbeddingWireResponse.VectorRow] = result.outputs.enumerated().map { idx, out in
            let winRows: [EmbeddingWireResponse.WindowPayload]? = out.windows.isEmpty
                ? nil
                : out.windows.map {
                    EmbeddingWireResponse.WindowPayload(
                        start: $0.start,
                        end: $0.end,
                        frameTimes: $0.frameTimes,
                        embedding: $0.vector
                    )
                }
            return EmbeddingWireResponse.VectorRow(
                index: idx,
                embedding: out.vector,
                promptName: out.promptName ?? request.items[idx].promptName,
                windows: winRows
            )
        }

        return EmbeddingWireResponse(
            object: "list",
            model: modelID,
            dimensions: result.dimensions,
            data: rows,
            inputTokens: result.inputTokens,
            requestID: requestID,
            elapsedMs: (elapsedMs * 100.0).rounded() / 100.0
        )
    }

    private func resolveEmbeddingEngine(forModel modelID: String) async throws -> (engine: any EmbeddingEngine, usesGPU: Bool) {
        if let desc = await scheduler.modelDescription(for: modelID) {
            return try resolveEmbeddingEngine(for: desc)
        }

        let defaultAliases: Set<String> = [
            "embeddinggemma-2",
            "embeddinggemma",
            "embeddinggemma-300m",
            "mlx-community/embeddinggemma-300m",
            "google/embeddinggemma-300m",
            "text-embedding",
            "local",
        ]
        if (hasCustomEmbeddingEngine && modelID == defaultEmbeddingEngine.modelID)
            || defaultAliases.contains(modelID.lowercased()) {
            let usesGPU = !(defaultEmbeddingEngine is HybridLocalEmbeddingEngine)
            return (defaultEmbeddingEngine, usesGPU)
        }

        throw WorkbenchError.notFound("Embedding model '\(modelID)' not found")
    }

    private func resolveEmbeddingEngine(for desc: EngineDescription) throws -> (engine: any EmbeddingEngine, usesGPU: Bool) {
        guard desc.supportsEmbeddings.value else {
            throw WorkbenchError.badRequest("Model '\(desc.modelID)' does not support embeddings")
        }
        let usesGPU = desc.provider == .gguf || desc.provider == .mlx || !desc.computePlacement.value.runsOnNeuralEngineLane
        if let cached = cachedEmbeddingEnginesByModel[desc.modelID] {
            return (cached, usesGPU)
        }
        if hasCustomEmbeddingEngine && desc.modelID == defaultEmbeddingEngine.modelID {
            return (defaultEmbeddingEngine, usesGPU)
        }
        if desc.provider == .gguf {
            #if os(macOS)
            let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
            let sidecar = HeylookConfig.loadSidecar(in: dirURL)
            let mmprojPath = sidecar?.extraConfig["mmproj_path"]?.stringValue
            let ggufEngine = try GGUFEmbeddingEngine(
                modelPath: URL(fileURLWithPath: desc.modelPath),
                mmprojPath: mmprojPath.map { URL(fileURLWithPath: $0) },
                modelID: desc.modelID,
                ctxSize: sidecar?.ctxSize,
                flashAttention: sidecar?.flashAttn ?? desc.flashAttention.value
            )
            cachedEmbeddingEnginesByModel[desc.modelID] = ggufEngine
            return (ggufEngine, true)
            #else
            throw WorkbenchError.serviceUnavailable("GGUF out-of-process embeddings require macOS")
            #endif
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: desc.modelPath)
        let engine = NativeMLXEmbeddingGemma2.makeBestAvailableEngine(
            checkpointURL: dirURL,
            computeOptions: desc.computePlacement.value.runsOnNeuralEngineLane ? .hybridANETowers : .allMLXGPU
        )
        cachedEmbeddingEnginesByModel[desc.modelID] = engine
        return (engine, usesGPU)
    }

    // MARK: - Conversations / Messages / Media CRUD (with 409 lock guards)

    public func listConversations() throws -> ConversationListResponseDTO {
        let records = try store.listConversations()
        let summaries = records.map { rec in
            ConversationSummaryDTO(
                from: rec,
                generating: registry.isConversationGenerating(rec.id)
            )
        }
        return ConversationListResponseDTO(conversations: summaries)
    }

    public func createConversation(_ req: ConversationCreateRequestDTO) throws -> ConversationDetailDTO {
        let now = SQLiteStore.nowMillis()
        let conv = ConversationRecord(
            id: UUID().uuidString,
            title: req.title,
            systemPrompt: req.systemPrompt ?? "",
            defaultModelID: req.modelID,
            params: req.params.settings,
            appliedPresetID: req.appliedPresetID,
            createdUnixMs: now,
            updatedUnixMs: now
        )
        try store.upsertConversation(conv)
        return ConversationDetailDTO(from: conv, messages: [], generating: false)
    }

    public func getConversation(id: String) throws -> ConversationDetailDTO {
        guard let detail = try store.getConversationDetail(id: id) else {
            throw WorkbenchError.notFound("Conversation '\(id)' not found")
        }
        return ConversationDetailDTO(
            from: detail.conversation,
            messages: detail.messages,
            generating: registry.isConversationGenerating(id)
        )
    }

    public func updateConversation(id: String, req: ConversationUpdateRequestDTO) throws -> ConversationDetailDTO {
        guard var detail = try store.getConversationDetail(id: id) else {
            throw WorkbenchError.notFound("Conversation '\(id)' not found")
        }
        var conv = detail.conversation
        if req.presentKeys.contains("title"), let title = req.title {
            conv.title = title
        }
        if req.presentKeys.contains("model_id") {
            conv.defaultModelID = req.modelID
        }
        if req.presentKeys.contains("system_prompt") {
            conv.systemPrompt = req.systemPrompt ?? ""
        }
        if req.presentKeys.contains("params") {
            conv.params = req.params?.settings
        }
        if req.presentKeys.contains("applied_preset_id") {
            conv.appliedPresetID = req.appliedPresetID
        }
        conv.updatedUnixMs = SQLiteStore.nowMillis()
        try store.upsertConversation(conv)
        detail = try store.getConversationDetail(id: id) ?? (conv, [])
        return ConversationDetailDTO(
            from: detail.conversation,
            messages: detail.messages,
            generating: registry.isConversationGenerating(id)
        )
    }

    public func deleteConversation(id: String) throws {
        try enforceConversationNotGenerating(id)
        guard (try store.getConversationDetail(id: id)) != nil else {
            throw WorkbenchError.notFound("Conversation '\(id)' not found")
        }
        try store.deleteConversation(id: id)
    }

    public func cloneConversation(id: String, req: ConversationCloneRequestDTO) throws -> ConversationDetailDTO {
        try enforceConversationNotGenerating(id)
        let cloned = try store.cloneConversationWithMessages(
            id: id,
            upToPosition: req.upToPosition,
            newTitle: req.title
        )
        return ConversationDetailDTO(from: cloned.conversation, messages: cloned.messages, generating: false)
    }

    public func listMessages(conversationID: String) throws -> [WireMessageDTO] {
        guard let detail = try store.getConversationDetail(id: conversationID) else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        return detail.messages.map(WireMessageDTO.init(from:))
    }

    public func addMessage(conversationID: String, req: MessageCreateRequestDTO) throws -> WireMessageDTO {
        try enforceConversationNotGenerating(conversationID)
        guard let detail = try store.getConversationDetail(id: conversationID) else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        let nextPos = (detail.messages.last?.position ?? 0) + 1
        let msg = MessageRecord(
            id: UUID().uuidString,
            conversationID: conversationID,
            position: nextPos,
            role: req.role,
            content: req.content,
            thinking: req.thinking,
            modelID: nil,
            contentBlocks: req.contentBlocks
        )
        let saved = try store.appendMessage(msg)
        return WireMessageDTO(from: saved)
    }

    public func updateMessage(
        conversationID: String,
        messageID: String,
        req: MessageUpdateRequestDTO
    ) throws -> WireMessageDTO {
        try enforceConversationNotGenerating(conversationID)
        guard let detail = try store.getConversationDetail(id: conversationID) else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        guard var existing = detail.messages.first(where: { $0.id == messageID }) else {
            throw WorkbenchError.notFound("Message '\(messageID)' not found in conversation '\(conversationID)'")
        }
        if req.presentKeys.contains("content") {
            if let blocks = req.contentBlocks {
                existing.contentBlocks = blocks
            }
            if let content = req.content {
                existing.content = content
            } else if let blocks = req.contentBlocks {
                existing.content = blocks.compactMap { part in
                    if case .text(let s) = part { return s }
                    return nil
                }.joined(separator: " ")
            }
        }
        if req.presentKeys.contains("thinking") {
            existing.thinking = req.thinking
        }
        existing.updatedUnixMs = SQLiteStore.nowMillis()
        let updated = try store.updateMessage(existing)
        if req.truncateAfter {
            _ = try store.deleteMessagesAfter(conversationID: conversationID, afterPosition: existing.position)
        }
        return WireMessageDTO(from: updated)
    }

    public func deleteMessagesAfter(conversationID: String, afterPosition: Int) throws -> Int {
        try enforceConversationNotGenerating(conversationID)
        guard (try store.getConversationDetail(id: conversationID)) != nil else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        return try store.deleteMessagesAfter(conversationID: conversationID, afterPosition: afterPosition)
    }

    public func deleteMessage(conversationID: String, messageID: String) throws {
        try enforceConversationNotGenerating(conversationID)
        guard (try store.getConversationDetail(id: conversationID)) != nil else {
            throw WorkbenchError.notFound("Conversation '\(conversationID)' not found")
        }
        let deleted = try store.deleteMessage(conversationID: conversationID, messageID: messageID)
        if !deleted {
            throw WorkbenchError.notFound("Message '\(messageID)' not found in conversation '\(conversationID)'")
        }
    }

    public func mediaBlob(conversationID: String, mediaID: String) throws -> MediaBlobRecord {
        guard let blob = try store.getMediaBlob(conversationID: conversationID, mediaID: mediaID) else {
            throw WorkbenchError.notFound("Media blob '\(mediaID)' not found in conversation '\(conversationID)'")
        }
        return blob
    }

    private func enforceConversationNotGenerating(_ conversationID: String) throws {
        if registry.isConversationGenerating(conversationID) {
            throw WorkbenchError.conflict(
                code: "generation_in_progress",
                message: "Generation is currently in progress for conversation '\(conversationID)'"
            )
        }
    }

    // MARK: - Notebooks & Presets

    public func listNotebooks() throws -> NotebookListResponseDTO {
        let records = try store.listNotebooks()
        return NotebookListResponseDTO(notebooks: records.map(NotebookSummaryDTO.init(from:)))
    }

    public func createNotebook(_ req: NotebookCreateRequestDTO) throws -> NotebookDetailDTO {
        let now = SQLiteStore.nowMillis()
        let record = NotebookRecord(
            id: UUID().uuidString,
            title: req.title,
            body: req.content,
            systemPrompt: req.systemPrompt ?? "",
            modelID: req.modelID,
            params: req.params.settings,
            appliedPresetID: req.appliedPresetID,
            updatedUnixMs: now,
            createdUnixMs: now
        )
        try store.upsertNotebook(record)
        return NotebookDetailDTO(from: record)
    }

    public func getNotebook(id: String) throws -> NotebookDetailDTO {
        guard let record = try store.getNotebook(id: id) else {
            throw WorkbenchError.notFound("Notebook '\(id)' not found")
        }
        return NotebookDetailDTO(from: record)
    }

    public func updateNotebook(id: String, req: NotebookUpdateRequestDTO) throws -> NotebookDetailDTO {
        guard var existing = try store.getNotebook(id: id) else {
            throw WorkbenchError.notFound("Notebook '\(id)' not found")
        }
        if req.presentKeys.contains("title"), let title = req.title { existing.title = title }
        if req.presentKeys.contains("content"), let content = req.content { existing.body = content }
        if req.presentKeys.contains("model_id") { existing.modelID = req.modelID }
        if req.presentKeys.contains("system_prompt") { existing.systemPrompt = req.systemPrompt ?? "" }
        if req.presentKeys.contains("params") { existing.params = req.params?.settings }
        if req.presentKeys.contains("applied_preset_id") { existing.appliedPresetID = req.appliedPresetID }
        existing.updatedUnixMs = SQLiteStore.nowMillis()
        try store.upsertNotebook(existing)
        return NotebookDetailDTO(from: existing)
    }

    public func deleteNotebook(id: String) throws {
        guard (try store.getNotebook(id: id)) != nil else {
            throw WorkbenchError.notFound("Notebook '\(id)' not found")
        }
        try store.deleteNotebook(id: id)
    }

    public func listPresets() throws -> PresetListResponseDTO {
        let records = try store.listPresets()
        return PresetListResponseDTO(presets: records.map(PresetDTO.init(from:)))
    }

    public func createPreset(_ req: PresetCreateRequestDTO) throws -> PresetDTO {
        let now = SQLiteStore.nowMillis()
        let record = PresetRecord(
            id: UUID().uuidString,
            name: req.name,
            description: "",
            scope: "sampler",
            systemPrompt: req.systemPrompt,
            settings: req.params.settings,
            updatedUnixMs: now,
            createdUnixMs: now
        )
        let saved = try store.createPreset(record)
        return PresetDTO(from: saved)
    }

    public func getPreset(id: String) throws -> PresetDTO {
        guard let record = try store.getPreset(id: id) else {
            throw WorkbenchError.notFound("Preset '\(id)' not found")
        }
        return PresetDTO(from: record)
    }

    public func updatePreset(id: String, req: PresetUpdateRequestDTO) throws -> PresetDTO {
        guard var existing = try store.getPreset(id: id) else {
            throw WorkbenchError.notFound("Preset '\(id)' not found")
        }
        if req.presentKeys.contains("name"), let name = req.name { existing.name = name }
        if req.presentKeys.contains("system_prompt") { existing.systemPrompt = req.systemPrompt }
        if req.presentKeys.contains("params"), let params = req.params { existing.settings = params.settings }
        existing.updatedUnixMs = SQLiteStore.nowMillis()
        let saved = try store.updatePreset(id: id, existing)
        return PresetDTO(from: saved)
    }

    public func deletePreset(id: String) throws {
        guard (try store.getPreset(id: id)) != nil else {
            throw WorkbenchError.notFound("Preset '\(id)' not found")
        }
        try store.deletePreset(id: id)
    }

    // MARK: - Operational Config, Logs, Metrics, Capabilities, Cache & Data Clear

    public func getOperationalConfig() -> [String: JSONValue] {
        buildOperationalConfigSnapshot()
    }

    public func updateOperationalConfig(patchData: Data) throws -> [String: JSONValue] {
        let rawObj = try JSONSerialization.jsonObject(with: patchData)
        guard let rawDict = rawObj as? [String: Any] else {
            throw WorkbenchError.unprocessable(message: "Operational config body must be a JSON object")
        }

        // Strict decode against OperationalSettings
        let currentEffectiveData = try JSONEncoder().encode(buildOperationalConfigSnapshot()["effective"] ?? .object([:]))
        var mergedDict = (try JSONSerialization.jsonObject(with: currentEffectiveData) as? [String: Any]) ?? [:]
        for (k, v) in rawDict {
            mergedDict[k] = v
        }
        let mergedData = try JSONSerialization.data(withJSONObject: mergedDict)
        let validated = try JSONDecoder().decode(OperationalSettings.self, from: mergedData)

        let patchJSON = rawDict.reduce(into: [String: JSONValue]()) { acc, pair in
            if let jv = JSONValue.fromAny(pair.value) {
                acc[pair.key] = jv
            }
        }

        if isReadOnlyConfig {
            for (k, v) in patchJSON {
                if v == .null {
                    memoryOnlySettings.removeValue(forKey: k)
                } else {
                    memoryOnlySettings[k] = v
                }
                memoryOnlyKeys.insert(k)
            }
        } else {
            _ = try HeylookConfig.updateSettingsSection(patchJSON)
        }

        cachedOperationalSettings = validated
        ObservabilityDiskLog.shared.apply(settings: validated)
        return buildOperationalConfigSnapshot()
    }

    public func resetOperationalConfigKey(_ key: String) throws -> [String: JSONValue] {
        let validKeys: Set<String> = ["observability_level", "observability_retention_days", "mlx_cache_limit_gb"]
        guard validKeys.contains(key) else {
            throw WorkbenchError.notFound("Unknown operational setting '\(key)'")
        }
        if isReadOnlyConfig {
            memoryOnlySettings.removeValue(forKey: key)
            memoryOnlyKeys.remove(key)
        } else {
            _ = try HeylookConfig.updateSettingsSection([key: .null])
        }
        let snap = buildOperationalConfigSnapshot()
        if let effectiveJSON = snap["effective"],
           let encoded = try? JSONEncoder().encode(effectiveJSON),
           let decoded = try? JSONDecoder().decode(OperationalSettings.self, from: encoded) {
            cachedOperationalSettings = decoded
            ObservabilityDiskLog.shared.apply(settings: decoded)
        }
        return snap
    }

    public func readObservabilityLogs(streamRaw: String?, limit: Int?) throws -> ObservabilityLogsResponseDTO {
        let streamName = streamRaw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "events"
        let stream: ObservabilityStream
        do {
            stream = try ObservabilityStream.parse(streamName)
        } catch {
            throw WorkbenchError.badRequest(error.localizedDescription)
        }
        let boundedLimit = max(1, min(1000, limit ?? 100))
        let rows = ObservabilityDiskLog.shared.readTail(stream: stream, limit: boundedLimit)
        return ObservabilityLogsResponseDTO(
            stream: stream,
            level: ObservabilityDiskLog.shared.currentLevel(),
            records: rows
        )
    }

    public func systemMetrics() async -> [String: JSONValue] {
        let mem = HardwareMemoryMonitor.snapshot()
        let schedulerMetrics = await scheduler.metrics()
        let models = await scheduler.listModels()
        let activeReqs = registry.activeRequests()

        var modelsJSON: [String: JSONValue] = [:]
        for desc in models where desc.isLoaded {
            var cap = desc.contextSize.value
            var used: Int? = nil
            if let provider = try? await scheduler.provider(for: desc.modelID) {
                let observed = await provider.observedContractState()
                if let runningCap = observed.runningContextLength { cap = runningCap }
                if let u = observed.contextUsed { used = u }
            }
            var mDict: [String: JSONValue] = [
                "context_capacity": .int(cap),
                "requests_active": .int(activeReqs.filter { $0.modelID == desc.modelID }.count),
                "requests_queued": .int(schedulerMetrics.gpuQueueDepth),
            ]
            if let used {
                mDict["context_used"] = .int(used)
                if cap > 0 {
                    mDict["context_percent"] = .double((Double(used) / Double(cap) * 1000.0).rounded() / 10.0)
                }
            }
            if desc.provider == .mlx {
                mDict["memory_mb"] = .double(Double(mem.metalCurrentAllocatedMB))
                mDict["memory_source"] = .string("mlx_metal_active")
            } else if desc.estimatedMemoryMB > 0 {
                mDict["memory_mb"] = .double(Double(desc.estimatedMemoryMB))
                mDict["memory_source"] = .string(desc.provider == .gguf ? "gguf_weights_estimate" : "coreai_estimate")
            }
            modelsJSON[desc.modelID] = .object(mDict)
        }

        return [
            "timestamp": .string(WireTimestamp.iso8601(fromUnixMs: SQLiteStore.nowMillis())),
            "system": .object([
                "ram_used_gb": .double(mem.ramUsedGB),
                "ram_available_gb": .double(mem.ramAvailableGB),
                "ram_total_gb": .double(mem.totalPhysicalRAMGB),
                "cpu_percent": .double(mem.cpuPercent),
                "metal_recommended_max_working_set_gb": .double(mem.metalRecommendedMaxWorkingSetGB),
                "metal_current_allocated_mb": .int(mem.metalCurrentAllocatedMB),
                "process_resident_mb": .int(mem.processResidentMB),
                "device_name": .string(mem.deviceName),
            ]),
            "models": .object(modelsJSON),
            "scheduler": .object([
                "gpu_gate_busy": .bool(schedulerMetrics.gpuGateBusy),
                "gpu_queue_depth": .int(schedulerMetrics.gpuQueueDepth),
                "max_queue_depth": .int(schedulerMetrics.maxQueueDepth),
                "active_ane_tasks": .int(schedulerMetrics.activeANETasks),
                "loaded_gpu_models": .array(schedulerMetrics.loadedGPUModels.map { .string($0) }),
                "loaded_ane_models": .array(schedulerMetrics.loadedANEModels.map { .string($0) }),
                "pinned_models": .array(schedulerMetrics.pinnedModels.map { .string($0) }),
            ]),
        ]
    }

    public func performanceProfile(range: String) throws -> SQLiteStore.PerformanceProfileSummary {
        try store.queryPerformanceProfile(range: range)
    }

    public func clearCaches(modelID: String?) async throws -> [String: JSONValue] {
        if let modelID, !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let desc = await scheduler.modelDescription(for: trimmed), desc.isLoaded,
                  let provider = try? await scheduler.provider(for: trimmed) else {
                return ["deleted_count": .int(0)]
            }
            let cleared = await provider.clearCache()
            return ["deleted_count": .int(cleared ? 1 : 0)]
        }
        let count = await scheduler.clearAllCaches()
        return ["deleted_count": .int(count)]
    }

    public func clearAllData() throws -> [String: JSONValue] {
        let activeConvs = registry.activeRequests().compactMap(\.conversationID)
        if !activeConvs.isEmpty {
            throw WorkbenchError.conflict(
                code: "generation_in_progress",
                message: "Cannot clear data while conversation generation is active"
            )
        }
        let convCount = (try? store.listConversations().count) ?? 0
        let nbCount = (try? store.listNotebooks().count) ?? 0
        try store.clearAllData()
        return [
            "conversations_deleted": .int(convCount),
            "notebooks_deleted": .int(nbCount),
        ]
    }

    public func ingestTelemetryEvents(_ payload: [String: Any]) -> [String: JSONValue] {
        guard let events = payload["events"] as? [Any] else {
            return [
                "accepted": .int(0),
                "error": .string("body must be {events: [...]}"),
            ]
        }
        var accepted = 0
        for rawEv in events.prefix(100) {
            guard let ev = rawEv as? [String: Any],
                  let typeVal = ev["type"] else {
                continue
            }
            let eventType = String(describing: typeVal)
            let level = (ev["level"] as? String) ?? "info"
            let minVerbosity = ObservabilitySeverity.minimumVerbosity(forSeverity: level).rawValue
            var sanitized: [String: JSONValue] = ["level": .string(level)]
            for (k, v) in ev where k != "type" && k != "level" {
                if let s = v as? String {
                    sanitized[k] = .string(String(s.prefix(2000)))
                } else if let jv = JSONValue.fromAny(v) {
                    sanitized[k] = jv
                }
            }
            ObservabilityDiskLog.shared.recordEvent(
                eventType,
                tier: .events,
                minLevel: minVerbosity,
                source: "frontend-v3",
                fields: sanitized
            )
            accepted += 1
        }
        return ["accepted": .int(accepted)]
    }

    public func capabilities() async -> [String: JSONValue] {
        let mem = HardwareMemoryMonitor.snapshot()
        let asrSummary = await scheduler.voiceASRSummary()
        let models = await scheduler.listModels()
        let embLimits = EmbeddingRequestLimits.default
        let embeddingModelLoaded = models.contains { $0.supportsEmbeddings.value } || isDefaultEmbeddingEngineReady()

        return [
            "server_version": .string("2.0.251-macos"),
            "uptime_ms": .int(Int(max(0, SQLiteStore.nowMillis() - startedAtUnixMs))),
            "metal": .object([
                "available": .bool(true),
                "device_name": .string(mem.deviceName),
                "max_recommended_working_set_size": .double(mem.metalRecommendedMaxWorkingSetGB * 1024.0 * 1024.0 * 1024.0),
            ]),
            "endpoints": .object([
                "messages": .object([
                    "endpoint": .string("/v1/messages"),
                    "supports_base64": .bool(true),
                    "server_side_image_resize": .bool(false),
                ]),
                "embeddings": .object([
                    "endpoint": .string("/v1/embeddings"),
                    "supports_multimodal": .bool(true),
                    "supports_matryoshka": .bool(true),
                    "dimensions": .array([768, 512, 256, 128].map { .int($0) }),
                    "limits": .object([
                        "items": .int(EmbeddingRequestLimits.maxItems),
                        "blocks_per_item": .int(EmbeddingRequestLimits.maxBlocksPerItem),
                        "text_chars": .int(EmbeddingRequestLimits.maxTextChars),
                        "request_media_bytes": .int(EmbeddingRequestLimits.maxRequestMediaBytes),
                    ]),
                    "limits_info": .object([
                        "items": .int(embLimits.items),
                        "blocks_per_item": .int(embLimits.blocksPerItem),
                        "text_chars": .int(embLimits.textChars),
                    ]),
                ]),
            ]),
            "features": .object([
                "streaming": .bool(true),
                "model_caching": .object([
                    "enabled": .bool(true),
                    "eviction_policy": .string("LRU"),
                ]),
                "vision_models": .bool(true),
                "audio_transcription": .bool(asrSummary.available),
                "embeddings_ready": .bool(embeddingModelLoaded),
            ]),
            "voice_asr": .object([
                "available": .bool(asrSummary.available),
                "backend": .string(asrSummary.backend),
                "models": .array(asrSummary.models.map { .string($0) }),
            ]),
        ]
    }

    // MARK: - Helpers

    private func isDefaultEmbeddingEngineReady() -> Bool {
        if hasCustomEmbeddingEngine {
            return true
        }
        if let hybrid = defaultEmbeddingEngine as? HybridLocalEmbeddingEngine {
            return hybrid.hasCompiledANETowers
                && CheckpointPieceTokenizer.load(from: hybrid.checkpointURL) != nil
        }
        let defaultDir = NativeMLXEmbeddingGemma2.defaultCheckpointURL
        return FileManager.default.fileExists(atPath: defaultDir.appendingPathComponent("config.json").path)
            || ANETowerInspector.locateBundle(for: .textBackbone, in: defaultDir) != nil
    }

    private static func requestContainsMedia(_ request: MessagesRequest, kind targetKind: String) -> Bool {
        request.messages.contains { turn in
            turn.attachments.contains(where: { $0.kind == targetKind })
                || turn.parts.contains(where: { part in
                    if case .media(let m) = part {
                        return m.kind == targetKind
                    }
                    return false
                })
        }
    }

    private func isModelGenerating(_ id: String) async -> Bool {
        let schedActive = await scheduler.activeGenerations(for: id)
        return schedActive > 0 || registry.activeRequests().contains { $0.modelID == id }
    }

    private static func readStoredOperationalSettings() -> (effective: OperationalSettings, stored: [String: JSONValue]) {
        let url = HeylookConfig.activeConfigURL()
        guard let text = try? String(contentsOf: url, encoding: .utf8),
              let root = try? TOMLCodec.parse(text),
              case .table(let sec)? = root["settings"] else {
            return (OperationalSettings(), [:])
        }
        var stored: [String: JSONValue] = [:]
        let validKeys: Set<String> = ["observability_level", "observability_retention_days", "mlx_cache_limit_gb"]
        for k in validKeys {
            if let v = sec[k] {
                stored[k] = v.toJSONValue()
            }
        }
        if let jsonData = try? JSONEncoder().encode(JSONValue.object(stored)),
           let decoded = try? JSONDecoder().decode(OperationalSettings.self, from: jsonData) {
            return (decoded, stored)
        }
        return (OperationalSettings(), stored)
    }

    private func buildOperationalConfigSnapshot() -> [String: JSONValue] {
        let fromDisk = Self.readStoredOperationalSettings()
        var mergedStored = fromDisk.stored
        for (k, v) in memoryOnlySettings {
            mergedStored[k] = v
        }
        var effective = fromDisk.effective
        if !memoryOnlySettings.isEmpty,
           let jsonData = try? JSONEncoder().encode(JSONValue.object(mergedStored)),
           let decoded = try? JSONDecoder().decode(OperationalSettings.self, from: jsonData) {
            effective = decoded
        }
        let effDict: [String: JSONValue] = [
            "observability_level": .string(effective.observabilityLevel.rawValue),
            "observability_retention_days": .int(effective.observabilityRetentionDays),
            "mlx_cache_limit_gb": effective.mlxCacheLimitGB.map { .double($0) } ?? .null,
        ]
        var result: [String: JSONValue] = [
            "effective": .object(effDict),
            "stored": .object(mergedStored),
        ]
        if !memoryOnlyKeys.isEmpty {
            result["memory_only"] = .array(memoryOnlyKeys.sorted().map { .string($0) })
        }
        return result
    }

    private static func removeExplicitModelBlock(id targetID: String, fromTOML raw: String) -> (String, Bool) {
        let lines = raw.components(separatedBy: .newlines)
        var out: [String] = []
        var i = 0
        var removed = false
        while i < lines.count {
            let trimmed = lines[i].trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed == "[[models]]" {
                var j = i + 1
                var blockLines = [lines[i]]
                var blockID: String? = nil
                while j < lines.count {
                    let subTrimmed = lines[j].trimmingCharacters(in: .whitespacesAndNewlines)
                    if subTrimmed.hasPrefix("[") {
                        break
                    }
                    blockLines.append(lines[j])
                    if subTrimmed.hasPrefix("id") && subTrimmed.contains("=") {
                        let rhs = subTrimmed.split(separator: "=", maxSplits: 1).last?
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                        if let rhs, !rhs.isEmpty {
                            blockID = rhs
                        }
                    }
                    j += 1
                }
                if blockID == targetID {
                    removed = true
                    i = j
                    continue
                }
                out.append(contentsOf: blockLines)
                i = j
            } else {
                out.append(lines[i])
                i += 1
            }
        }
        return (out.joined(separator: "\n"), removed)
    }
}

private extension JSONValue {
    var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    static func fromAny(_ any: Any) -> JSONValue? {
        if any is NSNull { return .null }
        if let b = any as? Bool { return .bool(b) }
        if let n = any as? NSNumber {
            // Distinguish Bool bridged into NSNumber
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                return .bool(n.boolValue)
            }
            if floor(n.doubleValue) == n.doubleValue && abs(n.doubleValue) <= Double(Int.max) {
                return .int(n.intValue)
            }
            return .double(n.doubleValue)
        }
        if let s = any as? String { return .string(s) }
        if let arr = any as? [Any] {
            return .array(arr.compactMap { JSONValue.fromAny($0) })
        }
        if let dict = any as? [String: Any] {
            var out: [String: JSONValue] = [:]
            for (k, v) in dict {
                if let jv = JSONValue.fromAny(v) {
                    out[k] = jv
                }
            }
            return .object(out)
        }
        return nil
    }
}
