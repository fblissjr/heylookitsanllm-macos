import Foundation
import Observation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#endif

/// Domain controller for model registry/provenance, RAM fit & sidecar/template editing,
/// embedded server & Bonjour peer connection (`RemoteWorkbenchBackend`), and on-demand
/// system metrics, performance profiles, and observability logs.
///
/// Never runs an unconditional polling loop: refreshes happen on demand (surface transitions,
/// explicit Refresh actions, model lifecycle operations, and completed generations).
@MainActor
@Observable
public final class ModelsAndPerfController {
    public let scheduler: ComputeScheduler
    public let store: SQLiteStore
    public let server: HeylookHTTPServer
    public let router: HeylookRouter
    public var config: HeylookConfig

    // MARK: - Model Registry & Selection

    public var registeredModels: [EngineDescription] = [] {
        didSet {
            recomputeFilteredModels()
            recomputeActiveModelDescription()
            onModelsChanged?(registeredModels)
        }
    }

    public private(set) var generationModels: [EngineDescription] = []
    public private(set) var filteredModels: [EngineDescription] = []
    public private(set) var modelCountsByProvider: [EngineProviderKind: Int] = [:]

    public var selectedModelID: String = "" {
        didSet {
            recomputeActiveModelDescription()
            if oldValue != selectedModelID {
                onSelectedModelChanged?(selectedModelID, activeModelDescription)
            }
        }
    }

    public private(set) var activeModelDescription: EngineDescription?
    public var loadingModelIDs: Set<String> = []
    public var selectedProviderFilter: EngineProviderKind? = nil {
        didSet { recomputeFilteredModels() }
    }

    // Inspector/Admin sheets owned cleanly in domain state
    public var editingSidecarModel: EngineDescription?
    public var showMLXConverterSheet: Bool = false
    public var convertingSourceModel: EngineDescription?

    // MARK: - Performance & Observability Telemetry (On-Demand)

    public var memorySnapshot: HardwareMemoryMonitor.MemorySnapshot = HardwareMemoryMonitor.snapshot()
    public var schedulerMetrics: ComputeScheduler.SchedulerMetrics = ComputeScheduler.SchedulerMetrics(
        gpuGateBusy: false,
        gpuQueueDepth: 0,
        maxQueueDepth: 2,
        activeANETasks: 0,
        loadedGPUModels: [],
        loadedANEModels: [],
        pinnedModels: []
    )

    public var selectedPerfWindow: PerfTimeWindow = .twentyFourHours {
        didSet { refreshPerfAndObservability() }
    }
    public private(set) var perfProfileSummary: SQLiteStore.PerformanceProfileSummary?

    public var observabilityMinLevelFilter: String? = nil {
        didSet { refreshObservabilityEntries() }
    }
    public var observabilityCategoryFilter: String = "" {
        didSet { refreshObservabilityEntries() }
    }
    public var observabilitySearchQuery: String = "" {
        didSet { refreshObservabilityEntries() }
    }
    public private(set) var observabilityEntries: [HeylookObservabilityLog.Entry] = []

    public var boundObservabilityLoggerLevel: String {
        get { HeylookObservabilityLog.shared.level }
        set {
            HeylookObservabilityLog.shared.setLevel(newValue)
            refreshObservabilityEntries()
        }
    }

    public var activeGGUFPorts: [String: Int] = [:] {
        didSet {
            sortedActiveGGUFPorts = activeGGUFPorts
                .sorted { $0.value < $1.value }
                .map { "\($0.key)::\($0.value)" }
        }
    }
    public private(set) var sortedActiveGGUFPorts: [String] = []
    public var activeSubprocessCount: Int = 0

    // MARK: - Embedded Server & Remote Peer Discovery

    public var isServerRunning: Bool = false
    public var serverPort: UInt16
    public var discoveredServers: [DiscoveredHeylookServer] = []
    public var isBrowsingBonjour: Bool = false
    public var remoteServerURLString: String = "http://mac-studio.local:8000"
    public var connectedRemoteServerURL: String?
    public private(set) var activeRemoteBackend: RemoteWorkbenchBackend?
    public private(set) var remoteConnectionError: String?

    @ObservationIgnored private let bonjourBrowser = BonjourServiceBrowser()
    @ObservationIgnored public var onStatusMessage: (String?) -> Void = { _ in }
    @ObservationIgnored public var onModelsChanged: (([EngineDescription]) -> Void)?
    @ObservationIgnored public var onSelectedModelChanged: ((_ modelID: String, _ description: EngineDescription?) -> Void)?
    @ObservationIgnored public var onRefreshANEReportsIfNeeded: (() async -> Void)?
    @ObservationIgnored public var onUnloadEmbeddingEngine: (() async -> Void)?
    @ObservationIgnored public var onSchedulePromptPreview: (() -> Void)?
    @ObservationIgnored public var onRemoteBackendChanged: ((RemoteWorkbenchBackend?) async -> Void)?

    public init(
        config: HeylookConfig,
        store: SQLiteStore,
        scheduler: ComputeScheduler,
        router: HeylookRouter,
        server: HeylookHTTPServer
    ) {
        self.config = config
        self.store = store
        self.scheduler = scheduler
        self.router = router
        self.server = server
        self.serverPort = UInt16(config.serverPort)
    }

    // MARK: - Model Selection Projections

    private func recomputeActiveModelDescription() {
        activeModelDescription = registeredModels.first(where: { $0.modelID == selectedModelID })
    }

    private func recomputeFilteredModels() {
        generationModels = registeredModels.filter { !$0.supportsEmbeddings.value && !$0.isSpeechRecognizer }
        var counts: [EngineProviderKind: Int] = [:]
        for model in registeredModels {
            counts[model.provider, default: 0] += 1
        }
        modelCountsByProvider = counts
        if let provider = selectedProviderFilter {
            filteredModels = registeredModels.filter { $0.provider == provider }
        } else {
            filteredModels = registeredModels
        }
    }

    // MARK: - Contract Lookup

    public func activeEngineContract(for modelID: String) async -> EngineContract? {
        if let remoteBackend = activeRemoteBackend {
            return await remoteBackend.engineContract(for: modelID)
        }
        return nil
    }

    // MARK: - On-Demand Metrics, Perf Profile & Observability

    public func refreshMetrics() async {
        if let remoteBackend = activeRemoteBackend {
            await refreshRemoteMetrics(using: remoteBackend)
            return
        }

        memorySnapshot = HardwareMemoryMonitor.snapshot()
        schedulerMetrics = await scheduler.metrics()
        registeredModels = await scheduler.listModels()
        activeGGUFPorts = GGUFPortPool.shared.activeAllocations()
        #if os(macOS)
        activeSubprocessCount = SubprocessLifecycleRegistry.shared.activeSubprocessCount
        #endif
        refreshPerfAndObservability()
        await onRefreshANEReportsIfNeeded?()
    }

    private func refreshRemoteMetrics(using remote: RemoteWorkbenchBackend) async {
        do {
            let remoteModels = try await remote.fetchModelDescriptions()
            for desc in remoteModels {
                let provider = RemoteHeylookProvider(baseURL: remote.baseURL, description: desc)
                await scheduler.register(provider: provider, description: desc)
            }
            let localNonRemote = await scheduler.listModels().filter { $0.provider != .remote }
            registeredModels = localNonRemote + remoteModels
        } catch {
            // Keep existing registered models if transient refresh fails, and surface error honestly.
            onStatusMessage("Remote model refresh warning: \(error.localizedDescription)")
        }

        if let metricsJSON = try? await remote.systemMetrics() {
            applyRemoteSystemMetrics(metricsJSON)
        }
        if let remoteProfile = try? await remote.performanceProfile(range: selectedPerfWindow.rawValue) {
            perfProfileSummary = remoteProfile
        }
        refreshObservabilityEntries()
    }

    private func applyRemoteSystemMetrics(_ json: [String: JSONValue]) {
        if case .object(let sys)? = json["system"] {
            let totalGB = sys["ram_total_gb"]?.doubleValue ?? memorySnapshot.totalPhysicalRAMGB
            let usedGB = sys["ram_used_gb"]?.doubleValue ?? memorySnapshot.ramUsedGB
            let availGB = sys["ram_available_gb"]?.doubleValue ?? memorySnapshot.ramAvailableGB
            let cpuPct = sys["cpu_percent"]?.doubleValue ?? 0
            let recGB = sys["metal_recommended_max_working_set_gb"]?.doubleValue ?? 0
            let metalMB = sys["metal_current_allocated_mb"]?.intValue ?? 0
            let rssMB = sys["process_resident_mb"]?.intValue ?? 0
            let deviceName = sys["device_name"]?.stringValue ?? (connectedRemoteServerURL ?? "Remote Server")
            memorySnapshot = HardwareMemoryMonitor.MemorySnapshot(
                totalPhysicalRAMGB: totalGB,
                metalRecommendedMaxWorkingSetGB: recGB,
                metalCurrentAllocatedMB: metalMB,
                processResidentMB: rssMB,
                deviceName: deviceName,
                ramUsedGB: usedGB,
                ramAvailableGB: availGB,
                cpuPercent: cpuPct
            )
        }

        if case .object(let sched)? = json["scheduler"] {
            let busy = sched["gpu_gate_busy"]?.boolValue ?? false
            let qDepth = sched["gpu_queue_depth"]?.intValue ?? 0
            let maxQ = sched["max_queue_depth"]?.intValue ?? 2
            let ane = sched["active_ane_tasks"]?.intValue ?? 0
            let gpuModels = sched["loaded_gpu_models"]?.stringArray ?? []
            let aneModels = sched["loaded_ane_models"]?.stringArray ?? []
            let pinned = sched["pinned_models"]?.stringArray ?? []
            schedulerMetrics = ComputeScheduler.SchedulerMetrics(
                gpuGateBusy: busy,
                gpuQueueDepth: qDepth,
                maxQueueDepth: maxQ,
                activeANETasks: ane,
                loadedGPUModels: gpuModels,
                loadedANEModels: aneModels,
                pinnedModels: pinned
            )
        }
    }

    public func refreshPerfAndObservability() {
        if let remoteBackend = activeRemoteBackend {
            let range = selectedPerfWindow.rawValue
            Task { [weak self] in
                if let profile = try? await remoteBackend.performanceProfile(range: range) {
                    self?.perfProfileSummary = profile
                }
            }
        } else {
            perfProfileSummary = try? store.queryPerformanceProfile(window: selectedPerfWindow.rawValue, recentLimit: 50)
        }
        refreshObservabilityEntries()
    }

    public func refreshObservabilityEntries() {
        let cat = observabilityCategoryFilter.trimmingCharacters(in: .whitespacesAndNewlines)
        let search = observabilitySearchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        observabilityEntries = HeylookObservabilityLog.shared.recentEntries(
            limit: 300,
            level: observabilityMinLevelFilter,
            category: cat.isEmpty ? nil : cat,
            search: search.isEmpty ? nil : search
        ).reversed()
    }

    public func clearObservabilityEntries() {
        HeylookObservabilityLog.shared.clear()
        refreshObservabilityEntries()
        onStatusMessage("Cleared observability event log.")
    }

    public func exportObservabilityJSONL() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let chronological = observabilityEntries.reversed()
        return chronological.compactMap { entry in
            guard let data = try? encoder.encode(entry) else { return nil }
            return String(data: data, encoding: .utf8)
        }.joined(separator: "\n")
    }

    // MARK: - Demo Workspace (Explicit Opt-In)

    public func loadSampleDemoWorkspace() async {
        let demos = Self.defaultDemoModels()
        await scheduler.registerDiscoveredModels(demos)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty || !generationModels.contains(where: { $0.modelID == selectedModelID }) {
            selectedModelID = generationModels.first?.modelID ?? registeredModels.first?.modelID ?? ""
        }
        await refreshMetrics()
        onStatusMessage("Loaded sample demo workspace ([DEMO] synthetic models — configure heylook.toml for local weights).")
    }

    public static func defaultDemoModels() -> [EngineDescription] {
        [
            EngineDescription(
                modelID: "gemma-4-26b-a4b-it-mlx",
                provider: .mlx,
                modelPath: "models/mlx/Gemma-4-26B-A4B-IT",
                architecture: ProvenanceField(value: "gemma4", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "4bit", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .hybridANEGPU, provenance: .configured, effect: .reload, computePlacement: .hybridANEGPU),
                isPinned: false,
                isLoaded: true,
                contextSize: ProvenanceField(value: 32768, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                thinkingControls: ThinkingControls(
                    supportsThinking: true,
                    canDisableThinking: true,
                    defaultEnabled: true,
                    switchVariable: "enable_thinking"
                ),
                estimatedMemoryMB: 14800,
                tags: ["demo", "chat", "vision", "thinking"]
            ),
            EngineDescription(
                modelID: "qwen3.5-0.8b-mlx-8bit",
                provider: .mlx,
                modelPath: "models/mlx/Qwen3.5-0.8B-MLX-8bit",
                architecture: ProvenanceField(value: "qwen3_5", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "8bit", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .mlxGPU, provenance: .derived, effect: .reload, computePlacement: .mlxGPU),
                isPinned: false,
                isLoaded: false,
                contextSize: ProvenanceField(value: 32768, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                thinkingControls: ThinkingControls(
                    supportsThinking: true,
                    canDisableThinking: true,
                    defaultEnabled: true,
                    switchVariable: "enable_thinking"
                ),
                estimatedMemoryMB: 980,
                tags: ["demo", "fast", "thinking_depth"]
            ),
            EngineDescription(
                modelID: "coreai-llama-3.2-3b-ane",
                provider: .coreai,
                modelPath: "models/coreai/Llama-3.2-3B.aimodel",
                architecture: ProvenanceField(value: "llama", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "int4-palettized", provenance: .derived, effect: .reload),
                computePlacement: ProvenanceField(value: .ane, provenance: .configured, effect: .reload, computePlacement: .ane),
                isPinned: true,
                isLoaded: true,
                contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: false, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: true, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 2100,
                tags: ["demo", "ane", "coreai"]
            ),
            EngineDescription(
                modelID: "embeddinggemma-2-MLX-BF16",
                provider: .mlx,
                modelPath: "models/embeddings/embeddinggemma-2-MLX-BF16",
                architecture: ProvenanceField(value: "embedding_gemma2", provenance: .derived, effect: .reload),
                quantization: ProvenanceField(value: "bf16+ane", provenance: .configured, effect: .reload),
                computePlacement: ProvenanceField(value: .ane, provenance: .configured, effect: .reload, computePlacement: .ane),
                isPinned: true,
                isLoaded: true,
                contextSize: ProvenanceField(value: 8192, provenance: .derived, effect: .reload),
                flashAttention: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsVision: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsAudio: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsEmbeddings: ProvenanceField(value: true, provenance: .derived, effect: .reload),
                supportsThinking: ProvenanceField(value: false, provenance: .derived, effect: .perRequest),
                estimatedMemoryMB: 1488,
                tags: ["demo", "embeddings", "ane", "multimodal"]
            )
        ]
    }

    // MARK: - Embedded Server & Bonjour Remote Connection

    public func toggleServer() async {
        do {
            if isServerRunning {
                await server.stop()
                isServerRunning = false
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "server",
                    message: "Embedded HTTP/SSE server stopped"
                )
                onStatusMessage("Embedded server stopped.")
            } else {
                try await server.start()
                isServerRunning = true
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "server",
                    message: "Embedded HTTP/SSE server listening on http://127.0.0.1:\(serverPort)"
                )
                onStatusMessage("Embedded server listening on http://127.0.0.1:\(serverPort)")
            }
            refreshObservabilityEntries()
        } catch {
            onStatusMessage("Server toggle error: \(error.localizedDescription)")
        }
    }

    public func startBonjourDiscovery() {
        isBrowsingBonjour = true
        bonjourBrowser.start { [weak self] servers in
            Task { @MainActor [weak self] in
                self?.discoveredServers = servers
            }
        }
    }

    public func stopBonjourDiscovery() {
        bonjourBrowser.stop()
        isBrowsingBonjour = false
    }

    /// Connects to a remote `HeylookServer` peer via `RemoteWorkbenchBackend`.
    /// Fails honestly with a descriptive error banner when `/v1/models` fails — never installs fake models.
    public func connectToRemoteServer(_ urlString: String) async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let baseURL = URL(string: trimmed),
              let scheme = baseURL.scheme?.lowercased(),
              (scheme == "http" || scheme == "https"),
              baseURL.host != nil else {
            let msg = "Invalid remote server URL: '\(urlString)'"
            remoteConnectionError = msg
            onStatusMessage(msg)
            return
        }

        remoteServerURLString = trimmed
        let token = config.adminToken ?? ProcessInfo.processInfo.environment["HEYLOOK_ADMIN_TOKEN"]
        let backend = RemoteWorkbenchBackend(baseURL: baseURL, adminToken: token)

        do {
            let remoteDescriptions = try await backend.fetchModelDescriptions()
            remoteConnectionError = nil
            activeRemoteBackend = backend
            connectedRemoteServerURL = trimmed

            for desc in remoteDescriptions {
                let provider = RemoteHeylookProvider(baseURL: baseURL, description: desc)
                await scheduler.register(provider: provider, description: desc)
            }
            registeredModels = await scheduler.listModels()
            if let firstRemote = remoteDescriptions.first(where: { !$0.supportsEmbeddings.value && !$0.isSpeechRecognizer })
                ?? remoteDescriptions.first {
                selectedModelID = firstRemote.modelID
            }

            HeylookObservabilityLog.shared.record(
                level: "INFO",
                category: "remote",
                message: "Connected to remote Heylook server at \(trimmed) (\(remoteDescriptions.count) models)"
            )
            await onRemoteBackendChanged?(backend)
            await refreshMetrics()
            onStatusMessage("Connected to remote server at \(trimmed) (\(remoteDescriptions.count) models).")
        } catch {
            remoteConnectionError = error.localizedDescription
            HeylookObservabilityLog.shared.record(
                level: "ERROR",
                category: "remote",
                message: "Failed connecting to \(trimmed): \(error.localizedDescription)"
            )
            refreshObservabilityEntries()
            onStatusMessage("Could not connect to \(trimmed): \(error.localizedDescription)")
        }
    }

    public func disconnectRemoteServer() async {
        let previouslyConnected = connectedRemoteServerURL
        activeRemoteBackend = nil
        connectedRemoteServerURL = nil
        remoteConnectionError = nil

        let currentList = await scheduler.listModels()
        for model in currentList where model.provider == .remote {
            _ = await scheduler.removeModel(model.modelID)
        }
        await rescanConfiguredModels()
        await onRemoteBackendChanged?(nil)
        if let previouslyConnected {
            onStatusMessage("Disconnected from remote server \(previouslyConnected).")
        }
    }

    // MARK: - Model Lifecycle, Scan Folders, RAM Fit, Sidecars & Templates

    public func assessModelRAMFit(
        _ model: EngineDescription,
        ctxSizeOverride: Int? = nil,
        sidecarOverride: ModelSidecarConfig? = nil
    ) -> GGUFSpawnConfig.RAMFitAssessment {
        GGUFSpawnConfig.assessRAMFit(
            for: model,
            ctxSize: ctxSizeOverride,
            sidecar: sidecarOverride
        )
    }

    public func applyQuickContextSize(_ ctxSize: Int, for model: EngineDescription) async {
        guard !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return
        }
        if model.provider == .remote, let remoteBackend = activeRemoteBackend {
            do {
                let patch = ModelConfigPatchPayload(config: ["max_kv_size": .int(ctxSize), "n_ctx": .int(ctxSize)])
                _ = try await remoteBackend.patchAdminModel(id: model.modelID, updates: patch)
                await refreshMetrics()
                onStatusMessage("Updated context size to \(ctxSize) on remote \(model.modelID).")
            } catch {
                onStatusMessage("Failed to update remote context size: \(error.localizedDescription)")
            }
            return
        }
        var sidecar = loadSidecarConfig(for: model)
        sidecar.ctxSize = ctxSize
        await saveModelSidecar(sidecar, for: model)
    }

    public func addModelScanFolder(_ url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
        guard !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return
        }
        do {
            config = try HeylookConfig.addScanFolder(url.standardizedFileURL.path)
            await rescanConfiguredModels()
            onStatusMessage("Added scan folder '\(url.lastPathComponent)' to heylook.toml.")
        } catch {
            onStatusMessage("Failed to add scan folder: \(error.localizedDescription)")
        }
    }

    public func removeModelScanFolder(_ folder: String) async {
        guard !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return
        }
        do {
            config = try HeylookConfig.removeScanFolder(folder)
            await rescanConfiguredModels()
            onStatusMessage("Removed scan folder '\(folder)' from heylook.toml.")
        } catch {
            onStatusMessage("Failed to remove scan folder: \(error.localizedDescription)")
        }
    }

    public func toggleModelPin(_ model: EngineDescription) async {
        await scheduler.setPinned(!model.isPinned, for: model.modelID)
        await refreshMetrics()
    }

    public func toggleModelLoaded(_ model: EngineDescription) async {
        if model.provider == .remote, let remoteBackend = activeRemoteBackend {
            guard !loadingModelIDs.contains(model.modelID) else { return }
            loadingModelIDs.insert(model.modelID)
            defer { loadingModelIDs.remove(model.modelID) }
            do {
                if model.isLoaded {
                    _ = try await remoteBackend.unloadModel(id: model.modelID)
                    onStatusMessage("Unloaded remote model \(model.modelID).")
                } else {
                    _ = try await remoteBackend.loadModel(id: model.modelID, warm: false)
                    onStatusMessage("Loaded remote model \(model.modelID).")
                }
                await refreshMetrics()
            } catch {
                await refreshMetrics()
                onStatusMessage("Remote model operation failed for \(model.modelID): \(error.localizedDescription)")
            }
            return
        }

        if model.isLoaded {
            await scheduler.unloadModel(model.modelID)
            await refreshMetrics()
            onStatusMessage("Unloaded \(model.modelID).")
        } else {
            guard !loadingModelIDs.contains(model.modelID) else { return }
            loadingModelIDs.insert(model.modelID)
            defer { loadingModelIDs.remove(model.modelID) }

            do {
                _ = try await scheduler.activateModel(model.modelID)
                await refreshMetrics()
                onStatusMessage("Loaded \(model.modelID).")
            } catch {
                await refreshMetrics()
                onStatusMessage("Failed to load \(model.modelID): \(error.localizedDescription)")
            }
        }
    }

    public func unloadAllModels() async {
        await scheduler.unloadAllModels()
        await onUnloadEmbeddingEngine?()
        await refreshMetrics()
        onStatusMessage("Unloaded all models, cleared Metal/ANE caches, and terminated background subprocesses.")
    }

    public func rescanConfiguredModels() async {
        config = HeylookConfig.loadDefault()
        let discovered = config.discoverConfiguredModels()
        await scheduler.registerDiscoveredModels(discovered)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty, let first = generationModels.first ?? registeredModels.first {
            selectedModelID = first.modelID
        }
        await refreshMetrics()
        onStatusMessage("Rescanned heylook.toml (\(registeredModels.count) models across \(config.scanFolders.count) scan folders).")
    }

    public func unloadGGUFAllocation(_ allocationEntry: String) async {
        let modelID = allocationEntry.components(separatedBy: "::").first ?? allocationEntry
        await scheduler.unloadModel(modelID)
        GGUFPortPool.shared.releasePort(for: modelID)
        await refreshMetrics()
        onStatusMessage("Unloaded GGUF subprocess for \(modelID).")
    }

    public func hasSidecarFileOnDisk(for model: EngineDescription) -> Bool {
        guard model.provider != .remote else { return false }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        return FileManager.default.fileExists(atPath: sidecarURL.path)
    }

    public func loadSidecarConfig(for model: EngineDescription) -> ModelSidecarConfig {
        guard model.provider != .remote else { return ModelSidecarConfig() }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        if let text = try? String(contentsOf: sidecarURL, encoding: .utf8) {
            return ModelSidecarConfig.parseTOML(text)
        }
        return ModelSidecarConfig()
    }

    public func saveModelSidecar(_ sidecar: ModelSidecarConfig, for model: EngineDescription) async {
        guard !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let sidecarURL = dirURL.appendingPathComponent("model.heylook.toml")
        var isDir: ObjCBool = false
        do {
            if sidecar == ModelSidecarConfig() {
                if FileManager.default.fileExists(atPath: sidecarURL.path) {
                    try FileManager.default.removeItem(at: sidecarURL)
                }
                await rescanConfiguredModels()
                onSchedulePromptPreview?()
                onStatusMessage("Removed model.heylook.toml overrides for \(model.modelID) (reverted to derived defaults).")
                return
            }

            if FileManager.default.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue {
                try sidecar.toTOML().write(to: sidecarURL, atomically: true, encoding: .utf8)
            }
            _ = try await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
            await refreshMetrics()
            onSchedulePromptPreview?()
            onStatusMessage("Saved model.heylook.toml sidecar for \(model.modelID).")
        } catch {
            onStatusMessage("Failed to write model.heylook.toml: \(error.localizedDescription)")
        }
    }

    public func loadChatTemplateState(for model: EngineDescription) -> HeylookConfig.ChatTemplateState {
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        return HeylookConfig.inspectChatTemplate(
            in: dirURL,
            fallbackSource: model.templateSource.value
        )
    }

    public func validateJinjaTemplateSyntax(_ template: String) -> String? {
        do {
            try HeylookJinjaRenderer.validateSyntax(template)
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    @discardableResult
    public func saveChatTemplateOverride(_ template: String, for model: EngineDescription) async throws -> HeylookConfig.ChatTemplateState {
        guard !HeylookConfig.isReadOnlyModelConfig else {
            throw ProviderExecutionError.invalidRequest(detail: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
        }
        if model.provider == .remote, let remoteBackend = activeRemoteBackend {
            let state = try await remoteBackend.setChatTemplate(id: model.modelID, template: template)
            await refreshMetrics()
            onSchedulePromptPreview?()
            onStatusMessage("Saved remote chat template override for \(model.modelID).")
            return state
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let state = try HeylookConfig.writeChatTemplateOverride(template, in: dirURL)
        let sidecar = loadSidecarConfig(for: model)
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        onSchedulePromptPreview?()
        onStatusMessage("Saved chat_template.heylook.jinja override for \(model.modelID).")
        return state
    }

    @discardableResult
    public func revertChatTemplateOverride(for model: EngineDescription) async -> HeylookConfig.ChatTemplateState {
        guard !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return loadChatTemplateState(for: model)
        }
        if model.provider == .remote, let remoteBackend = activeRemoteBackend {
            if let state = try? await remoteBackend.deleteChatTemplate(id: model.modelID) {
                await refreshMetrics()
                onSchedulePromptPreview?()
                onStatusMessage("Reverted remote chat template override for \(model.modelID) (active: \(state.source)).")
                return state
            }
        }
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: model.modelPath)
        let reverted = HeylookConfig.deleteChatTemplateOverride(
            in: dirURL,
            fallbackSource: model.templateSource.value == "chat_template.heylook.jinja"
                ? (model.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json")
                : model.templateSource.value
        )
        let sidecar = loadSidecarConfig(for: model)
        _ = try? await scheduler.applySidecarUpdate(sidecar, for: model.modelID)
        await refreshMetrics()
        onSchedulePromptPreview?()
        onStatusMessage("Reverted chat_template.heylook.jinja override for \(model.modelID) (active: \(reverted.source)).")
        return reverted
    }
}

private extension JSONValue {
    var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }

    var intValue: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d): return Int(d)
        default: return nil
        }
    }

    var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    var stringArray: [String]? {
        guard case .array(let items) = self else { return nil }
        return items.compactMap(\.stringValue)
    }
}
