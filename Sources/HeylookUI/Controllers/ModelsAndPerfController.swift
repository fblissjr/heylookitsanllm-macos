import Foundation
import Observation
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#endif

/// Per-loaded-model live telemetry projection parsed from `/v1/system/metrics`
/// (`WorkbenchService.systemMetrics()` locally or remote `GET /v1/system/metrics`).
public struct LoadedModelTelemetry: Identifiable, Sendable, Equatable {
    public var id: String { modelID }
    public let modelID: String
    public let providerName: String
    public let memoryMB: Double?
    public let memorySource: String?
    public let contextUsed: Int?
    public let contextCapacity: Int?
    public let contextPercent: Double?
    public let requestsActive: Int?
    public let requestsQueued: Int?

    public init(
        modelID: String,
        providerName: String,
        memoryMB: Double? = nil,
        memorySource: String? = nil,
        contextUsed: Int? = nil,
        contextCapacity: Int? = nil,
        contextPercent: Double? = nil,
        requestsActive: Int? = nil,
        requestsQueued: Int? = nil
    ) {
        self.modelID = modelID
        self.providerName = providerName
        self.memoryMB = memoryMB
        self.memorySource = memorySource
        self.contextUsed = contextUsed
        self.contextCapacity = contextCapacity
        self.contextPercent = contextPercent
        self.requestsActive = requestsActive
        self.requestsQueued = requestsQueued
    }
}

/// Domain controller for model registry/provenance, schema-driven config editing,
/// live RAM fit, multi-rung chat template inspection, embedded server & Bonjour peer
/// connection (`RemoteWorkbenchBackend`), and on-demand system metrics, performance
/// profiles, `[settings]` operational config, and observability logs.
///
/// Never runs an unconditional polling loop: refreshes happen strictly on demand
/// (surface mount, explicit Refresh actions, model lifecycle operations, and completed
/// generations).
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

    // Rich Admin Wire State, Schema Cache, Drafts, Fit Gates & Load Outcomes
    public private(set) var adminRowsByID: [String: AdminModelWireRow] = [:]
    public private(set) var publicRowsByID: [String: PublicModelWireRow] = [:]
    public private(set) var cachedProviderSchemas: [EngineProviderKind: [ConfigFieldDescriptor]] = [:]
    public var configDraftsByModelID: [String: [String: String]] = [:]
    public var templateDraftsByModelID: [String: String] = [:]
    public var fitGateReasonsByModelID: [String: String] = [:]
    public var loadWarmOutcomesByModelID: [String: String] = [:]
    public var modelsErrorBanner: String?
    public var modelsStatusNote: String?

    // Watch Folders (`scan-config`)
    public private(set) var scanConfig: ScanConfigPayload?
    public var scanFoldersDraftText: String = ""
    public var scanIntervalDraft: Int = 900
    public private(set) var isSavingScanConfig: Bool = false
    public private(set) var scanConfigNote: String?

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
    public private(set) var loadedModelTelemetry: [LoadedModelTelemetry] = []
    public private(set) var lastMetricsUpdatedAt: Date?
    public private(set) var perfLoadError: String?

    public var selectedPerfWindow: PerfTimeWindow = .twentyFourHours {
        didSet { refreshPerfAndObservability() }
    }
    public private(set) var perfProfileSummary: SQLiteStore.PerformanceProfileSummary?

    // Persistent Operational `[settings]` & Disk JSONL Streams (`events` / `metrics`)
    public private(set) var operationalSettings: OperationalSettings = OperationalSettings()
    public private(set) var operationalMemoryOnlyKeys: [String] = []
    public private(set) var operationalSettingsStatus: String?
    public private(set) var isSavingOperationalSettings: Bool = false
    public var selectedDiskLogStream: ObservabilityStream = .events
    public private(set) var diskLogLevel: String = "off"
    public private(set) var diskLogRecords: [[String: JSONValue]] = []

    // In-Memory Ring Observability Log (`HeylookObservabilityLog`)
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
    public private(set) var serverBindHost: String = "127.0.0.1"
    public private(set) var serverAdvertiseBonjourFlag: Bool = false
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
        self.scanFoldersDraftText = config.scanFolders.joined(separator: "\n")
    }

    // MARK: - Server & Bonjour UI Truth

    public var isServerLANReachable: Bool {
        let h = serverBindHost.trimmingCharacters(in: .whitespacesAndNewlines)
        return h.isEmpty || h == "0.0.0.0" || h == "::" || h == "*"
    }

    public var isBonjourAdvertisementActive: Bool {
        isServerRunning && serverAdvertiseBonjourFlag && isServerLANReachable
    }

    private func syncServerBindingTruth() async {
        serverBindHost = await server.host.trimmingCharacters(in: .whitespacesAndNewlines)
        serverAdvertiseBonjourFlag = await server.advertiseBonjour
    }

    // MARK: - Model Selection Projections

    private func recomputeActiveModelDescription() {
        activeModelDescription = registeredModels.first(where: { $0.modelID == selectedModelID })
    }

    private func recomputeFilteredModels() {
        generationModels = registeredModels.filter { !$0.supportsEmbeddings.value && !$0.isSpeechRecognizer }
        var counts: [EngineProviderKind: Int] = [:]
        for model in registeredModels {
            let effectiveProvider = adminRowsByID[model.modelID]?.provider ?? model.provider
            counts[effectiveProvider, default: 0] += 1
        }
        modelCountsByProvider = counts
        if let provider = selectedProviderFilter {
            filteredModels = registeredModels.filter { model in
                (adminRowsByID[model.modelID]?.provider ?? model.provider) == provider
            }
        } else {
            filteredModels = registeredModels
        }
    }

    // MARK: - Contract & Schema Lookup

    public func activeEngineContract(for modelID: String) async -> EngineContract? {
        guard !modelID.isEmpty else { return nil }
        if let remoteBackend = activeRemoteBackend {
            return await remoteBackend.engineContract(for: modelID)
        }
        if let cached = adminRowsByID[modelID]?.engine {
            return cached
        }
        return (try? await router.service.adminModel(id: modelID))?.engine
    }

    public func effectiveProvider(for model: EngineDescription) -> EngineProviderKind {
        adminRowsByID[model.modelID]?.provider ?? model.provider
    }

    public func schemaFields(for provider: EngineProviderKind) async throws -> [ConfigFieldDescriptor] {
        if let cached = cachedProviderSchemas[provider], !cached.isEmpty {
            return cached
        }
        if let remoteBackend = activeRemoteBackend {
            let raw = try await remoteBackend.getModelOptions()
            let parsed = Self.parseProviderSchemas(from: raw)
            for (kind, fields) in parsed {
                cachedProviderSchemas[kind] = fields
            }
            if let matched = parsed[provider] {
                return matched
            }
            if provider == .remote, let fallback = parsed[.mlx] ?? parsed[.gguf] {
                return fallback
            }
            return []
        }
        let localFields = ProviderConfigSchema.schema(for: provider).fields
        cachedProviderSchemas[provider] = localFields
        return localFields
    }

    public static func parseProviderSchemas(from payload: [String: JSONValue]) -> [EngineProviderKind: [ConfigFieldDescriptor]] {
        guard case .object(let providersObj)? = payload["providers"] else {
            return [:]
        }
        var result: [EngineProviderKind: [ConfigFieldDescriptor]] = [:]
        for (provKey, provVal) in providersObj {
            guard let kind = EngineProviderKind(rawValue: provKey.lowercased()),
                  case .object(let provDict) = provVal,
                  case .array(let fieldsArr)? = provDict["fields"] else {
                continue
            }
            let descriptors = fieldsArr.compactMap { Self.parseFieldDescriptor($0) }
            result[kind] = descriptors
        }
        return result
    }

    private static func parseFieldDescriptor(_ value: JSONValue) -> ConfigFieldDescriptor? {
        guard case .object(let obj) = value,
              let name = obj["name"]?.stringValue, !name.isEmpty,
              let effectRaw = obj["effect"]?.stringValue,
              let effect = ConfigFieldEffect(rawValue: effectRaw) else {
            return nil
        }

        let defaultValue = obj["default"]
        let rawTypeStr = obj["type"]?.stringValue
        let inferredType: ConfigFieldValueType = {
            if let rawTypeStr, let parsed = ConfigFieldValueType(rawValue: rawTypeStr) {
                return parsed
            }
            if case .bool = defaultValue { return .boolean }
            if case .int = defaultValue { return .integer }
            if case .double = defaultValue { return .number }
            if case .array = defaultValue { return .array }
            return .string
        }()

        let enumValues: [String]? = {
            guard case .array(let items)? = obj["enum"] else { return nil }
            let mapped: [String] = items.compactMap { item in
                switch item {
                case .string(let s): return s
                case .int(let i): return String(i)
                case .double(let d): return String(d)
                case .bool(let b): return b ? "true" : "false"
                default: return nil
                }
            }
            return mapped.isEmpty ? nil : mapped
        }()

        let itemType: ConfigFieldValueType? = {
            guard case .object(let itemsObj)? = obj["items"],
                  let itemTypeStr = itemsObj["type"]?.stringValue else {
                return nil
            }
            return ConfigFieldValueType(rawValue: itemTypeStr)
        }()

        let ui = obj["ui"]?.stringValue.flatMap(ConfigUIVisibility.init(rawValue:)) ?? .normal

        return ConfigFieldDescriptor(
            name: name,
            type: inferredType,
            defaultValue: defaultValue == .null ? nil : defaultValue,
            minimum: obj["minimum"]?.doubleValue,
            maximum: obj["maximum"]?.doubleValue,
            exclusiveMinimum: obj["exclusiveMinimum"]?.doubleValue,
            exclusiveMaximum: obj["exclusiveMaximum"]?.doubleValue,
            enumValues: enumValues,
            itemType: itemType,
            effect: effect,
            arg: obj["arg"]?.stringValue,
            ui: ui,
            shape: obj["shape"]?.stringValue,
            reason: obj["reason"]?.stringValue,
            isFileOnly: obj["file_only"]?.boolValue ?? false,
            isLoadSetting: obj["load_setting"]?.boolValue ?? false,
            required: obj["required"]?.boolValue ?? false,
            description: obj["description"]?.stringValue ?? ""
        )
    }

    // MARK: - On-Demand Metrics, Perf Profile & Observability

    public func refreshMetrics() async {
        await refreshMetrics(forceRefresh: false)
    }

    public func refreshMetrics(forceRefresh: Bool) async {
        if let remoteBackend = activeRemoteBackend {
            await refreshRemoteMetrics(using: remoteBackend, forceRefresh: forceRefresh)
            return
        }

        memorySnapshot = HardwareMemoryMonitor.snapshot()
        schedulerMetrics = await scheduler.metrics()
        registeredModels = await scheduler.listModels()
        activeGGUFPorts = GGUFPortPool.shared.activeAllocations()
        #if os(macOS)
        activeSubprocessCount = SubprocessLifecycleRegistry.shared.activeSubprocessCount
        #endif

        await syncServerBindingTruth()
        await refreshLocalAdminAndTelemetry()
        refreshPerfAndObservability()
        await onRefreshANEReportsIfNeeded?()
    }

    private func refreshLocalAdminAndTelemetry() async {
        let adminRows = await router.service.listAdminModels()
        var adminMap: [String: AdminModelWireRow] = [:]
        for row in adminRows {
            adminMap[row.id] = row
        }
        adminRowsByID = adminMap
        recomputeFilteredModels()

        let sysJSON = await router.service.systemMetrics()
        if case .object(let modelsObj)? = sysJSON["models"] {
            parseLoadedModelTelemetry(modelsObj)
        } else {
            loadedModelTelemetry = []
        }
        lastMetricsUpdatedAt = Date()
        perfLoadError = nil

        await refreshFitGatesForUnloadedModels()
    }

    private func refreshRemoteMetrics(using remote: RemoteWorkbenchBackend, forceRefresh: Bool = false) async {
        do {
            let remoteModels = try await remote.fetchModelDescriptions()
            for desc in remoteModels {
                let provider = RemoteHeylookProvider(baseURL: remote.baseURL, description: desc)
                await scheduler.register(provider: provider, description: desc)
            }
            let localNonRemote = await scheduler.listModels().filter { $0.provider != .remote }
            registeredModels = localNonRemote + remoteModels
            modelsErrorBanner = nil
        } catch {
            // Keep existing registered models if transient refresh fails, and surface error honestly.
            let msg = "Remote model refresh warning: \(error.localizedDescription)"
            modelsErrorBanner = msg
            onStatusMessage(msg)
        }

        let adminRows = await remote.listAdminModels()
        var adminMap: [String: AdminModelWireRow] = [:]
        for row in adminRows {
            adminMap[row.id] = row
        }
        adminRowsByID = adminMap

        var pubMap: [String: PublicModelWireRow] = [:]
        for model in registeredModels {
            if let pub = await remote.publicWireRow(for: model.modelID) {
                pubMap[model.modelID] = pub
            }
        }
        publicRowsByID = pubMap
        recomputeFilteredModels()

        do {
            let metricsJSON = try await fetchRemoteSystemMetrics(using: remote, forceRefresh: forceRefresh)
            applyRemoteSystemMetrics(metricsJSON)
            lastMetricsUpdatedAt = Date()
            perfLoadError = nil
        } catch {
            perfLoadError = "Could not load remote metrics: \(error.localizedDescription)"
        }

        do {
            let remoteProfile = try await remote.performanceProfile(range: selectedPerfWindow.rawValue)
            perfProfileSummary = remoteProfile
            perfLoadError = nil
        } catch {
            perfLoadError = "Could not load remote performance profile: \(error.localizedDescription)"
        }

        refreshObservabilityEntries()
    }

    private func fetchRemoteSystemMetrics(
        using remote: RemoteWorkbenchBackend,
        forceRefresh: Bool
    ) async throws -> [String: JSONValue] {
        guard forceRefresh else {
            return try await remote.systemMetrics()
        }
        var url = remote.baseURL
        url.appendPathComponent("v1")
        url.appendPathComponent("system")
        url.appendPathComponent("metrics")
        var comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
        comps?.queryItems = [URLQueryItem(name: "force_refresh", value: "true")]
        guard let targetURL = comps?.url else {
            return try await remote.systemMetrics()
        }
        var req = URLRequest(url: targetURL)
        req.httpMethod = "GET"
        let (data, resp) = try await remote.session.data(for: req)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            return try await remote.systemMetrics()
        }
        return try JSONDecoder().decode([String: JSONValue].self, from: data)
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

        if case .object(let modelsObj)? = json["models"] {
            parseLoadedModelTelemetry(modelsObj)
        } else {
            loadedModelTelemetry = []
        }
    }

    private func parseLoadedModelTelemetry(_ modelsObj: [String: JSONValue]) {
        var list: [LoadedModelTelemetry] = []
        list.reserveCapacity(modelsObj.count)
        for id in modelsObj.keys.sorted(by: { $0.localizedStandardCompare($1) == .orderedAscending }) {
            guard case .object(let dict)? = modelsObj[id] else { continue }
            let resolvedProvider: String = dict["provider"]?.stringValue
                ?? adminRowsByID[id]?.provider.displayName
                ?? publicRowsByID[id]?.provider.displayName
                ?? registeredModels.first(where: { $0.modelID == id })?.provider.displayName
                ?? "model"
            let rawMemory = dict["memory_mb"]?.doubleValue
            let memoryMB = (rawMemory != nil && rawMemory! > 0) ? rawMemory : nil
            let memorySource = dict["memory_source"]?.stringValue
            let ctxUsed = dict["context_used"]?.intValue
            let ctxCap = dict["context_capacity"]?.intValue
            let ctxPct = dict["context_percent"]?.doubleValue
            let reqActive = dict["requests_active"]?.intValue
            let reqQueued = dict["requests_queued"]?.intValue
            list.append(
                LoadedModelTelemetry(
                    modelID: id,
                    providerName: resolvedProvider,
                    memoryMB: memoryMB,
                    memorySource: memorySource,
                    contextUsed: ctxUsed,
                    contextCapacity: ctxCap,
                    contextPercent: ctxPct,
                    requestsActive: reqActive,
                    requestsQueued: reqQueued
                )
            )
        }
        loadedModelTelemetry = list
    }

    public func refreshPerfAndObservability() {
        if let remoteBackend = activeRemoteBackend {
            let range = selectedPerfWindow.rawValue
            Task { [weak self] in
                guard let self else { return }
                do {
                    let profile = try await remoteBackend.performanceProfile(range: range)
                    self.perfProfileSummary = profile
                    self.perfLoadError = nil
                } catch {
                    self.perfLoadError = "Could not load performance profile: \(error.localizedDescription)"
                }
            }
        } else {
            do {
                perfProfileSummary = try store.queryPerformanceProfile(window: selectedPerfWindow.rawValue, recentLimit: 50)
                perfLoadError = nil
            } catch {
                perfLoadError = "Could not query performance profile: \(error.localizedDescription)"
            }
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

    // MARK: - Operational `[settings]` & Disk JSONL Logs (`events` vs `metrics`)

    public func loadOperationalConfigAndDiskLogs() async {
        if let remoteBackend = activeRemoteBackend {
            do {
                let cfg = try await remoteBackend.getOperationalConfig()
                applyOperationalConfigResponse(cfg)
            } catch {
                operationalSettingsStatus = "Config unavailable: \(error.localizedDescription)"
            }
            await loadDiskObservabilityLogs(stream: selectedDiskLogStream)
            return
        }

        let snap = await router.service.getOperationalConfig()
        applyOperationalConfigResponse(snap)
        await loadDiskObservabilityLogs(stream: selectedDiskLogStream)
    }

    public func updateOperationalSetting(_ patch: [String: JSONValue]) async {
        isSavingOperationalSettings = true
        operationalSettingsStatus = "Saving…"
        defer { isSavingOperationalSettings = false }

        do {
            if let remoteBackend = activeRemoteBackend {
                let updated = try await remoteBackend.updateOperationalConfig(patch)
                applyOperationalConfigResponse(updated)
            } else {
                let patchData = try JSONEncoder().encode(patch)
                let updated = try await router.service.updateOperationalConfig(patchData: patchData)
                applyOperationalConfigResponse(updated)
            }
            operationalSettingsStatus = operationalMemoryOnlyKeys.isEmpty
                ? "Saved to heylook.toml [settings]"
                : "Applied in memory only (read-only config)"
            await loadDiskObservabilityLogs(stream: selectedDiskLogStream)
        } catch {
            operationalSettingsStatus = "Save failed: \(error.localizedDescription)"
        }
    }

    public func resetOperationalSettingKey(_ key: String) async {
        isSavingOperationalSettings = true
        operationalSettingsStatus = "Resetting \(key)…"
        defer { isSavingOperationalSettings = false }

        do {
            if let remoteBackend = activeRemoteBackend {
                let updated = try await remoteBackend.resetOperationalConfigKey(key)
                applyOperationalConfigResponse(updated)
            } else {
                let updated = try await router.service.resetOperationalConfigKey(key)
                applyOperationalConfigResponse(updated)
            }
            operationalSettingsStatus = "Reset \(key) to default."
            await loadDiskObservabilityLogs(stream: selectedDiskLogStream)
        } catch {
            operationalSettingsStatus = "Reset failed: \(error.localizedDescription)"
        }
    }

    public func loadDiskObservabilityLogs(stream: ObservabilityStream? = nil, limit: Int = 50) async {
        let targetStream = stream ?? selectedDiskLogStream
        selectedDiskLogStream = targetStream

        do {
            let dto: ObservabilityLogsResponseDTO
            if let remoteBackend = activeRemoteBackend {
                dto = try await remoteBackend.readObservabilityLogs(stream: targetStream, limit: limit)
            } else {
                dto = try await router.service.readObservabilityLogs(streamRaw: targetStream.rawValue, limit: limit)
            }
            guard selectedDiskLogStream == targetStream else { return }
            diskLogLevel = dto.level
            diskLogRecords = dto.records
        } catch {
            guard selectedDiskLogStream == targetStream else { return }
            diskLogRecords = []
            operationalSettingsStatus = "Log read failed: \(error.localizedDescription)"
        }
    }

    private func applyOperationalConfigResponse(_ snap: [String: JSONValue]) {
        if case .object(let eff)? = snap["effective"] {
            let level = eff["observability_level"]?.stringValue
                .map(ObservabilityVerbosity.parse) ?? .off
            let retention = eff["observability_retention_days"]?.intValue ?? 30
            let cacheLimit = eff["mlx_cache_limit_gb"]?.doubleValue
            operationalSettings = OperationalSettings(
                observabilityLevel: level,
                observabilityRetentionDays: retention,
                mlxCacheLimitGB: cacheLimit
            )
        }
        operationalMemoryOnlyKeys = snap["memory_only"]?.stringArray ?? []
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
                await syncServerBindingTruth()
                let bindLabel = isServerLANReachable
                    ? "http://\(serverBindHost.isEmpty ? "0.0.0.0" : serverBindHost):\(serverPort) (LAN\(isBonjourAdvertisementActive ? " + Bonjour" : ""))"
                    : "http://\(serverBindHost):\(serverPort) (localhost-only)"
                HeylookObservabilityLog.shared.record(
                    level: "INFO",
                    category: "server",
                    message: "Embedded HTTP/SSE server listening on \(bindLabel)"
                )
                onStatusMessage("Embedded server listening on \(bindLabel)")
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
            cachedProviderSchemas.removeAll()
            fitGateReasonsByModelID.removeAll()
            loadWarmOutcomesByModelID.removeAll()

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
            await refreshMetrics(forceRefresh: true)
            await loadScanConfig()
            await loadOperationalConfigAndDiskLogs()
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
        cachedProviderSchemas.removeAll()
        fitGateReasonsByModelID.removeAll()
        loadWarmOutcomesByModelID.removeAll()

        let currentList = await scheduler.listModels()
        for model in currentList where model.provider == .remote {
            _ = await scheduler.removeModel(model.modelID)
        }
        await rescanConfiguredModels()
        await onRemoteBackendChanged?(nil)
        await loadScanConfig()
        await loadOperationalConfigAndDiskLogs()
        if let previouslyConnected {
            onStatusMessage("Disconnected from remote server \(previouslyConnected).")
        }
    }

    // MARK: - Model Lifecycle, Watch Folders (`scan-config`), RAM Fit, Config Patch & Templates

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

    public func assessModelFitReport(
        id: String,
        configOverrides: [String: JSONValue],
        headroomGB: Double = 8.0
    ) async throws -> RAMFitEvaluator.FitReport {
        if let remoteBackend = activeRemoteBackend {
            return try await remoteBackend.assessModelFit(
                id: id,
                configOverrides: configOverrides,
                headroomGB: headroomGB
            )
        }
        return try await router.service.assessModelFit(
            id: id,
            configOverrides: configOverrides,
            headroomGB: headroomGB
        )
    }

    public func fetchModelStatusReading(id: String) async -> (memoryMB: Double?, memorySource: String?) {
        let status: [String: JSONValue]
        if let remoteBackend = activeRemoteBackend {
            status = await remoteBackend.modelStatus(id: id)
        } else {
            status = await router.service.modelStatus(id: id)
        }
        let rawMB = status["memory_mb"]?.doubleValue
        let validMB = (rawMB != nil && rawMB! > 0) ? rawMB : nil
        return (validMB, status["memory_source"]?.stringValue)
    }

    public func updateFitGate(modelID: String, reason: String?) {
        if let reason, !reason.isEmpty {
            fitGateReasonsByModelID[modelID] = reason
        } else {
            fitGateReasonsByModelID.removeValue(forKey: modelID)
        }
    }

    private func refreshFitGatesForUnloadedModels() async {
        let snapshot = registeredModels
        var updated: [String: String] = [:]
        for model in snapshot where !model.isLoaded && !model.isSpeechRecognizer {
            do {
                let report = try await assessModelFitReport(id: model.modelID, configOverrides: [:], headroomGB: 8.0)
                if report.verdict == "fail" {
                    updated[model.modelID] = "Does not fit memory (see Configure)"
                }
            } catch {
                // Never block Load on missing fit info (422 / unavailable).
            }
        }
        fitGateReasonsByModelID = updated
    }

    public func applyQuickContextSize(_ ctxSize: Int, for model: EngineDescription) async {
        guard activeRemoteBackend != nil || !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return
        }
        if model.provider == .remote || activeRemoteBackend != nil, let remoteBackend = activeRemoteBackend {
            do {
                let patch = ModelConfigPatchPayload(config: ["ctx_size": .int(ctxSize)])
                let res = try await remoteBackend.patchAdminModel(id: model.modelID, updates: patch)
                adminRowsByID[model.modelID] = res.model
                await refreshMetrics(forceRefresh: true)
                onStatusMessage("Updated ctx_size to \(ctxSize) on remote \(model.modelID).")
            } catch {
                modelsErrorBanner = "Failed to update remote context size: \(error.localizedDescription)"
                onStatusMessage(modelsErrorBanner)
            }
            return
        }
        do {
            _ = try await patchModelConfig(id: model.modelID, patch: ["ctx_size": .int(ctxSize)])
            onStatusMessage("Updated ctx_size to \(ctxSize) in model.heylook.toml for \(model.modelID).")
        } catch {
            var sidecar = loadSidecarConfig(for: model)
            sidecar.ctxSize = ctxSize
            await saveModelSidecar(sidecar, for: model)
        }
    }

    public func loadScanConfig() async {
        let payload: ScanConfigPayload
        if let remoteBackend = activeRemoteBackend {
            do {
                payload = try await remoteBackend.getScanConfig()
            } catch {
                scanConfigNote = "Could not read watch folders: \(error.localizedDescription)"
                return
            }
        } else {
            payload = await router.service.getScanConfig()
        }
        scanConfig = payload
        scanFoldersDraftText = payload.folders.joined(separator: "\n")
        scanIntervalDraft = payload.scanIntervalSeconds
        if payload.scanIntervalSeconds == 0 {
            scanConfigNote = "Discovery is off (scan_interval_seconds = 0) — manual Re-scan only."
        } else {
            scanConfigNote = nil
        }
    }

    public func saveScanConfig(folders: [String]?, scanIntervalSeconds: Int?) async {
        guard activeRemoteBackend != nil || !HeylookConfig.isReadOnlyModelConfig else {
            let msg = "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1)."
            scanConfigNote = msg
            onStatusMessage(msg)
            return
        }
        guard !isSavingScanConfig else { return }
        isSavingScanConfig = true
        scanConfigNote = "Saving…"
        defer { isSavingScanConfig = false }

        do {
            let updated: ScanConfigPayload
            if let remoteBackend = activeRemoteBackend {
                updated = try await remoteBackend.updateScanConfig(
                    folders: folders,
                    scanIntervalSeconds: scanIntervalSeconds
                )
                await refreshMetrics(forceRefresh: true)
            } else {
                updated = try await router.service.updateScanConfig(
                    folders: folders,
                    scanIntervalSeconds: scanIntervalSeconds
                )
                config = HeylookConfig.loadDefault()
                await refreshMetrics(forceRefresh: true)
            }
            scanConfig = updated
            scanFoldersDraftText = updated.folders.joined(separator: "\n")
            scanIntervalDraft = updated.scanIntervalSeconds
            modelsErrorBanner = nil
            let baseNote = updated.warning ?? "Saved — \(updated.modelsServed) models served."
            scanConfigNote = updated.scanIntervalSeconds == 0
                ? "\(baseNote) Discovery is off (scan_interval_seconds = 0)."
                : baseNote
            onStatusMessage(scanConfigNote)
        } catch {
            let msg = "Save failed: \(error.localizedDescription)"
            scanConfigNote = msg
            modelsErrorBanner = msg
            onStatusMessage(msg)
        }
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
            await loadScanConfig()
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
            await loadScanConfig()
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
        if !model.isLoaded, let gateReason = fitGateReasonsByModelID[model.modelID] {
            let msg = "Not loading \(model.modelID): \(gateReason)"
            modelsErrorBanner = msg
            onStatusMessage(msg)
            return
        }

        if model.provider == .remote || activeRemoteBackend != nil, let remoteBackend = activeRemoteBackend {
            guard !loadingModelIDs.contains(model.modelID) else { return }
            loadingModelIDs.insert(model.modelID)
            defer { loadingModelIDs.remove(model.modelID) }
            do {
                if model.isLoaded {
                    _ = try await remoteBackend.unloadModel(id: model.modelID)
                    loadWarmOutcomesByModelID.removeValue(forKey: model.modelID)
                    modelsErrorBanner = nil
                    modelsStatusNote = "Unloaded remote model \(model.modelID)."
                    onStatusMessage(modelsStatusNote)
                } else {
                    let result = try await remoteBackend.loadModel(id: model.modelID, warm: true)
                    recordLoadOutcome(result, actionLabel: "Loaded", modelID: model.modelID)
                }
                await refreshMetrics(forceRefresh: true)
            } catch {
                await refreshMetrics(forceRefresh: true)
                let msg = "Remote model operation failed for \(model.modelID): \(error.localizedDescription)"
                modelsErrorBanner = msg
                onStatusMessage(msg)
            }
            return
        }

        if model.isLoaded {
            do {
                _ = try await router.service.unloadModel(id: model.modelID)
                loadWarmOutcomesByModelID.removeValue(forKey: model.modelID)
                modelsErrorBanner = nil
                modelsStatusNote = "Unloaded \(model.modelID)."
                onStatusMessage(modelsStatusNote)
            } catch {
                let msg = "Failed to unload \(model.modelID): \(error.localizedDescription)"
                modelsErrorBanner = msg
                onStatusMessage(msg)
            }
            await refreshMetrics(forceRefresh: true)
        } else {
            guard !loadingModelIDs.contains(model.modelID) else { return }
            loadingModelIDs.insert(model.modelID)
            defer { loadingModelIDs.remove(model.modelID) }

            do {
                let result = try await router.service.loadModel(id: model.modelID, warm: true)
                recordLoadOutcome(result, actionLabel: "Loaded", modelID: model.modelID)
                await refreshMetrics(forceRefresh: true)
            } catch {
                await refreshMetrics(forceRefresh: true)
                let msg = "Failed to load \(model.modelID): \(error.localizedDescription)"
                modelsErrorBanner = msg
                onStatusMessage(msg)
            }
        }
    }

    public func reloadModel(_ model: EngineDescription, warm: Bool = true) async {
        guard !loadingModelIDs.contains(model.modelID) else { return }
        loadingModelIDs.insert(model.modelID)
        defer { loadingModelIDs.remove(model.modelID) }

        do {
            let result: ModelLoadResult
            if model.provider == .remote || activeRemoteBackend != nil, let remoteBackend = activeRemoteBackend {
                result = try await remoteBackend.reloadModel(id: model.modelID, loadSettings: nil, warm: warm)
            } else {
                result = try await router.service.reloadModel(id: model.modelID, loadSettings: nil, warm: warm)
            }
            recordLoadOutcome(result, actionLabel: "Reloaded", modelID: model.modelID)
            await refreshMetrics(forceRefresh: true)
        } catch {
            await refreshMetrics(forceRefresh: true)
            let msg = "Reload failed for \(model.modelID): \(error.localizedDescription)"
            modelsErrorBanner = msg
            onStatusMessage(msg)
        }
    }

    private func recordLoadOutcome(_ result: ModelLoadResult, actionLabel: String, modelID: String) {
        if let warmErr = result.warmError, !warmErr.isEmpty {
            let msg = "\(actionLabel), but warm-up failed: \(warmErr)"
            loadWarmOutcomesByModelID[modelID] = msg
            modelsErrorBanner = msg
            onStatusMessage(msg)
        } else if let warmMs = result.warmMs {
            let note = "Loaded & warmed in \(warmMs)ms"
            loadWarmOutcomesByModelID[modelID] = note
            modelsErrorBanner = nil
            modelsStatusNote = "\(modelID): \(note)"
            onStatusMessage("\(modelID): \(note)")
        } else {
            loadWarmOutcomesByModelID[modelID] = actionLabel
            modelsErrorBanner = nil
            modelsStatusNote = "\(actionLabel) \(modelID)."
            onStatusMessage("\(actionLabel) \(modelID).")
        }
    }

    public func unloadAllModels() async {
        await scheduler.unloadAllModels()
        await onUnloadEmbeddingEngine?()
        loadWarmOutcomesByModelID.removeAll()
        await refreshMetrics(forceRefresh: true)
        onStatusMessage("Unloaded all models, cleared Metal/ANE caches, and terminated background subprocesses.")
    }

    public func rescanConfiguredModels() async {
        if let remoteBackend = activeRemoteBackend {
            await refreshRemoteMetrics(using: remoteBackend, forceRefresh: true)
            await loadScanConfig()
            onStatusMessage("Refreshed remote models from \(connectedRemoteServerURL ?? "remote server").")
            return
        }
        config = HeylookConfig.loadDefault()
        let discovered = config.discoverConfiguredModels()
        await scheduler.reloadRegistry(discovered)
        registeredModels = await scheduler.listModels()
        if selectedModelID.isEmpty, let first = generationModels.first ?? registeredModels.first {
            selectedModelID = first.modelID
        }
        await refreshMetrics(forceRefresh: true)
        onStatusMessage("Rescanned heylook.toml (\(registeredModels.count) models across \(config.scanFolders.count) scan folders).")
    }

    public func unloadGGUFAllocation(_ allocationEntry: String) async {
        let modelID = allocationEntry.components(separatedBy: "::").first ?? allocationEntry
        await scheduler.unloadModel(modelID)
        GGUFPortPool.shared.releasePort(for: modelID)
        loadWarmOutcomesByModelID.removeValue(forKey: modelID)
        await refreshMetrics(forceRefresh: true)
        onStatusMessage("Unloaded GGUF subprocess for \(modelID).")
    }

    @discardableResult
    public func patchModelConfig(id: String, patch: [String: JSONValue]) async throws -> AdminModelPatchResult {
        if let remoteBackend = activeRemoteBackend {
            let provider = adminRowsByID[id]?.provider ?? registeredModels.first(where: { $0.modelID == id })?.provider ?? .mlx
            let fields = (try? await schemaFields(for: provider)) ?? []
            let fileOnlySet = Set(fields.filter(\.isFileOnly).map(\.name))
            let rejected = patch.keys.filter { fileOnlySet.contains($0) }.sorted()
            if !rejected.isEmpty {
                throw WorkbenchError.unprocessable(
                    message: "Refused over remote admin: \(rejected.joined(separator: ", ")) is file-only and must be edited in model.heylook.toml or heylook.toml on the host.",
                    field: rejected.first
                )
            }
            let payload = ModelConfigPatchPayload(config: patch)
            let result = try await remoteBackend.patchAdminModel(id: id, updates: payload)
            adminRowsByID[id] = result.model
            modelsErrorBanner = nil
            await refreshMetrics(forceRefresh: true)
            onSchedulePromptPreview?()
            return result
        }

        guard !HeylookConfig.isReadOnlyModelConfig else {
            throw WorkbenchError.conflict(
                code: "readonly_config",
                message: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1)."
            )
        }
        let payload = ModelConfigPatchPayload(config: patch)
        let result = try await router.service.patchAdminModel(id: id, updates: payload)
        if patch["pinned"] == .null {
            await scheduler.setPinned(false, for: id)
        }
        adminRowsByID[id] = result.model
        modelsErrorBanner = nil
        await refreshMetrics(forceRefresh: true)
        onSchedulePromptPreview?()
        return result
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
            await refreshMetrics(forceRefresh: true)
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

    public func fetchChatTemplateState(for model: EngineDescription) async throws -> HeylookConfig.ChatTemplateState {
        if model.provider == .remote || activeRemoteBackend != nil, let remoteBackend = activeRemoteBackend {
            return try await remoteBackend.getChatTemplate(id: model.modelID)
        }
        return try await router.service.getChatTemplate(id: model.modelID)
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
        if model.provider == .remote || activeRemoteBackend != nil, let remoteBackend = activeRemoteBackend {
            let state = try await remoteBackend.setChatTemplate(id: model.modelID, template: template)
            templateDraftsByModelID.removeValue(forKey: model.modelID)
            await refreshMetrics(forceRefresh: true)
            onSchedulePromptPreview?()
            onStatusMessage("Saved remote chat template override for \(model.modelID).")
            return state
        }
        guard !HeylookConfig.isReadOnlyModelConfig else {
            throw ProviderExecutionError.invalidRequest(detail: "Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
        }
        let state = try await router.service.setChatTemplate(id: model.modelID, template: template)
        templateDraftsByModelID.removeValue(forKey: model.modelID)
        await refreshMetrics(forceRefresh: true)
        onSchedulePromptPreview?()
        onStatusMessage("Saved chat_template.heylook.jinja override for \(model.modelID).")
        return state
    }

    @discardableResult
    public func revertChatTemplateOverride(for model: EngineDescription) async -> HeylookConfig.ChatTemplateState {
        if model.provider == .remote || activeRemoteBackend != nil, let remoteBackend = activeRemoteBackend {
            if let state = try? await remoteBackend.deleteChatTemplate(id: model.modelID) {
                templateDraftsByModelID.removeValue(forKey: model.modelID)
                await refreshMetrics(forceRefresh: true)
                onSchedulePromptPreview?()
                onStatusMessage("Reverted remote chat template override for \(model.modelID) (active: \(state.source)).")
                return state
            }
        }
        guard !HeylookConfig.isReadOnlyModelConfig else {
            onStatusMessage("Model configuration is read-only (HEYLOOK_READONLY_MODEL_CONFIG=1).")
            return loadChatTemplateState(for: model)
        }
        if let state = try? await router.service.deleteChatTemplate(id: model.modelID) {
            templateDraftsByModelID.removeValue(forKey: model.modelID)
            await refreshMetrics(forceRefresh: true)
            onSchedulePromptPreview?()
            onStatusMessage("Reverted chat_template.heylook.jinja override for \(model.modelID) (active: \(state.source)).")
            return state
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
        templateDraftsByModelID.removeValue(forKey: model.modelID)
        await refreshMetrics(forceRefresh: true)
        onSchedulePromptPreview?()
        onStatusMessage("Reverted chat_template.heylook.jinja override for \(model.modelID) (active: \(reverted.source)).")
        return reverted
    }

    public func hasUnsavedDraft(for modelID: String) -> Bool {
        if let tmpl = templateDraftsByModelID[modelID],
           !tmpl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        if let cfgDraft = configDraftsByModelID[modelID], !cfgDraft.isEmpty {
            return true
        }
        return false
    }

    public func clearDrafts(for modelID: String) {
        configDraftsByModelID.removeValue(forKey: modelID)
        templateDraftsByModelID.removeValue(forKey: modelID)
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
