import Darwin
import Foundation
import HeylookCore
import Metal

public enum ComputeSchedulerError: LocalizedError, Sendable, Equatable {
    case queueOverloaded(depth: Int, maxDepth: Int)
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case let .queueOverloaded(depth, maxDepth):
            return "model_overloaded: GPU generation queue is full (\(depth)/\(maxDepth))"
        case let .modelNotFound(modelID):
            return "model_not_found: '\(modelID)' is not registered"
        }
    }
}

public enum ProviderExecutionError: LocalizedError, Sendable, Equatable {
    case checkpointNotFound(modelID: String, path: String)
    case binaryNotFound(modelID: String, binaryPath: String)
    case runtimeUnavailable(modelID: String, detail: String)
    case invalidRequest(detail: String)

    public var errorDescription: String? {
        switch self {
        case let .checkpointNotFound(modelID, path):
            return "checkpoint_not_found: Model '\(modelID)' weights not found at '\(path)'"
        case let .binaryNotFound(modelID, binaryPath):
            return "binary_not_found: Executable for '\(modelID)' not found at '\(binaryPath)'"
        case let .runtimeUnavailable(modelID, detail):
            return "runtime_unavailable: Cannot execute '\(modelID)' (\(detail))"
        case let .invalidRequest(detail):
            return "invalid_request: \(detail)"
        }
    }
}

/// Unified provider protocol implemented by `MLXSwiftProvider`, `CoreAIProvider`,
/// `GGUFSubprocessProvider`, and `RemoteHeylookProvider`.
public protocol InferenceProvider: Sendable {
    var modelID: String { get }
    var providerKind: EngineProviderKind { get }

    func describe() async -> EngineDescription

    /// Eagerly loads weights / spawns runtime subprocesses and verifies readiness.
    func preload() async throws

    /// Clears KV / prompt / Metal caches without unloading model weights.
    func clearCache() async -> Bool

    func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) async -> AsyncThrowingStream<GenerationChunk, Error>

    /// Releases loaded weights, clears KV/Metal caches, and terminates any child subprocesses.
    func unload() async
}

public extension InferenceProvider {
    func preload() async throws {}
    func clearCache() async -> Bool { false }
    func unload() async {}
}

/// Hardware memory and CPU telemetry for Apple Silicon Unified Memory.
public enum HardwareMemoryMonitor {
    public struct MemorySnapshot: Codable, Sendable, Equatable {
        public let totalPhysicalRAMGB: Double
        public let metalRecommendedMaxWorkingSetGB: Double
        public let metalCurrentAllocatedMB: Int
        public let processResidentMB: Int
        public let deviceName: String
        public let ramUsedGB: Double
        public let ramAvailableGB: Double
        public let cpuPercent: Double

        public init(
            totalPhysicalRAMGB: Double,
            metalRecommendedMaxWorkingSetGB: Double,
            metalCurrentAllocatedMB: Int,
            processResidentMB: Int,
            deviceName: String,
            ramUsedGB: Double = 0,
            ramAvailableGB: Double = 0,
            cpuPercent: Double = 0
        ) {
            self.totalPhysicalRAMGB = totalPhysicalRAMGB
            self.metalRecommendedMaxWorkingSetGB = metalRecommendedMaxWorkingSetGB
            self.metalCurrentAllocatedMB = metalCurrentAllocatedMB
            self.processResidentMB = processResidentMB
            self.deviceName = deviceName
            self.ramUsedGB = ramUsedGB
            self.ramAvailableGB = ramAvailableGB
            self.cpuPercent = cpuPercent
        }
    }

    public static func snapshot() -> MemorySnapshot {
        let physBytes = ProcessInfo.processInfo.physicalMemory
        let physGB = Double(physBytes) / (1024.0 * 1024.0 * 1024.0)

        let device = MTLCreateSystemDefaultDevice()
        let recommendedGB = Double(device?.recommendedMaxWorkingSetSize ?? 0) / (1024.0 * 1024.0 * 1024.0)
        let currentMetalMB = Int((device?.currentAllocatedSize ?? 0) / (1024 * 1024))
        let devName = device?.name ?? "Apple Silicon"

        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let rssMB = kerr == KERN_SUCCESS ? Int(info.resident_size / (1024 * 1024)) : 0

        // Query system-wide Mach VM page counters for real ramUsedGB & ramAvailableGB
        var vmStats = vm_statistics64()
        var vmCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        var pageSize: vm_size_t = 16384
        host_page_size(mach_host_self(), &pageSize)
        let vmRes: kern_return_t = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &vmCount)
            }
        }
        let ramUsedGB: Double
        let ramAvailableGB: Double
        if vmRes == KERN_SUCCESS {
            let pageBytes = Double(pageSize)
            let usedPages = Double(vmStats.active_count)
                + Double(vmStats.wire_count)
                + Double(vmStats.compressor_page_count)
            let used = (usedPages * pageBytes) / (1024.0 * 1024.0 * 1024.0)
            ramUsedGB = min(physGB, (used * 10).rounded() / 10)
            ramAvailableGB = max(0, ((physGB - ramUsedGB) * 10).rounded() / 10)
        } else {
            let approxUsed = Double(rssMB) / 1024.0
            ramUsedGB = (approxUsed * 10).rounded() / 10
            ramAvailableGB = max(0, ((physGB - approxUsed) * 10).rounded() / 10)
        }

        // Normalized CPU utilization from 1-minute loadavg across active logical cores
        var loadAvg: Double = 0
        let nCores = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let cpuPct: Double
        if getloadavg(&loadAvg, 1) == 1 {
            cpuPct = min(100.0, max(0.0, ((loadAvg / Double(nCores)) * 1000.0).rounded() / 10.0))
        } else {
            cpuPct = 0.0
        }

        return MemorySnapshot(
            totalPhysicalRAMGB: (physGB * 10).rounded() / 10,
            metalRecommendedMaxWorkingSetGB: (recommendedGB * 10).rounded() / 10,
            metalCurrentAllocatedMB: currentMetalMB,
            processResidentMB: rssMB,
            deviceName: devName,
            ramUsedGB: ramUsedGB,
            ramAvailableGB: ramAvailableGB,
            cpuPercent: cpuPct
        )
    }
}

/// Dual-lane Heterogeneous Compute Scheduler (`GPU GenerationGate` + `ANE Co-Residency Lane`)
/// and LRU / Pinned model lifecycle manager.
public actor ComputeScheduler {
    public struct SchedulerMetrics: Codable, Sendable, Equatable {
        public let gpuGateBusy: Bool
        public let gpuQueueDepth: Int
        public let maxQueueDepth: Int
        public let activeANETasks: Int
        public let loadedGPUModels: [String]
        public let loadedANEModels: [String]
        public let pinnedModels: [String]

        public init(
            gpuGateBusy: Bool = false,
            gpuQueueDepth: Int = 0,
            maxQueueDepth: Int = 2,
            activeANETasks: Int = 0,
            loadedGPUModels: [String] = [],
            loadedANEModels: [String] = [],
            pinnedModels: [String] = []
        ) {
            self.gpuGateBusy = gpuGateBusy
            self.gpuQueueDepth = gpuQueueDepth
            self.maxQueueDepth = maxQueueDepth
            self.activeANETasks = activeANETasks
            self.loadedGPUModels = loadedGPUModels
            self.loadedANEModels = loadedANEModels
            self.pinnedModels = pinnedModels
        }
    }

    private var providers: [String: any InferenceProvider] = [:]
    private var descriptions: [String: EngineDescription] = [:]
    private var lruOrderGPU: [String] = []
    private var loadedANE: Set<String> = []
    private var maxLoadedGPUModels: Int
    private var maxQueueDepth: Int

    private var gpuGateBusy: Bool = false
    private var gpuWaiters: [CheckedContinuation<Void, Never>] = []
    private var activeANETasks: Int = 0

    public init(maxLoadedGPUModels: Int = 3, maxQueueDepth: Int = 2) {
        self.maxLoadedGPUModels = max(1, maxLoadedGPUModels)
        self.maxQueueDepth = max(1, maxQueueDepth)
    }

    public func reconfigureLimits(maxLoadedGPUModels: Int? = nil, maxQueueDepth: Int? = nil) {
        if let maxLoadedGPUModels {
            self.maxLoadedGPUModels = max(1, maxLoadedGPUModels)
        }
        if let maxQueueDepth {
            self.maxQueueDepth = max(1, maxQueueDepth)
        }
    }

    public func register(provider: any InferenceProvider, description: EngineDescription) async {
        providers[description.modelID] = provider
        descriptions[description.modelID] = description
        if description.isLoaded {
            let evicted = recordResidency(for: description)
            for evictedID in evicted {
                await providers[evictedID]?.unload()
            }
        }
    }

    /// Registers a batch of discovered `EngineDescription` entries using `ProviderFactory`.
    public func registerDiscoveredModels(
        _ discovered: [EngineDescription],
        allowSyntheticFallback: Bool = false
    ) async {
        for desc in discovered {
            let provider = ProviderFactory.makeProvider(for: desc, allowSyntheticFallback: allowSyntheticFallback)
            await register(provider: provider, description: desc)
        }
    }

    /// Replaces the registry with `discovered` (preserving currently loaded providers whose IDs match
    /// unless `unloadRemoved` or `forceReload` is requested).
    public func reloadRegistry(
        _ discovered: [EngineDescription],
        allowSyntheticFallback: Bool = false
    ) async {
        let newIDs = Set(discovered.map(\.modelID))
        let removedIDs = Set(descriptions.keys).subtracting(newIDs)
        for id in removedIDs {
            await unloadModel(id)
            providers.removeValue(forKey: id)
            descriptions.removeValue(forKey: id)
        }
        for var desc in discovered {
            if let existing = descriptions[desc.modelID], existing.isLoaded {
                desc.isLoaded = true
                descriptions[desc.modelID] = desc
            } else {
                let provider = ProviderFactory.makeProvider(for: desc, allowSyntheticFallback: allowSyntheticFallback)
                await register(provider: provider, description: desc)
            }
        }
    }

    public func removeModel(_ modelID: String) async -> Bool {
        guard descriptions[modelID] != nil else { return false }
        await unloadModel(modelID)
        providers.removeValue(forKey: modelID)
        descriptions.removeValue(forKey: modelID)
        return true
    }

    public func clearAllCaches() async -> Int {
        var cleared = 0
        for (_, provider) in providers {
            if await provider.clearCache() {
                cleared += 1
            }
        }
        return cleared
    }

    public func listModels() -> [EngineDescription] {
        descriptions.values.sorted { $0.modelID.localizedStandardCompare($1.modelID) == .orderedAscending }
    }

    public func modelDescription(for id: String) -> EngineDescription? {
        descriptions[id]
    }

    public func setPinned(_ pinned: Bool, for modelID: String) {
        guard var desc = descriptions[modelID] else { return }
        desc.isPinned = pinned
        descriptions[modelID] = desc
    }

    /// Applies an updated `ModelSidecarConfig` (`model.heylook.toml`) to `modelID` in-place,
    /// marking modified fields with `.configured` provenance and preserving `reasoningFormat`
    /// and `thinkingControls`.
    @discardableResult
    public func applySidecarUpdate(_ sidecar: ModelSidecarConfig, for modelID: String) throws -> EngineDescription {
        guard let existing = descriptions[modelID] else {
            throw ComputeSchedulerError.modelNotFound(modelID)
        }
        let updatedControls: ThinkingControls?
        if let explicitThinking = sidecar.supportsThinking, let controls = existing.thinkingControls {
            updatedControls = ThinkingControls(
                supportsThinking: explicitThinking,
                canDisableThinking: controls.canDisableThinking,
                defaultEnabled: controls.defaultEnabled,
                switchVariable: controls.switchVariable,
                depthVariable: controls.depthVariable,
                depthValues: controls.depthValues
            )
        } else {
            updatedControls = existing.thinkingControls
        }
        let updatedTemplateSource: ProvenanceField<String>
        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: existing.modelPath)
        let fallbackTemplateSource = existing.templateSource.value == "chat_template.heylook.jinja"
            ? (existing.provider == .gguf ? "gguf_embedded" : "tokenizer_config.json")
            : existing.templateSource.value
        let inspectedChatTemplate = HeylookConfig.inspectChatTemplate(
            in: dirURL,
            fallbackSource: fallbackTemplateSource
        )
        if let customTemplate = sidecar.chatTemplatePath, !customTemplate.isEmpty {
            updatedTemplateSource = ProvenanceField(
                value: customTemplate,
                provenance: .configured,
                effect: .reload
            )
        } else if inspectedChatTemplate.hasOverride {
            updatedTemplateSource = ProvenanceField(
                value: inspectedChatTemplate.source,
                provenance: .configured,
                effect: .reload
            )
        } else if existing.templateSource.value == "chat_template.heylook.jinja" {
            updatedTemplateSource = ProvenanceField(
                value: inspectedChatTemplate.source,
                provenance: .derived,
                effect: .reload
            )
        } else {
            updatedTemplateSource = existing.templateSource
        }

        let updatedDrafter: ProvenanceField<String?>
        if sidecar.unsetKeys.contains("draft_model") || sidecar.unsetKeys.contains("draft_model_path") {
            updatedDrafter = ProvenanceField(value: nil, provenance: .configured, effect: .reload)
        } else if let draftModel = sidecar.draftModel, !draftModel.isEmpty {
            updatedDrafter = ProvenanceField(value: draftModel, provenance: .configured, effect: .reload)
        } else {
            updatedDrafter = existing.speculativeDrafter
        }

        let updated = EngineDescription(
            modelID: existing.modelID,
            provider: sidecar.provider ?? existing.provider,
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
            thinkingBudget: sidecar.sampling.thinkingBudget.map {
                ProvenanceField(value: $0, provenance: .configured, effect: .perRequest)
            } ?? existing.thinkingBudget,
            reasoningFormat: existing.reasoningFormat,
            thinkingControls: updatedControls,
            templateSource: updatedTemplateSource,
            speculativeDrafter: updatedDrafter,
            estimatedMemoryMB: existing.estimatedMemoryMB,
            tags: sidecar.tags.isEmpty ? existing.tags : sidecar.tags
        )
        descriptions[modelID] = updated
        return updated
    }

    /// Marks `modelID` as loaded (and when `warm == true`, invokes `provider.preload()` first so
    /// weights / subprocesses are genuinely loaded before `isLoaded` becomes `true`) and enforces
    /// LRU eviction on unpinned GPU models while leaving ANE-resident models untouched.
    public func activateModel(_ modelID: String, warm: Bool = false) async throws -> EngineDescription {
        guard var desc = descriptions[modelID] else {
            throw ComputeSchedulerError.modelNotFound(modelID)
        }
        let evictedIDs = planEvictionsIfActivating(desc)
        for evictedID in evictedIDs {
            await unloadModel(evictedID)
        }
        if warm, let provider = providers[modelID] {
            try await provider.preload()
            var refreshed = await provider.describe()
            refreshed.isLoaded = true
            refreshed.isPinned = desc.isPinned
            desc = refreshed
        } else {
            desc.isLoaded = true
        }
        descriptions[modelID] = desc
        let postEvicted = recordResidency(for: desc)
        for evictedID in postEvicted {
            await providers[evictedID]?.unload()
        }
        return desc
    }

    public func unloadModel(_ modelID: String) async {
        guard var desc = descriptions[modelID] else { return }
        desc.isLoaded = false
        descriptions[modelID] = desc
        lruOrderGPU.removeAll { $0 == modelID }
        loadedANE.remove(modelID)
        await providers[modelID]?.unload()
    }

    /// Unloads all registered models, freeing Metal GPU memory, clearing KV caches,
    /// and terminating all `llama-server` subprocesses.
    public func unloadAllModels() async {
        let allIDs = Array(descriptions.keys)
        for id in allIDs {
            if var desc = descriptions[id] {
                desc.isLoaded = false
                descriptions[id] = desc
            }
        }
        lruOrderGPU.removeAll()
        loadedANE.removeAll()
        for (_, provider) in providers {
            await provider.unload()
        }
    }

    /// Acquires a compute lease on either the **ANE Co-Resident Lane** (concurrent, zero GPU contention)
    /// or the **Metal GPU GenerationGate** (serialized FIFO with bounded `maxQueueDepth`).
    public func withComputeLease<T: Sendable>(
        for modelID: String,
        operation: @Sendable () async throws -> T
    ) async throws -> T {
        let desc = try await activateModel(modelID, warm: false)
        let usesANELane = desc.computePlacement.value.runsOnNeuralEngineLane

        if usesANELane {
            activeANETasks += 1
            defer { activeANETasks -= 1 }
            return try await operation()
        }

        // Metal GPU FIFO Gate
        if gpuGateBusy {
            guard gpuWaiters.count < maxQueueDepth else {
                throw ComputeSchedulerError.queueOverloaded(
                    depth: gpuWaiters.count,
                    maxDepth: maxQueueDepth
                )
            }
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                gpuWaiters.append(cont)
            }
        } else {
            gpuGateBusy = true
        }

        defer {
            if !gpuWaiters.isEmpty {
                let next = gpuWaiters.removeFirst()
                next.resume()
            } else {
                gpuGateBusy = false
            }
        }
        return try await operation()
    }

    public func provider(for modelID: String) throws -> any InferenceProvider {
        guard let p = providers[modelID] else {
            throw ComputeSchedulerError.modelNotFound(modelID)
        }
        return p
    }

    public func metrics() -> SchedulerMetrics {
        let pinned = descriptions.values.filter(\.isPinned).map(\.modelID).sorted()
        return SchedulerMetrics(
            gpuGateBusy: gpuGateBusy,
            gpuQueueDepth: gpuWaiters.count,
            maxQueueDepth: maxQueueDepth,
            activeANETasks: activeANETasks,
            loadedGPUModels: lruOrderGPU,
            loadedANEModels: loadedANE.sorted(),
            pinnedModels: pinned
        )
    }

    private func planEvictionsIfActivating(_ desc: EngineDescription) -> [String] {
        guard !desc.computePlacement.value.runsOnNeuralEngineLane else { return [] }
        if lruOrderGPU.contains(desc.modelID) { return [] }
        var simulated = lruOrderGPU + [desc.modelID]
        var toEvict: [String] = []
        while simulated.count > maxLoadedGPUModels {
            guard let candidateIdx = simulated.firstIndex(where: { id in
                descriptions[id]?.isPinned != true && id != desc.modelID
            }) else {
                break
            }
            toEvict.append(simulated.remove(at: candidateIdx))
        }
        return toEvict
    }

    @discardableResult
    private func recordResidency(for desc: EngineDescription) -> [String] {
        if desc.computePlacement.value.runsOnNeuralEngineLane {
            loadedANE.insert(desc.modelID)
            return []
        }
        lruOrderGPU.removeAll { $0 == desc.modelID }
        lruOrderGPU.append(desc.modelID)

        var evictedIDs: [String] = []
        while lruOrderGPU.count > maxLoadedGPUModels {
            guard let candidateIdx = lruOrderGPU.firstIndex(where: { id in
                descriptions[id]?.isPinned != true && id != desc.modelID
            }) else {
                break
            }
            let evictedID = lruOrderGPU.remove(at: candidateIdx)
            if var evicted = descriptions[evictedID] {
                evicted.isLoaded = false
                descriptions[evictedID] = evicted
            }
            evictedIDs.append(evictedID)
        }
        return evictedIDs
    }
}
