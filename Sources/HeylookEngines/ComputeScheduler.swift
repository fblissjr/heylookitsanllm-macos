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

/// Unified provider protocol implemented by `MLXSwiftProvider`, `CoreAIProvider`,
/// `GGUFSubprocessProvider`, and `RemoteHeylookProvider`.
public protocol InferenceProvider: Sendable {
    var modelID: String { get }
    var providerKind: EngineProviderKind { get }

    func describe() async -> EngineDescription

    func generate(
        requestID: String,
        request: MessagesRequest,
        effectiveSampling: SamplerSettings,
        registry: RequestRegistry
    ) async -> AsyncThrowingStream<GenerationChunk, Error>
}

/// Hardware memory telemetry for Apple Silicon Unified Memory (e.g. 192 GB M2 Ultra Mac Studio).
public enum HardwareMemoryMonitor {
    public struct MemorySnapshot: Codable, Sendable, Equatable {
        public let totalPhysicalRAMGB: Double
        public let metalRecommendedMaxWorkingSetGB: Double
        public let metalCurrentAllocatedMB: Int
        public let processResidentMB: Int
        public let deviceName: String

        public init(
            totalPhysicalRAMGB: Double,
            metalRecommendedMaxWorkingSetGB: Double,
            metalCurrentAllocatedMB: Int,
            processResidentMB: Int,
            deviceName: String
        ) {
            self.totalPhysicalRAMGB = totalPhysicalRAMGB
            self.metalRecommendedMaxWorkingSetGB = metalRecommendedMaxWorkingSetGB
            self.metalCurrentAllocatedMB = metalCurrentAllocatedMB
            self.processResidentMB = processResidentMB
            self.deviceName = deviceName
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

        return MemorySnapshot(
            totalPhysicalRAMGB: (physGB * 10).rounded() / 10,
            metalRecommendedMaxWorkingSetGB: (recommendedGB * 10).rounded() / 10,
            metalCurrentAllocatedMB: currentMetalMB,
            processResidentMB: rssMB,
            deviceName: devName
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

    public func register(provider: any InferenceProvider, description: EngineDescription) {
        providers[description.modelID] = provider
        descriptions[description.modelID] = description
        if description.isLoaded {
            recordResidency(for: description)
        }
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

    /// Marks `modelID` as loaded and enforces LRU eviction on unpinned GPU models while leaving
    /// ANE-resident models (`computePlacement.runsOnNeuralEngineLane`) untouched.
    public func activateModel(_ modelID: String) throws -> EngineDescription {
        guard var desc = descriptions[modelID] else {
            throw ComputeSchedulerError.modelNotFound(modelID)
        }
        desc.isLoaded = true
        descriptions[modelID] = desc
        recordResidency(for: desc)
        return desc
    }

    public func unloadModel(_ modelID: String) {
        guard var desc = descriptions[modelID] else { return }
        desc.isLoaded = false
        descriptions[modelID] = desc
        lruOrderGPU.removeAll { $0 == modelID }
        loadedANE.remove(modelID)
    }

    /// Acquires a compute lease on either the **ANE Co-Resident Lane** (concurrent, zero GPU contention)
    /// or the **Metal GPU GenerationGate** (serialized FIFO with bounded `maxQueueDepth`).
    public func withComputeLease<T: Sendable>(
        for modelID: String,
        operation: @Sendable () async throws -> T
    ) async throws -> T {
        let desc = try activateModel(modelID)
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

    private func recordResidency(for desc: EngineDescription) {
        if desc.computePlacement.value.runsOnNeuralEngineLane {
            loadedANE.insert(desc.modelID)
            return
        }
        lruOrderGPU.removeAll { $0 == desc.modelID }
        lruOrderGPU.append(desc.modelID)

        // Evict oldest unpinned GPU models when unpinned + pinned GPU models exceed maxLoadedGPUModels
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
        }
    }
}
