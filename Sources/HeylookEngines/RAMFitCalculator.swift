import Darwin
import Foundation
import HeylookCore
#if canImport(Metal)
import Metal
#endif

/// Override inputs for interactive `POST /v1/admin/models/{id}/fit` what-if evaluation.
public struct RAMFitOverrides: Sendable, Codable, Equatable {
    public var ctxSize: Int?
    public var cacheTypeK: String?
    public var cacheTypeV: String?
    public var kvBits: Int?
    public var kvScheme: String?
    public var drafterPath: String?
    public var includeDrafter: Bool?

    enum CodingKeys: String, CodingKey {
        case ctxSize = "ctx_size"
        case cacheTypeK = "cache_type_k"
        case cacheTypeV = "cache_type_v"
        case kvBits = "kv_bits"
        case kvScheme = "kv_scheme"
        case drafterPath = "drafter_path"
        case includeDrafter = "include_drafter"
    }

    public init(
        ctxSize: Int? = nil,
        cacheTypeK: String? = nil,
        cacheTypeV: String? = nil,
        kvBits: Int? = nil,
        kvScheme: String? = nil,
        drafterPath: String? = nil,
        includeDrafter: Bool? = nil
    ) {
        self.ctxSize = ctxSize
        self.cacheTypeK = cacheTypeK
        self.cacheTypeV = cacheTypeV
        self.kvBits = kvBits
        self.kvScheme = kvScheme
        self.drafterPath = drafterPath
        self.includeDrafter = includeDrafter
    }
}

/// Live Metal/Mach memory telemetry snapshot consumed by `RAMFitCalculator`.
public struct RAMFitMemorySnapshot: Sendable, Equatable, Codable {
    public let totalGB: Double
    public let reclaimableGB: Double
    public let workingSetGB: Double
    public let maxBufferGB: Double?
    public let wiredLimitMB: Int?

    public init(
        totalGB: Double,
        reclaimableGB: Double,
        workingSetGB: Double,
        maxBufferGB: Double? = nil,
        wiredLimitMB: Int? = nil
    ) {
        self.totalGB = totalGB
        self.reclaimableGB = reclaimableGB
        self.workingSetGB = workingSetGB
        self.maxBufferGB = maxBufferGB
        self.wiredLimitMB = wiredLimitMB
    }
}

/// Truthful Apple Silicon unified-memory fit report matching `ram_fit.py` semantics.
public struct RAMFitReport: Sendable, Equatable, Codable {
    public enum Decision: String, Sendable, Codable {
        case ok
        case thin
        case fail
    }

    public enum WorkingSetDecision: String, Sendable, Codable {
        case ok
        case warn
        case fail
    }

    public let modelID: String
    public let provider: EngineProviderKind
    public let contextLength: Int
    public let weightsGB: Double
    public let mmprojGB: Double
    public let drafterGB: Double
    public let kvGB: Double
    public let totalRequiredGB: Double
    public let reclaimableGB: Double
    public let totalMemoryGB: Double
    public let workingSetGB: Double
    public let maxBufferGB: Double?
    public let wiredLimitMB: Int?
    public let headroomGB: Double
    public let kvHeadroomGB: Double
    public let fits: Bool
    public let decision: Decision
    public let workingSetDecision: WorkingSetDecision
    public let hardWorkingSet: Bool
    public let sysctlSuggestMB: Int?
    public let droppedDrafterForMemory: Bool
    public let recommendedCtxSize: Int
    public let notes: [String]

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case provider
        case contextLength = "context_length"
        case weightsGB = "weights_gb"
        case mmprojGB = "mmproj_gb"
        case drafterGB = "drafter_gb"
        case kvGB = "kv_gb"
        case totalRequiredGB = "total_required_gb"
        case reclaimableGB = "reclaimable_gb"
        case totalMemoryGB = "total_memory_gb"
        case workingSetGB = "working_set_gb"
        case maxBufferGB = "max_buffer_gb"
        case wiredLimitMB = "wired_limit_mb"
        case headroomGB = "headroom_gb"
        case kvHeadroomGB = "kv_headroom_gb"
        case fits
        case decision
        case workingSetDecision = "working_set_decision"
        case hardWorkingSet = "hard_working_set"
        case sysctlSuggestMB = "sysctl_suggest_mb"
        case droppedDrafterForMemory = "dropped_drafter_for_memory"
        case recommendedCtxSize = "recommended_ctx_size"
        case notes
    }
}

/// Unified-memory fit calculator modeled on `src/heylook_llm/ram_fit.py`.
/// Reads Mach `host_statistics64` reclaimable pages (`free + inactive + purgeable`),
/// Metal `recommendedMaxWorkingSetSize` and `maxBufferLength`, and `sysctl iogpu.wired_limit_mb`.
/// Metal working-set overrun is a hard `"fail"` on MLX (weights + KV must remain GPU-wired) and
/// an advisory `"warn"` on GGUF (unless headroom drops below minimum safety bounds).
public enum RAMFitCalculator {
    public static let thinHeadroomGB: Double = 24.0
    private static let gibBytes: Double = 1_073_741_824.0
    private static let mibBytes: Double = 1_048_576.0
    private static let workingSetFloorFraction: Double = 0.72
    private static let workingSetHeadroomReserveGB: Double = 4.0

    public static func liveMemorySnapshot() -> RAMFitMemorySnapshot {
        let totalBytes = Double(ProcessInfo.processInfo.physicalMemory)
        let totalGB = totalBytes / gibBytes

        let reclaimableBytes = liveReclaimableBytes()
        let reclaimableGB = reclaimableBytes > 0 ? Double(reclaimableBytes) / gibBytes : totalGB * 0.75

        var workingSetGB = totalGB * workingSetFloorFraction
        var maxBufferGB: Double?
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            if device.recommendedMaxWorkingSetSize > 0 {
                workingSetGB = Double(device.recommendedMaxWorkingSetSize) / gibBytes
            }
            if device.maxBufferLength > 0 {
                maxBufferGB = Double(device.maxBufferLength) / gibBytes
            }
        }
        #endif

        let wiredLimitMB = sysctlInt32(named: "iogpu.wired_limit_mb").flatMap { $0 > 0 ? $0 : nil }

        return RAMFitMemorySnapshot(
            totalGB: totalGB,
            reclaimableGB: reclaimableGB,
            workingSetGB: workingSetGB,
            maxBufferGB: maxBufferGB,
            wiredLimitMB: wiredLimitMB
        )
    }

    public static func assess(
        for description: EngineDescription,
        overrides: RAMFitOverrides = .init(),
        headroomOverrideGB: Double? = nil,
        snapshot: RAMFitMemorySnapshot? = nil,
        sidecar: ModelSidecarConfig? = nil
    ) -> RAMFitReport {
        let mem = snapshot ?? liveMemorySnapshot()
        let effectiveReclaimableGB = headroomOverrideGB ?? mem.reclaimableGB
        let effectiveWorkingSetGB = headroomOverrideGB ?? mem.workingSetGB

        let modelURL = URL(fileURLWithPath: description.modelPath)
        let resolvedSidecar = sidecar ?? HeylookConfig.loadSidecar(in: modelURL)
        let effectiveCtx = overrides.ctxSize
            ?? resolvedSidecar?.ctxSize
            ?? description.contextSize.value

        let fileSizes = inspectAssetSizesGB(for: description, modelURL: modelURL, sidecar: resolvedSidecar, overrides: overrides)
        let rawWeightsGB = fileSizes.weightsGB
        let mmprojGB = fileSizes.mmprojGB
        let drafterGB = fileSizes.drafterGB

        let fullKVGB = estimateKVBytes(
            for: description,
            modelURL: modelURL,
            contextLength: effectiveCtx,
            overrides: overrides,
            sidecar: resolvedSidecar
        ) / gibBytes

        let hardWorkingSet = description.provider == .mlx

        var includedDrafterGB = drafterGB
        var droppedDrafterForMemory = false
        let baseFootprintWithDrafter = rawWeightsGB + mmprojGB + includedDrafterGB + fullKVGB
        if includedDrafterGB > 0 && baseFootprintWithDrafter > min(effectiveReclaimableGB, effectiveWorkingSetGB) {
            includedDrafterGB = 0
            droppedDrafterForMemory = true
        }

        let totalRequiredGB = rawWeightsGB + mmprojGB + includedDrafterGB + fullKVGB
        let headroomGB = effectiveReclaimableGB - totalRequiredGB
        let kvHeadroomGB = effectiveWorkingSetGB - (rawWeightsGB + mmprojGB + includedDrafterGB + fullKVGB)

        var notes: [String] = []
        if droppedDrafterForMemory {
            notes.append("Draft model excluded from active footprint because target + KV + drafter exceeds available working set.")
        }

        // Max buffer check on single huge tensors / GGUF monolithic file shards
        if let maxBuf = mem.maxBufferGB, fileSizes.largestShardGB > maxBuf {
            notes.append(
                String(
                    format: "Largest weight shard (%.1f GiB) exceeds Metal maxBufferLength (%.1f GiB).",
                    fileSizes.largestShardGB,
                    maxBuf
                )
            )
        }

        // Sysctl wired-limit suggestion when iogpu.wired_limit_mb is active and tight
        var sysctlSuggestMB: Int?
        if let wiredMB = mem.wiredLimitMB, wiredMB > 0 {
            let neededWorkingSetMB = Int(ceil((rawWeightsGB + mmprojGB + includedDrafterGB + fullKVGB + workingSetHeadroomReserveGB) * 1024.0))
            let totalCapMB = Int(floor(mem.totalGB * 1024.0 * 0.90))
            if neededWorkingSetMB > wiredMB && wiredMB < totalCapMB {
                let targetMB = min(totalCapMB, ((neededWorkingSetMB + 1023) / 1024) * 1024)
                if targetMB > wiredMB {
                    sysctlSuggestMB = targetMB
                    notes.append("iogpu.wired_limit_mb=\(wiredMB) MiB is below estimated GPU working set (\(neededWorkingSetMB) MiB); raising to \(targetMB) MiB gives more KV headroom.")
                }
            }
        }

        let exceedsWorkingSet = totalRequiredGB > effectiveWorkingSetGB
        let workingSetDecision: RAMFitReport.WorkingSetDecision
        if exceedsWorkingSet {
            workingSetDecision = hardWorkingSet ? .fail : .warn
            if hardWorkingSet {
                notes.append(
                    String(
                        format: "Required footprint %.1f GiB exceeds Metal working set %.1f GiB (MLX hard fail).",
                        totalRequiredGB,
                        effectiveWorkingSetGB
                    )
                )
            } else {
                notes.append(
                    String(
                        format: "Estimated footprint %.1f GiB exceeds Metal working set %.1f GiB (GGUF warn; may spill or stall at decode).",
                        totalRequiredGB,
                        effectiveWorkingSetGB
                    )
                )
            }
        } else if kvHeadroomGB < 4.0 {
            workingSetDecision = .warn
        } else {
            workingSetDecision = .ok
        }

        let decision: RAMFitReport.Decision
        let fits: Bool
        if headroomGB < 0 || (hardWorkingSet && exceedsWorkingSet) {
            decision = .fail
            fits = false
        } else if headroomGB < thinHeadroomGB || workingSetDecision == .warn {
            decision = .thin
            fits = true
        } else {
            decision = .ok
            fits = true
        }

        // Compute conservative recommended context window within both reclaimable RAM and Metal working set
        let fixedWeightsGB = rawWeightsGB + mmprojGB + includedDrafterGB
        let budgetForKV = max(0.25, min(effectiveReclaimableGB, effectiveWorkingSetGB) - fixedWeightsGB - 1.5)
        let recommendedCtx: Int
        if fullKVGB <= budgetForKV {
            recommendedCtx = effectiveCtx
        } else if fullKVGB > 0 {
            let scaled = Int(Double(effectiveCtx) * (budgetForKV / fullKVGB))
            let buckets = [131_072, 65_536, 32_768, 16_384, 8_192, 4_096, 2_048]
            recommendedCtx = buckets.first(where: { $0 <= scaled }) ?? max(1_024, (scaled / 1_024) * 1_024)
        } else {
            recommendedCtx = 4_096
        }

        return RAMFitReport(
            modelID: description.modelID,
            provider: description.provider,
            contextLength: effectiveCtx,
            weightsGB: weightsGBRounded(rawWeightsGB),
            mmprojGB: weightsGBRounded(mmprojGB),
            drafterGB: weightsGBRounded(includedDrafterGB),
            kvGB: weightsGBRounded(fullKVGB),
            totalRequiredGB: weightsGBRounded(totalRequiredGB),
            reclaimableGB: weightsGBRounded(effectiveReclaimableGB),
            totalMemoryGB: weightsGBRounded(mem.totalGB),
            workingSetGB: weightsGBRounded(effectiveWorkingSetGB),
            maxBufferGB: mem.maxBufferGB.map(weightsGBRounded),
            wiredLimitMB: mem.wiredLimitMB,
            headroomGB: weightsGBRounded(headroomGB),
            kvHeadroomGB: weightsGBRounded(kvHeadroomGB),
            fits: fits,
            decision: decision,
            workingSetDecision: workingSetDecision,
            hardWorkingSet: hardWorkingSet,
            sysctlSuggestMB: sysctlSuggestMB,
            droppedDrafterForMemory: droppedDrafterForMemory,
            recommendedCtxSize: recommendedCtx,
            notes: notes
        )
    }

    private static func weightsGBRounded(_ value: Double) -> Double {
        (value * 100.0).rounded() / 100.0
    }

    private static func liveReclaimableBytes() -> UInt64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let host = mach_host_self()
        let result = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(host, HOST_VM_INFO64, intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        var pageSize: vm_size_t = 0
        host_page_size(host, &pageSize)
        let resolvedPage = pageSize > 0 ? UInt64(pageSize) : 16_384
        let reclaimablePages = UInt64(stats.free_count)
            + UInt64(stats.inactive_count)
            + UInt64(stats.purgeable_count)
        return reclaimablePages * resolvedPage
    }

    private static func sysctlInt32(named name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let rc = name.withCString { cName in
            sysctlbyname(cName, &value, &size, nil, 0)
        }
        guard rc == 0 else { return nil }
        return Int(value)
    }

    private struct AssetSizes {
        var weightsGB: Double
        var mmprojGB: Double
        var drafterGB: Double
        var largestShardGB: Double
    }

    private static func inspectAssetSizesGB(
        for description: EngineDescription,
        modelURL: URL,
        sidecar: ModelSidecarConfig?,
        overrides: RAMFitOverrides
    ) -> AssetSizes {
        let fm = FileManager.default
        var weightsBytes: UInt64 = 0
        var mmprojBytes: UInt64 = 0
        var drafterBytes: UInt64 = 0
        var largestShardBytes: UInt64 = 0

        let ggufSummary = fileSummaryIfGGUF(at: modelURL)
        let sidecarMMProj: String? = if case .string(let s)? = sidecar?.extraConfig["mmproj_path"] { s } else { nil }
        let mmprojPath = sidecarMMProj ?? HeylookConfig.discoverMMProj(in: modelURL.deletingLastPathComponent())

        var isDir: ObjCBool = false
        if fm.fileExists(atPath: modelURL.path, isDirectory: &isDir) {
            if isDir.boolValue {
                let contents = (try? fm.contentsOfDirectory(at: modelURL, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])) ?? []
                for file in contents {
                    let ext = file.pathExtension.lowercased()
                    guard ["safetensors", "gguf", "bin", "aimodel"].contains(ext) else { continue }
                    let nameLower = file.lastPathComponent.lowercased()
                    if let sz = fileSizeBytes(at: file) {
                        if nameLower.contains("mmproj") {
                            mmprojBytes += sz
                        } else {
                            weightsBytes += sz
                            largestShardBytes = max(largestShardBytes, sz)
                        }
                    }
                }
            } else if let sz = fileSizeBytes(at: modelURL) {
                weightsBytes = sz
                largestShardBytes = sz
            }
        } else {
            weightsBytes = UInt64(max(256, description.estimatedMemoryMB)) * UInt64(mibBytes)
        }

        // GGUF multi-split shard awareness (e.g. *-00001-of-00003.gguf)
        let splitCount = ggufSummary?.uintMetadata["split.count"] ?? 0
        let looksSplitByFilename = modelURL.lastPathComponent.contains("-0000")
            && modelURL.lastPathComponent.contains("-of-")
        if (splitCount > 1 || looksSplitByFilename), weightsBytes > 0 {
            let parent = modelURL.deletingLastPathComponent()
            if let entries = try? fm.contentsOfDirectory(at: parent, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
                let stemPrefix = modelURL.lastPathComponent.split(separator: "-0000").first.map(String.init) ?? ""
                if !stemPrefix.isEmpty {
                    let shardFiles = entries.filter {
                        $0.pathExtension.lowercased() == "gguf"
                            && $0.lastPathComponent.hasPrefix(stemPrefix)
                            && !$0.lastPathComponent.lowercased().contains("mmproj")
                    }
                    let sumShards = shardFiles.reduce(UInt64(0)) { $0 + (fileSizeBytes(at: $1) ?? 0) }
                    if sumShards > weightsBytes {
                        weightsBytes = sumShards
                        largestShardBytes = max(largestShardBytes, shardFiles.compactMap(fileSizeBytes(at:)).max() ?? 0)
                    }
                }
            }
        }

        if let mmprojPath, mmprojBytes == 0 {
            mmprojBytes = directoryOrFileSizeBytes(at: URL(fileURLWithPath: mmprojPath))
        }

        let includeDrafter = overrides.includeDrafter ?? true
        if includeDrafter {
            let draftPath = overrides.drafterPath
                ?? sidecar?.draftModel
                ?? description.speculativeDrafter.value
            if let draftPath, !draftPath.isEmpty {
                drafterBytes = directoryOrFileSizeBytes(at: URL(fileURLWithPath: draftPath))
            }
        }

        let resolvedWeights = weightsBytes > 0
            ? Double(weightsBytes) / gibBytes
            : Double(max(256, description.estimatedMemoryMB)) / 1024.0

        return AssetSizes(
            weightsGB: resolvedWeights,
            mmprojGB: Double(mmprojBytes) / gibBytes,
            drafterGB: Double(drafterBytes) / gibBytes,
            largestShardGB: Double(largestShardBytes) / gibBytes
        )
    }

    private static func fileSummaryIfGGUF(at url: URL) -> GGUFHeaderReader.HeaderSummary? {
        guard url.pathExtension.lowercased() == "gguf" else { return nil }
        return GGUFHeaderReader.readFileHeader(at: url)
    }

    private static func fileSizeBytes(at url: URL) -> UInt64? {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
              values.isRegularFile == true,
              let size = values.fileSize else {
            return nil
        }
        return UInt64(max(0, size))
    }

    private static func directoryOrFileSizeBytes(at url: URL) -> UInt64 {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { return 0 }
        if !isDir.boolValue {
            return fileSizeBytes(at: url) ?? 0
        }
        let entries = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])) ?? []
        return entries.reduce(UInt64(0)) { acc, entry in
            let ext = entry.pathExtension.lowercased()
            guard ["safetensors", "gguf", "bin", "aimodel"].contains(ext) else { return acc }
            return acc + (fileSizeBytes(at: entry) ?? 0)
        }
    }

    private static func estimateKVBytes(
        for description: EngineDescription,
        modelURL: URL,
        contextLength: Int,
        overrides: RAMFitOverrides,
        sidecar: ModelSidecarConfig?
    ) -> Double {
        let clampedCtx = max(512, contextLength)

        // KV dtype bytes multiplier
        let sidecarK: String? = if case .string(let s)? = sidecar?.extraConfig["cache_type_k"] { s } else { nil }
        let sidecarV: String? = if case .string(let s)? = sidecar?.extraConfig["cache_type_v"] { s } else { nil }
        let kType = (overrides.cacheTypeK ?? sidecarK ?? "").lowercased()
        let vType = (overrides.cacheTypeV ?? sidecarV ?? "").lowercased()
        let kvScheme = (overrides.kvScheme ?? "").lowercased()
        let kvBits = overrides.kvBits

        let elemBytes: Double
        if let kvBits, kvBits > 0 {
            elemBytes = Double(kvBits) / 8.0
        } else if kvScheme.contains("2") || kType.contains("q2") {
            elemBytes = 0.5
        } else if kvScheme.contains("4") || kType.contains("q4") || vType.contains("q4") {
            elemBytes = 0.65
        } else if kType.contains("q8") || vType.contains("q8") || kvScheme.contains("8") {
            elemBytes = 1.0
        } else if kType == "f32" {
            elemBytes = 4.0
        } else {
            elemBytes = 2.0 // FP16 / BF16 default
        }

        // Read architectural geometry from config.json when present
        let modelDir = modelURL.hasDirectoryPath ? modelURL : modelURL.deletingLastPathComponent()
        if let cfgData = try? Data(contentsOf: modelDir.appendingPathComponent("config.json")),
           let raw = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any] {
            let textCfg = (raw["text_config"] as? [String: Any]) ?? raw
            let layers = (textCfg["num_hidden_layers"] as? Int) ?? (textCfg["n_layer"] as? Int) ?? 32
            let kvHeads = (textCfg["num_key_value_heads"] as? Int)
                ?? (textCfg["num_attention_heads"] as? Int)
                ?? 8
            let headDim = (textCfg["head_dim"] as? Int)
                ?? (((textCfg["hidden_size"] as? Int) ?? 4096) / max(1, (textCfg["num_attention_heads"] as? Int) ?? 32))
            let layerTypes = textCfg["layer_types"] as? [String]
            let slidingWindow = (textCfg["sliding_window"] as? Int) ?? 4096

            let totalBytes: Double
            if let layerTypes, !layerTypes.isEmpty {
                var accum = 0.0
                for lt in layerTypes {
                    let lower = lt.lowercased()
                    if lower.contains("linear") || lower.contains("mamba") || lower.contains("ssm") {
                        // Fixed recurrent state (~4 MiB per SSM layer) independent of context length
                        accum += 4.0 * mibBytes
                    } else if lower.contains("sliding") {
                        let effWindow = Double(min(clampedCtx, max(512, slidingWindow)))
                        accum += effWindow * 2.0 * Double(kvHeads) * Double(headDim) * elemBytes
                    } else {
                        accum += Double(clampedCtx) * 2.0 * Double(kvHeads) * Double(headDim) * elemBytes
                    }
                }
                totalBytes = accum
            } else {
                totalBytes = Double(clampedCtx) * 2.0 * Double(layers) * Double(kvHeads) * Double(headDim) * elemBytes
            }
            return totalBytes * 1.15
        }

        // Fallback geometry estimate based on model size if config.json is absent (standalone .gguf / remote)
        let bytesPerTokenApprox: Double = {
            if let summary = fileSummaryIfGGUF(at: modelURL) {
                let effectiveCtxFactor = summary.hasSlidingWindowAttention ? 0.45 : 1.0
                return 262_144.0 * (elemBytes / 2.0) * effectiveCtxFactor
            }
            return 262_144.0 * (elemBytes / 2.0)
        }()
        return Double(clampedCtx) * bytesPerTokenApprox
    }
}
