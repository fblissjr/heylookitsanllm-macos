import Foundation
import HeylookCore
import HeylookEngines

/// Wraps `HeylookEngines.RAMFitCalculator` into the `/v1/admin/models/{id}/fit` wire response
/// (`heylook_llm.ram_fit` parity), failing closed (`422`) when a checkpoint has zero measurable weight bytes.
public enum RAMFitEvaluator {
    private static let thinKVHeadroomGB: Double = 4.0

    public struct FitLine: Codable, Sendable, Equatable {
        public let ceiling: String
        public let verdict: String
        public let needGB: Double
        public let haveGB: Double
        public let note: String

        enum CodingKeys: String, CodingKey {
            case ceiling
            case verdict
            case needGB = "need_gb"
            case haveGB = "have_gb"
            case note
        }

        public init(ceiling: String, verdict: String, needGB: Double, haveGB: Double, note: String = "") {
            self.ceiling = ceiling
            self.verdict = verdict
            self.needGB = needGB
            self.haveGB = haveGB
            self.note = note
        }
    }

    public struct FitReport: Codable, Sendable, Equatable {
        public let weightsGB: Double
        public let headroomGB: Double
        public let reclaimableGB: Double
        public let workingSetGB: Double?
        public let maxBufferGB: Double?
        public let sysctlWiredMB: Int?
        public let sysctlSuggestMB: Int?
        public let kvHeadroomGB: Double?
        public let headroomThin: Bool
        public let hardWorkingSet: Bool
        public let verdict: String
        public let lines: [FitLine]
        public let sizingNotes: [String]
        public let recommendedCtxSize: Int

        enum CodingKeys: String, CodingKey {
            case weightsGB = "weights_gb"
            case headroomGB = "headroom_gb"
            case reclaimableGB = "reclaimable_gb"
            case workingSetGB = "working_set_gb"
            case maxBufferGB = "max_buffer_gb"
            case sysctlWiredMB = "sysctl_wired_mb"
            case sysctlSuggestMB = "sysctl_suggest_mb"
            case kvHeadroomGB = "kv_headroom_gb"
            case headroomThin = "headroom_thin"
            case hardWorkingSet = "hard_working_set"
            case verdict
            case lines
            case sizingNotes = "sizing_notes"
            case recommendedCtxSize = "recommended_ctx_size"
        }

        public init(
            weightsGB: Double,
            headroomGB: Double,
            reclaimableGB: Double,
            workingSetGB: Double?,
            maxBufferGB: Double?,
            sysctlWiredMB: Int?,
            sysctlSuggestMB: Int?,
            kvHeadroomGB: Double?,
            headroomThin: Bool,
            hardWorkingSet: Bool,
            verdict: String,
            lines: [FitLine],
            sizingNotes: [String],
            recommendedCtxSize: Int
        ) {
            self.weightsGB = weightsGB
            self.headroomGB = headroomGB
            self.reclaimableGB = reclaimableGB
            self.workingSetGB = workingSetGB
            self.maxBufferGB = maxBufferGB
            self.sysctlWiredMB = sysctlWiredMB
            self.sysctlSuggestMB = sysctlSuggestMB
            self.kvHeadroomGB = kvHeadroomGB
            self.headroomThin = headroomThin
            self.hardWorkingSet = hardWorkingSet
            self.verdict = verdict
            self.lines = lines
            self.sizingNotes = sizingNotes
            self.recommendedCtxSize = recommendedCtxSize
        }
    }

    public static func evaluate(
        description: EngineDescription,
        configOverrides: [String: JSONValue] = [:],
        headroomGB: Double = 8.0
    ) throws -> FitReport {
        guard headroomGB.isFinite && headroomGB >= 0 && headroomGB <= 256 else {
            throw WorkbenchError.unprocessable(
                message: "headroom_gb must be between 0 and 256",
                field: "headroom_gb"
            )
        }

        let dirURL = HeylookConfig.resolveModelDirectoryURL(from: description.modelPath)
        var effectiveSidecar = HeylookConfig.loadSidecar(in: dirURL) ?? ModelSidecarConfig()
        var effectiveDesc = description
        var ramOverrides = RAMFitOverrides()

        for (key, val) in configOverrides {
            switch key {
            case "ctx_size", "context_length":
                switch val {
                case .null:
                    effectiveSidecar.ctxSize = nil
                    ramOverrides.ctxSize = nil
                case .int(let n):
                    ramOverrides.ctxSize = n
                case .double(let d):
                    ramOverrides.ctxSize = Int(d)
                default:
                    break
                }
            case "cache_type_k":
                if case .string(let s) = val { ramOverrides.cacheTypeK = s }
                else if case .null = val { ramOverrides.cacheTypeK = nil }
            case "cache_type_v":
                if case .string(let s) = val { ramOverrides.cacheTypeV = s }
                else if case .null = val { ramOverrides.cacheTypeV = nil }
            case "kv_bits":
                if case .int(let n) = val { ramOverrides.kvBits = n }
                else if case .null = val { ramOverrides.kvBits = nil }
            case "kv_quant_scheme", "kv_scheme":
                if case .string(let s) = val { ramOverrides.kvScheme = s }
                else if case .null = val { ramOverrides.kvScheme = nil }
            case "draft_model_path", "draft_model":
                if case .string(let s) = val {
                    ramOverrides.drafterPath = s
                    ramOverrides.includeDrafter = true
                } else if case .null = val {
                    ramOverrides.drafterPath = nil
                    ramOverrides.includeDrafter = false
                    effectiveSidecar.draftModel = nil
                }
            case "mmproj_path":
                if case .string(let s) = val {
                    effectiveSidecar.extraConfig["mmproj_path"] = .string(s)
                } else if case .null = val {
                    effectiveSidecar.extraConfig.removeValue(forKey: "mmproj_path")
                }
            case "model_path":
                if case .string(let p) = val, !p.isEmpty {
                    effectiveDesc = EngineDescription(
                        modelID: description.modelID,
                        provider: description.provider,
                        modelPath: p,
                        architecture: description.architecture,
                        quantization: description.quantization,
                        computePlacement: description.computePlacement,
                        isPinned: description.isPinned,
                        isLoaded: description.isLoaded,
                        contextSize: description.contextSize,
                        flashAttention: description.flashAttention,
                        supportsVision: description.supportsVision,
                        supportsAudio: description.supportsAudio,
                        supportsEmbeddings: description.supportsEmbeddings,
                        supportsThinking: description.supportsThinking,
                        thinkingBudget: description.thinkingBudget,
                        reasoningFormat: description.reasoningFormat,
                        thinkingControls: description.thinkingControls,
                        templateSource: description.templateSource,
                        speculativeDrafter: description.speculativeDrafter,
                        estimatedMemoryMB: description.estimatedMemoryMB,
                        tags: description.tags
                    )
                }
            default:
                break
            }
        }

        let calcReport = RAMFitCalculator.assess(
            for: effectiveDesc,
            overrides: ramOverrides,
            headroomOverrideGB: nil,
            snapshot: nil,
            sidecar: effectiveSidecar
        )

        let totalWeightsGB = round2(calcReport.weightsGB + calcReport.mmprojGB + calcReport.drafterGB)
        guard totalWeightsGB > 0 else {
            let reason = calcReport.notes.isEmpty
                ? "no weight files found"
                : calcReport.notes.joined(separator: "; ")
            throw WorkbenchError.unprocessable(message: reason, field: "model_path")
        }

        let needGB = round2(totalWeightsGB + headroomGB)
        let reclaimableGB = round2(calcReport.reclaimableGB)
        let workingSetGB = round2(calcReport.workingSetGB)
        let kvHeadroom = round2(workingSetGB - totalWeightsGB)
        let headroomThin = kvHeadroom < thinKVHeadroomGB

        var lines: [FitLine] = [
            FitLine(
                ceiling: "reclaimable_ram",
                verdict: needGB <= reclaimableGB ? "pass" : "fail",
                needGB: needGB,
                haveGB: reclaimableGB,
                note: "total - anonymous - wired; what an mmap-backed load can reach"
            ),
            FitLine(
                ceiling: "metal_working_set",
                verdict: needGB <= workingSetGB
                    ? "pass"
                    : (calcReport.hardWorkingSet ? "fail" : "warn"),
                needGB: needGB,
                haveGB: workingSetGB,
                note: calcReport.hardWorkingSet
                    ? "hard limit, MLX refuses above it"
                    : "advisory for llama.cpp -- it loads past it and degrades into paging"
            ),
        ]

        if let maxBuf = calcReport.maxBufferGB,
           calcReport.notes.contains(where: { $0.contains("maxBufferLength") }) {
            lines.append(
                FitLine(
                    ceiling: "metal_max_buffer",
                    verdict: "warn",
                    needGB: round2(totalWeightsGB),
                    haveGB: round2(maxBuf),
                    note: "per-allocation cap -- a single weight file exceeds Metal maxBufferLength"
                )
            )
        }

        var verdict = "pass"
        if lines.contains(where: { $0.verdict == "warn" }) { verdict = "warn" }
        if lines.contains(where: { $0.verdict == "fail" }) { verdict = "fail" }

        return FitReport(
            weightsGB: totalWeightsGB,
            headroomGB: round2(headroomGB),
            reclaimableGB: reclaimableGB,
            workingSetGB: workingSetGB,
            maxBufferGB: calcReport.maxBufferGB.map(round2),
            sysctlWiredMB: calcReport.wiredLimitMB ?? 0,
            sysctlSuggestMB: calcReport.sysctlSuggestMB,
            kvHeadroomGB: kvHeadroom,
            headroomThin: headroomThin,
            hardWorkingSet: calcReport.hardWorkingSet,
            verdict: verdict,
            lines: lines,
            sizingNotes: calcReport.notes,
            recommendedCtxSize: calcReport.recommendedCtxSize
        )
    }

    private static func round2(_ value: Double) -> Double {
        (value * 100.0).rounded() / 100.0
    }
}
