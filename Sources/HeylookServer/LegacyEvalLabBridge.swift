import Foundation
import HeylookEmbeddings

/// Deprecated Wave 2 compile bridge keeping existing `HeylookUI` / `HeylookTests` references
/// compiling until Waves 3 and 4 migrate callers directly to `heylook-cli`.
@available(*, deprecated, message: "Migrated to HeylookCLI; remove once HeylookUI/HeylookTests migrate in Waves 3-4")
public enum EvalLabSuite {
    public struct ProbeSummary: Codable, Sendable, Equatable, Identifiable {
        public var id: String { name }
        public let name: String
        public let passed: Bool
        public let detail: String

        public init(name: String, passed: Bool, detail: String) {
            self.name = name
            self.passed = passed
            self.detail = detail
        }
    }

    public struct SuiteReport: Codable, Sendable, Equatable {
        public let allPassed: Bool
        public let probes: [ProbeSummary]
        public let matryoshkaBenchmarks: [EmbeddingLabRunner.DimensionMetrics]

        public init(
            allPassed: Bool,
            probes: [ProbeSummary],
            matryoshkaBenchmarks: [EmbeddingLabRunner.DimensionMetrics]
        ) {
            self.allPassed = allPassed
            self.probes = probes
            self.matryoshkaBenchmarks = matryoshkaBenchmarks
        }
    }

    public static func runAll(allowSyntheticSelfTest _: Bool = false) async throws -> SuiteReport {
        let message = "EvalLabSuite migrated to heylook-cli; no live model evaluation in deprecated server bridge"
        let probes = [
            ProbeSummary(name: "chain_probe", passed: false, detail: message),
            ProbeSummary(name: "vlm_parity_probe", passed: false, detail: message),
            ProbeSummary(name: "matryoshka_recall_probe", passed: false, detail: message),
        ]
        return SuiteReport(
            allPassed: false,
            probes: probes,
            matryoshkaBenchmarks: []
        )
    }
}
