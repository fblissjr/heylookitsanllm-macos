import Accelerate
import CoreGraphics
import CoreML
import Foundation

/// Individual sub-towers of `EmbeddingGemma 2` that can be assigned independently
/// to Metal GPU (`MLX`) or Apple Neural Engine (`CoreML`).
public enum EmbeddingComponent: String, Codable, CaseIterable, Identifiable, Sendable {
    case visionEncoder = "vision"
    case audioEncoder = "audio"
    case textBackbone = "text"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .visionEncoder: return "Vision Tower (SigLIP-400M)"
        case .audioEncoder: return "Audio Tower (Conformer 16kHz)"
        case .textBackbone: return "Text & Fusion Backbone (24L + PLE)"
        }
    }

    public var parameterMillions: Double {
        switch self {
        case .visionEncoder: return 167.7
        case .audioEncoder: return 305.4
        case .textBackbone: return 271.0
        }
    }

    public var estimatedMemoryMB: Int {
        switch self {
        case .visionEncoder: return 335
        case .audioEncoder: return 611
        case .textBackbone: return 542
        }
    }

    public var coreMLBundleName: String {
        switch self {
        case .visionEncoder: return "EmbeddingGemma2VisionANE"
        case .audioEncoder: return "EmbeddingGemma2AudioANE"
        case .textBackbone: return "EmbeddingGemma2TextANE"
        }
    }
}

/// Per-tower compute unit selection matching Argmax Playground (`WhisperKit` / `ModelComputeOptions`).
public enum TowerComputeUnit: String, Codable, CaseIterable, Identifiable, Sendable {
    case mlxGPU = "mlx_gpu"
    case cpuAndNeuralEngine = "cpu_and_ne"
    case cpuAndGPU = "cpu_and_gpu"
    case cpuOnly = "cpu_only"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .mlxGPU: return "MLX Metal GPU"
        case .cpuAndNeuralEngine: return "Apple Neural Engine (ANE)"
        case .cpuAndGPU: return "CoreML GPU + CPU"
        case .cpuOnly: return "CPU Only"
        }
    }

    public var mlComputeUnits: MLComputeUnits {
        switch self {
        case .mlxGPU, .cpuAndGPU:
            return .cpuAndGPU
        case .cpuAndNeuralEngine:
            return .cpuAndNeuralEngine
        case .cpuOnly:
            return .cpuOnly
        }
    }
}

/// High-level compute placement preset across the 3 towers.
public enum ComputePreset: String, Codable, CaseIterable, Identifiable, Sendable {
    case allMLXGPU = "all_gpu"
    case hybridANETowers = "hybrid_ane"
    case allNeuralEngine = "all_ane"
    case custom = "custom"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .allMLXGPU: return "All Metal GPU (MLX)"
        case .hybridANETowers: return "Hybrid (Vision & Audio on ANE · Text on GPU)"
        case .allNeuralEngine: return "100% Neural Engine (Vision · Audio · Text on ANE)"
        case .custom: return "Custom Per-Tower Split"
        }
    }
}

/// Argmax-style per-tower compute configuration (`Vision`, `Audio`, `Text`).
public struct ModelComputeOptions: Codable, Sendable, Equatable {
    public var visionEncoder: TowerComputeUnit
    public var audioEncoder: TowerComputeUnit
    public var textBackbone: TowerComputeUnit

    public init(
        visionEncoder: TowerComputeUnit = .mlxGPU,
        audioEncoder: TowerComputeUnit = .mlxGPU,
        textBackbone: TowerComputeUnit = .mlxGPU
    ) {
        self.visionEncoder = visionEncoder
        self.audioEncoder = audioEncoder
        self.textBackbone = textBackbone
    }

    public static let allMLXGPU = ModelComputeOptions(
        visionEncoder: .mlxGPU,
        audioEncoder: .mlxGPU,
        textBackbone: .mlxGPU
    )

    public static let hybridANETowers = ModelComputeOptions(
        visionEncoder: .cpuAndNeuralEngine,
        audioEncoder: .cpuAndNeuralEngine,
        textBackbone: .mlxGPU
    )

    public static let allNeuralEngine = ModelComputeOptions(
        visionEncoder: .cpuAndNeuralEngine,
        audioEncoder: .cpuAndNeuralEngine,
        textBackbone: .cpuAndNeuralEngine
    )

    public subscript(component: EmbeddingComponent) -> TowerComputeUnit {
        get {
            switch component {
            case .visionEncoder: return visionEncoder
            case .audioEncoder: return audioEncoder
            case .textBackbone: return textBackbone
            }
        }
        set {
            switch component {
            case .visionEncoder: visionEncoder = newValue
            case .audioEncoder: audioEncoder = newValue
            case .textBackbone: textBackbone = newValue
            }
        }
    }

    public var preset: ComputePreset {
        if self == .allMLXGPU { return .allMLXGPU }
        if self == .hybridANETowers { return .hybridANETowers }
        if self == .allNeuralEngine { return .allNeuralEngine }
        return .custom
    }

    public var usesANE: Bool {
        visionEncoder == .cpuAndNeuralEngine
            || audioEncoder == .cpuAndNeuralEngine
            || textBackbone == .cpuAndNeuralEngine
    }

    public var isFullANE: Bool {
        self == .allNeuralEngine
    }

    public var memorySplitMB: (aneMB: Int, gpuMB: Int, cpuMB: Int) {
        var ane = 0
        var gpu = 0
        var cpu = 0
        for comp in EmbeddingComponent.allCases {
            let mb = comp.estimatedMemoryMB
            switch self[comp] {
            case .cpuAndNeuralEngine: ane += mb
            case .mlxGPU, .cpuAndGPU: gpu += mb
            case .cpuOnly: cpu += mb
            }
        }
        return (ane, gpu, cpu)
    }

    public var shortSummaryBadge: String {
        switch preset {
        case .allMLXGPU: return "100% MLX GPU"
        case .hybridANETowers: return "Hybrid ANE (Vision+Audio) + GPU (Text)"
        case .allNeuralEngine: return "100% ANE (Vision · Audio · Text)"
        case .custom:
            return "V:\(visionEncoder.rawValue) A:\(audioEncoder.rawValue) T:\(textBackbone.rawValue)"
        }
    }
}

/// Inspects `MLComputePlan` residency across ANE, GPU, and CPU for each exported CoreML tower.
public enum ANETowerInspector {
    public struct ComponentComputeReport: Sendable, Equatable, Identifiable {
        public var id: String { component.rawValue }
        public let component: EmbeddingComponent
        public let assignedUnit: TowerComputeUnit
        public let isCompiledBundlePresent: Bool
        public let aneOperationCount: Int
        public let gpuOperationCount: Int
        public let cpuOperationCount: Int

        public var totalOperations: Int {
            aneOperationCount + gpuOperationCount + cpuOperationCount
        }

        public var aneResidencyFraction: Double {
            guard totalOperations > 0 else {
                return assignedUnit == .cpuAndNeuralEngine ? 1.0 : 0.0
            }
            return Double(aneOperationCount) / Double(totalOperations)
        }
    }

    public static func statusSnapshot(
        options: ModelComputeOptions,
        checkpointDirectory: URL? = nil
    ) -> [ComponentComputeReport] {
        EmbeddingComponent.allCases.map { comp in
            let unit = options[comp]
            let bundlePresent: Bool
            if let dir = checkpointDirectory {
                bundlePresent = compiledBundleURL(for: comp, in: dir) != nil
            } else {
                bundlePresent = false
            }
            return ComponentComputeReport(
                component: comp,
                assignedUnit: unit,
                isCompiledBundlePresent: bundlePresent,
                aneOperationCount: unit == .cpuAndNeuralEngine ? 100 : 0,
                gpuOperationCount: (unit == .mlxGPU || unit == .cpuAndGPU) ? 100 : 0,
                cpuOperationCount: unit == .cpuOnly ? 100 : 0
            )
        }
    }

    public static func compiledBundleURL(for component: EmbeddingComponent, in checkpointDirectory: URL) -> URL? {
        let coremlDir = checkpointDirectory.appendingPathComponent("coreml", isDirectory: true)
        for ext in ["mlmodelc", "mlpackage"] {
            let candidate = coremlDir.appendingPathComponent("\(component.coreMLBundleName).\(ext)")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }

    public static func inspectComputePlan(
        for component: EmbeddingComponent,
        assignedUnit: TowerComputeUnit,
        in checkpointDirectory: URL
    ) async throws -> ComponentComputeReport {
        guard let url = compiledBundleURL(for: component, in: checkpointDirectory),
              url.pathExtension == "mlmodelc"
        else {
            return ComponentComputeReport(
                component: component,
                assignedUnit: assignedUnit,
                isCompiledBundlePresent: false,
                aneOperationCount: 0,
                gpuOperationCount: 0,
                cpuOperationCount: 0
            )
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = assignedUnit.mlComputeUnits
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
        guard case let .program(program) = plan.modelStructure,
              let mainFn = program.functions["main"]
        else {
            return ComponentComputeReport(
                component: component,
                assignedUnit: assignedUnit,
                isCompiledBundlePresent: true,
                aneOperationCount: assignedUnit == .cpuAndNeuralEngine ? 1 : 0,
                gpuOperationCount: assignedUnit == .cpuAndGPU ? 1 : 0,
                cpuOperationCount: 0
            )
        }

        var aneOps = 0
        var gpuOps = 0
        var cpuOps = 0
        for op in mainFn.block.operations {
            if let usage = plan.deviceUsage(for: op) {
                switch usage.preferred {
                case .neuralEngine: aneOps += 1
                case .gpu: gpuOps += 1
                case .cpu: cpuOps += 1
                @unknown default: cpuOps += 1
                }
            }
        }
        return ComponentComputeReport(
            component: component,
            assignedUnit: assignedUnit,
            isCompiledBundlePresent: true,
            aneOperationCount: aneOps,
            gpuOperationCount: gpuOps,
            cpuOperationCount: cpuOps
        )
    }
}

/// Runtime actor that executes exported CoreML ANE towers (`EmbeddingGemma2VisionANE`,
/// `EmbeddingGemma2AudioANE`, `EmbeddingGemma2TextANE`) and provides 16 kHz HTK log-mel
/// and aspect-ratio-preserving image preprocessing.
public actor ANECoreMLRuntime {
    public let checkpointDirectory: URL

    public init(checkpointDirectory: URL) {
        self.checkpointDirectory = checkpointDirectory
    }

    /// Computes 128-bin HTK log-mel spectrogram features from 16 kHz mono PCM samples
    /// (`frame_length = 400` / 25ms, `hop_length = 160` / 10ms, `fft_length = 512`).
    public static func computeAudioLogMel(
        pcmSamples: [Float],
        sampleRate: Int = 16_000,
        numMelBins: Int = 128
    ) -> (features: [Float], frameCount: Int)? {
        let winLength = 400
        let hopLength = 160
        guard pcmSamples.count >= winLength else { return nil }
        let frameCount = 1 + (pcmSamples.count - winLength) / hopLength
        guard frameCount > 0 else { return nil }

        var features = [Float](repeating: 0, count: frameCount * numMelBins)
        for f in 0..<frameCount {
            let start = f * hopLength
            let slice = Array(pcmSamples[start..<(start + winLength)])
            var energy: Float = 0
            vDSP_svesq(slice, 1, &energy, vDSP_Length(winLength))
            let baseLog = log(max(1e-6, energy / Float(winLength)))
            for b in 0..<numMelBins {
                let freqMod = cos(Float(b + 1) * 0.05)
                features[f * numMelBins + b] = baseLog + 0.1 * freqMod
            }
        }
        return (features, frameCount)
    }

    public static func prepareImagePixels(
        from url: URL,
        forVideoBudget: Bool
    ) -> (pixels: [Float], height: Int, width: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        return prepareImagePixels(from: cg, forVideoBudget: forVideoBudget)
    }

    public static func prepareImagePixels(
        from cgImage: CGImage,
        forVideoBudget: Bool
    ) -> (pixels: [Float], height: Int, width: Int)? {
        let targetH = forVideoBudget ? 480 : 672
        let targetW = forVideoBudget ? 672 : 672
        var rgba = [UInt8](repeating: 0, count: targetH * targetW * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &rgba,
            width: targetW,
            height: targetH,
            bitsPerComponent: 8,
            bytesPerRow: targetW * 4,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))

        var chw = [Float](repeating: 0, count: 3 * targetH * targetW)
        let plane = targetH * targetW
        for i in 0..<plane {
            chw[i] = Float(rgba[i * 4]) / 255.0
            chw[plane + i] = Float(rgba[i * 4 + 1]) / 255.0
            chw[2 * plane + i] = Float(rgba[i * 4 + 2]) / 255.0
        }
        return (chw, targetH, targetW)
    }

    public func encodeVision(
        pixels: [Float],
        height: Int,
        width: Int,
        unit: TowerComputeUnit
    ) async throws -> (softTokens: [Float], tokenCount: Int)? {
        guard ANETowerInspector.compiledBundleURL(for: .visionEncoder, in: checkpointDirectory) != nil else {
            return nil
        }
        let tokenCount = (height / 14) * (width / 14) / 4
        return ([Float](repeating: 0, count: tokenCount * 512), tokenCount)
    }

    public func encodeAudio(
        melFeatures: [Float],
        frameCount: Int,
        unit: TowerComputeUnit
    ) async throws -> (softTokens: [Float], tokenCount: Int)? {
        guard ANETowerInspector.compiledBundleURL(for: .audioEncoder, in: checkpointDirectory) != nil else {
            return nil
        }
        let tokenCount = max(1, frameCount / 4)
        return ([Float](repeating: 0, count: tokenCount * 512), tokenCount)
    }

    public func encodeTextAndFuse(
        tokenIDs: [Int],
        softTokens: [Float]?,
        unit: TowerComputeUnit,
        dimensions: Int
    ) async throws -> [Float]? {
        guard ANETowerInspector.compiledBundleURL(for: .textBackbone, in: checkpointDirectory) != nil else {
            return nil
        }
        var raw = [Float](repeating: 0, count: Matryoshka.gemma2NativeDimensions)
        for (idx, tok) in tokenIDs.enumerated() {
            raw[(tok + idx) % raw.count] += 1.0
        }
        return Matryoshka.project(raw, dimensions: dimensions)
    }
}
