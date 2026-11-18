import Accelerate
import AVFoundation
import CoreMedia
import Foundation
import HeylookCore
import HeylookEmbeddings
import Speech

#if canImport(CoreAISpeech)
import CoreAISpeech
#endif

/// Result emitted by ANE-lane speech transcription (`CoreAISpeech` Whisper / Parakeet TDT or system `SpeechTranscriber`).
public struct ANESpeechTranscriptionResult: Sendable, Equatable, Codable {
    public let text: String
    public let durationSeconds: Double
    public let sampleRateHz: Int
    public let rmsEnergy: Float
    public let melFrames: Int
    public let computePlacement: ComputePlacement
    public let backend: String
    public let modelID: String

    enum CodingKeys: String, CodingKey {
        case text
        case durationSeconds
        case sampleRateHz
        case rmsEnergy
        case melFrames
        case computePlacement
        case backend
        case modelID = "model_id"
    }

    public init(
        text: String,
        durationSeconds: Double,
        sampleRateHz: Int = 16_000,
        rmsEnergy: Float = 0,
        melFrames: Int = 0,
        computePlacement: ComputePlacement = .ane,
        backend: String = "coreai_speech",
        modelID: String = ""
    ) {
        self.text = text
        self.durationSeconds = durationSeconds
        self.sampleRateHz = sampleRateHz
        self.rmsEnergy = rmsEnergy
        self.melFrames = melFrames
        self.computePlacement = computePlacement
        self.backend = backend
        self.modelID = modelID
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.text = try c.decode(String.self, forKey: .text)
        self.durationSeconds = try c.decode(Double.self, forKey: .durationSeconds)
        self.sampleRateHz = try c.decodeIfPresent(Int.self, forKey: .sampleRateHz) ?? 16_000
        self.rmsEnergy = try c.decodeIfPresent(Float.self, forKey: .rmsEnergy) ?? 0
        self.melFrames = try c.decodeIfPresent(Int.self, forKey: .melFrames) ?? 0
        self.computePlacement = try c.decodeIfPresent(ComputePlacement.self, forKey: .computePlacement) ?? .ane
        self.backend = try c.decodeIfPresent(String.self, forKey: .backend) ?? "coreai_speech"
        self.modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? ""
    }
}

/// Dedicated ANE-lane speech recognition service cleanly decoupled from `.aimodel` LLM inference.
/// Supports `CoreAISpeech.SpeechRecognitionModel` (Whisper & Parakeet TDT `.aimodel` bundles) and
/// Apple's on-device `SpeechAnalyzer` + `SpeechTranscriber`. Throws `ProviderExecutionError` when
/// no real speech runtime or compiled weights are available; never fabricates or echoes audio markers.
public actor SpeechTranscriptionService {
    public static let systemSpeechModelID = "apple-speech-transcriber"

    public static var isSystemSpeechAvailable: Bool {
        SpeechTranscriber.isAvailable
    }

    public static var isCoreAISpeechLinked: Bool {
        #if canImport(CoreAISpeech)
        return true
        #else
        return false
        #endif
    }

    public static func systemSpeechDescription() -> EngineDescription {
        EngineDescription(
            modelID: systemSpeechModelID,
            provider: .coreai,
            modelPath: "system://speech-transcriber",
            architecture: ProvenanceField(value: "apple_speech_transcriber", provenance: .observed, effect: .reload),
            quantization: ProvenanceField(value: "native", provenance: .observed, effect: .reload),
            computePlacement: ProvenanceField(value: .ane, provenance: .observed, effect: .reload, computePlacement: .ane),
            isPinned: false,
            isLoaded: isSystemSpeechAvailable,
            contextSize: ProvenanceField(value: 448, provenance: .notApplicable, effect: .reload),
            flashAttention: ProvenanceField(value: false, provenance: .notApplicable, effect: .reload),
            supportsVision: ProvenanceField(value: false, provenance: .observed, effect: .reload),
            supportsAudio: ProvenanceField(value: true, provenance: .observed, effect: .reload),
            supportsEmbeddings: ProvenanceField(value: false, provenance: .observed, effect: .reload),
            supportsThinking: ProvenanceField(value: false, provenance: .observed, effect: .perRequest),
            thinkingBudget: ProvenanceField(value: nil, provenance: .notApplicable, effect: .perRequest),
            reasoningFormat: ProvenanceField(value: "none", provenance: .notApplicable, effect: .reload),
            thinkingControls: nil,
            templateSource: ProvenanceField(value: "speech_recognizer", provenance: .notApplicable, effect: .reload),
            speculativeDrafter: ProvenanceField(value: nil, provenance: .notApplicable, effect: .reload),
            estimatedMemoryMB: 120,
            tags: ["asr", "ane", "speech-transcriber", "system"]
        )
    }

    public let modelID: String
    public let isSystemSpeechProvider: Bool
    private var descriptionState: EngineDescription
    private let bundleManifest: CoreAIBundleManifest?

    #if canImport(CoreAISpeech)
    private var loadedSpeechModel: SpeechRecognitionModel?
    #endif

    public init(description: EngineDescription, manifest: CoreAIBundleManifest? = nil) {
        self.modelID = description.modelID
        self.descriptionState = description
        self.isSystemSpeechProvider = description.modelID == Self.systemSpeechModelID
            || description.modelPath == "system://speech-transcriber"
        if let manifest {
            self.bundleManifest = manifest
        } else if !self.isSystemSpeechProvider {
            let bundleURL = URL(fileURLWithPath: description.modelPath)
            self.bundleManifest = try? CoreAIBundleManifest.inspect(at: bundleURL)
        } else {
            self.bundleManifest = nil
        }
    }

    public func preload() async throws {
        if isSystemSpeechProvider {
            guard SpeechTranscriber.isAvailable else {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "Apple SpeechTranscriber is unavailable on this host."
                )
            }
            descriptionState.isLoaded = true
            return
        }

        #if canImport(CoreAISpeech)
        if bundleManifest?.isSpeechRecognizer == true {
            _ = try await ensureCoreAISpeechLoaded()
            return
        }
        #endif

        throw ProviderExecutionError.runtimeUnavailable(
            modelID: modelID,
            detail: "Speech recognizer runtime unavailable for '\(modelID)' at \(descriptionState.modelPath)."
        )
    }

    public func unload() async {
        #if canImport(CoreAISpeech)
        loadedSpeechModel = nil
        #endif
        descriptionState.isLoaded = false
    }

    public func describe() -> EngineDescription {
        descriptionState
    }

    public func transcribeAudio(
        wavData: Data,
        languageHint: String = "en"
    ) async throws -> ANESpeechTranscriptionResult {
        #if canImport(CoreAISpeech)
        if !isSystemSpeechProvider, bundleManifest?.isSpeechRecognizer == true {
            let speechModel = try await ensureCoreAISpeechLoaded()
            let modelSampleRate = await speechModel.sampleRate
            let targetRate = max(8_000, Int(modelSampleRate.rounded()))
            let samples = await Self.decodeAudioMonoPCM(from: wavData, targetSampleRate: targetRate)
            let stats = Self.computeAudioStats(samples: samples, sampleRate: targetRate)
            guard !samples.isEmpty else {
                return ANESpeechTranscriptionResult(
                    text: "",
                    durationSeconds: 0,
                    sampleRateHz: targetRate,
                    rmsEnergy: 0,
                    melFrames: 0,
                    computePlacement: descriptionState.computePlacement.value,
                    backend: "coreai_speech",
                    modelID: modelID
                )
            }
            do {
                let (text, _) = try await speechModel.transcribe(pcm: samples)
                return ANESpeechTranscriptionResult(
                    text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                    durationSeconds: stats.durationSeconds,
                    sampleRateHz: targetRate,
                    rmsEnergy: stats.rmsEnergy,
                    melFrames: stats.melFrames,
                    computePlacement: descriptionState.computePlacement.value,
                    backend: "coreai_speech",
                    modelID: modelID
                )
            } catch {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "CoreAISpeech transcription failed: \(error.localizedDescription)"
                )
            }
        }
        #endif

        if isSystemSpeechProvider {
            return try await transcribeWithSystemSpeechAnalyzer(
                wavData: wavData,
                languageHint: languageHint
            )
        }

        if bundleManifest == nil && !FileManager.default.fileExists(atPath: descriptionState.modelPath) {
            throw ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
        }
        let detail: String
        if bundleManifest?.isSpeechRecognizer == true {
            detail = "Speech recognizer bundle found at \(descriptionState.modelPath), but CoreAISpeech is not linked (build with HEYLOOK_ENABLE_COREAI=1)."
        } else {
            detail = "Speech ASR model runtime is not loaded at \(descriptionState.modelPath)"
        }
        throw ProviderExecutionError.runtimeUnavailable(modelID: modelID, detail: detail)
    }

    #if canImport(CoreAISpeech)
    private func ensureCoreAISpeechLoaded() async throws -> SpeechRecognitionModel {
        if let existing = loadedSpeechModel {
            return existing
        }
        let bundleURL = URL(fileURLWithPath: descriptionState.modelPath)
        guard FileManager.default.fileExists(atPath: bundleURL.path) else {
            throw ProviderExecutionError.checkpointNotFound(
                modelID: modelID,
                path: descriptionState.modelPath
            )
        }
        do {
            let model = try await SpeechRecognitionModel(resourcesAt: bundleURL)
            loadedSpeechModel = model
            descriptionState.isLoaded = true
            return model
        } catch {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Failed to load CoreAISpeech bundle at \(descriptionState.modelPath): \(error.localizedDescription)"
            )
        }
    }
    #endif

    private func transcribeWithSystemSpeechAnalyzer(
        wavData: Data,
        languageHint: String
    ) async throws -> ANESpeechTranscriptionResult {
        guard SpeechTranscriber.isAvailable else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Apple SpeechTranscriber is unavailable on this host."
            )
        }

        let sampleRate = 16_000
        let samples = await Self.decodeAudioMonoPCM(from: wavData, targetSampleRate: sampleRate)
        let stats = Self.computeAudioStats(samples: samples, sampleRate: sampleRate)
        guard samples.count >= 160 else {
            return ANESpeechTranscriptionResult(
                text: "",
                durationSeconds: stats.durationSeconds,
                sampleRateHz: sampleRate,
                rmsEnergy: stats.rmsEnergy,
                melFrames: stats.melFrames,
                computePlacement: .ane,
                backend: "apple_speech_transcriber",
                modelID: modelID
            )
        }

        let requestedLocale = Locale(identifier: languageHint)
        var resolvedLocale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale)
        if resolvedLocale == nil {
            let langPrefix = languageHint
                .split(whereSeparator: { $0 == "-" || $0 == "_" })
                .first
                .map(String.init)?
                .lowercased() ?? "en"
            let supported = await SpeechTranscriber.supportedLocales
            resolvedLocale = supported.first { $0.identifier.lowercased().hasPrefix(langPrefix) }
        }
        guard let targetLocale = resolvedLocale else {
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "Locale '\(languageHint)' is not supported by Apple SpeechTranscriber."
            )
        }

        let transcriber = SpeechTranscriber(locale: targetLocale, preset: .transcription)
        let assetStatus = await AssetInventory.status(forModules: [transcriber])
        switch assetStatus {
        case .unsupported:
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "On-device SpeechTranscriber assets are unsupported for locale '\(targetLocale.identifier)'."
            )
        case .supported:
            do {
                if let installReq = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                    try await installReq.downloadAndInstall()
                }
            } catch {
                throw ProviderExecutionError.runtimeUnavailable(
                    modelID: modelID,
                    detail: "On-device SpeechTranscriber assets for '\(targetLocale.identifier)' are not installed (\(error.localizedDescription))."
                )
            }
        case .downloading, .installed:
            break
        @unknown default:
            break
        }

        let canonicalWAV = Self.encodeCanonicalPCM16WAV(samples: samples, sampleRate: sampleRate)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-speech-analyzer-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try canonicalWAV.write(to: tempURL, options: .atomic)

        let audioFile = try AVAudioFile(forReading: tempURL)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let collectTask = Task<[String], Error> {
            var finals: [String] = []
            for try await result in transcriber.results {
                if result.isFinal {
                    let seg = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !seg.isEmpty {
                        finals.append(seg)
                    }
                }
            }
            return finals
        }

        do {
            if let lastTime = try await analyzer.analyzeSequence(from: audioFile) {
                try await analyzer.finalizeAndFinish(through: lastTime)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
        } catch {
            await analyzer.cancelAndFinishNow()
            collectTask.cancel()
            throw ProviderExecutionError.runtimeUnavailable(
                modelID: modelID,
                detail: "SpeechAnalyzer transcription failed: \(error.localizedDescription)"
            )
        }

        let segments = (try? await collectTask.value) ?? []
        descriptionState.isLoaded = true
        return ANESpeechTranscriptionResult(
            text: segments.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            durationSeconds: stats.durationSeconds,
            sampleRateHz: sampleRate,
            rmsEnergy: stats.rmsEnergy,
            melFrames: stats.melFrames,
            computePlacement: .ane,
            backend: "apple_speech_transcriber",
            modelID: modelID
        )
    }

    private static func computeAudioStats(
        samples: [Float],
        sampleRate: Int
    ) -> (durationSeconds: Double, rmsEnergy: Float, melFrames: Int) {
        guard !samples.isEmpty, sampleRate > 0 else { return (0, 0, 0) }
        let duration = Double(samples.count) / Double(sampleRate)
        var meanSq: Float = 0
        vDSP_measqv(samples, 1, &meanSq, vDSP_Length(samples.count))
        let rms = meanSq.squareRoot()
        let melFramesCount = ANECoreMLRuntime.computeAudioLogMel(samples: samples).count
        return (duration, rms, melFramesCount)
    }

    /// Decodes arbitrary audio containers (`WAV` with `AVAudioRecorder` 4 KB `FLLR` alignment chunks,
    /// `24/44.1/48 kHz`, stereo, `M4A`, `CAF`, `FLAC`, `MP3`) into resampled mono `Float32` PCM at `targetSampleRate`.
    public static func decodeAudioMonoPCM(
        from audioData: Data,
        targetSampleRate: Int = 16_000
    ) async -> [Float] {
        guard !audioData.isEmpty else { return [] }

        if let riffSamples = decodeRIFFChunksMonoPCM(from: audioData, targetSampleRate: targetSampleRate),
           !riffSamples.isEmpty {
            return riffSamples
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("heylook-audio-decode-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        do {
            try audioData.write(to: tempURL, options: .atomic)
            let decoded = try await ArtifactProcessor.decodeMonoPCM(for: tempURL, sampleRate: targetSampleRate)
            if !decoded.isEmpty {
                return decoded
            }
        } catch {
            // Fall through cleanly
        }
        return []
    }

    /// Parses standard RIFF/WAVE containers by walking subchunks (`fmt `, `FLLR`, `JUNK`, `LIST`, `data`)
    /// rather than assuming a fixed 44-byte header offset.
    public static func decodeRIFFChunksMonoPCM(
        from data: Data,
        targetSampleRate: Int = 16_000
    ) -> [Float]? {
        guard data.count >= 12,
              String(decoding: data.prefix(4), as: UTF8.self) == "RIFF",
              String(decoding: data.subdata(in: 8..<12), as: UTF8.self) == "WAVE"
        else { return nil }

        func readUInt16LE(_ offset: Int) -> UInt16 {
            UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
        }
        func readUInt32LE(_ offset: Int) -> UInt32 {
            UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
        }

        var audioFormat: UInt16 = 1
        var channels: Int = 1
        var srcSampleRate: Int = 16_000
        var bitsPerSample: Int = 16
        var dataRange: Range<Int>?

        var cursor = 12
        while cursor + 8 <= data.count {
            let chunkID = String(decoding: data.subdata(in: cursor..<(cursor + 4)), as: UTF8.self)
            let rawChunkSize = Int(readUInt32LE(cursor + 4))
            let payloadStart = cursor + 8
            let clampedEnd = min(data.count, payloadStart + max(0, rawChunkSize))

            if chunkID == "fmt " && clampedEnd - payloadStart >= 16 {
                audioFormat = readUInt16LE(payloadStart)
                channels = max(1, Int(readUInt16LE(payloadStart + 2)))
                srcSampleRate = max(1_000, Int(readUInt32LE(payloadStart + 4)))
                bitsPerSample = max(8, Int(readUInt16LE(payloadStart + 14)))
                if audioFormat == 0xFFFE && clampedEnd - payloadStart >= 26 {
                    audioFormat = readUInt16LE(payloadStart + 24)
                }
            } else if chunkID == "data" {
                dataRange = payloadStart..<clampedEnd
                break
            }

            let paddedSize = rawChunkSize + (rawChunkSize % 2)
            guard paddedSize >= 0, payloadStart + paddedSize > cursor else { break }
            cursor = payloadStart + paddedSize
        }

        guard let resolvedRange = dataRange else {
            return nil
        }

        let bytesPerSample = max(1, bitsPerSample / 8)
        let frameStride = bytesPerSample * channels
        let frameCount = resolvedRange.count / frameStride
        guard frameCount > 0 else { return [] }

        guard audioFormat == 1 || audioFormat == 3 else { return nil }

        var mono = [Float]()
        mono.reserveCapacity(frameCount)
        for frame in 0..<frameCount {
            let frameBase = resolvedRange.lowerBound + frame * frameStride
            var accum: Float = 0
            for ch in 0..<channels {
                let off = frameBase + ch * bytesPerSample
                let sample: Float
                if audioFormat == 3 && bytesPerSample == 4 {
                    let bits = readUInt32LE(off)
                    sample = Float(bitPattern: bits)
                } else if bytesPerSample == 1 {
                    sample = (Float(data[off]) - 128.0) / 128.0
                } else if bytesPerSample == 2 {
                    let s16 = Int16(bitPattern: readUInt16LE(off))
                    sample = Float(s16) / 32768.0
                } else if bytesPerSample == 3 {
                    let b0 = Int32(data[off])
                    let b1 = Int32(data[off + 1]) << 8
                    let b2 = Int32(Int8(bitPattern: data[off + 2])) << 16
                    sample = Float(b0 | b1 | b2) / 8_388_608.0
                } else if bytesPerSample == 4 {
                    let s32 = Int32(bitPattern: readUInt32LE(off))
                    sample = Float(s32) / 2_147_483_648.0
                } else {
                    sample = 0
                }
                accum += sample
            }
            mono.append(accum / Float(channels))
        }

        if abs(Double(srcSampleRate - targetSampleRate)) < 0.5 {
            return mono
        }
        return ArtifactProcessor.resampleMonoFloatPCM(
            samples: mono,
            sourceSampleRate: Double(srcSampleRate),
            targetSampleRate: Double(max(1_000, targetSampleRate))
        )
    }

    /// Encodes mono `Float32` samples (`[-1, 1]`) into a canonical 44-byte 16-bit PCM RIFF WAV container.
    public static func encodeCanonicalPCM16WAV(samples: [Float], sampleRate: Int = 16_000) -> Data {
        let clampedRate = UInt32(max(1_000, sampleRate))
        let dataBytesCount = UInt32(samples.count * 2)
        let riffChunkSize = 36 + dataBytesCount
        let byteRate = clampedRate * 2

        var out = Data()
        out.reserveCapacity(44 + Int(dataBytesCount))

        func appendU16(_ v: UInt16) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
        }
        func appendU32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
        }

        out.append(contentsOf: "RIFF".utf8)
        appendU32(riffChunkSize)
        out.append(contentsOf: "WAVE".utf8)
        out.append(contentsOf: "fmt ".utf8)
        appendU32(16)
        appendU16(1)
        appendU16(1)
        appendU32(clampedRate)
        appendU32(byteRate)
        appendU16(2)
        appendU16(16)
        out.append(contentsOf: "data".utf8)
        appendU32(dataBytesCount)

        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            let i16 = Int16((clamped * 32767.0).rounded())
            appendU16(UInt16(bitPattern: i16))
        }
        return out
    }
}
