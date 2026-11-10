import Foundation
import Observation
import SwiftUI
import UniformTypeIdentifiers
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

// MARK: - 4. Models & Provenance Inspector Surface

struct ModelsInspectorSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var showScanFolderImporter: Bool = false

    var body: some View {
        let filteredModels = viewModel.filteredModels
        VStack(spacing: 0) {
            ModelsTopActionBar(
                viewModel: viewModel,
                modelCount: filteredModels.count,
                onAddScanFolder: { showScanFolderImporter = true }
            )

            Divider()

            ScanFoldersStripView(
                viewModel: viewModel,
                onAddScanFolder: { showScanFolderImporter = true }
            )

            Divider()

            if filteredModels.isEmpty {
                ContentUnavailableView {
                    Label("No Models Registered", systemImage: "cpu")
                } description: {
                    Text("Configure scan_folders in heylook.toml to discover MLX, CoreAI, or GGUF checkpoints, or load the sample demo workspace.")
                } actions: {
                    GlassEffectContainer(spacing: 10) {
                        HStack(spacing: 10) {
                            Button("Re-scan heylook.toml") {
                                Task { await viewModel.rescanConfiguredModels() }
                            }
                            .buttonStyle(.glass)
                            .controlSize(.small)

                            Button("Load Sample Demo Workspace") {
                                Task { await viewModel.loadSampleDemoWorkspace() }
                            }
                            .buttonStyle(.glassProminent)
                            .controlSize(.small)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(filteredModels) { model in
                    ModelRegistryRowView(model: model, viewModel: viewModel)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(item: $viewModel.editingSidecarModel) { model in
            ModelSidecarEditorSheet(viewModel: viewModel, model: model)
        }
        .sheet(isPresented: $viewModel.showMLXConverterSheet) {
            MLXModelConverterSheet(
                viewModel: viewModel,
                initialModel: viewModel.convertingSourceModel
            )
        }
        .fileImporter(
            isPresented: $showScanFolderImporter,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                Task {
                    for url in urls {
                        await viewModel.addModelScanFolder(url)
                    }
                }
            }
        }
    }
}

private struct ModelsTopActionBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let modelCount: Int
    let onAddScanFolder: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Text("Registered Models (\(modelCount))")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            if HeylookConfig.isModelConfigReadOnly {
                Label("READ-ONLY", systemImage: "lock.fill")
                    .font(.caption2.monospaced().weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.orange.opacity(0.15), in: Capsule())
                    .foregroundStyle(.orange)
            }
            if let filter = viewModel.selectedProviderFilter {
                Button {
                    viewModel.selectedProviderFilter = nil
                } label: {
                    HStack(spacing: 4) {
                        Text(filter.displayName)
                        Image(systemName: "xmark.circle.fill")
                    }
                    .font(.caption2.monospaced())
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
            Spacer()
            Button(action: onAddScanFolder) {
                Label("Add Folder…", systemImage: "folder.badge.plus")
            }
            .buttonStyle(.glass)
            .controlSize(.small)
            .disabled(HeylookConfig.isModelConfigReadOnly)
            Button {
                viewModel.convertingSourceModel = nil
                viewModel.showMLXConverterSheet = true
            } label: {
                Label("Quantize / Convert MLX…", systemImage: "cube.transparent")
            }
            .buttonStyle(.glass)
            .controlSize(.small)

            Button {
                Task { await viewModel.rescanConfiguredModels() }
            } label: {
                Label("Re-scan", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.glass)
            .controlSize(.small)

            Button {
                let hasLoaded = viewModel.registeredModels.contains(where: \.isLoaded)
                    || !viewModel.schedulerMetrics.loadedGPUModels.isEmpty
                    || !viewModel.schedulerMetrics.loadedANEModels.isEmpty
                if hasLoaded {
                    viewModel.pendingConfirmation = .unloadAllModels
                } else {
                    Task { await viewModel.unloadAllModels() }
                }
            } label: {
                Label("Unload All Models…", systemImage: "eject.circle")
            }
            .buttonStyle(.glass)
            .controlSize(.small)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.bar)
    }
}

private struct ScanFoldersStripView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let onAddScanFolder: () -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                Label("scan_folders:", systemImage: "folder")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)

                if viewModel.config.scanFolders.isEmpty {
                    Text("None configured in heylook.toml")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                } else {
                    ForEach(viewModel.config.scanFolders, id: \.self) { folder in
                        HStack(spacing: 4) {
                            Text(folder)
                                .font(.system(size: 10.5, design: .monospaced))
                                .lineLimit(1)
                            if !HeylookConfig.isModelConfigReadOnly {
                                Button {
                                    Task { await viewModel.removeModelScanFolder(folder) }
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .frame(minWidth: 28, minHeight: 28)
                                        .contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                                .help("Remove scan folder from heylook.toml")
                            }
                        }
                        .padding(.leading, 8)
                        .padding(.trailing, HeylookConfig.isModelConfigReadOnly ? 8 : 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.11)))
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 4)
        }
        .background(Color.secondary.opacity(0.03))
    }
}

private struct ModelRegistryRowView: View {
    let model: EngineDescription
    @Bindable var viewModel: HeylookWorkbenchViewModel

    private static let quickContextPresets: [Int] = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    var body: some View {
        let isLoading = viewModel.loadingModelIDs.contains(model.modelID)
        let isSelected = viewModel.selectedModelID == model.modelID
        let isPureEmbedding = model.supportsEmbeddings.value
        let hasSidecar = viewModel.hasSidecarFileOnDisk(for: model)
        let fit = viewModel.assessModelRAMFit(model)
        let fitBadge: (label: String, color: Color) = {
            if !fit.fitsInMemory {
                return ("EXCEEDS RAM", .red)
            } else if fit.remainingHeadroomGB < 2.0 {
                return (String(format: "TIGHT (+%.1fGB)", fit.remainingHeadroomGB), .orange)
            } else {
                return (String(format: "FITS (+%.1fGB)", fit.remainingHeadroomGB), .green)
            }
        }()

        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .controlSize(.mini)
                } else {
                    Circle()
                        .fill(model.isLoaded ? Color.green : Color.secondary.opacity(0.35))
                        .frame(width: 8, height: 8)
                }

                Text(model.modelID)
                    .font(.headline)

                Text(model.provider.displayName)
                    .font(.caption2.monospaced().weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.15), in: Capsule())

                Text(model.computePlacement.value.label)
                    .font(.caption2.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.green.opacity(0.15), in: Capsule())

                Text(fitBadge.label)
                    .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(fitBadge.color.opacity(0.15), in: Capsule())
                    .foregroundStyle(fitBadge.color)
                    .help(String(format: "Weights %.1f GB + KV %.2f GB vs %.1f GB working set (recommended ctx %dk)", fit.modelWeightGB + fit.mmprojWeightGB + fit.drafterWeightGB, fit.estimatedKVGB, fit.availableWorkingSetGB, fit.recommendedCtxSize / 1024))

                if fit.droppedDrafterForMemory {
                    Text("MTP DROPPED")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(0.15), in: Capsule())
                        .foregroundStyle(.orange)
                }

                if model.isPinned {
                    Text("PINNED")
                        .font(.caption2.monospaced().weight(.bold))
                        .foregroundStyle(.orange)
                }

                if hasSidecar {
                    Text("SIDECAR")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.purple.opacity(0.15), in: Capsule())
                        .foregroundStyle(.purple)
                }

                Spacer()

                if !isPureEmbedding {
                    Menu {
                        ForEach(Self.quickContextPresets, id: \.self) { size in
                            Button {
                                Task { await viewModel.applyQuickContextSize(size, for: model) }
                            } label: {
                                HStack {
                                    Text("\(size / 1024)k (\(size) tokens)")
                                    if model.contextSize.value == size {
                                        Image(systemName: "checkmark")
                                    }
                                }
                            }
                        }
                    } label: {
                        Text("ctx \(max(1, model.contextSize.value / 1024))k")
                            .font(.caption2.monospaced())
                    }
                    #if os(macOS)
                    .menuStyle(.borderedButton)
                    #else
                    .menuStyle(.button)
                    #endif
                    .controlSize(.small)
                    .fixedSize()
                    .disabled(HeylookConfig.isModelConfigReadOnly)
                    .help("Quick-set context window in model.heylook.toml")
                }

                Button(model.isPinned ? "Unpin" : "Pin") {
                    Task { await viewModel.toggleModelPin(model) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    Task { await viewModel.toggleModelLoaded(model) }
                } label: {
                    if isLoading {
                        HStack(spacing: 4) {
                            ProgressView().controlSize(.mini)
                            Text("Loading…")
                        }
                    } else {
                        Text(model.isLoaded ? "Unload" : "Load")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(isLoading)

                if model.provider == .mlx {
                    Button {
                        viewModel.convertingSourceModel = model
                        viewModel.showMLXConverterSheet = true
                    } label: {
                        Label("Quantize…", systemImage: "cube.transparent")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Button {
                    viewModel.editingSidecarModel = model
                } label: {
                    Label("Sidecar", systemImage: "doc.badge.gearshape")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                if isPureEmbedding {
                    Text("Embeddings")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.purple.opacity(0.14), in: Capsule())
                        .foregroundStyle(.purple)
                        .help("Embedding models run on the Embeddings & ANE surface and /v1/embeddings")
                } else {
                    Button(isSelected ? "Active" : "Select") {
                        viewModel.selectGenerationModel(model.modelID)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(isSelected)
                }
            }

            Text(
                "Path: \(model.modelPath) • Arch: \(model.architecture.value) (\(model.architecture.provenance.rawValue)) • Quant: \(model.quantization.value) • Ctx: \(model.contextSize.value) (\(model.contextSize.provenance.rawValue)) • Reasoning: \(model.reasoningFormat.value) • Template: \(model.templateSource.value)"
            )
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)

            if !model.tags.isEmpty {
                HStack(spacing: 8) {
                    ForEach(model.tags, id: \.self) { tag in
                        Text(tag)
                            .font(.caption2.monospaced())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.12), in: Capsule())
                    }
                }
            }

            DisclosureGroup("Engine Provenance & Capabilities (\(model.estimatedMemoryMB) MB est.)") {
                VStack(alignment: .leading, spacing: 5) {
                    ProvenanceDetailRow(
                        label: "Architecture",
                        value: model.architecture.value,
                        source: model.architecture.provenance,
                        effect: model.architecture.effect
                    )
                    ProvenanceDetailRow(
                        label: "Quantization",
                        value: model.quantization.value,
                        source: model.quantization.provenance,
                        effect: model.quantization.effect
                    )
                    ProvenanceDetailRow(
                        label: "Compute Placement",
                        value: model.computePlacement.value.label,
                        source: model.computePlacement.provenance,
                        effect: model.computePlacement.effect
                    )
                    ProvenanceDetailRow(
                        label: "Context Window",
                        value: "\(model.contextSize.value) tokens",
                        source: model.contextSize.provenance,
                        effect: model.contextSize.effect
                    )
                    ProvenanceDetailRow(
                        label: "Flash Attention",
                        value: model.flashAttention.value ? "true" : "false",
                        source: model.flashAttention.provenance,
                        effect: model.flashAttention.effect
                    )
                    ProvenanceDetailRow(
                        label: "Vision / Audio / Embeddings",
                        value: "V:\(model.supportsVision.value) A:\(model.supportsAudio.value) E:\(model.supportsEmbeddings.value)",
                        source: model.supportsVision.provenance,
                        effect: model.supportsVision.effect
                    )
                    ProvenanceDetailRow(
                        label: "Thinking & Format",
                        value: "\(model.supportsThinking.value) (\(model.reasoningFormat.value))",
                        source: model.supportsThinking.provenance,
                        effect: model.supportsThinking.effect
                    )
                    ProvenanceDetailRow(
                        label: "Chat Template Source",
                        value: model.templateSource.value,
                        source: model.templateSource.provenance,
                        effect: model.templateSource.effect
                    )
                    if let drafter = model.speculativeDrafter.value, !drafter.isEmpty {
                        ProvenanceDetailRow(
                            label: "Speculative Drafter",
                            value: drafter,
                            source: model.speculativeDrafter.provenance,
                            effect: model.speculativeDrafter.effect
                        )
                    }
                    if let controls = model.thinkingControls {
                        let depthList = controls.depthValues.isEmpty
                            ? "[]"
                            : "[\(controls.depthValues.joined(separator: ","))]"
                        let disableLabel = controls.canDisableThinking ? "true" : "false"
                        let controlsLine = "ThinkingControls: switch=\(controls.switchVariable ?? "none") • depth=\(controls.depthVariable ?? "none") \(depthList) • canDisable=\(disableLabel)"
                        Text(controlsLine)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.top, 6)
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 6)
    }
}

private struct ProvenanceDetailRow: View {
    let label: String
    let value: String
    let source: ProvenanceSource
    let effect: EffectClass

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
                .frame(width: 170, alignment: .leading)
            Text(value)
                .font(.caption2.monospaced())
                .foregroundStyle(.primary)
                .lineLimit(1)
            Spacer()
            Text(effect.rawValue)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.secondary.opacity(0.1), in: Capsule())
            SamplerProvenancePill(source: source, isRequestOverride: false)
        }
    }
}

// MARK: - Pure-Swift MLX Checkpoint Quantizer Sheet

private struct MLXModelConverterSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let initialModel: EngineDescription?
    @Environment(\.dismiss) private var dismiss

    @State private var sourcePath: String = ""
    @State private var outputPath: String = ""
    @State private var mode: MLXQuantizationMode = .affine
    @State private var bits: Int = 4
    @State private var groupSize: Int = 64
    @State private var calibration: MLXQuantizationCalibration = .standard
    @State private var overwriteExisting: Bool = false
    @State private var isConverting: Bool = false
    @State private var progressMessage: String = ""
    @State private var conversionError: String?
    @State private var showFolderImporter: Bool = false
    @State private var pickingOutputFolder: Bool = false

    private var activeSpec: MLXConversionSpec {
        MLXConversionSpec(
            bits: bits,
            groupSize: groupSize,
            mode: mode,
            calibration: calibration,
            overwriteExistingOutput: overwriteExisting
        )
    }

    private var preflightResult: Result<MLXConversionPlan, Error>? {
        let src = sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
        let dst = outputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty, !dst.isEmpty else { return nil }
        return Result {
            try MLXModelConverter.planConversion(
                from: URL(fileURLWithPath: src),
                to: URL(fileURLWithPath: dst),
                spec: activeSpec
            )
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Pure-Swift MLX Checkpoint Quantizer (MLXLMCommon.convert)")
                        .font(.headline)
                    Text("Quantize unquantized BF16/FP16 .safetensors checkpoints in-process without Python")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Source .safetensors directory", text: $sourcePath)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                        .onChange(of: sourcePath) {
                            autoUpdateOutputPathIfEmptyOrDefault()
                        }
                    Button("Browse…") {
                        pickingOutputFolder = false
                        showFolderImporter = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                HStack {
                    TextField("Output quantized directory", text: $outputPath)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                    Button("Browse…") {
                        pickingOutputFolder = true
                        showFolderImporter = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            Divider()

            HStack(spacing: 14) {
                Picker("Mode", selection: $mode) {
                    ForEach(MLXQuantizationMode.allCases, id: \.self) { qMode in
                        Text(qMode.rawValue).tag(qMode)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: mode) {
                    bits = mode.defaultBits
                    groupSize = mode.defaultGroupSize
                }

                Picker("Bits", selection: $bits) {
                    Text("4-bit").tag(4)
                    Text("8-bit").tag(8)
                }
                .pickerStyle(.segmented)
                .disabled(calibration == .q4Zero)

                Picker("Group Size", selection: $groupSize) {
                    Text("32").tag(32)
                    Text("64").tag(64)
                    Text("128").tag(128)
                }
                .pickerStyle(.segmented)
                .disabled(calibration == .q4Zero)
            }

            HStack(spacing: 14) {
                Picker("Calibration", selection: $calibration) {
                    Text("standard").tag(MLXQuantizationCalibration.standard)
                    Text("q4_zero (4-bit gs=32 affine)").tag(MLXQuantizationCalibration.q4Zero)
                }
                .pickerStyle(.menu)
                .onChange(of: calibration) {
                    if calibration == .q4Zero {
                        mode = .affine
                        bits = 4
                        groupSize = 32
                    }
                }

                Toggle("Overwrite existing output directory", isOn: $overwriteExisting)
                    .font(.caption)
            }

            if let preflightResult {
                switch preflightResult {
                case .success(let plan):
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "checkmark.seal.fill")
                                .foregroundStyle(.green)
                            Text("Preflight Ready: \(plan.modelType) (\(plan.isVisionModel ? "VLM" : "LLM")) • \(plan.resolvedBits)-bit \(plan.mode.rawValue) (group_size=\(plan.resolvedGroupSize), \(plan.calibration.rawValue))")
                                .font(.caption.weight(.semibold))
                        }
                        Text("Shards: \(plan.safetensorsFiles.count) .safetensors • Sidecars: \(plan.sidecarFiles.count) files • Reasoning: \(plan.reasoningFormat)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.green.opacity(0.1)))

                case .failure(let err):
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(err.localizedDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.1)))
                }
            }

            if !progressMessage.isEmpty {
                HStack(spacing: 8) {
                    if isConverting {
                        ProgressView().controlSize(.small)
                    }
                    Text(progressMessage)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            if let conversionError {
                Text(conversionError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack {
                Button("Close") { dismiss() }
                    .buttonStyle(.bordered)
                    .disabled(isConverting)
                Spacer()
                Button("Quantize Checkpoint") {
                    runConversion()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isConverting || (try? preflightResult?.get()) == nil)
            }
        }
        .padding(20)
        .frame(minWidth: 580)
        .onAppear {
            if let initialModel {
                sourcePath = initialModel.modelPath
                outputPath = "\(initialModel.modelPath)-MLX-\(bits)bit"
            }
        }
        .fileImporter(
            isPresented: $showFolderImporter,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            _ = url.startAccessingSecurityScopedResource()
            defer { url.stopAccessingSecurityScopedResource() }
            if pickingOutputFolder {
                outputPath = url.path
            } else {
                sourcePath = url.path
                autoUpdateOutputPathIfEmptyOrDefault()
            }
        }
    }

    private func autoUpdateOutputPathIfEmptyOrDefault() {
        let trimmed = sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if outputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            outputPath = "\(trimmed)-MLX-\(activeSpec.resolvedBits)bit"
        }
    }

    private func runConversion() {
        isConverting = true
        conversionError = nil
        progressMessage = "Starting MLX quantization…"
        let srcURL = URL(fileURLWithPath: sourcePath.trimmingCharacters(in: .whitespacesAndNewlines))
        let dstURL = URL(fileURLWithPath: outputPath.trimmingCharacters(in: .whitespacesAndNewlines))
        let spec = activeSpec

        Task {
            do {
                let (_, weights) = try await MLXModelConverter.convert(
                    from: srcURL,
                    to: dstURL,
                    spec: spec,
                    progressHandler: { msg in
                        Task { @MainActor in
                            self.progressMessage = msg
                        }
                    }
                )
                await viewModel.rescanConfiguredModels()
                isConverting = false
                progressMessage = "Completed: wrote \(weights.count) quantized .safetensors shard(s) to \(dstURL.lastPathComponent)."
            } catch {
                isConverting = false
                conversionError = error.localizedDescription
            }
        }
    }
}

// MARK: - Override-Only Sidecar (`model.heylook.toml`) & Chat Template Editor Sheet

private enum TriStateOverride: String, CaseIterable, Identifiable {
    case inherit = "inherit"
    case enabled = "true"
    case disabled = "false"

    var id: String { rawValue }

    init(boolValue: Bool?) {
        switch boolValue {
        case nil: self = .inherit
        case true?: self = .enabled
        case false?: self = .disabled
        }
    }

    var boolValue: Bool? {
        switch self {
        case .inherit: return nil
        case .enabled: return true
        case .disabled: return false
        }
    }
}

private struct ModelSidecarEditorSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let model: EngineDescription
    @Environment(\.dismiss) private var dismiss

    @State private var loadedBaseSidecar: ModelSidecarConfig = ModelSidecarConfig()
    @State private var sidecarExistsOnDisk: Bool = false

    @State private var pinnedState: TriStateOverride = .inherit
    @State private var computePlacementTag: String = "__inherit__"
    @State private var ctxSizeText: String = ""
    @State private var flashAttnState: TriStateOverride = .inherit
    @State private var supportsVisionState: TriStateOverride = .inherit
    @State private var supportsThinkingState: TriStateOverride = .inherit
    @State private var imageMaxTokensText: String = ""
    @State private var draftModelText: String = ""
    @State private var specTypeText: String = ""

    @State private var temperatureText: String = ""
    @State private var topPText: String = ""
    @State private var topKText: String = ""
    @State private var minPText: String = ""
    @State private var repetitionPenaltyText: String = ""
    @State private var presencePenaltyText: String = ""
    @State private var maxTokensText: String = ""
    @State private var thinkingBudgetText: String = ""
    @State private var tagsText: String = ""
    @State private var unsetText: String = ""

    @State private var templateSourceLabel: String = "default_chatml"
    @State private var hasTemplateOverride: Bool = false
    @State private var isAuthoringTemplateOverride: Bool = false
    @State private var chatTemplateText: String = ""
    @State private var templateState: HeylookConfig.ChatTemplateState? = nil
    @State private var templateSyntaxError: String? = nil
    @State private var validationError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text("Override-Only Sidecar (model.heylook.toml) & Chat Template")
                            .font(.headline)
                        Text(sidecarExistsOnDisk ? "On Disk" : "Derived Only (No Sidecar)")
                            .font(.caption2.monospaced().weight(.semibold))
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(
                                (sidecarExistsOnDisk ? Color.orange : Color.secondary).opacity(0.15),
                                in: Capsule()
                            )
                            .foregroundStyle(sidecarExistsOnDisk ? .orange : .secondary)
                    }
                    Text("\(model.modelID) — Leave fields blank or 'Inherit' so untouched vendor/derived values are never frozen.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(templateSourceLabel)
                    .font(.caption2.monospaced().weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(
                        (hasTemplateOverride ? Color.purple : Color.secondary).opacity(0.15),
                        in: Capsule()
                    )
                    .foregroundStyle(hasTemplateOverride ? .purple : .secondary)
            }

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    // Load & Capability Overrides
                    GroupBox("Load-Time & Capability Overrides (Effect: reload)") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 14) {
                                Picker("Memory Pin (pinned)", selection: $pinnedState) {
                                    Text("Inherit (\(model.isPinned ? "true" : "false"))").tag(TriStateOverride.inherit)
                                    Text("Pinned (true)").tag(TriStateOverride.enabled)
                                    Text("LRU Evictable (false)").tag(TriStateOverride.disabled)
                                }
                                .pickerStyle(.menu)

                                Picker("Compute Unit", selection: $computePlacementTag) {
                                    Text("Inherit (\(model.computePlacement.value.label))").tag("__inherit__")
                                    ForEach(ComputePlacement.allCases, id: \.rawValue) { placement in
                                        Text(placement.label).tag(placement.rawValue)
                                    }
                                }
                                .pickerStyle(.menu)
                            }

                            HStack(spacing: 12) {
                                SidecarOptionalField(
                                    title: "Context Window (ctx_size)",
                                    placeholder: "Inherit (\(model.contextSize.value))",
                                    text: $ctxSizeText
                                )
                                SidecarOptionalField(
                                    title: "Image Max Tokens",
                                    placeholder: "Inherit (auto)",
                                    text: $imageMaxTokensText
                                )
                            }

                            let previewCtx = Int(ctxSizeText.trimmingCharacters(in: .whitespacesAndNewlines))
                            let fit = viewModel.assessModelRAMFit(
                                model,
                                ctxSizeOverride: previewCtx,
                                sidecarOverride: loadedBaseSidecar
                            )
                            HStack(spacing: 8) {
                                Image(systemName: fit.fitsInMemory ? "memorychip" : "exclamationmark.triangle.fill")
                                    .foregroundStyle(fit.fitsInMemory ? (fit.remainingHeadroomGB < 2.0 ? .orange : .green) : .red)
                                    .font(.caption)
                                Text(
                                    String(
                                        format: "RAM Fit Preview: %.1f GB weights + %.2f GB KV vs %.1f GB working set (%+.1f GB headroom • recommended ctx %dk)",
                                        fit.modelWeightGB + fit.mmprojWeightGB + fit.drafterWeightGB,
                                        fit.estimatedKVGB,
                                        fit.availableWorkingSetGB,
                                        fit.remainingHeadroomGB,
                                        max(1, fit.recommendedCtxSize / 1024)
                                    )
                                )
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            }

                            HStack(spacing: 12) {
                                Picker("Flash Attn", selection: $flashAttnState) {
                                    Text("Inherit (\(model.flashAttention.value ? "true" : "false"))").tag(TriStateOverride.inherit)
                                    Text("true").tag(TriStateOverride.enabled)
                                    Text("false").tag(TriStateOverride.disabled)
                                }
                                .pickerStyle(.menu)

                                Picker("Vision", selection: $supportsVisionState) {
                                    Text("Inherit (\(model.supportsVision.value ? "true" : "false"))").tag(TriStateOverride.inherit)
                                    Text("true").tag(TriStateOverride.enabled)
                                    Text("false").tag(TriStateOverride.disabled)
                                }
                                .pickerStyle(.menu)

                                Picker("Thinking", selection: $supportsThinkingState) {
                                    Text("Inherit (\(model.supportsThinking.value ? "true" : "false"))").tag(TriStateOverride.inherit)
                                    Text("true").tag(TriStateOverride.enabled)
                                    Text("false").tag(TriStateOverride.disabled)
                                }
                                .pickerStyle(.menu)
                            }

                            HStack(spacing: 12) {
                                SidecarOptionalField(
                                    title: "Speculative Drafter (draft_model)",
                                    placeholder: model.speculativeDrafter.value ?? "Inherit (none)",
                                    text: $draftModelText
                                )
                                SidecarOptionalField(
                                    title: "Spec Type (spec_type)",
                                    placeholder: "Inherit (none)",
                                    text: $specTypeText
                                )
                            }
                        }
                        .padding(6)
                    }

                    // Default Sampler Overrides
                    GroupBox("Model-Level Sampler Overrides (Effect: next_turn)") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 10) {
                                SidecarOptionalField(title: "temperature", placeholder: "Inherit", text: $temperatureText)
                                SidecarOptionalField(title: "top_p", placeholder: "Inherit", text: $topPText)
                                SidecarOptionalField(title: "top_k", placeholder: "Inherit", text: $topKText)
                                SidecarOptionalField(title: "min_p", placeholder: "Inherit", text: $minPText)
                            }
                            HStack(spacing: 10) {
                                SidecarOptionalField(title: "repetition_penalty", placeholder: "Inherit", text: $repetitionPenaltyText)
                                SidecarOptionalField(title: "presence_penalty", placeholder: "Inherit", text: $presencePenaltyText)
                                SidecarOptionalField(title: "max_tokens", placeholder: "Inherit", text: $maxTokensText)
                                SidecarOptionalField(
                                    title: "thinking_budget",
                                    placeholder: model.thinkingBudget.value.map { "Inherit (\($0))" } ?? "Inherit",
                                    text: $thinkingBudgetText
                                )
                            }
                            HStack(spacing: 10) {
                                SidecarOptionalField(
                                    title: "Tags Override (comma-separated)",
                                    placeholder: model.tags.isEmpty ? "Inherit (none)" : "Inherit (\(model.tags.joined(separator: ", ")))",
                                    text: $tagsText
                                )
                                SidecarOptionalField(
                                    title: "Unset Keys (unset = [...])",
                                    placeholder: "e.g. draft_model, spec_type",
                                    text: $unsetText
                                )
                            }
                        }
                        .padding(6)
                    }

                    // 4-Rung Chat Template Override
                    GroupBox("4-Rung Chat Template Override (chat_template.heylook.jinja)") {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(spacing: 6) {
                                Text(
                                    hasTemplateOverride
                                        ? "Active local override: chat_template.heylook.jinja"
                                        : "Read-only preview from \(templateSourceLabel)"
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)

                                if let state = templateState {
                                    if !state.writable {
                                        Text("READ-ONLY")
                                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                                            .padding(.horizontal, 5)
                                            .padding(.vertical, 1)
                                            .background(Capsule().fill(Color.orange.opacity(0.15)))
                                            .foregroundStyle(.orange)
                                    }
                                    if state.modifiedSinceDownload == true {
                                        Text("MODIFIED SINCE DOWNLOAD")
                                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                                            .padding(.horizontal, 5)
                                            .padding(.vertical, 1)
                                            .background(Capsule().fill(Color.yellow.opacity(0.2)))
                                            .foregroundStyle(.orange)
                                    }
                                }

                                Spacer()

                                if !hasTemplateOverride && !isAuthoringTemplateOverride {
                                    Button("Start Override from This") {
                                        isAuthoringTemplateOverride = true
                                        templateSyntaxError = viewModel.validateJinjaTemplateSyntax(chatTemplateText)
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                    .disabled(templateState?.writable == false)
                                }

                                if hasTemplateOverride {
                                    Button("Revert Override", role: .destructive) {
                                        Task {
                                            let reverted = await viewModel.revertChatTemplateOverride(for: model)
                                            templateState = reverted
                                            templateSourceLabel = reverted.source
                                            hasTemplateOverride = reverted.hasOverride
                                            isAuthoringTemplateOverride = false
                                            chatTemplateText = reverted.template
                                            templateSyntaxError = nil
                                        }
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                    .disabled(templateState?.writable == false)
                                }

                                if hasTemplateOverride || isAuthoringTemplateOverride {
                                    Button("Save Template Override") {
                                        Task {
                                            do {
                                                let saved = try await viewModel.saveChatTemplateOverride(chatTemplateText, for: model)
                                                templateState = saved
                                                templateSourceLabel = saved.source
                                                hasTemplateOverride = saved.hasOverride
                                                isAuthoringTemplateOverride = true
                                                templateSyntaxError = nil
                                            } catch {
                                                templateSyntaxError = error.localizedDescription
                                            }
                                        }
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .controlSize(.mini)
                                    .disabled(
                                        chatTemplateText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                            || templateState?.writable == false
                                            || templateSyntaxError != nil
                                    )
                                }
                            }

                            if let state = templateState {
                                if let inert = state.inertReason, !inert.isEmpty {
                                    Label("Template Inert: \(inert)", systemImage: "info.circle.fill")
                                        .font(.caption2)
                                        .foregroundStyle(.orange)
                                }
                                if state.stale == true {
                                    Label("Local override may be stale relative to updated checkpoint tokenizer_config.json.", systemImage: "exclamationmark.triangle.fill")
                                        .font(.caption2)
                                        .foregroundStyle(.orange)
                                }
                                if state.prefixStable == false {
                                    Label(
                                        "Prefix Unstable (KV cache invalidated across turns): \(state.prefixNote ?? "template mutates history prefix")",
                                        systemImage: "bolt.trianglebadge.exclamationmark.fill"
                                    )
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                                }
                                TemplateCapabilitiesStripView(caps: state.caps)
                            }

                            TextEditor(text: $chatTemplateText)
                                .font(.system(.caption, design: .monospaced))
                                .disabled(!hasTemplateOverride && !isAuthoringTemplateOverride)
                                .opacity((hasTemplateOverride || isAuthoringTemplateOverride) ? 1.0 : 0.78)
                                .frame(minHeight: 110, maxHeight: 165)
                                .overlay {
                                    RoundedRectangle(cornerRadius: 6)
                                        .strokeBorder(
                                            templateSyntaxError != nil
                                                ? Color.red.opacity(0.6)
                                                : Color.secondary.opacity(0.22),
                                            lineWidth: 1
                                        )
                                }
                                .onChange(of: chatTemplateText) { _, newValue in
                                    if hasTemplateOverride || isAuthoringTemplateOverride {
                                        templateSyntaxError = viewModel.validateJinjaTemplateSyntax(newValue)
                                    }
                                }

                            if let syntaxErr = templateSyntaxError {
                                Label(syntaxErr, systemImage: "xmark.octagon.fill")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.red)
                            }
                        }
                        .padding(6)
                    }
                }
            }

            if let validationError {
                Text(validationError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.bordered)

                if sidecarExistsOnDisk {
                    Button("Remove All Sidecar Overrides", role: .destructive) {
                        Task {
                            await viewModel.saveModelSidecar(ModelSidecarConfig(), for: model)
                            dismiss()
                        }
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()

                Button("Save model.heylook.toml") {
                    saveSidecarOverrides()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 660, idealWidth: 720, minHeight: 580)
        .onAppear {
            sidecarExistsOnDisk = viewModel.hasSidecarFileOnDisk(for: model)
            let loaded = viewModel.loadSidecarConfig(for: model)
            loadedBaseSidecar = loaded

            pinnedState = TriStateOverride(boolValue: loaded.pinned)
            computePlacementTag = loaded.computePlacement?.rawValue ?? "__inherit__"
            ctxSizeText = loaded.ctxSize.map(String.init) ?? ""
            flashAttnState = TriStateOverride(boolValue: loaded.flashAttn)
            supportsVisionState = TriStateOverride(boolValue: loaded.supportsVision)
            supportsThinkingState = TriStateOverride(boolValue: loaded.supportsThinking)
            imageMaxTokensText = loaded.imageMaxTokens.map(String.init) ?? ""
            draftModelText = loaded.draftModel ?? ""
            specTypeText = loaded.specType ?? ""

            temperatureText = loaded.sampling.temperature.map { String($0) } ?? ""
            topPText = loaded.sampling.topP.map { String($0) } ?? ""
            topKText = loaded.sampling.topK.map(String.init) ?? ""
            minPText = loaded.sampling.minP.map { String($0) } ?? ""
            repetitionPenaltyText = loaded.sampling.repetitionPenalty.map { String($0) } ?? ""
            presencePenaltyText = loaded.sampling.presencePenalty.map { String($0) } ?? ""
            maxTokensText = loaded.sampling.maxTokens.map(String.init) ?? ""
            thinkingBudgetText = loaded.sampling.thinkingBudget.map(String.init) ?? ""
            tagsText = loaded.tags.joined(separator: ", ")
            unsetText = loaded.unsetKeys.joined(separator: ", ")

            let tmplState = viewModel.loadChatTemplateState(for: model)
            templateState = tmplState
            templateSourceLabel = tmplState.source
            hasTemplateOverride = tmplState.hasOverride
            isAuthoringTemplateOverride = tmplState.hasOverride
            chatTemplateText = tmplState.template
            templateSyntaxError = nil
        }
    }

    private func parseOptionalInt(_ raw: String, fieldName: String) throws -> Int? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let v = Int(trimmed) else {
            throw SidecarValidationError(message: "Invalid integer for \(fieldName): '\(trimmed)'")
        }
        return v
    }

    private func parseOptionalDouble(_ raw: String, fieldName: String) throws -> Double? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let v = Double(trimmed), v.isFinite else {
            throw SidecarValidationError(message: "Invalid number for \(fieldName): '\(trimmed)'")
        }
        return v
    }

    private func saveSidecarOverrides() {
        validationError = nil
        do {
            var updated = loadedBaseSidecar
            // Preserve explicit id/provider ONLY if already authored in model.heylook.toml!
            updated.id = loadedBaseSidecar.id
            updated.provider = loadedBaseSidecar.provider
            updated.pinned = pinnedState.boolValue
            updated.computePlacement = computePlacementTag == "__inherit__" ? nil : ComputePlacement(rawValue: computePlacementTag)
            updated.ctxSize = try parseOptionalInt(ctxSizeText, fieldName: "ctx_size")
            updated.flashAttn = flashAttnState.boolValue
            updated.supportsVision = supportsVisionState.boolValue
            updated.supportsThinking = supportsThinkingState.boolValue
            updated.imageMaxTokens = try parseOptionalInt(imageMaxTokensText, fieldName: "image_max_tokens")

            let trimmedDraft = draftModelText.trimmingCharacters(in: .whitespacesAndNewlines)
            updated.draftModel = trimmedDraft.isEmpty ? nil : trimmedDraft
            let trimmedSpec = specTypeText.trimmingCharacters(in: .whitespacesAndNewlines)
            updated.specType = trimmedSpec.isEmpty ? nil : trimmedSpec

            updated.sampling.temperature = try parseOptionalDouble(temperatureText, fieldName: "temperature")
            updated.sampling.topP = try parseOptionalDouble(topPText, fieldName: "top_p")
            updated.sampling.topK = try parseOptionalInt(topKText, fieldName: "top_k")
            updated.sampling.minP = try parseOptionalDouble(minPText, fieldName: "min_p")
            updated.sampling.repetitionPenalty = try parseOptionalDouble(repetitionPenaltyText, fieldName: "repetition_penalty")
            updated.sampling.presencePenalty = try parseOptionalDouble(presencePenaltyText, fieldName: "presence_penalty")
            updated.sampling.maxTokens = try parseOptionalInt(maxTokensText, fieldName: "max_tokens")
            updated.sampling.thinkingBudget = try parseOptionalInt(thinkingBudgetText, fieldName: "thinking_budget")

            updated.tags = tagsText
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            updated.unsetKeys = unsetText
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }

            Task {
                await viewModel.saveModelSidecar(updated, for: model)
                dismiss()
            }
        } catch let err as SidecarValidationError {
            validationError = err.message
        } catch {
            validationError = error.localizedDescription
        }
    }
}

private struct SidecarValidationError: Error {
    let message: String
}

private struct SidecarOptionalField: View {
    let title: String
    let placeholder: String
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button {
                        text = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear override (revert to Inherit)")
                }
            }
            TextField(placeholder, text: $text)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospaced())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct TemplateCapabilitiesStripView: View {
    let caps: TemplateCapabilities

    private var badges: [(label: String, active: Bool)] {
        [
            ("system", caps.supportsSystemRole),
            ("tools", caps.supportsTools),
            ("vision", caps.supportsVision),
            ("audio", caps.supportsAudio),
            ("thinking", caps.supportsThinking),
            ("alternating", caps.requiresAlternatingRoles),
            ("typed_content", caps.requiresTypedContent)
        ]
    }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 5) {
                Text("caps:")
                    .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)
                ForEach(badges, id: \.label) { item in
                    Text(item.label)
                        .font(.system(size: 9, weight: item.active ? .bold : .regular, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule().fill(
                                item.active
                                    ? Color.teal.opacity(0.16)
                                    : Color.secondary.opacity(0.08)
                            )
                        )
                        .foregroundStyle(item.active ? .teal : .secondary)
                }
            }
        }
    }
}

// MARK: - 5. Perf & Server Surface

struct PerfServerSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        Form {
            Section("Unified Memory & Dual-Lane Compute Residency (\(viewModel.memorySnapshot.deviceName))") {
                LabeledContent("Physical Unified RAM") {
                    Text(String(format: "%.1f GB total • %.1f GB used • %.1f GB avail", viewModel.memorySnapshot.totalPhysicalRAMGB, viewModel.memorySnapshot.ramUsedGB, viewModel.memorySnapshot.ramAvailableGB))
                        .font(.caption.monospaced().weight(.semibold))
                }
                LabeledContent("Metal Recommended Max Working Set") {
                    Text(String(format: "%.1f GB", viewModel.memorySnapshot.metalRecommendedMaxWorkingSetGB))
                        .font(.subheadline.monospaced().weight(.semibold))
                }
                LabeledContent("Metal Allocated / Process RSS / Host CPU") {
                    Text(String(format: "%d MB Metal • %d MB RSS • %.1f%% CPU", viewModel.memorySnapshot.metalCurrentAllocatedMB, viewModel.memorySnapshot.processResidentMB, viewModel.memorySnapshot.cpuPercent))
                        .font(.caption.monospaced().weight(.semibold))
                }
                LabeledContent("Resident GPU / ANE Models") {
                    HStack(spacing: 10) {
                        Text("\(viewModel.schedulerMetrics.loadedGPUModels.count) GPU • \(viewModel.schedulerMetrics.loadedANEModels.count) ANE")
                            .font(.subheadline.monospaced().weight(.semibold))
                        Button("Refresh Telemetry") {
                            Task { await viewModel.refreshMetrics() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }

            Section {
                LabeledContent("Active llama-server Subprocesses") {
                    Text("\(viewModel.activeSubprocessCount)")
                        .font(.subheadline.monospaced().weight(.semibold))
                        .foregroundStyle(viewModel.activeSubprocessCount > 0 ? .orange : .green)
                }
                LabeledContent("Allocated GGUF Loopback Ports") {
                    if viewModel.sortedActiveGGUFPorts.isEmpty {
                        Text("None (18080+ free)")
                            .font(.caption.monospaced().weight(.medium))
                            .foregroundStyle(.secondary)
                    } else {
                        HStack(spacing: 6) {
                            ForEach(viewModel.sortedActiveGGUFPorts, id: \.self) { allocation in
                                HStack(spacing: 4) {
                                    Text(allocation)
                                        .font(.caption2.monospaced().weight(.medium))
                                    Button("Unload") {
                                        Task { await viewModel.unloadGGUFAllocation(allocation) }
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                }
                            }
                        }
                    }
                }
                LabeledContent("Memory & Subprocess Reclamation") {
                    Button("Unload All & Reclaim Memory…", role: .destructive) {
                        let hasLoaded = viewModel.registeredModels.contains(where: \.isLoaded)
                            || !viewModel.schedulerMetrics.loadedGPUModels.isEmpty
                            || !viewModel.schedulerMetrics.loadedANEModels.isEmpty
                        if hasLoaded {
                            viewModel.pendingConfirmation = .unloadAllModels
                        } else {
                            Task { await viewModel.unloadAllModels() }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
            } header: {
                Text("Lifecycle Reclamation, Subprocess Registry & GGUF Port Pool")
            } footer: {
                Text("Calls `provider.unload()`, `MLX.Memory.clearCache()`, `ANECoreMLRuntime.unload()`, and `SubprocessLifecycleRegistry.shared.terminateAll()` (`SIGTERM` -> `SIGKILL`).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.aneInspectionReports.isEmpty {
                Section("ANE Tower Compute Plan (EmbeddingGemma 2 MLComputePlan)") {
                    ForEach(EmbeddingComponent.allCases) { component in
                        if let report = viewModel.aneInspectionReports.first(where: { $0.component == component }) {
                            let unitBadge = viewModel[computeUnitFor: component].shortBadge
                            let residencyPct = Int((report.aneResidencyFraction * 100).rounded())
                            let summaryText = report.isCompiledBundlePresent
                                ? "\(residencyPct)% ANE ops (\(unitBadge))"
                                : "MLX GPU fallback (\(unitBadge))"
                            LabeledContent(component.title) {
                                HStack(spacing: 8) {
                                    Circle()
                                        .fill(report.isCompiledBundlePresent ? Color.green : Color.orange)
                                        .frame(width: 7, height: 7)
                                    Text(summaryText)
                                        .font(.caption.monospaced())
                                }
                            }
                        }
                    }
                }
            }

            Section("Recent Request Telemetry (TPS, KV Prefix Cache & Speculative Acceptance)") {
                let assistantTurns = viewModel.currentMessages.filter { $0.role == "assistant" }.suffix(6)
                if assistantTurns.isEmpty {
                    Text("No assistant turns recorded in the active conversation yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(assistantTurns)) { turn in
                        HStack(spacing: 10) {
                            Text(turn.modelID ?? "assistant")
                                .font(.caption2.monospaced().weight(.semibold))
                                .lineLimit(1)
                            if let cache = turn.cacheReport {
                                Text(cache.summaryBadge)
                                    .font(.caption2.monospaced())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.blue.opacity(0.12), in: Capsule())
                            }
                            if let spec = turn.speculativeReport {
                                Text(spec.summaryBadge)
                                    .font(.caption2.monospaced())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.purple.opacity(0.14), in: Capsule())
                            }
                            Spacer()
                            if let tps = turn.generationTPS, tps > 0 {
                                Text(String(format: "%.1f tok/s", tps))
                                    .font(.caption2.monospaced().weight(.bold))
                                    .foregroundStyle(.green)
                            }
                        }
                    }
                }
            }

            Section {
                LabeledContent("HTTP & SSE Status") {
                    HStack(spacing: 10) {
                        Circle()
                            .fill(viewModel.isServerRunning ? Color.green : Color.secondary)
                            .frame(width: 8, height: 8)
                        Text(
                            viewModel.isServerRunning
                                ? "Listening on http://127.0.0.1:\(viewModel.serverPort)"
                                : "Embedded server stopped"
                        )
                        .font(.caption.monospaced())
                        Button(viewModel.isServerRunning ? "Stop Server" : "Start Server") {
                            Task { await viewModel.toggleServer() }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                }
            } header: {
                Text("Embedded HTTP & SSE Server (/v1/messages, /v1/embeddings, /v1/conversations)")
            } footer: {
                Text("Supports disconnect-resilient background generation (`heylook_saved`) for iOS 27.2+ mobile clients over LAN/Tailscale.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Bonjour (_heylook._tcp) Auto-Discovery & Remote Mac Studio Connection") {
                LabeledContent("LAN Discovery") {
                    HStack(spacing: 8) {
                        Text(viewModel.isBrowsingBonjour ? "Scanning LAN for _heylook._tcp peers…" : "Idle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Button(viewModel.isBrowsingBonjour ? "Stop Scan" : "Scan Bonjour") {
                            if viewModel.isBrowsingBonjour {
                                viewModel.stopBonjourDiscovery()
                            } else {
                                viewModel.startBonjourDiscovery()
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }

                ForEach(viewModel.discoveredServers) { peer in
                    LabeledContent {
                        Button("Connect") {
                            Task { await viewModel.connectToRemoteServer(peer.baseURLString) }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "desktopcomputer")
                                .foregroundStyle(.tint)
                            Text(peer.name)
                                .font(.caption.weight(.semibold))
                            Text(peer.baseURLString)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                HStack(spacing: 8) {
                    TextField("http://mac-studio.local:8000 or Tailscale URL", text: $viewModel.remoteServerURLString)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                    Button("Connect Remote") {
                        Task { await viewModel.connectToRemoteServer(viewModel.remoteServerURLString) }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }

                if let connected = viewModel.connectedRemoteServerURL {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Connected to remote Mac Studio: \(connected)")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // Persisted Performance Profile (`1h / 6h / 24h / 7d`)
            Section {
                HStack {
                    Picker("Time Window", selection: $viewModel.selectedPerfWindow) {
                        ForEach(PerfTimeWindow.allCases) { window in
                            Text(window.label).tag(window)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 260)

                    Spacer()

                    Button {
                        viewModel.refreshPerfAndObservability()
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                if let profile = viewModel.perfProfileSummary, profile.sampleCount > 0 {
                    let cacheRatio = profile.totalPromptTokens > 0
                        ? (Double(profile.totalCachedTokens) / Double(profile.totalPromptTokens)) * 100.0
                        : 0.0
                    HStack(spacing: 16) {
                        PerfStatBadge(title: "Samples", value: "\(profile.sampleCount)")
                        PerfStatBadge(title: "Avg Gen", value: String(format: "%.1f tok/s", profile.averageGenerationTPS))
                        PerfStatBadge(title: "Avg Prompt", value: String(format: "%.0f tok/s", profile.averagePromptTPS))
                        PerfStatBadge(title: "Avg TTFT", value: String(format: "%.0f ms", profile.averageTTFTMs))
                        PerfStatBadge(title: "Output Tok", value: "\(profile.totalOutputTokens)")
                        PerfStatBadge(title: "KV Cache Hit", value: String(format: "%.0f%%", cacheRatio))
                    }

                    ForEach(profile.byModel) { row in
                        HStack(spacing: 10) {
                            Text(row.modelID)
                                .font(.caption.monospaced().weight(.semibold))
                                .lineLimit(1)
                            Text("\(row.sampleCount) turns")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(String(format: "Gen %.1f tok/s • Prompt %.0f tok/s • TTFT %.0f ms", row.averageGenerationTPS, row.averagePromptTPS, row.averageTTFTMs))
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                } else {
                    Text("No persisted turn statistics recorded in SQLite within the \(viewModel.selectedPerfWindow.label) window yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Persisted Performance Profile (/v1/performance/profile)")
            }

            // Structured JSONL Observability Event Log (`HeylookObservabilityLog`)
            Section {
                HStack(spacing: 10) {
                    Picker("Record Threshold", selection: $viewModel.boundObservabilityLoggerLevel) {
                        Text("Record: DEBUG+").tag("DEBUG")
                        Text("Record: INFO+").tag("INFO")
                        Text("Record: WARN+").tag("WARN")
                        Text("Record: ERROR").tag("ERROR")
                    }
                    .controlSize(.small)
                    .frame(maxWidth: 175)

                    Picker(
                        "Filter",
                        selection: Binding(
                            get: { viewModel.observabilityMinLevelFilter ?? "__all__" },
                            set: { viewModel.observabilityMinLevelFilter = ($0 == "__all__") ? nil : $0 }
                        )
                    ) {
                        Text("View: All").tag("__all__")
                        Text("INFO+").tag("INFO")
                        Text("WARN+").tag("WARN")
                        Text("ERROR").tag("ERROR")
                    }
                    .controlSize(.small)
                    .frame(maxWidth: 140)

                    TextField("Search message, category, or request ID…", text: $viewModel.observabilitySearchQuery)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())

                    Button {
                        copyTextToPasteboard(viewModel.exportObservabilityJSONL())
                        viewModel.statusBanner = "Copied \(viewModel.observabilityEntries.count) observability events as JSONL."
                    } label: {
                        Label("Copy JSONL", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.observabilityEntries.isEmpty)

                    Button("Clear") {
                        viewModel.clearObservabilityEntries()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.observabilityEntries.isEmpty)
                }

                if viewModel.observabilityEntries.isEmpty {
                    Text("No structured observability events match the active filter.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(viewModel.observabilityEntries.prefix(80))) { entry in
                                ObservabilityLogRowView(entry: entry)
                            }
                        }
                        .padding(4)
                    }
                    .frame(maxHeight: 240)
                }
            } header: {
                Text("Structured Observability Event Log (JSONL Ring)")
            }

            // Data Maintenance & Store Reset
            Section("SQLite Data Maintenance") {
                LabeledContent("Persisted Store Records") {
                    HStack(spacing: 12) {
                        Text("\(viewModel.conversations.count) conversations • \(viewModel.notebooks.count) notebooks • \(viewModel.presets.count) presets")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                        Button("Clear All Conversations & Notebooks…", role: .destructive) {
                            viewModel.pendingConfirmation = .clearAllConversationsAndNotebooks
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            viewModel.refreshPerfAndObservability()
        }
    }

    private func copyTextToPasteboard(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = text
        #endif
    }
}

private struct PerfStatBadge: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 9.5, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .bold, design: .monospaced))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.09)))
    }
}

private struct ObservabilityLogRowView: View {
    let entry: HeylookObservabilityLog.Entry

    private static let timeFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss.SSS"
        return df
    }()

    private var levelColor: Color {
        switch entry.level {
        case "ERROR": return .red
        case "WARN": return .orange
        case "DEBUG": return .secondary
        default: return .blue
        }
    }

    var body: some View {
        let date = Date(timeIntervalSince1970: Double(entry.timestampMs) / 1000.0)
        let metaSummary = entry.metadata
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ")

        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(Self.timeFormatter.string(from: date))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.tertiary)

            Text(entry.level)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Capsule().fill(levelColor.opacity(0.15)))
                .foregroundStyle(levelColor)

            Text("[\(entry.category)]")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)

            Text(entry.message)
                .font(.system(size: 10.5, design: .monospaced))
                .lineLimit(2)

            if !metaSummary.isEmpty {
                Text(metaSummary)
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)
        }
        .padding(.vertical, 2)
    }
}
