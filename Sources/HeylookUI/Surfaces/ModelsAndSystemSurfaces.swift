import Foundation
import Observation
import SwiftUI
import HeylookCore
import HeylookEmbeddings
import HeylookEngines
import HeylookServer

#if os(macOS)
import AppKit
#endif

// MARK: - 4. Models & Provenance Inspector Surface

struct ModelsInspectorSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        let filteredModels = viewModel.filteredModels
        VStack(spacing: 0) {
            ModelsTopActionBar(
                viewModel: viewModel,
                modelCount: filteredModels.count
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
    }
}

private struct ModelsTopActionBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let modelCount: Int

    var body: some View {
        HStack(spacing: 8) {
            Text("Registered Models (\(modelCount))")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
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

private struct ModelRegistryRowView: View {
    let model: EngineDescription
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle()
                    .fill(model.isLoaded ? Color.green : Color.secondary.opacity(0.35))
                    .frame(width: 8, height: 8)
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
                if model.isPinned {
                    Text("PINNED")
                        .font(.caption2.monospaced().weight(.bold))
                        .foregroundStyle(.orange)
                }
                Spacer()

                Button(model.isPinned ? "Unpin" : "Pin") {
                    Task { await viewModel.toggleModelPin(model) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(model.isLoaded ? "Unload" : "Load") {
                    Task { await viewModel.toggleModelLoaded(model) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

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

                Button("Select") {
                    viewModel.selectedModelID = model.modelID
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            Text(
                "Path: \(model.modelPath) • Arch: \(model.architecture.value) (\(model.architecture.provenance.rawValue)) • Quant: \(model.quantization.value) • Ctx: \(model.contextSize.value) (\(model.contextSize.provenance.rawValue)) • Reasoning: \(model.reasoningFormat.value) • Template: \(model.templateSource.value)"
            )
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                ForEach(model.tags, id: \.self) { tag in
                    Text(tag)
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.12), in: Capsule())
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
                    #if os(macOS)
                    Button("Browse…") {
                        pickFolder(isOutput: false)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    #endif
                }

                HStack {
                    TextField("Output quantized directory", text: $outputPath)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                    #if os(macOS)
                    Button("Browse…") {
                        pickFolder(isOutput: true)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    #endif
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

    #if os(macOS)
    private func pickFolder(isOutput: Bool) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = isOutput
        panel.allowsMultipleSelection = false
        panel.prompt = isOutput ? "Select Output Folder" : "Select Source Checkpoint"
        if panel.runModal() == .OK, let url = panel.url {
            if isOutput {
                outputPath = url.path
            } else {
                sourcePath = url.path
                autoUpdateOutputPathIfEmptyOrDefault()
            }
        }
    }
    #endif
}

private struct ModelSidecarEditorSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let model: EngineDescription
    @Environment(\.dismiss) private var dismiss

    @State private var loadedBaseSidecar: ModelSidecarConfig = ModelSidecarConfig()
    @State private var pinned: Bool = false
    @State private var computePlacement: ComputePlacement = .mlxGPU
    @State private var ctxSize: Int = 8192
    @State private var flashAttn: Bool = true
    @State private var supportsVision: Bool = false
    @State private var supportsThinking: Bool = true
    @State private var imageMaxTokensText: String = ""
    @State private var draftModelText: String = ""
    @State private var temperature: Double = 0.7
    @State private var topP: Double = 0.95
    @State private var thinkingBudget: Int = 4096
    @State private var tagsText: String = ""
    @State private var unsetText: String = ""
    @State private var templateSourceLabel: String = "default_chatml"
    @State private var hasTemplateOverride: Bool = false
    @State private var chatTemplateText: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Edit model.heylook.toml Sidecar & Chat Template")
                        .font(.headline)
                    Text(model.modelID)
                        .font(.caption.monospaced())
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

            Toggle("Pin Model in Unified Memory (Exempt from GPU LRU Eviction)", isOn: $pinned)
            HStack(spacing: 16) {
                Picker("Compute Placement (compute_unit)", selection: $computePlacement) {
                    ForEach(ComputePlacement.allCases, id: \.self) { placement in
                        Text(placement.label).tag(placement)
                    }
                }
                .pickerStyle(.menu)

                Stepper("Context Size (ctx_size): \(ctxSize)", value: $ctxSize, in: 2048...1_048_576, step: 4096)
                    .font(.subheadline.monospacedDigit())
            }
            HStack(spacing: 16) {
                Toggle("Flash Attention (flash_attn)", isOn: $flashAttn)
                Toggle("Supports Vision (vision)", isOn: $supportsVision)
                Toggle("Supports Thinking (thinking_capable)", isOn: $supportsThinking)
            }
            HStack(spacing: 10) {
                TextField("Image Max Tokens (image_max_tokens, optional)", text: $imageMaxTokensText)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption.monospaced())
                TextField("Speculative Drafter Path (draft_model, optional)", text: $draftModelText)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption.monospaced())
            }

            Divider()

            HStack {
                Text("Default Temperature: \(String(format: "%.2f", temperature))")
                    .font(.subheadline.monospacedDigit())
                Slider(value: $temperature, in: 0...2.0)
            }
            HStack {
                Text("Default Top-P: \(String(format: "%.2f", topP))")
                    .font(.subheadline.monospacedDigit())
                Slider(value: $topP, in: 0.05...1.0)
            }
            Stepper("Default Thinking Budget: \(thinkingBudget)", value: $thinkingBudget, in: 256...65536, step: 512)
                .font(.subheadline.monospacedDigit())

            TextField("Tags (comma-separated, e.g. chat, vision, mtp)", text: $tagsText)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospaced())

            TextField("Unset Keys (comma-separated, e.g. draft_model_path, spec_type)", text: $unsetText)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospaced())

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("4-Rung Chat Template Override (chat_template.heylook.jinja)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if hasTemplateOverride {
                        Button("Revert Override", role: .destructive) {
                            Task {
                                let reverted = await viewModel.revertChatTemplateOverride(for: model)
                                templateSourceLabel = reverted.source
                                hasTemplateOverride = reverted.hasOverride
                                chatTemplateText = reverted.template
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                    Button("Save Template Override") {
                        Task {
                            let saved = await viewModel.saveChatTemplateOverride(chatTemplateText, for: model)
                            templateSourceLabel = saved.source
                            hasTemplateOverride = saved.hasOverride
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(chatTemplateText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                TextEditor(text: $chatTemplateText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(minHeight: 100, maxHeight: 140)
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(Color.secondary.opacity(0.22), lineWidth: 1)
                    }
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Save model.heylook.toml") {
                    let parsedTags = tagsText
                        .split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    let parsedUnset = unsetText
                        .split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    let parsedImageMaxTokens = Int(imageMaxTokensText.trimmingCharacters(in: .whitespacesAndNewlines))
                    let trimmedDraft = draftModelText.trimmingCharacters(in: .whitespacesAndNewlines)

                    var updatedSidecar = loadedBaseSidecar
                    updatedSidecar.id = model.modelID
                    updatedSidecar.provider = model.provider
                    updatedSidecar.pinned = pinned
                    updatedSidecar.computePlacement = computePlacement
                    updatedSidecar.ctxSize = ctxSize
                    updatedSidecar.flashAttn = flashAttn
                    updatedSidecar.supportsVision = supportsVision
                    updatedSidecar.supportsThinking = supportsThinking
                    updatedSidecar.imageMaxTokens = parsedImageMaxTokens
                    updatedSidecar.draftModel = trimmedDraft.isEmpty ? nil : trimmedDraft
                    updatedSidecar.unsetKeys = parsedUnset
                    updatedSidecar.sampling.temperature = temperature
                    updatedSidecar.sampling.topP = topP
                    updatedSidecar.sampling.thinkingBudget = thinkingBudget
                    updatedSidecar.tags = parsedTags

                    Task {
                        await viewModel.saveModelSidecar(updatedSidecar, for: model)
                        dismiss()
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 580)
        .onAppear {
            let loaded = viewModel.loadSidecarConfig(for: model)
            loadedBaseSidecar = loaded
            pinned = loaded.pinned ?? model.isPinned
            computePlacement = loaded.computePlacement ?? model.computePlacement.value
            ctxSize = loaded.ctxSize ?? model.contextSize.value
            flashAttn = loaded.flashAttn ?? model.flashAttention.value
            supportsVision = loaded.supportsVision ?? model.supportsVision.value
            supportsThinking = loaded.supportsThinking ?? model.supportsThinking.value
            imageMaxTokensText = loaded.imageMaxTokens.map(String.init) ?? ""
            draftModelText = loaded.draftModel ?? (model.speculativeDrafter.value ?? "")
            temperature = loaded.sampling.temperature ?? 0.7
            topP = loaded.sampling.topP ?? 0.95
            thinkingBudget = loaded.sampling.thinkingBudget ?? (model.thinkingBudget.value ?? 4096)
            tagsText = (loaded.tags.isEmpty ? model.tags : loaded.tags).joined(separator: ", ")
            unsetText = loaded.unsetKeys.joined(separator: ", ")
            let tmplState = viewModel.loadChatTemplateState(for: model)
            templateSourceLabel = tmplState.source
            hasTemplateOverride = tmplState.hasOverride
            chatTemplateText = tmplState.template
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
                    Text(String(format: "%.1f GB", viewModel.memorySnapshot.totalPhysicalRAMGB))
                        .font(.subheadline.monospaced().weight(.semibold))
                }
                LabeledContent("Metal Recommended Max Working Set") {
                    Text(String(format: "%.1f GB", viewModel.memorySnapshot.metalRecommendedMaxWorkingSetGB))
                        .font(.subheadline.monospaced().weight(.semibold))
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
        }
        .formStyle(.grouped)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
