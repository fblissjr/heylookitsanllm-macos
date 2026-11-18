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
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var showScanFolderImporter: Bool = false

    private var modelsPerf: ModelsAndPerfController { viewModel.modelsPerf }

    var body: some View {
        let filteredModels = viewModel.filteredModels
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        let isCompact = horizontalSizeClass == .compact

        VStack(spacing: 0) {
            ModelsTopActionBar(
                viewModel: viewModel,
                modelCount: filteredModels.count,
                isCompact: isCompact,
                onAddScanFolder: { showScanFolderImporter = true }
            )

            if let errorBanner = modelsPerf.modelsErrorBanner, !errorBanner.isEmpty {
                Divider()
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(errorBanner)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                    Spacer()
                    Button {
                        modelsPerf.modelsErrorBanner = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
                .background(Color.red.opacity(contrastStyle.bannerFillOpacity))
            }

            if let listNote = modelsPerf.modelsStatusNote, !listNote.isEmpty {
                Divider()
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text(listNote)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        modelsPerf.modelsStatusNote = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 5)
                .background(Color.green.opacity(contrastStyle.bannerFillOpacity * 0.6))
            }

            Divider()

            WatchFoldersSectionView(
                viewModel: viewModel,
                contrastStyle: contrastStyle,
                onAddScanFolder: { showScanFolderImporter = true }
            )

            Divider()

            if filteredModels.isEmpty {
                ContentUnavailableView {
                    Label("No Models Registered", systemImage: "cpu")
                } description: {
                    Text("Configure watch folders above (heylook.toml [scan]) to discover MLX, CoreAI, or GGUF checkpoints, or connect to a Bonjour peer in Perf & Server.")
                } actions: {
                    GlassEffectContainer(spacing: 10) {
                        HStack(spacing: 10) {
                            Button("Re-scan Models") {
                                Task { await viewModel.rescanConfiguredModels() }
                            }
                            .buttonStyle(.glassProminent)
                            .controlSize(.small)

                            Button("Perf & Server") {
                                viewModel.activeSurface = .perf
                            }
                            .buttonStyle(.glass)
                            .controlSize(.small)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(filteredModels) { model in
                    ModelRegistryRowView(
                        model: model,
                        adminRow: modelsPerf.adminRowsByID[model.modelID],
                        publicRow: modelsPerf.publicRowsByID[model.modelID],
                        viewModel: viewModel,
                        contrastStyle: contrastStyle
                    )
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task {
            await modelsPerf.loadScanConfig()
        }
        .sheet(item: $viewModel.editingSidecarModel) { model in
            ModelConfigEditorSheet(viewModel: viewModel, model: model)
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
    let isCompact: Bool
    let onAddScanFolder: () -> Void

    private var isRemoteConnected: Bool {
        viewModel.modelsPerf.activeRemoteBackend != nil
    }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                Text("Registered Models (\(modelCount))")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)

                if let remote = viewModel.connectedRemoteServerURL {
                    Label("REMOTE: \(remote)", systemImage: "network")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.blue.opacity(0.15), in: Capsule())
                        .foregroundStyle(.blue)
                }

                if HeylookConfig.isModelConfigReadOnly && !isRemoteConnected {
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

                if !isCompact {
                    Spacer(minLength: 16)
                }

                if !isRemoteConnected {
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
                }

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
            .frame(minWidth: isCompact ? nil : 0, maxWidth: isCompact ? nil : .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .background(.bar)
    }
}

private struct WatchFoldersSectionView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let contrastStyle: AdaptiveContrastStyle
    let onAddScanFolder: () -> Void
    @State private var isExpanded: Bool = false

    private var modelsPerf: ModelsAndPerfController { viewModel.modelsPerf }
    private var isLocalReadOnly: Bool {
        modelsPerf.activeRemoteBackend == nil && HeylookConfig.isModelConfigReadOnly
    }

    var body: some View {
        let activeFolders = modelsPerf.scanConfig?.folders ?? viewModel.config.scanFolders
        let activeInterval = modelsPerf.scanConfig?.scanIntervalSeconds ?? modelsPerf.scanIntervalDraft

        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        isExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption2.weight(.semibold))
                        Label("Watch folders (\(activeFolders.count))", systemImage: "folder")
                            .font(.caption2.weight(.semibold))
                    }
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                if activeInterval == 0 {
                    Text("manual only (interval 0s)")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.orange)
                } else {
                    Text("scan every \(activeInterval)s")
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        if activeFolders.isEmpty {
                            Text("None configured in heylook.toml")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        } else {
                            ForEach(activeFolders, id: \.self) { folder in
                                HStack(spacing: 4) {
                                    Text(folder)
                                        .font(.system(size: 10.5, design: .monospaced))
                                        .lineLimit(1)
                                    if !isLocalReadOnly && modelsPerf.activeRemoteBackend == nil {
                                        Button {
                                            Task { await viewModel.removeModelScanFolder(folder) }
                                        } label: {
                                            Image(systemName: "xmark.circle.fill")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                                .frame(minWidth: 22, minHeight: 22)
                                                .contentShape(Rectangle())
                                        }
                                        .buttonStyle(.plain)
                                        .help("Remove watch folder from heylook.toml")
                                    }
                                }
                                .padding(.leading, 8)
                                .padding(.trailing, (!isLocalReadOnly && modelsPerf.activeRemoteBackend == nil) ? 3 : 8)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.secondary.opacity(contrastStyle.chipFillOpacity)))
                            }
                        }
                    }
                }
            }

            if let note = modelsPerf.scanConfigNote, !note.isEmpty {
                Text(note)
                    .font(.caption2.monospaced())
                    .foregroundStyle(activeInterval == 0 ? .orange : .secondary)
            }

            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Text("One folder per line (read on the server). Every model under a watch folder is served; per-model overrides live in each folder's model.heylook.toml.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    TextEditor(text: Bindable(modelsPerf).scanFoldersDraftText)
                        .font(.system(.caption, design: .monospaced))
                        .frame(minHeight: 56, maxHeight: 90)
                        .disabled(isLocalReadOnly)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(contrastStyle.borderColor(.secondary), lineWidth: contrastStyle.borderWidth)
                        )

                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 10) {
                            scanIntervalControls
                            Spacer()
                            scanActionButtons
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            scanIntervalControls
                            scanActionButtons
                        }
                    }

                    if modelsPerf.scanIntervalDraft == 0 {
                        Label("Discovery is off (scan_interval_seconds = 0) — models update only on manual Re-scan.", systemImage: "info.circle")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.03))
    }

    private var scanIntervalControls: some View {
        HStack(spacing: 8) {
            Text("scan_interval_seconds:")
                .font(.caption2.monospaced().weight(.semibold))
                .foregroundStyle(.secondary)

            TextField(
                "900",
                value: Bindable(modelsPerf).scanIntervalDraft,
                format: .number
            )
            .textFieldStyle(.roundedBorder)
            .font(.caption.monospaced())
            .frame(width: 76)
            .disabled(isLocalReadOnly)

            Menu("Presets") {
                Button("0 (Manual only)") { modelsPerf.scanIntervalDraft = 0 }
                Button("60s") { modelsPerf.scanIntervalDraft = 60 }
                Button("300s (5m)") { modelsPerf.scanIntervalDraft = 300 }
                Button("900s (15m)") { modelsPerf.scanIntervalDraft = 900 }
            }
            .controlSize(.mini)
            .disabled(isLocalReadOnly)
        }
    }

    private var scanActionButtons: some View {
        HStack(spacing: 8) {
            if modelsPerf.activeRemoteBackend == nil {
                Button(action: onAddScanFolder) {
                    Label("Browse Folder…", systemImage: "folder.badge.plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(isLocalReadOnly)
            }

            Button {
                let folders = modelsPerf.scanFoldersDraftText
                    .components(separatedBy: .newlines)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                let interval = max(0, modelsPerf.scanIntervalDraft)
                Task {
                    await modelsPerf.saveScanConfig(folders: folders, scanIntervalSeconds: interval)
                }
            } label: {
                if modelsPerf.isSavingScanConfig {
                    ProgressView().controlSize(.mini)
                } else {
                    Text("Save Watch Folders")
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(modelsPerf.isSavingScanConfig || isLocalReadOnly)
        }
    }
}

private struct ModelRegistryRowView: View {
    let model: EngineDescription
    let adminRow: AdminModelWireRow?
    let publicRow: PublicModelWireRow?
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let contrastStyle: AdaptiveContrastStyle

    private static let quickContextPresets: [Int] = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    private var modelsPerf: ModelsAndPerfController { viewModel.modelsPerf }

    private var effectiveProvider: EngineProviderKind {
        adminRow?.provider ?? publicRow?.provider ?? model.provider
    }

    private var effectiveOrigin: AdminModelWireRow.ConfigOrigin {
        if let adminRow { return adminRow.source }
        return viewModel.hasSidecarFileOnDisk(for: model) ? .config : .discovered
    }

    private var effectiveContract: EngineContract? {
        adminRow?.engine ?? publicRow?.engine
    }

    private var staleFields: [String] {
        guard model.isLoaded else { return [] }
        return adminRow?.staleReloadFields ?? []
    }

    var body: some View {
        let isLoading = viewModel.loadingModelIDs.contains(model.modelID)
        let isSelected = viewModel.selectedModelID == model.modelID
        let isPureEmbedding = model.supportsEmbeddings.value
        let fit = viewModel.assessModelRAMFit(model)
        let fitGateReason = !model.isLoaded ? modelsPerf.fitGateReasonsByModelID[model.modelID] : nil
        let fitBadge: (label: String, color: Color) = {
            if fitGateReason != nil || !fit.fitsInMemory {
                return ("EXCEEDS RAM", .red)
            } else if fit.remainingHeadroomGB < 2.0 {
                return (String(format: "TIGHT (+%.1fGB)", fit.remainingHeadroomGB), .orange)
            } else {
                return (String(format: "FITS (+%.1fGB)", fit.remainingHeadroomGB), .green)
            }
        }()

        VStack(alignment: .leading, spacing: 6) {
            // Row 1: Identity & Badges + Actions
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    rowIdentityBadges(fitBadge: fitBadge, fit: fit, isLoading: isLoading)
                    Spacer(minLength: 8)
                    rowActionButtons(
                        isLoading: isLoading,
                        isSelected: isSelected,
                        isPureEmbedding: isPureEmbedding,
                        fitGateReason: fitGateReason
                    )
                }

                VStack(alignment: .leading, spacing: 6) {
                    rowIdentityBadges(fitBadge: fitBadge, fit: fit, isLoading: isLoading)
                    ScrollView(.horizontal, showsIndicators: false) {
                        rowActionButtons(
                            isLoading: isLoading,
                            isSelected: isSelected,
                            isPureEmbedding: isPureEmbedding,
                            fitGateReason: fitGateReason
                        )
                    }
                }
            }

            // Row 2: Truthful Meta Line (`models.js` modelMetaLine parity)
            Text(metaLineText)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            if let desc = adminRow?.description, !desc.isEmpty {
                Text(desc)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Row 3: Non-default scalar config summary chip (`model-config.js` configSummary)
            if let summary = configSummaryText {
                HStack(spacing: 6) {
                    Image(systemName: "slider.horizontal.2.square")
                        .font(.caption2)
                        .foregroundStyle(.blue)
                    Text(summary)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.primary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Color.blue.opacity(contrastStyle.chipFillOpacity * 0.7), in: RoundedRectangle(cornerRadius: 6))
            }

            // Row 4: `stale_reload_fields` warning + immediate `Reload now` (`warm: true`)
            if !staleFields.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text("config changed — reload to apply (\(staleFields.joined(separator: ", ")))")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                    Spacer()
                    Button {
                        Task { await modelsPerf.reloadModel(model, warm: true) }
                    } label: {
                        if isLoading {
                            ProgressView().controlSize(.mini)
                        } else {
                            Label("Reload now", systemImage: "arrow.triangle.2.circlepath")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .controlSize(.mini)
                    .disabled(isLoading)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.orange.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(contrastStyle.borderColor(.orange), lineWidth: contrastStyle.borderWidth)
                )
            }

            if let warmOutcome = modelsPerf.loadWarmOutcomesByModelID[model.modelID] {
                Text(warmOutcome)
                    .font(.caption2.monospaced())
                    .foregroundStyle(warmOutcome.localizedCaseInsensitiveContains("fail") ? .red : .green)
            }

            if !model.tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(model.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        }
                    }
                }
            }

            // Row 5: Full EngineContract + Sampler Defaults (`model`/`vendor`/`default`) Disclosure
            EngineContractDisclosureView(
                model: model,
                adminRow: adminRow,
                publicRow: publicRow,
                contract: effectiveContract,
                schemaFields: modelsPerf.cachedProviderSchemas[effectiveProvider] ?? ProviderConfigSchema.schema(for: effectiveProvider).fields,
                contrastStyle: contrastStyle
            )
        }
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func rowIdentityBadges(
        fitBadge: (label: String, color: Color),
        fit: GGUFSpawnConfig.RAMFitAssessment,
        isLoading: Bool
    ) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                if isLoading {
                    ProgressView().controlSize(.mini)
                } else {
                    Circle()
                        .fill(model.isLoaded ? Color.green : Color.secondary.opacity(0.35))
                        .frame(width: 8, height: 8)
                }

                Text(model.modelID)
                    .font(.headline)

                Text(model.isLoaded ? "Loaded" : "Idle")
                    .font(.caption2.monospaced().weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        (model.isLoaded ? Color.green : Color.secondary).opacity(contrastStyle.chipFillOpacity),
                        in: Capsule()
                    )
                    .foregroundStyle(model.isLoaded ? .green : .secondary)

                Text(effectiveProvider.displayName)
                    .font(.caption2.monospaced().weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())

                if effectiveOrigin == .discovered {
                    Text("no entry")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.secondary)
                        .help("Discovered in a watch folder with all settings derived; no model.heylook.toml or [[models]] entry yet.")
                } else {
                    Text("configured")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.purple.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.purple)
                        .help("Has explicit configuration in model.heylook.toml or heylook.toml.")
                }

                Text(model.computePlacement.value.label)
                    .font(.caption2.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.teal.opacity(contrastStyle.chipFillOpacity), in: Capsule())

                Text(fitBadge.label)
                    .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(fitBadge.color.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                    .foregroundStyle(fitBadge.color)
                    .help(String(format: "Weights %.1f GB + KV %.2f GB vs %.1f GB working set (recommended ctx %dk)", fit.modelWeightGB + fit.mmprojWeightGB + fit.drafterWeightGB, fit.estimatedKVGB, fit.availableWorkingSetGB, fit.recommendedCtxSize / 1024))

                if fit.droppedDrafterForMemory {
                    Text("MTP DROPPED")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.orange)
                }

                if model.isPinned {
                    Text("PINNED")
                        .font(.caption2.monospaced().weight(.bold))
                        .foregroundStyle(.orange)
                }

                if modelsPerf.hasUnsavedDraft(for: model.modelID) {
                    Text("UNSAVED DRAFT")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.yellow.opacity(0.22), in: Capsule())
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    @ViewBuilder
    private func rowActionButtons(
        isLoading: Bool,
        isSelected: Bool,
        isPureEmbedding: Bool,
        fitGateReason: String?
    ) -> some View {
        HStack(spacing: 6) {
            // Load/Unload remains FIRST action (`models.js` invariant)
            Button {
                Task { await viewModel.toggleModelLoaded(model) }
            } label: {
                if isLoading {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text(model.isLoaded ? "Unloading…" : "Loading…")
                    }
                } else {
                    Text(model.isLoaded ? "Unload" : "Load")
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isLoading || (!model.isLoaded && fitGateReason != nil))
            .help(fitGateReason ?? (model.isLoaded ? "Unload model and release memory" : "Load & warm 1-token JIT"))

            Button {
                viewModel.editingSidecarModel = model
            } label: {
                Label("Configure…", systemImage: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

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
                    Text("ctx \(formatTokenCount(model.contextSize.value))")
                        .font(.caption2.monospaced())
                }
                #if os(macOS)
                .menuStyle(.borderedButton)
                #else
                .menuStyle(.button)
                #endif
                .controlSize(.small)
                .fixedSize()
                .disabled(modelsPerf.activeRemoteBackend == nil && HeylookConfig.isModelConfigReadOnly)
                .help("Quick-set context window in model.heylook.toml")
            }

            Button(model.isPinned ? "Unpin" : "Pin") {
                Task { await viewModel.toggleModelPin(model) }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            if effectiveProvider == .mlx && modelsPerf.activeRemoteBackend == nil {
                Button {
                    viewModel.convertingSourceModel = model
                    viewModel.showMLXConverterSheet = true
                } label: {
                    Label("Quantize…", systemImage: "cube.transparent")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            if isPureEmbedding {
                Text("Embeddings")
                    .font(.caption2.monospaced().weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.purple.opacity(contrastStyle.chipFillOpacity), in: Capsule())
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
    }

    private var metaLineText: String {
        var parts: [String] = [effectiveProvider.rawValue]
        let caps = adminRow?.capabilities ?? publicRow?.capabilities ?? []
        if !caps.isEmpty {
            parts.append(caps.joined(separator: ", "))
        }
        let ceiling = effectiveContract?.context.length.value ?? model.contextSize.value
        if ceiling > 0 {
            if let running = effectiveContract?.context.running.value, running > 0, running != ceiling {
                parts.append("ctx \(formatTokenCount(running)) / \(formatTokenCount(ceiling))")
            } else {
                parts.append("ctx \(formatTokenCount(ceiling))")
            }
        }
        parts.append("arch \(model.architecture.value)")
        parts.append("quant \(model.quantization.value)")
        if effectiveOrigin == .discovered {
            parts.append("no entry")
        } else {
            parts.append("model.heylook.toml")
        }
        return parts.joined(separator: " · ")
    }

    private var configSummaryText: String? {
        guard let config = adminRow?.config, !config.isEmpty else { return nil }
        let fields = modelsPerf.cachedProviderSchemas[effectiveProvider]
            ?? ProviderConfigSchema.schema(for: effectiveProvider).fields
        var parts: [String] = []
        for field in fields {
            if field.effect == .descriptive || field.ui == .hidden { continue }
            guard let val = config[field.name], val != .null else { continue }
            switch val {
            case .array, .object:
                continue
            case .string(let s) where s.count > 16:
                continue
            case .bool(let b):
                parts.append("\(field.name) \(b ? "on" : "off")")
            default:
                parts.append("\(field.name) \(formatJSONScalar(val))")
            }
        }
        guard !parts.isEmpty else { return nil }
        var shown = Array(parts.prefix(4))
        if parts.count > 4 {
            shown.append("+\(parts.count - 4) more")
        }
        return shown.joined(separator: " · ")
    }
}

// MARK: - Engine Contract & Sampler Provenance Renderer (`engine.js` parity)

private struct EngineContractDisclosureView: View {
    let model: EngineDescription
    let adminRow: AdminModelWireRow?
    let publicRow: PublicModelWireRow?
    let contract: EngineContract?
    let schemaFields: [ConfigFieldDescriptor]
    let contrastStyle: AdaptiveContrastStyle

    @State private var isExpanded: Bool = false

    private var summaryLine: String {
        if let contract {
            var parts: [String] = []
            if let rt = contract.runtime.value {
                parts.append(rt)
            }
            let ceiling = contract.context.length.value
            let running = contract.context.running.value
            if let running {
                parts.append("ctx \(formatTokenCount(running)) of \(ceiling.map(formatTokenCount) ?? "?")")
            } else if let ceiling {
                parts.append("ctx \(formatTokenCount(ceiling))")
            }
            if let origin = contract.template.origin.value {
                parts.append("template \(origin)")
            }
            if let drafter = contract.speculative.drafter.value, !drafter.isEmpty {
                let inForceWord: String = switch contract.speculative.inForce.value {
                case true?: "on"
                case false?: "not in use"
                case nil: "ready"
                }
                parts.append("spec decode \(inForceWord)")
            }
            if let mode = contract.decoding.mode.value {
                parts.append(mode)
            }
            return parts.joined(separator: " · ")
        }
        return "Path: \(model.modelPath) • \(model.estimatedMemoryMB) MB est."
    }

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            if let contract {
                VStack(alignment: .leading, spacing: 10) {
                    // 1. Engine (runtime + context.length + context.running)
                    contractSection("Engine") {
                        EngineFactRow(
                            label: "runtime",
                            value: contract.runtime.value ?? "—",
                            provenance: contract.runtime.provenance,
                            reason: contract.runtime.source
                        )
                        EngineFactRow(
                            label: "context length",
                            value: contract.context.length.value.map { "\($0)" } ?? "—",
                            provenance: contract.context.length.provenance,
                            reason: contract.context.length.source
                        )
                        EngineFactRow(
                            label: "context running",
                            value: contract.context.running.value.map { "\($0)" } ?? "—",
                            provenance: contract.context.running.provenance,
                            reason: contract.context.running.source
                        )
                    }

                    // 2. Chat template
                    contractSection("Chat template") {
                        EngineFactRow(
                            label: "origin",
                            value: contract.template.origin.value ?? "—",
                            provenance: contract.template.origin.provenance,
                            reason: contract.template.origin.source
                        )
                        EngineFactRow(
                            label: "path",
                            value: contract.template.path.value ?? "—",
                            provenance: contract.template.path.provenance,
                            reason: contract.template.path.source
                        )
                        EngineFactRow(
                            label: "sha256",
                            value: contract.template.sha256.value.map { String($0.prefix(12)) } ?? "—",
                            provenance: contract.template.sha256.provenance,
                            reason: contract.template.sha256.source
                        )
                        EngineFactRow(
                            label: "running sha256",
                            value: contract.template.runningSha256.value.map { String($0.prefix(12)) } ?? "—",
                            provenance: contract.template.runningSha256.provenance,
                            reason: contract.template.runningSha256.source
                        )
                        if let onDisk = contract.template.sha256.value,
                           let running = contract.template.runningSha256.value {
                            let same = onDisk == running
                            EngineFactRow(
                                label: "running template",
                                value: same ? "matches the file" : "differs from the file",
                                provenance: .observed,
                                reason: same
                                    ? "the template this process loaded is the one on disk"
                                    : "the file changed after this model loaded; reload it to apply the file"
                            )
                        }
                        EngineFactRow(
                            label: "prefix stable",
                            value: formatBoolOpt(contract.template.prefixStable.value),
                            provenance: contract.template.prefixStable.provenance,
                            reason: contract.template.prefixStable.source
                        )
                        let capsStr: String = {
                            guard let caps = contract.template.caps.value else { return "—" }
                            let active = caps.filter(\.value).map {
                                $0.key.replacingOccurrences(of: "supports_", with: "").replacingOccurrences(of: "_", with: " ")
                            }.sorted()
                            return active.isEmpty ? "none" : active.joined(separator: ", ")
                        }()
                        EngineFactRow(
                            label: "caps",
                            value: capsStr,
                            provenance: contract.template.caps.provenance,
                            reason: contract.template.caps.source
                        )
                    }

                    // 3. Thinking
                    if let thinking = contract.thinking {
                        contractSection("Thinking") {
                            let from = thinking.template.map { "rendered from \($0)" } ?? "rendered from the in-force template"
                            EngineFactRow(
                                label: "switch",
                                value: thinking.switchVariable ?? "none",
                                provenance: .derived,
                                reason: thinking.switchVariable != nil
                                    ? "\(from): the template reads \(thinking.switchVariable!)"
                                    : "\(from): the template reads no thinking switch"
                            )
                            if let d = thinking.depth {
                                let valuesText = d.unknown == "verbatim"
                                    ? "any word\(d.values.isEmpty ? "" : " (names \(d.values.joined(separator: ", ")))")"
                                    : d.values.joined(separator: ", ")
                                let aliases = d.aliases.sorted { $0.key < $1.key }.map { "\($0.key) = \($0.value)" }.joined(separator: ", ")
                                let reasonParts: [String?] = [
                                    "\(from).",
                                    d.defaultValue.map { "Default \($0)." } ?? "No value is the default: unset sends nothing.",
                                    aliases.isEmpty ? nil : "Also accepted: \(aliases).",
                                    "Unknown value \(d.unknown ?? "ignored").",
                                    d.changesPrefix == true ? "Changing it re-processes the whole conversation." : nil
                                ]
                                EngineFactRow(
                                    label: "depth \(d.variable)",
                                    value: valuesText,
                                    provenance: .derived,
                                    reason: reasonParts.compactMap { $0 }.joined(separator: " ")
                                )
                            } else {
                                EngineFactRow(
                                    label: "depth",
                                    value: "none",
                                    provenance: .derived,
                                    reason: "\(from): no depth levels\(thinking.switchVariable != nil ? ", so thinking is on or off only" : "")."
                                )
                            }
                            if let budget = thinking.budget {
                                EngineFactRow(
                                    label: "budget",
                                    value: budget.enforced ? "enforced" : "not enforced",
                                    provenance: .derived,
                                    reason: budget.reason
                                )
                            }
                        }
                    }

                    // 4. Cache
                    if !contract.cache.isEmpty {
                        contractSection("Cache") {
                            ForEach(contract.cache.keys.sorted(), id: \.self) { key in
                                if let fact = contract.cache[key] {
                                    EngineFactRow(
                                        label: key.replacingOccurrences(of: "_", with: " "),
                                        value: fact.value.map(formatJSONScalar) ?? "—",
                                        provenance: fact.provenance,
                                        reason: fact.source
                                    )
                                }
                            }
                        }
                    }

                    // 5. Speculative (drafter, type, in_force)
                    contractSection("Speculative") {
                        EngineFactRow(
                            label: "drafter",
                            value: contract.speculative.drafter.value ?? "—",
                            provenance: contract.speculative.drafter.provenance,
                            reason: contract.speculative.drafter.source
                        )
                        EngineFactRow(
                            label: "type",
                            value: contract.speculative.type.value ?? "—",
                            provenance: contract.speculative.type.provenance,
                            reason: contract.speculative.type.source
                        )
                        EngineFactRow(
                            label: "in force",
                            value: formatBoolOpt(contract.speculative.inForce.value),
                            provenance: contract.speculative.inForce.provenance,
                            reason: contract.speculative.inForce.source
                        )
                    }

                    // 6. Decoding
                    contractSection("Decoding") {
                        EngineFactRow(
                            label: "mode",
                            value: contract.decoding.mode.value ?? "—",
                            provenance: contract.decoding.mode.provenance,
                            reason: contract.decoding.mode.source
                        )
                        EngineFactRow(
                            label: "request fields",
                            value: contract.decoding.requestFields.value?.joined(separator: ", ") ?? "—",
                            provenance: contract.decoding.requestFields.provenance,
                            reason: contract.decoding.requestFields.source
                        )
                    }

                    // 7. Embedding facts (when embedding model)
                    if let emb = contract.embedding {
                        contractSection("Embedding") {
                            EngineFactRow(
                                label: "modalities",
                                value: emb.modalities.joined(separator: ", "),
                                provenance: .derived,
                                reason: emb.provenance
                            )
                            EngineFactRow(
                                label: "dimensions",
                                value: emb.dimensions.map(String.init).joined(separator: ", "),
                                provenance: .derived,
                                reason: "native \(emb.nativeDimensions)d, normalization \(emb.normalization)"
                            )
                            EngineFactRow(
                                label: "max input tokens",
                                value: "\(emb.maxInputTokens)",
                                provenance: .derived,
                                reason: emb.computePlacement.map { "placement \($0)" }
                            )
                        }
                    }

                    // 8. Sampler Defaults & 3-Way `sampler_sources` (`model` / `vendor` / `default`)
                    if let defaults = adminRow?.samplerDefaults ?? publicRow?.samplerDefaults, !defaults.isEmpty {
                        let sources = adminRow?.samplerSources ?? publicRow?.samplerSources ?? [:]
                        let thinkingDefault = adminRow?.thinkingDefault ?? publicRow?.thinkingDefault ?? model.supportsThinking.value
                        contractSection("Sampler Defaults (model / vendor / default)") {
                            HStack(spacing: 8) {
                                Text("thinking_default")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 170, alignment: .leading)
                                Text(thinkingDefault ? "on" : "off")
                                    .font(.caption2.monospaced().weight(.semibold))
                                Spacer()
                                let thinkSrc = sources["enable_thinking"]
                                    ?? (model.supportsThinking.provenance == .configured ? "model" : "default")
                                SamplerSourceOriginBadge(origin: thinkSrc)
                            }
                            ForEach(SamplerCascade.SamplerFieldKey.allCases, id: \.rawValue) { key in
                                let wireKey = key.canonicalWireKey
                                if wireKey != "enable_thinking", let val = defaults[wireKey] {
                                    let origin = sources[wireKey] ?? "default"
                                    HStack(spacing: 8) {
                                        Text(wireKey)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                            .frame(width: 170, alignment: .leading)
                                        Text(formatJSONScalar(val))
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.primary)
                                            .lineLimit(1)
                                        Spacer()
                                        SamplerSourceOriginBadge(origin: origin)
                                    }
                                }
                            }
                        }
                    }

                    // 9. Settings grouped by Effect Class (`engine.js` EFFECT_GROUPS)
                    if !contract.settings.isEmpty {
                        settingsSections(contract.settings)
                    }
                }
                .padding(.top, 6)
            } else {
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
                        label: "Context Window",
                        value: "\(model.contextSize.value) tokens",
                        source: model.contextSize.provenance,
                        effect: model.contextSize.effect
                    )
                    ProvenanceDetailRow(
                        label: "Chat Template Source",
                        value: model.templateSource.value,
                        source: model.templateSource.provenance,
                        effect: model.templateSource.effect
                    )
                }
                .padding(.top, 6)
            }
        } label: {
            HStack(spacing: 6) {
                Text("Engine")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                Text(summaryLine)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func contractSection<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 3) {
                content()
            }
            .padding(8)
            .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(contrastStyle.subtleBorderColor(.secondary), lineWidth: contrastStyle.borderWidth)
            )
        }
    }

    @ViewBuilder
    private func settingsSections(_ settings: [String: SettingFact]) -> some View {
        let hiddenNames = Set(schemaFields.filter { $0.ui == .hidden }.map(\.name))
        let visibleEntries = settings.filter { name, fact in
            !(hiddenNames.contains(name) && fact.provenance != .configured)
        }
        let effectGroups: [(effect: ConfigFieldEffect?, title: String)] = [
            (nil, "Load decisions"),
            (.requiresReload, "Applies at the next load"),
            (.loadTimeOnly, "Fixed at load"),
            (.appliesLive, "Applies live"),
            (.perRequest, "Per-request defaults"),
            (.descriptive, "Descriptive")
        ]
        ForEach(effectGroups, id: \.title) { group in
            let matches = visibleEntries
                .filter { $0.value.effect == group.effect }
                .sorted { $0.key < $1.key }
            if !matches.isEmpty {
                contractSection(group.title) {
                    ForEach(matches, id: \.key) { name, fact in
                        let autoNote: String? = fact.provenance == .configured
                            ? "auto would be \(fact.auto.map(formatJSONScalar) ?? "—")"
                            : nil
                        EngineFactRow(
                            label: name,
                            value: fact.value.map(formatJSONScalar) ?? "—",
                            provenance: fact.provenance,
                            reason: fact.reason,
                            extraNote: autoNote
                        )
                    }
                }
            }
        }
    }

    private func formatBoolOpt(_ v: Bool?) -> String {
        switch v {
        case true?: return "on"
        case false?: return "off"
        case nil: return "—"
        }
    }
}

private struct EngineFactRow: View {
    let label: String
    let value: String
    let provenance: ProvenanceSource
    let reason: String?
    var extraNote: String? = nil

    @State private var showReason: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Button {
                showReason.toggle()
            } label: {
                HStack(spacing: 8) {
                    Text(label)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 150, alignment: .leading)
                    Text(value)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Spacer()
                    EngineProvenanceLabelPill(provenance: provenance)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if showReason {
                VStack(alignment: .leading, spacing: 1) {
                    if let reason, !reason.isEmpty {
                        Text(reason)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    if let extraNote {
                        Text(extraNote)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding(.leading, 8)
            }
        }
    }
}

private struct EngineProvenanceLabelPill: View {
    let provenance: ProvenanceSource

    private var label: String {
        switch provenance {
        case .derived: return "derived"
        case .configured: return "set"
        case .observed: return "live"
        case .observedCached: return "cached"
        case .unknown: return "unknown"
        case .notApplicable: return "n/a"
        }
    }

    private var color: Color {
        switch provenance {
        case .derived: return .secondary
        case .configured: return .blue
        case .observed: return .green
        case .observedCached: return .mint
        case .unknown: return .orange
        case .notApplicable: return .secondary
        }
    }

    var body: some View {
        Text(label)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(Capsule().fill(color.opacity(0.16)))
            .foregroundStyle(color)
    }
}

/// Renders the exact 3-way `sampler_sources` (`model` / `vendor` / `default`) without ever
/// collapsing `vendor` and `default` together.
private struct SamplerSourceOriginBadge: View {
    let origin: String

    private var style: (label: String, color: Color) {
        switch origin.lowercased() {
        case "model":
            return ("model", .blue)
        case "vendor":
            return ("vendor", .teal)
        default:
            return ("default", .secondary)
        }
    }

    var body: some View {
        Text(style.label)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .padding(.horizontal, 6)
            .padding(.vertical, 1.5)
            .background(Capsule().fill(style.color.opacity(0.16)))
            .foregroundStyle(style.color)
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
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

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
        let isCompact = horizontalSizeClass == .compact
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

            ScrollView(.horizontal, showsIndicators: false) {
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
        .frame(minWidth: isCompact ? nil : 580)
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

// MARK: - 100% Schema-Driven Config, Live RAM Fit & Multi-Rung Chat Template Sheet

private struct ModelConfigEditorSheet: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let model: EngineDescription

    @Environment(\.dismiss) private var dismiss
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    @State private var schemaFields: [ConfigFieldDescriptor] = []
    @State private var schemaLoadError: String?
    @State private var isLoadingSchema: Bool = true
    @State private var savedBaseline: [String: JSONValue] = [:]

    @State private var isSavingConfig: Bool = false
    @State private var configStatusNote: String?
    @State private var configErrorNote: String?
    @State private var showReloadOfferAfterSave: Bool = false

    @State private var showRemoveAllOverridesConfirm: Bool = false
    @State private var showDiscardUnsavedConfirm: Bool = false

    // Live fit meter state
    @State private var fitReport: RAMFitEvaluator.FitReport?
    @State private var fitErrorMessage: String?
    @State private var residentReading: (memoryMB: Double?, memorySource: String?) = (nil, nil)
    @State private var fitDebounceTask: Task<Void, Never>?

    // Chat template baseline + unsaved state
    @State private var templateBaseline: String = ""
    @State private var templateLoaded: Bool = false

    private var modelsPerf: ModelsAndPerfController { viewModel.modelsPerf }

    private var liveModel: EngineDescription {
        viewModel.registeredModels.first(where: { $0.modelID == model.modelID }) ?? model
    }

    private var adminRow: AdminModelWireRow? {
        modelsPerf.adminRowsByID[model.modelID]
    }

    private var effectiveProvider: EngineProviderKind {
        adminRow?.provider ?? liveModel.provider
    }

    private var isRemoteModel: Bool {
        modelsPerf.activeRemoteBackend != nil || liveModel.provider == .remote
    }

    private var isLocalReadOnly: Bool {
        !isRemoteModel && HeylookConfig.isModelConfigReadOnly
    }

    private var draftMap: [String: String] {
        modelsPerf.configDraftsByModelID[model.modelID] ?? [:]
    }

    private var visibleFields: [ConfigFieldDescriptor] {
        schemaFields.filter { $0.ui != .hidden }
    }

    private var dirtyFields: [ConfigFieldDescriptor] {
        visibleFields.filter { field in
            guard let raw = draftMap[field.name] else { return false }
            return raw != savedRawValue(for: field)
        }
    }

    private var hasUnsavedTemplateDraft: Bool {
        guard templateLoaded, let draft = modelsPerf.templateDraftsByModelID[model.modelID] else {
            return false
        }
        let normalizedDraft = normalizeEOL(draft)
        return normalizedDraft != normalizeEOL(templateBaseline)
            && !normalizedDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var hasUnsavedChanges: Bool {
        hasUnsavedTemplateDraft || !dirtyFields.isEmpty
    }

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        let isCompact = horizontalSizeClass == .compact
        let origin = adminRow?.source ?? (viewModel.hasSidecarFileOnDisk(for: liveModel) ? .config : .discovered)
        let staleSet = Set(liveModel.isLoaded ? (adminRow?.staleReloadFields ?? []) : [])

        VStack(alignment: .leading, spacing: 12) {
            sheetHeader(origin: origin, contrastStyle: contrastStyle)

            Divider()

            if origin == .discovered {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle.fill")
                        .foregroundStyle(.blue)
                    Text("Found by a watch folder, with every setting detected at load. Saving a setting here writes it to model.heylook.toml in the model's folder; everything else keeps being detected.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.blue.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 8))
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    RAMFitMeterView(
                        model: liveModel,
                        fitReport: fitReport,
                        fitErrorMessage: fitErrorMessage,
                        residentMemoryMB: residentReading.memoryMB,
                        residentMemorySource: residentReading.memorySource,
                        contrastStyle: contrastStyle
                    )

                    SchemaConfigFormView(
                        model: liveModel,
                        adminRow: adminRow,
                        visibleFields: visibleFields,
                        isLoadingSchema: isLoadingSchema,
                        schemaLoadError: schemaLoadError,
                        isRemote: isRemoteModel,
                        isLocalReadOnly: isLocalReadOnly,
                        staleFields: staleSet,
                        contrastStyle: contrastStyle,
                        rawValueProvider: { field in currentRawValue(for: field) },
                        isDirtyProvider: { field in
                            guard let raw = draftMap[field.name] else { return false }
                            return raw != savedRawValue(for: field)
                        },
                        onEditField: { field, nextRaw in
                            var updated = modelsPerf.configDraftsByModelID[model.modelID] ?? [:]
                            if nextRaw == savedRawValue(for: field) {
                                updated.removeValue(forKey: field.name)
                            } else {
                                updated[field.name] = nextRaw
                            }
                            if updated.isEmpty {
                                modelsPerf.configDraftsByModelID.removeValue(forKey: model.modelID)
                            } else {
                                modelsPerf.configDraftsByModelID[model.modelID] = updated
                            }
                            scheduleFitRefresh()
                        }
                    )

                    if let note = configStatusNote ?? pendingPreSaveNote {
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(staleSet.isEmpty ? Color.secondary : Color.orange)
                    }

                    if let configErrorNote {
                        Text(configErrorNote)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    configActionBar(staleSet: staleSet)

                    ChatTemplateInspectorView(
                        viewModel: viewModel,
                        model: liveModel,
                        contrastStyle: contrastStyle,
                        onLoadedBaselineChanged: { baseline, loaded in
                            templateBaseline = baseline
                            templateLoaded = loaded
                        }
                    )
                }
            }

            Divider()

            HStack {
                Button("Close") {
                    attemptDismiss()
                }
                .buttonStyle(.bordered)

                if hasRemovableOverrides(origin: origin) && !isLocalReadOnly {
                    Button("Remove All Overrides…", role: .destructive) {
                        showRemoveAllOverridesConfirm = true
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()

                if hasUnsavedChanges {
                    Label("Unsaved changes", systemImage: "pencil.circle.fill")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.orange)
                }
            }
        }
        .padding(isCompact ? 14 : 20)
        .frame(
            minWidth: isCompact ? nil : 700,
            idealWidth: isCompact ? nil : 780,
            minHeight: isCompact ? nil : 620
        )
        .interactiveDismissDisabled(hasUnsavedChanges)
        .onKeyPress(.escape) {
            attemptDismiss()
            return .handled
        }
        .confirmationDialog(
            "Discard Unsaved Changes?",
            isPresented: $showDiscardUnsavedConfirm,
            titleVisibility: .visible
        ) {
            Button("Discard Changes", role: .destructive) {
                modelsPerf.clearDrafts(for: model.modelID)
                dismiss()
            }
            Button("Keep Editing", role: .cancel) {}
        } message: {
            Text("You have unsaved edits in this model's configuration or chat template override.")
        }
        .confirmationDialog(
            "Remove All Model Overrides?",
            isPresented: $showRemoveAllOverridesConfirm,
            titleVisibility: .visible
        ) {
            Button("Remove All Overrides", role: .destructive) {
                Task { await removeAllOverrides() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This clears all saved settings in model.heylook.toml for \(model.modelID) and reverts to derived defaults.")
        }
        .task(id: model.modelID) {
            await loadInitialData()
        }
    }

    @ViewBuilder
    private func sheetHeader(origin: AdminModelWireRow.ConfigOrigin, contrastStyle: AdaptiveContrastStyle) -> some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(liveModel.modelID)
                        .font(.headline)
                    Text(effectiveProvider.displayName)
                        .font(.caption2.monospaced().weight(.bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                    Text(origin == .discovered ? "no entry" : "model.heylook.toml")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            (origin == .discovered ? Color.secondary : Color.purple).opacity(contrastStyle.chipFillOpacity),
                            in: Capsule()
                        )
                        .foregroundStyle(origin == .discovered ? Color.secondary : Color.purple)
                    Text(liveModel.isLoaded ? "Loaded" : "Idle")
                        .font(.caption2.monospaced().weight(.bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            (liveModel.isLoaded ? Color.green : Color.secondary).opacity(contrastStyle.chipFillOpacity),
                            in: Capsule()
                        )
                        .foregroundStyle(liveModel.isLoaded ? .green : .secondary)
                }
                Text("Schema-driven model configuration — blank or 'default' unsets keys (sending null) so untouched derived/vendor values are never frozen.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    @ViewBuilder
    private func configActionBar(staleSet: Set<String>) -> some View {
        let dirty = dirtyFields
        let shouldShowReload = liveModel.isLoaded && (!staleSet.isEmpty || showReloadOfferAfterSave)

        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                actionButtons(dirty: dirty, shouldShowReload: shouldShowReload)
                Spacer()
            }
            VStack(alignment: .leading, spacing: 8) {
                actionButtons(dirty: dirty, shouldShowReload: shouldShowReload)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func actionButtons(dirty: [ConfigFieldDescriptor], shouldShowReload: Bool) -> some View {
        HStack(spacing: 8) {
            Button {
                Task { await saveSchemaConfig() }
            } label: {
                if isSavingConfig {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text("Saving…")
                    }
                } else {
                    Text("Save Config")
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isSavingConfig || dirty.isEmpty || isLocalReadOnly)

            Button("Reset") {
                modelsPerf.configDraftsByModelID.removeValue(forKey: model.modelID)
                configErrorNote = nil
                configStatusNote = nil
                scheduleFitRefresh()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isSavingConfig || dirty.isEmpty)

            if shouldShowReload {
                Button {
                    Task {
                        await modelsPerf.reloadModel(liveModel, warm: true)
                        showReloadOfferAfterSave = false
                        await refreshFitAndResidentNow()
                        if let outcome = modelsPerf.loadWarmOutcomesByModelID[model.modelID] {
                            configStatusNote = outcome
                        }
                    }
                } label: {
                    Label("Reload now", systemImage: "arrow.triangle.2.circlepath")
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .controlSize(.small)
                .disabled(viewModel.loadingModelIDs.contains(model.modelID))
            }
        }
    }

    private var pendingPreSaveNote: String? {
        let dirty = dirtyFields
        guard !dirty.isEmpty else {
            let stale = liveModel.isLoaded ? (adminRow?.staleReloadFields ?? []) : []
            return stale.isEmpty ? nil : "Saved changes pending reload: \(stale.joined(separator: ", "))."
        }
        let reloadNames = dirty.filter { $0.effect == .requiresReload }.map(\.name)
        let liveNames = dirty.filter { $0.effect != .requiresReload }.map(\.name)
        var parts: [String] = []
        if !reloadNames.isEmpty {
            let suffix = liveModel.isLoaded
                ? "needs a reload after save (the loaded model keeps running as-is)"
                : "applies on next load"
            parts.append("\(reloadNames.joined(separator: ", ")) — \(suffix)")
        }
        if !liveNames.isEmpty {
            parts.append("\(liveNames.joined(separator: ", ")) — applies on save")
        }
        return "Changed: \(parts.joined(separator: "; "))."
    }

    private func hasRemovableOverrides(origin: AdminModelWireRow.ConfigOrigin) -> Bool {
        if !isRemoteModel {
            return viewModel.hasSidecarFileOnDisk(for: liveModel)
        }
        return origin == .config && !(adminRow?.config.isEmpty ?? true)
    }

    private func attemptDismiss() {
        if hasUnsavedChanges {
            showDiscardUnsavedConfirm = true
        } else {
            dismiss()
        }
    }

    private func loadInitialData() async {
        isLoadingSchema = true
        schemaLoadError = nil

        // Sync baseline from adminRow or sidecar
        if let row = adminRow {
            savedBaseline = row.config
        } else if !isRemoteModel {
            savedBaseline = viewModel.loadSidecarConfig(for: liveModel).toExplicitJSONDictionary()
        } else {
            savedBaseline = [:]
        }

        do {
            schemaFields = try await modelsPerf.schemaFields(for: effectiveProvider)
        } catch {
            schemaLoadError = "Could not load option schema: \(error.localizedDescription)"
        }
        isLoadingSchema = false

        await refreshFitAndResidentNow()
    }

    private func savedRawValue(for field: ConfigFieldDescriptor) -> String {
        guard let val = savedBaseline[field.name], val != .null else { return "" }
        if field.type == .string, field.enumValues != nil, case .bool(let b) = val {
            // Normalize legacy boolean flash_attn values onto GGUF enum choices ("on" / "off")
            if field.name == "flash_attn" {
                return b ? "on" : "off"
            }
            return b ? "true" : "false"
        }
        return fieldControlString(for: field, value: val)
    }

    private func currentRawValue(for field: ConfigFieldDescriptor) -> String {
        if let drafted = draftMap[field.name] {
            return drafted
        }
        return savedRawValue(for: field)
    }

    private func computeFitOverrides() -> [String: JSONValue] {
        var overrides: [String: JSONValue] = [:]
        for field in dirtyFields {
            let raw = draftMap[field.name] ?? ""
            if case .success(let val) = parseControlValue(field: field, raw: raw) {
                overrides[field.name] = val
            }
        }
        return overrides
    }

    private func scheduleFitRefresh() {
        fitDebounceTask?.cancel()
        fitDebounceTask = Task {
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            await refreshFitAndResidentNow()
        }
    }

    private func refreshFitAndResidentNow() async {
        let overrides = computeFitOverrides()
        do {
            let report = try await modelsPerf.assessModelFitReport(
                id: liveModel.modelID,
                configOverrides: overrides,
                headroomGB: 8.0
            )
            fitReport = report
            fitErrorMessage = nil
            let gateReason = (report.verdict == "fail" && !liveModel.isLoaded)
                ? "Does not fit memory (see Configure)"
                : nil
            modelsPerf.updateFitGate(modelID: liveModel.modelID, reason: gateReason)
        } catch {
            fitReport = nil
            fitErrorMessage = error.localizedDescription
            modelsPerf.updateFitGate(modelID: liveModel.modelID, reason: nil)
        }

        if liveModel.isLoaded {
            residentReading = await modelsPerf.fetchModelStatusReading(id: liveModel.modelID)
        } else {
            residentReading = (nil, nil)
        }
    }

    private func saveSchemaConfig() async {
        let dirty = dirtyFields
        guard !isSavingConfig, !dirty.isEmpty else { return }

        configErrorNote = nil
        var patch: [String: JSONValue] = [:]
        for field in dirty {
            if field.isFileOnly && isRemoteModel {
                configErrorNote = "Not saved — '\(field.name)' is file-only and must be edited directly in model.heylook.toml or heylook.toml on the host."
                return
            }
            let raw = draftMap[field.name] ?? ""
            switch parseControlValue(field: field, raw: raw) {
            case .success(let value):
                patch[field.name] = value
            case .failure(let err):
                configErrorNote = "Not saved — \(err.message)"
                return
            }
        }

        isSavingConfig = true
        defer { isSavingConfig = false }

        do {
            let result = try await modelsPerf.patchModelConfig(id: liveModel.modelID, patch: patch)
            savedBaseline = result.model.config
            var nextDraft = modelsPerf.configDraftsByModelID[model.modelID] ?? [:]
            for field in dirty {
                nextDraft.removeValue(forKey: field.name)
            }
            if nextDraft.isEmpty {
                modelsPerf.configDraftsByModelID.removeValue(forKey: model.modelID)
            } else {
                modelsPerf.configDraftsByModelID[model.modelID] = nextDraft
            }

            var notes: [String] = ["Saved."]
            if !result.reloadRequiredFields.isEmpty {
                let joined = result.reloadRequiredFields.joined(separator: ", ")
                if liveModel.isLoaded {
                    notes.append("Takes effect on reload: \(joined).")
                    showReloadOfferAfterSave = true
                } else {
                    notes.append("Applies on next load: \(joined).")
                }
            } else {
                showReloadOfferAfterSave = false
            }
            if let warn = result.warning, !warn.isEmpty {
                notes.append(warn)
            }
            configStatusNote = notes.joined(separator: " ")
            await refreshFitAndResidentNow()
        } catch {
            configErrorNote = "Save failed: \(error.localizedDescription)"
        }
    }

    private func removeAllOverrides() async {
        if !isRemoteModel {
            await viewModel.saveModelSidecar(ModelSidecarConfig(), for: liveModel)
            modelsPerf.configDraftsByModelID.removeValue(forKey: model.modelID)
            savedBaseline = [:]
            configStatusNote = "Removed model.heylook.toml overrides."
            await refreshFitAndResidentNow()
            return
        }

        var clearPatch: [String: JSONValue] = [:]
        let fileOnlySet = Set(schemaFields.filter(\.isFileOnly).map(\.name))
        for key in savedBaseline.keys where !fileOnlySet.contains(key) {
            clearPatch[key] = .null
        }
        guard !clearPatch.isEmpty else { return }
        do {
            let res = try await modelsPerf.patchModelConfig(id: liveModel.modelID, patch: clearPatch)
            savedBaseline = res.model.config
            modelsPerf.configDraftsByModelID.removeValue(forKey: model.modelID)
            configStatusNote = "Cleared remote model overrides."
            await refreshFitAndResidentNow()
        } catch {
            configErrorNote = "Failed clearing overrides: \(error.localizedDescription)"
        }
    }
}

// MARK: - Live Debounced RAM Fit Meter View

private struct RAMFitMeterView: View {
    let model: EngineDescription
    let fitReport: RAMFitEvaluator.FitReport?
    let fitErrorMessage: String?
    let residentMemoryMB: Double?
    let residentMemorySource: String?
    let contrastStyle: AdaptiveContrastStyle

    @State private var copiedSysctlCommand: Bool = false

    private var verdictColor: Color {
        switch fitReport?.verdict {
        case "fail": return .red
        case "warn": return .orange
        case "pass": return .green
        default: return .secondary
        }
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text("Memory Fit")
                        .font(.subheadline.weight(.semibold))
                    if let fitReport {
                        Text(fitReport.verdict.uppercased())
                            .font(.caption2.monospaced().weight(.bold))
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(verdictColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                            .foregroundStyle(verdictColor)
                    }
                    Spacer()
                    if let recCtx = fitReport?.recommendedCtxSize, recCtx > 0 {
                        Text("Recommended ctx: \(formatTokenCount(recCtx))")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }

                if let fitReport {
                    VStack(alignment: .leading, spacing: 4) {
                        fitRow(label: "Weights", value: formatGiB(fitReport.weightsGB), note: nil)

                        ForEach(fitReport.sizingNotes, id: \.self) { note in
                            Text(note)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .padding(.leading, 8)
                        }

                        if let wsGB = fitReport.workingSetGB {
                            fitRow(
                                label: "Metal working set",
                                value: formatGiB(wsGB),
                                note: fitReport.hardWorkingSet
                                    ? "hard limit — MLX refuses above it"
                                    : "advisory — llama.cpp pages past it"
                            )
                        }

                        if let kvHeadroom = fitReport.kvHeadroomGB {
                            fitRow(
                                label: "Headroom for KV",
                                value: formatGiB(kvHeadroom),
                                note: nil,
                                isDanger: kvHeadroom < 0
                            )
                        }

                        fitRow(
                            label: "Reclaimable RAM",
                            value: formatGiB(fitReport.reclaimableGB),
                            note: "total − anonymous − wired"
                        )

                        if let workerWS = fitReport.lines.first(where: { $0.ceiling == "worker_metal_working_set" }) {
                            fitRow(
                                label: "Worker Metal working set",
                                value: "\(formatGiB(workerWS.needGB)) / \(formatGiB(workerWS.haveGB))",
                                note: workerWS.note.isEmpty ? "remote rank 1 Metal working set" : workerWS.note,
                                isDanger: workerWS.verdict == "fail"
                            )
                        }

                        if model.isLoaded, let residentMB = residentMemoryMB, residentMB > 0 {
                            fitRow(
                                label: "Resident now",
                                value: formatGiB(residentMB / 1024.0),
                                note: residentMemorySource ?? "measured after load"
                            )
                        }
                    }

                    // Verdict explanation (`model-config.js` lines 624-677)
                    Text(verdictSummaryText(fitReport))
                        .font(.caption)
                        .foregroundStyle(fitReport.verdict == "pass" ? .green : verdictColor)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(verdictColor.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 6))

                    if let suggestMB = fitReport.sysctlSuggestMB {
                        let cmd = "sudo sysctl iogpu.wired_limit_mb=\(suggestMB)"
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(spacing: 8) {
                                Text("Raise the GPU wired limit:")
                                    .font(.caption2.weight(.semibold))
                                Text(cmd)
                                    .font(.caption2.monospaced())
                                    .textSelection(.enabled)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 4))
                                Spacer()
                                Button {
                                    copyToPasteboard(cmd)
                                    copiedSysctlCommand = true
                                } label: {
                                    Label(copiedSysctlCommand ? "Copied" : "Copy", systemImage: copiedSysctlCommand ? "checkmark" : "doc.on.doc")
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                            }
                            Text("Resets at reboot (scripts/gpu_wired_limit.sh install persists it). Restart the server afterwards so it sizes against the new ceiling.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .padding(8)
                        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
                    }
                } else if let fitErrorMessage {
                    Text("fit unavailable — \(fitErrorMessage)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.mini)
                        Text("Calculating RAM fit…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(6)
        }
    }

    @ViewBuilder
    private func fitRow(label: String, value: String, note: String?, isDanger: Bool = false) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
                .frame(width: 150, alignment: .leading)
            Text(value)
                .font(.caption2.monospaced().weight(.semibold))
                .foregroundStyle(isDanger ? .red : .primary)
            if let note, !note.isEmpty {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
        }
    }

    private func verdictSummaryText(_ r: RAMFitEvaluator.FitReport) -> String {
        var pieces: [String] = []
        let workerWs = r.lines.first(where: { $0.ceiling == "worker_metal_working_set" })
        let ws = r.lines.first(where: { $0.ceiling == "metal_working_set" })
        let ram = r.lines.first(where: { $0.ceiling == "reclaimable_ram" })
        let buf = r.lines.first(where: { $0.ceiling == "metal_max_buffer" })

        if let workerWs, workerWs.verdict == "fail" {
            let over = formatGiB(workerWs.needGB - workerWs.haveGB)
            pieces.append("Worker slice exceeds \(workerWs.note.isEmpty ? "worker Metal working set" : workerWs.note) by \(over) (needs \(formatGiB(workerWs.needGB)), has \(formatGiB(workerWs.haveGB))).")
        }
        if let ram, ram.verdict == "fail" {
            if model.isLoaded {
                pieces.append("Loaded now, so this model's own memory is excluded from the ~\(formatGiB(ram.haveGB)) reclaimable figure — unload it to see what a fresh load would find.")
            } else {
                pieces.append("Won't fit: needs \(formatGiB(ram.needGB)) (weights + headroom), ~\(formatGiB(ram.haveGB)) reclaimable.")
            }
        }
        if let ws, ws.verdict != "pass" {
            let over = formatGiB(ws.needGB - ws.haveGB)
            if ws.verdict == "fail" {
                pieces.append("Over the Metal working set by \(over) — MLX refuses above it.")
            } else {
                pieces.append("Over the Metal working set by \(over) — llama.cpp will still load this; Metal stops guaranteeing residency and you degrade into paging.")
            }
        }
        if let buf {
            pieces.append("Weights exceed the \(formatGiB(buf.haveGB)) per-allocation cap — \(buf.note).")
        }
        if r.headroomThin, let kvHeadroom = r.kvHeadroomGB {
            let ggufNote = model.provider == .gguf
                ? "Spawns with llama-server's own micro-batch instead of heylook's larger auto one (slower prefill); "
                : ""
            pieces.append("Thin headroom: \(formatGiB(kvHeadroom)) left for KV + compute. \(ggufNote)a Metal out-of-memory at full context is possible.")
        }
        return pieces.isEmpty ? "Fits." : pieces.joined(separator: " ")
    }
}

// MARK: - 100% Schema-Driven Config Form (`model-config.js` parity)

private struct SchemaConfigFormView: View {
    let model: EngineDescription
    let adminRow: AdminModelWireRow?
    let visibleFields: [ConfigFieldDescriptor]
    let isLoadingSchema: Bool
    let schemaLoadError: String?
    let isRemote: Bool
    let isLocalReadOnly: Bool
    let staleFields: Set<String>
    let contrastStyle: AdaptiveContrastStyle
    let rawValueProvider: (ConfigFieldDescriptor) -> String
    let isDirtyProvider: (ConfigFieldDescriptor) -> Bool
    let onEditField: (ConfigFieldDescriptor, String) -> Void

    @State private var showAdvanced: Bool = false

    private var liveFields: [ConfigFieldDescriptor] {
        visibleFields.filter {
            ($0.effect == .perRequest || $0.effect == .appliesLive || $0.effect == .descriptive)
                && $0.ui != .advanced
        }
    }

    private var reloadFields: [ConfigFieldDescriptor] {
        visibleFields.filter { $0.effect == .requiresReload && $0.ui != .advanced }
    }

    private var advancedFields: [ConfigFieldDescriptor] {
        visibleFields.filter { $0.ui == .advanced && $0.effect != .loadTimeOnly }
    }

    private var fixedFields: [ConfigFieldDescriptor] {
        visibleFields.filter { $0.effect == .loadTimeOnly }
    }

    var body: some View {
        if isLoadingSchema {
            GroupBox("Provider Configuration Schema") {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.mini)
                    Text("Loading option schema…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
            }
        } else if let schemaLoadError {
            GroupBox("Provider Configuration Schema") {
                Text(schemaLoadError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(8)
            }
        } else if visibleFields.isEmpty {
            GroupBox("Provider Configuration Schema") {
                Text("No editable options for provider \"\(model.provider.rawValue)\".")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(8)
            }
        } else {
            VStack(alignment: .leading, spacing: 12) {
                if !liveFields.isEmpty {
                    GroupBox("Applies immediately") {
                        fieldList(liveFields)
                            .padding(6)
                    }
                }

                if !reloadFields.isEmpty {
                    let note = model.isLoaded
                        ? "Saved to the model's model.heylook.toml now; the loaded model keeps running as-is until reloaded."
                        : "Saved to the model's model.heylook.toml; applies when the model loads."
                    GroupBox("Requires reload") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(note)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            fieldList(reloadFields)
                        }
                        .padding(6)
                    }
                }

                if !advancedFields.isEmpty {
                    let allReload = advancedFields.allSatisfy { $0.effect == .requiresReload }
                    GroupBox {
                        DisclosureGroup(
                            allReload ? "Advanced (requires reload)" : "Advanced",
                            isExpanded: $showAdvanced
                        ) {
                            fieldList(advancedFields)
                                .padding(.top, 6)
                        }
                        .font(.subheadline.weight(.medium))
                        .padding(6)
                    }
                }

                if !fixedFields.isEmpty {
                    GroupBox("Fixed for this process (Load-time only)") {
                        fieldList(fixedFields)
                            .padding(6)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func fieldList(_ fields: [ConfigFieldDescriptor]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(fields, id: \.name) { field in
                SchemaFieldRowView(
                    field: field,
                    settingFact: adminRow?.engine.settings[field.name],
                    rawValue: rawValueProvider(field),
                    isDirty: isDirtyProvider(field),
                    isStale: staleFields.contains(field.name),
                    isRemote: isRemote,
                    isLocalReadOnly: isLocalReadOnly,
                    contrastStyle: contrastStyle,
                    onEdit: { next in onEditField(field, next) }
                )
            }
        }
    }
}

private struct SchemaFieldRowView: View {
    let field: ConfigFieldDescriptor
    let settingFact: SettingFact?
    let rawValue: String
    let isDirty: Bool
    let isStale: Bool
    let isRemote: Bool
    let isLocalReadOnly: Bool
    let contrastStyle: AdaptiveContrastStyle
    let onEdit: (String) -> Void

    private var isDisabled: Bool {
        field.effect == .loadTimeOnly || field.effect == .identity || isLocalReadOnly || (field.isFileOnly && isRemote)
    }

    private var autoDefaultLabel: String? {
        guard let value = settingFact?.auto ?? field.defaultValue, value != .null else {
            return nil
        }
        if field.type == .boolean {
            if case .bool(let b) = value { return b ? "on" : "off" }
        }
        if case .bool(let b) = value {
            return b ? "on" : "off"
        }
        if case .array(let items) = value {
            return items.map(formatJSONScalar).joined(separator: ", ")
        }
        return formatJSONScalar(value)
    }

    private var boundsText: String? {
        var parts: [String] = []
        if let min = field.minimum { parts.append("min \(formatBoundNumber(min))") }
        if let exMin = field.exclusiveMinimum { parts.append("> \(formatBoundNumber(exMin))") }
        if let max = field.maximum { parts.append("max \(formatBoundNumber(max))") }
        if let exMax = field.exclusiveMaximum { parts.append("< \(formatBoundNumber(exMax))") }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(field.name)
                    .font(.caption.monospaced().weight(.semibold))
                    .foregroundStyle(isDisabled ? .secondary : .primary)

                if let arg = field.arg, !arg.isEmpty {
                    Text(arg)
                        .font(.system(size: 9.5, design: .monospaced))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.secondary.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.secondary)
                }

                if isStale {
                    Text("Pending reload")
                        .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Color.orange.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.orange)
                }

                if isDirty {
                    Text("Modified")
                        .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Color.blue.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.blue)
                }

                if field.isFileOnly && isRemote {
                    Text("file_only")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Color.orange.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                        .foregroundStyle(.orange)
                }

                Spacer()

                if !rawValue.isEmpty && !isDisabled {
                    Button {
                        onEdit("")
                    } label: {
                        Label("Unset", systemImage: "xmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear override (send null to unset)")
                }
            }

            controlBody

            HStack(spacing: 6) {
                if let def = autoDefaultLabel {
                    Text("default \(def)")
                }
                if let boundsText {
                    Text("· \(boundsText)")
                }
                if field.type == .array {
                    let arrayHint = (field.itemType == .number || field.itemType == .integer)
                        ? "one per line or comma-separated"
                        : "one element per line"
                    Text("· \(arrayHint)")
                }
            }
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)

            if field.isFileOnly && isRemote {
                Text("File-only: must be edited in model.heylook.toml or heylook.toml on the host; refused over remote admin API.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            } else if let reason = field.reason, !reason.isEmpty {
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }

            if !field.description.isEmpty {
                Text(field.description)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isStale ? Color.orange.opacity(0.06) : Color.secondary.opacity(0.04))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(
                    isStale ? contrastStyle.borderColor(.orange) : contrastStyle.subtleBorderColor(.secondary),
                    lineWidth: contrastStyle.borderWidth
                )
        )
    }

    @ViewBuilder
    private var controlBody: some View {
        if field.type == .boolean || field.enumValues != nil {
            let defaultTitle = autoDefaultLabel.map { "default (\($0))" } ?? "default"
            let choices: [(tag: String, label: String)] = field.type == .boolean
                ? [("true", "on"), ("false", "off")]
                : (field.enumValues ?? []).map { ($0, $0) }
            Picker(
                field.name,
                selection: Binding(
                    get: { rawValue },
                    set: { onEdit($0) }
                )
            ) {
                Text(defaultTitle).tag("")
                ForEach(choices, id: \.tag) { item in
                    Text(item.label).tag(item.tag)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .disabled(isDisabled)
        } else if field.type == .array {
            TextEditor(
                text: Binding(
                    get: { rawValue },
                    set: { onEdit($0) }
                )
            )
            .font(.system(.caption, design: .monospaced))
            .frame(minHeight: 54, maxHeight: 96)
            .disabled(isDisabled)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(contrastStyle.borderColor(.secondary), lineWidth: contrastStyle.borderWidth)
            )
        } else {
            TextField(
                autoDefaultLabel ?? "default",
                text: Binding(
                    get: { rawValue },
                    set: { onEdit($0) }
                )
            )
            .textFieldStyle(.roundedBorder)
            .font(.caption.monospaced())
            .disabled(isDisabled)
        }
    }

    private func formatBoundNumber(_ v: Double) -> String {
        v.rounded() == v ? String(Int(v)) : String(v)
    }
}

// MARK: - Full Multi-Rung Chat Template Inspector (`model-config.js` parity)

private struct ChatTemplateInspectorView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let model: EngineDescription
    let contrastStyle: AdaptiveContrastStyle
    let onLoadedBaselineChanged: (_ baseline: String, _ loaded: Bool) -> Void

    @State private var templateState: HeylookConfig.ChatTemplateState?
    @State private var loadError: String?
    @State private var isLoading: Bool = false
    @State private var isSaving: Bool = false
    @State private var isExpanded: Bool = true
    @State private var showWinningWinnerBody: Bool = false
    @State private var showRevertConfirm: Bool = false
    @State private var statusBanner: String?
    @State private var syntaxError: String?

    private var modelsPerf: ModelsAndPerfController { viewModel.modelsPerf }

    private var serverBaselineBody: String {
        guard let state = templateState else { return "" }
        if state.hasOverride {
            return state.overrideTemplate ?? state.template
        }
        return state.template
    }

    private var editorTextBinding: Binding<String> {
        Binding(
            get: {
                modelsPerf.templateDraftsByModelID[model.modelID] ?? serverBaselineBody
            },
            set: { next in
                if normalizeEOL(next) == normalizeEOL(serverBaselineBody) {
                    modelsPerf.templateDraftsByModelID.removeValue(forKey: model.modelID)
                } else {
                    modelsPerf.templateDraftsByModelID[model.modelID] = next
                }
                syntaxError = viewModel.validateJinjaTemplateSyntax(next)
            }
        )
    }

    private var currentBody: String {
        modelsPerf.templateDraftsByModelID[model.modelID] ?? serverBaselineBody
    }

    private var isDirty: Bool {
        guard templateState != nil else { return false }
        let normalizedCurrent = normalizeEOL(currentBody)
        return normalizedCurrent != normalizeEOL(serverBaselineBody)
            && !normalizedCurrent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        GroupBox {
            DisclosureGroup("Multi-Rung Chat Template (chat_template.heylook.jinja)", isExpanded: $isExpanded) {
                VStack(alignment: .leading, spacing: 10) {
                    if isLoading {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.mini)
                            Text("Loading chat template ladder…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } else if let loadError {
                        Text("Could not read chat template: \(loadError)")
                            .font(.caption)
                            .foregroundStyle(.red)
                    } else if let state = templateState {
                        templateStatusAndOriginHeader(state)
                        sourcesLadderView(state)
                    }

                    // Always display & edit `override_template` (even when `inert_reason` explains why an explicit path outranks it!)
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(templateState?.hasOverride == true ? "Override Template Body (chat_template.heylook.jinja)" : "Template Editor")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Spacer()
                            if let state = templateState, !state.writable {
                                Text("READ-ONLY")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Capsule().fill(Color.orange.opacity(contrastStyle.chipFillOpacity)))
                                    .foregroundStyle(.orange)
                            }
                        }

                        TextEditor(text: editorTextBinding)
                            .font(.system(.caption, design: .monospaced))
                            .disabled(templateState?.writable == false)
                            .frame(minHeight: 130, maxHeight: 200)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .strokeBorder(
                                        syntaxError != nil ? Color.red.opacity(0.6) : contrastStyle.borderColor(.secondary),
                                        lineWidth: contrastStyle.borderWidth
                                    )
                            )

                        if let syntaxError {
                            Label(syntaxError, systemImage: "xmark.octagon.fill")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.red)
                        }

                        if let state = templateState,
                           state.hasOverride,
                           state.inertReason != nil,
                           let ovBody = state.overrideTemplate,
                           ovBody != state.template,
                           !state.template.isEmpty {
                            DisclosureGroup("Active Winning Template Body (\(state.source))", isExpanded: $showWinningWinnerBody) {
                                ScrollView {
                                    Text(state.template)
                                        .font(.system(.caption2, design: .monospaced))
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .textSelection(.enabled)
                                }
                                .frame(maxHeight: 110)
                                .padding(6)
                                .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))
                            }
                            .font(.caption2.weight(.semibold))
                        }

                        HStack(spacing: 8) {
                            Button {
                                Task { await saveTemplate() }
                            } label: {
                                if isSaving {
                                    ProgressView().controlSize(.mini)
                                } else {
                                    Text("Save Template")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(
                                isSaving
                                    || templateState == nil
                                    || templateState?.writable == false
                                    || !isDirty
                                    || syntaxError != nil
                            )

                            if templateState?.hasOverride == true {
                                Button("Revert Override…", role: .destructive) {
                                    showRevertConfirm = true
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                .disabled(isSaving || templateState?.writable == false)
                            }

                            Spacer()
                        }

                        if let statusBanner, !statusBanner.isEmpty {
                            Text(statusBanner)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.top, 8)
            }
            .font(.subheadline.weight(.medium))
            .padding(6)
        }
        .confirmationDialog(
            "Discard Chat Template Override?",
            isPresented: $showRevertConfirm,
            titleVisibility: .visible
        ) {
            Button("Revert to Model Default", role: .destructive) {
                Task { await revertTemplate() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Deletes chat_template.heylook.jinja for \(model.modelID) so the vendor template ladder takes effect again.")
        }
        .task(id: model.modelID) {
            await loadTemplateState()
        }
    }

    @ViewBuilder
    private func templateStatusAndOriginHeader(_ state: HeylookConfig.ChatTemplateState) -> some View {
        let inForceEntry = state.sources.first(where: \.inForce)
        let inForceIndex = state.sources.firstIndex(where: \.inForce) ?? -1
        let belowFiles: [String] = {
            guard inForceEntry?.source == "heylook override", inForceIndex >= 0 else { return [] }
            return state.sources.dropFirst(inForceIndex + 1).compactMap(\.file)
        }()

        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("In force: \(state.origin) (\(state.source))")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)

                if state.hasOverride {
                    Text("override on disk")
                        .font(.caption2.monospaced().weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.purple.opacity(contrastStyle.chipFillOpacity)))
                        .foregroundStyle(.purple)
                }

                // Tri-state `stale` (`observedStale`: true / false / nil)
                staleBadge(for: state.observedStale)

                if state.modifiedSinceDownload == true {
                    Text("MODIFIED SINCE DOWNLOAD")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.orange.opacity(contrastStyle.chipFillOpacity)))
                        .foregroundStyle(.orange)
                }

                Spacer()
            }

            if !belowFiles.isEmpty {
                Text("Your override hides \(belowFiles.joined(separator: ", ")); Revert to use \(belowFiles.count > 1 ? "them" : "it").")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }

            let thinkingFacts = inForceEntry?.thinking
            Text(thinkingSummaryLine(thinkingFacts, fallbackSource: inForceEntry?.file ?? inForceEntry?.source))
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let inert = state.inertReason, !inert.isEmpty {
                Label("Override Inert: \(inert)", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .padding(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 6))
            }

            if !state.refusedShapes.isEmpty {
                Label(
                    "This template refuses \(state.refusedShapes.count == 1 ? "a message shape" : "some message shapes") — requests of that shape will fail: \(state.refusedShapes.joined(separator: "; "))",
                    systemImage: "xmark.shield.fill"
                )
                .font(.caption2)
                .foregroundStyle(.red)
                .padding(6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.red.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 6))
            }

            if !state.prefixStable {
                Label(
                    "This template breaks multi-turn prompt caching: \(state.prefixNote ?? "template mutates history prefix")",
                    systemImage: "bolt.trianglebadge.exclamationmark.fill"
                )
                .font(.caption2)
                .foregroundStyle(.orange)
            }

            TemplateCapabilitiesStripView(caps: state.caps)
        }
    }

    @ViewBuilder
    private func staleBadge(for observedStale: Bool?) -> some View {
        switch observedStale {
        case true?:
            Text("STALE (RELOAD NEEDED)")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.orange.opacity(contrastStyle.chipFillOpacity)))
                .foregroundStyle(.orange)
        case false?:
            Text("MATCHES RUNNING")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.green.opacity(contrastStyle.chipFillOpacity)))
                .foregroundStyle(.green)
        case nil:
            Text("NOT LOADED")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.secondary.opacity(contrastStyle.chipFillOpacity)))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func sourcesLadderView(_ state: HeylookConfig.ChatTemplateState) -> some View {
        if !state.sources.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("Template Ladder Copies (\(state.sources.count))")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)

                ForEach(Array(state.sources.enumerated()), id: \.offset) { _, src in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Text(src.source)
                                .font(.caption2.weight(.semibold))
                            if let file = src.file, !file.isEmpty, file != src.source {
                                Text("(\(file))")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            if src.inForce {
                                Text("IN FORCE")
                                    .font(.system(size: 8.5, weight: .bold, design: .monospaced))
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(Capsule().fill(Color.green.opacity(contrastStyle.chipFillOpacity)))
                                    .foregroundStyle(.green)
                            }
                            if src.provenance == "modified since download" {
                                Text("MODIFIED SINCE DOWNLOAD")
                                    .font(.system(size: 8.5, weight: .bold, design: .monospaced))
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(Capsule().fill(Color.orange.opacity(contrastStyle.chipFillOpacity)))
                                    .foregroundStyle(.orange)
                            }
                            Spacer()
                            if src.source != "heylook override" {
                                if src.inForce && !state.hasOverride {
                                    Text("shown in editor below")
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                } else {
                                    Button("Start an override from this") {
                                        modelsPerf.templateDraftsByModelID[model.modelID] = src.template
                                        syntaxError = viewModel.validateJinjaTemplateSyntax(src.template)
                                        statusBanner = normalizeEOL(src.template) == normalizeEOL(serverBaselineBody)
                                            ? "\(src.source) is identical to what the editor holds; edit it to save an override."
                                            : "Editing a copy of \(src.source). Save to write this model's override."
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                    .disabled(!state.writable)
                                }
                            }
                        }

                        let commitTag = src.downloadCommit.map { " @\(String($0.prefix(8)))" } ?? ""
                        let shaShort = String(src.sha256.prefix(8))
                        Text("\(src.provenance)\(commitTag) · sha \(shaShort)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)

                        if let tFacts = src.thinking {
                            Text(thinkingSummaryLine(tFacts, fallbackSource: src.file ?? src.source, prefixLabel: "thinking"))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }

                        if src.prefixStable == false {
                            Text("breaks prompt caching: \(src.prefixNote ?? "template mutates history prefix")")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }
                    .padding(7)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(src.inForce ? Color.green.opacity(0.06) : Color.secondary.opacity(0.04))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(
                                src.inForce ? contrastStyle.borderColor(.green) : contrastStyle.subtleBorderColor(.secondary),
                                lineWidth: contrastStyle.borderWidth
                            )
                    )
                }
            }
        }
    }

    private func loadTemplateState() async {
        isLoading = true
        loadError = nil
        do {
            let state = try await modelsPerf.fetchChatTemplateState(for: model)
            templateState = state
            let baseline = state.hasOverride ? (state.overrideTemplate ?? state.template) : state.template
            onLoadedBaselineChanged(baseline, true)
            syntaxError = nil
        } catch {
            loadError = error.localizedDescription
            onLoadedBaselineChanged("", false)
        }
        isLoading = false
    }

    private func saveTemplate() async {
        guard !isSaving else { return }
        isSaving = true
        statusBanner = "Saving template…"
        defer { isSaving = false }
        do {
            let saved = try await viewModel.saveChatTemplateOverride(currentBody, for: model)
            templateState = saved
            let baseline = saved.hasOverride ? (saved.overrideTemplate ?? saved.template) : saved.template
            onLoadedBaselineChanged(baseline, true)
            syntaxError = nil
            statusBanner = saved.observedStale == true
                ? "Saved. Reload the model to apply the new template."
                : "Saved chat_template.heylook.jinja."
        } catch {
            syntaxError = error.localizedDescription
            statusBanner = "Save failed: \(error.localizedDescription)"
        }
    }

    private func revertTemplate() async {
        guard !isSaving else { return }
        isSaving = true
        statusBanner = "Reverting override…"
        defer { isSaving = false }
        let reverted = await viewModel.revertChatTemplateOverride(for: model)
        templateState = reverted
        let baseline = reverted.hasOverride ? (reverted.overrideTemplate ?? reverted.template) : reverted.template
        onLoadedBaselineChanged(baseline, true)
        syntaxError = nil
        statusBanner = "Reverted override (active: \(reverted.source))."
    }

    private func thinkingSummaryLine(
        _ thinking: EngineThinkingFacts?,
        fallbackSource: String?,
        prefixLabel: String = "Thinking controls"
    ) -> String {
        guard let thinking else {
            return "\(prefixLabel): unknown (no template to judge)."
        }
        var parts: [String] = [
            thinking.switchVariable.map { "switch \($0)" } ?? "no thinking switch"
        ]
        if let d = thinking.depth {
            let vals: String = if d.unknown == "verbatim" {
                "any word\(d.values.isEmpty ? "" : " (names \(d.values.joined(separator: ", ")))")"
            } else {
                d.values.joined(separator: ", ")
            }
            let defPart = d.defaultValue.map { ", default \($0)" } ?? ""
            parts.append("depth \(d.variable): \(vals)\(defPart)")
        } else {
            parts.append("no depth control")
        }
        let from = (thinking.template ?? fallbackSource).map { " (from \($0))" } ?? ""
        return "\(prefixLabel)\(from): \(parts.joined(separator: "; "))."
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

// MARK: - 5. Perf & Server Surface (`perf.js` parity, zero background polling)

struct PerfServerSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    @State private var retentionDaysDraft: String = "30"
    @State private var mlxCacheLimitDraft: String = ""
    @State private var obsInputValidationError: String?

    private var modelsPerf: ModelsAndPerfController { viewModel.modelsPerf }

    private static let clockFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss"
        return df
    }()

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)

        Form {
            if let perfError = modelsPerf.perfLoadError, !perfError.isEmpty {
                Section {
                    Label(perfError, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            // 1. System Strip + Per-Loaded-Model Telemetry Cards
            Section {
                HStack(spacing: 10) {
                    if let updatedAt = modelsPerf.lastMetricsUpdatedAt {
                        Text("Updated \(Self.clockFormatter.string(from: updatedAt))")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    } else {
                        Text("On-demand telemetry (no background polling)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        Task {
                            await modelsPerf.refreshMetrics(forceRefresh: true)
                            viewModel.refreshPerfAndObservability()
                            await modelsPerf.loadOperationalConfigAndDiskLogs()
                        }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }

                LabeledContent("Physical Unified RAM") {
                    Text(String(
                        format: "%.1f GB used • %.1f GB avail • %.1f GB total",
                        viewModel.memorySnapshot.ramUsedGB,
                        viewModel.memorySnapshot.ramAvailableGB,
                        viewModel.memorySnapshot.totalPhysicalRAMGB
                    ))
                    .font(.caption.monospaced().weight(.semibold))
                }
                LabeledContent("Metal Working Set / Allocated / RSS / CPU") {
                    Text(String(
                        format: "%.1f GB max • %d MB Metal • %d MB RSS • %.1f%% CPU",
                        viewModel.memorySnapshot.metalRecommendedMaxWorkingSetGB,
                        viewModel.memorySnapshot.metalCurrentAllocatedMB,
                        viewModel.memorySnapshot.processResidentMB,
                        viewModel.memorySnapshot.cpuPercent
                    ))
                    .font(.caption.monospaced().weight(.semibold))
                }
                LabeledContent("Resident GPU / ANE Models") {
                    Text("\(viewModel.schedulerMetrics.loadedGPUModels.count) GPU • \(viewModel.schedulerMetrics.loadedANEModels.count) ANE")
                        .font(.subheadline.monospaced().weight(.semibold))
                }

                if modelsPerf.loadedModelTelemetry.isEmpty {
                    Text("No models loaded right now.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(modelsPerf.loadedModelTelemetry) { item in
                        LoadedModelTelemetryCardView(item: item, contrastStyle: contrastStyle)
                    }
                }
            } header: {
                Text("System & Per-Loaded-Model Telemetry (\(viewModel.memorySnapshot.deviceName))")
            }

            // 2. Lifecycle Reclamation & GGUF Port Pool
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
                        ScrollView(.horizontal, showsIndicators: false) {
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

            // 3. ANE Tower Compute Plan (when populated)
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

            // 4. 3 Performance Profile Tables (`1h` / `6h` / `24h` / `7d`)
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
                        Label("Refresh Profile", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                if let profile = viewModel.perfProfileSummary,
                   (profile.sampleCount > 0 || !profile.cache.isEmpty || !profile.trends.isEmpty || profile.timingBreakdown.contains(where: { $0.count > 0 })) {
                    let cacheRatio = profile.totalPromptTokens > 0
                        ? (Double(profile.totalCachedTokens) / Double(profile.totalPromptTokens)) * 100.0
                        : 0.0
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            PerfStatBadge(title: "Samples", value: "\(profile.sampleCount)")
                            PerfStatBadge(title: "Avg Gen", value: String(format: "%.1f tok/s", profile.averageGenerationTPS))
                            PerfStatBadge(title: "Avg Prompt", value: String(format: "%.0f tok/s", profile.averagePromptTPS))
                            PerfStatBadge(title: "Avg TTFT", value: String(format: "%.0f ms", profile.averageTTFTMs))
                            PerfStatBadge(title: "Output Tok", value: "\(profile.totalOutputTokens)")
                            PerfStatBadge(title: "KV Cache Hit", value: String(format: "%.0f%%", cacheRatio))
                        }
                    }

                    if !profile.timingBreakdown.isEmpty {
                        TimingBreakdownTableView(rows: profile.timingBreakdown, contrastStyle: contrastStyle)
                    }
                    if !profile.cache.isEmpty {
                        PerModelCacheTableView(rows: profile.cache, contrastStyle: contrastStyle)
                    }
                    if !profile.trends.isEmpty {
                        RecentTrendsTableView(rows: Array(profile.trends.suffix(8)), contrastStyle: contrastStyle)
                    }
                } else {
                    Text("No persisted turn statistics recorded in SQLite within the \(viewModel.selectedPerfWindow.label) window yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Performance Profile (/v1/performance/profile/\(viewModel.selectedPerfWindow.rawValue))")
            }

            // 5. Embedded Server & Bonjour UI Truth
            Section {
                LabeledContent("HTTP & SSE Listener") {
                    VStack(alignment: .trailing, spacing: 4) {
                        HStack(spacing: 8) {
                            Circle()
                                .fill(viewModel.isServerRunning ? Color.green : Color.secondary)
                                .frame(width: 8, height: 8)
                            Text(modelsPerf.isServerLANReachable ? "LAN + BONJOUR" : "LOCALHOST ONLY")
                                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(
                                    (modelsPerf.isServerLANReachable ? Color.blue : Color.orange).opacity(contrastStyle.chipFillOpacity),
                                    in: Capsule()
                                )
                                .foregroundStyle(modelsPerf.isServerLANReachable ? .blue : .orange)
                            Button(viewModel.isServerRunning ? "Stop Server" : "Start Server") {
                                Task { await viewModel.toggleServer() }
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        }
                        Text(serverListenerTruthLine)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.trailing)
                    }
                }
            } header: {
                Text("Embedded HTTP & SSE Server (/v1/messages, /v1/embeddings, /v1/conversations)")
            } footer: {
                Text("Supports disconnect-resilient background generation (`heylook_saved`) for iOS 27.2+ clients. Bonjour `_heylook._tcp` LAN advertisement activates only when bound to wildcard host (`0.0.0.0`).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // 6. Bonjour Discovery & Remote Server Connection (truthful state)
            Section("Bonjour (_heylook._tcp) Discovery & Remote Heylook Server") {
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

                if let remoteError = modelsPerf.remoteConnectionError, !remoteError.isEmpty {
                    Label(remoteError, systemImage: "xmark.octagon.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                if let connected = viewModel.connectedRemoteServerURL {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Connected to remote server: \(connected)")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Disconnect", role: .destructive) {
                            Task { await viewModel.disconnectRemoteServer() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                } else {
                    Text("Not connected to a remote server (operating on local workspace).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            // 7. Persistent `[settings]` Observability Config + `events` vs `metrics` Disk JSONL Log Reader
            Section {
                if !modelsPerf.operationalMemoryOnlyKeys.isEmpty || (modelsPerf.activeRemoteBackend == nil && HeylookConfig.isReadOnlyModelConfig) {
                    let keysStr = modelsPerf.operationalMemoryOnlyKeys.isEmpty
                        ? "all keys"
                        : modelsPerf.operationalMemoryOnlyKeys.joined(separator: ", ")
                    Label(
                        "Read-only config mode (memory_only: \(keysStr)) — updates apply in memory for this session only and are not written to heylook.toml.",
                        systemImage: "lock.fill"
                    )
                    .font(.caption2)
                    .foregroundStyle(.orange)
                }

                LabeledContent("observability_level") {
                    Picker(
                        "Level",
                        selection: Binding(
                            get: { modelsPerf.operationalSettings.observabilityLevel },
                            set: { next in
                                Task {
                                    await modelsPerf.updateOperationalSetting([
                                        "observability_level": .string(next.rawValue)
                                    ])
                                }
                            }
                        )
                    ) {
                        ForEach(ObservabilityVerbosity.allCases, id: \.self) { level in
                            Text(level.rawValue).tag(level)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 280)
                    .disabled(modelsPerf.isSavingOperationalSettings)
                }

                LabeledContent("observability_retention_days") {
                    HStack(spacing: 6) {
                        TextField("30", text: $retentionDaysDraft)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption.monospaced())
                            .frame(width: 72)
                        Button("Apply") {
                            obsInputValidationError = nil
                            guard let days = Int(retentionDaysDraft.trimmingCharacters(in: .whitespacesAndNewlines)), days >= 0 else {
                                obsInputValidationError = "Retention days must be a whole number >= 0."
                                return
                            }
                            Task {
                                await modelsPerf.updateOperationalSetting(["observability_retention_days": .int(days)])
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(modelsPerf.isSavingOperationalSettings)
                    }
                }

                LabeledContent("mlx_cache_limit_gb") {
                    HStack(spacing: 6) {
                        TextField("auto (unlimited)", text: $mlxCacheLimitDraft)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption.monospaced())
                            .frame(width: 120)
                        Button("Apply") {
                            obsInputValidationError = nil
                            let trimmed = mlxCacheLimitDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                            if trimmed.isEmpty {
                                Task {
                                    await modelsPerf.updateOperationalSetting(["mlx_cache_limit_gb": .null])
                                }
                                return
                            }
                            guard let gb = Double(trimmed), gb.isFinite, gb > 0 else {
                                obsInputValidationError = "mlx_cache_limit_gb must be a positive number (or empty for auto)."
                                return
                            }
                            Task {
                                await modelsPerf.updateOperationalSetting(["mlx_cache_limit_gb": .double(gb)])
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(modelsPerf.isSavingOperationalSettings)

                        if modelsPerf.operationalSettings.mlxCacheLimitGB != nil {
                            Button("Reset") {
                                Task {
                                    await modelsPerf.resetOperationalSettingKey("mlx_cache_limit_gb")
                                    mlxCacheLimitDraft = ""
                                }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                            .disabled(modelsPerf.isSavingOperationalSettings)
                        }
                    }
                }

                if let obsInputValidationError {
                    Text(obsInputValidationError)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }

                if let status = modelsPerf.operationalSettingsStatus {
                    Text(status)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }

                Divider()

                HStack(spacing: 10) {
                    Text("JSONL Disk Stream:")
                        .font(.caption.weight(.semibold))
                    Picker(
                        "Stream",
                        selection: Binding(
                            get: { modelsPerf.selectedDiskLogStream },
                            set: { next in
                                Task { await modelsPerf.loadDiskObservabilityLogs(stream: next) }
                            }
                        )
                    ) {
                        Text("events.jsonl").tag(ObservabilityStream.events)
                        Text("metrics.jsonl").tag(ObservabilityStream.metrics)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 230)

                    Spacer()

                    Button {
                        Task { await modelsPerf.loadDiskObservabilityLogs() }
                    } label: {
                        Label("Reload Logs", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }

                if modelsPerf.diskLogRecords.isEmpty {
                    let msg = modelsPerf.diskLogLevel == "off"
                        ? "Observability is off — choose minimal, standard, or debug above to record structured JSONL telemetry."
                        : "No \(modelsPerf.selectedDiskLogStream.rawValue) records written yet."
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    DiskJSONLLogTableView(
                        stream: modelsPerf.selectedDiskLogStream,
                        records: Array(modelsPerf.diskLogRecords.prefix(50)),
                        contrastStyle: contrastStyle
                    )
                }
            } header: {
                Text("Operational [settings] & Disk JSONL Streams (/v1/admin/config, /v1/admin/config/logs)")
            }

            // 8. In-Memory Observability Ring (`HeylookObservabilityLog`)
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
                        copyToPasteboard(viewModel.exportObservabilityJSONL())
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
                    .frame(maxHeight: 220)
                }
            } header: {
                Text("In-Memory Observability Event Ring (HeylookObservabilityLog)")
            }

            // 9. Data Maintenance & Store Reset
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
            Task {
                await modelsPerf.refreshMetrics(forceRefresh: false)
                await modelsPerf.loadOperationalConfigAndDiskLogs()
                syncOperationalDraftsFromController()
            }
        }
        .onChange(of: modelsPerf.operationalSettings) {
            syncOperationalDraftsFromController()
        }
    }

    private var serverListenerTruthLine: String {
        let hostDisplay = modelsPerf.serverBindHost.isEmpty ? "0.0.0.0" : modelsPerf.serverBindHost
        if viewModel.isServerRunning {
            if modelsPerf.isServerLANReachable {
                let bonjourMsg = modelsPerf.isBonjourAdvertisementActive
                    ? "Bonjour _heylook._tcp active"
                    : "Bonjour disabled"
                return "Listening on http://\(hostDisplay):\(viewModel.serverPort) — LAN reachable (\(bonjourMsg))"
            } else {
                return "Listening on http://\(hostDisplay):\(viewModel.serverPort) — localhost-only; Bonjour LAN broadcast inactive"
            }
        } else {
            let mode = modelsPerf.isServerLANReachable
                ? "LAN reachable + Bonjour _heylook._tcp when started"
                : "localhost-only (Bonjour LAN broadcast inactive)"
            return "Stopped (configured \(hostDisplay):\(viewModel.serverPort) — \(mode))"
        }
    }

    private func syncOperationalDraftsFromController() {
        retentionDaysDraft = "\(modelsPerf.operationalSettings.observabilityRetentionDays)"
        if let limit = modelsPerf.operationalSettings.mlxCacheLimitGB {
            mlxCacheLimitDraft = limit.rounded() == limit ? String(Int(limit)) : String(limit)
        } else {
            mlxCacheLimitDraft = ""
        }
    }
}

// MARK: - Perf Surface Subviews

private struct LoadedModelTelemetryCardView: View {
    let item: LoadedModelTelemetry
    let contrastStyle: AdaptiveContrastStyle

    private var memoryLabel: String {
        guard let mb = item.memoryMB, mb > 0 else { return "memory --" }
        if mb >= 1024.0 {
            return String(format: "%.1f GiB", mb / 1024.0)
        }
        return String(format: "%.0f MB", mb)
    }

    private var contextBarFraction: Double {
        guard let pct = item.contextPercent else {
            if let used = item.contextUsed, let cap = item.contextCapacity, cap > 0 {
                return min(1.0, max(0.0, Double(used) / Double(cap)))
            }
            return 0.0
        }
        return min(1.0, max(0.0, pct / 100.0))
    }

    private var contextSummaryText: String {
        let capStr = item.contextCapacity.map { $0.formatted() } ?? "--"
        guard let used = item.contextUsed else {
            return "ctx -- / \(capStr)"
        }
        let pctVal = item.contextPercent ?? (item.contextCapacity.flatMap { $0 > 0 ? (Double(used) / Double($0)) * 100.0 : nil })
        if let pctVal {
            return String(format: "ctx %@ / %@ (%.1f%%)", used.formatted(), capStr, pctVal)
        }
        return "ctx \(used.formatted()) / \(capStr)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(item.modelID)
                    .font(.caption.monospaced().weight(.semibold))
                    .lineLimit(1)
                Text(item.providerName)
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1.5)
                    .background(Color.accentColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                Spacer()
                VStack(alignment: .trailing, spacing: 1) {
                    Text(memoryLabel)
                        .font(.caption.monospaced().weight(.semibold))
                    if let source = item.memorySource, !source.isEmpty {
                        Text(source)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            ProgressView(value: contextBarFraction, total: 1.0)
                .tint(contextBarFraction > 0.85 ? .orange : .accentColor)

            HStack {
                Text(contextSummaryText)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(item.requestsActive.map(String.init) ?? "--") active · \(item.requestsQueued.map(String.init) ?? "--") queued")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(contrastStyle.subtleBorderColor(.secondary), lineWidth: contrastStyle.borderWidth)
        )
    }
}

private struct TimingBreakdownTableView: View {
    let rows: [SQLiteStore.TimingBreakdownEntry]
    let contrastStyle: AdaptiveContrastStyle

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Timing Breakdown")
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 12) {
                        Text("Operation").frame(width: 150, alignment: .leading)
                        Text("Avg ms").frame(width: 74, alignment: .trailing)
                        Text("Count").frame(width: 60, alignment: .trailing)
                        Text("%").frame(width: 150, alignment: .leading)
                    }
                    .font(.caption2.monospaced().weight(.bold))
                    .foregroundStyle(.secondary)

                    Divider()

                    ForEach(rows, id: \.operation) { row in
                        let pct = min(100.0, max(0.0, row.percentage * 100.0))
                        HStack(spacing: 12) {
                            Text(row.operation)
                                .frame(width: 150, alignment: .leading)
                            Text(String(format: "%.1f", row.avgTimeMs))
                                .frame(width: 74, alignment: .trailing)
                            Text("\(row.count)")
                                .frame(width: 60, alignment: .trailing)
                            HStack(spacing: 6) {
                                ProgressView(value: pct, total: 100.0)
                                    .frame(width: 80)
                                Text(String(format: "%.1f%%", pct))
                                    .frame(width: 54, alignment: .trailing)
                            }
                            .frame(width: 150, alignment: .leading)
                        }
                        .font(.caption2.monospaced())
                    }
                }
                .padding(8)
            }
            .background(Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(contrastStyle.subtleBorderColor(.secondary), lineWidth: contrastStyle.borderWidth)
            )
        }
    }
}

private struct PerModelCacheTableView: View {
    let rows: [SQLiteStore.ModelCacheProfile]
    let contrastStyle: AdaptiveContrastStyle

    private static let outcomeOrder = ["reused", "miss", "ineligible"]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Per-Model Cache")
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 12) {
                        Text("Model").frame(width: 180, alignment: .leading)
                        Text("Reqs").frame(width: 48, alignment: .trailing)
                        Text("Hit %").frame(width: 64, alignment: .trailing)
                        Text("Saved tok").frame(width: 84, alignment: .trailing)
                        Text("Outcomes").frame(width: 190, alignment: .leading)
                        Text("Why").frame(width: 240, alignment: .leading)
                    }
                    .font(.caption2.monospaced().weight(.bold))
                    .foregroundStyle(.secondary)

                    Divider()

                    ForEach(rows, id: \.model) { row in
                        let hitPct = row.cacheShare.map { String(format: "%.0f%%", $0 * 100.0) } ?? "--"
                        let outcomesText = formatOutcomes(row.outcomes)
                        let whyText = formatCauses(row.causes)
                        HStack(spacing: 12) {
                            Text(row.model)
                                .lineLimit(1)
                                .frame(width: 180, alignment: .leading)
                            Text("\(row.requests)")
                                .frame(width: 48, alignment: .trailing)
                            Text(hitPct)
                                .frame(width: 64, alignment: .trailing)
                            Text(formatTokenCount(row.cachedTokens))
                                .frame(width: 84, alignment: .trailing)
                            Text(outcomesText)
                                .lineLimit(1)
                                .frame(width: 190, alignment: .leading)
                            Text(whyText)
                                .lineLimit(2)
                                .frame(width: 240, alignment: .leading)
                        }
                        .font(.caption2.monospaced())
                    }
                }
                .padding(8)
            }
            .background(Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(contrastStyle.subtleBorderColor(.secondary), lineWidth: contrastStyle.borderWidth)
            )
        }
    }

    private func formatOutcomes(_ bag: [String: Int]) -> String {
        guard !bag.isEmpty else { return "--" }
        var ordered: [String] = []
        for k in Self.outcomeOrder {
            if let v = bag[k] {
                ordered.append("\(k) \(v)")
            }
        }
        for k in bag.keys.sorted() where !Self.outcomeOrder.contains(k) {
            if let v = bag[k] {
                ordered.append("\(k) \(v)")
            }
        }
        return ordered.isEmpty ? "--" : ordered.joined(separator: " · ")
    }

    private func formatCauses(_ bag: [String: Int]) -> String {
        guard !bag.isEmpty else { return "--" }
        let items = bag
            .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .map { "\(CacheReport.causeWords($0.key)) \($0.value)" }
        return items.joined(separator: " · ")
    }
}

private struct RecentTrendsTableView: View {
    let rows: [SQLiteStore.HourlyTrendEntry]
    let contrastStyle: AdaptiveContrastStyle

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Recent Trends (Last \(rows.count) Hourly Buckets)")
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 10) {
                        Text("Hour").frame(width: 120, alignment: .leading)
                        Text("Resp ms").frame(width: 68, alignment: .trailing)
                        Text("Tok/s").frame(width: 56, alignment: .trailing)
                        Text("Reqs").frame(width: 44, alignment: .trailing)
                        Text("Errs").frame(width: 40, alignment: .trailing)
                        Text("Cache %").frame(width: 64, alignment: .trailing)
                        Text("Draft acc %").frame(width: 78, alignment: .trailing)
                        Text("Draft share %").frame(width: 88, alignment: .trailing)
                    }
                    .font(.caption2.monospaced().weight(.bold))
                    .foregroundStyle(.secondary)

                    Divider()

                    ForEach(rows, id: \.hour) { row in
                        HStack(spacing: 10) {
                            Text(formatHourString(row.hour))
                                .frame(width: 120, alignment: .leading)
                            Text(String(format: "%.1f", row.responseTimeMs))
                                .frame(width: 68, alignment: .trailing)
                            Text(String(format: "%.1f", row.tokensPerSecond))
                                .frame(width: 56, alignment: .trailing)
                            Text("\(row.requests)")
                                .frame(width: 44, alignment: .trailing)
                            Text("\(row.errors)")
                                .foregroundStyle(row.errors > 0 ? .red : .primary)
                                .frame(width: 40, alignment: .trailing)
                            Text(row.cacheShare.map { String(format: "%.0f%%", $0 * 100.0) } ?? "--")
                                .frame(width: 64, alignment: .trailing)
                            Text(row.draftAcceptance.map { String(format: "%.0f%%", $0 * 100.0) } ?? "--")
                                .frame(width: 78, alignment: .trailing)
                            Text(row.draftShare.map { String(format: "%.0f%%", $0 * 100.0) } ?? "--")
                                .frame(width: 88, alignment: .trailing)
                        }
                        .font(.caption2.monospaced())
                    }
                }
                .padding(8)
            }
            .background(Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(contrastStyle.subtleBorderColor(.secondary), lineWidth: contrastStyle.borderWidth)
            )
        }
    }

    private func formatHourString(_ iso: String) -> String {
        let replaced = iso.replacingOccurrences(of: "T", with: " ")
        return String(replaced.prefix(16))
    }
}

private struct DiskJSONLLogTableView: View {
    let stream: ObservabilityStream
    let records: [[String: JSONValue]]
    let contrastStyle: AdaptiveContrastStyle

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                ForEach(Array(records.enumerated()), id: \.offset) { _, rec in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(clockString(from: rec))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .frame(width: 64, alignment: .leading)
                        Text(eventTypeString(from: rec))
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 130, alignment: .leading)
                            .lineLimit(1)
                        Text(summarizeRecord(rec))
                            .font(.system(size: 10.5, design: .monospaced))
                            .lineLimit(2)
                        Spacer(minLength: 0)
                    }
                    .padding(.vertical, 2)
                }
            }
            .padding(6)
        }
        .frame(maxHeight: 220)
        .background(Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(contrastStyle.subtleBorderColor(.secondary), lineWidth: contrastStyle.borderWidth)
        )
    }

    private func clockString(from rec: [String: JSONValue]) -> String {
        if case .string(let iso)? = rec["iso"] {
            if let tIdx = iso.firstIndex(of: "T") {
                return String(iso[iso.index(after: tIdx)...].prefix(8))
            }
        }
        if case .double(let ts)? = rec["ts"] {
            let df = DateFormatter()
            df.dateFormat = "HH:mm:ss"
            return df.string(from: Date(timeIntervalSince1970: ts))
        }
        return "--:--:--"
    }

    private func eventTypeString(from rec: [String: JSONValue]) -> String {
        if case .string(let s)? = rec["event_type"] { return s }
        if case .string(let s)? = rec["type"] { return s }
        return stream.rawValue
    }

    private func summarizeRecord(_ rec: [String: JSONValue]) -> String {
        var parts: [String] = []
        if case .string(let m)? = rec["model_id"] ?? rec["model"] { parts.append(m) }
        if case .string(let ep)? = rec["endpoint"] { parts.append(ep) }
        if let code = scalarNumberString(rec["status_code"]) { parts.append("HTTP \(code)") }
        if let tps = scalarDouble(rec["tokens_per_second"]) { parts.append(String(format: "%.1f tok/s", tps)) }
        if let ms = scalarDouble(rec["total_ms"]) { parts.append(String(format: "%.0f ms", ms)) }
        if let ram = scalarDouble(rec["system_ram_used_gb"]) { parts.append(String(format: "RAM %.1f GB", ram)) }
        if case .string(let err)? = rec["error_type"] ?? rec["error"] { parts.append(err) }
        if !parts.isEmpty {
            return parts.joined(separator: " · ")
        }
        let fallback = rec
            .filter { !["ts", "iso", "type", "source"].contains($0.key) }
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\(formatJSONScalar($0.value))" }
        return fallback.isEmpty ? "{}" : fallback.joined(separator: " ")
    }

    private func scalarNumberString(_ val: JSONValue?) -> String? {
        switch val {
        case .int(let i)?: return String(i)
        case .double(let d)?: return String(Int(d))
        default: return nil
        }
    }

    private func scalarDouble(_ val: JSONValue?) -> Double? {
        switch val {
        case .double(let d)?: return d
        case .int(let i)?: return Double(i)
        default: return nil
        }
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

// MARK: - Shared Formatting & Control Parsing Helpers

private struct ControlParseError: Error {
    let message: String
}

private func parseControlValue(field: ConfigFieldDescriptor, raw: String) -> Result<JSONValue, ControlParseError> {
    if raw.isEmpty {
        return .success(.null)
    }
    switch field.type {
    case .integer:
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return .success(.null) }
        guard let n = Int(trimmed) else {
            return .failure(ControlParseError(message: "\(field.name): not a whole number"))
        }
        return .success(.int(n))
    case .number:
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return .success(.null) }
        guard let n = Double(trimmed), n.isFinite else {
            return .failure(ControlParseError(message: "\(field.name): not a number"))
        }
        return .success(.double(n))
    case .boolean:
        return .success(.bool(raw == "true"))
    case .array:
        if field.itemType == .number || field.itemType == .integer {
            let tokens = raw
                .components(separatedBy: CharacterSet(charactersIn: "\n,"))
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if tokens.isEmpty {
                return .success(.null)
            }
            if field.itemType == .integer {
                var ints: [JSONValue] = []
                for tok in tokens {
                    guard let v = Int(tok) else {
                        return .failure(ControlParseError(message: "\(field.name): \"\(tok)\" is not a valid integer"))
                    }
                    ints.append(.int(v))
                }
                return .success(.array(ints))
            }
            var nums: [JSONValue] = []
            for tok in tokens {
                guard let v = Double(tok), v.isFinite else {
                    return .failure(ControlParseError(message: "\(field.name): \"\(tok)\" is not a valid number"))
                }
                nums.append(.double(v))
            }
            return .success(.array(nums))
        }
        // String arrays (e.g. extra_args, tags, stop_sequences): strictly one element per line
        let lines = raw
            .components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if lines.isEmpty {
            return .success(.null)
        }
        return .success(.array(lines.map { .string($0) }))
    case .string, .object:
        return .success(.string(raw))
    }
}

private func fieldControlString(for field: ConfigFieldDescriptor, value: JSONValue) -> String {
    if value == .null { return "" }
    if field.type == .array, case .array(let items) = value {
        return items.map(formatJSONScalar).joined(separator: "\n")
    }
    return formatJSONScalar(value)
}

private func formatJSONScalar(_ value: JSONValue) -> String {
    switch value {
    case .null:
        return "—"
    case .bool(let b):
        return b ? "true" : "false"
    case .int(let i):
        return String(i)
    case .double(let d):
        if d.rounded() == d && abs(d) < Double(Int.max) {
            return String(Int(d))
        }
        return String(d)
    case .string(let s):
        if s.count == 64 && s.allSatisfy(\.isHexDigit) {
            return String(s.prefix(12))
        }
        return s
    case .array(let arr):
        return arr.isEmpty ? "—" : arr.map(formatJSONScalar).joined(separator: ", ")
    case .object(let obj):
        if let data = try? JSONEncoder().encode(JSONValue.object(obj)),
           let s = String(data: data, encoding: .utf8) {
            return s
        }
        return "{...}"
    }
}

private func formatTokenCount(_ count: Int) -> String {
    if count >= 1_048_576 {
        let m = Double(count) / 1_048_576.0
        return count % 1_048_576 == 0 ? "\(Int(m))M" : String(format: "%.1fM", m)
    }
    if count >= 1024 {
        return "\(Int((Double(count) / 1024.0).rounded()))k"
    }
    return "\(count)"
}

private func formatGiB(_ value: Double) -> String {
    String(format: "%.1f GiB", value)
}

private func normalizeEOL(_ text: String) -> String {
    text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
}

private func copyToPasteboard(_ text: String) {
    #if os(macOS)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
    #elseif canImport(UIKit)
    UIPasteboard.general.string = text
    #endif
}
