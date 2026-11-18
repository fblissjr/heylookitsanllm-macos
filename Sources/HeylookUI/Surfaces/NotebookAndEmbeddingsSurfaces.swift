import AVFoundation
import AVKit
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

// MARK: - 2. Notebook Surface

struct NotebookSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @Environment(\.scenePhase) private var scenePhase
    @State private var showNotebookSystemPrompt: Bool = false

    private var wordCount: Int {
        viewModel.notebookBody
            .split { $0.isWhitespace || $0.isNewline }
            .count
    }

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        let promptState = viewModel.notebookPresets.notebookPromptChipState
        let models = viewModel.notebookPresets.availableGenerationModels
        let activeNbModelID = viewModel.notebookPresets.activeNotebookModelID

        VStack(alignment: .leading, spacing: 10) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    notebookTitleAndStatusCluster(contrastStyle: contrastStyle)
                    notebookModelChip(models: models, activeModelID: activeNbModelID, contrastStyle: contrastStyle)
                    notebookPresetMenu(contrastStyle: contrastStyle)
                    notebookSystemPromptChip(state: promptState, contrastStyle: contrastStyle)
                    Spacer(minLength: 8)
                    notebookActionCluster
                }

                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        notebookTitleAndStatusCluster(contrastStyle: contrastStyle)
                        Spacer(minLength: 4)
                        notebookPrimaryCompactAction
                    }

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            notebookModelChip(models: models, activeModelID: activeNbModelID, contrastStyle: contrastStyle)
                            notebookPresetMenu(contrastStyle: contrastStyle)
                            notebookSystemPromptChip(state: promptState, contrastStyle: contrastStyle)
                            notebookCompactOverflowActions
                        }
                    }
                }
            }

            if showNotebookSystemPrompt {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label(
                            promptState.chipLabel,
                            systemImage: promptState.isActiveOrModified ? "terminal.fill" : "terminal"
                        )
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        Spacer()
                        if !viewModel.notebookSystemPrompt.isEmpty {
                            Button("Clear") {
                                viewModel.notebookSystemPrompt = ""
                                viewModel.notebookSettings.flushPendingPromptSave()
                            }
                            .font(.caption2)
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                        }
                        Button("Done") {
                            viewModel.notebookSettings.flushPendingPromptSave()
                            withAnimation(.easeInOut(duration: 0.15)) {
                                showNotebookSystemPrompt = false
                            }
                        }
                        .font(.caption2.weight(.medium))
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                    TextEditor(text: $viewModel.notebookSystemPrompt)
                        .font(.system(size: 12, design: .monospaced))
                        .frame(minHeight: 62, maxHeight: 120)
                        .padding(6)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(contrastStyle.chipFillOpacity * 0.65))
                        )
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .strokeBorder(contrastStyle.borderColor(.secondary), lineWidth: contrastStyle.isIncreased ? 1 : 0.5)
                        }
                }
                .padding(8)
                .background(
                    Color.secondary.opacity(contrastStyle.chipFillOpacity * 0.45),
                    in: RoundedRectangle(cornerRadius: 10)
                )
            }

            ZStack(alignment: .topLeading) {
                CursorAwareTextEditor(
                    text: $viewModel.notebookBody,
                    selectedRange: $viewModel.notebookSelectedRange,
                    isEditable: !viewModel.isNotebookGenerating,
                    onUserEdit: {
                        viewModel.noteNotebookUserEdited()
                    },
                    onCommitBlur: {
                        viewModel.flushNotebookSaveIfDirty()
                    }
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if viewModel.notebookBody.isEmpty && !viewModel.isNotebookGenerating {
                    Text("Start writing, or click Continue at Cursor to generate into an empty notebook…")
                        .font(.system(size: 13.5, design: .monospaced))
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 16)
                        .allowsHitTesting(false)
                }
            }
            .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        viewModel.isNotebookGenerating
                            ? Color.accentColor.opacity(0.55)
                            : contrastStyle.borderColor(.secondary),
                        lineWidth: contrastStyle.isIncreased ? 1.2 : 1
                    )
            }

            HStack {
                let cursorSummary = viewModel.hasNotebookSelection
                    ? "Sel \(viewModel.notebookSelectedRange.length) chars at \(viewModel.notebookSelectedRange.location)"
                    : "Cursor \(viewModel.notebookSelectedRange.location)"
                Text("\(wordCount) words • \(viewModel.notebookBody.count) chars • \(cursorSummary)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                Spacer()
                if let activeModel = viewModel.activeModelDescription {
                    Text("Model: \(activeModel.modelID) (\(activeModel.computePlacement.value.label))")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .padding(horizontalSizeClass == .compact ? 10 : 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: scenePhase) { _, phase in
            if phase == .background || phase == .inactive {
                viewModel.flushNotebookSaveIfDirty()
            }
        }
        .onChange(of: showNotebookSystemPrompt) { _, isVisible in
            if !isVisible {
                viewModel.notebookSettings.flushPendingPromptSave()
            }
        }
        .onDisappear {
            viewModel.flushNotebookSaveIfDirty()
        }
    }

    @ViewBuilder
    private func notebookTitleAndStatusCluster(contrastStyle: AdaptiveContrastStyle) -> some View {
        TextField("Notebook Title", text: $viewModel.notebookTitle)
            .font(.title3.weight(.semibold))
            .textFieldStyle(.roundedBorder)
            .frame(minWidth: 120, maxWidth: horizontalSizeClass == .compact ? 220 : 280)

        Text(viewModel.isNotebookDirty ? "Saving…" : "Saved")
            .font(.caption2.monospaced())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                (viewModel.isNotebookDirty ? Color.orange : Color.secondary)
                    .opacity(contrastStyle.chipFillOpacity * 0.65),
                in: Capsule()
            )
            .foregroundStyle(viewModel.isNotebookDirty ? .orange : .secondary)
    }

    @ViewBuilder
    private func notebookModelChip(
        models: [EngineDescription],
        activeModelID: String,
        contrastStyle: AdaptiveContrastStyle
    ) -> some View {
        Menu {
            if models.isEmpty {
                Text("No models discovered")
            } else {
                ForEach(models, id: \.modelID) { model in
                    Button {
                        viewModel.notebookPresets.selectNotebookModel(model.modelID)
                    } label: {
                        HStack {
                            Text(model.modelID)
                            if model.modelID == activeModelID {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "cpu")
                    .font(.system(size: 10))
                Text(activeModelID.isEmpty ? "No Model" : activeModelID)
                    .font(.caption2.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 140, alignment: .leading)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(Color.secondary.opacity(contrastStyle.chipFillOpacity))
            )
            .overlay {
                Capsule().strokeBorder(contrastStyle.borderColor(.secondary), lineWidth: contrastStyle.isIncreased ? 1 : 0)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    @ViewBuilder
    private func notebookPresetMenu(contrastStyle: AdaptiveContrastStyle) -> some View {
        Menu {
            Button {
                viewModel.clearPresetSelection()
            } label: {
                HStack {
                    Text("Model Defaults (No Preset)")
                    if viewModel.selectedPresetName == nil {
                        Image(systemName: "checkmark")
                    }
                }
            }
            Divider()
            ForEach(viewModel.presets) { preset in
                Button {
                    viewModel.applyPreset(preset)
                } label: {
                    HStack {
                        Text(preset.name)
                        if viewModel.selectedPresetName == preset.name {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }
            Divider()
            Button {
                viewModel.activeSurface = .presets
            } label: {
                Label("Manage Presets…", systemImage: "slider.horizontal.3")
                    .labelStyle(.titleAndIcon)
            }
        } label: {
            let labelText: String = {
                if let name = viewModel.selectedPresetName {
                    return viewModel.isPresetModified ? "\(name) (modified)" : name
                }
                return viewModel.hasAnyManualSamplerOverride ? "Custom Samplers" : "Model Defaults"
            }()
            HStack(spacing: 4) {
                Image(systemName: "dial.medium")
                    .font(.system(size: 10))
                Text(labelText)
                    .font(.caption2.weight(.medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(
                    (viewModel.selectedPresetName != nil ? Color.indigo : Color.secondary)
                        .opacity(contrastStyle.chipFillOpacity)
                )
            )
            .overlay {
                Capsule().strokeBorder(
                    contrastStyle.borderColor(viewModel.selectedPresetName != nil ? .indigo : .secondary),
                    lineWidth: contrastStyle.isIncreased ? 1 : 0
                )
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    @ViewBuilder
    private func notebookSystemPromptChip(
        state: NotebookPromptChipState,
        contrastStyle: AdaptiveContrastStyle
    ) -> some View {
        let tintColor: Color = switch state {
        case .empty: .secondary
        case .custom: .accentColor
        case .presetUnmodified: .teal
        case .presetModified: .orange
        }
        Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                showNotebookSystemPrompt.toggle()
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: state.isActiveOrModified ? "terminal.fill" : "terminal")
                Text(state.chipLabel)
                    .lineLimit(1)
            }
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(tintColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
            .overlay {
                Capsule().strokeBorder(contrastStyle.borderColor(tintColor), lineWidth: contrastStyle.isIncreased ? 1 : 0)
            }
        }
        .buttonStyle(.plain)
        .foregroundStyle(state == .empty ? .secondary : tintColor)
    }

    @ViewBuilder
    private var notebookPrimaryCompactAction: some View {
        if viewModel.isNotebookGenerating {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Button(role: .destructive) {
                    viewModel.cancelNotebookGeneration()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.small)
            }
        } else {
            Button {
                Task { await viewModel.continueNotebookGeneration() }
            } label: {
                Label("Continue", systemImage: "sparkles")
            }
            .buttonStyle(.glassProminent)
            .controlSize(.small)
        }
    }

    private var notebookCompactOverflowActions: some View {
        HStack(spacing: 6) {
            Button {
                viewModel.saveNotebook()
            } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button {
                Task { await viewModel.summarizeNotebook() }
            } label: {
                Label("Summarize", systemImage: "list.bullet.rectangle")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.isNotebookGenerating || viewModel.notebookBody.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

            Button {
                Task { await viewModel.analyzeNotebookSelection() }
            } label: {
                Label("Analyze", systemImage: "wand.and.stars")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.isNotebookGenerating || !viewModel.hasNotebookSelection)

            Button {
                viewModel.saveNotebookToConversations()
            } label: {
                Label("Send to Chat", systemImage: "bubble.left.and.bubble.right")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    private var notebookActionCluster: some View {
        HStack(spacing: 8) {
            if viewModel.isNotebookGenerating {
                ProgressView()
                    .controlSize(.small)
            }

            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    Button {
                        Task { await viewModel.summarizeNotebook() }
                    } label: {
                        Label("Insert Summary", systemImage: "list.bullet.rectangle")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)
                    .disabled(viewModel.isNotebookGenerating || viewModel.notebookBody.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button {
                        Task { await viewModel.analyzeNotebookSelection() }
                    } label: {
                        Label("Analyze Selection", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)
                    .disabled(viewModel.isNotebookGenerating || !viewModel.hasNotebookSelection)
                    .help(viewModel.hasNotebookSelection ? "Critique highlighted selection" : "Highlight a passage in the editor to enable Analyze Selection")

                    if viewModel.isNotebookGenerating {
                        Button(role: .destructive) {
                            viewModel.cancelNotebookGeneration()
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                        .controlSize(.small)
                        .help("Stop inline generation and keep partial text (Esc)")
                    } else {
                        Button {
                            Task { await viewModel.continueNotebookGeneration() }
                        } label: {
                            Label("Continue at Cursor", systemImage: "sparkles")
                        }
                        .buttonStyle(.glassProminent)
                        .controlSize(.small)
                        .keyboardShortcut(.return, modifiers: [.command])
                    }

                    Button {
                        viewModel.saveNotebook()
                    } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)
                    .keyboardShortcut("s", modifiers: [.command])

                    Button {
                        viewModel.saveNotebookToConversations()
                    } label: {
                        Label("Send to Chat", systemImage: "bubble.left.and.bubble.right")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.small)
                }
            }
        }
    }
}

// MARK: - 3. Embeddings & ANE Media Finder Studio Surface

struct EmbeddingsStudioSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @FocusState private var isSearchFocused: Bool
    @State private var showIndexingImporter = false

    private let orderedModes: [EmbeddingsSubMode] = [.compareMatrix, .evalLab, .search]

    private func modeTitle(for mode: EmbeddingsSubMode) -> String {
        switch mode {
        case .compareMatrix: return "Compare"
        case .evalLab: return "Embedding Lab"
        case .search: return "Media Corpus (mbed)"
        }
    }

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        VStack(spacing: 0) {
            // Top Mode Switcher Bar (Compare | Embedding Lab | Media Corpus (mbed))
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    modePicker
                    Spacer()
                    engineStatusBadge(contrastStyle: contrastStyle)
                }
                VStack(alignment: .leading, spacing: 6) {
                    modePicker
                    engineStatusBadge(contrastStyle: contrastStyle)
                }
            }
            .padding(.horizontal, horizontalSizeClass == .compact ? 10 : 14)
            .padding(.vertical, 8)
            .background(.bar)

            Divider()

            switch viewModel.embeddingsSubMode {
            case .search:
                EmbeddingsSearchHeaderBar(
                    viewModel: viewModel,
                    isSearchFocused: $isSearchFocused
                )

                Divider()

                EmbeddingsStatusToolbar(
                    viewModel: viewModel,
                    onAddFilesOrFolders: triggerIndexingPicker
                )

                Divider()

                // Results List or Empty State
                if viewModel.searchHits.isEmpty {
                    ContentUnavailableView {
                        Label("No Matching Artifacts", systemImage: "doc.text.magnifyingglass")
                    } description: {
                        Text("Type a search query above, attach an image or audio clip, or add local files and folders to your corpus.")
                    } actions: {
                        Button {
                            triggerIndexingPicker()
                        } label: {
                            Label("Add Files or Folders…", systemImage: "plus.rectangle.on.folder")
                        }
                        .buttonStyle(.glassProminent)
                        .controlSize(.small)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(selection: $viewModel.selectedHit) {
                        ForEach(viewModel.searchHits) { hit in
                            HitRowView(hit: hit)
                                .tag(hit)
                        }
                    }
                    .listStyle(.inset)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onChange(of: viewModel.selectedHit) { _, newHit in
                        if newHit != nil {
                            viewModel.showInspector = true
                        }
                    }
                }

                if viewModel.showEvalLabDrawer {
                    Divider()
                    EmbeddingsEvalLabDrawer(viewModel: viewModel)
                }

            case .compareMatrix:
                EmbeddingsCompareMatrixView(viewModel: viewModel)

            case .evalLab:
                EmbeddingsMultiConfigLabView(viewModel: viewModel)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            if viewModel.embeddingsSubMode == .search {
                isSearchFocused = true
            }
            Task { await viewModel.embeddings.refreshEmbeddingModels() }
        }
        .fileImporter(
            isPresented: $showIndexingImporter,
            allowedContentTypes: [.item, .folder],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                viewModel.startIndexing(urls)
            }
        }
    }

    private var modePicker: some View {
        Picker("Embeddings Mode", selection: $viewModel.embeddingsSubMode) {
            ForEach(orderedModes) { mode in
                Label(modeTitle(for: mode), systemImage: mode.icon)
                    .tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: horizontalSizeClass == .compact ? .infinity : 480)
    }

    @ViewBuilder
    private func engineStatusBadge(contrastStyle: AdaptiveContrastStyle) -> some View {
        if viewModel.isActiveEmbeddingEngineSynthetic && viewModel.embeddings.selectedBackend == .nativeMLX {
            HStack(spacing: 5) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text("No real embedding checkpoint loaded — synthetic fallback blocked")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.orange)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Capsule().fill(Color.orange.opacity(contrastStyle.bannerFillOpacity)))
            .overlay {
                Capsule().strokeBorder(contrastStyle.borderColor(.orange), lineWidth: contrastStyle.isIncreased ? 1 : 0)
            }
        } else {
            HStack(spacing: 5) {
                Circle()
                    .fill(Color.green)
                    .frame(width: 7, height: 7)
                Text(
                    viewModel.embeddings.selectedEmbeddingModelID.isEmpty
                        ? viewModel.activeQueryTowersLabel
                        : "\(viewModel.embeddings.selectedEmbeddingModelID) • \(viewModel.activeQueryTowersLabel)"
                )
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            }
        }
    }

    private func triggerIndexingPicker() {
        showIndexingImporter = true
    }
}

private struct EmbeddingsSearchHeaderBar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    var isSearchFocused: FocusState<Bool>.Binding
    @State private var showQueryMediaImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .font(.system(size: 15, weight: .semibold))

                TextField("Search across text, code, PDFs, images, audio & video…", text: $viewModel.embeddingQuery)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 14))
                    .focused(isSearchFocused)
                    .defaultFocus(isSearchFocused, true)
                    .onSubmit {
                        Task { await viewModel.runEmbeddingSearch() }
                    }
                    .onChange(of: viewModel.embeddingQuery) { _, _ in
                        viewModel.scheduleEmbeddingSearchDebounced()
                    }

                Button {
                    showQueryMediaImporter = true
                } label: {
                    Label("Attach Media", systemImage: "paperclip")
                }
                .buttonStyle(.glass)
                .controlSize(.small)
                .help("Attach an image or audio clip for multimodal query search")

                Button("Search") {
                    Task { await viewModel.runEmbeddingSearch() }
                }
                .buttonStyle(.glassProminent)
                .controlSize(.small)

                if !viewModel.embeddingQuery.isEmpty || viewModel.queryAttachmentURL != nil {
                    Button {
                        viewModel.embeddingQuery = ""
                        viewModel.queryAttachmentURL = nil
                        Task { await viewModel.runEmbeddingSearch() }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .frame(minWidth: 28, minHeight: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Clear search query")
                }
            }

            if let attachment = viewModel.queryAttachmentURL {
                let kind = ArtifactKind.detect(from: attachment) ?? .image
                HStack(spacing: 6) {
                    Image(systemName: kind.systemImage)
                        .font(.caption)
                        .foregroundStyle(.tint)
                    Text("Query Media: \(attachment.lastPathComponent)")
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                    Button {
                        viewModel.queryAttachmentURL = nil
                        Task { await viewModel.runEmbeddingSearch() }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(minWidth: 28, minHeight: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Capsule().fill(Color.accentColor.opacity(0.14)))
            }

            ScoringPipelineBanner(
                dimension: viewModel.selectedDimension,
                task: viewModel.effectiveQueryTask,
                hasQueryMedia: viewModel.queryAttachmentURL != nil,
                includeSoundtrack: viewModel.loadAudioFromVideo,
                videoSampling: viewModel.selectedSampling,
                towersLabel: viewModel.activeQueryTowersLabel
            )
        }
        .padding(14)
        .background(.bar)
        .fileImporter(
            isPresented: $showQueryMediaImporter,
            allowedContentTypes: [.image, .audiovisualContent, .audio],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                viewModel.queryAttachmentURL = url
                Task { await viewModel.runEmbeddingSearch() }
            }
        }
    }
}

private struct EmbeddingsStatusToolbar: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let onAddFilesOrFolders: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            if viewModel.isIndexing || viewModel.isEmbeddingBusy {
                ProgressView()
                    .controlSize(.small)
            }
            Text(viewModel.embeddingStatusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Spacer()

            if viewModel.isIndexing {
                Button(role: .destructive) {
                    viewModel.cancelIndexing()
                } label: {
                    Label("Stop Indexing", systemImage: "stop.circle.fill")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }

            Button {
                onAddFilesOrFolders()
            } label: {
                Label("Add Files/Folders…", systemImage: "plus.rectangle.on.folder")
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)

            Toggle(isOn: $viewModel.showEvalLabDrawer) {
                Label("Eval Lab", systemImage: "chart.bar.xaxis")
            }
            .toggleStyle(.button)
            .controlSize(.mini)

            Text("\(viewModel.searchHits.count) results • \(viewModel.artifactCount) indexed")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.05))
    }
}

private struct EmbeddingsEvalLabDrawer: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Quick Pairwise Compare & Embedding Lab", systemImage: "chart.bar.xaxis")
                    .font(.caption.weight(.semibold))
                Spacer()
                if viewModel.isRunningEvalLab {
                    Button(role: .destructive) {
                        viewModel.embeddings.cancelActiveEmbeddingRun()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .controlSize(.mini)
                } else {
                    Button {
                        Task { await viewModel.embeddings.runMultimodalEmbeddingLab() }
                    } label: {
                        Label("Run Embedding Lab", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
                }

                ForEach(viewModel.embeddings.labSweepResults) { run in
                    Text(
                        String(
                            format: "%dd R@1: %.0f%% (MRR %.2f)",
                            run.config.dimensions,
                            run.recallAt1 * 100,
                            run.mrr
                        )
                    )
                    .font(.caption2.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.secondary.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                }
            }

            HStack(spacing: 10) {
                TextField("Left text", text: $viewModel.compareLeftText)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                TextField("Right text", text: $viewModel.compareRightText)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                Button("Compare") {
                    viewModel.runPairwiseCompare()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    Task { await viewModel.promoteComparePairToRetrievalLab() }
                } label: {
                    Label("Send to Embedding Lab", systemImage: "arrow.right.square")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                if let score = viewModel.compareCosineScore {
                    Text(String(format: "%dd Cosine: %.4f", viewModel.selectedDimension, score))
                        .font(.caption.monospaced().weight(.bold))
                        .foregroundStyle(.tint)
                }
            }
        }
        .padding(12)
        .background(Color.secondary.opacity(0.04))
    }
}

// MARK: - Embeddings Contextual Sidebar Sections (Rendered in Unified Sidebar List)

struct EmbeddingsSidebarSections: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var showIndexingImporter = false

    var body: some View {
        Section("Modality Filter") {
            ModalityFilterRow(
                title: "All Artifacts",
                icon: "square.grid.2x2",
                count: viewModel.artifactCount,
                isSelected: viewModel.selectedKindFilter == nil
            ) {
                viewModel.selectedKindFilter = nil
                Task { await viewModel.runEmbeddingSearch() }
            }
            ForEach(ArtifactKind.allCases) { kind in
                ModalityFilterRow(
                    title: kind.displayName,
                    icon: kind.systemImage,
                    count: viewModel.countsByKind[kind] ?? 0,
                    isSelected: viewModel.selectedKindFilter == kind
                ) {
                    viewModel.selectedKindFilter = kind
                    Task { await viewModel.runEmbeddingSearch() }
                }
            }
        }

        Section("Index & Watched Folders") {
            Button {
                showIndexingImporter = true
            } label: {
                Label("Add Files or Folders…", systemImage: "plus.rectangle.on.folder")
                    .fontWeight(.medium)
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)
            .fileImporter(
                isPresented: $showIndexingImporter,
                allowedContentTypes: [.item, .folder],
                allowsMultipleSelection: true
            ) { result in
                if case .success(let urls) = result {
                    viewModel.startIndexing(urls)
                }
            }

            if !viewModel.watchedFolders.isEmpty {
                Button {
                    viewModel.reindexWatchedFolders()
                } label: {
                    Label("Re-scan Watched Folders", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.plain)
                .disabled(viewModel.isIndexing)
            }

            Button(role: .destructive) {
                viewModel.pendingConfirmation = .clearEmbeddingIndex
            } label: {
                Label("Clear Index…", systemImage: "trash")
            }
            .buttonStyle(.plain)
            .disabled(viewModel.artifactCount == 0 || viewModel.isIndexing)

            ForEach(viewModel.watchedFolders) { folder in
                HStack(spacing: 6) {
                    Image(systemName: "folder.fill")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(folder.url.lastPathComponent)
                            .font(.caption)
                            .lineLimit(1)
                        Text(folder.path)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    Button {
                        viewModel.removeWatchedFolder(folder)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(minWidth: 28, minHeight: 28)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Stop watching \(folder.path)")
                }
                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                    Button(role: .destructive) {
                        viewModel.removeWatchedFolder(folder)
                    } label: {
                        Label("Remove", systemImage: "trash")
                    }
                }
            }
        }

        Section("MLX EmbeddingGemma 2") {
            Picker("Engine", selection: $viewModel.selectedBackend) {
                ForEach(EngineBackend.allCases, id: \.self) { engine in
                    Text(engine.label).tag(engine)
                }
            }
            .onChange(of: viewModel.selectedBackend) { _, _ in
                viewModel.setupEmbeddingCoordinator()
            }

            Picker("Matryoshka Dim", selection: $viewModel.selectedDimension) {
                Text("768d (Full)").tag(768)
                Text("512d").tag(512)
                Text("256d (Fast)").tag(256)
                Text("128d (Compact)").tag(128)
            }
            .onChange(of: viewModel.selectedDimension) { _, _ in
                Task { await viewModel.runEmbeddingSearch() }
            }

            Picker("Query Task", selection: $viewModel.boundEffectiveQueryTask) {
                Text("Retrieval Query").tag(EmbeddingTask.query)
                Text("Semantic Similarity").tag(EmbeddingTask.sts)
                Text("Code Retrieval").tag(EmbeddingTask.codeRetrieval)
                Text("Question Answering").tag(EmbeddingTask.questionAnswering)
                Text("Fact Verification").tag(EmbeddingTask.factVerification)
            }

            Picker("Video Sampling", selection: $viewModel.selectedSampling) {
                ForEach(VideoWindowSampling.allCases, id: \.self) { mode in
                    Text(mode.label).tag(mode)
                }
            }

            Toggle("Include Video Soundtrack", isOn: $viewModel.loadAudioFromVideo)

            if viewModel.selectedBackend == .heylook {
                TextField("Server URL", text: $viewModel.embeddingServerURLString)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .onSubmit {
                        viewModel.setupEmbeddingCoordinator()
                    }
            } else {
                TextField("Checkpoint Path", text: $viewModel.checkpointPath)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .onSubmit {
                        viewModel.setupEmbeddingCoordinator()
                    }
            }
        }

        Section("Compute Placement (ANE / GPU)") {
            Picker("Preset", selection: $viewModel.boundComputePreset) {
                ForEach(ComputePreset.allCases, id: \.self) { preset in
                    Text(preset.label).tag(preset)
                }
            }

            ForEach(EmbeddingComponent.allCases) { component in
                VStack(alignment: .leading, spacing: 3) {
                    Picker(
                        component.title,
                        selection: $viewModel[computeUnitFor: component]
                    ) {
                        ForEach(TowerComputeUnit.allCases, id: \.self) { unit in
                            Text(unit.shortBadge).tag(unit)
                        }
                    }

                    if let report = viewModel.aneInspectionReports.first(where: { $0.component == component }) {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(report.isCompiledBundlePresent ? Color.green : Color.orange)
                                .frame(width: 6, height: 6)
                            Text(
                                report.isCompiledBundlePresent
                                    ? "CoreML Ready (\(Int((report.aneResidencyFraction * 100).rounded()))% ANE ops)"
                                    : "MLX GPU fallback (no .mlmodelc)"
                            )
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        }
                    }
                }
                .padding(.vertical, 2)
            }

            ComputeMemorySplitBar(split: viewModel.modelComputeOptions.memorySplitMB)
                .padding(.vertical, 4)
        }
    }
}

private struct ModalityFilterRow: View {
    let title: String
    let icon: String
    let count: Int
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button {
            onSelect()
        } label: {
            HStack {
                Label(title, systemImage: icon)
                    .fontWeight(isSelected ? .semibold : .regular)
                Spacer()
                Text("\(count)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .listRowBackground(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.16) : Color.clear)
        )
    }
}

private struct ScoringPipelineBanner: View {
    let dimension: Int
    let task: EmbeddingTask
    let hasQueryMedia: Bool
    let includeSoundtrack: Bool
    let videoSampling: VideoWindowSampling
    let towersLabel: String

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 6) {
                    FeaturePill(
                        icon: "cpu",
                        label: towersLabel,
                        tint: .purple
                    )
                    FeaturePill(
                        icon: "ruler",
                        label: "Matryoshka \(dimension)d L2 Cosine",
                        tint: .blue
                    )
                    FeaturePill(
                        icon: "arrow.triangle.branch",
                        label: task == .query ? "Asymmetric Query→Doc" : "Symmetric Similarity",
                        tint: .indigo
                    )
                    FeaturePill(
                        icon: "film.stack",
                        label: "Video: \(videoSampling.label)",
                        tint: .orange
                    )
                    if includeSoundtrack {
                        FeaturePill(
                            icon: "waveform.and.magnifyingglass",
                            label: "Dual Video+Soundtrack Spans",
                            tint: .pink
                        )
                    }
                    if hasQueryMedia {
                        FeaturePill(
                            icon: "photo.on.rectangle.angled",
                            label: "Cross-Modal Media Query",
                            tint: .green
                        )
                    }
                }
            }
        }
    }
}

private struct ComputeMemorySplitBar: View {
    let split: (aneMB: Int, gpuMB: Int, cpuMB: Int)

    var body: some View {
        let total = max(1, split.aneMB + split.gpuMB + split.cpuMB)
        let aneFrac = CGFloat(split.aneMB) / CGFloat(total)
        let gpuFrac = CGFloat(split.gpuMB) / CGFloat(total)

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Est. Residency Split")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("ANE ~\(split.aneMB)MB • GPU ~\(split.gpuMB)MB")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            GeometryReader { geo in
                HStack(spacing: 1) {
                    if split.aneMB > 0 {
                        Rectangle()
                            .fill(Color.green.gradient)
                            .frame(width: max(4, geo.size.width * aneFrac))
                    }
                    if split.gpuMB > 0 {
                        Rectangle()
                            .fill(Color.purple.gradient)
                            .frame(width: max(4, geo.size.width * gpuFrac))
                    }
                    if split.cpuMB > 0 {
                        Rectangle()
                            .fill(Color.blue.gradient)
                    }
                }
                .clipShape(Capsule())
            }
            .frame(height: 6)
        }
    }
}

private struct FeaturePill: View {
    let icon: String
    let label: String
    let tint: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .semibold))
            Text(label)
                .font(.system(size: 10, weight: .medium))
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .foregroundStyle(tint)
        .background(
            Capsule()
                .fill(tint.opacity(0.12))
        )
    }
}

private struct HitRowView: View {
    let hit: SearchHit

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ThumbnailBadge(artifact: hit.artifact)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(hit.artifact.title)
                        .font(.system(size: 14, weight: .semibold))
                        .lineLimit(1)

                    if let spanBadge = hit.bestSpan.displayBadge {
                        Text(spanBadge)
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.accentColor.opacity(0.15)))
                            .foregroundStyle(Color.accentColor)
                    }

                    Text(hit.bestScoredFeature.title)
                        .font(.system(size: 10, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.14)))
                        .foregroundStyle(.secondary)

                    if hit.hasMultiFeatureMatch {
                        Text("+ Audio & Vision")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.pink.opacity(0.15)))
                            .foregroundStyle(.pink)
                    }

                    Spacer()
                    ScoreBadgeView(score: hit.bestScore)
                }

                if let preview = hit.previewText, !preview.isEmpty {
                    Text(preview.replacingOccurrences(of: "\n", with: " "))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                HStack(spacing: 6) {
                    Text(hit.artifact.kind.displayName)
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.secondary.opacity(0.15)))

                    if hit.spanMatches.count > 1 {
                        Text("\(hit.spanMatches.count) spans matched")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    Text(hit.artifact.url.path)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .padding(.vertical, 6)
    }
}

private struct ScoreBadgeView: View {
    let score: Float

    var body: some View {
        let clamped = max(0, min(1, score))
        let color: Color = clamped > 0.55 ? .green : (clamped > 0.35 ? .blue : .secondary)
        return Text(String(format: "%.3f", score))
            .font(.caption2.monospacedDigit().weight(.semibold))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.16)))
            .foregroundStyle(color)
    }
}

private struct ThumbnailBadge: View {
    let artifact: Artifact

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.12))
                .frame(width: 46, height: 46)

            if artifact.kind == .image {
                AsyncImage(url: artifact.url) { phase in
                    switch phase {
                    case .success(let image):
                        image
                            .resizable()
                            .scaledToFill()
                            .frame(width: 46, height: 46)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    default:
                        Image(systemName: artifact.kind.systemImage)
                            .font(.system(size: 20))
                            .foregroundStyle(.tint)
                    }
                }
            } else {
                Image(systemName: artifact.kind.systemImage)
                    .font(.system(size: 20))
                    .foregroundStyle(.tint)
            }
        }
    }
}

private struct KeyframeStamp: Identifiable, Equatable {
    let index: Int
    let timeSeconds: Double
    var id: String { "\(index)-\(timeSeconds)" }
}

// MARK: - Embeddings Artifact & Matryoshka Inspector (Rendered in Native .inspector)

struct EmbeddingsInspectorView: View {
    let hit: SearchHit?
    let selectedDimension: Int
    var onSelectDimension: ((Int) -> Void)? = nil
    var onSendToCompare: ((SearchHit) -> Void)? = nil
    var onPinToLab: ((SearchHit) -> Void)? = nil
    @State private var player: AVPlayer?
    @State private var selectedChunkID: String?

    private var activeSpanMatch: SearchHit.SpanMatch? {
        guard let hit else { return nil }
        if let id = selectedChunkID, let match = hit.spanMatches.first(where: { $0.chunkID == id }) {
            return match
        }
        return hit.spanMatches.first
    }

    private var activeSpan: ArtifactSpan? {
        activeSpanMatch?.span ?? hit?.bestSpan
    }

    var body: some View {
        if let hit {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    HStack(spacing: 12) {
                        ThumbnailBadge(artifact: hit.artifact)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(hit.artifact.title)
                                .font(.headline)
                            HStack(spacing: 6) {
                                Text(hit.artifact.kind.displayName)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                ScoreBadgeView(score: hit.bestScore)
                            }
                        }
                        Spacer()
                    }

                    // Compare-to-Lab Bridge Actions
                    if onSendToCompare != nil || onPinToLab != nil {
                        HStack(spacing: 8) {
                            if let onSendToCompare {
                                Button {
                                    onSendToCompare(hit)
                                } label: {
                                    Label("Compare in Lab", systemImage: "arrow.left.and.right.square")
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                            if let onPinToLab {
                                Button {
                                    onPinToLab(hit)
                                } label: {
                                    Label("Pin to Lab", systemImage: "chart.bar.doc.horizontal")
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                            Spacer()
                        }
                    }

                    Divider()

                    // Transparent Feature & Score Attribution Card
                    if let match = activeSpanMatch {
                        ScoredFeatureBreakdownCard(
                            artifact: hit.artifact,
                            spanMatch: match,
                            selectedDimension: selectedDimension,
                            onSelectDimension: onSelectDimension
                        )
                    }

                    // Preview by Modality
                    switch hit.artifact.kind {
                    case .image:
                        AsyncImage(url: hit.artifact.url) { phase in
                            switch phase {
                            case .success(let image):
                                image
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 280)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            case .failure:
                                Label("Unable to load image preview", systemImage: "photo.badge.exclamationmark")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            default:
                                ProgressView()
                                    .controlSize(.small)
                                    .frame(maxWidth: .infinity, minHeight: 120)
                            }
                        }

                    case .video, .audio:
                        VStack(alignment: .leading, spacing: 8) {
                            if let player {
                                #if os(macOS)
                                MacVideoPlayerView(player: player)
                                    .frame(height: hit.artifact.kind == .video ? 240 : 70)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                #elseif canImport(UIKit)
                                IOSVideoPlayerView(player: player)
                                    .frame(height: hit.artifact.kind == .video ? 240 : 70)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                #endif
                            }

                            if let span = activeSpan, !span.frameTimes.isEmpty {
                                let keyframes = span.frameTimes.enumerated().map {
                                    KeyframeStamp(index: $0.offset, timeSeconds: $0.element)
                                }
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("EXTRACTED KEYFRAME TIMESTAMPS (CLICK TO SEEK)")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 6) {
                                            ForEach(keyframes) { stamp in
                                                Button {
                                                    seekAndPlay(url: hit.artifact.url, startTime: stamp.timeSeconds)
                                                } label: {
                                                    HStack(spacing: 3) {
                                                        Image(systemName: "camera.viewfinder")
                                                            .font(.system(size: 9))
                                                        Text("#\(stamp.index + 1) \(ArtifactSpan.formatTimestampMillis(stamp.timeSeconds))")
                                                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                                                    }
                                                    .padding(.horizontal, 7)
                                                    .padding(.vertical, 3)
                                                    .background(Capsule().fill(Color.orange.opacity(0.16)))
                                                    .foregroundStyle(.orange)
                                                }
                                                .buttonStyle(.plain)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        .onAppear {
                            seekAndPlay(url: hit.artifact.url, startTime: activeSpan?.startTimeSeconds)
                        }
                        .onChange(of: hit.artifact.id) { _, _ in
                            selectedChunkID = nil
                            seekAndPlay(url: hit.artifact.url, startTime: hit.bestSpan.startTimeSeconds)
                        }

                    case .text, .code, .document:
                        if let snippet = activeSpanMatch?.previewText ?? hit.previewText {
                            ScrollView {
                                Text(snippet)
                                    .font(.system(size: 12, design: hit.artifact.kind == .code ? .monospaced : .default))
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(10)
                            }
                            .frame(maxHeight: 240)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.secondary.opacity(0.08))
                            )
                        }
                    }

                    // Matching Spans Timeline / Pages / Lines
                    if !hit.spanMatches.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("SCORED SPANS & FEATURES (\(hit.spanMatches.count))")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text("Click span to inspect / seek")
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }

                            ForEach(hit.spanMatches.prefix(8)) { spanMatch in
                                let isSelected = (selectedChunkID ?? hit.spanMatches.first?.chunkID) == spanMatch.chunkID
                                let feature = spanMatch.scoredFeature(for: hit.artifact)
                                Button {
                                    selectedChunkID = spanMatch.chunkID
                                    if let startTime = spanMatch.span.startTimeSeconds {
                                        seekAndPlay(url: hit.artifact.url, startTime: startTime)
                                    }
                                } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Image(systemName: feature.systemImage)
                                                .font(.caption)
                                                .foregroundStyle(isSelected ? .white : Color.accentColor)
                                            Text(spanMatch.span.displayBadge ?? "Full Artifact")
                                                .font(.caption.monospacedDigit().weight(.medium))
                                                .foregroundStyle(isSelected ? .white : .primary)
                                            Text(feature.title)
                                                .font(.system(size: 10))
                                                .padding(.horizontal, 5)
                                                .padding(.vertical, 1)
                                                .background(
                                                    Capsule().fill(
                                                        isSelected ? Color.white.opacity(0.2) : Color.secondary.opacity(0.15)
                                                    )
                                                )
                                                .foregroundStyle(isSelected ? .white : .secondary)
                                            Spacer()
                                            Text(String(format: "%.3f", spanMatch.score))
                                                .font(.caption2.monospacedDigit().weight(.bold))
                                                .foregroundStyle(isSelected ? .white : .secondary)
                                        }

                                        if let snippet = spanMatch.previewText, !snippet.isEmpty {
                                            Text(snippet.replacingOccurrences(of: "\n", with: " "))
                                                .font(.caption2)
                                                .foregroundStyle(isSelected ? .white.opacity(0.85) : .secondary)
                                                .lineLimit(1)
                                        }
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 7)
                                    .background(
                                        RoundedRectangle(cornerRadius: 7)
                                            .fill(isSelected ? Color.accentColor : Color.secondary.opacity(0.08))
                                    )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    Divider()

                    // Metadata
                    let sortedAttributes = hit.artifact.attributes.sorted { $0.key < $1.key }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("METADATA")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        LabeledContent("Path", value: hit.artifact.url.path)
                            .font(.caption)
                        LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: hit.artifact.fingerprint.byteCount, countStyle: .file))
                            .font(.caption)

                        ForEach(sortedAttributes, id: \.key) { key, value in
                            LabeledContent(key.capitalized, value: value)
                                .font(.caption)
                        }
                    }

                    #if os(macOS)
                    HStack {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([hit.artifact.url])
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                        }

                        Button {
                            NSWorkspace.shared.open(hit.artifact.url)
                        } label: {
                            Label("Open File", systemImage: "arrow.up.forward.app")
                        }
                    }
                    .padding(.top, 6)
                    #endif
                }
                .padding(16)
            }
        } else {
            ContentUnavailableView(
                "Select an Artifact",
                systemImage: "sidebar.right",
                description: Text("Select a search result to inspect multimodal previews, Matryoshka cosine attribution, and matching timestamps or pages.")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func seekAndPlay(url: URL, startTime: Double?) {
        let avPlayer: AVPlayer
        if let existing = player,
           let asset = existing.currentItem?.asset as? AVURLAsset,
           asset.url == url {
            avPlayer = existing
        } else {
            avPlayer = AVPlayer(url: url)
            player = avPlayer
        }
        if let startTime, startTime > 0 {
            let target = CMTime(seconds: startTime, preferredTimescale: 600)
            avPlayer.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        }
    }
}

private struct ScoredFeatureBreakdownCard: View {
    let artifact: Artifact
    let spanMatch: SearchHit.SpanMatch
    let selectedDimension: Int
    var onSelectDimension: ((Int) -> Void)? = nil

    var body: some View {
        let feature = spanMatch.scoredFeature(for: artifact)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("WHY THIS MATCHED (EMBEDDING ATTRIBUTION)", systemImage: "sparkles.rectangle.stack")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(feature.subtitle)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }

            HStack(spacing: 8) {
                Text(feature.title)
                    .font(.caption.weight(.semibold))
                if let badge = spanMatch.span.displayBadge {
                    Text("• \(badge)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Matryoshka Cosine Across Subspaces (Click to Switch)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                ForEach([128, 256, 512, 768], id: \.self) { dim in
                    if let dimScore = spanMatch.scoresByDimension[dim] {
                        Button {
                            onSelectDimension?(dim)
                        } label: {
                            HStack(spacing: 8) {
                                Text("\(dim)d")
                                    .font(.system(size: 10, weight: dim == selectedDimension ? .bold : .regular, design: .monospaced))
                                    .frame(width: 34, alignment: .leading)
                                    .foregroundStyle(dim == selectedDimension ? Color.accentColor : .secondary)

                                GeometryReader { geo in
                                    ZStack(alignment: .leading) {
                                        Capsule()
                                            .fill(Color.secondary.opacity(0.15))
                                        Capsule()
                                            .fill(dim == selectedDimension ? Color.accentColor : Color.blue.opacity(0.6))
                                            .frame(width: max(4, geo.size.width * CGFloat(max(0, min(1, dimScore)))))
                                    }
                                }
                                .frame(height: 6)

                                Text(String(format: "%.3f", dimScore))
                                    .font(.system(size: 10, weight: dim == selectedDimension ? .bold : .regular, design: .monospaced))
                                    .frame(width: 42, alignment: .trailing)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(Color.accentColor.opacity(0.06))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 9)
                .strokeBorder(Color.accentColor.opacity(0.18), lineWidth: 1)
        }
    }
}

#if os(macOS)
private struct MacVideoPlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        view.player = player
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}
#elseif canImport(UIKit)
private struct IOSVideoPlayerView: UIViewControllerRepresentable {
    let player: AVPlayer

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        controller.player = player
        controller.showsPlaybackControls = true
        return controller
    }

    func updateUIViewController(_ uiViewController: AVPlayerViewController, context: Context) {
        if uiViewController.player !== player {
            uiViewController.player = player
        }
    }
}
#endif

// MARK: - N×N Multimodal Cosine Similarity Matrix View (`≤16` Items)

private struct JSONExportDocument: FileDocument {
    static let readableContentTypes: [UTType] = [.json]
    let data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        self.data = configuration.file.regularFileContents ?? Data("{}".utf8)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

private struct EmbeddingsCompareMatrixView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    @State private var showMediaImporter: Bool = false
    @State private var pendingBlockTargetItemID: String?
    @State private var showCompareExporter: Bool = false

    private var embeddings: EmbeddingsStudioController { viewModel.embeddings }

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                compareControlsHeader(contrastStyle: contrastStyle)

                if let preflightErr = embeddings.comparePreflightError {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(preflightErr)
                            .font(.caption)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(contrastStyle.borderColor(.orange), lineWidth: contrastStyle.isIncreased ? 1 : 0.5)
                    }
                }

                // Ordered Multi-Block Staging List (`≤16` Items)
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        ViewThatFits(in: .horizontal) {
                            HStack {
                                Text("COMPARE ITEMS (\(embeddings.compareRichItems.count) / \(HeylookWorkbenchViewModel.maxCompareMatrixItems))")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                stagingActionButtons
                            }
                            VStack(alignment: .leading, spacing: 6) {
                                Text("COMPARE ITEMS (\(embeddings.compareRichItems.count) / \(HeylookWorkbenchViewModel.maxCompareMatrixItems))")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                stagingActionButtons
                            }
                        }

                        ForEach(Array(embeddings.compareRichItems.enumerated()), id: \.element.id) { idx, item in
                            CompareRichItemEditorCard(
                                index: idx,
                                item: bindingForCompareItem(id: item.id),
                                totalItems: embeddings.compareRichItems.count,
                                promptNames: embeddings.availableEmbeddingPromptNames,
                                supportsVideoWindows: embeddings.supportsVideoWindowingForSelectedModel,
                                maxWindowSeconds: embeddings.activeEmbeddingFacts.videoWindows?.maxSeconds ?? 32,
                                contrastStyle: contrastStyle,
                                onMoveItem: { delta in embeddings.moveCompareItem(id: item.id, delta: delta) },
                                onRemoveItem: { embeddings.removeCompareItem(id: item.id) },
                                onAddTextBlock: { embeddings.addBlockToCompareItem(itemID: item.id, kind: .text) },
                                onAddMediaBlock: {
                                    pendingBlockTargetItemID = item.id
                                    showMediaImporter = true
                                },
                                onMoveBlock: { blockID, delta in
                                    embeddings.moveBlockInCompareItem(itemID: item.id, blockID: blockID, delta: delta)
                                },
                                onRemoveBlock: { blockID in
                                    embeddings.removeBlockFromCompareItem(itemID: item.id, blockID: blockID)
                                },
                                onPromptChanged: { promptName in
                                    embeddings.setCompareItemPrompt(itemID: item.id, promptName: promptName)
                                }
                            )
                        }
                    }
                    .padding(6)
                }

                if !embeddings.compareMatrixScores.isEmpty,
                   embeddings.compareMatrixScores.count == embeddings.compareRichItems.count {
                    compareRankingsSection(contrastStyle: contrastStyle)

                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 16) {
                            CompareMatrixHeatmapGridView(viewModel: viewModel)
                            CompareSelectedPairProfileCard(
                                viewModel: viewModel,
                                contrastStyle: contrastStyle,
                                onCopyText: copyToPasteboard,
                                onExportJSON: { showCompareExporter = true }
                            )
                        }
                        VStack(alignment: .leading, spacing: 14) {
                            CompareMatrixHeatmapGridView(viewModel: viewModel)
                            CompareSelectedPairProfileCard(
                                viewModel: viewModel,
                                contrastStyle: contrastStyle,
                                onCopyText: copyToPasteboard,
                                onExportJSON: { showCompareExporter = true }
                            )
                        }
                    }
                } else {
                    ContentUnavailableView(
                        "Matrix Not Computed Yet",
                        systemImage: "square.grid.3x3",
                        description: Text("Configure 2–16 ordered multi-block items above (with optional task prompts and video windowing) and click Compute Matrix.")
                    )
                    .frame(maxWidth: .infinity, minHeight: 160)
                }
            }
            .padding(horizontalSizeClass == .compact ? 10 : 16)
        }
        .fileImporter(
            isPresented: $showMediaImporter,
            allowedContentTypes: [.image, .audiovisualContent, .audio, .pdf, .plainText, .sourceCode, .item],
            allowsMultipleSelection: pendingBlockTargetItemID == nil
        ) { result in
            defer { pendingBlockTargetItemID = nil }
            guard case .success(let urls) = result else { return }
            if let targetItemID = pendingBlockTargetItemID, let first = urls.first {
                embeddings.addMediaBlockToCompareItem(itemID: targetItemID, url: first)
            } else {
                embeddings.addCompareMediaFiles(urls)
            }
        }
        .fileExporter(
            isPresented: $showCompareExporter,
            document: JSONExportDocument(data: embeddings.compareMatrixExportData()),
            contentType: .json,
            defaultFilename: "heylook-compare-matrix.json"
        ) { _ in }
        .sheet(
            item: Binding(
                get: { embeddings.activeComparePreviewWindow },
                set: { embeddings.activeComparePreviewWindow = $0 }
            )
        ) { windowHit in
            CompareVideoWindowPreviewSheet(hit: windowHit, contrastStyle: contrastStyle)
        }
    }

    @ViewBuilder
    private func compareControlsHeader(contrastStyle: AdaptiveContrastStyle) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                compareHeaderTitle
                Spacer()
                compareHeaderControls(contrastStyle: contrastStyle)
            }
            VStack(alignment: .leading, spacing: 8) {
                compareHeaderTitle
                ScrollView(.horizontal, showsIndicators: false) {
                    compareHeaderControls(contrastStyle: contrastStyle)
                }
            }
        }
    }

    private var compareHeaderTitle: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Multimodal Compare Matrix")
                .font(.headline)
            Text("Ordered text & media blocks, dynamic task prompts, video windowing, and raw vector inspection")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func compareHeaderControls(contrastStyle: AdaptiveContrastStyle) -> some View {
        HStack(spacing: 8) {
            if !embeddings.availableEmbeddingModels.isEmpty {
                Picker(
                    "Model",
                    selection: Binding(
                        get: { embeddings.selectedEmbeddingModelID },
                        set: { embeddings.selectEmbeddingModel($0) }
                    )
                ) {
                    ForEach(embeddings.availableEmbeddingModels) { entry in
                        Text(entry.label).tag(entry.id)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 210)
            }

            let dims = embeddings.availableEmbeddingDimensions
            Picker("Dim", selection: $viewModel.compareMatrixDimension) {
                ForEach(dims, id: \.self) { d in
                    Text("\(d)d").tag(d)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 220)

            Toggle(
                viewModel.compareAsymmetricMode ? "Asymmetric" : "Per-Item / STS",
                isOn: $viewModel.compareAsymmetricMode
            )
            .toggleStyle(.button)
            .controlSize(.small)

            Button {
                Task { await viewModel.promoteComparePairToRetrievalLab() }
            } label: {
                Label("Send to Embedding Lab", systemImage: "arrow.right.square")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(embeddings.compareRichItems.count < 2)

            Button {
                showCompareExporter = true
            } label: {
                Label("Export JSON", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(embeddings.compareMatrixScores.count < 2)

            if viewModel.isEmbeddingBusy {
                Button(role: .destructive) {
                    embeddings.cancelActiveEmbeddingRun()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.small)
            } else {
                Button {
                    Task { await viewModel.runCompareMatrix() }
                } label: {
                    Label("Compute Matrix", systemImage: "square.grid.3x3.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(embeddings.compareRichItems.count < 2)
            }
        }
    }

    private var stagingActionButtons: some View {
        HStack(spacing: 8) {
            Button {
                embeddings.addCompareTextItem()
            } label: {
                Label("Add Text Item", systemImage: "plus.bubble")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(embeddings.compareRichItems.count >= HeylookWorkbenchViewModel.maxCompareMatrixItems)

            Button {
                pendingBlockTargetItemID = nil
                showMediaImporter = true
            } label: {
                Label("Add Media Item…", systemImage: "plus.rectangle.on.folder")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(embeddings.compareRichItems.count >= HeylookWorkbenchViewModel.maxCompareMatrixItems)
        }
    }

    @ViewBuilder
    private func compareRankingsSection(contrastStyle: AdaptiveContrastStyle) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 12) {
                closestItemsRankCard(contrastStyle: contrastStyle)
                if !embeddings.compareVideoWindowHits.isEmpty {
                    closestVideoWindowsCard(contrastStyle: contrastStyle)
                }
            }
            VStack(alignment: .leading, spacing: 12) {
                closestItemsRankCard(contrastStyle: contrastStyle)
                if !embeddings.compareVideoWindowHits.isEmpty {
                    closestVideoWindowsCard(contrastStyle: contrastStyle)
                }
            }
        }
    }

    private func closestItemsRankCard(contrastStyle: AdaptiveContrastStyle) -> some View {
        GroupBox("Closest Items to Item 1") {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(embeddings.compareRankedMatchesToItem1.enumerated()), id: \.element.index) { rankIdx, match in
                    Button {
                        viewModel.selectedCompareCellRow = 0
                        viewModel.selectedCompareCellCol = match.index
                    } label: {
                        HStack(spacing: 8) {
                            Text("#\(rankIdx + 1)")
                                .font(.caption2.monospaced().weight(.bold))
                                .foregroundStyle(.secondary)
                            Text(match.label)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Spacer()
                            Text(String(format: "%.4f", match.score))
                                .font(.caption.monospaced().weight(.semibold))
                                .foregroundStyle(match.score > 0.65 ? Color.teal : Color.primary)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(
                            Color.secondary.opacity(contrastStyle.chipFillOpacity * 0.65),
                            in: RoundedRectangle(cornerRadius: 6)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(4)
        }
    }

    private func closestVideoWindowsCard(contrastStyle: AdaptiveContrastStyle) -> some View {
        GroupBox("Closest video windows to Item 1") {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(embeddings.compareVideoWindowHits.prefix(8).enumerated()), id: \.element.id) { idx, hit in
                    Button {
                        embeddings.activeComparePreviewWindow = hit
                    } label: {
                        HStack(spacing: 8) {
                            Text("#\(idx + 1)")
                                .font(.caption2.monospaced().weight(.bold))
                                .foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(hit.itemLabel)
                                    .font(.caption.weight(.medium))
                                    .lineLimit(1)
                                Text(String(format: "%.1fs – %.1fs • %d frame(s)", hit.start, hit.end, hit.frameTimes.count))
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(String(format: "%.4f", hit.score))
                                .font(.caption.monospaced().weight(.semibold))
                                .foregroundStyle(.tint)
                            Image(systemName: "play.circle.fill")
                                .foregroundStyle(.tint)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(
                            Color.accentColor.opacity(contrastStyle.chipFillOpacity * 0.5),
                            in: RoundedRectangle(cornerRadius: 6)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(4)
        }
    }

    private func bindingForCompareItem(id: String) -> Binding<CompareRichItem> {
        Binding(
            get: {
                embeddings.compareRichItems.first(where: { $0.id == id })
                    ?? CompareRichItem(id: id, label: "Item", blocks: [CompareBlock(kind: .text)])
            },
            set: { updated in
                if let idx = embeddings.compareRichItems.firstIndex(where: { $0.id == id }) {
                    embeddings.compareRichItems[idx] = updated
                    embeddings.comparePreflightError = embeddings.validateCompareItemsForLimits()
                }
            }
        )
    }

    private func copyToPasteboard(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = text
        #endif
    }
}

private struct CompareRichItemEditorCard: View {
    let index: Int
    @Binding var item: CompareRichItem
    let totalItems: Int
    let promptNames: [String]
    let supportsVideoWindows: Bool
    let maxWindowSeconds: Double
    let contrastStyle: AdaptiveContrastStyle

    let onMoveItem: (Int) -> Void
    let onRemoveItem: () -> Void
    let onAddTextBlock: () -> Void
    let onAddMediaBlock: () -> Void
    let onMoveBlock: (String, Int) -> Void
    let onRemoveBlock: (String) -> Void
    let onPromptChanged: (String?) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    itemHeaderFields
                    Spacer()
                    itemReorderAndDeleteButtons
                }
                VStack(alignment: .leading, spacing: 6) {
                    itemHeaderFields
                    itemReorderAndDeleteButtons
                }
            }

            ForEach(Array(item.blocks.enumerated()), id: \.element.id) { blockIdx, block in
                CompareBlockRowEditor(
                    block: bindingForBlock(id: block.id),
                    blockIndex: blockIdx,
                    blockCount: item.blocks.count,
                    supportsVideoWindows: supportsVideoWindows,
                    maxWindowSeconds: maxWindowSeconds,
                    contrastStyle: contrastStyle,
                    onMoveUp: { onMoveBlock(block.id, -1) },
                    onMoveDown: { onMoveBlock(block.id, 1) },
                    onRemove: { onRemoveBlock(block.id) }
                )
            }

            HStack(spacing: 8) {
                Button(action: onAddTextBlock) {
                    Label("+ Text", systemImage: "text.badge.plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)

                Button(action: onAddMediaBlock) {
                    Label("+ Media", systemImage: "photo.badge.plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)

                Spacer()
                Text("\(item.blocks.count) block(s)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(Color.secondary.opacity(contrastStyle.chipFillOpacity * 0.4), in: RoundedRectangle(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(contrastStyle.borderColor(.secondary), lineWidth: contrastStyle.isIncreased ? 1 : 0.6)
        }
    }

    @ViewBuilder
    private var itemHeaderFields: some View {
        HStack(spacing: 8) {
            Text("#\(index + 1)")
                .font(.caption.monospaced().weight(.bold))
                .foregroundStyle(.secondary)

            TextField("Item Label", text: $item.label)
                .textFieldStyle(.roundedBorder)
                .font(.caption.weight(.semibold))
                .frame(minWidth: 100, maxWidth: 180)

            Picker(
                "Task",
                selection: Binding(
                    get: { item.promptName ?? "" },
                    set: { val in onPromptChanged(val.isEmpty ? nil : val) }
                )
            ) {
                Text("Task: None").tag("")
                ForEach(promptNames, id: \.self) { name in
                    Text("Task: \(name)").tag(name)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 200)
        }
    }

    private var itemReorderAndDeleteButtons: some View {
        HStack(spacing: 4) {
            Button { onMoveItem(-1) } label: {
                Image(systemName: "arrow.up")
                    .frame(minWidth: 28, minHeight: 28)
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
            .disabled(index == 0)

            Button { onMoveItem(1) } label: {
                Image(systemName: "arrow.down")
                    .frame(minWidth: 28, minHeight: 28)
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
            .disabled(index >= totalItems - 1)

            Button(role: .destructive, action: onRemoveItem) {
                Image(systemName: "xmark")
                    .frame(minWidth: 28, minHeight: 28)
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
            .disabled(totalItems <= 2)
        }
    }

    private func bindingForBlock(id: String) -> Binding<CompareBlock> {
        Binding(
            get: { item.blocks.first(where: { $0.id == id }) ?? CompareBlock(kind: .text) },
            set: { updated in
                if let idx = item.blocks.firstIndex(where: { $0.id == id }) {
                    item.blocks[idx] = updated
                }
            }
        )
    }
}

private struct CompareBlockRowEditor: View {
    @Binding var block: CompareBlock
    let blockIndex: Int
    let blockCount: Int
    let supportsVideoWindows: Bool
    let maxWindowSeconds: Double
    let contrastStyle: AdaptiveContrastStyle
    let onMoveUp: () -> Void
    let onMoveDown: () -> Void
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                switch block.kind {
                case .text:
                    Image(systemName: "text.alignleft")
                        .foregroundStyle(.secondary)
                    TextField("Enter text block…", text: $block.text, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        .lineLimit(1...4)
                case .media:
                    Image(systemName: (block.mediaKind ?? .document).systemImage)
                        .foregroundStyle(.tint)
                    Text((block.mediaKind ?? .document).displayName)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                    Text(block.fileURL?.lastPathComponent ?? "No media attached")
                        .font(.caption.monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                }

                HStack(spacing: 3) {
                    Button(action: onMoveUp) {
                        Image(systemName: "chevron.up")
                            .font(.caption2)
                            .frame(minWidth: 28, minHeight: 28)
                    }
                    .buttonStyle(.plain)
                    .disabled(blockIndex == 0)

                    Button(action: onMoveDown) {
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                            .frame(minWidth: 28, minHeight: 28)
                    }
                    .buttonStyle(.plain)
                    .disabled(blockIndex >= blockCount - 1)

                    Button(role: .destructive, action: onRemove) {
                        Image(systemName: "trash")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(minWidth: 28, minHeight: 28)
                    }
                    .buttonStyle(.plain)
                }
            }

            if block.kind == .media && block.mediaKind == .video && supportsVideoWindows {
                HStack(spacing: 8) {
                    Text("Video mode:")
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(.secondary)

                    Picker("Windows", selection: $block.videoWindowMode) {
                        ForEach(CompareVideoWindowMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 210)

                    if block.videoWindowMode != .wholeVideo {
                        TextField("Window s (≤\(Int(maxWindowSeconds)))", text: $block.videoWindowSecondsText)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption2.monospaced())
                            .frame(maxWidth: 110)
                    }
                    Spacer()
                }
            }
        }
        .padding(6)
        .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct CompareVideoWindowPreviewSheet: View {
    let hit: CompareVideoWindowHit
    let contrastStyle: AdaptiveContrastStyle
    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer?
    @State private var currentSeconds: Double = 0

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(hit.itemLabel)
                            .font(.headline)
                        Text(
                            String(
                                format: "Window %.1fs – %.1fs • Similarity %.4f • Playhead %.1fs",
                                hit.start,
                                hit.end,
                                hit.score,
                                currentSeconds
                            )
                        )
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                    }
                    Spacer()
                }

                if let player {
                    Group {
                        #if os(macOS)
                        MacVideoPlayerView(player: player)
                        #elseif canImport(UIKit)
                        IOSVideoPlayerView(player: player)
                        #endif
                    }
                    .frame(minHeight: 220, maxHeight: 360)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                } else {
                    ContentUnavailableView(
                        "Video File Unavailable",
                        systemImage: "film.stack",
                        description: Text("The original local video URL is no longer accessible.")
                    )
                    .frame(height: 220)
                }

                if !hit.frameTimes.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Sampled frame_times (tap to seek):")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(Array(hit.frameTimes.enumerated()), id: \.offset) { idx, t in
                                    Button {
                                        seekAndPlay(to: t)
                                    } label: {
                                        Text(String(format: "#%d %.2fs", idx + 1, t))
                                            .font(.caption2.monospaced())
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(Color.accentColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                }
            }
            .padding(16)
            .navigationTitle("Video Window Preview")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 420)
        #endif
        .task(id: hit.id) {
            guard let url = hit.fileURL else { return }
            let avPlayer = AVPlayer(url: url)
            player = avPlayer
            currentSeconds = hit.start
            let startCM = CMTime(seconds: hit.start, preferredTimescale: 600)
            await avPlayer.seek(to: startCM, toleranceBefore: .zero, toleranceAfter: .zero)
            avPlayer.play()
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(120))
                let sec = avPlayer.currentTime().seconds
                if sec.isFinite {
                    currentSeconds = sec
                    if sec >= hit.end && avPlayer.rate > 0 {
                        avPlayer.pause()
                        await avPlayer.seek(to: startCM, toleranceBefore: .zero, toleranceAfter: .zero)
                        currentSeconds = hit.start
                    } else if sec < hit.start - 0.25 && avPlayer.rate > 0 {
                        await avPlayer.seek(to: startCM, toleranceBefore: .zero, toleranceAfter: .zero)
                    }
                }
            }
            avPlayer.pause()
        }
    }

    private func seekAndPlay(to seconds: Double) {
        guard let player else { return }
        let clamped = min(max(hit.start, seconds), hit.end)
        let target = CMTime(seconds: clamped, preferredTimescale: 600)
        Task {
            await player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
            currentSeconds = clamped
            player.play()
        }
    }
}

private struct CompareMatrixHeatmapGridView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        let items = viewModel.embeddings.compareRichItems
        let scores = viewModel.compareMatrixScores
        ScrollView([.horizontal, .vertical]) {
            Grid(horizontalSpacing: 4, verticalSpacing: 4) {
                GridRow {
                    Text(viewModel.compareAsymmetricMode ? "Row (Q) \\ Col (D)" : "Item \\ Item")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 108, alignment: .leading)

                    ForEach(0..<items.count, id: \.self) { cIdx in
                        let colItem = items[cIdx]
                        Text(colItem.label)
                            .font(.system(size: 10, weight: .semibold))
                            .lineLimit(1)
                            .frame(width: 66)
                            .help(colItem.primaryFileURL?.lastPathComponent ?? colItem.textSummary)
                            .foregroundStyle(cIdx == viewModel.selectedCompareCellCol ? Color.accentColor : .secondary)
                    }
                }

                ForEach(0..<items.count, id: \.self) { rIdx in
                    GridRow {
                        Text(items[rIdx].label)
                            .font(.system(size: 10, weight: .semibold))
                            .lineLimit(1)
                            .frame(width: 108, alignment: .leading)
                            .foregroundStyle(rIdx == viewModel.selectedCompareCellRow ? Color.accentColor : .primary)

                        ForEach(0..<items.count, id: \.self) { cIdx in
                            CompareMatrixCellButton(
                                score: scores[rIdx][cIdx],
                                isSelected: rIdx == viewModel.selectedCompareCellRow && cIdx == viewModel.selectedCompareCellCol,
                                onSelect: {
                                    viewModel.selectedCompareCellRow = rIdx
                                    viewModel.selectedCompareCellCol = cIdx
                                }
                            )
                        }
                    }
                }
            }
            .padding(8)
        }
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.06)))
    }
}

private struct CompareMatrixCellButton: View {
    let score: Float
    let isSelected: Bool
    let onSelect: () -> Void

    private var cellColor: Color {
        let clamped = Double(max(-0.2, min(1.0, score)))
        if clamped >= 0.75 {
            return Color.teal.opacity(0.88)
        } else if clamped >= 0.50 {
            return Color.blue.opacity(0.55)
        } else if clamped >= 0.25 {
            return Color.indigo.opacity(0.28)
        } else {
            return Color.secondary.opacity(0.14)
        }
    }

    var body: some View {
        Button(action: onSelect) {
            Text(String(format: "%.3f", score))
                .font(.system(size: 10.5, weight: isSelected ? .bold : .medium, design: .monospaced))
                .foregroundStyle(score > 0.65 ? .white : .primary)
                .frame(width: 66, height: 34)
                .background(RoundedRectangle(cornerRadius: 6).fill(cellColor))
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 6).strokeBorder(Color.primary, lineWidth: 2)
                    }
                }
        }
        .buttonStyle(.plain)
    }
}

private struct CompareSelectedPairProfileCard: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let contrastStyle: AdaptiveContrastStyle
    let onCopyText: (String) -> Void
    let onExportJSON: () -> Void

    var body: some View {
        let items = viewModel.embeddings.compareRichItems
        let count = max(1, items.count)
        let r = min(max(0, viewModel.selectedCompareCellRow), count - 1)
        let c = min(max(0, viewModel.selectedCompareCellCol), count - 1)
        let dimScores = viewModel.scoresAcrossDimensionsForCompareCell(row: r, col: c)
        let rowVec = viewModel.embeddings.inspectCompareVector(itemIndex: r, useColumnVector: false)
        let colVec = viewModel.embeddings.inspectCompareVector(itemIndex: c, useColumnVector: true)

        VStack(alignment: .leading, spacing: 10) {
            Text("CELL & RAW VECTOR INSPECTOR")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.secondary)

            Text("\(items[r].label) ↔ \(items[c].label)")
                .font(.subheadline.weight(.semibold))

            ForEach(dimScores, id: \.dim) { entry in
                HStack(spacing: 8) {
                    Text("\(entry.dim)d")
                        .font(.system(size: 10.5, weight: entry.dim == viewModel.compareMatrixDimension ? .bold : .regular, design: .monospaced))
                        .frame(width: 36, alignment: .leading)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.secondary.opacity(0.15))
                            Capsule()
                                .fill(entry.dim == viewModel.compareMatrixDimension ? Color.accentColor : Color.teal)
                                .frame(width: max(4, geo.size.width * CGFloat(max(0, min(1, entry.score)))))
                        }
                    }
                    .frame(height: 6)
                    Text(String(format: "%.4f", entry.score))
                        .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                        .frame(width: 52, alignment: .trailing)
                }
            }

            if let rowVec {
                Divider()
                rawVectorPreview(title: "Row #\(r + 1) (\(rowVec.label))", inspection: rowVec)
            }
            if let colVec, r != c || viewModel.compareAsymmetricMode {
                rawVectorPreview(title: "Col #\(c + 1) (\(colVec.label))", inspection: colVec)
            }

            Divider()

            Button {
                Task { await viewModel.sendComparePairToLab(row: r, col: c) }
            } label: {
                Label("Send Pair to Embedding Lab", systemImage: "chart.bar.doc.horizontal")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)

            HStack(spacing: 8) {
                Button {
                    onCopyText(viewModel.compareMatrixCSV())
                    viewModel.statusBanner = "Copied \(viewModel.compareMatrixDimension)d similarity matrix as CSV."
                } label: {
                    Label("Copy CSV", systemImage: "tablecells")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(action: onExportJSON) {
                    Label("Export JSON", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(12)
        .frame(maxWidth: 320, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(contrastStyle.chipFillOpacity * 0.5)))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(contrastStyle.borderColor(.secondary), lineWidth: contrastStyle.isIncreased ? 1 : 0.5)
        }
    }

    private func rawVectorPreview(title: String, inspection: CompareVectorInspection) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title)
                    .font(.caption2.weight(.semibold))
                    .lineLimit(1)
                Spacer()
                Text(String(format: "dim %d • ‖v‖=%.4f", inspection.dim, inspection.l2Norm))
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Text("[" + inspection.first16.map { String(format: "%.3f", $0) }.joined(separator: ", ") + ", …]")
                .font(.system(size: 9.5, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }
}

// MARK: - Multimodal Embedding Lab (`set.json` + 8-Case CoreGraphics Sample + Multi-Config Sweep)

private struct EmbeddingsMultiConfigLabView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast

    @State private var showSetJSONImporter: Bool = false
    @State private var showLabExporter: Bool = false
    @State private var draftModelID: String = ""
    @State private var draftDimensions: Int = Matryoshka.gemma2NativeDimensions
    @State private var draftQueryPrompt: String = ""
    @State private var draftDocPrompt: String = ""
    @State private var selectedRunID: String?

    private var embeddings: EmbeddingsStudioController { viewModel.embeddings }

    private var selectedRun: EmbeddingsLabRunResult? {
        if let id = selectedRunID, let match = embeddings.labSweepResults.first(where: { $0.id == id }) {
            return match
        }
        return embeddings.labSweepResults.first
    }

    var body: some View {
        let contrastStyle = AdaptiveContrastStyle(colorSchemeContrast)
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                labHeaderBar

                if let progress = embeddings.labProgressText {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(progress)
                            .font(.caption.monospaced())
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.accentColor.opacity(contrastStyle.bannerFillOpacity), in: RoundedRectangle(cornerRadius: 8))
                }

                // Multi-Config Queue Builder
                GroupBox("Config Sweep Queue (Model × Dimensions × Prompts)") {
                    VStack(alignment: .leading, spacing: 10) {
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 8) {
                                labConfigPickers
                                Spacer()
                                labConfigButtons
                            }
                            VStack(alignment: .leading, spacing: 8) {
                                labConfigPickers
                                labConfigButtons
                            }
                        }

                        #if os(macOS)
                        if embeddings.selectedBackend == .nativeMLX {
                            Toggle(
                                "macOS ANE Placement Benchmark (sweep All GPU / Hybrid ANE Towers / All ANE)",
                                isOn: Binding(
                                    get: { embeddings.labSweepANELanePresets },
                                    set: { embeddings.labSweepANELanePresets = $0 }
                                )
                            )
                            .font(.caption)
                            .toggleStyle(.switch)
                            .controlSize(.small)
                        }
                        #endif

                        if embeddings.labConfigQueue.isEmpty {
                            Text("Queue is empty — running evaluates the current model & dimensions, or tap '+ Add Config' / 'Add prompts on vs. off'.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 6) {
                                    ForEach(embeddings.labConfigQueue) { spec in
                                        HStack(spacing: 6) {
                                            Text(spec.label)
                                                .font(.caption2.monospaced())
                                            Button {
                                                embeddings.removeLabConfig(id: spec.id)
                                            } label: {
                                                Image(systemName: "xmark.circle.fill")
                                                    .foregroundStyle(.secondary)
                                                    .frame(minWidth: 28, minHeight: 28)
                                            }
                                            .buttonStyle(.plain)
                                        }
                                        .padding(.leading, 8)
                                        .padding(.trailing, 4)
                                        .padding(.vertical, 2)
                                        .background(Color.accentColor.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                                    }
                                    Button("Clear Queue") {
                                        embeddings.clearLabConfigQueue()
                                    }
                                    .font(.caption2)
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                }
                            }
                        }
                    }
                    .padding(6)
                }

                // Sweep Metrics Summary Table
                GroupBox("Evaluation Results (\(embeddings.labSweepResults.count) run(s))") {
                    if embeddings.labSweepResults.isEmpty {
                        Text("No Embedding Lab evaluation runs yet. Use the built-in 8-case multimodal sample or import a set.json folder, then click Run Evaluation.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(8)
                    } else {
                        ScrollView(.horizontal) {
                            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                                GridRow {
                                    Text("Config").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Recall@1").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Recall@3").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("MRR").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Mean Margin").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Worst Margin").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Latency").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                }
                                Divider()
                                ForEach(embeddings.labSweepResults) { run in
                                    let isSelected = selectedRun?.id == run.id
                                    GridRow {
                                        Button {
                                            selectedRunID = run.id
                                        } label: {
                                            HStack(spacing: 5) {
                                                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                                                    .font(.caption2)
                                                    .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                                                Text(run.config.label)
                                                    .font(.caption.weight(.medium))
                                            }
                                        }
                                        .buttonStyle(.plain)
                                        Text(String(format: "%.0f%%", run.recallAt1 * 100)).font(.caption.monospaced())
                                        Text(String(format: "%.0f%%", run.recallAt3 * 100)).font(.caption.monospaced())
                                        Text(String(format: "%.3f", run.mrr)).font(.caption.monospaced())
                                        Text(String(format: "%+.4f", run.meanMargin)).font(.caption.monospaced())
                                        Text(String(format: "%+.4f", run.worstMargin))
                                            .font(.caption.monospaced().weight(.semibold))
                                            .foregroundStyle(run.worstMargin > 0 ? Color.teal : Color.orange)
                                        Text("\(Int(run.elapsedMs.rounded())) ms")
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                        .padding(6)
                    }
                }

                // Per-Case Hit / Miss Breakdown
                if let activeRun = selectedRun {
                    GroupBox("Per-Case Breakdown — \(activeRun.config.label)") {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(activeRun.caseResults) { c in
                                HStack(alignment: .center, spacing: 8) {
                                    Text(c.hitAt1 ? "HIT" : "MISS")
                                        .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background((c.hitAt1 ? Color.green : Color.orange).opacity(contrastStyle.chipFillOpacity), in: Capsule())
                                        .foregroundStyle(c.hitAt1 ? .green : .orange)

                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("\(c.id): \(c.querySummary)")
                                            .font(.caption.weight(.medium))
                                            .lineLimit(1)
                                        if let beater = c.beatenByLabel {
                                            Text("beaten by: \(beater)")
                                                .font(.caption2.monospaced())
                                                .foregroundStyle(.orange)
                                        }
                                    }
                                    Spacer()
                                    Text(
                                        String(
                                            format: "r=%d • pos %.3f • neg %.3f • Δ %+.3f",
                                            c.rank,
                                            c.bestPositiveScore,
                                            c.bestNegativeScore,
                                            c.margin
                                        )
                                    )
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                }
                                .padding(.horizontal, 6)
                                .padding(.vertical, 4)
                                .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 6))
                            }
                        }
                        .padding(4)
                    }
                }

                // Active Set Preview (`set.json` / 8-Case Sample)
                GroupBox("Active Benchmark Set: \(embeddings.labSet.name) (\(embeddings.labSet.cases.count) cases)") {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(embeddings.labSet.cases) { bench in
                            HStack(spacing: 8) {
                                Text(bench.id)
                                    .font(.caption2.monospaced().weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .frame(minWidth: 96, alignment: .leading)
                                Text(bench.query.summaryLabel)
                                    .font(.caption)
                                    .lineLimit(1)
                                Spacer()
                                Text("+\(bench.positives.count) / -\(bench.negatives.count)")
                                    .font(.caption2.monospaced())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.secondary.opacity(contrastStyle.chipFillOpacity), in: Capsule())
                            }
                        }
                    }
                    .padding(6)
                }
            }
            .padding(horizontalSizeClass == .compact ? 10 : 16)
        }
        .onAppear {
            if draftModelID.isEmpty {
                draftModelID = embeddings.selectedEmbeddingModelID
            }
            draftDimensions = embeddings.compareMatrixDimension
        }
        .fileImporter(
            isPresented: $showSetJSONImporter,
            allowedContentTypes: [.json, .folder],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                embeddings.importBenchmarkSetJSON(from: url)
            }
        }
        .fileExporter(
            isPresented: $showLabExporter,
            document: JSONExportDocument(data: embeddings.embeddingLabExportData()),
            contentType: .json,
            defaultFilename: "heylook-embedding-lab.json"
        ) { _ in }
    }

    private var labHeaderBar: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) {
                labHeaderTitle
                Spacer()
                labHeaderActions
            }
            VStack(alignment: .leading, spacing: 8) {
                labHeaderTitle
                ScrollView(.horizontal, showsIndicators: false) {
                    labHeaderActions
                }
            }
        }
    }

    private var labHeaderTitle: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Embedding Lab")
                .font(.headline)
            Text("Multimodal retrieval sweeps: Recall@1, Recall@3, MRR, Mean/Worst margin, and per-case diagnostics")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var labHeaderActions: some View {
        HStack(spacing: 8) {
            Button {
                embeddings.loadBuiltInMultimodalSampleSet()
            } label: {
                Label("Sample (8 cases)", systemImage: "shapes")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button {
                showSetJSONImporter = true
            } label: {
                Label("Import set.json / Folder…", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button {
                showLabExporter = true
            } label: {
                Label("Export JSON", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(embeddings.labSweepResults.isEmpty)

            if viewModel.isRunningEvalLab {
                Button(role: .destructive) {
                    embeddings.cancelActiveEmbeddingRun()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.small)
            } else {
                Button {
                    Task { await embeddings.runMultimodalEmbeddingLab() }
                } label: {
                    Label("Run Evaluation", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(embeddings.labSet.cases.isEmpty)
            }
        }
    }

    private var labConfigPickers: some View {
        HStack(spacing: 8) {
            if !embeddings.availableEmbeddingModels.isEmpty {
                Picker("Model", selection: $draftModelID) {
                    ForEach(embeddings.availableEmbeddingModels) { entry in
                        Text(entry.label).tag(entry.id)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 180)
            }

            Picker("Dim", selection: $draftDimensions) {
                ForEach(embeddings.availableEmbeddingDimensions, id: \.self) { dim in
                    Text("\(dim)d").tag(dim)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 90)

            Picker("Query prompt", selection: $draftQueryPrompt) {
                Text("Q: None").tag("")
                ForEach(embeddings.availableEmbeddingPromptNames, id: \.self) { name in
                    Text("Q: \(name)").tag(name)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 160)

            Picker("Doc prompt", selection: $draftDocPrompt) {
                Text("D: None").tag("")
                ForEach(embeddings.availableEmbeddingPromptNames, id: \.self) { name in
                    Text("D: \(name)").tag(name)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 160)
        }
    }

    private var labConfigButtons: some View {
        HStack(spacing: 8) {
            Button {
                embeddings.addCurrentConfigToLabQueue(
                    modelID: draftModelID.isEmpty ? embeddings.selectedEmbeddingModelID : draftModelID,
                    dimensions: draftDimensions,
                    queryPromptName: draftQueryPrompt.isEmpty ? nil : draftQueryPrompt,
                    docPromptName: draftDocPrompt.isEmpty ? nil : draftDocPrompt
                )
            } label: {
                Label("Add Config", systemImage: "plus")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button {
                embeddings.addPromptsOnVsOffAblation()
            } label: {
                Label("Add prompts on vs. off", systemImage: "switch.2")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }
}
