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
    @State private var showNotebookSystemPrompt: Bool = false

    private var wordCount: Int {
        viewModel.notebookBody
            .split { $0.isWhitespace || $0.isNewline }
            .count
    }

    var body: some View {
        let hasNotebookSysPrompt = !viewModel.notebookSystemPrompt
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty

        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                TextField("Notebook Title", text: $viewModel.notebookTitle)
                    .font(.title3.weight(.semibold))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 300)

                Text(viewModel.isNotebookDirty ? "Saving…" : "Saved")
                    .font(.caption2.monospaced())
                    .foregroundStyle(viewModel.isNotebookDirty ? .orange : .secondary)

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
                            viewModel.selectedPresetName != nil
                                ? Color.indigo.opacity(0.15)
                                : Color.secondary.opacity(0.12)
                        )
                    )
                }
                .menuStyle(.borderlessButton)
                .fixedSize()

                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        showNotebookSystemPrompt.toggle()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: hasNotebookSysPrompt ? "terminal.fill" : "terminal")
                        Text(hasNotebookSysPrompt ? "System ●" : "System")
                    }
                    .font(.caption)
                }
                .buttonStyle(.bordered)
                .tint(hasNotebookSysPrompt ? .accentColor : nil)
                .controlSize(.small)

                Spacer()

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
                            .disabled(viewModel.notebookBody.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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

            if showNotebookSystemPrompt {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Label("Notebook System Prompt (persisted per notebook)", systemImage: "terminal")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if !viewModel.notebookSystemPrompt.isEmpty {
                            Button("Clear") {
                                viewModel.notebookSystemPrompt = ""
                            }
                            .font(.caption2)
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                        }
                    }
                    TextEditor(text: $viewModel.notebookSystemPrompt)
                        .font(.system(size: 12, design: .monospaced))
                        .frame(minHeight: 56, maxHeight: 110)
                        .padding(6)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(0.08))
                        )
                }
            }

            CursorAwareTextEditor(
                text: $viewModel.notebookBody,
                selectedRange: $viewModel.notebookSelectedRange,
                isEditable: !viewModel.isNotebookGenerating,
                onUserEdit: {
                    viewModel.noteNotebookUserEdited()
                }
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(
                        viewModel.isNotebookGenerating
                            ? Color.accentColor.opacity(0.45)
                            : Color.secondary.opacity(0.18),
                        lineWidth: 1
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
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - 3. Embeddings & ANE Media Finder Studio Surface

struct EmbeddingsStudioSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @FocusState private var isSearchFocused: Bool
    @State private var showIndexingImporter = false

    var body: some View {
        VStack(spacing: 0) {
            // Top Mode Switcher Bar (Media Search | N×N Compare | Embedding Lab)
            HStack(spacing: 12) {
                Picker("Embeddings Mode", selection: $viewModel.embeddingsSubMode) {
                    ForEach(EmbeddingsSubMode.allCases) { mode in
                        Label(mode.rawValue, systemImage: mode.icon)
                            .tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 440)

                Spacer()

                if viewModel.isActiveEmbeddingEngineSynthetic {
                    HStack(spacing: 5) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text("No real embedding checkpoint loaded — synthetic fallback blocked")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.orange)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color.orange.opacity(0.14)))
                } else {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 7, height: 7)
                        Text(viewModel.activeQueryTowersLabel)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 14)
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
                        Text("Type a search query above, attach an image or audio clip, or index local folders from the sidebar.")
                    } actions: {
                        GlassEffectContainer(spacing: 10) {
                            HStack(spacing: 10) {
                                Button {
                                    triggerIndexingPicker()
                                } label: {
                                    Label("Add Files or Folders…", systemImage: "plus.rectangle.on.folder")
                                }
                                .buttonStyle(.glassProminent)
                                .controlSize(.small)

                                Button {
                                    Task { await viewModel.seedSampleCorpusIfNeeded() }
                                } label: {
                                    Label("Load Sample Demo Corpus", systemImage: "sparkles")
                                }
                                .buttonStyle(.glass)
                                .controlSize(.small)
                            }
                        }
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

                // Optional Pairwise & Retrieval Lab Bottom Drawer
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

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Pairwise Embedding Cosine & Retrieval Lab (Recall@1 / MRR)", systemImage: "chart.bar.xaxis")
                    .font(.caption.weight(.semibold))
                Spacer()
                Button {
                    Task { await viewModel.runFullEvalLabSuite() }
                } label: {
                    Label(
                        viewModel.isRunningEvalLab ? "Running Probes…" : "Run Eval Suite (chain + VLM + MRL)",
                        systemImage: "checkmark.shield"
                    )
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.mini)
                .disabled(viewModel.isRunningEvalLab)

                ForEach(viewModel.labMetrics) { metric in
                    Text(String(format: "%dd R@1: %.0f%% (MRR %.2f)", metric.dimensions, metric.recallAt1 * 100, metric.mrr))
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.12), in: Capsule())
                }
            }

            if let suite = viewModel.evalLabSuiteReport {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(suite.probes) { probe in
                            HStack(spacing: 4) {
                                Image(systemName: probe.passed ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                    .foregroundStyle(probe.passed ? .green : .red)
                                    .font(.caption2)
                                Text("\(probe.name): \(probe.detail)")
                                    .font(.caption2.monospaced())
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(
                                (probe.passed ? Color.green : Color.red).opacity(0.12),
                                in: Capsule()
                            )
                        }
                    }
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
                    Label("Add to Lab (\(viewModel.benchmarkCases.count))", systemImage: "plus.square.on.square")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(viewModel.selectedHit == nil && viewModel.searchHits.isEmpty)
                .help("Promote Left query + selected artifact into the Retrieval Lab benchmark suite")

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

private struct EmbeddingsCompareMatrixView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var showMediaImporter: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Controls Header
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("N×N Multimodal Cosine Similarity Matrix")
                            .font(.headline)
                        Text("Compare up to \(HeylookWorkbenchViewModel.maxCompareMatrixItems) text, code, PDF, image, audio, or video items with live Matryoshka subspace slicing")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Toggle(
                        viewModel.compareAsymmetricMode ? "Asymmetric (Query→Doc)" : "Symmetric (STS)",
                        isOn: $viewModel.compareAsymmetricMode
                    )
                    .toggleStyle(.button)
                    .controlSize(.small)

                    Picker("Subspace", selection: $viewModel.compareMatrixDimension) {
                        Text("768d").tag(768)
                        Text("512d").tag(512)
                        Text("256d").tag(256)
                        Text("128d").tag(128)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 210)

                    Button {
                        Task { await viewModel.runCompareMatrix() }
                    } label: {
                        HStack(spacing: 5) {
                            if viewModel.isEmbeddingBusy {
                                ProgressView().controlSize(.mini)
                            }
                            Label("Compute Matrix", systemImage: "square.grid.3x3.fill")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(viewModel.isEmbeddingBusy || viewModel.compareItems.count < 2)
                }

                // Item Staging List (`≤16`)
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("STAGED ITEMS (\(viewModel.compareItems.count) / \(HeylookWorkbenchViewModel.maxCompareMatrixItems))")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button {
                                viewModel.addCompareTextItem()
                            } label: {
                                Label("Add Text", systemImage: "plus.bubble")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(viewModel.compareItems.count >= HeylookWorkbenchViewModel.maxCompareMatrixItems)

                            Button {
                                showMediaImporter = true
                            } label: {
                                Label("Add Media / Files…", systemImage: "plus.rectangle.on.folder")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(viewModel.compareItems.count >= HeylookWorkbenchViewModel.maxCompareMatrixItems)
                        }

                        ForEach($viewModel.compareItems) { $item in
                            HStack(spacing: 8) {
                                Image(systemName: item.kind.systemImage)
                                    .foregroundStyle(.tint)
                                    .frame(width: 18)

                                TextField("Label", text: $item.label)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption.weight(.semibold))
                                    .frame(width: 120)

                                if let fileURL = item.fileURL {
                                    HStack(spacing: 5) {
                                        Text(item.kind.displayName)
                                            .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                                            .foregroundStyle(Color.accentColor)
                                        Text(fileURL.lastPathComponent)
                                            .font(.caption.monospaced())
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                        Spacer()
                                    }
                                } else {
                                    TextField("Enter text or code snippet to embed…", text: $item.textContent)
                                        .textFieldStyle(.roundedBorder)
                                        .font(.caption)
                                }

                                Button(role: .destructive) {
                                    viewModel.removeCompareItem(id: item.id)
                                } label: {
                                    Image(systemName: "minus.circle.fill")
                                        .foregroundStyle(.secondary)
                                        .frame(minWidth: 28, minHeight: 28)
                                        .contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                                .disabled(viewModel.compareItems.count <= 2)
                                .help("Remove item")
                            }
                        }
                    }
                    .padding(6)
                }

                // Heatmap Grid + Selected Pair Attribution
                if !viewModel.compareMatrixScores.isEmpty,
                   viewModel.compareMatrixScores.count == viewModel.compareItems.count {
                    HStack(alignment: .top, spacing: 16) {
                        CompareMatrixHeatmapGridView(viewModel: viewModel)
                        CompareSelectedPairProfileCard(
                            viewModel: viewModel,
                            onCopyText: copyToPasteboard
                        )
                    }
                } else {
                    ContentUnavailableView(
                        "Matrix Not Computed Yet",
                        systemImage: "square.grid.3x3",
                        description: Text("Stage 2–16 text or media items above and click Compute Matrix to inspect cross-modal Matryoshka cosine similarity.")
                    )
                    .frame(maxWidth: .infinity, minHeight: 180)
                }
            }
            .padding(16)
        }
        .fileImporter(
            isPresented: $showMediaImporter,
            allowedContentTypes: [.image, .audiovisualContent, .audio, .pdf, .plainText, .sourceCode, .item],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                viewModel.addCompareMediaFiles(urls)
            }
        }
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

private struct CompareMatrixHeatmapGridView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        let items = viewModel.compareItems
        let scores = viewModel.compareMatrixScores
        ScrollView([.horizontal, .vertical]) {
            Grid(horizontalSpacing: 4, verticalSpacing: 4) {
                GridRow {
                    Text(viewModel.compareAsymmetricMode ? "Row (Q) \\ Col (D)" : "Symmetric")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 108, alignment: .leading)

                    ForEach(0..<items.count, id: \.self) { cIdx in
                        let colItem = items[cIdx]
                        Text(colItem.label)
                            .font(.system(size: 10, weight: .semibold))
                            .lineLimit(1)
                            .frame(width: 66)
                            .help(colItem.fileURL?.lastPathComponent ?? colItem.textContent)
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
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(cellColor)
                )
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(Color.primary, lineWidth: 2)
                    }
                }
        }
        .buttonStyle(.plain)
    }
}

private struct CompareSelectedPairProfileCard: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    let onCopyText: (String) -> Void

    var body: some View {
        let count = max(1, viewModel.compareItems.count)
        let r = min(max(0, viewModel.selectedCompareCellRow), count - 1)
        let c = min(max(0, viewModel.selectedCompareCellCol), count - 1)
        let dimScores = viewModel.scoresAcrossDimensionsForCompareCell(row: r, col: c)

        VStack(alignment: .leading, spacing: 10) {
            Text("SELECTED PAIR SUBSPACE PROFILE")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.secondary)

            Text("\(viewModel.compareItems[r].label) ↔ \(viewModel.compareItems[c].label)")
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

            Divider()

            Button {
                Task { await viewModel.sendComparePairToLab(row: r, col: c) }
            } label: {
                Label("Send Pair to Retrieval Lab", systemImage: "chart.bar.doc.horizontal")
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

                Button {
                    onCopyText(viewModel.compareMatrixJSON())
                    viewModel.statusBanner = "Copied \(viewModel.compareMatrixDimension)d similarity matrix as JSON."
                } label: {
                    Label("Copy JSON", systemImage: "curlybraces")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(12)
        .frame(width: 280)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.07)))
    }
}

// MARK: - Multi-Config Embedding Retrieval & ANE/GPU Placement Lab

private struct EmbeddingsMultiConfigLabView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var showSetJSONImporter: Bool = false
    @State private var newCaseQuery: String = ""
    @State private var newCaseExpectedID: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header & Action Bar
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Multi-Config Retrieval & ANE/GPU Placement Lab")
                            .font(.headline)
                        Text("Evaluate Recall@1, MRR, Top-1 Cosine Margin, and latency across Matryoshka subspaces and compute placements")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Button {
                        showSetJSONImporter = true
                    } label: {
                        Label("Import set.json…", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button {
                        Task { await viewModel.evaluateCurrentConfigInLab() }
                    } label: {
                        Label("Eval Current Config", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(viewModel.isRunningEvalLab || viewModel.artifactCount == 0)

                    Button {
                        Task { await viewModel.sweepComputePresetsInLab() }
                    } label: {
                        Label("Sweep GPU / Hybrid / ANE", systemImage: "bolt.horizontal.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.isRunningEvalLab || viewModel.artifactCount == 0)

                    Button {
                        Task { await viewModel.runFullEvalLabSuite() }
                    } label: {
                        Label("Full Probe Suite", systemImage: "checkmark.shield")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.isRunningEvalLab)

                    if !viewModel.labConfigRuns.isEmpty {
                        Button("Clear Runs") {
                            viewModel.clearLabConfigRuns()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }

                if viewModel.artifactCount == 0 {
                    HStack(spacing: 10) {
                        Image(systemName: "info.circle.fill")
                            .foregroundStyle(.blue)
                        Text("VectorStore has 0 indexed artifacts. Load the Sample Demo Corpus or index a folder first to run Retrieval Lab evaluations.")
                            .font(.caption)
                        Spacer()
                        Button("Load Sample Demo Corpus") {
                            Task { await viewModel.seedSampleCorpusIfNeeded() }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.blue.opacity(0.1)))
                }

                // Multi-Config Comparison Table
                GroupBox("Config Comparison Matrix (\(viewModel.labConfigRuns.count) run(s))") {
                    if viewModel.labConfigRuns.isEmpty {
                        Text("No configuration runs recorded yet. Click 'Eval Current Config' or 'Sweep GPU / Hybrid / ANE' above.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(8)
                    } else {
                        ScrollView(.horizontal) {
                            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                                GridRow {
                                    Text("Config / Checkpoint").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Placement").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("768d R@1").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("512d R@1").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("256d R@1").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("128d R@1").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("768d MRR").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Margin (Mean / Worst)").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Latency").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                    Text("Est. ANE / GPU").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                                }

                                Divider()

                                ForEach(viewModel.labConfigRuns) { run in
                                    GridRow {
                                        Text(run.label)
                                            .font(.caption.weight(.semibold))
                                            .lineLimit(1)
                                        Text(run.shortPlacementBadge)
                                            .font(.caption2.monospaced())
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Capsule().fill(Color.purple.opacity(0.14)))
                                            .foregroundStyle(.purple)
                                        Text(recallLabel(run, dim: 768)).font(.caption.monospaced())
                                        Text(recallLabel(run, dim: 512)).font(.caption.monospaced())
                                        Text(recallLabel(run, dim: 256)).font(.caption.monospaced())
                                        Text(recallLabel(run, dim: 128)).font(.caption.monospaced())
                                        Text(mrrLabel(run, dim: 768)).font(.caption.monospaced())
                                        Text(String(format: "%+.3f / %+.3f", run.meanTop1Margin, run.worstTop1Margin))
                                            .font(.caption.monospaced())
                                            .foregroundStyle(run.worstTop1Margin >= 0 ? .green : .orange)
                                        Text(String(format: "%.1f ms", run.evalLatencyMs))
                                            .font(.caption.monospaced())
                                        Text("\(run.aneMemoryMB)MB / \(run.gpuMemoryMB)MB")
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .padding(6)
                        }
                    }
                }

                //Probe Suite Summary
                if let suite = viewModel.evalLabSuiteReport {
                    GroupBox("Verification Probe Suite") {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(suite.probes) { probe in
                                HStack(spacing: 8) {
                                    Image(systemName: probe.passed ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                        .foregroundStyle(probe.passed ? .green : .red)
                                    Text(probe.name)
                                        .font(.caption.weight(.semibold))
                                    Text(probe.detail)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                }
                            }
                        }
                        .padding(6)
                    }
                }

                // Benchmark Cases (`set.json`) Editor
                GroupBox("Benchmark Ground-Truth Cases (\(viewModel.benchmarkCases.count))") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 8) {
                            TextField("Benchmark query…", text: $newCaseQuery)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                            TextField("Expected artifact ID (e.g. doc-vision)", text: $newCaseExpectedID)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption.monospaced())
                                .frame(maxWidth: 240)
                            Button("Add Case") {
                                let q = newCaseQuery.trimmingCharacters(in: .whitespacesAndNewlines)
                                let exp = newCaseExpectedID.trimmingCharacters(in: .whitespacesAndNewlines)
                                guard !q.isEmpty, !exp.isEmpty else { return }
                                viewModel.benchmarkCases.append(.init(query: q, expectedArtifactID: exp))
                                newCaseQuery = ""
                                newCaseExpectedID = ""
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(
                                newCaseQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                    || newCaseExpectedID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            )
                        }

                        ForEach(Array(viewModel.benchmarkCases.enumerated()), id: \.offset) { idx, bench in
                            HStack(spacing: 8) {
                                Text("#\(idx + 1)")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 28, alignment: .leading)
                                Text(bench.query)
                                    .font(.caption)
                                Spacer()
                                Text("→ \(bench.expectedArtifactID)")
                                    .font(.caption2.monospaced())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Capsule().fill(Color.secondary.opacity(0.12)))
                                Button(role: .destructive) {
                                    viewModel.benchmarkCases.remove(at: idx)
                                } label: {
                                    Image(systemName: "trash")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .frame(minWidth: 28, minHeight: 28)
                                        .contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .padding(6)
                }
            }
            .padding(16)
        }
        .fileImporter(
            isPresented: $showSetJSONImporter,
            allowedContentTypes: [.json],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                viewModel.importBenchmarkSetJSON(from: url)
            }
        }
    }

    private func recallLabel(_ run: EmbeddingLabConfigRun, dim: Int) -> String {
        guard let m = run.metricsByDim.first(where: { $0.dimensions == dim }) else { return "—" }
        return String(format: "%.0f%%", m.recallAt1 * 100)
    }

    private func mrrLabel(_ run: EmbeddingLabConfigRun, dim: Int) -> String {
        guard let m = run.metricsByDim.first(where: { $0.dimensions == dim }) else { return "—" }
        return String(format: "%.2f", m.mrr)
    }
}
