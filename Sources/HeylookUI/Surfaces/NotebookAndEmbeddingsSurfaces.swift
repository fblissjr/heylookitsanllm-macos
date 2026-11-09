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
#endif

// MARK: - 2. Notebook Surface

struct NotebookSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @FocusState private var isBodyFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                TextField("Notebook Title", text: $viewModel.notebookTitle)
                    .font(.title3.weight(.semibold))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 360)

                Spacer()

                if viewModel.isNotebookGenerating {
                    ProgressView()
                        .controlSize(.small)
                }

                GlassEffectContainer(spacing: 10) {
                    HStack(spacing: 8) {
                        Button {
                            Task { await viewModel.continueNotebookGeneration() }
                        } label: {
                            Label("Continue Writing", systemImage: "sparkles")
                        }
                        .buttonStyle(.glassProminent)
                        .disabled(viewModel.isNotebookGenerating)

                        Button {
                            viewModel.saveNotebook()
                        } label: {
                            Label("Save", systemImage: "square.and.arrow.down")
                        }
                        .buttonStyle(.glass)
                        .keyboardShortcut("s", modifiers: [.command])

                        Button {
                            viewModel.saveNotebookToConversations()
                        } label: {
                            Label("Send to Chat", systemImage: "bubble.left.and.bubble.right")
                        }
                        .buttonStyle(.glass)
                    }
                }
            }

            TextEditor(text: $viewModel.notebookBody)
                .font(.system(.body, design: .monospaced))
                .focused($isBodyFocused)
                .defaultFocus($isBodyFocused, true)
                .padding(8)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(Color.secondary.opacity(0.18), lineWidth: 1)
                }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            isBodyFocused = true
        }
    }
}

// MARK: - 3. Embeddings & ANE Media Finder Studio Surface

struct EmbeddingsStudioSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @FocusState private var isSearchFocused: Bool
    @State private var showIndexingImporter = false

    var body: some View {
        VStack(spacing: 0) {
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
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            isSearchFocused = true
        }
        #if !os(macOS)
        .fileImporter(
            isPresented: $showIndexingImporter,
            allowedContentTypes: [.item, .folder],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                viewModel.startIndexing(urls)
            }
        }
        #endif
    }

    private func triggerIndexingPicker() {
        #if os(macOS)
        openFilePickerForIndexing()
        #else
        showIndexingImporter = true
        #endif
    }

    #if os(macOS)
    private func openFilePickerForIndexing() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Index"
        if panel.runModal() == .OK {
            viewModel.startIndexing(panel.urls)
        }
    }
    #endif
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
                        Task { await viewModel.runEmbeddingSearch() }
                    }

                Button {
                    #if os(macOS)
                    pickQueryAttachment()
                    #else
                    showQueryMediaImporter = true
                    #endif
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
        #if !os(macOS)
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
        #endif
    }

    #if os(macOS)
    private func pickQueryAttachment() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.image, .audiovisualContent, .audio]
        panel.prompt = "Attach Query Media"
        if panel.runModal() == .OK, let url = panel.url {
            viewModel.queryAttachmentURL = url
            Task { await viewModel.runEmbeddingSearch() }
        }
    }
    #endif
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
                #if os(macOS)
                openFilePickerForIndexing()
                #else
                showIndexingImporter = true
                #endif
            } label: {
                Label("Add Files or Folders…", systemImage: "plus.rectangle.on.folder")
                    .fontWeight(.medium)
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)
            #if !os(macOS)
            .fileImporter(
                isPresented: $showIndexingImporter,
                allowedContentTypes: [.item, .folder],
                allowsMultipleSelection: true
            ) { result in
                if case .success(let urls) = result {
                    viewModel.startIndexing(urls)
                }
            }
            #endif

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

    #if os(macOS)
    private func openFilePickerForIndexing() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Index"
        if panel.runModal() == .OK {
            viewModel.startIndexing(panel.urls)
        }
    }
    #endif
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
                                #else
                                VideoPlayer(player: player)
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
#endif
