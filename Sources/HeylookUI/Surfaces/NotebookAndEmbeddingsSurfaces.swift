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

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                TextField("Notebook Title", text: $viewModel.notebookTitle)
                    .font(.title3.weight(.semibold))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 340)

                Spacer()

                Button("Continue Writing") {
                    Task { await viewModel.continueNotebookGeneration() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isNotebookGenerating)

                Button("Save Notebook") {
                    viewModel.saveNotebook()
                }
                .buttonStyle(.bordered)

                Button("Save to Conversations") {
                    viewModel.saveNotebookToConversations()
                }
                .buttonStyle(.bordered)
            }

            TextEditor(text: $viewModel.notebookBody)
                .font(.system(.body, design: .monospaced))
                .padding(8)
                .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
        }
        .padding(16)
    }
}

// MARK: - 3. Embeddings & ANE Media Finder Studio Surface

struct EmbeddingsStudioSurface: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        #if os(macOS)
        HSplitView {
            EmbeddingsSidebarView(viewModel: viewModel)
                .frame(minWidth: 260, idealWidth: 290, maxWidth: 340)

            EmbeddingsSearchAndResultsView(viewModel: viewModel)
                .frame(minWidth: 420, idealWidth: 560)

            EmbeddingsInspectorView(
                hit: viewModel.selectedHit,
                selectedDimension: viewModel.selectedDimension,
                onSendToCompare: { hit in
                    viewModel.sendHitToPairwiseCompare(hit)
                },
                onPinToLab: { hit in
                    Task { await viewModel.pinHitToRetrievalLab(hit) }
                }
            )
            .frame(minWidth: 340, idealWidth: 400)
        }
        #else
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                EmbeddingsSidebarView(viewModel: viewModel)
                EmbeddingsSearchAndResultsView(viewModel: viewModel)
                EmbeddingsInspectorView(
                    hit: viewModel.selectedHit,
                    selectedDimension: viewModel.selectedDimension,
                    onSendToCompare: { hit in
                        viewModel.sendHitToPairwiseCompare(hit)
                    },
                    onPinToLab: { hit in
                        Task { await viewModel.pinHitToRetrievalLab(hit) }
                    }
                )
            }
            .padding(12)
        }
        #endif
    }
}

private struct EmbeddingsSidebarView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel
    @State private var isTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "sparkle.magnifyingglass")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text("mbed Studio")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                    Text("Multimodal ANE / GPU Finder")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Modality Filter
                    VStack(alignment: .leading, spacing: 4) {
                        Text("MODALITY FILTER")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 14)

                        modalityFilterButton(title: "All Artifacts", icon: "square.grid.2x2", kind: nil, count: viewModel.artifactCount)

                        ForEach(ArtifactKind.allCases) { kind in
                            modalityFilterButton(
                                title: kind.displayName,
                                icon: kind.systemImage,
                                kind: kind,
                                count: viewModel.countsByKind[kind] ?? 0
                            )
                        }
                    }

                    Divider()

                    // Engine & Compute Placement
                    VStack(alignment: .leading, spacing: 10) {
                        Text("ENGINE & COMPUTE PLACEMENT")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        Picker("Engine", selection: $viewModel.selectedBackend) {
                            ForEach(EngineBackend.allCases, id: \.self) { engine in
                                Text(engine.label).tag(engine)
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: viewModel.selectedBackend) {
                            viewModel.setupEmbeddingCoordinator()
                        }

                        Picker("Dimensions", selection: $viewModel.selectedDimension) {
                            Text("768d (Full)").tag(768)
                            Text("512d").tag(512)
                            Text("256d (Fast)").tag(256)
                            Text("128d (Compact)").tag(128)
                        }
                        .pickerStyle(.menu)
                        .onChange(of: viewModel.selectedDimension) {
                            Task { await viewModel.runEmbeddingSearch() }
                        }

                        Picker(
                            "Query Task",
                            selection: Binding(
                                get: { viewModel.effectiveQueryTask },
                                set: { newTask in
                                    viewModel.selectedTaskOverride = newTask
                                    Task { await viewModel.runEmbeddingSearch() }
                                }
                            )
                        ) {
                            Text("Retrieval Query (Asymmetric)").tag(EmbeddingTask.query)
                            Text("Semantic Similarity (Symmetric)").tag(EmbeddingTask.sts)
                            Text("Code Retrieval").tag(EmbeddingTask.codeRetrieval)
                            Text("Question Answering").tag(EmbeddingTask.questionAnswering)
                            Text("Fact Verification").tag(EmbeddingTask.factVerification)
                        }
                        .pickerStyle(.menu)

                        Picker("Video Sampling", selection: $viewModel.selectedSampling) {
                            ForEach(VideoWindowSampling.allCases, id: \.self) { mode in
                                Text(mode.label).tag(mode)
                            }
                        }
                        .pickerStyle(.menu)

                        Toggle("Include Video Soundtrack", isOn: $viewModel.loadAudioFromVideo)
                            .font(.caption)
                            .toggleStyle(.switch)
                            .controlSize(.small)

                        Divider()

                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Label("Per-Tower Compute", systemImage: "cpu.fill")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(viewModel.modelComputeOptions.shortSummaryBadge)
                                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }

                            Picker(
                                "Preset",
                                selection: Binding(
                                    get: { viewModel.modelComputeOptions.preset },
                                    set: { viewModel.applyComputePreset($0) }
                                )
                            ) {
                                ForEach(ComputePreset.allCases, id: \.self) { preset in
                                    Text(preset.label).tag(preset)
                                }
                            }
                            .pickerStyle(.menu)
                            .controlSize(.small)

                            ForEach(EmbeddingComponent.allCases) { component in
                                HStack(spacing: 6) {
                                    Image(systemName: component.systemImage)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .frame(width: 14)
                                    Text(component.title)
                                        .font(.caption2.weight(.medium))
                                    Spacer()
                                    Picker(
                                        "",
                                        selection: Binding(
                                            get: { viewModel.modelComputeOptions[component] },
                                            set: { newUnit in
                                                viewModel.modelComputeOptions[component] = newUnit
                                                viewModel.setupEmbeddingCoordinator()
                                            }
                                        )
                                    ) {
                                        ForEach(TowerComputeUnit.allCases, id: \.self) { unit in
                                            Text(unit.label).tag(unit)
                                        }
                                    }
                                    .labelsHidden()
                                    .pickerStyle(.menu)
                                    .controlSize(.mini)
                                    .frame(width: 122)
                                }

                                if let report = viewModel.aneInspectionReports.first(where: { $0.component == component }) {
                                    HStack(spacing: 4) {
                                        Circle()
                                            .fill(report.isCompiledBundlePresent ? Color.green : Color.orange)
                                            .frame(width: 6, height: 6)
                                        Text(
                                            report.isCompiledBundlePresent
                                                ? "CoreML Ready (\(Int((report.aneResidencyFraction * 100).rounded()))% ANE ops)"
                                                : "CoreML missing — run scripts/export_ane_coreml.py"
                                        )
                                        .font(.system(size: 9))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    }
                                    .padding(.leading, 20)
                                }
                            }

                            ComputeMemorySplitBar(split: viewModel.modelComputeOptions.memorySplitMB)
                                .padding(.top, 4)
                        }

                        if viewModel.selectedBackend == .heylook {
                            TextField("Server URL", text: $viewModel.embeddingServerURLString)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                                .onSubmit {
                                    viewModel.setupEmbeddingCoordinator()
                                }
                        } else if viewModel.selectedBackend == .nativeMLX {
                            TextField("Checkpoint Path", text: $viewModel.checkpointPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.caption)
                                .onSubmit {
                                    viewModel.setupEmbeddingCoordinator()
                                }
                        }
                    }
                    .padding(.horizontal, 14)

                    Divider()

                    // Index Actions
                    VStack(alignment: .leading, spacing: 8) {
                        Text("INDEXING & EVALUATION")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        #if os(macOS)
                        Button {
                            openFilePickerForIndexing()
                        } label: {
                            Label("Add Files or Folders…", systemImage: "plus.rectangle.on.folder")
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.regular)
                        #endif

                        HStack(spacing: 6) {
                            Button {
                                viewModel.reindexWatchedFolders()
                            } label: {
                                Label("Re-scan", systemImage: "arrow.clockwise")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(viewModel.watchedFolders.isEmpty || viewModel.isIndexing)

                            Button(role: .destructive) {
                                viewModel.clearEmbeddingIndex()
                            } label: {
                                Label("Clear", systemImage: "trash")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(viewModel.artifactCount == 0 || viewModel.isIndexing)
                        }

                        Button {
                            viewModel.showEvalLabDrawer.toggle()
                        } label: {
                            Label(
                                viewModel.showEvalLabDrawer ? "Hide Pairwise & Eval Lab" : "Show Pairwise & Eval Lab",
                                systemImage: "chart.bar.xaxis"
                            )
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)

                        if !viewModel.watchedFolders.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Image(systemName: "eye.fill")
                                        .font(.caption2)
                                        .foregroundStyle(.green)
                                    Text("WATCHED FOLDERS (FSEVENTS)")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.top, 4)

                                ForEach(viewModel.watchedFolders) { folder in
                                    HStack(spacing: 4) {
                                        Image(systemName: "folder")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Text(folder.url.lastPathComponent)
                                            .font(.caption2)
                                            .lineLimit(1)
                                            .help(folder.path)
                                        Spacer()
                                        Button {
                                            viewModel.removeWatchedFolder(folder)
                                        } label: {
                                            Image(systemName: "xmark.circle.fill")
                                                .font(.caption2)
                                                .foregroundStyle(.tertiary)
                                        }
                                        .buttonStyle(.plain)
                                        .help("Stop watching \(folder.path)")
                                    }
                                }
                            }
                        }

                        Text("Drop folders, videos, audio, PDFs, images, or code here to index.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 14)
                    .padding(.bottom, 14)
                }
            }
        }
        .background(isTargeted ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.04))
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private func modalityFilterButton(title: String, icon: String, kind: ArtifactKind?, count: Int) -> some View {
        let isSelected = viewModel.selectedKindFilter == kind
        return Button {
            viewModel.selectedKindFilter = kind
            Task { await viewModel.runEmbeddingSearch() }
        } label: {
            HStack {
                Image(systemName: icon)
                    .frame(width: 18)
                    .foregroundStyle(isSelected ? .white : .primary)
                Text(title)
                    .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? .white : .primary)
                Spacer()
                Text("\(count)")
                    .font(.caption2.monospacedDigit())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(isSelected ? Color.white.opacity(0.25) : Color.secondary.opacity(0.15))
                    )
                    .foregroundStyle(isSelected ? .white : .secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(isSelected ? Color.accentColor : Color.clear)
            )
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 8)
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

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    viewModel.startIndexing([url])
                }
            }
        }
        return true
    }
}

private struct EmbeddingsSearchAndResultsView: View {
    @Bindable var viewModel: HeylookWorkbenchViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Search Bar + Query Attachment
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 15, weight: .medium))

                    TextField("Search across text, code, PDFs, images, audio & video…", text: $viewModel.embeddingQuery)
                        .textFieldStyle(.plain)
                        .font(.system(size: 15))
                        .onSubmit {
                            Task { await viewModel.runEmbeddingSearch() }
                        }
                        .onChange(of: viewModel.embeddingQuery) {
                            Task { await viewModel.runEmbeddingSearch() }
                        }

                    #if os(macOS)
                    Button {
                        pickQueryAttachment()
                    } label: {
                        Image(systemName: "paperclip")
                            .foregroundStyle(viewModel.queryAttachmentURL != nil ? Color.accentColor : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Attach an image or audio clip for multimodal query search")
                    #endif

                    if !viewModel.embeddingQuery.isEmpty || viewModel.queryAttachmentURL != nil {
                        Button {
                            viewModel.embeddingQuery = ""
                            viewModel.queryAttachmentURL = nil
                            Task { await viewModel.runEmbeddingSearch() }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
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
                    videoSampling: viewModel.selectedSampling
                )
            }
            .padding(14)

            Divider()

            // Status / Progress Bar
            HStack {
                if viewModel.isIndexing || viewModel.isEmbeddingBusy {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.75)
                }
                Text(viewModel.embeddingStatusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                if viewModel.isIndexing {
                    Button {
                        viewModel.cancelIndexing()
                    } label: {
                        Label("Stop", systemImage: "stop.circle.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                    .help("Cancel current indexing job")
                }
                Text("\(viewModel.searchHits.count) results")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(Color.secondary.opacity(0.05))

            Divider()

            // Results List
            if viewModel.searchHits.isEmpty {
                ContentUnavailableView(
                    "No Matching Artifacts",
                    systemImage: "doc.text.magnifyingglass",
                    description: Text("Try another query, attach an image/audio clip, or drop folders onto the sidebar to index.")
                )
                .frame(maxHeight: .infinity)
            } else {
                List(selection: $viewModel.selectedHit) {
                    ForEach(viewModel.searchHits) { hit in
                        HitRowView(hit: hit)
                            .tag(hit)
                    }
                }
                .listStyle(.inset)
            }

            if viewModel.showEvalLabDrawer {
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("Pairwise Embedding Cosine & Retrieval Lab (Recall@1 / MRR)", systemImage: "chart.bar.xaxis")
                            .font(.caption.weight(.semibold))
                        Spacer()
                        Button {
                            Task { await viewModel.runFullEvalLabSuite() }
                        } label: {
                            Label(
                                viewModel.isRunningEvalLab ? "Running Probes..." : "Run Eval Suite (chain + VLM + MRL)",
                                systemImage: "checkmark.shield"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)
                        .disabled(viewModel.isRunningEvalLab)

                        ForEach(viewModel.labMetrics, id: \.dimensions) { metric in
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
                                ForEach(suite.probes, id: \.name) { probe in
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
    }

    #if os(macOS)
    private func pickQueryAttachment() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Attach Query Media"
        if panel.runModal() == .OK, let url = panel.url {
            viewModel.queryAttachmentURL = url
            Task { await viewModel.runEmbeddingSearch() }
        }
    }
    #endif
}

private struct ScoringPipelineBanner: View {
    let dimension: Int
    let task: EmbeddingTask
    let hasQueryMedia: Bool
    let includeSoundtrack: Bool
    let videoSampling: VideoWindowSampling

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                FeaturePill(
                    icon: "cpu",
                    label: "Gemma-2 2B Multimodal",
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

            #if os(macOS)
            if artifact.kind == .image,
               let nsImage = NSImage(contentsOf: artifact.url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 46, height: 46)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                Image(systemName: artifact.kind.systemImage)
                    .font(.system(size: 20))
                    .foregroundStyle(.tint)
            }
            #else
            Image(systemName: artifact.kind.systemImage)
                .font(.system(size: 20))
                .foregroundStyle(.tint)
            #endif
        }
    }
}

private struct EmbeddingsInspectorView: View {
    let hit: SearchHit?
    let selectedDimension: Int
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
                                    Label("Pin to Retrieval Lab", systemImage: "chart.bar.doc.horizontal")
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
                            selectedDimension: selectedDimension
                        )
                    }

                    // Preview by Modality
                    switch hit.artifact.kind {
                    case .image:
                        #if os(macOS)
                        if let nsImage = NSImage(contentsOf: hit.artifact.url) {
                            Image(nsImage: nsImage)
                                .resizable()
                                .scaledToFit()
                                .frame(maxHeight: 280)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                        #endif

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
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("EXTRACTED KEYFRAME TIMESTAMPS (CLICK TO SEEK)")
                                        .font(.system(size: 10, weight: .semibold))
                                        .foregroundStyle(.secondary)

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 6) {
                                            ForEach(Array(span.frameTimes.enumerated()), id: \.offset) { idx, frameTime in
                                                Button {
                                                    seekAndPlay(url: hit.artifact.url, startTime: frameTime)
                                                } label: {
                                                    HStack(spacing: 3) {
                                                        Image(systemName: "camera.viewfinder")
                                                            .font(.system(size: 9))
                                                        Text("#\(idx + 1) \(ArtifactSpan.formatTimestampMillis(frameTime))")
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
                        .onChange(of: hit.artifact.id) {
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
                    VStack(alignment: .leading, spacing: 6) {
                        Text("METADATA")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        LabeledContent("Path", value: hit.artifact.url.path)
                            .font(.caption)
                        LabeledContent("Size", value: ByteCountFormatter.string(fromByteCount: hit.artifact.fingerprint.byteCount, countStyle: .file))
                            .font(.caption)

                        ForEach(hit.artifact.attributes.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
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
                description: Text("Inspect multimodal previews, Matryoshka cosine attribution, and jump directly to matching timestamps or pages.")
            )
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
                Text("Matryoshka Cosine Across Subspaces (L2 Re-Normalized)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                ForEach([128, 256, 512, 768], id: \.self) { dim in
                    if let dimScore = spanMatch.scoresByDimension[dim] {
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
                    }
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(Color.accentColor.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 9)
                .strokeBorder(Color.accentColor.opacity(0.18), lineWidth: 1)
        )
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
