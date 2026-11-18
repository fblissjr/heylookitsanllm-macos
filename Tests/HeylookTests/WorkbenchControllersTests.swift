import Foundation
import Testing
@testable import HeylookCore
@testable import HeylookEmbeddings
@testable import HeylookEngines
@testable import HeylookServer
@testable import HeylookUI

@MainActor
@Suite("Workbench ViewModel, DocumentSettings, Chat & Embeddings Controllers Suite", .serialized)
struct WorkbenchControllersTests {

    private static func makeIsolatedViewModel(
        sandbox: TestTempSandbox,
        embeddingEngine: (any EmbeddingEngine)? = nil
    ) throws -> HeylookWorkbenchViewModel {
        let store = try SQLiteStore(databaseURL: sandbox.dbURL)
        let vectorStore = try VectorStore(databaseURL: sandbox.vectorDBURL)
        let cfg = HeylookConfig(scanFolders: [sandbox.modelsDir.path])
        return HeylookWorkbenchViewModel(
            autoBootstrap: false,
            store: store,
            vectorStore: vectorStore,
            config: cfg,
            embeddingEngine: embeddingEngine ?? TestDeterministicEmbeddingEngine()
        )
    }

    @Test("HeylookWorkbenchViewModel shares one WorkbenchService spine, starts with zero background polling, and caches @Observable projections")
    func testViewModelSingleServiceSpineZeroPollingAndCachedProjections() async throws {
        UserDefaults.standard.removeObject(forKey: "heylook.draftPrompt")
        let sandbox = try TestTempSandbox(prefix: "heylook_vm_spine")
        defer {
            sandbox.cleanup()
            UserDefaults.standard.removeObject(forKey: "heylook.draftPrompt")
        }

        let vm = try Self.makeIsolatedViewModel(sandbox: sandbox)

        // 1. Single WorkbenchService actor instance shared between HTTP router and ChatController
        #expect(vm.chat.localService === vm.router.service)

        // 2. Zero background metrics/system polling tasks spawned at init
        let vmChildLabels = Mirror(reflecting: vm).children.compactMap(\.label)
        let perfChildLabels = Mirror(reflecting: vm.modelsPerf).children.compactMap(\.label)
        #expect(!vmChildLabels.contains(where: { $0.localizedCaseInsensitiveContains("metricsLoop") }))
        #expect(!perfChildLabels.contains(where: { $0.localizedCaseInsensitiveContains("metricsLoop") }))
        #expect(vm.modelsPerf.lastMetricsUpdatedAt == nil)
        try await Task.sleep(nanoseconds: 80_000_000)
        #expect(vm.modelsPerf.lastMetricsUpdatedAt == nil)

        // 3. Default UI state & cached @Observable projections + ASR filter
        #expect(vm.autoSendVoiceTurn == false)
        #expect(vm.activeModelDescription == nil)
        #expect(vm.filteredModels.isEmpty)
        #expect(vm.sortedActiveGGUFPorts.isEmpty)

        let mlxChat = EngineDescription(
            modelID: "qwen3.5-0.8b-mlx",
            provider: .mlx,
            modelPath: sandbox.modelsDir.appendingPathComponent("qwen3.5").path,
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let ggufVLM = EngineDescription(
            modelID: "gemma4-26b-gguf",
            provider: .gguf,
            modelPath: sandbox.modelsDir.appendingPathComponent("gemma4.gguf").path,
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        let whisperASR = EngineDescription(
            modelID: "whisper-large-v3-turbo",
            provider: .coreai,
            modelPath: sandbox.modelsDir.appendingPathComponent("whisper").path,
            architecture: ProvenanceField(value: "whisper", provenance: .observed),
            computePlacement: ProvenanceField(value: .ane, provenance: .observed, computePlacement: .ane),
            supportsAudio: ProvenanceField(value: true, provenance: .observed),
            tags: ["asr", "speech_recognizer"]
        )

        vm.registeredModels = [mlxChat, ggufVLM, whisperASR]
        #expect(vm.modelCountsByProvider[.mlx] == 1)
        #expect(vm.modelCountsByProvider[.gguf] == 1)
        #expect(vm.modelCountsByProvider[.coreai] == 1)
        #expect(vm.generationModels.map(\.modelID) == ["qwen3.5-0.8b-mlx", "gemma4-26b-gguf"])
        #expect(vm.availableASRModels.map(\.modelID).contains("whisper-large-v3-turbo"))

        vm.selectedModelID = "gemma4-26b-gguf"
        #expect(vm.activeModelDescription?.modelID == "gemma4-26b-gguf")
        #expect(vm.activeModelDescription?.supportsVision.value == true)

        vm.selectedProviderFilter = .mlx
        #expect(vm.filteredModels.map(\.modelID) == ["qwen3.5-0.8b-mlx"])
        vm.selectedProviderFilter = nil
        #expect(vm.filteredModels.count == 3)

        vm.activeGGUFPorts = ["z-model": 18082, "a-model": 18080, "m-model": 18081]
        #expect(vm.sortedActiveGGUFPorts == ["a-model::18080", "m-model::18081", "z-model::18082"])

        vm.pendingConfirmation = .clearEmbeddingIndex
        #expect(vm.pendingConfirmation?.id == "clearEmbeddingIndex")
        #expect(vm.pendingConfirmation?.confirmLabel == "Clear Index")
        vm.pendingConfirmation = .unloadAllModels
        #expect(vm.pendingConfirmation?.id == "unloadAllModels")
        #expect(vm.pendingConfirmation?.confirmLabel == "Unload All Models")
    }

    @Test("DocumentSettingsController isolates chat vs notebook drawers, saves sparse Null-Means-Cascade presets, enforces Override-Box prompt rule, and supports presetForNewDoc")
    func testDocumentSettingsIsolationSparsePresetsOverrideBoxAndNewDoc() async throws {
        UserDefaults.standard.removeObject(forKey: "heylook.draftPrompt")
        let sandbox = try TestTempSandbox(prefix: "heylook_doc_settings")
        defer {
            sandbox.cleanup()
            UserDefaults.standard.removeObject(forKey: "heylook.draftPrompt")
        }

        let vm = try Self.makeIsolatedViewModel(sandbox: sandbox)
        let modelDesc = EngineDescription(
            modelID: "think-mlx",
            provider: .mlx,
            modelPath: sandbox.modelsDir.appendingPathComponent("think-mlx").path,
            supportsThinking: ProvenanceField(value: true, provenance: .derived)
        )
        vm.registeredModels = [modelDesc]
        vm.selectedModelID = modelDesc.modelID

        // 1. Separate chatSettings vs notebookSettings isolation across surface switch
        vm.chatSettings.systemPrompt = "Chat system prompt"
        vm.chatSettings.temperatureOverride = 0.45
        vm.activeSurface = .notebook
        #expect(vm.activeDocumentSettings === vm.notebookSettings)
        #expect(vm.notebookSettings.systemPrompt.isEmpty)
        #expect(vm.notebookSettings.temperatureOverride == nil)
        vm.notebookSettings.systemPrompt = "Notebook prompt"
        vm.notebookSettings.topPOverride = 0.7
        vm.activeSurface = .chat
        #expect(vm.chatSettings.systemPrompt == "Chat system prompt")
        #expect(vm.chatSettings.temperatureOverride == 0.45)

        // 2. Sparse preset save (`Null-Means-Cascade`): effective defaults (`maxTokens = 16384`, `topP = 0.95`) stay nil in saved preset!
        vm.chatSettings.hydrateDocument(
            id: "conv-1",
            systemPrompt: "Keep existing prompt",
            params: SamplerSettings(temperature: 0.3),
            appliedPresetID: nil,
            activeModel: modelDesc
        )
        let saved = try #require(vm.chatSettings.saveAsPreset(name: "Concise Sparse", scope: "sampler"))
        #expect(saved.settings.temperature == 0.3)
        #expect(saved.settings.maxTokens == nil)
        #expect(saved.settings.topP == nil)
        #expect(saved.systemPrompt == nil)

        // 3. Override-Box rule:
        //    (a) Sampler-only preset with empty prompt preserves existing systemPrompt & reports zero drift
        let appliedImmediate = vm.chatSettings.applyPreset(saved)
        #expect(appliedImmediate == true)
        #expect(vm.chatSettings.systemPrompt == "Keep existing prompt")
        #expect(vm.chatSettings.pendingPromptReplacement == nil)
        let drift = vm.chatSettings.driftReport()
        #expect(drift.promptChanged == false)
        #expect(drift.hasChanges == false)

        //    (b) Update preset saves sparse delta cleanly; stale-timestamp guard blocks overwriting externally updated preset
        vm.chatSettings.temperatureOverride = 0.25
        let updated = try #require(vm.chatSettings.updateActivePreset())
        #expect(updated.settings.temperature == 0.25)
        #expect(updated.settings.maxTokens == nil)

        var bumped = updated
        bumped.description = "Edited elsewhere"
        bumped.updatedUnixMs = updated.updatedUnixMs + 10
        _ = try vm.store.updatePreset(id: updated.id, bumped)
        #expect(vm.chatSettings.updateActivePreset() == nil)

        //    (c) Full preset with DIFFERENT non-empty prompt arms `pendingPromptReplacement` instead of silently overwriting
        let fullPreset = try vm.store.createPreset(
            PresetRecord(
                name: "Strict Reviewer",
                scope: "full",
                settings: SamplerSettings(temperature: 0.1),
                systemPrompt: "Audit thoroughly."
            )
        )
        vm.notebookPresets.seedDefaultPresetsIfNeeded()
        let didImmediateApply = vm.chatSettings.applyPreset(fullPreset)
        #expect(didImmediateApply == false)
        #expect(vm.chatSettings.pendingPromptReplacement?.preset.id == fullPreset.id)
        #expect(vm.chatSettings.systemPrompt == "Keep existing prompt")
        vm.chatSettings.confirmPendingPromptReplacement(replacePrompt: true)
        #expect(vm.chatSettings.systemPrompt == "Audit thoroughly.")
        #expect(vm.chatSettings.pendingPromptReplacement == nil)

        // 4. `presetForNewDoc()` carries stored preset cleanly without leaking drifted edits, and carries pre-doc draft when no doc is open
        vm.chatSettings.temperatureOverride = 0.88 // drift after applying fullPreset
        let inherited = vm.chatSettings.presetForNewDoc()
        #expect(inherited.appliedPresetID == fullPreset.id)
        #expect(inherited.systemPrompt == "Audit thoroughly.")
        #expect(inherited.params?.temperature == 0.1)

        vm.chatSettings.hydrateDocument(id: nil, systemPrompt: "Scratch system draft", params: SamplerSettings(topK: 16), appliedPresetID: nil)
        let fromScratch = vm.chatSettings.presetForNewDoc()
        #expect(fromScratch.appliedPresetID == nil)
        #expect(fromScratch.systemPrompt == "Scratch system draft")
        #expect(fromScratch.params?.topK == 16)
        #expect(UserDefaults.standard.string(forKey: "heylook.draftPrompt") == nil)
    }

    @Test("ChatController distinguishes persisted-history vs staged media on model switch and hard-blocks send on incompatible staged media")
    func testChatControllerSwitchHistoryWarningAndHardMediaBlock() async throws {
        UserDefaults.standard.removeObject(forKey: "heylook.draftPrompt")
        let sandbox = try TestTempSandbox(prefix: "heylook_chat_ctrl")
        defer {
            sandbox.cleanup()
            UserDefaults.standard.removeObject(forKey: "heylook.draftPrompt")
        }

        let vm = try Self.makeIsolatedViewModel(sandbox: sandbox)
        let visionModel = EngineDescription(
            modelID: "vlm-mlx",
            provider: .mlx,
            modelPath: sandbox.modelsDir.appendingPathComponent("vlm").path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: true, provenance: .derived),
            supportsAudio: ProvenanceField(value: false, provenance: .derived)
        )
        let textOnlyModel = EngineDescription(
            modelID: "text-mlx",
            provider: .mlx,
            modelPath: sandbox.modelsDir.appendingPathComponent("text").path,
            isLoaded: true,
            supportsVision: ProvenanceField(value: false, provenance: .derived),
            supportsAudio: ProvenanceField(value: false, provenance: .derived)
        )
        await vm.scheduler.register(provider: TestStubInferenceProvider(description: visionModel), description: visionModel)
        await vm.scheduler.register(provider: TestStubInferenceProvider(description: textOnlyModel), description: textOnlyModel)
        vm.registeredModels = [visionModel, textOnlyModel]
        vm.selectedModelID = "vlm-mlx"

        let pngBytes = try TestFixtures.makeMinimalPNGData()

        // 1. Staged-only media in composer (empty history) does NOT trigger history-loss pendingModelSwitchTargetID
        #expect(vm.chat.stageMediaDataIfAllowed(kind: .image, mimeType: "image/png", filename: "staged.png", data: pngBytes) == true)
        vm.chat.selectGenerationModel("text-mlx")
        #expect(vm.chat.pendingModelSwitchTargetID == nil)
        #expect(vm.selectedModelID == "text-mlx")

        // 2. While on text-only model: staging new media fails AND existing staged image triggers warning & hard send block!
        #expect(vm.chat.stageMediaDataIfAllowed(kind: .image, mimeType: "image/png", filename: "blocked.png", data: pngBytes) == false)
        #expect(vm.chat.chatAttachmentCapabilityWarning != nil)
        vm.chat.draftPrompt = "Describe staged photo"
        await vm.chat.sendChatPrompt()
        #expect(vm.store.listMessagesCountSafe() == 0)

        // 3. Switch back cleanly, create a persisted conversation containing an image message, then try switching -> pendingModelSwitchTargetID MUST arm!
        vm.chat.stagedAttachments.removeAll()
        vm.chat.selectGenerationModel("vlm-mlx")
        let conv = ConversationRecord(title: "VLM Thread", defaultModelID: "vlm-mlx")
        try vm.store.upsertConversation(conv)
        _ = try vm.store.appendMessage(
            MessageRecord(
                conversationID: conv.id,
                position: 0,
                role: "user",
                content: "Look at this",
                contentBlocks: [
                    .text("Look at this"),
                    .media(MessagesRequest.MediaAttachment(kind: .image, mimeType: "image/png", base64Data: pngBytes.base64EncodedString())),
                ]
            )
        )
        vm.chat.selectConversation(conv.id)
        vm.chat.selectGenerationModel("text-mlx")
        #expect(vm.chat.pendingModelSwitchTargetID == "text-mlx")
        #expect(vm.selectedModelID == "vlm-mlx")
        vm.chat.cancelPendingModelSwitch()
        #expect(vm.chat.pendingModelSwitchTargetID == nil)
        #expect(vm.selectedModelID == "vlm-mlx")
    }

    @Test("EmbeddingsStudioController defaults to compareMatrix, promotes multi-block Compare items to Lab, and evaluates 8-case CoreGraphics sample set")
    func testEmbeddingsStudioComparePromotionAnd8CaseMultimodalLab() async throws {
        let sandbox = try TestTempSandbox(prefix: "heylook_emb_ctrl")
        defer { sandbox.cleanup() }

        let engine = TestDeterministicEmbeddingEngine(modelID: "embeddinggemma-2-300m-test")
        let vm = try Self.makeIsolatedViewModel(sandbox: sandbox, embeddingEngine: engine)
        let studio = vm.embeddings

        // 1. Default sub-mode is Compare matrix + 8-case built-in sample set loaded at init
        #expect(studio.embeddingsSubMode == .compareMatrix)
        #expect(studio.labSet.cases.count == 8)
        for labCase in studio.labSet.cases {
            #expect(!labCase.id.isEmpty)
            #expect(!labCase.positives.isEmpty)
            #expect(!labCase.negatives.isEmpty)
        }
        #expect(Set(studio.labSet.cases.map(\.id)).contains("text-to-image-triangle"))

        // 2. Build a multi-block Compare item, run Compare matrix, then promote pair -> standalone Lab set
        let item1ID = try #require(studio.compareRichItems.first?.id)
        studio.addBlockToCompareItem(itemID: item1ID, kind: .text)
        #expect(studio.compareRichItems[0].blocks.count == 2)
        await studio.runCompareMatrix()
        #expect(studio.compareMatrixScores.count >= 2)

        await studio.promoteComparePairToRetrievalLab()
        #expect(studio.embeddingsSubMode == .evalLab)
        #expect(studio.labSet.cases.count == 1)
        #expect(studio.labSet.cases.first?.id == "compare-case-1-to-2")

        // 3. Reload built-in 8-case CoreGraphics multimodal benchmark set & run Lab evaluation
        studio.loadBuiltInMultimodalSampleSet()
        #expect(studio.labSet.cases.count == 8)
        studio.clearLabConfigQueue()
        studio.addCurrentConfigToLabQueue()
        await studio.runMultimodalEmbeddingLab()
        #expect(studio.labSweepResults.count == 1)
        #expect(studio.labSweepResults.first?.caseResults.count == 8)
        #expect(studio.labSweepResults.first?.recallAt1 ?? -1 >= 0.0)
    }
}

private extension SQLiteStore {
    func listMessagesCountSafe() -> Int {
        guard let convs = try? listConversations() else { return 0 }
        return convs.reduce(0) { sum, c in
            sum + ((try? listMessages(conversationID: c.id).count) ?? 0)
        }
    }
}
