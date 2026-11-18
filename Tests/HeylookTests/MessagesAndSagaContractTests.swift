import Darwin
import Foundation
import Testing
@testable import HeylookCore
@testable import HeylookEngines
@testable import HeylookServer

@Suite("Messages Wire, Reasoning Protocols, Loopback HTTP/SSE & Conversation Saga Suite")
struct MessagesAndSagaContractTests {

    @Test("POST /v1/messages validates thinking shapes, 422 retired/misspelled fields, media source normalization, 400 unknown model, and 503 overload")
    func testMessagesValidationNormalizationAndAdmission() async throws {
        let harness = try TestRouterHarness(maxQueueDepth: 8)
        let textStub = await harness.registerStub(
            id: "text-only-mlx",
            provider: .mlx,
            supportsVision: false,
            supportsAudio: false,
            supportsThinking: true,
            chunks: [
                GenerationChunk(thinking: "plan ", tokenCount: 1, thinkingTokens: 1, promptTokens: 5, generationTPS: 30, promptTPS: 90, ttftMs: 11),
                GenerationChunk(text: "done", tokenCount: 2, contentTokens: 1, promptTokens: 5, generationTPS: 30, promptTPS: 90, ttftMs: 11, stopReason: .endTurn),
            ]
        )
        _ = await harness.registerStub(
            id: "vision-audio-mlx",
            provider: .mlx,
            supportsVision: true,
            supportsAudio: true,
            supportsThinking: true,
            chunks: [
                GenerationChunk(text: "saw media", tokenCount: 1, contentTokens: 1, promptTokens: 8, generationTPS: 25, promptTPS: 80, ttftMs: 14, stopReason: .endTurn),
            ]
        )

        // 1. Boolean thinking & {type: "enabled", budget_tokens: 128}
        let boolThinkResp = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: [
                "model": "text-only-mlx",
                "messages": [["role": "user", "content": "hi"]],
                "thinking": true,
                "stream": false,
            ]
        )
        #expect(boolThinkResp.status == 200)

        let objThinkResp = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: [
                "model": "text-only-mlx",
                "messages": [["role": "user", "content": "hi"]],
                "thinking": ["type": "enabled", "budget_tokens": 128],
                "show_special_tokens": false,
                "stream": false,
            ]
        )
        #expect(objThinkResp.status == 200)
        let lastReq = await textStub.invocations.last
        #expect(lastReq?.request.thinking == true)
        #expect(lastReq?.sampling.thinkingBudget == 128)

        // 2. 422 on every retired or misspelled field (`show_special_tokens: true` rejected while `false` above passed)
        let rejectedPayloads: [[String: Any]] = [
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "sampler": "balanced"],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "preset": "default"],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "chat_template_kwargs": ["enable_thinking": true]],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "show_special_tokens": true],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "vision_tokens": 256],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "tools": []],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "tool_choice": "auto"],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "logprobs": true],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "enable_thinking": true],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "max_new_tokens": 64],
            ["model": "text-only-mlx", "messages": [["role": "user", "content": "x"]], "system_prompt": "be terse"],
        ]
        for bad in rejectedPayloads {
            let r = try await harness.sendJSON("POST", "/v1/messages", object: bad)
            #expect(r.status == 422, "Expected 422 for payload keys \(Array(bad.keys))")
        }

        // 3. 400 on missing or unknown model
        let missingModel = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: ["messages": [["role": "user", "content": "hi"]]]
        )
        #expect(missingModel.status == 400 || missingModel.status == 422)

        let unknownModel = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: ["model": "nonexistent-model", "messages": [["role": "user", "content": "hi"]]]
        )
        #expect(unknownModel.status == 400 || unknownModel.status == 404)

        // 4. Nested + flat media normalization (" " text join) + 422 payload-less block + capability gate
        let pngB64 = try TestFixtures.makeMinimalPNGData().base64EncodedString()
        let nestedAndFlat = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: [
                "model": "vision-audio-mlx",
                "messages": [[
                    "role": "user",
                    "content": [
                        ["type": "text", "text": "alpha"],
                        ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": pngB64]],
                        ["type": "image", "source_type": "base64", "data": pngB64, "media_type": "image/png"],
                        ["type": "text", "text": "beta"],
                    ],
                ]],
                "stream": false,
            ]
        )
        #expect(nestedAndFlat.status == 200)
        // Capability gate refuses image on text-only model
        let gatedImg = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: [
                "model": "text-only-mlx",
                "messages": [[
                    "role": "user",
                    "content": [["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": pngB64]]],
                ]],
            ]
        )
        #expect(gatedImg.status == 422)

        // Normalized " " joined text check on decoded MessageTurn
        let decodedTurn = try JSONDecoder().decode(
            MessagesRequest.MessageTurn.self,
            from: JSONSerialization.data(withJSONObject: [
                "role": "user",
                "content": [
                    ["type": "text", "text": "hello"],
                    ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": pngB64]],
                    ["type": "text", "text": "world"],
                ],
            ])
        )
        #expect(decodedTurn.content == "hello world")
        #expect(decodedTurn.attachments.count == 1)

        let emptyMediaBlock = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: [
                "model": "vision-audio-mlx",
                "messages": [[
                    "role": "user",
                    "content": [["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": ""]]],
                ]],
            ]
        )
        #expect(emptyMediaBlock.status == 422)

        // 5. 503 model_overloaded + Retry-After: 1 + X-RateLimit-* when maxQueueDepth (1) is saturated
        let overloadedHarness = try TestRouterHarness(maxQueueDepth: 1)
        _ = await overloadedHarness.registerStub(
            id: "busy-model",
            chunks: [GenerationChunk(text: "slow", tokenCount: 1, contentTokens: 1, stopReason: .endTurn)],
            chunkDelayNanoseconds: 350_000_000
        )
        let firstTask = Task {
            try await overloadedHarness.sendJSON(
                "POST",
                "/v1/messages",
                object: ["model": "busy-model", "messages": [["role": "user", "content": "first"]], "stream": false]
            )
        }
        var enteredLease = false
        for _ in 0..<40 {
            if !overloadedHarness.registry.activeRequests().isEmpty {
                enteredLease = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(enteredLease)

        // Second request occupies the single queue slot (`maxQueueDepth = 1`)
        let queuedTask = Task {
            try await overloadedHarness.sendJSON(
                "POST",
                "/v1/messages",
                object: ["model": "busy-model", "messages": [["role": "user", "content": "queued"]], "stream": false]
            )
        }
        var queueFull = false
        for _ in 0..<40 {
            if await overloadedHarness.scheduler.metrics().gpuQueueDepth >= 1 {
                queueFull = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(queueFull)

        let thirdResp = try await overloadedHarness.sendJSON(
            "POST",
            "/v1/messages",
            object: ["model": "busy-model", "messages": [["role": "user", "content": "overloaded"]], "stream": false]
        )
        #expect(thirdResp.status == 503)
        let thirdErr = try TestFixtures.decodeJSONObject(thirdResp.body)
        #expect((thirdErr["error"] as? [String: Any])?["code"] as? String == "model_overloaded")
        #expect(RequestMiddleware.headerValue(named: "Retry-After", in: thirdResp.headers) == "1")
        #expect(RequestMiddleware.headerValue(named: "X-RateLimit-Remaining", in: thirdResp.headers) == "0")
        #expect(RequestMiddleware.headerValue(named: "X-RateLimit-Limit", in: thirdResp.headers) == "1")
        _ = try await firstTask.value
        _ = try await queuedTask.value
    }

    @Test("ReasoningParser handles all 4 protocols, StopSequenceFilter cuts reply only, and SSE contract emits thinking_delta, progress & ping")
    func testReasoningProtocolsStopFilterAndSSEWireContract() async throws {
        // 1. All 4 ReasoningParser protocols + StripSpecials
        var tagParser = ReasoningParser(format: .tagPair(open: "<think>", close: "</think>", prefillsThinking: false))
        let tA = tagParser.feed("<think>reason ")
        let tB = tagParser.feed("more</think>answer")
        #expect(tA.thinking + tB.thinking == "reason more")
        #expect(tA.text + tB.text == "answer")

        var gemmaParser = ReasoningParser(format: .gemmaChannel(openPrefix: "<|channel>", closeTag: "<channel|>"))
        let gA = gemmaParser.feed("<|channel>thought\nstep 1<channel|>Final")
        let gTail = gemmaParser.finalize()
        #expect(gA.thinking + gTail.thinking == "step 1")
        #expect(gA.text + gTail.text == "Final")

        var harmonyParser = ReasoningParser(format: .harmonyChannel)
        let hOut = harmonyParser.feed("<|channel|>analysis<|message|>plan here<|end|><|start|>assistant<|channel|>final<|message|>Done<|return|>")
        let hTail = harmonyParser.finalize()
        #expect((hOut.thinking + hTail.thinking) == "plan here")
        #expect((hOut.text + hTail.text).contains("Done"))

        var recipientParser = ReasoningParser(format: .recipientChannel)
        let rOut = recipientParser.feed("to=self<|message|>private thought<|eom|><|start|>assistant to=user<|message|>public reply<|eot|>")
        let rTail = recipientParser.finalize()
        var scrubber = StripSpecials(tokens: ["<|eot|>"])
        let cleanRecipient = scrubber.feed(rOut.text + rTail.text) + scrubber.finalize()
        #expect(rOut.thinking + rTail.thinking == "private thought")
        #expect(cleanRecipient == "public reply")

        // 2. StopSequenceFilter inside StreamingPipeline cuts ONLY visible reply (NEVER thinking!) and echoes `stop_sequence`
        let harness = try TestRouterHarness()
        _ = await harness.registerStub(
            id: "sse-model",
            supportsThinking: true,
            chunks: [
                GenerationChunk(thinking: "thinking contains STOP_HERE harmlessly ", tokenCount: 1, thinkingTokens: 1, promptTokens: 4),
                GenerationChunk(text: "Visible answerSTOP_HEREtrailing_cut", tokenCount: 2, contentTokens: 1, promptTokens: 4, generationTPS: 50, promptTPS: 100, ttftMs: 9.0, stopReason: .endTurn),
            ],
            prefillSteps: [(processed: 2, total: 4), (processed: 4, total: 4)]
        )

        let collector = SSEChunkCollector()
        let sseResp = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            headers: ["X-Request-ID": "req_sse_probe"],
            object: [
                "model": "sse-model",
                "messages": [["role": "user", "content": "hi"]],
                "stop_sequences": ["STOP_HERE"],
                "thinking": true,
                "stream": true,
            ],
            onSSEFrame: { frame in
                await collector.append(frame)
            }
        )
        #expect(sseResp.status == 200)
        let rawSSE = await collector.joined()
        let frames = TestFixtures.parseSSEFrames(rawSSE)

        let progressFrames = frames.filter { $0.event == "heylook_progress" }
        #expect(progressFrames.count == 2)

        let deltaObjs = try frames
            .filter { $0.event == "content_block_delta" }
            .compactMap { try $0.jsonObject()?["delta"] as? [String: Any] }
        let thinkingDeltas = deltaObjs.filter { $0["type"] as? String == "thinking_delta" }
        #expect(!thinkingDeltas.isEmpty)
        for td in thinkingDeltas {
            #expect(td["thinking"] as? String == td["text"] as? String)
            #expect((td["thinking"] as? String ?? "").contains("STOP_HERE"))
        }

        let textDeltas = deltaObjs
            .filter { $0["type"] as? String == "text_delta" }
            .compactMap { $0["text"] as? String }
            .joined()
        #expect(textDeltas == "Visible answer")

        let messageDeltaFrame = try #require(frames.first(where: { $0.event == "message_delta" }))
        let messageDeltaObj = try #require(try messageDeltaFrame.jsonObject())
        let mdDelta = try #require(messageDeltaObj["delta"] as? [String: Any])
        #expect(mdDelta["stop_reason"] as? String == "stop_sequence")
        #expect(mdDelta["stop_sequence"] as? String == "STOP_HERE")

        let messageStopFrame = try #require(frames.first(where: { $0.event == "message_stop" }))
        let messageStopObj = try #require(try messageStopFrame.jsonObject())
        #expect(messageStopObj["performance"] is [String: Any])

        // 3. Non-stream performance omits unmeasured (0 / nil) throughput metrics rather than fabricating 0.0
        _ = await harness.registerStub(
            id: "zero-rate-model",
            chunks: [GenerationChunk(text: "ok", tokenCount: 1, contentTokens: 1, generationTPS: 0, promptTPS: 0, stopReason: .endTurn)]
        )
        let zeroRateResp = try await harness.sendJSON(
            "POST",
            "/v1/messages",
            object: ["model": "zero-rate-model", "messages": [["role": "user", "content": "hi"]], "stream": false]
        )
        #expect(zeroRateResp.status == 200)
        let zeroPerf = (try TestFixtures.decodeJSONObject(zeroRateResp.body)["performance"] as? [String: Any])
        #expect(zeroPerf?["generation_tps"] == nil || zeroPerf?["generation_tps"] is NSNull)

        // 4. SSEConnectionStreamer 5.0s default ping constant + deterministic ping emission during idle wait
        #expect(SSEConnectionStreamer.defaultKeepaliveIntervalSeconds == 5.0)
        actor ByteAccumulator {
            var chunks: [String] = []
            func append(_ d: Data) {
                if let s = String(data: d, encoding: .utf8) { chunks.append(s) }
            }
            func joined() -> String { chunks.joined() }
        }
        let acc = ByteAccumulator()
        let streamer = SSEConnectionStreamer(
            requestID: "req_ping_probe",
            keepaliveIntervalSeconds: 0.06,
            sendBytes: { d in await acc.append(d) },
            closeConnection: {}
        )
        await streamer.sendFrame("event: message_start\ndata: {}\n\n")
        try await Task.sleep(nanoseconds: 160_000_000)
        await streamer.finish()
        let streamedText = await acc.joined()
        #expect(streamedText.contains("event: ping\ndata: {\"type\":\"ping\"}"))

        // 5. DELETE /v1/requests/{id} (422 malformed, 404 absent) & POST /v1/requests/{id}/reasoning-end
        let badCancel = await harness.send("DELETE", "/v1/requests/bad$id")
        #expect(badCancel.status == 422)
        let missingCancel = await harness.send("DELETE", "/v1/requests/unknown_req_99")
        #expect(missingCancel.status == 404)
    }

    @Test("HTTPConnectionHandler serves HTTP/1.1 Keep-Alive buffer reuse, Transfer-Encoding chunked, greedy *model_id, constant-time 401 admin token, Host 403, Request-ID replacement, and chunked SSE framing")
    func testHTTPTransportKeepAliveChunkedGreedyModelAuthHostRequestIDAndSSE() async throws {
        let harness = try TestRouterHarness(
            adminToken: "secret-admin-token-42",
            allowedHosts: ["localhost"],
            bindHost: "0.0.0.0"
        )
        _ = await harness.registerStub(
            id: "org/live-model",
            provider: .mlx,
            isLoaded: false,
            chunks: [
                GenerationChunk(text: "tcp-stream-", tokenCount: 1, contentTokens: 1, promptTokens: 3),
                GenerationChunk(text: "complete", tokenCount: 2, contentTokens: 1, promptTokens: 3, generationTPS: 40, promptTPS: 80, ttftMs: 8.0, stopReason: .endTurn),
            ]
        )

        // 1. Multi-request HTTP/1.1 Keep-Alive buffer reuse inside ConnectionBufferActor:
        //    Req 1: greedy *model_id load + invalid X-Request-ID silent replacement (`bad id with spaces!` -> `req-...`)
        //    Req 2: Transfer-Encoding: chunked POST /v1/messages pipelined immediately after Req 1
        //    (Directly exercises Data slice index normalization on `buffer.dropFirst(consumedBytes)`!)
        let req1Raw = [
            "POST /v1/models/org/live-model/load HTTP/1.1",
            "Host: 127.0.0.1:8000",
            "X-Request-ID: bad id with spaces!",
            "Content-Length: 0",
            "Connection: keep-alive",
            "",
            "",
        ].joined(separator: "\r\n")

        let msgBody = #"{"model":"org/live-model","messages":[{"role":"user","content":"chunked"}],"stream":false}"#
        let mid = msgBody.index(msgBody.startIndex, offsetBy: msgBody.count / 2)
        let partA = String(msgBody[..<mid])
        let partB = String(msgBody[mid...])
        let chunkedPayload =
            "\(String(partA.utf8.count, radix: 16))\r\n\(partA)\r\n" +
            "\(String(partB.utf8.count, radix: 16))\r\n\(partB)\r\n" +
            "0\r\n\r\n"
        let req2Raw = [
            "POST /v1/messages HTTP/1.1",
            "Host: localhost:8000",
            "Content-Type: application/json",
            "Transfer-Encoding: chunked",
            "Connection: close",
            "",
            "",
        ].joined(separator: "\r\n") + chunkedPayload

        let connActor = HTTPConnectionHandler.ConnectionBufferActor()
        await connActor.append(Data((req1Raw + req2Raw).utf8), isComplete: false, hadError: false)

        let parsed1 = try #require(await connActor.nextRequest().request)
        #expect(parsed1.method == "POST")
        #expect(parsed1.path == "/v1/models/org/live-model/load")
        #expect(parsed1.wantsKeepAlive == true)
        let res1 = await harness.router.handle(
            method: parsed1.method,
            path: parsed1.path,
            queryItems: parsed1.queryItems,
            headers: parsed1.headers,
            body: parsed1.body
        )
        #expect(res1.status == 200)
        let replacedID = RequestMiddleware.headerValue(named: "X-Request-ID", in: res1.headers) ?? ""
        #expect(!replacedID.isEmpty && !replacedID.contains("bad"))
        let res1Obj = try TestFixtures.decodeJSONObject(res1.body)
        #expect(res1Obj["model_id"] as? String == "org/live-model")

        let parsed2 = try #require(await connActor.nextRequest().request)
        #expect(parsed2.method == "POST")
        #expect(parsed2.path == "/v1/messages")
        #expect(parsed2.wantsKeepAlive == false)
        let res2 = await harness.router.handle(
            method: parsed2.method,
            path: parsed2.path,
            queryItems: parsed2.queryItems,
            headers: parsed2.headers,
            body: parsed2.body
        )
        #expect(res2.status == 200)
        let res2Obj = try TestFixtures.decodeJSONObject(res2.body)
        let contentArr = try #require(res2Obj["content"] as? [[String: Any]])
        #expect(contentArr.first?["text"] as? String == "tcp-stream-complete")

        // 2. Constant-time 401 admin token guard + 403 DNS-rebinding Host check via dispatchRawHTTPBuffer
        actor WireCapture {
            var chunks: [Data] = []
            func append(_ d: Data) { chunks.append(d) }
            func text() -> String { String(decoding: chunks.reduce(into: Data(), +=), as: UTF8.self) }
        }

        let unauthRaw = Data("GET /v1/admin/models HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nContent-Length: 0\r\n\r\n".utf8)
        let unauthCap = WireCapture()
        await HeylookHTTPServer.dispatchRawHTTPBuffer(
            unauthRaw,
            router: harness.router,
            sendBytes: { d in await unauthCap.append(d) },
            closeConnection: {}
        )
        let unauthWire = await unauthCap.text()
        #expect(unauthWire.hasPrefix("HTTP/1.1 401 Unauthorized"))

        let authResp = await harness.send(
            "GET",
            "/v1/admin/models",
            headers: ["Host": "127.0.0.1:8000", "X-Heylook-Admin-Token": "secret-admin-token-42"]
        )
        #expect(authResp.status == 200)
        #expect(RequestMiddleware.constantTimeEqual("secret-admin-token-42", "secret-admin-token-42") == true)
        #expect(RequestMiddleware.constantTimeEqual("secret-admin-token-42", "secret-admin-token-99") == false)
        #expect(RequestMiddleware.constantTimeEqual("short", "longer-token") == false)

        let badHostRaw = Data("GET /v1/models HTTP/1.1\r\nHost: evil.attacker.example:8000\r\nContent-Length: 0\r\n\r\n".utf8)
        let badHostCap = WireCapture()
        await HeylookHTTPServer.dispatchRawHTTPBuffer(
            badHostRaw,
            router: harness.router,
            sendBytes: { d in await badHostCap.append(d) },
            closeConnection: {}
        )
        let badHostWire = await badHostCap.text()
        #expect(badHostWire.hasPrefix("HTTP/1.1 403 Forbidden"))

        // 3. Direct chunked Transfer-Encoding SSE response framing via HeylookHTTPServer.dispatchRawHTTPBuffer
        let sseBody = #"{"model":"org/live-model","messages":[{"role":"user","content":"stream"}],"stream":true}"#
        let sseRaw = Data((
            "POST /v1/messages HTTP/1.1\r\n" +
            "Host: 127.0.0.1:8000\r\n" +
            "Content-Type: application/json\r\n" +
            "Content-Length: \(sseBody.utf8.count)\r\n\r\n" +
            sseBody
        ).utf8)
        let sseCap = WireCapture()
        await HeylookHTTPServer.dispatchRawHTTPBuffer(
            sseRaw,
            router: harness.router,
            sendBytes: { d in await sseCap.append(d) },
            closeConnection: {}
        )
        let sseWire = await sseCap.text()
        #expect(sseWire.contains("HTTP/1.1 200 OK"))
        #expect(sseWire.contains("Transfer-Encoding: chunked"))
        #expect(sseWire.contains("event: message_start"))
        #expect(sseWire.contains("tcp-stream-"))
        #expect(sseWire.contains("event: message_stop"))
        #expect(sseWire.hasSuffix("0\r\n\r\n"))
    }

    @Test("ConversationOrchestrator enforces atomic tail preservation, open-thinking resume, anchor model_id preservation, user-role continue, 409 lock, and media_blobs isolation")
    func testConversationSagaTailPreservationThinkingResumeLocksAndMediaBlobs() async throws {
        let harness = try TestRouterHarness()
        let alphaDir = harness.sandbox.modelsDir.appendingPathComponent("model-alpha", isDirectory: true)
        let betaDir = harness.sandbox.modelsDir.appendingPathComponent("model-beta", isDirectory: true)
        try FileManager.default.createDirectory(at: alphaDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: betaDir, withIntermediateDirectories: true)
        let thinkTokCfg = #"{"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}{% if enable_thinking %}<think></think>{% endif %}"}"#
        try thinkTokCfg.write(to: alphaDir.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        try thinkTokCfg.write(to: betaDir.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)

        let stubAlpha = await harness.registerStub(
            id: "model-alpha",
            provider: .mlx,
            modelPath: alphaDir.path,
            supportsVision: true,
            supportsThinking: true,
            chunks: [
                GenerationChunk(text: "resumed_text", tokenCount: 1, contentTokens: 1, promptTokens: 4, generationTPS: 30, promptTPS: 80, ttftMs: 10, stopReason: .endTurn),
            ]
        )
        let stubBeta = await harness.registerStub(
            id: "model-beta",
            provider: .mlx,
            modelPath: betaDir.path,
            supportsVision: false,
            supportsThinking: true,
            chunks: [
                GenerationChunk(text: "beta_reply", tokenCount: 1, contentTokens: 1, promptTokens: 4, generationTPS: 30, promptTPS: 80, ttftMs: 10, stopReason: .endTurn),
            ],
            chunkDelayNanoseconds: 250_000_000
        )

        let pngBytes = try TestFixtures.makeMinimalPNGData()
        let convResp = try await harness.sendJSON(
            "POST",
            "/v1/conversations",
            object: ["title": "Saga Thread", "model_id": "model-alpha"]
        )
        #expect(convResp.status == 201)
        let convID = try #require(TestFixtures.decodeJSONObject(convResp.body)["id"] as? String)

        // 1. Store media in user message -> 64-hex SHA-256 media_blobs URL relocation + nosniff + cross-conv isolation
        let userMsgResp = try await harness.sendJSON(
            "POST",
            "/v1/conversations/\(convID)/messages",
            object: [
                "role": "user",
                "content": [
                    ["type": "text", "text": "look"],
                    ["type": "image", "source_type": "base64", "data": pngBytes.base64EncodedString(), "media_type": "image/png"],
                ],
            ]
        )
        #expect(userMsgResp.status == 201)
        let userMsgObj = try TestFixtures.decodeJSONObject(userMsgResp.body)
        let userBlocks = try #require(userMsgObj["content_blocks"] as? [[String: Any]])
        let imgBlock = try #require(userBlocks.last)
        let imgSource = try #require(imgBlock["source"] as? [String: Any])
        #expect(imgSource["type"] as? String == "url")
        let mediaURLPath = try #require(imgSource["url"] as? String)
        #expect(imgBlock["type"] as? String == "image")
        #expect(mediaURLPath.hasPrefix("/v1/conversations/\(convID)/media/"))
        let mediaID = URL(fileURLWithPath: mediaURLPath).lastPathComponent
        #expect(mediaID.count == 64)

        let blobFetch = await harness.send("GET", mediaURLPath)
        #expect(blobFetch.status == 200)
        #expect(blobFetch.body == pngBytes)
        #expect(RequestMiddleware.headerValue(named: "X-Content-Type-Options", in: blobFetch.headers) == "nosniff")

        // Cross-conversation blob fetch must return 404
        let conv2Resp = try await harness.sendJSON("POST", "/v1/conversations", object: ["title": "Other"])
        let conv2ID = try #require(TestFixtures.decodeJSONObject(conv2Resp.body)["id"] as? String)
        let crossFetch = await harness.send("GET", "/v1/conversations/\(conv2ID)/media/\(mediaID)")
        #expect(crossFetch.status == 404)

        // Clone copies media_blobs cleanly into cloned conversation scope
        let cloneResp = try await harness.sendJSON("POST", "/v1/conversations/\(convID)/clone", object: [String: Any]())
        #expect(cloneResp.status == 201)
        let cloneID = try #require(TestFixtures.decodeJSONObject(cloneResp.body)["id"] as? String)
        let cloneBlob = await harness.send("GET", "/v1/conversations/\(cloneID)/media/\(mediaID)")
        #expect(cloneBlob.status == 200)
        #expect(cloneBlob.body == pngBytes)

        // 2. Open-thinking assistant anchor + trailing user turn -> regenerate/continue atomic tail preservation on mid-stream error!
        let basePos = try #require(userMsgObj["position"] as? Int)
        let openAsst = try harness.store.appendMessage(
            MessageRecord(
                conversationID: convID,
                position: basePos + 1,
                role: "assistant",
                content: "",
                thinking: "open thought prefix ",
                modelID: "model-alpha"
            )
        )
        let tailTurn = try harness.store.appendMessage(
            MessageRecord(conversationID: convID, position: basePos + 2, role: "user", content: "follow up question")
        )
        _ = tailTurn

        // Configure failure mid-stream during regenerate of `openAsst` -> tail (`basePos + 2`) MUST NOT be wiped!
        await stubAlpha.setMidStreamFailure(after: 0, error: ProviderExecutionError.runtimeUnavailable(modelID: "model-alpha", detail: "transient boom"))
        let failedRegen = try await harness.sendJSON(
            "POST",
            "/v1/conversations/\(convID)/generate",
            object: ["mode": "regenerate", "message_id": openAsst.id]
        )
        #expect(failedRegen.status == 200)
        let afterFailMsgs = try harness.store.listMessages(conversationID: convID)
        #expect(afterFailMsgs.count == 3)
        #expect(afterFailMsgs.last?.content == "follow up question")

        // Clear failure -> open-thinking resume (`startsInThinking = true`) closes thinking then commits reply,
        // preserving anchor `model_id` ("model-alpha") even when continue requests "model-beta"!
        await stubAlpha.setMidStreamFailure(after: nil, error: nil)
        await stubBeta.updateChunks([
            GenerationChunk(text: "extra thinking</think>final answer", tokenCount: 2, contentTokens: 2, promptTokens: 4, generationTPS: 35, promptTPS: 90, ttftMs: 9, stopReason: .endTurn),
        ])
        let continueResp = try await harness.sendJSON(
            "POST",
            "/v1/conversations/\(convID)/generate",
            object: [
                "mode": "continue",
                "message_id": openAsst.id,
                "model": "model-beta",
            ]
        )
        #expect(continueResp.status == 200)
        let afterContinueMsgs = try harness.store.listMessages(conversationID: convID)
        #expect(afterContinueMsgs.count == 2)
        let updatedAnchor = try #require(afterContinueMsgs.last)
        #expect(updatedAnchor.id == openAsst.id)
        #expect(updatedAnchor.modelID == "model-alpha")
        #expect(updatedAnchor.thinking == "open thought prefix extra thinking")
        #expect(updatedAnchor.content == "final answer")

        // 3. `continue` from user role works cleanly on MLX, whereas GGUF rejects user-role `continue_final_message=true` with `invalid_request_error`
        let userAnchor = try harness.store.appendMessage(
            MessageRecord(conversationID: convID, position: basePos + 2, role: "user", content: "continue from user")
        )
        let mlxUserCont = try await harness.sendJSON(
            "POST",
            "/v1/conversations/\(convID)/generate",
            object: ["mode": "continue", "message_id": userAnchor.id]
        )
        #expect(mlxUserCont.status == 200)
        #expect((try harness.store.listMessages(conversationID: convID)).count == 4)

        let ggufDesc = EngineDescription(modelID: "test-gguf", provider: .gguf, modelPath: harness.sandbox.modelsDir.path)
        let ggufProvider = GGUFSubprocessProvider(description: ggufDesc)
        let ggufErrStream = await ggufProvider.generate(
            requestID: "req_gguf_cont_err",
            request: MessagesRequest(
                model: "test-gguf",
                messages: [.init(role: "user", content: "trailing user")],
                metadata: ["continue_final_message": "true"]
            ),
            effectiveSampling: SamplerSettings(),
            registry: harness.registry
        )
        let ggufCollector = SSEChunkCollector()
        _ = await SSEWirePipeline.consumeStreamAndFormatSSE(
            stream: ggufErrStream,
            modelID: "test-gguf",
            requestID: "req_gguf_cont_err",
            reasoningFormat: .none,
            startsInThinking: false,
            thinkingEnabled: false,
            stopSequences: [],
            requestStartedAt: Date(),
            registry: harness.registry,
            onEvent: { frame in await ggufCollector.append(frame) }
        )
        let ggufSSE = await ggufCollector.joined()
        let ggufErrParsed = try #require(TestFixtures.parseSSEFrames(ggufSSE).first(where: { $0.event == "error" }))
        let ggufErrFrame = try #require(try ggufErrParsed.jsonObject())
        #expect((ggufErrFrame["error"] as? [String: Any])?["type"] as? String == "invalid_request_error")

        // 4. 409 generation_in_progress lock blocks ALL conversation & message mutations while generation holds lock
        let busyTask = Task {
            try await harness.sendJSON(
                "POST",
                "/v1/conversations/\(convID)/generate",
                object: [
                    "mode": "append",
                    "content": "hold lock",
                    "model": "model-beta",
                ]
            )
        }
        var lockAcquired = false
        for _ in 0..<40 {
            if harness.registry.isConversationGenerating(convID) {
                lockAcquired = true
                break
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(lockAcquired)

        #expect(await harness.send("DELETE", "/v1/conversations/\(convID)").status == 409)
        #expect(try await harness.sendJSON("POST", "/v1/conversations/\(convID)/clone", object: [String: Any]()).status == 409)
        #expect(try await harness.sendJSON("POST", "/v1/conversations/\(convID)/messages", object: ["role": "user", "content": "x"]).status == 409)
        #expect(try await harness.sendJSON("PATCH", "/v1/conversations/\(convID)/messages/\(openAsst.id)", object: ["content": "x"]).status == 409)
        #expect(await harness.send("DELETE", "/v1/conversations/\(convID)/messages/\(openAsst.id)").status == 409)
        #expect(await harness.send("DELETE", "/v1/conversations/\(convID)/messages", queryItems: [URLQueryItem(name: "after", value: "0")]).status == 409)

        _ = try await busyTask.value
        _ = await stubBeta.unloadCount
    }

    @Test("SQLiteStore DELETE ?after=, non-compacting single DELETE, unseeded presets, performance profiles, ObservabilityDiskLog, and WAV 48kHz->16kHz + ASR gates")
    func testSQLiteDeleteSemanticsPresetsProfileObservabilityAndASRGates() async throws {
        let harness = try TestRouterHarness()

        // 1. Fresh SQLiteStore starts with 0 unseeded presets, and user presets survive POST /v1/data/clear
        #expect(try harness.store.listPresets().isEmpty)
        let createPreset = try await harness.sendJSON(
            "POST",
            "/v1/presets",
            object: ["name": "Custom Test Preset", "scope": "sampler", "params": ["temperature": 0.4]]
        )
        #expect(createPreset.status == 201)

        // 2. Non-compacting single DELETE [0, 1, 2] -> delete 1 -> [0, 2]; DELETE ?after=0 preserves pos 0
        let conv = ConversationRecord(title: "Delete Check", defaultModelID: "stub")
        try harness.store.upsertConversation(conv)
        let m0 = try harness.store.appendMessage(MessageRecord(conversationID: conv.id, position: 0, role: "user", content: "p0"))
        let m1 = try harness.store.appendMessage(MessageRecord(conversationID: conv.id, position: 1, role: "assistant", content: "p1"))
        let m2 = try harness.store.appendMessage(MessageRecord(conversationID: conv.id, position: 2, role: "user", content: "p2"))

        let delSingle = await harness.send("DELETE", "/v1/conversations/\(conv.id)/messages/\(m1.id)")
        #expect(delSingle.status == 200)
        let afterSingle = try harness.store.listMessages(conversationID: conv.id)
        #expect(afterSingle.map(\.position) == [0, 2])
        #expect(afterSingle.map(\.id) == [m0.id, m2.id])

        let delAfter = await harness.send(
            "DELETE",
            "/v1/conversations/\(conv.id)/messages",
            queryItems: [URLQueryItem(name: "after", value: "0")]
        )
        #expect(delAfter.status == 200)
        let afterTruncate = try harness.store.listMessages(conversationID: conv.id)
        #expect(afterTruncate.map(\.position) == [0])
        #expect(afterTruncate.first?.id == m0.id)

        // /v1/data/clear removes conversations & notebooks but keeps user presets
        let clearResp = await harness.send("POST", "/v1/data/clear")
        #expect(clearResp.status == 200)
        #expect(try harness.store.listConversations().isEmpty)
        #expect(try harness.store.listPresets().count == 1)

        // 3. Performance profile 1h|6h|24h|7d (400 invalid range) + timing_breakdown/cache/trends
        try harness.store.upsertMessageStats(
            MessageStatsRecord(
                messageID: "m_stat_1",
                conversationID: "c_stat",
                modelID: "model-x",
                promptTokens: 32,
                outputTokens: 64,
                cachedTokens: 16,
                promptTPS: 220.0,
                generationTPS: 55.0,
                ttftMs: 120.0,
                requestDurationMs: 1200.0
            )
        )
        for validRange in ["1h", "6h", "24h", "7d"] {
            let profResp = await harness.send("GET", "/v1/performance/profile/\(validRange)")
            #expect(profResp.status == 200)
            let obj = try TestFixtures.decodeJSONObject(profResp.body)
            #expect(obj["timing_breakdown"] != nil)
            #expect(obj["cache"] != nil)
            #expect(obj["trends"] != nil)
        }
        #expect(await harness.send("GET", "/v1/performance/profile/99y").status == 400)

        // 4. Isolated ObservabilityDiskLog writes events & metrics JSONL streams cleanly
        let diskLog = ObservabilityDiskLog(level: "standard", logDirectory: harness.sandbox.rootURL.appendingPathComponent("logs", isDirectory: true))
        diskLog.recordEvent("turn_complete", tier: .events, minLevel: "standard", fields: ["tokens": .int(42)])
        diskLog.recordEvent("sample_metric", tier: .metrics, minLevel: "standard", fields: ["tps": .double(88.5)])
        diskLog.recordEvent("debug_only", tier: .events, minLevel: "debug", fields: ["skip": .bool(true)])
        let eventsTail = diskLog.readTail(stream: .events, limit: 10)
        let metricsTail = diskLog.readTail(stream: .metrics, limit: 10)
        #expect(eventsTail.count == 1)
        #expect(metricsTail.count == 1)
        #expect(eventsTail.first?["type"] == .string("turn_complete"))

        // 5. RIFF/WAVE FLLR 4 KB padding skip + 48kHz stereo -> 16kHz mono resampling + ASR availability & voice-generate 422
        let fllrWav = TestFixtures.makeWAVData(sampleRate: 48_000, channels: 2, durationSeconds: 0.25, fllrPaddingBytes: 4096)
        let decoded16k = await SpeechTranscriptionService.decodeAudioMonoPCM(from: fllrWav, targetSampleRate: 16_000)
        #expect(abs(decoded16k.count - 4_000) <= 4)
        let earlySlice = decoded16k.prefix(200)
        let earlyRMS = sqrt(earlySlice.reduce(0.0) { $0 + Double($1 * $1) } / Double(max(1, earlySlice.count)))
        #expect(earlyRMS > 0.15)

        let capObj = try TestFixtures.decodeJSONObject(await harness.send("GET", "/v1/capabilities").body)
        let endpointsObj = try #require(capObj["endpoints"] as? [String: Any])
        #expect(endpointsObj["messages"] != nil)
        #expect(endpointsObj["embeddings"] != nil)
        #expect((capObj["voice_asr"] as? [String: Any]) != nil)

        // Missing explicit ASR model returns 404
        let missingASR = try await harness.sendJSON(
            "POST",
            "/v1/audio/transcriptions",
            object: ["audio_base64": fllrWav.base64EncodedString(), "model": "nonexistent-asr"]
        )
        #expect(missingASR.status == 404)

        // When system SpeechTranscriber is available on host, empty/silent WAV on voice-generate returns 400/422 without saving an empty user turn
        if SpeechTranscriptionService.isSystemSpeechAvailable {
            _ = await harness.registerStub(id: "voice-chat-mlx", isLoaded: true)
            let convVoiceResp = try await harness.sendJSON("POST", "/v1/conversations", object: ["title": "Voice Conv", "model_id": "voice-chat-mlx"])
            let voiceConvID = try #require(TestFixtures.decodeJSONObject(convVoiceResp.body)["id"] as? String)
            let silentWAV = SpeechTranscriptionService.encodeCanonicalPCM16WAV(samples: [Float](repeating: 0, count: 3_200), sampleRate: 16_000)
            let voiceEmpty = try await harness.sendJSON(
                "POST",
                "/v1/conversations/\(voiceConvID)/voice-generate",
                object: [
                    "model": "voice-chat-mlx",
                    "audio_base64": silentWAV.base64EncodedString(),
                    "asr_model": SpeechTranscriptionService.systemSpeechModelID,
                ]
            )
            #expect(voiceEmpty.status == 400)
            #expect(try harness.store.listMessages(conversationID: voiceConvID).isEmpty)
        }
    }

    @Test("PrefixCacheTracker classifies cold, reused, system_changed, history_diverged, and token-ID branch truncation")
    func testPrefixCacheTrackerLifecycleAndTokenIDs() {
        var tracker = PrefixCacheTracker()
        let wordCount: @Sendable (String) -> Int = { max(1, $0.split(separator: " ").count) }

        let turn1 = [MessagesRequest.MessageTurn(role: "user", content: "hello world")]
        let cold = tracker.evaluate(system: "be helpful", messages: turn1, tokenCounter: wordCount)
        #expect(cold.outcome == .miss)
        #expect(cold.cause == "cold")
        #expect(cold.cachedTokens == 0)
        #expect(tracker.isWarm == true)

        let turn2 = [
            MessagesRequest.MessageTurn(role: "user", content: "hello world"),
            MessagesRequest.MessageTurn(role: "assistant", content: "hi there friend"),
            MessagesRequest.MessageTurn(role: "user", content: "follow up question"),
        ]
        let reused = tracker.evaluate(system: "be helpful", messages: turn2, tokenCounter: wordCount)
        #expect(reused.outcome == .reused)
        #expect(reused.cause == nil)
        #expect(reused.cachedTokens > 0)

        let sysChanged = tracker.evaluate(system: "new system prompt", messages: turn2, tokenCounter: wordCount)
        #expect(sysChanged.outcome == .miss)
        #expect(sysChanged.cause == "system_changed")

        let diverged = tracker.evaluate(
            system: "new system prompt",
            messages: [
                MessagesRequest.MessageTurn(role: "user", content: "edited root turn"),
                MessagesRequest.MessageTurn(role: "assistant", content: "different reply"),
                MessagesRequest.MessageTurn(role: "user", content: "branch"),
            ],
            tokenCounter: wordCount
        )
        #expect(diverged.outcome == .miss)
        #expect(diverged.cause == "history_diverged")

        let tokCold = tracker.evaluateTokenIDs([10, 20, 30, 40])
        #expect(tokCold.commonPrefixCount == 0)
        tracker.appendGeneratedTokenIDs([50, 60])
        let tokBranch = tracker.evaluateTokenIDs([10, 20, 30, 99, 100])
        #expect(tokBranch.commonPrefixCount == 3)
        #expect(tokBranch.truncatedTrailingCount == 3)
        #expect(tracker.branchTruncationsCount == 1)

        tracker.reset()
        #expect(tracker.isWarm == false)
        #expect(tracker.cachedTurnCount == 0)
    }
}
