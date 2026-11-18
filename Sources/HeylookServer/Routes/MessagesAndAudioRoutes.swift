import Foundation
import HeylookCore

public enum MessagesAndAudioRoutes {
    public static func register(on router: inout PatternRouter, service: WorkbenchService) {
        router.register("POST", "/v1/messages") { ctx in
            let req = try JSONDecoder().decode(MessagesRequest.self, from: ctx.body)
            if let response = try await service.createMessageResponse(
                request: req,
                requestID: ctx.requestID,
                cancellation: ctx.cancellationSignal,
                onSSEFrame: ctx.onSSEFrame
            ) {
                return try .json(response, status: 200)
            }
            return HeylookHTTPResponse(status: 200, headers: ["Content-Type": "text/event-stream"], body: Data())
        }

        router.register("POST", "/v1/audio/transcriptions") { ctx in
            let req = try JSONDecoder().decode(AudioTranscriptionRequest.self, from: ctx.body)
            let result = try await service.transcribeAudio(req, requestID: ctx.requestID)
            return try .json(result, status: 200)
        }

        router.register("POST", "/v1/conversations/:id/voice-generate") { ctx in
            guard let convID = ctx.param("id"), !convID.isEmpty else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let req = try JSONDecoder().decode(ConversationVoiceGenerateRequest.self, from: ctx.body)
            let result = try await service.voiceGenerate(
                conversationID: convID,
                request: req,
                requestID: ctx.requestID,
                cancellationSignal: ctx.cancellationSignal,
                onSSEFrame: ctx.onSSEFrame
            )
            return try .json(result, status: 200)
        }
    }
}
