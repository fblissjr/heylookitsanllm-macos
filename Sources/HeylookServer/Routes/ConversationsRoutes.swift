import Foundation
import HeylookCore

public enum ConversationsRoutes {
    public static func register(on router: inout PatternRouter, service: WorkbenchService) {
        router.register("GET", "/v1/conversations") { _ in
            let list = try await service.listConversations()
            return try .json(list, status: 200)
        }

        router.register("POST", "/v1/conversations") { ctx in
            let req = ctx.body.isEmpty
                ? try JSONDecoder().decode(ConversationCreateRequestDTO.self, from: Data("{}".utf8))
                : try JSONDecoder().decode(ConversationCreateRequestDTO.self, from: ctx.body)
            let created = try await service.createConversation(req)
            return try .json(created, status: 201)
        }

        router.register("GET", "/v1/conversations/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let detail = try await service.getConversation(id: id)
            return try .json(detail, status: 200)
        }

        router.registerMethods(["PUT", "PATCH"], "/v1/conversations/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let req = try JSONDecoder().decode(ConversationUpdateRequestDTO.self, from: ctx.body)
            let updated = try await service.updateConversation(id: id, req: req)
            return try .json(updated, status: 200)
        }

        router.register("DELETE", "/v1/conversations/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            try await service.deleteConversation(id: id)
            return try .jsonObject(["status": "deleted", "id": id], status: 200)
        }

        router.register("POST", "/v1/conversations/:id/clone") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let req = ctx.body.isEmpty
                ? try JSONDecoder().decode(ConversationCloneRequestDTO.self, from: Data("{}".utf8))
                : try JSONDecoder().decode(ConversationCloneRequestDTO.self, from: ctx.body)
            let cloned = try await service.cloneConversation(id: id, req: req)
            return try .json(cloned, status: 201)
        }

        router.register("POST", "/v1/conversations/:id/prompt") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let req = ctx.body.isEmpty
                ? PromptPreviewRequestPayload()
                : try JSONDecoder().decode(PromptPreviewRequestPayload.self, from: ctx.body)
            let preview = try await service.previewPrompt(conversationID: id, request: req)
            return try .json(preview, status: 200)
        }

        router.register("POST", "/v1/conversations/:id/generate") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let req = try JSONDecoder().decode(ConversationGenerateRequest.self, from: ctx.body)
            let saved = try await service.generateInConversation(
                conversationID: id,
                request: req,
                requestID: ctx.requestID,
                cancellationSignal: ctx.cancellationSignal,
                detachOnSubscriberExit: true,
                onSSEFrame: ctx.onSSEFrame
            )
            if ctx.onSSEFrame != nil {
                return HeylookHTTPResponse(status: 200, headers: ["Content-Type": "text/event-stream"], body: Data())
            }
            return try .json(saved, status: 200)
        }

        router.register("GET", "/v1/conversations/:id/messages") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let messages = try await service.listMessages(conversationID: id)
            return try .json(messages, status: 200)
        }

        router.register("POST", "/v1/conversations/:id/messages") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            let req = try JSONDecoder().decode(MessageCreateRequestDTO.self, from: ctx.body)
            let saved = try await service.addMessage(conversationID: id, req: req)
            return try .json(saved, status: 201)
        }

        router.registerMethods(["PUT", "PATCH"], "/v1/conversations/:id/messages/:msg_id") { ctx in
            guard let id = ctx.param("id"), let msgID = ctx.param("msg_id") else {
                throw WorkbenchError.badRequest("Missing conversation or message id")
            }
            let req = try JSONDecoder().decode(MessageUpdateRequestDTO.self, from: ctx.body)
            let updated = try await service.updateMessage(conversationID: id, messageID: msgID, req: req)
            return try .json(updated, status: 200)
        }

        router.register("DELETE", "/v1/conversations/:id/messages") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing conversation id")
            }
            guard let after = ctx.intQuery("after") else {
                throw WorkbenchError.unprocessable(
                    message: "Missing or invalid required query parameter 'after' (Int)",
                    field: "after"
                )
            }
            let deletedCount = try await service.deleteMessagesAfter(conversationID: id, afterPosition: after)
            return try .jsonObject([
                "deleted": deletedCount,
                "after_position": after,
            ], status: 200)
        }

        router.register("DELETE", "/v1/conversations/:id/messages/:msg_id") { ctx in
            guard let id = ctx.param("id"), let msgID = ctx.param("msg_id") else {
                throw WorkbenchError.badRequest("Missing conversation or message id")
            }
            try await service.deleteMessage(conversationID: id, messageID: msgID)
            return try .jsonObject(["deleted": 1, "id": msgID], status: 200)
        }

        router.register("GET", "/v1/conversations/:id/media/:media_id") { ctx in
            guard let id = ctx.param("id"), let mediaID = ctx.param("media_id") else {
                throw WorkbenchError.badRequest("Missing conversation or media id")
            }
            let blob = try await service.mediaBlob(conversationID: id, mediaID: mediaID)
            return HeylookHTTPResponse(
                status: 200,
                headers: [
                    "Content-Type": blob.mediaType,
                    "Cache-Control": "private, max-age=31536000, immutable",
                    "X-Content-Type-Options": "nosniff",
                ],
                body: blob.data
            )
        }
    }
}
