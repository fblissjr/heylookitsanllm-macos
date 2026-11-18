import Foundation
import HeylookCore
import HeylookEmbeddings

public enum EmbeddingsRoutes {
    public static func register(on router: inout PatternRouter, service: WorkbenchService) {
        router.register("POST", "/v1/embeddings") { ctx in
            let parsed = try decodeStrictRequest(from: ctx.body)
            let response = try await service.embed(request: parsed, requestID: ctx.requestID)
            return try .json(response, status: 200)
        }
    }

    public static func decodeStrictRequest(from data: Data) throws -> EmbeddingWireRequest {
        guard let rawObj = try? JSONSerialization.jsonObject(with: data),
              let root = rawObj as? [String: Any] else {
            throw WorkbenchError.unprocessable(message: "Request body must be a JSON object")
        }

        let allowedTopKeys: Set<String> = ["model", "input", "dimensions"]
        let extraTop = root.keys.filter { !allowedTopKeys.contains($0) }.sorted()
        if !extraTop.isEmpty {
            throw WorkbenchError.unprocessable(message: "Unknown field(s): \(extraTop.joined(separator: ", "))")
        }

        guard let model = root["model"] as? String,
              !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkbenchError.unprocessable(message: "'model' must be a non-empty string", field: "model")
        }

        var dimensions: Int? = nil
        if let rawDims = root["dimensions"], !(rawDims is NSNull) {
            guard let num = rawDims as? NSNumber,
                  CFGetTypeID(num) != CFBooleanGetTypeID(),
                  floor(num.doubleValue) == num.doubleValue else {
                throw WorkbenchError.unprocessable(message: "'dimensions' must be a positive integer", field: "dimensions")
            }
            let dimInt = num.intValue
            guard dimInt > 0 else {
                throw WorkbenchError.unprocessable(message: "'dimensions' must be > 0", field: "dimensions")
            }
            dimensions = dimInt
        }

        guard let rawInput = root["input"] else {
            throw WorkbenchError.unprocessable(message: "'input' is required", field: "input")
        }

        let rawItems: [Any]
        if let str = rawInput as? String {
            rawItems = [str]
        } else if let arr = rawInput as? [Any] {
            rawItems = arr
        } else {
            throw WorkbenchError.unprocessable(message: "'input' must be a string or list of items", field: "input")
        }

        guard !rawItems.isEmpty && rawItems.count <= EmbeddingRequestLimits.maxItems else {
            throw WorkbenchError.unprocessable(
                message: "'input' must contain 1...\(EmbeddingRequestLimits.maxItems) items",
                field: "input"
            )
        }

        var parsedItems: [EmbeddingWireRequest.Item] = []
        parsedItems.reserveCapacity(rawItems.count)

        for (itemIdx, rawItem) in rawItems.enumerated() {
            if let str = rawItem as? String {
                guard str.count <= EmbeddingRequestLimits.maxTextChars else {
                    throw WorkbenchError.unprocessable(
                        message: "input[\(itemIdx)] exceeds \(EmbeddingRequestLimits.maxTextChars) characters"
                    )
                }
                parsedItems.append(
                    EmbeddingWireRequest.Item(
                        content: [.text(str)],
                        promptName: nil,
                        hasExplicitPromptName: false,
                        videoWindowsByBlockIndex: [:],
                        hasInvalidMediaMIME: nil
                    )
                )
                continue
            }

            guard let itemDict = rawItem as? [String: Any] else {
                throw WorkbenchError.unprocessable(
                    message: "input[\(itemIdx)] must be a string or content object"
                )
            }
            let allowedItemKeys: Set<String> = ["content", "prompt_name"]
            let extraItem = itemDict.keys.filter { !allowedItemKeys.contains($0) }.sorted()
            if !extraItem.isEmpty {
                throw WorkbenchError.unprocessable(
                    message: "input[\(itemIdx)] has unknown key(s): \(extraItem.joined(separator: ", "))"
                )
            }

            var promptName: String? = nil
            var hasExplicitPromptName = false
            if let rawPN = itemDict["prompt_name"] {
                hasExplicitPromptName = true
                if !(rawPN is NSNull) {
                    guard let s = rawPN as? String, s.count <= 128 else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].prompt_name must be a string <= 128 chars"
                        )
                    }
                    promptName = s
                }
            }

            guard let rawContent = itemDict["content"] as? [Any],
                  !rawContent.isEmpty,
                  rawContent.count <= EmbeddingRequestLimits.maxBlocksPerItem else {
                throw WorkbenchError.unprocessable(
                    message: "input[\(itemIdx)].content must have 1...\(EmbeddingRequestLimits.maxBlocksPerItem) blocks"
                )
            }

            var parts: [MessagesRequest.MessageTurn.ContentPart] = []
            var videoWindows: [Int: VideoWindowOptions] = [:]

            for (blockIdx, rawBlock) in rawContent.enumerated() {
                guard let blockDict = rawBlock as? [String: Any],
                      let blockType = blockDict["type"] as? String else {
                    throw WorkbenchError.unprocessable(
                        message: "input[\(itemIdx)].content[\(blockIdx)] requires string 'type'"
                    )
                }

                switch blockType {
                case "text":
                    let extraBlock = blockDict.keys.filter { $0 != "type" && $0 != "text" }.sorted()
                    if !extraBlock.isEmpty {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].content[\(blockIdx)] unknown key(s): \(extraBlock.joined(separator: ", "))"
                        )
                    }
                    guard let text = blockDict["text"] as? String else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].content[\(blockIdx)].text must be a string"
                        )
                    }
                    guard text.count <= EmbeddingRequestLimits.maxTextChars else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)] text exceeds \(EmbeddingRequestLimits.maxTextChars) characters"
                        )
                    }
                    parts.append(.text(text))

                case "image", "audio", "video":
                    let extraBlock = blockDict.keys.filter { $0 != "type" && $0 != "source" && $0 != "windows" }.sorted()
                    if !extraBlock.isEmpty {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].content[\(blockIdx)] unknown key(s): \(extraBlock.joined(separator: ", "))"
                        )
                    }
                    guard let srcDict = blockDict["source"] as? [String: Any] else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].content[\(blockIdx)].source is required"
                        )
                    }
                    let allowedSrc: Set<String> = ["type", "media_type", "data", "url"]
                    let extraSrc = srcDict.keys.filter { !allowedSrc.contains($0) }.sorted()
                    if !extraSrc.isEmpty {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].content[\(blockIdx)].source unknown key(s): \(extraSrc.joined(separator: ", "))"
                        )
                    }
                    guard let srcType = srcDict["type"] as? String,
                          srcType == "base64" || srcType == "url",
                          let mediaType = srcDict["media_type"] as? String else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)].content[\(blockIdx)].source requires valid type and media_type"
                        )
                    }
                    guard mediaType.lowercased().hasPrefix(blockType + "/") else {
                        throw WorkbenchError.unprocessable(
                            message: "input[\(itemIdx)]: media_type must match content block type '\(blockType)'"
                        )
                    }

                    let rawData = srcDict["data"] as? String
                    let rawURL = srcDict["url"] as? String
                    if srcType == "base64" {
                        guard let rawData, !rawData.isEmpty, srcDict["url"] == nil || srcDict["url"] is NSNull else {
                            throw WorkbenchError.unprocessable(
                                message: "input[\(itemIdx)]: base64 source requires non-empty data and no url"
                            )
                        }
                        parts.append(.media(.init(kind: blockType, mediaType: mediaType, dataBase64: rawData)))
                    } else {
                        guard let rawURL,
                              srcDict["data"] == nil || srcDict["data"] is NSNull,
                              let parsedURL = URL(string: rawURL),
                              let scheme = parsedURL.scheme?.lowercased(),
                              scheme == "http" || scheme == "https" else {
                            throw WorkbenchError.unprocessable(
                                message: "input[\(itemIdx)]: url source requires an http(s) url and no data"
                            )
                        }
                        parts.append(.media(.init(kind: blockType, mediaType: mediaType, dataBase64: "", url: rawURL)))
                    }

                    if let rawWindows = blockDict["windows"], !(rawWindows is NSNull) {
                        guard blockType == "video" else {
                            throw WorkbenchError.unprocessable(
                                message: "input[\(itemIdx)]: windows applies to video blocks only"
                            )
                        }
                        guard rawContent.count == 1 else {
                            throw WorkbenchError.unprocessable(
                                message: "input[\(itemIdx)]: an item with video windows must contain only that video block"
                            )
                        }
                        guard let winDict = rawWindows as? [String: Any] else {
                            throw WorkbenchError.unprocessable(message: "windows must be a JSON object")
                        }
                        let extraWin = winDict.keys.filter { $0 != "seconds" && $0 != "sampling" }.sorted()
                        if !extraWin.isEmpty {
                            throw WorkbenchError.unprocessable(
                                message: "windows unknown key(s): \(extraWin.joined(separator: ", "))"
                            )
                        }
                        var secs: Double? = nil
                        if let rawSecs = winDict["seconds"], !(rawSecs is NSNull) {
                            guard let num = rawSecs as? NSNumber,
                                  CFGetTypeID(num) != CFBooleanGetTypeID() else {
                                throw WorkbenchError.unprocessable(message: "windows.seconds must be > 0")
                            }
                            secs = num.doubleValue
                        }
                        let samplingRaw = winDict["sampling"] as? String
                        do {
                            let validatedWin = try VideoWindowOptions.validated(seconds: secs, samplingRaw: samplingRaw)
                            videoWindows[blockIdx] = validatedWin
                        } catch let valErr as EmbeddingValidationError {
                            throw valErr.wireValidationError
                        }
                    }

                default:
                    throw WorkbenchError.unprocessable(
                        message: "input[\(itemIdx)].content[\(blockIdx)] unsupported type '\(blockType)'"
                    )
                }
            }

            parsedItems.append(
                EmbeddingWireRequest.Item(
                    content: parts,
                    promptName: promptName,
                    hasExplicitPromptName: hasExplicitPromptName,
                    videoWindowsByBlockIndex: videoWindows,
                    hasInvalidMediaMIME: nil
                )
            )
        }

        return EmbeddingWireRequest(model: model, items: parsedItems, dimensions: dimensions)
    }
}
