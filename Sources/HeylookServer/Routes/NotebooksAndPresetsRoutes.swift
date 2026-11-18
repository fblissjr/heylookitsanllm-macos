import Foundation
import HeylookCore

public enum NotebooksAndPresetsRoutes {
    public static func register(on router: inout PatternRouter, service: WorkbenchService) {
        // MARK: Notebooks

        router.register("GET", "/v1/notebooks") { _ in
            let list = try await service.listNotebooks()
            return try .json(list, status: 200)
        }

        router.register("POST", "/v1/notebooks") { ctx in
            let req = ctx.body.isEmpty
                ? try JSONDecoder().decode(NotebookCreateRequestDTO.self, from: Data("{}".utf8))
                : try JSONDecoder().decode(NotebookCreateRequestDTO.self, from: ctx.body)
            let created = try await service.createNotebook(req)
            return try .json(created, status: 201)
        }

        router.register("GET", "/v1/notebooks/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing notebook id")
            }
            let nb = try await service.getNotebook(id: id)
            return try .json(nb, status: 200)
        }

        router.registerMethods(["PUT", "PATCH"], "/v1/notebooks/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing notebook id")
            }
            let req = try JSONDecoder().decode(NotebookUpdateRequestDTO.self, from: ctx.body)
            let updated = try await service.updateNotebook(id: id, req: req)
            return try .json(updated, status: 200)
        }

        router.register("DELETE", "/v1/notebooks/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing notebook id")
            }
            try await service.deleteNotebook(id: id)
            return try .jsonObject(["status": "deleted", "id": id], status: 200)
        }

        // MARK: Presets

        router.register("GET", "/v1/presets") { _ in
            let list = try await service.listPresets()
            return try .json(list, status: 200)
        }

        router.register("POST", "/v1/presets") { ctx in
            let req = try JSONDecoder().decode(PresetCreateRequestDTO.self, from: ctx.body)
            let created = try await service.createPreset(req)
            return try .json(created, status: 201)
        }

        router.register("GET", "/v1/presets/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing preset id")
            }
            let preset = try await service.getPreset(id: id)
            return try .json(preset, status: 200)
        }

        router.registerMethods(["PUT", "PATCH"], "/v1/presets/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing preset id")
            }
            let req = try JSONDecoder().decode(PresetUpdateRequestDTO.self, from: ctx.body)
            let updated = try await service.updatePreset(id: id, req: req)
            return try .json(updated, status: 200)
        }

        router.register("DELETE", "/v1/presets/:id") { ctx in
            guard let id = ctx.param("id") else {
                throw WorkbenchError.badRequest("Missing preset id")
            }
            try await service.deletePreset(id: id)
            return try .jsonObject(["status": "deleted", "id": id], status: 200)
        }
    }
}
