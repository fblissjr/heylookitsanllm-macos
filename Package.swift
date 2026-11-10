// swift-tools-version: 6.2
import Foundation
import PackageDescription

let env = ProcessInfo.processInfo.environment
let isXcode = (env["__CFBundleIdentifier"]?.contains("Xcode") == true)
    || (env["XPC_SERVICE_NAME"]?.contains("Xcode") == true)
let enableAll = env["HEYLOOK_ENABLE_ALL_PACKAGES"] == "1"
let enableNativeMLX: Bool = {
    if env["HEYLOOK_DISABLE_NATIVE_MLX"] == "1" || env["HEYLOOK_ENABLE_NATIVE_MLX"] == "0" {
        return false
    }
    return enableAll || env["HEYLOOK_ENABLE_NATIVE_MLX"] == "1" || isXcode
}()
let enableCoreAI = enableAll || env["HEYLOOK_ENABLE_COREAI"] == "1"
let enableArgmax = enableAll || env["HEYLOOK_ENABLE_ARGMAX"] == "1"
let enableHummingbird = enableAll || env["HEYLOOK_ENABLE_HUMMINGBIRD"] == "1"

var packageDependencies: [Package.Dependency] = []
var embeddingsDependencies: [Target.Dependency] = ["HeylookCore"]
var enginesDependencies: [Target.Dependency] = ["HeylookCore", "HeylookEmbeddings"]
var serverDependencies: [Target.Dependency] = ["HeylookCore", "HeylookEmbeddings", "HeylookEngines"]

if enableNativeMLX {
    packageDependencies.append(contentsOf: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.33.0")
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            branch: "main"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMajor(from: "1.1.0")
        ),
    ])
    let mlxProducts: [Target.Dependency] = [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXVLM", package: "mlx-swift-lm"),
        .product(name: "Transformers", package: "swift-transformers"),
    ]
    embeddingsDependencies.append(contentsOf: mlxProducts)
    enginesDependencies.append(contentsOf: mlxProducts)
}

if enableCoreAI {
    packageDependencies.append(
        .package(
            url: "https://github.com/apple/coreai-models",
            branch: "main"
        )
    )
    enginesDependencies.append(contentsOf: [
        .product(name: "CoreAILM", package: "coreai-models"),
        .product(name: "CoreAISpeech", package: "coreai-models"),
    ])
}

if enableArgmax {
    packageDependencies.append(
        .package(
            url: "https://github.com/argmaxinc/argmax-oss-swift",
            branch: "main"
        )
    )
    enginesDependencies.append(contentsOf: [
        .product(name: "WhisperKit", package: "argmax-oss-swift"),
        .product(name: "TTSKit", package: "argmax-oss-swift"),
    ])
}

if enableHummingbird {
    packageDependencies.append(
        .package(
            url: "https://github.com/hummingbird-project/hummingbird",
            .upToNextMajor(from: "2.22.0")
        )
    )
    serverDependencies.append(
        .product(name: "Hummingbird", package: "hummingbird")
    )
}

let package = Package(
    name: "heylookitsanllm-macos",
    platforms: [
        .macOS("27.2"),
        .iOS("27.2"),
    ],
    products: [
        .library(name: "HeylookCore", targets: ["HeylookCore"]),
        .library(name: "HeylookEmbeddings", targets: ["HeylookEmbeddings"]),
        .library(name: "HeylookEngines", targets: ["HeylookEngines"]),
        .library(name: "HeylookServer", targets: ["HeylookServer"]),
        .library(name: "HeylookUI", targets: ["HeylookUI"]),
        .executable(name: "heylook-app", targets: ["HeylookApp"]),
        .executable(name: "heylook-cli", targets: ["HeylookCLI"]),
    ],
    dependencies: packageDependencies,
    targets: [
        .target(
            name: "HeylookCore",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "HeylookEmbeddings",
            dependencies: embeddingsDependencies,
            linkerSettings: [
                .linkedLibrary("sqlite3"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreML"),
                .linkedFramework("ImageIO"),
                .linkedFramework("PDFKit"),
                .linkedFramework("CoreServices", .when(platforms: [.macOS])),
            ]
        ),
        .target(
            name: "HeylookEngines",
            dependencies: enginesDependencies,
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
            ]
        ),
        .target(
            name: "HeylookServer",
            dependencies: serverDependencies,
            linkerSettings: [
                .linkedFramework("Network"),
            ]
        ),
        .target(
            name: "HeylookUI",
            dependencies: [
                "HeylookCore",
                "HeylookEmbeddings",
                "HeylookEngines",
                "HeylookServer",
            ],
            linkerSettings: [
                .linkedFramework("SwiftUI"),
                .linkedFramework("AVKit"),
                .linkedFramework("UniformTypeIdentifiers"),
                .linkedFramework("AppKit", .when(platforms: [.macOS])),
            ]
        ),
        .executableTarget(
            name: "HeylookApp",
            dependencies: [
                "HeylookCore",
                "HeylookEmbeddings",
                "HeylookEngines",
                "HeylookServer",
                "HeylookUI",
            ]
        ),
        .executableTarget(
            name: "HeylookCLI",
            dependencies: [
                "HeylookCore",
                "HeylookEmbeddings",
                "HeylookEngines",
                "HeylookServer",
            ]
        ),
        .testTarget(
            name: "HeylookTests",
            dependencies: [
                "HeylookCore",
                "HeylookEmbeddings",
                "HeylookEngines",
                "HeylookServer",
                "HeylookUI",
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
