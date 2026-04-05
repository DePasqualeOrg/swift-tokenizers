import BenchmarkHelpers
import Foundation
import HFAPI
import MLXLMCommon
import Tokenizers
import TokenizersCore

#if TOKENIZERS_SWIFT_BACKEND
import TokenizersSwiftBackend
#endif

#if Rust
import TokenizersRustBackend
#endif

private enum SubsystemBenchmarkDefaults {
    static let configuration = MLXLMCommon.ModelConfiguration(id: "mlx-community/Qwen3-0.6B-4bit")
    static let sidecarRuns = 25
    static let tokenizerCoreRuns = BenchmarkDefaults.loadingRuns
    static let templateRenderRuns = 25
}

private let subsystemTokenizerDownloadPatterns = ["*.json", "*.jinja"]

private let templateBenchmarkMessages: [Tokenizers.Message] = [
    ["role": "system", "content": "You are a concise assistant."],
    ["role": "user", "content": "Explain why Rust-backed tokenization can be faster than a pure Swift implementation."],
]

private enum SubsystemBenchmarkError: LocalizedError {
    case unsupportedTokenizerType(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedTokenizerType(let type):
            return "Unexpected tokenizer benchmark type: \(type)"
        }
    }
}

private func resolveBenchmarkTokenizerDirectory(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration = SubsystemBenchmarkDefaults.configuration,
    useLatest: Bool = false
) async throws -> URL {
    switch configuration.tokenizerSource {
    case .id(let id, let revision):
        return try await downloader.download(
            id: id,
            revision: revision,
            matching: subsystemTokenizerDownloadPatterns,
            useLatest: useLatest,
            progressHandler: { _ in }
        )
    case .directory(let directory):
        return directory
    case nil:
        switch configuration.id {
        case .id(let id, let revision):
            return try await downloader.download(
                id: id,
                revision: revision,
                matching: subsystemTokenizerDownloadPatterns,
                useLatest: useLatest,
                progressHandler: { _ in }
            )
        case .directory(let directory):
            return directory
        }
    }
}

func benchmarkSidecarLoading(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration = SubsystemBenchmarkDefaults.configuration,
    useLatest: Bool = false,
    runs: Int = SubsystemBenchmarkDefaults.sidecarRuns
) async throws -> BenchmarkStats {
    let tokenizerDirectory = try await resolveBenchmarkTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )

    #if Rust
    _ = try RustAutoTokenizerDirectoryLoader.loadRuntimeConfiguration(from: tokenizerDirectory)
    #elseif TOKENIZERS_SWIFT_BACKEND
    _ = try SwiftAutoTokenizerDirectoryLoader.loadRuntimeConfiguration(from: tokenizerDirectory)
    #else
    #error("No tokenizer backend selected")
    #endif

    var times: [Double] = []
    for i in 1...runs {
        let start = CFAbsoluteTimeGetCurrent()
        #if Rust
        _ = try RustAutoTokenizerDirectoryLoader.loadRuntimeConfiguration(from: tokenizerDirectory)
        #elseif TOKENIZERS_SWIFT_BACKEND
        _ = try SwiftAutoTokenizerDirectoryLoader.loadRuntimeConfiguration(from: tokenizerDirectory)
        #endif
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Sidecar load run \(i): \(String(format: "%.1f", elapsed))ms")
    }

    return BenchmarkStats(times: times)
}

func benchmarkTokenizerCoreLoading(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration = SubsystemBenchmarkDefaults.configuration,
    useLatest: Bool = false,
    runs: Int = SubsystemBenchmarkDefaults.tokenizerCoreRuns
) async throws -> BenchmarkStats {
    let tokenizerDirectory = try await resolveBenchmarkTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )

    #if Rust
    let runtimeConfiguration = try RustAutoTokenizerDirectoryLoader.loadRuntimeConfiguration(from: tokenizerDirectory)
    _ = try await RustAutoTokenizerDirectoryLoader.loadTokenizerCore(
        from: tokenizerDirectory,
        runtimeConfiguration: runtimeConfiguration,
        strict: true
    )
    #elseif TOKENIZERS_SWIFT_BACKEND
    let tokenizerConfig = try SwiftAutoTokenizerDirectoryLoader.loadTokenizerConfig(from: tokenizerDirectory)
    _ = try await SwiftAutoTokenizerDirectoryLoader.loadTokenizerCore(
        from: tokenizerDirectory,
        tokenizerConfig: tokenizerConfig,
        strict: true
    )
    #else
    #error("No tokenizer backend selected")
    #endif

    var times: [Double] = []
    for i in 1...runs {
        let start = CFAbsoluteTimeGetCurrent()
        #if Rust
        _ = try await RustAutoTokenizerDirectoryLoader.loadTokenizerCore(
            from: tokenizerDirectory,
            runtimeConfiguration: runtimeConfiguration,
            strict: true
        )
        #elseif TOKENIZERS_SWIFT_BACKEND
        _ = try await SwiftAutoTokenizerDirectoryLoader.loadTokenizerCore(
            from: tokenizerDirectory,
            tokenizerConfig: tokenizerConfig,
            strict: true
        )
        #endif
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("Tokenizer core load run \(i): \(String(format: "%.1f", elapsed))ms")
    }

    return BenchmarkStats(times: times)
}

private func loadPreTrainedTokenizerForTemplateBenchmark(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration = SubsystemBenchmarkDefaults.configuration,
    useLatest: Bool = false
) async throws -> PreTrainedTokenizer {
    let tokenizerDirectory = try await resolveBenchmarkTokenizerDirectory(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )
    let tokenizer = try await AutoTokenizer.from(directory: tokenizerDirectory)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
        throw SubsystemBenchmarkError.unsupportedTokenizerType(String(describing: type(of: tokenizer)))
    }
    return tokenizer
}

func benchmarkChatTemplateRendering(
    from downloader: any Downloader,
    configuration: MLXLMCommon.ModelConfiguration = SubsystemBenchmarkDefaults.configuration,
    useLatest: Bool = false,
    runs: Int = SubsystemBenchmarkDefaults.templateRenderRuns
) async throws -> BenchmarkStats {
    let tokenizer = try await loadPreTrainedTokenizerForTemplateBenchmark(
        from: downloader,
        configuration: configuration,
        useLatest: useLatest
    )

    let template = try tokenizer.selectedChatTemplate(chatTemplate: nil, tools: nil)
    let contextObject = try tokenizer.chatTemplateContextObject(
        messages: templateBenchmarkMessages,
        addGenerationPrompt: true,
        tools: nil,
        additionalContext: nil
    )

    _ = try tokenizer.renderChatTemplateToString(template: template, contextObject: contextObject)

    var times: [Double] = []
    for i in 1...runs {
        let start = CFAbsoluteTimeGetCurrent()
        let rendered = try tokenizer.renderChatTemplateToString(
            template: template,
            contextObject: contextObject
        )
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print(
            "Chat template render run \(i): \(String(format: "%.1f", elapsed))ms "
                + "(\(rendered.count) chars)"
        )
    }

    return BenchmarkStats(times: times)
}
