import BenchmarkHelpers
import Foundation
import HFAPI
import Testing

private let benchmarksEnabled = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"
private let modelBenchmarksEnabled = ProcessInfo.processInfo.environment["RUN_MODEL_BENCHMARKS"] == "1"
private let benchmarkDownloader = HubClientDownloader(.default)

@Suite(.serialized, .enabled(if: benchmarksEnabled))
struct Benchmarks {
    @Test func loadSidecars() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkSidecarLoading(
            from: benchmarkDownloader
        )
        stats.printSummary(label: "Sidecar load (\(backend.label))")
    }

    @Test func loadTokenizerCore() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkTokenizerCoreLoading(
            from: benchmarkDownloader
        )
        stats.printSummary(label: "Tokenizer core load (\(backend.label))")
    }

    @Test func loadTokenizer() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkTokenizerLoading(
            from: benchmarkDownloader,
            using: backend.loader
        )
        stats.printSummary(label: "Tokenizer load (\(backend.label))")
    }

    @Test func tokenizeText() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let sampleText = try await loadTokenizationBenchmarkText()
        let stats = try await benchmarkTokenization(
            from: benchmarkDownloader,
            using: backend.loader,
            text: sampleText
        )
        stats.printSummary(label: "Tokenization (\(backend.label))")
    }

    @Test func decodeText() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let sampleText = try await loadDecodingBenchmarkText()
        let stats = try await benchmarkDecoding(
            from: benchmarkDownloader,
            using: backend.loader,
            text: sampleText
        )
        stats.printSummary(label: "Decoding (\(backend.label))")
    }

    @Test func renderChatTemplate() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkChatTemplateRendering(
            from: benchmarkDownloader
        )
        stats.printSummary(label: "Chat template render (\(backend.label))")
    }

    @Test(.enabled(if: modelBenchmarksEnabled)) func loadLLM() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkLLMLoading(
            from: benchmarkDownloader,
            using: backend.loader
        )
        stats.printSummary(label: "LLM load (\(backend.label))")
    }

    @Test(.enabled(if: modelBenchmarksEnabled)) func loadVLM() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkVLMLoading(
            from: benchmarkDownloader,
            using: backend.loader
        )
        stats.printSummary(label: "VLM load (\(backend.label))")
    }

    @Test(.enabled(if: modelBenchmarksEnabled)) func loadEmbedding() async throws {
        let backend = activeBenchmarkTokenizerBackend()
        let stats = try await benchmarkEmbeddingLoading(
            from: benchmarkDownloader,
            using: backend.loader
        )
        stats.printSummary(label: "Embedding load (\(backend.label))")
    }
}
