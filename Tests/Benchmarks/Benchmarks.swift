import BenchmarkHelpers
import HFAPI
import Testing
import Foundation

private let benchmarksEnabled = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"

@Suite(.serialized, .enabled(if: benchmarksEnabled))
struct Benchmarks {
    @Test func loadTokenizer() async throws {
        let stats = try await benchmarkTokenizerLoading(
            from: HubClient.default,
            using: TokenizersLoader()
        )
        stats.printSummary(label: "Tokenizer load (swift-tokenizers)")
    }

    @Test func tokenizeText() async throws {
        let sampleText = try await loadTokenizationBenchmarkText()
        let stats = try await benchmarkTokenization(
            from: HubClient.default,
            using: TokenizersLoader(),
            text: sampleText
        )
        stats.printSummary(label: "Tokenization (swift-tokenizers)")
    }

    @Test func decodeText() async throws {
        let sampleText = try await loadDecodingBenchmarkText()
        let stats = try await benchmarkDecoding(
            from: HubClient.default,
            using: TokenizersLoader(),
            text: sampleText
        )
        stats.printSummary(label: "Decoding (swift-tokenizers)")
    }

    @Test func loadLLM() async throws {
        let stats = try await benchmarkLLMLoading(
            from: HubClient.default,
            using: TokenizersLoader()
        )
        stats.printSummary(label: "LLM load (swift-tokenizers)")
    }

    @Test func loadVLM() async throws {
        let stats = try await benchmarkVLMLoading(
            from: HubClient.default,
            using: TokenizersLoader()
        )
        stats.printSummary(label: "VLM load (swift-tokenizers)")
    }

    @Test func loadEmbedding() async throws {
        let stats = try await benchmarkEmbeddingLoading(
            from: HubClient.default,
            using: TokenizersLoader()
        )
        stats.printSummary(label: "Embedding load (swift-tokenizers)")
    }
}
