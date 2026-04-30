// Copyright © Anthony DePasquale

import Foundation
import HFAPI
import Testing

@testable import Tokenizers

#if Swift
@testable import TokenizersSwiftBackend
#endif

#if Rust
@testable import TokenizersRustBackend
#endif

/// Cross-backend encoding snapshots that pin canonical behavior along two axes:
///
/// - **Multi-script encoding**: a shared Latin/CJK/Cyrillic/Arabic/Thai/emoji input,
///   tokenized against each of the eight models in `TokenizerTests.tokenizer`. Swift
///   and Rust agree on most models, but a few have genuine per-backend differences
///   (Thai-script handling in Llama-2 and the Swift path for the NLLB BPE model).
///   Those live as separate per-backend snapshots so they don't masquerade as
///   regressions.
///
/// - **Parity snapshot**: a fixed phrase (`"Who are you?"`) against a curated set of
///   models. All five entries agree across backends.
///
/// Recording workflow: delete an entry to force re-capture; run the test; paste the
/// `Issue.record` value back in.
@Suite("Baseline", .serialized)
struct BaselineTests {
    // MARK: - Shared fixtures

    private static let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-baseline-tests")
    }()

    private static let hubClient = HubClient()
    private static let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json", "config.json"]

    private static func downloadModel(_ modelName: String) async throws -> URL {
        guard let repoId = Repo.ID(rawValue: modelName) else {
            struct InvalidRepoID: Error { let name: String }
            throw InvalidRepoID(name: modelName)
        }
        return try await hubClient.downloadSnapshot(
            of: repoId,
            matching: tokenizerFiles,
            to: downloadDestination.appending(path: modelName)
        )
    }

    private static func makeTokenizer(hubModelName: String) async throws -> PreTrainedTokenizer {
        let modelDirectory = try await downloadModel(hubModelName)
        let tokenizer = try await AutoTokenizer.from(directory: modelDirectory)
        struct Unexpected: Error { let type: String }
        guard let pretrained = tokenizer as? PreTrainedTokenizer else {
            throw Unexpected(type: "\(type(of: tokenizer))")
        }
        return pretrained
    }

    /// Shared input mixing scripts (Latin, CJK, Cyrillic, Arabic, Thai, emoji) with
    /// whitespace-heavy runs (double spaces, tab, newline). Mirrors the shared-input
    /// style that upstream transformers uses across its tokenizer test matrices.
    private static let multiScriptInput =
        "Hello 世界! Привет🌍 مرحبا بالعالم  ราชอาณาจักรไทย   tabs\tand\nnewlines."

    private static func backendLabel() -> String {
        #if Rust
        "Rust"
        #else
        "Swift"
        #endif
    }

    /// Assert a snapshot: if no expected value is recorded for this (backend, model) pair,
    /// record the actual output so it can be pasted back in; otherwise compare.
    private static func expectSnapshot<Value: Equatable>(
        _ actual: Value,
        equals expected: Value?,
        label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        guard let expected else {
            Issue.record(
                "Baseline missing for \(label) [\(backendLabel())]. Captured value: \(String(describing: actual))",
                sourceLocation: sourceLocation
            )
            return
        }
        let comment: Comment = "\(label) [\(backendLabel())]"
        #expect(actual == expected, comment, sourceLocation: sourceLocation)
    }

    // MARK: - Multi-script encoding snapshot

    private static func multiScriptExpected(model: String) -> [Int]? {
        #if Rust
        return multiScriptExpectedRust[model]
        #else
        return multiScriptExpectedSwift[model]
        #endif
    }

    private static let multiScriptExpectedSwift: [String: [Int]] = [
        "coreml-projects/Llama-2-7b-chat-coreml": [
            1, 15043, 29871, 30793, 30967, 29991, 7203, 7616, 31494, 29871, 30159, 30156,
            30240, 30177, 30112, 29871, 30177, 19233, 30218, 19233, 30159, 259, 30297,
            30289, 30913, 30351, 30289, 31796, 30289, 227, 187, 139, 227, 187, 180, 30425,
            30297, 31252, 30595, 30549, 259, 18859, 12, 392, 13, 1482, 9012, 29889,
        ],
        "distilbert/distilbert-base-multilingual-cased": [
            101, 31178, 2087, 5621, 106, 100, 788, 51052, 30877, 10909, 10961, 54422,
            22887, 1427, 17344, 42407, 33178, 17344, 62914, 17344, 46856, 69365, 22765,
            41362, 100781, 10107, 10111, 10751, 31782, 119, 102,
        ],
        "distilbert/distilgpt2": [
            15496, 220, 10310, 244, 45911, 234, 0, 12466, 253, 21169, 18849, 38857, 16843,
            20375, 8582, 234, 235, 47048, 26897, 148, 255, 39848, 12919, 17550, 101,
            23525, 44690, 23525, 25405, 220, 220, 19567, 96, 19567, 110, 19567, 232,
            19567, 255, 19567, 110, 19567, 241, 19567, 110, 19567, 230, 19567, 109, 19567,
            223, 19567, 96, 31479, 226, 19567, 245, 19567, 95, 220, 220, 22524, 197, 392,
            198, 3605, 6615, 13,
        ],
        "openai/whisper-large-v2": [
            50258, 50363, 15947, 220, 24486, 0, 38932, 4222, 234, 235, 3714, 2288, 5016,
            3555, 995, 20666, 3615, 45340, 220, 220, 7067, 4943, 17080, 5104, 4943, 38206,
            4943, 14190, 5981, 6223, 7067, 8809, 11099, 7044, 220, 220, 20743, 197, 474,
            198, 7686, 11045, 13, 50257,
        ],
        "openai/whisper-tiny.en": [
            50257, 50362, 15496, 220, 10310, 244, 45911, 234, 0, 12466, 253, 21169, 18849,
            38857, 16843, 20375, 8582, 234, 235, 47048, 26897, 148, 255, 39848, 12919,
            17550, 101, 23525, 44690, 23525, 25405, 220, 220, 19567, 96, 19567, 110,
            19567, 232, 19567, 255, 19567, 110, 19567, 241, 19567, 110, 19567, 230, 19567,
            109, 19567, 223, 19567, 96, 31479, 226, 19567, 245, 19567, 95, 220, 220,
            22524, 197, 392, 198, 3605, 6615, 13, 50256,
        ],
        "pcuenq/Llama-3.2-1B-Instruct-tokenizer": [
            128000, 9906, 127365, 0, 80584, 28089, 8341, 9468, 234, 235, 111895, 103645,
            100700, 24102, 101952, 220, 100839, 101286, 23780, 125221, 109313, 102938,
            256, 23204, 53577, 198, 943, 8128, 13,
        ],
        "google-t5/t5-base": [
            8774, 3, 2, 55, 3, 2, 14709, 6609, 15042, 2, 3, 2, 3, 2, 3, 2, 3808, 7, 11,
            126, 6972, 5, 1,
        ],
        "tiiuae/falcon-7b": [
            9856, 204, 22602, 12, 5052, 237, 9310, 18945, 122, 37770, 48778, 219, 64951,
            28996, 158, 239, 59390, 15040, 25349, 113, 31254, 158, 129, 31254, 29663, 204,
            204, 46702, 6965, 122, 6965, 216, 40661, 6965, 122, 6965, 225, 6965, 122,
            6965, 214, 63624, 6965, 207, 46702, 166, 129, 210, 6965, 229, 6965, 107, 258,
            22883, 192, 373, 193, 1785, 7423, 25,
        ],
    ]

    private static let multiScriptExpectedRust: [String: [Int]] = [
        "coreml-projects/Llama-2-7b-chat-coreml": [
            1, 15043, 29871, 30793, 30967, 29991, 7203, 7616, 31494, 29871, 30159, 30156,
            30240, 30177, 30112, 29871, 30177, 19233, 30218, 19233, 30159, 259, 30297,
            30289, 30913, 30351, 30289, 31796, 30289, 30991, 30510, 30425, 30297, 31252,
            30595, 30549, 259, 18859, 12, 392, 13, 1482, 9012, 29889,
        ],
        "distilbert/distilbert-base-multilingual-cased": [
            101, 31178, 2087, 5621, 106, 100, 788, 51052, 30877, 10909, 10961, 54422,
            22887, 1427, 17344, 42407, 33178, 17344, 62914, 17344, 46856, 69365, 22765,
            41362, 100781, 10107, 10111, 10751, 31782, 119, 102,
        ],
        "distilbert/distilgpt2": [
            15496, 220, 10310, 244, 45911, 234, 0, 12466, 253, 21169, 18849, 38857, 16843,
            20375, 8582, 234, 235, 47048, 26897, 148, 255, 39848, 12919, 17550, 101,
            23525, 44690, 23525, 25405, 220, 220, 19567, 96, 19567, 110, 19567, 232,
            19567, 255, 19567, 110, 19567, 241, 19567, 110, 19567, 230, 19567, 109, 19567,
            223, 19567, 96, 31479, 226, 19567, 245, 19567, 95, 220, 220, 22524, 197, 392,
            198, 3605, 6615, 13,
        ],
        "openai/whisper-large-v2": [
            50258, 50363, 15947, 220, 24486, 0, 38932, 4222, 234, 235, 3714, 2288, 5016,
            3555, 995, 20666, 3615, 45340, 220, 220, 7067, 4943, 17080, 5104, 4943, 38206,
            4943, 14190, 5981, 6223, 7067, 8809, 11099, 7044, 220, 220, 20743, 197, 474,
            198, 7686, 11045, 13, 50257,
        ],
        "openai/whisper-tiny.en": [
            50257, 50362, 15496, 220, 10310, 244, 45911, 234, 0, 12466, 253, 21169, 18849,
            38857, 16843, 20375, 8582, 234, 235, 47048, 26897, 148, 255, 39848, 12919,
            17550, 101, 23525, 44690, 23525, 25405, 220, 220, 19567, 96, 19567, 110,
            19567, 232, 19567, 255, 19567, 110, 19567, 241, 19567, 110, 19567, 230, 19567,
            109, 19567, 223, 19567, 96, 31479, 226, 19567, 245, 19567, 95, 220, 220,
            22524, 197, 392, 198, 3605, 6615, 13, 50256,
        ],
        "pcuenq/Llama-3.2-1B-Instruct-tokenizer": [
            128000, 9906, 127365, 0, 80584, 28089, 8341, 9468, 234, 235, 111895, 103645,
            100700, 24102, 101952, 220, 100839, 101286, 23780, 125221, 109313, 102938,
            256, 23204, 53577, 198, 943, 8128, 13,
        ],
        "google-t5/t5-base": [
            8774, 3, 2, 55, 3, 2, 14709, 6609, 15042, 2, 3, 2, 3, 2, 3, 2, 3808, 7, 11,
            126, 6972, 5, 1,
        ],
        "tiiuae/falcon-7b": [
            9856, 204, 22602, 12, 5052, 237, 9310, 18945, 122, 37770, 48778, 219, 64951,
            28996, 158, 239, 59390, 15040, 25349, 113, 31254, 158, 129, 31254, 29663, 204,
            204, 46702, 6965, 122, 6965, 216, 40661, 6965, 122, 6965, 225, 6965, 122,
            6965, 214, 63624, 6965, 207, 46702, 166, 129, 210, 6965, 229, 6965, 107, 258,
            22883, 192, 373, 193, 1785, 7423, 25,
        ],
    ]

    @Test(arguments: [
        "coreml-projects/Llama-2-7b-chat-coreml",
        "distilbert/distilbert-base-multilingual-cased",
        "distilbert/distilgpt2",
        "openai/whisper-large-v2",
        "openai/whisper-tiny.en",
        "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
        "google-t5/t5-base",
        "tiiuae/falcon-7b",
    ])
    func multiScriptEncoding(hubModelName: String) async throws {
        let tokenizer = try await Self.makeTokenizer(hubModelName: hubModelName)
        let encoded = tokenizer.encode(text: Self.multiScriptInput)
        Self.expectSnapshot(
            encoded,
            equals: Self.multiScriptExpected(model: hubModelName),
            label: "multi-script encode \(hubModelName)"
        )
    }

    // MARK: - Parity snapshot

    private static func parityExpected(model: String) -> [Int]? {
        #if Rust
        return parityExpectedRust[model]
        #else
        return parityExpectedSwift[model]
        #endif
    }

    private static let parityExpectedSwift: [String: [Int]] = [
        "coreml-projects/Llama-2-7b-chat-coreml": [1, 11644, 526, 366, 29973],
        "google-bert/bert-base-uncased": [101, 2040, 2024, 2017, 1029, 102],
        // DeepSeek's tokenizer.json has a ByteLevel post-processor, so there is no
        // leading bos. This matches canonical Python transformers v5 and upstream
        // `huggingface/tokenizers`.
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [15191, 525, 498, 30],
        "FacebookAI/xlm-roberta-base": [0, 40469, 621, 398, 32, 2],
        "Xenova/nllb-200-distilled-600M": [256047, 26133, 2442, 1259, 248130, 2],
    ]

    private static let parityExpectedRust: [String: [Int]] = [
        "coreml-projects/Llama-2-7b-chat-coreml": [1, 11644, 526, 366, 29973],
        "google-bert/bert-base-uncased": [101, 2040, 2024, 2017, 1029, 102],
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [15191, 525, 498, 30],
        "FacebookAI/xlm-roberta-base": [0, 40469, 621, 398, 32, 2],
        "Xenova/nllb-200-distilled-600M": [256047, 26133, 2442, 1259, 248130, 2],
    ]

    @Test(arguments: [
        // BPE + TemplateProcessing post-processor.
        "coreml-projects/Llama-2-7b-chat-coreml",
        // WordPiece.
        "google-bert/bert-base-uncased",
        // BPE with a ByteLevel post-processor. `tokenizer_config.json` sets
        // `add_bos_token: true` but the expected output has no leading bos token
        // because the post-processor baked into `tokenizer.json` is the only
        // source of prepended specials — see the rationale on
        // `TokenizerRuntimeConfiguration` / the Rust core's `RuntimeConfiguration`.
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        // Unigram via XLM-R.
        "FacebookAI/xlm-roberta-base",
        // BPE loaded via `tokenizer.json`'s `model.type` (wrapper name is
        // untagged / unknown; model dispatch does not depend on it).
        "Xenova/nllb-200-distilled-600M",
    ])
    func paritySnapshot(hubModelName: String) async throws {
        let tokenizer = try await Self.makeTokenizer(hubModelName: hubModelName)
        let encoded = tokenizer.encode(text: "Who are you?")
        Self.expectSnapshot(
            encoded,
            equals: Self.parityExpected(model: hubModelName),
            label: "parity snapshot \(hubModelName)"
        )
    }
}
