// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale
// Based on GPT2TokenizerTests by Julien Chaumond.

import Foundation
import HFAPI
import Testing

@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

private let hubClient = HFClient.default

private enum TestError: Error { case unsupportedTokenizer }

private struct Dataset: Decodable {
    let text: String
    // Bad naming, not just for bpe.
    // We are going to replace this testing method anyway.
    let bpe_tokens: [String]
    let token_ids: [Int]
    let decoded_text: String
}

private func loadDataset(filename: String) throws -> Dataset {
    let url = Bundle.module.url(forResource: filename, withExtension: "json")!
    let json = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(Dataset.self, from: json)
}

private struct EdgeCase: Decodable {
    let input: String

    struct EncodedData: Decodable {
        let input_ids: [Int]
        let token_type_ids: [Int]?
        let attention_mask: [Int]
    }

    let encoded: EncodedData
    let decoded_with_special: String
    let decoded_without_special: String
}

private func loadEdgeCases(for hubModelName: String) throws -> [EdgeCase]? {
    let url = Bundle.module.url(forResource: "tokenizer_tests", withExtension: "json")!
    let json = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let cases = try decoder.decode([String: [EdgeCase]].self, from: json)
    return cases[hubModelName]
}

private let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json", "config.json"]

private func downloadModel(_ modelName: String) async throws -> URL {
    guard let repoId = RepositoryID(modelName) else {
        throw TestError.unsupportedTokenizer
    }
    return try await hubClient.model(repoId).snapshotDownload(
        allowPatterns: tokenizerFiles,
        localDir: downloadDestination.appending(path: modelName)
    )
}

private func makeTokenizer(hubModelName: String) async throws -> PreTrainedTokenizer {
    let modelDirectory = try await downloadModel(hubModelName)
    let tokenizer = try await AutoTokenizer.from(directory: modelDirectory)
    guard let pretrained = tokenizer as? PreTrainedTokenizer else {
        throw TestError.unsupportedTokenizer
    }
    return pretrained
}

// MARK: -

struct ModelSpec: Sendable, CustomStringConvertible {
    let hubModelName: String
    let encodedSamplesFilename: String
    let unknownTokenId: Int?

    var description: String {
        hubModelName
    }

    init(_ hubModelName: String, _ encodedSamplesFilename: String, _ unknownTokenId: Int? = nil) {
        self.hubModelName = hubModelName
        self.encodedSamplesFilename = encodedSamplesFilename
        self.unknownTokenId = unknownTokenId
    }
}

// MARK: -

@Suite("Tokenizer Tests", .serialized)
struct TokenizerTests {
    @Test(arguments: [
        ModelSpec("enterprise-explorers/Llama-2-7b-chat-coreml", "llama_encoded", 0),
        ModelSpec("distilbert/distilbert-base-multilingual-cased", "distilbert_cased_encoded", 100),
        ModelSpec("distilbert/distilgpt2", "gpt2_encoded_tokens"),
        ModelSpec("openai/whisper-large-v2", "whisper_large_v2_encoded", 50257),
        ModelSpec("openai/whisper-tiny.en", "whisper_tiny_en_encoded", 50256),
        ModelSpec("pcuenq/Llama-3.2-1B-Instruct-tokenizer", "llama_3.2_encoded"),
        ModelSpec("google-t5/t5-base", "t5_base_encoded", 2),
        ModelSpec("tiiuae/falcon-7b", "falcon_encoded"),
    ])
    func tokenizer(spec: ModelSpec) async throws {
        let tokenizer = try await makeTokenizer(hubModelName: spec.hubModelName)
        let dataset = try loadDataset(filename: spec.encodedSamplesFilename)

        #expect(try tokenizer.tokenize(text: dataset.text) == dataset.bpe_tokens)
        #expect(try tokenizer.encode(text: dataset.text) == dataset.token_ids)
        #expect(try tokenizer.decode(tokenIds: dataset.token_ids) == dataset.decoded_text)

        // Edge cases (if available)
        if let edgeCases = try? loadEdgeCases(for: spec.hubModelName) {
            for edgeCase in edgeCases {
                #expect(try tokenizer.encode(text: edgeCase.input) == edgeCase.encoded.input_ids)
                #expect(try tokenizer.decode(tokenIds: edgeCase.encoded.input_ids) == edgeCase.decoded_with_special)
                #expect(try tokenizer.decode(tokenIds: edgeCase.encoded.input_ids, skipSpecialTokens: true) == edgeCase.decoded_without_special)
            }
        }

        // Unknown token checks. `tokenizer.unknownTokenId` composes the
        // sidecar-style fallback (`tokenizer_config.json.unk_token` first, then
        // the model's own `unk_token`), matching Python's user-facing
        // `tokenizer.unk_token_id`. The underlying `rustTokenizer.unknownTokenId`
        // is the model's own field only (e.g. nil for Whisper, whose
        // `tokenizer.json.model.unk_token` is null).
        #expect(tokenizer.unknownTokenId == spec.unknownTokenId)
        // `convertTokenToId` reports vocabulary membership; missing tokens
        // resolve to `nil` rather than the unk id.
        #expect(tokenizer.rustTokenizer.convertTokenToId("_this_token_does_not_exist_") == nil)
        if let modelUnknownId = tokenizer.rustTokenizer.unknownTokenId {
            #expect(tokenizer.rustTokenizer.unknownToken == tokenizer.rustTokenizer.convertIdToToken(modelUnknownId))
        }
    }

    @Test
    func gemmaUnicode() async throws {
        let modelDirectory = try await downloadModel("pcuenq/gemma-tokenizer")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        // These are two different characters
        let cases = [
            "\u{0061}\u{0300}", // NFD: a + combining grave accent
            "\u{00E0}", // NFC: precomposed à
        ]
        let expected = [217138, 1305]
        for (s, expected) in zip(cases, expected) {
            let encoded = try tokenizer.encode(text: " " + s)
            #expect(encoded == [2, expected])
        }

        // Keys that start with BOM sequence
        // https://github.com/huggingface/swift-transformers/issues/88
        // https://github.com/ml-explore/mlx-swift-examples/issues/50#issuecomment-2046592213
        #expect(tokenizer.convertIdToToken(122661) == "\u{feff}#")
        #expect(tokenizer.convertIdToToken(235345) == "#")

        // Verifies all expected entries are parsed
        #expect(tokenizer.getVocabSize(withAddedTokens: true) == 256_000)

        // Test added tokens
        let inputIds = try tokenizer("This\n\nis\na\ntest.")
        #expect(inputIds == [2, 1596, 109, 502, 108, 235250, 108, 2195, 235265])
        let decoded = try tokenizer.decode(tokenIds: inputIds)
        #expect(decoded == "<bos>This\n\nis\na\ntest.")
    }

    /// Pins both flavors of `getVocabSize` against a tokenizer whose added vocabulary
    /// is non-empty, so the `withAddedTokens: false` and `withAddedTokens: true` results
    /// differ. Llama 3.2 has a 128_000-entry base vocabulary and 256 added reserved
    /// special tokens, matching `tokenizer.vocab_size` and `len(tokenizer)` in the
    /// Hugging Face Python library.
    @Test
    func llama3VocabSize() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "pcuenq/Llama-3.2-1B-Instruct-tokenizer")
        #expect(tokenizer.getVocabSize(withAddedTokens: false) == 128_000)
        #expect(tokenizer.getVocabSize(withAddedTokens: true) == 128_256)
    }

    /// Verifies sentence-pair (`Dual`) input flows through the protocol and returns an
    /// encoding with both sequences distinguishable. DistilBERT's WordPiece + post-processor
    /// produces `[CLS] A [SEP] B [SEP]` and assigns `sequence_ids` of `[None, 0, 0, 0, None, 1, 1, 1, None]`.
    /// Expected values cross-checked against the upstream Hugging Face `tokenizers` Python bindings.
    @Test
    func pairInputEncoding() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "distilbert/distilbert-base-multilingual-cased")
        let encoding = try tokenizer.encodeWithMetadata(
            text: "Sequence A",
            textPair: "Sequence B",
            addSpecialTokens: true,
            offsetUnit: .unicodeScalar
        )
        #expect(encoding.tokenIds == [101, 11045, 72494, 138, 102, 11045, 72494, 139, 102])
        #expect(encoding.sequenceCount == 2)
        #expect(encoding.sequenceIndices == [nil, 0, 0, 0, nil, 1, 1, 1, nil])
        #expect(encoding.tokenTypeIds == [0, 0, 0, 0, 0, 1, 1, 1, 1])

        // Fast pair encode returns just the IDs.
        let ids = try tokenizer.encode(text: "Sequence A", textPair: "Sequence B", addSpecialTokens: true)
        #expect(ids == encoding.tokenIds)
    }

    /// Verifies batch encoding produces the same IDs as encoding each input individually,
    /// and that the metadata-rich batch path returns one ``TokenizerEncoding`` per input.
    @Test
    func batchEncoding() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "distilbert/distilbert-base-multilingual-cased")
        let texts = ["hello", "world goodbye"]
        let batch = try tokenizer.encodeBatch(texts: texts, addSpecialTokens: true)
        #expect(batch.count == texts.count)
        for (index, text) in texts.enumerated() {
            let single = try tokenizer.encode(text: text, addSpecialTokens: true)
            #expect(batch[index] == single)
        }

        let metadataBatch = try tokenizer.encodeBatchWithMetadata(
            texts: texts,
            addSpecialTokens: true,
            offsetUnit: .unicodeScalar
        )
        #expect(metadataBatch.count == texts.count)
        for (index, text) in texts.enumerated() {
            let single = try tokenizer.encodeWithMetadata(
                text: text,
                addSpecialTokens: true,
                offsetUnit: .unicodeScalar
            )
            #expect(metadataBatch[index].tokenIds == single.tokenIds)
            #expect(metadataBatch[index].sequenceIndices == single.sequenceIndices)
        }
    }

    @Test
    func phi4() async throws {
        let modelDirectory = try await downloadModel("microsoft/phi-4")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(try tokenizer.encode(text: "hello") == [15339])
        #expect(try tokenizer.encode(text: "hello world") == [15339, 1917])
        #expect(try tokenizer.encode(text: "<|im_start|>user<|im_sep|>Who are you?<|im_end|><|im_start|>assistant<|im_sep|>") == [100264, 882, 100266, 15546, 527, 499, 30, 100265, 100264, 78191, 100266])
    }

    /// Compact hub-integration coverage for model families on Python transformers
    /// v5's `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` force-map list. Each test
    /// loads the model via `AutoTokenizer.from(directory:)` and asserts a small
    /// encode output to pin the canonical behavior.
    @Test
    func qwen25() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "Qwen/Qwen2.5-0.5B-Instruct")
        #expect(try tokenizer.encode(text: "Who are you?") == [15191, 525, 498, 30])
        #expect(
            try tokenizer.encode(text: "<|im_start|>user\nHello<|im_end|>")
                == [151644, 872, 198, 9707, 151645]
        )
    }

    @Test
    func modernBert() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "answerdotai/ModernBERT-base")
        #expect(try tokenizer.encode(text: "Hello world") == [50281, 12092, 1533, 50282])
    }

    @Test
    func gemma2() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "mlx-community/gemma-2-2b-it-4bit")
        #expect(try tokenizer.encode(text: "Who are you?") == [2, 6571, 708, 692, 235336])
    }

    @Test
    func mistralV03() async throws {
        let tokenizer = try await makeTokenizer(
            hubModelName: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        )
        #expect(try tokenizer.encode(text: "Who are you?") == [1, 7294, 1228, 1136, 29572])
    }

    @Test
    func ayaExpanse() async throws {
        let tokenizer = try await makeTokenizer(hubModelName: "mlx-community/aya-expanse-8b-4bit")
        #expect(try tokenizer.encode(text: "Who are you?") == [5, 33668, 1955, 1933, 38])
    }

    @Test
    func tokenizerFromLocalDirectory() async throws {
        let bundle = Bundle.module
        guard
            let tokenizerConfigURL = bundle.url(
                forResource: "tokenizer_config",
                withExtension: "json"
            ),
            bundle.url(
                forResource: "tokenizer",
                withExtension: "json"
            ) != nil
        else {
            Issue.record("Missing offline tokenizer fixtures")
            return
        }

        let tokenizer = try await AutoTokenizer.from(directory: tokenizerConfigURL.deletingLastPathComponent())

        let encoded = try tokenizer.encode(text: "offline path")
        #expect(!encoded.isEmpty)
    }

    /// https://github.com/huggingface/swift-transformers/issues/96
    @Test
    func legacyLlamaBehaviour() async throws {
        let modelDirectory = try await downloadModel("mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let inputIds = try tokenizer(" Hi")
        #expect(inputIds == [1, 29871, 6324])
    }

    /// https://github.com/huggingface/swift-transformers/issues/99
    @Test
    func robertaXLMTokenizer() async throws {
        let modelDirectory = try await downloadModel("intfloat/multilingual-e5-small")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = try tokenizer.encode(text: "query: how much protein should a female eat")
        let expected = [0, 41, 1294, 12, 3642, 5045, 21308, 5608, 10, 117776, 73203, 2]
        #expect(ids == expected)
    }

    /// https://github.com/huggingface/swift-transformers/issues/318
    @Test
    func kredorPunctuateAllTokenizer() async throws {
        let modelDirectory = try await downloadModel("kredor/punctuate-all")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = try tokenizer.encode(text: "okay so lets get started")
        let expected = [0, 68403, 221, 2633, 7, 2046, 26859, 2]
        #expect(ids == expected)
    }

    @Test
    func robertaXLMCanonicalTokenizer() async throws {
        let modelDirectory = try await downloadModel("FacebookAI/xlm-roberta-base")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = try tokenizer.encode(text: "okay so lets get started")
        let expected = [0, 68403, 221, 2633, 7, 2046, 26859, 2]
        #expect(ids == expected)
    }

    @Test
    func nllbTokenizer() async throws {
        let modelDirectory = try await downloadModel("Xenova/nllb-200-distilled-600M")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = try tokenizer.encode(text: "Why did the chicken cross the road?")
        let expected = [256047, 24185, 4077, 349, 1001, 22690, 83580, 349, 82801, 248130, 2]
        #expect(ids == expected)
    }

    /// DeepSeek's `tokenizer.json` has a `ByteLevel` post-processor, which does not add
    /// bos or eos. This matches canonical Python transformers v5 and upstream
    /// `huggingface/tokenizers`. Users wanting the leading bos can switch to a
    /// community re-upload whose `tokenizer.json` already uses `TemplateProcessing`.
    ///
    /// Reproduce with Python:
    ///     from transformers import AutoTokenizer
    ///     tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ///     assert tok.encode("Who are you?") == [15191, 525, 498, 30]
    @Test
    func deepSeekPostProcessor() async throws {
        let modelDirectory = try await downloadModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!
        #expect(try tokenizer.encode(text: "Who are you?") == [15191, 525, 498, 30])
    }

    /// Some Llama tokenizers already use a bos-prepending Template post-processor
    @Test
    func llamaPostProcessor() async throws {
        let modelDirectory = try await downloadModel("enterprise-explorers/Llama-2-7b-chat-coreml")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!
        #expect(try tokenizer.encode(text: "Who are you?") == [1, 11644, 526, 366, 29973])
    }

    @Test
    func localTokenizerFromDownload() async throws {
        let modelDirectory = try await downloadModel("pcuenq/gemma-tokenizer")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
    }

    @Test
    func bertCased() async throws {
        let modelDirectory = try await downloadModel("distilbert/distilbert-base-multilingual-cased")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(try tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
        #expect(try tokenizer.tokenize(text: "Car") == ["Car"])
    }

    @Test
    func bertCasedResaved() async throws {
        let modelDirectory = try await downloadModel("pcuenq/distilbert-base-multilingual-cased-tokenizer")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(try tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
    }

    @Test
    func bertUncased() async throws {
        let modelDirectory = try await downloadModel("google-bert/bert-base-uncased")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(try tokenizer.tokenize(text: "mąka") == ["ma", "##ka"])
        #expect(try tokenizer.encode(text: "mąka") == [101, 5003, 2912, 102])
        #expect(try tokenizer.tokenize(text: "département") == ["depart", "##ement"])
        #expect(try tokenizer.encode(text: "département") == [101, 18280, 13665, 102])
        #expect(try tokenizer.tokenize(text: "Car") == ["car"])

        #expect(try tokenizer.tokenize(text: "€4") == ["€", "##4"])
        #expect(try tokenizer.tokenize(text: "test $1 R2 #3 €4 £5 ¥6 ₣7 ₹8 ₱9 test") == ["test", "$", "1", "r", "##2", "#", "3", "€", "##4", "£5", "¥", "##6", "[UNK]", "₹", "##8", "₱", "##9", "test"])

        let text = "l'eure"
        let tokenized = try tokenizer.tokenize(text: text)
        #expect(tokenized == ["l", "'", "eu", "##re"])
        let encoded = try tokenizer.encode(text: text)
        #expect(encoded == [101, 1048, 1005, 7327, 2890, 102])
        let decoded = try tokenizer.decode(tokenIds: encoded, skipSpecialTokens: true)
        // Note: this matches the behaviour of the Python "slow" tokenizer, but the fast one produces "l ' eure"
        #expect(decoded == "l'eure")

        // Reading added_tokens from tokenizer.json
        #expect(tokenizer.convertTokenToId("[PAD]") == 0)
        #expect(tokenizer.convertTokenToId("[UNK]") == 100)
        #expect(tokenizer.convertTokenToId("[CLS]") == 101)
        #expect(tokenizer.convertTokenToId("[SEP]") == 102)
        #expect(tokenizer.convertTokenToId("[MASK]") == 103)
    }

    @Test
    func robertaEncodeDecode() async throws {
        let modelDirectory = try await downloadModel("FacebookAI/roberta-base")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(try tokenizer.tokenize(text: "l'eure") == ["l", "'", "e", "ure"])
        #expect(try tokenizer.encode(text: "l'eure") == [0, 462, 108, 242, 2407, 2])
        #expect(try tokenizer.decode(tokenIds: try tokenizer.encode(text: "l'eure"), skipSpecialTokens: true) == "l'eure")

        #expect(try tokenizer.tokenize(text: "mąka") == ["m", "Ä", "ħ", "ka"])
        #expect(try tokenizer.encode(text: "mąka") == [0, 119, 649, 5782, 2348, 2])

        #expect(try tokenizer.tokenize(text: "département") == ["d", "Ã©", "part", "ement"])
        #expect(try tokenizer.encode(text: "département") == [0, 417, 1140, 7755, 6285, 2])

        #expect(try tokenizer.tokenize(text: "Who are you?") == ["Who", "Ġare", "Ġyou", "?"])
        #expect(try tokenizer.encode(text: "Who are you?") == [0, 12375, 32, 47, 116, 2])

        #expect(try tokenizer.tokenize(text: " Who are you? ") == ["ĠWho", "Ġare", "Ġyou", "?", "Ġ"])
        #expect(try tokenizer.encode(text: " Who are you? ") == [0, 3394, 32, 47, 116, 1437, 2])

        #expect(try tokenizer.tokenize(text: "<s>Who are you?</s>") == ["<s>", "Who", "Ġare", "Ġyou", "?", "</s>"])
        #expect(try tokenizer.encode(text: "<s>Who are you?</s>") == [0, 0, 12375, 32, 47, 116, 2, 2])
    }

    @Test
    func tokenizerBackend() async throws {
        let modelDirectory = try await downloadModel("mlx-community/Ministral-3-3B-Instruct-2512-4bit")
        let tokenizerOpt = try await AutoTokenizer.from(directory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(try tokenizer.encode(text: "She took a train to the West") == [6284, 5244, 1261, 10018, 1317, 1278, 5046])
    }

}
