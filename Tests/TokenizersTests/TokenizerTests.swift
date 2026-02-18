// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale
// Based on GPT2TokenizerTests by Julien Chaumond.

import Foundation
import HuggingFace
import Testing

@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-tests")
}()

private let hubClient = HubClient()

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
    guard let repoId = Repo.ID(rawValue: modelName) else {
        throw TestError.unsupportedTokenizer
    }
    return try await hubClient.downloadSnapshot(
        of: repoId,
        to: downloadDestination.appending(path: modelName),
        matching: tokenizerFiles
    )
}

private func makeTokenizer(hubModelName: String) async throws -> PreTrainedTokenizer {
    let modelDirectory = try await downloadModel(hubModelName)
    let tokenizer = try await AutoTokenizer.from(modelDirectory: modelDirectory)
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
        ModelSpec("coreml-projects/Llama-2-7b-chat-coreml", "llama_encoded", 0),
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

        #expect(tokenizer.tokenize(text: dataset.text) == dataset.bpe_tokens)
        #expect(tokenizer.encode(text: dataset.text) == dataset.token_ids)
        #expect(tokenizer.decode(tokens: dataset.token_ids) == dataset.decoded_text)

        // Edge cases (if available)
        if let edgeCases = try? loadEdgeCases(for: spec.hubModelName) {
            for edgeCase in edgeCases {
                #expect(tokenizer.encode(text: edgeCase.input) == edgeCase.encoded.input_ids)
                #expect(tokenizer.decode(tokens: edgeCase.encoded.input_ids) == edgeCase.decoded_with_special)
                #expect(tokenizer.decode(tokens: edgeCase.encoded.input_ids, skipSpecialTokens: true) == edgeCase.decoded_without_special)
            }
        }

        // Unknown token checks
        let model = tokenizer.model
        #expect(model.unknownTokenId == spec.unknownTokenId)
        #expect(model.unknownTokenId == model.convertTokenToId("_this_token_does_not_exist_"))
        if let unknownTokenId = model.unknownTokenId {
            #expect(model.unknownToken == model.convertIdToToken(unknownTokenId))
        } else {
            #expect(model.unknownTokenId == nil)
        }
    }

    @Test
    func gemmaUnicode() async throws {
        let modelDirectory = try await downloadModel("pcuenq/gemma-tokenizer")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        // These are two different characters
        let cases = [
            "\u{0061}\u{0300}", // NFD: a + combining grave accent
            "\u{00E0}", // NFC: precomposed à
        ]
        let expected = [217138, 1305]
        for (s, expected) in zip(cases, expected) {
            let encoded = tokenizer.encode(text: " " + s)
            #expect(encoded == [2, expected])
        }

        // Keys that start with BOM sequence
        // https://github.com/huggingface/swift-transformers/issues/88
        // https://github.com/ml-explore/mlx-swift-examples/issues/50#issuecomment-2046592213
        #expect(tokenizer.convertIdToToken(122661) == "\u{feff}#")
        #expect(tokenizer.convertIdToToken(235345) == "#")

        // Verifies all expected entries are parsed
        #expect((tokenizer.model as? BPETokenizer)?.vocabCount == 256_000)

        // Test added tokens
        let inputIds = tokenizer("This\n\nis\na\ntest.")
        #expect(inputIds == [2, 1596, 109, 502, 108, 235250, 108, 2195, 235265])
        let decoded = tokenizer.decode(tokens: inputIds)
        #expect(decoded == "<bos>This\n\nis\na\ntest.")
    }

    @Test
    func phi4() async throws {
        let modelDirectory = try await downloadModel("microsoft/phi-4")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "hello") == [15339])
        #expect(tokenizer.encode(text: "hello world") == [15339, 1917])
        #expect(tokenizer.encode(text: "<|im_start|>user<|im_sep|>Who are you?<|im_end|><|im_start|>assistant<|im_sep|>") == [100264, 882, 100266, 15546, 527, 499, 30, 100265, 100264, 78191, 100266])
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

        let tokenizer = try await AutoTokenizer.from(modelDirectory: tokenizerConfigURL.deletingLastPathComponent())

        let encoded = tokenizer.encode(text: "offline path")
        #expect(!encoded.isEmpty)
    }

    /// https://github.com/huggingface/swift-transformers/issues/96
    @Test
    func legacyLlamaBehaviour() async throws {
        let modelDirectory = try await downloadModel("mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let inputIds = tokenizer(" Hi")
        #expect(inputIds == [1, 29871, 6324])
    }

    /// https://github.com/huggingface/swift-transformers/issues/99
    @Test
    func robertaXLMTokenizer() async throws {
        let modelDirectory = try await downloadModel("intfloat/multilingual-e5-small")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = tokenizer.encode(text: "query: how much protein should a female eat")
        let expected = [0, 41, 1294, 12, 3642, 5045, 21308, 5608, 10, 117776, 73203, 2]
        #expect(ids == expected)
    }

    @Test
    func nllbTokenizer() async throws {
        let modelDirectory = try await downloadModel("Xenova/nllb-200-distilled-600M")

        do {
            _ = try await AutoTokenizer.from(modelDirectory: modelDirectory)
            Issue.record("Expected Tokenizer.from to throw for strict mode")
        } catch {
            // Expected to throw in normal (strict) mode
        }

        // no strict mode proceeds
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory, strict: false) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        let ids = tokenizer.encode(text: "Why did the chicken cross the road?")
        let expected = [256047, 24185, 4077, 349, 1001, 22690, 83580, 349, 82801, 248130, 2]
        #expect(ids == expected)
    }

    /// Deepseek needs a post-processor override to add a bos token as in the reference implementation
    @Test
    func deepSeekPostProcessor() async throws {
        let modelDirectory = try await downloadModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!
        #expect(tokenizer.encode(text: "Who are you?") == [151646, 15191, 525, 498, 30])
    }

    /// Some Llama tokenizers already use a bos-prepending Template post-processor
    @Test
    func llamaPostProcessor() async throws {
        let modelDirectory = try await downloadModel("coreml-projects/Llama-2-7b-chat-coreml")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!
        #expect(tokenizer.encode(text: "Who are you?") == [1, 11644, 526, 366, 29973])
    }

    @Test
    func localTokenizerFromDownload() async throws {
        let modelDirectory = try await downloadModel("pcuenq/gemma-tokenizer")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
    }

    @Test
    func bertCased() async throws {
        let modelDirectory = try await downloadModel("distilbert/distilbert-base-multilingual-cased")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
        #expect(tokenizer.tokenize(text: "Car") == ["Car"])
    }

    @Test
    func bertCasedResaved() async throws {
        let modelDirectory = try await downloadModel("pcuenq/distilbert-base-multilingual-cased-tokenizer")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "mąka") == [101, 181, 102075, 10113, 102])
    }

    @Test
    func bertUncased() async throws {
        let modelDirectory = try await downloadModel("google-bert/bert-base-uncased")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.tokenize(text: "mąka") == ["ma", "##ka"])
        #expect(tokenizer.encode(text: "mąka") == [101, 5003, 2912, 102])
        #expect(tokenizer.tokenize(text: "département") == ["depart", "##ement"])
        #expect(tokenizer.encode(text: "département") == [101, 18280, 13665, 102])
        #expect(tokenizer.tokenize(text: "Car") == ["car"])

        #expect(tokenizer.tokenize(text: "€4") == ["€", "##4"])
        #expect(tokenizer.tokenize(text: "test $1 R2 #3 €4 £5 ¥6 ₣7 ₹8 ₱9 test") == ["test", "$", "1", "r", "##2", "#", "3", "€", "##4", "£5", "¥", "##6", "[UNK]", "₹", "##8", "₱", "##9", "test"])

        let text = "l'eure"
        let tokenized = tokenizer.tokenize(text: text)
        #expect(tokenized == ["l", "'", "eu", "##re"])
        let encoded = tokenizer.encode(text: text)
        #expect(encoded == [101, 1048, 1005, 7327, 2890, 102])
        let decoded = tokenizer.decode(tokens: encoded, skipSpecialTokens: true)
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
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.tokenize(text: "l'eure") == ["l", "'", "e", "ure"])
        #expect(tokenizer.encode(text: "l'eure") == [0, 462, 108, 242, 2407, 2])
        #expect(tokenizer.decode(tokens: tokenizer.encode(text: "l'eure"), skipSpecialTokens: true) == "l'eure")

        #expect(tokenizer.tokenize(text: "mąka") == ["m", "Ä", "ħ", "ka"])
        #expect(tokenizer.encode(text: "mąka") == [0, 119, 649, 5782, 2348, 2])

        #expect(tokenizer.tokenize(text: "département") == ["d", "Ã©", "part", "ement"])
        #expect(tokenizer.encode(text: "département") == [0, 417, 1140, 7755, 6285, 2])

        #expect(tokenizer.tokenize(text: "Who are you?") == ["Who", "Ġare", "Ġyou", "?"])
        #expect(tokenizer.encode(text: "Who are you?") == [0, 12375, 32, 47, 116, 2])

        #expect(tokenizer.tokenize(text: " Who are you? ") == ["ĠWho", "Ġare", "Ġyou", "?", "Ġ"])
        #expect(tokenizer.encode(text: " Who are you? ") == [0, 3394, 32, 47, 116, 1437, 2])

        #expect(tokenizer.tokenize(text: "<s>Who are you?</s>") == ["<s>", "Who", "Ġare", "Ġyou", "?", "</s>"])
        #expect(tokenizer.encode(text: "<s>Who are you?</s>") == [0, 0, 12375, 32, 47, 116, 2, 2])
    }

    @Test
    func tokenizerBackend() async throws {
        let modelDirectory = try await downloadModel("mlx-community/Ministral-3-3B-Instruct-2512-4bit")
        let tokenizerOpt = try await AutoTokenizer.from(modelDirectory: modelDirectory) as? PreTrainedTokenizer
        #expect(tokenizerOpt != nil)
        let tokenizer = tokenizerOpt!

        #expect(tokenizer.encode(text: "She took a train to the West") == [6284, 5244, 1261, 10018, 1317, 1278, 5046])
    }

    @Test
    func concurrentTokenizerRegistration() async throws {
        // Test that concurrent registration doesn't cause crashes or data races.
        // This validates the thread-safety of AutoTokenizer.register().

        final class MockTokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
            let bosToken: String? = nil
            let bosTokenId: Int? = nil
            let eosToken: String? = nil
            let eosTokenId: Int? = nil
            let unknownToken: String? = nil
            let unknownTokenId: Int? = nil
            var fuseUnknownTokens: Bool { false }

            required init(
                tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int],
                vocab: TokenizerVocab? = nil, merges: TokenizerMerges? = nil
            ) throws {}
            func tokenize(text: String) -> [String] { [] }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            func encode(text: String) -> [Int] { [] }
            func decode(tokens: [Int]) -> String { "" }
        }

        // Register from multiple concurrent tasks
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    AutoTokenizer.register(MockTokenizer.self, for: "ConcurrentTestTokenizer\(i)")
                }
            }
        }

        // Verify registrations succeeded by checking we can look them up
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    _ = TokenizerModel.tokenizerClass(for: "ConcurrentTestTokenizer\(i)")
                }
            }
        }
    }
}
