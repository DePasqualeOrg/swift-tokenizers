// Copyright © Hugging Face SAS

#if Swift
import Foundation
import Testing

@testable import Tokenizers
@testable import TokenizersSwiftBackend

@Suite("Post-processor functionality tests")
struct PostProcessorTests {
    @Suite("RoBERTa post-processing behavior")
    struct RoBERTaProcessingTests {
        @Test("Should keep spaces; uneven spaces; ignore addPrefixSpace")
        func keepsSpacesUnevenIgnoresAddPrefixSpace() throws {
            let config = Config([
                "cls": ["[HEAD]", 0 as UInt],
                "sep": ["[END]", 0 as UInt],
                "trimOffset": false,
                "addPrefixSpace": true,
            ])
            let tokens = [" The", " sun", "sets ", "  in  ", "   the  ", "west"]
            let expect = ["[HEAD]", " The", " sun", "sets ", "  in  ", "   the  ", "west", "[END]"]
            let processor = try RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: nil)
            #expect(output == expect)
        }

        @Test("Should leave only one space around each token")
        func normalizesSpacesAroundTokens() throws {
            let config = Config([
                "cls": ["[START]", 0 as UInt],
                "sep": ["[BREAK]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let expect = ["[START]", " The ", " sun", "sets ", " in ", " the ", "west", "[BREAK]"]
            let processor = try RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: nil)
            #expect(output == expect)
        }

        @Test("Should ignore empty tokens pair")
        func ignoresEmptyTokensPair() throws {
            let config = Config([
                "cls": ["[START]", 0 as UInt],
                "sep": ["[BREAK]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let tokensPair: [String] = []
            let expect = ["[START]", " The ", " sun", "sets ", " in ", " the ", "west", "[BREAK]"]
            let processor = try RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            #expect(output == expect)
        }

        @Test("Should trim all whitespace")
        func trimsAllWhitespace() throws {
            let config = Config([
                "cls": ["[CLS]", 0 as UInt],
                "sep": ["[SEP]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": false,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let expect = ["[CLS]", "The", "sun", "sets", "in", "the", "west", "[SEP]"]
            let processor = try RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: nil)
            #expect(output == expect)
        }

        @Test("Should add tokens")
        func addsTokensEnglish() throws {
            let config = Config([
                "cls": ["[CLS]", 0 as UInt],
                "sep": ["[SEP]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let tokensPair = [".", "The", " cat ", "   is ", " sitting  ", " on", "the ", "mat"]
            let expect = [
                "[CLS]", " The ", " sun", "sets ", " in ", " the ", "west", "[SEP]",
                "[SEP]", ".", "The", " cat ", " is ", " sitting ", " on", "the ",
                "mat", "[SEP]",
            ]
            let processor = try RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            #expect(output == expect)
        }

        @Test("Should add tokens (CJK)")
        func addsTokensCJK() throws {
            let config = Config([
                "cls": ["[CLS]", 0 as UInt],
                "sep": ["[SEP]", 0 as UInt],
                "trimOffset": true,
                "addPrefixSpace": true,
            ])
            let tokens = [" 你 ", " 好 ", ","]
            let tokensPair = [" 凯  ", "  蒂  ", "!"]
            let expect = ["[CLS]", " 你 ", " 好 ", ",", "[SEP]", "[SEP]", " 凯 ", " 蒂 ", "!", "[SEP]"]
            let processor = try RobertaProcessing(config: config)
            let output = processor.postProcess(tokens: tokens, tokensPair: tokensPair)
            #expect(output == expect)
        }
    }

    @Suite("TemplateProcessing expansion")
    struct TemplateProcessingTests {
        private static let llamaStyleConfig: Config = .init([
            "type": "TemplateProcessing",
            "single": [
                ["SpecialToken": ["id": "<s>", "type_id": 0]],
                ["Sequence": ["id": "A", "type_id": 0]],
            ],
            "pair": [
                ["SpecialToken": ["id": "<s>", "type_id": 0]],
                ["Sequence": ["id": "A", "type_id": 0]],
                ["SpecialToken": ["id": "<s>", "type_id": 1]],
                ["Sequence": ["id": "B", "type_id": 1]],
            ],
        ])

        @Test("Single template prepends bos to sequence A")
        func singleSequenceExpansion() throws {
            let processor = try TemplateProcessing(config: Self.llamaStyleConfig)
            let result = processor.postProcess(
                tokens: ["Hello", ",", "world"],
                tokensPair: nil,
                addSpecialTokens: true
            )
            #expect(result == ["<s>", "Hello", ",", "world"])
        }

        @Test("Pair template expands sequence A and B with both bos tokens")
        func pairSequenceExpansion() throws {
            let processor = try TemplateProcessing(config: Self.llamaStyleConfig)
            let result = processor.postProcess(
                tokens: ["first"],
                tokensPair: ["second"],
                addSpecialTokens: true
            )
            #expect(result == ["<s>", "first", "<s>", "second"])
        }

        @Test("addSpecialTokens false suppresses SpecialToken entries but keeps Sequences")
        func specialTokenSuppression() throws {
            let processor = try TemplateProcessing(config: Self.llamaStyleConfig)
            let single = processor.postProcess(
                tokens: ["x", "y"],
                tokensPair: nil,
                addSpecialTokens: false
            )
            #expect(single == ["x", "y"])

            let pair = processor.postProcess(
                tokens: ["a"],
                tokensPair: ["b"],
                addSpecialTokens: false
            )
            #expect(pair == ["a", "b"])
        }

        @Test("BERT-style cls/sep substitution around both sequences")
        func bertStyleSubstitution() throws {
            let config = Config([
                "type": "TemplateProcessing",
                "single": [
                    ["SpecialToken": ["id": "[CLS]", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                    ["SpecialToken": ["id": "[SEP]", "type_id": 0]],
                ],
                "pair": [
                    ["SpecialToken": ["id": "[CLS]", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                    ["SpecialToken": ["id": "[SEP]", "type_id": 0]],
                    ["Sequence": ["id": "B", "type_id": 1]],
                    ["SpecialToken": ["id": "[SEP]", "type_id": 1]],
                ],
            ])
            let processor = try TemplateProcessing(config: config)
            let result = processor.postProcess(
                tokens: ["Q"],
                tokensPair: ["A"],
                addSpecialTokens: true
            )
            #expect(result == ["[CLS]", "Q", "[SEP]", "A", "[SEP]"])
        }
    }

    @Suite("Post-processor error handling")
    struct PostProcessorErrorTests {
        @Test("Unsupported post-processor type throws unsupportedComponent")
        func unsupportedPostProcessorType() throws {
            let config = Config(["type": "NonExistentPostProcessor"])
            #expect(throws: TokenizerError.unsupportedComponent(kind: "PostProcessor", type: "NonExistentPostProcessor")) {
                try PostProcessorFactory.fromConfig(config: config)
            }
        }

        @Test("TemplateProcessing throws on missing single or pair")
        func templateMissingSingleOrPair() throws {
            #expect(throws: TokenizerError.missingConfigField(field: "single", component: "TemplateProcessing")) {
                try TemplateProcessing(config: Config(["pair": [] as [String]]))
            }

            #expect(throws: TokenizerError.missingConfigField(field: "pair", component: "TemplateProcessing")) {
                try TemplateProcessing(config: Config(["single": [] as [String]]))
            }
        }

        @Test("RobertaProcessing throws on missing sep or cls")
        func robertaMissingSepOrCls() throws {
            #expect(throws: TokenizerError.missingConfigField(field: "sep", component: "RobertaProcessing")) {
                try RobertaProcessing(config: Config(["cls": ["[CLS]", 0 as UInt]]))
            }

            #expect(throws: TokenizerError.missingConfigField(field: "cls", component: "RobertaProcessing")) {
                try RobertaProcessing(config: Config(["sep": ["[SEP]", 0 as UInt]]))
            }
        }

        @Test("BertProcessing throws on missing sep or cls")
        func bertMissingSepOrCls() throws {
            #expect(throws: TokenizerError.missingConfigField(field: "sep", component: "BertProcessing")) {
                try BertProcessing(config: Config(["cls": ["[CLS]", 0 as UInt]]))
            }

            #expect(throws: TokenizerError.missingConfigField(field: "cls", component: "BertProcessing")) {
                try BertProcessing(config: Config(["sep": ["[SEP]", 0 as UInt]]))
            }
        }

        @Test("Sequence post-processor throws on missing processors")
        func sequenceMissingProcessors() throws {
            let config = Config(["type": "Sequence"])
            #expect(throws: TokenizerError.missingConfigField(field: "processors", component: "Sequence post-processor")) {
                try SequenceProcessing(config: config)
            }
        }
    }
}
#endif
