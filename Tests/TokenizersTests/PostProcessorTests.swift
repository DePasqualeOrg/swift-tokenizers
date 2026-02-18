// Copyright © Hugging Face SAS

import Foundation
import Testing

@testable import Tokenizers

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
