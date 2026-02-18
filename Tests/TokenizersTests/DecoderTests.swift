// Copyright © Hugging Face SAS

import Foundation
import Testing

@testable import Tokenizers

@Suite("Tokenizer Decoder Tests")
struct DecoderTests {
    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test("Metaspace decoder with prefix space replacement")
    func metaspaceDecoder() {
        let decoder = MetaspaceDecoder(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
            ]))

        let tokens = ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        let decoded = decoder.decode(tokens: tokens)

        #expect(
            decoded == ["Hey", " my", " friend", " ", " <s>", " how", " are", " you"]
        )
    }

    @Test("WordPiece decoder with prefix and cleanup")
    func wordPieceDecoder() throws {
        let config = Config(["prefix": "##", "cleanup": true])
        let decoder = try WordPieceDecoder(config: config)

        let testCases: [([String], String)] = [
            (["##inter", "##national", "##ization"], "##internationalization"),
            (["##auto", "##mat", "##ic", "transmission"], "##automatic transmission"),
            (["who", "do", "##n't", "does", "n't", "can't"], "who don't doesn't can't"),
            (["##un", "##believ", "##able", "##fa", "##ntastic"], "##unbelievablefantastic"),
            (
                ["this", "is", "un", "##believ", "##able", "fa", "##ntastic"],
                "this is unbelievable fantastic"
            ),
            (["The", "##quick", "##brown", "fox"], "Thequickbrown fox"),
        ]

        for (tokens, expected) in testCases {
            let output = decoder.decode(tokens: tokens)
            #expect(output.joined() == expected)
        }
    }

    @Suite("Decoder error handling")
    struct DecoderErrorTests {
        @Test("Unsupported decoder type throws unsupportedComponent")
        func unsupportedDecoderType() throws {
            let config = Config(["type": "NonExistentDecoder"])
            #expect(throws: TokenizerError.unsupportedComponent(kind: "Decoder", type: "NonExistentDecoder")) {
                try DecoderFactory.fromConfig(config: config)
            }
        }

        @Test("WordPieceDecoder throws on missing prefix")
        func wordPieceMissingPrefix() throws {
            let config = Config(["cleanup": true])
            #expect(throws: TokenizerError.missingConfigField(field: "prefix", component: "WordPieceDecoder")) {
                try WordPieceDecoder(config: config)
            }
        }

        @Test("Sequence decoder throws on missing decoders")
        func sequenceMissingDecoders() throws {
            let config = Config(["type": "Sequence"])
            #expect(throws: TokenizerError.missingConfigField(field: "decoders", component: "Sequence decoder")) {
                try DecoderSequence(config: config)
            }
        }

        @Test("StripDecoder throws on missing content, start, or stop")
        func stripMissingFields() throws {
            #expect(throws: TokenizerError.missingConfigField(field: "content", component: "StripDecoder")) {
                try StripDecoder(config: Config(["start": 1, "stop": 1]))
            }

            #expect(throws: TokenizerError.missingConfigField(field: "start", component: "StripDecoder")) {
                try StripDecoder(config: Config(["content": " ", "stop": 1]))
            }

            #expect(throws: TokenizerError.missingConfigField(field: "stop", component: "StripDecoder")) {
                try StripDecoder(config: Config(["content": " ", "start": 1]))
            }
        }
    }
}
