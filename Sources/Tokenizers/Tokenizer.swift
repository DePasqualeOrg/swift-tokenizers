// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation

/// A type alias for chat messages, represented as key-value pairs.
public typealias Message = [String: any Sendable]

/// A type alias for tool specifications used in chat templating.
public typealias ToolSpec = [String: any Sendable]

/// Errors that can occur during tokenizer operations.
public enum TokenizerError: LocalizedError, Equatable, Sendable {
    case missingConfig
    case chatTemplate(String)
    case missingChatTemplate
    case invalidConfiguration(String)
    case internalError(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            "Tokenizer configuration is missing."
        case let .chatTemplate(message):
            "Chat template error: \(message)"
        case .missingChatTemplate:
            "This tokenizer does not have a chat template, and no template was passed."
        case let .invalidConfiguration(message):
            "Invalid tokenizer configuration: \(message)"
        case let .internalError(message):
            "Internal tokenizer error: \(message)"
        }
    }
}

package enum JSONBridge {
    package static func foundationObject(from value: Any) throws -> Any {
        switch value {
        case let value as String:
            return value
        case let value as Bool:
            return value
        case let value as Int:
            return value
        case let value as Int8:
            return Int(value)
        case let value as Int16:
            return Int(value)
        case let value as Int32:
            return Int(value)
        case let value as Int64:
            return Int(value)
        case let value as UInt:
            return value
        case let value as UInt8:
            return Int(value)
        case let value as UInt16:
            return Int(value)
        case let value as UInt32:
            return Int(value)
        case let value as UInt64:
            return Int(value)
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as NSNumber:
            return value
        case is NSNull:
            return NSNull()
        case let value as [String: any Sendable]:
            return try Dictionary(
                uniqueKeysWithValues: value.map { key, nestedValue in
                    (key, try foundationObject(from: nestedValue))
                })
        case let value as [String: Any]:
            return try Dictionary(
                uniqueKeysWithValues: value.map { key, nestedValue in
                    (key, try foundationObject(from: nestedValue))
                })
        case let value as [any Sendable]:
            return try value.map { try foundationObject(from: $0) }
        case let value as [Any]:
            return try value.map { try foundationObject(from: $0) }
        default:
            let mirror = Mirror(reflecting: value)
            switch mirror.displayStyle {
            case .optional:
                guard let child = mirror.children.first else {
                    return NSNull()
                }
                return try foundationObject(from: child.value)
            case .collection, .set:
                return try mirror.children.map { try foundationObject(from: $0.value) }
            case .dictionary:
                var result: [String: Any] = [:]
                for child in mirror.children {
                    let entryMirror = Mirror(reflecting: child.value)
                    let entryChildren = Array(entryMirror.children)
                    guard
                        entryChildren.count == 2,
                        let key = entryChildren[0].value as? String
                    else {
                        throw TokenizerError.invalidConfiguration(
                            "Tokenizer JSON bridge only supports string-keyed dictionaries"
                        )
                    }
                    result[key] = try foundationObject(from: entryChildren[1].value)
                }
                return result
            default:
                throw TokenizerError.invalidConfiguration(
                    "Tokenizer JSON bridge cannot encode value of type \(type(of: value))"
                )
            }
        }
    }

    package static func jsonString(from value: Any) throws -> String {
        let object = try foundationObject(from: value)
        let data = try JSONSerialization.data(withJSONObject: object)
        return String(decoding: data, as: UTF8.self)
    }
}

/// Units used by token offset spans in a metadata-rich encoding.
public enum OffsetUnit: String, Codable, Sendable {
    /// Offsets are UTF-8 byte positions in the original input.
    case utf8
    /// Offsets are Unicode scalar positions in the original input, matching Rust/Python "char" offsets rather than Swift `Character` indices.
    case unicodeScalar
}

/// A half-open offset span associated with a token, expressed in its encoding's `OffsetUnit`.
///
/// The span values are positions in the original input expressed in `OffsetUnit` (UTF-8 bytes or
/// Unicode scalars). They are *not* directly usable to index into a Swift `String`, whose natural
/// indices address extended grapheme clusters. To slice the original text, index into the matching
/// view: `text.utf8` for `.utf8` offsets or `text.unicodeScalars` for `.unicodeScalar` offsets.
public struct OffsetSpan: Codable, Equatable, Sendable {
    /// Inclusive lower bound of the span, in the encoding's `OffsetUnit`.
    public let start: Int
    /// Exclusive upper bound of the span, in the encoding's `OffsetUnit`.
    public let end: Int

    /// Creates a span covering `start..<end`.
    public init(start: Int, end: Int) {
        self.start = start
        self.end = end
    }

    /// The span as a half-open `Range<Int>` in the encoding's `OffsetUnit`.
    ///
    /// This range is not a `Range<String.Index>` — see the type's documentation for how to slice
    /// the original input correctly.
    public var range: Range<Int> {
        start..<end
    }
}

/// A metadata-rich tokenizer encoding.
public struct TokenizerEncoding: Codable, Equatable, Sendable {
    /// Numeric vocabulary IDs for each token.
    public let tokenIds: [Int]
    /// Token type IDs (segment IDs) for each token. Used by sentence-pair models; typically `0` for single-sequence input.
    public let tokenTypeIds: [Int]
    /// The subword token strings produced by the tokenizer, parallel to `tokenIds`.
    public let tokens: [String]
    /// Pre-tokenizer word indices associated with each token. Special tokens and tokens without an input word have `nil`.
    public let wordIndices: [Int?]
    /// Span in the original input covered by each token, expressed in `offsetUnit`. Special tokens have a zero-length span.
    public let offsetSpans: [OffsetSpan]
    /// Per-token mask whose value is `1` for special tokens added by the tokenizer and `0` otherwise.
    public let specialTokensMask: [Int]
    /// Per-token attention mask whose value is `1` for real tokens and `0` for padding.
    public let attentionMask: [Int]
    /// Index of the input sequence each token belongs to. `nil` for tokens that are not associated with any input sequence.
    public let sequenceIndices: [Int?]
    /// Number of distinct input sequences represented in this encoding (typically `1`, or `2` for sentence-pair input).
    public let sequenceCount: Int
    /// Additional encodings produced when the input was truncated and the tokenizer is configured to return overflow.
    public let overflowEncodings: [TokenizerEncoding]
    /// Unit used by every value in `offsetSpans`.
    public let offsetUnit: OffsetUnit

    /// Creates an encoding from already-extracted backend output.
    public init(
        tokenIds: [Int],
        tokenTypeIds: [Int],
        tokens: [String],
        wordIndices: [Int?],
        offsetSpans: [OffsetSpan],
        specialTokensMask: [Int],
        attentionMask: [Int],
        sequenceIndices: [Int?],
        sequenceCount: Int,
        overflowEncodings: [TokenizerEncoding] = [],
        offsetUnit: OffsetUnit = .unicodeScalar
    ) {
        self.tokenIds = tokenIds
        self.tokenTypeIds = tokenTypeIds
        self.tokens = tokens
        self.wordIndices = wordIndices
        self.offsetSpans = offsetSpans
        self.specialTokensMask = specialTokensMask
        self.attentionMask = attentionMask
        self.sequenceIndices = sequenceIndices
        self.sequenceCount = sequenceCount
        self.overflowEncodings = overflowEncodings
        self.offsetUnit = offsetUnit
    }

    /// `true` when the encoding contains no tokens.
    public var isEmpty: Bool {
        tokenIds.isEmpty
    }

    /// Total number of tokens in the encoding, including special tokens.
    public var count: Int {
        tokenIds.count
    }

    /// Returns the offset span for the token at `tokenIndex`, or `nil` if the index is out of range or the token is not associated with any input sequence.
    public func offsetSpan(forTokenIndex tokenIndex: Int) -> OffsetSpan? {
        guard sequenceIndex(forTokenIndex: tokenIndex) != nil else { return nil }
        guard tokenIndex >= 0, tokenIndex < offsetSpans.count else { return nil }
        return offsetSpans[tokenIndex]
    }

    /// Returns the input sequence index the token at `tokenIndex` belongs to, or `nil` if the index is out of range or the token is not associated with any input sequence.
    public func sequenceIndex(forTokenIndex tokenIndex: Int) -> Int? {
        guard tokenIndex >= 0, tokenIndex < sequenceIndices.count else { return nil }
        return sequenceIndices[tokenIndex]
    }

    /// Returns the pre-tokenizer word index for the token at `tokenIndex`, or `nil` if the index is out of range, the token is not associated with any input sequence, or the token has no source word (for example, a special token).
    public func wordIndex(forTokenIndex tokenIndex: Int) -> Int? {
        guard sequenceIndex(forTokenIndex: tokenIndex) != nil else { return nil }
        guard tokenIndex >= 0, tokenIndex < wordIndices.count else { return nil }
        return wordIndices[tokenIndex]
    }

    /// Returns the input sequence index and offset span for the token at `tokenIndex`, or `nil` if the index is out of range or the token is not associated with any input sequence.
    ///
    /// Mirrors the upstream Rust `token_to_chars(token) -> Option<(sequence_id, offsets)>` API.
    public func tokenInfo(forTokenIndex tokenIndex: Int) -> (sequenceIndex: Int, span: OffsetSpan)? {
        guard let sequenceIndex = sequenceIndex(forTokenIndex: tokenIndex) else { return nil }
        guard tokenIndex >= 0, tokenIndex < offsetSpans.count else { return nil }
        return (sequenceIndex, offsetSpans[tokenIndex])
    }

    /// Returns the input sequence index and pre-tokenizer word index for the token at `tokenIndex`, or `nil` if the index is out of range, the token is not associated with any input sequence, or the token has no source word.
    ///
    /// Mirrors the upstream Rust `token_to_word(token) -> Option<(sequence_id, word)>` API.
    public func wordInfo(forTokenIndex tokenIndex: Int) -> (sequenceIndex: Int, wordIndex: Int)? {
        guard let sequenceIndex = sequenceIndex(forTokenIndex: tokenIndex) else { return nil }
        guard tokenIndex >= 0, tokenIndex < wordIndices.count else { return nil }
        guard let wordIndex = wordIndices[tokenIndex] else { return nil }
        return (sequenceIndex, wordIndex)
    }

    /// Returns the index of the first token in the given sequence whose offset span contains `offsetValue`, or `nil` if no token does.
    ///
    /// `offsetValue` is interpreted in the encoding's `offsetUnit`. The `sequenceIndex` argument is required
    /// because the same offset value can refer to different positions in different input sequences (matching the
    /// upstream Rust `char_to_token(pos, sequence_id)` API).
    public func tokenIndex(containingOffset offsetValue: Int, sequenceIndex: Int) -> Int? {
        offsetSpans.indices.first { tokenIndex in
            guard sequenceIndices.indices.contains(tokenIndex), sequenceIndices[tokenIndex] == sequenceIndex else {
                return false
            }
            let offset = offsetSpans[tokenIndex]
            return offsetValue >= offset.start && offsetValue < offset.end
        }
    }

    /// Returns the pre-tokenizer word index that contains `offsetValue` in the given sequence, or `nil` if no token covers that offset.
    ///
    /// `offsetValue` is interpreted in the encoding's `offsetUnit`.
    public func wordIndex(containingOffset offsetValue: Int, sequenceIndex: Int) -> Int? {
        guard let tokenIndex = tokenIndex(containingOffset: offsetValue, sequenceIndex: sequenceIndex) else {
            return nil
        }
        return wordIndex(forTokenIndex: tokenIndex)
    }

    /// Returns the contiguous half-open range of token indices produced from `wordIndex` in the given sequence, or `nil` if no token maps to that word.
    ///
    /// Mirrors the upstream Rust `word_to_tokens(word, sequence_id)` API.
    public func tokenRange(forWordIndex wordIndex: Int, sequenceIndex: Int) -> Range<Int>? {
        var lower: Int?
        var upper: Int?

        for tokenIndex in wordIndices.indices {
            guard sequenceIndices.indices.contains(tokenIndex), sequenceIndices[tokenIndex] == sequenceIndex else {
                continue
            }
            guard wordIndices[tokenIndex] == wordIndex else {
                continue
            }
            lower = min(lower ?? tokenIndex, tokenIndex)
            upper = max(upper ?? (tokenIndex + 1), tokenIndex + 1)
        }

        guard let lower, let upper else {
            return nil
        }
        return lower..<upper
    }

    /// Returns the offset span covering the original input range that produced `wordIndex` in the given sequence, or `nil` if no token maps to that word.
    public func offsetSpan(forWordIndex wordIndex: Int, sequenceIndex: Int) -> OffsetSpan? {
        guard let tokenRange = tokenRange(forWordIndex: wordIndex, sequenceIndex: sequenceIndex) else {
            return nil
        }
        guard tokenRange.lowerBound < tokenRange.upperBound else {
            return nil
        }
        guard
            offsetSpans.indices.contains(tokenRange.lowerBound),
            offsetSpans.indices.contains(tokenRange.upperBound - 1)
        else {
            return nil
        }

        return OffsetSpan(
            start: offsetSpans[tokenRange.lowerBound].start,
            end: offsetSpans[tokenRange.upperBound - 1].end
        )
    }
}

/// Overrides for selecting which chat template to use when applying chat formatting.
public enum ChatTemplateOverride: Sendable {
    /// A Jinja template to use for the conversation.
    ///
    /// Normally it is not necessary to provide a template, since it will be read from the tokenizer config.
    case literal(String)

    /// For models whose tokenizer config includes multiple chat templates, the template can be specified by name.
    ///
    /// Normally this is not necessary.
    case name(String)
}

/// A complete tokenizer interface supporting encoding, decoding, and chat template functionality.
///
/// This is the main protocol that defines all tokenizer operations, including text processing,
/// chat template application, and special token handling.
public protocol Tokenizer: Sendable {
    /// Splits `text` into the subword token strings the tokenizer's model produces, without converting them to IDs and without adding special tokens.
    func tokenize(text: String) -> [String]

    /// Encodes `text` (optionally paired with a second sequence) to vocabulary IDs.
    ///
    /// - Parameters:
    ///   - text: Primary text to encode.
    ///   - textPair: Secondary text for sentence-pair models, or `nil` for single-sequence encoding. Maps to Rust `EncodeInput::Dual` when supplied.
    ///   - addSpecialTokens: When `true`, applies the tokenizer's post-processor so that any required special tokens (such as bos/eos) are inserted.
    /// - Returns: Vocabulary IDs for the encoded input.
    func encode(text: String, textPair: String?, addSpecialTokens: Bool) -> [Int]

    /// Encodes a batch of inputs to vocabulary IDs in parallel. Each pair element supplies the primary text and an optional `textPair` for sentence-pair encoding.
    ///
    /// - Parameters:
    ///   - inputs: The batch of inputs.
    ///   - addSpecialTokens: When `true`, applies the tokenizer's post-processor to every entry.
    /// - Returns: One token-ID array per input, in the same order.
    func encodeBatch(_ inputs: [(text: String, textPair: String?)], addSpecialTokens: Bool) -> [[Int]]

    /// Encodes `text` (optionally paired with a second sequence) and returns a ``TokenizerEncoding`` containing token IDs along with token strings, masks, offset spans, and word/sequence indices.
    ///
    /// - Parameters:
    ///   - text: Primary text to encode.
    ///   - textPair: Secondary text for sentence-pair models, or `nil` for single-sequence encoding.
    ///   - addSpecialTokens: When `true`, applies the tokenizer's post-processor.
    ///   - offsetUnit: Unit used by the returned offset spans.
    /// - Throws: ``TokenizerError`` if encoding fails.
    /// - Returns: Token IDs and associated metadata for the encoded input.
    func encodeWithMetadata(
        text: String,
        textPair: String?,
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> TokenizerEncoding

    /// Encodes a batch of inputs in parallel and returns a ``TokenizerEncoding`` for each.
    ///
    /// - Parameters:
    ///   - inputs: The batch of inputs.
    ///   - addSpecialTokens: When `true`, applies the tokenizer's post-processor to every entry.
    ///   - offsetUnit: Unit used by the returned offset spans.
    /// - Throws: ``TokenizerError`` if any entry fails to encode.
    /// - Returns: One ``TokenizerEncoding`` per input, in the same order.
    func encodeBatchWithMetadata(
        _ inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> [TokenizerEncoding]

    /// Decodes a sequence of vocabulary IDs back to text.
    ///
    /// - Parameters:
    ///   - tokenIds: Vocabulary IDs to decode.
    ///   - skipSpecialTokens: When `true`, omits any special tokens from the decoded string.
    /// - Returns: Decoded text.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String

    /// Returns the vocabulary ID for `token`, or `nil` if it is not in the vocabulary.
    func convertTokenToId(_ token: String) -> Int?

    /// Returns the token string for vocabulary ID `id`, or `nil` if no such ID exists.
    func convertIdToToken(_ id: Int) -> String?

    /// The beginning-of-sequence token, if the tokenizer defines one.
    var bosToken: String? { get }

    /// The end-of-sequence token, if the tokenizer defines one.
    var eosToken: String? { get }

    /// The unknown token, if the tokenizer defines one.
    var unknownToken: String? { get }

    /// The size of the tokenizer's vocabulary.
    ///
    /// - Parameter withAddedTokens: When `true`, the result is the base model vocabulary size plus
    ///   any added tokens that are not already present in the base vocabulary, matching
    ///   `len(tokenizer)` in the Hugging Face Python library. When `false`, the result is the base
    ///   model vocabulary size only, matching Python's `vocab_size` property and the
    ///   `get_vocab_size(with_added_tokens: false)` flavor of the upstream Rust API.
    /// - Returns: The vocabulary size.
    func getVocabSize(withAddedTokens: Bool) -> Int

    /// `true` when the tokenizer carries at least one chat template that ``applyChatTemplate(messages:chatTemplate:addGenerationPrompt:truncation:maxLength:tools:additionalContext:)`` can use without an override.
    var hasChatTemplate: Bool { get }

    /// Renders `messages` through the selected chat template and tokenizes the resulting string.
    ///
    /// - Parameters:
    ///   - messages: The conversation to render. Each message is a dictionary; the keys expected depend on the template (typically `"role"` and `"content"`).
    ///   - chatTemplate: Optional override that selects either a literal Jinja template or a named template from the tokenizer config. Pass `nil` to use the tokenizer's default.
    ///   - addGenerationPrompt: When `true`, appends the prompt suffix that cues the model to begin a new assistant turn.
    ///   - truncation: When `true`, truncates the output to fit within `maxLength` (or the model's max length).
    ///   - maxLength: Maximum token count when truncation is enabled. Capped by the model's max length when both are set.
    ///   - tools: Optional list of tool specifications passed to the template (used by tool-calling models).
    ///   - additionalContext: Extra key-value pairs merged into the template's render context.
    /// - Throws: ``TokenizerError`` if no template is available, the override cannot be resolved, or the template fails to render.
    /// - Returns: Vocabulary IDs for the rendered chat prompt.
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateOverride?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public var hasChatTemplate: Bool { false }

    /// Convenience overload that supplies default values for every parameter except `messages`.
    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateOverride? = nil,
        addGenerationPrompt: Bool = true,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            truncation: truncation,
            maxLength: maxLength,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    /// Convenience overload that takes a literal Jinja template string instead of a ``ChatTemplateOverride``.
    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: String,
        addGenerationPrompt: Bool = true,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages,
            chatTemplate: .literal(chatTemplate),
            addGenerationPrompt: addGenerationPrompt,
            truncation: truncation,
            maxLength: maxLength,
            tools: tools,
            additionalContext: additionalContext
        )
    }
}

public extension Tokenizer {
    /// Single-sequence convenience: encodes `text` to vocabulary IDs.
    func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, textPair: nil, addSpecialTokens: addSpecialTokens)
    }

    /// Single-sequence convenience: encodes `text` with metadata using ``OffsetUnit/unicodeScalar`` for offset spans.
    func encodeWithMetadata(
        text: String,
        addSpecialTokens: Bool = true,
        offsetUnit: OffsetUnit = .unicodeScalar
    ) throws -> TokenizerEncoding {
        try encodeWithMetadata(
            text: text,
            textPair: nil,
            addSpecialTokens: addSpecialTokens,
            offsetUnit: offsetUnit
        )
    }

    /// Single-sequence batch convenience: encodes each text without a pair sequence.
    func encodeBatch(texts: [String], addSpecialTokens: Bool = true) -> [[Int]] {
        encodeBatch(texts.map { ($0, nil) }, addSpecialTokens: addSpecialTokens)
    }

    /// Single-sequence batch convenience: encodes each text with metadata.
    func encodeBatchWithMetadata(
        texts: [String],
        addSpecialTokens: Bool = true,
        offsetUnit: OffsetUnit = .unicodeScalar
    ) throws -> [TokenizerEncoding] {
        try encodeBatchWithMetadata(
            texts.map { ($0, nil) },
            addSpecialTokens: addSpecialTokens,
            offsetUnit: offsetUnit
        )
    }

    /// Allows the tokenizer to be invoked as a function, equivalent to calling ``encode(text:addSpecialTokens:)``.
    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    /// Decodes a sequence of vocabulary IDs back to text without skipping special tokens.
    func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    /// Returns the result of ``convertTokenToId(_:)`` for each input token.
    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    /// Returns the result of ``convertIdToToken(_:)`` for each input ID.
    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }

    /// The vocabulary ID of ``bosToken``, if present.
    var bosTokenId: Int? { bosToken.flatMap { convertTokenToId($0) } }
    /// The vocabulary ID of ``eosToken``, if present.
    var eosTokenId: Int? { eosToken.flatMap { convertTokenToId($0) } }
    /// The vocabulary ID of ``unknownToken``, if present.
    var unknownTokenId: Int? { unknownToken.flatMap { convertTokenToId($0) } }
}

/// Concrete ``Tokenizer`` conformance returned by ``AutoTokenizer``. Wraps a
/// `RustTokenizer` and applies chat-template policy from the runtime configuration.
package final class PreTrainedTokenizer: Sendable, Tokenizer {
    package let rustTokenizer: RustTokenizer
    package let runtimeConfiguration: TokenizerRuntimeConfiguration

    public var bosToken: String? { runtimeConfiguration.bosToken ?? rustTokenizer.bosToken }
    public var eosToken: String? { runtimeConfiguration.eosToken ?? rustTokenizer.eosToken }
    public var unknownToken: String? { runtimeConfiguration.unknownToken ?? rustTokenizer.unknownToken }
    public var bosTokenId: Int? { rustTokenizer.bosTokenId }
    public var eosTokenId: Int? { rustTokenizer.eosTokenId }
    public var unknownTokenId: Int? { rustTokenizer.unknownTokenId }

    public func getVocabSize(withAddedTokens: Bool) -> Int {
        rustTokenizer.getVocabSize(withAddedTokens: withAddedTokens)
    }

    package init(
        rustTokenizer: RustTokenizer,
        runtimeConfiguration: TokenizerRuntimeConfiguration
    ) {
        self.rustTokenizer = rustTokenizer
        self.runtimeConfiguration = runtimeConfiguration
    }

    package func selectedChatTemplate(
        chatTemplate: ChatTemplateOverride?,
        tools: [ToolSpec]?
    ) throws -> String {
        try runtimeConfiguration.selectedChatTemplate(chatTemplate: chatTemplate, tools: tools)
    }

    package func chatTemplateContextObject(
        messages: [Message],
        addGenerationPrompt: Bool,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String: Any] {
        try runtimeConfiguration.chatTemplateContextObject(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    package func effectiveChatTemplateMaxLength(_ maxLength: Int?) -> Int? {
        runtimeConfiguration.effectiveChatTemplateMaxLength(maxLength)
    }

    package func renderChatTemplateToString(
        template: String,
        contextObject: [String: Any]
    ) throws -> String {
        try rustTokenizer.renderChatTemplate(template: template, contextObject: contextObject)
    }

    package func renderChatTemplateToString(
        messages: [Message],
        chatTemplate: ChatTemplateOverride?,
        addGenerationPrompt: Bool,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> String {
        let selectedTemplate = try selectedChatTemplate(chatTemplate: chatTemplate, tools: tools)
        let contextObject = try chatTemplateContextObject(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            tools: tools,
            additionalContext: additionalContext
        )
        return try renderChatTemplateToString(template: selectedTemplate, contextObject: contextObject)
    }

    package func cleanUp(text: String) -> String {
        guard runtimeConfiguration.cleanUpTokenizationSpaces else { return text }

        return
            text
            .replacingOccurrences(of: " .", with: ".")
            .replacingOccurrences(of: " ?", with: "?")
            .replacingOccurrences(of: " !", with: "!")
            .replacingOccurrences(of: " ,", with: ",")
            .replacingOccurrences(of: " ' ", with: "'")
            .replacingOccurrences(of: " n't", with: "n't")
            .replacingOccurrences(of: " 'm", with: "'m")
            .replacingOccurrences(of: " 's", with: "'s")
            .replacingOccurrences(of: " 've", with: "'ve")
            .replacingOccurrences(of: " 're", with: "'re")
    }

    public func tokenize(text: String) -> [String] {
        rustTokenizer.tokenize(text: text)
    }

    public func encode(text: String, textPair: String?, addSpecialTokens: Bool) -> [Int] {
        rustTokenizer.encode(text: text, textPair: textPair, addSpecialTokens: addSpecialTokens)
    }

    public func encodeBatch(_ inputs: [(text: String, textPair: String?)], addSpecialTokens: Bool) -> [[Int]] {
        rustTokenizer.encodeBatch(inputs, addSpecialTokens: addSpecialTokens)
    }

    public func encodeWithMetadata(
        text: String,
        textPair: String?,
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> TokenizerEncoding {
        try rustTokenizer.encodeWithMetadata(
            text: text,
            textPair: textPair,
            addSpecialTokens: addSpecialTokens,
            offsetUnit: offsetUnit
        )
    }

    public func encodeBatchWithMetadata(
        _ inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> [TokenizerEncoding] {
        try rustTokenizer.encodeBatchWithMetadata(
            inputs,
            addSpecialTokens: addSpecialTokens,
            offsetUnit: offsetUnit
        )
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let decoded = rustTokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        return cleanUp(text: decoded)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        rustTokenizer.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        rustTokenizer.convertIdToToken(id)
    }

    public var hasChatTemplate: Bool {
        runtimeConfiguration.hasChatTemplate
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateOverride?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let selectedTemplate = try selectedChatTemplate(chatTemplate: chatTemplate, tools: tools)
        let contextObject = try chatTemplateContextObject(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            tools: tools,
            additionalContext: additionalContext
        )
        return try rustTokenizer.applyChatTemplate(
            template: selectedTemplate,
            contextObject: contextObject,
            truncation: truncation,
            maxLength: effectiveChatTemplateMaxLength(maxLength)
        )
    }
}

/// A namespace for automatically creating appropriate tokenizer instances.
public enum AutoTokenizer {}
