// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation
import Jinja

/// Vocabulary extracted from tokenizer.json for fast BPE or Unigram initialization.
///
/// - Note: `@unchecked Sendable` is safe because the underlying data is immutable after extraction from JSON.
public enum TokenizerVocab: @unchecked Sendable {
    /// BPE vocabulary: dictionary mapping token strings to token IDs.
    case bpe(NSDictionary)
    /// Unigram vocabulary: array of [token, score] pairs.
    case unigram(NSArray)
}

/// Merge rules extracted from tokenizer.json for fast BPE initialization.
///
/// - Note: `@unchecked Sendable` is safe because the underlying data is immutable after extraction from JSON.
public struct TokenizerMerges: @unchecked Sendable {
    /// The raw merge rules as extracted from JSON.
    public let rules: [Any]

    public init(_ rules: [Any]) {
        self.rules = rules
    }
}

/// A type alias for chat messages, represented as key-value pairs.
public typealias Message = [String: any Sendable]

/// A type alias for tool specifications used in chat templating.
public typealias ToolSpec = [String: any Sendable]

/// Errors that can occur during tokenizer operations.
public enum TokenizerError: LocalizedError, Equatable {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case chatTemplate(String)
    case missingChatTemplate
    case tooLong(String)
    case mismatchedConfig(String)
    case unsupportedComponent(kind: String, type: String)
    case missingConfigField(field: String, component: String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            "Tokenizer configuration is missing."
        case .missingTokenizerClassInConfig:
            "The tokenizer class is not specified in the configuration."
        case let .unsupportedTokenizer(name):
            "The tokenizer type '\(name)' is not supported."
        case .missingVocab:
            "Vocabulary file is missing from the tokenizer configuration."
        case .malformedVocab:
            "The vocabulary file is malformed or corrupted."
        case let .chatTemplate(message):
            "Chat template error: \(message)"
        case .missingChatTemplate:
            "This tokenizer does not have a chat template, and no template was passed."
        case let .tooLong(message):
            "Input is too long: \(message)"
        case let .mismatchedConfig(message):
            "Tokenizer configuration mismatch: \(message)"
        case let .unsupportedComponent(kind, type):
            "Unsupported \(kind) type: '\(type)'"
        case let .missingConfigField(field, component):
            "Missing '\(field)' in \(component) configuration"
        }
    }
}

/// A protocol defining the core tokenization functionality.
///
/// This protocol defines the fundamental operations that any tokenization model must support,
/// including converting between text and tokens, and between tokens and their numeric IDs.
public protocol TokenizingModel {
    /// Tokenizes the input text into a sequence of tokens.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func tokenize(text: String) -> [String]

    /// Alias for `tokenize` that allows the instance to be called as a function.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func callAsFunction(_ text: String) -> [String]

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if the token is not in the vocabulary
    func convertTokenToId(_ token: String) -> Int?

    /// Converts multiple token strings to their corresponding numeric IDs.
    ///
    /// - Parameter tokens: An array of token strings to convert
    /// - Returns: An array of numeric IDs, with nil values for tokens not in the vocabulary
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is not valid
    func convertIdToToken(_ id: Int) -> String?

    /// Converts multiple numeric token IDs back to their string representations.
    ///
    /// - Parameter ids: An array of numeric token IDs to convert
    /// - Returns: An array of token strings, with nil values for invalid IDs
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    /// The beginning-of-sequence token string, if defined.
    var bosToken: String? { get }

    /// The numeric ID of the beginning-of-sequence token, if defined.
    var bosTokenId: Int? { get }

    /// The end-of-sequence token string, if defined.
    var eosToken: String? { get }

    /// The numeric ID of the end-of-sequence token, if defined.
    var eosTokenId: Int? { get }

    /// The unknown token string used for out-of-vocabulary words.
    var unknownToken: String? { get }

    /// The numeric ID of the unknown token.
    var unknownTokenId: Int? { get }

    /// Whether consecutive unknown tokens should be fused together.
    var fuseUnknownTokens: Bool { get }
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] {
        tokenize(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

/// A tokenizer model that can be initialized from Hugging Face Hub configuration data.
///
/// This protocol extends `TokenizingModel` with the ability to be created from configuration
/// files typically found in tokenizer repositories on the Hugging Face Hub.
public protocol PreTrainedTokenizerModel: TokenizingModel {
    /// Initializes a tokenizer model from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration (typically from tokenizer_config.json)
    ///   - tokenizerData: The tokenizer data (typically from tokenizer.json)
    ///   - addedTokens: A dictionary mapping added token strings to their IDs
    ///   - vocab: Pre-extracted vocabulary for fast initialization (nil to parse from Config)
    ///   - merges: Pre-extracted merge rules for fast initialization (nil to parse from Config)
    /// - Throws: `TokenizerError` if the configuration is invalid or missing required data
    init(
        tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int],
        vocab: TokenizerVocab?, merges: TokenizerMerges?) throws
}

enum TokenizerModel {
    /// User-registered custom tokenizer classes.
    /// Register via `AutoTokenizer.register(_:for:)`.
    /// Uses `NSLock` instead of `OSAllocatedUnfairLock` because protocol metatypes
    /// don't conform to `Sendable`, which `OSAllocatedUnfairLock` requires for its state.
    private static var _registeredTokenizers: [String: PreTrainedTokenizerModel.Type] = [:]
    private static let registrationLock = NSLock()

    static func registerTokenizer(_ tokenizerClass: PreTrainedTokenizerModel.Type, for name: String) {
        registrationLock.lock()
        defer { registrationLock.unlock() }
        _registeredTokenizers[name] = tokenizerClass
    }

    /// Returns the tokenizer class for the given name, checking registered tokenizers first.
    static func tokenizerClass(for name: String) -> PreTrainedTokenizerModel.Type? {
        registrationLock.lock()
        defer { registrationLock.unlock() }
        return _registeredTokenizers[name] ?? knownTokenizers[name]
    }

    static let knownTokenizers: [String: PreTrainedTokenizerModel.Type] = [
        "BertTokenizer": BertTokenizer.self,
        "CodeGenTokenizer": BPETokenizer.self,
        "CodeLlamaTokenizer": BPETokenizer.self,
        "CohereTokenizer": BPETokenizer.self,
        "DistilbertTokenizer": BertTokenizer.self,
        "DistilBertTokenizer": BertTokenizer.self,
        "FalconTokenizer": BPETokenizer.self,
        "GemmaTokenizer": BPETokenizer.self,
        "GPT2Tokenizer": BPETokenizer.self,
        "GPTNeoXTokenizer": BPETokenizer.self,
        "InternLM2Tokenizer": BPETokenizer.self,
        "LlamaTokenizer": BPETokenizer.self,
        "PreTrainedTokenizer": BPETokenizer.self,
        "Qwen2Tokenizer": BPETokenizer.self,
        "Qwen3Tokenizer": BPETokenizer.self,
        "RobertaTokenizer": BPETokenizer.self,
        "T5Tokenizer": T5Tokenizer.self,
        "TokenizersBackend": BPETokenizer.self,
        "WhisperTokenizer": BPETokenizer.self,
        "XLMRobertaTokenizer": UnigramTokenizer.self,
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    /// Async factory that creates the appropriate tokenizer model.
    ///
    /// For BPE and Unigram tokenizers with pre-extracted vocab/merges, uses parallel
    /// dictionary building for faster loading. For other tokenizer types, falls back
    /// to the protocol init.
    static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool = true
    ) async throws -> TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        let tokenizerClass = TokenizerModel.tokenizerClass(for: tokenizerName) ?? BPETokenizer.self
        if TokenizerModel.tokenizerClass(for: tokenizerName) == nil {
            if strict {
                throw TokenizerError.unsupportedTokenizer(tokenizerName)
            } else {
                print("Warning: Tokenizer model class \(tokenizerName) is not registered, falling back to a standard BPE implementation.")
            }
        }

        // Use async parallel building for BPE tokenizers
        if tokenizerClass is BPETokenizer.Type,
            case .bpe(let rawVocab) = tokenizerVocab,
            let rawMerges = tokenizerMerges?.rules
        {
            return await BPETokenizer.create(
                tokenizerConfig: tokenizerConfig,
                rawVocab: rawVocab,
                rawMerges: rawMerges,
                addedTokens: addedTokens
            )
        }

        // Use async parallel building for Unigram tokenizers
        if tokenizerClass is UnigramTokenizer.Type,
            case .unigram(let rawVocabArray) = tokenizerVocab,
            let rawVocab = rawVocabArray as? [[Any]]
        {
            return try await UnigramTokenizer.create(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                rawVocab: rawVocab,
                addedTokens: addedTokens
            )
        }

        // Fallback to protocol init (handles both fast and Config-based paths)
        return try tokenizerClass.init(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: addedTokens,
            vocab: tokenizerVocab,
            merges: tokenizerMerges
        )
    }
}

/// Arguments for specifying chat templates when applying chat formatting.
public enum ChatTemplateArgument {
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
    /// Tokenizes the input text into a sequence of tokens.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func tokenize(text: String) -> [String]

    /// Encodes text into token IDs with special tokens included by default.
    ///
    /// This is the main entry point for most tokenization tasks.
    ///
    /// - Parameter text: The input text to encode
    /// - Returns: An array of token IDs
    func encode(text: String) -> [Int]

    /// Encodes text into token IDs with optional special token handling.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens (e.g., BOS, EOS)
    /// - Returns: An array of token IDs
    func encode(text: String, addSpecialTokens: Bool) -> [Int]

    /// Function call syntax for encoding text.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens
    /// - Returns: An array of token IDs
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]

    /// Decodes token IDs back into text with special tokens included.
    ///
    /// - Parameter tokens: The token IDs to decode
    /// - Returns: The decoded text string
    func decode(tokens: [Int]) -> String

    /// Decodes token IDs back into text with optional special token handling.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to decode
    ///   - skipSpecialTokens: Whether to skip special tokens in the output
    /// - Returns: The decoded text string
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if not found
    func convertTokenToId(_ token: String) -> Int?

    /// Converts multiple token strings to their corresponding numeric IDs.
    ///
    /// - Parameter tokens: An array of token strings to convert
    /// - Returns: An array of numeric IDs, with nil values for unknown tokens
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String?

    /// Converts multiple numeric token IDs back to their string representations.
    ///
    /// - Parameter ids: An array of numeric token IDs to convert
    /// - Returns: An array of token strings, with nil values for invalid IDs
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    /// The beginning-of-sequence token string, if defined.
    var bosToken: String? { get }

    /// The numeric ID of the beginning-of-sequence token, if defined.
    var bosTokenId: Int? { get }

    /// The end-of-sequence token string, if defined.
    var eosToken: String? { get }

    /// The numeric ID of the end-of-sequence token, if defined.
    var eosTokenId: Int? { get }

    /// The unknown token string used for out-of-vocabulary words.
    var unknownToken: String? { get }

    /// The numeric ID of the unknown token.
    var unknownTokenId: Int? { get }

    /// Whether this tokenizer has a chat template configured.
    var hasChatTemplate: Bool { get }

    /// Applies a chat template to format messages for model input.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries representing the conversation
    ///   - chatTemplate: Optional chat template specification (literal or named)
    ///   - addGenerationPrompt: Whether to add a generation prompt for the assistant
    ///   - truncation: Whether to truncate if the result exceeds maximum length
    ///   - maxLength: Maximum allowed token length
    ///   - tools: Optional array of tool specifications for function calling
    ///   - additionalContext: Additional context variables for template rendering
    /// - Returns: Token IDs for the formatted conversation
    /// - Throws: `TokenizerError` if template application fails
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

extension Tokenizer {
    public var hasChatTemplate: Bool { false }

    /// Convenience with default parameter values for the protocol requirement.
    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
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

    /// Convenience that accepts a template string directly.
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
    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

/// A comprehensive tokenizer implementation supporting pre-trained models from Hugging Face.
///
/// This class provides a complete tokenizer implementation that can be initialized from
/// Hugging Face Hub configuration files and supports all standard tokenization operations
/// including chat template application, normalization, pre-tokenization, and post-processing.
public class PreTrainedTokenizer: @unchecked Sendable, Tokenizer {
    private static let specialTokenAttributes: Set<String> = [
        "bos_token", "eos_token", "unk_token", "sep_token",
        "pad_token", "cls_token", "mask_token", "additional_special_tokens",
    ]

    let model: TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }
    public var fuseUnknownTokens: Bool { model.fuseUnknownTokens }

    let addedTokens: Set<String>
    let specialTokens: [String: Int]
    let addedTokensRegex: NSRegularExpression?

    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?
    private let tokenizerConfig: Config

    private let cleanUpTokenizationSpaces: Bool

    /// Thread-safe cache for compiled Jinja templates keyed by their literal template string
    private var _templateCache = [String: Template]()
    private let _templateCacheLock = NSLock()

    /// Parses addedTokens from tokenizerData, returning (addedTokens dict, specialTokens dict).
    internal static func parseAddedTokens(from tokenizerData: Config) -> (tokens: [String: Int], special: [String: Int]) {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue }
            guard let content = addedToken.content.string() else { continue }
            addedTokens[content] = id
            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
            }
        }
        return (addedTokens, specialTokens)
    }

    /// Internal init accepting a pre-built model (used by async factory).
    /// Subclasses can use this to support async parallel building.
    internal init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        model: TokenizingModel
    ) throws {
        self.model = model

        let parsed = Self.parseAddedTokens(from: tokenizerData)
        self.specialTokens = parsed.special
        self.addedTokens = Set(parsed.tokens.keys)

        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData["addedTokens"].array(or: [])).compactMap { addedToken -> (String, Bool, Bool)? in
            guard let content = addedToken.content.string() else { return nil }
            let prefix = addedToken["lstrip"].boolean(or: false)
            let suffix = addedToken["rstrip"].boolean(or: false)
            return (content: content, prefix: prefix, suffix: suffix)
        }.sorted {
            $0.content.count > $1.content.count
        }

        let addedTokensRegexString = unwrappedAddedTokens.map {
            let token = NSRegularExpression.escapedPattern(for: $0.content)
            let prefix = $0.prefix ? #"\s*"# : ""
            let suffix = $0.suffix ? #"\s*"# : ""
            return "\(prefix)(\(token))\(suffix)"
        }.joined(separator: "|")
        addedTokensRegex = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])

        preTokenizer = try PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        normalizer = try NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        postProcessor = try PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        decoder = try DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: self.addedTokens)
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)
        self.tokenizerConfig = tokenizerConfig
    }

    /// Async factory that builds tokenizer model with parallel dictionary building.
    class func create(
        tokenizerConfig: Config,
        tokenizerData: Config,
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool = true
    ) async throws -> PreTrainedTokenizer {
        // Parse addedTokens (small data, used for model init)
        let parsed = parseAddedTokens(from: tokenizerData)

        // Build model with parallel dictionary building where applicable
        let model = try await TokenizerModel.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: parsed.tokens,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges,
            strict: strict
        )

        return try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            model: model
        )
    }

    private func compiledTemplate(for templateString: String) throws -> Template {
        // Fast path: check cache
        _templateCacheLock.lock()
        if let cached = _templateCache[templateString] {
            _templateCacheLock.unlock()
            return cached
        }
        _templateCacheLock.unlock()

        // Compile template outside of lock to avoid holding lock during expensive operation
        let compiled = try Template(templateString)

        // Insert into cache (double-checked in case another thread compiled the same template)
        _templateCacheLock.lock()
        defer { _templateCacheLock.unlock() }
        if let cached = _templateCache[templateString] {
            return cached
        }
        _templateCache[templateString] = compiled
        return compiled
    }

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor else { return tokens }
        return postProcessor(tokens: tokens, addSpecialTokens: addSpecialTokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

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

    func fuseUnknown(_ tokens: [String]) -> [String] {
        guard fuseUnknownTokens else { return tokens }
        let (fused, _) = tokens.reduce((fused: [String](), previousIsUnknown: false)) { result, token in
            var (fused, previousIsUnknown) = result
            let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
            if isUnknown {
                if !previousIsUnknown { fused.append(token) }
            } else {
                fused.append(token)
            }
            return (fused, isUnknown)
        }
        return fused
    }

    /// Tokenizes input text using the configured normalization and pre-tokenization steps.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of token strings
    public func tokenize(text: String) -> [String] {
        // Take care of special tokens first
        let sections: [String] =
            if let regex = addedTokensRegex {
                text.split(by: regex)
            } else {
                [text]
            }
        return sections.enumerated().map { section, x in
            if addedTokens.contains(x) { return [x] }
            return preTokenize(normalize(x), options: section == 0 ? [.firstSection] : []).flatMap { model($0) }
        }.flatMap { fuseUnknown($0) }
    }

    /// Encodes input text into token IDs with optional special token handling.
    ///
    /// This is the main entry point for text encoding operations.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens during post-processing
    /// - Returns: An array of token IDs
    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map { model.convertTokenToId($0)! }
    }

    /// Encodes input text into token IDs with special tokens included by default.
    ///
    /// - Parameter text: The input text to encode
    /// - Returns: An array of token IDs
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    /// Decodes token IDs back into human-readable text.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to decode
    ///   - skipSpecialTokens: Whether to exclude special tokens from the output text
    /// - Returns: The decoded text string
    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        // IDs to tokens
        let tokenStrings: [String]
        if skipSpecialTokens {
            let specialTokenIDs = Set(specialTokens.values)
            tokenStrings =
                tokens
                .filter { !specialTokenIDs.contains($0) }
                .compactMap { model.convertIdToToken($0) }
        } else {
            tokenStrings = tokens.compactMap { model.convertIdToToken($0) }
        }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if not found in the vocabulary
    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }

    /// Whether this tokenizer has a chat template configured.
    public var hasChatTemplate: Bool {
        !tokenizerConfig.chatTemplate.isNull()
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = true,
        truncation: Bool = false,
        maxLength: Int? = nil,
        // A list of tools (callable functions) that will be accessible to the model. If the template does not
        // support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        // giving the name, description and argument types for the tool. See the
        // [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
        // for more information.
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        var selectedChatTemplate: String?
        if let chatTemplate, case let .literal(template) = chatTemplate {
            // Use chat template from argument
            selectedChatTemplate = template
        } else if !tokenizerConfig.chatTemplate.isNull() {
            let valueFromConfig: Config = tokenizerConfig.chatTemplate
            if let arrayValue = valueFromConfig.array() {
                // If the config specifies a list of chat templates, convert them to a dictionary
                let templateDict = [String: String](
                    uniqueKeysWithValues: arrayValue.compactMap { item in
                        guard let name = item["name"].string(), let template = item["template"].string() else {
                            return nil
                        }
                        return (name, template)
                    })
                if let chatTemplate, case let .name(name) = chatTemplate {
                    // Select chat template from config by name
                    if let matchingDictEntry = templateDict[name] {
                        selectedChatTemplate = matchingDictEntry
                    } else {
                        throw TokenizerError.chatTemplate("No chat template named \"\(name)\" was found in the tokenizer config")
                    }
                } else if let tools, !tools.isEmpty, let toolUseTemplate = templateDict["tool_use"] {
                    // Use tool use chat template from config
                    selectedChatTemplate = toolUseTemplate
                } else if let defaultChatTemplate = templateDict["default"] {
                    // Use default chat template from config
                    selectedChatTemplate = defaultChatTemplate
                }
            } else if let stringValue = valueFromConfig.string() {
                // Use chat template from config
                selectedChatTemplate = stringValue
            }
        }

        guard let selectedChatTemplate else {
            throw TokenizerError.missingChatTemplate
        }

        let template = try compiledTemplate(for: selectedChatTemplate)
        var context: [String: Jinja.Value] = try [
            "messages": .array(messages.map { try Value(any: $0) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        if let tools {
            context["tools"] = try .array(tools.map { try Value(any: $0) })
        }
        if let additionalContext {
            // Additional keys and values to be added to the context provided to the prompt templating engine.
            // For example, the app could set "tools_in_user_message" to false for Llama 3.1 and 3.2 if a system message is provided.
            // The default value is true in the Llama 3.1 and 3.2 chat templates, but these models will perform better if the tools are included in a system message.
            for (key, value) in additionalContext {
                context[key] = try Value(any: value)
            }
        }

        for (key, value) in tokenizerConfig.dictionary(or: [:]) {
            if Self.specialTokenAttributes.contains(key.string), !value.isNull() {
                if let stringValue = value.string() {
                    context[key.string] = .string(stringValue)
                } else if let dictionary = value.dictionary() {
                    if let addedTokenString = Config(dictionary).tokenString {
                        context[key.string] = .string(addedTokenString)
                    }
                } else if let array: [String] = value.get() {
                    context[key.string] = .array(array.map { .string($0) })
                } else {
                    context[key.string] = try Value(any: value)
                }
            }
        }

        let rendered = try template.render(context)
        var encodedTokens = encode(text: rendered, addSpecialTokens: false)
        var maxLength = maxLength ?? encodedTokens.count
        maxLength = min(maxLength, tokenizerConfig.modelMaxLength.integer() ?? maxLength)
        if encodedTokens.count > maxLength {
            if truncation {
                encodedTokens = Array(encodedTokens.prefix(maxLength))
            }
        }

        return encodedTokens
    }
}

// MARK: - Building

/// A namespace for automatically creating appropriate tokenizer instances.
///
/// `AutoTokenizer` provides static methods for loading pre-trained tokenizers
/// from local directories. It automatically selects the appropriate tokenizer
/// class based on the configuration.
public enum AutoTokenizer {
    /// Registers a custom tokenizer class for a given tokenizer name.
    ///
    /// Use this to add support for tokenizer types not included in the library.
    /// Registration should be done at app launch, before loading any tokenizers.
    ///
    /// This mirrors Python transformers' `AutoTokenizer.register()`.
    ///
    /// - Parameters:
    ///   - tokenizerClass: The tokenizer class to register
    ///   - name: The tokenizer class name as it appears in `tokenizer_config.json`
    ///           (e.g., "MyCustomTokenizer"). The "Fast" suffix is automatically stripped
    ///           during lookup.
    ///
    /// Example:
    /// ```swift
    /// AutoTokenizer.register(MyTokenizer.self, for: "MyCustomTokenizer")
    /// ```
    public static func register(_ tokenizerClass: PreTrainedTokenizerModel.Type, for name: String) {
        TokenizerModel.registerTokenizer(tokenizerClass, for: name)
    }
}

enum PreTrainedTokenizerClasses {
    /// Class overrides for custom behavior.
    /// Not to be confused with the TokenizerModel classes defined in TokenizerModel.
    static let tokenizerClasses: [String: PreTrainedTokenizer.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizer.self
    ]
}

/// Maps `model_type` from config.json to the corresponding tokenizer class name.
/// Mirrors Python's `TOKENIZER_MAPPING_NAMES` from `transformers.models.auto.tokenization_auto`.
private let modelTypeToTokenizerClass: [String: String] = [
    "bert": "BertTokenizer",
    "code_llama": "CodeLlamaTokenizer",
    "codegen": "GPT2Tokenizer",
    "cohere": "CohereTokenizer",
    "distilbert": "BertTokenizer",
    "gemma": "GemmaTokenizer",
    "gemma2": "GemmaTokenizer",
    "gpt2": "GPT2Tokenizer",
    "llama": "LlamaTokenizer",
    "qwen2": "Qwen2Tokenizer",
    "roberta": "RobertaTokenizer",
    "t5": "T5Tokenizer",
    "whisper": "WhisperTokenizer",
    "xlm-roberta": "XLMRobertaTokenizer",
]

public extension AutoTokenizer {
    /// Determines the appropriate tokenizer class for the given configuration.
    ///
    /// - Parameter tokenizerConfig: The tokenizer configuration
    /// - Returns: The appropriate `PreTrainedTokenizer` subclass
    internal static func tokenizerClass(for tokenizerConfig: Config) -> PreTrainedTokenizer.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            return PreTrainedTokenizer.self
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        if let tokenizerClass = PreTrainedTokenizerClasses.tokenizerClasses[tokenizerName] {
            return tokenizerClass
        }

        return PreTrainedTokenizer.self
    }

    /// Loads a tokenizer from a local directory containing tokenizer configuration files.
    ///
    /// The directory must contain `tokenizer.json`. `tokenizer_config.json` is optional.
    /// If a `chat_template.jinja` or `chat_template.json` file is present, its contents
    /// will be merged into the tokenizer configuration.
    ///
    /// - Parameters:
    ///   - directory: Path to a local directory containing tokenizer files
    ///   - strict: Whether to enforce strict validation of tokenizer types
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if required files are missing or configuration is invalid
    static func from(directory: URL, strict: Bool = true) async throws -> Tokenizer {
        // Load and parse tokenizer data (required)
        let tokenizerDataURL = directory.appending(path: "tokenizer.json")
        let tokenizerDataRaw: NSDictionary
        do {
            let data = try Data(contentsOf: tokenizerDataURL)
            tokenizerDataRaw = try YYJSONParser.parseToNSDictionary(data)
        } catch {
            throw TokenizerError.missingConfig
        }

        // Extract vocab/merges from raw JSON before wrapping in Config.
        // This preserves NSString byte-level keys (important for tokenizers
        // whose vocab contains different Unicode normalization forms).
        // Only extract for BPE and Unigram model types; other types (e.g.
        // WordPiece) read vocab directly from Config.
        var tokenizerVocab: TokenizerVocab?
        var tokenizerMerges: TokenizerMerges?
        let parsed = tokenizerDataRaw.mutableCopy() as! NSMutableDictionary
        if let modelDict = parsed["model"] as? NSDictionary {
            let model = modelDict.mutableCopy() as! NSMutableDictionary
            let modelType = model["type"] as? String

            if modelType == "BPE", let vocab = model["vocab"] as? NSDictionary {
                tokenizerVocab = .bpe(vocab)
                if let merges = model["merges"] as? [Any] {
                    tokenizerMerges = TokenizerMerges(merges)
                }
                model.removeObject(forKey: "vocab")
                model.removeObject(forKey: "merges")
                parsed["model"] = model
            } else if modelType == "Unigram", let vocab = model["vocab"] as? NSArray {
                tokenizerVocab = .unigram(vocab)
                model.removeObject(forKey: "vocab")
                parsed["model"] = model
            }
        }
        let tokenizerData = Config(parsed as! [NSString: Any])

        // Load tokenizer config (optional — some models only have tokenizer.json)
        let tokenizerConfigURL = directory.appending(path: "tokenizer_config.json")
        var tokenizerConfig: Config
        if let data = try? Data(contentsOf: tokenizerConfigURL),
            let parsed = try? YYJSONParser.parseToConfig(data)
        {
            tokenizerConfig = parsed
        } else {
            tokenizerConfig = Config([:] as [NSString: Any])
        }

        // Resolve tokenizer_class if missing from tokenizer_config.json.
        // Mirrors Python's AutoTokenizer resolution: check config.json for
        // tokenizer_class directly, then fall back to model_type mapping.
        if tokenizerConfig.tokenizerClass.string() == nil {
            let modelConfigURL = directory.appending(path: "config.json")
            if let modelConfigData = try? Data(contentsOf: modelConfigURL),
                let modelConfig = try? YYJSONParser.parseToConfig(modelConfigData)
            {
                var resolvedClass: String?

                // Stage 1: Check config.json for tokenizer_class
                if let tokenizerClassName = modelConfig.tokenizerClass.string() {
                    resolvedClass = tokenizerClassName
                }
                // Stage 2: Use model_type to look up tokenizer class
                else if let modelType = modelConfig.modelType.string() {
                    resolvedClass = modelTypeToTokenizerClass[modelType]
                }

                if let resolvedClass {
                    var configDict = tokenizerConfig.dictionary() ?? [:]
                    configDict["tokenizer_class"] = Config(resolvedClass)
                    tokenizerConfig = Config(configDict)
                }
            }
        }

        // Load chat template if available (optional)
        // Prefer .jinja template over .json template
        let chatTemplateJinjaURL = directory.appending(path: "chat_template.jinja")
        let chatTemplateJsonURL = directory.appending(path: "chat_template.json")

        var chatTemplate: String? = nil
        if FileManager.default.fileExists(atPath: chatTemplateJinjaURL.path) {
            chatTemplate = try? String(contentsOf: chatTemplateJinjaURL, encoding: .utf8)
        } else if FileManager.default.fileExists(atPath: chatTemplateJsonURL.path),
            let chatTemplateData = try? Data(contentsOf: chatTemplateJsonURL),
            let chatTemplateConfig = try? YYJSONParser.parseToConfig(chatTemplateData)
        {
            chatTemplate = chatTemplateConfig.chatTemplate.string()
        }

        if let chatTemplate {
            if var configDict = tokenizerConfig.dictionary() {
                configDict["chat_template"] = Config(chatTemplate)
                tokenizerConfig = Config(configDict)
            } else {
                tokenizerConfig = Config(["chat_template": Config(chatTemplate)])
            }
        }

        return try await from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges,
            strict: strict
        )
    }

    /// Creates a tokenizer from configuration objects with optional pre-extracted vocab and merges.
    ///
    /// When vocab and merges are pre-extracted, BPE and Unigram tokenizers use parallel
    /// dictionary building for faster loading.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration (from tokenizer_config.json)
    ///   - tokenizerData: The tokenizer data (from tokenizer.json)
    ///   - tokenizerVocab: Pre-extracted vocabulary for fast initialization. Pass nil to
    ///     fall back to parsing from Config (slower for large vocabularies).
    ///   - tokenizerMerges: Pre-extracted merges for fast initialization. Pass nil to
    ///     fall back to parsing from Config (slower for large merge lists).
    ///   - strict: Whether to enforce strict validation
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if configuration is invalid
    static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool = true
    ) async throws -> Tokenizer {
        let selectedClass = tokenizerClass(for: tokenizerConfig)

        // Use async factory with dynamic dispatch (subclasses can override create)
        return try await selectedClass.create(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges,
            strict: strict
        )
    }
}

// MARK: - Tokenizer model classes

class T5Tokenizer: UnigramTokenizer, @unchecked Sendable {}

// MARK: - PreTrainedTokenizer classes

/// See https://github.com/xenova/transformers.js/blob/1a9964fb09b8f54fcbeac46dc6aae8d76795809d/src/tokenizers.js#L3203 for these exceptions
class LlamaPreTrainedTokenizer: PreTrainedTokenizer, @unchecked Sendable {
    private static let sentencePieceUnderline = "▁"

    /// Builds an updated post-processor config for Llama tokenizers that use bos/eos tokens.
    /// See https://github.com/huggingface/transformers/blob/bcb841f0073fcd7a4fb88ea8064313c17dcab04a/src/transformers/models/llama/tokenization_llama_fast.py#L181
    /// Returns updated config, or nil if the existing post-processor is already correct.
    private static func updatedPostProcessorConfig(tokenizerConfig: Config, processorConfig: Config?) throws -> Config? {
        // If it's already a Template processor (instead of a ByteLevel one), assume it's correct
        let postProcessor = try PostProcessorFactory.fromConfig(config: processorConfig)
        guard !(postProcessor is TemplateProcessing) else { return nil }

        let addBosToken = tokenizerConfig.addBosToken.boolean(or: false)
        let bosToken = tokenizerConfig.bosToken.tokenString
        if addBosToken, bosToken == nil {
            throw TokenizerError.mismatchedConfig("add_bos_token is True but bos_token is nil")
        }

        let addEosToken = tokenizerConfig.addEosToken.boolean(or: false)
        let eosToken = tokenizerConfig.eosToken.tokenString
        if addEosToken, eosToken == nil {
            throw TokenizerError.mismatchedConfig("add_eos_token is True but eos_token is nil")
        }

        var single: [[String: Any]] = []
        if addBosToken {
            single = single + [["SpecialToken": ["id": bosToken!, "type_id": 0]]]
        }
        single = single + [["Sequence": ["id": "A", "type_id": 0]]]
        if addEosToken {
            single = single + [["SpecialToken": ["id": eosToken!, "type_id": 0]]]
        }

        var pair: [[String: Any]] = single
        if addBosToken {
            pair = pair + [["SpecialToken": ["id": bosToken!, "type_id": 1]]]
        }
        pair = pair + [["Sequence": ["id": "B", "type_id": 1]]]
        if addEosToken {
            pair = pair + [["SpecialToken": ["id": eosToken!, "type_id": 1]]]
        }

        let postProcessorConfig = Config(["type": PostProcessorType.TemplateProcessing.rawValue, "single": single, "pair": pair])
        return postProcessorConfig
    }
    let isLegacy: Bool

    /// Internal init accepting a pre-built model (used by async factory).
    internal init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        model: TokenizingModel,
        isLegacy: Bool
    ) throws {
        self.isLegacy = isLegacy
        try super.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, model: model)
    }

    /// Async factory that builds model with parallel dictionary building.
    override class func create(
        tokenizerConfig: Config,
        tokenizerData: Config,
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool = true
    ) async throws -> PreTrainedTokenizer {
        let isLegacy = tokenizerConfig.legacy.boolean(or: true)
        let updatedData = try buildUpdatedConfig(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            isLegacy: isLegacy
        )

        // Parse addedTokens for model init
        let parsed = parseAddedTokens(from: updatedData)

        // Build model with parallel dictionary building where applicable
        let model = try await TokenizerModel.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: updatedData,
            addedTokens: parsed.tokens,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges,
            strict: strict
        )

        return try LlamaPreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: updatedData,
            model: model,
            isLegacy: isLegacy
        )
    }

    /// Builds the modified config for Llama tokenizers.
    private static func buildUpdatedConfig(
        tokenizerConfig: Config,
        tokenizerData: Config,
        isLegacy: Bool
    ) throws -> Config {
        var configDictionary = tokenizerData.dictionary(or: [:])
        if !isLegacy {
            _ = configDictionary.removeValue(forKey: "normalizer")
            configDictionary["pre_tokenizer"] = [
                "type": "Metaspace", "replacement": .init(sentencePieceUnderline), "add_prefix_space": true, "prepend_scheme": "first",
            ]
        }

        if let postProcessorConfig = try updatedPostProcessorConfig(tokenizerConfig: tokenizerConfig, processorConfig: tokenizerData["postProcessor"]) {
            configDictionary["post_processor"] = .init(postProcessorConfig.dictionary(or: [:]))
        }

        return Config(configDictionary)
    }

    /// If `isLegacy` is `False`, a prefix token is added unless the first token is special.
    /// https://github.com/huggingface/transformers/blob/e6dcf8abd6f65bb4b6dfc1831b20d9ba49ce00e2/src/transformers/models/t5/tokenization_t5.py#L374-L387
    override func tokenize(text: String) -> [String] {
        if isLegacy || text.isEmpty {
            return super.tokenize(text: text)
        }

        let tokens = super.tokenize(text: Self.sentencePieceUnderline + text.replacingOccurrences(of: Self.sentencePieceUnderline, with: " "))
        if tokens.first == Self.sentencePieceUnderline, let second = tokens.dropFirst().first, specialTokens[second] != nil {
            return Array(tokens[1...])
        }
        return tokens
    }
}
