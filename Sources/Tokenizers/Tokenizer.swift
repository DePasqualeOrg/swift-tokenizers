// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation

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

package enum JSONBridge {
    package static func foundationObject(from value: Any) throws -> Any {
        switch value {
        case let value as Config:
            return value.foundationObject()
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
                        throw TokenizerError.mismatchedConfig(
                            "Tokenizer JSON bridge only supports string-keyed dictionaries"
                        )
                    }
                    result[key] = try foundationObject(from: entryChildren[1].value)
                }
                return result
            default:
                throw TokenizerError.mismatchedConfig(
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

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if the token is not in the vocabulary
    func convertTokenToId(_ token: String) -> Int?

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is not valid
    func convertIdToToken(_ id: Int) -> String?

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
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        vocab: TokenizerVocab?,
        merges: TokenizerMerges?
    ) throws
}

package protocol TokenizerExecutionBackend: Sendable {
    var performsCleanup: Bool { get }
    func tokenize(text: String) -> [String]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func renderChatTemplate(template: String, contextObject: [String: Any]) throws -> String
    func applyChatTemplate(
        template: String,
        contextObject: [String: Any],
        truncation: Bool,
        maxLength: Int?
    ) throws -> [Int]
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
    func tokenize(text: String) -> [String]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?
    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }
    var hasChatTemplate: Bool { get }
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
    func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int]) -> String {
        decode(tokenIds: tokenIds, skipSpecialTokens: false)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }

    var bosTokenId: Int? { bosToken.flatMap { convertTokenToId($0) } }
    var eosTokenId: Int? { eosToken.flatMap { convertTokenToId($0) } }
    var unknownTokenId: Int? { unknownToken.flatMap { convertTokenToId($0) } }
}

/// A comprehensive tokenizer implementation supporting pre-trained models from Hugging Face.
///
/// The heavy tokenization logic lives in a backend-specific execution object. This shared wrapper
/// owns the public API surface and backend-agnostic chat-template policy.
public class PreTrainedTokenizer: @unchecked Sendable, Tokenizer {
    package let model: any TokenizingModel
    package let runtimeConfiguration: TokenizerRuntimeConfiguration

    private let backend: any TokenizerExecutionBackend

    public var bosToken: String? { runtimeConfiguration.bosToken ?? model.bosToken }
    public var eosToken: String? { runtimeConfiguration.eosToken ?? model.eosToken }
    public var unknownToken: String? { runtimeConfiguration.unknownToken ?? model.unknownToken }

    package init(
        model: some TokenizingModel,
        runtimeConfiguration: TokenizerRuntimeConfiguration,
        backend: some TokenizerExecutionBackend
    ) {
        self.model = model
        self.runtimeConfiguration = runtimeConfiguration
        self.backend = backend
    }

    package func selectedChatTemplate(
        chatTemplate: ChatTemplateArgument?,
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
        try backend.renderChatTemplate(template: template, contextObject: contextObject)
    }

    package func renderChatTemplateToString(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
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
        backend.tokenize(text: text)
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        backend.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let decoded = backend.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        if backend.performsCleanup {
            return decoded
        }
        return cleanUp(text: decoded)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }

    public var hasChatTemplate: Bool {
        runtimeConfiguration.hasChatTemplate
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
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
        return try backend.applyChatTemplate(
            template: selectedTemplate,
            contextObject: contextObject,
            truncation: truncation,
            maxLength: effectiveChatTemplateMaxLength(maxLength)
        )
    }
}

/// A namespace for automatically creating appropriate tokenizer instances.
public enum AutoTokenizer {}
