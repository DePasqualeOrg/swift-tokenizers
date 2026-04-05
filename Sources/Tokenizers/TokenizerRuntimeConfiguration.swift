import Foundation

package struct TokenizerRuntimeConfiguration: Codable, Sendable {
    package struct NamedChatTemplate: Codable, Hashable, Sendable {
        package let name: String
        package let template: String
    }

    package enum ChatTemplateSource: Codable, Hashable, Sendable {
        case none
        case literal(String)
        case named([NamedChatTemplate])

        package init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .none
            } else if let template = try? container.decode(String.self) {
                self = .literal(template)
            } else if let templates = try? container.decode([NamedChatTemplate].self) {
                self = .named(templates)
            } else {
                throw DecodingError.typeMismatch(
                    ChatTemplateSource.self,
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "Expected a string, an array of named templates, or null"
                    )
                )
            }
        }

        package func encode(to encoder: any Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .none:
                try container.encodeNil()
            case .literal(let template):
                try container.encode(template)
            case .named(let templates):
                try container.encode(templates)
            }
        }
    }

    package let bosToken: String?
    package let eosToken: String?
    package let unknownToken: String?
    package let addBosToken: Bool
    package let addEosToken: Bool
    package let legacy: Bool?
    package let tokenizerClass: String?
    package let modelType: String?
    package let sepToken: String?
    package let padToken: String?
    package let clsToken: String?
    package let maskToken: String?
    package let additionalSpecialTokens: [String]
    package let cleanUpTokenizationSpaces: Bool
    package let modelMaxLength: Int?
    package let chatTemplate: ChatTemplateSource
    package let fuseUnknownTokens: Bool

    package init(tokenizerConfig: Config) {
        bosToken = tokenizerConfig.bosToken.tokenString
        eosToken = tokenizerConfig.eosToken.tokenString
        unknownToken = tokenizerConfig.unkToken.tokenString
        addBosToken = tokenizerConfig.addBosToken.boolean(or: false)
        addEosToken = tokenizerConfig.addEosToken.boolean(or: false)
        legacy = tokenizerConfig.legacy.boolean()
        tokenizerClass = tokenizerConfig.tokenizerClass.string()
        modelType = tokenizerConfig.modelType.string()
        sepToken = tokenizerConfig.sepToken.tokenString
        padToken = tokenizerConfig.padToken.tokenString
        clsToken = tokenizerConfig.clsToken.tokenString
        maskToken = tokenizerConfig.maskToken.tokenString
        additionalSpecialTokens = tokenizerConfig.additionalSpecialTokens.array(or: []).compactMap(\.tokenString)
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)
        modelMaxLength = tokenizerConfig.modelMaxLength.integer()
        chatTemplate = TokenizerRuntimeConfiguration.chatTemplateSource(from: tokenizerConfig.chatTemplate)
        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    package var hasChatTemplate: Bool {
        if case .none = chatTemplate {
            return false
        }
        return true
    }

    package func selectedChatTemplate(
        chatTemplate argument: ChatTemplateArgument?,
        tools: [ToolSpec]?
    ) throws -> String {
        if let argument, case let .literal(template) = argument {
            return template
        }

        switch chatTemplate {
        case .none:
            throw TokenizerError.missingChatTemplate
        case let .literal(template):
            return template
        case let .named(templates):
            let templateDictionary = Dictionary(uniqueKeysWithValues: templates.map { ($0.name, $0.template) })
            if let argument, case let .name(name) = argument {
                guard let matchingTemplate = templateDictionary[name] else {
                    throw TokenizerError.chatTemplate(
                        "No chat template named \"\(name)\" was found in the tokenizer config"
                    )
                }
                return matchingTemplate
            }
            if let tools, !tools.isEmpty, let toolUseTemplate = templateDictionary["tool_use"] {
                return toolUseTemplate
            }
            if let defaultTemplate = templateDictionary["default"] {
                return defaultTemplate
            }
            throw TokenizerError.missingChatTemplate
        }
    }

    package func chatTemplateContextObject(
        messages: [Message],
        addGenerationPrompt: Bool,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String: Any] {
        var context: [String: Any] = [
            "messages": try JSONBridge.foundationObject(from: messages),
            "add_generation_prompt": addGenerationPrompt,
        ]

        if let tools {
            context["tools"] = try JSONBridge.foundationObject(from: tools)
        }
        if let additionalContext {
            for (key, value) in additionalContext {
                context[key] = try JSONBridge.foundationObject(from: value)
            }
        }

        if let bosToken {
            context["bos_token"] = bosToken
        }
        if let eosToken {
            context["eos_token"] = eosToken
        }
        if let unknownToken {
            context["unk_token"] = unknownToken
        }
        if let sepToken {
            context["sep_token"] = sepToken
        }
        if let padToken {
            context["pad_token"] = padToken
        }
        if let clsToken {
            context["cls_token"] = clsToken
        }
        if let maskToken {
            context["mask_token"] = maskToken
        }
        if !additionalSpecialTokens.isEmpty {
            context["additional_special_tokens"] = additionalSpecialTokens
        }

        return context
    }

    package func effectiveChatTemplateMaxLength(_ maxLength: Int?) -> Int? {
        switch (maxLength, modelMaxLength) {
        case let (.some(requested), .some(modelMaxLength)):
            return min(requested, modelMaxLength)
        case let (.some(requested), nil):
            return requested
        case let (nil, .some(modelMaxLength)):
            return modelMaxLength
        case (nil, nil):
            return nil
        }
    }

    private static func chatTemplateSource(from config: Config) -> ChatTemplateSource {
        if let templates = config.array() {
            let namedTemplates = templates.compactMap { item -> NamedChatTemplate? in
                guard let name = item["name"].string(), let template = item["template"].string() else {
                    return nil
                }
                return NamedChatTemplate(name: name, template: template)
            }
            if !namedTemplates.isEmpty {
                return .named(namedTemplates)
            }
        }

        if let template = config.string() {
            return .literal(template)
        }

        return .none
    }
}
