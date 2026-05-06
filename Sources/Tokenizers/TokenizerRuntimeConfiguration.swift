import Foundation

package struct TokenizerRuntimeConfiguration: Codable, Sendable {
    package struct NamedChatTemplate: Codable, Hashable, Sendable {
        package let name: String
        package let template: String

        package init(name: String, template: String) {
            self.name = name
            self.template = template
        }
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

    // `tokenizer.json`'s `post_processor` is the sole source of truth for which
    // special tokens get prepended or appended. Python `transformers` v5 takes
    // the same stance: `PreTrainedTokenizerBase.from_pretrained` pops
    // `add_bos_token` / `add_eos_token` from `init_kwargs` whenever a
    // `tokenizer.json` is present, so they never reach
    // `PreTrainedTokenizerFast.update_post_processor`. A tokenizer that needs a
    // leading bos must carry it in its `tokenizer.json` post-processor.
    package let bosToken: String?
    package let eosToken: String?
    package let unknownToken: String?
    package let sepToken: String?
    package let padToken: String?
    package let clsToken: String?
    package let maskToken: String?
    package let additionalSpecialTokens: [String]
    package let cleanUpTokenizationSpaces: Bool
    package let modelMaxLength: Int?
    package let chatTemplate: ChatTemplateSource

    package init(
        bosToken: String?,
        eosToken: String?,
        unknownToken: String?,
        sepToken: String?,
        padToken: String?,
        clsToken: String?,
        maskToken: String?,
        additionalSpecialTokens: [String],
        cleanUpTokenizationSpaces: Bool,
        modelMaxLength: Int?,
        chatTemplate: ChatTemplateSource
    ) {
        self.bosToken = bosToken
        self.eosToken = eosToken
        self.unknownToken = unknownToken
        self.sepToken = sepToken
        self.padToken = padToken
        self.clsToken = clsToken
        self.maskToken = maskToken
        self.additionalSpecialTokens = additionalSpecialTokens
        self.cleanUpTokenizationSpaces = cleanUpTokenizationSpaces
        self.modelMaxLength = modelMaxLength
        self.chatTemplate = chatTemplate
    }

    package var hasChatTemplate: Bool {
        if case .none = chatTemplate {
            return false
        }
        return true
    }

    package func selectedChatTemplate(
        chatTemplate argument: ChatTemplateOverride?,
        tools: [ToolSpec]?
    ) throws(TokenizerError) -> String {
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
    ) throws(TokenizerError) -> [String: Any] {
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
}
