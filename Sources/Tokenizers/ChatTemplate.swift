// Copyright © Anthony DePasquale

import Foundation
import TokenizersFFI

/// Standalone chat-template rendering against the same Rust minijinja engine
/// `Tokenizer.applyChatTemplate` uses, without a `Tokenizer` instance.
/// Use when the template + special-token strings come from another source
/// (e.g. GGUF metadata).
public enum ChatTemplate {
    /// Render `template` against a pre-built `context` dict (the minijinja
    /// variables). Use the messages-based overload if you have raw
    /// messages/tools/specialTokens.
    public static func render(
        template: String,
        context: [String: any Sendable]
    ) throws(TokenizerError) -> String {
        try renderJSON(template: template, contextValue: context)
    }

    /// Compose a context object from `messages` + `tools` + `additionalContext`
    /// + `specialTokens` (matching `PreTrainedTokenizer.applyChatTemplate`)
    /// and render.
    public static func render(
        template: String,
        messages: [Message],
        addGenerationPrompt: Bool = true,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        specialTokens: SpecialTokens = .init()
    ) throws(TokenizerError) -> String {
        let ctx = try context(
            messages: messages,
            addGenerationPrompt: addGenerationPrompt,
            tools: tools,
            additionalContext: additionalContext,
            specialTokens: specialTokens
        )
        return try renderJSON(template: template, contextValue: ctx)
    }

    /// Build the context dict the messages-based `render` overload passes
    /// to minijinja. Exposed for inspection or mutation before rendering;
    /// pass the result back through `render(template:context:)`.
    ///
    /// Returns `[String: Any]` because the dict mixes caller-supplied
    /// Sendable values with `JSONBridge`-produced Foundation containers
    /// (NSArray / NSDictionary) that aren't `Sendable` themselves.
    public static func context(
        messages: [Message],
        addGenerationPrompt: Bool = true,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        specialTokens: SpecialTokens = .init()
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

        if let bos = specialTokens.bos {
            context["bos_token"] = bos
        }
        if let eos = specialTokens.eos {
            context["eos_token"] = eos
        }
        if let unk = specialTokens.unk {
            context["unk_token"] = unk
        }
        if let sep = specialTokens.sep {
            context["sep_token"] = sep
        }
        if let pad = specialTokens.pad {
            context["pad_token"] = pad
        }
        if let cls = specialTokens.cls {
            context["cls_token"] = cls
        }
        if let mask = specialTokens.mask {
            context["mask_token"] = mask
        }
        if !specialTokens.additional.isEmpty {
            context["additional_special_tokens"] = specialTokens.additional
        }
        return context
    }

    /// Shared back-end accepting either `[String: any Sendable]` (public
    /// `render(template:context:)`) or `[String: Any]` (internal output
    /// of `context(messages:)`). Both upcast to `Any` for `JSONBridge`.
    private static func renderJSON(
        template: String,
        contextValue: Any
    ) throws(TokenizerError) -> String {
        let contextJSON = try JSONBridge.jsonString(from: contextValue)
        do {
            return try TokenizersFFI.renderTemplate(template: template, contextJson: contextJSON)
        } catch let error as TokenizersFFI.TokenizerError {
            throw error.bridged
        } catch let error as TokenizerError {
            throw error
        } catch {
            throw TokenizerError.internalError(error.localizedDescription)
        }
    }

    /// Special-token strings the template may reference. Populate only what
    /// the template uses (typically `bos` and `eos`).
    public struct SpecialTokens: Sendable {
        public var bos: String?
        public var eos: String?
        public var unk: String?
        public var sep: String?
        public var pad: String?
        public var cls: String?
        public var mask: String?
        public var additional: [String]

        public init(
            bos: String? = nil,
            eos: String? = nil,
            unk: String? = nil,
            sep: String? = nil,
            pad: String? = nil,
            cls: String? = nil,
            mask: String? = nil,
            additional: [String] = []
        ) {
            self.bos = bos
            self.eos = eos
            self.unk = unk
            self.sep = sep
            self.pad = pad
            self.cls = cls
            self.mask = mask
            self.additional = additional
        }
    }
}
