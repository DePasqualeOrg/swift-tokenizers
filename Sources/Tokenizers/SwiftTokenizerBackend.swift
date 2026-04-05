#if TOKENIZERS_SWIFT_BACKEND
// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation
import Jinja
import TokenizersCore

private struct TokenizerTypeBox<Value>: @unchecked Sendable {
    let type: Value
}

private final class RegisteredTokenizerStore: @unchecked Sendable {
    private var registeredTokenizers: [String: TokenizerTypeBox<PreTrainedTokenizerModel.Type>] = [:]
    private let lock = NSLock()

    func register(_ tokenizerClass: PreTrainedTokenizerModel.Type, for name: String) {
        lock.lock()
        defer { lock.unlock() }
        registeredTokenizers[name] = TokenizerTypeBox(type: tokenizerClass)
    }

    func tokenizerClass(
        for name: String,
        fallback: [String: TokenizerTypeBox<PreTrainedTokenizerModel.Type>]
    ) -> PreTrainedTokenizerModel.Type? {
        lock.lock()
        defer { lock.unlock() }
        return registeredTokenizers[name]?.type ?? fallback[name]?.type
    }
}

package enum TokenizerModel {
    private static let registeredTokenizers = RegisteredTokenizerStore()

    package static func registerTokenizer(_ tokenizerClass: PreTrainedTokenizerModel.Type, for name: String) {
        registeredTokenizers.register(tokenizerClass, for: name)
    }

    package static func tokenizerClass(for name: String) -> PreTrainedTokenizerModel.Type? {
        registeredTokenizers.tokenizerClass(for: name, fallback: knownTokenizers)
    }

    private static let knownTokenizers: [String: TokenizerTypeBox<PreTrainedTokenizerModel.Type>] = [
        "BertTokenizer": TokenizerTypeBox(type: BertTokenizer.self),
        "CodeGenTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "CodeLlamaTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "CohereTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "DistilbertTokenizer": TokenizerTypeBox(type: BertTokenizer.self),
        "DistilBertTokenizer": TokenizerTypeBox(type: BertTokenizer.self),
        "FalconTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "GemmaTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "GPT2Tokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "GPTNeoXTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "InternLM2Tokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "LlamaTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "PreTrainedTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "Qwen2Tokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "Qwen3Tokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "RobertaTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "T5Tokenizer": TokenizerTypeBox(type: T5Tokenizer.self),
        "TokenizersBackend": TokenizerTypeBox(type: BPETokenizer.self),
        "WhisperTokenizer": TokenizerTypeBox(type: BPETokenizer.self),
        "XLMRobertaTokenizer": TokenizerTypeBox(type: UnigramTokenizer.self),
        "Xlm-RobertaTokenizer": TokenizerTypeBox(type: UnigramTokenizer.self),
    ]

    package static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    package static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool = true
    ) async throws -> any TokenizingModel {
        let tokenizerName = try TokenizerCompatibility.validateResolvedTokenizerName(
            tokenizerClass: tokenizerConfig.tokenizerClass.string(),
            modelType: tokenizerConfig.modelType.string(),
            strict: strict
        ) { tokenizerName in
            TokenizerModel.tokenizerClass(for: tokenizerName) != nil
        }
        let tokenizerClass = TokenizerModel.tokenizerClass(for: tokenizerName) ?? BPETokenizer.self
        if TokenizerModel.tokenizerClass(for: tokenizerName) == nil {
            if strict {
                throw TokenizerError.unsupportedTokenizer(tokenizerName)
            } else {
                // This fallback keeps the Swift path usable for transformers-side wrapper names that
                // we have not registered locally. The underlying tokenizer engine may still just be
                // BPE/Unigram/etc. even when Python exposes a model-specific class like
                // `NllbTokenizer`.
                print(
                    "Warning: Tokenizer model class \(tokenizerName) is not registered, falling back to a standard BPE implementation."
                )
            }
        }

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

        return try tokenizerClass.init(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: addedTokens,
            vocab: tokenizerVocab,
            merges: tokenizerMerges
        )
    }
}

package class SwiftTokenizerBackend: TokenizerExecutionBackend, @unchecked Sendable {
    package let model: any TokenizingModel
    package let specialTokens: [String: Int]
    package let performsCleanup = true

    private let addedTokens: Set<String>
    private let addedTokensRegex: NSRegularExpression?
    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?
    private let cleanUpTokenizationSpaces: Bool
    private let fuseUnknownTokens: Bool

    private var templateCache = [String: Template]()
    private let templateCacheLock = NSLock()

    package static func parseAddedTokens(from tokenizerData: Config) -> (tokens: [String: Int], special: [String: Int]) {
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

    package init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        model: any TokenizingModel,
        runtimeConfiguration: TokenizerRuntimeConfiguration
    ) throws {
        self.model = model
        let parsed = Self.parseAddedTokens(from: tokenizerData)
        self.specialTokens = parsed.special
        self.addedTokens = Set(parsed.tokens.keys)
        self.fuseUnknownTokens = model.fuseUnknownTokens
        self.cleanUpTokenizationSpaces = runtimeConfiguration.cleanUpTokenizationSpaces

        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = tokenizerData["addedTokens"]
            .array(or: [])
            .compactMap { addedToken -> (String, Bool, Bool)? in
                guard let content = addedToken.content.string() else { return nil }
                let prefix = addedToken["lstrip"].boolean(or: false)
                let suffix = addedToken["rstrip"].boolean(or: false)
                return (content, prefix, suffix)
            }
            .sorted { $0.content.count > $1.content.count }

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
        _ = tokenizerConfig
    }

    private func compiledTemplate(for templateString: String) throws -> Template {
        templateCacheLock.lock()
        if let cached = templateCache[templateString] {
            templateCacheLock.unlock()
            return cached
        }
        templateCacheLock.unlock()

        let compiled = try Template(templateString, with: .init(lstripBlocks: true, trimBlocks: true))

        templateCacheLock.lock()
        defer { templateCacheLock.unlock() }
        if let cached = templateCache[templateString] {
            return cached
        }
        templateCache[templateString] = compiled
        return compiled
    }

    private func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    private func normalize(_ text: String) -> String {
        guard let normalizer else { return text }
        return normalizer(text: text)
    }

    private func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor else { return tokens }
        return postProcessor(tokens: tokens, addSpecialTokens: addSpecialTokens)
    }

    private func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    private func cleanUp(text: String) -> String {
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

    private func fuseUnknown(_ tokens: [String]) -> [String] {
        guard fuseUnknownTokens else { return tokens }
        let (fused, _) = tokens.reduce((fused: [String](), previousIsUnknown: false)) { result, token in
            var (fused, previousIsUnknown) = result
            let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
            if isUnknown {
                if !previousIsUnknown {
                    fused.append(token)
                }
            } else {
                fused.append(token)
            }
            return (fused, isUnknown)
        }
        return fused
    }

    package func tokenize(text: String) -> [String] {
        let sections: [String] =
            if let regex = addedTokensRegex {
                text.split(by: regex)
            } else {
                [text]
            }

        return sections.enumerated().map { section, text in
            if addedTokens.contains(text) {
                return [text]
            }
            return preTokenize(normalize(text), options: section == 0 ? [.firstSection] : [])
                .flatMap { model($0) }
        }
        .flatMap { fuseUnknown($0) }
    }

    package func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map {
            model.convertTokenToId($0)!
        }
    }

    package func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let tokenStrings: [String]
        if skipSpecialTokens {
            let specialTokenIDs = Set(specialTokens.values)
            tokenStrings =
                tokenIds
                .filter { !specialTokenIDs.contains($0) }
                .compactMap { model.convertIdToToken($0) }
        } else {
            tokenStrings = tokenIds.compactMap { model.convertIdToToken($0) }
        }

        let decoded = decodeTokens(tokenStrings)
        return cleanUp(text: decoded.joined(separator: ""))
    }

    package func renderChatTemplate(template: String, contextObject: [String: Any]) throws -> String {
        let compiledTemplate = try compiledTemplate(for: template)
        let context = try Dictionary(
            uniqueKeysWithValues: contextObject.map { key, value in
                (key, try Value(any: value))
            })
        return try compiledTemplate.render(context)
    }

    package func applyChatTemplate(
        template: String,
        contextObject: [String: Any],
        truncation: Bool,
        maxLength: Int?
    ) throws -> [Int] {
        let rendered = try renderChatTemplate(template: template, contextObject: contextObject)
        var encoded = encode(text: rendered, addSpecialTokens: false)
        if let maxLength, encoded.count > maxLength, truncation {
            encoded = Array(encoded.prefix(maxLength))
        }
        return encoded
    }
}

private final class LlamaSwiftTokenizerBackend: SwiftTokenizerBackend, @unchecked Sendable {
    private static let sentencePieceUnderline = "▁"

    private let isLegacy: Bool

    init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        model: any TokenizingModel,
        runtimeConfiguration: TokenizerRuntimeConfiguration,
        isLegacy: Bool
    ) throws {
        self.isLegacy = isLegacy
        try super.init(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            model: model,
            runtimeConfiguration: runtimeConfiguration
        )
    }

    override package func tokenize(text: String) -> [String] {
        if isLegacy || text.isEmpty {
            return super.tokenize(text: text)
        }

        let tokens = super.tokenize(
            text: Self.sentencePieceUnderline + text.replacingOccurrences(of: Self.sentencePieceUnderline, with: " ")
        )
        if tokens.first == Self.sentencePieceUnderline,
            let second = tokens.dropFirst().first,
            specialTokens[second] != nil
        {
            return Array(tokens[1...])
        }
        return tokens
    }
}

private enum LlamaTokenizerConfig {
    private static let sentencePieceUnderline = "▁"

    private static func updatedPostProcessorConfig(
        tokenizerConfig: Config,
        processorConfig: Config?
    ) throws -> Config? {
        // Keep the Swift Llama compatibility path aligned with Python transformers:
        // `TokenizersBackend.update_post_processor` in `tokenization_utils_tokenizers.py`
        // rebuilds `TemplateProcessing` from `add_bos_token` / `add_eos_token` for
        // Llama-family fast tokenizers instead of trusting the raw tokenizer JSON.
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
            single += [["SpecialToken": ["id": bosToken!, "type_id": 0]]]
        }
        single += [["Sequence": ["id": "A", "type_id": 0]]]
        if addEosToken {
            single += [["SpecialToken": ["id": eosToken!, "type_id": 0]]]
        }

        var pair = single
        if addBosToken {
            pair += [["SpecialToken": ["id": bosToken!, "type_id": 1]]]
        }
        pair += [["Sequence": ["id": "B", "type_id": 1]]]
        if addEosToken {
            pair += [["SpecialToken": ["id": eosToken!, "type_id": 1]]]
        }

        return Config([
            "type": PostProcessorType.TemplateProcessing.rawValue,
            "single": single,
            "pair": pair,
        ])
    }

    static func buildUpdatedConfig(
        tokenizerConfig: Config,
        tokenizerData: Config,
        isLegacy: Bool
    ) throws -> Config {
        var configDictionary = tokenizerData.dictionary(or: [:])
        if !isLegacy {
            // Python `models/llama/tokenization_llama.py` swaps Llama fast tokenizers to
            // a `Metaspace` pre-tokenizer that uses `_get_prepend_scheme(...)` for the
            // non-legacy path. Keep the Swift backend aligned with that behavior.
            _ = configDictionary.removeValue(forKey: "normalizer")
            configDictionary["pre_tokenizer"] = [
                "type": "Metaspace",
                "replacement": .init(sentencePieceUnderline),
                "add_prefix_space": true,
                "prepend_scheme": "first",
            ]
        }

        if let postProcessorConfig = try updatedPostProcessorConfig(
            tokenizerConfig: tokenizerConfig,
            processorConfig: tokenizerData["postProcessor"]
        ) {
            configDictionary["post_processor"] = .init(postProcessorConfig.dictionary(or: [:]))
        }

        return Config(configDictionary)
    }
}

package enum AutoTokenizerDirectorySidecars {
    static func load(from directory: URL) throws -> Config {
        var tokenizerConfig = loadOptionalConfig(from: directory.appending(path: "tokenizer_config.json"))

        if tokenizerConfig.tokenizerClass.string() == nil {
            let modelConfig = loadOptionalConfig(from: directory.appending(path: "config.json"))
            let resolvedClass = TokenizerCompatibility.resolvedTokenizerClass(
                tokenizerClass: modelConfig.tokenizerClass.string(),
                modelType: modelConfig.modelType.string()
            )
            if let resolvedClass {
                tokenizerConfig = merging(tokenizerConfig, key: "tokenizer_class", value: Config(resolvedClass))
            }
        }

        if let chatTemplate = loadChatTemplateOverride(from: directory) {
            tokenizerConfig = merging(tokenizerConfig, key: "chat_template", value: chatTemplate)
        }

        return tokenizerConfig
    }

    private static func loadOptionalConfig(from url: URL) -> Config {
        guard let data = try? Data(contentsOf: url), let parsed = try? YYJSONParser.parseToConfig(data) else {
            return Config([:] as [NSString: Any])
        }
        return parsed
    }

    private static func loadChatTemplateOverride(from directory: URL) -> Config? {
        let chatTemplateJinjaURL = directory.appending(path: "chat_template.jinja")
        if FileManager.default.fileExists(atPath: chatTemplateJinjaURL.path),
            let chatTemplate = try? String(contentsOf: chatTemplateJinjaURL, encoding: .utf8)
        {
            return Config(chatTemplate)
        }

        let chatTemplateJsonURL = directory.appending(path: "chat_template.json")
        guard FileManager.default.fileExists(atPath: chatTemplateJsonURL.path),
            let chatTemplateData = try? Data(contentsOf: chatTemplateJsonURL),
            let chatTemplateConfig = try? YYJSONParser.parseToConfig(chatTemplateData)
        else {
            return nil
        }

        let chatTemplate = chatTemplateConfig[Config.Key("chat_template")]
        return chatTemplate.isNull() ? nil : chatTemplate
    }

    private static func merging(_ config: Config, key: String, value: Config) -> Config {
        var dictionary = config.dictionary() ?? [:]
        dictionary[Config.Key(key)] = value
        return Config(dictionary)
    }
}

package struct SwiftTokenizerDirectoryArtifacts {
    package let tokenizerData: Config
    package let tokenizerVocab: TokenizerVocab?
    package let tokenizerMerges: TokenizerMerges?
}

package enum SwiftAutoTokenizerDirectoryLoader {
    package static func loadRuntimeConfiguration(from directory: URL) throws -> TokenizerRuntimeConfiguration {
        let tokenizerConfig = try AutoTokenizerDirectorySidecars.load(from: directory)
        return TokenizerRuntimeConfiguration(tokenizerConfig: tokenizerConfig)
    }

    package static func loadTokenizerConfig(from directory: URL) throws -> Config {
        try AutoTokenizerDirectorySidecars.load(from: directory)
    }

    package static func loadTokenizerArtifacts(from directory: URL) throws -> SwiftTokenizerDirectoryArtifacts {
        let tokenizerDataURL = directory.appending(path: "tokenizer.json")
        let tokenizerDataRaw: NSDictionary
        do {
            let data = try Data(contentsOf: tokenizerDataURL)
            tokenizerDataRaw = try YYJSONParser.parseToNSDictionary(data)
        } catch {
            throw TokenizerError.missingConfig
        }

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

        return SwiftTokenizerDirectoryArtifacts(
            tokenizerData: Config(parsed as! [NSString: Any]),
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges
        )
    }

    package static func loadTokenizerCore(
        from directory: URL,
        tokenizerConfig: Config,
        strict: Bool
    ) async throws -> any Tokenizer {
        let artifacts = try loadTokenizerArtifacts(from: directory)
        return try await SwiftAutoTokenizerFactory.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: artifacts.tokenizerData,
            tokenizerVocab: artifacts.tokenizerVocab,
            tokenizerMerges: artifacts.tokenizerMerges,
            strict: strict
        )
    }

    package static func load(from directory: URL, strict: Bool) async throws -> any Tokenizer {
        let tokenizerConfig = try loadTokenizerConfig(from: directory)
        return try await loadTokenizerCore(from: directory, tokenizerConfig: tokenizerConfig, strict: strict)
    }
}

private enum SwiftAutoTokenizerFactory {
    static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool = true
    ) async throws -> any Tokenizer {
        if tokenizerName(from: tokenizerConfig) == "LlamaTokenizer" {
            return try await makeLlamaTokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                tokenizerVocab: tokenizerVocab,
                tokenizerMerges: tokenizerMerges,
                strict: strict
            )
        }

        let parsed = SwiftTokenizerBackend.parseAddedTokens(from: tokenizerData)
        let model = try await TokenizerModel.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: parsed.tokens,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges,
            strict: strict
        )
        let runtimeConfiguration = TokenizerRuntimeConfiguration(tokenizerConfig: tokenizerConfig)
        let backend = try SwiftTokenizerBackend(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            model: model,
            runtimeConfiguration: runtimeConfiguration
        )
        return PreTrainedTokenizer(
            model: model,
            runtimeConfiguration: runtimeConfiguration,
            backend: backend
        )
    }

    private static func makeLlamaTokenizer(
        tokenizerConfig: Config,
        tokenizerData: Config,
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?,
        strict: Bool
    ) async throws -> any Tokenizer {
        let isLegacy = tokenizerConfig.legacy.boolean(or: true)
        let updatedData = try LlamaTokenizerConfig.buildUpdatedConfig(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            isLegacy: isLegacy
        )
        let parsed = SwiftTokenizerBackend.parseAddedTokens(from: updatedData)
        let model = try await TokenizerModel.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: updatedData,
            addedTokens: parsed.tokens,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges,
            strict: strict
        )
        let runtimeConfiguration = TokenizerRuntimeConfiguration(tokenizerConfig: tokenizerConfig)
        let backend = try LlamaSwiftTokenizerBackend(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: updatedData,
            model: model,
            runtimeConfiguration: runtimeConfiguration,
            isLegacy: isLegacy
        )
        return PreTrainedTokenizer(
            model: model,
            runtimeConfiguration: runtimeConfiguration,
            backend: backend
        )
    }

    private static func tokenizerName(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.tokenizerClass.string()?.replacingOccurrences(of: "Fast", with: "")
    }
}

public extension AutoTokenizer {
    static func register(_ tokenizerClass: PreTrainedTokenizerModel.Type, for name: String) {
        TokenizerModel.registerTokenizer(tokenizerClass, for: name)
    }
}

private final class T5Tokenizer: UnigramTokenizer, @unchecked Sendable {}
#endif
