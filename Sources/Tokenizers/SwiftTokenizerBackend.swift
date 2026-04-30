#if Swift
// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation
import Jinja
import TokenizersCore

package enum TokenizerModel {
    package static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?
    ) async throws -> any TokenizingModel {
        // Select the model implementation from `tokenizer.json`'s `model.type`,
        // matching upstream's tagged enum in `tokenizers/src/models/mod.rs`
        // (`ModelWrapper`). When `type` is absent, fall back to shape-based
        // detection — the "legacy untagged" path in `ModelWrapper`'s custom
        // `Deserialize` impl. Widely-used models like `bert-base-uncased` ship
        // tokenizer.json without `model.type`, so strict rejection isn't viable;
        // empty / unknown values still throw.
        let rawModelType =
            tokenizerData["model"]["type"].string()
            ?? inferLegacyModelType(
                from: tokenizerData,
                extractedMerges: tokenizerMerges
            )

        switch rawModelType {
        case "BPE":
            if case .bpe(let rawVocab) = tokenizerVocab,
                let rawMerges = tokenizerMerges?.rules
            {
                return await BPETokenizer.create(
                    tokenizerConfig: tokenizerConfig,
                    rawVocab: rawVocab,
                    rawMerges: rawMerges,
                    addedTokens: addedTokens,
                    unknownToken: tokenizerData.model["unkToken"].string(),
                    fuseUnknownTokens: tokenizerData.model["fuseUnk"].boolean(or: false),
                    byteFallback: tokenizerData.model["byteFallback"].boolean(or: false)
                )
            }
            return try BPETokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                addedTokens: addedTokens,
                vocab: tokenizerVocab,
                merges: tokenizerMerges
            )

        case "WordPiece":
            return try WordPieceTokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                addedTokens: addedTokens,
                vocab: tokenizerVocab,
                merges: tokenizerMerges
            )

        case "Unigram":
            if case .unigram(let rawVocabArray) = tokenizerVocab,
                let rawVocab = rawVocabArray as? [[Any]]
            {
                return try await UnigramTokenizer.create(
                    tokenizerConfig: tokenizerConfig,
                    tokenizerData: tokenizerData,
                    rawVocab: rawVocab,
                    addedTokens: addedTokens
                )
            }
            return try UnigramTokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                addedTokens: addedTokens,
                vocab: tokenizerVocab,
                merges: tokenizerMerges
            )

        case "WordLevel":
            return try WordLevelTokenizer(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                addedTokens: addedTokens,
                vocab: tokenizerVocab,
                merges: tokenizerMerges
            )

        default:
            throw TokenizerError.unsupportedModelType(rawModelType ?? "")
        }
    }
}

/// Shape-based fallback matching the order upstream's `ModelWrapper` uses when the
/// `type` tag is missing (see the custom `Deserialize` impl in
/// `tokenizers/src/models/mod.rs`): BPE first, then WordPiece, WordLevel, Unigram.
/// Returns `nil` if the shape matches none.
private func inferLegacyModelType(
    from tokenizerData: Config,
    extractedMerges: TokenizerMerges?
) -> String? {
    let model = tokenizerData["model"]
    // BPE: presence of `merges`. Upstream's `ModelWrapper::Deserialize` untagged
    // fallback attempts BPE first, so we match that priority.
    if extractedMerges != nil || model["merges"].array() != nil {
        return "BPE"
    }
    // WordPiece is distinguished from WordLevel by fields that only WordPiece has.
    if model["continuing_subword_prefix"].string() != nil
        || model["max_input_chars_per_word"].integer() != nil
    {
        return "WordPiece"
    }
    // Vocab shape decides the remaining two: array ⇒ Unigram, dict ⇒ WordLevel.
    if model["vocab"].array() != nil {
        return "Unigram"
    }
    if model["vocab"].dictionary() != nil {
        return "WordLevel"
    }
    return nil
}

package class SwiftTokenizerBackend: TokenizerExecutionBackend, @unchecked Sendable {
    /// Behavior flags for an entry in `tokenizer.json`'s `added_tokens` array. Mirrors
    /// `AddedToken` in `tokenizers/src/tokenizer/added_vocabulary.rs`.
    private struct AddedTokenInfo: Sendable {
        let content: String
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
        let singleWord: Bool
        let normalized: Bool
    }

    /// Alias for `PostProcessorToken` within the backend. Upstream Rust tracks
    /// added-token matches as `Token { id, value, offsets }` so the widened
    /// match slice (e.g. `" <mask>"` when `lstrip`/`rstrip` are set) flows
    /// through post-processing as the token string while still encoding to the
    /// added-token's ID. `PostProcessorToken` carries that pair positionally,
    /// so the override survives the post-processor even when another position
    /// in the same sequence produces an identical value via model tokenization.
    private typealias CoreToken = PostProcessorToken

    package let model: any TokenizingModel
    package let specialTokens: [String: Int]
    package let performsCleanup = true

    private let addedTokens: Set<String>
    /// Phase-1 lookup: raw matched text → original added token info. Keys are the raw
    /// token contents since phase-1 runs before normalization.
    private let nonNormalizedAddedInfo: [String: AddedTokenInfo]
    /// Phase-2 lookup: normalized matched text → original added token info. Keys are
    /// the *normalized* form of each token's content, matching upstream's
    /// `AddedVocabulary::refresh_added_tokens`, which populates `normalized_cache`
    /// by running the tokenizer's normalizer over each token's content. The
    /// `AddedTokenInfo.content` value in this dict is still the original
    /// (pre-normalization) string, so emitted tokens carry the vocab-key form.
    private let normalizedAddedInfo: [String: AddedTokenInfo]
    private let nonNormalizedAddedRegex: NSRegularExpression?
    private let normalizedAddedRegex: NSRegularExpression?
    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?
    private let cleanUpTokenizationSpaces: Bool

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
        tokenizerData: Config,
        model: any TokenizingModel,
        runtimeConfiguration: TokenizerRuntimeConfiguration
    ) throws {
        self.model = model
        let parsed = Self.parseAddedTokens(from: tokenizerData)
        self.specialTokens = parsed.special
        self.addedTokens = Set(parsed.tokens.keys)
        self.cleanUpTokenizationSpaces = runtimeConfiguration.cleanUpTokenizationSpaces

        let addedTokenInfo: [AddedTokenInfo] = tokenizerData["addedTokens"]
            .array(or: [])
            .compactMap { addedToken -> AddedTokenInfo? in
                guard let content = addedToken.content.string() else { return nil }
                guard let id = addedToken["id"].integer() else { return nil }
                let special = addedToken["special"].boolean(or: false)
                return AddedTokenInfo(
                    content: content,
                    id: id,
                    lstrip: addedToken["lstrip"].boolean(or: false),
                    rstrip: addedToken["rstrip"].boolean(or: false),
                    // Config's subscript normalizes camelCase → snake_case, so
                    // `addedToken["singleWord"]` reads `tokenizer.json`'s `single_word`.
                    singleWord: addedToken["singleWord"].boolean(or: false),
                    // Upstream default: `normalized` is `!special` when not explicitly set.
                    normalized: addedToken["normalized"].boolean() ?? !special
                )
            }
            .sorted { $0.content.count > $1.content.count }

        let resolvedPreTokenizer = try PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        let resolvedNormalizer = try NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        let resolvedPostProcessor = try PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        let resolvedDecoder = try DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: self.addedTokens)

        preTokenizer = resolvedPreTokenizer
        normalizer = resolvedNormalizer
        postProcessor = resolvedPostProcessor
        decoder = resolvedDecoder

        // Phase-1 matches non-normalized tokens against raw input. Pattern keys are
        // the raw contents and map directly to each info.
        let nonNormalizedInfos = addedTokenInfo.filter { !$0.normalized }
        self.nonNormalizedAddedInfo = Dictionary(
            nonNormalizedInfos.map { ($0.content, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        self.nonNormalizedAddedRegex = Self.buildAddedTokensRegex(
            patternsWithFlags: nonNormalizedInfos.map {
                (pattern: $0.content, lstrip: $0.lstrip, rstrip: $0.rstrip)
            }
        )

        // Phase-2 matches normalized tokens against normalized input. Build the
        // pattern from each token's *normalized* content and key the lookup dict
        // by the same normalized form, mirroring upstream's `normalized_cache`.
        // Sort again after normalization because normalization can change length
        // and length-descending ordering is what makes the `|`-alternation match
        // the longest candidate at each position.
        let normalizedInfos = addedTokenInfo.filter { $0.normalized }
        let normalizedPairs: [(normalizedContent: String, info: AddedTokenInfo)] =
            normalizedInfos
            .map { info in
                let normalized = resolvedNormalizer.map { $0(text: info.content) } ?? info.content
                return (normalized, info)
            }
            .sorted { $0.normalizedContent.count > $1.normalizedContent.count }
        self.normalizedAddedInfo = Dictionary(
            normalizedPairs.map { ($0.normalizedContent, $0.info) },
            uniquingKeysWith: { first, _ in first }
        )
        self.normalizedAddedRegex = Self.buildAddedTokensRegex(
            patternsWithFlags: normalizedPairs.map {
                (pattern: $0.normalizedContent, lstrip: $0.info.lstrip, rstrip: $0.info.rstrip)
            }
        )
    }

    private static func buildAddedTokensRegex(
        patternsWithFlags: [(pattern: String, lstrip: Bool, rstrip: Bool)]
    ) -> NSRegularExpression? {
        guard !patternsWithFlags.isEmpty else { return nil }
        let pattern = patternsWithFlags.map { entry in
            let escaped = NSRegularExpression.escapedPattern(for: entry.pattern)
            let prefix = entry.lstrip ? #"\s*"# : ""
            let suffix = entry.rstrip ? #"\s*"# : ""
            return "\(prefix)(\(escaped))\(suffix)"
        }.joined(separator: "|")
        return try? NSRegularExpression(pattern: pattern, options: [])
    }

    /// Unicode-aware word-character check matching the Rust `regex` crate's default
    /// `\w`, used by upstream `AddedVocabulary`'s `single_word` boundary regex
    /// (`tokenizers/src/tokenizer/added_vocabulary.rs`). The Rust crate's `\w` is
    /// `\p{word}` = `\p{L} ∪ \p{N} ∪ \p{M} ∪ \p{Pc}`. `CharacterSet.alphanumerics`
    /// covers only `\p{L} ∪ \p{N}`, so combining marks and connector punctuation
    /// need explicit checks; `_` is already included in `\p{Pc}`.
    private static func isWordChar(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
            .modifierLetter, .otherLetter:
            return true
        case .decimalNumber, .letterNumber, .otherNumber:
            return true
        case .nonspacingMark, .spacingMark, .enclosingMark:
            return true
        case .connectorPunctuation:
            return true
        default:
            return false
        }
    }

    private func findAddedTokenMatches(
        in text: String,
        regex: NSRegularExpression?,
        infoByMatchedContent: [String: AddedTokenInfo]
    ) -> [(outer: NSRange, value: String, id: Int)] {
        guard let regex else { return [] }
        let ns = text as NSString
        let fullRange = NSRange(location: 0, length: ns.length)
        var results: [(NSRange, String, Int)] = []
        let scalars = text.unicodeScalars
        for match in regex.matches(in: text, options: [], range: fullRange) {
            var captured: NSRange?
            for groupIndex in 1..<match.numberOfRanges {
                let range = match.range(at: groupIndex)
                if range.location != NSNotFound {
                    captured = range
                    break
                }
            }
            guard let captured else { continue }
            let matchedContent = ns.substring(with: captured)
            guard let info = infoByMatchedContent[matchedContent] else { continue }

            if info.singleWord {
                // Upstream checks `single_word` against the raw content bounds from the
                // Aho-Corasick matcher, *before* extending for lstrip / rstrip. See
                // `AddedVocabulary::find_matches` in upstream's `added_vocabulary.rs`:
                // the regex uses `start` / `stop` (the content bounds) for the
                // surrounding-word check, then separately widens `start` / `stop` for
                // lstrip / rstrip. We mirror that by reading the captured (inner)
                // group's NSRange rather than `match.range` (which may include
                // consumed whitespace from `\s*` bookends). Using the outer range
                // would reject `<mask>` in `"a <mask>"` because `a` is a word
                // character immediately before the consumed leading space.
                guard let contentRange = Range(captured, in: text) else { continue }
                let leftOK: Bool
                if contentRange.lowerBound == text.startIndex {
                    leftOK = true
                } else {
                    let before = scalars.index(before: contentRange.lowerBound)
                    leftOK = !Self.isWordChar(scalars[before])
                }
                let rightOK: Bool
                if contentRange.upperBound == text.endIndex {
                    rightOK = true
                } else {
                    rightOK = !Self.isWordChar(scalars[contentRange.upperBound])
                }
                guard leftOK && rightOK else { continue }
            }

            // Emit the outer (widened) slice as the token value, matching upstream
            // `AddedVocabulary::split_with_indices` (`added_vocabulary.rs`) which
            // stores `slice.get().to_owned()` — the lstrip/rstrip-widened text in
            // phase 1 and the normalized-input slice in phase 2 — as the Token's
            // value. The paired `id` keeps the added-token association so `encode`
            // resolves the correct ID even though the value is not a vocab key.
            results.append((match.range, ns.substring(with: match.range), info.id))
        }
        return results
    }

    private func splitByMatches(
        _ text: String,
        matches: [(outer: NSRange, value: String, id: Int)]
    ) -> [(value: String, idOverride: Int?)] {
        let ns = text as NSString
        var sections: [(String, Int?)] = []
        var cursor = 0
        for (outer, value, id) in matches {
            if cursor < outer.location {
                let sliceRange = NSRange(location: cursor, length: outer.location - cursor)
                sections.append((ns.substring(with: sliceRange), nil))
            }
            sections.append((value, id))
            cursor = outer.location + outer.length
        }
        if cursor < ns.length {
            let sliceRange = NSRange(location: cursor, length: ns.length - cursor)
            sections.append((ns.substring(with: sliceRange), nil))
        }
        return sections
    }

    private func compiledTemplate(for templateString: String) throws -> Template {
        templateCacheLock.lock()
        if let cached = templateCache[templateString] {
            templateCacheLock.unlock()
            return cached
        }
        templateCacheLock.unlock()

        let compiled: Template
        do {
            compiled = try Template(templateString, with: .init(lstripBlocks: true, trimBlocks: true))
        } catch let jinjaError as JinjaError {
            throw TokenizerError.chatTemplate("Failed to compile chat template: \(jinjaError.localizedDescription)")
        }

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

    /// Id-aware post-processing: preserves `idOverride` on passthrough tokens
    /// positionally, so added-token matches still encode to their added-token
    /// id even when the widened value isn't a vocab key. Built-in
    /// post-processors override the `[PostProcessorToken]` requirement to
    /// forward overrides; the default extension delegates to the `[String]`
    /// method and drops overrides for any user-supplied conformer.
    private func postProcess(_ tokens: [CoreToken], addSpecialTokens: Bool) -> [CoreToken] {
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

    /// The pre-post-processor tokenization: added-tokens matching, pre-tokenization,
    /// model step, and unknown fusion. `tokenize(text:)` and `encode(text:addSpecialTokens:)`
    /// both call this and then decide whether to run the post-processor (and with
    /// which `addSpecialTokens` setting).
    private func tokenizeCore(text: String) -> [CoreToken] {
        // Two-pass matcher over `added_tokens`, matching upstream
        // `tokenizers/src/tokenizer/added_vocabulary.rs`:
        //   pass 1 — `normalized: false` tokens matched against raw input
        //   pass 2 — `normalized: true` tokens matched against normalized input
        // Each pass filters `single_word` matches using the content-only bounds
        // (the regex's inner capture group), not the outer match range that may
        // include whitespace consumed by lstrip/rstrip.
        var sections: [[CoreToken]] = []
        var isFirstSection = true

        let phase1Matches = findAddedTokenMatches(
            in: text,
            regex: nonNormalizedAddedRegex,
            infoByMatchedContent: nonNormalizedAddedInfo
        )
        for phase1 in splitByMatches(text, matches: phase1Matches) {
            if let phase1Id = phase1.idOverride {
                sections.append([CoreToken(value: phase1.value, idOverride: phase1Id)])
                isFirstSection = false
                continue
            }

            let normalized = normalize(phase1.value)
            let phase2Matches = findAddedTokenMatches(
                in: normalized,
                regex: normalizedAddedRegex,
                infoByMatchedContent: normalizedAddedInfo
            )
            for phase2 in splitByMatches(normalized, matches: phase2Matches) {
                if let phase2Id = phase2.idOverride {
                    sections.append([CoreToken(value: phase2.value, idOverride: phase2Id)])
                    isFirstSection = false
                    continue
                }
                let pre = preTokenize(
                    phase2.value,
                    options: isFirstSection ? [.firstSection] : []
                )
                sections.append(pre.flatMap { model($0).map { CoreToken(value: $0, idOverride: nil) } })
                isFirstSection = false
            }
        }

        return sections.flatMap { $0 }
    }

    /// Runs the full tokenization pipeline (including the post-processor with
    /// specials suppressed), matching Python transformers v5's `TokenizersBackend.tokenize`
    /// in `src/transformers/tokenization_utils_tokenizers.py`:
    ///
    ///     def tokenize(self, text, pair=None, add_special_tokens=False, **kwargs):
    ///         return self._encode_plus(text=text, text_pair=pair,
    ///                                  add_special_tokens=add_special_tokens, ...).tokens()
    ///
    /// and the Rust backend's `tokenizer.encode(text, false)` call. The post-processor
    /// runs because non-special transformations (e.g. Roberta's offset trimming) apply
    /// regardless of the `addSpecialTokens` flag.
    package func tokenize(text: String) -> [String] {
        postProcess(tokenizeCore(text: text), addSpecialTokens: false).map { $0.value }
    }

    package func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        postProcess(tokenizeCore(text: text), addSpecialTokens: addSpecialTokens).map { token in
            token.idOverride ?? model.convertTokenToId(token.value)!
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

        // Upstream (`tokenizers/src/tokenizer/mod.rs`) joins raw tokens with a space when
        // no decoder is configured. When a decoder ran, its output is already text parts
        // (e.g. ByteLevel resolves byte-encoded whitespace) and should be concatenated
        // without adding a separator.
        let separator = decoder == nil ? " " : ""
        let decoded = decodeTokens(tokenStrings).joined(separator: separator)
        return cleanUp(text: decoded)
    }

    package func renderChatTemplate(template: String, contextObject: [String: Any]) throws -> String {
        // Compile errors are already wrapped by compiledTemplate(for:); only catch
        // JinjaError from Value(any:) and render here.
        let compiledTemplate = try compiledTemplate(for: template)
        do {
            let context = try Dictionary(
                uniqueKeysWithValues: contextObject.map { key, value in
                    (key, try Value(any: value))
                })
            return try compiledTemplate.render(context)
        } catch let jinjaError as JinjaError {
            throw TokenizerError.chatTemplate("Failed to render chat template: \(jinjaError.localizedDescription)")
        }
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

package enum AutoTokenizerDirectorySidecars {
    static func load(from directory: URL) throws -> Config {
        var tokenizerConfig = loadOptionalConfig(from: directory.appending(path: "tokenizer_config.json"))

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

            // Shape-based inference for untagged `tokenizer.json` files uses the
            // same priority as upstream `ModelWrapper`'s custom `Deserialize` impl
            // in `tokenizers/src/models/mod.rs`: BPE (merges present), then Unigram
            // (vocab is an array). WordPiece / WordLevel go through the slow path.
            // When inference resolves a type, stamp it into `model.type` so the
            // model-selection switch in `TokenizerModel.from` sees it without
            // needing to re-read the vocab (which we strip below to avoid
            // duplicate parsing).
            let hasMerges = model["merges"] is [Any]
            let vocabIsArray = model["vocab"] is NSArray

            let isBPE = modelType == "BPE" || (modelType == nil && hasMerges)
            let isUnigram =
                modelType == "Unigram" || (modelType == nil && !hasMerges && vocabIsArray)

            if isBPE, let vocab = model["vocab"] as? NSDictionary {
                tokenizerVocab = .bpe(vocab)
                if let merges = model["merges"] as? [Any] {
                    tokenizerMerges = TokenizerMerges(merges)
                }
                model.removeObject(forKey: "vocab")
                model.removeObject(forKey: "merges")
                if modelType == nil {
                    model["type"] = "BPE"
                }
                parsed["model"] = model
            } else if isUnigram, let vocab = model["vocab"] as? NSArray {
                tokenizerVocab = .unigram(vocab)
                model.removeObject(forKey: "vocab")
                if modelType == nil {
                    model["type"] = "Unigram"
                }
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
        tokenizerConfig: Config
    ) async throws -> any Tokenizer {
        let artifacts = try loadTokenizerArtifacts(from: directory)
        return try await SwiftAutoTokenizerFactory.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: artifacts.tokenizerData,
            tokenizerVocab: artifacts.tokenizerVocab,
            tokenizerMerges: artifacts.tokenizerMerges
        )
    }

    package static func load(from directory: URL) async throws -> any Tokenizer {
        let tokenizerConfig = try loadTokenizerConfig(from: directory)
        return try await loadTokenizerCore(from: directory, tokenizerConfig: tokenizerConfig)
    }
}

private enum SwiftAutoTokenizerFactory {
    static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        tokenizerVocab: TokenizerVocab?,
        tokenizerMerges: TokenizerMerges?
    ) async throws -> any Tokenizer {
        let parsed = SwiftTokenizerBackend.parseAddedTokens(from: tokenizerData)
        let model = try await TokenizerModel.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: parsed.tokens,
            tokenizerVocab: tokenizerVocab,
            tokenizerMerges: tokenizerMerges
        )
        let runtimeConfiguration = TokenizerRuntimeConfiguration(tokenizerConfig: tokenizerConfig)
        let backend = try SwiftTokenizerBackend(
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
}
#endif
