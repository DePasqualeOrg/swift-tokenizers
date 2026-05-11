// Copyright © Anthony DePasquale

import Foundation
import TokenizersFFI

private struct RustTokenizerDescriptor {
    let runtimeConfiguration: TokenizerRuntimeConfiguration
    let bosTokenId: Int?
    let eosTokenId: Int?
    let unknownTokenId: Int?
    let baseVocabSize: Int
    let totalVocabSize: Int
}

/// Owns a UniFFI-managed Rust tokenizer handle and exposes the per-method
/// surface used by the public ``Tokenizer`` protocol implementation.
///
/// Conformance to `Sendable` (not `@unchecked`) holds because every stored
/// property is `let` — the compiler verifies the structural part. Concurrent
/// calls into the stored `TokenizersFFI.Tokenizer` are themselves safe because
/// the generated type is declared `@unchecked Sendable` with `NSLock`
/// synchronization in UniFFI's `UniffiHandleMap`.
package final class RustTokenizer: Sendable {
    private let inner: TokenizersFFI.Tokenizer

    package let bosToken: String?
    package let eosToken: String?
    package let unknownToken: String?
    package let bosTokenId: Int?
    package let eosTokenId: Int?
    package let unknownTokenId: Int?
    package let baseVocabSize: Int
    package let totalVocabSize: Int

    fileprivate init(inner: TokenizersFFI.Tokenizer, descriptor: RustTokenizerDescriptor) {
        self.inner = inner
        bosToken = descriptor.runtimeConfiguration.bosToken
        eosToken = descriptor.runtimeConfiguration.eosToken
        unknownToken = descriptor.runtimeConfiguration.unknownToken
        bosTokenId = descriptor.bosTokenId
        eosTokenId = descriptor.eosTokenId
        unknownTokenId = descriptor.unknownTokenId
        baseVocabSize = descriptor.baseVocabSize
        totalVocabSize = descriptor.totalVocabSize
    }

    package func getVocabSize(withAddedTokens: Bool) -> Int {
        withAddedTokens ? totalVocabSize : baseVocabSize
    }

    package func tokenize(text: String) throws(TokenizerError) -> [String] {
        try bridgeFFIErrors {
            try inner.tokenize(text: text)
        }
    }

    package func encode(text: String, textPair: String?, addSpecialTokens: Bool) throws(TokenizerError) -> [Int] {
        try bridgeFFIErrors {
            try inner.encode(
                text: text,
                textPair: textPair,
                addSpecialTokens: addSpecialTokens
            ).map(Int.init)
        }
    }

    package func encodeBatch(
        _ inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool
    ) throws(TokenizerError) -> [[Int]] {
        let ffiInputs = inputs.map { TokenizersFFI.EncodeInput(text: $0.text, textPair: $0.textPair) }
        return try bridgeFFIErrors {
            try inner.encodeBatch(inputs: ffiInputs, addSpecialTokens: addSpecialTokens)
                .map { $0.map(Int.init) }
        }
    }

    package func encodeWithMetadata(
        text: String,
        textPair: String?,
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws(TokenizerError) -> TokenizerEncoding {
        try bridgeFFIErrors {
            try inner.encodeWithMetadata(
                text: text,
                textPair: textPair,
                addSpecialTokens: addSpecialTokens,
                offsetUnit: offsetUnit.ffi
            ).materialized
        }
    }

    package func encodeBatchWithMetadata(
        _ inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws(TokenizerError) -> [TokenizerEncoding] {
        let ffiInputs = inputs.map { TokenizersFFI.EncodeInput(text: $0.text, textPair: $0.textPair) }
        return try bridgeFFIErrors {
            try inner.encodeBatchWithMetadata(
                inputs: ffiInputs,
                addSpecialTokens: addSpecialTokens,
                offsetUnit: offsetUnit.ffi
            ).map { $0.materialized }
        }
    }

    package func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws(TokenizerError) -> String {
        // Reject out-of-range IDs at the boundary. Rust's tokenizer API takes
        // `u32`, so anything below `0` is semantically invalid; the FFI bridge
        // uses `Int32`, so anything above `Int32.max` is rejected here.
        var ffiIds: [Int32] = []
        ffiIds.reserveCapacity(tokenIds.count)
        for id in tokenIds {
            guard id >= 0, let value = Int32(exactly: id) else {
                throw TokenizerError.invalidTokenId(id)
            }
            ffiIds.append(value)
        }
        return try bridgeFFIErrors {
            try inner.decode(tokenIds: ffiIds, skipSpecialTokens: skipSpecialTokens)
        }
    }

    package func convertTokenToId(_ token: String) -> Int? {
        inner.convertTokenToId(token: token).map(Int.init)
    }

    package func convertIdToToken(_ id: Int) -> String? {
        guard let id32 = Int32(exactly: id) else { return nil }
        return inner.convertIdToToken(tokenId: id32)
    }

    package func renderChatTemplate(template: String, contextObject: [String: Any]) throws(TokenizerError) -> String {
        try bridgeFFIErrors {
            let contextJSON = try JSONBridge.jsonString(from: contextObject)
            return try TokenizersFFI.renderTemplate(template: template, contextJson: contextJSON)
        }
    }

    package func applyChatTemplate(
        template: String,
        contextObject: [String: Any],
        truncation: Bool,
        maxLength: Int?
    ) throws(TokenizerError) -> [Int] {
        let ffiMaxLength: UInt64?
        if let maxLength {
            guard maxLength >= 0 else {
                throw TokenizerError.invalidConfiguration("maxLength must be non-negative")
            }
            ffiMaxLength = UInt64(maxLength)
        } else {
            ffiMaxLength = nil
        }
        return try bridgeFFIErrors {
            let contextJSON = try JSONBridge.jsonString(from: contextObject)
            return try inner.applyChatTemplate(
                template: template,
                contextJson: contextJSON,
                truncation: truncation,
                maxLength: ffiMaxLength
            ).map(Int.init)
        }
    }

    /// Decode without any post-processing (no `cleanUp(text:)` rewrite). Used by
    /// `StreamingDetokenizer` to maintain byte-prefix-monotonic decodes.
    package func rawDecode(tokenIds: [Int], skipSpecialTokens: Bool) throws(TokenizerError) -> String {
        try decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
}

/// Wraps a UniFFI-throwing call so any error becomes a public ``TokenizerError``.
///
/// The generated `TokenizersFFI` wrapper throws two kinds of errors: the
/// project-defined `TokenizersFFI.TokenizerError` (which we map onto our public
/// shape via `.bridged`) and `UniffiInternalError` for buffer overflows, stale
/// handles, unexpected enum tags, and propagated Rust panics. The latter is
/// `fileprivate` in the generated wrapper, so callers cannot pattern-match it;
/// remap it onto `.internalError` so all public throws stay inside the
/// declared `TokenizerError` shape. Other errors thrown from inside `body`
/// (for example, `NSError` from `JSONSerialization`) are remapped the same way.
private func bridgeFFIErrors<T>(_ body: () throws -> T) throws(TokenizerError) -> T {
    do {
        return try body()
    } catch let error as TokenizersFFI.TokenizerError {
        throw error.bridged
    } catch let error as Tokenizers.TokenizerError {
        throw error
    } catch {
        throw Tokenizers.TokenizerError.internalError(error.localizedDescription)
    }
}

private extension OffsetUnit {
    var ffi: TokenizersFFI.OffsetUnit {
        switch self {
        case .utf8: .utf8
        case .unicodeScalar: .unicodeScalar
        }
    }

    init(_ ffi: TokenizersFFI.OffsetUnit) {
        self =
            switch ffi {
            case .utf8: .utf8
            case .unicodeScalar: .unicodeScalar
            }
    }
}

private extension TokenizersFFI.OffsetSpan {
    var bridged: OffsetSpan {
        OffsetSpan(start: Int(start), end: Int(end))
    }
}

private extension TokenizersFFI.EncodingMetadata {
    func materialized(overflowEncodings: [TokenizerEncoding] = []) -> TokenizerEncoding {
        TokenizerEncoding(
            tokenIds: tokenIds.map(Int.init),
            tokenTypeIds: tokenTypeIds.map(Int.init),
            tokens: tokens,
            wordIndices: wordIndices.map { $0.map(Int.init) },
            offsetSpans: offsetSpans.map(\.bridged),
            specialTokensMask: specialTokensMask.map(Int.init),
            attentionMask: attentionMask.map(Int.init),
            sequenceIndices: sequenceIndices.map { $0.map(Int.init) },
            sequenceCount: Int(sequenceCount),
            overflowEncodings: overflowEncodings,
            offsetUnit: OffsetUnit(offsetUnit)
        )
    }
}

private extension TokenizersFFI.Encoding {
    /// Materializes the public `TokenizerEncoding`. Per the Option B flattening
    /// documented on `Encoding` in `rust/src/core/uniffi_api.rs`, child
    /// encodings never carry their own overflow children, so each child uses
    /// the default empty `overflowEncodings: []`.
    var materialized: TokenizerEncoding {
        let overflowEncodings = overflowing.map { $0.materialized() }
        return primary.materialized(overflowEncodings: overflowEncodings)
    }
}

private extension TokenizersFFI.NamedChatTemplate {
    var bridged: TokenizerRuntimeConfiguration.NamedChatTemplate {
        TokenizerRuntimeConfiguration.NamedChatTemplate(name: name, template: template)
    }
}

private extension TokenizerRuntimeConfiguration.NamedChatTemplate {
    var ffi: TokenizersFFI.NamedChatTemplate {
        TokenizersFFI.NamedChatTemplate(name: name, template: template)
    }
}

private extension TokenizersFFI.ChatTemplateSource {
    var bridged: TokenizerRuntimeConfiguration.ChatTemplateSource {
        switch self {
        case .none: .none
        case let .literal(template): .literal(template)
        case let .named(templates): .named(templates.map(\.bridged))
        }
    }
}

private extension TokenizerRuntimeConfiguration.ChatTemplateSource {
    var ffi: TokenizersFFI.ChatTemplateSource {
        switch self {
        case .none: .none
        case let .literal(template): .literal(template: template)
        case let .named(templates): .named(templates: templates.map(\.ffi))
        }
    }
}

private extension TokenizersFFI.RuntimeConfiguration {
    var bridged: TokenizerRuntimeConfiguration {
        get throws(TokenizerError) {
            let modelMaxLengthInt: Int?
            if let modelMaxLength {
                guard modelMaxLength <= UInt64(Int.max) else {
                    throw TokenizerError.invalidConfiguration("model_max_length exceeds Swift Int.max")
                }
                modelMaxLengthInt = Int(modelMaxLength)
            } else {
                modelMaxLengthInt = nil
            }
            return TokenizerRuntimeConfiguration(
                bosToken: bosToken,
                eosToken: eosToken,
                unknownToken: unknownToken,
                sepToken: sepToken,
                padToken: padToken,
                clsToken: clsToken,
                maskToken: maskToken,
                additionalSpecialTokens: additionalSpecialTokens,
                cleanUpTokenizationSpaces: cleanUpTokenizationSpaces,
                modelMaxLength: modelMaxLengthInt,
                chatTemplate: chatTemplate.bridged
            )
        }
    }
}

private extension TokenizerRuntimeConfiguration {
    var ffi: TokenizersFFI.RuntimeConfiguration {
        get throws(TokenizerError) {
            let ffiModelMaxLength: UInt64?
            if let modelMaxLength {
                guard modelMaxLength >= 0 else {
                    throw TokenizerError.invalidConfiguration("modelMaxLength must be non-negative")
                }
                ffiModelMaxLength = UInt64(modelMaxLength)
            } else {
                ffiModelMaxLength = nil
            }
            return TokenizersFFI.RuntimeConfiguration(
                bosToken: bosToken,
                eosToken: eosToken,
                unknownToken: unknownToken,
                sepToken: sepToken,
                padToken: padToken,
                clsToken: clsToken,
                maskToken: maskToken,
                additionalSpecialTokens: additionalSpecialTokens,
                cleanUpTokenizationSpaces: cleanUpTokenizationSpaces,
                modelMaxLength: ffiModelMaxLength,
                chatTemplate: chatTemplate.ffi
            )
        }
    }
}

private extension TokenizersFFI.TokenizerDescriptor {
    var bridged: RustTokenizerDescriptor {
        get throws(TokenizerError) {
            try RustTokenizerDescriptor(
                runtimeConfiguration: runtimeConfiguration.bridged,
                bosTokenId: bosTokenId.map(Int.init),
                eosTokenId: eosTokenId.map(Int.init),
                unknownTokenId: unknownTokenId.map(Int.init),
                baseVocabSize: Int(baseVocabSize),
                totalVocabSize: Int(totalVocabSize)
            )
        }
    }
}

private extension TokenizersFFI.TokenizerError {
    var bridged: Tokenizers.TokenizerError {
        // UniFFI 0.31.x's Swift bindgen quirk: error-enum variants are emitted
        // in PascalCase (Rust `MissingConfig` → Swift `.MissingConfig`), unlike
        // `uniffi::Enum` variants, which are lowerCamelCase. The exhaustive
        // switch makes any future bindgen rename a build break, and the
        // wrapper-drift CI guard surfaces it on bindgen upgrades.
        switch self {
        case .MissingConfig:
            return .missingConfig
        case let .ChatTemplate(message):
            return .chatTemplate(message)
        case let .MismatchedConfig(message):
            return .invalidConfiguration(message)
        case let .Internal(message):
            return .internalError(message)
        }
    }
}

package enum RustAutoTokenizerDirectoryLoader {
    private static func makeTokenizer(inner: TokenizersFFI.Tokenizer) throws -> any Tokenizer {
        let descriptor = try inner.descriptor().bridged
        let runtimeConfiguration = descriptor.runtimeConfiguration
        let rustTokenizer = RustTokenizer(inner: inner, descriptor: descriptor)
        return PreTrainedTokenizer(
            rustTokenizer: rustTokenizer,
            runtimeConfiguration: runtimeConfiguration
        )
    }

    package static func loadRuntimeConfiguration(from directory: URL) throws(TokenizerError) -> TokenizerRuntimeConfiguration {
        try bridgeFFIErrors {
            try TokenizersFFI.loadRuntimeConfiguration(directoryPath: directory.path).bridged
        }
    }

    package static func loadTokenizerCore(
        from directory: URL,
        runtimeConfiguration: TokenizerRuntimeConfiguration
    ) async throws(TokenizerError) -> any Tokenizer {
        let ffiConfiguration = try runtimeConfiguration.ffi
        return try bridgeFFIErrors {
            let inner = try TokenizersFFI.Tokenizer.fromDirectoryWithRuntimeConfiguration(
                directoryPath: directory.path,
                runtimeConfiguration: ffiConfiguration
            )
            return try makeTokenizer(inner: inner)
        }
    }

    package static func load(from directory: URL) async throws(TokenizerError) -> any Tokenizer {
        try bridgeFFIErrors {
            let inner = try TokenizersFFI.Tokenizer.fromDirectory(directoryPath: directory.path)
            return try makeTokenizer(inner: inner)
        }
    }
}
