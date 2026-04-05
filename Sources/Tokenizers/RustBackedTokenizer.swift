import Foundation
import TokenizersCore

#if Rust
import TokenizersRustCore

private struct RustTokenizerDescriptor: Decodable {
    let runtimeConfiguration: TokenizerRuntimeConfiguration
    let bosTokenId: Int?
    let eosTokenId: Int?
    let unknownTokenId: Int?
}

private enum RustFFI {
    static func emptyBuffer() -> st_owned_buffer_t {
        st_owned_buffer_t(data: nil, len: 0)
    }

    static func emptyError() -> st_error_t {
        st_error_t(code: 0, message: emptyBuffer())
    }

    static func data(from buffer: st_owned_buffer_t) -> Data {
        guard let data = buffer.data, buffer.len > 0 else {
            return Data()
        }
        return Data(bytes: data, count: Int(buffer.len))
    }

    static func string(from buffer: st_owned_buffer_t) -> String {
        String(decoding: data(from: buffer), as: UTF8.self)
    }

    static func takeString(from buffer: st_owned_buffer_t) -> String {
        defer { st_free_owned_buffer(buffer) }
        return string(from: buffer)
    }

    static func takeData(from buffer: st_owned_buffer_t) -> Data {
        defer { st_free_owned_buffer(buffer) }
        return data(from: buffer)
    }

    static func takeIntArray(pointer: UnsafeMutablePointer<Int32>?, count: Int) -> [Int] {
        guard let pointer, count > 0 else {
            return []
        }
        defer { st_free_int32_array(pointer, count) }
        let buffer = UnsafeBufferPointer(start: pointer, count: count)
        return buffer.map(Int.init)
    }

    static func tokenizerError(from error: st_error_t) -> TokenizerError {
        defer { st_free_owned_buffer(error.message) }
        let message = string(from: error.message)
        switch error.code {
        case 1:
            return .missingConfig
        case 6:
            return .chatTemplate(message)
        case 7:
            return .missingChatTemplate
        case 9:
            return .mismatchedConfig(message)
        default:
            let fallback = message.isEmpty ? "Rust tokenizer error code \(error.code)" : message
            return .mismatchedConfig(fallback)
        }
    }
}

package final class RustProxyModel: TokenizingModel, @unchecked Sendable {
    private let handle: OpaquePointer
    private let vocabCountLock = NSLock()
    private var cachedVocabCount: Int?

    package let bosToken: String?
    package let eosToken: String?
    package let unknownToken: String?
    package let fuseUnknownTokens: Bool
    package let bosTokenId: Int?
    package let eosTokenId: Int?
    package let unknownTokenId: Int?

    fileprivate init(handle: OpaquePointer, descriptor: RustTokenizerDescriptor) {
        self.handle = handle
        bosToken = descriptor.runtimeConfiguration.bosToken
        eosToken = descriptor.runtimeConfiguration.eosToken
        unknownToken = descriptor.runtimeConfiguration.unknownToken
        fuseUnknownTokens = descriptor.runtimeConfiguration.fuseUnknownTokens
        bosTokenId =
            descriptor.bosTokenId
            ?? bosToken.flatMap { try? RustTokenizerBackend.convertTokenToId(handle: handle, token: $0) }
        eosTokenId =
            descriptor.eosTokenId
            ?? eosToken.flatMap { try? RustTokenizerBackend.convertTokenToId(handle: handle, token: $0) }
        unknownTokenId =
            descriptor.unknownTokenId
            ?? unknownToken.flatMap { try? RustTokenizerBackend.convertTokenToId(handle: handle, token: $0) }
    }

    // Keep vocab count off the Rust load path. This mirrors Python fast tokenizers, which query
    // vocabulary size lazily from the backend in `tokenization_utils_tokenizers.py`, and lets us
    // revisit the eager Swift design later without paying this cost during tokenizer creation.
    package var vocabCount: Int {
        vocabCountLock.lock()
        defer { vocabCountLock.unlock() }

        if let cachedVocabCount {
            return cachedVocabCount
        }

        let resolvedVocabCount = (try? RustTokenizerBackend.vocabCount(handle: handle)) ?? 0
        cachedVocabCount = resolvedVocabCount
        return resolvedVocabCount
    }

    package func tokenize(text: String) -> [String] {
        (try? RustTokenizerBackend.tokenize(handle: handle, text: text)) ?? []
    }

    package func convertTokenToId(_ token: String) -> Int? {
        if let tokenID = try? RustTokenizerBackend.convertTokenToId(handle: handle, token: token) {
            return tokenID
        }
        return unknownTokenId
    }

    package func convertIdToToken(_ id: Int) -> String? {
        try? RustTokenizerBackend.convertIdToToken(handle: handle, id: id)
    }
}

private final class RustTokenizerBackend: TokenizerExecutionBackend, @unchecked Sendable {
    private let handle: OpaquePointer
    let performsCleanup = false

    init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        st_tokenizer_destroy(handle)
    }

    package func tokenize(text: String) -> [String] {
        (try? Self.tokenize(handle: handle, text: text)) ?? []
    }

    package func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        (try? Self.encode(handle: handle, text: text, addSpecialTokens: addSpecialTokens)) ?? []
    }

    package func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        (try? Self.decode(handle: handle, tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)) ?? ""
    }

    package func renderChatTemplate(template: String, contextObject: [String: Any]) throws -> String {
        let contextJSON = try JSONBridge.jsonString(from: contextObject)
        return try Self.renderTemplate(template: template, contextJSON: contextJSON)
    }

    package func applyChatTemplate(
        template: String,
        contextObject: [String: Any],
        truncation: Bool,
        maxLength: Int?
    ) throws -> [Int] {
        let contextJSON = try JSONBridge.jsonString(from: contextObject)
        return try Self.renderChatTemplate(
            handle: handle,
            template: template,
            contextJSON: contextJSON,
            truncation: truncation,
            maxLength: maxLength
        )
    }

    static func tokenize(handle: OpaquePointer, text: String) throws -> [String] {
        var tokensBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()
        let success = text.withCString { textPointer in
            st_tokenizer_tokenize_to_json(handle, textPointer, &tokensBuffer, &error)
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        defer { st_free_owned_buffer(tokensBuffer) }
        let data = RustFFI.data(from: tokensBuffer)
        return try JSONDecoder().decode([String].self, from: data)
    }

    static func encode(handle: OpaquePointer, text: String, addSpecialTokens: Bool) throws -> [Int] {
        var tokenIDs: UnsafeMutablePointer<Int32>?
        var count: Int = 0
        var error = RustFFI.emptyError()
        let success = text.withCString { textPointer in
            st_tokenizer_encode(handle, textPointer, addSpecialTokens, &tokenIDs, &count, &error)
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return RustFFI.takeIntArray(pointer: tokenIDs, count: count)
    }

    static func decode(handle: OpaquePointer, tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
        let ids = tokenIds.map(Int32.init)
        var textBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()
        let success = ids.withUnsafeBufferPointer { idsPointer in
            st_tokenizer_decode(
                handle,
                idsPointer.baseAddress,
                idsPointer.count,
                skipSpecialTokens,
                &textBuffer,
                &error
            )
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return RustFFI.takeString(from: textBuffer)
    }

    static func convertTokenToId(handle: OpaquePointer, token: String) throws -> Int? {
        var found = false
        var tokenID: Int32 = 0
        var error = RustFFI.emptyError()
        let success = token.withCString { tokenPointer in
            st_tokenizer_convert_token_to_id(handle, tokenPointer, &found, &tokenID, &error)
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return found ? Int(tokenID) : nil
    }

    static func convertIdToToken(handle: OpaquePointer, id: Int) throws -> String? {
        var found = false
        var tokenBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()
        let success = st_tokenizer_convert_id_to_token(
            handle,
            Int32(id),
            &found,
            &tokenBuffer,
            &error
        )
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        guard found else {
            return nil
        }
        return RustFFI.takeString(from: tokenBuffer)
    }

    static func vocabCount(handle: OpaquePointer) throws -> Int {
        var vocabCount: Int = 0
        var error = RustFFI.emptyError()
        let success = st_tokenizer_vocab_count(handle, &vocabCount, &error)
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return vocabCount
    }

    static func renderChatTemplate(
        handle: OpaquePointer,
        template: String,
        contextJSON: String,
        truncation: Bool,
        maxLength: Int?
    ) throws -> [Int] {
        var tokenIDs: UnsafeMutablePointer<Int32>?
        var count: Int = 0
        var error = RustFFI.emptyError()

        let success = template.withCString { templatePointer in
            contextJSON.withCString { contextPointer in
                st_tokenizer_apply_chat_template(
                    handle,
                    templatePointer,
                    contextPointer,
                    truncation,
                    maxLength != nil,
                    UInt32(maxLength ?? 0),
                    &tokenIDs,
                    &count,
                    &error
                )
            }
        }

        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return RustFFI.takeIntArray(pointer: tokenIDs, count: count)
    }

    static func renderTemplate(template: String, contextJSON: String) throws -> String {
        var textBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()

        let success = template.withCString { templatePointer in
            contextJSON.withCString { contextPointer in
                st_render_template(templatePointer, contextPointer, &textBuffer, &error)
            }
        }

        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return RustFFI.takeString(from: textBuffer)
    }
}

package enum RustAutoTokenizerDirectoryLoader {
    private static func decodeDescriptor(from metadataData: Data) throws -> RustTokenizerDescriptor {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(RustTokenizerDescriptor.self, from: metadataData)
    }

    private static func makeTokenizer(
        handle: OpaquePointer,
        metadataData: Data
    ) throws -> any Tokenizer {
        let descriptor = try decodeDescriptor(from: metadataData)
        let model = RustProxyModel(handle: handle, descriptor: descriptor)
        let backend = RustTokenizerBackend(handle: handle)
        return PreTrainedTokenizer(
            model: model,
            runtimeConfiguration: descriptor.runtimeConfiguration,
            backend: backend
        )
    }

    private static func encodeRuntimeConfiguration(_ runtimeConfiguration: TokenizerRuntimeConfiguration) throws
        -> String
    {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(runtimeConfiguration)
        return String(decoding: data, as: UTF8.self)
    }

    package static func loadRuntimeConfiguration(from directory: URL) throws -> TokenizerRuntimeConfiguration {
        var configurationBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()

        let success = directory.path.withCString { directoryPath in
            st_load_tokenizer_runtime_configuration(directoryPath, &configurationBuffer, &error)
        }

        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }

        let configurationData = RustFFI.takeData(from: configurationBuffer)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(TokenizerRuntimeConfiguration.self, from: configurationData)
    }

    private static func validateStrictCompatibility(
        runtimeConfiguration: TokenizerRuntimeConfiguration,
        strict: Bool
    ) throws {
        // Unlike the Swift backend, Rust does not dispatch through a local `tokenizer_class`
        // registry. It validates the transformers-side name in strict mode to preserve the public
        // contract, then loads `tokenizer.json` through upstream `huggingface/tokenizers` and adds
        // only the compatibility behavior we explicitly need.
        _ = try TokenizerCompatibility.validateResolvedTokenizerName(
            tokenizerClass: runtimeConfiguration.tokenizerClass,
            modelType: runtimeConfiguration.modelType,
            strict: strict,
            isSupported: TokenizerCompatibility.isRustSupportedTokenizer(named:)
        )
    }

    package static func loadTokenizerCore(
        from directory: URL,
        runtimeConfiguration: TokenizerRuntimeConfiguration,
        strict: Bool
    ) async throws -> any Tokenizer {
        try validateStrictCompatibility(runtimeConfiguration: runtimeConfiguration, strict: strict)
        let runtimeConfigurationJSON = try encodeRuntimeConfiguration(runtimeConfiguration)

        var handle: OpaquePointer?
        var metadataBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()

        let success = directory.path.withCString { directoryPath in
            runtimeConfigurationJSON.withCString { runtimeConfigurationPointer in
                st_tokenizer_create_from_tokenizer_json(
                    directoryPath,
                    runtimeConfigurationPointer,
                    &handle,
                    &metadataBuffer,
                    &error
                )
            }
        }

        guard success, let handle else {
            throw RustFFI.tokenizerError(from: error)
        }

        return try makeTokenizer(handle: handle, metadataData: RustFFI.takeData(from: metadataBuffer))
    }

    package static func load(from directory: URL, strict: Bool) async throws -> any Tokenizer {
        let runtimeConfiguration = try loadRuntimeConfiguration(from: directory)
        try validateStrictCompatibility(runtimeConfiguration: runtimeConfiguration, strict: strict)

        var handle: OpaquePointer?
        var metadataBuffer = RustFFI.emptyBuffer()
        var error = RustFFI.emptyError()

        let success = directory.path.withCString { directoryPath in
            st_tokenizer_create_from_directory(directoryPath, &handle, &metadataBuffer, &error)
        }

        guard success, let handle else {
            throw RustFFI.tokenizerError(from: error)
        }

        return try makeTokenizer(handle: handle, metadataData: RustFFI.takeData(from: metadataBuffer))
    }
}
#endif
