import Foundation
import TokenizersRust

/// Calls `body` with a C-string pointer for `value`, or with `nil` when `value` is `nil`.
/// Used by FFI calls that take optional `const char *` arguments.
private func withOptionalCString<Result>(
    _ value: String?,
    _ body: (UnsafePointer<CChar>?) throws -> Result
) rethrows -> Result {
    guard let value else {
        return try body(nil)
    }
    return try value.withCString { pointer in
        try body(pointer)
    }
}

/// Materializes a batch of `(text, textPair?)` inputs as a `[st_encode_input_t]` whose
/// `const char *` pointers stay valid for the duration of `body`. Allocations are released
/// before `body` returns.
private func withEncodeInputs<Result>(
    _ inputs: [(text: String, textPair: String?)],
    _ body: ([st_encode_input_t]) throws -> Result
) rethrows -> Result {
    var allocations: [UnsafeMutablePointer<CChar>] = []
    allocations.reserveCapacity(inputs.count * 2)
    defer {
        for pointer in allocations { pointer.deallocate() }
    }

    func makeCString(_ value: String) -> UnsafePointer<CChar> {
        let utf8 = value.utf8CString
        let pointer = UnsafeMutablePointer<CChar>.allocate(capacity: utf8.count)
        utf8.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            pointer.initialize(from: base, count: utf8.count)
        }
        allocations.append(pointer)
        return UnsafePointer(pointer)
    }

    var ffiInputs: [st_encode_input_t] = []
    ffiInputs.reserveCapacity(inputs.count)
    for (text, textPair) in inputs {
        let textPointer = makeCString(text)
        let textPairPointer: UnsafePointer<CChar>? = textPair.map(makeCString)
        ffiInputs.append(st_encode_input_t(text: textPointer, text_pair: textPairPointer))
    }
    return try body(ffiInputs)
}

private struct RustTokenizerDescriptor {
    let runtimeConfiguration: TokenizerRuntimeConfiguration
    let bosTokenId: Int?
    let eosTokenId: Int?
    let unknownTokenId: Int?
    let baseVocabSize: Int
    let totalVocabSize: Int
}

private enum RustFFI {
    static func emptyBuffer() -> st_owned_buffer_t {
        st_owned_buffer_t(data: nil, len: 0)
    }

    static func emptyError() -> st_error_t {
        st_error_t(code: 0, message: emptyBuffer())
    }

    static func emptyRuntimeConfiguration() -> st_runtime_configuration_t {
        st_runtime_configuration_t(
            bos_token: emptyBuffer(),
            has_bos_token: false,
            eos_token: emptyBuffer(),
            has_eos_token: false,
            unknown_token: emptyBuffer(),
            has_unknown_token: false,
            sep_token: emptyBuffer(),
            has_sep_token: false,
            pad_token: emptyBuffer(),
            has_pad_token: false,
            cls_token: emptyBuffer(),
            has_cls_token: false,
            mask_token: emptyBuffer(),
            has_mask_token: false,
            additional_special_tokens: nil,
            additional_special_tokens_len: 0,
            clean_up_tokenization_spaces: true,
            model_max_length: 0,
            has_model_max_length: false,
            chat_template_kind: 0,
            chat_template_literal: emptyBuffer(),
            named_chat_templates: nil,
            named_chat_templates_len: 0
        )
    }

    static func emptyTokenizerDescriptor() -> st_tokenizer_descriptor_t {
        st_tokenizer_descriptor_t(
            runtime_configuration: emptyRuntimeConfiguration(),
            bos_token_id: 0,
            has_bos_token_id: false,
            eos_token_id: 0,
            has_eos_token_id: false,
            unknown_token_id: 0,
            has_unknown_token_id: false,
            base_vocab_size: 0,
            total_vocab_size: 0
        )
    }

    static func emptyEncoding() -> st_encoding_t {
        st_encoding_t(
            token_ids: nil,
            token_ids_len: 0,
            token_type_ids: nil,
            token_type_ids_len: 0,
            tokens: nil,
            tokens_len: 0,
            word_indices: nil,
            word_indices_present: nil,
            word_indices_len: 0,
            offset_spans: nil,
            offset_spans_len: 0,
            special_token_mask: nil,
            special_token_mask_len: 0,
            attention_mask: nil,
            attention_mask_len: 0,
            sequence_indices: nil,
            sequence_indices_present: nil,
            sequence_indices_len: 0,
            sequence_count: 0,
            overflow_encodings: nil,
            overflow_encodings_len: 0,
            offset_unit: 0
        )
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

    static func array<Element>(
        pointer: UnsafeMutablePointer<Element>?,
        count: Int,
        field: String
    ) throws -> [Element] {
        guard count > 0 else {
            return []
        }
        guard let pointer else {
            throw TokenizerError.invalidConfiguration("Rust encoding field \(field) was null with count \(count)")
        }
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    static func intArray(
        pointer: UnsafeMutablePointer<Int32>?,
        count: Int,
        field: String
    ) throws -> [Int] {
        try array(pointer: pointer, count: count, field: field).map(Int.init)
    }

    static func optionalIntArray(
        values: UnsafeMutablePointer<Int32>?,
        present: UnsafeMutablePointer<Bool>?,
        count: Int,
        field: String
    ) throws -> [Int?] {
        let values = try intArray(pointer: values, count: count, field: field)
        let present = try array(pointer: present, count: count, field: "\(field)_present")
        return zip(values, present).map { value, hasValue in
            hasValue ? value : nil
        }
    }

    static func stringArray(
        pointer: UnsafeMutablePointer<st_owned_buffer_t>?,
        count: Int,
        field: String
    ) throws -> [String] {
        try array(pointer: pointer, count: count, field: field).map(string(from:))
    }

    static func optionalString(
        buffer: st_owned_buffer_t,
        present: Bool
    ) -> String? {
        present ? string(from: buffer) : nil
    }

    static func namedChatTemplateArray(
        pointer: UnsafeMutablePointer<st_named_chat_template_t>?,
        count: Int,
        field: String
    ) throws -> [TokenizerRuntimeConfiguration.NamedChatTemplate] {
        try array(pointer: pointer, count: count, field: field).map { template in
            TokenizerRuntimeConfiguration.NamedChatTemplate(
                name: string(from: template.name),
                template: string(from: template.template_text)
            )
        }
    }

    static func chatTemplateSource(from configuration: st_runtime_configuration_t) throws
        -> TokenizerRuntimeConfiguration.ChatTemplateSource
    {
        guard let kind = RustChatTemplateKind(rawValue: configuration.chat_template_kind) else {
            throw TokenizerError.invalidConfiguration(
                "Unsupported Rust chat template kind: \(configuration.chat_template_kind)"
            )
        }
        switch kind {
        case .none:
            return .none
        case .literal:
            return .literal(string(from: configuration.chat_template_literal))
        case .named:
            return .named(
                try namedChatTemplateArray(
                    pointer: configuration.named_chat_templates,
                    count: Int(configuration.named_chat_templates_len),
                    field: "named_chat_templates"
                )
            )
        }
    }

    static func runtimeConfiguration(from configuration: st_runtime_configuration_t) throws
        -> TokenizerRuntimeConfiguration
    {
        let modelMaxLength: Int?
        if configuration.has_model_max_length {
            guard configuration.model_max_length <= UInt64(Int.max) else {
                throw TokenizerError.invalidConfiguration(
                    "model_max_length exceeds Swift Int.max"
                )
            }
            modelMaxLength = Int(configuration.model_max_length)
        } else {
            modelMaxLength = nil
        }

        return try TokenizerRuntimeConfiguration(
            bosToken: optionalString(
                buffer: configuration.bos_token,
                present: configuration.has_bos_token
            ),
            eosToken: optionalString(
                buffer: configuration.eos_token,
                present: configuration.has_eos_token
            ),
            unknownToken: optionalString(
                buffer: configuration.unknown_token,
                present: configuration.has_unknown_token
            ),
            sepToken: optionalString(
                buffer: configuration.sep_token,
                present: configuration.has_sep_token
            ),
            padToken: optionalString(
                buffer: configuration.pad_token,
                present: configuration.has_pad_token
            ),
            clsToken: optionalString(
                buffer: configuration.cls_token,
                present: configuration.has_cls_token
            ),
            maskToken: optionalString(
                buffer: configuration.mask_token,
                present: configuration.has_mask_token
            ),
            additionalSpecialTokens: stringArray(
                pointer: configuration.additional_special_tokens,
                count: Int(configuration.additional_special_tokens_len),
                field: "additional_special_tokens"
            ),
            cleanUpTokenizationSpaces: configuration.clean_up_tokenization_spaces,
            modelMaxLength: modelMaxLength,
            chatTemplate: chatTemplateSource(from: configuration)
        )
    }

    static func tokenizerDescriptor(from descriptor: st_tokenizer_descriptor_t) throws -> RustTokenizerDescriptor {
        RustTokenizerDescriptor(
            runtimeConfiguration: try runtimeConfiguration(from: descriptor.runtime_configuration),
            bosTokenId: descriptor.has_bos_token_id ? Int(descriptor.bos_token_id) : nil,
            eosTokenId: descriptor.has_eos_token_id ? Int(descriptor.eos_token_id) : nil,
            unknownTokenId: descriptor.has_unknown_token_id ? Int(descriptor.unknown_token_id) : nil,
            baseVocabSize: Int(descriptor.base_vocab_size),
            totalVocabSize: Int(descriptor.total_vocab_size)
        )
    }

    static func offsetSpanArray(
        pointer: UnsafeMutablePointer<st_encoding_offset_span_t>?,
        count: Int,
        field: String
    ) throws -> [OffsetSpan] {
        try array(pointer: pointer, count: count, field: field).map { span in
            OffsetSpan(start: Int(span.start), end: Int(span.end))
        }
    }

    static func encodingArray(
        pointer: UnsafeMutablePointer<st_encoding_t>?,
        count: Int,
        field: String
    ) throws -> [TokenizerEncoding] {
        try array(pointer: pointer, count: count, field: field).map { encoding in
            try Self.encoding(from: encoding)
        }
    }

    static func encoding(from encoding: st_encoding_t) throws -> TokenizerEncoding {
        let tokenIds = try intArray(
            pointer: encoding.token_ids,
            count: Int(encoding.token_ids_len),
            field: "token_ids"
        )
        let tokenTypeIds = try intArray(
            pointer: encoding.token_type_ids,
            count: Int(encoding.token_type_ids_len),
            field: "token_type_ids"
        )
        let tokens = try stringArray(
            pointer: encoding.tokens,
            count: Int(encoding.tokens_len),
            field: "tokens"
        )
        let wordIndices = try optionalIntArray(
            values: encoding.word_indices,
            present: encoding.word_indices_present,
            count: Int(encoding.word_indices_len),
            field: "word_indices"
        )
        let offsetSpans = try offsetSpanArray(
            pointer: encoding.offset_spans,
            count: Int(encoding.offset_spans_len),
            field: "offset_spans"
        )
        let specialTokensMask = try intArray(
            pointer: encoding.special_token_mask,
            count: Int(encoding.special_token_mask_len),
            field: "special_token_mask"
        )
        let attentionMask = try intArray(
            pointer: encoding.attention_mask,
            count: Int(encoding.attention_mask_len),
            field: "attention_mask"
        )
        let sequenceIndices = try optionalIntArray(
            values: encoding.sequence_indices,
            present: encoding.sequence_indices_present,
            count: Int(encoding.sequence_indices_len),
            field: "sequence_indices"
        )
        let overflowEncodings = try encodingArray(
            pointer: encoding.overflow_encodings,
            count: Int(encoding.overflow_encodings_len),
            field: "overflow_encodings"
        )
        let offsetUnit = try OffsetUnit.fromRustRawValue(encoding.offset_unit)

        return TokenizerEncoding(
            tokenIds: tokenIds,
            tokenTypeIds: tokenTypeIds,
            tokens: tokens,
            wordIndices: wordIndices,
            offsetSpans: offsetSpans,
            specialTokensMask: specialTokensMask,
            attentionMask: attentionMask,
            sequenceIndices: sequenceIndices,
            sequenceCount: Int(encoding.sequence_count),
            overflowEncodings: overflowEncodings,
            offsetUnit: offsetUnit
        )
    }

    static func tokenizerError(from error: st_error_t) -> TokenizerError {
        defer { st_free_owned_buffer(error.message) }
        let message = string(from: error.message)
        switch error.code {
        case RustErrorCode.missingConfig:
            return .missingConfig
        case RustErrorCode.chatTemplate:
            return .chatTemplate(message)
        case RustErrorCode.mismatchedConfig:
            return .invalidConfiguration(message)
        case RustErrorCode.internal:
            return .internalError(message)
        default:
            let fallback = message.isEmpty ? "Rust tokenizer error code \(error.code)" : message
            return .internalError(fallback)
        }
    }
}

// Mirrors the integer codes returned by `CoreError::code` on the Rust side
// (rust/src/core/error.rs). Keep these in sync with the Rust enum.
private enum RustErrorCode {
    static let missingConfig: Int32 = 1
    static let chatTemplate: Int32 = 6
    static let mismatchedConfig: Int32 = 9
    static let `internal`: Int32 = 100
}

// Mirrors `FfiChatTemplateKind` in `rust/src/core/abi/config.rs`. Keep raw
// values in sync with the Rust enum.
private enum RustChatTemplateKind: UInt8 {
    case none = 0
    case literal = 1
    case named = 2
}

/// Owns the byte buffers backing every string in `ffi`. The instance must outlive
/// any Rust call that reads `ffi` or any field within it; the borrowed pointers
/// inside `ffi` become invalid when this deinits, so do not stash `ffi` separately
/// or pass it to an async Rust call that may outlive the parent scope.
private final class RustRuntimeConfigurationInput {
    private let bytePointers: [UnsafeMutablePointer<UInt8>]
    private let additionalSpecialTokensPointer: UnsafeMutablePointer<st_owned_buffer_t>?
    private let additionalSpecialTokensCount: Int
    private let namedChatTemplatesPointer: UnsafeMutablePointer<st_named_chat_template_t>?
    private let namedChatTemplatesCount: Int

    let ffi: st_runtime_configuration_t

    init(_ configuration: TokenizerRuntimeConfiguration) throws {
        var bytePointers: [UnsafeMutablePointer<UInt8>] = []

        func makeBuffer(_ string: String) -> st_owned_buffer_t {
            let bytes = Array(string.utf8)
            guard !bytes.isEmpty else {
                return RustFFI.emptyBuffer()
            }

            let pointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
            pointer.initialize(from: bytes, count: bytes.count)
            bytePointers.append(pointer)
            return st_owned_buffer_t(data: pointer, len: bytes.count)
        }

        func makeOptionalBuffer(_ string: String?) -> (st_owned_buffer_t, Bool) {
            guard let string else {
                return (RustFFI.emptyBuffer(), false)
            }
            return (makeBuffer(string), true)
        }

        func makeBufferArray(_ strings: [String]) -> (UnsafeMutablePointer<st_owned_buffer_t>?, Int) {
            let buffers = strings.map(makeBuffer)
            guard !buffers.isEmpty else {
                return (nil, 0)
            }

            let pointer = UnsafeMutablePointer<st_owned_buffer_t>.allocate(capacity: buffers.count)
            pointer.initialize(from: buffers, count: buffers.count)
            return (pointer, buffers.count)
        }

        let (bosToken, hasBosToken) = makeOptionalBuffer(configuration.bosToken)
        let (eosToken, hasEosToken) = makeOptionalBuffer(configuration.eosToken)
        let (unknownToken, hasUnknownToken) = makeOptionalBuffer(configuration.unknownToken)
        let (sepToken, hasSepToken) = makeOptionalBuffer(configuration.sepToken)
        let (padToken, hasPadToken) = makeOptionalBuffer(configuration.padToken)
        let (clsToken, hasClsToken) = makeOptionalBuffer(configuration.clsToken)
        let (maskToken, hasMaskToken) = makeOptionalBuffer(configuration.maskToken)
        let (additionalSpecialTokens, additionalSpecialTokensCount) = makeBufferArray(
            configuration.additionalSpecialTokens
        )

        let modelMaxLength: UInt64
        let hasModelMaxLength: Bool
        if let maxLength = configuration.modelMaxLength {
            guard maxLength >= 0 else {
                throw TokenizerError.invalidConfiguration("modelMaxLength must be non-negative")
            }
            modelMaxLength = UInt64(maxLength)
            hasModelMaxLength = true
        } else {
            modelMaxLength = 0
            hasModelMaxLength = false
        }

        let chatTemplateKind: UInt8
        let chatTemplateLiteral: st_owned_buffer_t
        let namedChatTemplates: UnsafeMutablePointer<st_named_chat_template_t>?
        let namedChatTemplatesCount: Int

        switch configuration.chatTemplate {
        case .none:
            chatTemplateKind = RustChatTemplateKind.none.rawValue
            chatTemplateLiteral = RustFFI.emptyBuffer()
            namedChatTemplates = nil
            namedChatTemplatesCount = 0
        case .literal(let template):
            chatTemplateKind = RustChatTemplateKind.literal.rawValue
            chatTemplateLiteral = makeBuffer(template)
            namedChatTemplates = nil
            namedChatTemplatesCount = 0
        case .named(let templates):
            chatTemplateKind = RustChatTemplateKind.named.rawValue
            chatTemplateLiteral = RustFFI.emptyBuffer()
            namedChatTemplatesCount = templates.count
            if templates.isEmpty {
                namedChatTemplates = nil
            } else {
                let ffiTemplates = templates.map { template in
                    st_named_chat_template_t(
                        name: makeBuffer(template.name),
                        template_text: makeBuffer(template.template)
                    )
                }
                let pointer = UnsafeMutablePointer<st_named_chat_template_t>.allocate(
                    capacity: ffiTemplates.count
                )
                pointer.initialize(from: ffiTemplates, count: ffiTemplates.count)
                namedChatTemplates = pointer
            }
        }

        self.bytePointers = bytePointers
        self.additionalSpecialTokensPointer = additionalSpecialTokens
        self.additionalSpecialTokensCount = additionalSpecialTokensCount
        self.namedChatTemplatesPointer = namedChatTemplates
        self.namedChatTemplatesCount = namedChatTemplatesCount
        self.ffi = st_runtime_configuration_t(
            bos_token: bosToken,
            has_bos_token: hasBosToken,
            eos_token: eosToken,
            has_eos_token: hasEosToken,
            unknown_token: unknownToken,
            has_unknown_token: hasUnknownToken,
            sep_token: sepToken,
            has_sep_token: hasSepToken,
            pad_token: padToken,
            has_pad_token: hasPadToken,
            cls_token: clsToken,
            has_cls_token: hasClsToken,
            mask_token: maskToken,
            has_mask_token: hasMaskToken,
            additional_special_tokens: additionalSpecialTokens,
            additional_special_tokens_len: additionalSpecialTokensCount,
            clean_up_tokenization_spaces: configuration.cleanUpTokenizationSpaces,
            model_max_length: modelMaxLength,
            has_model_max_length: hasModelMaxLength,
            chat_template_kind: chatTemplateKind,
            chat_template_literal: chatTemplateLiteral,
            named_chat_templates: namedChatTemplates,
            named_chat_templates_len: namedChatTemplatesCount
        )
    }

    deinit {
        additionalSpecialTokensPointer?.deinitialize(count: additionalSpecialTokensCount)
        additionalSpecialTokensPointer?.deallocate()
        namedChatTemplatesPointer?.deinitialize(count: namedChatTemplatesCount)
        namedChatTemplatesPointer?.deallocate()
        for pointer in bytePointers {
            pointer.deallocate()
        }
    }
}

// Raw values must match Rust `EncodingOffsetUnit` in
// `rust/src/core/tokenizer_core.rs`.
private extension OffsetUnit {
    var rustRawValue: UInt8 {
        switch self {
        case .utf8:
            0
        case .unicodeScalar:
            1
        }
    }

    static func fromRustRawValue(_ rawValue: UInt8) throws -> OffsetUnit {
        switch rawValue {
        case 0:
            .utf8
        case 1:
            .unicodeScalar
        default:
            throw TokenizerError.invalidConfiguration("Unsupported Rust offset unit: \(rawValue)")
        }
    }
}

/// Owns the Rust tokenizer handle and exposes both per-token operations
/// (`tokenize`, `convertTokenToId`, …) and text-level operations (`encode`,
/// `encodeWithMetadata`, `decode`, `applyChatTemplate`, …).
///
/// `@unchecked Sendable` is sound because the underlying `tokenizers::Tokenizer`
/// is `Send + Sync`, every FFI entry point reads through the handle without
/// mutating shared state, and every Swift-side stored property is `let`.
package final class RustTokenizer: @unchecked Sendable {
    private let handle: OpaquePointer

    package let bosToken: String?
    package let eosToken: String?
    package let unknownToken: String?
    package let bosTokenId: Int?
    package let eosTokenId: Int?
    package let unknownTokenId: Int?
    package let baseVocabSize: Int
    package let totalVocabSize: Int

    fileprivate init(handle: OpaquePointer, descriptor: RustTokenizerDescriptor) {
        self.handle = handle
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

    deinit {
        st_tokenizer_destroy(handle)
    }

    package func tokenize(text: String) -> [String] {
        do {
            return try RustFFICalls.tokenize(handle: handle, text: text)
        } catch {
            assertionFailure("RustFFICalls.tokenize failed: \(error)")
            return []
        }
    }

    package func encode(text: String, textPair: String?, addSpecialTokens: Bool) -> [Int] {
        do {
            return try RustFFICalls.encode(
                handle: handle,
                text: text,
                textPair: textPair,
                addSpecialTokens: addSpecialTokens
            )
        } catch {
            assertionFailure("RustFFICalls.encode failed: \(error)")
            return []
        }
    }

    package func encodeBatch(
        _ inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool
    ) -> [[Int]] {
        do {
            return try RustFFICalls.encodeBatch(
                handle: handle,
                inputs: inputs,
                addSpecialTokens: addSpecialTokens
            )
        } catch {
            assertionFailure("RustFFICalls.encodeBatch failed: \(error)")
            return []
        }
    }

    package func encodeWithMetadata(
        text: String,
        textPair: String?,
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> TokenizerEncoding {
        try RustFFICalls.encodeWithMetadata(
            handle: handle,
            text: text,
            textPair: textPair,
            addSpecialTokens: addSpecialTokens,
            offsetUnit: offsetUnit
        )
    }

    package func encodeBatchWithMetadata(
        _ inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> [TokenizerEncoding] {
        try RustFFICalls.encodeBatchWithMetadata(
            handle: handle,
            inputs: inputs,
            addSpecialTokens: addSpecialTokens,
            offsetUnit: offsetUnit
        )
    }

    package func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        do {
            return try RustFFICalls.decode(handle: handle, tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
        } catch {
            assertionFailure("RustFFICalls.decode failed: \(error)")
            return ""
        }
    }

    package func convertTokenToId(_ token: String) -> Int? {
        do {
            return try RustFFICalls.convertTokenToId(handle: handle, token: token)
        } catch {
            assertionFailure("RustFFICalls.convertTokenToId failed: \(error)")
            return nil
        }
    }

    package func convertIdToToken(_ id: Int) -> String? {
        do {
            return try RustFFICalls.convertIdToToken(handle: handle, id: id)
        } catch {
            assertionFailure("RustFFICalls.convertIdToToken failed: \(error)")
            return nil
        }
    }

    package func renderChatTemplate(template: String, contextObject: [String: Any]) throws -> String {
        let contextJSON = try JSONBridge.jsonString(from: contextObject)
        return try RustFFICalls.renderTemplate(template: template, contextJSON: contextJSON)
    }

    package func applyChatTemplate(
        template: String,
        contextObject: [String: Any],
        truncation: Bool,
        maxLength: Int?
    ) throws -> [Int] {
        let contextJSON = try JSONBridge.jsonString(from: contextObject)
        return try RustFFICalls.applyChatTemplate(
            handle: handle,
            template: template,
            contextJSON: contextJSON,
            truncation: truncation,
            maxLength: maxLength
        )
    }
}

private enum RustFFICalls {
    static func tokenize(handle: OpaquePointer, text: String) throws -> [String] {
        var tokens: UnsafeMutablePointer<st_owned_buffer_t>?
        var count = 0
        var error = RustFFI.emptyError()
        let success = text.withCString { textPointer in
            st_tokenizer_tokenize(handle, textPointer, &tokens, &count, &error)
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        defer { st_free_owned_buffer_array(tokens, count) }
        return try RustFFI.stringArray(pointer: tokens, count: count, field: "tokens")
    }

    static func encode(
        handle: OpaquePointer,
        text: String,
        textPair: String?,
        addSpecialTokens: Bool
    ) throws -> [Int] {
        var tokenIDs: UnsafeMutablePointer<Int32>?
        var count: Int = 0
        var error = RustFFI.emptyError()
        let success = withOptionalCString(textPair) { textPairPointer in
            text.withCString { textPointer in
                st_tokenizer_encode(handle, textPointer, textPairPointer, addSpecialTokens, &tokenIDs, &count, &error)
            }
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        return RustFFI.takeIntArray(pointer: tokenIDs, count: count)
    }

    static func encodeWithMetadata(
        handle: OpaquePointer,
        text: String,
        textPair: String?,
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> TokenizerEncoding {
        var encoding = RustFFI.emptyEncoding()
        var error = RustFFI.emptyError()
        let success = withOptionalCString(textPair) { textPairPointer in
            text.withCString { textPointer in
                st_tokenizer_encode_with_metadata(
                    handle,
                    textPointer,
                    textPairPointer,
                    addSpecialTokens,
                    offsetUnit.rustRawValue,
                    &encoding,
                    &error
                )
            }
        }
        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        defer { st_free_encoding(encoding) }
        return try RustFFI.encoding(from: encoding)
    }

    static func encodeBatch(
        handle: OpaquePointer,
        inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool
    ) throws -> [[Int]] {
        var arrays: UnsafeMutablePointer<st_owned_int32_array_t>?
        var arraysLen: Int = 0
        var error = RustFFI.emptyError()

        let success = withEncodeInputs(inputs) { ffiInputs in
            ffiInputs.withUnsafeBufferPointer { buffer in
                st_tokenizer_encode_batch(
                    handle,
                    buffer.baseAddress,
                    buffer.count,
                    addSpecialTokens,
                    &arrays,
                    &arraysLen,
                    &error
                )
            }
        }

        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        defer { st_free_owned_int32_array_array(arrays, arraysLen) }

        guard arraysLen > 0, let arrays else { return [] }
        let bufferPointer = UnsafeBufferPointer(start: arrays, count: arraysLen)
        return bufferPointer.map { array -> [Int] in
            guard array.len > 0, let data = array.data else { return [] }
            let ids = UnsafeBufferPointer(start: data, count: array.len)
            return ids.map(Int.init)
        }
    }

    static func encodeBatchWithMetadata(
        handle: OpaquePointer,
        inputs: [(text: String, textPair: String?)],
        addSpecialTokens: Bool,
        offsetUnit: OffsetUnit
    ) throws -> [TokenizerEncoding] {
        var encodings: UnsafeMutablePointer<st_encoding_t>?
        var encodingsLen: Int = 0
        var error = RustFFI.emptyError()

        let success = withEncodeInputs(inputs) { ffiInputs in
            ffiInputs.withUnsafeBufferPointer { buffer in
                st_tokenizer_encode_batch_with_metadata(
                    handle,
                    buffer.baseAddress,
                    buffer.count,
                    addSpecialTokens,
                    offsetUnit.rustRawValue,
                    &encodings,
                    &encodingsLen,
                    &error
                )
            }
        }

        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        defer { st_free_encoding_array(encodings, encodingsLen) }

        guard encodingsLen > 0, let encodings else { return [] }
        let bufferPointer = UnsafeBufferPointer(start: encodings, count: encodingsLen)
        return try bufferPointer.map { try RustFFI.encoding(from: $0) }
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

    static func applyChatTemplate(
        handle: OpaquePointer,
        template: String,
        contextJSON: String,
        truncation: Bool,
        maxLength: Int?
    ) throws -> [Int] {
        var tokenIDs: UnsafeMutablePointer<Int32>?
        var count: Int = 0
        var error = RustFFI.emptyError()

        let ffiMaxLength: UInt64
        if let maxLength {
            guard maxLength >= 0 else {
                throw TokenizerError.invalidConfiguration("maxLength must be non-negative")
            }
            ffiMaxLength = UInt64(maxLength)
        } else {
            ffiMaxLength = 0
        }

        let success = template.withCString { templatePointer in
            contextJSON.withCString { contextPointer in
                st_tokenizer_apply_chat_template(
                    handle,
                    templatePointer,
                    contextPointer,
                    truncation,
                    maxLength != nil,
                    ffiMaxLength,
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
    private static func makeTokenizer(
        handle: OpaquePointer,
        descriptor: RustTokenizerDescriptor
    ) -> any Tokenizer {
        let rustTokenizer = RustTokenizer(handle: handle, descriptor: descriptor)
        return PreTrainedTokenizer(
            rustTokenizer: rustTokenizer,
            runtimeConfiguration: descriptor.runtimeConfiguration
        )
    }

    package static func loadRuntimeConfiguration(from directory: URL) throws -> TokenizerRuntimeConfiguration {
        var configuration = RustFFI.emptyRuntimeConfiguration()
        var error = RustFFI.emptyError()

        let success = directory.path.withCString { directoryPath in
            st_load_tokenizer_runtime_configuration(directoryPath, &configuration, &error)
        }

        guard success else {
            throw RustFFI.tokenizerError(from: error)
        }
        defer { st_free_runtime_configuration(configuration) }
        return try RustFFI.runtimeConfiguration(from: configuration)
    }

    package static func loadTokenizerCore(
        from directory: URL,
        runtimeConfiguration: TokenizerRuntimeConfiguration
    ) async throws -> any Tokenizer {
        let runtimeConfigurationInput = try RustRuntimeConfigurationInput(runtimeConfiguration)

        var handle: OpaquePointer?
        var descriptor = RustFFI.emptyTokenizerDescriptor()
        var error = RustFFI.emptyError()

        let success = withUnsafePointer(to: runtimeConfigurationInput.ffi) { runtimeConfigurationPointer in
            directory.path.withCString { directoryPath in
                st_tokenizer_create_with_runtime_configuration(
                    directoryPath,
                    runtimeConfigurationPointer,
                    &handle,
                    &descriptor,
                    &error
                )
            }
        }
        defer { st_free_tokenizer_descriptor(descriptor) }

        guard success, let handle else {
            throw RustFFI.tokenizerError(from: error)
        }

        do {
            return try makeTokenizer(
                handle: handle,
                descriptor: RustFFI.tokenizerDescriptor(from: descriptor)
            )
        } catch {
            st_tokenizer_destroy(handle)
            throw error
        }
    }

    package static func load(from directory: URL) async throws -> any Tokenizer {
        var handle: OpaquePointer?
        var descriptor = RustFFI.emptyTokenizerDescriptor()
        var error = RustFFI.emptyError()

        let success = directory.path.withCString { directoryPath in
            st_tokenizer_create_from_directory(directoryPath, &handle, &descriptor, &error)
        }
        defer { st_free_tokenizer_descriptor(descriptor) }

        guard success, let handle else {
            throw RustFFI.tokenizerError(from: error)
        }

        do {
            return try makeTokenizer(
                handle: handle,
                descriptor: RustFFI.tokenizerDescriptor(from: descriptor)
            )
        } catch {
            st_tokenizer_destroy(handle)
            throw error
        }
    }
}
