import Foundation

#if TOKENIZERS_SWIFT_BACKEND
import TokenizersSwiftBackend
#endif

#if Rust
import TokenizersRustBackend
#endif

public extension AutoTokenizer {
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
        #if Rust
        return try await RustAutoTokenizerDirectoryLoader.load(from: directory, strict: strict)
        #elseif TOKENIZERS_SWIFT_BACKEND
        return try await SwiftAutoTokenizerDirectoryLoader.load(from: directory, strict: strict)
        #else
        fatalError("No tokenizer backend is enabled")
        #endif
    }
}
