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
    /// Algorithm dispatch is driven by `tokenizer.json`'s `model.type`
    /// (`BPE` / `WordPiece` / `Unigram` / `WordLevel`). An unrecognized or missing
    /// value throws `TokenizerError.unsupportedModelType`.
    ///
    /// - Parameter directory: Path to a local directory containing tokenizer files
    /// - Returns: A configured `Tokenizer` instance
    /// - Throws: `TokenizerError` if required files are missing or configuration is invalid
    static func from(directory: URL) async throws -> Tokenizer {
        #if Rust
        return try await RustAutoTokenizerDirectoryLoader.load(from: directory)
        #elseif TOKENIZERS_SWIFT_BACKEND
        return try await SwiftAutoTokenizerDirectoryLoader.load(from: directory)
        #else
        fatalError("No tokenizer backend is enabled")
        #endif
    }
}
