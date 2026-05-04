import Foundation

public extension AutoTokenizer {
    /// Loads a tokenizer from a local directory containing tokenizer configuration files.
    ///
    /// The directory must contain `tokenizer.json`. `tokenizer_config.json` is optional. If a
    /// `chat_template.jinja` file is present its contents are used as the chat template; otherwise
    /// a `chat_template.json` file's `chat_template` field is used. Either sidecar takes
    /// precedence over the `chat_template` field in `tokenizer_config.json`.
    ///
    /// Algorithm dispatch is handled by the Rust tokenizers backend.
    ///
    /// - Parameter directory: Path to a local directory containing tokenizer files.
    /// - Returns: A configured ``Tokenizer`` instance.
    /// - Throws: ``TokenizerError`` if required files are missing or configuration is invalid.
    static func from(directory: URL) async throws -> Tokenizer {
        return try await RustAutoTokenizerDirectoryLoader.load(from: directory)
    }
}
