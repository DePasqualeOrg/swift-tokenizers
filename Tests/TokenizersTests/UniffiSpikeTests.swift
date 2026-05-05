// Copyright © Anthony DePasquale

import Foundation
import Testing
import TokenizersFFI

/// Spike-level smoke tests that exercise the UniFFI-generated wrapper directly,
/// bypassing the public `Tokenizers` API. They validate the bindgen pipeline
/// end to end: bindings metadata → static lib → generated `TokenizersFFI`
/// wrapper → call into Rust.
@Suite("UniFFI spike")
struct UniffiSpikeTests {
    private static func fixtureDirectory() -> URL {
        let resourceURL = Bundle.module.url(forResource: "tokenizer", withExtension: "json")!
        return resourceURL.deletingLastPathComponent()
    }

    @Test("Tokenizer.fromDirectory + tokenize round-trips through UniFFI")
    func tokenizeViaUniffi() throws {
        let tokenizer = try TokenizersFFI.Tokenizer.fromDirectory(
            directoryPath: Self.fixtureDirectory().path
        )
        let tokens = try tokenizer.tokenize(text: "hello world")
        #expect(!tokens.isEmpty)
    }
}
