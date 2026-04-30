#if TOKENIZERS_DOCS_BUILD
// swift-docc-plugin builds the facade with both traits enabled for symbol-graph
// coverage. Skip the mutex guard in that case; it still applies to every other build.
#elseif Rust && Swift
#error("Swift and Rust tokenizer backends are mutually exclusive. Enable only one package trait.")
#elseif !Rust && !Swift
#error("No tokenizer backend selected. Enable either the default Swift trait or the Rust trait.")
#endif
