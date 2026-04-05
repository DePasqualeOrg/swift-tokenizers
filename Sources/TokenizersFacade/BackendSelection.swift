#if Rust && TOKENIZERS_SWIFT_BACKEND
#error("Swift and Rust tokenizer backends are mutually exclusive. Enable only one package trait.")
#elseif !Rust && !TOKENIZERS_SWIFT_BACKEND
#error("No tokenizer backend selected. Enable either the default Swift trait or the Rust trait.")
#endif
