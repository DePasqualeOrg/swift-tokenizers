# Rust Load Optimization Notes

## Purpose

These notes capture the Rust load-path changes that materially improved tokenizer creation time.

## Implemented Changes

### Original-bytes fast path

The Rust loader now tries upstream `Tokenizer::from_bytes` on the original `tokenizer.json` bytes first.

It only falls back to parsing into `serde_json::Value`, normalizing `added_tokens`, and retrying if the original bytes fail.

This removed an unconditional extra parse-and-reserialize step from the common path.

### Remove eager token-id map

The Rust loader no longer builds a full token-to-id map during tokenizer creation.

Instead, it uses upstream `token_to_id` and `id_to_token` for conversions and only resolves the small set of special token ids needed up front.

This moved non-essential work off the load path and reused the tokenizer's own internal lookup structures.

### Lazy `vocabCount`

Rust `vocabCount` is intentionally lazy.

- It is not part of tokenizer creation metadata.
- It is resolved on demand from the Rust handle.
- It is cached on the Swift side after first access.

This follows the Python fast-tokenizer direction more closely than the current Swift implementation and keeps introspection work off the Rust load path.

## Current Benchmark State

Warm release benchmark results after the optimization pass:

- Swift sidecar load: `0.3ms`
- Rust sidecar load: `0.1ms` median
- Swift tokenizer core load: `210.2ms`
- Rust tokenizer core load: `193.4ms`
- Swift full tokenizer load: `201.7ms`
- Rust full tokenizer load: `190.4ms`
- Swift tokenize: `26.2ms`
- Rust tokenize: `4.0ms`
- Swift decode: `16.5ms`
- Rust decode: `4.6ms`

On warm runs, the Rust backend is now ahead on load, tokenization, and decode.

## Remaining Notes

- Further Rust load optimization should be profiling-driven, not speculative.
- MiniJinja environment setup is a possible later optimization, but current template-render timings do not make it a priority.
