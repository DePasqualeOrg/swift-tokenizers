# Rust Backend Design Notes

## Purpose

`swift-tokenizers` supports two backend modes behind the same Swift API:

- default `Swift` trait: pure Swift backend
- opt-in `Rust` trait: Rust-backed backend

The goal of the Rust backend is to improve performance while keeping Rust fully out of the default build.

## Current Design

- One public product: `Tokenizers`
- Build-time backend selection only
- Default trait set enables `Swift`
- `Rust` is opt-in and mutually exclusive with `Swift` in code
- Swift remains the public API layer
- Rust mode uses one Rust core artifact behind a coarse C ABI

## Rust Backend Responsibilities

In `Rust` builds, the Rust core owns:

- `tokenizer.json` loading via `huggingface/tokenizers`
- sidecar loading for `tokenizer_config.json`, `config.json`, and chat template files
- tokenization, encoding, decoding, and token/id conversion
- chat template rendering via `minijinja`

Swift owns:

- the public `Tokenizer` and `AutoTokenizer` API
- shared runtime configuration types
- error mapping into the Swift-facing error model

## Upstream Components

- `huggingface/tokenizers`: core tokenizer engine
- `minijinja`: chat templating in Rust mode
- `serde` / `serde_json`: internal Rust decoding only

Package-specific Rust should stay minimal and focus on ABI glue and package-shaped orchestration.

## Constraints

- No runtime backend selection
- No mixed public Swift/Rust mode
- No tokenizer training APIs
- `AutoTokenizer.register(...)` remains a Swift-path escape hatch, not a Rust registry model
- Rust mode should stay lean:
  - one Rust core artifact
  - minimal Cargo features
  - binary size tracked and justified

## Packaging

- Swift package traits require Swift 6.1+
- Official Rust distribution is a prebuilt Apple artifact consumed through SwiftPM
- Rust should not require consumers to install or manage a Rust toolchain

## Verification

Keep both backend modes validated by:

- parity tests on representative tokenizer fixtures
- end-to-end benchmarks for load, tokenize, and decode
- subsystem benchmarks for load-path work where needed
- binary size checks for the Rust artifact

See [rust-load-optimization-plan.md](/Users/anthony/files/projects/forked/swift-tokenizers/docs/rust-load-optimization-plan.md) for the load-performance optimization history.
