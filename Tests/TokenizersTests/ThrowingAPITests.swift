// Copyright © Anthony DePasquale

import Foundation
import HFAPI
import Testing

@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-throwing-api-tests")
}()

private let hubClient = HFClient.default
private let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json", "config.json"]

private func downloadModel(_ name: String) async throws -> URL {
    guard let repoId = RepositoryID(name) else {
        struct InvalidRepoID: Error { let name: String }
        throw InvalidRepoID(name: name)
    }
    return try await hubClient.model(repoId).snapshotDownload(
        allowPatterns: tokenizerFiles,
        localDir: downloadDestination.appending(path: name)
    )
}

private func makeTokenizer(_ name: String) async throws -> PreTrainedTokenizer {
    let directory = try await downloadModel(name)
    let tokenizer = try await AutoTokenizer.from(directory: directory)
    struct UnexpectedType: Error { let type: String }
    guard let pretrained = tokenizer as? PreTrainedTokenizer else {
        throw UnexpectedType(type: "\(type(of: tokenizer))")
    }
    return pretrained
}

@Suite("Throwing decode error paths", .serialized)
struct DecodeErrorPathTests {
    @Test
    func negativeTokenIdThrows() async throws {
        let tokenizer = try await makeTokenizer("openai/whisper-tiny.en")
        do {
            _ = try tokenizer.decode(tokenIds: [-1], skipSpecialTokens: false)
            Issue.record("Expected .invalidTokenId for negative id")
        } catch {
            guard case .invalidTokenId(-1) = error else {
                Issue.record("Expected .invalidTokenId(-1), got \(error)")
                return
            }
        }
    }

    @Test
    func overflowTokenIdThrows() async throws {
        let tokenizer = try await makeTokenizer("openai/whisper-tiny.en")
        do {
            _ = try tokenizer.decode(tokenIds: [Int.max], skipSpecialTokens: false)
            Issue.record("Expected .invalidTokenId for overflow id")
        } catch {
            guard case .invalidTokenId(Int.max) = error else {
                Issue.record("Expected .invalidTokenId(Int.max), got \(error)")
                return
            }
        }
    }
}

@Suite("StreamingDetokenizer", .serialized)
struct StreamingDetokenizerTests {
    @Test
    func emitsAsciiPerToken() async throws {
        let tokenizer = try await makeTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        let ids = try tokenizer.encode(text: "Hello world", addSpecialTokens: false)
        let stream = tokenizer.streamingDetokenizer()

        var collected = ""
        for id in ids {
            if let chunk = try stream.consume(id) {
                #expect(!chunk.isEmpty)
                collected.append(chunk)
            }
        }

        let oneShot = try tokenizer.rawDecode(tokenIds: ids, skipSpecialTokens: false)
        #expect(collected == oneShot)
    }

    @Test
    func multiByteScalarIsBufferedAcrossTokens() async throws {
        // distilgpt2 is byte-level BPE without dedicated emoji tokens, so
        // multi-byte scalars are reliably fragmented into multiple sub-byte
        // tokens — the canonical "wait for more bytes" case.
        let tokenizer = try await makeTokenizer("distilbert/distilgpt2")
        // U+1F30D 🌍 is encoded as 4 UTF-8 bytes that distilgpt2 splits into
        // multiple tokens (see `multiScriptExpected` in BaselineTests).
        let ids = try tokenizer.encode(text: "🌍", addSpecialTokens: false)
        try #require(ids.count >= 2, "Expected emoji to fragment across tokens for byte-level BPE")

        let stream = tokenizer.streamingDetokenizer()
        var nilCount = 0
        var combined = ""
        for id in ids {
            if let chunk = try stream.consume(id) {
                combined.append(chunk)
            } else {
                nilCount += 1
            }
        }

        #expect(nilCount >= 1, "Expected at least one nil emission while bytes accumulate")

        let oneShot = try tokenizer.rawDecode(tokenIds: ids, skipSpecialTokens: false)
        #expect(combined == oneShot)
    }

    @Test
    func bufferStaysBoundedAcrossLongStream() async throws {
        let tokenizer = try await makeTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        // A long stream without natural reset points (no newlines).
        let text = String(repeating: "lorem ipsum dolor sit amet ", count: 200)
        let ids = try tokenizer.encode(text: text, addSpecialTokens: false)
        let stream = tokenizer.streamingDetokenizer()

        var combined = ""
        for id in ids {
            if let chunk = try stream.consume(id) {
                combined.append(chunk)
            }
        }
        let oneShot = try tokenizer.rawDecode(tokenIds: ids, skipSpecialTokens: false)
        #expect(combined == oneShot)

        // The internal id buffer should not have grown unboundedly. After every
        // successful emission the algorithm trims to ids since the last emission,
        // so for a stream of ~thousands of ASCII tokens the buffer should be small.
        let mirror = Mirror(reflecting: stream)
        let storedIds = mirror.children.first { $0.label == "ids" }?.value as? [Int]
        #expect(storedIds != nil)
        if let storedIds {
            #expect(storedIds.count < 50, "Streaming buffer grew to \(storedIds.count) ids — algorithm should keep it bounded")
        }
    }

    @Test
    func seedingFromInitialIds() async throws {
        let tokenizer = try await makeTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        let prefix = try tokenizer.encode(text: "Hello", addSpecialTokens: false)
        let suffix = try tokenizer.encode(text: " world", addSpecialTokens: false)

        let stream = tokenizer.streamingDetokenizer(initialTokenIds: prefix)
        var collected = ""
        for id in suffix {
            if let chunk = try stream.consume(id) {
                collected.append(chunk)
            }
        }

        let oneShot = try tokenizer.rawDecode(tokenIds: prefix + suffix, skipSpecialTokens: false)
        let prefixDecoded = try tokenizer.rawDecode(tokenIds: prefix, skipSpecialTokens: false)
        // The seeded stream should emit only the tail that was added after the
        // initial ids; the prefix bytes were established by the seeding step.
        #expect(collected == String(oneShot.dropFirst(prefixDecoded.count)))
    }

    @Test
    func consumeBatchReturnsConcatenation() async throws {
        let tokenizer = try await makeTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        let ids = try tokenizer.encode(text: "The quick brown fox jumps", addSpecialTokens: false)
        let stream = tokenizer.streamingDetokenizer()
        let combined = try stream.consume(ids)
        let oneShot = try tokenizer.rawDecode(tokenIds: ids, skipSpecialTokens: false)
        #expect(combined == oneShot)
    }

    @Test
    func skippingSpecialTokensYieldsNilForUnchangedDecode() async throws {
        // When special-token-skipping causes a decode to leave the cached
        // prefix unchanged (no new visible bytes), `consume` must withhold
        // rather than tripping the prefix-invariant check or emitting "".
        let tokenizer = try await makeTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        let bosTokenId = try #require(tokenizer.bosTokenId ?? tokenizer.eosTokenId)

        let stream = tokenizer.streamingDetokenizer(skipSpecialTokens: true)
        let chunk = try stream.consume(bosTokenId)
        // Skipped tokens contribute no visible bytes, so the stream emits nil.
        #expect(chunk == nil)
    }

    @Test
    func cleanupCompatibilityCanary() async throws {
        // BERT uncased applies `cleanUpTokenizationSpaces`. Streaming routes
        // through `rawDecode`, so retroactive rewrites do not break the
        // prefix invariant. If a future implementation regresses to using the
        // public `decode`, this test will trip on a contraction-like edge
        // case once the cleanup pass rewrites whitespace mid-stream.
        let tokenizer = try await makeTokenizer("google-bert/bert-base-uncased")
        let ids = try tokenizer.encode(text: "l'eure", addSpecialTokens: false)
        let stream = tokenizer.streamingDetokenizer()
        var combined = ""
        for id in ids {
            if let chunk = try stream.consume(id) {
                combined.append(chunk)
            }
        }
        // The streamed concatenation matches the raw (uncleaned) decode.
        let raw = try tokenizer.rawDecode(tokenIds: ids, skipSpecialTokens: false)
        #expect(combined == raw)
    }

    @Test
    func transactionalStateOnInvalidTokenIdThrow() async throws {
        let tokenizer = try await makeTokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        let ids = try tokenizer.encode(text: "Hello", addSpecialTokens: false)
        let stream = tokenizer.streamingDetokenizer()

        // Feed valid tokens first.
        for id in ids {
            _ = try stream.consume(id)
        }

        // Snapshot internal state via Mirror.
        func snapshot() -> ([Int], String, Int)? {
            let mirror = Mirror(reflecting: stream)
            guard
                let ids = mirror.children.first(where: { $0.label == "ids" })?.value as? [Int],
                let prefix = mirror.children.first(where: { $0.label == "prefix" })?.value as? String,
                let prefixIndex = mirror.children.first(where: { $0.label == "prefixIndex" })?.value as? Int
            else { return nil }
            return (ids, prefix, prefixIndex)
        }

        let before = snapshot()
        try #require(before != nil)

        // Feed a bogus id that the boundary guard will reject.
        do {
            _ = try stream.consume(-1)
            Issue.record("Expected throw on negative id")
        } catch {
            // Expected.
        }

        let after = snapshot()
        try #require(after != nil)

        if let before, let after {
            #expect(before.0 == after.0)
            #expect(before.1 == after.1)
            #expect(before.2 == after.2)
        }

        // After rollback the next valid consume should still produce the right text.
        let nextIds = try tokenizer.encode(text: " world", addSpecialTokens: false)
        var continuation = ""
        for id in nextIds {
            if let chunk = try stream.consume(id) {
                continuation.append(chunk)
            }
        }
        let allIds = ids + nextIds
        let oneShot = try tokenizer.rawDecode(tokenIds: allIds, skipSpecialTokens: false)
        let prefixDecoded = try tokenizer.rawDecode(tokenIds: ids, skipSpecialTokens: false)
        #expect(continuation == String(oneShot.dropFirst(prefixDecoded.count)))
    }
}
