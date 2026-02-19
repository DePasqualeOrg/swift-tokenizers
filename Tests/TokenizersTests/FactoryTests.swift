// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation
import HuggingFace
import Testing

@testable import Tokenizers

private let downloadDestination: URL = {
    let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return base.appending(component: "huggingface-factory-tests")
}()

private let hubClient = HubClient()
private let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json", "config.json"]

private func downloadModel(_ model: Repo.ID) async throws -> URL {
    try await hubClient.downloadSnapshot(
        of: model,
        to: downloadDestination.appending(path: "\(model)"),
        matching: tokenizerFiles
    )
}

@Suite("Factory", .serialized)
struct FactoryTests {
    @Test
    func llama() async throws {
        let modelDirectory = try await downloadModel("coreml-projects/Llama-2-7b-chat-coreml")

        let tokenizer = try await AutoTokenizer.from(directory: modelDirectory)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test
    func whisper() async throws {
        let modelDirectory = try await downloadModel("openai/whisper-large-v2")

        let tokenizer = try await AutoTokenizer.from(directory: modelDirectory)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }
}
