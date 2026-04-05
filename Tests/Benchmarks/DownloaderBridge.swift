import Foundation
import HFAPI
import MLXLMCommon

enum DownloaderBridgeError: LocalizedError {
    case invalidRepositoryID(String)

    var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id):
            return "Invalid Hugging Face repository ID: '\(id)'. Expected format 'namespace/name'."
        }
    }
}

struct HubClientDownloader: Downloader, @unchecked Sendable {
    private let upstream: HubClient

    init(_ upstream: HubClient) {
        self.upstream = upstream
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: id) else {
            throw DownloaderBridgeError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        if !useLatest,
            let cached = try? await upstream.downloadSnapshot(
                of: repoID,
                revision: revision,
                matching: patterns,
                localFilesOnly: true,
                progressHandler: progressHandler
            )
        {
            return cached
        }

        return try await upstream.downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: progressHandler
        )
    }
}
