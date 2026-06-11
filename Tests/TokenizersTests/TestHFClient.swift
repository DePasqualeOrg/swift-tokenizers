import Foundation
import HFAPI

extension HFClient {
    /// Shared zero-config client for tests. `HFClient.init` throws only on
    /// environment misconfiguration (e.g. a malformed `HF_ENDPOINT`), so a
    /// trap here surfaces the configuration error directly.
    static let `default`: HFClient = {
        do {
            return try HFClient()
        } catch {
            fatalError("Failed to create default HFClient: \(error.localizedDescription)")
        }
    }()
}
