# Upstream commits analysis

Ledger of upstream commits from [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) and how they have been evaluated and handled.

**Last sync date:** 2026-04-01
**Last checked upstream commit:** `b38443e`
**Previous upstream commit (already synced):** `e5e227b` – Replace HubApi downloader with swift-huggingface HubClient (#315)

## Commits

| Hash | PR | Description | Scope | Action |
|------|-----|-------------|-------|--------|
| `f3d5cbf` | #319 | Fix MetaspacePreTokenizer: prepend_scheme no longer gated by addPrefixSpace | Tokenizers | Merged (#11) |
| `31be7a5` | #320 | Support tokenizer from FacebookAI/xlm-roberta-base | Tokenizers | Merged (#11) |
| `67baef8` | #321 | Add integration test for MetaspacePreTokenizer fix | Tokenizers | Merged (#11) |
| `150169b` | #324 | Add JSON5 flag (YYJSON_READ_ALLOW_INF_AND_NAN) | Tokenizers | Merged (#13) |
| `4f6c72b` | #323 | Enable lstripBlocks and trimBlocks for chat templates | Tokenizers | Merged (#12) |
| `e5e227b` | #315 | Replace HubApi downloader with swift-huggingface HubClient | Hub only | Skip |
| `5926983` | #328 | Fix yyjson parity with JSONSerialization | Tokenizers | PR pending |
| `eed7264` | #326 | Remove `Downloader` and update `HubApi.download` to throw `HubClientError` | Hub only | Skip |
| `d7eb87f` | #338 | Create background Hub clients only if necessary | Hub only | Skip |
| `58c4bc1` | #336 | Fix progress handler regression | Hub only | Skip |
| `b38443e` | #334 | Add Swift 6.1 manifest to forward Xet trait to `swift-huggingface` | CI/Package | Skip |
