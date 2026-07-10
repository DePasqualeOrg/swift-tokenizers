---
name: sync-upstream
description: Evaluate and adapt tokenizer changes from huggingface/swift-transformers for the independently maintained swift-tokenizers fork. Use when the user explicitly asks to inspect, plan, or perform an upstream synchronization.
---

# Sync Upstream

## Preserve the fork model and scope

This repository has no `upstream` remote. Fetch the upstream default branch directly into the pull-only `upstream-main` branch:

```text
git fetch https://github.com/huggingface/swift-transformers.git main:upstream-main
```

Never push `upstream-main`.

Only tokenizer code is in scope: `Sources/Tokenizers` and `Tests/TokenizersTests`. Skip upstream changes that affect only Hub downloads, Core ML support, generation, or other components outside those paths.

## Maintain the tracking documents

Use `docs/upstream-commits-analysis.md` as the concise persistent ledger, with one row per upstream pull request and its commit, description, judgment, and action. Use a separate dated working document for detailed analysis during each sync, update the ledger's last-sync date and last-checked commit, and delete the working document when the sync is complete.

## Evaluate before applying

1. Read each in-scope upstream diff and identify its behavioral intent.
2. Check whether this fork already provides equivalent behavior or uses a different architecture.
3. Cross-check tokenizer and API behavior against Python `tokenizers` and `huggingface_hub`.
4. Test against real tokenizer outputs when mocks cannot establish correctness.
5. Record the judgment before applying a change and explain skipped or adapted work.

Files moved from upstream paths with `git mv` retain rename history. Try the cherry-pick before assuming the directory differences make it incompatible.

## Adapt selected changes

Use `git cherry-pick <hash> --no-commit` to inspect and adapt a selected change without creating a commit. Preserve the fork's organization, remove dead code introduced by incomplete adaptations, use realistic test data, and fix tokenizer edge cases when upstream behavior is incomplete.

Adapt upstream tests that use `AutoTokenizer.from(pretrained:)` to the fork's `downloadModel` plus `AutoTokenizer.from(directory:)` workflow.

Build and test through the project's supported containerized workflow. Do not create a commit unless the user explicitly authorizes it in the current conversation. For an authorized cherry-pick commit, preserve the original author, document nontrivial adaptations, and reference the upstream pull request and commit. Keep separate fork-authored improvements in a separate authorized commit when that history is useful.

Creating or pushing branches, opening pull requests, or changing GitHub state requires separate explicit authorization. When asked, prepare one temporary root-level pull request description per logical change, use full upstream pull request URLs, and do not commit those temporary description files.

## Copyright headers

Ensure modified Swift files have the correct headers:

- Upstream-only content: `// Copyright © Hugging Face SAS`
- Anthony-only content: `// Copyright © Anthony DePasquale`
- Mixed content: Hugging Face first, then Anthony

Add Anthony's line when the fork makes substantive modifications to upstream code. Do not add AI attribution or AI co-author metadata.
