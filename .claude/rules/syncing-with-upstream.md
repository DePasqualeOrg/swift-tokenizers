# Syncing with Upstream

This repo is a fork of [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers). There is no `upstream` remote — syncing is done by fetching directly from the URL to avoid `gh` and GitHub Desktop associating the repo with the upstream account:

```
git fetch https://github.com/huggingface/swift-transformers.git main:upstream-main
```

The local `upstream-main` branch is a pull-only reference to the original repo's main branch.

## Scope

This fork only includes the tokenizer portion of upstream (`Sources/Tokenizers` and its tests). Upstream also contains Hub download utilities, Core ML model support, generation logic, and other components that are out of scope here. When reviewing upstream commits, skip any changes that only touch files outside `Sources/Tokenizers` or `Tests/TokenizersTests`.

## Tracking documents

**`docs/upstream-commits-analysis.md`** is the persistent ledger of all upstream commits and our judgment/action on each. Keep it concise — one table row per upstream PR with commit hash, description, and action taken.

For each sync session, create a fresh working document (e.g., `docs/upstream-sync-2026-03-15.md`) for detailed analysis, comparisons with Python, API investigation notes, etc. Update the main ledger with judgments/actions as you go. Delete the working document once the sync is complete.

After completing a sync, update the last sync date and last checked commit hash at the top of the ledger.

## Philosophy

Our goal is to align closely with the Python `huggingface_hub` and `tokenizers` libraries. When upstream introduces new designs or strategies that diverge from how Python does things, we prefer to follow Python's approach instead. Some differences from upstream are also due to improvements we've made on top of cherry-picked commits.

## Evaluating upstream changes

Before cherry-picking, evaluate each upstream PR:

1. **Read the diff** to understand what it does.
2. **Check if we already have equivalent functionality.** Our architecture may differ from upstream in some areas.
3. **Cross-reference with Python** (`tokenizers`, `huggingface_hub`) when the change involves API behavior or tokenizer logic. Verify that the Swift implementation matches what Python does.
4. **Test against real tokenizer outputs** when mock tests alone can't verify correctness.

## File history and cherry-pick compatibility

Files in this fork were moved from their upstream paths using `git mv` (e.g., `Sources/Hub/YYJSONParser.swift` → `Sources/Tokenizers/YYJSONParser.swift`). Because `git mv` preserves rename history, `git cherry-pick` can automatically resolve path differences for these files. Do not assume cherry-picks will fail due to different directory structures – try the cherry-pick first.

## Cherry-picking workflow

1. Fetch the latest upstream commits into `upstream-main` before starting:
   ```
   git fetch https://github.com/huggingface/swift-transformers.git main:upstream-main
   ```
2. Use `git cherry-pick <hash> --no-commit` to inspect changes before committing.
2. Resolve conflicts:
   - Keep our version when upstream changes don't apply to our architecture.
   - For non-trivial conflict resolution, note what was dropped in the commit message.
4. Build and run tests before committing.
5. Commit the cherry-pick preserving original authorship with `--author="Name <email>"`. Add `Co-Authored-By: Anthony DePasquale <anthony@depasquale.org>` when we make modifications beyond trivial conflict resolution. Reference the upstream PR in the commit message: `Cherry-picked from huggingface/swift-transformers#XX (hash).`
6. If we need to make additional improvements on top of the cherry-pick (e.g., fixing tests, handling edge cases the upstream missed), commit those as **separate commits** under Anthony's authorship (the default git config).

## PRs and documentation

- Create a **separate PR for each logical change** (one upstream PR or a small group of related changes).
- Write a PR description as a markdown file in the project root (e.g., `pr-fix-metaspace-prepend-scheme.md`) before opening the PR. These are temporary files for copying into GitHub and are not committed. Use full URLs for upstream PR references (e.g., `[#319](https://github.com/huggingface/swift-transformers/pull/319)`) to avoid GitHub linking within our repo.
- Update the checklist in `docs/upstream-commits-analysis.md` after each item.

## Copyright headers

All `.swift` files should have copyright headers. If a file is missing them, add them at the top of the file. The rules for which copyright lines to include:

- **Upstream-only content** (pure cherry-picks with no significant modifications): `// Copyright © Hugging Face SAS`
- **Anthony-only content** (entirely new files written by us): `// Copyright © Anthony DePasquale`
- **Mixed content** (upstream code with our modifications, or files with contributions from both): both lines, with Hugging Face first:
  ```swift
  // Copyright © Hugging Face SAS
  // Copyright © Anthony DePasquale
  ```

For any files modified during the sync process, check the copyright headers. If `// Copyright © Anthony DePasquale` is not already present, add it below the existing copyright lines.

## Common adaptations

Upstream tests often use `AutoTokenizer.from(pretrained:)` which doesn't exist in our fork. Adapt these to use `downloadModel` + `AutoTokenizer.from(directory:)`.

## Changes on top of cherry-picks

When upstream changes are incomplete or incorrect, fix them in the same PR rather than deferring:
- Update mock tests to use realistic data.
- Remove dead code left by the cherry-pick (e.g., properties that are no longer read after a refactor).
- Fix logic to match actual behavior when upstream missed edge cases.
