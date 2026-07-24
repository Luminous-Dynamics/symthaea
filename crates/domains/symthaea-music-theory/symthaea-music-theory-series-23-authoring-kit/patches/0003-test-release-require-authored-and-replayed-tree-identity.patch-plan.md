# Patch 0003: Require authored and replayed tree identity

**Series:** 23

## Objective

Prevent a release tree from containing unrepresented edits or generated artifacts absent from the patch lineage.

## Intended changes

- Compare final Git trees, tracked file inventories, executable bits, symlink targets, and submodule metadata.
- Fail on untracked source-like files and ignored generated Rust sources.
- Produce a bounded diff inventory without silently normalizing mismatches.

## Required tests

- Detect one uncommitted source edit.
- Detect executable-bit drift.
- Detect a generated file referenced by Cargo but absent from Git.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
