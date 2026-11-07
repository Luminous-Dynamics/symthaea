# CEB Repository Twin — public development agent view V0

**Task:** CEB-REPO-001  
**Status:** `public_development_only_not_sealed`  
**Pinned source revision:** `7a563cd61f4bda3347119ea39169a2e2d1c4f30b`

This directory is a source-only materialization of the synthetic five-crate Rust workspace. It contains the allowlisted Cargo manifests, Rust source files and integration-test source needed to inspect the workspace.

## Deliberate package exclusions

This package subtree does not contain the gold graph, mutation scenarios, feature/target gold worlds, hidden labels, reference patches or evaluator code. See `manifest.v0.json` for the exact path inventory and Git blob object IDs.

## Critical custody warning

The evaluator/design artifacts remain public elsewhere in the same Git repository. This is **not a sealed agent/evaluator split**, must not be scored as a blind holdout, and must not be represented as confidential. Use it only for public development, allowlist review, packaging-layout review and deterministic-inventory design.

The manifest records Git SHA-1 blob object IDs for byte-preservation checks. These are not the CEB-TREE-V1 SHA-256 tree digest, not signatures and not isolation evidence.

No source, build script, procedural macro, test, or binary was executed to produce this package. Execution remains blocked by #10064.
