# Patch 0009: test run clean room mail series replay

**Series:** 27

## Objective

Prove the real Series 22–26 patch chain applies and reproduces one exact source tree.

## Intended changes

- Replay from the exact Series 21 final tree under sanitized Git configuration.
- Verify per-patch and final tree identities.
- Build and test only from the replayed tree.

## Required tests

- No manual edit or hidden generated file is required.
- Replayed and authored final trees are identical.
- Patch and source archives reproduce deterministically.

## Non-claims

- Does not accept a semantically similar but different tree.
- Does not reuse build artifacts from the authoring workspace.
