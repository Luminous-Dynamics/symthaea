# CEB Repository Twin — public development agent view V0

**Task:** CEB-REPO-001  
**Status:** `public_development_only_not_sealed`  
**Pinned source revision:** `7a563cd61f4bda3347119ea39169a2e2d1c4f30b`  
**Payload:** `workspace/`  
**Payload CEB-TREE-V1:** `9eb74b8c51a72e44328f55f8dbba4c3f340bb96fc3038764c88571b8466a7a92`
**Manifest SHA-256:** `2de350d2195b2a8c5d667551f91baef3b214545aead8468c98259b3acc8808a7`

The `workspace/` directory contains 15 allowlisted files from the synthetic five-crate Rust fixture: Cargo manifests, Rust source and integration-test source. The manifest is outside the payload tree to avoid self-reference. It records each file's mode, byte length and SHA-256 digest and pins the existing CEB-TREE-V1 payload digest.

Verify the static inventory with the Rust receipt verifier's `verify-agent-view` command, supplying both the exact manifest SHA-256 and payload tree SHA-256 from the trusted task registry.

## Deliberate exclusions

The payload does not contain the gold graph, mutation scenarios, feature/target gold worlds, hidden labels, reference patches or evaluator code.

## Critical custody warning

The evaluator/design artifacts remain public elsewhere in this repository. This is **not** a sealed agent/evaluator split, must not be scored as a blind holdout, and must not be represented as confidential. Use it only for public development, allowlist review, packaging-layout review and deterministic-inventory design.

The manifest's SHA-256 and CEB-TREE-V1 digest are content commitments, not signatures, rights grants, isolation evidence or proof of hidden-answer custody. The digest values printed above are published for reproducibility only: **no separate trusted task registry or external pin service has been provisioned yet**. A production caller must obtain both expected values through an independently controlled channel; a package README or manifest cannot authenticate itself.

No source, build script, procedural macro, test, or binary was executed to produce this package. Execution remains blocked by #10064.
