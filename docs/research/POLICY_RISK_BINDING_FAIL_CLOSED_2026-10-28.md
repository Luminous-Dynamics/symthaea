# Fail-Closed Policy-to-Risk-Matrix Binding

**Date:** 2026-10-28  
**Status:** Python validation helper plus a typed Rust validator and inline unit tests are committed. The Rust tests have not been executed in this environment because no Rust/Cargo toolchain is installed. The verifier is not yet wired into production action dispatch.

## Why this gate is needed

The decision-sufficient ambiguity experiment showed a large failure under an intentional policy-row identity mismatch. Mean regret increased from 0.00152 bits to 0.09781 bits, with 31 pooled-covered and 26 group-max-covered test cases violating the conditional regret <= 2r property when measured against the true deployed action losses. Statistical calibration cannot correct a risk matrix whose rows do not mean the actions the executor actually runs.

The existing Symthaea subterranean policy-governance protocol already describes stable candidate identity/epoch and source-evidence digests, shadow-only transfer, and safety precedence. This validator targets a narrower invariant between those objects: the exact ordered action set used by the planner to construct the risk matrix must match the ordered policy set evaluated and executed.

## Binding schema

The new validation/policy_risk_binding.py helper defines a deterministic manifest with:

- schema version;
- ordered unique policy IDs;
- ordered unique hypothesis IDs;
- risk matrix shape;
- SHA-256 of canonical row-major IEEE-754 binary64 big-endian loss bytes;
- SHA-256 of canonical JSON manifest payload.

At a planning boundary, verify_binding rejects:

- invalid or duplicate identifiers;
- policy ID or hypothesis ID reordering;
- changed risk-matrix values, changed row order, or shape mismatch;
- mismatch between planned policy order and execution policy order;
- altered manifest content, schema, self digest, or trusted manifest digest;
- NaN/infinity and negative losses.

The matrix is normalized to a stable canonical binary representation, so C- vs Fortran-contiguous storage does not change the digest.

## Rust implementation

A typed implementation now lives at crates/domains/symthaea-subterranean/src/policy_risk_binding.rs and is exposed by the subterranean crate. The crate manifest and Cargo.lock were updated for the workspace's existing sha2 dependency. Its eight inline unit tests cover canonical manifest round-trip, row permutations, policy and hypothesis order, execution order, trusted-anchor/self-digest tampering, invalid matrix values/shape, and signed-zero normalization.

This Rust module is **authored but not locally compiled or tested**: the current environment has no cargo or rustc executable, and the PR's latest CI run was skipped. Its committed file has Git blob 63c4e9aacfe5b7db5971e701c42ae38ac226c4c3; the crate manifest blob is 45de998db181d6459983e417942db27aee528e0e; lib.rs blob is 525647bb48110711bddecc813879b66be5e7e707; Cargo.lock blob is 5701d38370bb363508872871a15eb27537f8d645.

## Trust model caveat

A digest is an integrity binding, not authentication. The trusted manifest SHA-256 must come from a separately authenticated signed release/receipt or trusted control-plane record. If an attacker can replace both the matrix and the purported trusted digest, this helper alone cannot establish authenticity. The helper does not itself execute actions or prevent a compromised caller from bypassing the gate.

## Validation

Eight local tests pass, covering deterministic canonicalization, row permutation, policy-order mismatch, execution-order mismatch, hypothesis-order mismatch, trust-anchor/self-digest tampering, duplicate IDs, non-finite values, negative losses, and the real finite-game risk tensor's policy-row ordering. The combined decision-sufficient, negative-control, and policy-risk-binding suites pass **16/16 local tests** in 11.773 seconds. The source/test files were committed and their Git blob hashes fetched from the PR branch exactly match the executed local files:

- validation/policy_risk_binding.py Git blob: 66cccbbf9b6bbe9d04fce05ac998a3fc0e576b27; SHA-256: c2c709906a91dc6b9c80d313c6d5497be96c2ea55b04977a5ab977951a1b4c4b.
- validation/test_policy_risk_binding.py Git blob: 30a9dcfaf181b2cb4a6c6069ebc5648a09e25168; SHA-256: e26b2da17d6057361fd6e63eb3970cb504ffd1f5610b15181a55494643121186.

## Next production integration gate

1. Compile and run the new Rust unit tests, then add trusted signature/release-receipt verification (the current Rust module checks integrity digests but not authenticity).
2. Bind the risk tensor digest, policy ID order, hypothesis order, objective/schema version, simulator build identity, model revision, calibration receipt, and execution artifact digest into one signed decision receipt.
3. Verify this receipt immediately before optimization and again before dispatch; any mismatch should reject the proposed action and use a prequalified safe baseline or abstain.
4. Add mutation/property tests: reorder rows/IDs, substitute a policy while retaining its label, replay stale matrices, mutate a hypothesis schema, and replace the risk tensor after certification.
5. Add runtime counters for mismatch, stale evidence, OOD shift, fallback and abstention; test both fail-closed correctness and fallback availability.
6. Run the gates on exact PR head and actual crate APIs. This standalone validation helper is not proof that the production system is currently protected.

No production readiness, certification, remote CI, or SOTA claim is made.
