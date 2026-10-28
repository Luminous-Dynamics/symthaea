# Fail-Closed Policy-to-Risk-Matrix Binding

**Date:** 2026-10-28  
**Status:** executable validation helper and tests; not yet integrated into production Rust control path.

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

## Trust model caveat

A digest is an integrity binding, not authentication. The trusted manifest SHA-256 must come from a separately authenticated signed release/receipt or trusted control-plane record. If an attacker can replace both the matrix and the purported trusted digest, this helper alone cannot establish authenticity. The helper does not itself execute actions or prevent a compromised caller from bypassing the gate.

## Validation

Seven local tests pass, covering deterministic canonicalization, row permutation, policy-order mismatch, execution-order mismatch, hypothesis-order mismatch, trust-anchor/self-digest tampering, duplicate IDs, non-finite values, and negative losses. The combined decision-sufficient, negative-control, and policy-risk-binding suites pass **15/15 local tests** in 12.85 seconds. The source/test files were committed and their Git blob hashes fetched from the PR branch exactly match the executed local files:

- validation/policy_risk_binding.py Git blob: 66cccbbf9b6bbe9d04fce05ac998a3fc0e576b27; SHA-256: c2c709906a91dc6b9c80d313c6d5497be96c2ea55b04977a5ab977951a1b4c4b.
- validation/test_policy_risk_binding.py Git blob: 29b8176acd6dae0b6ab674c8f3537fffef786374; SHA-256: bb180baed9c4eb4092ed6f9dd10ae523439df90e48ae430ee2560316f96a4241.

## Next production integration gate

1. Add a typed Rust manifest with a trusted signature/release-receipt binding—not Python-only enforcement.
2. Bind the risk tensor digest, policy ID order, hypothesis order, objective/schema version, simulator build identity, model revision, calibration receipt, and execution artifact digest into one signed decision receipt.
3. Verify this receipt immediately before optimization and again before dispatch; any mismatch should reject the proposed action and use a prequalified safe baseline or abstain.
4. Add mutation/property tests: reorder rows/IDs, substitute a policy while retaining its label, replay stale matrices, mutate a hypothesis schema, and replace the risk tensor after certification.
5. Add runtime counters for mismatch, stale evidence, OOD shift, fallback and abstention; test both fail-closed correctness and fallback availability.
6. Run the gates on exact PR head and actual crate APIs. This standalone validation helper is not proof that the production system is currently protected.

No production readiness, certification, remote CI, or SOTA claim is made.
