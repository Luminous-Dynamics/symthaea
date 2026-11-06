# Independent replay and byte binding for adaptive connectome challenges V1

**Tracking:** #10661 — SCI-CONNECTOMICS-091
**Composes:** #10603, #10656, #10657, #10658, #10660
**Status:** executable replay bridge and synthetic known-answer receipt; no biological claim is made.

## 1. Gap this closes

The discovery-bundle verifier in #10660 checks the structure of a challenge reference and consistency of reported before/after fields. That is not sufficient to establish that the upstream #10603 planner actually processed the supplied base input and challenge bytes.

This bridge binds the exact raw base input and exact raw upstream report, recomputes the upstream `input_sha256`, independently recomputes the full #10603 report using the separate Python tuple-filtering oracle, and checks the selected challenge result against a frozen binding manifest.

## 2. Exact artifact contract

For each selected scenario challenge, the replay receipt binds:
- raw base-input SHA-256 and the `input_sha256` embedded in the exact upstream report;
- raw upstream report SHA-256 (the original report bytes, not reserialized JSON);
- frozen dataset/query/base scenario/action/joint-support identities;
- challenge ID, exact-once presence in both base input and report, and a recomputed challenge digest;
- expanded scenario-set and expanded joint-support SHA-256 values from the independently replayed challenge result;
- before/after adaptive policy status, effort and monotonicity result;
- exact upstream Rust planner Git blob SHA-1 and independent oracle Git blob SHA-1;
- exact commit SHA checked out for the upstream run.

The challenge digest is recomputed from the raw input's parsed challenge object using the independent oracle's pinned V1 digest function. The entire upstream report is independently recomputed and compared field-for-field with the report JSON. This is not a digest-only attestation.

Git blob identifiers in the binding manifest are explicitly Git object SHA-1 identities, not SHA-256 content digests. Raw input/report SHA-256 values are emitted in the receipt.

## 3. Independent replay algorithm

The bridge imports `verify_adaptive_query_policy_oracle.py`, which independently implements the adaptive planner using tuple filtering and memoized minimax recursion. It does not call the Rust planner or reuse its equality-mask state representation.

1. Read exact input, report, binding, Rust source and oracle source bytes.
2. Verify the binding manifest's frozen query and base digest identities.
3. Parse the report JSON but retain and hash its exact raw bytes.
4. Verify the report's `input_sha256` equals SHA-256 over the exact raw input bytes.
5. Require `search_complete: true` and a recognized exact terminal status.
6. Require the challenge ID to appear exactly once in the input and exactly once in the upstream report.
7. Recompute the challenge digest from the base input and its exact challenge record.
8. Recompute the entire expected report with the independent Python oracle; require exact JSON equality.
9. Require the exact selected challenge entry to match the oracle result and frozen expected semantic outcome.
10. Verify the upstream planner and oracle source Git blob identities; emit an exact-head receipt binding both raw artifact hashes and source identities.

Any missing/duplicate challenge, stale report, cross-query report, altered source, changed policy cost, failed monotonicity, incomplete exact search or oracle mismatch rejects the binding.

## 4. Synthetic known-answer challenge

Fixture: `CONNECTOME_ADAPTIVE_HYPOTHESIS_EXPANSION.example.json`.

Challenge: `append-refuting-world-sharing-observation-tuple`.

Expected result:
- one refuting reconstruction world appended;
- baseline adaptive policy exists at 4 effort units;
- after expansion, no adaptive policy exists within the frozen action universe;
- append-only monotonicity passes;
- the base query, action universe and joint support are unchanged.

This synthetic result demonstrates how an omitted opposing reconstruction can invalidate an exact baseline policy. It does not establish that a real biological alternative is admissible or that all alternatives have been discovered.

## 5. Claim ceiling

A passing receipt establishes byte binding and independent software-semantic reproduction for the declared finite #10603 input and challenge. It does not authenticate source evidence producers, prove biological adjudication, establish connectome completeness, or prove that the declared finite world set covers biological reality.