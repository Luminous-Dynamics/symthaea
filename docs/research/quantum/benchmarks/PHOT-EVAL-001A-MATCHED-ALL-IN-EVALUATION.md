# PHOT-EVAL-001A: matched all-in evaluation contract

Status: implementation draft. The standalone Rust test runner is added for rapid, isolated validation; application experiments remain to be run.

Issue: https://github.com/Luminous-Dynamics/symthaea/issues/8734

## Why an evaluation contract is the next algorithmic step

A feature can exist and still be useless. A controller can reduce sampled trajectories and still make the end-to-end solver slower. A candidate can have a high objective and still be infeasible. A benchmark can accidentally report a speedup by omitting a failed run, a transfer stage, repair, or verification.

QOBLIB separates application, algorithm, and system benchmarking; it publishes instance records and accepts both better solutions and useful negative results. Its paper explicitly says its released classical baseline runs are not necessarily the best solutions achievable with extensive classical engineering. Therefore public QOBLIB instances supply reproducibility, not a license to compare against only weak reference runs:
- https://doi.org/10.1038/s43588-026-00991-1
- https://zib-aopt.github.io/QOBLIB/participate.html

For MIS, the external comparison should include specialized graph methods such as KaMIS/ReduMIS and suitable exact or local-search methods where available and license-eligible:
- https://github.com/KarlsruheMIS

## Implemented contract

The Rust module matched_evaluation.rs defines:

- Matched arm identity bound to SHA-256 of canonical original graph bytes and a frozen comparison protocol.
- Same target cardinality, per-trial solver budget, tuning budget, and independent verifier digest across the two arms.
- Exact pairing by pair ID, with seed identity required to match. No silent inner join discards unmatched attempts.
- An explicit termination state and candidate-verification state for every trial, including failed, timed-out, aborted, unverified, conflicting, and verifier-failed candidates.
- Target attainment derived only when the same declared independent verifier confirms a conflict-free independent set whose cardinality meets the target.
- A fixed cost vector for queue wait, acquisition, preprocessing, feature extraction, transfer, amortized compilation/initialization, solve, repair, verification, and receipt persistence.
- Two non-interchangeable timing conventions: deployment all-in latency includes queue delay and amortized compilation/initialization; standardized algorithm runtime excludes those two fields to match common benchmark submission rules. Both require every stage in the respective convention to be measured.
- Unknown costs represented as unknown. If any all-in stage is unknown, deployment latency and all-in speedup are unavailable; if any algorithm-runtime stage is unknown, standardized runtime is unavailable. A missing stage is never a zero-cost stage.
- Energy records with explicit NotMeasured, DeviceOnly, PartialSystem, or FullSystem boundaries. Energy comparisons are emitted only when both successful paired trials have comparable measured boundaries.
- Per-pair details plus target-hit counts and rates (candidate, baseline, both, either-only, neither), candidate-minus-baseline hit-rate delta, all-in-costed and algorithm-runtime-costed pair counts, uncosted target hits, energy-scope mismatches, and separate medians for all-in latency, standardized runtime, and measured energy ratios.

The ratio baseline/candidate is greater than 1 when the candidate uses less of the measured resource. Ratios are reported only for pairs where both methods attain the same verified target and the relevant cost vectors are complete. These are conditional diagnostics and can be survivor-biased when hard pairs are missed, so they are always accompanied by both arms' hit rates and paired hit-rate delta. Deployment latency and standardized algorithm runtime are reported separately; do not call either one the other. Missed targets remain in target-hit counts and do not receive an invented ratio.

## Standalone smoke runner

The helper script scripts/research/run-photonic-contract-tests.sh compiles the four dependency-free Rust contracts directly with rustc --test rather than invoking Cargo. This provides a fast, narrow test path while the repository's wider workspace-discovery issue is being resolved. It does not replace full crate compilation, cargo fmt, clippy, repository CI, exact MIS integration, or application-level comparison.

## Candidate and verifier-receipt evidence boundary

The stacked follow-up implementation in `mis_evidence.rs` now defines a narrow canonical format for graph bytes and candidate bytes and checks them independently:

- Recomputes SHA-256 from the exact original graph and candidate byte slices.
- Rejects non-canonical line endings, ambiguous decimal encodings, duplicate/unsorted edges, self-loops, out-of-range vertices, duplicate/unsorted candidate vertices, and declared-count mismatches.
- Counts selected-edge conflicts directly from the canonical graph. A candidate with a conflict gets a conflict receipt, never a target hit; a feasible candidate below the frozen target also is not a hit.
- Constructs a deterministic receipt from the recomputed graph/candidate digests, verifier digest, target, size, conflict count, and verdict.
- `verify_mis_candidate_with_receipt` re-runs verification and requires the actual persisted receipt bytes to exactly match the reconstructed receipt bytes; it also recomputes the receipt SHA-256.
- Returns an opaque `VerifiedMisCandidate` whose fields cannot be constructed or modified through the crate's safe public API. The matched evaluator requires this evidence token and rebinds its metadata to the trial and report.

The SHA-256 implementation is dependency-free and has known-answer vectors for empty input, `abc`, the standard quick-brown-fox vector, multi-block input, and lengths around SHA-256 padding boundaries. These tests are authored but have not yet been executed in Rust at the time of this draft; exact-head CI remains required.

**Trust limit:** hashes establish byte identity/integrity, not signer identity or trusted execution. The verifier digest is supplied by the caller and must be pinned to the actual reviewed verifier build/source outside this module. This module does not sign receipts or authenticate the machine that executed it. The existing `ResearchArtifactReceipt` still uses a non-cryptographic FNV fingerprint and is not suitable as an authenticated verifier receipt. Do not claim malicious-producer resistance, independent lab reproduction, or cryptographic attestation until a separate trust mechanism is implemented and audited.

## Negative controls in unit tests

- Unknown acquisition time suppresses all-in time and speedup rather than making the candidate look cheaper.
- A large unverified candidate is not a target hit.
- Missing verifier receipt, malformed receipt digest, and receipt metadata disagreement with graph/verifier/candidate/target are rejected.
- A verifier failure without a receipt remains recordable but is never a target hit.
- Supplying joules under NotMeasured is rejected.
- Device-only energy cannot be compared directly with full-system energy.
- Different graph digests, target size, solver budget, verifier digest, pair ID sets, or paired seed identity are rejected.
- An inconsistent verified-independent-set record and a complete zero-cost record are rejected.
- Speedup is only emitted for both-hit, fully costed pairs.

## Application experiment protocol

For each canonical graph digest, freeze the target cardinality and per-trial solver budget. Compare:
1. Strong classical solver without trajectory calls.
2. Same graph-reduction pipeline with final-spin-frequency ordering.
3. Same pipeline with time-resolved features.
4. Same pipeline with shuffled feature scores and seeded-random order.
5. Fixed-K trajectory acquisition.
6. Adaptive acquisition gate.
7. A hybrid portfolio including an eligible specialist solver such as KaMIS/ReduMIS.

Use a development split for tuning and a confirmation split frozen by graph family. QOBLIB's contribution guidance calls for at least five runs for stochastic methods and recommends ten or more; follow the pinned benchmark revision's current rule. Pair stochastic repeats by explicit trial IDs and seeds, but treat graph instances—not repeated seeds on one graph—as the independent units for generalization claims.

Report feasible objective, best known/proven bound, proof status, target-hit fraction, all-in verified time-to-target, failures/timeouts, stage breakdown, energy scope, and graph-level paired effects with uncertainty. Keep solution-quality comparisons separate from optimality certification.

## Go/no-go

Continue the trajectory controller only if held-out gains persist against final-spin, structural, shuffled, random, and strong classical baselines after acquisition and all downstream stages are charged. If the gains disappear in all-in time, reduce quality, or fail to transfer across graph families, publish the negative result and stop tuning the controller family.

## No claims

The module and smoke runner are test infrastructure. They do not establish improved MIS quality, speedup, SOTA, physical energy advantage, hardware advantage, or quantum advantage.
