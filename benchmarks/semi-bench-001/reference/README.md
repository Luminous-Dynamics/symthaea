# SEMI-BENCH-001 reference evaluator (v0.1)

This directory contains a deliberately small, independent reference evaluator for binary risk scores. It locks metric semantics and known answers before a production Rust evaluator is implemented in Symthaea.

**The Python implementation is test-only reference code.** It uses only the Python standard library, imports no Symthaea code, does not train a model, does not fetch datasets and does not control hardware. It is not part of a production semiconductor pipeline. The intended production implementation remains Rust-first and must reproduce the frozen contract and answers independently.

## Scope

V1 scores binary probability/risk predictions, such as SECOM failure-risk predictions. The positive label must be declared by the calling benchmark protocol. For SECOM the intended positive class is **fail** (the source dataset encodes pass as -1 and fail as +1).

Implemented metrics:

- Threshold confusion matrix, sensitivity, specificity, precision, accuracy and balanced accuracy.
- Non-interpolated average precision with equal-score grouping.
- Pairwise ROC AUC with half credit for ties.
- Brier score.
- Fixed-width expected calibration error (ECE).
- False-positive rate at a requested recall target.
- Risk-versus-coverage curve with equal-confidence ties grouped.

Input validation rejects empty datasets, duplicate sample identities, non-boolean labels, non-finite/out-of-range scores, invalid thresholds, invalid bin counts and one-class-only samples. It does not silently impute, clip, reorder, relabel or repair inputs.

## Known-answer fixture

- Input: fixtures/binary-metrics-known-answer-v1.json
- Expected output: fixtures/binary-metrics-known-answer-v1.expected.json
- Input SHA-256: aa9e2511b5a43f5fe0353b5d8c7c2d203f6de3f69865655b88ce776fd59e2614
- Expected output SHA-256: 177123d1a255c52936bcc31f0d300696cf6d84da281e7d04d2b57421c2b0c8c8

The 8-row fixture is synthetic and balanced by design. It tests ties, calibration bins, threshold metrics, recall-constrained false-positive rate and selective prediction. It is **not** a SECOM result and must not be reported as an empirical model score.

## Reproduce locally

From the repository root, run:

    python3 -m unittest discover -s benchmarks/semi-bench-001/reference/python -p 'test_*.py' -v
    python3 benchmarks/semi-bench-001/reference/python/semi_binary_metrics_reference.py

No third-party Python packages are required.

## Metric conventions

- Probability score is finite and in [0, 1].
- A sample is predicted positive when score is greater than or equal to the decision threshold.
- Average precision uses descending unique score groups; ties are admitted together.
- ROC AUC is pairwise and gives half credit for ties.
- ECE uses fixed-width bins [lo, hi), with the final bin including 1.0.
- Risk is classification error at the 0.5 argmax decision; confidence is max(p, 1-p); equal-confidence ties are admitted together.
- FPR-at-recall chooses the strictest observed score threshold reaching the requested recall.

These conventions are part of the versioned contract. A future Rust implementation must match them exactly, or introduce a new schema/version rather than silently changing score meanings.

## Claim ceiling

Passing these tests establishes only that this reference implementation reproduces the checked-in synthetic known answers and rejects the tested malformed inputs. It does not establish predictive performance, SECOM generalization, fab yield improvement, process causality, metrology capability, physical semiconductor fabrication or production readiness.


## Validation-selected threshold policy

The separate helper `python/semi_threshold_policy_reference.py` selects the highest validation recall subject to a declared precision floor, with equal-score thresholds grouped. It then applies the chosen threshold once to a disjoint test subject. Validation/test identity overlap is rejected. If no threshold satisfies the floor, the specified fallback is reported rather than silently choosing one.

The synthetic 10-validation / 10-test known-answer case is frozen in `fixtures/validation-threshold-known-answer-v1.json` and `fixtures/validation-threshold-known-answer-v1.expected.json`. Its 3 tests bring the suite to 14 tests at that stage. The tiny fixture demonstrates semantics only; it says nothing about real threshold reliability on rare-failure datasets.


## Leakage-resistant split-manifest validator

The independent helper `python/semi_split_manifest_reference.py` validates a frozen split assignment without reading or distributing raw dataset rows. It supports three explicitly distinct policies: `group_disjoint`, `forward_time`, and `stratified_sample`. It rejects duplicate sample identities, empty splits, cross-split group reuse, non-chronological/overlapping time windows, timezone-free timestamps and malformed dataset digests. It emits a deterministic SHA-256 identity for the sorted assignment set and states which independence claims are—and are not—established.

The synthetic group-disjoint known-answer input/output are `fixtures/split-manifest-known-answer-v1.json` and `fixtures/split-manifest-known-answer-v1.expected.json`. Seven tests cover known-answer stability and adversarial leakage controls. Together with binary metrics, threshold policy, and inspection-decision policy, the current suite contains 31 tests. Hosted GitHub Actions run [#44890353550](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44890353550) completed successfully, including unit tests and byte-for-byte reproduction of the binary-metrics and inspection-decision reports. This is a hosted CI result; no local test execution is claimed.


## Budget-aware inspection decision reference

The independent helper `python/semi_inspection_decision_reference.py` scores a finite set of inspection candidates using a declared expected-net-value model. It accounts for defect-risk probability, test sensitivity/specificity, actionability, exposed population, avoidable loss, false-hold cost, direct inspection cost and delay penalty. It selects the exact best subset under inspection-time, cash-cost and per-group capacity constraints by exhaustive enumeration, capped at 18 candidates.

A matched-budget risk-only greedy policy is emitted as a comparator. The synthetic fixture is in `fixtures/inspection-decision-known-answer-v1.json` with frozen output in `fixtures/inspection-decision-known-answer-v1.expected.json`. Ten tests cover known answers, budget/group constraints, input-order invariance, invalid probabilities/costs, duplicate identities, excessive candidate count and empty-positive-value behavior.

This is an exact finite-set reference oracle, not a production scheduling algorithm. All expected values are model-based; uninspected wafers remain unknown, and the fixture is not evidence of physical savings or fab yield.


## Adaptive inspection decision oracle

The independent helper `python/semi_inspection_decision_reference.py` evaluates a finite candidate set using an explicit expected-net-value formula. It includes risk probability, test sensitivity/specificity, actionability, exposed volume, avoidable loss, false-hold cost, inspection cost and delay penalty. It selects the exact best subset subject to cash, time and per-group capacity limits, using exhaustive enumeration capped at 18 candidates. The matched-budget risk-only comparator is emitted separately.

The synthetic fixture selects A/C/F under the 45-minute / 20-cost-unit budget for modeled net value 1,286.16. The risk-only greedy comparator selects A/B/D/F with modeled net value 607.05. These are fixture outputs, not observed factory savings. Ten tests cover known answers, budget/group constraints, deterministic ordering and malformed/empty cases.


## Source-local naive chronology

The split-manifest validator supports two explicit time bases for forward-time manifests:
- offset_aware: timestamps must include an explicit timezone offset;
- source_local_naive: timestamps must be timezone-free and the manifest must declare a non-empty clock_domain_id.

The second mode proves only ordered, non-overlapping chronology within that one source clock. It does not convert timestamps to UTC or permit comparisons across sites/clock domains. This is needed for sources such as SECOM whose timestamps provide a date/time string but no timezone offset. Tests reject silent timezone assumptions and require the clock-domain declaration.


Implementation compatibility note: non-temporal v1 split manifests retain their original canonical payload and known-answer digest. The `time_basis` and `clock_domain_id` fields are included only for `forward_time` manifests, preserving compatibility for existing group-disjoint and stratified fixtures.


## Frozen real-source split manifest

A source-level chronological SECOM manifest now accompanies the synthetic split fixtures at `docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json`. It binds sample row IDs, labels, source-local naive timestamps, source commit and a canonical file-manifest digest. A dedicated test validates the actual manifest and asserts the 940/313/314 partition counts and 76/11/17 failure support. The 11 split-manifest tests bring the full reference suite to **35 tests**.


## Latest hosted verification

The updated reference suite contains **35 tests** and passed in hosted GitHub Actions run [#44899113316](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44899113316). This run includes the frozen row-level SECOM source-clock manifest test. It also reproduced the binary-metric and inspection-decision reports byte-for-byte. No local test execution is claimed.


## Split-local contrastive-pair validator

The independent helper `python/semi_pair_manifest_reference.py` validates pair provenance and split boundaries for graph/contrastive wafer learning. It requires each endpoint to exist in the frozen split manifest, both endpoints to share the declared pair scope, unique sample/pair/unordered-edge identities, no self-pairs, and valid positive/negative relations. If a pair uses production-proximity evidence, its reason must be supported by matching lot identity or timestamps within a declared embargo and clock domain.

The known-answer input/output are `fixtures/contrastive-pair-manifest-known-answer-v1.json` and `fixtures/contrastive-pair-manifest-known-answer-v1.expected.json`. Eleven tests cover cross-split pairs, scope mismatch, duplicate identities/edges, orphan references, self-pairs, same-lot evidence and time-embargo violations.

This establishes pair-manifest integrity only. It does not establish that a positive or negative pair is semantically correct, or that a trained embedding/model performs well.


Reproduce the pair-manifest validation report:

    python benchmarks/semi-bench-001/reference/python/semi_pair_manifest_reference.py \
      benchmarks/semi-bench-001/reference/fixtures/contrastive-pair-manifest-known-answer-v1.json


## Contrastive-pair provenance validator

The current branch adds the split-local pair manifest validator and an executable CLI. Local verification of the new component: **11/11 unit tests passed**, and CLI output matched the frozen expected report byte-for-byte. The combined suite contains 46 tests; the last hosted CI run passed the previous 35-test suite, before this addition. The new full-suite CI result is not yet verified.


## Current complete hosted verification

The clean-base PR #13126 run [#44917273962](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44917273962) passed all 46 unit tests. The job also reproduced the binary-metric, inspection-decision and contrastive-pair validation reports byte-for-byte. The exact tested commit is `5f8ffa91bac2cb8cee923d0f1968d2f57fd01149`.


## Latest Rust-native integrity gate

Hosted GitHub Actions run [#44934333447](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44934333447) passed on commit `4a7f7029fa8694914f0c85650053d879e399b402`:

- Rust format check: pass.
- Clippy with warnings denied: pass.
- Rust unit tests: **18 passed, 0 failed**.
- Rust binary metrics CLI report: byte-for-byte match with the frozen independent Python reference.
- The Rust split validator verified the synthetic group-disjoint known answer and the actual 1,567-row source-local SECOM chronological manifest, including 940/313/314 row counts and 76/11/17 failure support.
- The Rust contrastive-pair validator passed known-answer, cross-split, duplicate-edge, same-lot, time-embargo and clock-domain tests.
- The Python/reference job in the same run passed 46/46 tests and reproduced its three frozen reports.

These checks ran in a minimal temporary Cargo workspace because the repository root workspace currently references a missing `crates/core/symthaea/Cargo.toml`. The isolated workspace avoided changing unrelated root workspace state. This verifies contract and leakage logic, not a trained model's predictive performance or physical fab outcome.


## Latest complete Rust verification

Hosted run [#44935276557](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44935276557) passed on code commit `162a96cd423daa1f4f9297f00a32e434cd19a5f9`:

- Rustfmt passed.
- Clippy passed with warnings denied.
- **23/23 Rust tests passed** across binary metrics, split-manifest integrity, contrastive-pair integrity and SECOM raw-file parsing.
- The binary-metrics CLI report matched its frozen reference byte-for-byte.
- The SECOM raw-audit CLI report matched its synthetic expected JSON byte-for-byte.
- The Python/reference lane passed **46/46 tests** and its three frozen report comparisons.

The latest PR branch head includes later documentation/manifest/source-register updates; the verified Rust code, fixtures and workflow are unchanged from the tested code commit. The full 1,567×590 feature matrix was not retrieved in this runtime, so the parser was verified with synthetic bytes and the frozen real-source split manifest, not with a full real-file model run.


## Current scale-aware Rust metric evaluator

The Rust binary-metrics scorer uses tie-group rank accumulation for AUROC: O(n log n) sorting/aggregation and exact half-credit for tied positive-negative score pairs. A 40,000-row synthetic test with balanced tied score groups is included. Hosted run [#44988247262](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44988247262) passed 24/24 Rust tests and 46/46 Python reference tests, along with fmt, Clippy and both Rust CLI report parity checks.


## Rust DTCO evidence-lineage contract

The new `crates/domains/symthaea-semiconductor/src/benchmark/dtco_manifest.rs` validates a multi-fidelity DTCO evidence graph. It binds run identity, stage, evidence kind, tool/configuration/model digests, parent run IDs and exact artifact digests. Every parent output must be present among the child inputs; run graphs must be acyclic. Physical measurement nodes require instrument identity, calibration identity and an uncertainty artifact. Simulated/model evidence cannot claim to be a physical measurement.

Frozen input/output:
- `benchmarks/semi-bench-001/reference/fixtures/dtco-lineage-known-answer-v1.json`
- `benchmarks/semi-bench-001/reference/fixtures/dtco-lineage-known-answer-v1.expected.json`

Eight Rust tests cover the known answer and adversarial cases (missing parents, disconnected digest edges, cycles, missing calibration, evidence-kind mismatch, missing model digest and malformed SHA-256). Hosted run [#44988500047](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44988500047) passed all **32/32 Rust tests**, the 46-test Python reference suite, rustfmt, Clippy and both Rust CLI report reproductions.

This is a provenance/lineage contract only. It does not validate physics, the accuracy of any model, a PDK, a fab process or fabricated silicon.


## Rust DTCO lineage CLI

The crate now provides `semi-dtco-lineage`, a standalone CLI that consumes a versioned DTCO lineage manifest and emits a canonical key-sorted JSON report. CI runs the CLI against `dtco-lineage-known-answer-v1.json` and checks the full output byte-for-byte.

Hosted run [#44988794403](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44988794403) passed 32 Rust tests, 46 Python reference tests, fmt, Clippy, and the binary-metric, SECOM raw-audit and DTCO-lineage CLI report checks.
