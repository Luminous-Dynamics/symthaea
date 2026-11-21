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

The synthetic 10-validation / 10-test known-answer case is frozen in `fixtures/validation-threshold-known-answer-v1.json` and `fixtures/validation-threshold-known-answer-v1.expected.json`. Its 3 tests bring this reference suite to 14 tests in total. The tiny fixture demonstrates semantics only; it says nothing about real threshold reliability on rare-failure datasets.
