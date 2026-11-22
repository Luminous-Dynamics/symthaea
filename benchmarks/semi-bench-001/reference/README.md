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


## DTCO lineage gate — uncertainty, stage direction and root identity

The current Rust contract has been strengthened beyond parent/output digest matching:
- model-prediction evidence requires both a model digest and a prediction-uncertainty artifact digest;
- a child may not name a later-fidelity stage as its parent;
- each root run must bind the declared immutable design-input digest;
- the declared design-constraint digest must be consumed by at least one run;
- physical measurements still require instrument, calibration and measurement-uncertainty references.

The frozen DTCO fixture now carries the design and constraint digests as root-run inputs (seven distinct artifacts). Eleven DTCO-specific tests cover the known answer and adversarial lineage, stage, uncertainty and binding cases.

Hosted run [#44989349728](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44989349728) passed **35/35 Rust tests** and **46/46 Python reference tests**, Rust fmt and Clippy, plus byte-for-byte output comparisons for Python binary/inspection/pair reports and Rust binary/SECOM/DTCO CLI reports. The root workspace limitation remains isolated from this crate-level run.


## Current exact-head verification

Hosted run [#44989451987](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44989451987) passed on the latest PR head: 46/46 Python reference tests, 35/35 Rust tests, rustfmt, Clippy and all six report comparisons. The DTCO lineage known answer covers design/constraint binding, model uncertainty, monotonic fidelity, acyclic parent graph and measurement provenance.


## Budgeted multi-fidelity acquisition CLI

The `semi-mf-acquire` CLI consumes a versioned subject with exact prediction packet, model, uncertainty-calibration and objective-specification digests and emits a canonical report for the exact budgeted subset oracle. The known-answer five-candidate example selects B/E under budget 13, with modeled utility 0.3115.

Hosted run [#44989958104](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44989958104) passed 43/43 Rust tests, 46/46 Python reference tests, fmt, Clippy and the multi-fidelity acquisition report diff. Synthetic fixture only; it is not an actual SPICE/TCAD run or design approval.


## Verified budgeted multi-fidelity acquisition baseline

The Rust crate now contains `mf_acquisition.rs` and CLI `semi-mf-acquire`. It exactly selects a candidate subset under a high-fidelity cost budget (up to 18 candidates) using:

`expected_feasible_hv_gain + exploration_weight × epistemic_stddev × frontier_relevance`

The acquisition subject binds the prediction packet, model, uncertainty-calibration artifact and objective specification by SHA-256. The output is a scheduling recommendation only; it does not execute a simulator or approve the design.

On the frozen five-candidate synthetic fixture, budget 13 selects B/E (cost 13; modeled utility 0.3115). This is only a known-answer result.

Hosted run [#44990125209](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44990125209) passed **43/43 Rust tests**, **46/46 Python reference tests**, rustfmt, Clippy and all seven CLI/report comparisons (three Python and four Rust).


## Latest exact-head verification

Hosted run [#44990424342](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44990424342) passed on commit `f5073aab67d8aaf987bfe42d74d0ebd43365cf6e`: **43/43 Rust tests**, **46/46 Python reference tests**, rustfmt, Clippy with warnings denied, and all seven byte-for-byte report checks. The four Rust CLI reports include the multi-fidelity acquisition selector and its calibration-digest-bound fixture.


## SECOM missingness-only shortcut baseline

The Rust crate includes \`semi-secom-missingness\`, which verifies the digest-bound raw files and split manifest, then compares two held-out baselines:

- **B0 prevalence-only:** a constant failure probability fitted from training labels.
- **B1 missingness-only:** a smoothed Bernoulli naive-Bayes model fitted from the training partition's per-sensor missingness mask. It does not read numeric sensor values.

The report includes validation/test PR-AUC, ROC-AUC, Brier/ECE, threshold metrics, risk-coverage, per-split missingness profiles, per-sensor missing fractions, and row-level scores. It also binds the raw feature/label digests, split-manifest digest and fitted-model digest. The held-out labels are used only for evaluation; changing them cannot change model parameters or row scores.

Run against the pinned SECOM mirror after obtaining its raw files:

\`\`\`sh
cargo run -p symthaea-semiconductor --bin semi-secom-missingness -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

The checked-in eight-row fixture is synthetic and exists only to verify scorer semantics and leakage controls. It is not a SECOM model result. B1 is a simple diagnostic comparator, not a competitive production classifier; Bernoulli conditional independence is unlikely to hold across correlated sensors. A high missingness-only score is evidence of an observation-pattern shortcut on the evaluated split, not evidence that missingness is a physical failure cause. A low score does not prove the mask is harmless under other acquisition regimes.

The chronological test contains only 17 failures in the pinned mirror. Report uncertainty and avoid model ranking or threshold claims from that small support. This baseline does not replace the remaining B2/B3 imputation, shift, MAR/MNAR and independent-dataset comparisons in SEMI-SECOM-MISS-001.


## SECOM observation-regime drift diagnostic

The Rust CLI \`semi-secom-observation-drift\` compares the training observation mask with validation and test without fitting a predictive model. It reports:

- pooled missing-cell fraction and signed train-to-held-out change;
- mean and maximum per-sensor missing-rate differences;
- mean and maximum per-sensor Bernoulli Jensen–Shannon divergence (normalized to bits in [0, 1]);
- number of sensors whose missing rate changes by at least five percentage points;
- change in mean missing features per row;
- Jensen–Shannon divergence of the row-level missing-feature-count distribution.

The five-percentage-point alert threshold is an explicit diagnostic setting, not a learned threshold or a process alarm. The JS metrics are marginal summaries; they do not capture the full joint dependence structure of the 590-dimensional missingness mask. A large drift statistic indicates that the observed measurement pattern changed. It does not identify whether the cause was sensor failure, measurement policy, process change, data handling or chance variation.

Run against the same pinned raw-file and split-manifest identities:

\`\`\`sh
cargo run -p symthaea-semiconductor --bin semi-secom-observation-drift -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

The synthetic fixture checks zero-drift reproduction, a one-sensor observation shift, label invariance, feature-permutation invariance and divergence bounds. It is not a real-SECOM drift result. The report is an unsupervised monitoring diagnostic and must not be used alone to trigger a production process change.


## SECOM train-only value baselines B2–B4

The Rust CLI \`semi-secom-tabular-baselines\` evaluates three additional baselines on the same digest-bound raw files and frozen split:

- **B2 — values-only:** per-feature median imputation fitted on training rows, train-only value normalization, and Gaussian naive Bayes. No missingness-indicator likelihood is used.
- **B3 — values + mask:** same value path as B2 plus smoothed Bernoulli likelihoods for each sensor's missingness state.
- **B4 — zero + mask:** zero-imputed values plus the same type of missingness-indicator likelihood.

Each model has an explicit variance floor, train-fitted class priors and class-conditional statistics, a serialized model SHA-256, and held-out validation/test metrics from the existing binary scorer. A feature with no observed training values uses a deterministic zero fallback and is counted in the report. Non-finite standardized values or likelihoods are rejected rather than clipped.

Run against the pinned mirror:

\`\`\`sh
cargo run -p symthaea-semiconductor --bin semi-secom-tabular-baselines -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

B2–B4 use Gaussian/Bernoulli conditional-independence assumptions and are deliberately transparent statistical comparators, not production models or claims of SOTA. B3/B4's mask likelihood can be miscalibrated when sensor missingness is strongly correlated; judge calibration and transfer on held-out domains, not accuracy alone. The 17 failures in the current chronological test require support counts and uncertainty intervals before ranking. No feature has known physical semantics in the public SECOM mirror, and no association establishes root cause.


### B2–B4 comparability and interpretation

All three baselines share the exact same frozen rows and binary scoring contract. The Gaussian likelihoods operate on values standardized by training-only population mean/scale; per-class variance uses training-only MLE variance with an explicit floor of \`1e-6\`. B3/B4 add independent Bernoulli missingness likelihoods with symmetric Laplace smoothing \`alpha=1\`. These assumptions are exposed in the fitted-model digest/report so they can be reproduced or replaced by a new model schema.

For comparison, report B0/B1/B2/B3/B4 side by side. The prevalence-only AP reference equals the positive prevalence on the scored split; a B1 AP below this reference is not useful ranking performance. A high B1 score with poor held-out calibration must be treated as a failure, not converted into a threshold recommendation.

The B2/B3/B4 chronological test has only 17 failures. Before leaderboard claims, add lot/tool/time groups from a source that actually carries those identities, multiple independent time periods, confidence intervals and a second independent dataset. The anonymous SECOM mirror cannot establish fab transfer.


## SECOM discriminative logistic baseline B5

The Rust CLI \`semi-secom-logistic\` adds two train-only FTRL-Proximal logistic models over a shared feature transform:

- **B5 unweighted:** L2-regularized logistic regression, with training-median value imputation and training-standardized values plus per-sensor missingness indicators.
- **B5-balanced:** the same model and transform with inverse-frequency training loss weights. Its emitted log-odds receive the analytic training-prior correction \`ln(weight_negative / weight_positive)\`; the correction is fitted from training class counts only.

The model uses a frozen FTRL configuration (80 epochs, alpha 0.05, beta 1.0, L1 0, L2 1.0). Training rows are sorted by stable sample identity before gradient accumulation, making the serialized weights reproducible under input-row permutation. The report binds the raw feature/label digests, split-manifest digest and model SHA-256.

Run against the pinned mirror:

\`\`\`sh
cargo run -p symthaea-semiconductor --bin semi-secom-logistic -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

B5 is a strong discriminative challenger to B2–B4, but it is still a linear model over anonymous sensor columns. Class weighting changes the training objective; the prior correction restores the training class prior under the ideal weighted-likelihood identity, but does not guarantee calibration under covariate/observation shift. Validate Brier/log-loss and high-confidence errors in addition to AUROC/AP. Do not tune the frozen epoch count, regularization or thresholds on the final chronological test.


### Sparse Elastic-Net B6 extension

The same CLI now also evaluates two B6 variants using a frozen FTRL elastic-net setting:

- **B6 sparse:** unweighted FTRL-Proximal with L1 = 0.01 and L2 = 1.0.
- **B6 sparse-balanced:** inverse-frequency class-weighted FTRL with the same L1/L2 penalty and training-prior correction.

B5 retains L1 = 0 and L2 = 1.0. B6 reports the count of coefficients whose absolute value exceeds \`1e-12\`, so sparsity is measurable rather than inferred from the selected algorithm name. The synthetic and full pinned-source workflow checks assert both B6 variants and their declared L1 configuration.

B6 is a fixed research comparator, not a validation-selected hyperparameter winner. The L1 penalty has not been tuned on the final chronological test. Feature selection/zero coefficients do not establish physical sensor importance because the public SECOM mirror does not carry interpretable sensor semantics.


### Sparse Elastic-Net B6 extension

The \`semi-secom-logistic\` CLI also evaluates:

- **B6 sparse:** unweighted FTRL-Proximal with L1=0.01 and L2=1.0.
- **B6-balanced:** the same sparse penalty with inverse-frequency class weights and the training-prior correction.

B5 retains L1=0 and L2=1.0. B6 reports the number of nonzero coefficients (absolute weight > 1e-12), so sparsity is measured rather than assumed from the algorithm name. The synthetic and full pinned-source workflow checks assert the two B6 variants and their fixed L1 configuration.

B6 is a preregistered research comparator, not a validation-selected hyperparameter winner. No tuning on the final chronological test is permitted. Zero coefficients do not establish physical sensor importance because the anonymous public SECOM mirror does not provide sensor semantics.


### B6 sparsity audit and stronger regularization arm

The first B6 arm used L1=0.01 and L2=1.0. Its full pinned-source report showed 894 nonzero coefficients for all four B5/B6 variants and the identical coefficient-support SHA-256:

\`d9d8dcd4e3d330408be5f34ffd79211090dd8f94e69126b63cd8e6cac9dfb460\`

Therefore L1=0.01 did not change feature support; it is retained as a near-zero-penalty control, not evidence of meaningful sparsity.

A second fixed arm is now added:
- **B6-strong:** unweighted FTRL with L1=1.0, L2=1.0.
- **B6-strong-balanced:** class-weighted FTRL with L1=1.0, L2=1.0 and the same training-prior correction.

Both arms report nonzero count and SHA-256 over the binary coefficient-support mask. The exact-head workflow evaluates both on the same chronological split and archives model support and metrics. The stronger setting is preregistered, not selected by test performance. If it produces severe sparsity, predictive collapse or calibration failure, that is a negative result to preserve.


## Repeated stratified SECOM 80/20 manifests

The Rust CLI \`semi-secom-stratified-splits\` creates five two-way stratified train/test manifests for the published-comparability track in issue #13234. It first audits the pinned raw feature/label files against the exact expectations digests and binds the existing dataset snapshot digest.

The split algorithm:
- canonicalizes sample identity by sorting sample IDs before randomization;
- stratifies pass and fail rows independently;
- uses SplitMix64 plus Fisher–Yates with separate deterministic class salts;
- rounds each class's test count independently (for the 1,567-row mirror this yields 21/104 fails and 293/1,463 passes in test);
- emits train/test assignment manifests sorted by sample ID;
- binds each manifest's seed, PRNG identifier, dataset snapshot and assignments with SHA-256.

The current reconstruction uses seeds 42–46 and test fraction 0.2. The 2026 article reports five repeated stratified 80/20 splits but does not state the original five SECOM seeds. These are therefore Luminous reconstruction seeds, not claimed to be the authors' seeds. No exact-reproduction claim is possible until all unpublished model/preprocessing details are resolved.

Run on the pinned source:

\`\`\`sh
cargo run -p symthaea-semiconductor --bin semi-secom-stratified-splits -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

Each report records per-repeat train/test sample counts and positive/negative support, plus independent manifest digests. Input ordering does not change manifests. The published-comparability split has no validation partition; thresholds must not be selected on its test split. Any hyperparameter selection requires nested CV or a separately declared inner split within the training partition.

These stratified repeats measure interpolation across historical samples, not time-forward generalization, and the five repeats are not five independent lots or fabs. Continue to report the original chronological track separately.


## Train-only preprocessing for the five-repeat stratified SECOM track

The Rust CLI \`semi-secom-fold-preprocess\` consumes the digest-verified raw SECOM files, the five-repeat stratified report and the source-identity manifest. It validates the snapshot-manifest digest and every source row's sample ID, fail/pass label and naive source-local timestamp before creating per-repeat transforms.

For each repeat, all preprocessing parameters are fitted on that repeat's 1,253 training rows only:
- drop a feature when its training missing fraction is strictly greater than 0.50;
- median-impute using observed training values only;
- drop near-zero-variance features using the declared minimum population variance (1e-12) and minimum distinct-value count (2);
- fit a complete empirical quantile reference over training-only median-imputed values;
- apply the quantile map to training and held-out rows without refitting.

The default output is an approximate normal-score empirical quantile transform using midranks and half-rank tail clipping. Uniform-rank output is also available through the typed configuration. The article does not disclose all quantile-transform defaults or exact variance thresholds, so these settings are Luminous reconstruction parameters, not claimed to be the authors' exact implementation.

Two matrices are hashed independently:
- **Feature Set A:** quantile-transformed retained numeric values; the paper-comparability path.
- **Feature Set B:** Feature Set A plus retained-feature missingness indicators and one missing ratio over all original columns. This is a Luminous ablation, not attributed to Dong et al.'s exact pipeline.

The fitted pipeline digest binds train sample IDs, config, retained feature identities, train medians, variances, distinct counts and full sorted quantile references. Reports bind source snapshot/identity-manifest digests, repeat index/seed and train/test matrix digests. Test values/labels do not affect the fitted pipeline digest.

Run:

\`\`\`sh
cargo run -p symthaea-semiconductor --bin semi-secom-fold-preprocess -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  stratified-80-20-repeats.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

This verifies fold-specific preprocessing and source binding; it does not train XGBoost or Random Forest. The 2026 paper's exact five seeds, full quantile-transform defaults and near-zero variance threshold are not fully disclosed in the accessible protocol, so direct numerical reproduction remains pending. Results from this stratified protocol must not be compared directly to the frozen chronological 940/313/314 results.


## Native deterministic Random Forest on repeated stratified folds

The Rust module \`benchmark::random_forest\` implements a deterministic bootstrap CART forest with a fresh \`mtry\` sample at each split. It exposes continuous scores from the unweighted positive-class frequency in each reached leaf, averaged over trees. It does not convert hard class labels into fabricated probabilities.

The preregistered real-source runner uses:
- 200 bootstrap trees per fold;
- max depth 12;
- minimum leaf size 4;
- \`mtry = floor(sqrt(p))\` at every split;
- fixed seed inherited from each reconstructed stratified repeat;
- Gini split criterion;
- two variants: unweighted split impurity and inverse-frequency class-balanced split impurity;
- raw empirical leaf frequencies for scoring in both arms (therefore scores are not asserted to be calibrated).

Class balancing modifies split selection only. The leaf score retains unweighted empirical positive frequency from bootstrap samples. Brier/ECE are evaluated rather than assumed to be valid posterior probabilities.

Each fitted model binds training IDs, resolved mtry, parameters, bootstrap trees and their split structure into a model SHA-256. Each score report binds sorted sample IDs, source labels and scores into a score-vector SHA-256. Fit and inference milliseconds are recorded outside the model digest. The existing independent binary metric scorer computes AP, AUROC, threshold metrics, Brier, ECE, FPR at target recall and risk/coverage.

Run on the digest-verified pinned source:

\`\`\`sh
cargo run --release -p symthaea-semiconductor --bin semi-secom-random-forest -- \\
  data/secom.data \\
  data/secom_labels.data \\
  docs/research/semi-bench-001/secom-raw-expectations-v1.json \\
  stratified-80-20-repeats.json \\
  docs/research/semi-bench-001/splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json
\`\`\`

The CLI reruns train-only preprocessing per repeat from #13238, then fits/evaluates both RF variants on Feature Set A. Feature Set B remains a separate Luminous missingness-indicator ablation. No hyperparameter or threshold is selected on test data.

**Claim limit:** this is a native Rust Random Forest comparator, not an XGBoost substitute. It evaluates stratified interpolation on the historic SECOM mirror, not chronological or fab transfer. Leaf-frequency scores are not calibrated defect probabilities. The five seeds and preprocessing defaults are Luminous reconstructions because the cited article does not disclose every setting. For benchmark interpretation, keep these scores separate from the frozen source-clock chronological results.


### First exact-head five-repeat Random Forest result (run #45062680083)

The full 200-tree real-source experiment passed exact-head CI with 88/88 Rust tests, format, strict Clippy, raw source audit, split/preprocessor verification and all ten model runs (five folds × unweighted/balanced).

| Variant | AP mean ± SD | AUROC mean ± SD | Brier mean ± SD | ECE mean ± SD |
|---|---:|---:|---:|---:|
| RF 200 unweighted | 0.1459 ± 0.0386 | 0.7004 ± 0.0865 | 0.06119 ± 0.00261 | 0.02066 ± 0.01657 |
| RF 200 balanced splits | 0.1967 ± 0.0807 | 0.7030 ± 0.0939 | 0.05981 ± 0.00237 | 0.01384 ± 0.00719 |

For every fold, the training prevalence was 83/1,253 = 0.06624 and outer-test prevalence was 21/314 = 0.06688. A constant predictor using the train-fold prevalence has Brier 0.06241 on this test support. Both RF arms are modestly better on average; the balanced arm's mean Brier improvement is approximately 0.00259. This is a descriptive difference only; paired confidence intervals and repeated-fab evidence are not yet available.

The balanced model's AP spans 0.1179–0.3021 across the five repeats (sample SD 0.0807). The AP result is therefore substantially more variable than the unweighted arm (0.0816–0.1803). At the fixed diagnostic threshold 0.5, both arms have TP=0 / FN=21 in every repeat. This demonstrates that ranking/probability diagnostics and operational threshold utility are different questions.

The paper-reported 2026 Dong et al. SECOM RF AP is 0.431 ± 0.041; the local reconstruction remains materially below that published number. The original paper split seeds and complete preprocessing defaults are not disclosed, so these are not matched partitions and no SOTA claim is made. Park & Baek's distinct 2026 RF result is also a different split/imputation protocol. Preserve these as external comparators, not directly reproducible equivalence claims.

Artifact #14769912968 (SHA-256 \`6c0590e4725f1507e5aece893e46196ea31241dce9bf81ccb6d9525509b14f8d\`) contains the archived run reports, including fold-level prediction scores and model digests.
