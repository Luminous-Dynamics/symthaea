# Cognitive Loop Evidence & Adaptation Benchmark

**Status:** proposal / benchmark contract, not measured results  
**Owner:** Symthaea  
**Purpose:** test whether the cognitive loop learns from valid outcomes, calibrates uncertainty, reacts to real distribution shifts, and avoids adapting to missing or corrupted evidence.

## Executive thesis

The next architectural step is to make the cognitive loop **evidence-driven rather than scalar-driven**. A prediction error (PE) is not a self-authenticating truth: it needs a declared comparison space, a target, a validity state, and provenance. Treating `1.0` as interchangeable between genuine maximum error and a missing/zero-norm prediction makes downstream behavior impossible to interpret and can contaminate attention, urgency, learning, and replay.

The proposed improvement is a typed evidence contract feeding an explicit, single-writer phase schedule:

1. **Measure:** produce a PE estimate plus validity/reason, comparison-space identity, and source/target provenance.
2. **Interpret:** derive surprise, uncertainty, and urgency only from evidence that supports those interpretations; carry prior estimates as priors, never as new observations.
3. **Act:** allow each controller to declare which evidence types it consumes.
4. **Observe outcome:** distinguish an action, a predicted outcome, and the outcome actually observed.
5. **Learn:** update a given model or policy only when the required evidence is present.
6. **Audit:** emit a replayable per-cycle record of evidence, gate decisions, stateful update counts, and downstream effects.

The core hypothesis is falsifiable: **relative to the legacy raw-scalar baseline, the validity-aware loop will reduce false adaptation and improve calibration/recovery under missingness and distribution shift without hurting nominal task performance or exceeding a small latency budget.**

## What must be compared

Do not compare this internal loop directly to a model's public task-success leaderboard and call it an apples-to-apples SOTA result. Use a layered benchmark:

- **Core mechanism:** synthetic, deterministic PE streams with known ground truth.
- **Continual adaptation:** task sequences with a stateless control and identical resources.
- **End-to-end capability:** established agent benchmarks under their published configurations.
- **Systems behavior:** latency, memory, resource cost, stability, and replayable evidence.

## Baseline matrix

Every result must include these configurations, with the same model weights, data, seeds, and task budget where applicable:

| ID | Configuration | What it isolates |
|---|---|---|
| B0 | Legacy raw-scalar behavior | Current/reference failure mode; preserve as an ablation, not production behavior |
| B1 | Validity gate only | Whether invalid PE can affect learning/adaptation |
| B2 | Single-update phase schedule only | Whether repeated stateful calls were affecting behavior |
| B3 | Validity + single-update + gated replay | Candidate implementation |
| B4 | B3 with confidence/uncertainty calibration | Value of calibrated predictive confidence |
| B5 | Stateless/no-online-learning control | Whether memory/adaptation adds value instead of merely changing scores |
| B6 | Persistence/last-valid-PE control | Simple non-learned reference for temporal prediction |
| B7 | Randomized or shuffled outcome control | Detects accidental leakage or spurious gains |

For each ablation, state explicitly which code path changed. Do not silently change models or task difficulty between rows.

## Benchmark suites

### S0 — Evidence validity and invariant tests (hard gate)

Use deterministic generated cases with a known oracle: first cycle/no prediction, empty prediction, zero-norm prediction, dimensional mismatch, non-finite input, valid identical prediction, valid orthogonal prediction, valid anti-parallel prediction, and ordinary distribution shift.

Required assertions:

- Invalid/missing comparisons do **not** enter PE EMA/history/high-error counters.
- Invalid comparisons do not update PE-trained attention, confidence, exploration, flow-state, curiosity, self-reflection, or plan-effectiveness state.
- Invalid transitions do not become replay experiences labelled as zero-error successes.
- Current state/prediction windows still advance where needed for the next valid comparison.
- A valid comparison still updates each intended consumer.
- The raw diagnostic sentinel, control estimate, and validity flag remain distinguishable.
- Each stateful phase runs at most once per cycle. A duplicate call should fail a counter assertion in the harness.
- NaN/Inf never cause NaN/Inf to enter the evidence record or control outputs.

**Initial release target:** zero invariant violations across at least 10,000 generated cycles × 32 deterministic seeds. Any invariant violation fails the run; averages cannot hide a safety failure.

### S1 — Calibrated prediction and surprise detection

Generate stable, periodic, noisy, non-stationary, and abrupt-change sequences with labelled change points. Include missing-data bursts and corrupt predictor outputs as separate conditions, not as “high-error” labels.

Record:

- PE error under the metric appropriate to the comparison space (e.g. cosine/angular error for unit vectors; normalized squared error only where magnitude is meaningful).
- Brier score and negative log-likelihood for actual probabilistic event forecasts.
- Reliability diagram and ECE with bins fixed on a validation split; report bin count and sample count.
- Precision/recall, false-positive rate, detection delay, and AUROC/AUPRC for change detection.
- Risk–coverage curve: how task error changes when the system abstains or requests more evidence at lower confidence.
- False-surprise rate on stable streams and missed-change rate on real shifts.

Do **not** call raw PE “calibrated uncertainty.” Calibration scores are valid only when Symthaea emits a probability/forecast for a defined event and is scored against the observed event label.

**Initial release targets (aspirational, must be measured):**

- False PE-driven surprise rate ≤ 1% on stable and missingness-only streams, measured separately.
- Change-point recall ≥ 95% on the pre-registered synthetic shift suite at the chosen operating point.
- At least 10% relative Brier-score improvement over B0/B6 on held-out probabilistic forecasts, without worsening log loss.
- Report ECE; target ≤ 0.05 after calibration on the held-out in-distribution validation set. Separately report OOD performance, where this target may not hold.

### S2 — Continual learning and recovery

Use sequential tasks with repeated motifs, distractors, delayed rewards, and controlled distribution shifts. Each task sequence must have a stateless baseline, and the same sequence must be run with adaptation disabled and enabled.

Measure:

- Reward / completion trajectory by task index.
- **Gain over the same system's stateless baseline**, not just aggregate reward.
- Samples/interactions to recover a previous performance level after a shift.
- Catastrophic forgetting on previously mastered tasks.
- Replay-buffer contamination by invalid or unlabelled outcomes.
- Learning curves, not only final scores; cost, wall time, and memory per solved task.

**Initial release targets (aspirational):**

- ≥ 10% relative gain over B5 after 50 labelled interactions on held-out task sequences.
- ≥ 20% lower median recovery time after distribution shift vs B0.
- No increase in catastrophic forgetting; quantify it instead of treating it as “not observed.”
- Zero replay records generated from invalid comparison events.

### S3 — Cognitive phase causality / ablation

Run counterfactual paired trials. Hold input, state, seed, and all non-PE signals fixed; inject only an invalid PE sentinel. Compare with a control in which the PE measurement is absent. Repeat with a genuine surprise signal.

The key metric is **causal leakage**: the fraction of PE-dependent state changes attributable to invalid evidence. The desired value is exactly zero for PE-gated consumers. Independently driven mechanisms (for example, safety, task deadlines, or sensory novelty) are allowed to act and must be logged as such.

Required ablations: validity gate off/on, duplicate stateful update off/on, replay admission off/on, calibrated threshold off/on, and each downstream controller individually. Emit a table linking each metric to the exact state fields it checks.

### S4 — End-to-end capability and industry comparability

Use external benchmarks for the questions they actually answer:

- **Continual Learning Bench (CLBench):** relevant to sequential learning because it compares stateful task performance against a stateless baseline and reports reward gain and cost.
- **SWE-bench Verified:** the benchmark describes a human-validated 500-instance subset and offers a controlled bash-only comparison setting. However, Epoch AI's September 2026 benchmark-review index labels SWE-bench Verified *Flawed*. Treat its leaderboard as a diagnostic reference, audit the pinned tasks/test patches and disagreement cases, and report results both on the published set and a clearly declared audited subset. Do not quote the raw score as uncontested ground truth.
- **OSWorld / OSWorld-Verified:** pin the exact release, environment, task set, grading logic, and budget. Epoch AI's September 28, 2026 review of OSWorld 2.1 marked it *Flawed*: its audit of 50 of 108 tasks found scoring issues in 36, with a possible impact up to 50.4% of the audited total score. Prefer a reviewed/pinned task subset and retain task-level traces rather than relying on a leaderboard delta alone.

These suites measure task execution, not PE validity itself. The core suites S0–S3 are needed to show *why* a cognitive-loop change helps or harms. Benchmark audits are part of the evaluation, not optional paperwork. No cross-suite combined score should be reported without a declared weighting and justification.

### S5 — Systems efficiency and stability

On fixed hardware, report p50/p95/p99 cycle latency, throughput, peak resident memory, replay-buffer size, allocations if instrumented, and the compute budget spent during stable vs surprise-heavy periods.

**Initial release target:** ≤ 2% p95 latency regression against B0 for the same configuration; separately report gains from removing duplicate calls. If it regresses beyond the budget, identify the feature responsible rather than hiding it in aggregate performance.

## Experimental protocol

1. Pin repository HEAD, configuration, toolchain, feature flags, hardware, dataset/task snapshot, and baseline revision.
2. Pre-register hypotheses, primary metrics, exclusion rules, seeds, and success thresholds before reading final results.
3. Use at least 32 deterministic seeds for synthetic suites; publish all per-seed results. For end-to-end task samples, use paired trials where possible.
4. Report mean and median, spread, bootstrap 95% confidence intervals, effect size, and failure counts. A hard-gate violation fails independently of averages.
5. Keep a held-out calibration split separate from evaluation tasks. Do not tune thresholds on the final test stream.
6. Retain raw records and digests for inputs, outcomes, model/config, metric version, and gate decisions. Every displayed aggregate must be reproducible from those records.
7. Compare against simple baselines before frontier agents; then run public benchmarks using matched, documented scaffolds. State when a comparison is not apples-to-apples.
8. CI being queued, skipped, or missing is **not** validation. A report records the exact head SHA and executed command/result.

## Result record contract

Each cycle record should make the measurement-to-action chain inspectable:

- run/sequence/seed and exact code/config identifiers;
- cycle ID and a monotonic event/update sequence;
- input and target digests plus comparison-space/model version;
- raw diagnostic PE, validity, invalid reason, control estimate, and prior source;
- each consumer's declared decision: accepted/rejected evidence and reason;
- stateful update counts by subsystem;
- training/replay admission decision and outcome label provenance;
- task outcome, latency, resource use, and output digest.

Use structured typed fields in code; don't implement this as log-string parsing. A record's existence is not proof that the recorded claim is correct—validate it independently.

## Definition of “better than SOTA”

For this project, “better” means a statistically supported Pareto improvement on a declared task and budget: higher task reward and/or lower error at matched resources, better calibrated uncertainty, fewer invalid-evidence adaptations, faster recovery, and no material regression on nominal tasks. We will not claim industry-leading results until the tests run against pinned public baselines and the full artifacts are available.

The differentiator is a **causal, evidence-audited cognitive loop**: one observation has one explicit validity state, each adaptive update can be traced to an accepted observation/outcome, and an independent evaluator can replay the causal chain.

## External references

- NIST, [AI measurement and evaluation](https://www.nist.gov/ai-measurement-and-evaluation): measurement science, task design, and evaluation methodology.
- Silva Filho et al., [Classifier calibration: a survey on how to assess and improve predicted class probabilities](https://link.springer.com/article/10.1007/s10994-023-06336-7): calibration, proper scoring rules, metrics, and reliability diagnostics.
- [Continual Learning Bench](https://continual-learning-bench.com/): sequential task learning, gain over stateless baseline, and cost.
- [SWE-bench Verified](https://www.swebench.com/verified): its stated human-validated subset and controlled bash-only comparison setup; pair it with the [Epoch AI benchmark-review index](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks), which marks SWE-bench Verified Flawed (2026-09-03).
- [OSWorld](https://os-world.github.io/): execution-based computer-use tasks; pin the exact version and task set.
- Epoch AI, [OSWorld 2.1 benchmark review](https://epoch.ai/benchmarks/osworld-2-1/review) (2026-09-28): review finding scoring issues in 36 of 50 audited tasks, underscoring that grader reliability must be measured.

## Machine-readable scenario manifest

`benchmarks/cognitive-loop-evidence/scenarios.json` is the implementation-facing companion to this contract. It currently defines **38 cases across six suites (S0–S5)**: validity invariants, calibration and distribution shift, continual learning, phase causality, end-to-end benchmark adapters, and system behavior. It includes the B0–B7 baseline matrix, including B4 for calibrated uncertainty. These are proposed cases and configurations; the manifest is not yet a runnable benchmark suite and no results are implied.


## External drift-detector baselines

B0–B7 are **internal ablations**, not state-of-the-art comparators. The shift-detection suite therefore also declares a separate external-reference track in the manifest. This distinction prevents a win over legacy Symthaea from being reported as a win over specialist drift detectors.

| Reference | What it tests | Input/label caveat | Status |
|---|---|---|---|
| Fixed threshold | Minimum viable alarm policy | Valid PE/residual only; threshold calibrated off scored seeds | Planned |
| EWMA control chart | Smoothed deviation and sensitivity/noise trade-off | Unlabelled scalar stream; calibration separated from evaluation | Planned |
| Page–Hinkley | Sequential mean-shift detection | Unlabelled numeric stream; primarily a mean-change test | Planned |
| ADWIN | Adaptive window-size change detection | Unlabelled numeric stream; valid-observation cadence must match | Planned |
| KSWIN | Two-window distributional change test | Unlabelled numeric stream; pin window/significance configuration | Planned |
| Discriminative Drift Detector (D3) | Fully unsupervised multivariate feature-space drift | Requires a pinned external adapter and common feature contract | Adapter required |
| Image-Based Drift Detector (IBDD) | Drift from a window-to-image representation | Transformation and representation costs must be included | Adapter required |
| Semi-Parametric Log-Likelihood (SPLL) | Likelihood change against a reference distribution | Numerical features; clustering/covariance assumptions must be documented | Adapter required |

The D3/IBDD/SPLL candidates come from a survey and real-world benchmark of fully unsupervised concept-drift detectors. That study emphasizes an important limit: feature-space methods cannot detect a pure change in the posterior (P(y\mid X)) unless it also changes the feature distribution, and it finds current drift metrics imperfect for incremental/gradual changes. See [Lukats et al. (2024/2025)](https://doi.org/10.1007/s41060-024-00620-y) and its [reproducibility repository](https://github.com/DFKI-NI/unsupervised-concept-drift-detection).

The harness must pre-register a disjoint calibration stream, pin versions/configs/seeds, and use the same valid observations. Do not map invalid PE to a numerical value for a detector to consume. Report false alarms per 1,000 valid observations, missed-drift rate, detection-delay distributions, precision/recall stratified by abrupt, gradual and recurring drift, coverage, throughput, peak memory, and per-seed event traces. Do not rank only by one aggregate such as Mean Time Ratio: detector-quality metrics have assumptions, and the cited review reports limitations for both MTR and lift-per-drift.

These are **planned comparators**, not implementations or results. The new Rust command validates that their contracts are present; it does not execute them.

## Manifest validator

The JSON manifest is now checked by a small Rust validator in `xtask`; this validates the experiment contract, **not benchmark performance**. Run it from the repository root:

```bash
cargo run -p xtask -- validate-cognitive-loop-evidence
cargo test -p xtask cognitive_loop_evidence
```

The validator fails closed on duplicate suite/case/baseline identifiers, missing B0–B7 controls, incomplete validity-case expectations, invalid evidence configured to update PE training or enter replay, hard-gate cases without assertions, missing result-provenance fields, and a CI policy that does not explicitly reject skipped checks. It prints a suite/case count only after the manifest contract passes. Keep `status` at `proposal-not-measured` until actual pinned runs and raw records exist. The command does not execute the scenarios and produces no SOTA claim.

## Current state-of-the-art drift comparisons and evaluation protocol

A KDD 2026 evaluation framework compares 14 detector families on seven real-world data streams across class-prior, label-swap, feature-permutation, and feature-filtering drifts, each abrupt and gradual. It recommends randomized drift onset and leave-one-dataset-out hyperparameter tuning; SEED and STEPD are strong F1 references, while ABCD offers a useful false-alarm trade-off. [Paper](https://doi.org/10.1145/3770855.3819070) · [Reproduction repository](https://github.com/vcerqueira/experiments-drift_evaluation).

The manifest pins the exact 14-family inventory and a seven-dataset name list for later version/licence verification: Asfault, Electricity, Covertype, GasSensorArray, NOAA, Posture, and Rialto. There are now **26 comparator definitions**, because several published families have separate error-stream and feature-stream contracts, and Symthaea also needs PE-only controls plus separate unsupervised and adaptive-threshold references. The published KDD-compatible error-stream track uses a pinned Hoeffding Tree with prequential test-then-train and an observed per-instance correctness/error stream; do not compare those scores directly to a different Symthaea PE stream. To compare detector algorithms fairly, run them over the same pinned error stream with identical label timing and update order.

The manifest now includes the complete eight-cell core drift matrix and the published KDD error-stream reference set (ADWIN, CUSUM, DDM, EDDM, EWMA, GMA, HDDMA, HDDMW, PH, RDDM, SEED, STEPD, ABCD and STUDD+ADWIN), with ABCD(X) explicitly represented as a feature-only variant. Dynamic Threshold Determination (DTD), D3/IBDD/SPLL, and simpler PE-only controls remain separate. These are not yet runnable adapters or measured results. DTD is especially relevant because its AAAI 2026 work argues that adaptive threshold selection can outperform any single fixed threshold in its studied settings ([paper](https://doi.org/10.1609/aaai.v40i29.39586)).

Detector comparisons are separated by **evidence access**: valid-PE-only, feature-only unsupervised, auxiliary-loss unsupervised, and observed-outcome. A feature-only detector cannot detect a pure target mapping change with no feature-distribution change. That should be measured as a boundary of observability, not “solved” by leaking labels.

## Timing-aware drift scoring

The Rust `xtask score-cognitive-loop-drift` command scores alarm traces against registered ground-truth episodes independently of the full cognitive loop:

```bash
cargo run --locked -p xtask -- score-cognitive-loop-drift --input path/to/drift-trace.json --out reports/drift-score.json
cargo test --locked -p xtask cognitive_drift_score
```

Trace schema v1 requires a detector ID and version, comparison/access track, outcome-label delay, full commit hash, configuration digest, seed, stream length, valid-observation count, maximum delay, ground-truth intervals, and alarm cycle indices with per-alarm evidence validity. Each event is scored against the interval `[start_cycle, end_cycle + max_detection_delay_cycles]`; one alarm can match at most one event, duplicate/unmatched alarms count as false positives, and unmatched events count as missed. Invalid/missing PE sentinels are not valid observations.

The scorer is intended to report event precision/recall/F1, false alarms per 1,000 valid observations, missed-drift rate, and a penalized normalized delay. The delay penalty is delay divided by transition width plus the post-window delay budget, clipped to `[0,1]`; missed events receive 1.0. This is our explicitly defined metric—not a claim to reproduce the paper's exact metric. Publish metric version and per-event matches beside aggregate results.

Incremental drift, recurring drift, and correlated sensor shifts are planned future extensions, not part of the current eight-cell core matrix.


## Paired comparison with uncertainty

Detector-level scores are not enough to justify a performance claim. The Rust `xtask compare-cognitive-loop-drift` command consumes paired metric records, one candidate and one reference value per identical `(dataset_id, seed, scenario_id)` key:

```bash
cargo run --locked -p xtask -- compare-cognitive-loop-drift --input path/to/paired-comparison.json --out reports/paired-comparison.json
cargo test --locked -p xtask cognitive_drift_compare
```

Each arm pins a version, repository head SHA, configuration digest and evidence-access track. The candidate and baseline must use the same track. Duplicate pairs, non-finite values, fewer than three dataset clusters, fewer than two seeds, and unbalanced scenario panels fail closed.

The effect is direction-normalized so positive always favors the candidate. It reports candidate and baseline dataset-macro means, a paired effect, median per-record delta, paired win fraction, per-dataset effects, and a deterministic 95% percentile interval. The interval uses a hierarchical paired bootstrap: sample datasets with replacement, then sample seeds within each selected dataset while retaining the complete scenario block for each sampled seed. It requires the same scenario panel across all datasets and gives each dataset equal final weight, avoiding the mistake of treating repeated seeds as independent datasets.

The bootstrap fraction above zero is descriptive, not a p-value. A confidence interval above zero supports an effect only for the pinned metric, dataset set, access track, and configurations. It does not establish universal state-of-the-art superiority. The command separately sets a **coverage-only eligibility gate** for any SOTA claim: at least seven independent datasets and 50 seeds per dataset, reflecting the scale of the current KDD drift benchmark. Passing that gate would not prove superiority; it only prevents small smoke runs from being mislabeled as SOTA evidence. Compare costs, false alarms, recovery and domain-specific slices independently, publish per-dataset results and uncertainty, and pre-register the primary metric before looking at test results.

A clearly labelled synthetic comparison fixture exercises the CLI in CI; its perfect candidate/reference separation is **not a measured result** and must never be added to benchmark leaderboards.


## Counterfactual adaptation utility

Drift-event F1 is not a proxy for better cognition: a detector can win on alarm timing and still trigger expensive, destabilizing or net-negative adaptation. The machine-readable `drift_protocol.adaptation_counterfactual_protocol` therefore defines a separate same-stream, five-arm policy comparison:

- **No adaptation** is the task-loss baseline and must have zero adaptation cost.
- **Detector-triggered** adapts when the candidate detector alarms.
- **Detector-benefit-gated** requires a registered estimated-benefit/risk rule before allowing an intervention.
- **Staged-verify-rollback** stages a candidate, verifies it against registered evidence, and commits or restores the previous state according to the declared transition contract.
- **Oracle-timed** uses ground-truth change timing as an unattainable diagnostic reference; it is not a deployable detector.

The Rust `xtask score-counterfactual-adaptation-utility` command compares pinned per-cycle outcomes from all five policy arms on the same exogenous stream digest. Schema v2 also requires one explicit `comparison_track` for the panel and rejects undeclared values; outputs from different information-access tracks must be analyzed separately, never pooled into the same confirmatory effect. All arms must have identical contiguous cycle indices, and each arm separately pins its version, repository head, configuration digest, and digest of the exogenous stream it actually consumed. Every arm's observed stream digest must equal the shared digest or scoring fails closed. This is a useful mismatch guard, not proof of honest runner execution or identical hidden state. The evaluator reports cumulative/mean task loss, adaptation resource cost, loss improvement against no adaptation, task-loss regret relative to oracle timing, and whether no adaptation dominates the candidate on both loss and resource cost.

```bash
cargo run --locked -p xtask -- score-counterfactual-adaptation-utility \
  --input path/to/counterfactual-adaptation.json \
  --out reports/counterfactual-adaptation.json
cargo test --locked -p xtask cognitive_adaptation_utility
```

Task loss and resource cost remain separate axes. The evaluator reports resource cost per task-loss unit reduced only when the candidate improves loss; it does not invent a universal exchange rate or hide a trade-off behind an unregistered composite score. The oracle-timed arm is an informative gap-to-reference, not a fair deployable baseline. Matching cycle indices and a shared-stream digest are necessary provenance, not proof that hidden state, random streams, environment transitions, or adaptation code were identical. Verify those properties with the runner's independent audit and retain per-cycle records.

### Counterfactual scorer v2: registered phase windows and recovery

The counterfactual input schema is now version 2 and requires outcomes for all five policy arms. It requires `evaluation_windows` with half-open pre-shift and post-shift cycle windows, a fixed absolute `recovery_loss_threshold`, and a number of consecutive below-threshold cycles required to declare recovery. The evaluator validates that the windows are non-empty, in range, and non-overlapping. It reports each arm's pre-shift mean loss, post-shift mean and cumulative loss, post-shift loss area above the registered recovery threshold, the first recovery cycle, and recovery delay. The direct detector-versus-no-adapt effect and detector-versus-oracle regret for **post-shift mean loss** are reported separately from whole-stream effects and resource cost.

The registered recovery threshold is part of the input digest; it must be frozen before the held-out run and expressed in the declared task-loss unit. A missing recovery cycle is emitted as null, not silently converted to the last cycle or an arbitrary penalty. If an analysis requires a finite recovery penalty, preregister that penalty and report censored/non-recovered runs separately. The schema-v1 fixtures are intentionally upgraded to v2 rather than silently assigning phase windows.

The v2 CLI smoke tests assert the preregistered staged-versus-detector post-shift contrast, recovery behavior, and the no-impact-shift negative control—including that benefit gating abstains when adaptation produces no task benefit. These values verify metric wiring only and are not evidence of performance.

The synthetic fixture at `benchmarks/cognitive-loop-evidence/synthetic-counterfactual-adaptation.json` exists solely for CLI regression tests. Its values are synthetic, not benchmark results. This evaluator is a scoring primitive, not the replay engine; actual task runs, pinned datasets, cost instrumentation, and counterfactual state-isolation checks remain necessary before reporting efficacy.


### Complete-panel aggregation and the confirmatory gate

The Rust `xtask compare-cognitive-adaptation-panel` command consumes v2 utility-score records and validates their per-record result digests, fixed policy revisions, per-dataset configuration stability, task-loss metric/unit, and information-access track before aggregation. The expected grid is pinned as seven dataset IDs × 50 seed IDs × the eight registered abrupt/gradual drift cells, for 2,800 unique records. Duplicate cells, unexpected identities, invalid metric digests, a mismatched primary effect, or revision/config drift fail closed.

If any required cell is absent, the command writes the exact missing-cell list and **does not emit a primary comparison or confidence interval**. For complete coverage only, it reuses the existing hierarchical paired bootstrap that resamples datasets first, then seeds while retaining each seed's scenario block. The primary estimate is the dataset-macro paired effect of staged-verify-rollback versus detector-triggered adaptation on post-shift mean task loss, with lower loss oriented as a positive candidate effect. The bootstrap fraction above zero is descriptive, not a p-value.

Example:

```bash
cargo run --locked -p xtask -- compare-cognitive-adaptation-panel \
  --input path/to/adaptation-panel.json \
  --out reports/adaptation-panel-comparison.json
cargo test --locked -p xtask cognitive_adaptation_panel
```

The workflow fixture `synthetic-adaptation-panel-incomplete.json` deliberately contains zero runs. It must report all 2,800 cells missing and `primary_comparison: null`; it tests fail-closed coverage wiring only. The panel aggregator validates scored summaries and their declared provenance, not the honesty of the original replay, the correctness of the model adapter, or causal execution. A full panel score is still not proof of SOTA superiority.


## Runnable scalar drift reference baselines

The Rust `xtask run-cognitive-drift-baseline` command now implements two executable baselines on the **valid-PE-only** track: a fixed threshold and a calibration-centred EWMA control chart. Both consume a disjoint calibration vector plus a cycle-indexed evaluation stream, then feed generated alarm timestamps into the common timing-aware scorer.

```bash
cargo run --locked -p xtask -- run-cognitive-drift-baseline \
  --input benchmarks/cognitive-loop-evidence/synthetic-fixed-threshold-run.json \
  --out reports/fixed-threshold-smoke.json
cargo test --locked -p xtask cognitive_drift_baselines
```

Invalid samples must be encoded as `evidence_valid: false, value: null`; they are skipped and cannot change detector state or re-arm an alarm latch. Valid samples must contain finite values. The fixed-threshold detector emits one alarm per above-threshold excursion, re-arming only after a valid below-threshold sample. The EWMA control chart estimates its center and sample standard deviation from calibration values, uses a declared standard-deviation floor, and applies a two-sided threshold. Threshold/configuration selection must occur outside the scored evaluation stream; the input digest captures the exact calibration data, evaluation sequence, ground-truth episodes, and detector config.

The CI fixture is synthetic and only tests execution/scoring integrity. These are minimal reference controls, not a full implementation of the 14-family KDD 2026 protocol. They do not include the public datasets, held-out-dataset tuning, classifier-error adapters, feature-space baselines, or actual Symthaea cognitive-loop replay. A benchmark result requires pinned detector revisions/configurations, actual held-out runs, raw per-cycle traces, and uncertainty/cost reporting.


## Third executable reference control: Page-Hinkley

The scalar baseline runner now also supports a **one-sided, upward-shift Page-Hinkley detector** for valid prediction-error streams. It initializes its sequential mean from the disjoint calibration observations, declares the tolerated mean movement `page_hinkley_delta`, and declares the cumulative-deviation threshold `page_hinkley_threshold`. An alarm is latched until a later valid PE returns at or below the calibration mean plus delta; invalid/missing observations neither change the statistic nor rearm the alarm.

The synthetic CLI fixture `benchmarks/cognitive-loop-evidence/synthetic-page-hinkley-run.json` exercises two separated upward shifts and asserts the alarm cycles. The fixture is labelled not measured; its exact outcomes test the implementation contract only.

```bash
cargo run --locked -p xtask -- run-cognitive-drift-baseline \
  --input benchmarks/cognitive-loop-evidence/synthetic-page-hinkley-run.json \
  --out reports/page-hinkley-smoke.json
cargo test --locked -p xtask cognitive_drift_baselines
```

This is deliberately described as a reference variant, not as a bit-for-bit reproduction of every Page-Hinkley library implementation. It is upward-shift-only for the PE use case, and the latch/rearm rule is part of the detector contract. Report its parameters and false alarms, delay, and task-level impact. The published KDD 2026 comparison still requires its independently pinned detector implementations, datasets, fold/tuning protocol, and shared classifier-error stream; this PE-only variant cannot substitute for that track.


## Fourth executable reference control: predictive rank martingale portfolio

A more substantive direction than adding another scalar threshold is to make the alarm boundary correspond to a false-alarm budget. Kuang and Xia's September 2026 preprint, *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales*, derives the exact sequential null predictive law for ranks computed against a fixed calibration set. It forms nonnegative martingales by betting on centered rank features; crossing wealth (1/\alpha) is an anytime-valid stopping rule under the paper's null and randomization assumptions. Its order feature targets directional shifts, while its dispersion feature targets center-versus-tail changes. See the [paper](https://arxiv.org/abs/2609.00536), especially its predictive-rank law and portfolio construction.

The Rust baseline runner now includes `predictive-rank-martingale-portfolio`. It uses a fixed calibration reference and four fixed-bet components: positive/negative order features and positive/negative dispersion features, combined with equal weights. Its configured `prm_alpha` defines the first-crossing boundary (1/\alpha); e.g. `0.05` means a wealth boundary of 20. The algorithm updates its predictive rank probabilities from the observed rank history, rather than treating repeated calibration-based p-values as independent.

The runner emits a `diagnostics` object with the statistic name, stopping boundary, final and maximum statistic, configured alpha, whether the run stopped at the first crossing, and the number of valid samples processed. The focused CLI smoke test checks that the recorded maximum reaches the boundary and that the reported crossing cycle matches the processed sample count. A unit test checks the defining conditional invariant: each fixed-bet component has conditional mean one under the predictive null rank probabilities.

After one-time feature-vector construction and calibration sorting, the implementation computes predictive feature expectations from base sums plus accumulated history sums. Each valid observation uses a binary-search rank lookup and constant-time updates, avoiding a full calibration-grid scan per cycle.

The implementation is deliberately scoped:

- It uses fixed betting coefficients, not the paper's adaptive Online Newton Step (ONS) bet updates.
- It produces only the **first alarm per calibration epoch**. Repeated alarms require a fresh, justified calibration epoch or a separately validated alpha-spending construction; resetting wealth informally would destroy the stated guarantee.
- Tie-breaking uses a deterministic PRNG stream derived from the recorded experiment seed, for reproducibility. The exact mathematical guarantee assumes independent randomized tie-breaking; do not claim the guarantee conditional on one fixed pseudorandom seed. Benchmark runs should treat the seed as independent experimental randomization and include multiple independent seeds.
- The guarantee assumes a stable IID null for the calibrated scalar scores. It does not automatically establish validity for autocorrelated PE streams, informative missingness, changing calibration distributions, or the complete cognitive loop.
- This implementation is a reference portfolio, not a reproduction of the paper's ONS algorithm or a measured superiority result. It is currently first-alarm-only per calibration epoch, with the PRNG assumptions described above. The report's `diagnostics` are evidence about the implemented statistic, not by themselves proof of the theorem's assumptions in a deployed autocorrelated cognitive loop.

A synthetic CLI fixture, `benchmarks/cognitive-loop-evidence/synthetic-predictive-rank-martingale-run.json`, tests the first-crossing behavior on one sustained upward shift. It is not benchmark evidence.

```bash
cargo run --locked -p xtask -- run-cognitive-drift-baseline \
  --input benchmarks/cognitive-loop-evidence/synthetic-predictive-rank-martingale-run.json \
  --out reports/predictive-rank-martingale-smoke.json
cargo test --locked -p xtask cognitive_drift_baselines
```

For a meaningful state-of-the-art comparison, compare both the fixed-bet control and the adaptive ONS portfolio against CCTM, ADWIN/Page-Hinkley, dynamic-threshold selection, and the KDD 2026 baselines using identical held-out streams. Report empirical anytime false-alarm rate over the entire stream (not per-cycle alarms treated as independent), detection delay, missed shifts, task-loss improvement, and intervention cost. Until an audited PRNG/independent-seed model and null-calibration study are complete, label this as an implementation prototype with a martingale-inspired contract, not a formally certified detector.

## Fifth executable reference control: predictive rank martingale with ONS

The runner now has a separate `predictive-rank-martingale-ons-portfolio` implementation rather than folding the research algorithm into the fixed-bet control. It maintains one Order PRM and one Dispersion PRM e-process, each with a predictable ONS coefficient, and combines them with equal weights. The coefficient update follows Kuang and Xia (2026): `g_t = Z_t / (1 + lambda_t * Z_t)`, `A_t = A_(t-1) + g_t^2`, and `lambda_(t+1)` is projected after the rank-count update using step size `(9/2) / A_t`. The predictable feasible radius is computed from the maximum possible centered feature magnitude, preserving a positive betting factor.

The separate fixture `benchmarks/cognitive-loop-evidence/synthetic-predictive-rank-martingale-ons-run.json` and focused CI smoke check cover the CLI contract, first-crossing boundary, score linkage, and output digests. Unit tests exercise the ONS execution path and projection-radius invariant. These checks are executable contracts, not a null-calibration study or performance claim.

```bash
cargo run --locked -p xtask -- run-cognitive-drift-baseline \
  --input benchmarks/cognitive-loop-evidence/synthetic-predictive-rank-martingale-ons-run.json \
  --out reports/predictive-rank-martingale-ons-smoke.json
cargo test --locked -p xtask cognitive_drift_baselines
```

The implementation retains the same explicit limitations as the fixed-bet PRM: the seed-derived tie-break is reproducible but not independent randomization conditional on a fixed seed; the theoretical anytime false-alarm guarantee assumes an IID null with the paper’s randomization conditions; informative missingness and autocorrelation need separate treatment; and the prototype has not reproduced the paper’s empirical results. Run both variants against identical null and shifted streams with independent seeds, include CCTM and adaptive-threshold baselines, and compare empirical false-alarm probability over the whole horizon, detection delay, shift-family power, and decision-level utility before claiming superiority.

## Independent receipt-to-panel audit

`audit-cognitive-adaptation-panel` goes beyond the panel’s receipt-link metadata. It reopens the protocol, panel input, each per-cell utility input, and every replay receipt; independently re-runs the receipt-bound scorer for each registered cell; requires exact equality with the stored score; rejects reused identities and unsafe paths; and recomputes the panel comparison.

The command requires an external `--expected-bundle-sha256` supplied by a preregistration record independent of the outcome bundle. It also checks the protocol metric version, declared 7 × 50 × 8 grid, and exact scenario set. This makes an audit reproducible against an independently pinned bundle rather than trusting the digest embedded in that same bundle.

```bash
cargo run --locked -p xtask -- audit-cognitive-adaptation-panel \
  --input path/to/audit-bundle.json \
  --expected-bundle-sha256 PUBLISHED_64_CHARACTER_SHA256 \
  --out reports/cognitive-adaptation-audit.json
```

A pass establishes consistency and reproducibility of the supplied artifacts; it does not prove honest runner execution, actual isolation of oracle information, dataset representativeness, or production superiority. The protocol and bundle digest should be frozen in an independent timestamped record before held-out outcomes are inspected.
