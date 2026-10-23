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

`benchmarks/cognitive-loop-evidence/scenarios.json` is the implementation-facing companion to this contract. It currently defines **30 cases across six suites (S0–S5)**: validity invariants, calibration and distribution shift, continual learning, phase causality, end-to-end benchmark adapters, and system behavior. It includes the B0–B7 baseline matrix, including B4 for calibrated uncertainty. These are proposed cases and configurations; the manifest is not yet a runnable benchmark suite and no results are implied.


## Manifest validator

The JSON manifest is now checked by a small Rust validator in `xtask`; this validates the experiment contract, **not benchmark performance**. Run it from the repository root:

```bash
cargo run -p xtask -- validate-cognitive-loop-evidence
cargo test -p xtask cognitive_loop_evidence
```

The validator fails closed on duplicate suite/case/baseline identifiers, missing B0–B7 controls, incomplete validity-case expectations, invalid evidence configured to update PE training or enter replay, hard-gate cases without assertions, missing result-provenance fields, and a CI policy that does not explicitly reject skipped checks. It prints a suite/case count only after the manifest contract passes. Keep `status` at `proposal-not-measured` until actual pinned runs and raw records exist. The command does not execute the scenarios and produces no SOTA claim.
