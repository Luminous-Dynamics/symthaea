# Symthaea Assistive BCI R&D

**Status:** proposal; no BCI decoder, acquisition driver, or clinical capability is implied by this document.  
**Tracking issue:** [BCI-RD-000](https://github.com/Luminous-Dynamics/symthaea/issues/8179)  
**Scope:** evidence-first assistive brain–computer interface software research.

## Executive thesis

Build a research path toward a BCI that improves a person's practical agency: reliable computer access, communication, and eventually carefully bounded assistive-device interaction. Start with public, offline data and deterministic replay. Only add live non-invasive acquisition after signal contracts, data provenance, baselines, and failure handling have been tested. Implant hardware and clinical translation are separate future programs requiring qualified neuroengineering/clinical partners, ethics review, risk management, and relevant regulatory pathways.

The differentiating hypothesis is **not** that HDC or CfC/LTC is automatically superior to current decoders. It is that a system which combines temporal evidence, explicit uncertainty, careful distribution-shift evaluation, personalized adaptation, and a strict action gate may achieve more reliable intent selection with less recalibration and fewer unintended activations. The hypothesis is falsifiable; the classical baseline may win.

## Repository findings

- Symthaea's existing `neural-bridge` feature is a semantic pathway from text/model-derived representations (including BGE-M3 embeddings) into HDC space. It is not a brain-signal acquisition interface.
- `src/perception/sensor_fusion.rs` provides an existing physical/simulated-sensor-to-`ContinuousHV` pattern, currently demonstrated with IMU inputs.
- [MOBILITY-ASSIST-001, issue #7766](https://github.com/Luminous-Dynamics/symthaea/issues/7766) covers disability-first, task-centered assistive mobility. A future BCI controller should supply user intent to that work rather than reimplement mobility evaluation in parallel.

Keep the BCI code and feature names distinct from `neural-bridge`. Prefer an opt-in, hardware-free core crate or feature such as `bci-core` / `neuro-signal`; choose the final name after a repository-wide collision check.

## Proposed pipeline

```text
source adapter
  -> versioned neural observation
  -> validation + signal-quality gate
  -> causal window/features
  -> decoder candidates + calibrated uncertainty
  -> intent hypotheses + explicit abstention
  -> independent action policy
  -> sandbox / accessible application
```

Each layer has a distinct responsibility:

1. **Source adapter:** first, deterministic replay of public EEG records. A later live EEG adapter is read-only, opt-in, and isolated from the cognitive loop so device I/O cannot block it.
2. **Canonical observation:** a versioned contract records modality, ordered channel identities and montage, sample rate, sequence/sample range, monotonic capture time, units, quality flags, preprocessing-manifest digest, and payload digest. Any device/session identity must be pseudonymous and purpose-limited. Do not put names, diagnoses, or raw signals into ordinary logs.
3. **Validation and quality:** detect non-finite samples, clipping/saturation, flatline channels, missing or duplicated frames, timestamp regressions/gaps, channel/montage mismatch, and stale data. Quality failure must be visible downstream; it cannot silently become a confident class.
4. **Causal representation:** no future-sample leakage. Keep each preprocessing step and its parameters in the run manifest. Model windows must preserve subject/session grouping and chronological boundaries.
5. **Decoder:** emit a set of candidate intents with scores, calibration metadata, supported action vocabulary, timestamp, and an explicit abstain/unknown output. Do not claim the representation directly reveals thoughts, truth, consciousness, or intentions outside the tested task.
6. **Independent policy:** decoder output is a proposal, never authority. The policy checks freshness, confidence/calibration limits, allowed action, rate/velocity constraints, and any required user confirmation. It supports immediate disengagement and a safe idle state. Initial outputs drive a deterministic sandbox only.
7. **Audit/provenance:** hash run configuration, data/split manifests, preprocessing, model artifact, code revision, and results. Provenance establishes what was tested; it does not prove safety, efficacy, or clinical benefit.

### Data minimization and user sovereignty

- Keep raw neural data local by default; no telemetry or remote upload by default.
- Require an explicit purpose, export choice, and retention policy before data leaves the local machine.
- Separate raw measurements, derived features, decoder estimates, user confirmation, and action outcomes.
- Make profile/adaptation reset and model deletion possible without losing the ability to operate the interface through an accessible fallback.
- Treat commands inferred from neural data as potentially erroneous. No medical diagnosis, emotion/personality inference, covert monitoring, or high-consequence action is in scope.

## Evaluation plan

### Phase 0 — lock the experiment before tuning

Use a public dataset version with documented terms and keep raw recordings out of Git. Store the data version/DOI, licence terms, exact subject/session split, preprocessing, seed, decoder configuration, model hashes, and environment identity in a reproducible manifest.

Useful initial sources:

- **PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB):** 64-channel EEG at 160 Hz with movement and motor-imagery tasks. [Dataset and terms](https://physionet.org/content/eegmmidb/1.0.0/)
- **BCI Competition IV, dataset 2a:** 22 EEG channels at 250 Hz, four-class cued motor imagery, nine subjects. Dataset 1 also provides continuous motor-imagery classification with idle periods. [Competition dataset descriptions](https://www.bbci.de/competition/iv/)

Start with one dataset/task/split, freeze it, and reproduce the same baseline twice before adding a novel architecture. Preserve continuous idle periods where available so that "do nothing" is explicitly evaluated.

### Phase 1 — baseline ladder

Compare methods on the same splits and preprocessing:

1. Chance, majority-class, and persistence/temporal baselines.
2. CSP + regularized LDA; add filter-bank CSP when appropriate.
3. A documented compact temporal model or Riemannian-geometry baseline suited to the dataset.
4. HDC-only encoding/decoding.
5. CfC/LTC-only temporal model.
6. HDC + CfC/LTC hybrid.
7. Frozen models versus carefully isolated online adaptation.

This is an ablation program, not a leaderboard by intuition. Include parameter/model-size and compute budgets so added complexity must justify itself.

### Phase 2 — robustness and adaptation

Freeze evaluation tasks that vary across sessions/days, channel availability, signal quality, artifact levels, and distribution shift. Separate calibration data from evaluation data. Adaptation may consume only events that would be available in live use, and results must include its time/data cost.

### Phase 3 — task-centered sandbox

Drive a simulated cursor, grid selection, or text-entry task. Score task completion and recovery, not merely class accuracy. The action policy must reject stale, malformed, repeated, ambiguous, and out-of-policy commands. Test that decoder failure cannot bypass the policy.

### Metrics to report

| Area | Required reporting |
|---|---|
| Classification | Balanced accuracy, macro-F1, Cohen's kappa; per-subject results and uncertainty |
| Effective control | Information-transfer rate where protocol permits; successful selections; time-to-target; task completion and error recovery |
| False activation | False activations per hour, idle-state rejection, command precision, and abstention coverage/risk curve |
| Calibration | Calibration time, calibration drift, confidence reliability / expected calibration error where appropriate |
| Generalization | Cross-session/day degradation; subject-specific vs pooled training; held-out users; channel-dropout and artifact robustness |
| Runtime | End-to-end p50/p95/p99 latency, deadline misses, CPU/GPU use, memory, and energy when measurable |
| User agency | Override/stop rate, fallback success, task accessibility, and eventual participant-reported burden in an ethically reviewed study |

Do not compare non-invasive EEG classification directly with implanted intracortical cursor-control bits-per-second without clearly identifying modality, participants, task, and protocol. These operating regimes are not interchangeable.

## Safety and security invariants

- **Fail closed:** invalid/stale/missing-quality input produces no actionable intent.
- **No direct actuator authority:** model outputs cannot skip the independent action policy.
- **Explicit idle and stop:** a valid idle output is part of the decoder contract; stop/disengagement is not probabilistic.
- **Replay protection:** repeated frames, sequence rollback, timestamp regression, and stale packets are rejected or explicitly quarantined.
- **No silent preprocessing:** resampling, filtering, reference changes, and channel remapping are named, versioned, and included in the run manifest.
- **No false assurance:** synthetic fixtures prove structural invariants only; public-dataset metrics prove performance only on the tested dataset/protocol.
- **No hidden-state claims:** HDC/CfC representations are computational representations of observed signals, not proof of subjective experience or privileged access to a person's unexpressed thoughts.

## Milestones and acceptance criteria

### BCI-RD-001 — contract + golden fixtures

Deliver a typed, versioned observation/frame contract and deterministic replay fixtures with no device I/O.

Acceptance:
- identical bytes + configuration produce identical validated event sequences and outputs;
- non-finite, out-of-order, stale, wrong-montage, missing-channel, and malformed frames fail closed;
- channel quality and preprocessing provenance survive the entire pipeline;
- a negative test proves stale/replayed frames cannot become actionable commands;
- raw signals are absent from ordinary logs.

### BCI-RD-002 — leakage-resistant benchmark harness

Deliver the data/split manifest, baseline implementations, score schema, and machine-readable result artifacts. A result must be reproducible from declared artifacts without access to private data.

Acceptance:
- subject/session leakage checks are automated;
- chance/simple and classical baselines are included;
- per-subject results and uncertainty are reported;
- the complete pipeline can be rerun without a live hardware device.

### BCI-RD-003 — HDC/CfC falsification study

Run paired ablations for representation and temporal model choices. Evaluate whether hybrid methods improve robustness, calibration burden, false activations, or cross-session transfer—not only in-split accuracy. Publish negative results.

### BCI-RD-004 — safe sandbox controller

Connect to a deterministic simulated cursor/selection task behind the independent policy. Run negative controls for stale, duplicated, contradictory, low-confidence, and corrupted decoder outputs.

### BCI-RD-005 — opt-in non-invasive acquisition

Only after prior gates pass: add read-only EEG adapter(s), clock/packet-loss diagnostics, local processing, and explicit user stop. Device-specific work must not become a default dependency.

### Clinical/implant gate

Any implanted-device, stimulation, human-participant, or clinical-effectiveness direction is a separate review. It requires qualified neuroengineering and clinical collaborators, ethics oversight, formal risk management, and the applicable regulatory process. This proposal does not provide or claim a surgical or implant construction design.

## Competitive context (October 2026)

Neuralink's [public updates](https://neuralink.com/updates/) report an October 1, 2026 result describing self-supervised pretraining on 50,000 hours of unlabeled brain data, decoder operation for weeks without recalibration, and an 11.32 bits-per-second record. Those are company-reported results, not an independent comparison against Symthaea.

Implications for our research:
- Self-supervised pretraining and longitudinal drift are must-test baselines, not optional stretch goals.
- Recalibration time and false activations are first-class outcomes.
- An open dataset benchmark is useful but cannot establish parity with an implanted system.
- A claim of "better than SOTA" requires matched protocol, independent reproduction, and explicit limitations.

The [FDA's implanted-BCI guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/implanted-brain-computer-interface-bci-devices-patients-paralysis-or-amputation-non-clinical-testing) describes non-clinical testing and clinical-study considerations for implanted BCIs. It is an important later planning reference, not a claim that this software-only project meets regulatory requirements.



## 2026 benchmark research and go/no-go targets

New research reinforces a multi-objective, deployment-oriented benchmark rather than a leaderboard based on one accuracy number:

- **EEG-FM-Bench (ICML / PMLR 2026):** evaluates EEG foundation models over 14 datasets and 10 paradigms; its analysis highlights multi-task regularization, negative transfer, objective alignment, and adaptation compatibility. https://proceedings.mlr.press/v306/xiong26f.html
- **EEG-FM-Compass (National Science Review, August 2026):** compares 12 open foundation models and specialist baselines across 13 datasets and nine paradigms, including leave-one-subject-out and few-shot calibration; larger models and linear probes do not reliably dominate. https://academic.oup.com/nsr/advance-article/doi/10.1093/nsr/nwag466/8750569
- **BEND-BCI (2026 preprint):** compares 23 decoder methods across 16 real/synthetic recordings and reports that accuracy rankings can change when perturbation robustness, compute, and cross-recording representation consistency are measured. This is a preprint, not yet peer-reviewed. https://www.biorxiv.org/content/10.64898/2026.07.21.739953v1
- **EEG/EMG Foundation Challenge 2026, Track 02:** the official page says registration remains open through October 24, 2026; the sealed phase runs October 28–November 21. The task tests three cued mental commands across later sessions and unseen participants, scoring cell-mean balanced accuracy over participant/session/context. Public warm-up scores are explicitly indicative because the data are public and leakage is possible. Treat it as a protocol check; only sealed scores qualify for final ranking. Work is tracked in [BCI-RD-003](https://github.com/Luminous-Dynamics/symthaea/issues/8195). https://neural-interfaces26.github.io/tracks.html
- **AdaBrain-Bench (2025 preprint/project):** evaluates EEG foundation-model transfer across seven application areas, including cross-subject, multi-subject, and few-shot settings. It is a comparison tool to consider—not a claim that foundation models always win. https://arxiv.org/abs/2507.09882
- **Neuralink (company-reported, 1 October 2026):** reports self-supervised pretraining on 50,000 hours of unlabeled neural data, weeks without recalibration, and 11.32 bits/s. Those intracortical results are not directly comparable with non-invasive EEG motor-imagery accuracy. https://neuralink.com/updates/pretraining-on-50000-hours/
- **PhysioNet EEGMMIDB:** gives a broader non-invasive transfer test with 64-channel recordings from 109 volunteers across movement and motor-imagery conditions. Dataset source: https://physionet.org/content/eegmmidb/1.0.0/

### Proposed internal gates (aspirational, not achieved results)

| Gate | Target and decision rule |
|---|---|
| Contract integrity | All deterministic fixtures pass; every known malformed/stale/low-quality case is rejected; a rejected frame never advances the stream. Follow with property/fuzz tests and at least 100,000 generated adversarial frames before considering live acquisition. |
| Fair baseline | Same frozen subject/session splits and preprocessing for CSP+regularized LDA, a compact temporal/Riemannian method, EEGNet or an openly available task-matched deep baseline, and each Symthaea ablation. Publish per-subject metrics and uncertainty. |
| Scientific improvement | Aim for at least **+3 absolute percentage points** in the preregistered primary metric over the strongest matched baseline, with participant-clustered uncertainty excluding zero, on two independent public dataset/task settings. If accuracy does not improve, a Pareto win can count only if it preserves task success while materially reducing calibration or false activations. These are go/no-go targets, not claims of superiority. |
| Calibration burden | Target **50% fewer labeled calibration examples or minutes** while remaining within a predeclared **2 percentage-point** non-inferiority margin versus full calibration, under a subject/session-held-out protocol. |
| Robustness | At matched task-success level, target **30% fewer false activations** under a preregistered artifact/channel-loss/temporal-drift battery, while reporting abstention coverage and risk. Do not tune perturbations against the test set. |
| Runtime | Provisional software target: p95 feature-plus-decoder computation below **20 ms** on a declared reference CPU for a fixed benchmark frame, reported alongside memory and processor load. Acquisition-window latency is reported separately. |
| Practical utility | Simulated task success, time-to-target, error recovery, stop/override use, and calibration time must improve or remain non-inferior. A model-score gain alone is not sufficient. |

Because BCI Competition IV 2a has only nine participants, use participant-level paired inference and avoid claiming broad generalization from this benchmark alone. Report both per-subject values and the aggregate; repeat promising results on EEGMMIDB or another genuinely independent, task-compatible dataset. Separate all model selection from the held-out sessions. A public dataset cannot certify safety or clinical efficacy.

### Paradigm shift to test

Treat a BCI as a **selective, adaptive control system under uncertainty**, not a classifier that must always output an action:

1. Keep input quality and uncertainty visible throughout the pipeline; "unknown" and "do nothing" are first-class outputs.
2. Optimize for **useful agency per calibration minute**: task success, safe command precision, false activations, time-to-target, and recovery.
3. Train representations on unlabeled data only when a credible source/license and leakage controls exist; evaluate transfer and few-shot adaptation against specialist baselines.
4. Adapt only under explicit, auditable constraints, with rollback and a frozen evaluation arm. Avoid self-training loops in which the decoder's own unverified predictions silently become ground truth.
5. Make personalization local-first and user-reversible. The user can stop, reset adaptation, and choose an accessible fallback.

## BCI-RD-001 implementation artifact

The hardware-free research core is staged at `research/bci-core/` as a standalone, dependency-free Rust package. It defines typed signal frames, explicit quality metadata and thresholds, freshness checks, stream continuity/montage/preprocessing invariants, typed rejection reasons, and a reset rule requiring the next accepted frame to use a strictly greater clock epoch. A constructor accepting the last durably accepted epoch lets the caller preserve this replay floor across process restarts. It also includes a validator-issued opaque frame receipt and an independent sandbox-only action gate that applies freshness, quality, confidence/margin, replay-sequence, cursor-step, allowed-action, and external user-confirmation checks. A metadata-only benchmark module audits subject/session split leakage and configurable temporal guard bands between signal windows. It also computes equal-weight participant × session × recording-context cell-mean balanced accuracy against a frozen expected-cell inventory, rejecting omitted/unexpected cells and missing ground-truth classes. Versioned run-manifest and scorecard contracts bind dataset, split, protocol, preprocessing, model, code revision, environment, and metric identities; scorecard uncertainty bounds must name a method. The strict cell-mean scorer (evaluate_cell_mean_balanced_accuracy) uses a fixed declared class set and rejects any cell missing ground-truth support for a declared class; it is intentionally stricter than the current Track 02 reference implementation. On 2026-10-20, we inspected the public objective and metric at [neural-interfaces26/2026-competition commit a66bc23d1a0e686b895c093a642b45c74970758b](https://github.com/neural-interfaces26/2026-competition/tree/a66bc23d1a0e686b895c093a642b45c74970758b/tracks/bci_decoding). Its balanced_accuracy macro-averages over the per-cell union of target and predicted classes: a predicted-only class contributes zero recall, while a class absent from both target and prediction is omitted. The additional evaluate_track02_reference_compatible_cell_mean helper mirrors that class-set behavior and equal-weight cell aggregation while retaining integer parts-per-million; because recalls, cell means, and the final mean are rounded down, its fixed-point result can differ from the organizer's float64 result by a few ppm. It is a semantic compatibility helper, not a bit-for-bit port or official score. Golden fixtures still need to be executed against the pinned reference evaluator before any claim of exact parity. Canonical JSON serialization and SHA-256 calculation remain the responsibility of an external reviewed layer. The `selective` module computes threshold-dependent coverage/risk, fixed-bin confidence calibration, and false action-onset rates per hour with explicit idle exposure; sustained command states are not supposed to count as repeated new activations, and the evaluator rejects a purportedly correct prediction paired with a new action onset during ground-truth idle. The gate returns sandbox action values only; it is not a hardware actuator API or a production safety certification.

Run once a Rust toolchain is available:

```bash
cargo test --manifest-path research/bci-core/Cargo.toml
cargo clippy --manifest-path research/bci-core/Cargo.toml -- -D warnings
```

The contract now includes a deterministic 100,000-case mutation loop that must reject every malformed/stale/identity-shifted frame without advancing stream state, followed by a valid-frame recovery check. It also exposes a constructor that restores a caller-supplied durable epoch floor after process restart. This is test code authored, not a result already executed.

The package has been authored but **has not been compiled or tested in the authoring environment because Cargo/rustc are unavailable here**. Its in-source tests and exact-head CI still need to qualify it. It remains isolated from the root Cargo workspace to avoid unreviewed dependency/lockfile churn; promote it into a workspace member only after the crate has been compiled and reviewed.

**Current evidence status:** this is a research specification and unverified initial contract implementation. No Symthaea BCI hardware, benchmark result, clinical validation, or superiority claim is made.


## Participant-clustered uncertainty for paired model comparisons

Repeated EEG windows from one person are not independent experimental participants. Reporting a confidence interval that treats thousands of overlapping windows as independent can make weak evidence look decisive. The statistics module therefore accepts one paired summary row per held-out participant and resamples participants with replacement.

Protocol:

1. Freeze the participant split, evaluation-cell inventory, preprocessing, score, and baseline/candidate pair before tuning.
2. Aggregate each method over the exact same declared evaluation cells for each participant. Persist the per-cell rows elsewhere in the benchmark artifact; this helper consumes participant aggregates only.
3. Pass one row per participant to paired_participant_bootstrap. Duplicate or blank participant IDs, scores outside [0, 1], fewer than two participants, and invalid bootstrap settings are rejected.
4. Report both method means, paired candidate-minus-baseline delta, the prespecified percentile interval (default 95%), seed, replication count, cohort size, and the number of bootstrap replicates whose mean difference was positive.
5. Treat the positive-replicate fraction as a descriptive resampling statistic, not a posterior probability or p-value. A non-positive lower interval is not evidence of superiority. Publish per-participant outcomes so a strong aggregate cannot hide harmed participants.

This helper estimates uncertainty for an equal-participant-weighted mean of participant-level summaries. It does **not** replace the official challenge metric, and it is not guaranteed to produce an interval for the official cell-weighted score when participants have differing numbers of evaluation cells. For competition reporting, preserve the organizer's exact score and separately implement a participant-cluster bootstrap that recomputes that official aggregation inside every resample from the frozen cell-level predictions. The score implementation must be cross-checked against the organizer's reference evaluator before describing it as exact.

A research go/no-go rule should be preregistered before observing sealed results. The current exploratory target remains at least a 3-percentage-point absolute gain with a paired 95% interval excluding zero in two independent evaluation settings, alongside separate limits for calibration burden, false activations, generalization, and latency. Choose sample size with a power or precision analysis; a fixed number of windows is not a substitute for a sufficient number of independent participants.



## Paired uncertainty for cell-mean BCI results

Both bootstrap helpers require baseline and candidate predictions to carry identical stable, pseudonymous window IDs, with the exact same evaluation-cell identity and target class for each ID. Duplicate or empty IDs are rejected. This prevents silent comparison of different labelled windows.

### Challenge-aligned cell-unit bootstrap

The challenge website's published post-evaluation analysis defines Track 02's independent units as participant × session × context cells. The helper paired_cell_mean_unit_bootstrap resamples those cell-level scores with replacement. Each draw uses the same cells for baseline and candidate and computes their paired score difference. It reports the original cell-mean scores, observed paired delta, bootstrap mean delta, percentile interval, unit and participant counts, seed, confidence level, replication count, and a versioned method ID.

This follows the website's described cell-unit bootstrap convention and the official cell-weighted point-score aggregation. It is still a project implementation using fixed-point scores and order-statistic quantiles; golden comparisons are required before claiming exact parity with the reference post-evaluation analysis. It does not yet implement pairwise bootstrap p-values, Holm correction or whole-leaderboard rank stability.

### Participant-cluster sensitivity analysis

The helper paired_cell_mean_participant_bootstrap is a second, deliberately different analysis. Each draw samples participants with replacement and retains all of every selected participant's session/context cells. This gives a sensitivity analysis when cells from the same participant may be correlated. Because it resamples participants as clusters, it should not be reported as the challenge website's planned default analysis.

The earlier paired_participant_bootstrap remains available for equal-participant-weighted summaries. Do not confuse its estimand with the official cell-weighted point score.

In all cases, the positive-delta replicate count is descriptive, not a posterior probability or p-value. A real superiority claim needs a prespecified effect size, uncertainty procedure, adequate independent units, and an evaluation set untouched by model selection.


## Full-field inference: significance, effect sizes and rank stability

The current public EEG/EMG Foundation Challenge leaderboard (snapshot refreshed October 19, 2026) shows many Track 02 warm-up entries at 1.00. The challenge page explicitly separates the public warm-up board from the sealed final ranking; the final uses the best of the last five submissions and includes a reproducibility audit. The public number is therefore a development signal, not proof of generalization or a durable rank.

The new dependency-free Rust API analyze_leaderboard requires the frozen model-ID and independent-unit-ID inventories from the benchmark manifest, plus one score for every model × unit pair. It rejects unknown IDs and a missing pair even when a unit is omitted from every model, preventing the scoring denominator from silently shrinking. It makes one shared bootstrap draw of evaluation units and re-scores every model on that same draw, so pairwise deltas are paired rather than inferred from separate confidence intervals. It reports:

- point scores and mean bootstrap scores for each model;
- pairwise observed score deltas, percentile intervals and two-sided bootstrap p-values based on the smaller signed tail;
- Holm step-down adjusted p-values across all declared model pairs, with a family-wise alpha recorded in the analysis configuration;
- mean ranks and rank distributions for every model, plus top-1/top-3/top-5 frequencies;
- the seed, replicate count, confidence level, independent-unit count, and a versioned method ID.

Rank ties use shared competition rank: a model's rank is one plus the number of models with strictly higher scores in that draw. Thus two models tied at the top can both have top-1 frequency 100%; top-k frequencies across models are not mutually exclusive. Full per-model rank-frequency vectors use largest-remainder apportionment, so each vector sums to exactly 1000 permille.

This analysis requires complete, frozen scores for every model and every declared unit. Higher scores are assumed better, which fits Track 02's balanced accuracy; lower-is-better metrics must be transformed or handled by an explicitly different protocol. Empty/duplicate model-unit observations, invalid scores, incomplete grids and computations exceeding a declared work budget are rejected.

The p-value estimator intentionally follows the public challenge methodology's doubled smaller tail, counting zero differences in both tails and using a conservative fixed-point rounding rule. The result is **not** a Bayesian posterior probability, and bootstrap significance alone does not prove practical utility. Pairwise comparisons must be prespecified, the evaluation set must remain untouched during model selection, and effect-size/uncertainty results should be reported together. Golden parity tests against the challenge's Python/NumPy reference are still needed; this helper is an implementation of the published analysis shape, not a verified replacement for the official evaluator.

Research reference: https://neural-interfaces26.github.io/leaderboard.html


## Calibration reduction as a preregistered non-inferiority test

The new Rust function evaluate_calibration_reduction evaluates a single prespecified contrast: a candidate decoder at a smaller calibration budget versus a reference decoder at a larger budget, on the same frozen participants. Inputs must declare the budget unit (for example labelled examples or minutes); these units must not be mixed. Exactly one row is required for every expected participant × model × budget. Unknown models/participants, duplicate rows, missing comparisons, unexpected budgets and invalid scores are rejected.

The result reports the budget reduction, paired candidate-minus-reference score delta, percentile confidence interval, participant count, seed/settings, non-inferiority margin and a three-way decision:
- Supports non-inferiority when the lower interval endpoint is at least the negative of the prespecified margin.
- Evidence of inferiority when the upper endpoint is below that boundary.
- Inconclusive otherwise.

The initial target can be preregistered as 50% fewer labelled calibration examples with no more than a 2-percentage-point absolute loss in the benchmark score, tested on matched held-out participants. This is an aspirational go/no-go criterion, not a result. The implementation evaluates exactly one declared budget comparison; do not search over many budgets on the final test set and report only the one that passed. If a calibration frontier is explored, select candidate budgets using development data and confirm the chosen comparison on a separately frozen test set.

Limitations: this contract operates on already computed scores; it does not perform online learning or measure wall-clock calibration time itself. It assumes higher scores are better and uses the same participant-resampling percentile interval helper as the paired evaluation module. The tests are synthetic and do not establish a real calibration reduction or any clinical outcome.
