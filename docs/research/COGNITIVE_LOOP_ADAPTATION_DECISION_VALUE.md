# Cognitive-loop adaptation decision value and recovery protocol

**Status:** proposal / preregistration template; not measured  
**Scope:** next increment after the Cognitive Loop Evidence Benchmark (#8495)  
**Normative principle:** an alarm is evidence for investigation, not authorization to adapt.

## Why this increment

The existing benchmark separates detector quality from downstream task loss and cost, which is the right foundation. However, an arm summary and its self-reported stream digest are not enough to establish a common replay. The scorer now supports binding each five-arm score to a raw replay receipt that it revalidates, while keeping the boundary explicit: a valid receipt is an internally consistent artifact, not proof that a runner executed honestly. The next experiment should make the replay driver—not each arm—the authority for the exogenous stream and action/event ledger.

This distinction matters because the research question is not just “did a detector find the change?” It is:

> Given only information available at decision time, can a controller improve held-out task outcomes by choosing whether to adapt, when to adapt, how much to adapt, and whether to keep or roll back the change—without violating evidence-validity or operational-risk constraints?

The KDD 2026 benchmark addresses detector comparability with controlled drifts injected into real-world streams, timing-aware metrics, and leave-one-dataset-out configuration selection. A September 2026 district-heating benchmark reports that the strongest detector-centric result and the strongest paired-learner forecasting improvement are not the same. These findings support separate detector, policy, and task-outcome tracks rather than an aggregate detector leaderboard.

## Research hypotheses (falsifiable)

- **H1 — decision value:** a policy using valid evidence plus an estimated benefit/cost gate improves post-shift task loss relative to detector-triggered adaptation, on held-out datasets, while respecting preregistered cost and risk constraints.
- **H2 — recoverability:** staged adaptation with explicit verification and rollback reduces cumulative loss after harmful interventions relative to the same detector with irreversible or unverified adaptation.
- **H3 — selective action:** a decision-value policy avoids unnecessary adaptations on nuisance shifts that change input distributions but do not degrade task performance.
- **H4 — evidence integrity:** invalid, missing, stale, non-finite, dimension-incompatible, or future information cannot cause a parameter update, replay admission, confidence escalation, or action authorization unless that consumer has an explicitly justified alternative contract.
- **H5 — generalization:** gains persist under leave-one-dataset-out evaluation, delayed labels, autocorrelation, gradual drift, recurring concepts, missingness, noise, and transition severity changes—not merely one synthetic change shape.

A hypothesis is supported only by its registered estimand, held-out data, paired uncertainty, and negative controls. The protocol itself is not evidence for any hypothesis.

## Recent evidence that changes the benchmark design

The October 2026 TPAMI work on latency-aware adaptation treats delayed supervision and concept drift as a coupled problem, not two independent implementation details. That reinforces a required design choice here: outcomes have both a value and a reveal time, and a policy may only consume them after that reveal time. The synthetic harness records a one-tick delay and uses the `observed-outcome` track; it must not be compared directly with a valid-prediction-error-only detector. The paper's reported results apply to its own evaluated settings and are motivation for a matched delayed-label ablation, not evidence that Symthaea outperforms it. [Paper record](https://pubmed.ncbi.nlm.nih.gov/42862614/).

## Conceptual shift: drift as a hypothesis, adaptation as a transaction

Model an adaptation attempt as a reversible transaction:

1. **Observe:** emit a timestamped evidence record with validity, provenance, calibration epoch, access track, and availability time.
2. **Diagnose:** estimate whether the change is task-relevant, not merely statistically detectable.
3. **Decide:** compare expected benefit with declared intervention cost and risk limits. If evidence is invalid or the expected benefit is not sufficiently supported, abstain.
4. **Stage:** snapshot the prior state; perform adaptation in shadow/canary mode where possible.
5. **Verify:** evaluate against a predeclared post-intervention window using only labels/outcomes available at that time.
6. **Commit or roll back:** retain the candidate only when the registered recovery criterion is met; otherwise restore the prior state and record the failed attempt.
7. **Learn:** update controller calibration only from permitted, time-correct evidence. Never quietly tune on the held-out test stream.

Every transition must be observable in the replay log. “Adaptation happened” is not equivalent to “adaptation succeeded.”

## Information-access tracks

Results must never mix tracks in a single ranking. Adapters, tuning, and policy decisions must obey the same declared information boundary.

| Track | Information available to detector/controller | Main question |
| --- | --- | --- |
| Valid prediction-error | Valid prequential prediction error only; invalid samples are explicitly missing | Can we react to degradation in the online predictor? |
| Unsupervised feature | Current features and declared calibration history; no labels or future outcomes | Can distribution change be detected without labels? |
| Auxiliary unsupervised | A registered auxiliary loss or self-supervised signal, with its availability timestamp | Does the auxiliary signal add value without label leakage? |
| Delayed observed outcome | Outcomes only after the declared label delay | Can intervention improve actual task performance under realistic feedback latency? |
| Oracle diagnostic | Ground-truth change timing and labels, unavailable to a deployable policy | Upper bound / debugging only; never presented as a deployable comparator |

For every decision, log the earliest availability time of every input. A controller must fail closed if a feature's timestamp exceeds the decision timestamp. A delayed-label track must preserve label delay, missing labels, and reveal order.

## Policy arms

The full benchmark should include detector and policy comparisons, not only an expanding detector list.

- **No adaptation:** task baseline and control for accidental regression.
- **Always adapt at registered boundaries:** upper comparator for adaptation frequency/cost, not an oracle.
- **Detector-triggered:** adapt immediately when each detector alarms.
- **Detector + benefit gate:** adapt only when the registered estimated gain criterion is met.
- **Staged verify/rollback:** benefit-gated adaptation with a saved state, canary/verification window, and rollback rule.
- **Periodic or scheduled adaptation:** practical operational comparator.
- **Oracle-timed adaptation:** ground-truth-timed diagnostic only; explicitly non-deployable.

Hold detector revisions fixed while comparing policy logic, and hold policy logic fixed while comparing detectors. This factorial separation helps distinguish a strong detector from a strong controller.

## Primary outcomes and report contract

There is no default composite score. The **single confirmatory comparison** is staged-verify-rollback versus detector-triggered adaptation on held-out post-shift task loss (lower is better), with the policy, endpoint, direction, and interpretation frozen before held-out evaluation. Other detector, policy, metric, and scenario comparisons are exploratory unless a family and a multiplicity-control procedure are preregistered. This prevents choosing the best-looking result across many combinations and presenting an unadjusted interval as confirmatory evidence.

Report these separately:

**Task outcomes**
- Pre-shift, transition-window, post-shift, and whole-stream task loss.
- Cumulative regret against no-adapt and the oracle-timed diagnostic.
- Recovery time, overshoot, and area under the post-shift loss curve.
- False adaptation rate on nuisance/no-impact shifts.
- Adaptation benefit persistence after the verification window.

The implemented counterfactual utility scorer is schema v3 and requires all five policy arms. It computes whole-stream and pre/post-window loss, post-shift loss area above the preregistered threshold, recovery cycle/delay, intervention cost, and the primary paired-record effect of staged-verify-rollback versus detector-triggered adaptation on post-shift mean task loss. It also reports benefit-gated and staged effects relative to no-adapt. These are metrics over supplied arm summaries, not yet outputs from a driver-owned real replay.

For receipt-bound scoring, pass the raw receipt to the same command:

`cargo run --locked -p xtask -- score-counterfactual-adaptation-utility --input path/to/counterfactual-adaptation.json --receipt path/to/replay-receipt.json --out reports/counterfactual-adaptation.json`

The scorer reruns receipt-schema-v2 validation and fails closed unless dataset, stream ID, environment seed, and the canonical event-ledger digest match the score, and each of the five policy arms matches the receipt's repository head and configuration digest. For `run_kind: replay`, it also requires per-cycle task loss, adaptation cost, and an outcome-availability tick; it compares all three values exactly against the scored input for each arm/cycle. The outcome timestamp cannot precede its decision tick. It then includes both the canonical receipt hash and receipt-validation-report hash in the score's own result digest. A score created without `--receipt` remains an unlinked exploratory summary.

The panel distinguishes grid coverage from receipt linkage. It emits the preregistered comparison only when all 2,800 cells are present and each score carries valid receipt-link fields marked `run_kind: "replay"`; synthetic/not-measured receipts never satisfy this gate. The panel checks digests and declared linkage rather than re-executing 2,800 receipts. Preserve the raw receipt artifacts and have an independent verifier revalidate them; neither hashes nor receipt structure prove that the runner was honest, oracle access was technically isolated, or task outcomes were genuine.

**Decision quality**
- Fraction of interventions that improve task loss.
- Unnecessary intervention rate, harmful intervention rate, missed opportunities, and abstention rate.
- Time from first available evidence to intervention; time to verified recovery.
- Rollbacks, failed verifications, repeat interventions, and oscillations.

**Detector quality**
- Event precision/recall/F1, missed-event rate, false alarms per 1,000 valid observations, normalized delay, and alarms during null periods.
- In an anytime-valid track, empirical probability of any false alarm over the full horizon; do not treat per-cycle alarms as independent trials.

**Cost and operational safety**
- Adaptation compute/time, memory, energy where measurable, label/review budget, state-snapshot and rollback costs.
- p50/p95/p99 decision latency and dropped/late events.
- Invalid-evidence propagation count; forbidden-state-update count; rollback correctness.
- Cost units and any exchange rate used to compare cost with task loss must be preregistered. If no defensible conversion exists, show Pareto frontiers and constrained comparisons rather than weighted totals.

## Replay runner: required next implementation

The runner should own the stream and execute all arms against one canonical, timestamped exogenous event sequence. It must not trust an arm's self-reported stream hash as sufficient proof.

Required design:

1. Canonical dataset artifact and transformation digest, ordered event ledger, explicit availability timestamps, label-delay model, deterministic environment seed, and replay-driver revision.
2. At each cycle, deliver exactly the information available then to each arm. Log each arm's action before delivering subsequent outcomes. Keep policy-specific state separate while deriving exogenous observations from the same immutable event source.
3. Capture input/event digests from the driver itself, the received-event sequence for each arm, action and adaptation records, pre/post state digests, loss observations, intervention costs, and rollback outcomes.
4. Separate environment randomness from policy randomness. Use matched environment randomness across arms; record policy PRNG seeds separately. Do not force identical internal stochastic states when the policies legitimately diverge.
5. Reject a run if a required event is missing, duplicated, reordered, delivered early, or differs across arms; if the arm reports a digest incompatible with the driver's event ledger; or if revision/configuration/artifact provenance is absent.
6. Emit raw per-cycle records before summary scores, with canonical serialization and hashes. A summary alone must not be accepted as source evidence.
7. Keep oracle labels/change points in a separate channel that is inaccessible to deployable arm code. Enforce this through the runner's interface, not just a documentation promise.
8. Include negative tests that deliberately leak a future label, alter one event in one arm, reorder one input, change one configuration, tamper with a reported cost, omit a rollback event, and inject invalid evidence.

The existing counterfactual scorer remains useful as a downstream summary auditor. It is not the replay orchestrator and should not be described as proving causal execution on its own.

### Replay receipt validator (implemented; not a replay engine)

The Rust `xtask validate-cognitive-replay-receipt` command now validates typed receipt schema v2 with a canonical event ledger and five required policy arms. In addition to identical delivery IDs/ticks, causal information availability, oracle-flag isolation, state-digest continuity, and stage → verify-with-evidence-digest → commit/rollback transitions, each executed (`run_kind: replay`) receipt must include per-cycle task loss, adaptation cost, and the tick when the outcome became available. Missing outcomes, non-finite/negative values, or an outcome timestamp earlier than its decision cycle fail closed.

The utility scorer's `--receipt` mode revalidates that raw receipt and matches every scored task-loss value, adaptation cost, cycle index, and outcome-availability timestamp against the corresponding receipt decision, for all five policy arms. It also checks dataset/stream/seed/event-ledger identity plus policy revisions/configuration digests. Tests cover outcome tampering, timestamp tampering, identity mismatch, and configuration drift. The panel still consumes score artifacts and receipt-link metadata rather than independently opening 2,800 raw receipts, so preserve the raw artifacts and independently audit them.

These controls establish internal consistency and prevent accidental mixing of scores and receipts; they do not prove that the runner executed honestly, that a sandbox technically prevented oracle access, or that the measured outcome is truthful. The included receipt fixture remains synthetic and marked `synthetic-not-measured`.

### Synthetic end-to-end replay smoke harness (implemented; not measured)

The new `xtask run-cognitive-adaptation-synthetic-replay` command exercises the receipt-bound scorer path without pretending to execute the production Symthaea cognitive loop:

```bash
cargo run --locked -p xtask -- run-cognitive-adaptation-synthetic-replay \
  --scenario abrupt --ticks 64 --seed 7 \
  --out-dir reports/cognitive-adaptation-synthetic
```

It deterministically builds a synthetic scalar-regression stream and five policy arms (no-adapt, detector-triggered, benefit-gated, staged verification/rollback, and oracle-timed diagnostic). The scenario choices are `abrupt`, `gradual`, `no-impact`, and `recurring`. The harness writes `receipt.json`, `utility-input.json`, `receipt-validation.json`, `utility-score.json`, and `summary.json`. The observed-outcome access track explicitly reflects the controller's one-tick delayed feedback; it is not labelled valid-PE-only.

The runner tags every receipt and summary as `synthetic-not-measured`; its run IDs carry the same classification, and the receipt validator rejects relabelling that synthetic run ID as `run_kind: replay`. The workflow includes a focused unit-test step and an end-to-end CLI smoke check that revalidates the generated receipt. These checks are intended to expose interface/schema bugs early, not establish task improvement, reproduce published detector results, or satisfy the 2,800-cell confirmatory panel.

This harness is deliberately **not** the production replay engine. It uses a simple scalar model, synthetic generated outcomes, and a fixed policy family; it does not run the actual Symthaea cognitive loop, isolate oracle access in a hardened process boundary, ingest pinned public datasets, or establish real-world causal performance. The next scientific milestone remains a driver-owned replay interface with a real policy adapter and immutable exogenous-event delivery. Until the Rust workflow passes on this exact commit, even the synthetic smoke path should be considered implemented but not yet validated.

## Reference datasets and comparators

Use the seven public datasets and eight drift cells registered by the current benchmark plan as the minimum detector-comparability panel. Pin data sources, licensing, transformations, and per-dataset checksums. Record all exclusions and all missing runs; never silently substitute a dataset.

Keep the following comparator families distinct:
- KDD 2026 reference implementations and protocol, including its supported drift methods and classifier-error access track.
- Conventional scalar controls: fixed threshold, EWMA, Page–Hinkley.
- Adaptive threshold comparator: AAAI 2026 Dynamic Threshold Determination (DTD).
- Window/stream comparators: ADWIN and KSWIN, plus the registered STUDD/ABCD/SEED/STEPD families where reproducible implementations and licences are available.
- Sequential-evidence comparators: CCTM and the published predictive-rank martingale procedure. The existing fixed-bet PRM prototype must not be labelled as the published adaptive ONS method.
- Model/policy controls: no adaptation, scheduled adaptation, detector-triggered adaptation, benefit gate, and staged rollback policy.

No implementation is “equivalent” merely because it uses a similarly named method. Record repository URL, immutable commit/release, licence, adapter revision, parameters, information access, calibration protocol, and deviations from the publication. If a comparator cannot be reproduced, mark it unavailable with a reason rather than substituting an unlabelled approximation.

## Statistical protocol

- Preserve the existing minimum of seven datasets and 50 seeds per dataset. For a full 8-cell factorial panel, explicitly report the realized dataset × seed × scenario cell counts; missing cells are coverage failures, not zero-valued results.
- Tune only on the training datasets within each leave-one-dataset-out fold. Freeze configuration before the held-out dataset run.
- Use identical registered seeds/scenarios and paired stream assignments for arms where meaningful. Keep dataset-level effects visible.
- Report per-dataset, per-scenario, and aggregate paired effects. Use the current hierarchical bootstrap as one interval estimator, but also show per-dataset signs and leave-one-dataset-out sensitivity. Seven dataset clusters are few; do not imply that one interval estimator eliminates uncertainty about cross-domain generalization.
- Pre-register primary metrics, direction, missing-data handling, exclusions, failure criteria, and the rule for interpreting intervals before measuring the candidate.
- Report results for null/no-impact shifts, abrupt and gradual changes, recurring concepts, delayed outcomes, autocorrelated streams, correlated errors, missingness bursts, predictor corruption, noisy labels, and class-/feature-space shifts.
- Preserve raw traces and failed runs. Run count, timeout, compilation, or artifact failures must not be converted into successful measurements.

## Candidate success criteria (aspirational; tune before test runs)

These are proposed research gates, not results or guarantees:

1. **Integrity gate:** zero forbidden state updates/replay admissions caused by invalid evidence in the registered hard-gate panel.
2. **Null calibration:** for anytime-valid methods, empirical false-alarm results and uncertainty are consistent with their stated assumptions and nominal error budget; separately disclose assumption violations such as autocorrelation or informative missingness.
3. **Decision-value gate:** on held-out datasets, the staged benefit-gated policy lowers task loss versus detector-triggered adaptation without exceeding preregistered resource/risk limits. Report the effect size and paired interval, not just a sign.
4. **Generalization gate:** no result is called a general win if it depends on one dataset or one drift shape. Publish all seven dataset effects, leave-one-dataset-out results, and per-cell uncertainty.
5. **Recovery gate:** every injected harmful adaptation must either be correctly rejected before commit or rolled back within its registered recovery bound; recovery limits must be specified before the run.
6. **Cost gate:** publish the quality/latency/memory/resource trade-off against each relevant baseline on the target hardware and in a controlled reference environment.

A full coverage panel is necessary for the registered comparison, but it is not proof of superiority. An interval entirely above zero is not, by itself, proof of broad state-of-the-art superiority if tuning leaked, datasets are not representative, hard gates fail, or the candidate wins only on an unregistered subset.

## Literature and reproducibility links

- Cerqueira et al., *A Framework for Evaluating and Benchmarking Concept Drift Detection Methods*, KDD 2026. https://doi.org/10.1145/3770855.3819070
- Lu et al., *Autonomous Concept Drift Threshold Determination*, AAAI 2026. https://doi.org/10.1609/aaai.v40i29.39586
- Kuang & Xia, *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales*, arXiv:2609.00536 (September 2026). https://arxiv.org/abs/2609.00536
- Shaer et al., *Testing for Distribution Shifts with Conditional Conformal Test Martingales*, ICML 2026. https://proceedings.mlr.press/v306/shaer26a.html
- Liu et al., *Latency-Aware Adaptive Learning for Nonstationary Data Streams: Coordinating Concept Drift and Delayed Supervision*, IEEE Transactions on Pattern Analysis and Machine Intelligence, ahead of print (12 October 2026). https://doi.org/10.1109/TPAMI.2026.3742576
- *Concept drift detection in district heating: A characterization framework and detector-centric benchmark*, Energy and AI 25 (September 2026), 100788. https://doi.org/10.1016/j.egyai.2026.100788

## Industry baseline and competitive thesis

The competitive baseline should include production observability systems, not only academic detectors. The public documentation reviewed describes mature capabilities for monitoring distribution/data quality, model quality when actual outcomes arrive, performance/latency/cost thresholds, alerts, and slice-level investigation:

| Industry reference | Publicly documented strength | Symthaea comparison to run |
| --- | --- | --- |
| [WhyLabs monitors](https://docs.whylabs.ai/docs/monitor-manager/) | Data drift, data quality, concept/label drift, and model-performance monitors. | Compare false alarms, detection latency, delayed-label behavior, missingness robustness, and monitoring overhead on the same streams. |
| [Arize AX monitoring](https://arize.com/docs/ax/get-started/get-started-production) | Metric thresholds, alerts, tracing, and analysis of performance/cost/latency signals. | Compare end-to-end detection-to-decision latency, cost per verified recovery, and fraction of interventions that improve held-out outcomes. |
| [Evidently](https://github.com/evidentlyai/evidently) | Broad evaluation library for data quality, drift tests, prediction quality, and LLM outputs. | Use relevant open-source tests as detector/quality adapters where feasible; record exact package version, metric settings, and runtime cost. |
| [Azure ML model monitoring](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring) | Multiple drift/data-quality signals and objective model-performance checks when actuals are available. | Compare feature/prediction drift and delayed actuals separately; do not equate a statistically significant shift with a need to adapt. |

This is not a claim that these systems cannot support custom interventions. The current comparison is based on what their public documentation describes, not an exhaustive audit of vendor internals. The benchmark should use product adapters or exportable traces when access is available, and otherwise compare against documented open baselines while clearly marking the gap.

**The proposed differentiation is an evidence-backed intervention contract.** Monitoring tells an operator what changed. Symthaea should also be able to show whether an intervention helped, what it cost, how it recovered, and whether the evidence used at decision time was valid. To substantiate that differentiation, measure:
- alarm-to-action and action-to-verified-recovery latency, not detector latency alone;
- task-loss improvement and harm rate per intervention, with no-adapt and benefit-gated controls;
- false adaptation under nuisance shifts, label delay, autocorrelation, and missingness;
- rollback correctness and recovery time after deliberately harmful candidate updates;
- total compute, memory, event throughput, and review/label cost at matched service-level objectives.

The fair comparison must hold input streams, label availability, decision budgets, workload, and declared access tracks constant. If a proprietary platform cannot run the same replay or expose decision traces, report an **interface/observability comparison** instead of inventing a numerical head-to-head score.

## Implementation order

1. Specify schema and event ledger; add red-team negative fixtures for leakage and arm mismatch.
2. Implement a driver-owned replay interface and raw per-cycle receipt. Do not enlarge the detector roster first.
3. Wire no-adapt, detector-triggered, benefit-gated, staged rollback, and oracle-timed diagnostic arms.
4. Re-run the scalar detectors and comparator adapters on the same access track and stream.
5. Publish a small dry-run trace to prove interface correctness only; label it synthetic/not measured.
6. Execute the preregistered held-out panel and publish raw records, hashes, uncertainty, failures, and cost.
7. Update claims only after independent review of the exact commit and artifacts.

**Non-claim:** this document introduces a research protocol only. It does not implement the replay engine, validate the current Rust branch, establish a statistical guarantee for the PRM prototype, or report performance gains.

## Independent receipt-to-panel audit

The panel aggregator previously checked receipt-link metadata without independently reopening every raw receipt. The new audit command reopens and hashes the preregistered protocol, panel input, every scorer input, and every replay receipt; invokes the receipt-bound scorer again for all 2,800 cells; requires exact score equality; checks that paths remain inside the bundle root; rejects duplicate identities, reused input or receipt bytes, reused run IDs, and non-replay receipts; then recomputes the full panel comparison.

Run it with:

```bash
cargo run --locked -p xtask -- audit-cognitive-adaptation-panel \
  --input path/to/audit-bundle.json \
  --expected-bundle-sha256 PUBLISHED_64_CHARACTER_SHA256 \
  --out reports/cognitive-adaptation-audit.json
```

The JSON bundle manifest is version 1 and includes `panel_path`, `panel_sha256`, `protocol_path`, `protocol_sha256`, and a `cells` array. Every cell declares `dataset_id`, `seed`, `scenario_id`, `run_id`, `utility_input_path`, `utility_input_sha256`, `replay_receipt_path`, and `replay_receipt_sha256`. Each SHA-256 value must be 64 lowercase hexadecimal characters and match raw bytes. Paths are relative to the manifest directory; absolute paths, `.` or `..` components, and symlinks that resolve outside that directory are rejected.

**Preregistration discipline:** publish the audit-bundle digest and exact protocol and input commitments before examining held-out outcomes. If the bundle is assembled after results are visible, it still enables reproducibility but does not by itself prevent post-hoc selection. A successful audit is not cryptographic proof that a runner executed honestly, that the oracle channel was technically inaccessible, or that the dataset represents deployment. Those remain separate controls and limitations.

## Driver-owned replay boundary and dual-latency evaluation (October 2026)

A Rust contract prototype now exists in `xtask/src/cognitive_replay_driver.rs`. It gives adapters only newly delivered features and newly revealed outcomes, keeps hidden targets in driver-owned state, records each decision's eligible and consumed evidence IDs, and independently audits delivery sets and reveal timing. It is not yet connected to the production cognitive loop or the confirmatory 2,800-cell panel. The typed API limits accidental information leakage; it is not a security sandbox against adapter code that has independent access to the source dataset.

The current prototype uses a fixed positive label delay. That is only Phase A. A research-grade replay driver should treat at least two clocks independently:

- **Label-reveal latency:** when the outcome associated with an earlier prediction or decision becomes observable.
- **Intervention-effect latency:** how long after staging/committing an intervention before its task-level benefit or harm can be estimated.

These clocks should be independently preregistered and varied. The driver should ultimately support per-outcome reveal schedules, out-of-order arrivals, explicitly missing labels, and right-censored outcomes. A label that has not arrived is not a zero loss, evidence of no benefit, or permission to substitute a later sample.

### Required latency and censoring experiment matrix

Cross the following factors without mixing information-access tracks:

1. Label delay: immediate-after-decision, short, long, variable/per-outcome, and out-of-order reveal.
2. Outcome completeness: complete labels, missing-at-random labels, missingness correlated with observed features, and explicitly censored outcomes. State the assumptions; do not label missing-not-at-random data as ignorable.
3. Intervention-effect horizon: immediate effect, delayed benefit, delayed harm, transient benefit, and effect not identifiable within the registered horizon.
4. Drift: abrupt, gradual, recurring, and nuisance/no-impact changes.
5. Policy: no adaptation, detector-triggered, benefit-gated, staged verify/rollback, periodic adaptation, and oracle-only diagnostic.

Preregister the target estimand for each cell. For outcomes censored before the evaluation horizon, use a declared censoring-aware estimand or classify the cell as not identified; do not silently drop it or count it as a policy failure. When policy actions influence which labels are observed, explicitly address observation-selection bias and the conditions needed for a causal comparison.

### Acceptance gates before confirmatory runs

- Prefix invariance: changing hidden future outcomes must not change earlier actions.
- Reveal-time correctness: no outcome may affect a decision before its registered availability tick, even under out-of-order arrivals or checkpoint/restore.
- Censoring discipline: missing, censored, delayed, and observed outcomes have distinct states and are never silently coerced to numeric sentinels.
- Intervention timing: verification begins only after the registered effect-evaluation window is observable; early evidence cannot stand in for a mature task outcome.
- Equal-information comparisons: candidate and baseline policies receive matched feature and label access, with any oracle channel separately typed and excluded from deployable rankings.
- Independent audit: rebuild eligible and consumed evidence sets from the driver-authored trace and recompute scores from raw per-cycle records.
- Positive and negative controls: a policy that ignores every valid outcome must fail a positive control, while nuisance shifts and future-data mutation must not create a false gain.

### Updated field comparison

Current monitoring platforms such as WhyLabs document performance tracking with delayed or partial ground truth, and Arize documents drift, performance, and data-quality monitoring. Symthaea should not claim superiority merely for timestamping labels or emitting provenance reports. The differentiating test is whether a policy improves held-out task outcomes **at the same information budget**, while accounting for intervention-effect latency, missing/censored feedback, recovery risk, compute and operational cost. Any numerical industry comparison requires matched workloads, comparable access to actual outcomes, and a reproducible measurement method.

Research motivation: the KDD 2026 drift benchmark advocates controlled perturbations of real streams and leave-one-dataset-out tuning ([paper](https://doi.org/10.1145/3770855.3819070)); the October 2026 TPAMI work couples drift with verification latency ([paper record](https://pubmed.ncbi.nlm.nih.gov/42862614/)); a recent experimental survey addresses delayed and partially labelled data streams ([paper](https://doi.org/10.1016/j.dsp.2026.106325)); and AAAI 2026 work on uplift modeling with delayed feedback discusses how delayed treatment effects distort outcome assessment ([paper](https://doi.org/10.1609/aaai.v40i19.38686)). These sources motivate experimental dimensions and safeguards; they do not establish that Symthaea outperforms those methods.
