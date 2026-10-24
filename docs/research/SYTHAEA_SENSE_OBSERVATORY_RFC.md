# Symthaea Sense: Evidence-Preserving Scientific Instrumentation

**Status:** RFC / proposed roadmap  
**Date:** 2026-10-17  
**Scope:** Public Symthaea repository; sensor data contracts, active experiment design, evaluation, and an incremental instrument program.

## Executive thesis

Build Symthaea as an **evidence-preserving, active scientific instrument system**, not a model that consumes sensor vectors and reports anomalies. The research loop is:

1. Observe a physical system with known instrument characteristics.
2. Preserve measurements, provenance, quality flags, calibration evidence, and uncertainty.
3. Construct competing explanations.
4. Select an affordable next measurement that distinguishes those explanations.
5. Run an experiment with preregistered predictions and controls.
6. Evaluate against references, negative controls, and held-out data.
7. Publish a replayable evidence bundle and update beliefs only as strongly as results justify.

This does not assume that an anomaly is new physics. The goal is to make unusual observations easier to notice and harder to misinterpret.

## Paradigm shifts

### P1. From sensor fusion to evidence fusion

Existing sensor pathways encode values into HDC representations. These can help perception, but an HDC vector is a derived feature, not the source of truth. Preserve raw or minimally processed samples, units, conversion metadata, calibration records, uncertainty, timestamps, instrument identity, and processing versions alongside learned representations.

A reasoning result should link back to exact observation IDs and code/model revisions. Missing uncertainty remains missing; it must never silently become zero.

### P2. From passive acquisition to active experiment design

Make the next-measurement choice an explicit policy. Given competing hypotheses, choose the next sensor, location, sampling interval, or intervention expected to reduce uncertainty per unit of cost, time, energy, or risk.

Compare Symthaea against random selection, fixed-period sampling, and a simple information-gain baseline. A complex policy earns adoption only if it beats those baselines on held-out scenarios.

### P3. From correlation to discriminating causal tests

When a relationship is detected, generate predictions that differ between plausible explanations. Where safe and appropriate, use controlled interventions; otherwise use natural experiments, negative controls, and preregistered temporal predictions. Report correlations as correlations until the experimental design supports a causal claim.

### P4. From single instruments to interoperable observations

Use an internal typed schema and open exchange formats rather than making each vendor define semantics. Align external adapters with OGC SensorThings API and FAIR data principles. Do not claim standard conformance before implementing and testing the relevant requirements.

### P5. From headline scores to statistical evidence

Every benchmark records code revision, configuration, dataset split, seeds, hardware, metric definition, sample count, and uncertainty interval. Report false alarms and negative-control outcomes alongside recall. NIST AI 800-3's 2026 discussion of statistical AI evaluation is a useful starting point for avoiding misleading point estimates and hidden assumptions.

### P6. From monolithic perception to multi-rate, fault-aware sensing

Sensors have different clocks, bandwidths, ranges, warm-up behavior, and failure modes. Keep sensor-local sequence numbers and monotonic time, record wall-clock quality separately, detect missing data and drift, and never conceal a saturated or stale sensor behind a fused average.

### P7. From opaque novelty to a claim ladder

Use explicit claim levels:

- **L0 — Raw observation:** recorded instrument output.
- **L1 — Validated measurement:** integrity and fitness-for-purpose checks passed.
- **L2 — Replicated pattern:** pattern reappears under independent or repeated acquisition.
- **L3 — Discriminated explanation:** key alternatives and negative controls tested.
- **L4 — Independently reproduced result:** an independent operator or setup reproduces the procedure and result.

Do not jump levels because an AI confidence score is high. Lower-level claims can be useful when clearly labeled.

## Existing starting points

The public repository already contains useful building blocks:

- `src/perception/sensor_fusion.rs`: IMU-to-HDC fusion.
- `crates/domains/symthaea-hal/src/sensor.rs`: embedded sensor interface and validation of decoded readings.
- `src/swarm/mesh/sensor.rs` and `src/swarm/mesh/sensor_iot.rs`: sensor registry and IoT input adapters.
- `docs/robotics/INFRASTRUCTURE_HAZARDS_AND_SENSORIUM.md` and `docs/robotics/CLIME_HAZARDS_AND_SENSORIUM.md`: domain-specific sensor bundles and hazards.

These are not yet a unified metrology layer. Numeric arrays and string metadata alone are insufficient to establish uncertainty-aware, traceable measurements.

## Proposed architecture

```text
Physical sensor / instrument
  -> vendor or bus adapter
  -> typed observation envelope
       - observation and instrument IDs
       - observed property, value, unit
       - capture wall clock + session-local monotonic time + sequence
       - standard uncertainty (or explicitly absent)
       - calibration reference / declared unknown state
       - quality flags and raw-sample digest
  -> append-only capture/replay bundle
  -> domain transformations (versioned)
  -> HDC / temporal / causal representations
  -> competing hypotheses + next-experiment policy
  -> protocol, controls, result, uncertainty, evidence bundle
```

The typed envelope is additive; it must not require immediate rewrites of existing adapters. The first software slice introduces the contract in `symthaea-hal` and tests structural validation. Follow-on work can adapt one existing sensor and add deterministic replay.

### Observation contract invariants

- A record has schema version, unique observation ID, stable sensor and instrument IDs, a named channel/property, numeric value, and unit.
- A measurement must be finite. If standard uncertainty is supplied, it must be finite and non-negative; absence is not zero.
- Timestamps distinguish wall clock from monotonic time. Monotonic timestamps are meaningful within the source session unless synchronization is established.
- Calibration state is explicit. A calibration record pointer is not, by itself, proof of metrological traceability.
- Quality flags (stale, saturated, out-of-range, sensor fault, unsynchronized clock, dropped samples, operator-suspect) are retained; consumers decide fitness for a specified task.
- A raw digest can reveal changes only when compared with a trusted digest. It does not authenticate a sensor or prove provenance on its own.
- Units should use UCUM-compatible codes where practical. Cross-sensor derived values must document conversions and uncertainty propagation.
- Every derived result references source observations, algorithm version, configuration, and relevant calibration context.

## Benchmarks and proposed gates

All numeric gates are **proposed targets**, not measured current performance. Establish a baseline first, publish uncertainty intervals, and revise a gate only through documented rationale—not to hide a regression.

| ID | Benchmark | Protocol | Proposed initial gate |
|---|---|---|---|
| SE-01 | Schema and integrity | Tests for empty IDs/units, NaN/infinity, uncertainty, schema version, malformed digest, and flagged readings | 100% of malformed fixtures rejected; valid flagged fixtures preserved |
| SE-02 | Deterministic replay | Replay identical bundles twice with identical code/config | Exact equality for deterministic transforms; record every input/output ID and config |
| SE-03 | Clock and gap handling | Inject offset, drift, out-of-order/repeated sequence, dropped samples, and wall-clock discontinuity | 100% injected faults detected or explicitly classified unsupported; no silent relabeling |
| SE-04 | Sensor fault detection | Inject stuck-at, dropout, saturation, bias drift, spikes, and sensor disagreement | Target >= 95% detection of defined fault episodes at <= 1 contiguous false-alarm episode per sensor-hour on held-out suite; report per-fault results and point-level false-positive samples separately |
| SE-05 | Calibration/uncertainty | Compare against synthetic reference values, then bench references where available | For nominal 95% intervals, empirical coverage target 90–98% on the declared test distribution; report width and bias |
| SE-06 | Multimodal value | Compare fused predictor to best single sensor and simple concatenation/linear baselines | Target >= 10% relative error reduction on a preselected held-out task without worsening calibration or safety-relevant false alarms |
| SE-07 | Active sensing | Compare with random, fixed cadence, and greedy information-gain baselines under equal budgets | Target >= 15% fewer measurements to reach a prespecified uncertainty/identification threshold on held-out scenarios |
| SE-08 | Causal discrimination | Paired scenarios with confounders, interventions, nulls, and negative controls | Report precision/recall and false causal-claim rate; no positive causal claim on designated null controls |
| SE-09 | Novelty detection | Known injected anomalies plus benign distribution shifts | Report PR-AUC, recall at fixed false-alarm budget, detection delay, and abstention rate; do not optimize accuracy alone |
| SE-10 | Resource efficiency | Measure CPU, RAM, storage, energy, throughput, latency on named hardware | Establish per-device baseline first; compare against non-HDC and single-sensor baselines |
| SE-11 | Cross-device generalization | Tune on one device/site; evaluate on unseen devices/sites and calibration histories | Report calibration drift and degradation; use >= 80% in-domain performance only where task/labels support comparison |
| SE-12 | Replicability | Independent replay, second capture after recalibration, eventually independent operator | 100% manifest completeness; surface every discrepancy; independent reproduction required before L4 |
| SE-13 | Likelihood-model robustness | Rank candidates across plausible likelihood scenarios; compare expected-gain and worst-case-gain policies on held-out misspecification cases | Report expected and worst-case regret; initial target >= 20% reduction in worst-case regret versus single-model ranking, while limiting expected-gain loss to <= 10% on correctly specified controls |

Thresholds are engineering proposals, not universal scientific standards. For streaming metrics, contiguous false-positive samples count as one alarm episode; sample-level false positives are reported separately. Safety-critical limits must come from the relevant hazard analysis. If trustworthy labels or reference measurements are unavailable, mark a benchmark exploratory or blocked instead of inventing a score.

## First controlled benchmark: SenseBench-0

Before hardware is purchased, create a deterministic synthetic environment with ground truth and replayable event manifests. Include:

1. Stable baselines and slow periodic changes.
2. A real shared cause affecting two sensors.
3. Correlated sensors with no direct causal relation.
4. Single-sensor step change, spike, stuck-at value, dropout, saturation, and bias drift.
5. Clock offset, sampling-rate mismatch, out-of-order messages, and dropped sequences.
6. A null dataset with no injected phenomenon.
7. A blind test set whose seed and event labels are hidden from the detection policy during evaluation.

First task: detect an injected step change while controlling false alarms on the null suite. Second: distinguish shared-cause from spurious-correlation scenarios. Third: test whether active sampling finds a change with fewer observations than random and fixed-rate baselines.

Define generator version, seed policy, sample counts, train/tuning/test separation, event labels, metric formulas, and exact commit/config for each result.

## Comparative evaluation beyond synthetic data

SenseBench-0 is a controllable correctness harness, not the final leaderboard. Once our stream adapter and evaluator contract stabilize, add external comparisons:

- **Numenta Anomaly Benchmark (NAB):** a complementary streaming time-series suite with labeled real-world and artificial series and explicit real-time scoring. Keep the upstream evaluation protocol intact; publish our adapter and exact configuration rather than cherry-picking subsets.
- **Domain-specific reference data:** for each real instrument modality, define known calibration/reference stimuli, benign shifts, and documented fault modes. Generic anomaly sets cannot establish metrological accuracy.
- **Active experiment design:** compare random selection, fixed cadence, greedy predictive variance, and expected information gain under equal acquisition budgets. Recent spectral-experiment work uses Bayesian active learning with explicit model selection; causal intervention-design research also demonstrates why interventions should be selected for discrimination rather than novelty volume alone.

Do not pool scores from different datasets into one number unless their task, label meaning, event prevalence, and cost model justify it. Publish per-dataset metrics and uncertainty first.

## Active experiment-ranking primitive

The first code slice adds `rank_experiments` to `symthaea-sensebench`. It accepts explicit hypothesis priors and candidate-specific outcome likelihood matrices, computes expected posterior-entropy reduction in bits, and ranks by information gain per declared cost. It fails closed on malformed priors, identifiers, cost, or likelihood distributions and uses deterministic tie-breaking.

This primitive narrows the gap between the roadmap and executable work, but it is not yet a complete active-science agent. A second API, `rank_experiments_robust`, accepts a weighted ensemble of plausible likelihood models and prioritizes experiments by worst-case expected information gain per cost, while exposing model-weighted expected gain and sensitivity across those models. Candidate identity, cost, and outcome-index semantics must be compatible across scenarios. The robust ranking is only as broad as the model set supplied; it cannot protect against omitted alternatives or make uncalibrated likelihoods trustworthy. The implementation does not produce hypotheses, connect to HDC/LTC, run experiments, or authorize physical actions. The scientific model supplying the probabilities must be separately evaluated and its assumptions recorded.

### Current synthetic causal-selection prototype

A first executable simulator now compares three policies across balanced trials for two candidate explanations, a direct-cause model and a shared-cause model. The candidate library includes a passive observation and a negative control whose outcome distributions are identical under both hypotheses, plus an intervention candidate whose outcome likelihoods differ. The active policy re-ranks candidates against the current posterior each round; random selection is a uniform candidate choice; passive-only always chooses the passive measurement.

The runner reports posterior probability on the true hypothesis, multiclass Brier score, top-hypothesis accuracy, threshold-reaching fraction, experiments needed among successful trials, and informative interventions used. It requires an even trial count to balance ground truth and uses separate action-selection and outcome streams (common random numbers across policies). Results are deterministic for a fixed seed/configuration.

The original `causal` suite is a *toy control*, not evidence that Symthaea can discover causal structure in real environments: it assumes correctly specified likelihoods and has a small hypothesis set. The new `blind-worlds` suite adds outcome-generating likelihood misspecification, a shared resource budget, a development-selected fixed-design baseline, predictive-variance and robust-selection policies, held-out regimes, calibration metrics, and null-truth causal-claim accounting. It remains synthetic and does not yet represent independently generated causal graph topologies or calibrated physical likelihoods.

### Model-misspecification regret fixture (software-only)

Run the synthetic fixture through the CLI:

```sh
SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
  cargo run -p symthaea-sensebench -- --suite robustness
```

The suite compares nominal single-model expected gain, nominal predictive variance, model-weighted expected gain, worst-case robust ranking, a predeclared fixed comparator, and the exact expected gain for uniform-random selection. It scores decisions against two declared toy worlds and one likelihood world held out from selector inputs. Reports include the chosen candidate, evaluator-oracle candidate, expected information gain under evaluator-only hidden-world likelihoods per declared cost, per-world regret, and mean/maximum regret.

The hand-authored fixture is parameterized to exercise the proposed SE-13 gate mechanics: formula-level reference arithmetic gives nominal maximum regret of about 0.2781 bits/cost and worst-case robust maximum regret of about 0.0101 bits/cost, a 96.36% reduction; information-gain loss on the correctly specified model-A control is about 3.51%. These are not Rust-executed results or empirical performance. They are expected values computed from deliberately constructed likelihood tables to test evaluator wiring and acceptance checks.

A crucial negative control remains: the fixed coarse comparator matches the robust policy's worst-case regret on this small fixture. It therefore does not establish active-policy superiority over strong fixed designs. It also does not yet meet the full research goal of hidden causal graph families, repeated-seed intervals, a sealed evaluator, calibrated likelihoods, or physical reference measurements. Those remain explicit next steps under the active-science benchmark issue.

### Outcome-generating blind-world benchmark

Run the new simulation suite:

```sh
SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
  cargo run -p symthaea-sensebench -- --suite blind-worlds --trials 512 --budget 8 --seed 42
```

This suite compares seven policies across six regime templates. Four deterministic, seed-derived parameter variants per template create 24 worlds: eight development worlds and 16 held-out worlds. The held-out regimes include intervention artifact, degraded sensing, effect reversal, and a null negative control. The fixed comparator is selected using development worlds only. Policy selection receives declared model scenarios; evaluator-only likelihood tables generate observed outcomes.

Per-world reports include multiclass Brier/log loss, top-hypothesis accuracy, calibration error, threshold rate and resource cost, declared-model versus evaluator-world expected information gain, myopic per-selection acquisition regret versus the hidden-world best affordable action, realized entropy change, and unsupported high-confidence causal claims when the true hypothesis is no direct causal effect. Negative entropy changes are not clamped. Acquisition regret is a local one-step comparator, not a globally optimal multi-step oracle. Trial-level intervals are conditional on each parameterized world; the variants are jittered versions of six hand-authored templates, not independently generated causal graph families.

The tables remain source-visible, so held-out is an evaluation split rather than secure blinding. This software-only suite makes no claim about physical causal discovery. Rust tests are authored but not executed in the current environment; exact-head workflow runs marked `skipped` and empty commit statuses do not establish a pass. See [the protocol](SENSEBENCH_BLIND_WORLD_PROTOCOL.md) for acceptance gates and limitations.

### Model adequacy and claim abstention

The v0.2 blind-world report adds prequential predictive log loss under the declared likelihood-model ensemble. Before each posterior update, the evaluator scores the observed outcome under the current hypothesis posterior marginalized over declared model scenarios. Each policy receives a model-adequacy cutoff calibrated at the 99th empirical percentile from independent-seed simulations generated only from the declared model scenarios. This calibration is disjoint from evaluation-world outcome streams.

A trial above the cutoff triggers a **post-trial claim-abstention diagnostic**. Reports show the model-adequacy alarm rate, fraction retained after gating, unsupported high-confidence causal claims on null truth before gating, and the corresponding rate after gating. A gate that prevents false claims by rejecting almost everything must expose that loss of coverage rather than receiving a free safety win.

This is an initial synthetic diagnostic, not a formally valid universal goodness-of-fit test. No alarm does not prove adequacy, and an alarm does not prove that every current hypothesis is false. The threshold is conditional on the declared model ensemble, policy, budget, posterior threshold and generator; any change to those dependencies requires requalification. Future work should compare with sequentially valid model checks/e-values, measure detection delay, and validate false-alarm and power trade-offs on independent reference streams. The separate [SENSE-EXP-002 follow-up](https://github.com/Luminous-Dynamics/symthaea/issues/7825) investigates a literature-faithful representative/informative/error-de-amplifying acquisition objective and must remain distinct from this heuristic gate.

### Generated SCM graph-recovery benchmark

The new `scm-graph` suite adds a graph-level complement to the three-hypothesis `blind-worlds` benchmark. It enumerates all 25 labeled DAGs over three observed variables. Seven graph-family templates each generate four topology instances, then every exact topology is reused under four separate mechanism regimes, producing 28 unique topology draws and 112 graph × mechanism worlds. The partition is factorized: two topology families and two positive-effect mechanism regimes are development-only; mixed-sign and near-null mechanisms are held out across every topology; any world with a held-out factor is held out. This leaves 16 development worlds and 96 held-out worlds. Separate deterministic RNG streams generate topology and mechanism parameters, and each exact graph variant is evaluated under every mechanism regime. The parameter draw for a given graph-variant index and mechanism regime is paired unchanged across graph families, so a family-specific coefficient signature is not encoded by the generator. The v0.6 report separates topology-only, mechanism-only and joint-shift subsets. For aggregate uncertainty, mechanism-regime rows are first averaged within each topology, then intervals are calculated across topology instances instead of incorrectly treating a repeated graph under four mechanisms as four independent topologies. It also reports paired policy deltas against nominal EIG, the development-fixed design, uniform-random action, and both asymptotic and exact-conditional PC baselines on matching worlds, with uncertainty across topology instances. The acceptance rule must compare against the strong fixed design and check coverage/false-claim regressions; no policy superiority is asserted until its paired interval and trade-offs support that conclusion. There are 28 unique topology draws nested within seven hand-authored families, so this interval is conditional on the fixed family design rather than generalization to the universe of SCMs. Eleven policies are compared under a shared measurement budget, including separate asymptotic and exact-conditional observational PC comparators. The exact-conditional comparator enumerates the finite G-squared null distribution conditional on observed binary-table margins; it is a small finite-sample reference, not a universal CI-testing solution. Both PC baselines use passive samples, project their finite-sample structure estimates to a valid three-node CPDAG, and abstain from fully oriented DAG claims. The asymptotic baseline's 5% cutoffs are a transparent starting point, not a calibrated finite-sample guarantee; the exact-conditional baseline is a finite-sample reference limited to the binary three-variable setting. Schema v6 reports the mean number of raw PC edge marks changed by projection to a valid CPDAG and the fraction of trials where projection changes any mark for both PC comparators, exposing the cost of the repair step without claiming the underlying statistical decisions are correct. Exact binary outcome distributions are enumerated from candidate graphs; outcomes are sampled from the hidden graph and mechanism parameters.

Policies include nominal expected information gain, predictive outcome entropy, model-weighted gain, worst-case robust selection, a fixed action tuned on development-family mechanism parameters, uniform random actions, passive-only observation, an optimistic known-mechanism oracle, and a structure-blind random-graph submission. The oracle is a diagnostic upper-bound comparator, not a deployable policy. The posterior-update rule remains nominal across action-selection policies, isolating selection differences from inference differences.

Per-world metrics score skeleton F1, directed-edge F1, a clearly defined structural Hamming distance, exact graph recovery, posterior mass on the true graph, graph-posterior Brier score, graph-claim coverage and false accepted-claim rate. The benchmark now also enumerates the 11 Markov-equivalence classes over three variables, selects the CPDAG with greatest summed posterior mass, reports compelled-direction F1 and CPDAG edge-mark distance, and separates full-DAG confidence from class-level claim coverage. This is a stronger causal-validity contract: observationally unidentifiable directions are not automatically treated as errors, while unsupported compelled-arrow claims are penalized. Held-out aggregates keep each generated world as a unit, rather than pooling trials as independent evidence of generalization. The report states that four parameter variants per hand-authored family are insufficient to establish broad transfer.

The scope is deliberately small and auditable: all three-node DAGs are represented, with binary linear-probability structural equations and no latent confounders. Markov equivalence uses the same-skeleton/same-unshielded-collider criterion; this assumes the causal-sufficiency/faithfulness regime used by the generator and must be cross-checked against an independent trusted CPDAG implementation before scaling. The benchmark now scores CPDAG equivalence for this three-variable catalogue, but does not yet support continuous/nonlinear SCMs, temporal graphs, latent confounders, or larger variable counts. Schema v6 also audits how often either PC baseline's raw edge marks need to be changed to fit a valid catalogue CPDAG; it does not yet establish CI-test calibration or expose every per-test p-value. The current implementation is a benchmark substrate, not a claim of scientific causal discovery. Protocol: [SENSEBENCH_SCM_GRAPH_PROTOCOL.md](SENSEBENCH_SCM_GRAPH_PROTOCOL.md).

### Domain-specific fixture: geophysics

The `symthaea-sensebench/src/geophysics.rs` fixture reuses the generic experiment-ranker with two abstract Earth-interior hypotheses and three named observation candidates. All probabilities and costs are hand-authored toy values used to check determinism, entropy arithmetic, expected-gain ranking, and robust ranking across two alternative likelihood tables. In one toy table the targeted magnetic candidate is highly discriminating; in the other it is deliberately made uninformative, while a coarse comparator stays moderately informative. This checks that the maximin API responds to model uncertainty. The values are **not** fit to geophysical measurements, validated physical forward models, or claims about the Earth's core. No real-world conclusion may be inferred from its ranking. Before a domain-specific scientific selector is meaningful, freeze a documented generative/forward model, derive or calibrate likelihoods against public observations, disclose uncertainty, and test alternatives with independent references.

### Required evaluation

- A perfectly discriminating binary experiment with equal priors should provide 1 bit of expected information gain.
- An outcome distribution identical under all hypotheses should provide 0 bits.
- Under a finite budget, ranking by information gain per cost should beat random choice on held-out tasks where informative actions differ.
- Compare against random, fixed-design, predictive-variance, and expected-information-gain baselines; do not compare only against intentionally weak methods.
- Report both identification accuracy and resource cost. A high information-gain estimate from a miscalibrated model is not a successful outcome.
- In physical environments, an independent safety layer must veto experiments outside predeclared limits; maximizing information is never sufficient justification for a risky action.

## Hardware roadmap

### Phase 0 — Software first (no new hardware required)

- Introduce the typed observation contract.
- Build deterministic capture/replay and synthetic fault fixtures.
- Validate quality flags, missing uncertainty, schema evolution, and timestamps.
- Run baseline comparisons locally; do not wait for CI to begin gathering evidence.

### Phase 1 — Bench-top SenseNode

Start with a small, synchronized environmental setup:

- Temperature/humidity reference sensor.
- CO2 sensor, clearly distinguished from general gas-response indicators.
- Multichannel visible/near-infrared spectral sensor.
- Low-resolution thermal array.
- Three-axis magnetometer.
- Host capable of timestamped acquisition and local storage.

Measure alongside a reference or controlled stimulus; log warm-up, placement, power supply, calibration state, and environmental conditions. Sensors do not share units, bandwidths, or response times by default.

### Phase 2 — Expand by research question

Add RF spectrum analysis, acoustic/vibration sensing, radiation instrumentation, or chemistry-specific detectors only when a documented hypothesis calls for them. Each modality requires its own safety requirements, bandwidth, calibration, and references. More modalities are not automatically better.

### Phase 3 — Active scientific loop

Allow a bounded policy to suggest the next observation or experiment. Start in simulation and human-approved bench experiments. Require predeclared constraints, abort limits, and a record of proposed and executed actions. Do not allow autonomous hazardous emissions, radiation exposure, biological intervention, or other risky experiments simply because they maximize information gain.

### Phase 4 — Shared observation network

Publish schema, provenance manifests, replayable datasets, benchmark reports, adapter contracts, and independent replication instructions. Target interoperable records and FAIR reuse; implement and test standards rather than claiming compliance by name alone.

## Benchmark report provenance

The CLI now requires `SENSEBENCH_CODE_REVISION` before producing benchmark reports. Each anomaly report embeds the complete scenario manifest (schema/generator version, scenario ID/type, seed, sample count, sample interval, and split) plus evaluator configuration. The causal report records generator version, code revision, seed, trial count, experiment budget, posterior threshold, and per-policy summaries. The `blind-worlds` report also records the root seed, number of variants per regime, development-selected fixed candidate, held-out counts, per-world policy summaries, and its interpretation boundary. `--allow-unversioned` is an explicit exploratory escape hatch; such output must not support benchmark claims. Operators should run from a clean worktree because the CLI records the caller-supplied commit string rather than inspecting Git state itself.

## Exit criteria for the first milestone

The first milestone is complete only when:

- The typed envelope is implemented without breaking existing sensor adapters.
- Unit tests cover structural validation and serialization.
- At least one existing stream can be adapted without losing the original reading.
- Deterministic synthetic fixtures cover normal, faulty, stale, and unsynchronized records.
- Replay outputs are bound to input manifests and exact code/config revisions.
- Baseline scores include a null suite and false-alarm reporting.
- No scientific novelty claim is made from synthetic-only results.

## Risks and safeguards

- **Instrument illusion:** interference, drift, or aliasing is mistaken for a new phenomenon. Mitigate with references, negative controls, independent sensors, and replication.
- **Calibration theater:** metadata says “calibrated” without a credible record or uncertainty. Treat references as verifiable evidence, not trust badges.
- **Selection bias:** repeated exploratory probing eventually makes some patterns look significant. Separate exploration from confirmatory tests and log iterations.
- **Metric gaming:** tune against the benchmark until its score stops predicting real performance. Use held-out generators, hidden seeds, external tasks, and versioned rules.
- **Overclaiming causality:** a predictive relation is called an explanation. Use an explicit claim ladder and intervention/negative-control tests.
- **Unsafe autonomy:** information-gain maximization chooses a harmful experiment. Safety constraints precede utility optimization; risk-bearing actions require human approval.

## References

- NIST, [Policy on Metrological Traceability](https://www.nist.gov/calibrations/traceability).
- Possolo, Bruce & Watters, [Metrological Traceability: FAQs and NIST Policy, Technical Note 2156](https://www.nist.gov/metrology/metrological-traceability).
- NIST, [AI 800-3: Expanding the AI Evaluation Toolbox with Statistical Models](https://www.nist.gov/news-events/news/2026/02/new-report-expanding-ai-evaluation-toolbox-with-statistical-models), 2026-02-19.
- OGC, [SensorThings API](https://www.ogc.org/standards/sensorthings/), including Part 1: Sensing v1.1.
- Wilkinson et al., [The FAIR Guiding Principles for scientific data management and stewardship](https://doi.org/10.1038/sdata.2016.18), Scientific Data, 2016.
- W3C, [PROV Model Primer](https://www.w3.org/TR/prov-primer/).
