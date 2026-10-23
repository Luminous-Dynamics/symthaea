# SPACE-TELEM-BENCH-001 — Spacecraft telemetry anomaly benchmark

**Status:** protocol proposal; no dataset has been downloaded or benchmark executed in this change.  
**Tracking issue:** [#7947](https://github.com/Luminous-Dynamics/symthaea/issues/7947)  
**Parent architecture:** [#6255](https://github.com/Luminous-Dynamics/symthaea/issues/6255), registry freeze [#6257](https://github.com/Luminous-Dynamics/symthaea/issues/6257)  
**Separate task family:** spacecraft pose/domain transfer [#6261](https://github.com/Luminous-Dynamics/symthaea/issues/6261)

## Why this work exists

The current Symthaea example `crates/domains/symthaea-physics/examples/spacecraft_sepsis_eval.rs` evaluates an HDC anomaly detector on legacy SMAP/MSL telemetry and clinical sepsis data. This is useful as historical research, but it should not be mistaken for a mission-qualified spacecraft fault detector.

The spaceflight engineering opportunity is broader than maximizing ROC-AUC on one dataset. We need to measure whether a detector finds operationally important anomaly episodes early enough, keeps alert burden manageable, remains calibrated under domain shift, and gives the operator enough evidence to investigate the fault. Later, we can assess whether it helps isolate likely failure modes. Those are different claims and require different labels.

A 2025 review of the OPSSAT-AD work notes that widely used SMAP/MSL collections have limitations as sole time-series anomaly benchmarks, including unrealistic anomaly density, label issues and distribution shifts. Therefore, keep those evaluations for continuity, but add independent sources rather than silently replacing or rewriting earlier results.

## Paradigm shifts

### 1. From point scores to operational decision quality

Report event/episode recall, missed-event severity where defensibly labeled, detection delay, false-alert burden, calibration and abstention. A detector with attractive aggregate accuracy but an impractical number of false alarms is not a useful operations tool.

### 2. From random splits to source- and mission-aware transfer

Reproduce each external source's frozen official split first. Add secondary temporal, episode-grouped, channel-held-out, or mission-held-out splits only when the metadata supports them. Public test labels that influenced design or threshold selection must be marked exposed; they are not blind evidence. Fit normalization, feature selection, thresholding and learned representations using training/validation data only.

### 3. From a single “universal” model to calibrated specialization

Evaluate three configurations separately: (a) per-channel or per-mission specialist; (b) shared model trained across available channels/missions; and (c) zero-training/reference-only HDC baseline. Do not assume any one will generalize. Report when a model abstains because the input is novel, incomplete, or outside its validated envelope.

### 4. From model output to proof-carrying evidence

A result must bind the dataset release and digest, rights decision, split identity, preprocessing scope, model and configuration digests, random seeds, runtime profile, evaluator version, metrics, confidence intervals, limitations and report digest. A chart or self-reported score is not an evidence receipt.

### 5. From simulation success to a transfer ladder

Keep the following evidence classes separate: public real telemetry; synthetic injected telemetry; simulator scenarios; software-in-the-loop; hardware-in-the-loop; and operational/on-orbit observations. The orbital simulator may later produce controlled fault campaigns, but success there does not count as real-telemetry benchmark success.

## Candidate source registry

The corresponding machine-readable candidate registry is `docs/research/SPACE_TELEM_BENCH_001_registry.json`.

### Preferred first source: ESA Anomaly Dataset (ESA-ADB)

The official [ESA repository](https://github.com/esa/anomaly-dataset) describes curated anomaly annotations from three ESA missions and links to the [data record](https://zenodo.org/records/12528696), [ESA-ADB paper](https://arxiv.org/abs/2406.17826), and [challenge](https://www.kaggle.com/competitions/esa-adb-challenge). The repository LICENSE states CC BY 3.0 IGO. Treat that as an initial lead, not permission to skip verification of the exact downloaded data version, metadata, license attribution and intended use. The source must be re-frozen from its actual archive metadata before ingestion.

**Why first:** multi-mission origin offers a route to stronger transfer testing than a single-mission-only evaluation. The exact channels, label granularity, split semantics and mission identifiers must be inspected before claiming that a leave-one-mission-out test is possible.

### Complementary source: OPSSAT-AD

The [2025 Scientific Data article](https://www.nature.com/articles/s41597-025-05035-3) describes 2,123 telemetry segments across nine channels, raw fragments plus 18 engineered features, an official 1,494-train / 529-test split and 30 published supervised/unsupervised baselines. The associated source package is linked from [KP Labs' repository](https://github.com/kplabs-pl/OPS-SAT-AD) and [Zenodo record](https://zenodo.org/records/15108715). The article received a correction on 18 August 2025; consult the corrected version at [doi:10.1038/s41597-025-05751-w](https://www.nature.com/articles/s41597-025-05751-w).

The paper reports seven classification metrics: accuracy, precision, recall, F1, Matthews correlation coefficient (MCC), ROC-AUC and PR-AUC. Calculate these on the exact published split as a reproduction baseline. Preserve the raw-fragment task and engineered-feature task as separate benchmark subjects. **Do not infer the data archive's license from the paper's open-access status**; admit use only after the exact data-record terms are inspected.

### Historical continuity only: SMAP/MSL

Retain the existing example and its old numbers as historical references. Add no implied equivalence between its split, metric definitions, preprocessing, or data representation and either new source. Do not use SMAP/MSL as the sole evidence for spacecraft anomaly performance.

## Benchmark contract

### Subject identity

Every run must pin:

- source and exact release/version/date;
- archive URL and cryptographic digest (once downloaded);
- source-specific data license, attribution and admitted use;
- official split file/digest and any secondary split generator/version;
- target definition and label granularity (sample, segment, interval, episode, or root cause);
- data channels and sensor modalities actually available to the detector;
- preprocessing implementation and which split fitted every learned statistic;
- model/algorithm generation, configuration, seed and dependency/build identity;
- evaluator version, host CPU/GPU, precision mode, latency measurement method and resource limits;
- test exposure status, missingness policy, applicability boundary and claim ceiling.

A changed archive, test split, label mapping, evaluator, input policy or benchmark metric creates a new subject generation rather than quietly overwriting the prior result.

### Required results

1. **Source fidelity:** source-provided evaluation is reproduced on the exact official split; report values and explain every discrepancy.
2. **Per-source transparency:** results are presented per dataset and per channel where available. Do not combine unlike datasets into a universal score.
3. **Classification metrics:** calculate the source-defined metrics; for OPSSAT-AD emit all seven published metrics.
4. **Operations metrics:** where timestamp and interval semantics permit, report event recall, detection delay, false alarms per operating hour/day, and alert episodes after a declared debouncing rule. On segment-only data, report false positives per 1,000 nominal segments instead; do not manufacture wall-clock rates from non-temporal samples.
5. **Calibration and abstention:** where supported, report Brier score or reliability plots plus coverage-risk curves. State whether confidence is empirically calibrated.
6. **Uncertainty:** report bootstrap confidence intervals using the meaningful independent unit (episode, time block, channel or mission) instead of treating correlated samples as independent.
7. **Compute:** report p50/p95/p99 inference latency, peak memory and device profile. Do not claim onboard viability until the target processor and real sampling deadline are specified.
8. **Robustness:** run supported missing-data, data-gap, sampling-rate, amplitude-shift and channel-dropout cases. Clearly distinguish naturally occurring conditions from synthetic interventions.

### Baselines

- Source-published baseline on the exact source split, where reproducible.
- A simple, interpretable baseline such as rolling median/MAD or EWMA.
- A frozen prior Symthaea generation.
- HDC zero-training/reference-only configuration as its own declared method, not as an assumed winner.
- A data-quality-only baseline using missingness, segment length and sampling frequency to expose shortcut learning.

### Negative controls / anti-leakage gates

- Shuffled labels should not yield credible performance above the declared chance/null distribution.
- A no-signal or constant-input baseline must be reported.
- Corrupting channel/time alignment should be detectable by the evaluator or produce a clearly degraded result; silently mixing labels and features is a hard failure.
- A deliberately test-fitted normalizer / threshold is a test fixture for the leakage detector and must never be admitted as a valid model run.
- Validation/test labels cannot be used to tune a threshold, select a model or repeatedly decide which variant to keep without marking that test as exposed.
- Test every preprocessing operation against a train-only fit contract; document how missing, infinite and out-of-range values are handled.

## Proposed gates

These are project targets, not aerospace standards and not results already achieved.

| Gate | Pass condition | Does not establish |
|---|---|---|
| G0 — source admission | Exact artifact identity, checksums, terms/rights, attribution, label semantics, split identity and data handling are recorded; unknown rights block ingestion. | Permission to use an unverified data copy or claim of data completeness. |
| G1 — baseline reproduction | Official split is unchanged; source-defined metrics are emitted; published baselines are reproduced or documented discrepancy-by-discrepancy. | Superiority or operational utility. |
| G2 — leakage and transfer | Train-only preprocessing, negative controls, confidence intervals and available temporal/channel/mission holdouts are reported; exposure is explicit. | Untested cross-mission generalization. |
| G3 — material improvement | After G1 freezes baselines, preregister a target such as ≥10% relative reduction in false-alert burden at matched event recall. Require a bootstrap interval supporting positive improvement on at least two independent source families, where semantics/sample sizes support the comparison. If they do not, revise the gate before inspecting held-out labels. | Flight safety, false-alarm rates in operations, or superiority on other missions. |
| G4 — bounded diagnosis | Where trusted root-cause labels exist, report fault-isolation top-k performance, calibration and abstention; synthetic fault campaigns remain separately reported. | Diagnosis where no labels exist or actuation authority. |
| G5 — hardware transfer | Only after a processor and deadline are frozen: repeat within the measured resource envelope, then proceed to SIL/HIL with an independent test oracle. | On-orbit or flight qualification. |

A queued CI run is not a pass. An unrun data source, missing required metric, unreviewed rights question, or a failed negative control is not a pass.

## Integration boundaries

- **Symthaea:** score telemetry, surface bounded hypotheses, track uncertainty and generate reproducible run evidence. It must not turn anomalous correlation into a verified root cause.
- **Symtropy / symthaea-orbital:** supply a future deterministic synthetic-fault and closed-loop scenario suite. Simulator assumptions, version and fault parameters must be explicit, and synthetic success stays a separate evidence class.
- **Mycelix:** can record signed provenance, approvals and cross-party evidence references. A signed record cannot make an inaccurate sensor reading true or grant operational command authority.
- **Sol-Atlas:** can display validated telemetry/mission context after coordinate, timestamp, provenance and missing-data semantics are defined. Visualization is not an evaluation oracle.
- **Independent evaluator / safety boundary:** the benchmark scorer and deterministic command interlock must remain independent of the model under test. No benchmark result in this program directly commands a flight vehicle.

## Execution order

1. Freeze source identity, license terms, checksums, split and metric semantics.
2. Audit the existing Rust SMAP/MSL example's preprocessing/split scope; preserve historical outputs and document limitations rather than silently rewriting them.
3. Add a Rust-first adapter for one admitted public source and the source-defined metric set.
4. Add an independent second source, group-based evaluation and the negative-control suite.
5. Compare HDC/CfC/FEP-related variants only against frozen simple and published baselines.
6. After offline evaluation passes, connect simulator fault injection as a separate SIL campaign; consider HIL only when a real target and independent oracle exist.

## Research references

- ESA Anomaly Dataset, official repository: https://github.com/esa/anomaly-dataset
- ESA-ADB data record: https://zenodo.org/records/12528696
- ESA-ADB paper: https://arxiv.org/abs/2406.17826
- OPSSAT-AD data descriptor, *Scientific Data* (2025): https://www.nature.com/articles/s41597-025-05035-3
- OPSSAT-AD correction (18 August 2025): https://www.nature.com/articles/s41597-025-05751-w
- OPSSAT-AD data record: https://zenodo.org/records/15108715
- OPSSAT-AD reference code: https://github.com/kplabs-pl/OPS-SAT-AD
- JPL F Prime, flight-software framework and test infrastructure: https://github.com/nasa/fprime

## Claim ceiling

A passing run establishes only reproducible performance on the exact named source, split, evaluator, configuration and compute profile. It does not establish flightworthiness, safe autonomous control, universal cross-mission transfer, root-cause diagnosis without reliable labels, regulatory/mission approval, or superiority to SpaceX or another aerospace organization.