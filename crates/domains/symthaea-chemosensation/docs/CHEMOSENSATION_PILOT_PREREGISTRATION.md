# Symthaea Chemosensation Pilot Preregistration

**Program ID:** `symthaea-chemosensation-pilot-2026-01`  
**Program version:** 0.1.0  
**Decision date:** 2026-08-26  
**Claim boundary:** artificial chemical sensing, representation, temporal context, and cross-modal binding  
**Current evidence state:** software/simulation only; no physical smell/taste performance claim

## 1. Decision being made

This program asks whether Symthaea's chemosensation architecture is worth advancing from deterministic software fixtures to physical olfactory and gustatory hardware.

The program separates four claims that must never be collapsed into one score:

1. **transduction correctness** — raw measurements, calibration, health, units, and acquisition context are preserved correctly;
2. **representation utility** — the chemical representation preserves useful similarity while suppressing nuisance variation;
3. **temporal utility** — rise/recovery, carryover, and novelty information improve discrimination or uncertainty estimates;
4. **multimodal utility** — smell/taste/flavor integration improves useful inference without erasing disagreement or provenance.

Success on one claim does not establish the others.

## 2. Evidence vocabulary

Every result is labeled with exactly one evidence source:

- `SimulatedFixture` — deterministic software simulator or analytical fixture;
- `RecordedReplay` — replay of a previously recorded sensor trace;
- `BenchPhysicalObservation` — new measurements from a development/bench setup;
- `HeldOutPhysicalObservation` — new measurements from a frozen holdout session/set.

These are categorical evidence classes, not interchangeable ranks. A simulator result cannot populate a decision protocol preregistered for held-out physical evidence.

Every result is also labeled with one evaluation partition:

- `Calibration` — may be used to choose nuisance models, thresholds, and hyperparameters permitted by this protocol;
- `Development` — may be used for implementation debugging and exploratory analysis;
- `Holdout` — outcome-bearing confirmatory evaluation only after the relevant protocol version is frozen.

Development and calibration results cannot satisfy a holdout decision rule.

## 3. Asymmetric decision semantics

Every confirmatory metric has a **confirmation threshold** and may also have a separately preregistered **practical-failure threshold**.

For a metric where larger is better:

- `value >= confirmation_threshold` -> `ConfirmationPass`;
- `value <= practical_failure_threshold` -> `PracticalFailure`;
- values between them -> `Indeterminate`.

For a metric where smaller is better, the inequalities are reversed.

The aggregate decision is:

- `Confirmed` only when every required gate passes;
- `NotConfirmed` only when at least one gate establishes a preregistered practical failure;
- `Inconclusive` otherwise.

**Failure to confirm is not itself a negative result.** This prevents a noisy, small, or underpowered pilot from being relabeled as evidence of absence.

## 4. Frozen anti-leakage rules

Before any holdout run:

- all holdout sample identities and outcome labels remain unavailable to threshold/model selection;
- preprocessing/calibration choices allowed to adapt must be explicitly listed in the protocol version;
- model/encoder hyperparameters are frozen after calibration;
- no holdout sample may be moved to development because it is difficult, anomalous, or unfavorable;
- failed reads, timeouts, saturation, contamination, and unavailable channels remain in the denominator or are labeled with a preregistered exclusion reason;
- a rerun after inspecting an outcome is a protocol deviation unless the rerun condition was frozen beforehand.

## 5. Acquisition integrity

Every outcome-bearing physical observation must carry `SamplingContext` and be admissible to a `ChemicalTrace`.

At minimum the trace binds:

- sampling protocol ID/version;
- run ID;
- modality;
- replicate;
- phase;
- monotonic timestamps;
- monotonic protocol step indices.

The trace is structural rather than prescriptive. Olfaction and gustation may use different phase recipes.

Raw channel values are preserved. Calibration, HDC encoding, learned labels, and semantic hypotheses are derived views and do not replace source evidence.

## 6. Pilot families

### OD-001 — odor identity under concentration shift

**Question:** Can a learned odor representation preserve identity when concentration changes, rather than primarily encoding intensity?

**Primary split:** train/calibrate on a subset of concentrations; evaluate identity on held-out concentrations from the same odor families.

**Primary metrics:**

- identity classification/retrieval performance;
- concentration leakage from the learned/encoded representation;
- confidence calibration on held-out concentrations.

**Required baselines:**

- raw calibrated sensor vector;
- standardized raw vector;
- PCA + simple linear classifier/retrieval;
- small dense embedding/MLP baseline;
- current HDC chemical fingerprint;
- at least one locality-preserving HDC alternative (for example level/thermometer encoding) before representation claims are frozen.

**Decision thresholds:** selected using Calibration only, then frozen in a versioned `ChemicalDecisionProtocol` before Holdout begins. No threshold may be chosen from holdout outcomes.

### OD-002 — humidity nuisance robustness

**Question:** Does the representation retain odor identity across humidity shifts without hiding genuine sensor uncertainty?

**Primary metrics:**

- identity degradation versus matched-humidity condition;
- representation shift attributable to humidity;
- confidence calibration under humidity shift.

Humidity compensation fitted on calibration sessions is permitted. Refitting after viewing holdout outcomes is not.

### OD-003 — temporal response utility

**Question:** Do rise/recovery dynamics add useful information beyond static snapshots?

Compare:

- static endpoint features;
- simple hand-engineered rise/recovery descriptors;
- the preregistered temporal/HDC representation;
- a small recurrent/temporal baseline when enough real sequences exist.

A temporal method must improve a frozen primary endpoint or uncertainty calibration; merely increasing model complexity is not evidence of utility.

### OD-004 — open-set novelty

**Question:** Can unfamiliar odors be recognized as unfamiliar without forcing a known-class label?

Known and unknown identities are split by identity, not by individual measurement. Novelty thresholds are selected on calibration unknowns that are disjoint from holdout unknown identities.

Report false-known and false-novel rates separately; do not hide them behind one AUROC value.

### GT-001 — gustatory concentration shift

**Question:** Can the electronic-tongue representation preserve sample/mixture identity across concentration changes?

Use the same anti-leakage rules as OD-001. Human taste categories such as sweet/bitter/umami are derived labels, not primary transducer coordinates.

### GT-002 — mixture discrimination

**Question:** Can mixtures with overlapping pH/conductivity be distinguished from distributed electrochemical response patterns?

Include pH-only and conductivity-only baselines so improvements cannot be credited to the full array when a single scalar explains them.

### GT-003 — temperature robustness

**Question:** Does temperature-aware processing reduce nuisance shift without erasing real Nernst-dependent response changes?

Analytical Nernst fixtures remain internal-correctness checks. Physical robustness claims require physical data.

### GT-004 — rinse and carryover

**Question:** Does the acquisition protocol return sufficiently close to baseline after exposure, and can residual carryover be detected rather than mistaken for a new sample?

Primary metrics should include:

- residual response after rinse/recovery;
- time to return inside a preregistered baseline band;
- next-sample error conditioned on prior sample identity.

A high next-sample accuracy does not excuse unacceptable carryover if the carryover gate crosses its practical-failure boundary.

### FL-001 — smell + taste complementarity

**Question:** Does conservative smell+taste fusion improve held-out sample inference relative to the stronger single modality?

Primary comparator is the **better** of olfaction-only and gustation-only on the same holdout cases, not their average.

Flavor is confirmed useful only if the frozen fusion endpoint improves while modality-specific error/disagreement reporting remains intact.

### FL-002 — cross-modal contradiction preservation

**Question:** When visual/contextual evidence and chemistry disagree, does the system preserve the disagreement rather than average it away?

Construct controlled contradiction cases only before outcomes are inspected. Success requires both component evidence records to remain recoverable and a contradiction signal to change in the expected direction.

## 7. Representation benchmark rules

No HDC representation receives privileged status because it is native to Symthaea.

At minimum, representation studies compare against appropriate simple baselines and report:

- task metric(s);
- calibration/uncertainty metric(s);
- memory footprint;
- inference/update cost;
- robustness across session/day/device where data permits.

Hyperparameters for every method use the same calibration/development budget.

A representation is not called superior from an in-sample or same-session result alone.

## 8. Statistical protocol

For physical studies, the exact sample count and resampling method are frozen after a calibration-only variance pilot and before holdout evaluation.

General rules:

- use grouped splits by physical sample/odor identity/session as appropriate; do not randomly split adjacent frames from the same exposure across train and test;
- treat one acquisition run/trace as the primary paired unit unless the protocol states otherwise;
- report uncertainty intervals for primary paired differences;
- retain all preregistered runs, including sensor failures, with explicit status;
- correct families of confirmatory comparisons when multiple primary gates test the same claim family;
- label analyses not frozen before holdout as exploratory.

## 9. Calibration-stage freedom

Before a holdout protocol version is frozen, Calibration data may be used to choose:

- baseline windows;
- permitted temperature/humidity compensation form;
- sensor-channel normalization;
- HDC level count / locality parameters;
- novelty threshold;
- classifier regularization and other baseline hyperparameters;
- practical confirmation/failure thresholds informed by observed variance and intended use.

The chosen values, code revision, seeds, and rationale are then frozen. Holdout outcomes may not alter them.

## 10. Deviations

After a holdout protocol version is frozen, every change that can affect an outcome requires a deviation record containing:

- protocol/version;
- timestamp;
- affected runs/fields;
- reason discovered;
- whether outcome data had been inspected;
- expected direction of bias if known;
- disposition: restart, new version, or exploratory-only analysis.

No adverse or anomalous sample may be silently removed.

## 11. Claim boundaries

The following claims are explicitly out of scope until supported by separately frozen evidence:

- human-equivalent smell or taste;
- arbitrary chemical identification;
- certified toxic-gas detection;
- food-safety certification;
- biological qualia or subjective smell/taste experience;
- superiority of HDC over conventional methods;
- transfer across unseen sensor hardware without a held-out device study.

## 12. Execution order

1. Complete software-integrity PRs and deterministic simulator checks.
2. Freeze acquisition hardware and typed sampling protocols.
3. Collect Calibration traces only.
4. Use Calibration results to freeze preprocessing, baselines, thresholds, sample counts, and protocol versions.
5. Seal holdout identities/sessions.
6. Run outcome-bearing holdout experiments once under the frozen version.
7. Emit machine-readable `ChemicalDecisionReceipt` records plus a human-readable result memo.
8. Treat any post-hoc analysis as exploratory unless a new protocol version is preregistered.

This document defines the program-level rules. Each outcome-bearing physical experiment receives a narrower versioned protocol containing its exact sample matrix, sensor hardware revision, preprocessing configuration, metric gates, and practical-failure thresholds before holdout acquisition begins.
