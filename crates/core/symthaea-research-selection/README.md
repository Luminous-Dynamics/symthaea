# symthaea-research-selection

Content-addressed provenance for **how one frozen candidate is selected before final Evaluation**.

A good train/calibration/evaluation split and leakage-safe fitting are not sufficient if model or hyperparameter choice can inspect Evaluation scores. This crate makes the selection contest explicit and fail-closed.

## Core rule

```text
Training
   ↓
fit candidate artifacts
   ↓
Calibration
   ↓
select one frozen candidate
   ↓
SELECTION RECEIPT FROZEN
   ↓
Evaluation
   ↓
final scientific result
```

Evaluation never participates in candidate choice.

## What is bound

`ResearchSelectionManifest` binds:

- selection id;
- exact `ResearchSplitManifest` digest;
- one declared scalar selection metric;
- minimize/maximize direction;
- allowed selection roles;
- deterministic tie-break policy;
- every candidate's fit-manifest digest;
- every candidate's fitted output-artifact digest;
- every selection observation and source sample digest;
- sample role for every observation;
- deterministic per-candidate aggregate;
- selected candidate id;
- selection timestamp;
- content-addressed manifest digest.

Persisted selections should also call `verify_against(...)` with the authoritative split and fit manifests before use.

## Role policies

### `CalibrationOnly`

Only the explicit Calibration partition may influence selection.

This is the preferred policy for a conventional train/calibration/evaluation contest.

### `TrainingAndCalibration`

Training and Calibration may influence selection when a preregistered design explicitly requires it. Evaluation is still forbidden.

The broader policy is not permission to erase the distinction between fitting and selection; the exact influence observations remain visible.

## Same-sample requirement

Every candidate must be evaluated on the **same selection sample ids**.

This prevents a candidate from looking artificially strong because it was only scored on an easier subset. Missing/abstaining model outputs should eventually be represented by an explicitly preregistered selective-prediction metric rather than silently dropping cases from one candidate.

## Tie breaking

v1 uses an intentionally boring rule:

```text
primary selection metric exactly tied
        ↓
lexicographically smallest candidate id
```

The rule is frozen into the manifest. A human does not get to choose the preferred tied model after inspecting final Evaluation performance.

## Fit lineage

Each candidate is constructed from a `FitArtifactManifest` that must already verify against the same frozen split.

The selection record therefore connects:

```text
ResearchSplitManifest
        ↓
FitArtifactManifest A
FitArtifactManifest B
...
        ↓
Calibration observations
        ↓
ResearchSelectionManifest
        ↓
selected frozen artifact
```

`verify_against(...)` rechecks both the split and fitted-artifact lineage.

## Sentinel / Planetary Perception example

A future Wetland Watch contest might prepare:

```text
candidate-a: persistence/statistical baseline
candidate-b: conventional EO model
candidate-c: HDC/Symthaea model
```

Candidate representations and weights are fitted using Training according to `symthaea-research-fit`.

Calibration scenes may then be used for a preregistered metric such as Brier score. The selection manifest freezes which candidate won **before** any final geographic/time holdout is revealed.

Only the selected candidate then enters the final Evaluation campaign.

This prevents a common hidden source of optimistic results: trying many models against the test set and reporting only the best one.

## What this does not prove

The crate does not prove that:

- the selection metric is scientifically appropriate;
- Calibration is large or representative enough;
- candidate implementations are trustworthy;
- selection observations were generated correctly;
- a chosen candidate will generalize;
- final Evaluation data was inaccessible through unrelated ambient channels;
- one scalar selection criterion captures all relevant societal/scientific tradeoffs.

The last point is intentional: plural real-world consequences belong in higher-level outcome analysis. This crate only makes a narrow engineering selection step auditable.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-selection --all-targets
cargo test -p symthaea-research-selection
cargo clippy -p symthaea-research-selection --all-targets -- -D warnings
```
