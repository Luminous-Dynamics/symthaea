# Evaluation Open Gate v0.1

This document defines the integration gate that must exist before final Evaluation predictor inputs are released to a selected model.

It is intentionally narrower than model selection and narrower than custody. It composes already-frozen evidence from both.

## Required inputs

An evaluation-open decision must bind:

- the authoritative `ResearchSplitManifest` digest;
- the frozen `ResearchSelectionManifest` digest;
- the selected candidate id;
- the selected fitted-artifact digest;
- the authoritative `ResearchCustodyManifest` digest;
- the exact evaluation predictor-input asset ids to be opened;
- an experiment/run identifier;
- the opening time;
- a content-addressed gate receipt.

## Preconditions

Before opening any final Evaluation predictor input:

1. the split manifest verifies;
2. the selection manifest verifies against the same split and exact fitted candidate manifests;
3. the selected candidate is frozen and its output artifact digest matches the artifact that will execute;
4. the custody manifest verifies against the same split;
5. every requested asset is an Evaluation `PredictorInput`;
6. custody policy permits `ModelProcess` `Read`/`Transform` no earlier than `EvaluationInputsOpen`;
7. no `GroundTruthLabel` or `VerificationOutcome` is included in the opening set;
8. the gate receipt is created before the first model access receipt;
9. later access receipts bind the gate receipt digest as phase evidence.

## Ordering

```text
fit candidates
    ↓
calibration-only selection
    ↓
selection manifest frozen
    ↓
verify split + fits + selection + custody
    ↓
EvaluationOpenReceipt
    ↓
open predictor inputs only
    ↓
model output / forecast
    ↓
commit output
    ↓
reveal verification outcome
    ↓
score
```

## Fail-closed cases

Do not issue an `EvaluationOpenReceipt` when:

- selection is missing, mutable, invalid, or references another split;
- the executing artifact differs from the selected artifact;
- a requested asset is not Evaluation data;
- a requested asset is a hidden label/outcome rather than predictor input;
- custody references another split;
- model access would precede `EvaluationInputsOpen`;
- required artifact digests or identifiers are empty;
- any prerequisite manifest fails authoritative verification.

## Evidence ordering

The receipt should bind the selection and custody digests directly, rather than merely copying a human-readable phase name. The first allowed `ModelProcess` access receipt for a final predictor input should then bind the `EvaluationOpenReceipt` digest as its `phase_evidence_digest`.

This creates an auditable chain:

```text
selection manifest
        +
custody manifest
        ↓
evaluation-open receipt
        ↓
predictor-input access receipt(s)
        ↓
model/forecast commitment
        ↓
outcome-reveal access receipt
        ↓
score/result manifest
```

## Security boundary

This gate is scientific provenance, not operating-system isolation. A later Xenia adapter should bind the same receipt to real principals and capabilities so only the selected executable can access the opened predictor inputs and hidden outcomes remain inaccessible until the output-commitment phase.

## Sentinel / Wetland Watch

The locked Sentinel witness in issue #194 should eventually require this ordering before real final-evaluation Sentinel-1/Sentinel-2 inputs are released.

For a classification-style witness, final predictor imagery may open after selection while labels remain sealed. For a forecasting witness, the future Sentinel observation itself may remain a `VerificationOutcome` and therefore must not open until the forecast/output is committed.

## Promotion boundary

Passing this gate proves only that the declared evaluation-opening evidence chain is internally consistent. It does not establish model quality, real isolation, label validity, sensor correctness, geographic independence, causal explanation, satellite compression benefit, or subsurface inference capability.
