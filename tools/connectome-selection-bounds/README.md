# Connectome Selection Bounds V1

A dependency-free Rust library for finite-cohort worst-case bounds when a declared bounded outcome is missing, with separate selection-rate and conditional-kernel summaries.

## Paradigm shift

Do not let a successful response detector define the population you claim to understand. The tool preserves three different objects:

1. **All-event outcome:** a scalar outcome defined for every planned stimulation unit. Missing values are bounded using only its declared range.
2. **Source and target detector states:** both are retained independently (detected, not detected, unobserved, classifier invalid, registration failed); the combined selection state is derived from the conjunctive rule and checked against those raw states.
3. **Conditional propagation/kernel:** a value defined only among units meeting the selection rule. Its denominator is the selected cohort. Missing kernels among selected units are bounded; non-selected units are outside this conditional estimand, not zero-valued kernels.

The output reports these quantities separately. It never collapses them into one score.

## Bounds

For N planned units, n observed outcomes, observed sum S, and declared outcome range [L,U], the finite-cohort mean is bounded by:

    [(S + (N-n)L)/N, (S + (N-n)U)/N]

These are sharp worst-case bounds for the finite-cohort mean when the only restriction on missing values is Y in [L,U]. They are not confidence intervals and do not include sampling uncertainty.

The descriptive treatment-minus-control arm-contrast interval is:

    [lower_treatment - upper_control, upper_treatment - lower_control]

It is not automatically an ATE or a causal effect. A separate design audit must establish randomization/assignment, consistency, interference and other identification conditions. This library does not authenticate randomization receipts or treatment receipt.

For a conditional kernel, the same bound is computed only over units with selection status Selected. It does not transport this selected-cohort estimand to non-selected units.

## Explicit selection states

- Selected: the frozen source/target response-selection rule was met.
- NotSelected: the rule was evaluated and not met. This is not a ground-truth neural zero.
- Unobserved: the trace or selection decision was unavailable.
- ClassifierInvalid: the detector did not produce a valid decision.
- RegistrationFailed: identity/alignment failed, so selection could not be evaluated.

The selection rate among evaluable units is Selected / (Selected + NotSelected). Unknown selection statuses are excluded from that denominator and reported separately. The selected fraction of all planned units is also reported, but it is not a response probability when some selection decisions are unknown.

## Input contract

Each planned unit has a globally unique ID, a selection status, an optional all-event outcome, and an optional conditional kernel.

- all_event_outcome may be present for any selection status because it represents a different outcome defined on the full planned cohort.
- conditional_kernel may be present only for Selected units.
- A missing all-event outcome is bounded over the complete cohort.
- A missing kernel among selected units is bounded over the selected cohort.
- A kernel on a non-selected/unevaluable unit is rejected.
- Values outside declared finite ranges and duplicate unit IDs fail closed.

The caller must define the all-event outcome independently of response selection. Do not insert a detector's negative output as zero unless that detector output itself is the declared outcome and its measurement-error limitations are explicit. This library does not correct sensitivity/specificity, response misclassification, or latent response-dependent trace missingness.

## Synthetic known answer

The fixture has six planned units per arm.

- Treatment: observed all-event values sum to 3.0 across four of six units; range [0,1] gives [0.500000, 0.833333].
- Control: observed all-event values sum to 1.8 across five of six units; range [0,1] gives [0.300000, 0.466667].
- Descriptive treatment-minus-control arm contrast: [0.033333, 0.533333].
- Treatment: four selected units and three observed kernels; kernel range [-1,1] gives selected-cohort bounds [0.25, 0.75].
- Control: two selected units and both kernels observed, giving selected-cohort mean -0.05.
- One treatment selection decision is unobserved; it is not recoded as not-selected.

All values are synthetic arithmetic checks, not biological findings.

## API and test

The library API is evaluate(&StudyInput) -> Result<StudyReport, EvalError>.

    cargo test --locked --manifest-path tools/connectome-selection-bounds/Cargo.toml

## Qualification limits and next work

This is an arithmetic kernel, not a full selection model. It does not provide detector sensitivity/specificity correction, MAR/MNAR point identification, sequential excursion-effect estimation, propensity/positivity analysis, cluster-robust confidence intervals, raw-trace reconstruction or prospective outcomes.

The next integration should connect these summaries to sequential DGP work in #9834 and the response/measurement schema in #9810. Calibrated detector-error intervals and sensitivity analyses should be separate layers rather than hidden assumptions inside an adjusted point estimate.


## Source/target status derivation

The input carries independent source-neuron and target-neuron response statuses. The combined selection status is:
- Selected only when both are Detected.
- NotSelected when either is known NotDetected (even if the other side is unknown); the unknown side remains visible in its own status field.
- Otherwise RegistrationFailed takes precedence over ClassifierInvalid, then Unobserved.

This deterministic summary is not a latent neural-response label. It states whether the declared conjunctive detector rule was met or could be evaluated. The report also counts every raw status separately for source and target.
