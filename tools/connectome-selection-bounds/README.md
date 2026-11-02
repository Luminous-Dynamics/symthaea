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


## Detector operating-range inversion

`latent_prevalence_bounds` provides separate sensitivity analysis for a binary source or target detector. Given an observed positive fraction q and externally qualified sensitivity/specificity ranges Se in [Se_L, Se_U] and Sp in [Sp_L, Sp_U], it returns the set of latent prevalence values p in [0,1] compatible with at least one operating point:

    q = Se*p + (1-Sp)*(1-p)

The implementation solves the two equivalent linear feasibility inequalities. It does not divide by a possibly near-zero Youden index or silently assume a perfect detector. Completely uninformative detector ranges can return the full [0,1] interval; incompatible ranges return an explicit error.

Sensitivity and specificity ranges must come from independent calibration evidence representative of the evaluation population, and should already include their own sampling/cluster uncertainty. This function treats them as fixed admissible ranges; it does not estimate them from validation counts or create confidence intervals. Apply it separately to source and target detectors. Do not multiply marginal prevalence intervals into a joint probability: source/target dependence is not identified by marginal rates. The result is conditional on the detector-evaluable population and does not correct trace missingness or transport across animals/sessions.

Analytic tests cover q=0.6 with Se in [0.8,0.9] and Sp in [0.9,0.95] (latent prevalence [0.625, 11/15]), a perfect detector, a completely uninformative detector returning [0,1], incompatible ranges, invalid probabilities and monotonicity under wider calibration uncertainty.


## Machine-readable contracts

- `schemas/selection-bounds-input.schema.json` defines the finite-cohort selection/bounds input.
- `schemas/selection-bounds-output.schema.json` defines the report and hard claim-boundary flags.
- `schemas/selection-bounds-fixture.schema.json` validates the input/expected wrapper for `fixtures/selection-bounds-known-answer.json`.
- `schemas/detector-prevalence-input.schema.json` defines the observed rate and sensitivity/specificity ranges for detector inversion.
- `schemas/detector-prevalence-fixture.schema.json` validates the wrapper for `fixtures/detector-prevalence-known-answer.json`.

The known-answer fixtures separate input from expected output; the expected values are never accepted as part of the strict evaluator input object.


## Joint source/target response prevalence

`joint_response_prevalence_bounds(source, target)` uses the sharp Fréchet-Hoeffding bounds when only source and target marginal prevalence intervals are available:

    P(source AND target) in
    [max(0, lower_source + lower_target - 1), min(upper_source, upper_target)]

This is an **identified interval over all couplings**, not a point estimate. It makes no independence assumption. Multiplying marginal response rates is unjustified unless a separately validated conditional-independence or dependence model supports it.

Known-answer examples:
- Point marginals 0.8 and 0.6 -> joint interval [0.4, 0.6], while the independence product 0.48 is merely one possible value.
- Marginals [0.7,0.8] and [0.6,0.9] -> [0.3,0.8].
- Marginals [0.2,0.4] and [0.3,0.5] -> [0,0.4].
- One completely unknown marginal [0,1] and another [0.4,0.7] -> [0,0.7].

The function accepts intervals, so it composes directly with separate source and target outputs from `latent_prevalence_bounds`. It does not recover the dependence structure. A joint-response model needs paired calibration observations, joint latent-state measurements, or explicitly defended structural assumptions.


The joint-response layer also has strict contracts:
- `schemas/joint-response-input.schema.json` defines source/target marginal-prevalence intervals.
- `schemas/joint-response-fixture.schema.json` validates the input/expected wrapper for `fixtures/joint-response-known-answer.json`.
- `schemas/joint-response-counts-input.schema.json` defines full-cohort source/target response-status counts and separate detector operating ranges.
- `schemas/joint-response-counts-fixture.schema.json` validates integrated known-answer fixtures in `fixtures/joint-response-counts-known-answer.json` and `fixtures/joint-response-counts-missing-known-answer.json`.


## Building detector ranges from validation counts

`wilson_interval(successes, trials, z)` computes a two-sided Wilson score interval. `detector_operating_bounds_from_counts` applies it to:
- sensitivity = TP / (TP + FN)
- specificity = TN / (TN + FP)

The reference-positive and reference-negative denominators must both be non-zero; invalid counts and invalid z-scores fail closed. The result can be passed directly to `latent_prevalence_bounds`.

**Statistical qualification:** the Wilson interval assumes independent Bernoulli validation units. It is suitable only when the validation sampling unit and design support that assumption. Repeated trials nested within animal/session are not independent simply because they are separate rows. For clustered calibration, use animal/session-level resampling or another design-appropriate method and pass the resulting sensitivity/specificity ranges into the prevalence inversion. The crate does not automatically infer clustering or turn a trial-level Wilson interval into a cluster-robust interval.

A synthetic validation fixture uses TP=90, FN=10, TN=95, FP=5 at z=1.9599639845. It produces sensitivity [0.825634, 0.944771] and specificity [0.888250, 0.978456]. These counts are illustrative, not empirical detector performance.


Detector-count calibration contracts:
- `schemas/detector-validation-counts-input.schema.json` defines the TP/FN/TN/FP counts and Wilson z-score.
- `schemas/detector-validation-counts-fixture.schema.json` validates the wrapper for `fixtures/detector-validation-counts-known-answer.json`.


## Source-to-target conditional response

`target_given_source_prevalence_bounds(source, target)` derives sharp bounds for the conditional target-response probability P(target | source) from the same marginal prevalence intervals:

    [max(0, s_L + t_L - 1) / s_L, min(1, t_U / s_L)]

This result is returned only when the source lower bound is strictly positive. If the source prevalence interval includes zero, the function returns `None` because the conditioning denominator is not guaranteed to exist. It is an associational probability bound, not a causal propagation effect or proof of a direct synapse.

For point marginals source=0.8 and target=0.6, the conditional target-response probability is bounded by [0.5,0.75]. With intervals source=[0.7,0.8] and target=[0.6,0.9], it is [3/7,1]. These are sharp over all compatible source/target couplings.

## Full-cohort missing-detector integration

The point-valued latent_prevalence_bounds(q, operating) remains available when q is observed for every planned unit. When source or target detector states are unknown, do not calculate q only over complete cases and silently transport that rate to the planned cohort. Use the added composition:

1. observed_positive_fraction_bounds(counts, planned_units) forms the full-cohort detector-positive interval:
   [detected / planned, (detected + unknown) / planned].
   Here unknown is the sum of unobserved, classifier-invalid and registration-failed units. All five status counts must sum exactly to the planned denominator; an empty or inconsistent cohort fails closed.
2. latent_prevalence_bounds_from_observed_fraction_interval(q_bounds, operating) inverts the full observed-rate interval against the channel's externally calibrated sensitivity/specificity ranges.
3. evaluate_joint_response_bounds(...) performs those steps independently for source and target, then applies the Fréchet bounds to their latent marginal intervals. It also returns target-given-source bounds when source prevalence is bounded strictly above zero.

The integrated report retains the planned denominator, both raw response-status count tables, both observed-positive intervals, both latent marginal intervals, the joint interval and the conditional interval (or None). Unknown states are not converted to negative detections, and complete-case denominators are not substituted for the planned cohort.

Synthetic known answers:
- With 10 planned units, source 8/10 and target 6/10, and perfect detector ranges, the marginal prevalences are 0.8 and 0.6; the sharp joint interval is [0.4,0.6] and target-given-source is [0.5,0.75].
- With source counts 6 detected / 2 negative / 2 unknown and target counts 3 detected / 5 negative / 2 unknown, the observed intervals are [0.6,0.8] and [0.3,0.5]; under perfect detectors the joint interval is [0,0.5].
- With all detector statuses unknown, each marginal remains [0,1], the joint remains [0,1], and target-given-source is not estimable.

These are finite-cohort identification bounds conditional on the supplied detector operating ranges. They are not sampling confidence intervals, do not solve MNAR trace availability beyond the declared range, and do not identify a causal effect or direct monosynaptic connection. Detector calibration must represent the relevant animal/session populations; trial-level Wilson intervals are not a substitute for cluster-aware calibration.


## Structural hypothesis response sets

The crate also propagates response bounds over alternative, provenance-pinned structural interpretations. This prevents an unresolved split/merge decision from being silently collapsed into one hard neuron identity before functional analysis.

The API is:

```rust
evaluate_structural_hypothesis_response_set(
    &[StructuralHypothesisInput]
) -> Result<StructuralResponseIdentifiedSet, EvalError>
```

Each input contains a stable hypothesis ID, source and target entity IDs, a SHA-256 evidence-manifest digest, an upstream admissibility flag, and already-qualified source/target prevalence intervals. The caller must freeze and justify the admissibility rule; this function does not inspect raw segmentation, validate graph topology, or determine which structural alternative is true.

For every admissible hypothesis, the library computes its Fréchet joint-response interval and target-given-source interval. It then returns a **canonical union of intervals**, not a weighted mean and not a convex hull. This distinction matters: two alternative connectomes can imply disjoint feasible response ranges. Filling the gap between them would claim values supported by no supplied hypothesis.

If any admissible structural hypothesis allows the source prevalence to be zero, the global target-given-source set is returned as not estimable. The implementation does not discard that alternative just because a conditional denominator is unavailable.

The report preserves the hypothesis IDs, entity mappings, evidence digests, marginal intervals, per-hypothesis outputs, excluded hypothesis IDs, and the explicit flag that no probability weights were applied. It validates all alternatives (including excluded records) and rejects duplicate IDs, malformed digests, invalid intervals and an empty admissible set.

Synthetic known answer:
- Hypothesis A has source/target marginals 0.1/0.1 and implies a joint interval [0,0.1].
- Hypothesis B has source/target marginals 0.9/0.9 and implies [0.8,0.9].
- Their union remains [[0,0.1],[0.8,0.9]]. The interval [0.1,0.8] is not filled in.
- A third recorded but excluded hypothesis remains in the provenance report and is not included in the identified set.

This is an identified set conditional on the supplied hypotheses and marginal intervals, not a posterior probability, a confidence interval, a complete enumeration of all possible connectomes, a causal effect, or proof of a direct synapse. The supplied hypotheses must be sufficiently complete for any broader scientific claim.
