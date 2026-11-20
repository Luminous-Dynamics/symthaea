# SC-CAUSAL-IDENTIFICATION-0.1

**Status:** synthetic causal-identification auditor and adversarial fixtures; no physical causal effect is established.  
**Research issue:** [#12969](https://github.com/Luminous-Dynamics/symthaea/issues/12969)  
**Planner:** [#12968](https://github.com/Luminous-Dynamics/symthaea/pull/12968)  
**Risk/resource layer:** [#12965](https://github.com/Luminous-Dynamics/symthaea/pull/12965)

## Objective

Qualify when transition/effect estimates are actually supported by intervention assignment and specimen-/batch-linked evidence. The causal-transition planner consumes a transition model; this auditor checks whether estimators and comparisons are supported by a declared assignment mechanism, treatment receipt, overlap, controls and complete outcome accounting.

The v0.1 estimand is the finite-population intention-to-treat average treatment effect (ATE):

\`ATE = N^{-1} sum_i (Y_i(1)-Y_i(0))\`

Potential outcomes are used only as held-out synthetic benchmark truth. Estimators read observed assignment, propensity, dose receipt, lineage and observed outcomes only.

## What is implemented

The Rust binary \`sc_causal_identification_audit\`:
- validates sample identity, batch and material-family lineage across every cohort;
- requires one assignment record per specimen;
- preserves treatment assignment separately from delivered dose;
- reports pooled difference-in-means and batch-blocked differences as descriptive/assumption-dependent comparisons;
- reports Horvitz–Thompson and normalized Hájek inverse-propensity estimates;
- reports a simple family-stratified AIPW comparator, with its small-sample/no-cross-fitting limitation explicit;
- fails closed for HT, Hájek and AIPW when outcomes are missing or assignment propensities leave the preregistered [0.05,0.95] overlap gate;
- reports propensity range, overlap violations and inverse-probability effective sample size by arm;
- reports baseline pairing/coherence proxy balance, batch/family assignment counts and delivered-dose summaries;
- exactly enumerates blocked treatment allocations under the declared fixed treated count per batch and reports a two-sided sharp-null randomization p-value;
- retains the count of missing outcomes and abstains from exact randomization inference if any outcome is missing;
- binds receipts to schema and fixture SHA-256 digests.

## Causal interpretation boundaries

- Assignment is the primary randomized estimand. Delivered dose is a separate treatment-receipt field; partial compliance does not remove a sample from its assigned ITT arm.
- The v0.1 implementation does not estimate a dose-response or per-protocol causal effect.
- Observational inverse-propensity weighting is not automatically causal. Interpretation requires a correct assignment propensity, consistency, positivity and conditional exchangeability given the adjustment set; unmeasured confounding is not resolved by weighting.
- The simple family-stratified AIPW model uses the same fixture to fit and evaluate its outcome regressions; it is not cross-fitted and is only a transparent comparator on this tiny fixture.
- The pooled and blocked complete-case differences remain descriptive if outcome missingness is present. They are not silently promoted to causal effects.
- Missing outcomes trigger abstention for v0.1 IPW/AIPW. No missing-at-random imputation is silently assumed.
- Exact randomization inference tests the sharp null under the declared blocked assignment mechanism. It is not a confidence interval for heterogeneous effects, and the fixture's 12 units do not support broad population claims.

## Frozen adversarial suite

Registry: \`data/research/superconductivity_causal_identification_suite_v0_1.json\`.

1. **Blocked randomization versus historical targeting** — 12 synthetic specimens across three batches and two material families. The ground-truth finite-population ATE is 3.5; family ATEs are 2 and 5. Historical targeting creates a large pooled contrast; known-propensity weighting and blocked comparisons are separate methods with different assumptions. One randomized unit is assigned treatment but receives only 60% of the declared dose; it stays in the treatment arm for ITT.
2. **Missing outcomes** — one randomized and one historical outcome are marked unobserved. Specimens remain in cohort denominators; exact inference and v0.1 inverse-propensity estimators abstain.
3. **Positivity failure** — one historical assignment propensity is set to 0.99, outside the preregistered [0.05,0.95] overlap gate. HT, Hájek and AIPW abstain and the receipt reports the violation.

All potential outcomes, assignments, propensities, dose receipts and outcomes are synthetic. Values are designed to test software semantics, not to approximate real superconductivity experiments.

## Comparators and next benchmarks

The suite compares:
- pooled difference-in-means;
- batch-blocked difference-in-means;
- Horvitz–Thompson;
- Hájek normalized IPW;
- simple family-stratified AIPW;
- exact blocked randomization inference.

Follow-on benchmark work should add cross-fitted AIPW/TMLE, cluster randomization and interference, missing-not-at-random sensitivity bounds, sequential response-adaptive randomization, dose-response with non-compliance, family/source transportability, and multi-knob factorial interaction recovery. GO-CBED, Causal-EPIG and budgeted active RCT sampling are methodological comparators, not superconductivity baselines until evaluated on the same frozen SCMs.

## Claim ceiling

This is a synthetic causal-identification benchmark only. It does not identify a real intervention effect, establish a superconducting material or mechanism, authorize a specimen intervention, or establish laboratory safety. Historical evidence must have rights-cleared specimen/protocol/measurement lineage; prospective campaigns require qualified scientific collaborators, facility review and institutional safety approval.

## Assignment-design receipts

Every cohort now carries a typed `allocation_design` record:
- experimental unit (`specimen`);
- allocation algorithm (`blocked_complete_randomization_fixed_counts` or `historical_targeting_policy`);
- blocking variable (`batch_id`);
- randomization seed for the randomized cohort (null for historical targeting);
- an explicit specimen-to-assignment receipt.

The auditor requires the assignment receipt to match every cohort record exactly. It fails closed if an assigned treatment differs from the recorded allocation receipt, if the randomized cohort lacks its seed/design/block declaration, or if a historical cohort claims to have a randomization seed. The receipt emits a canonical SHA-256 of sorted specimen-assignment pairs, independent of HashMap iteration order. The full input fixture digest binds that assignment record to the rest of the cohort evidence.

This v0.1 receipt records the seed and declared algorithm but does not regenerate the assignment from the seed. Exact inference is conditional on the declared complete-randomization-within-batch, fixed-treated-count design and enumerates every assignment compatible with the observed block counts. A future extension should pin the PRNG/version and independently replay the randomization seed to the allocation receipt.
