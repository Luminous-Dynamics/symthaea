# Calibrated Model-Adequacy Audits for Scientific Discovery v0.1

**Status:** exploratory Rust implementation and diagnostic pilot; no SOTA or performance claim.

This work is intentionally stacked on the scenario-robust EIG work in PR #8361. It does not reimplement robust acquisition. It adds the missing *decision and audit layer*: what evidence must be present before a discovery loop can say its candidate space is inadequate or accept a singleton?

## 1. Paradigm shift: separate discovery, adequacy, and validation

A discovery process has at least three logically different tasks:

1. **Discriminate:** which represented candidate best explains the observations under the current model?
2. **Audit adequacy:** do independent observations exhibit predictive misfit inconsistent with the calibrated clean reference?
3. **Validate a candidate:** after one candidate remains, does it survive a predeclared validation budget on fresh measurements?

High EIG addresses the first task only, under the assumed candidate and likelihood models. It does not establish the second or third. The implementation therefore uses typed state for the adequacy assessment, validation result, candidate count, and termination reason. Hidden ground-truth laws and discrepancy amplitudes remain evaluator-only.

## 2. Typed decision semantics

The Rust module `model_adequacy.rs` exposes a pure `next_discovery_action` function.

- Empty candidate set + further search budget: reopen candidate search; do not infer model misspecification from emptiness alone.
- Empty candidate set + exhausted search: report an unexplained empty set.
- Calibrated alarm from an independent audit: report model-space inadequacy.
- Alarm without calibration or audit independence: request better audit evidence.
- Singleton with no independent adequacy result: audit before acceptance.
- Singleton with clean calibrated audit but incomplete validation: validate provisionally.
- Contradictory validation: reopen search if budget remains.
- Singleton accepted only when independent adequacy assessment is clear and candidate validation passed.
- Invalid numerical/protocol state: fail closed.

The contract test enumerates 1,890 combinations of candidate count, evidence quality, validation state, and termination cause. These test semantics, not empirical accuracy of any adequacy detector.

## 3. Batch-level conformal adequacy calibration

`adequacy_calibration.rs` includes:

- `standardized_rms_residual`: one nonconformity score for a prediction/measurement batch;
- `ConformalAdequacyCalibrator`: a finite-sample rank p-value against clean calibration batches;
- explicit rejection of malformed scores, invalid `alpha`, insufficient p-value resolution, and non-independent audit inputs.

For standardized RMS score (s_*) from a fresh audit batch and calibration scores (s_1,ldots,s_n), the conservative p-value is:

```text
p = (1 + count_i[s_i >= s_*]) / (n + 1)
```

The detector raises a predictive-misfit alarm when `p <= alpha`; otherwise it reports only that this score did not raise an alarm. It does not prove the candidate is correct.

At `alpha = 0.05`, at least 19 independent calibration batches are needed for the minimum attainable p-value `1/(n+1)` to reach 0.05. This is a resolution constraint, not a power guarantee. The batch-level calibration result is valid only when clean calibration scores and the test score are exchangeable under the declared null, with the same predeclared scoring procedure. The Rust type cannot prove data independence or exchangeability: the caller must record the data split, score-function version, seed/source, and target distribution in the run receipt.

### Known limitations

- A scalar residual-magnitude score can miss structured residuals, autocorrelation, missing interactions, and regime changes.
- A non-alarm is not proof of adequacy. A misspecified law can have an ordinary residual score over one finite audit batch.
- Distribution shift can violate exchangeability and invalidate the finite-sample false-alarm claim.
- Choosing among multiple diagnostic tests after viewing their p-values inflates false alarms unless multiplicity or sequential testing is predeclared.
- Passing a boolean `independent_audit = true` is an assertion by the caller, not an automated provenance proof.

## 4. Integrated synthetic pilot

The diagnostic pilot now calibrates on 80 independently seeded clean batches and evaluates 50 clean plus 50 perturbed audit batches at a fixed discrepancy amplitude of 0.12. Every batch produces only a model-visible standardized residual score before the conformal calibrator sees it; the hidden amplitude is used by evaluator-side simulation and never passed to the decision function.

The pilot reports empirical clean false-alarm and perturbed-world detection rates. It asserts basic accounting/rate validity and a loose false-alarm regression guard, but it does not require detection to exceed false alarms and does not assert the policy wins. Its deterministic seed ranges are disjoint across calibration and audit batches.

These are simulation diagnostics, not confirmatory guarantees. Before using this detector for a scientific claim, the calibrator must be evaluated on held-out structural families, not just more seeds from the same generator.

## 5. Research direction: from batch alarms to anytime-valid regime monitoring

Batch conformal scoring is a useful first component, but the next high-value upgrade is sequential monitoring with a fixed null reference. Shaer et al. (ICML 2026) propose conditional conformal test martingales with a fixed reference set; the paper specifically addresses contamination of the reference pool after a change and targets anytime-valid type-I error control with bounded expected detection delay.

A sensible Symthaea sequence is:

1. Keep batch conformal as an auditable baseline.
2. Implement a separate sequential fixed-reference alarm with a predeclared stopping threshold and documented assumptions.
3. Compare both on clean streams and scripted regime changes, including post-change sample contamination.
4. Report false alarm probability, detection delay, missed-shift rate, calibration refresh cost, false-alarm recovery cost, and clean-world structural recovery after an alarm.
5. Only after those controls are validated, let an alarm influence experiment acquisition or candidate generation.

Do not let an adaptive policy continuously fold post-shift scores into the reference set and still claim the original null calibration remains valid.

## 6. Benchmark matrix and advancement goals

Compare the audit layer across:
- correctly specified candidate families and omitted structural terms;
- discrepancy amplitude `0, 0.03, 0.06, 0.12, 0.24` in a frozen truth-scaled unit;
- homoscedastic Gaussian, heteroscedastic, and heavy-tailed noise;
- clean held-out world families, shifted target-query distributions, and change-point streams;
- rare-region and extrapolation failures where a pooled residual score may look ordinary;
- invalid numerical predictions and malformed/incomplete audit data.

Use at least 60 held-out worlds across six structural families and at least 20 paired seeds per world for the confirmatory synthetic stage. Resample worlds first and seeds within worlds for uncertainty intervals. Keep policy-visible inputs and evaluator-only labels in separate artifacts.

Proposed targets (not results):
- clean-world false-alarm rate at or below 5% under the stated exchangeability assumptions, with an uncertainty interval;
- at least 80% detection sensitivity for predeclared, practically relevant discrepancies at matched budget;
- no more than a 5 percentage-point decline in clean-world structural recovery after introducing adequacy audits;
- at least 20% relative reduction in false-singleton outcomes compared with nominal EIG on held-out misspecified worlds;
- report median and 90th-percentile total measurement cost, including audit queries, model expansion attempts, and unsuccessful experiments.

Do not combine all these into one score. Detection sensitivity, false alarms, clean-world recovery, unresolved rate, candidate expansion cost, and time-to-valid-recovery must remain visible. A policy must not “win” by abstaining indefinitely; couple false-singleton reduction to clean recovery and total cost.

## 7. Required tests and receipts

Unit/contract tests currently cover:
- the rank p-value threshold and attainable resolution;
- conservative treatment of tied calibration scores;
- rejection of non-independent audit requests;
- invalid numeric inputs and overflow in residual calculations;
- acceptance only after a clear calibrated independent audit and passed validation;
- fail-closed behavior for invalid runs;
- the empty-set / candidate-search distinction;
- exhaustive decision-gate combinations.

A confirmatory run receipt should include protocol version, source commit, dirty-tree state, compiler/toolchain, world generator version, world ID, selector version, audit score function, calibration and audit batch IDs, target distribution, prior/calibration policy, random seeds, experiment order, actual costs, hidden truth held only in evaluator artifacts, terminal outcome, and raw receipt digest.

## 8. Industry and research comparators

Relevant references:

- Callahan & Catanach, AISTATS 2026, [On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html). Shows why information maximization can amplify inferential bias under misspecification and distinguishes inferential-error from model-error objectives.
- Tang, Sloman & Kaski, AISTATS 2026, [Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html). A distinct robust acquisition baseline that considers representativeness and error de-amplification; not reproduced by this detector.
- Shaer et al., ICML 2026, [Testing For Distribution Shifts with Conditional Conformal Test Martingales](https://proceedings.mlr.press/v306/shaer26a.html). A strong next sequential-shift baseline using a fixed null reference.
- Lee, UAI 2026, [Diagnosing Conformal Prediction Failures Under Distribution Shift](https://proceedings.mlr.press/v337/lee26g.html). Shows that the same type of shift can cause very different calibration failures for different predictors.
- [NIST Sequential Bayesian Experiment Design](https://www.nist.gov/programs-projects/sequential-bayesian-experiment-design). Practical reference for adaptive, cost-sensitive parameter estimation under assumed parametric models; adjacent, not a direct symbolic-law-discovery SOTA score.
- [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench). Interactive law discovery under changing complexity and noise.
- [SciLaws-Bench](https://github.com/yiyihum/SciLaws-Bench). External benchmark reporting numerical fit, scientific validity, and structural recovery separately.

External scores must be recorded with the date, task split/version, harness, compute/tool budget, evaluator, and license. They are contextual comparisons until the task and protocol genuinely match.

## Claim boundary

This PR adds a calibrated audit component and diagnostic pilot to the scenario-robust EIG line. It does not implement an anytime-valid sequential detector, automatically prove audit independence, or establish a SOTA improvement. The calibrator is not yet wired as a mandatory production gate in the complete active-discovery loop. Only exact-head passing Rust checks and confirmatory held-out-world evidence can justify stronger claims.
