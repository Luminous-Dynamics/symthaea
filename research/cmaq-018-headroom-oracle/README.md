# CMAQ-018 repeated-continuation oracle — finite reference v0.1

This dependency-free Rust crate is the first executable finite-oracle slice for the denoised selectable-headroom estimand in Symthaea #10625. It is not the complete repeated-reactive experiment.

## Distinct estimands

For a frozen history, baseline candidate b, candidate set A, selection continuation set S and disjoint evaluation continuation set E, the module emits:

1. **One-shot oracle.** Select the candidate with the best outcome on the first selection continuation and report both that same-continuation gain and the selected candidate's gain on held-out continuations.
2. **Repeated in-sample oracle.** Select by the mean over S and score on S. This is an in-sample evaluator diagnostic, not a deployable policy and not a formal finite-sample upper confidence bound.
3. **Held-out repeated oracle.** Select by the mean over S and score the selected candidate on E. This is a held-out evaluator-oracle diagnostic, not the learned D3 selector.
4. **Actual D3 selector.** Explicitly not measured by this reference. It must be a separately frozen selector using permitted predicted/readout information, then scored on a sealed continuation set unavailable to selection, calibration, tuning, candidate generation and stopping.

The selected candidate is always scored on the same held-out continuation batch as the baseline. Selection and evaluation continuation IDs must be disjoint. Higher integer utilities are better, and terminal/failure outcomes must remain in the denominator under a preregistered utility rule.

## Pairing and variance decomposition

The held-out batch also reports candidate-versus-baseline variance under two sampling designs:

- paired continuation differences, when the evaluation batch declares SharedInvariant or SensitivityOnly;
- hypothetical independent-mean variance, computed from the two marginal sample variances.

The module reports sample covariance, paired-difference sample variance, both variances of the mean, the paired/independent variance ratio, and `1 - ratio` as the CRN variance reduction fraction. Negative reduction values are retained: they mean that the declared pairing increased variance. Uncoupled batches report only the independent marginal estimate and explicitly mark the paired effect as not identified. Sensitivity-only pairing is never promoted to the primary lane. At least two evaluation continuations are required.

This remains a finite descriptive calculation. The coupling disposition and justification digest are caller-supplied; no causal invariance proof, external authentication, or sampling-distribution guarantee is inferred.

A candidate-by-continuation decomposition is computed only when the selection batch declares **SharedInvariant** and supplies a nonempty justification reference. In that case the column index denotes a common exogenous continuation across candidate branches.

**Uncoupled** and **SensitivityOnly** batches do not produce the primary decomposition. Equal seeds, equal column indexes or matching labels do not prove causal invariance. The justification digest is metadata supplied by the caller; this finite module does not authenticate it or establish that the coupling is scientifically valid.

The balanced two-way decomposition reports candidate, continuation and candidate × continuation sums of squares. The interaction fraction is:

    SS_interaction / (SS_candidate + SS_interaction)

It is undefined when the denominator is zero. This is a descriptive decomposition of the supplied finite matrix, not an inferential variance component estimate.

## Support-gated reference selector

The crate now includes a finite reference selector that is separate from the evaluator-oracle diagnostics above. It uses repeated selection-continuation means and a familywise two-sided Hoeffding gate over the supported baseline and supported alternatives. It selects an alternative only when that candidate's lower bound strictly exceeds the baseline's upper bound.

- Candidate support is explicit: Supported, Unsupported, Unknown, or Incomplete. Only Supported alternatives can enter the selection comparison.
- If the baseline's support is not qualified, the selector abstains. It does not silently fall back to a nominal baseline.
- If no alternative clears the simultaneous separation gate, it retains the supported baseline. This means "no improvement established", not "baseline proven optimal".
- An incomplete or budget-limited candidate universe is labelled `EnumeratedBankOnly`; the result is not represented as a global action optimum.
- The bounded utility range, familywise delta, selection continuation count and rollout counts are returned with the decision.
- The confidence formula assumes independent continuation draws across continuation IDs for each candidate. The caller must supply an IID sampling contract and justification reference. The crate retains that claim but does not authenticate it or validate the scientific sampling process.
- The selected candidate is scored only after selection on the disjoint evaluation continuation batch. These held-out values are descriptive in this reference and do not feed back into the choice.

This is a **support-gated reference rule**, not the learned D3 selector and not a vehicle policy. The familywise bounds are valid only under their declared bounded/IID assumptions; they are not an empirical guarantee for temporally dependent road scenarios.

## Exact finite fixtures

The tests include:
- A noise-only selection trap: one-shot best-of-three appears to gain 10 units, while repeated and held-out oracle gain is zero.
- A genuine candidate effect that survives held-out scoring.
- A selection case where the one-shot winner differs from the repeated winner; the one-shot held-out share is scored on its own selected candidate.
- Candidate × continuation rank reversal where all candidate main effects are zero and all candidate-distinguishing variation is interaction; paired CRN fixtures where positive covariance reduces variance and negative covariance increases it.
- Uncoupled and sensitivity-only pairing, both of which refuse primary variance reduction claims; single-continuation batches do not report sample variance.
- Selection/evaluation overlap, noncanonical candidate ordering, malformed outcome dimensions and missing coupling justification.
- Support-gated reference selection: strong supported improvement, noise-only non-selection, unsupported/unknown candidate exclusion, baseline-support abstention, incomplete-bank scope labelling, utility-bound checks, and rejection of unqualified IID continuation sampling.

## Deliberate limits

This reference does not implement a simulator, policy continuation, learned D3 selector, D0/D1/D2/D3 campaign, confidence sequence, cluster bootstrap, statistical power analysis, common-random-number causal proof, external receipt authentication, or physical-driving experiment. Its support-gated rule is a transparent finite reference selector, not an empirically qualified planner. It does not infer empirical performance from finite fixture results.

The full #10625 protocol remains responsible for repeated closed-loop continuations, separate calibration and sealed evaluation sets, matched-compute comparisons, support/abstention gates, correspondence hypotheses, causal stochastic receipts, family-clustered inference, and compute/latency/energy accounting.
