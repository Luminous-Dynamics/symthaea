# CMAQ-018 repeated-continuation oracle — finite reference v0.1

This module is a small, dependency-free evaluator oracle for the denoised selectable-headroom estimand in Symthaea #10625. It is a first executable slice, not the complete repeated-reactive experiment.

## Distinct estimands

For a frozen history, baseline candidate b, candidate set A, selection continuation set S and disjoint evaluation continuation set E, the module emits:

1. **One-shot oracle.** Select the candidate with the best outcome on the first selection continuation and report both that same-continuation gain and the selected candidate's gain on held-out continuations.
2. **Repeated in-sample oracle.** Select by the mean over S and score on S. This is an in-sample evaluator diagnostic, not a deployable policy and not a formal finite-sample upper confidence bound.
3. **Held-out repeated oracle.** Select by the mean over S and score the selected candidate on E. This is a held-out evaluator-oracle diagnostic, not the learned D3 selector.
4. **Actual D3 selector.** Explicitly not measured by this reference. It must be a separately frozen selector using permitted predicted/readout information, then scored on a sealed continuation set unavailable to selection, calibration, tuning, candidate generation and stopping.

The selected candidate is always scored on the same held-out continuation batch as the baseline. Selection and evaluation continuation IDs must be disjoint. Higher integer utilities are better, and terminal/failure outcomes must remain in the denominator under a preregistered utility rule.

## Pairing and variance decomposition

A candidate-by-continuation decomposition is computed only when the selection batch declares **SharedInvariant** and supplies a nonempty justification reference. In that case the column index denotes a common exogenous continuation across candidate branches.

**Uncoupled** and **SensitivityOnly** batches do not produce the primary decomposition. Equal seeds, equal column indexes or matching labels do not prove causal invariance. The justification digest is metadata supplied by the caller; this finite module does not authenticate it or establish that the coupling is scientifically valid.

The balanced two-way decomposition reports candidate, continuation and candidate × continuation sums of squares. The interaction fraction is:

    SS_interaction / (SS_candidate + SS_interaction)

It is undefined when the denominator is zero. This is a descriptive decomposition of the supplied finite matrix, not an inferential variance component estimate.

## Exact finite fixtures

The tests include:
- A noise-only selection trap: one-shot best-of-three appears to gain 10 units, while repeated and held-out oracle gain is zero.
- A genuine candidate effect that survives held-out scoring.
- Candidate × continuation rank reversal where all candidate main effects are zero and all candidate-distinguishing variation is interaction.
- Uncoupled and sensitivity-only pairing, both of which refuse the primary decomposition.
- Selection/evaluation overlap, noncanonical candidate ordering, malformed outcome dimensions and missing coupling justification.

## Deliberate limits

This reference does not implement a simulator, policy continuation, learned selector, D0/D1/D2/D3 comparison, confidence sequence, cluster bootstrap, statistical power analysis, common-random-number causal proof, external receipt authentication, or physical-driving experiment. It does not infer empirical performance from finite fixture results.

The full #10625 protocol remains responsible for repeated closed-loop continuations, separate calibration and sealed evaluation sets, matched-compute comparisons, support/abstention gates, correspondence hypotheses, causal stochastic receipts, family-clustered inference, and compute/latency/energy accounting.
