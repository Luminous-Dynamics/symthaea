# VIS-DVPC-059 — Falsifying entropy-based annotation stopping for OPE v0

**Status:** synthetic counterexample and finite-instance stopping-rule benchmark. This is not a clinical validation, public-data result, or simultaneous-confidence guarantee.

## Literature comparator

APRIL (Mandyam et al., *Annotations for Policy evaluation with Reliable Inference from LLMs*, PMLR 297, 2026) proposes an entropy-based metric to identify when further counterfactual annotations stop being useful. It uses the marginal action entropy:

\[
H(A)=-\\sum_{a\\in\\mathcal A}\\widehat p(a)\\log\\widehat p(a),
\\qquad H_{max}=\\log|\\mathcal A|.
\]

The paper reports that action entropy approaches its maximum as counterfactual annotations are added and that OPE gains plateau around the same region in its clinical examples. This is a useful descriptive heuristic and a directly relevant comparator: https://proceedings.mlr.press/v297/mandyam26a.html

VIS-DVPC-059 does not assume the heuristic is invalid in general. It asks whether global marginal entropy is a *sufficient stopping signal* for OPE when context probabilities, target-policy mass, logging propensities, source error and source dependence are heterogeneous.

## Falsification fixture

The registered fixture `global_entropy_saturated_rare_context` has:

- two contexts with probabilities 0.99 and 0.01;
- a common context with 100 factual fit observations for each of three actions;
- a rare context with only two fit observations for action 0 and none for actions 1 or 2;
- a target-policy portfolio whose aggressive policy puts 0.80 probability on action 2 in the rare context and balanced puts 0.90 there;
- a factual logger that still has full support in that rare context, but assigns action 2 propensity 0.001;
- three annotation sources and shared-error covariance inherited from the preceding benchmark.

At zero annotation spend, the aggregate action counts are `[102, 100, 100]`, so global action entropy is within the registered epsilon (0.001) of `log(3)`. The rare context is nevertheless almost uncovered for an action carrying large target-policy mass. This is an acquisition problem, not an identification bypass: the behavior propensity is positive, and annotations never replace the factual residual reward or propensity.

## Alternative observables

For the same data/allocation, the benchmark reports:

1. **Global entropy** `H(A)`, following APRIL's aggregate action-count heuristic.
2. **Context-weighted entropy** `E_X[H(A|X)]`, which preserves the rare-context deficit instead of pooling it away.
3. **Target-weighted evidence coverage**, the prior-weighted portfolio mass multiplied by `n_xa/(n_xa+1)` and summed over context-action cells. This is a simple descriptive coverage diagnostic, not a propensity or support certificate.
4. **Exact worst-profile worst-policy MSE**, from the coherent source-profile scenario set.
5. **Best next annotation**, selected by marginal reduction in exact robust worst-policy MSE per unit cost.

The code intentionally follows that value-optimal next annotation for two additional steps even after the global entropy stop would fire. This makes the false-stop observable instead of allowing the stopping rule to censor the evidence that would falsify it.

## Stop-rule comparison

- APRIL-style global entropy stop: `H(A) >= log(|A|) - 0.001`;
- value-based stop: stop only when no feasible one-step annotation has robust worst-policy MSE gain per cost greater than `1e-10` (v0 diagnostic threshold). The row reports both decisions separately.

Future iterations should compare marginal entropy gain, context entropy, target-weighted coverage thresholds, best-policy expected-simple-regret gain, and sequential batch acquisition. The current v0 comparison focuses on the falsification of aggregate entropy as a sufficient stop signal.

## Expected output

The Rust executable emits three `ENTROPY_STOP` rows for steps 0, 1 and 2. Each row records global entropy and maximum entropy, context-weighted entropy, target-weighted evidence coverage, entropy/value stop decisions, current and next exact robust risk, marginal gain and gain per cost, and the next context-action-source choice.

## Acceptance gates

- At step 0, global action entropy is within epsilon of its maximum and therefore triggers the APRIL-style entropy stop.
- Context-weighted entropy remains below that same saturation threshold.
- The exact robust-value oracle identifies the rare context's high-target-mass, low-propensity action as the next annotation opportunity.
- That next feasible annotation yields strictly positive exact robust worst-policy MSE reduction while aggregate entropy remains saturated.
- Deterministic stepwise output is schema-checked and source/output hashes are recorded.
- The common factual logger remains full-support; a zero propensity would remain NOT IDENTIFIED regardless of annotations.

## Research boundary

This one frozen synthetic counterexample can falsify a universal claim that global action entropy is sufficient for OPE stopping. It does not establish that entropy is never useful, nor that the chosen target-weighted coverage formula is universally optimal. Any production stop rule requires calibration across policy portfolios, context shift, correlation uncertainty, acquisition costs, and best-policy selection regret.