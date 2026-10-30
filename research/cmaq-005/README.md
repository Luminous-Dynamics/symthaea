# CMAQ-005: Held-Out Causal-Mechanism Transfer

**Status:** standalone Rust evaluator prototype; no Symthaea results or qualification claims.

## Research question

Does a system learn an intervention-relevant causal model that transfers to a genuinely new mechanism, or does it merely fit held-out cases drawn from mechanisms it already saw?

CMAQ-001 measures memory-to-action scoring; CMAQ-002 tests closed-loop model exploitation; CMAQ-003 tests selective trust under uncertainty; CMAQ-004 measures net value of active interventions. CMAQ-005 isolates the missing transfer claim: counterfactual effects and safety predictions under independently held-out causal-law families.

## Inputs and leakage boundary

Evaluator truth rows contain an opaque case ID, evaluator-only mechanism family, split (development or qualification), outcome under do(X=0) and do(X=1), and actual safety under both interventions.

Candidate predictions contain only case ID, predicted outcomes under the two interventions, and predicted safety probabilities. The candidate schema explicitly rejects mechanism labels and has no field for hidden mechanism family or true outcomes. Qualification cases must be generated/evaluated separately from development cases; this scorer fails closed if predictions include a case from the wrong split, a duplicate ID, an unknown ID, or omit any selected case. It also rejects any mechanism family that appears in both development and qualification, so a case-level split cannot accidentally masquerade as a held-out-mechanism split.

The outcomes under both interventions exist only because the current lane is a synthetic simulator. For a real-world intervention, never fabricate unchosen outcomes: evaluate observed interventions with logged propensities/identification assumptions or a validated simulator and label the evidence source.

The committed fixture is a public evaluator smoke test only. It is not a sealed qualification set: anyone can inspect the fixture answer key, so it must never be used to report blind benchmark performance. Real qualification requires separately held, evaluator-controlled mechanisms and keys, with candidate predictions frozen before scoring.

## Metrics

- Outcome MAE across both intervention conditions.
- Causal-effect MAE, comparing predicted Y(do=1) - predicted Y(do=0) with the evaluator effect.
- Effect-sign accuracy, excluding truly zero effects so a no-effect predictor cannot win by guessing ties.
- Safety Brier score across both intervention conditions.
- False-safe prediction count at the predeclared 0.5 safety-probability threshold, separately visible and non-compensatory (safety_gate_pass).
- Overall and evaluator-only per-mechanism-family rows to expose aggregation hiding transfer failures.

## Protocol

1. Fit/tune candidate models on development mechanisms only.
2. Freeze model, configuration, adapter and candidate predictions for qualification cases before evaluator truth is opened.
3. Report overall plus per-family metrics, clustered bootstrap intervals, mechanism count, case count, all exclusions, and resource budgets.
4. Compare at minimum: no-effect predictor, global action prior, factorized regression, nearest-case retrieval, HDC-only, recurrent-only, explicit memory, hybrid world model, and the integrated Symthaea system.
5. Include a label-shuffle control and a no-effect counterfactual control. If the hybrid retains causal-effect/sign performance when treatment labels are shuffled, the claimed causal transfer is unsupported.
6. Treat safe-action selection as a separate hard gate; do not allow reward accuracy to compensate for false-safe actions.

## Research basis

The 2026 ICML work on task-sufficient world models connects active exploration to structured, task-relevant representations and tests generalization to unseen skills and object-skill compositions: https://proceedings.mlr.press/v306/feng26aa.html. WorldTest evaluates whether world models answer global environment-level and counterfactual queries beyond observed interaction trajectories: https://proceedings.mlr.press/v306/warrier26a.html. The September 2026 Active Causal Discovery Benchmark evaluates causal recovery under budgeted hard interventions: https://arxiv.org/abs/2609.31675. The World Model Benchmarks survey classifies the field across target, protocol, metrics and data source and cautions against treating heterogeneous scores as equivalent: https://world-model-benchmarks.github.io/.

CMAQ-005 complements these benchmarks; it does not replace them. Run external benchmarks under their published protocols and report separate results.

## Limitations

The first implementation scores two binary intervention levels and deterministic synthetic outcomes. It does not infer a structural causal model, estimate uncertainty intervals, provide a formal causal-identification guarantee, or train a candidate. Family labels remain evaluator-side, but a real qualification package still needs independent family generation, split custody, cryptographic manifests and external review. This authoring environment has no local Rust compiler; no local test pass is claimed. Fixtures are known-answer evaluator tests, not Symthaea performance.
