# Causal-Integration Benchmark Scorecard

**Status:** Proposed experimental protocol; no performance result is claimed. This scorecard is a companion to `ENTROPY_CAUSAL_CUT_BENCHMARK.md`. It separates three questions that must not be collapsed into one leaderboard: (1) does a quantity correctly describe its mathematical target, (2) does it recover causal structure under interventions, and (3) is it computationally useful at scale?


## Candidate new metric to investigate

The exact toy evaluator is now `examples/task_conditioned_finite_state_eval.rs`: it exhausts finite-state inputs and fair-coin edge-cut outcomes. Six fixtures cover useful XOR, irrelevant connectivity, harmful coupling (sign controls only), an AND task under two reversed priors, and a role-matched XOR synergy negative control. For XOR, both task-essential edges cause 0.5 accuracy loss but TCCIP is zero, while H(target)=1 bit, each individual input MI is zero, and joint input MI is 1 bit. Thus TCCIP alone misses higher-order synergy. The AND pair reverses TCCIP from +0.3 to −0.3 with fixed rules/task/partition/edges by reversing the input-state prior. The schema-v3 `validation/fixtures/tccip_toy_expectations.tsv` is executable source for fixtures and expected outcomes; Rust and independent standard-library Python parse the same file. Python has twelve authored test definitions and Rust fifteen; neither committed suite has been run here. The new second-order task-utility difference \(\Delta^2J=J(\varnothing)-J(i)-J(j)+J(i,j)\) is calculated for all pairs and checked against `validation/fixtures/tccip_pairwise_utility_interactions.tsv` (16 pair cases). It reports non-additivity of task-utility loss—not information synergy, not PEID, and not IIT Φ. The Python suite has sixteen authored test definitions and the Rust suite nineteen; neither committed suite has been run here.

The companion proposal [Task-Conditioned Counterfactual Integration Profile v0](TASK_CONDITIONED_COUNTERFACTUAL_INTEGRATION_PROFILE_V0.md) investigates whether cross-component causal interactions improve a *declared task* under explicit counterfactual cuts, matched sham controls, and held-out environments. It is a profile (task competence, cut-response curve, worst-case utility, redundancy/recovery, and shift stability), not a single consciousness score. Existing effective-information, causal-integration, PID, SURD, and IIT work means novelty is not assumed; the candidate must outperform simple ablation and graph/information baselines before retention.

## Pairwise-intervention response surface

The evaluator reports \(\Delta^2J_{ij}=J(\varnothing)-J(i)-J(j)+J(i,j)\) across all intervention pairs. Positive values mean single-cut losses overlap/saturate; negative values mean compound cuts amplify damage beyond additivity. This is an engineering interaction contrast, not a synergy/integration score. Pairwise expectations are fixed in `validation/fixtures/tccip_pairwise_utility_interactions.tsv`; the Rust and Python test suites compare all 16 pairs.

## Paradigm shift: stop asking one scalar to answer every question

A single `phi`, entropy, similarity, or connectivity score cannot stand in for uncertainty, predictive dependence, causal influence, integration under a named formalism, calibration, and practical usefulness. Treat these as typed outputs with distinct estimands and failure modes.

Each result should carry:
- quantity name and semantic version;
- model digest and state-distribution/prior digest separately;
- observational or interventional regime and intervention identifier;
- estimator, formalism/version, configuration, and uncertainty;
- source revision, command, environment lock, runtime, peak memory, and machine description;
- qualification state: `PASS`, `FAIL`, or `NOT QUALIFIED`. Missing evidence is never a numeric zero.

## Benchmark tracks

| Track | Question | Required baselines | Primary metrics |
|---|---|---|---|
| T0 — mathematical invariants | Does the implementation compute its stated quantity? | Analytic toy systems; independent enumerator | Absolute error, row-normalization error, entropy bounds, permutation invariance |
| T1 — predictive dependence | Does it measure temporal/statistical dependence? | Shannon entropy, mutual information, conditional MI, transfer entropy | Bits, estimator bias, finite-sample error, calibration |
| T2 — causal intervention | Does a score change under interventions that alter causal influence, rather than merely correlation? | Known SCM/Boolean ground truth; observational-only baselines | Intervention-effect error, false-positive rate under confounding, sensitivity to edge deletion/addition |
| T3 — higher-order structure | Does it distinguish XOR-like synergy from pairwise-only dependence? | Pairwise MI, conditional MI, partial-information decomposition where applicable | Synergy detection, false discovery, robustness to noise |
| T4 — formal IIT agreement | Does an implementation reproduce a specified IIT formalism? | Stable pinned PyPhi; canonical published examples | Exact fixture agreement, absolute/relative error, rank correlation, tied-result handling |
| T5 — distribution shift | Does the output remain interpretable when the state prior or noise changes? | Fixed dynamics under multiple declared priors/noise levels | Calibration drift, rank reversals, sensitivity curves, uncertainty coverage |
| T6 — scaling and usefulness | What can be computed within a practical budget? | Exact small-system oracle; sparse/approximate variants | Runtime, peak memory, largest qualified system, error-vs-budget Pareto frontier |
| T7 — reproducibility and adversarial controls | Can a result be independently reproduced, and do negative controls fail? | Independent implementation and mutation tests | Byte-stable fixtures, digest match, zero silent fallbacks, zero false qualification |

These tracks are deliberately not collapsed into a weighted composite score. Publish a scorecard vector and Pareto frontier; a weighted score may be added only for a stated use case with pre-registered weights and sensitivity analysis.

## Minimum qualification gates

### Gate A — correctness of the reference kernel
- Every transition-probability row sums to 1 within (10^{-12}); every entry is finite and in ([0,1]).
- Entropy values remain within theoretical bounds to (10^{-12}).
- Deterministic and maximally noisy fixtures match analytic expectations within (10^{-12}).
- Node relabeling preserves permutation-invariant results within (10^{-12}).
- At least two independent implementations agree on every TPM entry and reported metric for the small-system fixture set.
- Source SHA, model digest, prior digest, and command are captured in a run receipt.

### Gate B — formalism-specific agreement
- Use a stable, exact dependency pin and explicitly name the formalism; never rely on a changing default.
- Reproduce canonical published examples at their stated precision before evaluating novel fixtures.
- Compare only identical transition models, state, prior/background conditions, and compatible measures.
- Report MAE, RMSE, maximum absolute error, relative error with a documented zero-value policy, Spearman/Pearson correlations when mathematically defined, and all excluded/failed cases.
- Correlation is supplementary, not sufficient: constant outputs, narrow value ranges, or a shared bug can make correlation misleading.
- No fallback score, omitted failure, or mismatched digest may contribute to a qualified result.

### Gate C — intervention validity
- Use synthetic structural causal models with known ground truth, including confounding, mediation, common drivers, feedback, and synergistic mechanisms.
- Separate observational prediction from interventional effect estimation.
- Pre-register intervention targets and expected direction/range before scoring.
- Include null interventions and edge-shuffle controls; a method that reports strong causal influence on randomized or disconnected controls fails this gate.
- Evaluate out-of-distribution priors and noise rather than selecting only favorable regimes.

### Gate D — useful approximation
- Compare exact and approximate methods on the same held-out model instances.
- Plot error against wall time and peak memory, not only throughput.
- Report Pareto dominance only when an approach is no worse on the predeclared error and resource dimensions and strictly better on at least one.
- A proposed stretch target is at least 10× lower runtime at matched error on a declared workload; it is a research target, not a current result or universal SOTA claim.
- If accuracy degrades, publish the degradation curve and abstention threshold rather than hiding it behind a single speedup.

## Initial test matrix

Keep the starter suite small enough to exhaustively enumerate and expand only after the reference kernel is qualified:

1. Independent fair-noise nodes — null temporal information.
2. Deterministic copy/ring dynamics — high predictability under a uniform prior.
3. Same dynamics, changed state prior — tests whether the ensemble is silently assumed.
4. Common driver — dependence without direct pairwise causal interpretation.
5. XOR and AND — higher-order/nonlinear response controls.
6. Feed-forward chain versus feedback ring — direction and recurrence controls.
7. Noise sweep from deterministic to maximally noisy — monotonicity and edge cases where analytically justified.
8. Edge deletion/addition and node permutation — intervention and representation invariance.
9. Confounded observational data with randomized interventions — separates association from causal effect.
10. Null and shuffled controls — measures false positives and qualification fail-closed behavior.

Each fixture must define its expected property, not just an expected scalar. Do not assume that every metric is monotonic under every noise or topology change; encode only mathematically justified invariants.

## Industry and research comparison

- **PyPhi** is the formal reference implementation for the *specified IIT version*, not a universal oracle for causal inference, consciousness, or practical system quality. PyPhi 2.0 supports IIT 3.0, IIT 4.0 (2023), and IIT 4.0 (2026); values from different formalisms are not interchangeable.
- **Information-theoretic baselines** (entropy, MI, conditional MI, transfer entropy) answer uncertainty/dependence questions, not automatically intervention-causal or IIT questions.
- **Causal-inference baselines** should use known synthetic SCM ground truth and randomized interventions where possible. Observational-only success on a correlated dataset does not establish causal identification.
- **Graph/spectral baselines** measure properties such as connectivity or mixing and must retain those names; they should not be relabeled as formal integrated information.
- **Approximate Symthaea estimators** earn a place only after they show measured error, calibration, compute cost, and abstention behavior against qualified targets.

The defensible path to outperforming existing systems is therefore not to claim a larger `phi` or a higher correlation on a handpicked set. It is to deliver a reproducible, adversarially tested, multi-objective benchmark where every method is evaluated on the same declared target, data, and budget—and where the system knows when it does not know.

## Evidence status

This document defines a protocol only. The Rust tests, Python tests, stable PyPhi cross-check, and maintained shared-model Rust adapter must be executed and their receipts attached before any result is called validated. CI queued, skipped, unavailable, or not run is not a pass.
