# CMAQ-008 — Real Symthaea CfC world-model qualification

CMAQ-008 is the first diagnostic in this research track that calls the actual
\`HierarchicalCfCWorldModel\` implementation from \`src/dynamics/crate_world_model.rs\`.
It is intentionally a small, reproducible bridge—not a claim of state-of-the-art
performance.

## Why this benchmark exists

Prediction error is necessary but insufficient. A model used for planning must
rank counterfactual actions correctly, improve realized outcomes over a direct
policy, and retain that advantage under distribution shift. A decision-making
evaluation therefore records both model diagnostics and downstream action
outcomes.

## Current probes

### v0.1 — next-state prediction

Binary: \`cargo run --bin cmaq-008-world-model-smoke\`

- Fits five explicit model seeds on a small action-conditioned source law.
- Reports next-state MSE/MAE under source holdout and a held-out action-effect reversal.
- Includes persistence and zero-state baselines.
- Establishes that the actual model API can be trained, replayed and scored by an
  independent deterministic evaluator.

### v0.2 — counterfactual action ranking (three-regime panel)

Binary: \`cargo run --bin cmaq-008-action-ranking\`

- Ten explicit model seeds; 192 logged training transitions; 12 epochs.
- Three candidate action transforms per state: zero, source-cancelling, and
  source-reversing.
- Uses the actual \`predict\` output to score candidate next states against a
  fixed zero-target task reward.
- Direct baseline: a no-model reactive policy that chooses one action transform
  using only logged training outcomes. It makes zero model forward calls at test time.
- Evaluator oracle: selects the best of the three actions using true next-state
  outcomes. This oracle never participates in model selection.
- Measures reward lift versus direct policy, model regret versus oracle, direct
  regret, fraction of states harmed/improved, oracle-action agreement, pairwise
  action-ranking accuracy, reward prediction MAE, next-state MSE and policy override
  rate.
- Runs source holdout, a held-out action-effect reversal, and a held-out interaction-sign reversal with paired model seeds. The latter preserves the main action coefficient while changing the state-action interaction from +0.06 to -0.12.

The world-model policy makes three forward calls per decision in this panel. The
direct baseline makes zero. This is a transparent query-count proxy, not a
measured latency, energy or dollar-cost comparison.

### v0.3 — identity-residual ablation

Binary: \`cargo run --bin cmaq-008-residual-ablation\`

The residual path is compared with the absolute-next-state objective under the
same seeds, training data and transition regimes, with a low-cost linear ARX
comparator and paired per-seed differences. The residual formulation tests the
identity prior \`next_state = state + delta\`; it is not assumed to be superior.
The focused workflow validates its report and confidence-interval identities.

## Reproducibility

\`HierarchicalCfCWorldModel::new_with_seed(config, seed)\` makes initialized
weights and CfC time constants reproducible. The legacy \`new(config)\` remains
available and samples one seed internally.

This guarantees deterministic initialization and permits training replay under
the same CPU/toolchain path. It does **not** promise bitwise-identical behavior
across different hardware, BLAS implementations, compiler versions or future
parallel/GPU kernels.

## Evidence boundaries

All current CMAQ-008 environments are public deterministic synthetic systems.
The evaluator and candidate still run in one binary; this is not a sealed or
process-isolated qualification benchmark. The action-ranking probe has only
three action candidates, a simple one-step quadratic reward, a small model
configuration and one deliberately constructed shift family. Its approximate
95% intervals are across ten paired model seeds, not across a representative
population of environments.

No external robotics/industrial suite, hardware latency/energy profile, safety
certificate, real-world causal law, SOTA ranking, or superiority over a VLA,
model-free policy, MPC or commercial world model is claimed.

## Graduation targets

A serious comparison should eventually include:

1. **Environment breadth:** at least 10 independently implemented dynamics
   families, with held-out combinations and mechanisms—not merely new seeds of
   one equation.
2. **Action breadth:** continuous and combinatorial controls, delayed effects,
   state constraints and irreversible actions.
3. **Decision utility:** closed-loop return, oracle regret, policy ranking,
   optimization lift, failure rate and CVaR95.
4. **Counterfactual fidelity:** intervention error, action-pair ranking,
   treatment-effect sign and calibration under support shift.
5. **Reliability:** risk–coverage curves, false-safe rates, abstention quality,
   fresh-vs-stale model comparisons, and post-selection exploitability.
6. **Efficiency:** quality versus matched train data, model forward calls,
   inference compute, wall-clock latency, peak memory and energy.
7. **Statistical strength:** at least 30 paired model seeds for exploratory
   stability, with family-level sampling as the inferential unit; publish every
   failure lane and confidence interval.
8. **External validity:** one independently maintained simulator or robotics
   benchmark and a model-free/MPC baseline implemented by an independent team.

The relevant target is not "lowest next-state MSE". It is positive, statistically
credible **net decision value at matched compute**, with no unacceptable safety
regression and robust performance on held-out dynamics.
