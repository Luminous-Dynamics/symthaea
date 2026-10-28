# SCI-PROG-001 — Progressive scientific-law discovery, intermediate knowledge, and negative transfer

**Status:** Benchmark proposal only. No new algorithm, result, or superiority claim is implemented here.

Issue: [#8671](https://github.com/Luminous-Dynamics/symthaea/issues/8671)

## Research question

When a physical explanation is genuinely hierarchical, does progressive discovery of reusable intermediate laws improve validated recovery, extrapolation, interpretability, and data efficiency over one-shot symbolic regression? Equally importantly, can a method recognize when a staged decomposition is not justified?

## Current literature and claim boundary

Xia & Zhang, *Data-driven progressive discovery of physical laws*, was accepted by Physical Review Research on 6 October 2026. It proposes Chain of Symbolic Regression (CoSR), a chain of semantically meaningful knowledge units, and applies the approach to Kepler/gravity, Rayleigh–Bénard convection, pipe flow, laser-metal interaction, and aerodynamic coefficient scaling: https://doi.org/10.1103/f2kp-pp4q.

Pang et al., *Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs*, ICML 2026, separates candidate generation from search control using adaptive operators, structural diagnosis, and trajectory-level memory. Its paper reports performance over representative LLM symbolic-regression baselines on LLM-SRBench using 40% of the standard sample budget: https://proceedings.mlr.press/v306/pang26f.html.

These establish relevant approaches to compare and replicate, not expected performance in this protocol. The narrow proposed contribution is a paired, falsification-first qualification of intermediate-unit validity, transfer, and fallback behavior.

## Existing Symthaea work to reuse

- [#8487 — scientific discovery recovery protocol](https://github.com/Luminous-Dynamics/symthaea/pull/8487) for task registry, comparison standards, and claim boundaries.
- [#8536/#8584] for truth-absent and held-out-operator recovery.
- [#8607/#8641] for observation-process noise fixtures and deterministic seeds.
- [#8628] for identifiability and costed diagnostic experiments.
- [#8667/#8668] for unit/representation invariance.
- [#6879] for ScientificModel / PhysicsIR identity, units, dimensions, and assumptions.

Do not create a second scientific IR, experiment-selection API, or observation-noise generator.

## Factorial task design

Cross the following axes:

1. **Hierarchy status:** genuinely compositional law; partial/weak hierarchy; non-hierarchical law; misleading decomposition that overfits.
2. **Intermediate evidence:** directly observed intermediate quantity; derived quantity with explicit uncertainty; latent knowledge unit; truth-absent or operator-held-out intermediate.
3. **Observation regime:** clean; registered noise/missingness fixtures; adaptive design; scale/unit transformation.
4. **Transfer demand:** new parameters in the same family; new operating regime; unseen combinations of known units; deeper composition than training; new structural family.
5. **Candidate support:** true primitives available; one primitive/operator omitted; incorrect assumed hierarchy.
6. **Budget:** equal observations and separately equal wall-clock/candidate-evaluation budget.

Pair methods on the same world, split, initial information and admissible observation budget. Evaluator-only hierarchy labels, hidden decomposition and target equations must not be policy inputs.

## Baseline ladder

- One-shot PySR/SymbolicRegression.jl.
- Compatible bounded exhaustive symbolic regression with MDL/complexity control.
- Faithful CoSR reproduction when code and data are available.
- Deliberate Evolution on its compatible LLM-SRBench track, keeping that original scorecard separate.
- Frozen human-authored intermediate units as an explicitly oracle-assisted upper bound, not as an autonomous baseline.
- Random valid decomposition and deliberately wrong-decomposition negative controls.
- Ablations removing intermediate memory, structural diagnosis, adaptive search control, or cross-task reuse.

Do not call a method a reproduction based on its name alone. Pin source revision, grammar/operators, model access, external tools, search settings, sample and compute budgets, and all attempted runs.

## Metrics

Report separately for hierarchy status × noise regime × transfer type × candidate-support condition:

- Final canonical law equivalence-class recovery.
- Independent in-domain, marginal-shift, joint-extrapolation, near-boundary and deeper-composition prediction.
- Correctness of each intermediate unit, not only final-equation fit.
- **Intermediate reuse precision:** fraction of reused units independently validated in held-out contexts.
- **Transfer gain:** change in independently validated recovery per accumulated observation and compute budget versus a fresh one-shot search.
- **Negative transfer:** degradation caused by reusing false, over-specialized, or out-of-applicability intermediate units.
- False confident-singleton, unresolved, invalid, and not-evaluable rates.
- Measurements, candidate evaluations, simplification operations, wall-clock, CPU/GPU use and memory.
- Variability across paired seeds and semantically equivalent formulations.

The primary endpoint should be pre-registered as independently validated final-law recovery per total budget. Intermediate correctness, negative transfer, calibration/false-confidence and unresolved behavior remain independent guardrails, not tradeable components of one reward.

## Falsification and acceptance

- Exact tiny-grammar known-answer tests validate the composition and scoring oracle before large experiments.
- On true hierarchy tasks, claim benefit only if progressive discovery improves the pre-registered endpoint versus the strongest compatible baseline under paired equal budgets.
- On non-hierarchical and wrong-hierarchy controls, a progressive method must not receive credit for inventing a plausible chain. It should fall back to one-shot search or preserve uncertainty when decomposition is unsupported.
- Intermediate units enter reusable memory only after independent holdout verification, with provenance, applicability conditions, and status separating hypothesis from validated knowledge.
- Reused intermediates are rechecked in the destination context; historical utility is not a universal validity certificate.
- Report every cell and full denominator. Seeds within one world are nested repeats, not independent worlds.
- Confirmatory superiority needs at least six structural families, a frozen world-level power/precision plan, paired uncertainty intervals, matched baselines and complete cost accounting.
- No performance claim before exact-head executable checks and independent evaluation.

## Non-goals

This is a benchmark proposal—not a CoSR reproduction, a new symbolic-search implementation, an established reusable-knowledge library, or a claim that progressive search beats one-shot discovery.
