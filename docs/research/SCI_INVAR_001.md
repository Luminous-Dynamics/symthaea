# SCI-INVAR-001 — Paired dimensional and unit-transformation invariance for scientific law discovery

**Status:** preregistered benchmark proposal only. No transformation engine, dimensional checker, benchmark result, or SOTA claim is implemented by this document.

## Motivation and novelty boundary

A physical law should not be counted as a different discovery merely because the same quantities are expressed in metres instead of millimetres, seconds instead of milliseconds, or canonical coordinates instead of an admissible transformed coordinate system. Conversely, an expression that violates declared physical dimensions should not count as a valid scientific explanation simply because it fits a finite sample.

Dimensional analysis and unit-aware symbolic regression are established areas. Dimensionally-consistent equation discovery using probabilistic attribute grammars reported discovering 60% more equations on the standard Feynman benchmark (Information Sciences, 2023; https://doi.org/10.1016/j.ins.2023.03.073). Buckingham-Pi methods discover dimensionless groups and can collapse systems across scales (Nature Computational Science, 2022; https://www.nature.com/articles/s43588-022-00355-5). The proposed contribution is not to reinvent either method. It is a paired, metamorphic qualification protocol for whether discovery systems preserve law identity and predictive meaning under controlled representation changes.

## Reuse existing Symthaea work

- Reuse [#6879 — ScientificModel / PhysicsIR](https://github.com/Luminous-Dynamics/symthaea/issues/6879) for dimensions, units, coordinate/reference frames, and deterministic model identity. Do not introduce a competing dimensional IR.
- Compose with [#8487 — scientific-discovery recovery benchmark](https://github.com/Luminous-Dynamics/symthaea/pull/8487), [#8607 — noise-robust law discovery](https://github.com/Luminous-Dynamics/symthaea/issues/8607), and [#8628 — law drift vs measurement drift](https://github.com/Luminous-Dynamics/symthaea/issues/8628).
- Reuse the compatible Feynman/SRBench tasks and existing dimensionally-aware symbolic-regression baselines. Pin versions, grammars, split definitions, and budgets.

## Core paired experiment

Freeze one evaluator world, its latent law, observation realization or paired-noise stream, physical dimension vectors, and discovery/evaluation split. Produce paired datasets through known unit conversions. Each policy sees the transformed dataset as an ordinary task; the evaluator keeps the correspondence and canonical law hidden until scoring.

### First transformation tranche

1. SI-to-scaled-unit changes, such as metres to millimetres and seconds to milliseconds.
2. Output-unit changes with consistent coefficient conversion.
3. Dimensionless normalization using frozen characteristic scales.
4. Variable permutations and sign changes only where the target law is mathematically equivariant.
5. Orthogonal coordinate rotations only for scalar/vector/tensor task families whose declared physical structure supports that transform.
6. Surface coordinate frame changes only where the surface task's transform contract explicitly allows them.

Do not apply arbitrary transformations and claim invariance if the physical problem is not invariant under them. Freeze the expected group/action and transformed form for every task family.

## Metrics

Report every metric by structural family × transformation family × noise regime × grammar-support condition:

- Dimensionally valid candidate rate, with full candidate denominator.
- Recovery of the same canonical dimensionless functional class across paired unit systems.
- Canonical-unit round-trip prediction error after converting each discovered expression back through the evaluator's independent unit map.
- Pairwise class-posterior distance between transformed representations, only where posterior semantics and equivalence contracts are comparable.
- False law-change rate caused solely by a unit transformation.
- False confidence where a method exploits a numerical scale artifact.
- Non-identifiable, unsupported, and out-of-support results retained explicitly.
- Data points, candidate evaluations, runtime, memory, and simplification work.

Predictive invariance and posterior invariance are separate claims. A system can predict invariantly without recovering or calibrating the correct law posterior.

## Baseline ladder

- Best compatible symbolic-regression method without dimensional priors.
- Compatible dimensionally constrained method, such as a probabilistic attribute grammar or Phi-SO-style search, where implementation and grammar can be matched.
- Buckingham-Pi/dimensionless-feature baseline where its assumptions apply.
- Exact tiny-grammar oracle for small tasks.
- Random/constant and deliberately dimensionally invalid negative controls.

Published Feynman benchmark scores are context, not paired replication of this metamorphic protocol. Run baselines under matched data, grammar, compute budgets, and evaluator splits.

## Acceptance gates

1. Unit converters have exact round-trip known-answer tests for every declared unit and physical dimension.
2. The evaluator independently checks dimensional consistency; the policy never receives target or equivalence labels.
3. Paired unit transformations preserve canonical predictions within preregistered numerical tolerances.
4. A candidate with a contradictory dimensional signature receives no scientific-validity credit even if its finite-sample fit is strong.
5. A unit-only transformation causes zero false law-change decisions on deterministic exact controls.
6. Posterior class-mass invariance is evaluated only where the posterior interpretation is comparable; predictive and posterior invariance remain distinct.
7. Confirmatory worlds cover at least six structural families, and each transformation cell has a preregistered precision/power denominator.
8. A superiority claim requires the strongest compatible dimensionally-aware and unconstrained baselines, matched budgets, independent uncertainty intervals, and full cost accounting.

## Claim ceiling

This is a benchmark proposal, not an implementation or a performance result. Reuse existing PhysicalType/PhysicsIR representations if they cover required semantics. Do not implement a new production dimensional representation as part of this work.