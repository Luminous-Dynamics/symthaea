# Evidence-qualified symmetry discovery benchmark v0.1

This directory freezes a matched benchmark protocol for the Symthaea/Symtropy symmetry programme. It separates proposal quality, mathematical qualification, physical qualification, task invariance and search reduction. It is deliberately not a single leaderboard.

## Why this benchmark exists

Symmetry systems increasingly use different discovery paradigms. They do not all solve the same task:

- **Group-space distribution discovery.** LieFlow learns a distribution over an ambient group and reads subgroup support from it. Its ICML 2026 study covers continuous and discrete subgroups, including synthetic 2D/3D point clouds, ModelNet10 and MI-Motion. A generator-first benchmark alone would miss this class of method. https://proceedings.mlr.press/v306/chen26dg.html
- **Spectral fingerprints.** SymFP uses singular-value/rank structure of equivariant representations for point-group templates and local/degree-resolved symmetry-breaking diagnostics. It is a proposal/fingerprint lane, not an action proof. https://arxiv.org/abs/2611.07410
- **Learnable soft equivariance.** Geometric Learnable Invariants Transformers learn which symmetries are useful for an image-classification task. That is a task-level fingerprint and must not be pooled with physical point-group identification. https://arxiv.org/abs/2611.20150
- **Exact finite-group statistical discovery.** Soleymani et al. (COLT 2026) study unknown symmetry discovery on the subgroup lattice of a finite reference group, with polynomial-time recovery and minimax-optimal sample complexity for regression over finite-dimensional feature spaces. Reproduce the paper's theorem assumptions exactly; do not generalize those guarantees to unknown ambient groups or physical dynamics. https://proceedings.mlr.press/v336/soleymani26a.html
- **Noisy active acquisition.** Cost-sensitive noisy optimal decision trees and Adaptive Submodular Ranking with Noise are established algorithms, including persistent noise. Our evidence layer should compare faithfully, not claim to invent those algorithms. https://www.jmlr.org/papers/v25/23-1484.html . AFABench is a relevant common evaluation framework for active feature acquisition. https://kdd2026.kdd.org/papers/

## Current research baselines — distinct native tasks

The comparison set must be versioned and matched by what each method actually returns:

- **LieFlow (ICML 2026):** learns a distribution over an ambient Lie group; support is used to propose continuous/discrete subgroup structure. Its published evaluation includes synthetic 2D/3D point clouds, ModelNet10 and MI-Motion. Evaluate proposal recall, subgroup recovery and cost on the point-cloud/action lane, then pass proposed actions through the independent finite/Lie and task-preservation verifiers. [Paper](https://proceedings.mlr.press/v306/chen26dg.html)
- **SymFP (arXiv, 5 November 2026):** singular-value/rank fingerprints of equivariant representations for point-group identification and local/degree-resolved symmetry breaking. This is a new preprint; first require faithful reproduction of the representation-rank templates, leakage scores and subgroup monotonicity before treating its results as independently established. [Preprint](https://arxiv.org/abs/2611.07410)
- **Geometric Learnable Invariants Transformers (arXiv, 15 November 2026):** a soft-equivariant model learns which candidate symmetries help an image-classification task and emits a task fingerprint. It should be evaluated in its image-task lane against matched equivariant and unconstrained models, not as an exact geometric stabilizer oracle. [Preprint](https://arxiv.org/abs/2611.20150)
- **Soleymani et al. (COLT 2026):** exact symmetry discovery for regression relative to a known finite ambient group, under the paper's identifiability assumptions. Reproduce the estimator and theorem-matched setting exactly; stress tests outside those assumptions are separate. [Paper](https://proceedings.mlr.press/v336/soleymani26a.html)
- **MLSD (Physical Review E, 5 February 2026):** learns continuous generators and Poisson/Lie structure from canonical trajectories, with demonstrations on 3D Kepler and isotropic harmonic oscillator. Keep those integrable Hamiltonian assumptions explicit. [Paper](https://journals.aps.org/pre/abstract/10.1103/jl75-7ynx)
- **LieStoNet (ICML 2026):** discovers Lie-point symmetries of stochastic dynamical systems from spatiotemporal data and treats drift and diffusion determining equations separately. Its lane must include held-out SDE trajectories and diffusion/noise-preservation controls, not just deterministic vector-field tests. [Paper](https://proceedings.mlr.press/v306/liu26aq.html)
- **One-parameter subgroup discovery (ICML 2026):** targets interpretable elliptic, hyperbolic and parabolic one-parameter subgroups. Include these templates as a subgroup-specific baseline where the ambient action is continuous and the data meet its scope. [Paper](https://proceedings.mlr.press/v306/karjol26a.html)
- **Noisy active acquisition:** compare against the approximation algorithms for noisy Optimal Decision Trees and Adaptive Submodular Ranking with Noise; do not substitute our evidence replay for those optimizers. [JMLR 2024](https://www.jmlr.org/papers/v25/23-1484.html). Use AFABench for comparable acquisition-policy data/protocol where its task semantics fit. [AFABench](https://arxiv.org/abs/2508.14734)

The benchmark must preserve output type: group distribution, exact subgroup label, spectral fingerprint, predictive task behavior, vector-field/SDE generator, physical step-map certificate, and quotient proof are distinct outcomes.

## The evidence ladder

Every evaluated method produces a proposal/report with a typed evidence level. The benchmark scores each stage separately:

1. **Candidate proposal:** what group/action/subgroup candidates are suggested, at what cost, with what recall?
2. **Mathematical qualification:** does the declared action compose correctly; are generators independent, closed and Jacobi-consistent; are representations and metrics covariant?
3. **Physical qualification:** does the same physical discrete step survive the declared coordinate change and physical action? This is a separate Symtropy result.
4. **Task qualification:** are boundary/initial conditions, admissible set, objective, constraints and outputs invariant under the proposed action?
5. **Search reduction:** only a task-qualified, proof-carrying quotient may prune. Solutions must be lifted and independently checked against the original task.

A learned score, spectral match or proposal confidence never bypasses a later stage. An unsupported or unresolved case is an abstention, not a negative proof.

## Non-pooled benchmark tracks

### A. Exact finite structural stabilizers

Ground truth: exact stabilizer subgroup of a finite ambient matrix group acting on frozen point sets/graphs.

Baselines: exact exhaustive stabilizer oracle, LieFlow/group-space support, SymFP where a faithful representation is available, learned soft fingerprint where task semantics apply, and simple augmentation/nearest-template baselines.

Measures: exact subgroup label, subgroup precision/recall, generator recall, closure defect, proposal recall@K, false exact promotions, orientation/point-order stability, runtime and memory.

### B. Exact finite-group regression symmetry discovery

Ground truth: invariance subgroup of a target function relative to a known finite reference group. This is the theorem-matched COLT 2026 lane.

Reproduce the paper's estimator, sampling, selection rule, theorem assumptions and sample-complexity accounting faithfully. Compare with exhaustive subgroup oracle on small groups, unconstrained regression, fixed-invariance models, and Symthaea's exact action verification. Report subgroup recovery, generator recall, held-out excess risk, sample complexity, query cost, false promotion and verification overhead.

### C. Spectral fingerprint / representation rank

Ground truth: known representation decomposition, exact rank template, local and degree-resolved symmetry-breaking labels. Compare faithful SymFP, exact harmonic/rank templates and learned fingerprints where the representation/task matches.

Measures: rank/template accuracy, leakage calibration, local defect localization, degree stability, orientation invariance, out-of-template abstention, and cost per fingerprint.

### D. Task-level soft equivariance

Keep image-classification or other predictive tasks in their own task lane. Compare the faithful soft-equivariant model, parameter/compute-matched non-equivariant transformer, and structurally equivariant baselines. Report task accuracy/calibration and task-relevant transformation behavior separately from exact object/physical symmetry.

### E. Dynamical symmetry

Separate scalar invariance, vector-field infinitesimal equivariance, flow-map equivariance and discrete-step equivariance. Compare generator-first MLSD/LieStoNet, LieFlow where the input/output semantics match, symbolic/exact oracles on supported cases, and held-out transformed trajectories. Never treat generator recovery as a discrete integrator proof.

### F. Physical coordinate and step-map covariance

Use matched physical models and frozen solver profiles. Compare Symtropy's direct dense-coordinate RATTLE/pHDAE paths with independent solver formulations. Report lifted state errors, multipliers, per-edge constraints, endpoint tangency, energy/work, momentum, conditioning, abstention and solver failures. This track does not pool with point-cloud W1 or ML classification accuracy.

### G. Proof-carrying search quotient

Compare unquotiented exhaustive/baseline search, static quotienting, task-aware quotienting and the evidence-gated compiler. Required gates: zero false hard-prunes, zero missed valid solutions after lift, 100% lift verification, byte-identical replay and complete resource accounting. Search reduction is scored only on fully qualified cases.

## Frozen protocol rules

- A case is identified by canonical manifest bytes and a digest generated by the harness; the fixture file itself is not the digest.
- Split by underlying structure, group family, orientation, transformation, parameter regime and trajectory—not by adjacent points/timesteps from the same object.
- Keep discovery, calibration/threshold selection and final holdout data distinct. The final holdout cannot select a threshold, rank template or candidate group.
- Use exact exhaustive or symbolic ground truth on small cases. For larger stochastic/learned lanes, preregister seeds and use at least 30 independent generated structures per family; timesteps/points within one structure are not independent replicates.
- Run parameter-matched and compute-matched comparisons where meaningful. Record hardware, software versions, wall time, peak memory and energy where measurable.
- Include all abstentions, failures, out-of-domain cases and rejected candidates in the denominator. Never reward a method for silently omitting hard cases.
- All solver/model parameters, split IDs, costs, stop criteria and comparator versions are frozen before the final holdout.

## Metrics and preregistered targets

All targets are prospective until the corresponding frozen benchmark has actually run.

| Metric | Prospective target |
|---|---:|
| Exact subgroup/action labels on the declared exact in-scope corpus | 100% |
| False exact/action promotions from learned or sampled proposals | 0 |
| Missed valid answers after a qualified quotient and lift | 0 |
| Lift verification for every returned quotient solution | 100% |
| Representation/orientation consistency where invariance is claimed | ≥99.9% |
| Subgroup proposal recall at a fixed candidate budget | ≥95% on held-out in-scope families |
| Bounded-noise false eliminations | 0 |
| Exact finite minimax agreement on small active-probe cases | 100% |
| Verified search-cost reduction | Aim for ≥2× on one substantial case; only after all safety gates |

Report risk-coverage and accuracy-cost Pareto curves rather than one pooled scalar. Report proposal recall separately from exact qualification acceptance. Report search reduction separately from false-prune and lift correctness.

## Industry comparison

Use matched semantics, not product-name comparisons. MuJoCo 4.0.1 (11 Nov 2026) includes a fix to a sparse Cholesky supernode bug that could give the Newton solver a wrong Hessian for sparse constraint Jacobians; MuJoCo 4.0.0 added speculative contacts and elasto-plasticity. Chrono 10.0.0 includes new Jacobian update strategies for implicit integrators. Pinocchio 4.1.0 adds URDF v1.2 and tangent-space acceleration/jerk limits. Drake documents where its discrete contact model is robust and where it can become unstable under large feedback gains, damping or external forcing.

- MuJoCo changelog: https://mujoco.readthedocs.io/en/4.0.1/changelog.html
- Chrono release notes: https://projectchrono.org/news/
- Pinocchio changelog: https://github.com/stack-of-tasks/pinocchio/blob/devel/CHANGELOG.md
- Drake MultibodyPlant: https://drake.mit.edu/pydrake/pydrake.multibody.plant.html

For mechanics, align masses, constraints, constitutive/contact models, integrator, timestep, frame conventions, tolerances and solver settings before measuring accuracy or speed. The differentiator under test is **qualified physical work per unit compute**, including failed/abstained cases and certificate overhead—not raw steps per second.

## Artifacts and reproducibility

- `fixtures/v0.1/cases.json`: five hand-authored exact controls plus 150 deterministic generated point-set instances (30 each for D4, C4, half-turn C2, reflection C2, and C1).
- `schemas/case_manifest.schema.json`: manifest validation.
- Future accepted run: baseline source/version, input and split digests, hardware/software record, raw predictions, verifier receipts, all failures, and scorecard.
- Never label a target as achieved from fixture unit tests alone.

## Claim boundary

This protocol is a benchmark contract, not evidence that Symthaea currently beats LieFlow, SymFP, COLT 2026 methods, soft-equivariant transformers, JMLR noisy ODT/ASRN, MuJoCo, Chrono, Pinocchio or Drake. Any superiority claim requires matched baseline executions on the frozen corpus.


## Exact fixture validator

Run the test-only validator (the production system remains Rust-first):

    python3 -m pip install jsonschema==4.26.0
    python3 -I research/symmetry-discovery-benchmark-001/tools/validate_fixtures.py

It validates the Draft 2020-12 manifest and then independently checks:
- all 64 entries of the frozen D4 multiplication table;
- exact identity, two-sided inverses, closure and associativity;
- unordered point-set stabilizers using exact decimal-to-rational parsing;
- polynomial invariance under each group matrix using exact rational coefficient expansion;
- the expected subgroup label for every frozen case.

The deterministic corpus is regenerated from the manifest's D4 matrices using xorshift32 orbit seeds. Reproduce or refresh it with:

    python3 -I research/symmetry-discovery-benchmark-001/tools/generate_d4_pointsets.py --write

The validator checks that the frozen 150 generated cases exactly match the generator output before computing their stabilizers. The workflow runs these checks against the exact triggering revision and is guarded against draft PR execution. This is fixture qualification only; no external learner baseline is run by it.


## Fixture arithmetic preflight (independent, not CI)

An independent deterministic replay of the current v0.1 manifest checked:
- all 150 generated point sets reproduce byte-for-byte from the declared seeds/attempt sequence and are unique;
- each generated set's exact D4 stabilizer equals its declared target subgroup (D4, C4, half-turn C2, reflection C2 or C1);
- the D4 matrix multiplication table is closed and every element has a two-sided inverse;
- the three hand-authored geometric labels (D4, C4, C1) match exact point-set stabilizers;
- the D4 and C4 polynomial targets have exactly their declared stabilizers under symbolic integer expansion.

This is a local fixture arithmetic preflight only. The exact-head CI fixture validator, full baseline adapters, holdout splits and benchmark runs remain separate gates. No accuracy or performance result is inferred from these fixture checks.
