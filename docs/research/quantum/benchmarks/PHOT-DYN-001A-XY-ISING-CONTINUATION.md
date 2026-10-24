# PHOT-DYN-001A: source-grounded XY-to-Ising continuation benchmark

Status: **protocol and equation transcription; no Symthaea reproduction or device result claimed**

Parent issue: [PHOT-DYN-001 #8627](https://github.com/Luminous-Dynamics/symthaea/issues/8627). Reuse the canonical MIS-to-QUBO/Ising representation in [PHOT-OPT-001A / PR #8618](https://github.com/Luminous-Dynamics/symthaea/pull/8618).

## 1. What the published result actually establishes

Kim & Yamamoto, *Accelerating a coherent Ising machine by XY-Ising spin transition*, Scientific Reports **16**, 10396 (2026), doi: [10.1038/s41598-026-41315-7](https://doi.org/10.1038/s41598-026-41315-7), numerically model classical phase-quadrature dynamics that interpolate between:

- phase-sensitive degenerate optical parametric amplification (DOPA), which suppresses the out-of-phase quadrature and yields Ising-like spins; and
- phase-insensitive non-degenerate optical parametric amplification (NOPA), which permits continuous phase / XY-like spins.

The article reports, for its N=60, alpha=0.8 demonstration, median time-to-solution values of about 5.2e3 round trips for Ising-only, 3.5e3 for XY-only, and 2.0e3 for a monotone XY-to-Ising transition (2.6x relative to Ising-only). Its harder-instance study reports roughly order-of-magnitude reductions in some conditions and a mean speedup factor around 3 for N=200. Its adaptive schedule returns to XY when trapped and is optimized using SPSA. These are the article's numerical results, not photonic hardware measurements by Symthaea; the reported model neglects quantum fluctuations.

Source details:
- Paper / equations / methods: https://www.nature.com/articles/s41598-026-41315-7
- Published PDF: https://www.nature.com/articles/s41598-026-41315-7.pdf

Do not compress the paper's multiple settings into one universal "10x" claim. The N=60, alpha=0.8 Figure 3 comparison is 2.6x; the larger factors belong to separate, harder-instance conditions.

## 2. Source-transcribed model to independently review

For each oscillator i, let x_i(t) and p_i(t) be in-phase and out-of-phase amplitudes; a_i(t)=x_i(t)+i p_i(t); J_ij be the symmetric Ising coupling matrix with zero diagonal; Δt the round-trip step; κ the momentum-like feedback parameter; G the small-signal power gain; and E_s the saturation pulse energy.

The paper's classical difference equations are:

```text
x_i(t+Δt) - x_i(t)
  = Δt * [-x_i(t)
          + g_x(t) * Σ_{j != i} J_ij x_j(t)
          + κ * (x_i(t) - x_i(t-Δt))]

p_i(t+Δt) - p_i(t)
  = Δt * [-p_i(t)
          + g_p(t) * Σ_{j != i} J_ij p_j(t)
          + κ * (p_i(t) - p_i(t-Δt))]
```

Define the complex feedback field F_i(t)=Σ_{j != i} J_ij a_j(t). The gain terms are transcribed as:

```text
g_x(t) = sqrt(G) / (1 + |F_i(t)|^2 / E_s)

g_p(t) = [(1 - c(t))*sqrt(G) + c(t)/sqrt(G)]
         / (1 + |F_i(t)|^2 / E_s)
```

Check the typography against Eqs. (1)-(4) in the paper before turning this text into executable code. In particular, the paper's displayed gain denominators are oscillator-specific via F_i; they should not be accidentally replaced by a single global denominator. A code-review fixture must confirm limiting cases:

- c(t)=1: phase-sensitive / Ising-like gain; g_p's numerator is 1/sqrt(G).
- c(t)=0: phase-insensitive / XY-like gain; x and p receive the same gain numerator sqrt(G).
- 0<=c(t)<=1 for any admissible schedule.
- J_ii=0 and the matrix has the declared symmetry/convention.
- When coupling is identically zero, feedback terms disappear; the trajectory should reflect only loss, momentum and the specified gain/saturation model.

The article defines its scored discrete spin from the sign of x_i, even while p_i can be nonzero in the XY regime. The projected assignment is a scoring output, **not a physical measurement unless a calibrated optical detector was actually used**.

### Published schedule forms

- Pure XY: c(t)=0.
- Pure Ising: c(t)=1.
- Monotone transition: c(t)=tanh(ε t / Δt), with the published experiment-specific ε.
- Adaptive schedule: the article parameterizes a tanh envelope multiplied by a piecewise Heaviside switching function; transition times are optimized by simultaneous perturbation stochastic approximation (SPSA) against time-to-solution.

Do not tune epsilon or adaptive schedule parameters on the held-out instances. Reproducing the paper's tuned values is a replication; tuning new values on separate train instances is a new experiment.

## 3. Milestone A — numerical transcription fidelity

Implement the classical equations in a deterministic, dependency-free Rust reference model before any optical adapter exists.

Required checks:

1. Validate symmetry, zero diagonal, dimensions, finite couplings, positive finite G and E_s, finite Δt>0, and finite κ.
2. Preserve the exact time-indexing: x(t-Δt) and p(t-Δt) are history state, not aliases of the current state.
3. Evaluate each feedback field and saturation term independently and reject non-finite intermediate values.
4. Test both limiting gain regimes and interpolation endpoints.
5. Test zero-coupling, one-spin, two-spin, and analytically controlled tiny systems.
6. Freeze the update order, initialization RNG, float precision and stop rule; require bit-identical replay where floating-point platform semantics allow it, and document any cross-platform deviations.
7. Keep the raw continuous-state trajectory separate from the projected spin vector and its Ising energy.
8. Preserve diverged, failed, non-finite, and timeout outcomes as failures in the run manifest.

The implementation is a **classical numerical model inspired by an optical system**. It does not reproduce quantum fluctuations, shot noise, optical detection, or hardware control unless those terms are separately modeled and independently validated.

## 4. Milestone B — reproduce one published benchmark before extending it

Primary replication anchor:
- N=60, alpha=0.8 Wishart-planted instances.
- Compare pure Ising, pure XY, and the paper's monotone XY-to-Ising schedule.
- Freeze Chook generator/version, all generator parameters, planted ground state, coupling-matrix construction, solver parameters, initial conditions, number of instances, total trials and success rule from the paper and its supplement.
- Candidate upstream audit pin for planning: `dilinanp/chook` current master commit `7f124bdfac44f1a6c26a21f00059bff12d52c22e`; the repository declares Apache-2.0. This is a source-provenance starting point, **not proof** that this exact revision matches the paper's run. Confirm the paper/supplement/package version and defaults first; then freeze the configuration bytes and hashes of generated instance/coupling/ground-truth records. Do not vendor Chook code merely to establish a source pin.
- Use the article's time-to-solution definition to reproduce published metrics. The article uses a 99% aggregate success target and selects run duration to minimize TTS.
- Report both the paper-comparable continuous estimate, T * ln(1-q) / ln(1-p_s), and the operational whole-run ceiling; never label them interchangeably.
- Also report uncertainty-aware TTS using the lower Wilson confidence bound on the single-run success probability. Zero observed successes means TTS is unavailable from the point estimate, not zero.
- Separately record all-in wall time including model setup, schedule/controller overhead and decoding.
- Start with a small implementation smoke cohort to validate arithmetic and output schemas; only call it a replication after the source-matched cohort and analysis rule have been run.

The article's harder-instance figure uses separate alpha levels and 10 independent instances per level, while its methods discuss 50-instance studies for other figures. Do not merge those different sample counts or figures under one pooled average.

## 5. Milestone C — schedule generalization and adaptive control

Use three data partitions by *instance*:
- development: tune the adaptive policy and numerical settings;
- validation: select among predeclared policy families;
- confirmation: one-time evaluation after all policy and stopping rules are frozen.

Candidate schedules:
1. pure Ising;
2. pure XY;
3. source-reported monotone tanh schedule;
4. piecewise schedule with one or more XY returns;
5. adaptive schedule using only previously specified trajectory diagnostics and explicitly charged controller overhead.

Negative controls:
- shuffled transition times retaining the same number/duration of XY intervals;
- random schedules drawn from the same timing range;
- an Ising-only schedule given the same total numerical steps and wall-clock budget;
- fixed schedules whose parameters are permuted between instances;
- an oracle-tuned per-instance schedule shown only as a clearly labelled upper bound, never as a fair deployable baseline.

The controller may not consult the planted solution, ground state, held-out outcomes, or future trajectory state to choose a schedule. Retain the whole tuning trace so post-selection is visible.

## 6. Milestone D — move to MIS without changing the question silently

Only after Milestone B has been validated should the dynamics be transferred to the MIS QUBO/Ising model.

- Generate/freeze the graph cohort once and record canonical graph digests.
- Map using PHOT-OPT-001A and verify every coefficient convention against the tiny exhaustive oracle.
- Use identical completed graph bytes for static Ising, continuation schedules and classical controls.
- Project x-quadrature to a binary assignment with a declared zero/tie rule, then run the independent MIS verifier.
- Score raw projected feasibility/objective before any repair. If repair is included, make it a separate stage with identical budgets across arms.
- Keep planted Wishart results, QOBLIB MIS results, and random-regular MIS results as distinct workloads with independent cohort-level results.

A gain on planted fully connected Wishart QUBOs does not imply gain on sparse MIS graphs; it is only evidence for a transferable mechanism if confirmed in the independent MIS track.

## 7. Metrics and inference

Primary:
- exact planted-ground-state success probability where the planted target is guaranteed;
- time-to-solution at 99% aggregate success using the article's definition;
- distribution of success probability versus run duration and full TTS curve;
- for MIS transfer, independently verified cardinality, independence ratio and target-hit probability;
- all-in wall time separately from model round trips or nominal device latency.

Secondary diagnostics:
- best-so-far projected Ising energy;
- rate of transition out of stagnating basins under a predeclared stagnation definition;
- amplitude distribution and finite/diverged trajectory rate;
- sensitivity to noise, finite precision, coupling scale, initialization and schedule parameters;
- full computation cost of SPSA, schedule selection, repeated trials, post-processing and verification.

For each instance, preserve every trial. Use problem instance as the independent statistical unit and repeated trajectories as nested runs. Report median and interquartile range plus confidence intervals for the graph-/instance-level TTS improvement; do not treat time steps, spins or shots as independent sample units. Predeclare multiplicity correction for multiple schedule and hardness comparisons.

## 8. Advancement gate

Advance to MIS hardware-neutral adapter design only if:

- the paper's equation transcription is source-reviewed;
- limiting-case and tiny-system tests pass;
- the published N=60, alpha=0.8 direction is reproduced within a predeclared tolerance or the discrepancy is documented and explained;
- the gain is not an artifact of extra numerical steps, an unequal retry count, tuning leakage, or different stopping semantics;
- any claimed schedule improvement survives held-out instances and shuffled/random schedule controls.

Do not claim a SOTA win from the paper's external speedup, a simulated trajectory, projected spin output, or a single tuned instance. No physical photonic result or Symthaea advantage is claimed by this document.
