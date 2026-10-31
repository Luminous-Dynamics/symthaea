# HDC-LTC Input-Trajectory Integration Under Irregular Sampling v0.1

**Status:** research protocol / preregistration; not executed  
**Coordinating issue:** [NEURO-SILICON-006 #9371](https://github.com/Luminous-Dynamics/symthaea/issues/9371)  
**Parent experiment:** [NEURO-SILICON-005 #9369](https://github.com/Luminous-Dynamics/symthaea/issues/9369)  
**Active implementation:** `crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs`  
**Inspected source SHA:** `40c3b0e75845a20b3ef2dbbe0ea3a0af0068f4b7`  
**External motivation:** NIALIM/DFA-CfN (IEEE TNNLS 37(9), 2026), without assuming the published method transfers.

## 1. Research hypothesis

A sequence of irregular observations should be treated as a sampled input trajectory, not simply as an unordered series of vectors with a scalar `dt`. The active HDC-LTC methods currently evaluate one input vector per call and assume that this input is held across the full interval.

On smooth high-curvature or irregularly sampled inputs, that zero-order/constant-input assumption may dominate trajectory error. On discrete events and abrupt steps, interpolation may be harmful because it smears a real transition. The experiment therefore tests *when to hold, interpolate, subdivide, or respect a discontinuity*.

## 2. Exact source behavior and important reference limitation

For the active implementation:

- `compute_equilibrium(input)` computes `activation(bundle(bind(weight_hv, state), bind(input_mask, input), optional_fourier_basis))`.
- `compute_tau(input)` uses current state norm and input similarity against `tau_modulator`.
- `compute_gating(input, dt)` forms a state/input gate base `sigma_base`, then `decay = exp(-dt/tau)` and `sigma = 1 - decay * (1 - sigma_base)`.
- `evolve_closed_form` evaluates equilibrium and gating once with the supplied input vector, then interpolates the state once.
- `evolve_rk4` passes the *same* input to all four RK stages.
- `evolve_closed_form_iterative` recomputes equilibrium and tau during substeps but still supplies the same fixed input to every substep.

### Fourier/reference caveat found in source

There is a potential mismatch that must be resolved before using the existing RK4 path as an oracle with Fourier features enabled:
- `compute_equilibrium` optionally bundles `compute_fourier_basis()`.
- `compute_derivative`, used by `evolve_rk4`, currently bundles only transformed state and masked input; it does not include the optional Fourier basis.
- Therefore the current RK4 method is not a same-equation reference for Fourier-enabled runs unless this is intentional and explicitly separated. The initial variable-input reference should either disable Fourier features and record that restriction, or implement a derivative whose equilibrium exactly matches the configured equation, with independent tests.

This is a source-level observation, not proof of a bug in all intended Fourier modes. The owner should adjudicate before any numerical reference is promoted.

## 3. Define a completed input segment

For adjacent observations ((t_{n-1}, u_{n-1})) and ((t_n,u_n)), define:
- (Delta t_n = t_n - t_{n-1});
- normalized segment coordinate (alpha(t) = (t-t_{n-1})/Delta t_n);
- a candidate piecewise-linear input path (u(t)=(1-alpha)u_{n-1}+alpha u_n).

This path is available only once the current endpoint (u_n) arrives. It does not use any sample after (t_n). For online inference, it is a causal retrospective integration of the interval that has just completed.

Do not assume an interpolated ContinuousHV is semantically equivalent to encoding an interpolated raw sensor vector. If the encoder is nonlinear, those paths differ and need separate arms. Do not normalize the interpolated HV unless that is a separate, explicitly versioned representation.

For asynchronous/event data, attach a discontinuity marker. Interpolation must not cross marked discrete events; use the declared event hold/reset policy on those segments.

## 4. Input-path strategy matrix

| Arm | Input path | Purpose |
|---|---|---|
| U0 | current right-endpoint zero-order hold | Exact incumbent |
| U1 | previous/left-endpoint zero-order hold | Endpoint bias control |
| U2 | midpoint hold | One extra path estimate |
| U3 | piecewise linear ContinuousHV endpoints, 2 substeps | Minimal linear path |
| U4 | piecewise linear ContinuousHV endpoints, 4/8/16 substeps | Refinement surface |
| U5 | interpolate raw feature endpoints then encode each stage | Distinguish raw interpolation from HV interpolation |
| U6 | causal degree-2 Lagrange path from current and historical endpoints | Curvature-aware candidate |
| U7 | causal degree-3 Lagrange path where enough valid history exists | Higher-order candidate; test overshoot |
| U8 | adaptive substeps from input curvature / compact HDC novelty | Resource-aware candidate |
| U9 | discontinuity-aware event hold; prohibit interpolation across event boundary | Event semantics control |

The NIALIM paper's method must be reproduced from its actual derivation as a separate arm. The method is not assumed to equal generic piecewise-linear or polynomial interpolation. The public DFA-CfN code snapshot is pinned and audited in #9369 / PR #9370, but exact correspondence to the final paper's equations remains unverified.

## 5. Two numerical tracks — different contracts

### Track A: underlying ODE reference

The source declares the ODE:

`dx/dt = (x∞(x,u(t)) - x) / τ(x,u(t))`.

Build an independent variable-input RK4 reference where the input is evaluated at each stage:
- k1 uses (u(t));
- k2 and k3 use (u(t+Delta t/2));
- k4 uses (u(t+Delta t)).

Use the declared path (ZOH or interpolated) to evaluate (u(t)), and refine with 1/2/4/8/16 or more substeps until successive solutions satisfy the frozen numerical tolerance. If the Fourier feature is configured, the derivative's equilibrium must use precisely the same Fourier term and activation as `compute_equilibrium`.

This is a numerical reference for the underlying ODE. It is **not** a claim that the learned-gating closed-form update is exactly the same equation.

### Track B: learned-gating closed-form execution

For U0, preserve the exact incumbent `evolve_closed_form` path. Candidate substeps may evaluate the same closed-form rule at (u(t_j)) for each segment, recomputing equilibrium, tau and gate. The candidate is a separately versioned numerical path.

Do not change the learned `sigma_base` law or interpret it as a new ODE without derivation. Avoid changing running statistics by accident: internal substeps and external observation intervals must be recorded separately. An implementation should aggregate per-observation statistics according to an explicit policy rather than letting the number of hidden substeps change the statistics update frequency.

## 6. Time, missingness, and event contract

- `dt` is derived from monotonic input timestamps; it is never wall-clock time or call count.
- Reject NaN/infinite timestamps, non-positive dt, and arithmetic overflow. No invalid input is converted to a valid transition via silent clamping.
- Preserve actual missingness masks and elapsed time. A missing input is not an interpolated observation unless the policy declares a causal imputation model.
- A large unobserved gap is not evidence of smooth evolution. Test long gaps and sudden jumps.
- At a declared discontinuity, do not interpolate across the event boundary.
- Adaptive schemes must have a deterministic maximum number of evaluations per interval and an explicit fallback/incomplete result when the budget is exceeded.
- If the input is held for a segment, name whether it is the left endpoint, right endpoint, or midpoint; do not write generic "ZOH" without stating the sample.
- State update ordering and timestamps must be identical across the candidate and reference.

## 7. Factorial experiment with fast/slow memory

Compose this work with #9369 rather than duplicating its runner. Cross:
- fast/slow arms A0–A9 from the parent protocol;
- input-path strategies U0–U9;
- task regime (smooth/fixed-dt, irregular, high curvature, event-driven, discontinuous, missing/large-gap);
- dimension and memory placement from QARCH-021/#9341.

A first screening matrix may hold slow context disabled and test U0–U9, then add only the strongest input-path candidates to the fast/slow factorial. Otherwise the cross product will be unnecessarily expensive.

## 8. Workload matrix and negative controls

Reuse pinned tasks from #9369: S-MNIST, PS-MNIST, SHD, SSC, DVS Gesture, and Symthaea temporal prediction/recall/control tasks with a real oracle. For external NIALIM tasks, verify paper preprocessing, grouped splits and pinned source.

Required synthetic and dataset-derived conditions:
- constant input: ZOH should be competitive and interpolation should not invent motion;
- linear ramp: tests first-order path handling;
- smooth sinusoidal/chirp: tests curvature and cadence;
- high-curvature smooth transitions: tests higher-order benefit;
- abrupt step/impulse: interpolation can smear the event;
- asynchronous event bursts: preserve event boundary semantics;
- missing intervals and irregular long gaps: preserve time and missingness;
- duplicate, out-of-order, zero, negative and non-finite timestamps: fail closed.

## 9. Metrics and promotion gates

Keep independent:
- trajectory error vs refined variable-input RK4;
- trajectory error vs constant-input RK4 (diagnostic only);
- exact incumbent output identity on U0;
- task quality (classification/prediction/control);
- state bounds, numerical stability, event timing and recovery;
- equilibrium/tau/gate evaluations per observation;
- substeps per interval and actual HDC encoding work;
- persistent/temporary/peak bytes and bytes moved;
- p50/p95/p99 end-to-end latency and deadline misses;
- directly metered energy per correct outcome with a complete system boundary.

Proposed goals, not results:
1. A variable-input RK4 oracle must show convergence under refinement before reference use.
2. On held-out smooth/high-curvature irregular tasks, a candidate should target at least 20% lower trajectory error than right-endpoint ZOH while satisfying frozen task-quality/stability thresholds.
3. On constant-input and discontinuity controls, candidate degradation must remain within predeclared tolerances; event smearing is a failure, not an acceptable side effect.
4. For deployment, target no more than 10% p95 latency increase at equivalent quality; an efficiency breakthrough requires at least 2× whole-task throughput or 30% lower directly metered energy per correct result.
5. The adaptive candidate must have a deterministic evaluation budget and demonstrate where it abstains or falls back.

## 10. Literature and external reference boundary

Liang et al., IEEE TNNLS 37(9), September 2026, DOI 10.1109/TNNLS.2026.3669345, propose NIALIM using Lagrange interpolation and temporal interval factors for LTC equations. The paper reports SOTA claims on six tasks and up to 69.2% improvement. Those are published claims, not Symthaea measurements:
- https://ieeexplore.ieee.org/document/11424014/
- https://pubmed.ncbi.nlm.nih.gov/41790819/

## Non-claims

This protocol does not show that input interpolation improves HDC-LTC, that a linear segment is correct for event data, that the current RK4 method is a same-equation oracle for Fourier-enabled configurations, or that NIALIM has been reproduced. It changes no production default.
