# SEMI-CAUSE-001 — Source-attributed drift and identifiability v0.1

**Issue:** [#13437](https://github.com/Luminous-Dynamics/symthaea/issues/13437)  
**Implementation:** `symthaea-semiconductor::benchmark::drift_identifiability`  
**Status:** executable small-instance structural-identifiability primitive; not yet exact-head validated.

## Research thesis

A VM residual is evidence of disagreement, not evidence of which subsystem changed. Process state, instrument bias and shared common-mode effects can be observationally equivalent. A useful system must determine which causal quantities are identifiable from the measurements actually available, and which additional measurement would make an intended contrast identifiable.

The initial model has four latent coordinates:

[
\theta =
\begin{bmatrix}
\Delta_{process} & b_A & b_B & c_{shared}
\end{bmatrix}^{T}
]

Rows in the observation-design matrix describe how an admissible measurement responds to each coordinate. The initial fixtures use:

| Observation | Process | Gauge A | Gauge B | Shared cause |
|---|---:|---:|---:|---:|
| Wafer measured on gauge A | 1 | 1 | 0 | 1 |
| Stable witness on gauge A | 0 | 1 | 0 | 1 |
| Wafer measured on gauge B | 1 | 0 | 1 | 1 |
| Stable witness on gauge B | 0 | 0 | 1 | 1 |
| Independent common-mode monitor | 0 | 0 | 0 | 1 |

These are synthetic sensitivity rows. They are not measured instrument characteristics or a universal semiconductor observation model.

## Structural identifiability

For a design matrix (A), the initial implementation determines whether a requested linear contrast (c^T\theta) is estimable using the row-space criterion:

[
\operatorname{rank}(A) =
\operatorname{rank}
\begin{bmatrix}
A \\
c^T
\end{bmatrix}
]

This is stronger than requiring the entire latent parameter vector to be identifiable. A physical process shift can be identifiable even when individual gauge and shared-reference offsets remain confounded.

Known-answer examples:

- Wafer-A measurement alone: process and gauge-A bias are not separately identifiable.
- Add stable witness on gauge A: the process-shift contrast becomes identifiable, because differencing the wafer and witness equations removes the gauge-A and shared components.
- Add an independent common-mode monitor: gauge-A bias and the shared common mode become separately identifiable.
- If that independent monitor is absent, the system must not invent a separation between gauge and shared-reference drift.

## Implemented Rust primitive

The module currently provides:

- typed source contrasts: process, gauge A, gauge B and shared common mode;
- typed observation-action evidence states: observed-qualified, available-qualified, unknown and rejected;
- finite sensitivity-row validation and duplicate-ID rejection;
- small-matrix rank / contrast-estimability evaluation;
- exhaustive minimum-cost action subset planning for up to 20 candidate actions;
- conservative sequential delay accounting and a hard incremental-cost budget;
- deterministic input and report digests;
- explicit no-plan / not-identifiable outcome;
- no measurement execution, control or authority field (execution authority is always false).

The planner uses action sensitivity rows supplied by the declared design. It does not learn or authenticate them. It does not infer a physical cause from measured outcomes. Correlated noise, reference-standard uncertainty and likelihood calibration are deliberately not implemented here; compose #9234 and the measurement-system owners.

## Authored tests

The initial module contains nine Rust tests for:

1. process/gauge confounding from a single production measurement;
2. process identifiability after a reference crossover while gauge/common-mode remain confounded;
3. separation of instrument bias from a shared cause after an independent monitor;
4. minimum-cost discriminating action selection;
5. rejection of unknown/ineligible actions from the plan;
6. hard-budget refusal;
7. duplicate action IDs and non-finite sensitivity rejection;
8. input-order-independent deterministic receipts;
9. multiple-target planning, requiring every requested contrast to be estimable.

Tests are authored, not yet reported as executed. They are not a reliability estimate or physical qualification result.

## Next benchmark layer

The next increments should add:
- a frozen JSON corpus with hidden source combinations and observation-equivalence controls;
- a test-only independent rank/contrast oracle;
- simultaneous process/gauge/common-mode scenarios;
- gauge-B crossover and reference-standard drift;
- covariance and non-Gaussian observation-model adapters from #9234;
- selection/delay/label maturity composition from #13328 and #13269;
- matched baselines: residual CUSUM/EWMA, multivariate SPC, M2D2-style multimodal drift, static source classifier, reference-check heuristics and decision-valued experiment selection;
- blinded laboratory crossover before any fab attribution claim.

A paper-reported multimodal feature attribution is a useful baseline, but it is not equivalent to a controlled identifiability experiment. The benchmark must distinguish detection, cause attribution, and selection of the next discriminating measurement as three separate outcomes.

## Claim ceiling

This is a synthetic structural-identifiability benchmark. It does not prove calibration traceability, independent reference chains, correct source attribution in a fab, physical MSA capability, production savings, or authority to correct calibration, alter a recipe, hold/release wafers, or control equipment.
