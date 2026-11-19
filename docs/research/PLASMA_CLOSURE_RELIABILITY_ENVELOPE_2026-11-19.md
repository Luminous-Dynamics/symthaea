# Plasma closure reliability envelope benchmark v0.1

**Date:** 2026-11-19  
**Status:** research protocol and task manifest only; no closure reproduction or physical result  
**Parent:** [PLASMA-LANDAU-CLOSURE-REPRO-001 #12907](https://github.com/Luminous-Dynamics/symthaea/issues/12907)  
**Substage work:** [Runge–Kutta memory consistency #12910](https://github.com/Luminous-Dynamics/symthaea/issues/12910)  
**Tracking:** [PLASMA-CLOSURE-ENVELOPE-001 #12914](https://github.com/Luminous-Dynamics/symthaea/issues/12914)

## 1. Why this benchmark exists

A learned closure is not qualified by snapshot accuracy. It participates in a feedback loop: closure output changes the fluid state, which changes later closure inputs. The evaluation object must therefore be the coupled trajectory and its physical/numerical validity, not just a collection of pointwise predictions.

Two 2026 results make the gap concrete:

- Jeß et al., *Properties and challenges of machine learning closures for ten-moment fluid models in the case of Landau damping*, arXiv:2610.23080 (2026-10-16), report interpolation-range competence but degradation with increasing extrapolation distance. They show that offline scores do not fully explain online performance and examine fluid-entropy evolution as an additional diagnostic when kinetic references are scarce.
- Burles, Camporeale & Pezzi, *A neural operator closure for Landau damping in electrostatic plasma*, JPP 92(5), E149, DOI [10.1017/S0022377826102517](https://doi.org/10.1017/S0022377826102517), published online 2026-10-30, train an FNO online in a differentiable fluid solver with an explicit trailing history. Their result is a meaningful advance for its stated 1D electrostatic setup. They identify solver-scheme dependence and solver-independent deployment as limitations for further work.

This protocol does not imply either paper is incorrect. It tests what lies beyond their demonstrated envelopes.

## 2. Claim architecture: no single score

Report four independent envelopes:

| Envelope | Question | A pass means |
|---|---|---|
| Accuracy | Does the coupled trajectory match the kinetic reference over declared conditions? | Predeclared field, phase, damping and horizon margins hold |
| Physical admissibility | Does the coupled model remain physically and numerically admissible? | Applicable conservation, realizability, finite-state, stability and well-posedness checks hold |
| Epistemic | Does the model recognise unsupported conditions and avoid false confidence? | Calibrated uncertainty / abstention meets the registered risk target |
| Portability | Does the model survive changes to numerical method? | Zero-shot transfer is reported separately from fine-tuned adaptation |

A mandatory failure in any envelope blocks an overall reliability claim. A weighted average cannot compensate for a violation of a physical constraint.

## 3. Scope separation

The initial benchmark is 1D electrostatic Landau damping. It must not be pooled with magnetic-reconnection pressure-tensor closure discovery, tokamak disruption prediction, multimodal discharge reconstruction, or dusty plasma. Those are separate equation systems, observations, datasets and metrics.

## 4. Candidate tournament

After source admission and paper-faithful reproduction, compare under the same kinetic data, solver, initial conditions, train/validation split, architecture/training budget where meaningful, and rollout horizon:

1. Analytic Hammett–Perkins closure.
2. Matched offline FNO.
3. Paper-exact online/a-posteriori FNO.
4. Accepted-state-only and stage-explicit memory variants from #12910.
5. Markovian FNO and fixed-window sequence ablations.
6. Sparse-regression closure only after its own code/data provenance and license audit.
7. Symthaea CfC/LTC and hybrid explicit-window/latent-memory candidates.

The 2025 PNAS and 2026 online-FNO studies are not compared by copying headline scores. Their task definitions, solvers, data, splits and deployment conditions must first be reproduced. CfC/LTC gets no preference because it is the in-house architecture.

## 5. Frozen intervention matrix

Each lane needs full-cohort splits: never partition adjacent time windows from one trajectory across train and test.

### In-distribution and generalisation
- trained amplitudes;
- held-out amplitudes strictly between training values;
- extrapolation below and above the training range;
- wavenumber and phase perturbations;
- translated initial conditions;
- linear, transition and nonlinear trapping strata separately.

### Observation and memory
- regular and irregular sample cadence;
- missing or stale history slots;
- explicit physical-time lag vs fixed number of steps;
- memory-lag removal/permutation;
- accepted/provisional stage semantics under #12910.

### Numerical transfer
- timestep changes;
- spatial grid / resolution changes;
- SSPRK3 versus a second independently implemented integrator;
- solver transfer when a matched second fluid solver can be established.

A no-finetuning transfer and a retrained/fine-tuned result are separate rows. Do not use "solver-independent" or "zero-shot" for a retrained model.

## 6. Metric contract

### A-priori closure fit
- heat-flux-divergence MAE, RMSE and predeclared normalised error;
- field-wise density, velocity and pressure error;
- residual structure and phase-conditioned error;
- error versus amplitude and distance from the training support;
- predictive interval coverage and width.

### A-posteriori coupled trajectories
- time-resolved relative L2 error for density, velocity, pressure and electric field;
- electric-field energy trajectory;
- oscillation phase and fitted damping-rate error;
- transition timing and nonlinear bounce/trapping fidelity;
- stable rollout fraction and time-to-first-failure;
- compute, memory and wall-time per stable simulated plasma time.

### Physical and numerical validity
- mass, momentum and energy residuals using the exact solver's declared conservation convention;
- pressure positivity / realizability where the model requires it;
- characteristic speeds and hyperbolicity when mathematically applicable;
- finite-state, timestep and long-horizon stability;
- spurious growing-mode checks.

### Entropy diagnostics — carefully bounded
Preserve the exact definitions of fluid entropy and kinetic entropy separately. Compare their time histories, oscillatory structure, drift and first-deviation time. Do not impose a universal monotonicity rule or treat low-moment fluid entropy as identical to fine-grained kinetic entropy. Non-Maxwellian velocity-space structure is not fully represented by a finite set of moments, and discrete Vlasov calculations can show resolution-dependent entropy growth as phase-space filaments reach the numerical grid.

Entropy-only anomaly detection without a kinetic reference is a candidate diagnostic, not a safety gate. First calibrate its thresholds on independently scored, held-out kinetic runs; quantify false positives and false negatives across grid and timestep refinements. A no-reference result must be labelled UNVALIDATED_ENTROPY_SENTINEL until that calibration exists.

### Epistemic risk and abstention
- OOD AUROC and AUPRC;
- false-safe rate outside declared support;
- selective risk versus coverage;
- abstention and unscoreable rates;
- calibrated interval coverage / width;
- lead time between the first detectable failure precursor and actual rollout failure.

Include per-regime/per-intervention denominators and uncertainty intervals. No aggregate metric may hide a catastrophic subset.

### Portability
Report error and stability degradation under every timestep, grid, integrator and solver change; distinguish zero-shot transfer, fine-tuned adaptation, and unsupported configuration. Include extra data, compute and wall-time required for adaptation.

## 7. Proposed gates (not results)

1. Reproduce the exact paper baseline before claiming superiority.
2. No overall reliability pass when a mandatory physical-admissibility or solver-stability gate fails.
3. Publish all four envelopes and all failed, censored and unscoreable cases.
4. Freeze OOD thresholds using validation data; final test data remain sealed.
5. A portability claim requires zero-finetuning evaluation on a second numerical method. Fine-tuning is a distinct result.
6. Entropy-only monitoring cannot authorize operation or stand in for physical admissibility.
7. A candidate improvement must beat the strongest matched baseline on a predeclared Pareto comparison without worsening conservation, stability, realizability or false-safe rate beyond registered margins.

Targets such as percentage error reduction, rollout speed-up or stable horizon extension are not set until the exact reference implementation, solver tolerance and baseline variance are established.

## 8. Public benchmark and industry context

### Closure research
The most direct current comparators are the 2025 offline FNO, 2026 online/a-posteriori FNO, Hammett–Perkins analytic baseline and admitted sparse-regression methods. These belong in this same 1D electrostatic task only under matched inputs and solver conditions.

### Tokamak foundation models — adjacent, not head-to-head
GenToka (arXiv:2611.08153, 2026-11-05) extends TokaMark to 20 MAST tasks in six groups and reports MAST-measurement improvements over TokaMind on selected task groups. The reported results are MAST-specific. This is important evidence of multimodal, schema-conditioned tokamak modelling, but it is not a plasma-closure benchmark and must not be mixed into the Landau-damping leaderboard.

A distinct experimental-plasma benchmark should compare cross-machine transfer, diagnostic corruption, causal warning lead time, false alarms per discharge/plasma-hour, calibration and actuator/diagnostic observability. It must use compatible shot-level splits and operational alarm definitions, not the closure metrics above.

### What would constitute a real advance
The credible differentiator is not another lower training-set error. It is a closure with matched or better trajectory fidelity, a wider measured physical-admissibility envelope, lower false-safe risk and calibrated abstention under OOD shifts, demonstrable zero-shot portability to a second solver/integrator, and lower total cost per stable, qualified simulated plasma time.

Only the final point can establish a practical efficiency advantage. Compare end-to-end runtime, memory, accelerator/CPU hardware, resolution, accuracy tolerance, stable horizon and all preprocessing/training costs against the strongest admissible kinetic and fluid baselines.

## 9. Data admission and reproducibility

### 2025 PNAS Vlasov data
Zenodo record [10.5281/zenodo.14915239](https://zenodo.org/records/14915239) lists two MAT files of approximately 41 MB each, but its visible record page reports the same MD5 for both names. Treat them as unresolved until record/version metadata and file bytes are lawfully acquired, SHA-256 is recomputed independently for each file, MAT headers/variables/dimensions/contents are inspected, and any byte identity or metadata error is documented with upstream clarification.

The visible record page has no displayed license value. Public availability is not a redistribution grant; no raw data should be mirrored pending rights review.

### 2026 online FNO artifacts
Zenodo [10.5281/zenodo.22800112](https://zenodo.org/records/22800112) is the code/data record identified by the publisher. Its version-specific artifact manifest, file hashes and record-level license must be captured before reproduction.

### Magnetic reconnection
Zenodo [10.5281/zenodo.15498576](https://zenodo.org/records/15498576) supplies OSIRIS input decks and Section 2 diagnostics (~3.6 GB). It does not include the approximately 500 GB Section 1 diagnostics described by the authors. The visible record page has no displayed license value. This is a separate Lorentz/Galilean symmetry PIC lane, not a Landau-damping closure substitute.

For every admitted run, pin the source record/version, file digests, code commit, runtime/environment, numerical settings, initial conditions, transformations, split manifest and solver-run receipt. Use the existing Symtropy #1685 receipt boundary; do not add a duplicate ledger.

## 10. Evidence status

| Item | Status |
|---|---|
| Research protocol and task manifest | Authored in this branch |
| 2025 PNAS files, duplicate MD5 | Observed on public record; unresolved |
| 2025 dataset license | Not displayed on visible record; unresolved |
| 2026 online-FNO artifact license / hashes | Not audited |
| Magnetic-reconnection data rights | Not audited |
| Rust evaluator / entropy implementation | Not implemented |
| Paper reproduction | Not performed |
| Plasma closure training / solver runs | Not performed |
| Physical or experimental validation | Not performed |

This is a specification increment. It does not claim that a Symthaea closure exists, beats SOTA, or has been validated on experimental plasma.
