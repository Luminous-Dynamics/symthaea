# CPG Numerical Integrity and Reproduction Audit V1

**Status:** source-level audit and receipt contract; no simulator reproduction or biological result is claimed.

**Target reference:** Pugliese et al., *Connectome simulations identify a central pattern generator for fly walking*, Nature, version of record 2026-10-28 ([DOI](https://doi.org/10.1038/s41586-026-11195-y)). The public implementation is [smpuglie/Pugliese_2026](https://github.com/smpuglie/Pugliese_2026). The exact source ref inspected here is `10e7661bf414ba7b4c2edf795cd36d0f878c17c0`.

## 1. Why this audit exists

The biological claim and its numerical implementation are different things. The published paper reports a connectome-derived three-interneuron rhythm-generating circuit across four Drosophila ventral nerve cord connectome datasets, and experimental optogenetic support for the DNb08 pathway. Those are the scientific results we aim to independently reproduce and test.

A simulator can produce plausible-looking output after a numerical failure, silently alter that output, or claim a perturbation that its equations never apply. The benchmark therefore needs an explicit numerical-integrity receipt for every in-silico result.

This audit does **not** infer that the paper's conclusions are false. It identifies source-level behaviors that deserve targeted tests and disallows treating sanitized output as an unqualified successful trajectory.

## 2. Static source findings

The immutable file inspected is [`src/simulation/vnc_sim.py` at the pinned ref](https://github.com/smpuglie/Pugliese_2026/blob/10e7661bf414ba7b4c2edf795cd36d0f878c17c0/src/simulation/vnc_sim.py). The observed source locations are:

| Lines | Source behavior | Reproduction risk / required test |
|---|---|---|
| 34–40 | The rate-equation function unpacks `noise_stdv`, but the input-noise addition is commented out at line 39 | A non-zero configured input-noise standard deviation may not perturb the state equation. Prove requested vs applied noise with sample digests and realized statistics. |
| 64–70 | A helper detects NaN/Inf state values | Detection exists as code, but existence of a helper does not mean it is active in a simulation. |
| 102, 109 | Event construction and `event=nan_inf_event` are commented out | The intended early-stop event is not attached to this solver call. |
| 104–110 | `diffeqsolve` is called with `throw=False` | A non-throwing call must still record and check the returned solver status/result. |
| 114–120 | Saved output is transposed, NaN/Inf values are replaced with zero, and values are clipped to [0, 1000] Hz | Silent sanitization and clipping can transform failed trajectories before downstream oscillation scoring. Preserve raw states and fail closed on non-finite values. |
| 1807–1810 | Noise settings are read from the simulation config and passed into the run | Configuration presence alone is not evidence that the corresponding noise was injected. |

These are source facts about this pinned implementation, not verified runtime observations. The follow-on reproduction must inspect the exact returned solver status and raw states on the actual execution head.

### Version and dataset pinning

The Nature version-of-record page lists the four connectome sources and points to its simulation output and Anipose data on [Zenodo record 22260923](https://doi.org/10.5281/zenodo.22260923). The current public repository README still describes the accompanying preprint and points to [Zenodo record 22260924](https://zenodo.org/records/22260924). These may be different deposits/revisions. Resolve their contents, checksums, dates and relationship before treating them as interchangeable.

Pin all of:
- paper version and DOI;
- code commit and per-file SHA-256;
- exact MANC/FANC/male CNS/BANC connectome release or CAVE materialization;
- simulation data release and file hashes;
- model equation/parameter manifest and resolved config;
- dependency lock, solver/library versions, hardware/backend/dtype;
- random seed and PRNG identity.

## 3. Fail-closed numerical contract

The receipt is `CPG_NUMERICAL_RUN_RECEIPT_V1`; the strict Draft 2020-12 schema is `CPG_NUMERICAL_RUN_RECEIPT_V1.schema.json`. The example `CPG_NUMERICAL_RUN_RECEIPT_EXAMPLE_V1.json` is intentionally incomplete.

A complete receipt requires:

1. **Solver completion:** explicit success status, positive accepted steps and function evaluations, expected saved timepoint count, and exact state dimensions.
2. **Raw-state verification:** independently scan raw solver states before any downstream transformation. The scan must be recorded and the non-finite count must equal zero.
3. **No hidden repair:** post-solver NaN/Inf replacement count, clipping count and other post-solver transformation count must all be zero. If a physical state constraint belongs in the differential equation, declare it in the equation manifest; do not use an undocumented post-hoc repair.
4. **Perturbation truth:** bind the exact applied stimulus waveform and protocol. For input or weight noise, a non-zero requested mean/stddev requires an applied flag, sample digest and realized statistics. A config value alone never qualifies as applied perturbation.
5. **Source and environment identity:** immutable code ref, exact file SHA-256, solver/library version, dependency-lock digest, resolved config and environment digest.
6. **Raw output identity:** raw state digest and raw output manifest digest. Do not hash only a sanitized or clipped array.
7. **Cost reporting:** wall time, peak memory and energy where measurable. Missing energy must remain explicitly null rather than estimated from a generic hardware label.

A failed or numerically invalid run is a result to report, not a value to repair into the valid-run population. Do not convert missing, failed or non-finite results to zero oscillation score.

## 4. Rust semantic verifier

The companion Rust binary checks cross-field conditions that JSON Schema cannot fully establish:
- claim ceiling matches evidence tier;
- complete status implies solver success and positive, consistent output shape/counts;
- raw-state finite scan was performed and observed zero non-finite values;
- no post-solver sanitization, clipping or unrecorded transformation occurred;
- non-zero requested noise cannot be marked unapplied;
- applied noise has a digest and positive finite realized standard deviation;
- exact code, solver, dependency, parameter, config, environment and output digests are present;
- in-silico receipts carry an immutable data manifest and documented permitted data state.

A syntactically valid receipt is not a numerical pass. A numerically valid receipt is not evidence that the model equations or biological interpretation are correct.

## 5. Required benchmark: solver and perturbation robustness

Use a frozen corpus of source-bound small/medium fixtures before running any full connectome workloads.

### A. Known-answer ODE fixtures

- Stable linear system with analytic solution.
- Damped oscillator with known frequency/decay.
- Nonlinear saturating rate equation with known reference integration.
- Unstable/divergent case that must fail explicitly.
- Injected NaN/Inf state that must be rejected.
- Maximum-step exhaustion that must be reported as incomplete/failed.
- Empty, truncated and mis-shaped output arrays.

Compare the production solver against an independent reference integrator. Use a preregistered convergence sweep over `rtol`, `atol`, precision and initial/max step. Compare state trajectory error, frequency, phase, amplitude, limit-cycle presence and final solver status—not only a single aggregate waveform distance.

### B. Perturbation application controls

- Requested zero input noise: no sample is injected and the state equation remains unchanged.
- Requested non-zero input noise: non-zero sample digest and realized statistics must be recorded; same seed/config reproduces the same sample.
- Requested non-zero weight noise: weight-matrix digest changes as declared; seed and distribution are pinned.
- Stimulus waveform: applied input hash and integrated input energy match the declared waveform.
- Negative control: a deliberately no-op noise implementation must fail the requested-vs-applied contract.

### C. Connectome and oscillator robustness

For each frozen connectome:
- compare baseline and exact declared perturbations;
- compare degree/type-preserving and spatially constrained rewires;
- evaluate selected neuron/edge ablations and rescue conditions;
- vary declared biological parameters and connectome uncertainty separately from numerical solver tolerances;
- test whether rhythmic/non-rhythmic and cross-leg coordination conclusions survive numerical refinement;
- retain each connectome dataset as a separate stratum.

### D. Metrics

Report per run and per stratum:
- solver completion/failure rate, max-step exhaustion, rejected/accepted steps, function evaluations;
- NaN/Inf count, sanitization/clipping count (must be zero for qualification);
- raw-state digest, state dimension/timepoint completeness;
- trajectory NRMSE and independent-reference error;
- estimated frequency and phase error, oscillation detection agreement;
- inter-leg phase-coordination error;
- perturbation application fidelity and input energy;
- wall-time p50/p95, peak memory and energy per simulated second.

Use tolerance sweeps to produce a **numerical robustness envelope**. Do not tune tolerances after seeing which setting preserves the desired biological conclusion.

## 6. Paradigm shifts beyond current SOTA

### 6.1 From one solver run to a numerical robustness envelope

A biological interpretation should not depend on one arbitrary integrator configuration. Treat solver/backend/precision/tolerance as controlled nuisance axes. Require the circuit-level conclusion to remain stable across a preregistered qualified region of this space, or report it as numerically sensitive.

### 6.2 From nominal stimulus to verified applied intervention

The intervention is the actual waveform delivered to the model, not a configuration field. Record the resolved sample array or immutable sample digest, applied channel, time support, total power/energy, and realized stochastic statistics.

### 6.3 From single best connectome to reconstruction-ensemble prediction

Run the same model family and stimulus across MANC, FANC, male CNS and BANC with their own exact source revisions. Preserve reconstruction uncertainty and do not pool away dataset-specific failures.

### 6.4 From post-hoc pruning to prospective model discrimination

The next experiment should be selected before opening held-out outcomes. Compare EIG (hypothesis discrimination), EVSI (decision usefulness) and Fisher-information/parameter-identification objectives separately. The 2026 eLife work on optimal perturbation design supports designing inputs to expose latent dynamics, while the 2026 Nature Reviews framework motivates selecting inputs on which candidate models disagree. Neither substitutes for prospective validation.

## 7. Comparison to research and industry practice

The current leading connectome-driven locomotor reference combines detailed VNC connectivity, simulations across four connectome datasets and targeted optogenetic evidence. NeuroMechFly v2 is a complementary embodied sensorimotor simulation framework with multimodal sensory feedback, complex terrains and learned navigation controllers; it is not the same benchmark task.

Symthaea should aim to add a qualification layer missing from a plain simulation leaderboard: exact provenance, raw numerical integrity, verified input delivery, solver robustness envelopes, model-disagreement experiment design, and held-out measured intervention prediction.

Do not claim superiority until the same connectome versions, observations, perturbation protocols, tasks and compute budgets have been reproduced.

## 8. Evidence ceiling

- A passing synthetic receipt validates the schema/verifier only.
- A passing in-silico receipt validates execution integrity of one declared simulator run only.
- A simulation result remains a model counterfactual; it does not establish measured causal effect.
- Only held-out, independently measured interventions can support the corresponding empirical prediction claim.

**No benchmark outcome, biological reproduction or SOTA result is claimed by this audit document.**
