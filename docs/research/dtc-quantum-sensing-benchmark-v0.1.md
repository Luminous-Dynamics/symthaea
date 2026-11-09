# Discrete time-crystal quantum sensing benchmark v0.1

Date: 2026-11-08  
Status: benchmark specification; no experimental reproduction or sensing-performance claim  
Parent: [DTC-SENSE-001 #11128](https://github.com/Luminous-Dynamics/symthaea/issues/11128)  
Programme: [CTGP #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Motivation and current frontier

The February 23 2026 Nature Physics paper, *Sensing with discrete time crystals* (Moon et al., DOI 10.1038/s41567-025-03163-6), demonstrates frequency-selective detection of AC magnetic fields in dipolar-coupled 13C nuclear spins in diamond. The published system operates over 0.5–50 kHz and reports resonant linewidth below 70 mHz. The AC field at resonance enhances the lifetime of the prethermal DTC order parameter; the linewidth is then set by the extended lifetime. The experiment reports an extension of the order-parameter lifetime by up to about three orders of magnitude in selected settings and sustained spin flips beyond 20 seconds.

This is a different sensor mechanism from simply detecting a half-frequency spectral peak. It exploits a resonant, interaction-assisted stabilization channel in a two-tone Floquet protocol and quasi-continuous rotating-frame readout. Strong inter-spin interactions are part of the sensing resource.

A complementary May 13 2026 Physical Review B study (Ghosh, Das and Mukherjee, DOI 10.1103/hym6-j655) explores DTC-transition-assisted field-strength sensing in a periodically driven Lipkin–Meshkov–Glick model using QFI, finite-size scaling, time-averaged inverse participation ratios and mean-field analysis.

Separately, Switzer et al. demonstrated a two-dimensional anisotropic-Heisenberg DTC on IBM hardware (Nature Communications 17, 605, 2026; DOI 10.1038/s41467-025-67787-1), while Shinjo et al. demonstrated a clean two-dimensional DTC and an incommensurately modulated response on a 133-qubit IBM Heron processor for up to 100 Floquet cycles (npj Quantum Information 12, 41, 2026; DOI 10.1038/s41534-026-01193-3).

These experiments establish the current comparison targets. They do not mean that a finite exact state-vector simulator reproduces the diamond sensor or large hardware experiments.

## Paradigm shift

Optimize **information per total experimental cost**, not a visually sharp peak or a large order-parameter amplitude.

Total cost includes state preparation, polarization/initialization, drive sequence, acquisition, readout, cooling/reset/reinitialization, calibration and failed-run fraction. A longer-lived response is useful only if it produces more calibrated information about the parameter of interest under a matched resource budget.

Keep the following quantities distinct:

- **Signal response:** amplitude, phase, lifetime and spectral lineshape.
- **Sensor performance:** bias, variance, sensitivity, bandwidth and information per unit time.
- **Quantum upper bound:** quantum Fisher information for the declared state family and parameter generator.
- **Realized readout information:** classical Fisher information for the actual measurement distribution.
- **Experimental qualification:** independently calibrated measurement with uncertainty and reproducible raw data.

Do not promote a QFI upper bound into an achieved sensitivity result.

## Model hierarchy

### Level 0 — analytic one- and two-spin limits

Start with an independently derived small system where a time-dependent field is introduced through an explicit Hamiltonian term, for example:

    H_sig(t) = g * B_ac * cos(2*pi*f*t + phi) * sum_i Z_i

State the sign, rotating frame, units, coupling constant, pulse ordering and whether the signal is applied during the interaction step or continuously. Derive a closed-form single-spin response and a two-spin interacting reference where tractable.

### Level 1 — exact closed finite graph

Use the existing finite kicked-Ising exact state vector as an oracle for small N. Add the time-dependent signal coupling in a distinct module or explicit drive object so existing static-field semantics cannot change silently. Keep arbitrary initial-state support and report state norm error at each period.

This level is a numerical toy model. It is not a microscopic reproduction of randomly positioned dipolar-coupled 13C spins in diamond.

### Level 2 — controlled open-system model

Add dephasing, relaxation, control error, inhomogeneity, readout noise, sensor loss and common-mode noise as separately declared channels. State whether the master equation is Markovian and whether the implementation is exact, Trotterized or a reduced-order approximation.

Check complete positivity and trace preservation for every finite-step channel. Do not apply a phenomenological amplitude-decay curve and label it a microscopic Lindblad derivation.

### Level 3 — diamond-mechanism surrogate

Represent dipolar interactions, two-tone protocol and rotating-frame readout using normalized parameters, then validate each mechanism independently against published descriptions and raw traces where accessible. Keep unavailable material unavailable; do not infer digitized raw values from an abstract or figure caption.

### Level 4 — published/experimental data bridge

A published-data comparison requires:
- exact source/version and license;
- units and calibration chain;
- raw or digitized data, including digitization uncertainty;
- pulse timing and acquisition windows;
- sample and field conditions;
- readout and fitting method;
- data exclusion rules and all failed runs.

Synthetic Level 0–2 passes are not field validation.

## Measurands and estimators

For each field amplitude B, frequency f and phase phi, estimate:

- complex response amplitude and phase;
- decay envelope and lifetime with uncertainty;
- resonance centre and FWHM/linewidth using a declared fit model;
- bias and variance of B and f;
- Allan deviation versus averaging time where time series permit it;
- SNR and bandwidth;
- classical Fisher information for declared outcomes;
- quantum Fisher information as a separately labelled upper bound;
- total experiment duration and duty cycle;
- sensitivity normalized by square-root bandwidth and total experiment time.

Compare the fit against at least a Lorentzian and a physically motivated lineshape. Report model residuals, parameter covariance, fit non-identifiability and alternate model selection. Do not assume a linewidth is exactly the inverse lifetime unless that relation is derived for the specific signal/envelope and measurement convention.

### Fisher information

For measurement outcomes y with probability p(y|B), calculate

    F_C(B) = sum_y p(y|B) * [d_B log p(y|B)]^2

or the integral analogue for continuous outcomes. Validate derivatives against finite differences and exact small-system calculations.

For a pure-state model, calculate QFI from the declared state derivative (or an equivalent generator formula only where its assumptions hold). For mixed/open states, use a correct mixed-state QFI definition or omit QFI. Always compare F_C against F_Q for the actual measurement basis.

## Baselines

All methods use the same Hamiltonian family, state preparation budget, observation time, measurement noise and calibration knowledge.

1. Noninteracting perfect-pi pulse control.
2. Noninteracting detuned-pulse control.
3. Interacting DTC-like protocol with no AC signal.
4. Off-resonant AC signal.
5. Resonant AC signal.
6. Conventional spin echo / dynamical-decoupling magnetometry with matched resources.
7. Matched coherent integration with no claimed DTC stabilization.
8. Exact small-N state vector versus converged Trotter / tensor-network approximations.
9. Oracle readout versus finite-shot/noisy readout.

A method may have a narrower linewidth but worse sensitivity after accounting for duty cycle, signal amplitude, readout noise or initialization costs. Report the complete Pareto surface rather than one headline metric.

## Perturbations and negative controls

- zero AC amplitude;
- detuning sweep that crosses and leaves resonance;
- no-interaction control;
- pulse-angle and pulse-timing errors;
- on-site and coupling disorder;
- initial-state ensemble (all-up/down, defects, random product, selected superpositions);
- dephasing, relaxation and correlated noise;
- finite-shot readout and sensor loss;
- calibration drift and frequency reference uncertainty;
- changed acquisition-window length;
- frequency/phase aliasing and spectral leakage.

The no-interaction perfect-pi control may show period doubling; it must not be counted as interaction-assisted sensing.

## Numerical integrity

- exact one-/two-spin oracle agreement;
- unitary norm preservation in the closed system;
- CPTP, trace and positivity checks in open-system dynamics;
- Trotter convergence on exact small-system comparators;
- stable resonance estimates under longer acquisition windows and independent window choices;
- distinguish FFT grid spacing from observation-limited resolution;
- leakage controls for frequencies near Nyquist;
- seeded reproducible ensembles, immutable source/configuration digests;
- independent code path for the analytic reference.

## Proposed acceptance gates

- Correct zero-signal and noninteracting controls.
- Analytic single-/two-spin response matches the independent closed-form result within derived numerical tolerance.
- The estimated resonance frequency is unbiased within the stated confidence interval on held-out seeds and signal phases.
- Reported linewidth uncertainty has nominal coverage on held-out synthetic data.
- A resonant lifetime extension is not inferred from selection of an analysis window or a post-hoc decay model.
- Sensitivity improvement is computed for the same total experiment budget, including reset/readout overhead.
- Classical FI matches an independent probability-distribution calculation; QFI is kept separate.
- Any claim of interaction-assisted advantage survives the predeclared perturbation grid and initial-state ensemble.
- A hardware/SOTA comparison matches the published observable, protocol, frequency band, readout and resource budget.

Thresholds must be frozen after pilot and null-distribution analysis, before qualification cases are revealed.

## Implementation order

1. Independent analytic oracle and model convention.
2. Time-dependent signal operator in a separate drive/field object.
3. One-/two-spin tests with exact solutions.
4. Readout probability and classical-FI primitive.
5. Frequency sweep, decay and linewidth fit with observation-window uncertainty.
6. Baseline and negative-control matrix.
7. Matched-resource sensitivity report.
8. Open-system channels and positivity checks.
9. 2D anisotropic-Heisenberg and fragmented-model sweeps.
10. Tensor-network comparison with bond-dimension/truncation convergence.
11. Published-data bridge when provenance and raw data are genuinely available.


## Joint amplitude–phase metrology and clock-offset nuisance — 2026-11-09

The initial finite-Floquet sensor calculation treated the probe's initial phase as known and estimated only the amplitude derivative. That is a conditional metrology problem. A field sensor with clock offset or unknown signal phase must instead treat phase/time as a nuisance parameter.

The new child workstream [DTC-CLOCK-001 #11261](https://github.com/Luminous-Dynamics/symthaea/issues/11261) adds a local two-parameter model:

`theta = [A, phi]`

where A is the dimensionless sampled probe-kick amplitude and phi is the initial phase at the first probe midpoint. At known probe frequency f, phase uncertainty is locally equivalent to a clock offset in drive-period units:

`delta_t = -delta_phi / (2 pi f)`

The conversion is undefined at f=0 because a static kick has no clock-offset interpretation. The phase remains a parameter through A cos(phi): at nonzero nominal phase, amplitude and phase are locally confounded, which is tested as a singular joint-Fisher control. At nominal phase zero, the first-order phase derivative vanishes, so the local FI alone does not reveal the global ambiguity.

The implementation now runs matched nominal, +/-amplitude and +/-phase state-vector trajectories. It reports:
- the 2x2 classical FI matrix for the exact N+1 total-down-spin readout outcomes;
- the 2x2 pure-state QFIM for the same amplitude/phase parameterization;
- amplitude information after eliminating phase as a nuisance using the Schur complement;
- joint local CRLB covariance matrices when the parameter pair is identifiable;
- the phase-to-clock-offset conversion in drive periods per radian (undefined for a static f=0 probe);
- a positive-semidefinite matrix comparison of CFI against QFIM, separate from the existing scalar diagonal check.

The known-phase scalar values remain available as the amplitude diagonal, but must be described as conditional. Do not use them as the amplitude sensitivity bound when phase is unknown. A singular joint matrix must not be hidden by falling back to the known-phase result.

A March 9, 2026 Physical Review A paper derives two-parameter QFI and Cramér–Rao bounds for generic quantum sensors while treating time as an uncertain nuisance parameter (DOI: 10.1103/nn67-9tph). This is a metrology-method comparator, not evidence about DTC sensors specifically. It reinforces the need to include clock uncertainty in the sensing model rather than assuming perfect timing.

An independent small-system Python reference for one representative six-spin configuration (J=0.5, pulse angle pi+0.08, 96 periods, amplitude 0.04, f=0.25 cycles/period, phase 0) gives:
- known-phase classical amplitude FI approximately 2.44e-2 per shot;
- joint classical nuisance-adjusted amplitude FI approximately 1.29e-4 per shot;
- known-phase pure-state QFI approximately 8.96e-1 per shot;
- joint pure-state nuisance-adjusted amplitude QFI approximately 8.83e-1 per shot.

This is one finite-system synthetic setting, not a universal ratio, experiment reproduction or sensitivity result. It illustrates that the actual collective readout and phase nuisance can change the inferential value substantially; the QFI remains an upper bound, not achieved information.

Acceptance requires independent analytic QFIM/CFI oracles, PSD checks, finite-difference convergence, phase-origin invariance, both zero-frequency nominal-phase and nonzero-phase confounding controls, and a matched known-phase versus unknown-phase comparison. The full experiment budget still needs preparation, readout, reset, dead time, time jitter, calibration drift and a conventional Ramsey/lock-in baseline before any sensitivity comparison.


The 2x2 joint covariance inversion now has an explicit maximum condition-number gate of `1e10`. Matrices that are mathematically positive but numerically too ill-conditioned are reported without a joint CRLB covariance; they are not inverted into a misleading finite precision. The zero-frequency, nonzero-phase test specifically checks the static-kick amplitude/phase confounding. The clock-offset conversion is a parameter mapping, not evidence that the simulator has an independent clock state.

## Claim ceiling

A numerical pass validates only a declared model and estimator. It does not reproduce the diamond experiment, show that Symthaea has achieved a physical DTC sensor, exceed current experimental sensitivity, or connect temporal order to UAP propulsion. A valid negative result is that no information advantage survives matched resources and noise.
