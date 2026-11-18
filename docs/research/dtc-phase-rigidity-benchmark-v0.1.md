# DTC phase-rigidity and metastability benchmark v0.1

Date: 2026-11-07  
Status: benchmark contract / no phase qualification result claimed  
Parent: [DTC-PHASE-001 #11033](https://github.com/Luminous-Dynamics/symthaea/issues/11033)  
Programme: [CTGP #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Scientific target

Distinguish a period-two signal from a discrete time-crystalline phase. A subharmonic Fourier peak is a necessary diagnostic in many driven spin models, but it is not sufficient: an exactly pi-pulsed noninteracting system can alternate trivially, and finite systems can exhibit initial-state-specific beating or long-lived metastability.

The target observable is not just a large finite-window amplitude. The benchmark must test frequency rigidity, phase coherence, perturbation robustness, initial-state dependence, lifetime scaling and numerical convergence as distinct claims.

## Current implementation boundary

The existing exact kicked-Ising module performs finite-dimensional complex state-vector evolution and is capped at 20 spins. The anisotropic-Heisenberg module uses a symmetric second-order Suzuki–Trotter decomposition over noncommuting interaction groups. It is exact only in commuting limits; elsewhere the Trotter slice count is a numerical approximation parameter that must be converged.

The independent v0.2 reference sweeps show a useful distinction:
- selected symmetry-broken all-up product states can retain a strong finite-window subharmonic response at moderate ZZ coupling;
- zero-interaction detuned-pulse controls lose phase coherence;
- a single spin-flipped initial state can lose the same response;
- stronger coupling is not monotonically better;
- the 2D response depends on size, topology and pulse detuning.

These are finite-system signal results, not evidence of thermodynamic-limit symmetry breaking.

## Phase classification ladder

The evaluator must emit separate levels:

1. `NoSubharmonicSignal`
2. `SubharmonicSignalDetected`
3. `FiniteWindowFrequencyLocked`
4. `RobustFiniteSizeRegime`
5. `DTCPhaseNotEstablished` or, only when supported by a full independent qualification, a suitably scoped phase claim.

No lower-level result may be silently promoted to a higher one. In particular, a finite-window FFT score cannot issue a phase label.

## Required controls

- Noninteracting perfect-pi-pulse control: period-two behavior is trivial and must not count as interaction-stabilized DTC.
- Noninteracting detuned-pulse control: tests whether the classifier detects loss of rigidity.
- Interacting generic-ergodic control.
- Metastable/scarred finite-time response control.
- Initial-state controls: all-up, all-down, random product, domain-wall, staggered and supported symmetric superpositions.
- Coupling, anisotropy, longitudinal-field and graph perturbation sweeps.
- Spin relabelling / edge-order invariance.
- Independent exact state-vector reference at small sizes.
- Heisenberg Trotter refinement and comparison against an exact small-graph comparator.

## Metrics (never collapsed into one score)

### Signal and spectral response
- demodulated `q_n=(-1)^n m_z(n)` amplitude and uncertainty;
- subharmonic peak prominence, linewidth and sideband structure;
- stroboscopic autocorrelation and connected correlation;
- phase coherence across independent initial states and perturbations.

### Frequency rigidity
- estimated response frequency versus drive detuning;
- width of the interval locked to half the drive frequency;
- amplitude and phase coherence over that interval;
- no per-sample retuning of the drive to preserve the response.

### Lifetime and finite-size scaling
- lifetime to a preregistered rolling-envelope threshold;
- censored lifetime intervals and ensemble variation;
- held-out-size prediction for candidate lifetime scaling laws;
- explicit comparison among exponential, power-law, saturating and unresolved models.

### Many-body diagnostics
- entanglement entropy and bipartite mutual information;
- participation ratios in a stated basis;
- quasienergy-pairing checks for small systems;
- fragmentation-sector and graph-connectivity diagnostics where applicable.

### Numerical and operational integrity
- norm drift / unitarity;
- Trotter convergence;
- exact-oracle discrepancy;
- wall time, peak memory and system-size scaling.

## Executable finite-sweep frequency diagnostic

The crate now contains `symthaea_fractal_time_lab::frequency_locking`:
- `estimate_subharmonic_peak` estimates a spectral peak in a declared band after mean removal and Hann windowing.
- `assess_frequency_locking` evaluates an ordered pulse-detuning sweep and reports per-point frequency, FFT-grid spacing, observation-limited frequency scale, amplitude, spectral prominence and demodulated sign coherence.
- The sweep report exposes locked-point count, longest contiguous locked run and the observed sampled detuning span.
- The measurement result is deliberately called **frequency locking**, not a DTC phase result.

The implementation reports both FFT bin spacing (which zero-padding reduces) and the observation-limited frequency scale (approximately inverse observation-window duration). Zero padding improves peak-grid interpolation; it does not add observation time or improve the fundamental frequency resolution.

A perfect-pi-pulse noninteracting trace can satisfy the finite-sweep frequency-locking threshold. This is an intentional negative-control result: the frequency metric measures frequency rigidity, while the phase evaluator must still require interaction stabilization, initial-state/perturbation robustness, finite-size/lifetime analysis and separation from the trivial pulse control.

The UAP Observatory's new `observability::analyze_jacobian` follows the same epistemic rule in another domain: return data-only rank/null directions before priors are considered. Neither module is itself a physical discovery claim.

## Executable AC-probe information benchmark

The fractal-time crate now includes a separate finite-system sensing path:

- `exact_floquet::KickedIsingFloquet::step_with_uniform_ac_probe` applies a declared sampled longitudinal probe kick coupled to the total (Z) magnetization.
- `dtc_sensing::evaluate_dtc_probe_sensing` compares a zero-probe baseline with a nominal probe and symmetric amplitude perturbations, all with the same initial state, Hamiltonian, pulse, frequency, duration and measurement.
- The final readout is the exact (N+1)-outcome distribution of total down-spin count, obtained by aggregating computational-basis probabilities by Hamming weight.
- The local classical Fisher information is calculated from the full distribution using a centered finite difference in probe-kick amplitude:
  
  (F(A)=\sum_k [\partial_A p_k(A)]^2/p_k(A)).

- The pure-state quantum Fisher information is also estimated from the centered state-vector derivative:
  
  \(F_Q(A)=4(\langle\partial_A\psi|\partial_A\psi\rangle-|\langle\psi|\partial_A\psi\rangle|^2)\).

- The report shows the CFI/QFI ratio and a numerical consistency flag. QFI is an upper bound for measurements on the declared pure state; it is not the achieved total-Z readout information.
- Both information measures are reported per final readout and divided by simulated Floquet periods, alongside finite-window subharmonic order and an explicitly right-censored lifetime estimate.

This is a closed-system, small-N, sampled-kick model. It is **not** a reproduction of the 13C diamond device, which uses dipolar-coupled spins, a two-tone Floquet protocol, quasi-continuous NMR readout, and an AC-field-mediated lifetime-extension mechanism. The experiment reports frequency-selective response with linewidth below 70 mHz and an interaction-assisted lifetime extension up to three orders of magnitude; those device results are experimental references, not validation targets that the toy model can claim to match. (Moon et al., *Nature Physics* 22, 367–373 (2026), DOI: 10.1038/s41567-025-03163-6.)

The intended initial benchmark is a matched-budget scan over coupling and AC frequency, including:
- (J=0) noninteracting control;
- interacting nearest-neighbour kicked-Ising controls;
- fixed spin count, pulse detuning, initial product state, probe amplitudes, total Floquet periods and readout;
- finite-difference step convergence for the amplitude derivative;
- probability normalization, state-vector norm, and singular readout-support checks;
- Fisher information per shot and per drive period, reported separately from the lifetime and subharmonic response.

A strong result would be a reproducible coupling/frequency region in which the interaction changes the calibrated readout information under matched resource constraints, with independent reproduction and robustness to initial-state family, finite-difference step, readout observable and noise. A large Fisher value at one parameter point, a narrow zero-padded FFT bin or a long-lived finite trace is not sufficient to claim a time-crystal sensor or a quantum advantage.

## Baselines

1. finite-window FFT maximum only;
2. noninteracting perfect-pi pulse;
3. noninteracting detuned pulse;
4. generic interacting ergodic Floquet model;
5. finite-time metastable/scarred response;
6. exact diagonalization/state vector on small systems;
7. converged Trotterized Heisenberg evolution;
8. later MPS/TEBD with bond-dimension and truncation convergence.

All methods share the same Hamiltonian, initial-state corpus, observables, parameter grid and compute accounting.

## Prospective qualification gates

- Zero phase false positives on perfect-pulse noninteracting and declared metastable negative controls.
- Frequency locking must persist over a finite predeclared perturbation interval, not one selected point.
- Initial-state failures remain visible; no post-hoc removal of inconvenient states.
- Lifetime scaling must predict held-out sizes and report uncertainty/model ambiguity.
- Exact Ising oracle agreement within declared floating-point tolerances.
- Heisenberg Trotter observables converge on a small-system comparator.
- Every report binds the source SHA, Hamiltonian/graph digest, Floquet convention, initial-state seed manifest, precision, Trotter slices and resource budget.
- Numerical thresholds are frozen after pilot/null-distribution analysis and before qualification labels are revealed.

## External frontier and comparison

The relevant public targets include the Jan 2026 2D anisotropic-Heisenberg DTC experiment (NIST research highlight: https://www.nist.gov/news-events/news/2026/01/researchers-demonstrate-two-dimensional-time-crystal), disorder-free fragmentation work (Phys. Rev. B 113, 214303, https://doi.org/10.1103/PhysRevB.113.214303), and Oct 2026 scarred-DTC work (Phys. Rev. B 114, 174302, https://doi.org/10.1103/PhysRevB.114.174302).

Symthaea should not claim to compete with hardware-scale demonstrations on qubit count. Its near-term competitive target is independent small-system correctness, phase-vs-signal discrimination, explicit metastability/initial-state controls, finite-size prediction and uncertainty-qualified Trotter/MPS comparisons.

## Claim ceiling

A benchmark pass validates only the declared finite-system numerical and classification contracts. It does not establish thermodynamic-limit spontaneous symmetry breaking, an experimental DTC phase, quantum information storage, biological time crystals or a UAP mechanism. The correct result may be: **subharmonic signal detected; DTC phase not established**.
