# Marine rotation-synchronous modulation audit

A deterministic order-synchronous envelope audit for carrier-band linear envelope power indexed by measured unwrapped shaft phase. This is a transparent order-domain estimator, not a reproduction of the spectral-coherence-weighted synchronous modulation spectrum (SWSMS) or a general cyclic spectral estimator.

## Run

~~~sh
cargo test --manifest-path tools/marine-rotation-synchronous-modulation-audit/Cargo.toml
cargo run --manifest-path tools/marine-rotation-synchronous-modulation-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-rotation-synchronous-modulation-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence.

## Estimator

For each run and carrier band, observations are assigned to equal-angle shaft-phase bins. Every phase bin must be populated to the predeclared minimum count; no phase interpolation is performed. The phase-bin mean is computed in linear envelope power.

For shaft order k, the audit estimates peak modulation amplitude and phase by the equal-angle Fourier coefficient:

`C_k = (2/N) Σ_j P_j exp(-i k θ_j)`

`A_k = |C_k|`, `φ_k = arg(C_k)`, and modulation index `m_k = A_k / mean(P)`.

The sign convention means an input envelope `P(θ)=μ+A cos(kθ+φ)` yields the reported phase `φ`. Mean envelope power is the equal-angle mean of the phase-bin means. The output also declares whether each selected order is an integer multiple of blade-passing order (order divided by blade count).

Phase is explicitly unwrapped and accompanied by revolution index, phase uncertainty, timestamps and measured RPM. The phase/time slope between adjacent observations is checked against the two adjacent RPM readings and their expanded uncertainties plus a predeclared consistency fraction. Phase resets, duplicate/reversed phase, non-monotonic timestamps and inconsistent phase/time/RPM are invalid. Phase uncertainty beyond the declared limit or incomplete phase-circle coverage is INCOMPLETE; missing bins are never treated as zero.

## Interpretation limits

The metric quantifies shaft-phase-locked modulation of the declared band envelope. It does not identify the unique physical source: wake loading, blade geometry, cavitation, structural coupling, machinery or other periodic forcings can all create order-synchronous modulation. It is not a complete spectral-coherence-weighted synchronous modulation spectrum, not cyclostationary source separation, not full-scale noise transfer and not ecological evidence.

Cavitation annotation coverage and window-presence fraction are reported separately. Missing annotations remain null, not zero. Runs are kept separate; no sample-level confidence interval or independent-sample assumption is made.

## SOTA comparator and benchmark

Wen et al. (Ocean Engineering 2026, DOI 10.1016/j.oceaneng.2026.125989) propose a spectral-coherence-weighted synchronous modulation spectrum. Hong et al. (International Journal of Multiphase Flow 2026, DOI 10.1016/j.ijmultiphaseflow.2026.105814) use synchronized high-speed imaging and hydroacoustics to identify wake-induced periodic gating. Pereira et al. (Aerospace Science and Technology 2026, DOI 10.1016/j.ast.2026.112963) report a blind isolated-propeller comparison with better agreement for early BPF harmonics than for higher harmonics and very-high-frequency broadband sound.

This audit deliberately does not claim to beat or reproduce those methods. Its first benchmark is an analytic phase-locked envelope with known order amplitude and phase; the next is an encoder/shaker bench, followed by a cavitation-tunnel held-out regime/facility benchmark.
