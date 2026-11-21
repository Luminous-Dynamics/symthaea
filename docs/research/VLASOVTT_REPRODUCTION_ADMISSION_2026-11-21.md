# VlasovTT tensor-train Vlasov–Poisson comparator: source admission

**Date:** 2026-11-21  
**Tracking:** [symthaea #13104](https://github.com/Luminous-Dynamics/symthaea/issues/13104)  
**Related trajectory cohort:** [symthaea #13072](https://github.com/Luminous-Dynamics/symthaea/issues/13072)

## Purpose

Admit an open-source tensor-train / matrix-product-state (TT/MPS) 1D1V Vlasov–Poisson solver as a serious matched comparator to the Symthaea dense distribution-space reference and future particle/learned representations.

This is source admission and reproduction planning, not a benchmark result. The paper's reported results are not independently reproduced here.

## Source identity

- Paper: *Tensor Network Compression for Fully Spectral Vlasov–Poisson Simulation*, arXiv:2602.13092.
- Inspected repository: [erikasgrim/VlasovTT.jl](https://github.com/erikasgrim/VlasovTT.jl).
- Inspected repository HEAD on 2026-11-21: \`99aa077b6f5e4ae4dfc8330590e502267caf298b\`.
- Repository README documents two paths: compressed TT/MPS and dense reference.
- Repository declares MIT licensing; the repository's exact \`LICENSE\` file is part of source admission.
- Julia environment uses ITensors/ITensorMPS, TensorCrossInterpolation, Quantics, QuanticsGrids, QuanticsTCI, FFTW, and CUDA among its dependencies.

The currently inspected HEAD is an immutable reproduction target for this workstream, but it is **not yet proven to be the exact repository revision used to generate the paper figures**. The original paper-linked source revision/data must be reconciled before describing the run as an exact reproduction of the publication.

## Critical source-to-paper discrepancy discovered

The paper states its linear Landau initial condition uses \(A=0.01\), \(v_{\rm th}=1.0\), \(x\in[-2\pi,2\pi]\), and \(v\in[-6,6]\). It gives the expected linear electric-field amplitude damping rate as approximately \(-0.151\,\omega_{pe}\).

However, both inspected current repository scripts:
- \`examples/landau_damping.jl\`
- \`examples/reference_landau_damping.jl\`

construct the initial condition with \`A = 0.1\`, not \`A = 0.01\`.

This tenfold amplitude mismatch is a reproduction blocker. Do not silently report the default repository script as paper-faithful. The reproduction harness must explicitly override the amplitude to the publisher's paper value for the paper-faithful lane and preserve \(A=0.1\) as a separately labelled source-default stress/compatibility lane. Do not patch the external project merely to hide the discrepancy.

## Paper-reported configuration and results

The paper reports:
- Quantics bit depth \(R=10\) for x and v, hence \(1024\times1024\) points (about \(10^6\) phase-space values).
- Fixed timestep \(\Delta t=0.1\).
- TCI approximation tolerance \(\tau=10^{-8}\), defined on adaptively queried entries rather than a uniform full-domain guarantee.
- Spectral filter retains \(2^7\) discrete Fourier modes; filter sharpness \(\beta=2\).
- Maximum electric-field TT bond dimension \(\chi_E=32\).
- Landau damping on \(x\in[-2\pi,2\pi]\), \(v\in[-6,6]\), \(v_{\rm th}=1\), \(A=0.01\), with expected amplitude damping \(-0.151\,\omega_{pe}\).
- Two-stream instability on \(x\in[-\pi/3.06,\pi/3.06]\), \(v\in[-0.6,0.6]\), \(v_{\rm th}=0.01\), \(v_0=0.2\), \(A=0.01\), with expected linear growth \(0.354\,\omega_{pe}\).
- TT truncation cutoff \(\epsilon\) is varied; the paper's study includes \(10^{-7}\) through \(10^{-10}\).
- Paper-reported maximum TT ranks are about 15–45 for Landau damping and 120–150 for two-stream in the studied runs.
- On Landau damping, the paper reports relative energy deviation around \(10^{-3}\), with tighter truncation improving the final deviation toward \(4\times10^{-4}\).
- Paper-reported negative distribution artifacts grow under more aggressive TT truncation. The paper reports Landau minima changing from about \(-3\times10^{-4}\) to \(-2\times10^{-3}\) over the cited cutoff range, and more severe localized negative artifacts for the two-stream experiment.
- The TCI fit after electric acceleration is reported as the dominant per-step cost, taking approximately an order of magnitude more time than all TT–MPO contractions combined.

These observations belong in the paper-reported column until a pinned independent reproduction is complete.

## Current repository script defaults (separate from paper claims)

The inspected TT Landau example defaults to:
- \(dt=0.1\), \(T_{\rm final}=30\);
- \(R=10\), \(k_{\rm cut}=2^6\), \(\beta=2\);
- \(x\in[-2\pi,2\pi]\), \(v\in[-6,6]\);
- TCI tolerance \(10^{-8}\), maximum TT rank 500, field maximum rank 32, SVD cutoff \(10^{-10}\);
- double GPU precision and GPU-enabled execution by default;
- **Landau amplitude \(A=0.1\)** in the inspected script.

The dense reference example uses the same phase-space grid defaults and timestep, but sets TCI tolerance and truncation cutoff to zero, maximum TT ranks to \`nothing\`, \`use_gpu=false\`, \`alg="reference"\`, and \`k_cut=-1\`. It also currently uses **Landau amplitude \(A=0.1\)**.

The TT Landau script and dense reference script are not identical workloads merely because they share an example name. Freeze and report all method-specific numerical settings, while matching physical initial conditions and final time.

The current two-stream script uses \(A=10^{-2}\), \(v_0=0.2\), \(v_{\rm th}=0.01\), consistent with the paper's stated initial-condition parameters.

## Required reproduction matrix

### A. Landau damping

Run three separately labelled conditions:
1. **Paper-faithful:** \(A=0.01\), \(v_{\rm th}=1\), \(x\in[-2\pi,2\pi]\), \(v\in[-6,6]\), mode 1.
2. **Repository-default compatibility:** \(A=0.1\), same domain and thermal speed.
3. **Symthaea cohort:** its frozen weak-Landau condition; compare only after translating all physical units and initial-condition parameters exactly.

For each condition, run:
- TT/MPS with multiple SVD truncation thresholds;
- dense reference at matched grid and timestep;
- Symthaea dense spectral reference on matched nodes and final time;
- at least one independent high-order/spectral reference when the common configuration is established.

### B. Two-stream instability

Match the paper parameters \(v_{\rm th}=0.01\), \(v_0=0.2\), \(A=0.01\), domain and mode exactly. Record growth rate, saturation timing, electric-field mode amplitude/phase, phase-space morphology, minimum f and long-horizon drift.

### C. Truncation and rank sweeps

Vary SVD cutoff and maximum bond dimension independently. Retain:
- rank history per bond and per time;
- TCI fit query/cross counts and tolerance;
- TCI time versus TT–MPO contraction time;
- reconstructed distribution and field errors;
- positivity/negative mass;
- moment and total-energy deviations;
- full runtime, peak memory and energy-to-solution.

Do not assume the SVD truncation norm bounds field error, observable error or long-time trajectory error without a separate derivation and empirical validation.

## Fair-comparison rules

- The same physical initial condition, normalization, phase-space nodes, time interval, field equation and observable definitions must be used.
- Compare CPU-to-CPU on the same CPU host and GPU-to-GPU only when hardware is available for both; separately report each method's best supported device path.
- Include all TCI fit and reconstruction, tensor conversion, MPO construction, transfer, truncation, projection, diagnostics and I/O costs.
- Separate setup/plan construction from steady-state step time; also publish amortized full-run time.
- Record Julia version, Manifest, package versions, GPU driver, GPU model, precision, thread settings and exact source SHA.
- Retain failed runs, maximum-rank saturation, TCI nonconvergence, invalid distributions and fallback.
- Report damping/growth-rate error, field spectra, phase-space distribution discrepancy, mass, momentum, kinetic/field/total energy, positivity, rank/memory/runtime and uncertainty. Do not collapse these into one score.

## Representation-adaptation research

The new opportunity is not “TT always beats dense.” It is to learn when a representation should change:
- dense spectral for states whose rank or TCI reconstruction overhead becomes excessive;
- TT/MPS when phase-space state remains compressible and rank stays bounded;
- Fourier–Hermite when velocity tails and phase mixing make periodic finite-v truncation unsuitable;
- particle measure for particle-native regimes;
- learned field proposals only when separately calibrated and correctable.

Dense↔TT↔Hermite↔particle conversion must be treated as a numerical operation with its own error, positivity, moment, energy and runtime receipt. A no-qualified-representation decision is permitted.

## Additional 2026 representation comparators

### Adaptive Fourier spectral expansion

Cho & Russo, [*An Adaptive Fourier Spectral Method for the Vlasov–Poisson system*](https://arxiv.org/abs/2607.12368) (14 July 2026), describe a high-order splitting method that monitors spectral tails and expands the wavenumber domain with adaptive zero-padding when phase-space filamentation emerges. The authors report avoiding artificial smoothing from conventional filters while preserving mass and the (L^2) norm within dynamically resolved scales.

This is a direct comparator to the Symthaea fixed-grid dense Fourier prototype and TT compression. It suggests that the next architectural choice is not merely dense versus compressed: **fixed versus dynamically expanded spectral support**. This is paper-reported, not reproduced here. Its source-faithful reproduction and spectral-tail/zero-padding transition oracle are tracked in [#13106](https://github.com/Luminous-Dynamics/symthaea/issues/13106).

### Positivity-preserving dynamical low-rank correction

Kormann, Nazarov & Wen, [*Positivity-preserving dynamical low-rank methods for the Vlasov equation*](https://arxiv.org/abs/2606.31662) (30 June 2026), formulate minimal constrained corrections to low-rank approximations as quadratic programmes. One proposed scheme additionally preserves total mass and momentum, with Vlasov–Poisson and Vlasov–Poisson–BGK examples.

This is a comparator against TT truncation and any future low-rank Symthaea representation. Positivity is relevant only for nonnegative physical (f); do not transfer it to signed (delta f) without a separate mathematical contract. It is paper-reported, not independently reproduced.

## Claim ceiling

No claim that TT/MPS is faster or more accurate than Symthaea's dense spectral prototype, no claim that the current repository default is an exact paper reproduction, no universal compression advantage, and no adaptive representation claim until the source mismatch is resolved and the matched trajectory suite is independently reproduced.
