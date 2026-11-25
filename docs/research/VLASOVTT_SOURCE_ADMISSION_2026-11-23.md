# VlasovTT.jl source admission — 23 November 2026

## Status

**Source pinned; independent reproduction not yet performed.** This admission fixes the exact public source snapshot and its environment. It does not claim that Symthaea has reproduced the VlasovTT paper, matched its physical cases, or independently confirmed its performance.

Machine-readable record: [`vlasovtt-jl-source-v0.1.json`](../../benchmarks/plasma/references/vlasovtt-jl-source-v0.1.json).

## Immutable upstream identity

- Repository: https://github.com/erikasgrim/VlasovTT.jl
- Commit: [`99aa077b6f5e4ae4dfc8330590e502267caf298b`](https://github.com/erikasgrim/VlasovTT.jl/commit/99aa077b6f5e4ae4dfc8330590e502267caf298b)
- Commit authored: 12 June 2026
- Repository last pushed: 20 August 2026
- License reported by repository metadata: MIT
- Paper: Åsgrim et al., *Tensor Network Compression for Fully Spectral Vlasov–Poisson Simulation*, *Journal of Computational Physics*, published online 26 August 2026, DOI [10.1016/j.jcp.2026.115330](https://doi.org/10.1016/j.jcp.2026.115330).

The repository has a compressed TT/MPS path based on Quantics, tensor-cross interpolation and ITensors, plus a dense reference path. It uses Strang splitting and spectral operators and records physical observables, rank and runtime diagnostics. The full `Project.toml`, `Manifest.toml`, license, relevant source, examples and tests are blob-pinned in the JSON record.

## Important reproduction discrepancy

The unmodified `examples/landau_damping.jl` sets:
- (dt=0.1), (T_{final}=30);
- (R=10) bits per axis (1024 points per dimension);
- (x\in[-2\pi,2\pi]), (v\in[-6,6]);
- TCI tolerance (10^{-8}), maximum TT rank 500, field rank 32, cutoff (10^{-10});
- double GPU precision and `use_gpu=true` by default;
- linear Landau initial-condition amplitude (A=0.1).

The paper configuration recorded in the existing Symthaea comparator notes specifies amplitude (A=0.01). These are not interchangeable. The paper's exact configuration must be resolved before declaring a paper-faithful reproduction. Keep the unmodified example profile and the paper profile as separate source-labelled runs until that is reconciled.

The two-stream example also has its own source defaults; they are recorded independently in the JSON manifest.

## Required reproduction sequence

1. Verify the pinned Julia environment and preserve dependency resolution.
2. Run the three standalone upstream test scripts at the pinned commit.
3. Execute the upstream Landau and two-stream examples unmodified, retaining original configs and outputs.
4. Run the dense-reference paths separately from TT/MPS.
5. Repeat CPU and GPU paths independently where supported. Record device model, software stack, precision, memory and transfer overhead.
6. Sweep timestep, phase-space resolution, TCI tolerance and maximum TT rank. Keep the raw results.
7. Only after source-faithful reproduction, adapt the common 1D1V echo and response-parity cohort.

## Fair comparison rules

The comparison must match governing equations, normalization, initial condition, physical domain, timestep, final time, output cadence and measured quantities. A TT rank or TCI residual is not automatically a bound on future echo response.

Include the complete cost of initialization, TT construction, tensor-cross interpolation, operator contractions, Poisson solve, truncation, conversion, diagnostics, HDF5 output and failures/retries. Publish CPU and GPU results as separate lanes. Report Pareto frontiers over physical fidelity, peak memory, energy-to-solution and wall time, rather than a single speedup.

## Claim ceiling

The source pin is not a result. No Symthaea-vs-VlasovTT speedup, accuracy advantage, or representation-selection win is claimed until the independent reproduction and matched-fidelity comparison pass.
