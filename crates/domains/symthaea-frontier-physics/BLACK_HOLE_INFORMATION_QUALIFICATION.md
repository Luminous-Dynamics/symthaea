# Black-hole quantum-information qualification boundary

This crate currently implements finite-dimensional quantum-information calculations and random-circuit experiments. It is not a gravitational black-hole solver.

## Distinct model families

### Unconstrained Haar/Page reference

A complex-Haar random pure state on an unconstrained bipartite Hilbert space. Use the finite-dimensional Page mean for the declared subsystem dimensions. This is an ensemble equilibrium mean, not an upper bound on each sample and not a dynamical model.

### Fixed-charge Haar/Page reference

For equal-gap qubits with conserved `N = sum_i |1><1|_i`, a state is Haar-random only within a declared total-excitation sector `K`. The exact reduced-state entropy mean is symmetry-resolved by charge blocks. It must not be compared to the unconstrained Page mean as though the two ensembles were interchangeable.

### Closed finite-depth scrambling circuit

Start from a low-entanglement product state in a declared fixed-charge sector. Evolve with local or all-to-all `U(1) x U(2) x U(1)` gates. Report subsystem entropy, purity, seed ensemble uncertainty, norm drift, charge expectation/variance, two-qubit gate count and gap to the matching symmetry-resolved Page *mean*. A depth curve is not a mixing-time result without depth/size/seed convergence.

### Sequential factor-transfer toy model

Start from a fixed-charge pure state. At every step apply a charge-preserving boundary gate to active register factors and transfer one factor to the radiation register. Previously emitted factors are excluded from subsequent gates. Report `S(R)`, `S(B)`, both purities, global norm, total charge, separate `N_R` and `N_B`, the additive ledger residual `|N-N_R-N_B|`, and emitted-factor identity.

This is a closed finite spin-chain protocol. Factor transfer is a register operation, not a derivation of Hawking emission.

## What is deliberately not claimed

None of the above is, by itself:

- Hawking radiation or a semiclassical black-hole evaporation calculation;
- a four-dimensional gravitational evolution or backreaction model;
- a Page-island or quantum-extremal-surface calculation;
- a Hayden–Preskill decoder, recovery channel, or efficient decoding result;
- evidence that a particular local circuit has reached its equilibrium ensemble.

## Next qualification stages

1. **QI-4 recovery:** introduce a reference system, early radiation, message input, remaining black-hole register and newly emitted radiation. Measure decoupling mutual information and actual recovery fidelity. Report decoder architecture, gate/operation count, memory and failure probability; an entropy curve alone does not prove recoverability.
2. **Energy-resolved emission:** replace equal-gap abstract qubits with a declared spectrum and exact energy-conserving local gates. Track emitted energy, residual energy and any work/bath source separately.
3. **Finite-size scaling:** sweep N, charge density, subsystem fraction, depth and topology with frozen independent seeds. Estimate finite-size transition width and uncertainty rather than extrapolating from N=8.
4. **Tensor networks:** compare dense state vectors against MPS/tensor-network evolution at matched entropy/fidelity error and disclose bond-dimension truncation.
5. **Gravity bridge:** keep JT gravity/island saddle calculations separate from spin-circuit dynamics, then compare only observables with a clearly stated mapping and assumptions.

## Required evidence receipt

A qualified result should preserve source SHA, manifest/fixture digest, model family, N, charge sector or energy spectrum, subsystem partition, topology, depth/emission count, seed schedule, sample count, gate count, numerical tolerances, norm/charge/energy residuals, entropies and purities, uncertainty, runtime, peak memory, compiler/toolchain and exact-head CI result.

Queued, skipped, missing or stale-head checks are not passing evidence.
