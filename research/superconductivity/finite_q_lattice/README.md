# Finite-Q lattice Ginzburg–Landau contract

This harness is a synthetic-only, one-dimensional periodic-ring reference for reconstructing a physical complex pair field from commensurate Fourier components.

For ring length \(L\), site \(x\), mode index \(q_m\), and site-dependent envelope \(\eta_m(x)\):

\[
\Delta_x=\sum_m \eta_m(x)e^{2\pi i q_m x/L}.
\]

The static evaluator calculates

\[
F=\sum_x\left[\alpha|\Delta_x|^2+\frac{\beta}{2}|\Delta_x|^4\right]
+K\sum_{x\to x+1}|\Delta_x-e^{iA_x}\Delta_{x+1}|^2.
\]

Every charged component shares one local electromagnetic U(1) transformation. The finite-Q modes are represented on the reciprocal ring, so the canonical integer momentum index is in \(0,\ldots,L-1\). Reciprocal-lattice aliases are rejected rather than silently accepted.

The auditor reports a discrete Fourier spectrum and a Parseval residual. Its quartic selection-rule audit enumerates ordered \(q_a-q_b+q_c-q_d=0\pmod L\) tuples. It is a symmetry bookkeeping check for constant-envelope mode couplings, not a substitute for a microscopic derivation of all allowed terms.

## Run

```sh
cargo test --manifest-path tools/superconductivity-finite-q-lattice/Cargo.toml
cargo run --manifest-path tools/superconductivity-finite-q-lattice/Cargo.toml --bin sc_finite_q_lattice_audit -- \
  --schema data/schemas/superconductivity_finite_q_lattice_v0_1.schema.json \
  --fixture data/research/superconductivity_finite_q_lattice_fixture_v0_1.json
```

## Claim boundary

A valid receipt establishes only that the synthetic commensurate finite-Q field was reconstructed and its declared static lattice functional was evaluated consistently. It does not calculate thermal helicity modulus, superfluid density, equilibrium stability, a finite-temperature phase diagram, microscopic pairing, a material-specific transition temperature, or synthesis feasibility. These are all explicitly marked not evaluated.
