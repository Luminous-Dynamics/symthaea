# LQCD-017D independent clover-topology / Wilson-action-flow oracle

## Subject

`scripts/lqcd-topology-flow-oracle.py`

SHA-256 of the executed subject before check-in:

`9233b58aa359061823606d1891fe63b7f2c525ebbbd091f2338c3d9ed6f03f02`

The subject is Python-standard-library only and imports no Symthaea or Rust implementation code.

## Scope

This oracle fixes the semantic conventions for the first topology observable and a deliberately slow reference flow integrator:

- periodic `2^4` pure-SU(3) gauge field;
- Gell-Mann basis with `Tr(lambda_a lambda_b)=2 delta_ab`;
- clover field strength `F_munu=(C_munu-C_munu^dagger)/(8 i)`, with the color singlet removed and lattice spacing `a=1`;
- topological density
  `q(x)=[Tr(F01 F23)-Tr(F02 F13)+Tr(F03 F12)]/(4 pi^2)`;
- total clover charge `Q=sum_x q(x)`;
- Wilson-action gradient estimated independently by central finite differences along all eight Gell-Mann directions;
- one simultaneous Lie-Euler descent step `U <- exp(-i dt sum_a dS/dtheta_a lambda_a) U`, with `dt=1e-3` and finite-difference epsilon `2e-6`.

The finite-difference flow is a semantic reference, not the intended production-performance implementation.

## Executed result

The exact subject executed successfully:

```text
ok
fixture_action=8.2223611910686003
fixture_q=-0.00041713340026960005
flowed_action=8.1318570997827067
flowed_q=-0.0004119078363035159
flow_action_delta=-0.090504091285893651
gauge_covariance_max_error=2.1794464265262063e-12
flowed_max_det_error=8.9203263506210716e-16
flowed_max_unitarity_error=8.8817841970012523e-16
```

## Invariants established by this oracle

1. Identity gauge field has exactly zero clover topological charge within the frozen tolerance.
2. The nontrivial deterministic fixture has a nonzero clover charge, preventing a trivial always-zero implementation from passing.
3. The Wilson action and clover charge are invariant under a deterministic non-Abelian local gauge transformation.
4. One finite-difference Wilson-action gradient-flow step lowers the Wilson action on the frozen fixture.
5. Flow commutes with the same gauge transformation to the declared numerical tolerance.
6. The Lie-group update preserves determinant one and unitarity to floating-point precision.

## Scientific boundary

This evidence does **not** establish:

- an integer-valued physical topological charge on a rough lattice;
- a physical flow scale `t0` or `w0`;
- equivalence of this Lie-Euler discretization to a higher-order production Wilson-flow integrator at finite step size;
- topological tunneling or adequate topology sampling in a Markov chain;
- continuum topological susceptibility;
- an instanton benchmark;
- any physical observable.

The production implementation must match these semantics independently. A later optimized staple-based flow must be parity-tested against a separately qualified reference rather than replacing the reference as the source of truth.

## Literature boundary

The architecture follows the standard use of Wilson/gradient flow as a smoothing/renormalized-field construction and clover topological charge as a flowed gluonic observable. The implementation-specific finite-difference gradient and Lie-Euler step are Symthaea qualification choices and are not claimed to reproduce a particular published high-order integrator.

Relevant background:

- M. Luescher, *Properties and uses of the Wilson flow in lattice QCD*, arXiv:1006.4518.
- C. Alexandrou et al., *Comparison of topological charge definitions in Lattice QCD*, arXiv:1708.00696.
