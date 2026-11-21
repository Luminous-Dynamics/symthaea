# Plasma phase-space measure diagnostics

A dependency-free Rust diagnostic for comparing weighted empirical measures in
one-dimensional periodic-position / unbounded-velocity phase space.

This is a benchmark primitive, not a plasma solver and not a complete metric on
probability measures.

## Definition

For each configured characteristic mode \((k,\eta)\), evaluate

\[
\widehat{\mu}(k,\eta)
= \sum_i w_i \exp\!\left(i\left[
2\pi k\frac{x_i \bmod L}{L}
+ \eta\frac{v_i-v_*}{v_s}
\right]\right).
\]

Here:

- \(L\) is the frozen periodic position-domain length.
- \(v_*\) is the frozen velocity reference.
- \(v_s>0\) is the frozen velocity scale.
- \(k\) is a non-negative integer spatial Fourier mode.
- \(\eta\) is a dimensionless velocity frequency.
- \(w_i\) is the signed effective particle weight.
- Each mode has a strictly positive feature importance.

The mode set must contain \((k,\eta)=(0,0)\), so net-weight mismatch is
always visible. Signed weights are supported; normalization uses total
variation \(\sum_i |w_i|\), not net charge/mass.

Given candidate and reference coefficient vectors, the implementation reports
a feature-importance-weighted RMS complex-coefficient error, normalized by the
larger total-variation weight. It also reports maximum per-mode error, net
weight mismatch, total-variation mismatch, per-mode values, and particle
counts. Particle count is metadata and is never part of the distance.

## Why this matters

Particle identities are usually not physical observables. Permuting particles
or replacing one macro-particle by co-located particles whose weights sum to
the original changes the representation without changing the represented
empirical measure. A particle-by-particle norm can therefore report an error
where the represented distribution is unchanged.

The characteristic signature is directly linear in the weighted empirical
measure, so it respects permutation and co-located weight splitting up to
floating-point summation error. Mixed position-velocity features distinguish
phase-space correlations that separate position-only and velocity-only
marginals could miss.

## Required use and limits

- Freeze the domain, velocity reference/scale, mode list, and mode importances
  in the benchmark manifest. They are part of the task identity.
- Report this diagnostic alongside field error, rollout/trajectory error,
  instability growth/saturation, particle noise, applicable conservation
  laws, Gauss/continuity residuals, positivity, and convergence.
- Use independent held-out trajectories and seeds.
- Keep the same species and signed-weight convention on both sides.
- Do not treat a low finite-feature error as proof that the full measures are
  equal: distinct measures can share a finite characteristic signature.
- This implementation is not a Wasserstein distance, total-variation distance
  between continuous distributions, uncertainty estimate, stability proof, or
  physics certificate. It must not independently authorize a solver proposal.

## Complexity

For \(N_p\) particles and \(M\) configured modes, evaluation is
\(O(N_p M)\), with \(O(M)\) output storage. The present implementation is
CPU scalar Rust and is a correctness-oriented baseline; no GPU or SIMD
performance claim is made.

## Validation status

The crate includes 14 source-authored unit tests covering permutation,
co-located split/merge, periodic representatives, velocity-sensitive and
weight-sensitive negative controls, signed weights, zero/empty measures,
invalid configurations, and non-finite inputs.

A source-authored test count is not a test pass. Run:

```sh
cargo fmt --manifest-path crates/tools/plasma-phase-space-measure/Cargo.toml -- --check
cargo test --manifest-path crates/tools/plasma-phase-space-measure/Cargo.toml
cargo clippy --manifest-path crates/tools/plasma-phase-space-measure/Cargo.toml --all-targets -- -D warnings
```

## References

- NEOPIC, *A Neural Operator Framework for Particle-based Kinetic Plasma
  Simulations* (2026): https://arxiv.org/abs/2610.40528
- Finn et al., *Structure preservation using discrete gradients in the
  Vlasov-Poisson-Landau system* (2026):
  https://doi.org/10.1016/j.jcp.2026.114749
- Qiao et al., *Intrinsic local Gauss's law preserving PIC method* (2026):
  https://doi.org/10.1016/j.cpc.2026.110296
