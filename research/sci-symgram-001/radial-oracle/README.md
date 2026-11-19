# SCI-SYMGRAM-001 radial polynomial oracle

A dependency-free Rust reference that checks the exact identities for the radial-force development family in:

`docs/research/SCI_SYMGRAM_001_RADIAL_FORCE_MINIPILOT_V0.md`

It uses sparse integer-coefficient polynomials in `(x1, x2, v1, v2)`. The implementation does not use floating-point arithmetic, a numerical solver, Symthaea's own symmetry finder, or the evaluator's hidden world state.

## Checks

- Infinitesimal SO(2) vector-field equivariance:
  `D F(z) [A z] - A F(z) = 0`, where `A = diag(J, J)`.
- Gram-determinant identity:
  `(x · x)(v · v) - (x · v)^2 - det(x, v)^2 = 0`.
- Exact reduced invariant dynamics for `a = x·x`, `b = x·v`, `c = v·v`, and `ell = det(x,v)`.
- SO(2) invariance checks for the four quotient observables and rejection of absolute external-frame coordinate `x1`.
- Rank-three differential of the invariant map on regular nonzero orbits, including `x = 0, v != 0`, and rank zero at the fixed origin.
- Conservation of signed angular momentum.
- Conservation of the integer-scaled energy `2(m+1)H`.
- Rejection of the unequal-stiffness oscillator as SO(2)-equivariant.

The exact reference accepts integer exponents `0..=8`, couplings `1..=1,000,000`, and state coordinates with magnitude at most `1,000,000` for the rank diagnostic. Unsupported inputs fail closed. This is a small polynomial fragment, not a general symbolic algebra engine.

## Run

```sh
cargo fmt --manifest-path research/sci-symgram-001/radial-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path research/sci-symgram-001/radial-oracle/Cargo.toml --all-targets
cargo clippy --locked --manifest-path research/sci-symgram-001/radial-oracle/Cargo.toml --all-targets -- -D warnings
```

The crate has no third-party dependencies. A green exact-head workflow qualifies these finite algebraic identities and the implementation contract; it does not qualify a production quotient compiler, prove physical truth or establish benchmark superiority.
