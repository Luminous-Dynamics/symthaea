# LQCD-017A — Independent rectangular Wilson-loop oracle

## Scope

This evidence qualifies path ordering, periodic closure, normalization and two simple downstream estimators for rectangular Wilson loops. It is **not** an equilibrium lattice-QCD calculation and makes no claim about a physical string tension or static potential.

## Independent subject

`scripts/lqcd-wilson-loop-oracle.py`

The subject is standard-library Python and imports no Symthaea/Rust code.

It independently implements:

- periodic 4D link storage;
- SU(3) matrix multiply/dagger/trace;
- rectangular path transport with `+mu,+nu,-mu,-nu` closure;
- normalized fundamental Wilson trace `Tr(W)/3`;
- lattice-site averaging;
- effective static-potential estimator `log[W(R,T)/W(R,T+1)] / a_t`;
- Creutz ratio `-log[W(R,T) W(R-1,T-1) / (W(R,T-1) W(R-1,T))]`.

## Executed exact-subject evidence

The exact local subject subsequently checked into the branch executed successfully:

```text
ok
localized_loop_re=0.97680241074829122
localized_loop_im=-0.00099418026018326566
average_1x1=0.99871124504157172
average_2x1=0.99742249008314354
average_1x2=0.99871124504157172
average_2x2=0.99742249008314354
area_law_creutz=0.22999999999999995
exponential_static_potential=0.41000000000000009
```

### Fixtures

1. **Identity field** on `3 x 3 x 2 x 2`: all tested rectangular Wilson loops equal one to `1e-14`.
2. **Localized diagonal SU(3) link**, `diag(exp(i0.3), exp(-i0.1), exp(-i0.2))`: pins a nontrivial complex loop and site-averaged `1x1`, `2x1`, `1x2`, `2x2` values.
3. **Synthetic area law**, `W(R,T)=exp(-sigma R T)` with `sigma=0.23`: Creutz estimator recovers `0.23` to floating precision.
4. **Synthetic single exponential**, `W(T)=exp(-V T)` with `V=0.41`: effective-potential estimator recovers `0.41` to floating precision.

## Authority boundary

The synthetic estimator tests establish algebra only. A physical static potential or string tension additionally requires:

- qualified equilibrium ensembles;
- sufficiently large spatial/temporal extents;
- autocorrelation-aware uncertainties and covariance;
- smearing/flow/operator choices with provenance;
- fit-window stability;
- finite-volume checks;
- scale setting;
- multiple lattice spacings and continuum control where a physical result is claimed.

This oracle is intended as an implementation-independent parity target for the Rust Wilson-loop observable layer.