# Finite probability and Markov validation v0

This is an isolated, dependency-free Rust harness that includes the exact
production source files from symthaea-markov by path. It does not copy the
implementation and does not replace full workspace integration tests.

## Run

```sh
cargo fmt --manifest-path research/markov-validation-001/Cargo.toml -- --check
cargo test --locked --manifest-path research/markov-validation-001/Cargo.toml
cargo clippy --locked --manifest-path research/markov-validation-001/Cargo.toml --all-targets -- -D warnings
```

## Semantics

- A FiniteDistribution created with new is validated as already normalized;
  it is never silently rescaled.
- FiniteDistribution::normalized is an explicit operation for non-negative
  finite weights with strictly positive total mass.
- MarkovChain::try_new validates matrix dimensions, finite/non-negative
  transition entries and row sums within an explicit tolerance.
- try_step and try_distribution_after reject malformed state distributions
  with typed errors. The old String-returning constructor and Vec-returning
  convenience methods remain for compatibility; the old propagation methods
  are deprecated in favour of typed fallible APIs.

## Negative controls

The in-module tests reject empty vectors, NaN, positive/negative infinity,
negative entries, zero mass, wrong dimensions, non-normalized vectors,
non-square matrices, row-sum drift and invalid tolerances.

## Next stochastic-process gates

1. Add exact finite-chain stationary-current and time-reversal fixtures.
2. Verify detailed balance and zero steady-state path irreversibility on a
   reversible two-state chain.
3. Verify positive stationary entropy production on a biased three-state ring.
4. Compare exact expected trajectory log-ratios with Monte Carlo estimates
   and confidence intervals.
5. Only after local detailed balance and physical energy assignments are
   supplied should a dimensionless path-irreversibility result be upgraded
   to physical entropy production or heat.

This harness establishes probability input validity only. It does not yet
compute Wasserstein distances, entropy production or physical thermodynamics.
