# Symthaea GameBench EVSI / Decision-Value Intervention Core v0.1

This is an isolated, dependency-free Rust research crate for exact finite decision-value-of-intervention calculations.

## What it measures

It separates:

- **Information gain:** reduction in entropy over candidate mechanisms.
- **Gross decision-value delta:** expected best downstream utility after the intervention minus the best utility available now.
- **Net intervention value:** gross decision-value delta minus intervention cost.

The model supports outcome-specific post-intervention utilities. This matters because an in-game probe can both reveal a rule and change the world (for example, opening a gate, consuming a key, or advancing a hazard).

For posterior p(h), baseline utility U0(a,h), intervention outcome o, outcome likelihood P(o|h,I), and post-intervention utility UI,o(a,h):

```
V0(p) = max_a sum_h p(h) U0(a,h)

VI(p) = sum_o P(o|p,I) *
          max_a sum_h P(h|o,p,I) UI,o(a,h)

gross_delta(I|p) = VI(p) - V0(p)
net_value(I|p) = gross_delta(I|p) - cost(I)

IG(I|p) = H(p) - sum_o P(o|p,I) H(P(h|o,p,I))
```

The reference recommends an intervention only when its net value is strictly positive. If all interventions are non-positive, it abstains.

## Scientific point

Entropy-greedy exploration is not equivalent to decision-focused exploration. The test suite includes an example in which a higher-information probe is irrelevant to the best action, while a lower-information probe changes the optimal decision and has positive value.

## Use boundary

This crate is evaluator-side reference mathematics. Do not pass the exact hypothesis likelihoods, outcome-conditional utility matrices, posterior, or oracle probe ranking to the candidate. The candidate must produce its own outcome forecasts and belief update through the task interface. The independent evaluator uses sealed ground truth to score forecast accuracy, intervention regret, downstream decision regret, and replay integrity.

This crate does not implement a Symthaea adapter, a hidden-rule game, a general POMDP solver, or a benchmark runner. It is a small exact reference primitive.

## Run locally

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
```

Tests in this repository are source-authored until they are actually executed by a recorded local or CI run.
