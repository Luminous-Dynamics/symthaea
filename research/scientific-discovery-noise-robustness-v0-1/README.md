# Scientific Discovery Noise Robustness Harness v0.1

This standalone, dependency-free Rust crate supplies deterministic observation-process fixtures for [SCI-NOISE-001](https://github.com/Luminous-Dynamics/symthaea/issues/8607).

## Run

```sh
cargo fmt --manifest-path research/scientific-discovery-noise-robustness-v0-1/Cargo.toml -- --check
cargo test --manifest-path research/scientific-discovery-noise-robustness-v0-1/Cargo.toml --locked
cargo clippy --manifest-path research/scientific-discovery-noise-robustness-v0-1/Cargo.toml --all-targets --locked -- -D warnings
```

## Versioned seed domains

Use `derive_stream_seed_v1(master_seed, domain, world_id)` to derive reproducible, domain-separated seeds for the seven domains in the scenario manifest: public design, target law, noise process, in-domain evaluation, marginal-shift evaluation, joint-extrapolation evaluation, and near-boundary evaluation.

A known-answer test freezes this function's integer outputs. Changing the derivation must be treated as a new seed-contract version, not a silent refactor. Domain separation helps prevent accidental seed reuse; this deterministic test PRNG is **not** cryptographically secure, does not conceal hidden labels, and does not mathematically guarantee statistical independence between streams.

## Implemented observation-process fixtures

- Homoscedastic Gaussian noise.
- Covariate-scaled heteroscedastic Gaussian noise.
- Student-t heavy-tailed noise with finite variance.
- Contamination/outlier mixture.
- AR(1)-correlated residuals.
- Explicit missing and left/right-censored outcomes.
- A predeclared noise-scale change point.
- A bounded interleaved-anchor schedule helper to reduce confounding between time/order drift and covariate shifts in fixed-design controls.

The combined-shift regime in the protocol manifest remains reserved for a later stress phase and is not implemented here. The protocol also now specifies a cause-discrimination track (stable law versus law change versus noise change); this crate does not yet implement the diagnosis policy or evaluator. The protocol also includes candidate-set omission and harmful-informativeness stress cases; no robust acquisition algorithm is implemented here.

## Boundary and limitations

The generator is evaluator-side synthetic infrastructure. Keep the model configuration and latent response out of the discovery policy; only admitted observations and permitted pre-outcome design variables belong on the policy side. The built-in pseudorandom generator is deterministic and suitable only for reproducible fixtures, not cryptography, secret labels, or secure blinding.

Passing unit tests would establish generator mechanics, diagnostic schedule invariants and deterministic replay only. It would not establish calibration, valid sequential guarantees, detection power, real-data robustness, law-recovery performance, or superiority over published methods. Those require separate statistical qualification and matched benchmark executions.
