# D2D Analytic v0.1

A dependency-free Rust fixture for the first executable Dream-to-Discovery protocol slice.

## Run

From the Symthaea repository root:

~~~bash
cargo test --manifest-path research/d2d-analytic-v0.1/Cargo.toml
cargo run --manifest-path research/d2d-analytic-v0.1/Cargo.toml
~~~

The crate declares its own Cargo workspace so it remains isolated from Symthaea's large workspace and does not add a new member or dependency to the production build.

## What it demonstrates

1. The agent receives two competing friction hypotheses.
2. It selects the affordable experiment with the highest declared discrimination-per-cost heuristic.
3. It seals predicted terminal velocities before the execution sequence.
4. An evaluator-only analytic world executes using a hidden friction parameter.
5. The evaluator compares the observed outcome to the sealed predictions.
6. A branch audit preserves missing, duplicate, unexpected, unsupported, failed, incomplete and resource-bound dispositions.

The fixture includes known-answer, budget, no-discrimination, seal-ordering, experiment-substitution and denominator-accounting tests. It also contains a deliberately harder selector-comparison case in which observation noise and rollout cost vary across candidate experiments.

### Selector comparator fixture

The new three-design case has two friction hypotheses, equal prior weights and Gaussian observation noise:

| Candidate | Steps | Noise σ | Predicted separation | Heuristic score |
|---|---:|---:|---:|---:|
| short-noisy | 1 | 0.15 | 0.0700 | 0.4667 |
| medium-precise | 2 | 0.05 | 0.1337 | 1.8908 |
| long-noisy | 5 | 0.20 | 0.2919 | 0.6527 |

All three fit the same five-step budget. The current discrimination/noise/√cost selector chooses **medium-precise**. A maximum-separation-only baseline chooses **long-noisy**; a cheapest-only baseline chooses **short-noisy**.

For this restricted equal-prior, two-Gaussian, equal-variance model, a deterministic Simpson-quadrature calculation of exact-family mutual information (expected information gain) gives approximately 0.0382, 0.6835 and 0.3056 bits, respectively. The principled EIG comparator also chooses medium-precise. Uniform-random selection has expected EIG of approximately 0.3424 bits across these three candidates.

These values are independently recalculated reference values for the declared analytic fixture, not observed experiment outcomes or a generalized Bayesian optimizer result. The EIG calculation is valid only for this stated binary Gaussian family. It should not be used to describe the current selector's heuristic score as EIG. The authored Rust regression exercises the ranking comparison; it has not yet been compiled or run.

## Explicit limitations

- This is a scalar analytic toy system, not a Symtropy adapter or physical simulator.
- The evaluator and learner APIs are separate, but the demo runs in one process. This is not privilege isolation.
- The prediction fingerprint uses a deterministic FNV-style hash only for fixture identity. It is not cryptographic, collision-resistant, signed, or suitable as a Mycelix evidence commitment.
- The discrimination-per-cost score is a transparent heuristic, not exact expected information gain.
- The model has fixed candidate friction hypotheses; it does not train HLB, learn new dynamics, update memory, or improve a policy.
- Passing the tests establishes only the behavior of this fixture.

## Next composition step

Replace the evaluator's analytic integrator with a narrow Symtropy scenario adapter while preserving the same proposal, prediction-before-reveal, execution-receipt and branch-accounting semantics. Use Mycelix's existing experiment-attestation owner for provenance later; do not add a second evidence ledger.
