# NEO observation value-of-information finite oracle v0.1

**Status:** synthetic exact finite decision-theory control; no NEO orbit model, observatory likelihood or operational decision capability is included.

## Why this case exists

An observation planner can maximize entropy reduction and still select the wrong observation for a consequential decision. It can also overstate evidence by treating repeated readings with a shared calibration/catalogue bias as conditionally independent.

This fixture is a deliberately small, exactly enumerable counterexample for Symthaea's [NEO-OBSERVATION-VOI-001](https://github.com/Luminous-Dynamics/symthaea/issues/11112). It gives a deterministic oracle before introducing real astrometry, telescope models or Monte Carlo.

## Decision model

There are two latent states:

- impact mode, prior probability 0.02;
- safe mode, prior probability 0.98.

There are two abstract actions. `protect` incurs unit loss in either state. `no_protect` incurs 100 loss units in impact mode and zero in safe mode. The baseline Bayes risk is therefore 1.0 and the baseline action is `protect`. The decision threshold is 1% impact probability.

The loss values are dimensionless teaching values. They do not represent real money, public-warning thresholds or civil-protection recommendations.

## Counterexample A: entropy is not decision value

The candidate `bright_nuisance_only` perfectly reveals an independent binary nuisance variable. It reduces total latent-state entropy by 1 bit, but it cannot change the impact posterior or action. Its expected value of sample information (EVSI) is exactly zero.

The candidate `impact_mode_perfect` perfectly reveals the decision-relevant impact/safe mode. Its entropy reduction is only:

`H_2(0.02) = 0.14144054254182067 bits`.

But its expected post-observation Bayes risk is 0.02 and its EVSI is 0.98. A correct decision-value selector must rank it above the candidate that yields seven times as much entropy reduction but no change in decision loss.

This is the minimal acceptance control for **decision-sensitive observation selection**.

## Counterexample B: repeated observations are not automatically independent

The synthetic sensor model has a 10% chance of a shared source-calibration inversion. Conditional on that shared calibration state, each reading has an independent 5% error. The marginal error probability of one reading is 14%, but two readings with the same `source_correlation_group` still share a latent failure mode.

Exact enumeration gives:

| Observation design | Mutual information about impact state | Expected post-observation Bayes risk | EVSI |
|---|---:|---:|---:|
| One reading | 0.03650383168720506 bits | 0.4344 | 0.5656 |
| Two readings, same source/calibration | 0.0434174064542725 bits | 0.3869 | 0.6131 |
| Two readings, independent source calibrations | 0.06897668656485645 bits | 0.3140 | 0.6860 |

If the evaluator ignores the common calibration latent variable and multiplies the two marginal likelihoods as if the same-source readings were independent, it predicts risk 0.314 instead of the correct 0.3869. That understates expected residual decision loss by 0.0729 and overstates EVSI by the same amount.

A measurement count is not an independence count. Correlation must be represented in the generative model, not just noted in prose.

## Exact acceptance gates

The JSON fixture is paired with `schemas/finite-case-v0.1.schema.json`. A future evaluator must:

1. Validate the fixture and reject non-finite, out-of-range or missing probabilities.
2. Enumerate every observation outcome exactly; outcome probabilities sum to one.
3. Apply Bayes' rule and minimize conditional expected loss for each outcome.
4. Reproduce every posterior, conditional risk, expected risk, mutual-information and EVSI oracle within the fixture tolerance.
5. Rank candidates by expected decision value, not entropy reduction alone.
6. Keep same-source shared calibration latent variables in the joint likelihood; do not multiply marginal likelihoods as if they were independent.
7. Report explicit abstention when required likelihoods, loss functions, source-correlation groups or uncertainty are unavailable.
8. Never label these synthetic outputs as real NEO impact probabilities.

Run the future evaluator against the fixture with deterministic arithmetic. A property-based test should additionally generate small finite state/observation models and compare the production evaluator with a brute-force reference enumerator.

## Baselines and extension path

For larger v1 campaign simulations, compare against:

- random feasible observation;
- brightest-object / highest-detection-probability first;
- maximum entropy reduction;
- cheapest observation first;
- expert-designed schedule;
- exact finite optimum where the scenario is still enumerable.

Report top-k decision value, schedule regret, realized loss, observation yield, cost, calibration, selection stability under prior perturbations and compute budget. Use posterior predictive log score / Brier score only on held-out truth with well-defined event labels.

The NEO-specific extension should test multi-modal orbit posteriors, different observing geometries, radar/astrometry complementarity, observatory outages, clock offsets, catalogue biases, duplicate observations, weather cancellation and model discrepancy. Only introduce real telescope sensitivity or astrometric likelihoods when source-backed, versioned and valid for the instrument/conditions.

The downstream composition remains:

- **Symthaea:** candidate ranking and value-of-information calculations.
- **Symtropy:** validated orbit/posterior propagation and decision-loss functions.
- **Mycelix:** raw-source identifiers/digests, calibration lineage, source correlations, correction history and audit record.
- **Sol-Atlas:** geometry and uncertainty visualization of supplied qualified candidates; no renderer-side official risk computation.

## Limitations

This v0.1 fixture tests the mathematics of finite decision value and shared-source dependence only. It is not an asteroid observation likelihood, orbit determination, Sentry reproduction, alerting service, or mitigation planner. No superiority over CNEOS or any operational authority is claimed.


## Rust reference evaluator (v0.1)

The standalone Rust package at `verifier/` recomputes the exact finite oracle rather than trusting the expected values in the fixture. It enumerates the one-reading, shared-calibration two-reading and independent-source two-reading outcome distributions, applies Bayes' rule, chooses the minimum conditional expected-loss action, checks every frozen outcome row, checks mutual information / EVSI / expected risk, and emits a SHA-256 receipt over the embedded fixture bytes.

The nested Cargo package is intentionally isolated from the root Symthaea workspace; it adds no dependency to the production workspace. Its direct dependencies are exact-pinned to versions already present in the root lockfile.

Run:

```sh
cargo generate-lockfile --manifest-path benchmarks/neo-observation-voi-v0/verifier/Cargo.toml
cargo fmt --manifest-path benchmarks/neo-observation-voi-v0/verifier/Cargo.toml -- --check
cargo clippy --manifest-path benchmarks/neo-observation-voi-v0/verifier/Cargo.toml --locked --all-targets -- -D warnings
cargo test --manifest-path benchmarks/neo-observation-voi-v0/verifier/Cargo.toml --locked
cargo run --manifest-path benchmarks/neo-observation-voi-v0/verifier/Cargo.toml --locked --release
```

The dedicated workflow is `.github/workflows/neo-observation-voi-v0.yml`. It is defined on this research branch and will run for subsequent benchmark changes once merged into the default branch. The existing root CI does not compile this independent nested package, so root-workspace CI success must not be reported as verifier test success.


The verifier also includes a deterministic 128-case finite-model sweep. For each generated likelihood pair it compares posterior-based evaluation against an independent joint-probability action-loss identity, then permutes outcome labels to check risk and information invariance. This adds a second path to the finite risk calculation without introducing a property-testing dependency.


## Unknown dependence is a first-class outcome

The fixture also links the synthetic unknown optical/radar source relation from Mycelix's `source-correlation-profile-v0.1`. When dependence assessment is `unknown`, the joint likelihood is unavailable and independence is not evidenced. The evaluator therefore requires `ABSTAIN_DEPENDENCE_UNRESOLVED`, rejects a numeric EVSI, and rejects an independent-likelihood product. It does not relabel unknown as dependent; it simply prevents the planner from pretending it knows the joint model.

This control is important in real campaigns: sensor modality, provider count or different observation IDs are not proofs of independence. A bounded, evidence-backed dependence model can restore quantitative evaluation; until then, abstention is the honest output.


The evaluator also pins the nuisance-only case to the unchanged baseline Bayes risk and exactly zero EVSI, and pins perfect-information post-observation risk to the baseline-minus-EVSI identity. This prevents a fixture edit from accidentally relabeling irrelevant entropy as decision value.
