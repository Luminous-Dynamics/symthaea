# Decision-Relevant Predictability Benchmark

**Status:** executable finite-world Bayesian and prior-free robust decision-value evaluator; broader chaotic-systems benchmark remains a protocol proposal.

Parent research protocol: [issue #12546](https://github.com/Luminous-Dynamics/symthaea/issues/12546).

Standalone Rust workspace: this directory. It has no external crate dependencies and uses a pinned Cargo.lock.

## Research question

Can a model preserve decision quality, calibrated uncertainty and useful dynamical information beyond the horizon at which exact pointwise trajectory prediction becomes unreliable, while spending fewer sensing and compute resources than fixed-cadence or prediction-only policies?

Pointwise forecast accuracy, autonomous dynamical fidelity, decision utility, epistemic reliability and full resource cost are distinct evaluation axes. There should be no unqualified composite score.

## Implemented finite-world evaluator

The current Rust library provides:

- Candidate-visible episode, observation, permitted-probe and submission types.
- Separate evaluator-side hidden state priors, decision-loss tables and evidence/probe outcome likelihoods.
- Explicit conditional-independence declaration for multiple evidence channels. Correlated observations require one joint channel; the evaluator does not silently multiply arbitrary marginal likelihoods.
- Posterior updating from authorized evidence.
- Information-set Bayes risk under a frozen worst-case decision-loss constraint.
- A separately labelled perfect-information lower bound and irreducible information gap.
- Candidate expected loss, worst-case risk-gate result and decision regret.
- Exact finite-outcome expected value of sample information (EVSI), acquisition cost, net EVSI and mutual information in bits.
- Probe permission, componentwise budget admission and decision-deadline checks.
- A separate prior-free robust mode in `src/robust.rs`: set-valued outcome supports, worst-case minimax decision loss before/after an observation, and guaranteed worst-case decision-loss reduction without priors or likelihood magnitudes.
- Permission, declared cost-budget and decision-deadline checks for the robust mode; acquisition cost remains separate from task loss and is never silently scalarized.
- Fail-closed validation of prior mass, outcome likelihood normalization, missing decision losses, unknown observations/probes and mismatched public probe contracts.

The current known-answer tests include:
- observationally equivalent passive evidence and irreducible decision risk;
- a perfectly discriminating probe with positive net decision value;
- a one-bit diagnostic with zero downstream decision value;
- a frozen worst-case risk veto;
- over-budget and late probe rejection;
- unknown/unpermitted probe rejection;
- malformed outcome likelihood mass;
- rejection of multiple evidence channels when conditional independence is not declared.

The evaluator is deterministic and does not simulate probe execution. It computes the value of a requested probe from the evaluator's declared likelihood model. The posterior and risk reports are evaluator outputs and must not be returned to the candidate during a confirmatory episode.

## Run

~~~bash
cargo fmt --manifest-path research/decision-predictability-benchmark/Cargo.toml -- --check
cargo test --manifest-path research/decision-predictability-benchmark/Cargo.toml --locked --all-targets
cargo clippy --manifest-path research/decision-predictability-benchmark/Cargo.toml --locked --all-targets -- -D warnings
~~~

Focused workflow: [Decision-Relevant Predictability Evaluator](https://github.com/Luminous-Dynamics/symthaea/actions/workflows/decision-predictability-benchmark.yml).

A passing run qualifies only this small finite-world evaluator. It does not qualify the full #12546 benchmark, chaotic-system reference generators, external baselines or physical sensor measurements.

Focused exact-head qualification: [workflow run 45382531423](https://github.com/Luminous-Dynamics/symthaea/actions/runs/45382531423), commit ee1bd6d62140f028b3f4909bfe9c056b6ba0b076. Rust 1.96.0 formatting, **14 tests (14 passed, 0 failed)** and strict Clippy all passed. This qualifies the finite-world Bayesian and robust decision-value primitives only; it is not the full #12546 benchmark.

## Decision-value modes and covariance integration

Use the Bayesian evaluator only when prior probabilities and outcome likelihoods are justified. Use the robust evaluator when the defensible information is a set of possible states/outcomes and no probability model is warranted. Keep their outputs in separate fields and separate benchmark tables.

The covariance-aware Stage-B planner (Symthaea PR #13453) supplies prior-free worst-case hypothesis-separation scores. Those distances can help rank experiments for discrimination, but they do not by themselves define observation outcome supports, calibrated likelihoods, or decision losses. A typed adapter must bind the actual experiment response model and downstream decision policy before any Stage-B score can feed the robust decision-value evaluator. Never relabel Mahalanobis separation as EVSI or EIG.

## Existing related implementations — separate owners

- [#9817](https://github.com/Luminous-Dynamics/symthaea/issues/9817) owns the active-sensing / decision-focused value-of-information task contract.
- [PR #9821](https://github.com/Luminous-Dynamics/symthaea/pull/9821) contains a separate Rust active-sensing planner under research/temporal-bench. It estimates expected task-loss reduction, rejects high-entropy but decision-irrelevant observations, checks six sensor-specific resource dimensions, and returns a Pareto frontier with provenance-bound receipts. It is not the paired-world evaluator described here.
- [#12407](https://github.com/Luminous-Dynamics/symthaea/issues/12407) owns observation time/mask/modality/event semantics.
- [#12502](https://github.com/Luminous-Dynamics/symthaea/issues/12502), [#12505](https://github.com/Luminous-Dynamics/symthaea/issues/12505), [#12506](https://github.com/Luminous-Dynamics/symthaea/issues/12506) and [#12624](https://github.com/Luminous-Dynamics/symthaea/issues/12624) own known-truth dynamics, Lyapunov qualification and predictor comparisons.
- [#5476](https://github.com/Luminous-Dynamics/symthaea/issues/5476), [#5477](https://github.com/Luminous-Dynamics/symthaea/issues/5477) and [#5478](https://github.com/Luminous-Dynamics/symthaea/issues/5478) own internal value-of-computation calibration, bounded enforcement and causal metareasoning ablations.

Do not create a competing scheduler or claim that PR #9821 supplies the paired-world evaluator, chaotic reference generators, or independent evaluator process.

## Evaluator security boundary

The candidate-visible and evaluator-side types make the interface explicit, but a Rust type boundary in one process is **not** a process-security boundary. Confirmatory runs must keep evaluator truth, priors, hidden likelihoods and loss tables in a separately permissioned process/environment and release only authorized observations to candidates.

Recorded high-rate traces can support an emulated acquisition model at requested timestamps; interpolation must not be presented as an actual observation. Physical energy claims require policy-controlled measurements on target hardware, including acquisition, wake, conversion, compute, storage and transmission.

## Remaining implementation gates

Before this becomes the complete #12546 benchmark:
1. add deterministic paired-world generation with evaluator-only truth and an independently permissioned execution boundary;
2. add multiple observation channels with joint likelihood models for correlated modalities;
3. add noisy, missing, delayed, duplicated and stale probe/observation controls;
4. add reference generators and qualified Lyapunov normalization for the planned dynamical systems;
5. add persistence, linear AR, ESN, NGRC, NGRC-MDS and eligible neural baselines under matched training/compute budgets;
6. implement the complete active-observation policy tournament and Pareto resource accounting;
7. publish paired confidence intervals, worst-regime results and full failure denominators.

Resource dimensions are comparable only when units, measurement boundaries, hardware and workloads match. This crate does not measure physical energy or hardware performance. Expected information gain about hidden state is reported separately from EVSI; a probe can reduce entropy without changing the optimal decision. The prior-free robust module is a different estimand again: it reports a guaranteed worst-case loss reduction over declared outcome-support sets, not an expectation.

No output from this crate establishes SOTA, physical reliability, causal reconstruction or consciousness.
