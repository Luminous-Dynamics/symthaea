# ICCB evaluator prototype

This is the first isolated Rust reference kernel for the Intergenerational Capability-Compounding Benchmark (ICCB) proposed in [Symthaea PR #7974](https://github.com/Luminous-Dynamics/symthaea/pull/7974).

## Run locally

From the repository root:

~~~sh
cargo test --manifest-path research/iccb/Cargo.toml
cargo run --manifest-path research/iccb/Cargo.toml
~~~

The prototype intentionally has no third-party dependencies. Its manifest defines a nested workspace boundary so it can be tested separately from Symthaea's larger workspace while its data contracts stabilize.

## What is implemented

- Candidate-facing synthetic structs for evidence claims, decision options, mandate constraints and scenarios.
- Five named, stable-ID public development fixtures: productive expansion, asset-price-only boom, hidden maintenance burden, declared dependency links, and stale evidence.
- Evidence usability checks for explicit source, observation time, validity window and qualification state.
- Hard gates for budget, liquidity, maintenance burden and delegated authority.
- A feasible cash-maximizing baseline and coordinate-wise Pareto frontier; valuation change is observable but does not dominate real capacity or cash-flow dimensions.
- An evidence-gated staged-tranche release contract that refuses release on failed milestones, stale evidence, unauthorized requests, commitment-cap breaches or arithmetic overflow.
- Reported valuation change, expected net cash, productive-capacity delta, maintenance burden and distinct declared dependency links are separate fields. Declared link count is descriptive only and is not a risk or criticality score.
- An outcome vector that keeps net cash, liquidity, productive capacity and maintenance burden separate; declared dependency-link count is descriptive metadata only.
- Unit tests for negative controls, stale/expired/missing-source evidence, repeated dependency IDs, constraints and deterministic equality.
- A demo in which the higher-cash option fails the maintenance gate while a lower-cash, higher-capacity option passes.

## Important limits

This is a **prototype contract and rule kernel**, not an investment model. It has no portfolio optimizer, probabilistic forecast, currency/period normalization, live-source adapter, real-world asset model, cryptographic receipt, benchmark corpus loader, hidden holdout evaluator, or integration with Symthaea, Sol-Atlas, Symtropy, Mycelix or Luminous Platform.

The EvidenceState::Qualified label is supplied by the fixture; it does not independently validate a source's truth. The current day fields are synthetic integer day indices. The candidate contract counts distinct declared dependency IDs only; it does not know which are critical and does not infer cascades. The evaluator-only harness scores candidate-reported dependencies against separate truth data. expected_net_cash_minor is an input value, not a modeled or validated forecast.

The demo should not be interpreted as proving superior decisions or a real investment result. A clean local test run and independent review are still required before treating the code as validated. The current authoring environment did not have rustc or cargo installed, so no local compile/test result is claimed in this change.

## Evaluator / candidate separation

A second isolated package, [`research/iccb-evaluator-harness/`](../iccb-evaluator-harness/src/lib.rs), holds evaluator-side dependency scoring. The candidate contract crate does not depend on it. The harness checks scenario IDs, reports true/false positives and false negatives independently, and marks precision/recall undefined when their denominators are empty rather than making empty cases look perfect.

This is a **code/data-flow boundary, not a sandbox**; source-code access to the scorer is compatible with benchmark transparency, but any actual held-out truth data must remain evaluator-controlled and must not be serialized into candidate-visible requests or artifacts. The public tests in the harness are examples, not a hidden benchmark.

## Cross-project physical-asset integration

Sol-Atlas owns the shared-resource, water/energy service, time-conservation and independent allocation-oracle work. ICCB should consume those versioned, evidence-qualified assessments rather than build a second water or resource allocator. See [ICCB-002](https://github.com/Luminous-Dynamics/symthaea/issues/8002) and the linked Sol-Atlas work: [#116](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/116), [#152](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/152), [#153](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/153), [#155](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/155), and [#131](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/131).

The adapter contract is not implemented and no cross-project integration is claimed. The boundary must preserve scenario digests, units, time scope, capacity bounds, uncertainty, protected-service floors and indeterminate states.

## Adaptive capital-policy comparison

The `policy.rs` module compares four explicit synthetic policy templates: abandon, commit the full amount now, wait for a demand signal and invest only when continuation value is positive, or commit a stage now and expand only when the signal supports continuation.

The case probabilities, cashflows, discount factor and waiting cost are inputs to the synthetic scenario—not empirically calibrated parameters. All money is represented in integer minor units, the probability mass must total exactly 10,000 basis points, negative-value continuation branches are clamped to zero only when the policy has an actual option to stop, and arithmetic uses conservative fixed-point rounding. The best policy is selected only among this finite catalog, with a deterministic tie-break in favor of abandoning at exact zero-value ties.

Tests intentionally include scenarios where staging wins, waiting wins, one-shot investment wins, and capital limits make abandonment the only feasible policy. This prevents the benchmark from baking in the assumption that more reversibility is always better.

This is an exhaustive comparison of four named policy templates, **not** a general dynamic-programming oracle or a validated real-options model. The next benchmark step is to independently enumerate small state/action trees with a reference oracle and compare the policy catalog against that upper bound.

## Value of information: deciding which uncertainty to reduce

The `value_of_information.rs` module compares the best action under current priors with the best action after each possible outcome of a specified noisy signal. It reports the baseline expected value, expected value after information, gross expected value of sample information (EVSI), acquisition cost, net EVSI, and expected value of perfect information (EVPI). Signals are not treated as valuable merely because they are informative: their value comes from improving the decision, and the estimated gross value cannot exceed the perfect-information ceiling under the same payoff model.

The inputs—state priors, signal likelihoods, payoffs, and information costs—are supplied assumptions. The prototype does not calibrate them or establish physical truth. It uses small finite state spaces and fixed-point arithmetic to make the decision calculation reproducible and hand-checkable. The initial prototype fail-closes above 16 states, 16 signals, or 16 actions to bound the cost of its simple table lookups; this is an implementation cap, not a benchmark or product requirement.

This work complements [Sol-Atlas issue #131](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/131), which already covers Pareto, sensitivity and value-of-information analysis for site selection. ICCB should consume those qualified investigation assessments rather than build a second site-selection engine. Further tests and independent validation are tracked in [ICCB-003](https://github.com/Luminous-Dynamics/symthaea/issues/8020).

The method follows established decision-analysis literature, including [Strong et al. (2015)](https://journals.sagepub.com/doi/10.1177/0272989X15575286) and [Jackson et al. (2019)](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1562932). ICCB's contribution is the experimental application to capital-plan and physical-asset evidence workflows—not a claim to have invented EVSI. The cislunar numeric reference case now matches [Symtropy #1738](https://github.com/Luminous-Dynamics/symtropy/issues/1738): prior 0.40/0.60, build payoffs +100/-150, import payoff 0, survey cost 10, sensitivity/specificity 0.90, gross EVSI 27, net EVSI 17, EVPI 40. These are synthetic conformance values, not real lunar-resource priors or project estimates. Cross-repository changes to probability or payoff semantics should update the versioned reference and fail both sides until reconciled.

## Staged commitments and reversibility

The staged-capital protocol is described in [REVERSIBILITY_AND_STAGED_CAPITAL.md](REVERSIBILITY_AND_STAGED_CAPITAL.md). The `staging.rs` module is a fail-closed tranche authorization gate, **not** a real-options valuation model. It makes required evidence, milestone satisfaction, delegated authority, and cumulative commitment limits explicit. The prototype also reports a cash-maximizing feasible baseline and a Pareto frontier without compressing outcomes into one score.

## Market validation

The buyer-side falsification plan is in [BUYER_WORKFLOW_VALIDATION.md](BUYER_WORKFLOW_VALIDATION.md). It recommends testing one asset-heavy operating-company capital-planning workflow rather than positioning ICCB as a replacement for incumbent family-office reporting or accounting products. No customer interviews or paid-pilot commitments are claimed.

## Next steps

1. Run cargo test and cargo fmt against this exact head; fix any compiler, formatting or contract issues.
2. Implement a simple spreadsheet-equivalent baseline over the public fixtures.
3. Move larger fixture definitions into versioned data files and separate candidate-visible observations from a separately controlled holdout evaluator.
4. Introduce deterministic run receipts, manifest/configuration hashes and replay.
5. Add adversarial fixtures and compare per-dimension outcomes against simple baselines, then extend only where results justify it.
