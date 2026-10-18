# ICCB evaluator prototype

This is the first isolated Rust reference kernel for the Intergenerational Capability-Compounding Benchmark (ICCB) proposed in [Symthaea PR #7974](https://github.com/Luminous-Dynamics/symthaea/pull/7974).

## Run locally

From the repository root:

~~~sh
cargo test --manifest-path research/iccb/Cargo.toml
cargo run --manifest-path research/iccb/Cargo.toml
~~~

The prototype is isolated from Symthaea's larger workspace. Its only third-party dependency is RustCrypto sha2 for SHA-256 receipts; avoiding a home-grown cryptographic implementation is preferable to maintaining an unreviewed hash algorithm.

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

## Robust decisions when probabilities are uncertain

The `robust_decision.rs` module is a complementary mode for cases where analysts can defend an interval for state probabilities but cannot defend a single prior distribution. It takes lower/upper probability bounds and a fixed action-by-state payoff table, then calculates each action's worst- and best-case expected payoff over every feasible probability vector in that box-constrained simplex. It does not use interval midpoints or invent probability weights.

For each action, the evaluator starts all state probabilities at their lower bounds, then allocates remaining probability mass to lowest-payoff states to find the minimum and highest-payoff states to find the maximum. Tie-breaking is stable by state ID. The report includes the exact numerator, conservative floor/ceiling in integer minor units, and the probability allocations that attain each bound. A unique robust winner is reported only if its exact worst-case value is strictly greater than every alternative's exact best-case value. Overlapping envelopes are reported as no unique robust winner. Separately, the module reports each action's worst-case expected regret and all actions tied for minimum worst-case regret; this provides another decision lens without pretending the ambiguity has disappeared.

This first version bounds **probability mass only**. Payoffs are fixed inputs. It does not yet account for uncertain engineering outputs, model-discrepancy envelopes, unknown/out-of-domain states, or correlations between separate survey actions. Those values must be bounded/qualified upstream; do not conflate this probability-ambiguity layer with Symtropy's model-discrepancy work.

The current test module also sweeps 48 deterministic three-state payoff tables and compares exact worst/best payoff numerators and minimax-regret numerators against a separate coarse probability-grid oracle. This broadens the sanity check beyond one hand-picked fixture but is still a bounded synthetic verification set, not proof for arbitrary instances.

The design complements [Symtropy #1769](https://github.com/Luminous-Dynamics/symtropy/issues/1769) on robust design under uncertainty and decision-focused surveys, [Symtropy #1812](https://github.com/Luminous-Dynamics/symtropy/issues/1812) on evidence-derived model-discrepancy envelopes, and is tracked in [ICCB-004](https://github.com/Luminous-Dynamics/symthaea/issues/8024). Proposed gates include separately implemented brute-force probability-grid oracles for both payoff extrema and minimax regret on small cases, exact input-permutation invariance, fail-closed infeasible intervals, conservative rounding for negative expectations, bound-valid extremizing allocations, and no robust-winner claim when action envelopes overlap.

## Typed uncertainty semantics and fail-closed routing

The `uncertainty_contract.rs` module prevents different uncertainty products from being silently treated as equivalent. It routes a declared, qualified point-probability distribution to expected-value/EVSI analysis; bounded state probabilities to the robust probability-rank evaluator; conservative compatibility sets to compatibility-preserving planning; model-discrepancy envelopes to discrepancy-aware assessment; and out-of-domain inputs to abstention.

Every route requires a stable scenario ID, producer version and source-manifest digest, plus the appropriate qualification/bounds/compatibility/discrepancy reference. The router checks SHA-256 digest *syntax* for the source manifest and for a point-probability validation corpus (64 hexadecimal characters, optionally prefixed with `sha256:`); malformed values fail closed. This does not recompute a digest, resolve a receipt, establish that a source is authentic, calibrate probabilities, validate model discrepancy, or prove an engineering result. The router remains a **semantic routing guard only**.

This boundary aligns with Symtropy's [robust design #1769](https://github.com/Luminous-Dynamics/symtropy/issues/1769), [model-discrepancy qualification #1812](https://github.com/Luminous-Dynamics/symtropy/issues/1812), and [adaptive compatibility-based investigations #1941](https://github.com/Luminous-Dynamics/symtropy/issues/1941). The planned consumer/contract scope is tracked in [ICCB-005](https://github.com/Luminous-Dynamics/symthaea/issues/8028). The central rule is: **do not turn a compatibility set or discrepancy envelope into probability weights just to force a single expected-value answer.**

## Value of information: deciding which uncertainty to reduce

The `value_of_information.rs` module evaluates one proposed measurement with a finite set of mutually exclusive outcomes. It compares the best action under current priors with the best action after each possible outcome. It reports baseline expected value, expected value after information, gross expected value of sample information (EVSI), direct measurement cost, delay cost, financing/carrying cost, net EVSI, and expected value of perfect information (EVPI). The current API models one measurement at a time; it does not yet represent sequential/adaptive surveys or correlations between separate measurements. Information is valuable only insofar as it improves the decision, and the estimated gross value cannot exceed the perfect-information ceiling under the same payoff model.

The inputs—state priors, signal likelihoods, payoffs, and information costs—are supplied assumptions. The prototype does not calibrate them or establish physical truth. It uses small finite state spaces and fixed-point arithmetic to make the decision calculation reproducible and hand-checkable. Survey/measurement cost, delay cost and financing/carrying cost are represented separately; net EVSI subtracts all three, and the total is checked for integer overflow. This avoids allowing a “cheap survey” to look worthwhile when its waiting or financing burden erases the benefit. The initial prototype fail-closes above 16 states, 16 outcomes of the same measurement, or 16 actions to bound the cost of its simple table lookups; this is an implementation cap, not a benchmark or product requirement.

This work complements [Sol-Atlas issue #131](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/131), which already covers Pareto, sensitivity and value-of-information analysis for site selection. ICCB should consume those qualified investigation assessments rather than build a second site-selection engine. Further tests and independent validation are tracked in [ICCB-003](https://github.com/Luminous-Dynamics/symthaea/issues/8020).

The method follows established decision-analysis literature, including [Strong et al. (2015)](https://journals.sagepub.com/doi/10.1177/0272989X15575286) and [Jackson et al. (2019)](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1562932). ICCB's contribution is the experimental application to capital-plan and physical-asset evidence workflows—not a claim to have invented EVSI. The cislunar numeric reference case now matches [Symtropy #1738](https://github.com/Luminous-Dynamics/symtropy/issues/1738): prior 0.40/0.60, build payoffs +100/-150, import payoff 0, survey cost 10, sensitivity/specificity 0.90, gross EVSI 27, net EVSI 17, EVPI 40. These are synthetic conformance values, not real lunar-resource priors or project estimates. Cross-repository changes to probability or payoff semantics should update the versioned reference and fail both sides until reconciled.

## Staged commitments and reversibility

The staged-capital protocol is described in [REVERSIBILITY_AND_STAGED_CAPITAL.md](REVERSIBILITY_AND_STAGED_CAPITAL.md). The `staging.rs` module is a fail-closed tranche authorization gate, **not** a real-options valuation model. It makes required evidence, milestone satisfaction, delegated authority, and cumulative commitment limits explicit. The prototype also reports a cash-maximizing feasible baseline and a Pareto frontier without compressing outcomes into one score.

## Market validation

The buyer-side falsification plan is in [BUYER_WORKFLOW_VALIDATION.md](BUYER_WORKFLOW_VALIDATION.md). It recommends testing one asset-heavy operating-company capital-planning workflow rather than positioning ICCB as a replacement for incumbent family-office reporting or accounting products. No customer interviews or paid-pilot commitments are claimed.

## Deterministic decision receipts and replay

The new `decision_receipt.rs` module creates a versioned receipt that binds the source-manifest digest, canonical scenario digest, canonical evaluator-output digest, producer commit, evaluator version, uncertainty route, evidence reference, claim ceiling and explicit limitation. SHA-256 is provided by RustCrypto `sha2`, not a locally handwritten cryptographic implementation.

The receipt uses a length-prefixed binary encoding tagged with `iccb-canonical-binary-v1`. Its scenario canonicalizer sorts options and evidence claims by stable IDs, and sorts/deduplicates dependency IDs because this evaluator only counts distinct dependency IDs. Duplicate option and evidence IDs fail closed. Output options and rejection reasons are canonicalized before hashing. The source manifest is **not** canonicalized by this module: the caller supplies exact manifest bytes plus a declared input-encoding version, and the receipt hashes those exact bytes. The typed assessment itself is routed inside the receipt builder; its source-manifest digest must match the computed digest, and its upstream producer version is preserved.

`verify_decision_receipt` re-evaluates the provided scenario with this evaluator and compares every receipt field. This is replay using the current implementation, not yet an independent second implementation. An independent verifier should implement the published versioned encoding separately. Digest equality proves byte identity under the declared encoding, not the truth of the sources, the physical validity of a model, or authority to execute a decision. The receipt is not a spending, transfer, engineering approval or execution authorization.

Synthetic unit tests cover the published SHA-256 test vector for `abc`, repeat-run equality, stale receipt rejection after manifest, scenario, evaluator-version or upstream-producer-version mutation, option/dependency-order invariance, duplicate IDs, route/manifest digest mismatch, missing encoding/version fields, receipt tampering and malformed pinned metadata. The current exact-head format, compile, tests and demo are still required before treating this code as verified.

## Next steps

1. Run cargo test and cargo fmt against this exact head; fix any compiler, formatting or contract issues.
2. Implement a simple spreadsheet-equivalent baseline over the public fixtures.
3. Move larger fixture definitions into versioned data files and separate candidate-visible observations from a separately controlled holdout evaluator.
4. Build an independent cross-implementation verifier and freeze a canonical golden receipt with expected digests.
5. Add adversarial fixtures and compare per-dimension outcomes against simple baselines, then extend only where results justify it.
