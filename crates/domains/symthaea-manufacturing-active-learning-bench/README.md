# Manufacturing Active-Learning Benchmark

This crate is the first executable slice for Symthaea issue [#7638](https://github.com/Luminous-Dynamics/symthaea/issues/7638), implementing parts of protocol [PR #7637](https://github.com/Luminous-Dynamics/symthaea/pull/7637). Statistical reporting follows the paired, censor-aware plan in [#7691](https://github.com/Luminous-Dynamics/symthaea/issues/7691).

It is **synthetic policy evidence only**. Its normalized factors, quality margin, throughput, resources and relative cost units do not represent real equipment, material properties, process settings, safety limits or manufacturing savings. No hardware, network, supplier or machine-control interface exists here.

## Run

From the Symthaea workspace:

```sh
cargo test -p symthaea-manufacturing-active-learning-bench
cargo run -p symthaea-manufacturing-active-learning-bench --example run_reference_campaigns > mfg-active-learning-synthetic-v1.json
```

The example runs six policies across five deterministic scenario families for 30 paired seeds (900 policy campaigns total), with matched budgets within each scenario/seed pair. It emits raw JSON transcripts, evaluator reports, and a statistical summary. The five reference-manifest SHA-256 digests are pinned in a unit test. The output file is an experiment artifact and should be hashed and preserved alongside the exact commit used to produce it.

The analysis groups results by scenario and policy, verifies every expected policy arm has the same paired-seed set, and reports:
- discovery rate separately from discovery cost conditional on success;
- restricted mean cost-to-discovery (the preregistered budget horizon is used for NotFound campaigns) with a descriptive normal-approximation 95% interval;
- fixed-design-relative acceleration factor and enhancement factor, with definitions and denominators included in the output;
- paired candidate-minus-fixed-design cost and utility differences with intervals;
- best conforming utility/throughput, resources, and design-rank rate as separate outcomes.

These summaries are synthetic and descriptive. The intervals are not a substitute for a preregistered domain-specific inferential analysis, and no metric establishes physical manufacturing performance.

## Finite-hypothesis acquisition kernel (issue #7670)

The source module `acquisition.rs` provides a separate finite-hypothesis EIG / decision-value kernel. It is intentionally **not yet wired into the six-arm campaign runner**; P3 and P4 remain explicitly named novelty/leverage proxies until a declared posterior and measurement-likelihood model is supplied.

A versioned model declares:
- prior hypothesis probabilities as integer parts per million that must sum exactly to one million;
- for each action and hypothesis, a complete categorical observation likelihood whose probabilities sum exactly to one million;
- exact measurement-model identity and relative experiment cost;
- a utility table for each decision across every hypothesis.

The kernel computes expected posterior entropy, EIG in nats, expected sample information value (the expected improvement in the best decision utility after observing the result), and both quantities per declared cost. It ranks only feasible actions; blocked and unknown actions appear as rejected records with reasons. Incomplete likelihoods and malformed priors do not produce a fabricated score. SHA-256 binds the exact prior, likelihoods, utilities, actions, cost and model version.

Posterior updates use `initial_posterior_state` followed by `condition_posterior`. Each observation requires a unique evidence ID and creates a receipt chained to the parent state digest, action, outcome, measurement-model identity and resulting posterior. Before every update, the kernel replays the existing conditioning history from the root prior and rejects duplicate evidence IDs, altered posterior weights, mismatched likelihoods or broken parent-state links. `expected_information_gain_for_state` and `rank_feasible_actions_for_state` evaluate subsequent experiments against that updated posterior rather than silently reverting to the original prior. Acquisition results report the snapshot's model digest separately from the posterior-history state digest, so a caller can bind both the probabilistic assumptions and the observations used to reach the current belief. This prevents accidentally reusing one measurement as independent evidence in normal API use; hashes still do not authenticate the external source of an observation.

Hand-checkable tests cover:
- a perfect binary observation with EIG = ln(2) and decision value = 500 synthetic utility milli-units;
- an uninformative observation with zero EIG and zero decision value;
- a highly informative observation that has zero decision value because it cannot change the best choice;
- a symmetric 90/10 noisy observation;
- a blocked high-information action that is not ranked;
- missing likelihood rows, invalid probability mass, and model-digest changes;
- posterior normalization after perfect/noisy observations, impossible-outcome rejection, duplicate evidence-ID rejection, tampered-state rejection, rehashed-receipt tampering, and acquisition recomputation under the updated posterior.

This remains synthetic mathematics. The likelihood must be supported and calibrated for a real process before the kernel can guide physical experimentation. Passing these tests would not establish physical information gain, causal identification, manufacturing utility, or savings.

## Current policy arms

- `P0-fixed-space-filling`: predeclared two-factor Latin-hypercube-like order over the normalized 11×11 grid; it does not adapt candidate order to observed outcomes.
- `P1-random-feasible`: deterministic pseudo-random selection from feasible affordable actions.
- `P2-uncertainty-only`: spatial novelty proxy.
- `P3-information-gain-proxy`: spatial novelty multiplied by a declared factor-space leverage heuristic. It is **not** Bayesian expected information gain.
- `P4-cost-aware-information-gain-proxy`: the same transparent heuristic divided by declared experiment cost.
- `P5-decision-aware`: nearest-observation response estimate with a novelty and reference-fidelity bias.

Every policy sees the same public candidate catalog, profile, costs, budget and already-observed measurements. It does not receive the hidden scenario family, oracle version, or oracle seed as selector fields. The candidate catalog does contain declared feasibility, measurement-fidelity and relative-cost information.

All arms share the same predeclared fidelity cadence: three proxy slots followed by a reference-fidelity slot, with fallback only when no candidate at the scheduled fidelity is affordable and available. The fixed-design arm also avoids re-measuring a design point already included in its predeclared sequence. This prevents a policy from appearing to discover conforming outputs without buying reference-quality confirmation. Proxy outcomes may inform selection, but **only reference-fidelity outcomes can qualify a synthetic conforming discovery or contribute synthetic conforming-product utility/throughput**. Here, “reference” means the hidden high-fidelity synthetic oracle—not physical metrology, certification, or real product conformity.

## Synthetic scenarios

- Smooth response.
- Narrow feasible window.
- Material/process interaction.
- Confounded factors, seeded with three shared diagonal-only prior observations that fail the synthetic target; an off-diagonal experiment is needed to find the synthetic feasible region. The same scenario-independent relative-cost surface applies to every family.
- Multi-fidelity discrepancy, where the low-cost proxy can be biased relative to the reference measurement.

The response oracle is deterministic and only invoked by the campaign runner for a selected action. A proxy observation is still synthetic, not physical evidence.

## Integrity and evaluator

Each manifest is serialized and SHA-256 hashed. Each observation receipt binds the manifest digest, exact action, profile ID and measurement payload. The evaluator independently recomputes cost, discovery cost, conformity, best observed utility, resource totals, budget compliance, action uniqueness and factor-design excitation from the manifest and transcript; it rejects missing required measurements and inconsistent claims.

The evaluator is in its own Rust source module and recomputes outcomes without calling policy selection or the synthetic oracle. It is **not yet a separately built executable or independent third-party verifier**. SHA-256 confirms receipt bytes match; it does not authenticate a physical source or prove scientific validity. Full cryptographic signing, source-qualified physical observation adapters, calibrated real-process likelihoods, and domain-specific inferential analysis remain future gates.

## Key distinctions

- `Feasible`, `Blocked`, and `Unknown` are separate. Only feasible candidates can be selected.
- The confounded scenario's initial observations are identical across policy arms and their declared costs count against the total campaign budget.
- Experiment cost and resource values are relative integer units, not currency or lifecycle costing.
- Quality margin is a toy constraint. A positive margin means synthetic conformance within this fixture only.
- Full-rank factor excitation is a geometric property, not proof of causal identifiability.
- A successful synthetic campaign does not establish physical manufacturability, product conformity, defect detection, production yield, safety, savings or cross-process transfer.

## Initial acceptance tests

Unit tests cover deterministic manifests and replay, all 30 scenario/policy combinations, feasibility gating, budget compliance, missing measurements, altered receipt data, falsified cost/conformity claims, blocked-candidate injection, proxy/reference discrepancy and factor-design rank. Tests have been authored but must be run against an exact commit before any pass is claimed.

## Household appliance active-diagnostics adapter (synthetic)

The appliance fan adapter in `src/appliance_adapter.rs` and its integration fixture are a domain consumer of the **same finite-hypothesis EIG / decision-value kernel above**. They do not implement a second entropy, posterior, or expected decision-value scorer.

Fixture: `fixtures/appliance_fan_active_diagnostics_v0_1.json`  
Integration tests: `tests/appliance_active_diagnostics.rs`

The scenario starts from a four-hypothesis tabletop DC-fan fault set (`bearing_wear`, `impeller_obstruction`, `motor_drive_fault`, `sensor_fault`) with visible integer-PPM observation likelihoods for RPM, airflow, and vibration checks. The repair/inspection utility table and test costs are explicitly synthetic. A registered fixture demonstrates that, under the same probability model and cost profile, EIG-per-cost prefers `rpm_check` while expected decision-value-per-cost prefers `airflow_check`. This is an illustration of objective choice, not evidence that airflow testing is better for a real fan.

The adapter adds only two kinds of domain policy:

- hard abstention for tripped/unknown protection or missing, stale, conflicting, and invalid-unit observations;
- an explicit `NoDecisionValue` result when the shared kernel finds no expected decision improvement, while preserving the full ranking receipt.

Unavailable, blocked, or unknown-feasibility tests remain rejected by the shared `Feasibility` gate. A posterior update reuses the same kernel's chained evidence receipt and posterior-state digest. The adapter has no sensor driver, network command, actuation, safe-to-operate verdict, or product-safety authority.

Cross-project trackers: [Symtropy APPLIANCE-QUAL-001 #1805](https://github.com/Luminous-Dynamics/symtropy/issues/1805), [Symthaea APPLIANCE-DIAG-001 #7819](https://github.com/Luminous-Dynamics/symthaea/issues/7819), and [APPLIANCE-DIAG-002 #7837](https://github.com/Luminous-Dynamics/symthaea/issues/7837).

