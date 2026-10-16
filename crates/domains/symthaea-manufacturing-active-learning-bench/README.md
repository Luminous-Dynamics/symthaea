# Manufacturing Active-Learning Benchmark

This crate is the first executable slice for Symthaea issue [#7638](https://github.com/Luminous-Dynamics/symthaea/issues/7638), implementing parts of protocol [PR #7637](https://github.com/Luminous-Dynamics/symthaea/pull/7637).

It is **synthetic policy evidence only**. Its normalized factors, quality margin, throughput, resources and relative cost units do not represent real equipment, material properties, process settings, safety limits or manufacturing savings. No hardware, network, supplier or machine-control interface exists here.

## Run

From the Symthaea workspace:

```sh
cargo test -p symthaea-manufacturing-active-learning-bench
cargo run -p symthaea-manufacturing-active-learning-bench --example run_reference_campaigns > mfg-active-learning-synthetic-v1.json
```

The example runs six policies across five deterministic scenario families for 30 paired seeds (900 policy campaigns total), with matched budgets within each scenario/seed pair. It emits JSON transcripts plus evaluator reports. The five reference-manifest SHA-256 digests are pinned in a unit test. The output file is an experiment artifact and should be hashed and preserved alongside the exact commit used to produce it.

## Current policy arms

- `P0-fixed-space-filling`: farthest-point design over the declared normalized candidate domain.
- `P1-random-feasible`: deterministic pseudo-random selection from feasible affordable actions.
- `P2-uncertainty-only`: spatial novelty proxy.
- `P3-information-gain-proxy`: spatial novelty multiplied by a declared factor-space leverage heuristic. It is **not** Bayesian expected information gain.
- `P4-cost-aware-information-gain-proxy`: the same transparent heuristic divided by declared experiment cost.
- `P5-decision-aware`: nearest-observation response estimate with a novelty and reference-fidelity bias.

Every policy sees the same public candidate catalog, profile, costs, budget and already-observed measurements. It does not receive the hidden scenario family, oracle version, or oracle seed as selector fields. The candidate catalog does contain declared feasibility, measurement-fidelity and relative-cost information.

## Synthetic scenarios

- Smooth response.
- Narrow feasible window.
- Material/process interaction.
- Confounded factors, seeded with three shared diagonal-only prior observations that fail the synthetic target; an off-diagonal experiment is needed to find the synthetic feasible region. The same scenario-independent relative-cost surface applies to every family.
- Multi-fidelity discrepancy, where the low-cost proxy can be biased relative to the reference measurement.

The response oracle is deterministic and only invoked by the campaign runner for a selected action. A proxy observation is still synthetic, not physical evidence.

## Integrity and evaluator

Each manifest is serialized and SHA-256 hashed. Each observation receipt binds the manifest digest, exact action, profile ID and measurement payload. The evaluator independently recomputes cost, discovery cost, conformity, best observed utility, resource totals, budget compliance, action uniqueness and factor-design excitation from the manifest and transcript; it rejects missing required measurements and inconsistent claims.

The evaluator currently lives in a separate module in the same crate. It is **not yet a separately built executable or independent third-party verifier**. SHA-256 confirms receipt bytes match; it does not authenticate a physical source or prove scientific validity. Full cryptographic signing, source-qualified physical observation adapters, calibrated probabilistic information gain, uncertainty intervals and statistical analysis are future gates.

## Key distinctions

- `Feasible`, `Blocked`, and `Unknown` are separate. Only feasible candidates can be selected.
- The confounded scenario's initial observations are identical across policy arms and their declared costs count against the total campaign budget.
- Experiment cost and resource values are relative integer units, not currency or lifecycle costing.
- Quality margin is a toy constraint. A positive margin means synthetic conformance within this fixture only.
- Full-rank factor excitation is a geometric property, not proof of causal identifiability.
- A successful synthetic campaign does not establish physical manufacturability, product conformity, defect detection, production yield, safety, savings or cross-process transfer.

## Initial acceptance tests

Unit tests cover deterministic manifests and replay, all 30 scenario/policy combinations, feasibility gating, budget compliance, missing measurements, altered receipt data, falsified cost/conformity claims, blocked-candidate injection, proxy/reference discrepancy and factor-design rank. Tests have been authored but must be run against an exact commit before any pass is claimed.
