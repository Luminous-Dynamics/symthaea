# Broca MACE Transfer Bench v0.1

**Status:** Deterministic synthetic research harness. Not the official MemoryArena or AMA-Bench runner, not a physical-world causal discovery claim, and not a measured SOTA result.

**Programme owner:** [Symthaea #9516 — causal experience transfer across interdependent agent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516)

## Research question

Does memory acquired in Session A improve a dependent Session B action, while correctly refusing to transfer when the apparent lesson is confounded, out of scope, retracted, untrusted, unreconciled, or supported only by correlated repetitions?

This v0.1 slice isolates one narrow bridge: **qualified transition evidence → policy-relevant transfer**. It does not implement a general agent, train a causal model, or call an external model API.

## Run

The benchmark is a Rust workspace with two separate packages:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-policy
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator
```

No external service, GPU, database or model API is required.

## Build-graph boundary

- `policy/` is the candidate-policy crate. It contains only candidate-visible experience/query types, policy logic and decision receipts. It has no dependency on the evaluator and no `OracleTruth` type.
- `evaluator/` owns scenarios, hidden target utilities, causal labels, scoring and reporting. It depends on the policy crate, not vice versa.

This is a stronger compile-time boundary than placing the oracle in another Rust type in the same source file. The evaluator source remains openly inspectable in this prototype; **it is not a secret or independently operated evaluator**. For a serious challenge, retain the evaluator in a separate controlled repository or service and expose only a versioned candidate API.

## Scenario families

The deterministic corpus includes thirteen two-session cases:

1. Identified interventional effect transfer.
2. Confounded observational association decoy.
3. Regime shift.
4. Objective shift.
5. Source retraction.
6. Tool success without reconciled external state.
7. Untrusted / poisoned source.
8. Incomplete delayed-consequence horizon.
9. Correlated duplicate echo from one source family.
10. Qualified negative experience.
11. Irrelevant-update control.
12. Under-supported candidate.
13. Positive control in a shifted regime.

## Compared policies

- `no_memory`: fixed conservative fallback.
- `frequency_only`: aggregate all observed outcomes, ignoring scope and evidence governance.
- `scope_aware`: restrict to exact regime and objective, but does not qualify source or causal evidence.
- `causal_qualified`: require exact scope, intervention, pre-registration, reconciled outcome, trusted non-retracted source, confounder control, completed delayed horizon and at least two independent source families.

The oracle is evaluator-only and is not linked as a dependency into the candidate-policy crate.

## Metrics (separate axes)

- Decision accuracy, mean utility and mean regret.
- Abstentions and the explicit conservative fallback used for utility/regret.
- Causal claims and causal overclaims.
- Invalid-evidence reuse and unsupported-A reuse.
- Evidence-bound decisions and evidence-reference counts.

## What v0.1 can and cannot establish

The harness can regression-test whether transparent policy rules reject these specific synthetic failure modes. The causal-qualified policy is deliberately hand-authored against visible fixture structure. A zero-overclaim or high-accuracy result on these thirteen cases is **not** evidence of causal discovery, real-world transfer, general intelligence or SOTA.

It does not yet include:
- hidden external evaluator custody or sealed labels;
- randomized multi-seed worlds and source-/seed-held-out intervals;
- Session C delayed corrections or a full Session A → consolidation → Session B → correction chain;
- official MemoryArena / AMA-Bench adapters or licensed benchmark data;
- learned policies, model training or model-provider comparisons;
- token, tool-call, CPU/GPU, energy and p95/p99 lifecycle cost measurement.

## Next qualification gates

1. Move evaluator code and hidden labels to evaluator-controlled custody (#11816).
2. Generate source- and seed-held-out worlds with randomized task prevalence and confounders.
3. Add Session C corrections and connect the output of consolidation #9139 / PR #12731 to Session B transfer.
4. Add causal-certificate mutation tests and replayable prediction-before-action receipts.
5. Reuse canonical source/claim/scope/action-effect contracts from #9075, #8265, #9016 and #9490.
6. Add official benchmark adapters only after dataset license and evaluator contracts are verified.
7. Measure paired transfer, worst-group regret, abstention coverage and full lifecycle cost under matched budgets (#8427).

## External comparison portfolio

- MemoryArena: interdependent multi-session agent–environment transfer.
- AMA-Bench: long-horizon state/action/observation/tool trajectories.
- TimlMemBench: raw-native lifecycle, revocation, scope and restraint.
- LongMemEval-V2: long-horizon retrieval and temporal question answering.
- ConsolidationBench: consolidation decision quality and governance.
- #8427: matched cross-benchmark quality/cost frontier.

Do not pool percentages across these tasks. Reproduce each official protocol separately and publish exact backbone, prompt, budget, data/evaluator revision and cost/latency.

## Related owners

- [#9516](https://github.com/Luminous-Dynamics/symthaea/issues/9516): causal experience transfer / MACE programme.
- [#9139](https://github.com/Luminous-Dynamics/symthaea/issues/9139): consolidation governance / #12731.
- [#8427](https://github.com/Luminous-Dynamics/symthaea/issues/8427): lifecycle cost-quality frontier.
- [#9075](https://github.com/Luminous-Dynamics/symthaea/issues/9075): bitemporal source-event ledger.
- [#8265](https://github.com/Luminous-Dynamics/symthaea/issues/8265): claim identity versus similarity.
- [#11816](https://github.com/Luminous-Dynamics/symthaea/issues/11816): evaluator and oracle independence.

## Generated stress tranche (MACE v0.3 research)

A second evaluator binary adds **256 deterministic clusters × 13 scenario families = 3,328 generated cases**. It rotates source-family and event identities and varies identifiable positive-control effects. Each cluster includes confounding, regime and objective shifts, source retraction, unreconciled tool success, untrusted poison, incomplete delayed outcomes, correlated duplicates, event-identity aliasing, qualified negative experience, few-shot invariant transport and a mechanism-break negative control.

Run it independently:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_generated_stress
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_generated_stress
```

The candidate gate requires at least two unique event IDs as well as two declared source families; every repeated event ID is excluded from its qualified evidence set, even when assigned multiple family labels. The runner reports aggregate metrics for all policies and per-family metrics for both exact-scope causal memory and invariant transport. It also reports paired utility lift versus no-memory with a deterministic **cluster-bootstrap 95% interval**, resampling world clusters rather than individual scenarios, and a Session-B / Session-C known-retraction recovery control.

This is a generated synthetic stress distribution—not an official MemoryArena/AMA-Bench adapter, learned-policy train/test split, sealed external oracle or real-world causal discovery. Source-family IDs vary to expose correlation/identity failures, but independence is not cryptographically attested. Bootstrap intervals quantify variation over these generated clusters only; they do not establish external validity. Session-C corrections are applied as observed source-state updates before the next decision.

## Invariant transport experiment (MACE v0.3 research)

The candidate crate adds an `invariant_transport` policy alongside exact-scope `causal_qualified`. It may pool prior-regime experience only when the objective matches exactly, the target regime has evidence for both actions, a prior regime has at least two unique qualified events from two source families for each action, source/target families are disjoint, and per-action utility means remain within a fixed 0.12 tolerance.

The generated suite includes a few-shot invariant positive control and a mechanism-break negative control. Exact-scope policy should abstain on the positive few-shot case; invariant transport may transfer. It must abstain when action effects move beyond the tolerance or source families overlap.

This is a transparent empirical-invariance heuristic, not a general transportability proof. The 0.12 tolerance is an uncalibrated prototype parameter. Future work should use preregistered equivalence margins, uncertainty intervals, explicit causal assumptions and independent target-regime witnesses.
