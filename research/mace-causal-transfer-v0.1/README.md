# Broca MACE — causal experience transfer reference v0.1

**Status:** synthetic research reference only. This file and its accompanying offline package do not establish an ML benchmark result, production capability, physical-world efficacy, or SOTA improvement.

Parent programme: [#9516 — qualify causal experience transfer across interdependent agent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516).

## Research shift

Move from **remembering which action preceded success** to **preserving a qualified transition model**:

```text
interaction trace
  -> outcome reconciliation against environment state
  -> observational / interventional evidence distinction
  -> regime-scoped transition estimate
  -> current-objective and constraint evaluation
  -> transfer, acquire evidence, or abstain
```

The governing distinctions are:
- temporal precedence is not causation;
- observational correlation is not an identified intervention effect;
- a causal effect is not itself a policy;
- a valid transition in one regime is not automatically valid in another;
- a tool-reported success is not proof that the environment changed;
- immediate reward is not lifecycle utility.

## Executed synthetic reference

The standalone package contains a deterministic reference policy, digest-bound experiment/outcome receipt payloads, separated reader/evaluator fixtures, and Go tests. It exercises six cases:

| Case | Failure/control | Expected qualified behaviour |
|---|---|---|
| MACE-C01 | Confounded action/outcome association | Do not promote observational correlation; choose baseline using intervention evidence |
| MACE-C02A | Productivity objective | Apply the same qualified effect to the current productivity objective |
| MACE-C02B | Safety objective, same causal evidence | Recompute policy; do not retain the previous objective |
| MACE-C03 | Historical and target regimes differ | Abstain rather than transfer an out-of-domain transition |
| MACE-C04 | Tool reports success; environment outcome disagrees | Reject the effect claim and abstain |
| MACE-C05 | Immediate gain is outweighed by delayed cost | Optimize the declared lifecycle objective |

The reader-input type is separate from evaluator-only utility/expected-result labels. A test serializes the reader input and rejects any leaked oracle/gold fields. Design and outcome receipt payloads are bound to event, regime and action identities by SHA-256; mutation and identity-mismatch tests must fail closed.

### Local execution record — 2026-11-10

- 13/13 top-level Go test functions passed.
- 6/6 synthetic scenario subtests passed.
- 128/128 deterministic confounder-history variants passed.
- `go test -race -count=1 ./...`: PASS.
- `go vet ./...`: PASS.
- Statement coverage: 79.3%.
- Go 1.23.2, standard library only.

In these six designed fixtures, a no-memory fixed baseline matches the evaluator's best-action labels in 5/6 cases; the intentionally weak immediate-productivity association policy matches in 1/6. The qualified reference matches its declared selection/abstention semantics in 6/6 (four selections and two abstentions). These are fixture counts, not statistical benchmark results or generalization measurements.

## External research alignment

MACE should complement, not replace, external suites:
- [MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html): interdependent multi-session Memory-Agent-Environment tasks.
- [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html): long-horizon agent trajectories, causal and objective-aware memory; AMA-Agent reports 57.22% average accuracy in its own protocol.
- [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/): active application of memory to tool choice and argument grounding.
- [UltraHorizon (ICML 2026)](https://proceedings.mlr.press/v306/luo26ai.html): partially observable, ultra-long-horizon exploration.

These datasets, answer models and scoring protocols are not directly comparable. Adapters should use their official harnesses, retain separate result tables, and freeze backbone, budgets and evaluator within paired comparisons.

## Proposed matched baseline ladder

B0 no persistent memory; B1 raw full-history ceiling; B2 episodic summary/latest-success; B3 BM25/dense/hybrid retrieval; B4 objective-aware state memory without causal links; B5 causal graph without provenance/regime gates; B6 outcome-reconciled, regime-scoped causal memory; B7 B6 plus current-objective recomputation, proof-carrying action rationale and abstention; B8 oracle structural causal model (diagnostic upper bound only).

Measure dependent-chain success, cumulative utility/regret, treatment-effect error where identifiable, false causal promotion, regime applicability, outcome reconciliation, stale-transition reuse, appropriate abstention, full memory bytes, model/tool calls, p50/p95/p99 latency and lifecycle cost. Report a vector scorecard and quality-cost Pareto curves; do not let average accuracy compensate for unsafe transfer.

## Proposed qualification targets — not achieved

- 100% deterministic oracle agreement on finite structural/lifecycle fixtures.
- Zero observational-only causal promotions in the conformance suite.
- Zero accepted mutated, wrong-action, cross-request or wrong-regime receipts.
- Zero action effects accepted without independent outcome reconciliation.
- At matched budgets, target +10 percentage points dependent-chain success over the strongest non-causal memory baseline, with paired cluster-aware 95% confidence intervals excluding zero.
- Under held-out regime shift, target at least 50% lower worst-group regret than unguarded causal-graph reuse, without hiding failure through excessive abstention.
- At 10x memory reduction, stretch target of at least 99% decision agreement on a frozen proof-sensitive panel.

These are project decision thresholds, not industry standards. Preregister splits, sample size, primary outcomes and stopping rules before model runs.

## Boundaries

The current reference consumes structured evidence; it does not infer causal graphs from arbitrary text, estimate uncertainty intervals, model interference, prove transportability, or run a real experiment. Digest consistency establishes byte integrity, not issuer authenticity or proof that randomization occurred. The synthetic design/outcome receipts are trusted inputs. Production would require authenticated design manifests, verified assignment, signed environment-side observations, causal-identification assumptions, and separate action authorization.

Rust/Cargo were unavailable in the execution environment. Go is a local reference only; it is not a proposed production dependency. No MemoryArena, AMA-Bench, Mem2ActBench, UltraHorizon, LongMemEval, MemDerive, StateMemBench or AML evaluation was run. No SOTA claim is made.

The accompanying downloadable research package contains the reference code, tests, protocol, results and separated JSON fixtures.