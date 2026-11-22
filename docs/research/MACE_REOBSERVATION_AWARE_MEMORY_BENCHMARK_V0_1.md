# MACE Re-observation-Aware Memory Benchmark v0.1

**Snapshot:** 2026-11-22  
**Status:** Research protocol and prototype module. No measured performance gain or SOTA claim.

## Research thesis

An agent should not retain every useful observation as a fact to inject into future context. Memory competes with live perception, tools, retrieval latency, staleness and privacy.

Separate what must be learned from what can be re-observed:

- **Current state:** often cheap to query again and dangerous to cache indefinitely.
- **Stable facts:** retain a compact fact or source locator, whichever has higher net value; unlike mutable current state, a stable fact need not have a short snapshot expiry.
- **Learned rules and procedures:** knowledge acquired through experience that may not be directly visible in the next environment state.
- **Preferences:** durable user- or principal-specific constraints, subject to scope and expiry.

The memory decision is therefore not binary retain/forget. It is a choice of representation:

1. **PersistKnowledge** — keep a learned rule, procedure or preference.
2. **StoreLocator** — preserve a verified way to query current state at use time.
3. **CacheSnapshot** — retain a bounded snapshot because a fresh observation is unavailable, too slow or too costly before the decision.
4. **Discard** — do not spend memory or tool budget on information with non-positive expected net value.

This operationalizes a key observation in Anthropic's November 6, 2026 report, [The components of agent memory](https://www.anthropic.com/research/components-of-agent-memory): their Pokémon Red experiment found that a smaller memory focused on strategy and tested rules could outperform a large archive of mutable game state; the same report found that a fixed failure-only recording instruction reduced the memory benefit for stronger models. Those are vendor-reported experiments, not direct MACE comparisons.

## Prototype implementation

The broca-mace-consolidation reobservation module:

- consumes only an opaque, independently verified future-use prediction and a separately verified re-observation profile;
- binds event ID, evidence digest, exact scope, as-known snapshot, source-profile digest, locator digest, source availability, latency, cost and snapshot expiry;
- rejects identity mismatch, expired verification, malformed source profiles and forged receipt bindings;
- accounts for re-observation cost and latency-to-deadline as well as context token cost;
- does not execute tools or authorize source access;
- emits a deterministic, digest-bound representation plan.

The source-availability verifier is an interface. The checked-in tests use fixture receipts; production must bind calibrated availability probability, failure consequence, latency, cost, locator scope and validity to Mycelix source profiles or Symtropy environment observations, with independent issuer verification.

### Principal, audience and purpose binding

Every re-observation proposal and plan now carries a typed `MemoryAuthorityScopeV1` containing opaque digests for:
- principal identity;
- capability/authority receipt;
- audience/recipient set;
- exact resource or tenant/project boundary;
- policy revision;
- memory purpose (`PersonalPreference`, `PrivateProjectKnowledge`, `SharedTeamKnowledge`, `PublicKnowledge`, or `EnvironmentState`).

The proposal digest and plan digest bind this scope. The injected independent source verifier receives the expected scope and must reject a mismatch; the planning policy repeats exact-scope equality before emitting a plan. This prevents a personal preference, private project note, shared team rule and public fact from being treated as interchangeable because their content or task regime matches.

This follows the principal/capability, resource, audience and purpose separation used in Mycelix's effectuation contracts. It does **not** implement a new policy engine or grant authority: production must resolve the capability and current policy revision through Mycelix, bind the resource/audience and recheck revocation at use. The fixture verifier only tests the contract's shape and equality behavior.

## Decision rule

For a predicted future-use value V:

- Snapshot net value: V minus (snapshot tokens multiplied by token cost).
- Locator net value: V × P(success) minus (1 − P(success)) × failure penalty, re-observation cost, latency penalty and locator-token cost.

A locator is eligible only when it can return before the declared decision deadline, its independent receipt remains valid strictly beyond that deadline, and its calibrated success probability justifies the expected failure penalty. To avoid optimistic quantization, the implementation rounds expected positive value down (and negative value toward negative infinity) while rounding expected failure loss up to the next micro-unit; zero-probability failure cost remains exactly zero. This direction is deliberate: integer rounding must not make an action appear safer or more valuable than its unquantized expectation. Its expected net value is `V × P(success) − (1 − P(success)) × failure_penalty − query_cost − latency_cost − locator_context_cost`. A receipt expiry equal to the deadline is expired because validity intervals are half-open. The profile and decision must use the exact same `(valid_at, known_at)` snapshot; advancing knowledge time requires re-verification, so a later correction or revocation cannot be hidden by replaying an old availability receipt. A receipt expiry equal to the deadline is expired because validity intervals are half-open. A current-state snapshot is eligible only when its verified validity interval covers the decision deadline. Stable facts may be retained as knowledge without a transient-state expiry, or represented by a cheaper verified locator. Learned rules, procedures and preferences are not replaced with a live-state locator merely because some source can be queried.

This is a per-item representation decision. A future global optimizer should jointly allocate context tokens, re-observation calls, latency and privacy budgets across a candidate set; it must preserve the exact salience selector as a matched baseline.

## Benchmark tracks

1. **Mutable visible state:** environment exposes current position, inventory, UI state, service status or configuration cheaply. Prefer a locator over stale snapshot injection.
2. **Learned strategy:** a tested rule is not visible in the current state. Retain it with exact evidence links.
3. **Intermittent source:** sensor/API is unavailable or delayed. Cache only while the snapshot's verified freshness interval covers the action deadline.
4. **Expensive observation:** compare the cost and latency of querying again with the context cost and stale-use risk of caching.
5. **Source correction/retraction:** profile or source receipt changes; invalidate the locator/snapshot plan and recalculate as-known eligibility.
6. **Model heterogeneity:** future-use prediction and selected representation must remain bound to the target model.
7. **Adversarial profile:** false availability, fabricated locator, stale cache expiry, overclaimed low query cost, source-scope mismatch and receipt replay.

## Baselines

- Store everything / full transcript.
- Frequency-first, recency-first and failure-only retention.
- Predictive salience without re-observation cost.
- Always re-observe mutable state.
- Always cache mutable state.
- Re-observation-aware representation selection.
- Oracle representation choice as an evaluator-only upper bound.

All systems receive the same environment observations, tools, memory-token budget, tool-call budget and decision deadlines.

## Metrics

Report separately:
- downstream utility and decision regret;
- stale-state activation after snapshot expiry;
- redundant snapshot tokens for cheaply re-observable state;
- calibrated re-observation success probability, expected failure loss, actual success rate and re-observation-to-decision latency;
- learned-rule retention and later task success;
- locator failure / stale locator use;
- context tokens, tool calls, query cost, storage and CPU;
- correction/retraction propagation latency;
- per-family and worst-family metrics.

Do not combine governance failures with utility into one weighted score.

## Initial qualification targets

- Zero locator use when the locator receipt is wrong-scope, expired at or before the decision deadline, source-retracted, bound to an older knowledge snapshot, or bound to a different principal/capability/audience/resource/purpose/policy revision.
- Zero snapshot activation after its verified expiry.
- At least 95% of high-value, non-reconstructible held-out rules retained under the declared budget.
- At least 90% fewer snapshot tokens for a fixture where state is cheaply re-observable, without a downstream utility regression.
- On an exact, matched Microsoft Research Memora reproduction, first fall within 1 percentage point of its published LoCoMo/LongMemEval results, then target +2 points on a held-out extension without a higher memory-token budget.
- Positive paired held-out utility lift over predictive-salience-only, with cluster-bootstrap 95% lower bound above zero.
- At least five independent seeds and held-out environment/task families before any external claim.

These are target thresholds, not current results. They should be recalibrated after the reference environment and tool-cost units are frozen.

### External reference target: Microsoft Research Memora

Microsoft Research reports 86.3% LoCoMo and 87.4% LongMemEval for its separate Memora representation, with about half as many memory entries as Mem0 and up to 98% token savings versus full-context baselines in its evaluated settings. These are protocol-specific research-report results, not scores from a shared public leaderboard. MACE should first reproduce the exact backbone, judge, task split, context/token accounting and memory API; then target at least +2 percentage points on a held-out extension with no higher memory-token budget. On mutable-state episodes, the additional target is at least 90% fewer retained snapshot tokens than store-everything without a downstream utility regression. Do not confuse this system with the separate ACL Memora/FAMA benchmark.

## External benchmark crosswalk — WorldMemArena

[WorldMemArena (2026)](https://arxiv.org/abs/2605.29341) and its [benchmark site](https://worldmemarena-mem.github.io/) provide a useful external transfer target: 400 multi-session multimodal tasks evaluated through an observable **write → maintain → retrieve → use** lifecycle, spanning Lifelong Evolution and Agentic Execution. The paper reports that storage quality alone does not guarantee downstream performance, that append-only memory maintenance remains a weakness, and that systems often lose visual, spatial and procedural detail when observations are reduced to captions.

Map the MACE representation factor to the benchmark lifecycle without replacing its task definitions or official evaluator:

| WorldMemArena stage | MACE representation experiment | Required evidence |
|---|---|---|
| Write | Choose persist-knowledge, locator, bounded snapshot or discard | Original observation/action evidence, exact source receipt, chosen representation and reason |
| Maintain | Re-evaluate after source changes, corrections, expiry and environment transitions | Bitemporal update/retraction chain; no stale resurrection |
| Retrieve | Resolve evidence or locator under the same memory/context/tool budget | Candidate IDs, source/scope checks, retrieval and re-observation latency |
| Use | Decide or act using the revalidated evidence | Actual environment outcome, action/evidence trace, correctness and utility at consumption |

The key multimodal control is **preserve evidence, not just captions**: a visual memory candidate should retain a stable pointer/digest to the original screenshot, frame, spatial relation or action/feedback trace, while a text description remains an optional index. A locator may re-observe mutable screen/world state; a learned procedure should preserve the tested action sequence and its outcome evidence. Do not allow a caption or a model-generated tool claim to become the evidence source.

External comparison protocol:
1. Keep WorldMemArena's official task split, action contract and evaluator unchanged. Do not tune thresholds on its held-out tasks.
2. Compare long-context/full-history, fixed summary/snapshot, a strong conventional external-memory baseline, salience-only retention, always-reobserve, and MACE representation selection.
3. Match backbone, tool permissions, prompt/context tokens, memory bytes, call budget and decision deadline; separately report unavoidable harness overhead.
4. Report stage-wise failure attribution and full task utility, plus stale-state use, visual evidence grounding, procedure transfer, correction recovery, token/storage footprint, calls, latency and cost.
5. Report separate per-domain/worst-family outcomes and paired uncertainty at the independent task/world unit. Never substitute synthetic MACE utility or internal fixture success for the official benchmark score.

This mapping is an evaluation plan, not an assertion that a WorldMemArena adapter or run exists. It makes the central research claim falsifiable: representation choice should improve future action under matched budgets, not merely improve the memory artefact.

### WorldMemArena crosswalk: representation choice across the action-world lifecycle

WorldMemArena defines four observable memory stages—write, maintain, retrieve and use—across 400 multi-session multimodal tasks. MACE's re-observation factor should map as follows:

| WorldMemArena stage | MACE representation experiment | Required receipt / metric |
|---|---|---|
| Write | Persist knowledge vs store locator vs cache snapshot vs discard | Source/evidence digest, target-model prediction, representation plan digest, tokens and expected costs |
| Maintain | Refresh, supersede, revoke or expire a locator/snapshot after source change | Current Mycelix lifecycle receipt; stale activation and correction-propagation latency |
| Retrieve | Retrieve a compact locator or learned rule, not an obsolete full-state copy | Candidate IDs, canonical registry resolution, evidence provenance and retrieval latency |
| Use | Revalidate exact `(valid_at, known_at)`, scope, source revision and deadline at decision consumption | Freshness-at-use, correct-on-time coverage, regret, stale/wrong-scope use and abstention |

The paired experiment should clone the same world state and current observation, then randomize these memory arms within world clusters:
- full-history / store-everything;
- predictive salience only;
- always re-observe mutable state;
- always cache mutable state;
- risk-adjusted representation selection;
- evaluator-only oracle representation as an upper bound.

Keep original screenshots, frames, spatial state and tool feedback as the evidence source; captions are candidate interpretations, not ground truth. Report write, maintain, retrieve and use failures separately. Preserve task/domain splits and score paired outcomes at the world-cluster unit. The MACE representation comparison must not claim to reproduce WorldMemArena until its official task loader, evidence artifacts, answer/action evaluator and resource budgets are matched.

## Limitations and next steps

- The module uses verifier-declared availability probability, failure penalty, normalized costs and a decision deadline; real deployments need independently calibrated tool latency, API availability, energy/cost and failure distributions. Stable-fact retention and mutable-state caching deliberately have different validity rules.
- Receipt authenticity is delegated to an injected verifier; fixture receipts do not establish production trust.
- The current selector makes per-item representation choices; a joint token/call/latency optimizer remains future work.
- The current module does not store or mutate persistent memory. It returns a plan for a governed consumer.
- The benchmark must include live state and learned strategy in the same dependent-session task, otherwise re-observe can win trivially by repeatedly querying an oracle.
- The raw Session-A ingress → future-use calibration → source-profile receipt → authority-scoped representation-plan path has an authored evaluator integration test; it remains fixture-backed and unexecuted until a Rust runner is available.

Next: connect the plan to receipt-qualified consolidation proposals, add environment-backed locators, and benchmark integrated Session-A learning → Session-B retrieval/re-observation → Session-C correction.


### Conservative fixed-point cost accounting

Expected success value is rounded conservatively: positive expected value is rounded down and negative expected value toward negative infinity. Expected failure loss is rounded up to the next micro-unit, while exact zero remains zero. This prevents fractional expected failure costs from disappearing and making an unreliable locator look artificially cheap. The re-observation module now contains **29 authored regression tests**, including sub-micro failure-risk and negative-value rounding controls. These are authored tests, not executed passes.


## Joint portfolio optimization

The per-item re-observation policy now exposes all individually valid representation alternatives to a portfolio allocator. The allocator chooses at most one option per event under shared context-token, re-observation-call, total latency-work, estimated-cost and expected-failure-loss budgets.

The exact bounded solver maximizes total per-item net value using deterministic branch-and-bound, with a 24-event-group and 2,000,000-node cap. It fails closed if the search budget is exhausted. A highest-individual-net-value greedy baseline is retained for matched comparison. The result is a digest-bound proposal, not a write or permission to query a source.

The resource model deliberately sums latency work rather than claiming parallel wall-clock latency. Shared query batching, parallel scheduling, privacy-exposure budgets and dynamic replanning remain outside v0.17.


### Trust-boundary tightening

Portfolio options are now admitted only when their representation-plan digest matches an option independently re-derived by `enumerate_memory_representation_options` from the same verified salience prediction, verified source profile and policy. A caller cannot mutate a plan's cost/disposition, recompute its digest, and present it as an eligible choice. The portfolio constructor still verifies the complete source/prediction/snapshot/authority binding before creating an option.

The exact optimizer assumes the supplied per-item `net_value_micros` are additive across distinct evidence payloads. It now rejects a byte-identical evidence digest being assigned multiple event IDs, but semantically overlapping facts with different payloads can still have interacting or diminishing value. This prototype is not a submodular/interaction-aware optimizer. A credible next version needs independently qualified marginal-value groups or a portfolio-level evaluator that measures the value of combinations against the same no-memory baseline; it must not simply sum correlated single-item predictions and call the result realized utility.
