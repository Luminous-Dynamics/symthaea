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

- Zero locator use when the locator receipt is wrong-scope, expired at or before the decision deadline, source-retracted or bound to an older knowledge snapshot.
- Zero snapshot activation after its verified expiry.
- At least 95% of high-value, non-reconstructible held-out rules retained under the declared budget.
- At least 90% fewer snapshot tokens for a fixture where state is cheaply re-observable, without a downstream utility regression.
- Positive paired held-out utility lift over predictive-salience-only, with cluster-bootstrap 95% lower bound above zero.
- At least five independent seeds and held-out environment/task families before any external claim.

These are target thresholds, not current results. They should be recalibrated after the reference environment and tool-cost units are frozen.

## Limitations and next steps

- The module uses verifier-declared availability probability, failure penalty, normalized costs and a decision deadline; real deployments need independently calibrated tool latency, API availability, energy/cost and failure distributions. Stable-fact retention and mutable-state caching deliberately have different validity rules.
- Receipt authenticity is delegated to an injected verifier; fixture receipts do not establish production trust.
- The current selector makes per-item representation choices; a joint token/call/latency optimizer remains future work.
- The current module does not store or mutate persistent memory. It returns a plan for a governed consumer.
- The benchmark must include live state and learned strategy in the same dependent-session task, otherwise re-observe can win trivially by repeatedly querying an oracle.
- The raw Session-A ingress → future-use calibration → source-profile receipt → representation-plan path has an authored evaluator integration test; it remains fixture-backed and unexecuted until a Rust runner is available.

Next: connect the plan to receipt-qualified consolidation proposals, add environment-backed locators, and benchmark integrated Session-A learning → Session-B retrieval/re-observation → Session-C correction.
