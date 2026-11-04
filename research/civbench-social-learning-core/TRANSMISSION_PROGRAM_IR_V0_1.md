# Bounded transmission-program IR v0.1

**Status:** executable Rust data-program interface; no evolutionary/LLM program search is implemented yet.  
**Implementation:** TransmissionProgram and TransmissionScoreWeights in src/adaptive_transmission.rs.  
**Design goal:** make the policy object inspectable, deterministic, resource-bounded and testable before adding program generation.

## 1. Why a program IR?

Fixed network policies specify possible senders and receivers. A state-aware program additionally conditions message selection and routing on the current recipient and collective state. The 2026 Colas et al. paper represents such protocols as stateful executable programs and uses LLM-guided evolutionary search. CivBench should reproduce the inspectable program boundary before introducing generated code.

The v0.1 Rust representation is intentionally smaller than the paper's Python protocol API. It is a bounded scoring program—not arbitrary code, not a general scripting language and not the paper's evolutionary algorithm.

## 2. Program object

A TransmissionProgram contains:
- immutable program ID and positive version;
- minimum recipient prerequisite match in thousandths;
- optional requirement that the output be novel to collective knowledge;
- maximum messages per step;
- maximum messages emitted by any one sender per step;
- non-negative integer weights for immediate executability, prerequisite match, collective novelty, downstream unlock potential and sender knowledge.

The score is an integer expression with stable tie-breaking over sender, receiver and recipe IDs. This makes the same input state and program produce the same schedule without floating point ranking drift.

The built-in frontier-v1 program is parameterized to reproduce the existing StateAwareFrontier heuristic's ranking. The unit test compares graph, final agent states and the entire metric history for the same seed/configuration. That is a conformance test between two implementations of the same heuristic, not proof that the heuristic is optimal. The simulator uses domain-separated streams and identity-keyed draws: individual discovery priority/outcome is keyed by (seed, step, agent, recipe), and acquisition is keyed by (seed, step, sender, receiver, route-slot), excluding payload identity. Thus content-shuffle ablations preserve route-slot acquisition draws. Candidate eligibility and top-k selection can still differ as inventories evolve, and distinct routes have no matched acquisition draw; this is stronger paired control, not universal action-level pairing.

## 3. Allowed observations and forbidden authority

The current IR reads only:
- graph neighbours;
- sender-held recipe IDs;
- recipient-known elements and recipes;
- collective-known elements;
- recipe input/output identifiers;
- recipe dependency depth/descendant count;
- message/attention limits in the supplied config.

It does not receive the hidden ground-truth recipe graph beyond the task rules required by the simulator, inspect the scorer's future outcomes, modify any agent's state directly, create messages outside the graph, bypass consent/authorization, or exceed budgets.

For real cross-project use, the event schema's recipient_state_basis and consent envelope must authorize the specific input features before a policy sees them. The current synthetic runner uses full simulation state for an upper-bound mechanism benchmark; it is not a privacy-preserving deployment.

## 4. Fail-closed validation

- Program ID must be non-empty.
- Version must be positive.
- Prerequisite threshold must be within 0..=1000.
- Per-step and per-sender message caps must be positive and no greater than the run's configured global message budget.
- Every score weight must be bounded by MAX_WEIGHT; at least one must be non-zero.
- Unknown feature names or new action types are not supported in v0.1.
- No arbitrary functions, dynamic imports, native code, shell commands or network calls are accepted.

The outer Adaptive Transmission Event binds an exact program digest and version. The current Rust IR does not itself canonicalize/hash a serialized program; that is an integration-layer responsibility and must be implemented before treating the digest as a runtime security boundary.

## 5. Program search plan

Begin with transparent parameter search:
1. Freeze the task family, candidate parameter grid and score metric.
2. Evaluate programs on training DAG families and seed set.
3. Select candidates using a separate validation family/seed set.
4. Evaluate one locked program on a test DAG family and seed range.
5. Compare against StateAwareFrontier, no-social, fixed-ring, random network/content and both content-shuffle ablations.
6. Keep all candidate programs, failed evaluations, configuration, seed schedule and output digests.

Only after the grid baseline is stable should evolutionary search or an LLM propose new bounded programs. Generated candidates must compile into the same typed IR; they must not emit arbitrary executable code. LLM calls, prompt/model version, token/compute budget, candidate lineage and repair attempts must be recorded.

## 6. Required program-search metrics

- held-out collective result coverage and depth;
- per-agent knowledge and complementarity;
- message/attention/compute cost;
- redundant messages and unexecuted/dormant acquired rules;
- fairness/lineage concentration across senders/recipients;
- false adoption and recovery under corrupted content;
- generalization across task families, population sizes and graph densities;
- privacy cost under centralized, coarse-signal and recipient-pull modes.

Do not select on the final test family. Do not use only one scalar fitness: preserve a Pareto frontier across performance, communication cost, robustness, autonomy and disclosure.

## 7. Proposed promotion gates

These are engineering targets, not achieved results:
- frontier-v1 exactly matches StateAwareFrontier on same seed/configuration;
- all generated programs pass structural, resource-bound and deterministic replay tests;
- at least one parameterized program beats the strongest fixed/random baseline on a pre-registered validation set;
- selected program improves on the locked unseen test family with uncertainty excluding zero under matched message/attention budgets;
- content-shuffled controls reduce or remove recipient-state advantage where expected;
- no privacy, consent or scope violations are permitted, regardless of performance.

A better score on one 14-recipe DAG family is not evidence of a universal coordination algorithm, human-group improvement, or civilizational superiority.
