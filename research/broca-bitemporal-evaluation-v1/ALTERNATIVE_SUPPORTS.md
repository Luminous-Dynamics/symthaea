# Broca Alternative-Support Truth Maintenance: Research Protocol

**Status:** proposed next architectural step; not implemented and not evaluated. This document identifies a semantic gap in the current derivation-reference graph and defines falsifiable cases before changing its API.

## Executive finding

The reference graph models one `DerivedFactInput` as one derivation record with a conjunctive set of dependencies. That is appropriate for a *proof instance*. It does not yet express the full truth-maintenance distinction between:

- **Claim/proposition:** the semantic statement being asserted (for example, `build-42.ready = true` over a validity interval).
- **Justification/derivation:** one particular proof of that claim.
- **Dependency:** a premise required by one justification.

Dependencies within a justification are conjunctive (AND). Different justifications for the same claim are alternatives (OR).

This distinction matters operationally. Suppose `ready` is justified either by `E1 AND E2` or independently by `E3`. Retracting `E1` should invalidate the first justification, but it should not make the claim unsupported while the `E3` justification remains valid. A downstream claim that depends on the *proposition* should remain current too. The existing API can represent separate derivation records with unique IDs, but its dependency references point to derived-fact IDs rather than a grouped proposition identity. Therefore alternative support is not yet a first-class semantic guarantee.

This is not a defect in the narrow promise that one derivation's descendants are invalidated. It is a boundary on treating the current module as a complete truth-maintenance system.

## Research grounding

Classic truth-maintenance systems explicitly track reasons for beliefs and support assumption revision; see Doyle's *A Truth Maintenance System* (1979/1981) and de Kleer's assumption-based TMS (1986):

- https://doi.org/10.1016/0004-3702(79)90008-0
- https://doi.org/10.1016/0004-3702(86)90080-9

Incremental Datalog materialisation research treats alternative derivations as a central deletion problem. The DRed algorithm can over-delete and then rederive; the Backward/Forward (B/F) algorithm targets workloads where facts have many alternative derivations and reported orders-of-magnitude improvements on some inputs:

- Motik et al., *Incremental Update of Datalog Materialisation: the Backward/Forward Algorithm*, AAAI 2015: https://ojs.aaai.org/index.php/AAAI/article/view/9409
- Hu, Motik & Horrocks, *Optimised Maintenance of Datalog Materialisations*, AAAI 2018: https://ojs.aaai.org/index.php/AAAI/article/view/11554
- Zhao et al., *Incremental Maintenance of DatalogMTL Materialisations*, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/39025

These are algorithmic references, not directly comparable Broca scores. They establish that alternative support, deletion, temporal rules, and the cost of rederivation are known research problems. Broca's opportunity is to combine them with evidence provenance, trusted-time/policy constraints, agent-facing explanations, and reproducible end-task evaluation—without claiming the underlying incremental-maintenance problem is new.

## Proposed model

Represent a proposition independently from the justifications that support it:

- `PropositionId -> semantic claim + temporal interval`
- `JustificationId -> PropositionId + rule ID/version + a set of dependencies`
- Dependencies refer to immutable source events or other **PropositionIds**, not to one particular justification record.
- One proposition may have zero, one, or many current justifications.
- A proposition is supported at a point in time if at least one currently valid, temporally applicable justification has all required premises supported.
- A retracted source invalidates the justifications that require it. A dependent proposition is invalidated only when it no longer has any valid alternative justification for the queried interval.
- Downstream derivations depend on proposition truth, not a particular proof ID. Otherwise one proof's loss can incorrectly cascade despite a surviving proof.
- Invalidated justifications remain in the audit trail. A later correction or new evidence adds a new justification; it does not mutate the old proof into a different proof.
- Two propositions with opposing values remain a conflict if both are supported under the active authority policy. Alternative support must not become an implicit conflict-resolution policy.
- The system must not permit an unsupported cycle to keep itself alive. For the first reference implementation, prohibit proposition-dependency cycles; recursive rules require a separately specified least-fixed-point/grounding semantics.

### Temporal semantics

Justification support is time-indexed. If alternative justifications cover intervals `[a,b)` and `[b,c)`, the proposition may be supported continuously over `[a,c)` even though no single justification covers all of it. A single interval field on a proposition is therefore insufficient for general support aggregation unless the system either:

1. stores a normalized union of supported intervals; or
2. evaluates support for a specific `valid_at` query and derives the supported interval set from justification intervals.

The first prototype should return supported interval segments and a witness set for each segment. It must preserve half-open interval behavior and report unknown for unsupported time segments.

## Independent oracle

The benchmark oracle must not call the implementation's reverse-index traversal. Given a finite acyclic justification hypergraph:

1. For each query time, determine eligible source events and policies under the query's trusted-known-at cutoff.
2. Mark a justification supported only if all source dependencies are admitted/current and all proposition dependencies are supported at that time.
3. Mark a proposition supported when at least one eligible justification is supported.
4. Propagate invalidation from source retraction or rule-version retirement until a fixed point.
5. Compare the implementation's support state, justification state, supported intervals, and witness provenance with the oracle after every event prefix.
6. Validate by full recomputation after each mutation. This slower oracle serves as a semantic control for the incremental implementation.

For the initial finite-DAG suite, the oracle can use independent iterative scanning; it must not read implementation-private indexes. Add a separate reference implementation for cyclic rules only after a formal fixed-point semantics is chosen.

## Required diagnostic scenarios

The companion `alternative_support_cases.json` file captures eight small symbolic cases with expected claim state. They must become unit/property tests before alternative support is integrated into Broca's worker.

| Case | Mutation | Required result |
|---|---|---|
| A01 | Retract one of two independent supports | Claim remains supported; dependents remain current |
| A02 | Retract the last remaining support | Claim becomes unsupported; dependent claims invalidate |
| A03 | One proof requires E1 AND E2; another requires E3 | Retracting E1 removes only the first proof |
| A04 | Retire a rule version that supplied one of two proofs | Claim remains supported by the other current rule |
| A05 | Two different values both have support | Conflict remains explicit; no winner by insertion order |
| A06 | Late evidence supports a historical interval | Current/as-known queries obey both valid time and trusted-known-at |
| A07 | Alternative support intervals are adjacent or overlapping | Supported intervals are unioned without gaps or half-open boundary errors |
| A08 | All leaf supports are removed and a dependent cycle is attempted | No unsupported self-sustaining cycle; cycle rejected or quarantined |

## Metrics and promotion gates

Report these independently of final answer QA:

- **Alternative-support recall:** fraction of gold valid support alternatives preserved after each mutation.
- **Over-invalidation rate:** fraction of still-supported propositions or dependents made unsupported by the implementation.
- **Under-invalidation rate:** fraction of propositions left supported despite having no valid justification.
- **Justification precision/recall:** correctness of the active proof set, including dependency edges and rule versions.
- **Temporal support accuracy:** exact supported interval segments and correct unknown/conflict state at each query time.
- **Witness completeness:** each supported proposition has at least one complete admissible support witness; each invalidated proposition has no eligible witness.
- **Update work:** justifications visited, propositions re-evaluated, storage growth, update latency, and full-recompute work avoided.
- **Recovery parity:** identical support state, intervals, witnesses, and digest after restart, replay and index rebuild.
- **End-task utility:** paired task-completion, reliability, answer correctness, latency, tokens and cost under fixed reader/context budgets.

Hard semantic gate: zero false unsupported claims, zero unsupported claims that remain current, zero arbitrary conflict winners, and exact oracle parity on deterministic generated DAGs. Also report useful-answer coverage; an implementation returning unknown for every query must not pass.

## Comparison strategy

Do not claim "better than SOTA" from the synthetic fixtures. Compare algorithms using the same generated hypergraphs and update traces:

- **Naive descendant invalidation** (the current reference behavior for one derivation): simple semantic baseline; likely over-invalidates when the semantic claim has alternative proofs.
- **Full recomputation**: correctness oracle; high work, simple semantics.
- **Counting maintenance**: include for suitable nonrecursive rule classes.
- **DRed / B-F / FBF**: implement or use a documented compatible reference for alternative-derivation deletion behavior; disclose rule-language differences.
- **Temporal incremental maintenance**: add DatalogMTL-style workloads only after the rule and interval semantics are specified.
- **Broca support graph**: evidence/policy-aware alternative support, auditable witnesses, trusted time, and an agent-facing API.

Vary graph size, fan-out, number of alternative proofs per proposition, overlap in premises, proportion of deleted supports, temporal interval fragmentation, and fraction of update-affected propositions. Report curves, not a single speedup. Algorithm performance depends on rule structure and update characteristics, so a mixed workload is essential.

## Implementation sequence

1. Keep `memory_derivation.rs` as a single-derivation reference until its existing exact-head tests are executed.
2. Implement a small separate `memory_supports.rs` prototype using proposition IDs and justification IDs. Do not silently redefine current dependency semantics.
3. Test the eight cases against an independent full-recompute oracle and generated DAGs.
4. Add interval-segment support and policy/time cutoffs.
5. Benchmark naive invalidation, full recomputation and a compatible incremental baseline.
6. Only then design a worker adapter and migration strategy. The existing worker must not silently change meaning while support semantics evolve.

**Non-claims:** This protocol is not an implementation, benchmark run, formal proof, durable transaction protocol, or SOTA result. Its purpose is to pin the semantic gap before changing production-facing interfaces.
