# MATH-SEARCH-MEMORY v1

Status: draft contract for MATH-SEARCH-001A.

## Purpose

Symthaea must remember mathematical truth claims and mathematical search experience without confusing the two.

The contract therefore separates:

- **ResultMemory** — mathematical outputs with explicit evidence state; and
- **SearchMemory** — strategies, representations, costs, failures, timeouts, abandoned lines of attack, and useful artifacts.

The authoritative append-only campaign record remains the ResearchGraph. These memories are retrieval-oriented projections and may be compacted without deleting the underlying provenance.

## Authority laws

```text
SearchScore != EvidenceScore != FormalAuthority
Phi != truth
similarity != truth
retrieval frequency != truth
Timeout != false
Abandoned != disproved
NoProgress != impossible
```

A `ResultEpisode` may reference formal evidence, but this contract does not authenticate that evidence. Kernel/comparator authority remains in MATH-EVID/MATH-VERIFY.

A `SearchEpisode` has no field capable of declaring theorem authority or a truth value. Attempts to add such fields fail the closed schema/validator.

## ResultMemory

Each result entry records:

- stable episode and claim identity;
- result kind;
- explicit evidence state;
- evidence references;
- method/representation;
- claim dependencies;
- timestamp;
- retrieval embedding reference;
- optional Phi-like search score as a bounded retrieval feature.

Formal-looking result kinds and formal evidence states require at least one evidence reference. The reference is still only a reference until independently authenticated by the formal evidence plane.

## SearchMemory

Each search entry records:

- research program and problem/subgoal identity;
- strategy, representation, and tactic family;
- parameter and budget commitments;
- typed outcome;
- failure class where the outcome is non-conclusive;
- resource cost;
- useful artifacts;
- exact source-provenance references;
- transfer context;
- timestamp and retrieval embedding;
- bounded Phi/search-quality score.

The following outcomes are explicitly non-truth outcomes and require a failure class:

```text
Timeout
ResourceExhausted
Unsupported
NoProgress
Abandoned
Superseded
RepresentationMismatch
SolverUnknown
```

They may guide future search but cannot refute a claim.

`CounterexampleFound` requires a useful artifact reference; the artifact still requires its own validation before becoming a mathematical counterexample result.

## Retention

This v1 contract deliberately does **not** define a retention policy. In particular, it does not permit lowest-Phi eviction to become the normative research-memory policy.

MATH-SEARCH-001C should qualify retention under multiple independent dimensions such as information gain, repeated-failure avoidance, cost saved, representation/strategy diversity, transfer usefulness, rarity, recency, and counterexample value, with negative controls including no-Phi, recency-only, random reservoir, and success-only baselines.

High-value negative results should be compacted or summarized rather than silently erased.

## HDC role

HDC is permitted to index and retrieve both result and search episodes. Retrieval output must preserve the episode class and evidence state so that a structurally similar failure cannot be presented as a fact.

A later causal experiment should compare conventional retrieval, HDC success-only retrieval, and HDC success+negative-search retrieval under equal budgets.

## Machine checks

- `.github/schemas/math-search-memory-v1.schema.json` defines the closed interchange shape.
- `.github/scripts/validate-math-search-memory.py` applies semantic invariants and contains adversarial self-tests.

The stdlib validator is normative where JSON Schema cannot express cross-field rules such as globally unique episode IDs, formal-result evidence references, timeout failure classes, and counterexample artifact requirements.

## Nonclaims

This contract does not:

- prove any theorem;
- authenticate an evidence receipt;
- establish formal authority;
- establish that Phi/HDC improves search;
- implement a Rust memory store;
- change the existing `MathMemory` runtime policy;
- establish a retention strategy.

It only makes the result/search distinction explicit and machine-checkable before runtime migration.
