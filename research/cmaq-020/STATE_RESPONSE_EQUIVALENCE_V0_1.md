# CMAQ-020 — State-response equivalence and query-sufficient abstraction v0.1

**Status:** research prototype / synthetic finite contract only.  
**Claim ceiling:** evaluator arithmetic and contract validation; no benchmark-performance, general causal-understanding, physical-safety or SOTA claim.

## Research object

A state abstraction is acceptable only relative to a frozen contract:

`C = (U, Q, T, I, D)`

where `U` is the finite set of permitted interventions, `Q` the registered queries/observables, `T` the response horizons, `I` the invariant signature and `D` the declared domain.

For each evaluator-owned raw state and every key in `U × Q × T`, the evaluator stores one response value. A candidate's abstraction label groups raw states. A false merge occurs when the candidate assigns one abstraction label to two states whose exact response signatures or invariant signatures differ. A false split occurs when two exactly equivalent states receive different labels.

The first Rust implementation deliberately uses canonical integer response units and exact finite coverage. It does not accept a candidate-supplied oracle response, discover the partition, or infer global equivalence from a finite probe set.

## Approximate response

For key `k`, normalize the absolute response difference by the contract's positive tolerance `tau_k`. Pairwise approximate equivalence is:

`max_k(|G_k(s,u)-G_k(s',u)| / tau_k) <= epsilon`

This relation may be non-transitive. The fixture `a=0, b=8, c=16` with tolerance 10 proves that `a~b` and `b~c` does not imply `a~c`. Do not use pairwise epsilon edges as a union-find equivalence partition. Any future cover must report its verified diameter, coverage and unresolved pairs.

## v0.1 executable slice

- Frozen intervention × query × horizon response grid.
- Exact invariant signature comparison.
- Candidate-assigned abstract labels.
- False-merge and false-split counts.
- Worst-case normalized pair distortion.
- Strict rejection of missing/extra response cells, missing invariant fields, duplicate state IDs, invalid contracts, zero tolerances and invalid epsilon.
- Deterministic test fixtures for true merges, false merges, false splits, invariant mismatch and non-transitive tolerance.

The initial slice is intentionally independent of HDC/CfC model internals. Subsequent adapters will consume actual Symthaea checkpoint/state fingerprints and evaluator-side response records only after the interface and custody contract are qualified.

## Next experimental matrix

Compare full history/full state, symbolic/tabular mechanism state, linear projection, matched recurrent latent state, HDC-only, CfC/LTC-only, HDC+CfC/LTC factorized state, and retrieval-backed event references. Match exploration, task split, query adapter, compute, interaction budget and all-in memory (model, codebook, index, journal, provenance sidecar and replay data).

Score response fidelity, false merge/split, invariants, selective uncertainty, downstream query/intervention regret, weak-cell transfer, memory bytes, update/replay work, p50/p95/p99 latency and measured energy separately. No pooled world-model score.

The confirming design crosses the 5 × 3 CMAQ-018 mechanism/query matrix and reuses CMAQ-019 paired-intervention outcomes. Independent environment/mechanism cluster is the inferential unit (#8578). Hidden oracle separation remains blocked on #9437.

## External positioning

- WorldTest / AutumnBench: exploration and query-transfer lane.
- What-If World / Causal-Discovery-Bench / ACDB / CausalWorld: separate intervention and experiment-design comparators.
- MRS.Q: adaptive search exploitation control, not memory-compression comparator.
- World-In-World / WorldArena 2.0: later closed-loop and embodied outcome lanes.
- TD-MPC2 / matched RSSM: separate continuous-control baseline when action and cost interfaces are genuinely comparable.

Predictive state representations, causal states, bisimulation and rate–distortion are established conceptual foundations. This work tests a task- and intervention-conditioned evaluation contract for Symthaea's action-conditioned world state; it does not claim to replace those theories.
