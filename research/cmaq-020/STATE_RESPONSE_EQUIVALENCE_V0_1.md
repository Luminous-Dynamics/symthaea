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


## Counterexample-guided refinement (new v0.2 evaluator increment)

The evaluator now supports a deterministic, cost-aware next-counterexample selector over the candidate's *currently merged* state pairs.

- An invariant mismatch is surfaced as an immediate, zero-acquisition-cost witness.
- Otherwise the evaluator selects the cheapest registered intervention × query × horizon key that distinguishes the pair; ties are resolved deterministically.
- Only the two opaque state IDs, contract ID, separating dimension and relative probe cost are returned.
- The hidden response values, invariant values and evaluator reference labels are not returned.
- The candidate must perform the permitted query/interaction and submit a revised abstraction on a subsequent evaluation. The evaluator does not auto-split the candidate's partition.
- If a candidate refines one false merge but leaves another, the next call selects a remaining counterexample. When no candidate-merged pair remains distinguishable within the finite contract, the selector returns no counterexample.

This makes refinement cost observable instead of granting free access to the oracle's entire response table. The relative probe costs are contract inputs and must be frozen before held-out evaluation; they are not a dynamic estimate derived from the hidden response. The counterexample selector is a finite diagnostic tool, not a proof that all possible future interventions have been covered.

Added deterministic tests cover: lowest-cost separating query selection, no disclosure of response values, invariant mismatch handling, ignored already-separated pairs, deterministic selection under input reordering, sequential refinement of multiple false merges, and missing/zero probe-cost rejection.

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
