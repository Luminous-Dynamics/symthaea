# CMAQ-020 — State-response equivalence and query-sufficient abstraction v0.1

**Status:** research prototype / synthetic finite contract only.  
**Claim ceiling:** evaluator arithmetic and contract validation; no benchmark-performance, general causal-understanding, physical-safety or SOTA claim.

## Research object

A state abstraction is acceptable only relative to a frozen contract:

`C = (U, Q, T, I, D)`

where `U` is the finite set of permitted interventions, `Q` the registered queries/observables, `T` the response horizons, `I` the invariant signature and `D` the declared domain.

For each evaluator-owned raw state and every key in `U × Q × T`, the evaluator stores one response value. A candidate's abstraction label groups raw states. A false merge occurs when the candidate assigns one abstraction label to two states whose exact response signatures or invariant signatures differ. A false split occurs when two exactly equivalent states receive different labels.

The first Rust implementation deliberately uses canonical integer response units and exact finite coverage. It does not accept a candidate-supplied oracle response, discover the partition, or infer global equivalence from a finite probe set.


## Provenance and domain qualification

The executable contract now binds:
- a validated SHA-256 for the canonical contract payload;
- a validated SHA-256 for the source dataset/generator manifest;
- an explicit allow-list of domain IDs.

Every state record binds:
- its domain ID;
- environment/mechanism family ID;
- independent environment-cluster ID;
- exact source-state snapshot SHA-256;
- frozen candidate checkpoint SHA-256.

The record-set validator rejects:
- a domain outside the frozen contract;
- malformed source-state or candidate-checkpoint digests;
- mixed candidate checkpoints in one evaluation;
- reuse of one environment-cluster ID across different domain/family identities.

The cluster ID is carried through to support #8578's cluster-level inference. A task/state row is not thereby declared an independent sample. State records from the same environment cluster may be compared descriptively, but the inferential unit remains the independently generated cluster.

These fields establish identity and custody only. In this dependency-free crate, SHA-256 fields are syntax-checked as 64-character hexadecimal strings; the crate does not recompute the digest from the external manifest/payload or authenticate its publisher. The runner/custody layer must independently recompute and compare each digest against the exact bytes and attach signed receipts. A SHA-256 digest alone does not prove the source data are true or establish independent generation. Source authenticity, evaluator isolation and signed receipts remain separate gates (#9437).

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
- Strict rejection of missing/extra response cells, missing invariant fields, duplicate state IDs, out-of-contract domains, malformed digests, mixed candidate checkpoints, cluster/domain/family aliasing, invalid contracts, zero tolerances and invalid epsilon.
- Deterministic test fixtures for true merges, false merges, false splits, invariant mismatch and non-transitive tolerance.

The initial slice is intentionally independent of HDC/CfC model internals. Subsequent adapters will consume actual Symthaea checkpoint/state fingerprints and evaluator-side response records only after the interface and custody contract are qualified.


## Counterexample-guided refinement (new v0.2 evaluator increment)

The evaluator now supports a deterministic, cost-aware next-counterexample selector over the candidate's *currently merged* state pairs.

- A structural invariant mismatch is surfaced as a metadata witness, not as a numeric or categorical invariant value. No intervention-query cost is assigned to this witness; verification and source-reconstruction work must still be separately charged in the run manifest.
- Otherwise the evaluator selects the cheapest registered intervention × query × horizon key that distinguishes the pair; ties are resolved deterministically.
- Only the two opaque state IDs, contract ID, separating dimension and relative probe cost are returned.
- The hidden response values, invariant values and evaluator reference labels are not returned.
- The candidate must perform the permitted query/interaction or source-evidence verification and submit a revised abstraction on a subsequent evaluation. The evaluator does not auto-split the candidate's partition.
- If a candidate refines one false merge but leaves another, the next call selects a remaining counterexample. When no candidate-merged pair remains distinguishable within the finite contract, the selector returns no counterexample.

This makes refinement cost observable instead of granting free access to the oracle's entire response table. The relative probe costs are contract inputs and must be frozen before held-out evaluation; they are not a dynamic estimate derived from the hidden response. The counterexample selector is a finite diagnostic tool, not a proof that all possible future interventions have been covered.

Added deterministic tests cover: lowest-cost separating query selection, no disclosure of response values, invariant mismatch handling, ignored already-separated pairs, deterministic selection under input reordering, sequential refinement of multiple false merges, missing/zero probe-cost rejection, domain and digest validation, single-checkpoint enforcement, cluster/family consistency, and a copy/no-op negative control where identical no-op response cannot hide a state-distinguishing action.


## External reliability anchors added to the evaluation matrix

CMAQ-020's next adapter layer should include distinct diagnostics inspired by current benchmarks. They are not pooled into one result:

- **ScratchWorld — changed-consequence correctness:** include a no-change/copy baseline and score only the value and identity of fields that actually changed under a replayed action. High unchanged-field overlap must not hide failure to predict action consequences.
- **MiraBench — action-following and optimism bias:** include actions that should fail, no-contact and partial-execution controls, and report false-success predictions separately from generic state error.
- **ContactWorld — representation compatibility:** where contact-rich robotics is evaluated, compare spatial geometry and tactile-force representations under the same model/planner budget. Do not assume that adding a modality helps without checking cross-modal compatibility and long-horizon task success.
- **Closed-loop advantage track:** compare the same downstream controller with and without the world model, plus a direct policy/VLA baseline. Measure executed success, recovery and cost; do not treat open-loop accuracy as proof of planner benefit.

Relevant references:
- ScratchWorld: https://arxiv.org/abs/2606.31689
- MiraBench: https://arxiv.org/abs/2605.29360
- ContactWorld: https://contact-world.github.io/
- World-model benchmark taxonomy: https://world-model-benchmarks.github.io/World-Model-Benchmarks/

## Next experimental matrix

Compare full history/full state, symbolic/tabular mechanism state, linear projection, matched recurrent latent state, HDC-only, CfC/LTC-only, HDC+CfC/LTC factorized state, and retrieval-backed event references. Match exploration, task split, query adapter, compute, interaction budget and all-in memory (model, codebook, index, journal, provenance sidecar and replay data).

Score response fidelity, false merge/split, invariants, selective uncertainty, downstream query/intervention regret, weak-cell transfer, memory bytes, update/replay work, p50/p95/p99 latency and measured energy separately. No pooled world-model score.

The confirming design crosses the 5 × 3 CMAQ-018 mechanism/query matrix and reuses CMAQ-019 paired-intervention outcomes. Independent environment/mechanism cluster is the inferential unit (#8578). Hidden oracle separation remains blocked on #9437.

## Comparator map

See [`WORLD_MODEL_COMPARATORS_2026.md`](WORLD_MODEL_COMPARATORS_2026.md) for the protocol-by-protocol benchmark and industry comparator map, including WAMs, RoboJEPA, Cosmos Policy, ContactWorld, MiraBench, ScratchWorld, direct VLA controls and matched resource rules.

## External positioning

- WorldTest / AutumnBench: exploration and query-transfer lane.
- What-If World / Causal-Discovery-Bench / ACDB / CausalWorld: separate intervention and experiment-design comparators.
- MRS.Q: adaptive search exploitation control, not memory-compression comparator.
- World-In-World / WorldArena 2.0: later closed-loop and embodied outcome lanes.
- TD-MPC2 / matched RSSM: separate continuous-control baseline when action and cost interfaces are genuinely comparable.

Predictive state representations, causal states, bisimulation and rate–distortion are established conceptual foundations. This work tests a task- and intervention-conditioned evaluation contract for Symthaea's action-conditioned world state; it does not claim to replace those theories.
