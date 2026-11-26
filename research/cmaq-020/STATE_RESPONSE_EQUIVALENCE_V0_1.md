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

Class count and singleton count are not memory-byte measurements. The confirming resource ledger must still charge vector payloads, codebooks, model/checkpoint assets, indices, provenance sidecars, replay data, update work and retrieval cost.

## Approximate response

For key `k`, normalize the absolute response difference by the contract's positive tolerance `tau_k`. Pairwise approximate equivalence is:

`max_k(|G_k(s,u)-G_k(s',u)| / tau_k) <= epsilon`

This relation may be non-transitive. The fixture `a=0, b=8, c=16` with tolerance 10 proves that `a~b` and `b~c` does not imply `a~c`. Do not use pairwise epsilon edges as a union-find equivalence partition. Any future cover must report its verified diameter, coverage and unresolved pairs.

## v0.1 executable slice

- Frozen intervention × query × horizon response grid.
- Exact invariant signature comparison.
- Candidate-assigned abstract labels.
- False-merge and false-split counts.
- Candidate abstraction class count and singleton-class count (coarse partition-cardinality diagnostics only).
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

Added deterministic tests cover: lowest-cost separating query selection, no disclosure of response values, invariant mismatch handling, ignored already-separated pairs, deterministic selection under input reordering, sequential refinement of multiple false merges, missing/zero probe-cost rejection, domain and digest validation, single-checkpoint enforcement, cluster/family consistency, a copy/no-op negative control where identical no-op response cannot hide a state-distinguishing action, and partition cardinality/singleton reporting.


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


## Failure fidelity is a separate qualification axis

A predicted state may match the final task-success label while getting the failure itself wrong. A world model can hallucinate a successful lift after missed contact, or reproduce failure but place it at the wrong execution stage. CMAQ-020 therefore adds the executable [failure-fidelity scorecard](FAILURE_FIDELITY_V0_1.md) in `src/failure_fidelity.rs`.

It reports separately:
- false-success rate among reference failures;
- failure capture recall;
- failure reason accuracy;
- execution-stage accuracy;
- false-failure rate among reference successes;
- boundary-flip detection across matched lower/upper perturbations;
- exact two-sided boundary outcome accuracy;
- per-independent-cluster counts.

A model that correctly predicts "failure" for the wrong reason does not get full credit. A model that detects a boundary flip but puts the failure on the wrong side does not get full boundary credit. The evaluator refuses incomplete boundary pairs, mixed candidate checkpoints and pairs that do not share exact initial-state/action/family/cluster identity.

External anchors:
- WorldFailBench tests failure reason, stage and boundary across LIBERO, RoboTwin and real G1 manipulation. https://arxiv.org/abs/2610.33565
- MiraBench tests physics adherence, action following and optimism bias. https://arxiv.org/abs/2605.29360
- AMD's 20 October 2026 BWM study reports 82.5% fake-success on simulator failures before failure-data fine-tuning, reduced to 26.3% on its held-out RoboTwin adjust_bottle cases; it also reports unresolved same-start action-fork errors. https://www.amd.com/en/developer/resources/technical-articles/2026/rl-inside-world-models.html
- LIBERO-RECOVER tests retry, action adaptation, object-state recovery and environmental recovery. https://arxiv.org/abs/2609.05178

These protocols must remain separate. Prediction of failure, recovery from failure, and downstream policy utility are different claims. No benchmark scores are pooled. Current CMAQ-020 failure-fidelity fixtures are synthetic known-answer tests only; no external benchmark adapter has been run.



### Replicated failure boundaries, not binary-only outcomes

WorldFailBench's released boundary subset contains 120 matched pairs with 10 reference rollouts per side (2,400 episodes). CMAQ-020 now has a separate rate-based boundary evaluator alongside the binary witness tests. Each pair records reference and candidate success counts/trial counts on the benchmark-designated success side and failure side, with shared initial-state/action/family/cluster/checkpoint identity.

It reports per-side success-rate MAE, success/failure rate-gap error, boundary-presence accuracy at a preregistered minimum gap and gap-direction accuracy. The threshold belongs to the frozen contract and must be pinned to the task protocol before external comparisons. The current synthetic fixture uses 2,000 basis points only as a test value. This scorecard is not the original WorldFailBench scorer and is not yet an external dataset adapter.

WorldFailBench reference failure annotations include an Unjudgable/indeterminate reason and unknown stage. CMAQ-020 represents those as explicit optional reason/stage fields rather than dropping the episode or inventing a known label. An indeterminate reference matched by an indeterminate prediction is preserved as such; guessing a specific reason or stage on an indeterminate reference is an error.

### Dataset admission evidence

- WorldFailBench paper scope: 7 world models, 32 tasks, 6,900 reference rollouts across LIBERO, RoboTwin and physical G1.
- Documented public simulator subset: 5,000 RoboTwin + 1,500 LIBERO = 6,500 episodes.
- Separate boundary subset: 120 pairs / 2,400 rollouts (10 per side); do not add these to the 6,500 base episodes without retaining the separate sampling design.
- The extra 400 G1 reference rollouts in paper scope are not part of the documented public simulator counts.
- The HF dataset card and dataset repository declare Apache-2.0; archive hash, file inventory and licenses for all linked assets still require local admission checks. The separate project site says its original evaluation code is not yet released.
- LIBERO-RECOVER's current site advertises 2,178 scenarios / 3,184 human demonstrations, while its GitHub README advertises 4,803 failure scenes / 1,446 raw rollouts. Treat as a release-manifest discrepancy until exact revisions and denominators are reconciled.

Published WorldFailBench RoboTwin results show Ref–Ref TFE/SFE/TPC/SPC of 0.061/0.094/0.814/0.903. Across the seven world-model rows, the best listed TFE is 0.159 (DW05), best TPC is 0.402 (DW05), and best world-model SPC is 0.095 (Ctrl). The reference SPC is 0.903. These are separate benchmark-specific metrics; they are not combined into one ranking. No Symthaea score is available yet.

## Benchmark validity is a promotion gate

A high score on a weak benchmark is not evidence of general capability. Before using an external task suite for a SOTA comparison, run an explicit validity audit for:

1. **Shortcut solvability:** simple/no-language/copy/no-op/heuristic baselines must not solve the benchmark through unintended correlations.
2. **Statistical significance:** use independent clusters, paired uncertainty and sufficient power for the target effect; repeated episodes within one environment are not independent samples.
3. **Creeping overfit:** hold out initial-state perturbations, object configurations, nuisance/context values and mechanism families, including near and out-of-range splits.
4. **Data-source dependence:** pin dataset/source revisions, hashes, licenses and training/test overlap; prevent near-duplicate source leakage.

These four audit dimensions are motivated by the June 2026 study *What Are We Actually Benchmarking in Robot Manipulation?* (LIBERO, CALVIN, SimplerEnv, RoboCasa and RoboTwin 2.0). https://arxiv.org/abs/2606.04233

CMAQ-020 must publish benchmark-validity results separately from model-performance scores. A benchmark failing an audit may remain useful as a diagnostic fixture but cannot substantiate a general-capability or SOTA claim.

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
