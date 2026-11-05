# Correlation-aware joint observation support and robust disambiguation V1

**Tracking:** [#10528 — joint observation support and correlation-aware disambiguation](https://github.com/Luminous-Dynamics/symthaea/issues/10528)  
**Upstream:** [#10506 — set-valued marginal observation disambiguation](https://github.com/Luminous-Dynamics/symthaea/pull/10506), [#10525 — adversarial support expansion](https://github.com/Luminous-Dynamics/symthaea/pull/10525).  
**Tool:** connectome-joint-support-audit.  
**Status:** Rust exact planner, JSON Schema, synthetic fixtures, independent Python oracle and isolated workflow authored. Exact-head qualification pending.

## 1. Why marginal supports are not enough

The #10506 planner treats each observation action separately. For action a and world w, it binds a set of permitted categorical outcomes O_a(w). That is conservative at the marginal level: two worlds are separated by one action only when their marginal sets are disjoint.

But the relation among observations can carry information that no single marginal contains. Conversely, assuming independence and multiplying marginal sets manufactures joint combinations that may be physically impossible.

The correct primitive is the *joint observation-support relation*:

    J(w) subset-of product over a in A of O_a(w)

Each row is one admissible joint configuration across the declared action universe. The relation is a finite set of tuples, not a probability distribution. Shared ancestry and model/source digests remain attached to the declared support.

## 2. Exact query-specific separation

For a selected action subset S subset-of A, project the joint relation onto those coordinates:

    pi_S(J(w)) = { r restricted to S : r in J(w) }

A pair of worlds with opposite frozen query decisions is robustly separated by S if and only if:

    pi_S(J(w_i)) intersect pi_S(J(w_j)) = empty

A scientific query is robust over the submitted world set only when one selected action set separates every opposing-decision pair.

This is not a Cartesian product of marginal sets. It preserves supplied correlations, including impossible combinations that must not be invented.

## 3. Paradigm shift: reduce joint projection to exact hitting set

For any tuple u in J(w_i) and v in J(w_j), define the action-coordinate difference set:

    D(u,v) = {a in A : u_a != v_a}

A selected action set S guarantees that this tuple pair cannot collide after projection exactly when:

    S intersect D(u,v) is non-empty

Therefore robust separation of every opposing world pair is equivalent to a hitting-set problem over all cross-world tuple-pair difference sets. The objective is minimum total effort subject to hitting every constraint.

This gives three useful properties:

1. No independence assumption is required.
2. Joint complementarity is expressible: two actions may jointly separate worlds even when each individual marginal overlaps.
3. A shared full joint tuple is a hard negative control. If two decision-opposed worlds admit the same full tuple, no action subset can separate them.

The implementation canonicalizes difference masks, removes redundant supersets using a subset-count transform, and marks invalid action subsets via complement submasks. It enumerates the remaining finite candidate subsets exactly. A fixed 50-million submask-marking cap fails closed with search_complete=false; it never relabels a partial search as an exact minimum.

## 4. Input/provenance contract

Every input binds:
- exact dataset revision, query ID and query-spec digest;
- scenario-set digest, action-set digest and joint-support-set digest;
- effort unit and review budget;
- each action's ID, type, positive integer effort, source-evidence digest and dependence ancestry;
- for each world, a joint observation-model digest, source-evidence digest, dependence ancestry and finite joint rows.

Each joint row must contain exactly one non-empty categorical value for every declared action. Unknown, missing or repeated action IDs fail closed. Duplicate complete rows fail closed. World and action ordering do not affect canonical digests.

The joint_support_scope is either exhaustive_declared_joint_support or sampled_joint_sensitivity_set. These labels define the supplied scope; neither proves all physically possible outcomes were included. A hash binds a declaration, not its scientific completeness.

Limits: 20 actions; 24 worlds; 64 joint rows per world; 1,536 rows total; 50 million subset-mark operations. The result includes explicit search completeness, all exact tied minima up to 128 returned plans, and a truncation flag.

## 5. Synthetic known-answer fixtures

### XOR/complementarity fixture

World alpha supports the joint rows (x=0,y=0,z=0) and (x=1,y=1,z=1). World beta supports (0,1,1) and (1,0,0).

Every single-action marginal is identical across worlds:
- x supports {0,1} in both;
- y supports {0,1} in both;
- z supports {0,1} in both.

No single action separates the query-conflict pair. The joint projection on {x,y} is {00,11} versus {01,10}, which is disjoint. The same is true for {x,z}; {y,z} overlaps. Efforts are x=2, y=3, z=5, so the exact minimum is {x,y} at 5 expert-review minutes. This is genuine complementarity, not independence.

### Cost-increase challenge fixture

Add the previously omitted beta joint row (x=0,y=0,z=1). The xy projection now overlaps at (0,0), but xz remains disjoint. The exact minimum rises to {x,z} at 7 minutes and exceeds the declared 5-minute budget. This tests whether a new plausible joint configuration invalidates a formerly cheapest observation plan.

### Shared-full-tuple negative control

Add beta's row (0,0,0), which is identical to an admissible alpha row. Every action projection now overlaps. No subset can robustly separate the worlds, and the tool must return the explicit shared-full-tuple pair with no minimum plan.

These are synthetic known-answer cases, not measurements from a biological connectome.

## 6. Independent verification and invariants

The Python oracle uses tuple projections and set intersection, then independently enumerates selected-action combinations. It does not reuse the Rust difference-mask / subset-marking optimizer.

Rust tests cover:
- complementarity absent from every single-action marginal;
- exact minimum selection of x+y rather than the more expensive x+z alternative;
- shared-full-tuple no-plan behavior;
- minimum effort increase after add-only joint-row expansion;
- monotonicity: support expansion cannot create a new robust projection for any action subset;
- missing action coordinates and stale digest rejection;
- canonical digest invariance under action/row ordering changes.

## 7. Compare against current industry and SOTA

The strongest 2026 predictive baseline located is **ConnectomeBench2**. Its release describes more than 716,485 expert-labelled proofreading decisions and over 4.5 million associated images across mouse, human, zebrafish and fly connectomes. It reports a multi-modal Vision Transformer reaching human-level split-error and merge-error proofreading accuracy across species, with in-distribution calibration and distribution-distance signals. We should reproduce its V2 official spatial holdouts before claiming predictive parity or superiority. [ConnectomeBench2 paper](https://arxiv.org/abs/2606.21116) · [benchmark release page](https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/)

**ConnectomeBench (NeurIPS 2025)** evaluated segment-type identification, split-error correction and merge-error detection; its abstract reports 52–82% balanced accuracy for segment identification and 75–85% split correction accuracy for evaluated VLMs, while merge detection was harder. ConnectomeBench2 is the stronger current multi-species proofreading comparator. [ConnectomeBench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4a70f0c8443593ca59a88ebc8a937ed6-Abstract-Datasets_and_Benchmarks_Track.html)

**NEURD (Nature 2025)** demonstrated high-confidence merge proofreading agreement of 99% for evaluated axon-on-axon edits and 95% for axon-on-dendrite edits, with large-scale automated edits. The paper also describes a precision/recall trade-off: axonal synapse recall in the evaluated MICrONS subset was materially lower than dendritic recall. This is why a single accuracy score is insufficient for scientific claims. [NEURD](https://www.nature.com/articles/s41586-025-08660-5)

**CAVE (Nature Methods 2025)** provides timestamped materialization and historical queryability while proofreading continues; it reports handling almost two billion annotations and over four million edits across deployed datasets. Our assurance receipts should bind exact reconstruction/materialization versions and lineage rather than compete with this infrastructure. [CAVE](https://www.nature.com/articles/s41592-024-02426-z)

### What this PR adds—and does not add

It does not improve split/merge classifier accuracy. It adds an auditable query-assurance capability: evaluate claims against a correlated finite observation relation and show the minimum review set needed to rule out every decision-reversing joint observation.

The product direction is a *claim-assurance plane* sitting alongside ConnectomeBench2-class classifiers, NEURD-class proofreading, and CAVE-class versioning. It should consume their evidence and preserve their provenance, not replace them.

## 8. Proposed benchmark targets (not achieved)

| Dimension | Target |
|---|---:|
| Joint-plan agreement with independent projection/combinations oracle (within bounds) | 100% |
| False robust separations on shared-full-tuple controls | 0 |
| Complementarity recovery on synthetic XOR cases | 100% |
| Monotonicity violations under add-only joint-row expansion | 0 |
| Missing/duplicate action coordinates or digest mismatches accepted | 0 |
| Unjustified independence/probability claims | 0 |
| Exact plan regret vs oracle | 0 |
| Predictive parity with ConnectomeBench2 V2 | within 1 percentage point on identical official spatial holdouts |
| Query-critical error recall | +15 percentage points at matched expert-review minutes |
| Expert review cost | 25% lower at matched adjudicated query quality |
| 90% / 95% calibrated set coverage | within ±3 percentage points on held-out spatial/acquisition domains |

The classifier and review-efficiency targets require separate prospective evaluations; this planner does not demonstrate them. Report per-species and per-task results, cluster-bootstrap confidence intervals, absolute counts, subgroup performance, precision/recall, and expert minutes. Never trade away query-critical minority cases for aggregate balanced accuracy.

## 9. Research next steps

1. Pair the marginal set-valued planner (#10506) and the joint relation planner on identical worlds/actions; quantify the complementarity gap and cost difference.
2. Extend #10525 support challenges to append joint rows and measure plan survival as support expands.
3. Add multi-reviewer, alternative preprocessing, acquisition-resolution and orthogonal-modality challenge sets with provenance ancestry.
4. Validate joint-support models prospectively against blinded expert adjudication and true orthogonal observations.
5. For action universes beyond 20, compare ILP/SMT and branch-and-bound against the exact bounded oracle. Approximate results must carry explicit optimality-gap certificates and must never be labelled exact minima.

No synthetic test establishes biological completeness, evidence independence, consciousness or biological truth.
