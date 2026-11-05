# Adversarial observation-support completeness envelope V1

**Tracking:** [#10519 — support-set completeness and robustness envelopes](https://github.com/Luminous-Dynamics/symthaea/issues/10519)  
**Upstream contract:** [#10506 — robust set-valued query disambiguation](https://github.com/Luminous-Dynamics/symthaea/pull/10506)  
**Earlier layers:** [#10502 — exact query-flipping witnesses](https://github.com/Luminous-Dynamics/symthaea/pull/10502), [#10505 — deterministic observation disambiguation](https://github.com/Luminous-Dynamics/symthaea/pull/10505).  
**Tool:** `connectome-observation-support-challenge-audit`.  
**Status:** Rust implementation, schema, three fixtures, independent oracle and isolated workflow authored. CI qualification remains pending.

## 1. Research gap

The robust planner in #10506 changes point outcomes into set-valued supports. It only credits an action for separating two worlds when their possible-outcome sets are disjoint:

`O_a(w_i) \cap O_a(w_j) = \varnothing`

This is conservative relative to the *declared* support sets. It is not conservative relative to reality unless the declarations include every plausible observation outcome. A cryptographic digest proves which declaration was evaluated; it cannot prove that the declaration is complete.

This layer challenges that assumption by admitting separately sourced outcomes and measuring how the robust plan changes. It is a sensitivity/failure-envelope audit, not a probabilistic uncertainty model.

## 2. Monotonicity theorem

A challenge may only add possible outcomes. For every action (a) and world (w), the challenged support satisfies:

`O'_a(w) \supseteq O_a(w)`

For any world pair, if the original supports overlap, expanding either or both supports cannot make them disjoint. Therefore, action-level robust separation can only stay the same or be lost.

The global pair-coverage predicate is the union over all available actions, so it also cannot increase. Further, the family of feasible action subsets after expansion is a subset of the baseline feasible family. With positive effort costs, if a challenged complete plan still exists, its exact minimum effort cannot be lower than the baseline minimum.

Expected invariant:

`coverage' \le coverage`

and, whenever both plans exist,

`minimum\_effort' \ge minimum\_effort`

The implementation checks these properties at runtime and emits a violation list. This is an executable invariant, not a statistical claim.

## 3. Input and binding contract

The CLI accepts the exact #10506 baseline input and a separate challenge JSON document. The challenge document binds:

- analysis ID, dataset revision commit, query ID and query-spec digest;
- the canonical scenario-set digest;
- the canonical action-set digest from the #10506 contract;
- a challenger-protocol digest;
- each challenge record's unique ID, target action/world, additional outcome categories, source-evidence digest and dependence ancestry labels.

Unknown worlds/actions, invalid digests, duplicate challenge IDs, empty outcome sets and malformed dependence groups fail closed. Challenge operations can add outcomes only; no deletion or support narrowing exists in the schema.

Multiple challenge records may target the same action/world. Duplicate outcome categories across records are counted only once as newly admitted support values. All records remain in the provenance trace. Dependence groups are retained and never converted into independence assumptions.

## 4. Report semantics

The report includes exact baseline and challenged plan summaries:

- decision-conflict pair coverage;
- undiscriminable pairs;
- exact minimum effort and number of tied minima;
- up to 128 deterministically ordered tied plans, with explicit truncation;
- whether the exact minimum is within the baseline budget;
- all action-level separation losses and all globally lost pair separations;
- per-record newly admitted outcomes;
- raw challenge-input hash and canonical challenge-set digest;
- monotonicity result and violations.

Outcome values:

| Outcome | Meaning |
|---|---|
| `NO_CHALLENGE_ADDITIONS` | The challenge corpus has no records. |
| `ROBUST_PLAN_PRESERVED` | A baseline robust plan remains at the same exact minimum cost. |
| `ROBUST_PLAN_COST_INCREASED` | A complete robust plan remains, but its exact minimum effort increased. |
| `ROBUST_PLAN_DESTROYED_BY_SUPPORT_EXPANSION` | At least one query-conflict pair is no longer separable by any declared action. |
| `BASELINE_ALREADY_HAS_NO_ROBUST_PLAN` | The baseline has no robust plan to preserve. |

The outcome does not replace the within-budget field. A plan can be preserved or become more expensive while still being over budget.

## 5. Known-answer fixtures

The three-world / five-action fixture from #10506 has a baseline exact robust minimum of four expert-review minutes.

1. **Cost increase:** add one shared, previously omitted outcome to the beta support for each of the two alternative synapse inspections, and one to gamma for morphology. Each cheap action loses one pair separation. The five-minute orthogonal measurement remains the only complete route. Expected minimum increases from 4 to 5; the plan remains mathematically feasible but exceeds the declared budget.
2. **Plan destroyed:** apply the three additions above and add the alpha orthogonal outcome to both beta and gamma. All query-conflict pairs become undiscriminable. Expected no-plan disposition.
3. **No-op:** an empty challenge record list leaves plan cost and action coverage unchanged.

These are synthetic fixtures, not observed biological measurements.

## 6. Independent oracle and controls

The Python standard-library oracle uses Python sets and `itertools.combinations`, rather than the Rust bit-mask cover implementation. It independently recomputes canonical digests, pair separation, exact minimum costs, tied plans, impact traces and monotonicity.

The Rust test suite covers cost increase, no-plan destruction, action-coverage monotonicity, baseline binding failure, required evidence provenance, record-order digest invariance and no-op challenges.

## 7. Benchmark program

| Metric | Target |
|---|---:|
| Monotonicity violations under add-only support expansion | 0 |
| Missed action-level pair-separation losses | 0 |
| Missed globally lost pair separations | 0 |
| Exact plan recovery against independent oracle (≤20 actions) | 100% |
| Digest/binding mismatch accepted | 0 |
| Challenge records missing source provenance accepted | 0 |
| False claims that challenged supports are complete | 0 |
| Output changes under equivalent record ordering | 0 (except raw-file hash) |

Research benchmark matrix:
- Expand each support with one shared ambiguous category at a time.
- Target the currently selected minimum plan first, then every alternative tied minimum.
- Challenge each action in isolation, and then all actions within a shared dependence group.
- Compare challenge sources: independent blinded reviewers, alternate segmentation/proofreading pipelines, changed image resolution/preprocessing, and re-acquisition.
- Plot minimum effort, complete-plan survival, lost pair coverage and reviewer burden against the number and provenance of new outcomes.
- Report challenge-source coverage and dependence structure. Do not collapse them into an uncertainty scalar.

Scalability comparison:
- Exact exhaustive search is the bounded oracle for ≤20 actions (up to 1,048,575 non-empty subsets).
- For larger sets, compare greedy weighted set cover, branch-and-bound, ILP/SMT and bounded local search against exact subsets on sampled small instances. Approximate plans must never be represented as exact minima.
- Runtime and memory are separate from evidence validity: a fast plan using incomplete outcome supports is still unsound.

## 8. Comparison with the current connectomics frontier

The assurance stack is complementary to, not a replacement for, state-of-the-art segmentation and proofreading systems.

### Prediction and proofreading systems

- **ConnectomeBench2 (arXiv:2606.21116, submitted 19 June 2026)** releases more than 716,485 expert-labelled proofreading decisions and over 4.5 million associated images across mouse, human, zebrafish and fly connectomes. Its shared-encoder Vision Transformer reports human-level accuracy for split-error correction and merge-error identification across species, good in-distribution calibration, and distribution-distance signals that predict degradation on unseen data. This is the current primary predictive benchmark to reproduce. Its paper description does not claim exhaustive query-specific support-set completeness or certificates that remain valid after admissible measurement outcomes are added. [Paper](https://arxiv.org/abs/2606.21116)
- **ConnectomeBench (NeurIPS 2025)** measures segment-type identification, split-error correction and merge-error detection. Its tested multimodal LLMs achieved 52–82% balanced accuracy for segment identification and 75–85% split correction accuracy, while merge detection remained difficult. That result is useful historical context, but ConnectomeBench2 is the stronger 2026 predictive baseline. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4a70f0c8443593ca59a88ebc8a937ed6-Abstract-Datasets_and_Benchmarks_Track.html)
- **NEURD (Nature, 2025)** uses graph and morphology filters to detect merge errors and automate proofreading. The paper reports 99% agreement for a high-confidence axon-on-axon split subset and 95% for axon-on-dendrite, nearly 150,000 high-confidence automatic edits in MICrONS, and more than threefold speedup for a difficult multi-soma splitting workflow. It also documents a precision/recall trade-off; its high-confidence result is not equivalent to perfect reconstruction. [Paper](https://www.nature.com/articles/s41586-025-08660-5)
- **Autoproof (2025 preprint)** reports obtaining 90% of the value of guided proofreading with 80% lower required cost, and automatically attaching 200,000 fragments (estimated as four proofreader-years), increasing connectivity completion by 1.3 percentage points in its Drosophila male central nervous system evaluation. These are important operational results on that dataset and workflow, not universal cross-species guarantees. [Paper](https://arxiv.org/abs/2509.26585)

### What this stack should do differently

We should not claim to surpass ConnectomeBench2 on split/merge classification until a frozen, same-split experiment demonstrates it. The intended contribution is a layer beyond classification:

1. **Version-bound scientific claims.** Each query-level result binds the dataset revision, reconstruction/proofreading lineage, query definition, relevant evidence, reviewer actions and exact report digests. Historical versions must remain materializable rather than silently overwritten.
2. **Question-specific stopping.** Do not stop because a global graph statistic, aggregate confidence score or proofreading queue looks stable. Stop only when the decision for a stated scientific query is invariant across the declared plausible reconstruction set, or report precisely which counterfactual worlds still change it.
3. **Adversarial measurement-model challenge.** Treat observation supports as hypotheses. Independent review, changed resolution/preprocessing, alternative pipelines or orthogonal acquisition can add plausible outcomes; the claim certificate records which separations and minimum review plans survive those additions.
4. **Evidence ancestry before independence.** A different model, file, reviewer or pipeline is not automatically an independent measurement. Preserve shared volume, segmentation, preprocessing, annotator and model ancestry; only a predeclared and scientifically justified dependence model may support independence claims.
5. **Precision and completeness as separate dimensions.** A workflow that removes false positives but also discards real synapses may be suitable for one query and harmful for another. Report query-specific consequences, false-positive and false-negative costs, and the evidence needed to justify the intended analysis.

The product direction is therefore a **claim-assurance layer for connectomics**: every scientific claim carries a reproducible evidence receipt and explicit conditions under which it would cease to hold. It is not another proofreading classifier, and no current synthetic test establishes biological validity.

### Benchmark targets (proposed, not achieved)

| Dimension | Benchmark target | Comparison discipline |
|---|---:|---|
| Predictive parity | Reproduce ConnectomeBench2 V2 split/merge results within 1 percentage point on its official held-out protocol | Same labels, species, spatial holdouts and aggregation; do not mix dataset-card row counts with expert-decision counts |
| OOD transfer | Improve query-critical error recall by at least 15 percentage points at matched expert-review minutes | Blinded spatial, acquisition, species and region shifts; report per-domain intervals |
| Review efficiency | At least 25% fewer expert minutes at matched adjudicated query quality | Prospective randomized or matched review cohorts; count rework and downstream correction |
| Calibration | Nominal 90% and 95% prediction-set coverage within ±3 percentage points on held-out domains | Cluster-level intervals; calibration and OOD detection remain separate |
| Query witness correctness | 100% exact minimum-witness recovery for finite cases within the declared 20-atom/action bound | Independent algorithm/oracle, mutation controls and exhaustive synthetic cases |
| Robust support challenge | Zero monotonicity violations; 100% detection of synthetic lost separations | Add-only support expansion, independent challenge corpus, adversarial shared outcomes |
| Provenance closure | 100% of promoted claims bind exact source bytes, dataset revision, query digest and review/measurement lineage | Fail closed on missing or stale bindings |
| Unsupported scientific promotion | Zero false claims of support completeness, measurement independence or biological truth | Blinded independent scientific adjudication and negative controls |

For any metric involving a target gain, report confidence intervals and absolute counts; a point estimate alone is not sufficient. The 15-point recall and 25% time-saving goals are research targets, not demonstrated results. A benchmark cannot be passed by improving aggregate accuracy while worsening a query-critical subgroup.

### Scaling beyond exact bounded search

The exact planners are bounded oracles, not the eventual production optimizer. For action universes above 20, compare weighted greedy set cover, branch-and-bound, ILP/SMT and bounded local search against exact solutions on matched small instances. Report approximation gap, optimality certificate availability, runtime, memory, and review effort separately. Never label an approximate plan an exact minimum.

## 9. Claim boundary

The report is conditional on supplied worlds, actions, challenge records, hashes and effort units. It cannot prove:
- that a challenge source is authentic or independent;
- that the challenge protocol is scientifically valid;
- that the challenge corpus contains every missing outcome;
- that baseline or expanded scenario spaces are biologically complete;
- that a graph edge, synapse, cell identity, or circuit interpretation is biologically true.

No probability, posterior, likelihood, entropy, expected-VOI or consciousness claim is made. No acquisition, edit, proofread correction or biological decision is executed.
