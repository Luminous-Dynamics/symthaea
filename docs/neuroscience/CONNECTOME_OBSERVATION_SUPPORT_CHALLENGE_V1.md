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

## 8. Comparison with connectomics industry benchmarks

ConnectomeBench (NeurIPS 2025) evaluates segment-type identification, split-error correction and merge-error detection. Its reported model ranges are task-specific predictive metrics; it does not by itself establish support-set completeness or decision stability under new plausible measurement outcomes.

Autoproof reports major reductions in guided proofreading effort on a Drosophila reconstruction. Those are workflow-automation and cost-efficiency results, not a substitute for independent evidence provenance or query-level robustness certificates.

This tool does not claim better segmentation or proofreading accuracy. It adds a complementary assurance capability: *show exactly which conclusions cease to be robust when the measurement model is challenged*. The proper comparison is therefore a paired evaluation on the same datasets and expert reviews: predictive quality and labor savings from the proofreading systems, plus plan survival, query-critical error recall, and support-expansion sensitivity from this assurance layer.

## 9. Claim boundary

The report is conditional on supplied worlds, actions, challenge records, hashes and effort units. It cannot prove:
- that a challenge source is authentic or independent;
- that the challenge protocol is scientifically valid;
- that the challenge corpus contains every missing outcome;
- that baseline or expanded scenario spaces are biologically complete;
- that a graph edge, synapse, cell identity, or circuit interpretation is biologically true.

No probability, posterior, likelihood, entropy, expected-VOI or consciousness claim is made. No acquisition, edit, proofread correction or biological decision is executed.
