# Correlation-aware joint observation support V1

**Tracking:** [#10528 — correlation-aware joint observation support](https://github.com/Luminous-Dynamics/symthaea/issues/10528)  
**Marginal robust predecessor:** [#10506 — set-valued observation disambiguation](https://github.com/Luminous-Dynamics/symthaea/pull/10506)  
**Support challenge predecessor:** [#10525 — adversarial support-set expansion](https://github.com/Luminous-Dynamics/symthaea/pull/10525)  
**Tool:** `connectome-joint-observation-disambiguation-audit`.  
**Status:** Rust implementation, schema, independent oracle, XOR complementarity fixture, shared-tuple negative control, and isolated CI authored. Exact-head qualification is pending.

## 1. Why marginal support is incomplete

The marginal robust planner treats observation actions separately. For an action (a), a world (w) has a set of possible outcomes (O_a(w)). It credits the action for distinguishing two worlds only when those marginal sets are disjoint.

That is appropriately conservative when actions are evaluated independently, but it can miss real information carried by correlations among observations.

Consider two actions (X) and (Y):

- Supporting world A permits joint outcomes ((0,0)) and ((1,1)).
- Refuting world B permits joint outcomes ((0,1)) and ((1,0)).

Each action alone has marginal support ({0,1}) in both worlds. The marginal planner cannot distinguish the worlds with either action. But the joint relations are disjoint: every admissible two-action pattern identifies which world is compatible with the observation.

This is not a probability argument. It is a statement about finite sets of admissible joint outcomes.

## 2. Paradigm shift: preserve the joint relation

For each world (w), the input now supplies a finite joint-support relation (J(w)) over the complete declared action universe. Each tuple contains exactly one categorical outcome for every action.

For an action subset (S), project the joint relation onto the selected coordinates:

`P_S(J(w)) = \{x|_S : x \in J(w)\}`

Two worlds are robustly separated by (S) exactly when:

`P_S(J(w_i)) \cap P_S(J(w_j)) = \varnothing`

The exact planner finds the minimum-effort action subset whose projected joint relations are disjoint for every pair of worlds that disagree on the frozen query decision.

This recovers complementarity without assuming that observations are independent.

### Why not multiply the marginals?

The Cartesian product of marginal supports is an independence-style construction of all combinations. It can include tuples that the joint observation process never permits. Replacing a measured or scientifically specified joint relation with a factorized product therefore discards correlation structure. This implementation never forms that product and never uses a factorized relation as a fallback.

A shared tuple is decisive in the opposite direction: even if most joint outcomes differ, one tuple admitted by both worlds means the selected observations do not guarantee discrimination. The shared-tuple fixture enforces that no false separation is claimed.

## 3. Exact algorithm and resource bounds

For each query-conflict world pair, the implementation constructs an equality mask for every cross-world pair of joint tuples. An equality mask records the actions on which those two tuples agree.

A selected subset (S) remains compatible with both worlds whenever at least one cross-world tuple pair agrees on every action in (S). The implementation computes that predicate for every action subset using a superset-OR zeta transform. The final query for a selected subset is a constant-time Boolean table lookup; it does not rebuild projected tuple sets for every candidate subset.

Bounds:

- Up to 16 observation actions (65,535 non-empty subsets).
- Up to 24 query worlds (at most 144 opposing-decision pairs).
- Up to 128 joint tuples per world.
- Exact subset search and all tied minimum-effort plans, returning at most 128 plans with an explicit truncation flag.
- Independent Python oracle uses direct tuple projection, set intersection and combinations enumeration rather than the Rust equality-mask/zeta-transform implementation.

The bounded exact planner is an oracle and a research instrument, not an unbounded production optimizer. Its current limits are explicit and fail closed.

## 4. Evidence and provenance contract

The input binds:

- exact dataset revision, query identifier and query-spec SHA-256;
- canonical scenario-set digest;
- canonical action-universe digest (action identity, type, effort, source/model hashes and dependence ancestry);
- canonical joint-support digest;
- per-world source-evidence and joint-observation-model digests;
- dependence ancestry groups at both action and joint-support levels.

Every joint tuple must have exactly one outcome for every declared action. Unknown action IDs, duplicate action assignments, empty outcomes, duplicate tuples, missing worlds, stale digests and over-limit inputs fail closed.

Different action IDs, reviewers, models or source files do not imply independent evidence. Dependence groups are retained as ancestry metadata; this V1 makes no probabilistic independence claim.

## 5. Known-answer fixtures

### XOR complementarity

Two worlds each permit two joint tuples over actions X and Y. Their marginal support sets are identical, so:

- marginally separable query-conflict pairs: 0;
- jointly separable query-conflict pairs: 1;
- the minimum exact action set is `{inspect-x, inspect-y}`;
- effort is 3 expert-review minutes;
- the single-action subsets do not separate the worlds.

This is a synthetic proof that the joint planner can recover complementarity that the marginal rule cannot.

### Shared-joint-tuple negative control

The refuting world's support contains a joint tuple also admitted by the supporting world. Even though the remaining tuples differ, the full projection overlaps. Expected result: no robust joint plan and the opposing-decision pair explicitly returned.

## 6. Benchmarks and promotion criteria

| Metric | Proposed target |
|---|---:|
| Exact subset-plan agreement with independent projection oracle (≤16 actions) | 100% |
| False joint separation when any shared projected tuple exists | 0 |
| Recovery of XOR / complementary-evidence controls | 100% |
| Missing, duplicate or unbound action outcomes accepted | 0 |
| Canonical digest drift under action/tuple ordering permutations | 0 |
| Unsupported independence/probability claims | 0 |
| Exact result determinism on frozen inputs | 100% |

Paired marginal-vs-joint experiments must use identical worlds, query definitions, action costs and raw source evidence. Report marginal-only coverage, joint-only coverage, robustly undiscriminable pairs, exact review effort, runtime, and sensitivity to challenged support additions. Do not compare a sampled support relation with an exhaustive one as if their guarantees were equivalent.

Future extension: challenge joint supports by adding admissible tuples, bound to a separate challenger digest. The same monotonicity rule should hold: adding joint tuples cannot create new robust separation or lower exact minimum effort. This will connect #10525 to the correlated joint planner.

## 7. Comparison with the current industry and research frontier

This work is complementary to the leading proofreading classifiers and reconstruction toolchains. We should not claim that a set-disambiguation planner outperforms a segmentation or proofreading classifier.

- **ConnectomeBench2 (2026)** provides over 716,485 expert-labelled proofreading decisions and more than 4.5 million images across mouse, human, zebrafish and fly. Its shared-encoder ViT reaches human-level split and merge proofreading performance and studies calibration and distribution shift. It is the primary predictive benchmark to reproduce, not a benchmark that this planner has beaten. [Project / dataset](https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/) · [arXiv:2606.21116](https://arxiv.org/abs/2606.21116)
- **The September 2026 VLM ConnectomeBench2 study** evaluated three open VLMs on the complete 101,929-operation test split. LoRA-adapted Gemma 4 12B reached 95.0% split and 90.9% merge balanced accuracy; the benchmark's EM-only ViT-B reported 94.2% and 91.1%. On unseen human and zebrafish species, the best adapted VLM merge-error results were 77.3% and 82.7%, above the reported ConvNeXt-Tiny specialist values of 66.9% and 74.7% for those rows. This is a preprint and is a moving baseline, not a settled universal ranking. [arXiv:2609.36492](https://arxiv.org/abs/2609.36492)
- **NEURD (Nature, 2025)** provides morphology/mesh-based automated proofreading, graph decomposition, feature extraction and downstream analyses. It demonstrates the practical value of integrating reconstruction geometry and proofreading workflows. [Nature paper](https://www.nature.com/articles/s41586-025-08660-5)
- **Uncertainty quantification for connectomics (Nature Methods, 2025)** argues that scalable reconstruction with minimal human supervision requires data- and model-uncertainty methods so downstream analyses can assess trust. [Nature Methods](https://www.nature.com/articles/s41592-025-02945-3)

The differentiator is therefore not a higher split/merge balanced-accuracy score. It is an independently checkable, query-level reliability layer that preserves the *joint* evidence structure, exposes exactly which observation combinations distinguish competing scientific interpretations, and refuses a robust claim when one admissible joint outcome remains shared.

## 8. Next experimental programme

1. Reproduce ConnectomeBench2 V2 on its official frozen splits and compare with the ViT-B, LoRA VLM and specialist baselines. Keep species, spatial splits, label definitions and class mix fixed.
2. Build query-level challenge sets from expert-adjudicated split/merge and synapse decisions. A query must state the circuit-level consequence of each competing reconstruction, not only whether a proofreading edit was accepted.
3. Record joint observations from real multi-view evidence (xy/xz/yz sections, morphology, synapse polarity, segmentation masks and independent review). Validate that joint tuples are scientifically admissible and that the joint support source is not an arbitrary product of marginals.
4. Evaluate the same query cases under marginal V1, robust set-valued V1, support-expansion challenge, and joint-support V1. Report where joint reasoning recovers complementarity and where shared outcomes prevent a guarantee.
5. Target zero false robust promotions on adversarial synthetic controls and independently adjudicated challenge sets; aim for at least 15 percentage points higher query-critical error recall and at least 25% fewer expert-review minutes at matched adjudicated query quality. These are future targets, not achieved results.

## 9. Claim boundary

This tool computes exact finite set projections and minimum-cost action subsets. It cannot prove that joint-support relations include every physically possible observation, that sources are authentic or independent, that measurement protocols are valid, that the query worlds are biologically complete, or that a biological claim is true.

No probability, posterior, likelihood, entropy, expected value of information, causal effect or consciousness claim is made. No acquisition, correction or biological decision is executed.
