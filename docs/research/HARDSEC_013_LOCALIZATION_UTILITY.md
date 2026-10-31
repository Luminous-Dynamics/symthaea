# HARDSEC-013 — Verified Localization Utility and Explanation Faithfulness

**Status:** research protocol / pre-registration draft  
**Scope:** explanation and localization evaluation; no model, explanation system, or human study is claimed to have been run  
**Parent:** [HARDSEC-013 issue #9377](https://github.com/Luminous-Dynamics/symthaea/issues/9377)

## 1. Thesis: optimize the engineering investigation, not the heatmap

A hardware-security detector does not help an engineer merely by returning a high score or painting a region red. Its output must identify a bounded, inspectable part of a specific design, support the mechanism it claims, preserve the relevant source-to-observation context, expose counterevidence and uncertainty, and reduce the cost of a correct investigation.

The primary research unit is therefore an **engineer-review task**: a ranked set of candidate regions with exact identities, source evidence, graph paths, uncertainty and an explicit review budget.

This protocol is additive to existing research:
- [HARDSEC-005 / #9089](https://github.com/Luminous-Dynamics/symthaea/issues/9089): design/source lineage and split integrity.
- [HARDSEC-007 / PR #9108](https://github.com/Luminous-Dynamics/symthaea/pull/9108): evidence-bearing challenge profiles, active/inactive witnesses and localization labels.
- [HARDSEC-009 / PR #9202](https://github.com/Luminous-Dynamics/symthaea/pull/9202): validation-frozen clear/suspect/abstain thresholds and selective risk.
- [HARDSEC-010 / issue #9313 and PR #9314](https://github.com/Luminous-Dynamics/symthaea/pull/9314): graph/path retention and industrial-scale graph accounting.
- [HARDSEC-012 / #9346](https://github.com/Luminous-Dynamics/symthaea/issues/9346): runtime telemetry active-evasion robustness.

This does not create another detector leaderboard or replace those tracks.

## 2. Literature motivation and comparison boundaries

Whitten, Wolff and Papachristou, “Explainability Methods for Hardware Trojan Detection: A Systematic Comparison,” *Journal of Electronic Testing* (published 3 August 2026), DOI [10.1007/s10836-026-06245-2](https://doi.org/10.1007/s10836-026-06245-2), compare domain-aware property analysis, case-based explanation, LIME, SHAP and gradient attribution on Trust-Hub.

Selected results from that paper:
- The held-out random-split test set contains 11,392 gates and is approximately 157:1 benign-to-Trojan.
- XGBoost reports 48.08% precision and 69.44% recall at a validation-selected threshold; Random Forest reports 58.46% precision and 52.78% recall.
- Gradient attribution is reported at 0.16 ms per explanation, SHAP at 1.10 ms, and LIME at 24.4 ms on the paper's evaluated setup. The authors note that gradient approximations can be sensitive near decision boundaries and to finite-difference settings.
- Case-based explanations show 96.51% correspondence between predictions and nearest training exemplars. This is prediction-to-exemplar correspondence, not proof that the exemplar is causally or structurally valid for the held-out design.
- LIME/SHAP per-gate attribution agreement is moderate (Spearman rho 0.30); the paper cautions that generic feature importance lacks circuit-specific context.
- Under leave-one-family-out, the five-scalar-feature XGBoost result falls to micro-F1 0.033 and Random Forest to 0.003. Within-RS232 LOCO is much stronger than cross-architecture ISCAS LOCO.
- The authors restrict their study to synthetic combinational Trust-Hub cases and note the absence of controlled human-subject studies with practicing hardware-security engineers.

Sources:
- Open article: https://link.springer.com/article/10.1007/s10836-026-06245-2
- DOI: https://doi.org/10.1007/s10836-026-06245-2

Additional context must stay separate:
- Xiao et al. (2026) report Random Forest F1 of 0.914 pooled, 0.636 leave-one-netlist-out and 0.460 leave-one-base-family-out across five Trust-Hub host families: https://arxiv.org/abs/2609.07199.
- SeqCube-HT reports graph-probe localization recall falling from 55.7% under instance-random evaluation to 31.4% under insertion-source holdout on its fixed four-circuit corpus: https://doi.org/10.1186/s42400-026-00676-2.

These studies use different datasets, labels, units and protocols. Their numbers are not pooled, averaged, or presented as a common leaderboard.

## 3. Paradigm shift: an explanation is a falsifiable engineering claim

Do not optimize for agreement between attribution methods or attractive saliency maps. Represent an explanation as a set of explicit claims:

1. **Identity:** exact source design/netlist digest and stable node, edge and region identities.
2. **Localization:** which gates, cells, cones, partitions or bounded regions are implicated.
3. **Mechanism:** which circuit-grounded properties and relationships support the localization.
4. **Path:** which declared trigger-to-observation paths are retained, cut, unknown or contradicted.
5. **Counterevidence:** benign explanations, equivalence controls and conflicting observations.
6. **Uncertainty:** calibration scope, OOD status, unsupported semantics and abstention.
7. **Review action:** the smallest useful next inspection, simulation or formal query, with cost measured or explicitly unavailable.

Confidence and fluent prose are not evidence of faithfulness. A source node absent from the pinned design invalidates the associated localization claim. A path not present in the supplied graph cannot be asserted as a verified route. Unknown or cut graph relations must remain unknown.

## 4. Workstream A — review-budget-aware localization

Predeclare review budgets at both region and gate-count levels. Example operating points:
- top 1, 5, 10, 25 and 50 candidate regions;
- fixed candidate gate budgets;
- 0.01%, 0.1% and 1% of design gates.

Report separately at gate, region, cone and declared path levels:
- precision@K and recall@K;
- average precision / PR-AUC;
- verified ground-truth coverage from the union of reviewed candidates;
- candidate gates / edges / regions needed to reach 50%, 80%, 90%, 95% and 99% of the available label footprint;
- false-positive density per million gates and per design;
- number and size of disconnected review fragments;
- cross-partition hops and path reconstruction status;
- evidence artifacts and query obligations presented to the reviewer.

Gate count is an operational proxy, not a measurement of engineer time. Do not report “hours saved” or “analyst effort reduced” without a controlled, preregistered study involving actual reviewers. Any automated review-complexity index must be named as a proxy, with its formula and limitations.

Where labels represent the inserted RTL/netlist footprint, call the endpoint **insertion-footprint localization**. Do not call it causal localization unless an independent functional/causal oracle establishes that the identified logic is necessary or sufficient for the tested effect.

## 5. Workstream B — explanation faithfulness and falsification

Every explainer/model pair should face independent controls:

### 5.1 Deletion, insertion and sufficiency

Test whether retaining top-ranked implicated features/regions preserves the prediction and whether removing them changes the score in the predicted direction. Apply only valid, well-defined interventions. Arbitrary graph deletion can create an invalid circuit and must not be treated as a meaningful counterfactual.

### 5.2 Equivalence-preserving benign transformations

Use exact, independently checked transformations such as identifier renaming, representation reorderings and Boolean gate decomposition where formal equivalence can be established. Bind source digest, output digest, transformation revision/configuration and equivalence result.

A model may change internal gate IDs after synthesis; compare semantic regions and mapped source identities rather than demanding literal node-ID equality. Equivalence without a valid source mapping does not establish localization stability.

### 5.3 Sensitivity and negative controls

Meaningful changes to the labelled or functionally validated malicious cone should change the explanation. Conversely, benign equivalent rewrites should preserve the mapped semantic explanation within a predeclared tolerance.

Include:
- high-attribution benign decoy regions;
- omitted or mutated path edges;
- source IDs that do not exist in the pinned input;
- explanations unchanged after the purported malicious cone is removed;
- case-based examples whose source lineage overlaps the held-out design;
- no-path / unsupported-semantics cases that require abstention.

### 5.4 Path support and completeness

Every claimed trigger-to-payload or source-to-observation route must bind exact edge identities and the graph/source digest. Reconcile it with the retention/exclusion receipt from [PR #9314](https://github.com/Luminous-Dynamics/symthaea/pull/9314).

A path cut but explicitly reported is a coverage limitation; silent path loss is a failure. A supplied witness proves only that supplied witness, not completeness of the path set. The benchmark must keep these claims separate:
- inserted/labelled logic footprint;
- functionally causal logic;
- activation path;
- observed payload effect.

They are different oracles and cannot substitute for one another.

## 6. Workstream C — structural and semantic explanations

Compare on identical held-out designs and review budgets:
1. SHAP, gradient and LIME where their exact implementation/setup is reproducible;
2. case-based retrieval with exact training-example lineage;
3. domain-aware circuit properties;
4. graph/subgraph localization;
5. HDC binding/retrieval of region, relation and evidence identity;
6. hybrid ranking plus explicit source/path/evidence receipts.

A nearest training example is a precedent, not independent evidence about a test design. A benchmark label cannot be used both to create the explanation and to validate that same explanation. Keep explainer training/tuning data and independent ground truth separate.

## 7. Workstream D — hierarchy-aware validation

Report separate results for every source-supported level:
- within-host diagnostic split;
- leave-one-netlist-out;
- leave-one-base-family-out;
- leave-one-Trojan-family-out;
- leave-insertion-source-out;
- leave-synthesis-tool / cell-library-out;
- unseen-scale stress cases where labels and source lineage support a valid comparison.

All feature selection, transform fitting, model/explainer tuning, threshold selection and review-budget selection must occur inside training/validation groups. Outer held-out designs/families must not influence any learned or selected choice.

Keep runtime physical telemetry, simulator telemetry, RTL labels, gate-level insertion labels, formal witnesses and synthetic graph stress fixtures in distinct evidence lanes.

## 8. Proposed success gates

The following are project decision targets, not established standards or achieved Symthaea results. A localization method can claim practical improvement only after source, lineage, path and equivalence controls pass and one or more of these are demonstrated:

1. At a fixed review budget, improve macro host-family verified localization recall by at least 10 percentage points over the strongest matched baseline, with paired family-level uncertainty and no worst-family regression beyond a predeclared margin.
2. At matched verified-positive recall, reduce candidate gates/edges or review regions by at least 25%, without losing declared trigger-to-observation paths.
3. At matched localization recall and false-alert budget, achieve at least 2x faster explanation generation on named hardware while passing the same fidelity/stability tests.
4. At least 95% of accepted explanations bind every stated source node/edge/path to independently verifiable source identities; 100% of cut/unknown path relations are surfaced.
5. No systematic loss of semantic localization under verified benign equivalent rewrites.

Human-effort claims require measured, consented, controlled studies. A gate-count proxy cannot substantiate human-time savings.

## 9. Matched baselines and reporting

Required comparisons, where their source protocols match:
- Hasegawa five-feature SVM and same-protocol RF/XGBoost;
- domain-aware 31-property analysis and case-based kNN from the 2026 explainability paper;
- SHAP, gradient and LIME under identical model/features/input;
- graph/GNN and saliency-localization baselines;
- threshold-only ranking at the same review budget;
- Symthaea HDC/CfC or hybrid region retrieval as an additional candidate, not a presumed winner.

Publish paired comparisons on identical held-out designs, identical review budgets, matched compute, exact supports and all failed/null results. Do not claim a model is better because it produces a smoother heatmap or a faster explanation if fidelity, coverage or false-positive burden worsens.

## 10. Initial safe fixtures

1. Known-answer labelled graph with multiple disconnected candidate regions.
2. Paired benign design with a structurally similar, formally equivalent rewrite.
3. Identifier renaming and gate decomposition with checked Boolean equivalence.
4. A labelled region whose declared source-to-observation path crosses multiple partitions.
5. A partitioned path explicitly cut and reported; completeness must become unknown.
6. A high-attribution benign decoy.
7. An explanation pointing to a node absent from the pinned source digest.
8. An explanation unchanged after its purported malicious cone is removed.
9. A retrieved training exemplar that shares lineage with a held-out test design.
10. A design with unsupported semantics/no supplied path; the explainer must abstain from a complete mechanism claim.

These fixtures qualify evaluator semantics only. They are not a hardware-detector score, physical-silicon result or security certification.

## 11. Machine-readable detector-facing contract

The initial JSON contract is at `schemas/hardsec-localization-explanation-v1.schema.json`, with a non-production known-answer example at `tests/fixtures/hardsec/localization-explanation-v1.json`.

The record binds source/netlist and node-edge-map digests, design/source grouping, model and training-lineage digests, explainer/configuration identity, frozen threshold profile, candidate regions, exact path references, counterevidence, uncertainty and review-budget/cost semantics.

Ground-truth Trojan labels and evaluator-oracle answers are intentionally absent from the detector-facing schema. They belong in a separately access-controlled evaluator input and must never be passed to the explainer/model being scored.

The JSON Schema validates record shape and local field constraints. It cannot by itself prove:
- region node/edge IDs exist in the source graph;
- edge endpoints form a continuous path;
- a path set is complete;
- group IDs truthfully describe source lineage;
- equivalence evidence is valid;
- the model or explanation was produced by the declared digest.

Those are semantic-auditor obligations. The fixture is a schema example, not a valid detector result or a benchmark score.

## 12. Implementation sequence

1. Freeze the metric dictionary and review budgets. Reuse evidence/profile/path structures from #9108, #9202 and #9314.
2. Implement a Rust explanation/localization receipt auditor with strict hashes, node/edge/path binding, review-budget curves and mutation tests.
3. Add source-to-output mapping for exact-identity and formally verified benign transformations. Do not create an equivalence oracle inside the explainer being evaluated.
4. Compare existing explanation outputs on identical lineage-aware holdouts and review budgets.
5. Run a human study only when actual reviewers, consent, preregistered tasks and enough independent designs are available.

## 13. Acceptance criteria

- [ ] Review budgets and metrics are frozen before confirmatory evaluation.
- [ ] Source/model/explainer/threshold/transform digests are bound in each receipt.
- [ ] Gate, region, cone and path localization are reported distinctly.
- [ ] Review-budget coverage and false-positive density are reported.
- [ ] Faithfulness, stability, sensitivity and sufficiency are separate endpoints.
- [ ] Benign-equivalence controls carry independent proof/equivalence evidence.
- [ ] Held-out design/family/source groups remain isolated through model and explainer tuning.
- [ ] Graph paths reconcile with #9314; unknown/cut paths block completeness claims.
- [ ] Human workload claims are withheld until measured in controlled engineer studies.
- [ ] Negative controls and independent receipt verification pass before explanation-quality/SOTA claims.

## 14. Claim ceiling

This work can establish explanation fidelity, stability, localization coverage and review-budget behavior for named models, datasets, transformations and source groups. It does not prove a Trojan-free design, explain arbitrary analog/sequential behavior, establish industrial generalization from Trust-Hub or certify a chip.
