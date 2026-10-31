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

### 2.1 Additional GNN and industry assurance comparators

A September 2026 *Microelectronics Journal* paper, “A unified graph neural network pipeline for hardware Trojan detection,” reports a two-layer GCN on 55 synthesized designs spanning the AES-T100–T2800 family and PIC16F84-T200. It reports 87.27% accuracy and ROC-AUC 0.9563 on its combined dataset, with saliency-guided subgraph localization assessed by node-level IoU. This is a relevant structural/localization baseline, not a transferable SOTA score: its 55-design/two-family setup must be re-evaluated under lineage-aware and source-held-out splits, matched review budgets, and the same independent path/equivalence oracles before any comparative claim. DOI: [10.1016/j.vlsi.2026.102775](https://doi.org/10.1016/j.vlsi.2026.102775).

Industry assurance is broader than a detector leaderboard. NIST IR 8615 (published 1 September 2026) records priorities for interoperable evidence-based trust models, lifecycle provenance/traceability, supply-chain assurance and scalable continuous verification/validation: https://doi.org/10.6028/NIST.IR.8615. OpenTitan is a concrete open silicon Root-of-Trust comparator: its documented security model covers provisioning, secure boot, attestation, lifecycle, firmware update, chip identity and ownership transfer, while its penetration-testing framework targets FPGA and silicon implementations with side-channel and fault-injection testing: https://opentitan.org/book/doc/security/.

The engineering opportunity is to make localization evidence consumable by this wider assurance chain—source-identified, path-bound, reproducible and explicit about unknowns—rather than claiming that an ML detector replaces formal verification, test, attestation, silicon validation or supply-chain controls.

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

## 11. Machine-readable contract and executable semantic audit

Three separate contracts are now versioned:

- Detector-facing explanation: `schemas/hardsec-localization-explanation-v1.schema.json`.
- Source graph: `schemas/hardsec-source-graph-v1.schema.json`.
- Evaluator-controlled insertion-footprint oracle: `schemas/hardsec-localization-oracle-v1.schema.json`.

The detector-facing explanation never includes ground-truth labels or oracle answers. The oracle is supplied as a separate evaluator input and must remain inaccessible to the detector/explainer during inference and explanation generation. Merely putting it in a different file does not enforce access control; the campaign runner/operator must enforce that trust boundary.

### 11.1 Rust receipt auditor

The first semantic auditor is implemented at `examples/hardsec_localization_audit.rs` and compiled through an isolated nested workspace at `tools/hardsec-localization-audit-harness`. It:
- hashes exact explanation, graph-document and oracle bytes;
- carries three distinct source identities: the original design/netlist digest (`source_artifact_sha256` declared in the graph manifest), the exact graph-document digest, and a canonical node/edge-map digest;
- requires explanation and evaluator oracle source digests to match the graph manifest's original design/netlist digest;
- verifies design and source identity bindings;
- rejects unknown candidate nodes/edges, candidate edges whose endpoints fall outside their declared region, duplicate region IDs/ranks, duplicate graph IDs and discontinuous retained paths;
- validates model, explainer and threshold digests, decision score bounds and supported path-coverage states;
- checks retained declared paths against the separately supplied reference-path oracle;
- only permits a `complete` path-coverage claim when the caller declares the oracle reference-path set complete and every supplied reference path is matched;
- reports insertion-footprint precision/recall over region, gate or edge review budgets, including a fixed review-budget curve;
- preserves unknown/contradicted path status and rejects causal/mechanism claims from an insertion-footprint-only oracle;
- writes a deterministic unsigned no-overwrite receipt outside input directories.

Example command:

```bash
cargo run --locked \
  --manifest-path tools/hardsec-localization-audit-harness/Cargo.toml \
  --features hardsec-localization-audit \
  --example hardsec_localization_audit -- \
  --explanation tests/fixtures/hardsec/localization-audit-v1/explanation.json \
  --graph tests/fixtures/hardsec/localization-audit-v1/source-graph.json \
  --oracle tests/fixtures/hardsec/localization-audit-v1/evaluator-oracle.json \
  --receipt /tmp/hardsec-localization-receipt.json
```

The receipt's score is an evaluator result, not a `pass`. The tool does not parse RTL, prove equivalence, independently validate a graph extractor, authenticate the oracle, or prove that the reference-path set is exhaustive. A `cut_and_reported` status is retained as a caller-declared status; this first version does not independently replay the partitioner or verify an external path-retention receipt.

### 11.2 Deterministic fixtures and focused workflow

The known-answer fixture labels nodes `g17` and `g18` as insertion-footprint positives; the top-ranked candidate region contains both, yielding precision 1.0 and recall 2/3 at a one-region review budget. A separate benign decoy region is included. This is synthetic evaluator-mechanics evidence only.

Eighteen focused unit tests are authored for known-answer scoring, source digest mutation, unknown node mutation, candidate-region endpoint mismatch, malformed model digest, out-of-range decision score, unsupported path-coverage state, discontinuous paths, incomplete-oracle completeness claims, causal-claim rejection, duplicate ranks, unknown oracle labels, retained-path mismatch, edge review budgets, unknown path handling, exact complete-path matching, empty path sets and review-curve emission. The dedicated workflow runs rustfmt, the locked Cargo suite and the known-answer CLI receipt assertion.

The JSON schemas describe record shape; the semantic auditor is responsible for cross-record ID/path checks. The graph manifest's `source_artifact_sha256` is a caller-declared commitment to the original design/netlist: this auditor does not receive or hash the original netlist bytes, so it checks that explanation/oracle agree with the graph manifest but cannot authenticate that source commitment. The graph-document digest and canonical node/edge-map digest are separate, independently recorded identities. Neither schemas nor receipt prove the authority of labels, equivalence certificates or completeness statements.

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
