# Connectomics Autopilot Research Program V2

**Date:** 2026-10-31  
**Status:** research and qualification programme; no new biological performance claim  
**Primary implementation:** PR [#8883](https://github.com/Luminous-Dynamics/symthaea/pull/8883)  
**Research tracks:** [#9241](https://github.com/Luminous-Dynamics/symthaea/issues/9241), [#9242](https://github.com/Luminous-Dynamics/symthaea/issues/9242), [#9248](https://github.com/Luminous-Dynamics/symthaea/issues/9248), [#9250](https://github.com/Luminous-Dynamics/symthaea/issues/9250), [#9255](https://github.com/Luminous-Dynamics/symthaea/issues/9255), [#9260](https://github.com/Luminous-Dynamics/symthaea/issues/9260), [#9265](https://github.com/Luminous-Dynamics/symthaea/issues/9265), [#9282](https://github.com/Luminous-Dynamics/symthaea/issues/9282), [#9322](https://github.com/Luminous-Dynamics/symthaea/issues/9322), [#9345](https://github.com/Luminous-Dynamics/symthaea/issues/9345), [#9344](https://github.com/Luminous-Dynamics/symthaea/issues/9344), [#9379](https://github.com/Luminous-Dynamics/symthaea/issues/9379), [#9403](https://github.com/Luminous-Dynamics/symthaea/issues/9403).

## 1. Strategic thesis

Symthaea should not attempt to replace every specialist connectomics system. It should orchestrate specialist reconstruction, proofreading, acquisition and annotation systems through an independently auditable scientific-evidence layer.

The target is not merely a cleaned graph. It is a **query-scoped, versioned, uncertainty-aware scientific result** whose evidence, population coverage, plausible alternatives, correction lineage and invalidation conditions can be independently checked.

The key distinction is:

> A reconstruction may be locally plausible while a scientific conclusion derived from it remains unsupported.

The system must separately establish:

1. **Identity and lineage:** which exact source data, segmentation, annotations, transforms, models and corrections were used?
2. **Candidate discovery:** what error hypotheses were generated, and what families/regions may the search channels have missed?
3. **Population coverage:** what declared eligible population was searched, and was its sampling frame complete?
4. **Correction validity:** which proposed edits are compatible, interacting, reversible and independently verified?
5. **Query robustness:** does the scientific result survive across plausible reconstructions and uncertainty assumptions?
6. **Currentness:** which new source events invalidate prior evidence or certificates?
7. **Operational value:** does the workflow improve independently adjudicated scientific outcomes per total resource cost?

These are independent assurance claims. Passing one does not imply the others.

## 2. Paradigm shifts

### 2.1 From one graph to a set of plausible reconstructions

A single best-scoring graph hides unresolved identity and synapse alternatives. Query results should be evaluated over a provenance-bound finite scenario set, set-valued envelope or empirically weighted ensemble. Probability language is allowed only when the ensemble construction, dependence assumptions and calibration justify it.

**V1 implementation added to PR #8883:** `connectome-query-ensemble-audit` evaluates the same frozen graph query over explicit finite scenarios, verifies canonical graph hashes, preserves shared evidence-lineage groups, sorts results deterministically, and reports the minimum/maximum value and decision stability. Its finite unweighted scenario semantics explicitly make no probability or posterior claim. The synthetic fixture includes a path query that changes across reconstructions and two stable controls. See `CONNECTOME_QUERY_ENSEMBLE_AUDIT.schema.json` and `CONNECTOME_QUERY_ENSEMBLE_AUDIT_PROTOCOL_V1.md`.

The implementation is tracked in [#9344](https://github.com/Luminous-Dynamics/symthaea/issues/9344). Its first Rust evaluator is deliberately finite/unweighted, includes canonical graph/scenario hashes and ten authored regression tests, and makes no probability claim. A separate follow-on, [#9379](https://github.com/Luminous-Dynamics/symthaea/issues/9379), adds the analysis-pipeline axis: query outcomes across a reconstruction × admissible-pipeline multiverse, with reconstruction effects, pipeline effects and their interaction kept separate. This is motivated by adjacent functional-connectomics pipeline-multiverse evidence, not claimed as an EM benchmark. The certificate structure in #9255 must consume these results without claiming that a DAG digest proves biological truth.

### 2.2 From graph nodes to continuous neurite evidence

A co-linear false merge can sit inside one non-branching skeleton segment. Sliding-window change-point detection is a first baseline, not a biological identity test. Use physical arclength, joint-noise calibration, independent identity evidence and explicit abstention. See [#9265](https://github.com/Luminous-Dynamics/symthaea/issues/9265) and [#9322](https://github.com/Luminous-Dynamics/symthaea/issues/9322).

### 2.3 From sample-index heuristics to physical measurement contracts

A detector must be stable to resampling of the same physical object. Window sizes, localization tolerances and false-proposal rates must be expressed in physical coordinates and tied to voxel anisotropy, transform provenance and annotation uncertainty. Sampling-density invariance is a qualification gate, not a cosmetic robustness metric.

### 2.4 From candidate ranking to joint action planning

Corrections interact. The edge-only exact optimizer in #9250 is a useful small-instance baseline, but biological operations require typed split/merge/reassign/continue actions, immutable object identities, preconditions, read/write sets, dependencies, conflicts, validation and rework costs.

### 2.5 From uncertainty scores to scientific value of information

Entropy or detector confidence is not scientific value. Choose between re-imaging, specialist tracing, morphology inspection and human review based on the expected reduction in query loss per total cost, with an independent audit of untouched regions. See [#9282](https://github.com/Luminous-Dynamics/symthaea/issues/9282).

### 2.6 From “proofread complete” to query-scoped assurance

A global graph invariant or completed queue cannot certify every path, motif, cell type, synapse family or circuit claim. Certificates bind a query, exact dependency closure, sources, sampled population, uncertainty and revision history. See #9255.

### 2.7 From global convergence to explicit biological completeness

Stable graph invariants are not synaptic completeness and are not proof that a query-critical circuit is correct. The July 2026 connectome-stopping preprint reports a zebrafish case in which its global invariant composite is estimated at roughly 85% of projected asymptote at 50,000 edits while synaptic completeness is only about 20%; the authors define the latter by whether both pre- and postsynaptic locations belong to fully proofread cells. This is a preprint result, not a universal benchmark or stopping threshold. It motivates an explicit false-stop control in [#9007](https://github.com/Luminous-Dynamics/symthaea/issues/9007).

**Executable increment in PR #8883:** stopping snapshots may now carry optional `synaptic_completeness` numerator/denominator evidence, bound to a population-definition digest, synapse-population digest and fully-proofread-cell-set digest. The audit reports the latest fraction and separate tail status. Missing records, partial tails and zero denominators are not silently dropped or converted to zero. A checked-in three-snapshot synthetic false-stop example has stable graph invariants, a stable 20% synaptic completeness and a rare query-critical path count changing from 12 to 1; the tool still recommends `CONTINUE_REVIEW`. The metric is never inferred from graph invariants, and the tool does not independently verify the endpoint memberships—the evidence producer must do that. The stopping-audit source now has 21 authored regression tests, including the synthetic false-stop fixture, missing-completeness, zero-denominator and population-drift controls; these are authored counts, not executed test results.

### 2.8 From static certification to event-driven currentness

A valid certificate is not valid forever. Reuse the repository's broader lifecycle work:
- [#9210](https://github.com/Luminous-Dynamics/symthaea/issues/9210): trusted workspace/source event capture;
- [#9239](https://github.com/Luminous-Dynamics/symthaea/issues/9239): dependency-observation completeness;
- PR [#9203](https://github.com/Luminous-Dynamics/symthaea/pull/9203): evidence timeline/invalidation consumer;
- [#2618](https://github.com/Luminous-Dynamics/symthaea/issues/2618): append-only qualification invalidation and requalification semantics;
- [#7412](https://github.com/Luminous-Dynamics/symthaea/issues/7412): measurement uncertainty provenance.

Do not create another event or invalidation model. Connectomics should supply typed dependency identities and query-specific invalidation policy to those shared primitives.

## 3. Dataset and evidence qualification rules

### 3.1 ConnectomeBench2 V2 is the local decision benchmark

The public dataset card identifies 401,170 rows across mouse, fly, human and zebrafish, with real merge/split edits and synthetic controls. It explicitly warns that `main` still points to the initially submitted V1 and that revision `v2` is the revision used by the arXiv paper. The exact immutable V2 commit and every downloaded shard/sidecar digest must be recorded before reproduction.

The card also reports substantial modality missingness:
- all rows have geometry;
- 251,499 rows have EM views (63% overall);
- only 38% of `merge_edit` and 23% of `split_edit` rows have EM views;
- synthetic controls have EM views, so a naïve pooled EM subset can alter task composition.

Therefore:
- report geometry-only and EM-eligible cohorts separately;
- compare models on identical examples and identical modality availability;
- report missing-modality rates by task, species and sample type;
- do not call an EM result representative of all real human edits without a justified sampling/weighting analysis;
- keep the all-geometry cohort as a distinct benchmark arm.

The card warns that `em_best` uses an oblique slice selected using both segment labels and can leak dual-label information for single-segment tasks. For single-segment tasks, fuse A/B EM mask channels and exclude `em_best`. Merge correction uses dual-segment geometry; split correction uses single-segment geometry. The intake auditor must preserve the exact task-routing/label mapping and reject mismatched representations.

The public split uses 50 µm spatial cubes. That does not by itself establish independence by neuron, source operation or annotation lineage. Audit `operation_id`, `source_operation_id`, root identities, physical regions and source archive relationships. Unknown linkage must remain unknown; it must not be recoded as independence.

Dataset card: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2  
VLM benchmark: https://arxiv.org/abs/2609.36492  
ConnectomeBench2 research: https://arxiv.org/abs/2606.21116

The row-level intake auditor checks `has_em` against metadata `present_slots` and now emits stratified modality coverage for overall, task route, sample type, species, split, source archive and the required cross-products. It reports any EM slot, missing EM, each of `em_xy`/`em_xz`/`em_yz`, full three-view availability, `em_best` separately, and fused geometry+usable-EM eligibility. The synthetic sidecar fixture now includes both a usable EM view and an `em_best`-only negative control. The dataset auditor has 28 authored source tests, not 28 passing tests. This report remains metadata-only: it does not verify sidecar claims against actual image/geometry bytes in Parquet shards. The byte-level join, immutable eligibility manifests and paired-cohort evaluator remain open in [#9345](https://github.com/Luminous-Dynamics/symthaea/issues/9345). Comparisons must use matched sample-ID intersections for pure modality ablations and separate full eligible-cohort outcomes. Do not infer population inclusion probabilities from missingness percentages.

### 3.2 Biological source artifacts are a separate qualification tier

Local CB2 renderings do not automatically provide complete neuron meshes, full skeletons, source EM volumes or a valid spatial eligibility frame. Resolve every biological morphology claim to the exact source volume, segmentation, mesh, skeleton, synapse table, coordinate transform and version. A task-level image benchmark cannot be used to claim whole-neuron reconstruction performance.

### 3.3 Distinguish evidence from truth

A content digest establishes identity/integrity, not biological correctness. Expert consensus is an adjudication reference, not automatically ground truth. Synthetic injected errors are controlled sensitivity data, not natural error prevalence. A high-resolution image is not automatically a gold standard. Keep each claim type separate in reports.

## 4. Industry comparison: matched tasks only

| System / benchmark | What it establishes | Symthaea comparison |
|---|---|---|
| ConnectomeBench2 | Operation-level split/merge and control decisions using mesh/EM renderings | Reproduce exactly on pinned V2 with matching inputs, split, task routing and modality coverage |
| Adapted VLMs (arXiv:2609.36492) | Open VLM LoRA/shot/zero-shot task-level comparison and species transfer | Include as an external baseline only where source data, model artifacts and protocol are reproducible |
| NEURD | Mesh/skeleton decomposition, morphology-conditioned merge proofreading and graph features | Matched source volumes and adjudications; test documented co-linear gaps, incomplete processes and synapse-preservation tradeoffs |
| RoboEM | 3D-EM neurite tracing/continuation | Compare recovered neurite path/completeness, false joins, expert time and compute—not unrelated binary merge accuracy |
| SmartEM | Adaptive SEM dwell-time allocation; reports up to ~7-fold acquisition acceleration in demonstrated settings | Compare imaging time/quality tradeoffs and downstream query value on compatible acquisition data |
| CAVE | Collaborative proofreading, annotation materialization, versioned historical queries and lineage | Integrate as a versioned data/action source; do not claim it is a candidate detector |
| SynAnno | Interactive guided synaptic proofreading workflow | Compare reviewer navigation and task time on matched annotation tasks; do not present it as a split/merge classifier |

Relevant sources:
- NEURD (Nature, 2025): https://www.nature.com/articles/s41586-025-08660-5
- RoboEM (Nature Methods, 2024): https://www.nature.com/articles/s41592-024-02226-5
- SmartEM (Nature Methods, 2026): https://www.nature.com/articles/s41592-025-02929-3
- CAVE (Nature Methods, 2025): https://www.nature.com/articles/s41592-024-02426-z
- SynAnno (IEEE TVCG, 2026): https://pubmed.ncbi.nlm.nih.gov/41396767/
- Uncertainty quantification in connectomics (Nature Methods, 2025): https://www.nature.com/articles/s41592-025-02945-3

The original ConnectomeBench2 paper reports its full multimodal ViT-B at 97.0% split-error and 93.0% merge-error balanced accuracy (ViT-L: 97.2% and 92.8%), compared with expert baselines of 93.0% and 84.1%. These are operation-level held-out benchmark results, not whole-connectome recall. The September/October 2026 VLM preprint reports on its full ConnectomeBench2 test split (101,929 samples) an **EM-only** ViT-B score of 94.2% split-error and 91.1% merge-error balanced accuracy; Gemma 4 12B LoRA reports 95.0% and 90.9%. The EM-only comparison must not be mistaken for the original full multimodal CB2 headline result. The VLM paper's unseen-species and balanced-subset results differ by task and species. These values are protocol-specific comparators, not whole-connectome recall or scientific-query correctness. Reproduction requires the exact cohort, modality eligibility, and evaluation procedure of the relevant paper.

## 5. Benchmark ladder and promotion gates

Do not jump directly from synthetic unit tests to biological SOTA claims. Promotion is a sequence of distinct evidence levels.

### Tier A — deterministic engineering correctness

- Exact source/schema/artifact identity and digest verification.
- Lattice tiling, boundary tile arithmetic and packed eligibility-mask integrity.
- Deterministic canonical hashes and input-order invariance.
- Fail-closed malformed, missing, duplicated, stale and out-of-range inputs.
- Exact synthetic dependency closure and invalidation oracle.
- Exact small-portfolio enumerator compared with an independent exhaustive oracle.
- Physical-arclength resampling metamorphic tests.
- Repeated execution produces identical report bytes when inputs and binary are fixed.

**Gate:** zero missed adversarial faults across the enumerated test suite. Report test count separately from executed/passing status.

### Tier B — statistical calibration on known finite populations

- Repeated-seed finite-population simulations for the actual sampling design.
- Confidence-interval coverage and estimator bias by stratum/error family.
- Controlled dependence across discovery lists and reviewers.
- Missing-not-at-random and heterogeneous detectability.
- False robust-certification and false probabilistic-certification tests.
- OOD, selective prediction and abstention under known distribution shifts.
- Multiplicity control for multiple change-point tests along a cable.

**Gate:** preregister calibration tolerances; refuse identification/estimation when assumptions or support fail. No IID claims from profile windows or image crops.

### Tier C — semi-synthetic and natural adjudicated data

- Controlled injections and natural errors reported separately.
- Development/calibration/lockbox split by volume, neuron/root, operation and physical region.
- Adjudication blinded to detector/policy identity, with disagreements preserved.
- Probability-sampled PSUs include zero-candidate regions and all detector-negative segments.
- All source modalities and exclusions are recorded.

**Gate:** no lockbox tuning, no outcome leakage, minimum per-family support, and independent sampling-frame audit.

### Tier D — prospective workflow utility

Use the block-randomized design in #9242. Keep interfaces and permissions identical when comparing generator/ranker effects. Use intention-to-treat, block-level randomization inference, independent adjudication, and independent audits outside the proposal pool.

**Gate:** improvement in independently verified scientific-query error reduction per actual person-minute, without increased incorrect accepted corrections or loss of critical-family recall.

### Tier E — scientific conclusion qualification

Use #9344 to evaluate the query across plausible reconstruction sets. Use #9255 to bind query semantics, source revisions, dependency closure, evidence, uncertainty and invalidation policy.

**Gate:** zero false robust certificates on exhaustive synthetic controls; uncertainty coverage only when its probability model is justified; explicit CONDITIONAL, SENSITIVE, NOT_IDENTIFIABLE or NOT_ESTIMABLE outcomes as appropriate.

## 6. Metrics that must be reported

### Local decision quality
- Balanced accuracy, precision, recall and PR-AUC with exact class/task definitions.
- Per-species, per-region, per-source archive and per-error-family denominators.
- Matched geometry-only, EM-only and fused-modality cohorts.
- Calibration (Brier, reliability curves, ECE with bin sensitivity) and selective risk/coverage.
- Label efficiency: expert labels needed to reach a frozen performance/risk target.

### Candidate-generation completeness
- Design-weighted candidate recall and impact-weighted recall from #9204.
- In-pool residual-risk estimates from #9187, reported separately.
- Multi-list unseen-error sensitivity from #9248, with model-dependence sensitivity and NOT_IDENTIFIABLE status.
- Family-specific support and lower confidence bounds.
- Empty PSUs, false-negative proposal counts and false proposals per neuron/cable-mm.

### Morphology and reconstruction quality
- Hidden-boundary localization error in physical units, including median/P90/P95 and tolerance-hit rate.
- False proposals per cable-mm and per review-minute.
- Recovered true neurite cable length, synapse precision/recall, branchpoint preservation and false-join rate.
- Axonal/dendritic and compartment-specific preservation outcomes.
- Performance under anisotropy, missing sections, folds/tears, low contrast and transform drift.

### Scientific query validity
- Query-output error and interval/range coverage across exact plausible reconstruction scenarios.
- Robust-decision stability and fraction of queries that become NOT_IDENTIFIABLE.
- False robust-certificate rate and stale-certificate rate after dependent source events.
- Query impact by error family and correction portfolio, including negative downstream effects.
- Proximity-conditioned null models for circuit motifs; raw motif counts are not evidence of biological enrichment.

### Operational utility and cost
- Independently adjudicated query-error reduction per actual expert person-minute.
- Time to fixed quality/risk target and expert minutes per verified correction.
- Rework, reversal, stale/conflicting operation and verification rates.
- Microscope time, high-resolution voxel volume, compute, GPU/CPU hours, memory, storage, data egress and adjudication costs.
- All-in cost per query meeting its declared error budget.

Never use one pooled score to hide a critical-family failure. Never substitute item accuracy for population recall or query validity.

## 7. Quantitative research targets

All numbers below are **proposed promotion targets, not current results**. Final tolerances and sample sizes must be preregistered using pilot variance and physical annotation uncertainty.

| Area | Target |
|---|---|
| Spatial frame | Zero missed/duplicate lattice units and mask inconsistencies in adversarial tests |
| CB2 V2 local decisions | Reproduce the pinned baseline within 1 percentage point before attempting improvements |
| Label efficiency | ≥20% fewer expert labels to a fixed held-out risk/quality target than the strongest eligible active-learning baseline |
| Candidate recall | ≥95% point estimate and design-adjusted lower 95% bound ≥90% for each critical, sufficiently supported error family |
| Hidden co-linear merges | ≥90% localization within preregistered physical tolerance; median ≤200 nm only when resolution/annotation uncertainty supports it |
| False proposals | No worse than strongest matched baseline at fixed recall; always report per cable-mm and per reviewer-minute |
| Neurite continuation | False-join rate no worse at matched recovered cable length; preserve axonal synapses and branchpoints |
| Portfolio selection | Zero regret against exhaustive optimum on fully enumerable small instances; ≥10% improvement over strongest frozen non-oracle policy on larger held-out instances |
| Certificate invalidation | Zero stale certificates after any dependent revision; no unnecessary invalidation for changes proven unrelated by a qualified dependency profile |
| Query robustness | Zero false robust certifications on exhaustive synthetic controls; 90/95% interval coverage only under justified calibrated ensemble semantics |
| Workflow utility | ≥20% fewer expert minutes at matched independently adjudicated scientific quality in a prospective trial |
| Acquisition policy | ≥20% more query-loss reduction per total cost than the strongest single-action baseline, with uncertainty excluding zero |
| Total cost | Target ≥20% lower all-in cost per query meeting its declared error budget |

A result is not promoted because it beats one baseline on one metric. Promotion requires a Pareto improvement or a justified tradeoff with explicit risk budget.

## 8. Experimental design and leakage barriers

- Pin ConnectomeBench2 revision `v2`, upstream code revision, shard list, every source-file digest and sidecar digest.
- Preserve exact task routing, label semantics, geometry version, EM channel transformations and modality eligibility.
- Exclude label-conditioned `em_best` where the dataset card warns of leakage.
- Group splits by source archive, source operation, neuron/root, physical cube and volume; keep alternative renders/profile resamples of one object in the same partition.
- Freeze model, feature extractor, detector configuration, probability frame, candidate pool, error-family vocabulary, query weights and adjudication protocol before lockbox outcomes.
- Do not allow reviewers or independent discovery channels to share proposals before their capture lists are frozen.
- Preserve raw capture histories for multi-list estimation; uncertain event linkage remains probabilistic/ambiguous.
- Sample all eligible segments/PSUs, not only regions where a model proposes a candidate.
- Use cluster-aware uncertainty at the volume/neuron/operation/PSU unit. Image crops and adjacent profile windows are not independent observations.
- Keep evaluator-only truth, adjudication outcomes and query references out of candidate rankers and acquisition policies.
- An unavailable comparator is reported as unavailable/not comparable, not as a failed baseline.

## 9. Implementation order

1. **Qualify the existing intake and frame foundation:** exact V2 manifest/artifacts, source and transform lineage, complete eligible lattice and mask, candidate recall integration.
2. **Qualify the co-linear detector:** #9322 physical-arclength windows, resampling invariance, null calibration, identity evidence and abstention. Keep V1 as baseline.
3. **Upgrade correction semantics:** #9250 typed biological actions and complete lifecycle cost; preserve the current edge-only exact enumerator as a baseline.
4. **Implement reconstruction/query ensembles:** #9344 finite scenario enumeration first, then justified weighted ensembles and partial-identification semantics.
5. **Connect certificate lifecycle:** #9255 certificate contents with #9210/#9239/#9203/#2618 event/currentness primitives; bind observation completeness and metric uncertainty provenance.
6. **Develop active evidence acquisition:** #9282 paired-resolution/repeat-acquisition calibration and replay before any live microscope control.
7. **Run the prospective workflow trial:** #9242 only after both arms, randomization blocks, sample size and safety constraints are frozen.

## 10. Current evidence boundary

PR #8883 includes source for the V2 intake/scorer, spatial frame/mask audit, candidate-generation coverage, metadata-sidecar modality coverage report, co-linear synthetic baseline, exact small graph-edge portfolio evaluator, query dependency DAG checks, and the finite reconstruction query ensemble evaluator. The query ensemble has a strict schema, synthetic fixture, protocol, README instructions, CI path triggers and ten authored unit tests. The dataset auditor has 35 authored unit tests including the new modality strata/EM-best control. The frozen proofreading scorer has 49 authored tests after adding optional domain-shift calibration, novelty diagnostics, fixed/adaptive reliability views, per-bin calibration intervals, cluster-bootstrap intervals, frozen detector provenance and strict split-manifest membership/leakage checks. JSON Schema/fixture validation and all three query-scenario graph digest recomputations were independently checked outside Rust. Authored counts are not passing counts, and the Rust compiler/test suite has not been run for this newest head.

The live PR head and its CI status must be checked at each qualification attempt. A missing workflow run/status is **unconfirmed**, not pass. Synthetic fixture behavior is not biological validation. No SOTA superiority, biological recall gain, causal reviewer-time gain or production certification is claimed by this document.

## 11. Promotion vocabulary

Use explicit, non-interchangeable states:

- `STRUCTURALLY_VALID`: schema, canonical identities and local consistency pass.
- `SOURCE_BOUND`: the required source bytes/revisions and lineage are independently bound.
- `FRAME_VALID_WITHIN_DECLARED_EXTENT`: complete tiling/mask checks pass for the declared spatial scope.
- `STATISTICALLY_QUALIFIED`: sampling and uncertainty methods pass declared simulation/calibration gates.
- `BIOLOGICALLY_EVALUATED`: independent natural-data evaluation has been performed on the declared population.
- `WORKFLOW_UTILITY_DEMONSTRATED`: prospective causal workflow endpoint passes.
- `ROBUST_WITHIN_DECLARED_SCOPE`: the query remains within its preregistered decision tolerance across the supported plausible reconstruction set.
- `CONDITIONAL`, `SENSITIVE`, `NOT_IDENTIFIABLE`, `NOT_ESTIMABLE`, `OOD_ABSTAIN`, `INVALIDATED_BY_REVISION`: explicit limitation/failure states.

No state may be inferred from a stronger-looking but different state. A source digest proves identity; it does not prove biology.

## 2.9 From confidence transfer to domain-conditional reliability and referral

[SCI-CONNECTOMICS-023 / #9403](https://github.com/Luminous-Dynamics/symthaea/issues/9403) makes model calibration and domain novelty separate evaluation axes. ConnectomeBench2 and the 2026 VLM work provide strong operation-level classification comparisons, but a pooled score does not prove that confidence is portable to a held-out species, region, source volume, acquisition protocol, segmentation lineage or modality regime.

The frozen scorer now accepts a complete optional domain tuple and a separate novelty score. It reports per-domain reliability bins, ECE/MCE alongside Brier/log loss, false-accept rates at high error-probability thresholds, low-confidence missed-error mass, and novelty AUROC/AUPRC. OOD labels come from a predeclared domain split, not from the model's own score. Inputs with partial domain/novelty metadata or mixed axes fail closed.

The scorer now reports ten fixed-width error-probability bins, a 15-bin adaptive confidence-versus-correctness calibration view and a 12-bin equal-mass confidence reliability curve; tied confidence values are kept together. It adds deterministic cluster-bootstrap percentile intervals for fixed/adaptive ECE, MCE and Brier score overall and per domain, plus per-bin observed-accuracy and mean-confidence intervals for the 12-bin equal-mass curve, and OOD AUROC/AUPRC when both OOD classes remain represented in a replicate. It resamples complete cluster_id units, rejects cluster IDs that span domain IDs, excludes one-class OOD replicates, and withholds intervals when fewer than two clusters or fewer than 80% valid replicates are available. An OOD run without both held-out and in-domain examples is `OOD_NOT_ESTIMABLE`. Every novelty-score run is bound to a frozen detector ID and profile SHA-256 in the output report. Domain holdout labels are now derived from and checked against the exact split-manifest file; the scorer hashes the file itself, verifies dataset/partition/sample/cluster/domain membership and rejects train/calibration or cluster-boundary leakage. Any OOD operating threshold must be selected on development and applied only to confirmatory/lockbox evaluation. This remains a research estimator: it does not validate that cluster_id is an independent sampling unit or that two clusters provide adequate power. The tool is not an automatic referral service and makes no biological transfer claim. The next promotion step is repeated-seed finite-population interval-coverage simulation and an independent design review of the sampling unit, followed by a locked volume/species transfer evaluation. The synthetic fixture deliberately includes confident false positives and missed positives in held-out domains to ensure aggregate scores cannot hide failure.

See the protocol, input schema and synthetic input fixture.
