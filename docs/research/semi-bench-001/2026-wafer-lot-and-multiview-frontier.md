# SEMI-BENCH-001 — 2026 lot generalization, open-world and multi-view wafer-map frontier

**Review date:** 2026-11-21  
**New workstreams:** [SEMI-WM-LOT-001 #13122](https://github.com/Luminous-Dynamics/symthaea/issues/13122), [SEMI-MULTIVIEW-001 #13123](https://github.com/Luminous-Dynamics/symthaea/issues/13123), [SEMI-WBM-COMP-001 #13113](https://github.com/Luminous-Dynamics/symthaea/issues/13113), [SEMI-OPENSET-001 #13121](https://github.com/Luminous-Dynamics/symthaea/issues/13121)

This update adds a sharper benchmark hierarchy for wafer maps: sample interpolation, unseen-lot transfer, open-set defect rejection, compositional mixed-defect recognition and multi-view test-item fusion. These are separate capabilities and must not collapse into a single wafer-classification score.

## 1. Graph contrastive learning: strong reported results, but random-split is not lot transfer

Awais, Postolache and Oliveira, *Graph-based contrastive learning for self-supervised semiconductor wafer defect detection*, Journal of Intelligent Manufacturing, published 4 July 2026:
- https://doi.org/10.1007/s10845-026-02906-3
- Full text: https://link.springer.com/article/10.1007/s10845-026-02906-3
- Released implementation: https://github.com/irjawais/wafer-gnn-contrastive

The paper reports WM-811K as 811,457 maps across 46,293 production lots, 25×25 maps, 3 die states and 9 pattern classes. It uses a 30% stratified sample (243,437 wafers), then 60/20/20 stratified train/validation/test partitions (146,062 / 48,687 / 48,688). The GNN uses 8-connectivity, a GIN with jumping-knowledge connections and a contrastive similarity that combines spatial pattern, defect density, structural descriptors and production context.

Paper-reported results include:
- unsupervised clustering: ARI 0.89, NMI 0.87;
- supervised fine-tuning: 98.6% accuracy, ARI 0.98, NMI 0.96;
- 95.7% accuracy with 25% labelled data;
- macro-F1 0.975, with Near-full F1 0.91 on 149 labelled examples and Donut F1 0.93 on 555;
- LOCO OOD: mean detection 89.3% at 95% ID retention, mean AUROC 0.927;
- reported inference latency 8.3 ms and memory 387 MB on the paper's hardware.

These are paper-reported figures, not Luminous reproductions. The primary paper's evaluation split is sample-level and stratified by defect type. It therefore does not itself establish transfer to wholly unseen lots, campaigns, tools or factories, even though lot and time context are used in pair construction.

### A useful artifact audit

The public implementation's data loader uses stratified train_test_split to produce 60/20/20 splits. Its OOD script appears to compute the MSP rejection threshold from test_in, while the paper's OOD method text says the threshold is calibrated on the ID validation set to retain 95% of ID samples.

This could be a repository/paper implementation mismatch; it does not prove that the published manuscript's results use test data for calibration. Our replication should preserve the released-script behavior as an audit track, but the faithful protocol must calibrate only on validation and leave test untouched. See [SEMI-WM-LOT-001 #13122](https://github.com/Luminous-Dynamics/symthaea/issues/13122).

### Paradigm shift: test production-context value under actual group independence

The key experiment is not simply "add lotName to the model." It is whether production context improves transfer after entire lots and their near-time neighborhoods are excluded from training and validation.

Required comparisons:
1. exact published stratified reproduction;
2. lot-disjoint held-out generalization;
3. source-time-forward or campaign-forward evaluation with a declared embargo;
4. same spatial GNN with and without the production-proximity term;
5. pair generation restricted to training-only lots and windows;
6. explicit context-blind vs context-aware inference;
7. HDC / multiscale relational HDC and CNN challengers under identical splits.

Report per-lot and worst-decile lot outcomes, rare-class support, confidence intervals and degradation versus the random-split track. Do not call random-split results contaminated unless overlap is directly measured; say instead that they do not establish unseen-lot transfer.

### OOD calibration integrity

The paper's OOD result is 89.3% detection at 95% ID retention (AUROC 0.927). The validation threshold must be frozen before scoring test rows. A robust challenger should compare maximum-softmax probability against energy, distance/prototype, conformal or HDC-based novelty scores, and report:
- FPR at 95% known/ID TPR;
- unknown recall and AUROC/AUPR;
- OSCR;
- known-class accuracy at fixed rejection;
- calibration and confidence intervals;
- per-held-out-pattern performance, especially visually similar class pairs.

Do not report the 89.3% as reproduced until the threshold protocol, splits and exact implementation are independently matched.

## 2. Industrial CP and DFC maps: preserve test-item semantics

Wu et al., *Adaptive open-world learning for analyzing wafer map variations*, Expert Systems with Applications, Volume 318, article 131994, published 1 July 2026:
- https://doi.org/10.1016/j.eswa.2026.131994

The paper describes item-wise layered processing and semantic analysis for two wafer-map types:
- Chip Probe (CP) maps;
- Delta Fuse Count (DFC) maps.

Its open-world architecture uses M3Net for known-class recognition (multi-task learning, multi-scale feature fusion and multi-label loss) and Parallel SVM (PSVM) to flag potential novel classes for expert review. The abstract reports evaluation on an industrial dataset; public access, licensing, sample lineage, complete label schema and exact metrics must be verified before direct reproduction.

### Paradigm shift: multi-view wafer evidence

A wafer is not necessarily one bitmap. Preserve a structured observation bundle:
- exact wafer and lot/campaign identity;
- test program and test-item identity;
- CP-map state/mask/coordinate semantics;
- DFC-map count/delta semantics;
- alignment and cross-view consistency evidence;
- missing/stale/contradictory view status;
- instrument/calibration context and review disposition.

Benchmark CP-only, DFC-only and multi-view fusion separately. Test what happens when one view is absent, delayed, corrupted, stale or contradictory. An absent map must not be represented as an all-zero map; that would confuse unavailable evidence with observed absence of defects.

See [SEMI-MULTIVIEW-001 #13123](https://github.com/Luminous-Dynamics/symthaea/issues/13123).

## 3. MixedWM38 and open-world research: separate interpolation from composition

MixedWM38 comprises 1 normal, 8 single-defect and 29 mixed-defect categories and includes GAN-generated samples to compensate for pattern imbalance. Random splits can therefore measure pipeline correctness while overstating real manufacturing generalization.

Yang, Jang and Kim, *Classifying mixed-defect from single-defect training in imbalanced wafer maps via diffusion and attention*, Expert Systems with Applications 299, article 130127, published 1 March 2026:
- https://doi.org/10.1016/j.eswa.2025.130127

The paper frames mixed-pattern recognition as generalized zero-shot learning and reports harmonic-mean accuracy of 74.3% in its most imbalanced setting, with a 3.92 percentage-point improvement over its strongest compared diffusion classifier. This is a paper-reported reference, not a reproduced result.

The correct hierarchy is:
1. sample-level interpolation reproduction;
2. leave-combination-out compositional recognition;
3. leave-component-out novelty rejection;
4. synthetic-to-real with sample-level source provenance;
5. cross-dataset transfer with a reviewed partial label mapping;
6. few-shot introduction and continual-learning retention.

See [SEMI-WBM-COMP-001 #13113](https://github.com/Luminous-Dynamics/symthaea/issues/13113) and [SEMI-OPENSET-001 #13121](https://github.com/Luminous-Dynamics/symthaea/issues/13121).

## 4. PPAPlace-5000: an actionable EDA benchmark artifact, with license gate

The official [PPAPlace-5000 Hugging Face dataset card](https://huggingface.co/datasets/ValleyC/PPAPlace-5000) describes:
- 5,000 complete mixed-size DREAMPlace placements;
- 10 open ChiPBench circuits, 500 placement samples each;
- matching post-global-routing WNS, TNS, power and area labels;
- 8.47 GiB compressed archives;
- a dataset card license field of other.

The [PPAPlace implementation](https://github.com/ValleyC/PPAPlace) publishes feature preparation, training, leave-one-circuit-out evaluation, refinement and audit scripts. Its docs pin ChiPBench commit ea28bebab06977b348a74d3fd07275bb828ca68e and OpenROAD v2.0-17198-g8396d0866.

The repository states that it does not include trained checkpoints or experiment outputs. Its documentation distinguishes within-circuit testing from unseen-circuit generalization and says surrogate PPA is not a physical timing measurement; baseline and refined placements require identical OpenROAD evaluation.

The dataset's other license metadata is not a blanket redistribution grant. Verify the dataset card's exact terms and any tool/design restrictions before copying, caching or redistributing archives. A local benchmark can use permitted metadata and licensed artifacts without committing the 8.47 GiB release.

See [SEMI-EDA-001 #13069](https://github.com/Luminous-Dynamics/symthaea/issues/13069).

## 5. Benchmark targets

| Track | Strongest current reference | Next claim we should pursue |
|---|---|---|
| GNN wafer-map classification | 98.6% accuracy / macro-F1 0.975 on sample-level WM-811K | Maintain performance on unseen lots and future windows |
| GNN open-set | 89.3% OOD detection at 95% ID retention | Improve OOD recall and keep FPR/threshold calibration honest |
| Mixed-defect composition | 74.3% harmonic mean under a generalized zero-shot protocol | Better unseen combinations without synthetic-to-real confusion |
| CP/DFC multi-view | M3Net + PSVM paper reference | Robust fusion and safe abstention when views are missing or contradictory |
| EDA | PPAPlace-5000 post-GRT labels | Better final routed PPA on held-out circuits, not merely surrogate score |
| SECOM | DCBBC 80.16% balanced accuracy and 2026 RF/LightGBM references | Time-forward + missingness-aware generalization |

Every result must be split- and protocol-specific. A higher random-split accuracy is not automatically a better engineering system.

## 6. Luminous integration

- **Symthaea:** spatial/relational representation, HDC prototypes, GNN/CNN/transformer challengers, open-set calibration, multi-view learning and experiment selection.
- **Symtropy:** deterministic process/inspection replay, scenario and uncertainty analysis; no conversion from an image score to a physical root cause.
- **Mycelix:** identity, provenance and lineage for wafer, lot, test item, map view, raw measurement, model run and review outcome.
- **Sol-Atlas:** canonical literature, standards, source snapshots, dataset rights and method/version traceability.

## Claim boundaries

- A lot-disjoint benchmark establishes transfer only across the declared lot identities present in that dataset.
- A sample-level split does not establish lot or campaign transfer.
- CP/DFC paper results do not grant raw industrial data access.
- A class-heldout WM-811K task is an open-set proxy, not proof of discovering a new physical defect mechanism.
- A surrogate EDA score is not final detailed-route or fabricated-silicon performance.
- None of these public benchmark results establishes yield improvement, fab process qualification or physical equipment capability.


## 3A. Corrected 2026 MixedWM38 comparison: closed-set transfer is not compositional SOTA

The full Awais et al. article adds a critical result not visible in the abstract. On its held-out MixedWM38 test set (7,603 wafers), the WM-811K-pretrained graph-contrastive pipeline reports 90.86% bit-wise accuracy and bit-wise macro-F1 0.815. The authors state that 36x36 downsampling loses spatial detail, the GIN was not tuned specifically for MixedWM38, and the supervised schedule was shorter than those in stronger benchmark methods. The paper cites WSCN at 98.20% and PIENet at 98.90% on its comparison context. These values use paper-specific protocols and must not be directly ranked until the label encoding, split and exact metric match.

This changes the next target: keep both per-defect 8-bit prediction and whole-composition recognition. Required outputs:
- bit-wise macro-F1 and per-defect recall;
- exact-set match / 38-way category accuracy;
- harmonic mean of seen- and unseen-combination accuracy;
- leave-combination-out and leave-component-out tasks;
- real-only, synthetic-only and source-mixed evaluations only when source identity is actually known;
- native resolution versus 36x36 and other downsampling ablations.

A separate July 5, 2026 FAD-Net paper reports 95.35–98.63% accuracy, F1 0.9255–0.9858 and AUC 0.9971–0.9997 across WM-811K and MixedWM38. Its public abstract does not expose the exact per-dataset split and source-identity protocol, and its data availability statement directs readers to the manuscript/author for further details. Treat it as an important reproduction lead, not an apples-to-apples leaderboard entry.

Sources:
- https://link.springer.com/article/10.1007/s10845-026-02906-3
- https://doi.org/10.1002/qre.70321


## 7. Frozen multimodal models and rule-ordered zero-shot reasoning

Cao & Tao, *Zero-Shot Multi-Class Wafer Map Classification with Rule-Ordered Prompts for Large Multimodal Models*, arXiv:2610.35080 (26 October 2026):
https://arxiv.org/abs/2610.35080

The preprint uses frozen GPT-4o (gpt-4o-2024-08-06), connected-component visual marks and class rules that specify positive morphology, spatial anchor and exclusion boundaries. It performs no task-specific training, fine-tuning or few-shot prompting. Its six-class subset results are:
- Mixed-WM38: 6,866 single-pattern maps; multi-region prompting reports 92.9% macro accuracy and 93.8% overall.
- WM-811K: 9,000 maps; multi-region prompting reports 79.1% macro and 81.3% overall.
- Changing only rule order moves Mixed-WM38 overall accuracy across 90.1–91.9% in the eight tested alternatives, versus 93.8% for the selected order.
- API inference is reported at roughly 0.5–1 second per image in the authors' environment.

This is a useful alternative to trained CNN/GNN/HDC approaches for the no-label stage of deployment, but it is not a full MixedWM38 result: only six single-pattern classes were retained. The WM-811K six-class evaluation is sample-selected, not lot-disjoint. A single call per image also does not establish repeatability or prompt-order invariance.

### New paradigm: semantic policy + deterministic geometric evidence

Treat class descriptions, positive features, spatial anchors and exclusion boundaries as a versioned, inspectable decision specification. Compute objective geometry—connected components, radial occupancy, angular continuity, elongation, cluster count, mask coverage—outside the LMM, then pass those grounded observations and selected regions to an interpreter. This lets us separate "what is visible" from "what the model thinks it means."

The crucial research issue is prompt-order sensitivity. Rather than trusting one favorable order, test all feasible permutations or a preregistered sample, pairwise tournament, order-ensemble, paraphrase perturbations and an order-invariant symbolic/HDC alternative. Report per-class flips and worst-order degradation; never tune order on the final test labels.

New follow-up: [SEMI-ZS-LLM-001 #13146](https://github.com/Luminous-Dynamics/symthaea/issues/13146), covering exact preprint reproduction, order robustness, evidence-grounded explanations, full mixed-composition generalization, open-set abstention, group/time transfer and LMM/HDC/CNN comparisons.

## 8. Semiconductor manufacturing should be benchmarked as a closed-loop decision system

Li et al., *Toward autonomous decision support in semiconductor manufacturing: a review of planning, scheduling, dispatching, and execution control*, published 6 November 2026:
https://doi.org/10.1007/s43684-026-00146-z

The review's central model connects fab state sensing, planning, scheduling, dispatching, execution, metrology feedback and explicit human oversight. It warns against treating a locally improved dispatching rule as a system improvement when it can create downstream congestion, yield risk or delivery instability. It also argues that learning and LLM orchestration must sit behind validated offline/simulation evaluation, bounded authority, safe fallbacks and auditable records.

This supplies a broader integration target for the programme:
- perception estimates evidence and uncertainty;
- process/material models constrain plausible state transitions;
- inspection policy decides what to measure next;
- scheduling/dispatching optimizes only within qualified process/tool, queue-time, capacity, material and quality constraints;
- approved execution produces evidence, not assumed state changes;
- metrology and quality feedback update future beliefs and decisions.

The benchmark should be a multi-fidelity ladder: synthetic DES/replay -> frozen historical trace -> offline counterfactual evaluation -> shadow-mode recommendations -> human-approved limited pilot -> controlled physical outcome study. Report system-level cycle time, WIP, OEE, excursion detection, inspection cost, throughput and delivery reliability jointly; every improvement must preserve quality/safety and respect hard constraints. No autonomous process authority is implied.

Source: https://link.springer.com/article/10.1007/s43684-026-00146-z


## 9. Cross-source pseudo-label alignment and boundary safety

The 16 October 2026 IEEE Access paper on pseudo-label-based integration of WM-811K and Mixed38 reports relative gains of 207.75% in silhouette coefficient and 16.19% in Ratkowsky-Lance index. Source: https://doi.org/10.1109/ACCESS.2026.3744016 . These are cluster-structure measures and do not establish pseudo-label precision, correctness or future-lot transfer. The follow-up [SEMI-WM-ALIGN-001 #13148](https://github.com/Luminous-Dynamics/symthaea/issues/13148) benchmarks partial label alignment, boundary-aware pseudo-label admission, unknown preservation, and source/lot-held-out performance.


## 10. Deployment Pareto: compact hybrid and ViT-Tiny challengers

Two additional 2026 references change the competitive target. Mohammad et al. (25 Aug 2026) report ViT-Tiny at 98.4% accuracy on WM-38K with 5.7M parameters, 1.77× faster inference and 4.47× lower memory than comparable transformers: https://doi.org/10.1038/s41598-026-63752-0 . Hu (10 Sep 2026) reports a hybrid global/local MobileNetV2 + handcrafted geometry model at 98.89% accuracy and 94.41% macro-F1 on WM-811K with 64,967 trainable parameters; the offline StyleGAN2-ADA generator and handcrafted feature extraction are additional lifecycle costs: https://doi.org/10.1038/s41598-026-68884-x . Neither result is an apples-to-apples comparison with the 2026 GNN or FAD-Net until split, label protocol, resolution, hardware and resource measurement are matched.

New follow-up [SEMI-PARETO-001 #13149](https://github.com/Luminous-Dynamics/symthaea/issues/13149) benchmarks accuracy, per-class recall, open-set calibration, p95 latency, peak memory, artifact size, training/update cost and measured energy where instrumented. The Luminous differentiator should be a measured resource/reliability Pareto win, not an unsupported claim of beating a 98–99% headline.
