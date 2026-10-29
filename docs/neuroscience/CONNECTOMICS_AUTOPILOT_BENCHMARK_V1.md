# Connectomics Autopilot: benchmark and research plan V1

> **State-of-the-art update (2026-10-29):** ConnectomeBench2 V2 is the primary proofreading benchmark candidate. The original ConnectomeBench (2025) is retained only as a historical/reproducibility comparison. Do not score the original dataset or ConnectomeBench2's `main` revision as though either were the current ConnectomeBench2 paper release.

**Status:** proposed research contract; no benchmark results claimed  
**Date:** 2026-10-29  
**Owner:** Symthaea neuroscience / scientific-workflow integration  
**Scope:** AI-assisted proofreading, error triage, and evidence-aware circuit analysis on existing connectomics reconstructions

## Executive decision

The first Symthaea connectomics milestone should **not** be a new general-purpose 3D segmentation model and should not claim autonomous brain mapping. It should test whether Symthaea can make existing reconstructions more useful by finding consequential errors sooner, allocating expert review intelligently, retaining correction provenance, and refusing to overstate what the data support.

The optimization target is:

> Maximize independently verified connectome quality and scientific utility per unit of expert-review time and compute, subject to non-negotiable accuracy and provenance gates.

This is a workflow/decision layer over specialist segmentation, tracing, mesh, and synapse tools. HDC retrieval, recurrent state, active inference, and scientific planning are candidate components to ablate—not presumptive sources of improvement. They do not replace 3D vision or microscopy expertise.

This document proposes a falsifiable benchmark. It does **not** report an implementation, performance result, or state-of-the-art win.

## Why this is a timely wedge

Existing results establish both meaningful automation and a sharper new frontier:

- **ConnectomeBench2 (2026)** reports more than 716,000 expert-labeled proofreading decisions and more than 4.5 million associated images across mouse, human, zebrafish, and fly connectomes. Its shared-encoder Vision Transformer reaches human-level accuracy in the paper's evaluated split/merge correction tasks, with within-distribution calibration, distribution-distance diagnostics, and preliminary benefits from connectomics-specific pretraining and active learning. This makes raw in-distribution accuracy an insufficient goal for Symthaea; we must target label efficiency, cross-region/species transfer, calibration under shift, reviewer utility, and downstream graph impact.
- **Benchmarking Vision-Language Models on Synapse and Neuron Proofreading in Connectomics (arXiv:2609.36492v2, 23 Oct 2026)** reports that most tested VLMs are near chance at zero-shot proofreading, while LoRA adaptation on a few thousand labels brings open VLMs close to specialist performance. On the full ConnectomeBench2 test, Gemma 4 12B LoRA reached 95.0% split-error and 90.9% merge-error balanced accuracy versus the dataset's EM-only ViT-B at 94.2% and 91.1%, respectively. In the paper's mouse/fly-to-unseen-human/zebrafish transfer setup, Gemma 4 12B reached 77.3% and 82.7% on merge-error identification, versus 66.9% and 74.7% for ConvNeXt-Tiny. These are preprint results on particular protocols, not universal rankings; split-error transfer did not show the same VLM win. The valuable target is to improve label/compute efficiency, calibration, human review utility, or OOD behavior—not just in-distribution accuracy.
- **ConnectomeBench (NeurIPS 2025)** is still useful as a historical multimodal-LLM benchmark. Its reported models reached 52–82% balanced accuracy on segment identification and 75–85% on split-error tasks, while merge-error identification remained difficult. Do not compare these values directly to ConnectomeBench2 results without aligning task definitions, data versions, metrics, and splits.
- **RoboEM (Nature Methods, 2024)** uses self-steering 3D tracing to correct difficult neurites. The authors report approximately 400-fold lower *computational annotation cost* than manual correction for evaluated cortical tasks—not a 400-fold end-to-end speedup for building a whole-brain connectome.
- **NEURD (Nature, 2025)** decomposes neuron meshes into annotated graphs and applies morphological rules for proofreading. The paper reports nearly 150,000 high-confidence edits in MICrONS and more than a threefold speedup for a semi-supervised multi-soma splitting workflow. This is an important baseline, not something to ignore or relabel as Symthaea's contribution.
- **H01** demonstrates the data scale: approximately 1 mm³ of human temporal cortex, 1.4 petabytes of images, tens of thousands of cells, and roughly 150 million synapses. Its proofread ground truth is sparse relative to the entire volume, so raw scale must not be confused with complete independent validation.
- **2026 whole-brain zebrafish resources** create a longer-term bridge between structure and function. Google Neural Mapping describes the Fire&Wire larval zebrafish wiring diagram and the ZAPBench activity-prediction resource, which covers electrical activity from more than 70,000 neurons. Use these to motivate a later structure/function phase; do not assume every record is already co-registered or that anatomical edges alone establish causal function.

References and scope notes:
- ConnectomeBench: https://proceedings.neurips.cc/paper_files/paper/2025/hash/4a70f0c8443593ca59a88ebc8a937ed6-Abstract-Datasets_and_Benchmarks_Track.html
- RoboEM: https://www.nature.com/articles/s41592-024-02226-5
- NEURD: https://www.nature.com/articles/s41586-025-08660-5
- H01 release: https://h01-release.storage.googleapis.com/landing.html
- Google Neural Mapping datasets: https://sites.research.google/gr/neural-mapping/datasets/
- ConnectomeBench2 paper (arXiv:2606.21116): https://arxiv.org/abs/2606.21116
- ConnectomeBench2 official project page: https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/
- ConnectomeBench2 training and data-generation code (MIT code license; upstream data licenses still apply): https://github.com/timfarkas/ConnectomeBench2
- ConnectomeBench2 Hugging Face dataset card; pin `revision="v2"`, do not use `main` for paper replication: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2
- VLM benchmark for synapse detection/proofreading (arXiv:2609.36492v2, 23 Oct 2026): https://arxiv.org/abs/2609.36492
- ConnectomeBench source code (MIT): https://github.com/jffbrwn2/ConnectomeBench
- ConnectomeBench dataset card/files (MIT): https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench/tree/main

### Required ConnectomeBench2 V2 intake and leakage audit

The Hugging Face dataset card explicitly warns that `main` points to the initially submitted V1; the revision used by the paper is `v2`. The card's current `main` snapshot describes 401,170 samples, while the paper abstract reports more than 716,485 expert-labeled decisions. These are version-dependent counts, not numbers to reconcile by assumption. Pin the dataset revision and file hashes before reporting counts or metrics.

The dataset schema and instructions introduce several high-priority leakage guards:

- `task_routing` defines which tasks a row is valid for. Filter through the verified routing semantics, not just a populated boolean label column.
- The card defines `false_split_correction_label = same_neuron` and `false_merge_identification_label = not same_neuron`. Verify those definitions against the paper/code and retain separate task metrics.
- For single-segment tasks, fuse A/B EM mask channels and zero the B mask as specified. Do not use `em_best` where it can leak ground-truth information because the oblique view was selected using both labels.
- Geometry has both `geometry` and `geometry_single`; the card notes they can differ in more than mask channels. Model inputs must be chosen to match the task without exposing post-edit geometry or hidden target state.
- Train/validation/test assignment uses spatial cubes around the proofreading location. Audit overlap in source operation, source archive, neuron identity, nearby regions, duplicated views, and annotation lineage. Spatially distinct cubes alone do not prove independent samples.
- The dataset's overall license field is `other`; upstream source datasets can have different licenses. Check use and redistribution conditions per source, not merely the MIT license on the data-generation code.

Required intake manifest:

A machine-readable schema is committed at [`CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json`](CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json). It pins the dataset revision and code commit, requires a reviewed task/label mapping reference, allowlists model-input columns, and structurally excludes `em_best`, labels, IDs, routing fields, and metadata from model inputs. Schema validity is only a declaration check; the audit utility must still inspect the actual Parquet content and verify the declared reference.
- exact dataset repository and revision (default target: `v2`), upstream code commit, source URL, file SHA-256 and byte size;
- Parquet schema, per-shard row count, unique sample IDs, class distribution, null coverage, and duplicate checks;
- task routing and label semantics verified against upstream training/evaluation code;
- per-modality input declaration that blocks `em_best` and post-edit / dual-mask leakage in incompatible tasks;
- spatial/source/operation/neuron split-overlap report; every unresolved lineage key is marked unknown rather than assumed independent;
- license/provenance audit by source archive;
- label and annotation consistency where a valid join key and independent reference exist.

Fail closed if the revision is ambiguous, input view contains known label leakage, labels do not match task routing, or sample identity is not stable. Do not silently relabel rows to resolve a mismatch.

### Historical ConnectomeBench V1 note

The older ConnectomeBench Hugging Face card was observed to associate some split/merge task names with filenames carrying the opposite error-family words. Treat this as a historical metadata warning only: verify task meaning against upstream code and row semantics if running that legacy benchmark. Do not let it distract from the primary ConnectomeBench2 V2 intake gate.

## Paradigm shifts worth testing

### 1. From pixel quality to scientific error cost

Segmentation overlap is necessary but not sufficient. A small local merge can create false graph edges; a broken thin neurite can erase a long-range connection. Rank candidate corrections by expected impact on a declared downstream quantity, not by voxel uncertainty alone.

A candidate impact score may combine:
- probability the proposed error is real;
- expected correction impact on neuron identity, synapse assignment, and graph connectivity;
- relevance to a preregistered scientific query;
- expected reviewer time;
- reversibility and blast radius of the proposed edit.

No biological-plausibility score may overrule image evidence or suppress an unusual but real structure. A low-confidence but high-impact edit should normally be reviewed, not auto-applied.

### 2. From uncertainty heatmaps to a budgeted scientific queue

Choose which region to inspect next under an explicit review budget. Compare this decision policy to random review, uncertainty-only ranking, model-disagreement ranking, and morphology-rule ranking. Evaluate at matched reviewer minutes and matched compute; include the cost of generating suggestions.

An uncertainty estimate is useful only to the extent that it predicts error and improves decisions. Uncalibrated model confidence must not be labeled probability.

### 3. From isolated AI calls to correction-aware memory

Record the exact pre-edit state, proposed operation, cited evidence, reviewer decision, post-edit graph delta, and tool/model versions. Retrieve analogous historical cases to rank future suggestions, but keep every prior result's provenance and distinguish accepted corrections from rejected hypotheses.

Memory should improve retrieval and ordering, not mutate old evidence, leak evaluation labels into inference, or make a previous model decision self-validating.

### 4. From one-model authority to diversity-aware evidence

Use independent evidence channels where available: local EM texture, 3D continuity, mesh/skeleton morphology, synapse geometry, and genuinely independent model outputs. Track shared raw-data, preprocessing, model, and annotation ancestry so duplicated views of one source do not masquerade as independent corroboration.

Disagreement is a reason to inspect or abstain, not an automatic rule to average scores or discard the inconvenient model.

### 5. From local edits to counterfactual graph impact

For each candidate split/merge/branch repair, compute the *proposed* graph delta and show which downstream connectivity summaries would change. This makes reviewer effort legible and lets the queue prioritize edits that affect the scientific question being asked.

The delta is a consequence of the proposed edit, not evidence that the edit is correct. Evaluate impact scores using held-out expert truth; never expose hidden oracle labels to the runtime ranker.

### 6. From wiring-only maps to structure/function hypotheses

After reconstruction quality gates pass, connect anatomical data to activity, cell identity, perturbations, or behavior where valid aligned datasets exist. Ask whether a proposed circuit explanation predicts held-out observations better than competing explanations. A connectome by itself is not a complete functional model and cannot establish the causal role of every edge.

## System boundary

The V1 system proposes, ranks, records, and evaluates. It does not autonomously mutate authoritative public connectome annotations.

```text
Dataset + immutable revision
        |
        v
Existing segmentation / tracing / synapse tools
        |
        v
Candidate error and repair proposals
        |
        +--> local image / 3D / mesh evidence
        +--> graph delta and scientific-query impact
        +--> model lineage and applicable-domain checks
        |
        v
Symthaea triage policy
  - calibrated error score where supported
  - expected impact and review cost
  - analogous reviewed cases
  - abstain / ask for more evidence
        |
        v
Human review queue (V1: authoritative decision remains human)
        |
        v
Versioned decision record + graph delta + independent scorer
        |
        v
Frozen evaluation report; no hidden labels in the runtime path
```

Use specialist tools as task-appropriate:
- ConnectomeBench tasks for a public, comparable first evaluation;
- NEURD's mesh-based proofreading as a merge-error comparator where its data interface and license permit;
- RoboEM as a tracing/correction comparator where the modality and task match;
- a simple rules baseline and existing segmentation output as required controls.

Do not compare an end-to-end system against a post-processing component as if their scopes were identical. Publish the component and end-to-end boundaries.

## Evaluation design

### Datasets and split policy

**Phase A — audit ConnectomeBench2 V2 first.** Pin the Hugging Face `v2` revision, verify the exact task routing and label semantics from upstream code, block known modality leakage, and audit split/source lineage before reproducing the paper's ViT baseline. Then add the 2025 ConnectomeBench protocol as a historical multimodal-LLM comparison if its task definitions and data version are recoverable.

**Phase B — test structural generalization.** Add an eligible MICrONS subset with expert-verified cases and NEURD-compatible morphology when data access permits. Use H01 as a human-tissue transfer case only where independent reference annotations support the metric. Do not treat automated labels or sparse proofreads as universal ground truth.

**Phase C — connectome-scale and functional reasoning.** Consider FlyWire or the male fly CNS for larger graph analyses; consider the zebrafish wiring/activity resources for structure/function hypotheses after confirming alignment and access constraints. This phase is downstream of successful error triage, not a substitute for it.

Splits must be grouped by source volume, spatial block, neuron/cell identity, and annotation lineage as appropriate. Never split adjacent crops or fragments of the same neuron across train and test and then call the result generalization. Keep the hidden test oracle separate from retrieval memory, prompts, model fitting, profile selection, and the live queue.

**Rare OOD groups require a separate uncertainty gate.** The ConnectomeBench2 paper's natural test split reports only 60 human and 54 zebrafish merge-error operations, compared with thousands of split-error examples in each species. Cross-species merge-error point estimates can therefore be unstable and clustered samples further reduce effective sample size. Report exact positive/negative counts, cluster-aware confidence intervals, and both natural-prevalence and balanced challenge metrics. If a rare subgroup cannot support a preregistered precision or calibration claim, mark it not estimable rather than infer reliability from pooled accuracy. If a curated hidden OOD set is added, freeze its selection protocol in advance and report it separately from the untouched natural test.


### Comparator matrix

1. **Published ConnectomeBench2 ViT baseline**, reproducing the pinned `v2` dataset, modality, and split protocol where possible.
2. **Specialist task models and morphology rules**, including NEURD where source data and operation semantics are compatible.
3. **VLM comparison arms:** zero-shot, few-shot, and LoRA-adapted configurations from the September 2026 VLM study where model/code/artifact access and license permit a faithful reproduction.
4. Existing segmentation/tracing output with no additional triage.
5. Random review at the same expert-review budget.
6. Simple uncertainty-only and calibrated-uncertainty-only ranking.
7. Model-disagreement ranking, with shared ancestry tracked where known.
8. Active-learning baseline matching the same annotation budget; report the labels needed to reach a fixed target quality.
9. An error/impact ranker using only directly observable, available features.
10. Symthaea orchestration with correction memory and budgeted selection.
11. Ablations: no HDC retrieval; no correction memory; no active selection; no graph-impact term; no calibration/abstention policy.
12. Expert review as a reference ceiling only where expert labels and reviewer effort are actually measured. Never present the ceiling as an automated baseline.

Keep model version, prompts, preprocessing, compute, reviewer instructions, candidate pool, and stopping rules fixed across paired comparisons. Give every baseline the same input access and review budget.

### Primary metrics

**Human-effort efficiency**
- reviewer minutes per independently verified high-impact correction;
- number of reviews required to reach a preregistered error/quality target;
- impact-weighted true-error yield at fixed review budgets;
- reviewer disagreement and escalation time;
- all-in latency/compute used to produce and rank the queue.

**Error quality**
- precision, recall, and PR-AUC by merge, split, branch discontinuity, orphan fragment, and synapse-assignment error when ground truth supports each class;
- high-impact merge recall at fixed review budget;
- false-correction rate (a suggested change that damages a correct reconstruction);
- abstention/risk-coverage curve, Brier score and calibration by error class/domain where sample sizes permit.

**Structural and scientific consequences**
- neuron identity / completeness and synapse precision-recall where an adequate reference exists;
- topology/connectivity differences versus held-out expert-corrected ground truth;
- task-specific downstream error (only for a declared scientific query with a valid reference);
- false circuit conclusions induced by known synthetic merge/split perturbations.

**Robustness and provenance**
- results by volume, brain region, cell morphology, image quality, error type, and in-domain/OOD status;
- results by exact species × error-family group, with positive/negative denominators and cluster-aware confidence intervals; never pool a tiny human/zebrafish merge subset into a broad cross-species claim;
- degradation under section artifacts, missing slices, anisotropy, thin neurites, and rare morphologies;
- deterministic replay of queue ordering from frozen inputs and profile;
- lineage completeness and rate of duplicated/common-source evidence being incorrectly counted as independent;
- hidden-oracle leakage and post-outcome rank rewriting (both must be zero).

Use paired, cluster-aware uncertainty estimates (cluster on volume/region/neuron, as applicable), report denominators and missing labels, and publish per-dataset results. A pooled average cannot hide a failed dataset or error family.

### Proposed promotion targets — targets, not results

After reproducing the pinned ConnectomeBench2 V2 baseline, validating the split/leakage audit, and estimating variance on development data, preregister these initial targets. Do not choose thresholds after viewing the lockbox:

- At a matched review budget, increase **impact-weighted true-error yield by at least 15% relative** to the strongest eligible non-oracle baseline, with the cluster-aware 95% confidence interval for the paired gain excluding zero.
- Alternatively, for a fixed high-impact merge-recall target, reduce reviewer minutes needed by at least 20% relative to the strongest eligible baseline, without more than a 1 percentage-point absolute loss in correction precision. If sample size is insufficient to support either target, report the estimate and uncertainty; do not lower thresholds after seeing test labels.
- On held-out cases, the triage policy must not be worse than the strongest comparator on the preregistered primary quality metric beyond its declared non-inferiority margin. Exact thresholds and the metric must be frozen before opening the evaluation lockbox.
- V1 automatically applies **zero** edits to authoritative data. Every suggestion carries evidence, predicted graph delta, source lineage, and an explicit uncertainty/abstention state.
- No hidden-label leakage, source/version mismatch, silent post hoc model/profile changes, or false claim that queued/unfinished runs passed. Any such failure invalidates the run regardless of average score.
- A cross-dataset or cross-species claim requires replication on at least two genuinely distinct dataset families with leakage-resistant splits. A single dataset win stays a single-dataset claim.
- **Label-efficiency target:** reduce expert labels/review decisions needed to reach a prespecified quality target by at least 20% relative to the strongest eligible active-learning baseline, under equal reviewer/compute budgets and with no critical error-class regression.
- **Shift-safety target:** on held-out region/species/source groups, report risk-coverage and calibration-vs-coverage; abstention must lower accepted-case error as coverage is reduced. If a subgroup lacks enough labels for a calibration claim, mark it not estimable and default to human review.
- Report two Pareto frontiers, not one leaderboard: (accuracy / accepted-case risk) vs expert-label budget, and (accepted-case risk) vs coverage, with all-in latency and compute reported alongside.

These targets are intentionally about efficient, safe proofreading, not unsupported claims of beating every segmenter or achieving whole-brain autonomy.

## Symthaea-specific ablation plan

The first implementation should be deliberately small. Begin with deterministic ranking and traceable artifacts; then test whether cognitive components add value.

- **Base ranker:** task-specific heuristics plus available direct evidence features.
- **HDC retrieval:** retrieve analogous reviewed error cases; compare against no retrieval and lexical/feature nearest-neighbor controls.
- **Correction memory:** use only training/development corrections; freeze memory snapshots per run and verify no test-label access.
- **Active selection:** choose next candidate by expected impact per review minute, versus uncertainty-only, impact-only, and random policies.
- **Temporal/recurrent state:** only include if the workflow has meaningful repeated observations or reviewer-history signals; otherwise omit it rather than adding LTC/CfC decoration.
- **FEP/value of information:** test whether the policy chooses cases that actually reduce uncertainty in the declared scientific quantity under a fixed budget; compare predicted versus realized value.
- **Independent checks:** score graph invariants and source lineage separately from the proposal producer. Agreement between two components sharing the same model or evidence root is not independent verification.

Promote a component only when its frozen ablation improves held-out reviewer efficiency or scientific error metrics without worsening calibration, subgroup performance, provenance, or abstention.

## Minimal implementation sequence

1. **Dataset/metric audit:** document licenses, source revisions, ground-truth coverage, exact ConnectomeBench tasks, and reproducible data acquisition. No data downloads or large compute are prerequisites for the design review.
2. **Benchmark contract:** define immutable run manifest, split identity, candidate edit schema, oracle boundary, and scorer outputs.
3. **Offline scorer:** evaluate predetermined candidate rankings against known expert annotations. First demonstrate correct scoring and negative controls before adding an adaptive ranker.
4. **Baselines:** random, uncertainty-only, disagreement-only, published/compatible rule-based baseline, and a small learned or HDC-assisted ranker if the data format permits.
5. **Active queue:** add review-minute budgeting, abstention, graph-impact explanation, and replayable deterministic ordering.
6. **Ablations and hidden lockbox:** freeze profiles and use independently held-out annotations; do not wait on an asynchronous runner to design or audit the protocol.
7. **Prospective pilot:** only after retrospective validation, measure actual expert time on a held-out queue. Record review time and outcome prospectively; never reconstruct them from model estimates.
8. **Functional circuit study:** only after data-quality exit gates, add aligned activity/perturbation evidence and test held-out predictions.

## Candidate record schema

The exact Rust types should be designed after auditing existing provenance and scientific-evidence conventions. The minimum conceptual record is:

- dataset ID, immutable volume/revision ID, image/annotation source references;
- model/tool/profile versions and declared applicability envelope;
- candidate region and candidate operation (split, merge, branch continuation, synapse reassignment, or abstain);
- evidence pointers and evidence-source ancestry;
- calibrated error probability only if calibration is supported for this task/domain;
- predicted graph delta and impact rationale tied to the declared query;
- estimated review cost and queue policy version;
- state: proposed, accepted by reviewer, rejected, deferred, or not estimable;
- reviewer decision/time and post-edit references (when actually observed);
- canonical digest for the input/profile/candidate and the resulting decision record.

Proposals, predictions, expert decisions, and actual annotation mutations must remain distinct types/events. A digest proves identity/integrity of recorded data, not biological truth.

## Explicit non-goals and kill conditions

V1 is not:
- a new universal 3D segmentation foundation model;
- proof that HDC, consciousness metrics, or a connectome graph alone can infer neuron identity or function;
- autonomous publication or mutation of authoritative annotations;
- a claim to map an entire living human brain or reconstruct subjective experience.

Stop or revise the approach if:
- gains disappear against random or uncertainty-only review at matched budgets;
- a ranking win depends on spatial/neuron leakage or test labels in memory;
- impact scores are correlated with model confidence but not with verified downstream error;
- higher aggregate yield hides unacceptable precision/recall loss in a critical error class;
- a model silently turns missing evidence into confidence;
- the workflow cannot reproduce its own queue from the exact frozen inputs and profile.

## Definition of success

The first credible success is narrow: on a public, leakage-resistant connectomics task, Symthaea's triage/orchestration reduces expert review effort or increases high-impact error yield against strong matched-budget baselines, while maintaining reconstruction quality, calibrated abstention where support exists, exact provenance, and zero unreviewed authoritative edits.

That would establish a real role for Symthaea in connectomics. Whole-brain mapping, cross-modal structure/function analysis, and greater autonomy remain later research questions whose feasibility must be earned through measured results.
