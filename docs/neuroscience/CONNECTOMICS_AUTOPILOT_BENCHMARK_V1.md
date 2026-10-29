# Connectomics Autopilot: benchmark and research plan V1

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

Existing results establish both meaningful automation and an unresolved proofing bottleneck:

- **ConnectomeBench (NeurIPS 2025)** evaluates segment-type identification, split-error correction, and merge-error detection against expert annotations from mouse visual cortex and the Drosophila brain. Its reported multimodal models reach 52–82% balanced accuracy on segment identification and 75–85% on split-error tasks, while merge-error identification remains a weakness. These numbers are task- and dataset-specific; they are not a global ranking of connectomics systems.
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

**Phase A — reproduce a public task.** Start with the published ConnectomeBench split/merge/segment-type tasks and verify the exact dataset versions, annotations, license, and data access before coding the adapter. Record task definitions and source revisions.

**Phase B — test structural generalization.** Add an eligible MICrONS subset with expert-verified cases and NEURD-compatible morphology when data access permits. Use H01 as a human-tissue transfer case only where independent reference annotations support the metric. Do not treat automated labels or sparse proofreads as universal ground truth.

**Phase C — connectome-scale and functional reasoning.** Consider FlyWire or the male fly CNS for larger graph analyses; consider the zebrafish wiring/activity resources for structure/function hypotheses after confirming alignment and access constraints. This phase is downstream of successful error triage, not a substitute for it.

Splits must be grouped by source volume, spatial block, neuron/cell identity, and annotation lineage as appropriate. Never split adjacent crops or fragments of the same neuron across train and test and then call the result generalization. Keep the hidden test oracle separate from retrieval memory, prompts, model fitting, profile selection, and the live queue.

### Comparator matrix

1. Existing segmentation/tracing output with no additional triage.
2. Random review at the same reviewer-minute budget.
3. Simple uncertainty-only ranking.
4. Model-disagreement ranking, with shared ancestry tracked where known.
5. Existing morphology rules, including NEURD on compatible data.
6. Published ConnectomeBench model/task baselines, using their stated protocols where reproducible.
7. An error/impact ranker using only directly observable, available features.
8. Symthaea orchestration with correction memory and budgeted selection.
9. Ablations: no HDC retrieval; no correction memory; no active selection; no graph-impact term; no calibration/abstention policy.
10. Expert review as a reference ceiling only where expert labels and reviewer effort are actually measured. Never present the ceiling as an automated baseline.

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
- degradation under section artifacts, missing slices, anisotropy, thin neurites, and rare morphologies;
- deterministic replay of queue ordering from frozen inputs and profile;
- lineage completeness and rate of duplicated/common-source evidence being incorrectly counted as independent;
- hidden-oracle leakage and post-outcome rank rewriting (both must be zero).

Use paired, cluster-aware uncertainty estimates (cluster on volume/region/neuron, as applicable), report denominators and missing labels, and publish per-dataset results. A pooled average cannot hide a failed dataset or error family.

### Proposed promotion targets — targets, not results

After reproducing the selected benchmark and measuring variance, preregister these initial targets:

- At a matched review budget, increase **impact-weighted true-error yield by at least 15% relative** to the strongest eligible non-oracle baseline, with the cluster-aware 95% confidence interval for the paired gain excluding zero.
- Alternatively, for a fixed high-impact merge-recall target, reduce reviewer minutes needed by at least 20% relative to the strongest eligible baseline, without more than a 1 percentage-point absolute loss in correction precision. If sample size is insufficient to support either target, report the estimate and uncertainty; do not lower thresholds after seeing test labels.
- On held-out cases, the triage policy must not be worse than the strongest comparator on the preregistered primary quality metric beyond its declared non-inferiority margin. Exact thresholds and the metric must be frozen before opening the evaluation lockbox.
- V1 automatically applies **zero** edits to authoritative data. Every suggestion carries evidence, predicted graph delta, source lineage, and an explicit uncertainty/abstention state.
- No hidden-label leakage, source/version mismatch, silent post hoc model/profile changes, or false claim that queued/unfinished runs passed. Any such failure invalidates the run regardless of average score.
- A cross-dataset or cross-species claim requires replication on at least two genuinely distinct dataset families with leakage-resistant splits. A single dataset win stays a single-dataset claim.

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
