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
- **NEURD (Nature, 2025)** decomposes neuron meshes into annotated graphs and applies morphological rules for proofreading. The paper reports nearly 150,000 high-confidence edits in MICrONS and more than a threefold speedup for a semi-supervised multi-soma splitting workflow. This is an important baseline, not something to ignore or relabel as Symthaea's contribution. It also highlights axon–dendrite proximity as a control for interpreting higher-order connectivity motifs; raw triangle counts alone are not evidence of biologically significant motif enrichment.
- **SynAnno (IEEE TVCG, 2026)** is a human-in-the-loop interface comparator rather than a directly comparable split/merge classifier. It uses neuron-centric guided traversal, a 3D mini-map, progress tracking, and separate detection/categorization/correction stages; its evaluation involved seven neuroscience experts and reports faster proofreading with lower cognitive load and annotation errors. Symthaea should compare reviewer time, error rates, context switching, interruptions, and provenance—not claim model superiority from this UX study. See https://pubmed.ncbi.nlm.nih.gov/41396767/.
- **H01** demonstrates the data scale: approximately 1 mm³ of human temporal cortex, 1.4 petabytes of images, tens of thousands of cells, and roughly 150 million synapses. Its proofread ground truth is sparse relative to the entire volume, so raw scale must not be confused with complete independent validation.
- **2026 whole-brain zebrafish resources** create a longer-term bridge between structure and function. Google Neural Mapping describes the Fire&Wire larval zebrafish wiring diagram and the ZAPBench activity-prediction resource, which covers electrical activity from more than 70,000 neurons. Use these to motivate a later structure/function phase; do not assume every record is already co-registered or that anatomical edges alone establish causal function.

References and scope notes:
- ConnectomeBench: https://proceedings.neurips.cc/paper_files/paper/2025/hash/4a70f0c8443593ca59a88ebc8a937ed6-Abstract-Datasets_and_Benchmarks_Track.html
- RoboEM: https://www.nature.com/articles/s41592-024-02226-5
- NEURD: https://www.nature.com/articles/s41586-025-08660-5
- H01 release: https://h01-release.storage.googleapis.com/landing.html
- Google Neural Mapping datasets: https://sites.research.google/gr/neural-mapping/datasets/
- ConnectomeBench2 paper (arXiv:2606.21116): https://arxiv.org/abs/2606.21116
- Martinez et al., *Connectome quality converges predictably to reveal optimal stopping points during proofreading* (bioRxiv, 2026): https://doi.org/10.64898/2026.06.30.735414
- ConnectomeBench2 official project page: https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/
- ConnectomeBench2 training and data-generation code (MIT code license; upstream data licenses still apply): https://github.com/timfarkas/ConnectomeBench2
- ConnectomeBench2 Hugging Face dataset card; pin `revision="v2"`, do not use `main` for paper replication: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2
- VLM benchmark for synapse detection/proofreading (arXiv:2609.36492v2, 23 Oct 2026): https://arxiv.org/abs/2609.36492
- ConnectomeBench source code (MIT): https://github.com/jffbrwn2/ConnectomeBench
- ConnectomeBench dataset card/files (MIT): https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench/tree/main

### Required ConnectomeBench2 V2 intake and leakage audit

The Hugging Face dataset card explicitly warns that `main` points to the initially submitted V1; the revision used by the paper is `v2`. The card's current `main` snapshot describes 401,170 samples, while the paper abstract reports more than 716,485 expert-labeled decisions. These are version-dependent counts, not numbers to reconcile by assumption. Pin the dataset revision and file hashes before reporting counts or metrics.

The upstream code was independently inspected at commit [`5679b39aa2641d8d0afeed88b441e696f87a66df`](https://github.com/timfarkas/ConnectomeBench2/commit/5679b39aa2641d8d0afeed88b441e696f87a66df). In `src/neurips_training/data.py::route_row`, `false_merge_identification` is routed to `false_merge_identification_label = !same_neuron`, while `false_split_correction` is routed to `false_split_correction_label = same_neuron`; that loader selects `geometry_single` for the former and dual `geometry` for the latter. `scripts/data_generation/neurips/postprocess_giga_parquet.py::derive_task_routing` defines the route membership from sample type and mask presence. The Rust intake utility and JSON Schema now pin this exact upstream code commit so a future source change cannot silently inherit the current task contract.

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

### 2b. From “proofread everything” to query-specific stopping

A July 2026 preprint, *Connectome quality converges predictably to reveal optimal stopping points during proofreading* (Martinez et al., https://www.biorxiv.org/content/10.64898/2026.06.30.735414v1), reports that selected global graph invariants can converge as proofreading progresses and that their trajectories can help forecast remaining effort on zebrafish spinal-cord and Drosophila hemibrain datasets. This complements error ranking: a system should estimate not only which edit to review next, but also when further proofreading is unlikely to change a specified downstream result enough to justify its cost.

Symthaea should test **query-specific stopping**, rather than treat a universal “connectome complete” score as biological truth:

1. Track versioned graph snapshots and edit counts using the initial contract in [`CONNECTOME_STOPPING_SNAPSHOT.schema.json`](CONNECTOME_STOPPING_SNAPSHOT.schema.json); compute a preregistered set of graph invariants and the actual scientific quantities the project intends to analyze. The schema defines per-snapshot inputs only; a sequential validator must still check monotone edit counts, parent digest continuity, matching revisions, and consistency of edit-class totals.
2. Fit candidate convergence curves using early snapshots only, then forecast later held-out snapshots without refitting on the future trajectory.
3. Compare predicted remaining edits and uncertainty with the observed effort required to reach prespecified stability bands.
4. Perturb the graph with held-out known split/merge errors to measure whether invariant stability can coexist with serious local or query-specific mistakes.
5. Compare against fixed-edit-count stopping, fixed-budget stopping, and expert-chosen stopping. Report false-stop rate, forecast interval coverage, edit-effort error, query-result stability, and the cost of additional proofreading.
6. Stop only for the declared analysis and declared stability tolerance; keep targeted review open for high-impact or rare pathways, and reopen the stopping decision when the scientific question or accepted graph state changes.

**Critical limitation:** global invariants can stabilize while local errors remain, and a stable graph statistic is not proof of ground-truth reconstruction, synaptic completeness, or causal biological correctness. Treat convergence as a resource-allocation signal; combine it with direct error estimates, graph-delta checks, scientific-query sensitivity, and expert-reviewed evidence.

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

### 7. From a single-model leaderboard to a risk-budgeted cascade

The October 2026 VLM study argues that specialist models can be cheaper per image for one fixed task and known dataset, while adapted VLMs can transfer better to unfamiliar species in some error classes. Symthaea should test a **cascade**, not assume one model should handle every case:

1. A compact specialist handles in-domain, well-calibrated, low-impact cases.
2. A second model or adapted VLM is invoked when calibrated uncertainty, detected distribution shift, or model disagreement justifies the extra cost.
3. High-impact unresolved cases are sent to expert review; insufficient evidence results in abstention.
4. The system records the route, features that triggered escalation, actual outcome, latency, GPU/memory cost, and reviewer time prospectively.

The routing decision must be task-specific. The cited VLM preprint's strongest unseen-species result is on merge-error identification; it did not show the same improvement on split-error transfer. Do not infer one universal species/OOD policy from the win on a single task.

Benchmark the risk–coverage–cost frontier, not just accuracy:
- accepted-case error at 50/70/80/90/95% automatic coverage, wherever sample size supports the estimate;
- end-to-end cost per 1,000 accepted decisions, including model inference, escalations, reviewer minutes, and failed/retried calls;
- p50/p95 latency, peak device memory, and throughput under the same hardware and batching;
- quality and calibration by task × species × modality, including abstention and escalation rates;
- the counterfactual cost and error of the same system if every item were sent to the VLM, specialist, or human reviewer.

Do not multiply uncalibrated confidence scores, and do not allow the router to hide hard cases by abstaining on an entire disadvantaged subgroup without reporting its coverage.

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


### Executable V2 intake-lineage gate (new)

The dataset sidecar contract is now audited before fitting: unique composite sample hashes are recomputed as MD5(`source_archive|source_archive_sample_hash`); the exact sample-type → `same_neuron` → task-label and task-routing derivations are rechecked; geometry-mask flags are checked against metadata image types and modality slots; and the upstream split-group keys are reconstructed per archive bank. The audit checks that the same operation/junction/synapse grouping key and the same physical 50 µm spatial cube do not occur in multiple partitions, even across different source banks. Edit/adjacent-control rows group on `operation_id`, junction controls on `source_operation_id`, and synapse controls on (`pre_pt_root_id`, `post_pt_root_id`, `synapse_ctr_pt_nm`), following the pinned upstream postprocessor.

Root/neuron identifiers appearing across partitions are emitted as a dependence-risk warning rather than an automatic failure: the declared spatial split can place different pieces of one biological neuron in different cubes, so the overlap requires a neuron-grouped sensitivity analysis and should not be described as proven leakage by itself. Rows with no coordinate are counted and warned about; a present but malformed/non-finite coordinate fails closed. Integer-valued IDs are normalized to match the upstream split builder's Python `int()` semantics, including numeric strings. The Rust auditor consumes metadata-only JSONL; a checked-in DuckDB SQL adapter exports only the three metadata Parquet sidecars. The CLI requires `--dataset-revision-commit-sha` and records the exact caller-supplied Hugging Face V2 commit alongside the pinned code commit. This is an operator attestation; the offline auditor does not resolve the Hub ref or prove that bytes came from that commit. Its synthetic five-sample fixture covers all five sample types but contains no biological observations.

This gate is a material improvement over a manifest-only validator, but it still does not independently resolve the Hugging Face `v2` branch to an immutable Hub commit, verify image-byte/Parquet row correspondence, or provide the hidden test lockbox. The exact V2 commit SHA and sidecar digests must be captured in the run manifest before model benchmarking. A clean audit is necessary, not sufficient, for a leakage-safe performance claim.

### Counterfactual query-impact benchmark (prototype added)

A correction is not equally valuable for every downstream scientific question. The next evaluation layer therefore freezes a weighted directed baseline graph and a declared bundle of scientific queries, then measures how each proposed graph mutation changes those query outputs. The new `connectome-query-impact` binary currently supports edge-weight, weighted in-degree, weighted out-degree, directed path-existence, weighted feed-forward-loop, and weighted directed three-cycle queries; candidate proposals can add or remove edge weight. Motif queries run only on frozen subgraphs of at most 4,096 directed edges. Feed-forward motifs sum the product of the three edge weights; directed cycles are normalized across their three rotational traversals. For candidate (c) and query (q), it calculates the normalized query delta `min(abs(Q_q(G ⊕ c) - Q_q(G)) / scale_q, 1)`, combines those deltas using preregistered query weights, and reports a probability-weighted query-sensitivity heuristic per review minute, not an estimate of net scientific benefit.

This is intentionally a **counterfactual sensitivity measure**, not a truth score. A high value means a declared scientific query would change substantially if the proposed mutation were applied; it does not show that the proposed edit is correct, that the supplied error probability is calibrated, or that the chosen query bundle captures scientific importance. Queries may also carry an optional `reference_value` from an independent corrected graph; the report then evaluates signed normalized query-error reduction (positive means closer to the oracle, negative means worse). These oracle values are evaluator-only, are excluded from the ranking formula, and must never be exposed to candidate generation. Missing oracle values remain not estimable. The tool includes a synthetic helpful-edit vs harmful-edit control to test that this distinction is preserved. The CLI does not write corrections.

The input now freezes a `review_budgets_minutes` schedule. At each budget, six policies are replayed on the same candidate pool: query-sensitivity × error probability per minute; error probability per minute; uncertainty per minute; local weighted-degree change per minute; query sensitivity without error probability per minute; and deterministic random. The local-degree comparator sums absolute changes in weighted in-/out-degree across nodes after each candidate mutation; it is a transparent local structural baseline, not a global centrality algorithm.

Replay is sequential: candidates are considered in policy order against an evolving in-memory graph. Selected proposals consume their declared review minutes. A proposal is applied only if valid against the current graph; conflicting/incompatible proposals remain charged as reviewed but are rejected from the simulated graph. The report compares the starting/final weighted query error only where independent oracle values exist. This is a retrospective policy simulation that assumes every selected, valid proposal is accepted; it does not model human rejection, deliberation, or evidence gathered during review. It is not measured reviewer-effort evidence. The real value-of-review claim still requires prospective human outcomes and matched-minute cluster-aware comparisons. Every candidate is marked `HUMAN_REVIEW_ONLY`. Candidates that touch the same baseline edge are flagged, since independent scores cannot be added when proposals interact. The current implementation does not yet simulate joint candidate sets, query uncertainty, mesh/skeleton morphology, motifs, path multiplicity, or measured after-review outcomes.

A synthetic fixture and regression tests include a meaningful negative control: removing a direct edge changes its edge-weight query but does **not** change reachability because an alternate path remains. Another control confirms that candidates sharing an edge are detected. The report retains each candidate's evidence SHA-256 and computes deterministic canonical SHA-256 digests for the baseline and counterfactual graphs from sorted directed edge triples encoded as big-endian node IDs and IEEE-754 weight bits. The caller-supplied source-graph digest is kept separate from the computed canonical digest. Next research gate: replace the synthetic graph with frozen query-relevant subgraphs and independent adjudicated corrections, then compare impact-per-minute ranking against error-probability-only, uncertainty-only, graph-centrality-only, random, and active-learning queues at matched expert-minute budgets. Require prospective review outcomes and paired, cluster-aware confidence intervals before making an efficiency claim. The full preregistered comparison is tracked in [SCI-CONNECTOMICS-002 / issue #9134](https://github.com/Luminous-Dynamics/symthaea/issues/9134). Spatially conditioned motif nulls are tracked separately in [SCI-CONNECTOMICS-003 / issue #9156](https://github.com/Luminous-Dynamics/symthaea/issues/9156).

### Post-hoc reviewer-outcome evaluation (isolated from ranking)

The initial matched-budget replay assumed every selected, structurally valid proposal was accepted. That is useful for testing graph mechanics but is not a realistic estimate of proofreading workflow quality. A separate `connectome-review-outcome-audit` binary now evaluates frozen policy rankings against a separately stored outcome corpus containing (a) independent adjudication of whether each proposed correction is correct, (b) the actual expert decision to accept/reject/abstain, and (c) measured review minutes.

The evaluator is a separate executable from `connectome-query-impact`. It binds the outcome file to the exact proposal-input SHA-256, exact frozen-ranking-report SHA-256, and an adjudication-protocol SHA-256. The ranker does not parse this schema or consume reviewer decisions/correctness labels. The evaluator replays the saved policy order using measured review minutes, applies only accepted mutations sequentially, applies accepted incorrect changes so query harm is visible, and records accepted proposals that become stale conflicts after earlier mutations.

Reported measures include accept/reject/abstain crossed with correct/incorrect/indeterminate adjudication; accepted-decision precision; applied-correction precision; candidate-pool correct-proposal review, acceptance, and application recall; incorrect-proposal acceptance rate; measured reviewer minutes; stale conflicts; and baseline/final oracle query error. Indeterminate adjudications are reported separately and excluded from adjudicated precision denominators. The three recall metrics have the **frozen candidate pool** as their denominator and must not be described as whole-connectome recall.

A real offline benchmark should first freeze its candidate pool, then use a blinded shadow-review corpus: reviewers do not see which policy ranked a candidate, and an independent adjudicator establishes proposal correctness. The cost of building this full outcome corpus is reported separately from simulated policy review budgets. Reviewer order/learning effects and the causal effect of different queue orders are not captured by retrospective replay; a randomized prospective policy trial is still required for causal claims. No authoritative annotations are modified.

The outcome schema and template are `docs/neuroscience/CONNECTOME_REVIEW_OUTCOMES.schema.json` and `docs/neuroscience/examples/CONNECTOME_REVIEW_OUTCOMES.example.json`. Operational details and CLI sequence are in the connectomics tool README.

### Spatially conditioned motif inference — next research gate

The current query-impact prototype reports weighted feed-forward-loop and directed three-cycle **counts only**. Those counts are sensitivity measures on the declared graph; they are not evidence of motif enrichment or biological significance. NEURD explicitly discusses axon–dendrite proximity controls for connectivity motifs, so raw counts and a global degree-preserving configuration model are not sufficient scientific baselines.

The next extension should add a *spatial opportunity null ladder*: directed degree-preserving configuration null; distance-bin/compartment-conditioned edge rewiring; and spatial + validated cell-type conditioned nulls where sample sizes support them. Use synapse/pre-post compartment locations or validated axonal/dendritic skeleton segments, not merely soma-centroid distance. Record anisotropy, boundaries, completeness and geometry provenance. Report observed weighted motif count, null mean/distribution, empirical p-value with finite-sample correction, standardized effect only where justified, Monte Carlo uncertainty, mixing diagnostics, and multiplicity correction across motif families.

Synthetic controls must show that degree-preserving rewiring can retain raw motif counts while changing spatial opportunity, and that location shuffling breaks the geometry correspondence. Missing/uncertain compartment labels remain explicit. The implementation and promotion gates are tracked in [SCI-CONNECTOMICS-003 / issue #9156](https://github.com/Luminous-Dynamics/symthaea/issues/9156). Until this gate exists, do not describe the prototype's motif outputs as proximity-corrected or biologically significant.

### Comparator matrix

1. **Published ConnectomeBench2 ViT baseline**, reproducing the pinned `v2` dataset, modality, and split protocol where possible.
2. **Specialist task models and morphology rules**, including NEURD where source data and operation semantics are compatible.
3. **VLM comparison arms:** zero-shot, few-shot, and LoRA-adapted configurations from the September/October 2026 VLM study where model/code/artifact access and license permit faithful reproduction.
4. **Risk-budgeted cascade:** specialist → uncertainty/OOD gate → adapted VLM → expert review/abstention, compared against always-run specialist, always-run VLM, and simple threshold routing at matched quality/coverage targets.
5. Existing segmentation/tracing output with no additional triage.
6. Random review at the same expert-review budget.
7. Simple uncertainty-only and calibrated-uncertainty-only ranking.
8. Model-disagreement ranking, with shared ancestry tracked where known.
9. Active-learning baseline matching the same annotation budget; report the labels needed to reach a fixed target quality.
10. An error/impact ranker using only directly observable, available features.
11. Symthaea orchestration with correction memory and budgeted selection.
12. Ablations: no HDC retrieval; no correction memory; no active selection; no graph-impact term; no calibration/abstention policy.
13. Expert review as a reference ceiling only where expert labels and reviewer effort are actually measured. Never present the ceiling as an automated baseline.
14. SynAnno-style guided neuron-centric proofreading workflow as an interactive human-review comparator. Compare task navigation, progress recovery, reviewer effort, cognitive load and annotation quality; do not equate this synapse-annotation study with merge/split model accuracy.

Keep model version, prompts, preprocessing, compute, reviewer instructions, candidate pool, and stopping rules fixed across paired comparisons. Give every baseline the same input access and review budget.

### Primary metrics

**Human-effort efficiency**
- reviewer minutes per independently verified high-impact correction;
- number of reviews required to reach a preregistered error/quality target;
- impact-weighted true-error yield at fixed review budgets;
- reviewer disagreement and escalation time;
- all-in latency/compute used to produce and rank the queue.
- all-in model + reviewer cost per 1,000 accepted decisions, with per-model inference cost, escalation rate, and reviewer minutes separated;
- p50/p95 end-to-end latency, peak memory, throughput, and fail/retry rate under matched hardware and batch assumptions;
- risk–coverage–cost Pareto frontier for specialist-only, VLM-only, human-only, and cascaded policies;

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

Use paired, cluster-aware uncertainty estimates (cluster on volume/region/neuron, as applicable), report denominators and missing labels, and publish per-dataset results. For budgeted queue comparisons, report paired cluster-bootstrap 95% intervals for submitted-policy minus each baseline at the same item/minute budgets (impact-weighted recall, error recall, and precision), not only separate point estimates. The scorer's intervals are conditional on frozen predictions and the declared cluster unit. A pooled average cannot hide a failed dataset or error family. The frozen scorer now calculates classification, balanced-accuracy cluster-bootstrap intervals, item/minute review yield, and risk–coverage checkpoints separately for each declared group, using the same operating budgets as the pooled report. It also evaluates three matched-budget controls—uncertainty-only, predicted-error-probability-only, and a seeded deterministic-random queue—overall and per group. Treat per-group operating curves and baseline comparisons as primary evidence, not decorative breakdowns; if support is too small, retain the denominator and mark the claim not estimable.

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
- **Cascade efficiency target:** at fixed accepted-case risk and automatic coverage, reduce all-in inference plus reviewer cost by at least 20% versus the strongest eligible always-run or simple-routing baseline. Cost accounting must be defined before lockbox evaluation; if expert time cannot be measured prospectively, mark the real-world cost target not estimable.

These targets are intentionally about efficient, safe proofreading, not unsupported claims of beating every segmenter or achieving whole-brain autonomy.

## Symthaea-specific ablation plan

The first implementation is deliberately small. The offline intake gate and frozen-prediction scorer are standalone Rust binaries. A sequential stopping-snapshot audit now validates lineage and edit-count continuity, reports descriptive tail stability, and performs one-step-ahead walk-forward predictions for graph invariants and declared query metrics. It compares an asymptotic curve fit with persistence and linear-trend baselines using only prior snapshots for each prediction. The relative-MAE comparison is computed on the same held-out target snapshots where the asymptotic fit was estimable, preventing the forecast from looking better simply by skipping difficult cases. This is a forecast diagnostic, not yet an estimator of time/edits remaining to a user-selected quality threshold; it emits no calibrated prediction intervals and always recommends `CONTINUE_REVIEW`. Authored unit tests exist, and an earlier revision of the intake/scorer/stopping audit passed CI; the newest walk-forward additions still require a clean CI result before they are considered validated. Next add Parquet/label/source-lineage audit, then test whether cognitive components add value.

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
   - **Stopping-curve diagnostic (prototype added):** `tools/connectomics-intake-audit/src/bin/connectome_stopping_audit.rs` validates sequential snapshot lineage and calculates walk-forward one-step predictions for graph invariants and declared query metrics. For each target snapshot, it fits an asymptotic curve and a linear trend only to earlier snapshots, then compares both against last-value persistence on the held-out value. The report includes per-series MAE and relative MAE improvement. This comparison conditions on the target's cumulative edit count known retrospectively; it does not yet forecast wall-clock time or edits needed to reach a chosen threshold, and has no calibrated prediction interval. It always returns `CONTINUE_REVIEW`. A synthetic fixture and no-leakage test are included; the new forecast code awaits successful CI verification.
3. **Offline scorer (prototype added):** `tools/connectomics-intake-audit/src/bin/proofreading_score.rs` scores frozen JSONL predictions with task/group breakdowns, balanced accuracy, AUROC, tie-aware average precision, Brier/log loss, item/minute-budget error yield, impact-weight capture, risk–coverage checkpoints, frozen policy/config hashes, and deterministic cluster-bootstrap intervals for balanced accuracy. The schema-version-4 report includes per-group operating metrics and matched-budget comparisons against uncertainty-only, probability-only, and deterministic-random queues. It also reports paired cluster-bootstrap 95% intervals for submitted-priority minus each control at every item/minute budget (impact-weighted error recall, ordinary error recall, and precision), so the result tests paired improvement rather than relying on separate point estimates. It is intentionally separate from runtime inference and always emits `SCORED_NOT_PROMOTED`. The intake tool still needs Parquet parsing and row/label/source-lineage audit before a real ConnectomeBench2 score can be produced. Fifteen intake tests and seventeen scorer tests are authored. CI previously passed these tools before the walk-forward forecasting changes; the latest changed scorer/stopping audit still need their own green run before promotion.
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


## Multi-rater reliability and reviewer calibration (SCI-CONNECTOMICS-004)

The first outcome evaluator retains one reviewer decision and one adjudication per candidate. That is a useful minimum but cannot quantify reviewer disagreement or uncertainty in the reference label. A separate multi-rater instrument now preserves the full panel:

- [Multi-rater protocol V1](CONNECTOME_MULTI_RATER_PROTOCOL_V1.md)
- [Outcome JSON Schema](CONNECTOME_MULTI_RATER_OUTCOMES.schema.json)
- [Synthetic outcome fixture](examples/CONNECTOME_MULTI_RATER_OUTCOMES.example.json)
- [Tracking issue #9186](https://github.com/Luminous-Dynamics/symthaea/issues/9186)

The standalone `connectome-multi-rater-audit` binary binds the outcome file to the exact proposal bytes, exact frozen ranking-report bytes, and adjudication-protocol digest. It requires at least two adjudicators and two blinded reviewer records per candidate; reviewer and adjudicator IDs must be disjoint. It retains every vote and checks complete frozen candidate-pool coverage. The ranker does not read this schema or these outcomes.

The evaluator emits separate nominal Krippendorff alpha, observed/expected disagreement, strict-majority coverage and vote fractions for reviewer decisions and adjudicator labels; candidate-level unresolved/tied outcomes; pooled and per-reviewer decision metrics; accept/reject confidence Brier score and 10-bin ECE on determinate candidates; and total panel-review minutes. Abstentions are explicit and confidence must be null for abstain. Confidence calibration excludes unresolved and indeterminate reference labels.

**Interpretation limits:** strict majority is a summary, not biological truth. Agreement is not correctness. Pooled reviewer events share candidates and must not be treated as independent samples. Panel minutes are summed person-minutes, not elapsed queue wall time. The implemented agreement statistic uses pair-count-weighted nominal disagreement; it is not a latent-rater truth model. Dawid–Skene-style aggregation remains future work and requires independent anchor cases and a validated protocol before it can influence any decision.

For collection, use at least three independent adjudicators per candidate where feasible (two is only the executable minimum), blind reviewers to policy/model identity and other votes, randomize candidate order, retain abstentions and unresolved cases, and account for the full-corpus adjudication cost separately. Causal queue-efficiency claims still require a prospective randomized or block-randomized policy trial. No biological multi-rater corpus has yet been evaluated, and no performance gain is claimed.


## Candidate-generation recall beyond the proposal pool (SCI-CONNECTOMICS-006)

The queue and unreviewed-tail evaluators condition on the system's proposal pool. That leaves candidate-generation false negatives unidentified: an error the model never proposes cannot appear in either precision or candidate-pool recall denominators. A high-performing queue could therefore optimize the wrong subset of the volume.

A separate, evaluator-only probability audit now defines the missing estimand:

- Sample complete primary sampling units (PSUs) from a declared, exhaustive volume/neuron search frame using stratified simple random sampling without replacement.
- Independently search every selected PSU exhaustively, retaining selected empty units rather than dropping them.
- Map each independently verified error to zero or more candidate IDs in the frozen proposal pool. Empty mapping means the generator missed the error; duplicate or unknown candidate references invalidate the run.
- Estimate finite-frame error totals, covered totals, missed totals, and independent panel person-time with stratified Horvitz–Thompson estimators.
- Require the complete error-family vocabulary to be preregistered; retain families with no sampled observations as zero-estimate/undefined-recall rows rather than silently omitting them. Their sampled counts must be visible; an unseen rare family is not proven absent.
- Withhold normal intervals for non-census totals/ratios when sample count is below 30 in any non-census stratum or observed residual sample variance is zero.
- Report candidate-generation recall and **impact-weighted candidate-generation recall** separately, with first-order Taylor ratio variance and conservative interval gating.
- Keep this evaluator isolated from candidate generation and ranking, and distinguish it from #9187's residual-risk sample within the generated candidate pool.

Artifacts:
- [Protocol V1](CONNECTOME_CANDIDATE_COVERAGE_AUDIT_PROTOCOL_V1.md)
- [JSON Schema](CONNECTOME_CANDIDATE_COVERAGE_AUDIT.schema.json)
- [Synthetic fixture](examples/CONNECTOME_CANDIDATE_COVERAGE_AUDIT.example.json)
- [Tracking issue #9204](https://github.com/Luminous-Dynamics/symthaea/issues/9204)

This estimates recall only over the finite declared PSU frame and only to the extent that the independent search protocol can detect and adjudicate errors. It is not whole-connectome recall. It does not measure causal queue-policy effects, because the policies do not control this independent sampling process. No biological coverage sample has been evaluated and no SOTA result is claimed.


## New research frontier: beyond single-candidate recall

The next phase adds four orthogonal research questions. They are deliberately separated so a gain in one layer cannot conceal a failure in another.

### Morphology-conditioned candidate discovery (SCI-CONNECTOMICS-007)

- [Protocol V1](CONNECTOME_MORPHOLOGY_CANDIDATE_GENERATION_V1.md)
- [Issue #9241](https://github.com/Luminous-Dynamics/symthaea/issues/9241)

Compare the current generator, EM-only, rendered-geometry-only, full-mesh/skeleton morphology rules, NEURD-compatible rules, and the union of independent specialists. Preserve detector provenance and use the #9204 independent PSU sample to estimate recall by error family, region, species, and impact. The generator and ranker must not receive audit outcomes.

**Representation limit:** the ConnectomeBench2 card describes local geometry renderings as three 2D views with seven channels at 224×224 and EM as 2D image slices. Those are not whole-neuron morphology graphs. Resolve full meshes, skeletons, and synapse sidecars from the original volume and pin their provenance before claiming NEURD-like morphology evaluation. The dataset card also reports that EM availability differs by sample type; modality comparisons must use a matched eligible subset and report missingness by task/class to avoid selection confounding. Verify these properties again against the exact v2 revision before intake promotion.

### Errors missed by every discovery channel (SCI-CONNECTOMICS-009)

- [Protocol V1](CONNECTOME_MULTI_LIST_UNSEEN_ERROR_SENSITIVITY_V1.md)
- [Issue #9248](https://github.com/Luminous-Dynamics/symthaea/issues/9248)

An expert-declared exhaustive search can itself miss errors. Use multiple independently frozen discovery lists—specialist image model, morphology rules, current generator, and blinded expert teams—and preserve each event's capture history. Multiple-systems estimation is a sensitivity layer only: list dependence, unequal detectability, identity linkage, and the unobserved all-zero cell are difficult or non-identifiable without assumptions. Report model-sensitivity ranges and explicit NOT_IDENTIFIABLE states. Do not replace the #9204 design-based estimator.

### Incomplete neurite extension and false-split recovery (SCI-CONNECTOMICS-010)

- [Protocol V1](CONNECTOME_NEURITE_EXTENSION_BENCHMARK_V1.md)
- [Issue #9249](https://github.com/Luminous-Dynamics/symthaea/issues/9249)

Separate cleaning false merges from recovering true incomplete processes. Benchmark against NEURD, RoboEM, mesh-tip extension methods, Autoproof where reproducible, and expert continuation. Evaluate endpoint recall, recovered cable length and synapses, false joins, branchpoint preservation, path-level uncertainty, and expert time. Boundary-truncated processes must not be mistaken for internal segmentation errors.

### Joint correction interactions and portfolio selection (SCI-CONNECTOMICS-011)

- [Protocol V1](CONNECTOME_JOINT_CORRECTION_PORTFOLIO_BENCHMARK_V1.md)
- [Issue #9250](https://github.com/Luminous-Dynamics/symthaea/issues/9250)

Individual counterfactual scores are not generally additive. Evaluate pairwise synergy/antagonism, higher-order residuals, edit order, stale references, and compatible correction portfolios. Compare isolated-score greedy ranking with conflict-aware, interaction-aware, and bounded portfolio search under identical review-minute budgets. Retrospective utility is not a causal human-efficiency claim.

### Prospective causal workflow evaluation (SCI-CONNECTOMICS-008)

- [Protocol V1](CONNECTOME_PROSPECTIVE_QUEUE_TRIAL_PROTOCOL_V1.md)
- [Issue #9242](https://github.com/Luminous-Dynamics/symthaea/issues/9242)

A blocked-randomized workflow trial holds the interface and safety permissions fixed while comparing the existing workflow with the frozen morphology-conditioned workflow. Use independent adjudication, intention-to-treat analysis, and both in-pool (#9187) and out-of-pool (#9204) probability audits. Count audit, adjudication, compute, storage, and reviewer time in the all-in cost.

## Industry comparison update

The key comparators are not interchangeable:

- **ConnectomeBench2** is the strongest broad local proofreading classification benchmark currently in this workstream: over 716,000 expert decisions and more than 4.5 million associated images in the paper. Its dataset card describes 401,170 released operation-level samples, with local image/geometry renderings. The 2026 VLM preprint reports Gemma 4 12B LoRA balanced accuracy of 95.0% for split-error and 90.9% for merge-error tasks, compared with 94.2% and 91.1% for the EM-only ViT-B comparator. This is task/protocol-specific; it is not a complete morphology or workflow comparison.
- **NEURD** provides decomposed neuron meshes and morphology/graph rules; the 99%/95% human agreement is restricted to two high-confidence merge-rule subsets. It also reports a trade-off in some axonal synapse recall after cleaning, and its current method does not repair incomplete neurites.
- **RoboEM** is a strong continuation/tracing comparator. Its paper reports approximately 400-fold lower computational annotation cost than manual correction for the evaluated cortical connectomes and 3.5-fold better split reduction than the compared agglomeration baseline. These are its paper-specific task and cost definitions.
- **Autoproof** reports, in a 2025 preprint, about 90% of the value of a guided workflow at 80% of its cost and large-scale fragment attachment on the Drosophila male central nervous system. Treat as a preprint comparator until independently replicated on matched tasks.
- **SynAnno** shows that guided neuron-centric navigation and review interfaces can materially change expert throughput and cognitive load; its user evaluation involved seven experts.

The strategic target is therefore not "beat every published headline." It is a fair Pareto comparison on matched data and endpoints: high-impact search coverage, safe false-join/false-correction control, completeness, scientific-query accuracy, actual expert minutes, label burden, and all-in cost. Preserve each comparator's task boundaries and report unmatched metrics as not comparable.

## Proposed integrated promotion gates

These remain research targets, not measured Symthaea results:

1. Candidate-generation recall ≥95% point estimate and design-adjusted lower 95% bound ≥90% on a declared frame, with family-specific support and bounds.
2. Incomplete-process extension: critical-family lower 95% endpoint-recall bound ≥90%, while false-join rate is no worse than the strongest compatible baseline at matched completeness.
3. Correction portfolio: zero regret on enumerable synthetic portfolios up to 20 candidates; ≥10% held-out query-error reduction over the strongest greedy baseline on larger portfolios at matched budget before any biological promotion.
4. MSE: calibrated synthetic sensitivity analysis with explicit refusal under non-identifiability; no unsupported natural-data point estimate.
5. Human workflow: ≥20% reduction in actual expert time at independently matched quality and coverage, or ≥15% relative gain in signed query-error reduction per expert-minute, confirmed prospectively with design-aware intervals.
6. No authoritative automatic mutations, no hidden audit-label leakage, and exact provenance for every input, model, policy, sample, correction, and output.

A pooled pass cannot override a failed critical-family, false-correction, false-join, or integrity gate.


### Query-scoped proof-carrying reliability certificates (SCI-CONNECTOMICS-012)

- [Protocol V1](CONNECTOME_QUERY_RELIABILITY_CERTIFICATE_V1.md)
- [Issue #9255](https://github.com/Luminous-Dynamics/symthaea/issues/9255)

Replace a global "proofread / not proofread" flag with a scoped certificate for each scientific question. Each certificate binds the exact query, code, data revisions, dependency closure, corrections, sampling evidence, family-specific coverage, rater disagreement, null models, and uncertainty. A relevant edit or source revision invalidates the certificate; provably unrelated edits need not.

The certificate must explicitly distinguish observed evidence, design-based sampling uncertainty, in-pool risk, multi-list unseen-error sensitivity, correction interactions, and query-model uncertainty. It may not convert these components into one aggregate confidence score unless a validated joint model exists. Its state is CERTIFIED_WITHIN_DECLARED_SCOPE, CONDITIONAL, NOT_CERTIFIED, or INVALIDATED_BY_REVISION. A digest establishes evidence identity, not biological truth.


- Machine-readable certificate contract: [schema](CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.schema.json) and [synthetic fixture](examples/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.example.json). The fixture is explicitly conditional and synthetic; it is not a biological certification.


## Proposed Symthaea integration architecture

Do not force one model to replace image segmentation, mesh decomposition, skeleton tracing, graph editing, and reviewer workflow. Compose specialist components behind a common evidence contract:

1. **Specialist perception and reconstruction:** task-matched EM/VLM classifiers, geometry models, NEURD-compatible decomposition, RoboEM-like continuation, and source-specific segmentation tools. Each contributes typed evidence and versioned hypotheses.
2. **Event-sourced evidence graph:** immutable source objects, transforms, candidate hypotheses, detector outputs, adjudication events, correction branches, and derived query values. Store provenance and dependency edges, not only final scores.
3. **Symthaea cognitive orchestration:** test HDC for compositional association/retrieval across typed evidence and prior reviewed cases; test CfC/liquid state for temporal change tracking across revisions and proofreading trajectories. These are candidate architectural roles, not assumed advantages. Compare against conventional indices, explicit event sourcing, calibrated ensembles, and simpler temporal models on latency, memory, retrieval accuracy, calibration, and cost. Do not use an HDC/CfC claim as a substitute for EM/mesh baseline performance.
4. **Independent assurance services:** keep design-based sampling, multiple-list sensitivity, and blinded adjudication outside generator/ranker information flow.
5. **Counterfactual portfolio engine:** evaluate correction dependencies, joint utility and order effects on immutable graph snapshots.
6. **Query-certificate verifier:** compute dependency closure, uncertainty components, quality gates, and automatic invalidation after dependent changes.
7. **Human review interface:** adopt useful neuron-centric navigation patterns and log real interaction time, decisions, reversals, and cognitive load.

No stage receives production write credentials. Only an accepted, independently validated correction may enter a reversible annotation branch. Every stage must have an independently tested contract and a kill condition.


### Executable query-certificate consistency gate

- [Certificate protocol](CONNECTOME_QUERY_RELIABILITY_CERTIFICATE_V1.md)
- [Schema](CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.schema.json)
- [Synthetic example](examples/CONNECTOME_QUERY_RELIABILITY_CERTIFICATE.example.json)
- `connectome-query-cert-audit` validates cross-field state, dependency, digest, correction-order, and invalidation invariants. The expected synthetic result is `PASS` for a `CONDITIONAL` certificate; this is not biological evidence.


### Spatial PSU frame integrity gate (SCI-CONNECTOMICS-013)

- [Protocol V1](CONNECTOME_SPATIAL_FRAME_AUDIT_PROTOCOL_V1.md)
- [JSON Schema](CONNECTOME_SPATIAL_FRAME.schema.json)
- [Synthetic frame](examples/CONNECTOME_SPATIAL_FRAME.example.json)
- [Issue #9260](https://github.com/Luminous-Dynamics/symthaea/issues/9260)

The `connectome-spatial-frame-audit` executable validates exact 3D lattice tiling inside a declared voxel bounding box, partitions every tile into eligible or explicitly excluded, and exports the eligible PSU population. #9204 now requires the exact spatial-frame report bytes and both the complete lattice and eligible-population digests before it will estimate candidate-generation recall. This proves tiling only within the declared box; source and eligibility-mask completeness remain external evidence gates.


The downstream #9204 invocation supplies both the exact spatial-frame manifest and the exact PASS report. It recomputes the manifest byte digest and checks it against the report, then verifies that the audit's eligible PSU/stratum rows exactly equal the report export. The coverage report retains separate hashes for the manifest, report, complete lattice, and eligible population.

### Hidden co-linear merge boundary detection (SCI-CONNECTOMICS-014)

- [Protocol V1](CONNECTOME_HIDDEN_COLINEAR_MERGE_BENCHMARK_V1.md)
- [Issue #9265](https://github.com/Luminous-Dynamics/symthaea/issues/9265)

NEURD's paper identifies a specific blind spot: some merge errors are co-linear in the skeleton and collapse into one non-branching segment, leaving no graph junction for ordinary node/branch rules to inspect. The proposed response is an intra-segment arclength scan using independent skeleton, mesh, EM, synapse and compartment evidence. Benchmark internal boundary localization in nanometres, candidate recall per independently sampled PSU, and false alarms per kilometre of cable. Preserve biological taper, boutons, spines, and true compartment transitions as negative controls.

### Spatial PSU frame integrity (SCI-CONNECTOMICS-013)

The spatial-frame auditor reports voxel *tile* volumes for eligible/excluded lattice cells; these are not counts of foreground voxels in the external eligibility mask. The mask itself is only digest-bound and must be independently inspected. The candidate coverage tool now takes both the exact frame manifest and exact frame-audit report and checks their byte digests, full lattice digest, and eligible population digest. This is a provenance/geometry gate, not proof of biological completeness.

## Executable advances: joint edits and mask-bound sampling frames (2026-10-30)

Two additional assurance components now have executable implementations on the Connectomics Autopilot branch:

- [Joint correction portfolio audit](CONNECTOME_CORRECTION_PORTFOLIO_AUDIT_PROTOCOL_V1.md): exact candidate-subset optimum for up to 20 graph edits, pairwise interaction terms, four non-oracle ranking baselines, matched minute budgets, regret, and reverse-order sensitivity. Synthetic only; no biological gain claimed.
- [Spatial eligibility-mask audit](CONNECTOME_SPATIAL_FRAME_AUDIT_PROTOCOL_V1.md): the frame report only emits PASS after streaming and hashing the exact mask artifact, validating bit order/length/padding, and checking that every eligible PSU contains mask foreground voxels while every excluded PSU contains none. This still does not certify that the mask's biological selection rule is correct or that the declared box is complete.

The combined target is an auditable chain from a declared tissue sampling frame to candidate-generation recall and downstream query reliability. A numerical recall estimate is not promoted if the frame or mask integrity gate fails.


- [Hidden co-linear merge executable baseline](CONNECTOME_COLINEAR_MERGE_AUDIT_PROTOCOL_V1.md): separate evaluator-only truth, a no-proposal control, radius-only versus multifeature change-point proposals, and negative controls for taper, bouton pulses, imaging artifacts, and compartment transitions. The synthetic fixture intentionally shows recall improving from 1/2 to 2/2 while false proposals rise from 1 to 7. Synthetic only; not a biological detector.


### Query-specific dependency closure hardening (2026-10-30)

The query-certificate contract now carries explicit `query.dependency_roots` and typed `dependency_closure[].depends_on` edges. The Rust verifier rejects unresolved references, duplicate roots/edges/nodes, directed cycles, and unreachable extra dependency nodes. The synthetic path query roots at its exact edge record and traverses to both endpoint neurons.

This is a graph-closure consistency proof relative to the supplied query roots and edges. It does **not** prove that the root manifest includes every scientifically relevant object; the query-specific extractor and source mapping still need independent validation. The verifier now has 23 authored tests, not yet compiled or run on this head.


### Acquisition–proofreading co-design research track

Issue [SCI-CONNECTOMICS-017 / #9282](https://github.com/Luminous-Dynamics/symthaea/issues/9282) extends the decision space upstream into adaptive EM acquisition. It proposes query-value-of-information comparisons between targeted re-imaging, specialist retracing, morphology evidence review, human proofreading, and abstention, with independent spatial sampling of untouched regions to prevent active-selection bias. This is a research track only; it does not authorize autonomous microscope control.


### Current implementation test inventory

Authored source tests at the current branch state: spatial-frame/mask 22; candidate-generation coverage with downstream geometry/mask re-derivation 35; correction portfolio 11; co-linear baseline/evaluator 9; query certificate and dependency DAG 23. **100 authored tests total; not yet compiled or executed.** The presence of these tests is not a passing CI result.

## Domain-shift calibration and abstention (SCI-CONNECTOMICS-023)

Issue [#9403](https://github.com/Luminous-Dynamics/symthaea/issues/9403) adds an optional domain-conditional evaluation path to the frozen proofreading-score tool. This closes a gap between pooled calibration and safe transfer: a model may remain well-ranked while becoming overconfident on a held-out species, source volume, region, acquisition protocol, segmentation lineage, modality-availability regime or morphology regime.

The scoring contract separates four quantities:
- p_error: calibrated probability of a genuine reconstruction error;
- review_priority_score: frozen queue-ordering score;
- ood_score: distinct novelty score;
- is_ood_domain: predeclared held-out status derived from the split protocol, never inferred from model confidence.

One run is restricted to one shift axis. Every row must carry the complete domain tuple or none; partial domain metadata, inconsistent held-out status within a domain, multiple axes in one file, and partial novelty scores fail closed. Per-domain reliability bins, ECE/MCE, Brier/log loss, high-confidence false-accept rates and low-confidence missed-error mass are emitted separately. OOD AUROC/AUPRC are computed only when every row has a novelty score; an optional fixed OOD threshold reports its false-referral and missed-OOD operating point.

This is an evaluator extension, not an OOD-aware production router. The current report includes independent cluster counts but does not yet add cluster-bootstrap intervals for ECE/MCE/OOD AUROC. Those point estimates are descriptive and cannot promote a policy. Next gates are a cluster-aware uncertainty implementation, known-finite-population shift simulations, and locked source-volume/species evaluation with no calibration leakage. No natural-data transfer or SOTA win is claimed.

Protocol: [CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_PROTOCOL_V1.md](CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_PROTOCOL_V1.md). Input schema: [CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_INPUT.schema.json](CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_INPUT.schema.json). Synthetic regression input: [example JSONL](examples/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT.example.jsonl).
