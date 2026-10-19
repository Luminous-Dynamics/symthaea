# VIS-REP-001 — Visual Representation Utility Benchmark

**Status:** proposed protocol; no encoder ranking or performance result is qualified by this document.  
**Owner:** Symthaea vision workstream  
**Issue:** [VIS-REP-001 #8152](https://github.com/Luminous-Dynamics/symthaea/issues/8152)

## 1. Purpose and claim boundary

Determine when the current Patch-HDC representation, a frozen learned dense encoder, or a hybrid representation produces better visual task outcomes under matched evidence, downstream logic, and resource constraints.

This is not a universal claim that one representation family is better. The result is conditional on the tested task, distribution, hardware, model revision, and budget.

Compose existing contracts instead of forking them:

- [Vision Mind #3934](https://github.com/Luminous-Dynamics/symthaea/issues/3934) — provider-neutral vision roadmap and provenance → entity → world model sequencing.
- [VIS-002 / PR #3973](https://github.com/Luminous-Dynamics/symthaea/pull/3973) — dense-feature evidence boundary.
- [VIS-EVAL-000A / #4107](https://github.com/Luminous-Dynamics/symthaea/issues/4107) — raw tracker-state baseline.
- [Symtropy SYM-EVAL-001A / PR #1214](https://github.com/Luminous-Dynamics/symtropy/pull/1214) and [anti-shortcut audit #1281](https://github.com/Luminous-Dynamics/symtropy/issues/1281) — hidden-truth evaluation and benchmark-validity controls.

An unqualified predecessor blocks any claim that depends on it. A new document or passing isolated test does not promote a queued or failed dependency.

## 2. Core hypotheses

H1. Frozen learned dense features improve appearance correspondence under nuisance transformations relative to the current Patch-HDC baseline.

H2. Hybrid features improve task-level temporal association in selected conditions, but may cost more latency and memory.

H3. The best representation depends on the failure regime: simple/local features may be competitive for constrained scenes, learned features may help semantic robustness, and temporal belief or active sensing may dominate raw representation changes on occlusion and ambiguity.

H4. A representation that scores well on static probes may not improve object persistence, prediction, or downstream task utility.

All four hypotheses are falsifiable. Publish null results and per-stratum regressions.

## 3. Comparison arms

| Arm | Encoder / representation | Purpose |
|---|---|---|
| A | Existing `PatchHdcEncoder`, frozen configuration + seed | Current system baseline |
| B | Deterministic local pixel / gradient descriptor | Shortcut and low-cost control |
| C | Frozen learned dense visual encoder using VIS-002 | Representation-strength baseline |
| D | Hybrid of A and C with preregistered fusion | Main research candidate |
| E | Applicable established video segmentation/tracking system (e.g. SAM 2) | Separate task-level practical comparator |

E is not an interchangeable encoder-only baseline; report it in a distinct comparison lane. Freeze model/version, preprocessing, numerical precision, feature extraction, hardware and license details. Model weights are optional offline inputs and must not be downloaded by CI. Confirm weight licenses against the intended distribution before using any candidate.

For the controlled A/C/D ablation, keep data splits, evaluator, downstream tracker/belief rules, and resource boundary identical where the adapters permit. Any learned fusion is trained only on training partitions with a declared parameter count and compute/data budget. Validation and test outcomes must not influence fusion, thresholds or checkpoint choice.

## 4. First corpus: 48 deterministic synthetic cases

Use Symtropy's existing hidden-truth scenario machinery. This corpus is a proposed workload plan, not evidence that the current evaluator already supports all case semantics.

Create six scenario families with eight fixed cases each, varying seed, object/texture identity and nuisance parameters within each family:

1. **Occlusion and reacquisition:** brief, prolonged, partial and complete occlusion; target reappears at the same or a changed location.
2. **Identity ambiguity:** visually similar distractors; crossing tracks; paired identity swaps; changed viewpoint.
3. **Appearance nuisance:** illumination, shadow, scale and background changes while object identity is held fixed.
4. **Camera/object motion:** static world with camera motion versus independently moving target; changes in apparent speed and direction.
5. **Lifecycle and causality:** true removal versus temporary invisibility; contact/displacement; paired scenes differing in one causal event.
6. **Negative and degraded evidence:** blank scene, absent target, ambiguous scene, frame drop, corrupt dimensions, stale calibration and missing evidence.

The case manifest freezes IDs, family, factor levels, seed, renderer/configuration identity, expected evaluator-only labels, and split assignment. Keep ground truth, branch assignment and outcome labels inaccessible to the producer.

Partition by generator parameters and object/texture identity families, not random adjacent frames. Any train/validation/test split that permits near-duplicate rendered scenes to cross boundaries must be rejected.

Before expanding to public video, prove basic behavior with these small fixtures. Later adapters may cover the [TAO open-world tracking dataset](https://taodataset.org/) or applicable video segmentation datasets; dataset-specific evidence must be kept distinct and honor each dataset's terms.

## 5. Causal delivery and leakage controls

At time step t, an evaluated producer may observe only frames 0 through t and their admitted metadata. The API must not expose full future trials.

At minimum, seed negatives for:

- future-frame access;
- branch/outcome label leakage through metadata, object IDs, filenames, ordering or timing;
- deterministic IDs correlated with hidden identity;
- producer/evaluator shared mutable state;
- hidden truth accidentally passed through an adapter;
- missing frames or missing calibration promoted to successful evidence;
- predicted/simulated/counterfactual state serialized as directly observed;
- modified input, model, split or output digests;
- test-set results used during threshold or fusion selection.

A leakage/manipulation-check failure invalidates the affected benchmark result; it is not a model-performance failure.

## 6. Metrics and eligibility

### Representation diagnostics

Report frozen nearest-neighbor retrieval / correspondence and, where labels justify it, linear-probe performance. Include feature stability under declared transformations, object-size strata, long-tail identity strata, and feature bandwidth/memory. These are diagnostics; linear decodability alone does not prove that cognition uses the feature causally.

### Tracking and geometry

The currently exported Patch-HDC tracker state includes tracker-local identity and patch-grid centroid/velocity semantics; it must not be treated as a qualified bounding-box or mask tracker.

- Until outputs meet the input contract, report raw tracking measures supported by the existing export: association decisions, identity switches, reacquisition frames and false persistence after removal.
- Use [TrackEval](https://github.com/JonathonLuiten/TrackEval), the reference implementation used by several tracking benchmarks, for HOTA / DetA / AssA / LocA only after supported detection/track geometry is available.
- Report IDF1 or other metrics only when their required ground truth and output semantics are satisfied. Never relabel centroid metrics as HOTA, IDF1 or MOTA.
- Keep point/centroid tracking and box/mask tracking results in separate tables.

### Beliefs, prediction and task utility

- Report explicit unknown/abstention behavior and error-versus-coverage curves.
- Compute Brier score, log loss or expected calibration error only for outputs with a defined probability semantics and a declared calibration procedure. Similarity and raw confidence scores are not automatically probabilities.
- Report prediction error by horizon and event type.
- For active perception, score realized information gain after a fresh admitted observation, and compare with random-view and surprise-only controls.
- Keep hidden-truth task success in the evaluator. Symthaea receives sensor evidence and its own outputs only.

### Resource boundary

Report p50/p95 end-to-end latency, throughput, peak RAM/VRAM and feature/memory footprint. Include preprocessing, host/device transfer, aggregation, temporal update, and tracking costs in the chosen end-to-end boundary. Also report encoder-only latency separately.

Energy per correct task outcome requires an actual calibrated power-measurement boundary covering the declared system. If no meter was used, set energy status to `NOT_MEASURED`; CPU/GPU time or FLOPs are not energy measurements.

## 7. Preregistered research targets

These are proposed experiment targets, not industry standards or achieved results.

### Qualification

- 100% agreement on independently specified known-answer fixtures.
- Zero false accepts for declared leakage, malformed identity/frame, missing-evidence and synthetic-as-physical negatives.
- Deterministic byte-stable canonical reports for identical pinned inputs.
- Exact commit, model/weight digest, configuration, corpus/split digest, toolchain, hardware, resource boundary and evidence class recorded for every run.

### Performance

After an exploratory pilot and a pre-run power/variance analysis, freeze the primary endpoint and analysis plan. Initial target for a candidate hybrid:

- At least **+5 absolute HOTA points** over the strongest eligible baseline on at least two preregistered occlusion/OOD strata, if and only if the output geometry qualifies for HOTA;
- paired 95% confidence intervals exclude zero for the primary comparison;
- no more than a 2-point absolute regression in the predeclared overall primary tracking score;
- all strata and negative results disclosed.

If HOTA is ineligible because the output is centroid-only, do not substitute a hand-built proxy and call it HOTA. Select an appropriate preregistered persistence/association endpoint instead and set its minimum meaningful effect from pilot variance.

### Efficiency

Target at least **25% lower measured energy per correct task outcome** versus the learned-only arm while remaining within a predeclared 2-point absolute HOTA non-inferiority margin. This target applies only on a calibrated instrumented system. Before that exists, evaluate the latency/memory Pareto frontier without claiming energy savings.

Statistical unit should be the independent scenario/seed cluster, not every highly correlated frame. Use paired comparisons, report intervals and all prespecified strata, and avoid choosing the best seed or most favorable subset after seeing the results.

## 8. Machine-readable run record

A future versioned result schema should bind, at minimum:

- schema ID/version;
- experiment-plan digest and corpus/split digest;
- exact source commit and working-tree cleanliness;
- arm ID and canonical encoder/model/weight/preprocessing identities;
- training data and compute budget, if any;
- seed set and scenario/stratum counts;
- evaluator version and benchmark-validity status;
- metrics plus denominators, exclusions, uncertainty intervals and per-stratum results;
- hardware/toolchain/runtime/resource-boundary metadata;
- latency/memory and instrumented power artifacts where applicable;
- raw output artifact digests and claim ceiling.

Statuses must distinguish `NOT_RUN`, `NOT_EVALUATED`, `FAILED`, `PASS_FOR_EXACT_SUBJECT` and other explicitly defined outcomes. Missing data is never coerced to zero or success.

Evidence classes remain distinct: `ANALYTIC_FIXTURE`, `SYNTHETIC_EVALUATION`, `OFFLINE_REAL_VIDEO`, `PHYSICAL_SENSOR_MEASUREMENT`, and `REAL_ROBOT_TASK`. No class upgrades another automatically.

## 9. Execution order

1. Review this protocol against VIS-002, VIS-EVAL-000A and the Symtropy anti-shortcut gate.
2. Freeze the first 48-case manifest and independent oracle before comparing candidate outputs.
3. Implement small, dependency-light Rust schema/report/reference tests; do not add foundation-model weights or heavy runtimes to the production workspace.
4. Validate the supported export semantics. Add HOTA/IDF1 only when emitted geometry and the ground truth fully support them.
5. Run analytic fixtures and leakage negatives locally; collect exact-head execution evidence when available. Queued CI is neither pass nor fail.
6. Add offline dense-feature ingestion and compare A/C/D with all provenance bindings intact.
7. Publish the complete resource/quality frontier and decide whether learned-feature, hybrid, active-vision, or memory changes merit a follow-up implementation.

## 10. Research anchors

- DINOv3 official repository / paper: https://github.com/facebookresearch/dinov3 and https://ai.meta.com/research/publications/dinov3/
- SAM 2: https://ai.meta.com/research/sam2/
- TAO open-world tracking: https://taodataset.org/
- TrackEval reference HOTA and tracking metrics: https://github.com/JonathonLuiten/TrackEval
- WorldArena evaluates video perceptual quality and downstream embodied utility: https://github.com/tsinghua-fib-lab/WorldArena
- WorldArena 2.0 extends evaluation to visuo-tactile settings, online RL environments and simulated / real robot platforms: https://github.com/WorldArena2/WorldArena-2.0

These references define comparison points only. No Symthaea result is implied by a paper's published score or by adding a baseline adapter.
