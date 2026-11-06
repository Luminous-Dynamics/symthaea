# ConnectomeBench2 provenance, task eligibility and denominator audit V1

**Tracking:** #10828 — SCI-CONNECTOMICS-095
**Status:** source-card metadata audit and synthetic known-answer fixture; not a full dataset reproduction.
**Sources checked:** 7 November 2026.

## 1. Why this audit is necessary

ConnectomeBench2 is the leading predictive proofreading baseline for this programme. It must be reproduced on the exact paper-current data revision and task routing, not on a convenient but stale materialization.

The currently visible Hugging Face `main` README says that `main` is the initially submitted V1 and that revision `v2` is the revision used by the current arXiv version of the paper. Therefore, `main` is useful for discovering the dataset-card contract and identifying a V1 inconsistency, but it is not a substitute for pinning and evaluating V2.

Sources:
- Paper/project page: https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/ and arXiv https://arxiv.org/abs/2606.21116
- Dataset card V1/main: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2/blob/main/README.md
- Dataset revision page: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2

## 2. Denominators are not interchangeable

The paper/project abstract reports **more than 716,485 expert-labelled proofreading decisions** and **more than 4.5 million associated images** across mouse, human, zebrafish and fly. The V1/main dataset card reports **401,170 samples/rows** and a set of row-level train/validation/test counts. These are different named units and may represent different versions, aggregation levels or view expansion. Do not divide, equate, or use one as the denominator for the other until the source authors' counting rules and exact V2 files have been reconciled.

The V1/main card gives 319,727 train rows, 43,517 validation rows and 37,926 test rows. These sum exactly to 401,170.

However, the five displayed V1/main sample-type counts are:

| Sample type | Card rows |
|---|---:|
| adjacent_control | 121,333 |
| junction_control | 38,272 |
| synapse_control | 18,182 |
| merge_edit | 146,461 |
| split_edit | 77,213 |
| **Arithmetic sum** | **401,461** |

The five values sum to 401,461, **291 more than the card's repeatedly stated 401,170 total**. The card itself defines five sample-type values, but the discrepancy's cause is unknown. Treat it as a source-card metadata inconsistency pending reconciliation against V2's sidecar metadata and shard-level records. Do not guess whether the cause is a typo, overlap, stale value, or revision mix.

The checked-in known-answer fixture records only the arithmetic over published V1/main card values. It is not a recount of all Parquet rows and does not assert that V2 shares the same discrepancy.

## 3. Exact revision and artifact custody

Before any paper-current score is reproduced:

1. Resolve the `v2` revision to an immutable full commit/object identity; the V1/main README's statement that `v2` is paper-current is not itself an immutable pin.
2. Download and verify the V2 README, `shards.csv`, all 506 Parquet shards, metadata sidecars, demo/figures needed for audit, and associated data manifests.
3. Record each shard path, byte size, row count and SHA-256, and compare the aggregate row count to the sidecar census.
4. Recompute per-sample-type, per-species, per-source-archive, per-split and per-modality counts from the actual V2 sidecars, then reconcile every sum and join cardinality.
5. Bind all results to exact artifact bytes and the exact benchmark-evaluation code/config revision.
6. Preserve upstream source provenance and access/license restrictions for MICrONS, FlyWire, H01 and zebrafish inputs.

**Current limitation:** this PR does not claim that the V2 revision's full commit SHA or all 506 shard digests have been independently obtained. The source-card `v2` reference is recorded as an explicit but unpinned revision until those bytes can be materialized and audited.

## 4. Task eligibility and leakage prevention

The card identifies `sample_type`, `has_single_mask`, `has_dual_mask`, `task_routing`, `has_em`, and the image-render fields as the row-level eligibility contract. A model evaluation must freeze its task and derive the valid population from these fields before feature construction.

| Task | Eligible sample types and masks | Correct geometry view |
|---|---|---|
| `false_split_correction` (merge correction) | `merge_edit`, `synapse_control`, `adjacent_control` AND `has_dual_mask` | dual-segment render |
| `false_merge_identification` | `split_edit`, `junction_control` AND `has_single_mask` | single-segment render |
| `split_mask_generation` | `split_edit` AND `has_single_mask` | single-segment input; corresponding dual render supplies target masks where documented |

EM controls:
- `em_xy`, `em_xz`, `em_yz` are distinct views; record modality presence and mask-channel processing.
- For single-segment tasks, merge segment A/B masks appropriately and zero the second mask channel.
- Never use `em_best` for a single-segment evaluation: its oblique view is chosen using both labels and can leak ground-truth information.
- Report EM-complete and geometry-only denominators separately. The V1/main card says EM is present on 251,499 of 401,170 rows, about 63% overall, but only about 38% of merge_edit and 23% of split_edit rows; controls have 100% EM. A multimodal score without a task-specific `has_em` denominator can be materially biased.
- Synthetic controls are not real human edit examples. Keep `adjacent_control`, `junction_control`, and `synapse_control` distinct from `merge_edit` and `split_edit` when reporting proofreading quality.

Spatial split:
- The card describes train/validation/test assignment by `interface_point_nm` within 50 µm cubes, rather than a random row split.
- Preserve cube identity, source volume and source archive ancestry. Never re-split individual rows or derived render views randomly across train and test.
- All preprocessing, feature selection, calibration, thresholding, model selection and active-learning decisions must be confined to the appropriate training/validation populations.

## 5. Scorecards must remain endpoint-specific

Do not create one scalar leaderboard combining incomparable tasks. Publish separate scorecards:

1. Split/merge proofreading: balanced accuracy, task-specific precision/recall, calibration, coverage and human-consensus uncertainty.
2. Split-mask generation: mask IoU/Dice and boundary/instance metrics, on eligible `split_edit` examples only.
3. Cross-species/region transfer: leave-one-species-out and spatial/source-held-out performance with exact eligible-row counts.
4. Modality ablations: geometry-only, EM-only, and multimodal on both native populations and explicitly coverage-matched subsets.
5. Query-critical alternative discovery (#10656): design-weighted alternative recall and query impact; this has a different denominator from classifier labels.
6. Expert workflow utility (#9242): blinded review quality, time, rework and cost; not a classifier score.
7. Global graph-invariant stopping: global convergence/cost forecast only; it is not a query-specific scientific certificate.

Candidate system comparison includes the ConnectomeBench2 multimodal ViT, adapted VLMs, NEURD and RoboEM, but only on their matched task, split, modality coverage and evidence tier. Do not infer classifier superiority from a query-assurance benchmark or compare computational annotation cost with expert label accuracy.

## 6. Acceptance and promotion

Before publishing a paper-current benchmark comparison:
- V2 exact commit and every consumed artifact hash are pinned.
- Paper-level decisions, image counts, V1 rows, V2 rows, valid task rows and model-level image/view counts are reported as different units.
- V2 per-task/per-species counts are recomputed from source metadata; no guessed/hardcoded totals.
- The 291 V1/main sample-type arithmetic mismatch is resolved against V1 source artifacts or explicitly remains an unresolved V1 card error; it is never silently copied into V2.
- Every excluded row has a machine-readable exclusion reason.
- 0 `em_best` leakage in single-segment split tasks.
- 0 synthetic control leakage into real-edit-only scorecards.
- 0 spatial-cube, source-archive or operation-lineage train/test leakage.
- Every reported model score states exact denominator and EM/geometry coverage.
- All upstream source licenses/access conditions remain preserved.

## 7. Claim ceiling

This audit can establish version identity, count reconciliation, task eligibility and benchmark split integrity. It cannot establish the correctness of the original expert labels, biological completeness, or a model's universal superiority. Synthetic fixture validation proves arithmetic/verifier behavior only; it is not a dataset reproduction.