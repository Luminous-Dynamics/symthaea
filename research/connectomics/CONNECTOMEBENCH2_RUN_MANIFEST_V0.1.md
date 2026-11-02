# ConnectomeBench2 immutable run manifest V0.1

**Status:** schema and denominator audit fixture authored; exact V2 source pin and raw count reconciliation are not yet complete.

## Purpose

The ConnectomeBench2 benchmark must distinguish three different units:

1. **Unique dataset rows** in the Hugging Face Parquet dataset.
2. **Task-routed operation instances** after expanding a row to each task in its task-routing field.
3. **Model-view instances** after expanding a routed operation across geometry and EM views.

Never compare or sum these units without an explicit expansion key and reconciliation artifact.

The V2 dataset-card pointer says to load revision `v2`; `main` is explicitly identified as the initial V1. The run manifest therefore requires the literal requested revision plus an immutable resolved commit SHA before a result may be called a V2 reproduction.

## Current documented count audit

The currently rendered Hugging Face card reports:

- 401,170 unique dataset rows.
- Train / validation / test row counts: 319,727 / 43,517 / 37,926 (sum 401,170).
- Sample-type counts: adjacent_control 121,333; junction_control 38,272; synapse_control 18,182; merge_edit 146,461; split_edit 77,213.
- The five displayed sample-type counts sum to 401,461, which is 291 more than the stated total.
- 251,499 rows with EM views and 401,170 with geometry.

The September 29, 2026 VLM paper separately reports 531,734 train, 82,822 validation and 101,929 test **task-operation instances**, totalling 716,485. These are not assumed to be unique Parquet rows. The benchmark must materialize task_routing and reconcile operation instances by unique combined_sample_hash × task before comparing those numbers.

This fixture deliberately records the row-count mismatch and labels these counts UNPINNED_RENDERED_CARD. They are values visible on the default Hugging Face page, which explicitly warns that main is V1; they are not asserted to be V2 counts. The V2 revision SHA and its own metadata counts remain unresolved. Revision/digests/task expansion are unmaterialized. This is a fail-closed audit fixture, not a pinned dataset release or completed reproduction.

## Artifact identity

Before scoring, freeze:
- requested revision (v2) and resolved immutable commit SHA;
- exact dataset-card digest;
- shards.csv digest and declared shard count;
- source-file manifest digest with every Parquet path, SHA-256, byte size and n_samples;
- metadata sidecar digests for train, validation and test;
- upstream license review by source archive/species.

No whole-dataset mirroring is required for the first audit. Start with small metadata and shard manifests, then a stratified smoke subset. The repository is approximately 121 GB; license is listed as `other`, with upstream constraints requiring separate review.

## Task and view policy

Task routing is the source of truth:
- false_split_correction: merge-correction task; input requires dual masks.
- false_merge_identification: merge-error identification; input requires single-segment support. Do not use em_best because the oblique view is selected using both segment masks and can leak post-edit information.
- split_mask_generation: a pixel-mask output task with its own supervision and metric; do not score it as a binary proofreading decision.

The September VLM paper used four EM views for split-error correction and dropped the oblique view for merge-error identification. Record that as a comparator-specific view policy, not as a universal image preprocessing rule.

## Benchmark requirements

Use the manifest with:
- a row-level count and a task-operation count;
- per-species, per-task and per-modality breakdowns;
- immutable V2 baseline/model checkpoint and preprocessing digests;
- the ViT-B/ViT-L baseline and VLM zero-shot / four-shot / LoRA comparators;
- operation/volume/species-aware holdouts;
- calibration, positive/negative controls, selective risk, abstention, reviewer cost and hardware/latency;
- exact task-specific view policy, including em_best exclusions.

Do not call the VLM paper's full natural-mix test, its balanced mouse/fly subset, and its unseen-species transfer split equivalent. Preserve the paper's small human and zebrafish merge-error support counts rather than pooling them away.

## Files

- `connectomebench2-run-manifest-v0.1.schema.json`: strict Draft 2020-12 run-manifest schema.
- `connectomebench2-denominator-audit-known-answer.json`: documented count arithmetic and explicit unresolved V2 pin.

## Acceptance

The benchmark may become `PINNED` only when:
1. v2 resolves to an immutable SHA;
2. all artifact digests are present;
3. row and sample-type totals reconcile to the Parquet shards;
4. task operation counts reconcile to unique source rows and task-routing expansion;
5. the VLM study's external operation counts are either reconciled or explicitly explained as a different unit;
6. license/source review is complete;
7. leakage-safe task/view policies and spatial split rules are frozen.

No SOTA comparison is valid while the source identity or denominators remain unresolved.

## References

- Dataset card: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2
- V2 VLM baseline study: https://arxiv.org/abs/2609.36492
- ConnectomeBench2 paper: https://arxiv.org/abs/2606.21116
- Metric calibration: https://doi.org/10.1038/s41587-026-03307-w
- OOD perturbation benchmark (preprint): https://doi.org/10.64898/2026.10.16.759604
