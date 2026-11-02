# ConnectomeBench2 denominator reconciliation V1

**Status:** executable metadata-audit extension; no pinned V2 dataset run or biological result is claimed.

## Why three denominators are mandatory

ConnectomeBench2 materials report quantities at different expansion levels. The dataset card reports 401,170 rows and approximately 2.2 million model-level samples (approximately 2.8 million when dual and single geometry are counted separately). The later VLM study reports task-operation counts. These numbers should not be compared as though they count the same unit.

This protocol fixes the units before any leaderboard or reviewer-efficiency comparison.

### R — unique dataset rows

A physical metadata-sidecar row identified by the upstream compatibility identifier `combined_sample_hash`. The auditor verifies its exact upstream derivation and rejects duplicate IDs. The count is the dataset-row denominator, not a count of downstream tasks.

### O — task-operation instances

One operation identity expanded to one declared task route:

`(source_identity_sha256, task_route)`

A row routed to both `false_merge_identification` and `split_mask_generation` produces two task-operation instances but remains one dataset row. Every task route must be validated against the upstream sample-type/mask rules. Count per route and total; total must equal the sum of route counts.

### V — available legal modality-view instances

One task-operation identity expanded to one available modality:

`(source_identity_sha256, task_route, modality)`

The metadata auditor counts task-appropriate geometry (`geometry` for false-split correction; `geometry_single` for false-merge identification and split-mask generation) and each available cardinal EM plane (`em_xy`, `em_xz`, `em_yz`). The current conservative denominator excludes `em_best` for every route and reports it separately; a later task-specific view policy may change this only with an explicit leakage review and versioned policy.

These are **available modality records**, not model forward passes, augmented examples, multimodal combinations, statistically independent samples, or a replication of the paper's model-level denominator. That requires the exact input contract and training/view expansion logic.

## Identity policy

The upstream source composes:

`combined_sample_hash = MD5(source_archive + "|" + source_archive_sample_hash)`

The MD5 value remains the compatibility key. It is not used as the source of cryptographic provenance. The audit additionally computes:

`SHA256("connectomebench2-source-identity-v1" || u64be(len(archive)) || archive || u64be(len(source_hash)) || source_hash)`

The source-hash input is currently a 12-hex value. The composite SHA-256 binds the two declared fields with unambiguous length-prefixing, but it does not increase the source hash's entropy or independently validate the upstream sample identity. The auditor explicitly detects one legacy MD5 identifier mapping to distinct source tuples and continues to reject duplicate combined identifiers.

## Required exact reconciliation

For a frozen V2 revision and sidecar digest:

- record (R), distinct source identities, and duplicate/collision findings;
- record (O), `task_operation_instances_by_route`, and require `O = sum(route counts)`;
- record (V), per-route/per-view counts, and require `V = sum(route-view counts)`;
- separately retain row-level split, species, sample-type, modality and spatial-lineage denominators;
- preserve the source-side natural prevalence and rare subgroup support without reweighting the primary test;
- reconcile task expansion by unique source identity × task route, not by assuming paper task-operation totals equal unique Parquet rows;
- reconcile model-view expansion only from the paper's exact model input contract, not from this metadata-sidecar availability count.

No value is imputed when a modality or identity field is absent. Missing task routes are not inferred from populated label columns. Empty/rare strata remain explicitly not estimable.

## Adversarial tests and acceptance targets

The metadata auditor adds deterministic fixtures for:

- one row routed to multiple task operations;
- the same modality being counted once per task-operation, not once per row;
- `em_best` being excluded from the conservative legal-view count;
- row/task/view denominator arithmetic;
- stable and length-delimited source identity SHA-256;
- the legacy MD5 compatibility ID and collision detection.

Promotion targets (not yet achieved on ConnectomeBench2 V2):

- exact deterministic replay under arbitrary sidecar row order;
- 100% equality between task-operation total and route-count sum;
- 100% equality between view-instance total and route/view-count sum;
- zero duplicate source identity / legacy-ID collision findings in the frozen eligible benchmark cohort;
- exact row and task-operation reconciliation to the relevant paper table or an explicit documented explanation of the different population/expansion rule;
- no SOTA or reviewer-productivity claim until the exact V2 SHA, shard digests, task expansion and view policy are pinned.

## Relationship to the model comparison

A useful ConnectomeBench2 comparison has separate arms and matched operation IDs:

- full multimodal mesh+EM specialist ViT-B / ViT-L;
- EM-only ViT-B comparator reported by the later VLM study;
- VLM zero-shot, few-shot and LoRA variants;
- geometry-only / EM-only ablations on the exact same eligible sample IDs;
- uncertainty, entropy, expected decision-loss VOI, and random review policies at the same reviewer-minute budgets.

Do not mix the full mesh+EM and EM-only ViT-B scores into one baseline. Do not treat row/view expansion counts as effective sample size. Report clustered intervals at the proper dependence level (source operation, cube/volume, neuron/session where available) and disclose insufficient support in rare strata.

## Sources inspected

- ConnectomeBench2 V2 dataset card: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2
- ConnectomeBench2 upstream code (label, routing, split and sidecar implementation): https://github.com/timfarkas/ConnectomeBench2/tree/main
- ConnectomeBench2 V2 VLM comparison paper: https://arxiv.org/abs/2609.36492
- ConnectomeBench2 benchmark paper: https://arxiv.org/abs/2606.21116

The dataset card explicitly instructs users to load `revision="v2"`; `main` is the initial V1. The exact immutable V2 commit SHA remains unresolved in this work, and the metadata audit must not run against an assumed `main` snapshot.
