# VIS-DVPC-005: Paired Real-Trace Admission Contract v1

**Status:** Draft contract and schema-only validator.  
**Issue:** [#8490](https://github.com/Luminous-Dynamics/symthaea/issues/8490)  
**Date:** 2026-10-23  
**Parent:** [VIS-DVPC-001 #8156](https://github.com/Luminous-Dynamics/symthaea/issues/8156)

## Objective

Prepare a path from synthetic DVPC mechanisms to reproducible real-trace evaluation without committing raw sensor data or leaking evaluator ground truth into an online policy. This is a dataset-neutral admission layer, not a dataset downloader, perception model, planner, or performance result.

Motivation: the September 2026 DEEP paper evaluates the decision value of escalating perception computation using KITTI and nuScenes perception outputs and nuPlan open-loop planner outcomes, with allocator overhead charged against resource budgets: https://arxiv.org/abs/2609.35910. The proposed contract borrows the experimental discipline (paired cheap/full outputs, downstream utility, actual resource cost) but makes no reproduction claim.

## Files

Each admitted experiment has two UTF-8, tab-separated files and an external provenance receipt:

1. **Observation file** — consumed by the online policy. Contains pre-escalation features and digests, declared resource estimates, the estimated DVPC, uncertainty and an optional producer-visible shift-signal digest.
2. **Evaluation file** — sealed from online policy selection. Contains paired cheap/full normalized utilities, safety violation counts, measured incremental latency/energy and evaluator-only shift labels.
3. **Receipt** — generated outside the validator and bound to the exact contract version, both input file byte digests, source revision, evaluator version and command. The focused workflow demonstrates file hashing with `sha256sum`.

No raw media, dataset labels or derived high-volume sensor artifacts belong in this repository. The validator checks digest *syntax and linkage*; it does not download source data or independently recompute dataset-snapshot hashes.

## Observation TSV schema

The first records are `META\tkey\tvalue` rows in the same order-independent set in both files, followed by the exact header and `OBS` rows. Unknown, missing or duplicate metadata keys fail closed.

Required metadata keys:

| Key | Meaning |
|---|---|
| `format` | Exact value `VIS-DVPC-TRACE-001-v1` |
| `dataset_id`, `dataset_revision` | Dataset identity and immutable release/revision token |
| `dataset_snapshot_sha256` | SHA-256 digest identifying the source snapshot/inventory |
| `source_manifest_sha256` | SHA-256 of the frozen source manifest |
| `split_manifest_sha256` | SHA-256 of the frozen group-level split manifest |
| `utility_spec_sha256` | SHA-256 of the normalized task utility definition |
| `license_id`, `license_scope` | Source license identifier plus declared permitted-use class |
| `higher_is_better` | Must be the literal `true` |

The exact observation header is:

```text
OBS_HEADER  trace_id  dataset_id  dataset_revision  split  group_id  step_index  sample_sha256  cheap_output_sha256  pre_features_sha256  utility_spec_sha256  declared_extra_latency_us  declared_extra_energy_uj  estimated_dvpc  reported_uncertainty  shift_signal_sha256
```

Fields are TAB-separated, not space-separated. SHA-256 fields must be 64 hexadecimal characters; optional shift signal uses `-` only when no observable signal exists. Trace/group IDs are tokens rather than filesystem paths. Split is one of `train`, `validation`, `test`, or `private_test`. Declared incremental latency and energy are strictly positive integer estimates, in microseconds and microjoules respectively. DVPC estimate is finite and may be negative; uncertainty is finite and non-negative.

The observation schema intentionally does not permit ground-truth shift labels, future features, cheap/full evaluator utilities, realized execution costs, or oracle decisions. Adding an undocumented column invalidates the file instead of being silently ignored.

## Evaluator TSV schema

The exact evaluation header is:

```text
EVAL_HEADER  trace_id  sample_sha256  cheap_utility_01  full_utility_01  cheap_safety_violations  full_safety_violations  actual_extra_latency_us  actual_extra_energy_uj  shift_label  shift_mechanism
```

- Utility is finite in `[0,1]`, normalized by the frozen utility specification, with higher being better.
- Safety violation counts are non-negative integer counts, not scores.
- Realized incremental latency and energy must be positive integers.
- Shift label is `in_domain`, `shifted` or `unknown`. Mechanism is one of `none`, `illumination`, `weather`, `occlusion`, `sensor`, `density`, `other`, or `unknown`. An `in_domain` row requires `none`; a `shifted` row requires a specific mechanism.
- The evaluator file has a strict one-to-one join on `trace_id`; it repeats `sample_sha256`, which must match the observation file exactly. Duplicate, extra, missing and mismatched pairs fail closed.

For a committed escalation choice, per-record decision value is computed by the evaluator as `full_utility_01 - cheap_utility_01`. The producer must never access this label before its action is committed. Safety outcomes remain a separate axis: a utility gain cannot cancel a safety regression in a single weighted score.

## Leakage and grouping invariants

- Each `trace_id` is unique; each `(dataset_id, revision, group_id)` belongs to exactly one split.
- `step_index` starts at zero and is contiguous within a group; duplicate indices fail closed.
- Frames within a group are not independent statistical replicates. Report paired deltas at the predeclared group/sequence level and use grouped split boundaries.
- Both paired files carry exactly identical metadata. The utility-spec digest on every observation must match the metadata.
- No missing-pair deletion without a failure/count receipt. Unsupported, skipped, invalid and over-budget rows remain visible in the downstream evaluator.
- This first validator admits exactly one dataset/revision and one utility profile per paired file set. Cross-dataset comparisons must be normalized and summarized by task/family after independently admitted source manifests.

## Resource accounting

Declared extra latency/energy are pre-escalation estimates available to the policy. Actual incremental latency/energy reside in the evaluator file and are used for score accounting. Allocator execution overhead must be recorded as its own measured channel in the separate run receipt; it must not be omitted from a latency/energy budget.

Latency and energy are separate constraints with explicit units. Do not add microseconds and microjoules or silently convert them to a single score. Any policy choice exceeding a declared budget is retained as a violation and excluded from a valid budget-relative regret claim; it must not be discarded from the incident/violation count.

## Dataset rights and availability boundary

The format permits metadata for different dataset licensing classes but does not grant permission to use or redistribute any dataset. Current source rules matter:
- nuScenes states the dataset is for non-commercial use and directs commercial users to request a separate license: https://www.nuscenes.org/data-format
- Waymo Open Dataset's terms define the provided dataset license as non-commercial and prohibit its use in production systems and for primarily commercial purposes: https://waymo.com/open/terms/
- KITTI maintains its own task-specific benchmark and submission process; its benchmark page notes that its servers were restored to an August 7, 2026 backup on September 7, 2026, so results in the affected interval require resubmission: https://www.cvlibs.net/datasets/kitti/eval_object.php
- nuPlan offers open- and closed-loop evaluation modes; those outcomes are not interchangeable: https://www.nuplan.org/

`license_scope` currently admits `synthetic_public_fixture`, `noncommercial_research_only`, or `commercial_use_authorized` as provenance assertions. The validator checks that an explicit class was written, not that the claim is legally correct. A real dataset adapter must complete the relevant license and data-access review before use. Do not put raw media, access tokens, or restricted labels in Git.

## Validator behavior and claim boundary

Run:

```sh
rustc --edition=2024 --test tools/vision-dvpc-trace-validator.rs -o /tmp/vis-dvpc-005-tests
/tmp/vis-dvpc-005-tests
rustc --edition=2024 -O tools/vision-dvpc-trace-validator.rs -o /tmp/vis-dvpc-005
/tmp/vis-dvpc-005 fixtures/vision-dvpc-observations.tsv fixtures/vision-dvpc-evaluations.tsv
```

The fixture is entirely synthetic and carries placeholder-format hashes. The validator is standard-library-only, fails closed on schema/join mistakes, and emits `PASS_SCHEMA_ONLY`; that label is deliberately not a performance claim.

Admission does not establish that a perception model is good, that an alarm is calibrated, that the loss function reflects safety, that a dataset license allows a proposed activity, or that any allocator beats random/full/learned baselines. Those require a frozen, independently reviewable dataset adapter, eligible rights, actual measured outputs, task-balanced scoring and exact-head execution receipts.

## Next stage

After exact-head qualification, add one dataset adapter at a time. Start by producing cheap/full perception output digests and downstream losses on a frozen validation subset, using only a licensed local copy. Use validation data for policy development, reserve the test split for a frozen confirmatory comparison, include allocator overhead, and compare cheap-only, full-only, random, simple score routers, conservative LCB, and the candidate policy. Do not publish a leaderboard claim until all baselines share the same source snapshot, evaluator, budgets and hardware/accounting boundary.
