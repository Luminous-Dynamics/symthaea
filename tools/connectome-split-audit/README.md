# connectome-split-audit

A dependency-free Rust auditor for group-disjoint train/validation/test splits and task-view leakage in connectomic proofreading benchmarks.

## Why random splits are not enough

ConnectomeBench2 V1 documents:
- `combined_sample_hash` as a 32-character MD5 hex primary key;
- `source_operation_id` and before/after root IDs in metadata;
- train/validation/test assignment by spatial location using randomly allocated 50 μm cubes;
- task routing where one sample may serve more than one task;
- `em_best` as an oblique view chosen using both masks, with an explicit warning against using it for single-segment false-merge identification.

These are important split and evidence dependencies. A high score on row-random splits is not a substitute for held-out source volume, operation lineage, neuron/root lineage, spatial cube, animal/session, archive, species or workflow.

The V1 card documents these facts; the exact V2 dataset identity and group-field availability must still be resolved before a V2 benchmark is claimed.

## Audit unit

The primary unit is a task-routed operation:
- one `combined_sample_hash` may legitimately produce distinct operations for different task routes;
- all task operations derived from one sample must remain in the same split;
- the same (sample hash, task route) pair may not occur twice;
- every view instance inherits the sample hash, split and task route from its parent operation;
- every view ID must be unique in the supplied manifest.

A view-instance ID should identify the model input instance (operation + modality/view), not only the underlying raw image bytes.

## Group identities

The policy declares which identities must be present and disjoint across splits:

- `SourceArchive`: upstream source archive/reconstruction grouping;
- `SourceVolume`: acquisition volume identity;
- `SourceOperation`: original upstream proofreading event identity, qualified by its source archive if the upstream ID is only locally unique;
- `SpatialCube`: stable 50 μm cube ID from the pinned split metadata/derivation, qualified by source volume;
- `Neuron`: union of relevant before/after neuron root IDs;
- `Animal`, `Session`, `Species`, `Workflow`: additional holdout dimensions.

For every required group key, a missing or empty identity fails closed. When one sample hash has multiple task routes, every supplied source-group field must be identical across those task records; a task expansion cannot silently rewrite volume, operation, cube, neuron or other provenance. Do not invent placeholders, infer IDs from unrelated fields or conflate null with a shared group. The adapter must preserve exact source IDs and pin its group-derivation code/configuration.

For neuron holdouts, populate `neuron_ids` with the deduplicated union of all relevant before/after root IDs from the source metadata. For spatial-cube holdouts, use the actual split-cube identity from the source split manifest or a separately documented, pinned derivation. Do not guess a cube origin from coordinates.

## Task-view contract

- The allowed views are passed in through the frozen policy rather than assumed globally.
- `EmBest` is hard-rejected for `FalseMergeIdentification`, even if a permission entry mistakenly includes it.
- Every view must inherit exactly the parent operation's sample hash, split and task route.
- The audit checks manifest consistency; it cannot prove the underlying image bytes correspond to the view ID or label. The ingestion adapter must hash and verify actual bytes.

## Output semantics

- `GroupDisjoint`: no overlap was observed for the policy's selected group keys.
- `LeakageDetected`: the report contains each overlapping (group key, value) and all affected splits. This is a failed audit; a training pipeline must refuse to score the manifest as group-disjoint.
- Missing required group values, duplicate identities, malformed sample/artifact hashes, duplicated operation/view IDs, cross-split sample assignment or task/view mismatch return an error.

A clean result proves only disjointness of the selected keys in the supplied manifest. It does not prove group IDs are biologically correct, exhaustive or resistant to source-provenance laundering.

## Synthetic fixture

- Input schema: `research/connectomics/connectome-split-audit-input-v0.1.schema.json`
- Known-answer cases: `research/connectomics/fixtures/connectome-split-audit-known-answer-v0.1.json`

The fixture covers:
- a clean leave-volume/neuron-out split;
- source-volume overlap;
- source-operation overlap;
- spatial-cube overlap;
- the same sample hash crossing splits under two different task routes;
- inconsistent source-group metadata across two task routes for the same sample;
- missing required group identity, invalid `em_best` task view and duplicate task operation.

The sample hashes and artifact hashes are synthetic test values, not real dataset digests.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-split-audit/Cargo.toml
```

Twenty-five Rust unit tests are authored for clean splits, volume and neuron overlap, multiple root IDs, source-operation and spatial-cube overlap, identical group provenance across task routes (with order-independent root-ID sets), duplicate root rejection, missing group values, cross-split sample leakage, duplicate operation/view identity, task-view permission, inherited sample/split/task view identity, `em_best` rejection, malformed hashes, stable output ordering and deterministic counts. They have **not** been compiled or executed in the current authoring environment.

## Benchmark ladder

Run the same frozen model/checkpoint across:
1. row-random split (diagnostic only);
2. held-out neuron/root lineage;
3. held-out spatial cube;
4. held-out source volume;
5. held-out source operation;
6. held-out animal/session;
7. leave-one-source-archive-out;
8. leave-one-species-out;
9. leave-one-workflow/protocol-out;
10. prospective temporal edit holdout.

Not every dataset supports every regime. If required metadata are unavailable, report `NOT_AUDITABLE` and exact missing keys rather than manufacturing a proxy regime.

All views and repeated crops inherit their operation's split. Calibration/adaptation groups must be disjoint from test groups. Few-shot label budgets are reported separately and may not draw from test groups.

See [SCI-CONNECTOMICS-058](https://github.com/Luminous-Dynamics/symthaea/issues/10023).