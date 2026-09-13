# symthaea-matbench-gap-exposure-plan

Deterministic exposure planning for the exact SHA-gated Matbench `matbench_expt_gap` composition universe.

The output is a **truth-free screening-universe partition**, not a truth slice and not a clean-holdout certificate.

## Why this is separate

Benchmark Zero must not make ranking code reopen truth-bearing structures merely to learn which candidate identities are admissible.

This crate therefore establishes a narrower planning boundary first:

```text
exact official compressed Matbench bytes
        ↓ SHA-gated #1722 parser
ordered normalized composition universe
        +
exact exposed Symthaea training snapshot
        ↓ composition-only comparison
complete Retained / ExcludedExposedTraining partition
        ↓
truth-free retained screening universe
```

Experimental gap values are never copied into the plan.

## Public authority path

The public official constructor accepts compressed bytes and invokes `symthaea-matbench-gap::parse_official_matbench_expt_gap` first. That parent parser requires the exact pinned compressed SHA-256 before decompression.

An arbitrary caller-created `MatbenchGapDataset` therefore cannot enter the public official-plan path.

`verify_official_exposure_plan(...)` separately requires:

1. structural plan validity;
2. the same current Symthaea training snapshot;
3. successful replay of the official SHA-gated Matbench parser; and
4. exact reproduced plan equality.

Structural `plan.validate()` is intentionally not source authentication.

## Complete partition instead of exclusions only

Every source row becomes exactly one `PlannedComposition` in canonical source order:

- source row index;
- candidate id;
- normalized composition fingerprint;
- disposition: `Retained` or `ExcludedExposedTraining { labels }`.

Validation requires the vector to cover `0..source_row_count` exactly once and requires candidate ids to be unique.

This closes an important architectural leak: a downstream screening method can consume `retained_rows()` / `retained_candidate_ids()` without touching `BandgapTruthSet` merely to reconstruct the candidate universe.

## Exact training-snapshot identity

The plan records `symthaea_training_snapshot_sha256`, a domain-separated digest of the exact exposed `symthaea-bandgap` training table used when the partition was derived.

`validate_against_current_training_snapshot()` distinguishes a stale plan from malformed plan structure. A changed training table is therefore not silently interpreted under an old exclusion decision.

This digest is provenance. The retain/exclude decision itself still depends on normalized composition overlap, not on benchmark truth values.

## Separate identities

The plan keeps several identities deliberately distinct:

- `source_compressed_sha256`: exact official compressed artifact bytes;
- `source_row_order_sha256`: parent adapter's full normalized rows, including experimental values;
- `source_composition_order_sha256`: ordered source-row/candidate/composition identity only;
- `partition_sha256`: complete retain/exclude partition;
- `retained_universe_sha256`: retained source-row/candidate/composition tuples only;
- `exclusion_set_sha256`: excluded rows plus the exposed-training labels that caused exclusion;
- `symthaea_training_snapshot_sha256`: exact exposed training-table snapshot.

A regression changes all fixture benchmark gap values while preserving composition/order. It requires:

```text
complete partition             unchanged
composition-order digest       unchanged
training-snapshot digest       unchanged
retained-universe digest       unchanged
exclusion-set digest           unchanged
compressed-artifact digest     changed
full row-order digest           changed
```

That is the noninterference theorem: benchmark truth values can change exact artifact identity, but cannot alter the composition-only screening-universe decision.

## Truth-label firewall

The serialized plan contains no `experimental_gap_ev` field and no benchmark target values.

The intended lifecycle is:

```text
verify official artifact
→ freeze truth-free candidate universe
→ rank without truth
→ freeze ranking/decisions
→ evaluate later against separately held truth
```

## What this still does not prove

Removing exact exposed-training composition collisions does **not** establish a globally clean holdout. It does not rule out:

- historical/manual knowledge used to write baseline equations;
- publication familiarity or prior experiments;
- external pretrained-model exposure;
- unpublished local data;
- near-neighbor/family leakage with different compositions;
- target-informed hand tuning;
- undeclared data sources.

The prior-knowledge disclosure is fixed and validation rejects attempts to relabel this plan as a clean-holdout certificate.

## Next scientific gate

Once the official compressed artifact is available and its pinned SHA verifies:

1. execute and replay-verify this plan;
2. record the training snapshot, partition and retained-universe digests;
3. independently review the exact exclusions;
4. preregister any additional leakage policy without reading benchmark truth labels;
5. freeze the final admissible screening universe;
6. only then construct a separate truth slice and run Benchmark Zero.

## Qualification status

Source/tests are authored only. This stack inherits #1722's unresolved exact-toolchain `Cargo.lock` / `flate2` boundary. Do not claim format/check/test/Clippy success until a pinned execution lane runs the exact head.
