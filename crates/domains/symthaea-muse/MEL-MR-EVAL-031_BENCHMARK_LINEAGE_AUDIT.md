# MEL-MR-EVAL-031 — benchmark lineage isolation and replay audit

**Status:** Rust preflight auditor implemented; source-level tests authored; no compile, test, dataset run, or benchmark score claimed yet.

## Research hypothesis

Paired-target evaluation is only meaningful when train, validation and test are independent at the level of the underlying recording/composition. A random sample split can leak a source composition through alternate renders, stems, MIDI exports, different edit targets, or round-tripped audio. This leakage can inflate apparent generalization without the editor learning a transferable operation.

The split should therefore be treated as a provenance graph rather than a collection of independent filenames.

## Implementation

Module: `src/benchmark_lineage.rs`

Public interface:

- `BenchmarkLineageManifest`: benchmark ID, immutable dataset revision, and manifest rows.
- `BenchmarkLineageRecord`: sample ID, split, operation family, source/target audio SHA-256, instruction SHA-256, optional source score/MIDI SHA-256, and required composition-lineage SHA-256.
- `audit_benchmark_lineage`: validates identities and checks cross-split reuse.
- `BenchmarkLineageAudit::is_clean`: clean only when there are no validation or leakage violations.
- `verify_benchmark_lineage_audit`: recomputes the manifest and audit; any altered count, row, violation, identity, or digest fails equality.

All fingerprints must be lowercase 64-character hexadecimal SHA-256 values. Invalid records remain committed in the manifest digest and are reported; they are not silently filtered out.

## Independent leakage surfaces

| Identity domain | Check |
|---|---|
| Audio assets | Source and target audio fingerprints share one namespace. Any fingerprint appearing in more than one split is rejected, including source-in-train reused as target-in-test. |
| Symbolic source | Optional MIDI/score fingerprint must not appear in multiple splits. |
| Composition lineage | Required lineage fingerprint must not appear in multiple splits, even when the audio renders and MIDI exports differ. |
| Sample identity | Duplicate sample IDs are rejected, including duplicates within one split. |

Repeated edits of one source are allowed within the same split. The benchmark may legitimately pair one source with multiple instructions or targets; those examples must remain grouped in the same partition.

Instruction fingerprints are required and committed, but are not used as split keys: generic instruction templates can legitimately recur across independent compositions.

## Determinism and replay

Manifest hashing is domain-separated and length-prefixed. Rows are canonically sorted by their complete stable record fields, so input row order does not change the manifest commitment. Audit hashing commits schema version, benchmark and dataset revision, counts, manifest digest, and the ordered violation set.

Replay verification recomputes all of those fields from the original manifest. It is not enough to recompute an audit's own digest: the verifier independently re-runs the checks and requires exact equality.

## Adversarial test matrix

Source tests cover:

1. Multiple edits using the same source within one split are permitted.
2. A source audio asset reused as another split's target is rejected.
3. A symbolic MIDI/score fingerprint crossing splits is rejected.
4. One composition lineage represented by different audio renders across splits is rejected.
5. Duplicate IDs and malformed required digests are reported.
6. Row ordering does not affect either commitment.
7. A tampered audit count fails replay verification.

Additional campaign controls should include alternate encodings, stems, remasters, time-stretched copies, transposed MIDI exports, and source/target role swaps. Exact SHA-256 alone will not detect perceptually or structurally equivalent content with different bytes.

## EVAL-030 integration

Run the audit before any model/editor executes and before calculating SMEBench or MMAE metrics. Do not use the test set to choose a mapping, prompt, threshold, DSP parameter, or model route.

For each benchmark report:

- benchmark ID and dataset revision;
- manifest SHA-256 and audit SHA-256;
- total / accepted / rejected sample counts;
- violations by class;
- source-lineage clustered confidence intervals;
- symbolic MIDI-derived and real-audio strata separately;
- operation-level scores and abstentions.

Use a source-lineage cluster—not a render, crop, stem or edit request—as the resampling unit for uncertainty estimates.

## Further qualification required

The implementation verifies the supplied declarations; it cannot prove the declarations are complete or truthful. Before publication, pair it with:

- curator-provided source-family mappings and split provenance;
- approximate nearest-neighbour screening over audio embeddings and symbolic fingerprints;
- spot-check listening and waveform/spectral comparison of nearest cross-split neighbours;
- a signed, immutable dataset release manifest;
- a frozen evaluator/version/settings record.

A lineage audit is necessary but not sufficient to establish external validity or SOTA.

## Validation boundary

The Rust module and seven tests are authored. The current execution environment does not have `cargo`, `rustc`, `rustfmt`, or the benchmark datasets, so compilation, execution, and dataset qualification remain unverified. Do not treat the PR as passing CI or as a measured benchmark improvement.
