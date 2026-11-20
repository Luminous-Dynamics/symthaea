# CDL-Bench: shared-dictionary independent telemetry groups (PZG3)

**Date:** 2026-11-20  
**Tracking:** [CDL-Bench-016 #12983](https://github.com/Luminous-Dynamics/symthaea/issues/12983), [CDL-Bench-014 #12964](https://github.com/Luminous-Dynamics/symthaea/issues/12964)  
**Status:** local C++20 PZG3 frame reference; synthetic fixtures; not production or SOTA evidence

## Hypothesis: decouple entropy context from decode granularity

PZG2 grouped series into one independently compressed Zstandard frame. Larger groups recovered some cross-series compression but forced unrelated series to be decompressed together. PZG3 keeps independently compressed groups but supplies the same trained Zstandard dictionary to every group.

This separates:
- **compression context:** one dictionary shared by all groups;
- **decompression context:** one independently checksummed group;
- **query context:** a directory maps a series key to its group and child byte range.

The dictionary is trained from a separate historical synthetic corpus, not the target fixture.

## PZG3 frame and validation

PZG3 is a self-contained binary frame:
- versioned `PZG3` magic, row/key/group counts, schema and Zstandard level;
- dictionary length and CRC-32, followed by the embedded dictionary;
- separately compressed directory with raw/compressed checksums;
- each group is an independent Zstandard frame compressed with the shared dictionary;
- each group has raw length, compressed length, raw CRC and compressed CRC;
- each child has entity/series identity, row count, offset, length and child CRC.

The decoder checks header, dictionary, directory, group and child integrity; validates offsets and row totals; supports selective key decode; and full-decodes groups before restoring global timestamp order. A corrupted group is rejected without invalidating an unrelated intact group. There is no implicit dictionary substitution.

Local compiler/test evidence:
- GCC 14.2.0, C++20, strict warnings with `-Werror` (excluding only the existing compact-source misleading-indentation warning), libzstd 1.5.7.
- Zstandard level 3.
- Empty frame; group sizes 1, 4, 8, 16, 64; full exact round-trip; selective key decode; group corruption isolation; dictionary CRC; header CRC; truncation all passed.

## Evaluation design

Dictionary training data: 4 synthetic workloads × 64 series, generated with seed `0x7e1e0000`. The evaluation fixture uses seed `0x7e1e0001`. This is out-of-seed evaluation, but the same synthetic generator families are used in train and test; it is not a blind public-data holdout.

Dictionary capacities tested: 1, 4, 8 and 16 KiB. Group sizes: 1, 4, 8, 16 and 64. Compression level: 3. Complete PZG3 bytes include the dictionary embedded in each frame.

## Selected cold-frame results

| Workload | Group size | Dictionary | PZG2 no-dictionary | PZG3 embedded dictionary | Difference |
|---|---:|---:|---:|---:|---:|
| Sparse | 1 | 1 KiB | 83,667 B | 79,120 B | -5.43% |
| Bursty | 1 | 4 KiB | 244,335 B | 241,023 B | -1.36% |
| Mostly idle | 8 | 4 KiB | 6,192 B | 9,798 B | +58.24% |
| Noisy | 8 | 16 KiB | 2,443,911 B | 2,459,312 B | +0.63% |

A shared dictionary is not universally beneficial. Sparse data and bursty data can benefit; small idle telemetry is highly sensitive to dictionary overhead; noisy data sees essentially no cold-frame gain.

## Dictionary-catalog amortization: separate scenario, not current PZG3 bytes

The experiment also calculates what happens if one dictionary is held once in an external content-addressed registry and reused across 16 objects. This amortized column subtracts the embedded dictionary and charges one-sixteenth of its byte size, retaining an 8-byte per-frame dictionary reference allowance.

Illustrative strongest amortized outcomes:
- Sparse, group size 1, 16 KiB dictionary: 77,259 B vs PZG2 83,667 B (-7.66%).
- Bursty, group size 1, 8 KiB dictionary: 233,518 B vs 244,335 B (-4.43%).
- Mostly idle, group size 4, 4 KiB dictionary: 6,227 B vs 6,533 B (-4.68%).
- Noisy, group size 1, 16 KiB dictionary: 2,495,813 B vs 2,509,825 B (-0.56%).

**Important:** these amortized values are a storage-accounting scenario for a future external dictionary catalog; they are not the size of the current self-contained PZG3 frame. The registry/reference format, content-addressed lookup, authentication, missing-dictionary behavior and cold/warm cache costs are not implemented yet. A standalone PZG3 file embeds the dictionary and pays its full cold cost.

## Conclusions and design implications

1. Shared dictionaries can recover compression context across independent groups without forcing all groups into one decompression unit.
2. Dictionary overhead can overwhelm small/idle data. Dictionary size must be selected from actual complete-frame cost, not compression ratio alone.
3. A future adaptive container should compare dictionary and no-dictionary options, with fail-closed dictionary IDs and digests.
4. External dictionary catalogs can amortize a reusable model over multiple objects, but introduce lifecycle, availability, cache and trust requirements.
5. PZG3 remains a synthetic C++ reference. No Rust validation, public-data results, encode/decode throughput, training CPU or cache-latency measurements are claimed.

## Reproduction

- `research/cdl-bench/reference/pzg3_shared_dictionary_reference.cpp`
- `scripts/research/cdl_pzg3_shared_dictionary_reference.sh`

Run `ZSTD_LEVEL=3 scripts/research/cdl_pzg3_shared_dictionary_reference.sh`. The C++ reference embeds the dictionary and decoder; it does not depend on CI or Rust toolchain availability.

## Next gates

- Per-group dictionary/no-dictionary fallback with full header/index/dictionary bytes counted.
- External dictionary registry: content ID + cryptographic digest, versioning, cache and fail-closed missing-ID behavior.
- Dictionary drift experiments across new workload families, schema revisions and service topology changes.
- Measure dictionary training time, decode CPU, peak RSS, dictionary-cache hit rate and cold/warm query latency.
- Reproduce on LO2v2 and Alibaba only after exact dataset downloads, checksums, schemas and split manifests are frozen.
