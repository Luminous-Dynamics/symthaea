# CDL-Bench: typed telemetry records and joint segmentation research note

**Date:** 2026-11-19  
**Status:** Research design and benchmark specification; no codec result is claimed  
**Scope:** Proposed CDL-Bench-009. This note does not change the existing F64C/F64S format or production Symthaea behavior.

## Executive finding

The current exact-f64 track is a useful bit-preserving transform laboratory, but it is not yet a production-telemetry benchmark. It represents one ordered vector of binary64 values. Its RLE transform groups adjacent identical scalar u64 patterns and then XOR-codes one value per run.

That must not be described as reproducing record-level RLE. The Dynatrace SIGMOD 2026 Industry Track study analyzes application-monitoring telemetry whose records often remain unchanged for long periods; it combines record-level RLE with XOR codecs and a block-level selector. The authors report approximately 23% compression-ratio improvement and 34% speed improvement over standalone methods, and more than 13% improvement over the best single RLE-enhanced compressor for their selector. These are the paper's measurements, not CDL-Bench results.

The next research increment should therefore move from a scalar stream to a typed, multi-field record stream. It should test repetition at three distinct levels: identical scalar values, repeated field vectors within one series, and repeated complete records across time. It should also make segmentation a first-class variable rather than fixing chunk lengths to 64/256/1024.

## What is already implemented—and what it does not establish

The current PR #12828 branch contains four exact-bit f64 transforms (raw, XOR-varint, reusable XOR window, scalar RLE+XOR), two byte entropy models, and fixed-chunk eight-candidate selectors (offline best-of-eight, full historical and sparse-probe). All frame headers, checksums and child frames are counted. These are meaningful controls.

They do not yet establish:
- record-level RLE across typed multi-field telemetry rows;
- preservation and compression of series/entity identity, timestamps, nullability and schema;
- cross-field or cross-series redundancy;
- variable-length segmentation jointly selected with a codec;
- matched performance against a reproduced ALP implementation;
- results on the production telemetry distribution studied by Dynatrace.

The offline best-of-eight selector remains an upper oracle over its fixed chunk boundaries, not an online or adaptive-segmentation result.

## New hypothesis: typed records expose a different redundancy surface

Represent a telemetry observation as a canonical record with a stable schema, for example:

`(timestamp, entity_id, series_id, cpu_f64, memory_f64, request_rate_f64, status_code, missing_bitmap)`

The exact schema is dataset-specific. The format must preserve:
- integer timestamps exactly, including irregular deltas and duplicate timestamps;
- entity and series identity (dictionary IDs are permitted only when dictionary bytes and ordering are encoded);
- field types, nulls, NaN payloads, signed zero and all IEEE-754 bits;
- original record order, or an explicit reversible permutation if sorting is tested;
- missingness separately from NaN and any application-specific sentinel values.

Do not silently assume that records are regularly sampled, that equal values mean equal records, or that records from different series are interchangeable.

### Three distinct run models

1. **Scalar run:** consecutive identical values in one typed field. For f64, equality is by exact u64 bit pattern, not floating-point equality.
2. **Field-vector run:** a run in which the selected value fields for one entity/series repeat, while timestamp and perhaps counters continue to advance. The repeat grammar must explicitly encode which fields are repeated and how timestamps advance; irregular timestamps cannot be synthesized from an assumed cadence.
3. **Whole-record run:** consecutive canonical records are identical under an explicitly declared equivalence relation. If timestamp is excluded from the repeated payload, its delta stream is still encoded and charged.

Run-length coding is not universally beneficial. Every run form needs a literal escape and a break-even rule that includes run length, type/tag overhead, timestamp cost and checksums. Short runs should remain literals when that is cheaper.

### Candidate A: state-delta event coding for sparse multi-field updates

For each stable `(entity_id, series_id)`, maintain the last decoded typed record and encode the next observation as an event:

`(time_delta, series_id, changed_field_bitmap, presence/null operations, changed_values)`

This changes the unit of coding from “a row of values” to “a state transition”. When a metric record changes only two of twelve fields, the unchanged ten fields do not need to be emitted again. When no value field changes, encode only the timestamp/heartbeat semantics; aggregate identical state intervals only when the timestamp rule is explicit and valid.

Important invariants:
- A zero bit in `changed_field_bitmap` means “same value as prior state”, not null.
- Null-to-value, value-to-null and null-to-NaN are explicit operations.
- Compare f64 fields by exact u64 bits; signed zero and different NaN payloads are different values.
- Field order and types come from a versioned schema; schema and dictionaries are charged.
- Interleaved entities require per-key prior state. Bound the state table and use deterministic eviction/checkpoint rules; never assume the decoder has hidden state from an earlier session.
- Checkpoints must permit bounded recovery and random access. Count their bytes and measure replay depth.
- A reset/restart is a semantic event, not silently interpreted as “all fields unchanged”.

A state-delta arm should be compared with scalar RLE, field-vector RLE, whole-record RLE, per-column ALP/Pco, and the current fixed-chunk codecs. Report update density (changed fields / total fields), state-table hit rate, checkpoint bytes, eviction frequency, decode replay depth, and random-access latency. This is a research candidate, not an implementation claim or novelty claim by itself; sparse updates and delta coding are established ideas. The proposed research contribution would be the decoder-verifiable integration with typed schema, bounded state, adaptive segmentation and a fully charged compute/size budget.

## Independent synthetic state-delta reference (2026-11-19)

A standalone C++20 transform-level reference is checked in at `research/cdl-bench/reference/typed_record_delta_reference.cpp`. It maintains prior typed field state per 32 keys and encodes each row as timestamp delta/absolute marker, key, changed-field bitmap, explicit presence/null operation, and exact u64 bits for changed present fields. The first observation for a key is a full checkpoint. It compares against a simple raw record layout with an 8-byte timestamp, 4-byte key, 1-byte presence bitmap and eight 8-byte field slots.

The reference compiled with:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/typed_record_delta_reference.cpp \
  -o /tmp/typed_record_delta_reference
/tmp/typed_record_delta_reference
```

Observed transform-only results for four deterministic synthetic workloads (65,536 records each):

| Workload | Raw bytes | State-delta bytes | Reduction |
|---|---:|---:|---:|
| Sparse updates | 5,046,284 | 339,876 | 93.26% |
| Bursty updates | 5,046,284 | 549,257 | 89.12% |
| Noisy updates | 5,046,284 | 3,160,871 | 37.36% |
| Mostly idle | 5,046,284 | 269,400 | 94.66% |

The reference also round-trips each fixture exactly and rejects five malformed streams: truncated payload, truncated header, unknown timestamp mode, invalid presence marker and trailing byte.

A second, separately labelled diagnostic applied **Zstandard CLI 1.5.7 at level 19** to both byte representations, using identical Zstd settings and no trained dictionary. The state-delta stream is thus a transform followed by Zstd; the control is Zstd on the raw canonical record serialization:

| Workload | Zstd(raw records) | Zstd(state-delta stream) | Change vs Zstd(raw) |
|---|---:|---:|---:|
| Sparse updates | 197,774 B | 90,001 B | -54.49% |
| Bursty updates | 434,115 B | 296,363 B | -31.73% |
| Noisy updates | 2,842,524 B | 2,780,315 B | -2.19% |
| Mostly idle | 75,071 B | 3,834 B | -94.89% |

The run also decompressed both Zstd outputs and byte-compared them with their respective inputs. The reproducer is `scripts/research/cdl_state_delta_zstd_reference.sh`; it compiles the reference with strict warnings, generates temporary raw/delta fixtures, compresses both at the configured level (default 19), verifies exact Zstd decompression, and prints CSV. The source can emit the two binary inputs with `--emit-dir`.

This diagnostic suggests that the state-delta grammar can expose additional structure to a general-purpose backend, especially at low update density. It does **not** show that the grammar alone beats Zstd: on the sparse fixture the uncompressed delta stream (339,876 B) is larger than Zstd(raw) (197,774 B), while Zstd(delta) is smaller after applying the second stage. No throughput comparison was collected, and Zstd+delta is not yet a complete random-access or bounded-state codec.

**Evidence boundary:** these are synthetic C++ transform-only bytes, not a Rust run, entropy-coded frame sizes, production telemetry measurements or a comparison to ALP/Pco/Gorilla/Chimp/ACTF/Rabbit/DeXOR. The raw baseline is an explicit simple record serialization; no schema dictionary, checksum, random-access index, checkpoint schedule, state-table bound, or general-purpose codec is included. The exceptionally high reduction therefore demonstrates that sparse per-key updates create exploitable structure in this generator, not expected end-to-end production compression. The next implementation must add bounded per-key state, checkpoint/recovery and a complete framed stream before these numbers can be interpreted operationally.

## Independent bounded-state reference and Zstandard control (2026-11-19)

A second, stricter C++20 reference is checked in at `research/cdl-bench/reference/typed_record_delta_bounded_reference.cpp`. Unlike the earlier 32-key/no-cap prototype above, this implementation mirrors the current TRD1 algorithm's core state rules: stable `(entity, series)` keys, bounded per-key LRU state, deterministic tie-breaking, segment-start state resets, full checkpoints for first-seen/evicted keys, timestamp delta/absolute handling including regressions, explicit null/presence changes, canonical changed-field masks, U32/Boolean range checks, and exact u64 payloads.

The reference compiled and executed with:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/typed_record_delta_bounded_reference.cpp \
  -o /tmp/typed_record_delta_bounded_reference
/tmp/typed_record_delta_bounded_reference
```

Result: 200 randomized round-trip trials passed across varying key counts, interleaved series, state capacities (1–12), segment intervals (1–50), field changes and timestamp regressions. Six malformed inputs were rejected (truncated timestamp, noncanonical timestamp varint, invalid timestamp mode, truncated control bytes, truncated final field payload and trailing data). Four deterministic 65,536-record synthetic workloads also round-tripped exactly.

### Transform-only and Zstandard results

All four workloads use 64 keys and eight typed lanes. The canonical raw representation includes eight schema-kind bytes plus 81 bytes per record (timestamp, entity, series, presence bitmap and eight u64 field slots). TRD1 state-delta bytes are the uncompressed changed-field transform, before arithmetic coding and outer frame/checksum bytes.

| Workload | Raw bytes | State-delta bytes | Transform reduction |
|---|---:|---:|---:|
| Sparse updates | 5,308,424 | 647,997 | 87.79% |
| Bursty updates | 5,308,424 | 832,949 | 84.31% |
| Noisy updates | 5,308,424 | 3,768,429 | 29.01% |
| Mostly idle | 5,308,424 | 590,645 | 88.87% |

Zstandard CLI 1.5.7 at level 19 was then applied to both the raw canonical records and the TRD1 state-delta transform. Both compressed outputs decompressed and byte-compared exactly with their inputs:

| Workload | Zstd(raw) | Zstd(state-delta) | Change vs Zstd(raw) |
|---|---:|---:|---:|
| Sparse | 223,103 B | 84,743 B | -62.02% |
| Bursty | 367,194 B | 223,915 B | -39.02% |
| Noisy | 2,603,078 B | 2,316,026 B | -11.03% |
| Mostly idle | 153,234 B | 5,596 B | -96.35% |

These values supersede neither the earlier simple reference results nor any Rust benchmark; they are a **separate, expanded fixture set and grammar**. The reproduction runner is `scripts/research/cdl_typed_record_delta_bounded_reference.sh`; it compiles with strict warnings, runs the randomized/malformed tests, writes temporary raw/transform fixtures, compresses both with the same Zstd version/level and checks exact decompression.

**Evidence boundary:** this is an independent C++ algorithm reference and transform+Zstd diagnostic, not the Rust implementation, not arithmetic-coded TRD1 frame sizes, and not production telemetry or a matched ALP/Pco/Gorilla/Chimp/Rabbit/WZS comparison. The high gains are from synthetic low-update workloads; noisy data is intentionally included and still benefits in this generated workload, but that is not evidence of production distributional generalization. No throughput, peak RSS, actual BTreeMap allocation, checkpoint random-access index or full schema-registry costs were measured.

## WZS-inspired LZV segmentation control (2026-11-19)

A deterministic, prefix-only segmentation control is now checked in at `research/cdl-bench/reference/lzv_segmentation_reference.cpp`, with runner `scripts/research/cdl_lzv_segmentation_reference.sh`. It is deliberately labelled **WZS-inspired**, not WZS. The exposed WZS repository does not yet provide a complete executable reference.

The control computes the leading-zero count of each adjacent XOR residual, then compares exact integer mean and population-variance statistics between a recent fast window (8 observations) and slow window (16 observations). It uses a latched change alarm, a minimum segment length (32), maximum length (512), and a 32-observation cooldown after a detected change. The detector uses only the observed prefix; no future values, branch labels or oracle boundary markers are supplied.

Local validation used `g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror` and passed six tests:
- Fixed min/max bounds and deterministic constant-stream splits.
- Empty-stream behavior and deterministic repeatability.
- 64 stationary-random streams of 8,192 values each: zero extra boundaries beyond required max-length splits.
- 64 abrupt constant-to-random / random-to-constant transitions: mean detection delay 1.5 observations, maximum 2, for this particular synthetic construction.
- Recurring stable/random/stable regimes.
- A slow LZV drift trace, included as a diagnostic rather than a claim that the detector optimally tracks gradual changes.

Representative segmentation-only output (default 32/512 bounds):
- Constant, 2,048 values: 4 segments, exactly 512 each.
- Stationary random, 4,096 values: 8 segments, exactly 512 each.
- Stable/random/stable, 1,536 values: 4 segments; boundaries 0, 512, 544, 1,026, 1,536.
- A synthetic 4,097-value slow-LZV-drift control: 17 segments, mostly maximum-sized; this shows the current detector does not create extra boundaries for that particular gradual drift.

**Evidence boundary:** these are segmentation diagnostics, not compression results. No child codec was run on segments, no segment header/selector metadata was measured, and no comparison was made with full WZS. The detector's parameters were tuned on synthetic development fixtures and require blind evaluation against CUSUM, fixed windows, MDL segmentation and WZS when its full artifact becomes available. Also, zero alarms in 64 random seeds is not a production false-alarm-rate guarantee.

## Rabbit-derived predictive branch-mapping diagnostic (2026-11-19)

A separate flag-mapping reference now exists at `research/cdl-bench/reference/rabbit_branch_mapper_reference.cpp`, with runner `scripts/research/cdl_rabbit_branch_mapper_reference.sh`. It implements only the paper-described four-branch prefix map `{0,10,110,111}`, previous-block branch-frequency ranking, and decoder-synchronous mapping reconstruction. It uses default identity ranking for the first block, then ranks branches by the previous block's counts. Ties break by branch ID ascending (the paper does not specify tie handling).

It deliberately does **not** implement ACTF's XOR branch classifier/payloads or Rabbit's TOE. The output contains only abstract branch flags, so it is not a full Rabbit codec, not an end-to-end size comparison, and not a Rabbit reproduction. It exists to isolate and test the synchronization/flag-length hypothesis before implementing complex payload branches.

A proposed guarded control is also tested (our extension, not part of Rabbit): it bootstraps with fixed two-bit flags; then it uses the dynamic map only when the previous histogram satisfies Rabbit's predicted-gain condition (x_1 > x_3+x_4); from the third block onward, the last two branch histograms must additionally have total-variation distance (le 0.10). Encoder and decoder derive this decision from the prior decoded prefix without transmitting a mapping or mode label.

Strict C++20 compilation (`-O2 -Wall -Wextra -Wconversion -Werror`) and seven tests passed, including 100 randomized exact encode/decode trials, the four expected prefix lengths, branch-gain lemma checks, deterministic output and malformed stream rejection. The 1,000-symbol block is the paper's default; 250- and 500-symbol sub-blocks are additional diagnostics.

### Flag-only synthetic results

All results below are **flag bits only**, over 4,000 abstract branch symbols, with no ACTF payload, XOR residuals, TOE, frame/header bytes or runtime claims:

| Regime | Fixed 2-bit | Previous-block mapper | Guarded control | Same-block fixed-or-dynamic oracle |
|---|---:|---:|---:|---:|
| Stationary skew | 8,000 | 5,200 | 5,900 | 5,200 |
| Uniform | 8,000 | 9,000 | 8,000 | 8,000 |
| One-block shift, block 1,000 | 8,000 | 6,760 | 8,160 | 5,200 |
| One-block shift, block 500 | 8,000 | 5,880 | 7,800 | 5,200 |
| One-block shift, block 250 | 8,000 | 6,800 | 7,020 | 4,960 |
| Alternating dominant, block 1,000 | 8,000 | 9,880 | 8,860 | 5,200 |
| Alternating dominant, block 500 | 8,000 | 6,820 | 7,800 | 5,200 |
| Alternating dominant, block 250 | 8,000 | 6,410 | 6,850 | 4,960 |

The paper-style mapping captures the expected gain on a stable skewed distribution, but its one-block predictive lag creates real regressions under uniform and alternating branch traffic. The guarded control prevents the uniform regression and reduces the alternating penalty, but it still loses to fixed flags on the first block after a sudden distribution change (8,160 vs 8,000 bits at 1,000-symbol blocks). Smaller blocks reduce the duration of the stale mapping, but change the number of block transitions and therefore require full framing/compute measurements.

**Evidence boundary:** this isolates the prefix-flag mechanism only. It does not validate Rabbit's ACTF-derived branch classification, TOE bit-elision proof, full byte/bitstream framing, paper datasets, implementation, or headline compression/time gains. The same-block oracle is an offline upper bound over only two flag choices (fixed 2-bit vs dynamic 1/2/3/3), using current-block counts; it does not include payloads or side-information costs. The guarded variant is our hypothesis and has not yet been evaluated on blind or production-derived data.

## Order-preserving per-series partitioning with adaptive timestamp merge (2026-11-19)

The grouped-versus-round-robin RLE ablation showed that whole-record runs disappear when records from many series are interleaved. The new partitioned frame restores per-series locality without silently changing caller-visible row order.

Files:
- `research/cdl-bench/reference/typed_record_partitioned_runs_reference.cpp`
- `scripts/research/cdl_partitioned_telemetry_reference.sh`

### Format and order-restoration modes

The prototype builds a stable dictionary of (entity_id, series_id) keys and encodes one chronological child stream per key. Each child carries exact timestamp deltas, changed-field masks, explicit presence/null state, exact 64-bit field values and repeated-state heartbeat runs. The outer frame charges key IDs, record counts, child lengths, schema, order metadata and CRC32.

It now supports three decoder-synchronous order modes:

1. **Timestamp merge (mode 0):** when the original global stream has strictly increasing, unique timestamps, the decoder reconstructs original interleaving with a deterministic k-way min-heap merge of the per-series streams. No order schedule is transmitted. The decoder verifies strict monotonicity in every child and across merged output.
2. **Tie-template schedule (mode 1):** when timestamps are nondecreasing but duplicated, the decoder identifies each equal-timestamp group from the child streams. The schedule contains only the tie-group key order. A bounded 16-entry move-to-front dictionary reuses recurring key-order templates; a literal template is emitted on a miss. Single-record timestamp groups require no schedule entries.
3. **Full schedule (mode 2):** when timestamps regress, the frame carries the original bit-packed key-ID schedule for every record. The decoder replays it to restore exact arrival order.

This is not a lossy reorder: each mode is selected only when its conditions are satisfied, and duplicate-time and regression cases preserve the original sequence. Tie templates are verified against actual per-key pending-record counts for each timestamp; malformed templates that would consume the wrong multiplicity are rejected.

Mode 1 decoding uses a min-heap merge frontier. It pops all rows at the next timestamp, including multiple rows from one series at that time, and pushes each per-series successor before advancing to the next timestamp. This changes timestamp-group discovery from a full scan over all keys for every group to (O(N \\log K)) heap work over (N) records and (K) active series (plus the template-map work). Tie counts are tracked sparsely by key in ordered maps rather than allocating and clearing a full key-count vector for each timestamp. The selected template is then replayed over the per-series cursors and checked against the actual multiplicity map.

A separate schedule-primitive reference and runner validate the move-to-front grammar:
- `research/cdl-bench/reference/tie_group_order_template_reference.cpp`
- `scripts/research/cdl_tie_group_order_template_reference.sh`

The primitive reference compiled locally with GCC 14.2 and strict warnings (`-O2 -std=c++20 -Wall -Wextra -Wconversion -Werror`). It passed 1,000 randomized exact tie-group schedule round-trips and five malformed-input checks. For 64 repeated groups of eight keys, the template schedule used 67 bytes versus 192 bytes for fixed-width per-event IDs (65.10% reduction). For 20 alternating groups across 16 keys, the 16-entry dictionary used 36 bytes versus 160 bytes for fixed-width IDs (77.50% reduction).

**Evidence boundary:** the schedule primitive tests only the tie-schedule subgrammar. The integrated three-mode PRN1 implementation has now also been compiled and run locally with GCC 14.2 under `-O2 -std=c++20 -Wall -Wextra -Wconversion -Werror`. The integrated test suite passed empty input; four deterministic 65,536-record mode-0 fixtures; mode-1 duplicate-time groups with a repeated template; alternating two-template groups; mode-2 out-of-order fallback; a same-series multi-row tie-group case; 100 randomized interleaved round-trips; and three malformed-frame cases. The first integrated run exposed an incorrect test bound: the 65,536-row fixture has 1,024 tie groups (not 64), and the correct repeated-template schedule is 1,072 bytes / 8,576 bits. The assertion was corrected from 5,000 to 9,000 bits. The next decoder revision replaced per-group full-key scans with a heap frontier and sparse per-key multiplicity maps; the same strict GCC 14.2 command passed with the new same-series multi-row tie test. A temporary local source copy used formatting-only line breaks to avoid misleading-indentation diagnostics; the checked-in algorithm and corrected threshold were tested. This is C++ reference validation only, not Rust CDL or CI evidence.


## Per-series field-columnar hybrid (PCL1) — 2026-11-20

The row-state PRN1 frame now has a second independently decoded child layout, PCL1, to test the strongest lesson from tparquet/ALP/Pco: numeric fields should be exposed as independent typed lanes rather than always serialized as row events.

PCL1 is deliberately narrow: it only accepts globally strictly increasing, unique timestamps. Duplicate-time and out-of-order cases remain on the PRN1 three-mode path.

### Layout

For each (entity, series) child, PCL1 stores:
1. A row count, first absolute timestamp and positive per-series timestamp deltas.
2. For each fixed typed field:
   - an LSB-first presence bitmap across child rows;
   - an LSB-first changed bitmap across child rows;
   - a contiguous exact u64 value lane, with one value only for each changed-and-present row.
3. The outer frame stores schema, key dictionary, row counts, child lengths and CRC32. The decoder validates bitmap padding, canonical update bits, exact child consumption, timestamps, field semantics, key uniqueness, lengths and CRC before merging.

The encoder and decoder reconstruct original records exactly; PCL1 is not a lossy numerical transform. The current version compresses the complete PCL1 frame with the same Zstandard level as the row-state controls. Pco and ALP have **not** yet been integrated into the lane codec selector.

### Tests and local measurements

The C++20 prototype and tests are in:
- `research/cdl-bench/reference/typed_record_partitioned_runs_reference.cpp`
- `scripts/research/cdl_partitioned_telemetry_reference.sh`

Local GCC 14.2 execution passed:
- empty columnar frame;
- four deterministic 65,536-record sparse, bursty, idle and noisy fixtures;
- randomized typed-record round-trips at 1, 8, 64, 1,024 and 4,096 distinct keys;
- corruption checks for bad magic/checksum, truncation and trailing bytes;
- the existing PRN1 three-mode suite, including duplicate/out-of-order and same-series multi-row tie groups.

The first run caught a field-layout mismatch: the encoder placed exact values before bitmaps while the decoder consumed bitmaps before values. The encoder now stages each lane's values and emits the presence bitmap, changed bitmap, then values; the corrected implementation passed the suite. No byte or compression result from the failing run was retained.

All sizes below are complete raw/global-delta/row-partitioned/field-columnar frames before Zstd, then complete frames compressed with Zstd CLI 1.5.7 level 19. Every compressed frame was decompressed and byte-compared with its original input.

| Workload | Raw frame | Global delta | PRN1 row-partitioned | PCL1 field-columnar | Zstd(raw) | Zstd(global delta) | Zstd(PRN1) | Zstd(PCL1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse | 5,308,448 | 648,493 | 140,879 | 392,931 | 223,774 | 85,931 | 61,864 | 59,042 |
| Bursty | 5,308,448 | 833,813 | 320,171 | 585,723 | 367,607 | 224,396 | 204,921 | 188,405 |
| Mostly idle | 5,308,448 | 590,677 | 8,308 | 334,243 | 153,209 | 5,577 | 4,784 | 5,692 |
| Noisy | 5,308,448 | 3,769,213 | 3,636,264 | 3,570,899 | 2,603,680 | 2,316,397 | 2,252,910 | 2,042,284 |

Relative to the complete Zstd(PRN1) frames, PCL1 is:
- **4.56% smaller** on sparse updates;
- **8.06% smaller** on bursty updates;
- **18.98% larger** on mostly-idle data;
- **9.35% smaller** on noisy full-update data.

Relative to Zstd(global state-delta), PCL1 is 31.29% smaller on sparse, 16.04% smaller on bursty, 2.06% larger on idle, and 11.83% smaller on noisy. Relative to Zstd(raw), it is 73.62%, 48.75%, 96.28% and 21.56% smaller, respectively.

This is the first direct signal that the best representation depends on workload regime. PRN1's whole-record heartbeat runs dominate mostly-idle data; PCL1's per-field lanes do better for sparse, bursty and noisy updates. **A per-series/segment row-vs-columnar selector is now more compelling than a single universal child layout.** The choice must account for complete frame size and decode/query cost, not just a local field-lane ratio.

### Evidence boundary and next gates

This is synthetic C++ transform-plus-Zstd evidence, not Rust CDL, not tparquet/ALP/Pco, not production telemetry, and not a SOTA result. The current PCL1 fixed eight-field schema has only an outer CRC (no independent child CRC yet), has no random-access index or checkpointed column lane decode, materializes full input and output in memory, and has only been measured at Zstd level 19. Query projection, per-field ALP/Pco/Zstd choice, throughput, peak RSS, and independent public dataset performance remain unmeasured. PCL1 rejects duplicate and out-of-order global timestamps rather than falling back to tie scheduling.

The next engineering step is an adaptive **per-series/per-segment layout selector** that compares complete PRN1 row-state and PCL1 field-columnar candidates on a bounded sample, records the choice in the frame, and preserves the exact PRN1 order fallback for tied or regressing timestamps. Issue #12956 tracks this lane-layout work; issue #12935 tracks bounded-memory streaming, child indexes, checkpoints and corruption isolation.


### Adaptive per-series selection: raw-size versus Zstd-aware child cost (2026-11-20)

Two additional frames now expose the cost of adaptive layout selection:

- **PAX1 raw-child selector:** stores each child uncompressed and selects PRN1 row-state or PCL1 field-columnar based on the smaller raw child length. The complete PAX1 frame is then compressed once with Zstd.
- **PZC1 independent-child selector:** compresses both candidates with Zstd 1.5.7 level 19, selects the smaller compressed child, and stores that child as its own Zstd frame with layout tag, raw length, compressed length and raw child CRC32. The outer frame has its own CRC32. This makes series payloads independently decompressible, but resets the Zstd context for each series.

Both formats are in `typed_record_partitioned_runs_reference.cpp`. The integrated C++ suite now checks empty frames, four fixture round-trips, row/columnar choice differences between idle and noisy streams, PZC1 child checksums, and corrupted/truncated frames. The local compile used GCC 14.2, `-O2 -std=c++20 -Wall -Wextra -Wconversion -Werror`, and linked libzstd 1.5.7. Tests passed.

| Workload | PAX1 chosen children (row / columnar) | Zstd(PAX1 whole frame) | PZC1 independently compressed frame | Zstd(global delta) | Zstd(PRN1) | Zstd(PCL1) |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 64 / 0 | 61,887 B | 72,574 B | 85,931 B | 61,864 B | 59,042 B |
| Bursty | 64 / 0 | 204,915 B | 219,720 B | 224,396 B | 204,921 B | 188,405 B |
| Mostly idle | 64 / 0 | 4,782 B | 9,095 B | 5,577 B | 4,784 B | 5,692 B |
| Noisy | 0 / 64 | 2,042,502 B | 2,173,924 B | 2,316,397 B | 2,252,910 B | 2,042,284 B |

**PAX1 finding:** selecting the raw-smaller child is not enough. It selects row-state for sparse and bursty streams, even though whole-frame Zstd(PCL1) is 4.56% and 8.06% smaller than Zstd(PRN1) there. It picks columnar for noisy streams, where PCL1 is better. PAX1 therefore nearly reproduces PRN1 on these first three workloads and PCL1 on noisy, but does not discover the PCL1 sparse/bursty advantage.

**PZC1 finding:** using actual Zstd-compressed child size changes the selection: it chooses columnar children for sparse, bursty and noisy data, and row-state for mostly idle data. Its complete independent-child frame beats Zstd(global delta) by 15.54% (sparse), 2.08% (bursty) and 6.15% (noisy), but is 63.08% larger on idle. It is still 17.31% larger than Zstd(PRN1) on sparse, 7.23% larger on bursty and 90.11% larger on idle; PCL1 whole-frame Zstd remains smaller on sparse, bursty and noisy. The per-child reset tax is visible.

This creates a clear next design choice:
- **Whole-frame/shared-context coding** wins bytes by letting Zstd exploit redundancy across series and metadata.
- **Independent child frames** allow per-series decompression and can support selective reads, but pay frame/reset cost for each series.
- A future frame should tune compression granularity: groups of related series, shared dictionaries, segment-level independent frames, and optional independent checkpoints. Do not assume one frame per series is the optimum.

PAX1's heuristic only compares raw child size; PZC1 compares independently compressed child size. Neither is a complete online optimum after global compression, query cost, memory or indexing. The prototype remains full-memory C++ synthetic research; no Pco/ALP lane integration, no direct indexed query benchmark, no bounded-memory streaming and no Rust validation.

## Current Rust implementation status (compiler validation pending)

The PR branch now contains `research/cdl-bench/src/telemetry.rs`, exported from `lib.rs`, with a `cdl-bench telemetry` synthetic command. The first frame has a fixed eight-lane schema, entity/series keys, exact u64 field payloads, explicit null/presence and changed-field masks, canonical u64 varints, deterministic bounded LRU state, segment-start checkpoints, and header/inner/raw-stream CRCs. The shared arithmetic coder is used for the transformed stream. Eight Rust tests are authored for round-trip, determinism, empty streams, truncation, single-bit mutations, resource limits, invalid field values and malformed varints.

The current environment has no Rust toolchain (`rustc`, `cargo`, `rustfmt` unavailable), so these tests have **not** run and Rust compilation/formatting are **not verified**. The C++ reference exercises the corresponding transform logic independently but is not a substitute for compiling or testing the Rust source. The Rust frame is still a fixed eight-lane prototype; it does not yet include a general schema registry, random-access index, schema evolution, per-segment independent child frames, or production recovery protocol.

## LZV segmentation control (WZS-inspired, not WZS) — 2026-11-19

A deterministic prefix-only LZV segmentation reference is now checked in:

- C++20 reference: `research/cdl-bench/reference/lzv_segmentation_reference.cpp`
- Reproduction script: `scripts/research/cdl_lzv_segmentation_reference.sh`
- Rust module (authored; compiler validation pending): `research/cdl-bench/src/segmentation.rs`
- Public API: `segment_lzv(values: &[u64], config: LzvConfig)`

For each new bit pattern, the detector computes the leading-zero count of the exact XOR residual against the prior decoded value. It compares the most recent fast-window distribution with an older slow window using exact integer mean and population-variance comparisons. A pending-change latch preserves a detected shift until the minimum segment length is legal; a cooldown reduces repeated alarms; maximum segment length forces a boundary even without a detected change. Decisions depend only on the reconstructed prefix and fixed config—no boundary labels are transmitted.

The independent C++20 reference passed strict-warning compilation (`-O2 -std=c++20 -Wall -Wextra -Wconversion -Werror`) and four test groups:
- constant stream: only forced max-length boundaries;
- stationary random: 64 deterministic seeds × 4,096 values each, with no unnecessary early splits beyond the declared tolerance;
- abrupt stable-to-random shift: boundary within 32 observations after the transition;
- deterministic and empty-input behavior.

The checked-in runner is `scripts/research/cdl_lzv_segmentation_reference.sh`; it passes `bash -n` and compiles/runs the reference with strict warnings. This is a reproducible boundary-control smoke suite, not a broad false-alarm-rate estimate.

Observed example boundaries with default configuration (fast=8, slow=16, min=32, max=512, cooldown=32):
- constant 2,048 values: `0,512,1024,1536,2048`;
- stationary-random 4,096 values: `0,512,1024,1536,2048,2560,3072,3584,4096`;
- stable→random 2,048 values: `0,512,1024,1056,1568,2048`;
- stable→random→stable 1,536 values: `0,512,544,1026,1536`.

These results validate deterministic segmentation behaviour only. They do not measure compression size, false alarms across a broad distribution suite, throughput or random access, and do not establish WZS performance or reproduce the WZS paper. The C++ reference is independent of the Rust translation. Six Rust unit tests are authored but unrun because no Rust toolchain is available in this environment.

## New hypothesis: jointly choose segmentation and codec

The current fixed chunks make boundary placement exogenous. Production data can contain long idle periods, sharp bursts and mixed regimes that do not align with 64/256/1024 boundaries.

Add two distinct policies:

### A. Offline joint-segmentation oracle (research upper bound)

For each possible segment start, consider a declared set of legal segment lengths and the available transform/codec candidates. Let (C(i,j,k)) be the complete encoded size for rows ([i,j)) under candidate (k), including child frame, headers, dictionaries, checksums and any model side information. Dynamic programming can find the minimum total:

`DP[j] = min_{i,k}(DP[i] + C(i,j,k) + boundary_cost)`

Constrain maximum lookahead/segment length and publish the candidate set. The oracle must not be presented as streaming. Its purpose is to quantify how much fixed boundaries leave on the table and to set a transparent upper bound for a declared candidate family.

### B. Bounded-compute decoder-synchronous policy

Build a deployable policy that makes decisions using either:
- statistics of the already-decoded prefix and past candidate costs; or
- a cheap, deterministic feature vector computed from the current block, with the selected child codec/length written into its self-describing header.

The second is a block-local selector, not a causal or prequential predictor. Its selection features, configuration/model bytes and child metadata must be counted. After decoding, the decoder can recompute the same features from the reconstructed block and verify the declared choice. The policy must use integer/fixed-point decisions where cross-platform determinism matters.

Candidate features worth ablation:
- scalar and whole-record run-length histograms;
- fraction of zero XOR residuals;
- leading/trailing-zero span distribution;
- delta and delta-of-delta magnitude histograms;
- decimal-roundtrip / ALP eligibility rate;
- per-field entropy and sparsity;
- missing/null density;
- cross-field repetition and same-entity persistence;
- block-local regime-change statistics.

Do not add all features at once. Begin with a cheap feature-only classifier and compare it with full candidate probing on the same blocks. Evaluate both ratio and decision compute.

## Public data track

Use at least two public production-derived sources before any claim about production monitoring:

1. **Latest Alibaba microservice time/call-rate/replica dataset (preferred compact fixture).** Zenodo record 14245634 (v3, published 2024-11-29) describes a filtered subset of Alibaba runtime traces. The documented columns include two timestamps, microservice ID, container ID, replica count, microservice time and call rate. The record contains an 8.2 MB CSV with published MD5 `148f0b2c9fc14bf65a25ae8d7ef3786b`, and documents that ten of the original 24 parts were extracted and filtered. Verify the downloaded file against that digest, then freeze the parser/schema version and a SHA-256 digest in the local manifest: https://zenodo.org/records/14245634
2. **Earlier Alibaba microservice time/call-rate artifact (legacy fixture).** The older Zenodo record 8310376 v1 describes a different 180,000-row file with microservice times and call rates over a 30-second interval. The record itself points to a newer version, so do not treat its CSV as the latest v3 dataset or merge the two artifacts without a versioned manifest: https://zenodo.org/records/8310376
2. **Alibaba production microservice trace (broader service population).** The official clusterdata repository describes 2021 traces covering more than 20,000 microservices over 12 hours, including runtime metrics and call dependencies: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021

A scale-oriented third source is Google's 2019 Borg cluster trace. Its official documentation describes eight cells for May 2019 and CPU-usage histograms per five-minute period; the full release is roughly 2.4 TiB compressed and is accessed through BigQuery, so use a documented sampled query before attempting full-scale runs: https://github.com/google/cluster-data/blob/master/ClusterData2019.md

The Dynatrace application-monitoring dataset described in the SIGMOD 2026 Industry Track paper is the most directly relevant target, but the public publication page does not provide an open dataset download. Record it as a non-reproduced production reference unless the authors provide access. Do not fabricate a public replica.

For every imported dataset, freeze a manifest with URL/DOI, license, version, source hash, download date, schema, parser version, raw row count, filtered row count, dtypes, missing-value semantics, timestamp units/order and exact train/validation/test split. Preserve raw files and publish normalized-output hashes. No random row shuffling.

## Matched baseline matrix

### Mandatory floating-point / numeric baselines
- ALP (SIGMOD 2024) through its official implementation and/or Apache Parquet 2.14.0 / Arrow Rust 60.0.0 support.
- Gorilla, Chimp and Chimp128; Elf/ElfPlus and Patas where buildable.
- ACTF and Rabbit (2026) for the adaptive floating-point flag/significant-bit family.
- Pco/Pcodec for typed numerical sequences.
- Zstandard on raw bytes and, where fair, on the same normalized column/record representation.
- Current CDL-Bench scalar F64C/F64S transforms and selectors.

ALP is not a niche paper baseline anymore: Parquet format 2.14.0 includes ALP, and Arrow Rust 60.0.0 announced an ALP encoder/decoder. ALP should be run on decimal-like sensor and monitoring columns, with its exceptions and page metadata counted. Its own design is not expected to dominate arbitrary full-range IEEE-754 bit patterns.

Rabbit's 2026 paper reports an average 4.15% improvement over ACTF on 28 datasets; reproduce its implementation and data settings before adopting that number as a comparison. WZS (September 2026) reports adaptive segmentation and selection results across 43 datasets, but treat abstract-level numbers as literature claims until code/configuration/data can be reproduced.

### Required fairness rules
- Same raw records, schema, row/column order, block limits and lossless semantics.
- No dropping timestamps, identifiers, null bitmaps, exceptions or indexes from the numerator.
- Separate value-payload-only size from end-to-end file size; use end-to-end size for superiority claims.
- Report cold-start model/code/config costs and amortized costs at several stream lengths.
- Give all methods the same hardware, core/thread budget and memory ceiling; publish versions, flags and CPU details.
- Report random access and partial-decode costs; smaller whole-file output can be worse for query workloads.
- Keep exact-lossless and bounded-error tracks separate.

## Metrics and gates

### Correctness (hard gate)
- 100% byte/bit-exact reconstruction of all values, timestamps, IDs, nulls and schema.
- Corruption/truncation/overflow/noncanonical encoding rejection.
- Deterministic output for a frozen configuration across supported targets.
- Decoder output hash equals the normalized source hash.
- No labels, hidden change boundaries or test-derived parameters enter the bitstream/predictor.

### Compression and compute
Report per-dataset and macro aggregates:
- bits/value, bits/record and complete bytes/record;
- end-to-end ratio including schema/dictionaries/indexes/metadata;
- encode/decode throughput and p50/p95/p99 block latency;
- peak RSS, allocations, bytes copied and energy where measurable;
- segment count, run-length distribution, candidate switches and selection regret;
- number of candidate encodes and decoder re-encodes;
- cold-start and amortized break-even length;
- online adaptation observations/bits after regime changes.

### Suggested targets (not achieved results)
- First gate: complete-frame size within 2% of the offline joint-segmentation oracle at no more than two extra candidate encodes per emitted segment on the declared mixed telemetry suite.
- Research win: at least 5% lower end-to-end size than the strongest reproduced matched baseline on at least two independent dataset families, with a paired 95% interval excluding zero and no exactness failures.
- Stability: no more than 1% median size regression against the best fixed policy on stationary/incompressible controls.
- Operational trade-off: publish the full size/throughput/memory Pareto frontier; do not hide an expensive oracle behind a headline ratio.
- Production-readiness: include bounded memory, streaming behavior, corruption recovery, version negotiation and independent decoder validation.

These are research targets, not results or promises.

## Paradigm-shift opportunity and novelty boundary

Do not claim novelty for RLE+XOR, ALP, adaptive block choice, online expert mixing or adaptive segmentation individually; these are all prior art. The credible research question is whether a **schema-aware, record-structured codec portfolio with a decoder-verifiable, compute-budgeted joint segmentation policy** can outperform specialized per-column codecs on real telemetry without sacrificing exactness, streaming constraints or queryability.

A stronger follow-on is mechanism-aware multi-series prediction: encode reusable cross-field/service mechanisms and their innovations, but retain per-field noncausal baselines and interventions as negative controls. A mechanism that only wins observationally but fails after service restarts, autoscaling or configuration changes is not a useful compressor.

## References

- Dynatrace SIGMOD 2026 Industry Track: https://www.dynatrace.com/research/publications/analyzing-and-improving-floating-point-compression-for-application-monitoring-time-series/ and https://doi.org/10.1145/3788853.3803097
- ALP SIGMOD 2024: https://github.com/cwida/ALP and https://2024.sigmod.org/toc-1-4.html
- Apache Parquet ALP adoption (2026-09-22): https://parquet.apache.org/blog/2026/09/22/alp-adaptive-lossless-floating-point-encoding-in-apache-parquet/
- Arrow Rust 60.0.0 ALP support (2026-09-29): https://arrow.apache.org/blog/2026/09/29/arrow-rs-60.0.0/
- Rabbit (Symmetry 2026, 18(4), 558): https://doi.org/10.3390/sym18040558
- WZS (Information Processing & Management, September 2026): https://doi.org/10.1016/j.ipm.2026.104697
- Alibaba microservice dataset: https://zenodo.org/records/8310376
- Alibaba production microservice trace: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021
- Google Borg 2019 trace: https://github.com/google/cluster-data/blob/master/ClusterData2019.md
