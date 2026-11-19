# CDL-Bench exact IEEE-754 f64 reference track v1

**Date:** 2026-11-19  
**Status:** Rust reference source, four reversible transforms, three eight-candidate segment policies and authored tests committed; independent C++/Python transform checks exercised; Rust compilation and arithmetic-coded size results pending.

## Why this track exists

The SCM/i16 lane does not represent the dominant specialized lossless floating-point time-series competition. By 2026, several purpose-built codecs target bit patterns, decimal prefixes, oscillation, branch flags and segment-local encoding choices. A credible SOTA comparison needs a typed IEEE-754 track; it must not infer that success on synthetic SCM i16 streams transfers to f64 sensor streams.

The branch adds correctness references and selector baselines, not a competitive codec:
- `research/cdl-bench/src/float64.rs`: typed `F64C` frame with raw bits, XOR-previous u64 varints, reusable XOR-leading/trailing windows, and exact-bit-pattern RLE+XOR run coding; `F64S` segment container with offline best-of-eight, full historical-cost and sparse-probe decoder-synchronous policies; inner CDL arithmetic coding, CRC-32 checksums and output limits.
- `research/cdl-bench/src/main.rs`: `float --values N --seed N` experiment command comparing single-stream, offline segment-portfolio, full historical-cost and sparse-probe policies across plateau/idle regimes as well as smooth, oscillating and random streams; it records oracle gap and probe compute.
- `research/cdl-bench/src/lib.rs`: public exports.
- `research/cdl-bench/reference/float64_window_reference.cpp`: independent C++20 bit-window transform reference.
- `research/cdl-bench/reference/float64_rle_xor_reference.cpp`: independent C++20 RLE+XOR reference and malformed-input checks.
- `research/cdl-bench/README.md`: usage and limits.
- Tracking: [CDL-Bench-007 #12874](https://github.com/Luminous-Dynamics/symthaea/issues/12874).

## Exactness contract

Values are converted only with `f64::to_bits` and `f64::from_bits`. The current transforms are:

| Transform | Definition | Purpose |
|---|---|---|
| Raw | Original u64 bit pattern, little-endian | Exact control |
| XOR previous | `bits_t XOR bits_(t-1)`, first previous = 0, canonical u64 varint | Simple temporal bit-pattern baseline |
| XOR leading/trailing window | XOR residual; define a new leading/trailing significant-bit window only when needed, otherwise reuse the prior window and write only its payload | Simple reversible Gorilla-style bit-window reference; not a Gorilla/Chimp implementation |
| RLE + XOR previous | Merge identical u64 bit patterns into runs; encode each run length and XOR residual of the run value against the previous run value | Transparent record-level RLE plus predictive-value baseline |

No float subtraction, rounding, decimal parsing, NaN canonicalization, or normalization occurs. Tests compare each reconstructed `to_bits()` result with the input. Edge fixtures include:
- positive and negative zero;
- positive and negative infinity;
- quiet NaNs with distinct payload/sign bits;
- minimum subnormal and minimum normal;
- repeated constants, smooth bit increments, sign-alternating/oscillating patterns;
- deterministic random 64-bit patterns.

The F64C frame has a 32-byte header: magic/version, transform ID, inner model ID, element count, inner frame length, CRC-32 of original raw u64 bytes, and CRC-32 of inner frame. The decoder applies the element-count limit before allocating and caps transformed output to twenty bytes per f64 for RLE+XOR (a ten-byte run length plus a ten-byte XOR residual); XOR-leading/trailing requires at most eleven bytes.

The u64 varint parser rejects truncation, overlong encodings and values exceeding 64 bits. The window transform uses markers 0/1/2 for zero residual, reuse prior window, or define new window. Reused residuals must fit the stored window; new windows must have canonical endpoint bits and valid spans. The RLE decoder rejects zero/overlong runs, runs beyond the output count, adjacent duplicate runs and trailing bytes. All transforms must reproduce every IEEE-754 bit exactly.

## Current CLI

From `research/cdl-bench`:

```sh
cargo run --release -- float --values 10000 --seed 42
```

The command emits frame lengths and encode/decode nanoseconds for the single-stream arms, the offline segment portfolio and the historical-cost online selector at fixed chunk sizes 64, 256 and 1,024, across:
- constant and smooth-ramp;
- oscillating/sign-alternating;
- high-precision decimal-like;
- long plateau steps;
- mostly-idle telemetry-like streams with long exact repeats;
- alternating positive/negative zero runs;
- repeated distinct NaN payloads, infinities and zero runs;
- random-bit-pattern streams.

Each stream is evaluated with raw, XOR-previous-varint, XOR-leading/trailing window, and RLE+XOR transforms under order-0 and order-1 entropy models. The segment portfolio tries all eight combinations per chunk and picks the smallest complete child F64C frame. The F64S container header, every F64C child header, child CRCs, and the inner CDL payloads are all counted in the reported frame size. Zstandard is measured on the original raw little-endian f64 bytes if installed. The CLI does not yet invoke DeXOR, ACTF, Gorilla, Chimp, Rabbit or Dolphin.

**Selection caveat:** the offline selector has the entire chunk available and tries all eight encodings before choosing one. This is an offline size oracle, not an online selector and not a WZS reproduction. It is useful as a transparent fixed-chunk upper bound for the specific eight candidate encodings; it does not establish that transmitted or inferred online segmentation would achieve the same result.

## Two segment-local selection policies

Both segmented policies use F64S outer framing, then concatenate self-describing F64C child frames. Each F64C child has its own transform/model IDs, inner frame length, and CRCs; the outer F64S header records version, policy ID, element count, chunk size, segment count, raw CRC and header CRC. The child headers and outer header are all included in reported bytes.

### Policy 1: offline best-of-eight size oracle

For each complete chunk the encoder tries eight candidates:
1. Raw / Order0
2. Raw / Order1
3. XOR-previous-varint / Order0
4. XOR-previous-varint / Order1
5. XOR-leading/trailing window / Order0
6. XOR-leading/trailing window / Order1
7. RLE+XOR / Order0
8. RLE+XOR / Order1

It stores the smallest complete F64C frame. Ties resolve by candidate order. The decoder independently reconstructs the chunk, recomputes all eight canonical F64C frames, and rejects a child that is not the exact smallest candidate frame. This is a size oracle with full current-chunk access, not a streaming selector or a WZS reproduction. It gives an upper reference for the eight candidates and fixed chunk size.

### Policy 2: decoder-synchronous historical-cost selection

The new `compress_f64_online_selector` / `decompress_f64_online_selector` pair uses an eight-candidate past-only policy:
- first chunk defaults deterministically to Raw / Order0;
- before each later chunk, select the candidate with the smallest mean complete F64C frame length over previous chunks;
- encode the current chunk with that candidate;
- after each chunk, both encoder and decoder measure the seven alternatives and reuse the selected child's actual complete-frame length to update the same integer totals;
- the child F64C header exposes the selected transform/model IDs; the decoder independently computes the expected candidate and rejects a child whose IDs violate the historical policy.

This transmits no additional selector symbols. The selected transform/model IDs already live in the child header and are counted as part of the F64C frame. The policy is decoder-synchronous and uses only prior chunks to select the current chunk. Both sides measure the seven alternatives after encoding/decoding the selected candidate and reuse the selected child's actual frame length, avoiding one redundant re-encode. It remains a diagnostic online baseline, not yet a practical throughput-optimized selector. The policy is not a learned causal mechanism, and it should be compared against simpler features, fixed-share and WZS-style selectors.

### Policy 3: sparse-probe historical-cost selection

The new `compress_f64_budgeted_selector` / `decompress_f64_budgeted_selector` pair keeps the same past-only eight-candidate policy but measures alternatives only on scheduled probe chunks:
- The first chunk is always a probe.
- Subsequent probes occur every (K) chunks, where (K\in\{1,2,4,8,16,32,64,128\}).
- The F64S flags byte stores `log2(K)`; the header CRC protects it. The decoder derives the same schedule.
- On a probe, both sides measure all eight candidate child sizes. The chosen child length is reused, so only seven additional candidate encodings are required.
- Between probes, the incumbent candidate is used but **no candidate cost statistics are updated**. This keeps every candidate mean based on the same probe chunk sample set. Adding incumbent-only observations would create selection-biased means because the candidates would be measured on different time regimes.
- The decoder still derives the candidate choice from prior probe data and rejects a child whose transform/model IDs differ from the expected policy. No per-segment selection symbols are emitted.

For (S) chunks, probes are (lceil S/K\rceil), and extra candidate encodes per side are (7\lceil S/K\rceil). Average additional candidate encodes per chunk are approximately (7/K): 7.0 at (K=1), 1.75 at (K=4), 0.4375 at (K=16), and 0.1094 at (K=64). The selected transform is still encoded once per chunk. Larger (K) reduces compute but increases adaptation staleness and potential regret.

The CLI evaluates offline oracle, full historical and sparse-probe policies at chunk sizes 64/256/1024. CSV rows separate `encoder_extra_candidate_encodes` from `decoder_candidate_reencodes`, and record probe interval/count, complete frame ratio, encode/decode nanoseconds and percentage gap to the offline oracle. Offline rows report 7 extra encoder encodes and 8 decoder re-encodes per segment (canonical minimality verification); full historical rows report 7/7 (reuse selected child length); sparse rows report 7/7 per probe and 0 between probes. Single-transform rows report zero selector work. An invariant checks that no online policy produces a smaller complete frame than the per-chunk offline best-of-eight oracle. At (K=1), sparse-probe child frames must equal the full historical selector's child frames; only the outer F64S policy ID/header CRC differ.

These are authored Rust tests, not compiled Rust results. The tests cover bitwise round-trip for all supported probe intervals, K=1 equivalence, invalid interval rejection, policy mismatch, header CRC and offline-oracle dominance. No actual Rust frame size, throughput or energy result is claimed.

## Independent transform-logic check

The reusable XOR-window implementation has an independent C++20 reference at
`research/cdl-bench/reference/float64_window_reference.cpp`. It was compiled
with strict warnings:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/float64_window_reference.cpp \
  -o /tmp/float64_window_reference
/tmp/float64_window_reference
```

It exercised 57,105 bit patterns across five fixture classes and rejected
seven malformed marker/span/payload cases. Exact output:

| Fixture | Samples | XOR-varint bytes | Reusable XOR-window bytes | Change |
|---|---:|---:|---:|---:|
| Edge values | 9 | 78 | 60 | -23.08% |
| Smooth bit increments | 2,048 | 2,329 | 6,155 | +164.28% |
| Oscillating | 2,048 | 20,479 | 18,430 | -10.01% |
| Random bit patterns | 50,000 | 474,699 | 450,002 | -5.20% |
| Repeated values | 3,000 | 3,016 | 3,006 | -0.33% |

These are **transform-only payload bytes**, before entropy coding, F64C/F64S
headers, checksums and chunk selection. The large smooth-sequence regression
is a real negative result; it is why this transform remains a selectable
specialist rather than a universal default. The C++ program also round-trips
every fixture and rejects malformed markers, reuse-before-window, invalid
leading/trailing spans, empty windows, zero residuals encoded as nonzero, and
nonzero unused payload bits.

An earlier independent Python translation exercised the initial transform
logic; the checked-in C++ run is the reproducible reference result reported
above. Neither validates the Rust F64C/F64S framing or runs the Rust arithmetic
coder.

Rust tests are authored for raw/XOR-varint/XOR-window/RLE+XOR exact reconstruction,
deterministic encoding, truncated-prefix rejection, single-bit mutation
rejection, output-limit enforcement, malformed varint/window rejection and
explicit window reuse. Segmented tests cover bitwise round-trip at chunk sizes
1/64/256/1024, truncation, zero chunk size, output limit, header flags/policy
validation, canonical best-of-eight child selection and mutation rejection.
Historical-cost policy tests cover initial choice, cumulative eight-candidate
mean updates, stable tie-breaking, exact round-trip, child-policy mismatch
rejection and corruption. **No Rust compilation, cargo test or rustfmt pass
has been obtained.** No SOTA or frame-size win is claimed.

## Independent RLE+XOR transform-logic check

The checked-in C++20 reference
`research/cdl-bench/reference/float64_rle_xor_reference.cpp` independently
implements exact-bit-pattern run grouping plus XOR-varint coding of run
values. It compiled under `-Wall -Wextra -Wconversion -Werror`, round-tripped
174,108 u64 bit patterns over edge, constant, plateau, mostly-idle and mixed
random-run fixtures, and rejected seven malformed run/varint streams.

Command:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/float64_rle_xor_reference.cpp \
  -o /tmp/float64_rle_xor_reference
/tmp/float64_rle_xor_reference
```

Transform-only output before entropy coding and F64C/F64S framing:

| Fixture | Values | XOR-varint bytes | RLE+XOR bytes | Change vs XOR-varint |
|---|---:|---:|---:|---:|
| IEEE-754 edge values | 12 | 81 | 87 | +7.41% |
| Constant run | 4,096 | 4,104 | 11 | -99.73% |
| Plateau steps | 20,000 | 20,663 | 899 | -95.65% |
| Mostly-idle telemetry | 50,000 | 51,762 | 2,219 | -95.71% |
| Mixed random runs | 100,000 | 185,978 | 106,196 | -42.90% |

The edge fixture regresses because per-run lengths add overhead. Repeated-value
regimes improve dramatically, so RLE+XOR should remain a specialist in the
portfolio, not the universal default. This is transform-level evidence only:
it does not run the Rust implementation, arithmetic coder, or full typed
frames, and it does not reproduce the Dynatrace implementation.

## 2026 competitive comparison

### Production telemetry: RLE+XOR and adaptive block selection

Urbanides, Weninger and Ertl (SIGMOD Companion 2026) identify a material
dataset-bias problem: application-monitoring values often have long runs of
unchanged samples, unlike many academic finance/weather benchmarks. Their
paper reports a multi-layer record-level RLE+XOR approach with about 23%
better compression ratio and 34% faster execution than standalone methods.
A lightweight per-block selector is reported within 2% of the optimal ratio
and over 13% better than their best single RLE-enhanced compressor. These
are author-reported figures, not measurements reproduced by CDL-Bench.

- Publication: https://doi.org/10.1145/3788853.3803097
- Dynatrace summary:
  https://www.dynatrace.com/research/publications/analyzing-and-improving-floating-point-compression-for-application-monitoring-time-series/

CDL-Bench adds `RleXorPrevious`: exact-bit-pattern runs followed by XOR-varint
coding of run values. It is a transparent comparison baseline, not a
reimplementation of the Dynatrace codec or its heuristic. New plateau-step
and idle-telemetry fixtures expose the opportunity, but production telemetry
still requires a real, licensed or public corpus.

This changes the novelty boundary: RLE+XOR and block-level adaptive selection
are not novel by themselves. A possible differentiation remains the
decoder-synchronous selector contract, exact end-to-end frame accounting, and
causal/interventional multichannel evaluation. Even this combination needs a
full related-work review and reproduction against the 2026 systems.

### DeXOR — direct reproduction priority

The PVLDB 2026 DeXOR paper uses decimal-space XOR and longest common decimal prefixes/suffixes, with exception handling for extreme exponents. The paper reports 22 datasets, 15% higher compression ratio and 20% faster decompression versus the compared schemes. It publishes code and data:
- Paper: https://www.vldb.org/pvldb/vol19/p849-li.pdf
- Repository: https://github.com/SuDIS-ZJU/DeXOR

The repository supports multiple algorithms, including Gorilla, Chimp, Chimp128, Elf, ElfPlus, Camel, DeXOR and ALP. Its documented reproduction environment includes Java 8, Maven and a TSFile integration. This is the first specialized baseline to reproduce because a public artifact exists.

### Rabbit — paper-reported comparator

Rabbit (25 March 2026) predicts temporal branch/flag distributions and has a dedicated significant-bit encoding. Its paper reports an average 4.15% improvement over ACTF on 28 datasets and a maximum relative ratio reduction of 43.04% on highly redundant Server31/34/41 traces, with modest compression/decompression speed improvements. The paper's data-availability statement does not provide implementation code:
- Paper: https://www.mdpi.com/2073-8994/18/4/558

Until an implementation/configuration is available, use these only as paper-reported context, not reproduced leaderboard values.

### Dolphin — oscillating-stream comparator

Dolphin (11 February 2026) specifically targets sign-alternating or amplitude-oscillating floating-point streams where XOR residuals tend to have few leading zeros. It adds prefix skipping and adaptive branch selection. Its paper reports 1.6–1.8% compression-ratio improvement over its compared state-of-the-art algorithms (maximum 3.73%) across 18 datasets, and 9.3–18.3% faster compression plus 8.5–9.7% faster decompression. The article lists code availability as not applicable:
- Paper: https://link.springer.com/article/10.1007/s44443-026-00553-5

Treat Dolphin as paper-only unless a reproducible implementation is obtained.

### Other required comparators

- ACTF (2024) as a floating-point streaming baseline: https://www.sciencedirect.com/science/article/pii/S1319157824003355
- Gorilla and Chimp/Chimp128 for XOR/leading-zero distributions.
- Elf/ElfPlus, Camel and ALP where supported.
- Pco for typed numerical sequences; keep its per-column and chunking semantics.
- WZS for adaptive segmentation and per-segment algorithm selection; its paper reports 43 datasets and strong gains, but exact implementation/configuration must be reproduced before adopting its numbers: https://www.sciencedirect.com/science/article/pii/S0306457326000889
- Zstandard/Brotli/XZ as general-purpose controls.

These reported ratios cannot be directly ranked: they use different data sets, precision distributions, chunk sizes, framing, baseline versions and ratio definitions. Report the source and protocol for each comparison.

## What could become a genuine paradigm shift

The current XOR-previous transform is a starting control. The next research direction should be a **decoder-synchronous typed codec portfolio**:
1. characterize each segment using bit-level features (sign flips, exponent transitions, XOR leading/trailing zeros, decimal-prefix length, residual entropy and local stationarity);
2. select among reversible transforms and specialized predictors per segment;
3. choose boundaries from decoder-visible features or explicitly encode and charge any segment markers;
4. keep fast local context and slow global priors, with resets derived deterministically from decoded observations;
5. select the model by total encoded size plus a compute/memory penalty, not by prediction quality alone;
6. expose random access, checkpointing and integrity guarantees as first-class properties.

The key is *matched end-to-end accounting*. A branch selector that improves residual size but emits costly metadata or consumes excessive CPU is not a win.

## Confirmatory benchmark design

Use at least 20 datasets and group by regime:
- smooth/stationary;
- constant/repeated;
- sign-alternating oscillation;
- exponent transitions and large amplitude changes;
- decimal-like high precision;
- abrupt shifts and recurring regimes;
- noisy/near-random bit patterns;
- highly redundant server-like streams.

For each:
- same binary source bytes, precision, channel order, time ordering and chunk size;
- bit-exact reconstruction (including NaN payloads and signed zero);
- total frame size including all metadata, model identifiers, resets and side information;
- compression and decompression MB/s;
- peak resident memory and per-stream state bytes;
- encoder/decoder energy where hardware instrumentation permits;
- cold-start and amortized model costs;
- per-dataset and geometric-mean results, with paired bootstrap intervals;
- explicit negative-control results where XOR/temporal locality should not help.

A future superiority claim should require at least 5% lower end-to-end bytes than the strongest reproduced matched baseline on a preregistered stream family, with exact reconstruction and a paired 95% confidence interval excluding zero. For an adaptive selector, also require a pre-registered compute budget and report selector regret versus the offline oracle; a size win obtained by unbounded candidate re-encoding is not a practical SOTA result. No broad SOTA claim should be made from the synthetic fixtures alone.

## Immediate next actions

1. Reproduce DeXOR on its public source/data using the documented environment.
2. Obtain or implement independent reproduction paths for ACTF, Gorilla, Chimp/Elf and Rabbit/Dolphin if source is unavailable.
3. Run the new float64 harness against Zstd and publish bit-level diagnostics per fixture.
4. Add a segment-local selector only after single-transform baselines have been measured.
5. Keep this f64 benchmark separate from SCM causal-transfer claims; join the tracks only when the same typed sensor stream and evidence contract can support both.
