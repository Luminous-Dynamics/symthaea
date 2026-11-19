# CDL-Bench exact IEEE-754 f64 reference track v1

**Date:** 2026-11-19  
**Status:** Rust reference source, segment-local selector and authored tests committed; independent Python transform translation exercised; Rust compilation and arithmetic-coded size results pending.

## Why this track exists

The SCM/i16 lane does not represent the dominant specialized lossless floating-point time-series competition. By 2026, several purpose-built codecs target bit patterns, decimal prefixes, oscillation, branch flags and segment-local encoding choices. A credible SOTA comparison needs a typed IEEE-754 track; it must not infer that success on synthetic SCM i16 streams transfers to f64 sensor streams.

The branch adds a small correctness reference, not a competitive codec:
- `research/cdl-bench/src/float64.rs`: typed `F64C` frame with raw bits or XOR-previous u64 varints; `F64S` segment container with policy IDs for offline best-of-four and decoder-synchronous historical-cost selection; inner CDL arithmetic coding, CRC-32 checksums and output limits.
- `research/cdl-bench/src/main.rs`: `float --values N --seed N` experiment command comparing single-stream, offline segment-portfolio, and online historical-cost rows with frame size and encode/decode timings.
- `research/cdl-bench/src/lib.rs`: public exports.
- `research/cdl-bench/README.md`: usage and limits.
- Tracking: [CDL-Bench-007 #12874](https://github.com/Luminous-Dynamics/symthaea/issues/12874).

## Exactness contract

Values are converted only with `f64::to_bits` and `f64::from_bits`. The current transforms are:

| Transform | Definition | Purpose |
|---|---|---|
| Raw | Original u64 bit pattern, little-endian | Exact control |
| XOR previous | `bits_t XOR bits_(t-1)`, first previous = 0, canonical u64 varint | Simple temporal bit-pattern baseline |
| XOR leading/trailing | XOR residual, then store leading-zero count, trailing-zero count and only the meaningful middle bits | Simple reversible bit-window baseline; not Gorilla/Chimp |

No float subtraction, rounding, decimal parsing, NaN canonicalization, or normalization occurs. Tests compare each reconstructed `to_bits()` result with the input. Edge fixtures include:
- positive and negative zero;
- positive and negative infinity;
- quiet NaNs with distinct payload/sign bits;
- minimum subnormal and minimum normal;
- repeated constants, smooth bit increments, sign-alternating/oscillating patterns;
- deterministic random 64-bit patterns.

The F64C frame has a 32-byte header: magic/version, transform ID, inner model ID, element count, inner frame length, CRC-32 of original raw u64 bytes, and CRC-32 of inner frame. The decoder applies the element-count limit before allocating and caps transformed output to eleven bytes per f64 for the XOR-leading/trailing transform (marker, two span bytes and up to eight payload bytes).

The u64 varint parser rejects truncation, overlong encodings and values exceeding 64 bits. All three transforms must reproduce every IEEE-754 bit exactly.

## Current CLI

From `research/cdl-bench`:

```sh
cargo run --release -- float --values 10000 --seed 42
```

The command emits frame lengths and encode/decode nanoseconds for the single-stream arms, the offline segment portfolio and the historical-cost online selector at fixed chunk sizes 64, 256 and 1,024, across:
- constant;
- smooth-ramp;
- oscillating/sign-alternating;
- high-precision decimal-like;
- random-bit-pattern streams.

Each stream is evaluated with raw, XOR-previous-varint and XOR-leading/trailing transforms under order-0 and order-1 entropy models. The segment portfolio tries all six combinations per chunk and picks the smallest complete child F64C frame. The F64S container header, every F64C child header, child CRCs, and the inner CDL payloads are all counted in the reported frame size. Zstandard is measured on the original raw little-endian f64 bytes if installed. The CLI does not yet invoke DeXOR, ACTF, Gorilla, Chimp, Rabbit or Dolphin.

**Selection caveat:** the offline selector has the entire chunk available and tries all six encodings before choosing one. This is an offline size oracle, not an online selector and not a WZS reproduction. It is useful as a transparent fixed-chunk upper bound for the specific four candidate encodings; it does not establish that transmitted or inferred online segmentation would achieve the same result.

## Two segment-local selection policies

Both segmented policies use F64S outer framing, then concatenate self-describing F64C child frames. Each F64C child has its own transform/model IDs, inner frame length, and CRCs; the outer F64S header records version, policy ID, element count, chunk size, segment count, raw CRC and header CRC. The child headers and outer header are all included in reported bytes.

### Policy 1: offline best-of-six size oracle

For each complete chunk the encoder tries six candidates:
1. Raw / Order0
2. Raw / Order1
3. XOR-previous-varint / Order0
4. XOR-previous-varint / Order1
5. XOR-leading/trailing / Order0
6. XOR-leading/trailing / Order1

It stores the smallest complete F64C frame. Ties resolve by candidate order. The decoder independently reconstructs the chunk, recomputes all six canonical F64C frames, and rejects a child that is not the exact smallest candidate frame. This is a size oracle with full current-chunk access, not a streaming selector or a WZS reproduction. It gives an upper reference for the six candidates and fixed chunk size.

### Policy 2: decoder-synchronous historical-cost selection

The new `compress_f64_online_selector` / `decompress_f64_online_selector` pair uses a six-candidate past-only policy:
- first chunk defaults deterministically to Raw / Order0;
- before each later chunk, select the candidate with the smallest mean complete F64C frame length over previous chunks;
- encode the current chunk with that candidate;
- after each chunk, both encoder and decoder recompute all six complete candidate sizes from that chunk and update the same integer totals;
- the child F64C header exposes the selected transform/model IDs; the decoder independently computes the expected candidate and rejects a child whose IDs violate the historical policy.

This transmits no additional selector symbols. The selected transform/model IDs already live in the child header and are counted as part of the F64C frame. The policy is decoder-synchronous and uses only prior chunks to select the current chunk, but it is **compute-heavy**: both sides perform six candidate encodings after every chunk to update historical cost. It is a diagnostic online baseline, not yet a practical throughput-optimized selector. The policy is not a learned causal mechanism, and it should be compared against simpler features, fixed share, and WZS-style selectors.

The float CLI evaluates both policies at chunk sizes 64, 256 and 1,024. It reports size ratio and encode/decode nanoseconds for each complete outer frame. Rust tests are authored for exact bitwise reconstruction, policy mismatch rejection, corruption/truncation, zero chunk size and output limits; they have not run.

## Independent transform-logic check

A separate Python translation exercised:
- five fixture classes with 10,249 f64 bit patterns total;
- exact u64-varint round trips at 0, 1, 127, 128, 255, 16,384, (2^{32}-1), (2^{63}), and (2^{64}-1);
- exact XOR/reconstruction on every fixture;
- rejection of three malformed varint cases (truncated, overlong and >64-bit).

All checks passed. This is **not** Rust compilation, F64C frame validation, arithmetic coder execution, or a compression-performance result.

Rust tests are authored for raw/XOR-varint/XOR-leading-trailing exact reconstruction, deterministic encoding, truncated-prefix rejection, single-bit mutation rejection, output-limit enforcement, malformed u64-varint rejection and malformed bit-window rejection. Segmented tests cover bitwise round-trip at chunk sizes 1/64/256/1024, truncation, zero chunk size, output limit, header flags/policy validation, canonical best-of-six child selection, and single-bit mutation rejection. Historical-cost policy tests cover initial choice, cumulative six-candidate mean updates, stable tie-breaking, exact round-trip, child-policy mismatch rejection and corruption. An independent Python transform translation exercised 57,103 bit patterns over five fixture families and rejected five malformed bit-window cases. A separate Python policy check reproduced the decision sequence [0, 3, 3] and stable tie-breaking. The Rust tests have not run because the local environment has no Rust toolchain. No SOTA or frame-size win is claimed.

## 2026 competitive comparison

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

A future superiority claim should require at least 5% lower end-to-end bytes than the strongest reproduced matched baseline on a preregistered stream family, with exact reconstruction and a paired 95% confidence interval excluding zero. No broad SOTA claim should be made from the synthetic fixtures alone.

## Immediate next actions

1. Reproduce DeXOR on its public source/data using the documented environment.
2. Obtain or implement independent reproduction paths for ACTF, Gorilla, Chimp/Elf and Rabbit/Dolphin if source is unavailable.
3. Run the new float64 harness against Zstd and publish bit-level diagnostics per fixture.
4. Add a segment-local selector only after single-transform baselines have been measured.
5. Keep this f64 benchmark separate from SCM causal-transfer claims; join the tracks only when the same typed sensor stream and evidence contract can support both.
