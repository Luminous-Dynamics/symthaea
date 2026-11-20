# CDL-Bench standalone codec laboratory

This is a dependency-free Rust crate under `research/cdl-bench`. It is intentionally
outside the root Symthaea Cargo workspace and has no production cognitive-loop
dependencies.

## Implemented in v0.1

- Deterministic adaptive order-0 and order-1 byte-frequency predictors.
- A 32-bit arithmetic entropy coder / decoder.
- A public `BytePredictor` / `ByteCdf` interface so external predictors can use the same coder with deterministic integer CDFs and decoder-synchronous updates.
- A versioned frame with uncompressed length, payload length, source CRC-32 and
  payload CRC-32.
- Output-size limits, strict frame-length checks, and checksum failure handling.
- Deterministic E0 sources: IID uniform, IID skewed, first-order Markov, periodic,
  bursty, long runs and pseudo-random incompressible bytes.
- E1 finite-state generator with evaluator-only latent state and noisy/aliased emissions.
- E2 five-variable fixed-point structural causal generator with evaluator-only intervention manifests kept outside learner-visible observations.
- E3 public development fixtures with held-out combinations assembled from individually seen mechanism components; metadata is not serialized into learner-visible streams. These fixed tuples are visible in source and are not a blind test split.
- CLI file encode/decode and an E0 length comparison against installed
  Zstandard, Brotli, XZ and gzip commands.
- Exact IEEE-754 f64 raw/XOR-varint/reusable XOR-window/RLE+XOR framing, bit-preserving test fixtures, a fixed-chunk best-of-eight size oracle, and decoder-synchronous full/sparse historical-cost selectors.
- SCM-specific typed-field, temporal-lag, oracle-parent, hybrid and
  shuffled-parent byte predictors, plus fixed-share and CUSUM change-point
  mixtures, all using the same arithmetic coder and exact decoder path.

## Exact IEEE-754 f64 reference track

The `float` command generates constant, smooth-ramp, oscillating,
high-precision, plateau-step, idle-telemetry, signed-zero-switch, NaN-payload
and random-bit-pattern streams, plus a deliberately misaligned alternating regime workload (1,537-sample repeated-value and random-bit-pattern blocks).
It compares:
- raw IEEE-754 binary64 bit patterns;
- XOR with the previous u64 bit pattern, encoded as a canonical u64 varint;
- XOR with the previous u64 bit pattern, defining and reusing a leading/trailing significant-bit window;
- exact-bit-pattern run-length encoding followed by XOR-varint coding of each run value;
- order-0 and order-1 byte CDF models;
- Zstandard on the original raw little-endian bytes, when installed.

The single-stream frames use a 32-byte typed header, inner CDL frame and
raw/inner CRC-32 checksums. Decoding reconstructs via `f64::from_bits`; it
does not perform floating-point subtraction or normalization. Tests compare
`to_bits()` for each sample, including NaN payloads, infinities, signed zero
and subnormals.

The command also emits three segmented-policy families at chunk sizes 64, 256 and 1,024:

- `segment_portfolio` is an **offline best-of-eight size oracle**. For each
  complete chunk, the encoder tries Raw/Order0, Raw/Order1, XOR-varint/Order0,
  XOR-varint/Order1, XOR-window/Order0, XOR-window/Order1, RLE-XOR/Order0
  and RLE-XOR/Order1,
  then keeps the smallest complete F64C child frame. Every child header and
  CRC, plus the outer F64S header, counts toward the total size. The decoder
  independently recomputes all eight canonical child frames after
  reconstruction and rejects a child that is not the exact smallest choice.
  The selector sees the current chunk before choosing and is not a streaming
  algorithm, online predictive gate, or WZS reproduction.
- `online_historical_cost` chooses the current chunk from the average complete
  child-frame sizes for all eight candidates measured on **previous** chunks only.
  The first chunk uses Raw/Order0. After each chunk, both encoder and decoder
  measure the seven alternatives and reuse the selected child's actual frame
  length to update the same integer history.
  The selected transform/model is recorded in the child's F64C header and the
  decoder rejects a child whose IDs do not match its historical policy. There
  is no additional selector symbol or intervention information.
- `budgeted_sparse_probe` uses the same historical policy, but only probes all
  eight candidates on the first chunk and then every `K`-th chunk. Supported
  `K` values are powers of two from 1 through 128 and the interval is recorded
  in the CRC-protected F64S header. Between probes, both sides retain the same
  candidate-cost history and encode only the incumbent. The decoder derives
  the same probe schedule and rejects a child whose selected candidate differs
  from its history-derived choice.
  Importantly, **only common probe chunks update cross-candidate means**.
  We do not add extra measurements for just the incumbent between probes,
  which would compare candidate means on different, selection-biased sample
  sets. The CLI reports `K`, probe count, separate encoder-extra and
  decoder-recompute counts, complete frame ratio and percentage size gap to
  the offline best-of-eight oracle.

The CLI CSV separates encoder and decoder work with
`encoder_extra_candidate_encodes` and `decoder_candidate_reencodes`, and also
reports `candidate_switches` plus `oracle_candidate_mismatch_segments` for
segmented policies:
- Single-transform rows: 0 / 0.
- Offline best-of-eight: 7 extra encoder encodes per segment (eight candidates
  are tried; one is selected), and 8 decoder candidate re-encodes per segment
  because the decoder reconstructs and recomputes all eight canonical frames
  before accepting the child.
- Full historical: 7 extra encoder encodes and 7 decoder candidate re-encodes
  per segment. The selected child's actual complete frame length is reused.
- Sparse historical: 7 extra encodes per side on each scheduled probe and 0
  between probes.

The CSV also includes `oracle_gap_pct`, `probe_interval`, `probe_count`,
`candidate_switches` and `oracle_candidate_mismatch_segments`. A candidate
mismatch is a different transform/model ID from the per-chunk offline choice;
it is a selection-behaviour diagnostic, not a byte-regret metric.

The CLI extracts candidate IDs from the self-describing F64C child headers in
each F64S frame. Its source tests verify trace length and K=1 equivalence for
offline/full/sparse policies, and reject malformed/truncated frames and
impossible segment counts before allocating the trace. Offline and full-historical rows are recorded as probing every
segment; sparse rows count only the scheduled common probes. Every online
frame is checked to be no smaller than the offline best-of-eight frame for the
same source and fixed chunking. For K=1, sparse-probe child frames must match
the full historical-cost selector's child frames exactly (only the outer F64S
policy ID/header CRC differ).

For a stream with S chunks and probe interval K, the number of probes is
`ceil(S/K)`; sparse encoder extras and decoder candidate re-encodes are each
`7*ceil(S/K)`. The amortized extra candidate encodes per segment are about
7/K: 7.0 at K=1, 1.75 at K=4, 0.4375 at K=16, and 0.1094 at K=64. The chosen
candidate is still encoded once per chunk and decoded once per chunk. The
offline decoder intentionally costs more because it verifies canonical
best-of-eight minimality.

The online policies offer different points on the size/compute frontier:
full historical measurement gives more frequent feedback, while sparse probes
reduce selector work but increase adaptation staleness. Neither is yet a
throughput-optimized production selector. All policies use the 32-byte F64S
outer header and count every F64C child header/CRC.

### Production telemetry and RLE context

A SIGMOD 2026 Dynatrace study reports that application-monitoring streams can
contain long unchanged runs that are underrepresented in academic finance and
weather datasets. Its paper reports an RLE+XOR stack with about 23% higher
compression ratio and 34% faster execution than standalone methods; a block
selector reportedly comes within 2% of optimal and improves over the best
single RLE-enhanced codec by more than 13%. These are paper-reported results,
not CDL-Bench measurements.

The new `RleXorPrevious` transform groups only exactly identical u64 IEEE-754
bit patterns (so signed zero and NaN payload distinctions remain exact), then
stores each run length and the XOR residual of its value against the preceding
run value. This is a transparent comparison point, not a reproduction of the
Dynatrace implementation. `float` now includes long plateaus and mostly-idle
telemetry-like fixtures, alongside smooth and random patterns.

Reference:
- Urbanides, Weninger and Ertl, SIGMOD Companion 2026:
  https://doi.org/10.1145/3788853.3803097

## Independent f64 bit-window reference

The checked-in C++20 reference
`research/cdl-bench/reference/float64_window_reference.cpp` independently
implements the reusable leading/trailing XOR window. It compares transform-only
payload bytes with XOR-varint and checks exact u64 reconstruction. Build and
run it with:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/float64_window_reference.cpp \
  -o /tmp/float64_window_reference
/tmp/float64_window_reference
```

The updated strict-warning reference source covers 57,105 values and includes eight malformed marker/span/payload cases, adding rejection of a redundant new-window marker when the previous window already fits. Transform-only byte counts versus XOR-varint were:
- edge values: 78 → 60 bytes (-23.08%);
- smooth bit increments: 2,329 → 6,155 (+164.28%);
- oscillating values: 20,479 → 18,430 (-10.01%);
- random patterns: 474,699 → 450,002 (-5.20%);
- repeated values: 3,016 → 3,006 (-0.33%).

These counts are before the byte entropy coder and all framing. The smooth
sequence is a significant regression, so this remains a selectable specialist,
not a universal default or a claim of Gorilla/Chimp parity.

## Independent RLE+XOR reference

The checked-in C++20 program
`research/cdl-bench/reference/float64_rle_xor_reference.cpp` independently
implements exact-bit-pattern run grouping plus XOR-varint coding of each run
value. It verifies exact u64 reconstruction and rejects seven malformed
streams. Build and run it with:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/float64_rle_xor_reference.cpp \
  -o /tmp/float64_rle_xor_reference
/tmp/float64_rle_xor_reference
```

The strict-warning reference run covered 174,108 values and rejected seven
malformed run/varint cases. Transform-only sizes, before the byte entropy
coder and all F64C/F64S framing, were:

| Fixture | XOR-varint | RLE+XOR | Change vs XOR-varint |
|---|---:|---:|---:|
| IEEE-754 edge values | 81 B | 87 B | +7.41% |
| 4,096-value constant run | 4,104 B | 11 B | -99.73% |
| 20,000-value plateaus | 20,663 B | 899 B | -95.65% |
| 50,000 mostly-idle values | 51,762 B | 2,219 B | -95.71% |
| 100,000 mixed random runs | 185,978 B | 106,196 B | -42.90% |

RLE+XOR is harmful on the tiny edge fixture due to per-run count overhead, but
dramatically improves repeated regimes. This supports retaining it as a
portfolio specialist rather than making it a universal transform. These are
transform-only reference measurements, not Rust arithmetic-coded results.

The reusable XOR-leading/trailing window is a small Gorilla-style reference transform,
not a Gorilla/Chimp implementation. Neither the single-stream XOR transforms
nor these segment selectors implement ACTF, DeXOR, Rabbit or Dolphin. These are transparent
correctness baselines and a harness for those specialized comparators.

## Independent fixed-share reference simulation

The checked-in C++20 program
`research/cdl-bench/reference/scm_expert_mixture_reference.cpp` independently
reproduces the SCM2 generator, exact-rational three-expert probability mixture,
posterior updates and fixed-share rule using `unsigned __int128`. It is a
diagnostic/reference implementation only; it does not link to or execute the
Rust crate and is not a substitute for Rust tests.

Build and run it with GCC or Clang that supports `unsigned __int128`:

```sh
g++ -O3 -std=c++20 -Wall -Wextra \
  research/cdl-bench/reference/scm_expert_mixture_reference.cpp \
  -o /tmp/cdl_mixture_ref
/tmp/cdl_mixture_ref
```

It prints per-stream ideal prequential bit costs and weight traces. The
reproducibility report is
`docs/research/CDL_EXPERT_MIXTURE_FIXED_SHARE_V1_2026-11-18.md`.
These ideal-bit diagnostics must not be confused with actual arithmetic-coded
frame lengths, which still require compiling and running the Rust CLI.

## Independent CUSUM detector reference

The controlled detector check is in
`research/cdl-bench/reference/cusum_detector_reference.cpp`. It tests
fixed-point log2 ratio and CUSUM behavior only; it does not replay SCM traces or
the Rust encoder/decoder.

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/cusum_detector_reference.cpp \
  -o /tmp/cdl_cusum_reference
/tmp/cdl_cusum_reference
```

The observed controlled checks are: zero alarms over 100,000 stationary
observations; detection of a 3-bit surprisal step after three observations;
and no alarm on a gradual 0.5-bit drift over 512 observations. These are
specific synthetic detector checks, not an empirical false-alarm guarantee.
See `docs/research/CDL_CHANGEPOINT_CUSUM_V1_2026-11-19.md`.
Implementation tracking: [CDL-Bench-006 #12873](https://github.com/Luminous-Dynamics/symthaea/issues/12873).


## Run

```sh
cd research/cdl-bench
cargo test
cargo run --release -- e0 --bytes 65536 --seed 42
cargo run --release -- causal --steps 10000 --seed 42
cargo run --release -- numeric --values 10000 --seed 42
cargo run --release -- float --values 10000 --seed 42
cargo run --release -- bounded --values 10000 --seed 42 --tolerance 5
cargo run --release -- encode --model order1 input.bin output.cdl
cargo run --release -- decode output.cdl reconstructed.bin
cmp input.bin reconstructed.bin
```

## Frame format v1

All integer fields are little-endian.

| Offset | Length | Field |
|---:|---:|---|
| 0 | 4 | Magic `CDL1` |
| 4 | 2 | Version (1) |
| 6 | 1 | Model id (0 = order-0, 1 = order-1) |
| 7 | 1 | Reserved flags (must be zero) |
| 8 | 8 | Original byte length |
| 16 | 8 | Payload length |
| 24 | 4 | CRC-32 of reconstructed source |
| 28 | 4 | CRC-32 of arithmetic-coded payload |
| 32 | variable | Arithmetic-coded payload |

The decoder reconstructs the same adaptive model from previously decoded bytes.
No probability table or training corpus is read from an undeclared external source.

## Accounting and limitations

- The E0 table compares complete CDL frames (including the 32-byte header) with
  native command outputs from installed general-purpose codecs.
- The predictor state byte count is reported separately. The runner executable
  size is printed by the file encoder, but excludes shared libraries and OS
  runtime dependencies. These values are **not** silently added to the per-stream
  frame ratio.
- The codec model itself is a very small adaptive byte model, not a claim of
  SOTA compression. Order-0 and order-1 are correctness and baseline arms.
- The pseudo-random generator is deterministic test data, not cryptographic
  randomness and not a substitute for an external incompressible corpus.
- The CLI E0 runner caps each generated stream at 64 MiB.
- CRC-32 detects accidental corruption; it is not authentication or tamper-proof
  provenance.
- The arithmetic coder and frame format are a research prototype. Do not use
  them for production archival, security, or untrusted data interchange.
- No compression result, external benchmark win, causal-transfer result, or
  SOTA claim is implied by source existence. Run the tests and record the exact
  toolchain and machine before interpreting performance.

The first milestone is decoder correctness and transparent accounting, not a
headline compression ratio.

## E1 and E2 generators

`cdl-bench causal --steps 10000 --seed 42` creates:
- A partially observed finite-state process (8 latent states, 4 emission symbols).
- An observational five-variable structural causal trace.
- A paired trace with `do(x1=20000)` active on the middle half of the stream.

The command checks exact codec round-trips and reports framed lengths for
the order-0/order-1, typed-field, temporal-lag, parent-oracle, hybrid and
shuffled-parent models, plus Zstandard when installed. The parent and hybrid
models receive the correct SCM graph by construction; they are topology-aware
coding controls, not learned causal predictors. The shuffled-parent model is a
topology negative control. The finite-state latent labels, intervention
manifests and mechanism parameters remain evaluator-side. The learner-visible
SCM2 stream contains schema/length metadata and observed i16 values only; it
does not expose intervention targets, windows or forced values. No
causal-transfer result is implied.

## Typed numerical i16 pipeline

`cdl-bench numeric --values 10000 --seed 42` compares four reversible
transforms (raw, delta, delta-of-delta and XOR-with-previous), each paired with
the order-0 and order-1 byte models. It also reports Zstandard size on the raw
little-endian i16 bytes when the command is installed.

The library API is:
- `compress_i16(values, transform, model)`
- `decompress_i16(frame, max_values)`

The numeric frame wraps the existing CDL frame, counts values explicitly, and
checks both the inner frame and reconstructed raw i16 data. This is a baseline
transform layer, not a reimplementation of Pco and not yet a causal predictor.
The next comparisons should include Pco, Sprintz, Gorilla/Chimp, TSCom-Bench
and SEP under matched dataset preparation and hardware.

## Separate bounded-error i16 experiment

`cdl-bench bounded --values 10000 --seed 42 --tolerance 5` runs a
**lossy** predictive coding experiment with a hard absolute error tolerance.
It is intentionally not part of the exact-lossless scorecard.

The bounded-error codec currently compares:
- Previous-value predictor.
- Linear predictor: `2 * previous - previous_previous`.
- Order-0 and order-1 adaptive byte models.
- Four synthetic source families: constant, ramp, random walk and demand-like.

For integer tolerance `tau`, it uses residual reconstruction step
`2 * tau + 1`. Each input sample is mapped to the nearest integer point on
the predictor-relative grid. The maximum quantization error before clipping is
`floor((2*tau + 1)/2) = tau`; clipping the reconstructed value to the i16
range cannot increase error for an i16 source sample.

Encoder and decoder both update predictors from reconstructed samples only.
This makes predictor state deterministic and decoder-synchronous. The format
stores the tolerance, predictor ID, entropy-model ID, sample count, payload
length and reconstructed-output checksum.

The command independently checks the maximum absolute error after decoding and
prints `PASS` only when every sample satisfies the configured tolerance. This
is a prototype baseline, not an implementation of Cadence, SZ3 or Pco; its
synthetic demand-like source is not an external electricity or transit dataset.

**Never compare its compressed size directly with an exact-lossless result
without reporting the fidelity difference.** A future bounded-error benchmark
must include Cadence and classical error-bounded baselines on identical data,
tolerance, entropy coder, context-bootstrap and resource accounting.

### Structure-aware SCM coding controls

The `causal` command additionally evaluates seven custom byte predictors on every
SCM2 stream, including observational, interventional, training and held-out E3
traces. All use the same arithmetic coder and 32-byte CDL frame as the built-in
order-0/order-1 arms. The learner-visible SCM2 header is 16 bytes and contains
only magic, version, flags, variable count and sequence length. Intervention
metadata is retained in the evaluator-side `ScmTrace` and never serialized:

- `field_frame`: separate adaptive base histograms for each variable and
  byte lane. This tests typed channel identity against one flattened byte
  stream.
- `lag_frame`: conditions each variable/lane on the previous timestep's
  reconstructed full i16 value for that same variable. This is a noncausal
  temporal baseline with no cross-variable graph.
- `parent_oracle_frame`: conditions each field/lane on the full reconstructed
  same-timestep parent values, quantized into four bins, using the known SCM
  graph.
- `hybrid_frame`: combines one-bit parent-sign context with the previous
  timestep's four-bin same-variable context.
- `shuffled_parent_frame`: preserves the context arity while deliberately
  misassigning downstream parent identities. This is a topology negative
  control, not a valid causal graph.

- `online_mixture_frame`: combines the field, lag and parent-oracle
  probabilities before each byte. After observing the byte, integer Bayesian
  expert weights update from each expert's likelihood. A fixed-share rule
  redistributes 1/256 of posterior mass across the three experts per byte, so
  a weak expert can recover after a regime change without receiving an
  intervention label or expert-selection symbol.

- `change_point_frame`: adds an integer surprisal-CUSUM over the prequential
  mixture. After a detector alarm, the model retains field-level base
  histograms, resets fast lag/parent context tables, and re-primes expert
  weights toward lag. Alarm decisions are derived from the same decoded prefix
  in encoder and decoder; no reset marker, intervention label or boundary is
  transmitted. The CLI reports alarm count and expert weights.

The detector uses a slow surprisal baseline (1/64 EMA), a 0.25-bit per-byte
CUSUM drift allowance, an 8-bit cumulative threshold, a 64-byte warmup and a
256-byte cooldown. These are fixed v1 defaults, not tuned SOTA parameters.
False alarms, detection delay, recovery time and extra coded bits must be
measured separately before the reset policy is considered useful.
Both mixture arms share one header and field/base histogram set across all experts;
only lag and parent context tables are specialist-specific. This avoids
duplicating identical per-stream statistics. Its state-size output exposes the
memory cost, while its final expert weights are printed as diagnostics. The
parent specialist still receives the known SCM topology and remains an oracle
expert, not a learned mechanism.

The mixtures combine the specialists' original CDF probabilities using an
exact common rational denominator (u128), then quantize only the final CDF to
the arithmetic coder's 32,768 frequency total. Model 7 uses a 1/256 fixed-share
weight update; model 8 additionally detects sustained surprisal increases and
resets fast context tables. Neither receives intervention labels or transmits
expert-selection/reset symbols. The fixed-share experiment is described in
[its report](../../docs/research/CDL_EXPERT_MIXTURE_FIXED_SHARE_V1_2026-11-18.md);
the CUSUM detector's controlled logic check is described in
[the CUSUM report](../../docs/research/CDL_CHANGEPOINT_CUSUM_V1_2026-11-19.md).

For parent, lag, hybrid and shuffled-parent arms, the predictor adds the
field/lane base frequencies to the selected conditional-context frequencies
when forming its integer CDF. Both histograms are updated after each decoded
symbol. This simple additive interpolation gives every context a shared
field-level backoff distribution rather than making a cold context predict
from a sparse local histogram alone. The typed-field arm uses only the
base histogram.

The parent and hybrid predictors are **oracle-topology controls**. The graph is
supplied in their implementation; this is not graph discovery, causal
mechanism learning, intervention generalization, or a Symthaea HDC/CfC result.
The conditional histograms are adaptive and decoder-synchronous, but the
conditioning structure is known in advance. The lag arm is deliberately a
strong noncausal comparator. The shuffled-parent arm tests whether a result is
specific to the correct variable identities.

The specialist predictors share header/field statistics within the online
mixture; every model resets to deterministic empty state for each stream. The
runner prints inline predictor-state sizes, mixture weights and CUSUM alarm
counts. Model 8 adds detector state and surprisal/CUSUM logic; code/executable
costs and shared model distributions must still be accounted for in cold-start
comparison. The per-stream frame size alone is not a complete deployed-system
size.

Interpretation order:
1. Compare field/lane with order-0/order-1 to measure typed channel identity.
2. Compare lag with field/lane to measure ordinary temporal persistence.
3. Compare parent-oracle with lag to test whether known graph structure helps.
4. Compare the fixed-share mixture with each specialist to measure adaptation
   and switching regret across observational/interventional regimes.
5. Compare the CUSUM-gated mixture with fixed-share without resets, measuring
   false alarms, detection delay, post-shift regret, recovery and total coded
   bits.
6. Only then test a learned mechanism model against the oracle and the matched
   noncausal temporal control.

No compression win is implied by source existence. The runner must be compiled
and executed, and total-size/resource accounting must be complete, before
reporting an encoded-bit result.

### E3 mechanism recombination

The same causal command also emits:
- Three training SCM configurations.
- Two held-out configurations that recombine mechanism component values seen in the training configurations, while keeping the full configuration tuple unseen.
- For each held-out configuration, a paired observational and `do(x1=20000)` trace with the same seed, so the intervention comparison is counterfactual while target/window/value remain evaluator-side.

The mechanism configuration and intervention manifest are kept out of the learner-visible SCM2 stream (truth fields are crate-private). The fixed E3 combinations are public development fixtures, not a blind test set; a confirmatory result requires an externally controlled split and seed manifest. The CLI labels the paired held-out streams and verifies codec round trips; it does not yet train a mechanism learner or measure transfer. A generator unit test also checks that the upstream x0 remains invariant while descendants x2 and x4 respond to the x1 intervention in each held-out world. This is a dataset-generation fixture only.

## Deterministic categorical CDF calibration

`categorical_byte_cdf_from_log2_logits` converts 256 signed Q4 log2 logits
into a `ByteCdf`. The adapter supplies scores calibrated on a separate
calibration split and a Q16 inverse-temperature parameter (65,536 = 1.0).
The converter uses a fixed Q24 `2^(-fraction)` lookup table, clips score
differences to 16 log2 units, and performs integer largest-remainder
normalization to a total frequency of 32,768. Every symbol receives a
configurable minimum frequency (1..=127); ties are resolved by ascending byte
symbol. No floating-point operation is used in the conversion.

This is only the deterministic final readout-to-CDF step. It does **not** map an
arbitrary `ContinuousHV` to byte logits, train a categorical readout, fit the
temperature, or prove calibration. Those steps require the separate HDC/CfC
adapter and held-out calibration protocol in
[CDL-Bench-005](https://github.com/Luminous-Dynamics/symthaea/issues/12847).
The codebook/readout, calibration parameters and decoder costs still count
toward model accounting.

## External codec reproduction

A pinned, method-aware reproduction setup is documented in
`docs/research/CDL_EXTERNAL_CODEC_REPRO_MANIFEST_V1_2026-11-19.md`.
It records the inspected DeXOR commit and 22-file `Overall` corpus, the
TerseTS commit and exact-lossless whitelist, upstream build constraints, and
the distinction between paper-reported results and actually reproduced
numbers.

The helper script is
`scripts/research/cdl_external_codec_repro.sh`. With pre-existing checkout
directories at the pinned commits, run:

```sh
DEXOR_DIR=/path/to/DeXOR TERSETS_DIR=/path/to/TerseTS \
  bash scripts/research/cdl_external_codec_repro.sh
```

It verifies revisions, builds, records environment/data hashes and explicitly
skips any external repository not supplied. For TerseTS it also compiles and
runs `scripts/research/tersets_bitwise_gate.c` against the pinned native C API:
Uncompressed/Chimp64/Chimp128/Elf must preserve every u64 bit on edge and
signed-zero fixtures; RLE and BitPackedDelta are reported separately as
conditional candidates. DeXOR requires Java 8 and Maven; TerseTS requires Zig
and a C compiler. Neither external codec has been built or run in the current
environment. TerseTS is beta, and lossy/bit-nonexact methods are excluded from
the exact-lossless comparison.

## Typed telemetry records and bounded state-delta coding

The experimental `telemetry` module adds a typed multi-field record frame:

- Eight schema-declared 64-bit lanes (`U64`, `I64`, `F64Bits`, `U32`, `Boolean`, `Opaque64`).
- Per-record timestamp, entity ID, series ID, presence bitmap and exact field payload bits.
- Explicit changed-field bitmap and null/presence operations; unchanged values are reconstructed from the prior state for that series.
- First-seen or re-admitted series are encoded as full checkpoints.
- A deterministic least-recently-seen eviction policy bounds state to the frame-declared capacity; ties break by entity/series key.
- Predictor state is reset at the declared segment-record interval, making every segment start an explicit checkpoint for all series.
- Header CRC, inner-frame CRC and canonical uncompressed-stream CRC; the inner payload uses the shared arithmetic coder.
- Decoder caps both record count and state capacity before allocation. Boolean/U32 field kinds are range-checked, absent field payloads must be zero, and varints/update masks are canonical.

From `research/cdl-bench`:

```sh
cargo run --release -- telemetry --records 65536 --keys 64 --state-capacity 64 --segment-records 4096 --workload sparse --seed 2111830017
```

Supported synthetic workloads are `sparse`, `bursty`, `idle`, and `noisy`. The CSV reports complete CDL telemetry-frame bytes against Zstandard on the same canonical raw typed-record serialization, plus encode/decode nanoseconds and a lower-bound state-payload estimate. The CLI path performs exact decode equality checks. This is a synthetic development diagnostic; use pinned production-derived data and matched ALP/Pco/Gorilla/Chimp/ACTF/Rabbit baselines before drawing compression claims.

The public API is:
- `TelemetryFieldKind`, `TelemetrySchema`, `TelemetryRecord`, `TelemetryStream`.
- `compress_telemetry_records(records, schema, state_capacity, segment_records, model)`.
- `decompress_telemetry_records(frame, max_records, max_state_entries)`.

An independent C++20 transform reference and Zstandard comparison runner are checked in at `research/cdl-bench/reference/typed_record_delta_bounded_reference.cpp` and `scripts/research/cdl_typed_record_delta_bounded_reference.sh`. The C++ runner exercises 200 randomized state/segment round trips, six malformed cases and four 65,536-record synthetic fixtures under strict compiler warnings. Its output is transform-level, not a Rust test or a complete CDL frame result. Run from the repository root with `bash scripts/research/cdl_typed_record_delta_bounded_reference.sh`; it requires `g++` and `zstd`.

This is an initial fixed eight-lane reference, not a general Arrow/Parquet schema container. The frame preserves row order and exact values, but this first version does not include a random-access index, timestamps dictionary, schema evolution, multi-segment independent child frames, or production recovery protocol. Segment resets bound predictor replay/state but do not by themselves make the monolithic outer frame randomly seekable. The frame stores only the field-kind IDs, not semantic field names; a higher-level schema registry is still required and must be included in total system accounting.

Authored Rust tests cover empty and populated streams, two entropy models, multiple state capacities, segment resets, bitwise f64 payloads, null transitions, truncated prefixes, single-bit mutations, malformed varints, invalid values, and record/state output limits. They have **not** been compiler-run in the current environment.

## Prefix-only LZV segmentation control

The experimental `segmentation` module provides `segment_lzv(&[u64], LzvConfig)`, returning deterministic segment-boundary indices from leading-zero counts of exact XOR residuals. It uses separate fast/slow windows, integer mean and population-variance comparisons, a pending-change latch, minimum segment length, cooldown and maximum segment length. It reads only the already-decoded prefix and transmits no boundary labels.

This is a **WZS-inspired control**, not the authors' WZS implementation. The independently compiled C++ reference is at `research/cdl-bench/reference/lzv_segmentation_reference.cpp`; run `bash scripts/research/cdl_lzv_segmentation_reference.sh` from the repository root. It checks stationary, constant and abrupt/recurring shift fixtures. It measures boundary behaviour only, not compressed bits or WZS performance.

The Rust module has authored tests for constant streams, stationary random streams, abrupt/recurring shifts, empty/singleton inputs, determinism and invalid configurations. They have not been compiler-run here because the Rust toolchain is unavailable.

From the standalone crate, the synthetic boundary report can be generated with:

```sh
cargo run --release -- segment --values 4096 --workload stationary-random --seed 2748 --min-segment 32 --max-segment 512
```

Supported fixtures are `constant`, `stationary-random`, `stable-random` and `stable-random-stable`. The command prints the config, segment lengths, detected LZV boundaries and forced maximum-length boundaries. This measures segmentation only; it does not yet encode each selected segment or establish a compression gain.

## Rabbit-derived branch-mapping diagnostic

The isolated flag-mapping reference is in `research/cdl-bench/reference/rabbit_branch_mapper_reference.cpp` with runner `scripts/research/cdl_rabbit_branch_mapper_reference.sh`.

This diagnostic models four abstract branch IDs and the paper-described prefix codes `0`, `10`, `110`, `111`, ranked using previous-block branch counts. It does not implement ACTF branch classification/payloads, XOR residual coding or TOE, and must not be called a full Rabbit reproduction. A guarded previous-block policy is included as an experimental CDL extension; it falls back to fixed two-bit flags when the predicted distribution is not sufficiently skewed or stable.

Run from the repository root:

```sh
bash scripts/research/cdl_rabbit_branch_mapper_reference.sh
```

The C++ runner uses a default block size of 1,000 symbols (Rabbit's paper default), checks exact flag round-trips including 100 randomized streams, and reports flag-only bits against fixed two-bit flags and an offline same-block fixed-or-dynamic oracle. These are synthetic flag costs only, not end-to-end Rabbit/ACTF or CDL compression results. See the research note for the ablation numbers and limitations.

## Per-series telemetry layouts: PRN1 row-state and PCL1 field-columnar

The C++20 reference at `research/cdl-bench/reference/typed_record_partitioned_runs_reference.cpp` now compares two exact-lossless per-series child layouts.

**PRN1 row-state layout** uses a stable dictionary of entity/series keys and one state-delta/heartbeat child stream per key. It restores global order through:
- Mode 0: strict unique timestamps, deterministic k-way merge, no order schedule.
- Mode 1: nondecreasing duplicate timestamps, tie-group order templates with a bounded 16-entry move-to-front dictionary.
- Mode 2: timestamp regressions, full bit-packed key-ID schedule for exact arrival-order replay.
Mode-1 uses a heap frontier and verifies per-key tie multiplicity, including multiple same-series rows at a timestamp.

**PCL1 field-columnar layout** is a second candidate for strictly increasing unique timestamps. For each series it stores timestamp deltas plus separate per-field presence bitmaps, changed bitmaps and contiguous exact-u64 value lanes. The decoder merges child streams to restore global order. It currently uses a fixed eight-field schema, one outer CRC, and Zstd on the full frame. It does not yet integrate Pco or ALP.

Run both layout ablations from the repository root:

```sh
bash scripts/research/cdl_partitioned_telemetry_reference.sh
```

The runner compiles with `-O2 -std=c++20 -Wall -Wextra -Wconversion -Wno-misleading-indentation -Werror`, runs PRN1 plus PCL1 exact-round-trip tests, tests 1/8/64/1,024/4,096 keys, and malformed frames. It then compares equally framed raw, global state-delta, PRN1 and PCL1 outputs with the same Zstd level and verifies exact decompression.

Selected synthetic Zstd 1.5.7 level-19 results (bytes; 65,536 records / 64 interleaved keys):

| Workload | Global state-delta | PRN1 row-state | PCL1 field-columnar |
|---|---:|---:|---:|
| Sparse | 85,931 | 61,864 | 59,042 |
| Bursty | 224,396 | 204,921 | 188,405 |
| Mostly idle | 5,577 | 4,784 | 5,692 |
| Noisy | 2,316,397 | 2,252,910 | 2,042,284 |

PCL1 is 4.56%, 8.06% and 9.35% smaller than PRN1 for sparse, bursty and noisy workloads, respectively, but 18.98% larger on mostly idle. The row-state heartbeat representation wins on idle data. This is evidence for adaptive layout selection, not a universal PCL1 win. The full byte tables, tests and evidence limitations are in `docs/research/CDL_TYPED_TELEMETRY_JOINT_SEGMENTATION_V1_2026-11-19.md`.

**Adaptive per-series variants** add two further ablations:
- **PAX1:** choose row-state or field-columnar per series by raw child size, store the child uncompressed, then compress the complete frame with Zstd.
- **PZC1:** compress each row/columnar candidate with libzstd at the requested Zstd level, select the smaller compressed child, and store each selected child as an independent Zstd frame with raw length, compressed length and child CRC32. This enables independent per-series decompression at the cost of resetting the compressor context.

The runner requires `pkg-config` and the libzstd development library in addition to the Zstd CLI, links with `-lzstd`, and passes the same numeric `ZSTD_LEVEL` to PZC1's child encoder and the whole-frame Zstd controls. PZC1 is reported as its already-compressed complete frame; PRN1/PCL1/PAX1 are reported after whole-frame Zstd.

Synthetic results (bytes, 65,536 records / 64 keys):

| Workload | Zstd(PRN1 row) | Zstd(PCL1 columnar) | Zstd(PAX1 raw-size choice) | PZC1 independent child-Zstd |
|---|---:|---:|---:|---:|
| Sparse | 61,864 | 59,042 | 61,887 | 72,574 |
| Bursty | 204,921 | 188,405 | 204,915 | 219,720 |
| Mostly idle | 4,784 | 5,692 | 4,782 | 9,095 |
| Noisy | 2,252,910 | 2,042,284 | 2,042,502 | 2,173,924 |

PAX1's raw-size heuristic picks row-state for sparse/bursty/idle and columnar for noisy; it misses PCL1's post-Zstd advantage on sparse/bursty. PZC1's actual per-child Zstd selector picks columnar for sparse/bursty/noisy and row-state for idle, but per-child compressor resets cost bytes compared with whole-frame Zstd. This demonstrates why layout, compression context and query granularity must be co-optimized.

The tie-schedule primitive is separately reproducible:

```sh
bash scripts/research/cdl_tie_group_order_template_reference.sh
```

Its independent reference passed 1,000 randomized tie-group round-trips and five malformed-input checks. It reduced the schedule from 192 to 67 bytes for 64 repeated groups of eight keys, and from 160 to 36 bytes for 20 alternating groups of 16 keys. This tests the schedule subgrammar separately from the full PRN1 frame.

**Evidence boundary:** all these results are independent synthetic C++ transform+Zstd diagnostics, not Rust CDL, not production data, and not ALP/Pco/tparquet comparisons. Neither layout has yet been measured for bounded-memory streaming, throughput, RSS, random access, indexes or checkpoint recovery. PRN1 supports tie/regression fallback; the current PCL1 prototype intentionally rejects duplicate or out-of-order global timestamps. Issue #12956 tracks field-columnar lane codecs and adaptive selection; #12935 tracks bounded-memory streaming and seekability.

## Industry benchmark tracks

The partitioned C++ reference is only a transform diagnostic. CDL-Bench now has two additional mandatory end-to-end competitor layers:

**Storage / column encoding:** Parquet default Zstd(9), tuned per-column Zstd, tuned+ALP (Parquet 2.14 preview), Pco/PCODEC and tparquet's per-column PCODEC/Zstd selector. The tparquet snapshot `aperiodic-io/tparquet@d1a40f6fa5508c99ee95e9609f3c38b111e21cf6` reports 31 public datasets and 151.5M rows: 478.6 MB tparquet, 649.5 MB tuned+ALP, 700.7 MB tuned Zstd and 957.1 MB default Zstd. This is a published external baseline, not a CDL reproduction.

**Transport / dataflow:** OTLP+Zstd versus OTAP (Arrow IPC over gRPC) and any future CDL transport. The audited OpenTelemetry Arrow Dataflow Engine is incubating and has continuous/nightly metrics for normal load (~100k records/sec per core), filtering, batch size, backpressure, saturation and multi-core scaling; its guide states a >0.90 scaling-efficiency target. These are transport results, not directly comparable with Parquet file bytes.

Full pins, methods, and caveats:
- `docs/research/CDL_EXTERNAL_CODEC_REPRO_MANIFEST_V1_2026-11-19.md`
- `docs/research/CDL_BENCH_COMPETITIVE_LANDSCAPE_2026-11-18.md`
- Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/12950

The benchmark rule is to compare complete storage frames and complete transport paths separately, with identical typed data and exact reconstruction. Report compression, encode/decode throughput, memory, access latency, CPU, network bytes and dropped-record counts; do not merge these into one score.

## Plugging in a custom predictor

The library exposes:
- `ByteCdf = [u32; 257]`: cumulative integer frequencies for all 256 byte symbols.
- `BytePredictor`: a stable model ID, `cumulative_frequency()`, and `observe(decoded_symbol)`.
- `compress_with_predictor(input, &mut predictor)`.
- `decompress_with_predictor(frame, max_output_bytes, &mut fresh_predictor)`.

CDFs must start at zero, be monotonically nondecreasing, and have a positive final total. A symbol with zero width cannot be encoded. Custom model IDs 2–255 are reserved for caller-supplied predictors; IDs 0 and 1 remain the built-in order-0/order-1 models.

The encoder and decoder must start from identical predictor configurations and update only after each symbol is coded/decoded. The frame includes the model ID but does not contain the custom checkpoint, configuration or initial state. Those are caller-supplied side information and must be counted separately in cold-start/amortized accounting.

This makes it possible to compare HDC/CfC, context trees, transformer/SSM predictors or a causal residual model under the same arithmetic coder without changing the entropy-coding implementation. It does not by itself provide a Symthaea model adapter or prove a compression benefit.


## Public telemetry data and query-aware storage

The benchmark programme now tracks public microservice observability data separately from synthetic codec fixtures:

- [CDL_PUBLIC_TELEMETRY_DATASETS_AND_QUERY_BENCH_V1_2026-11-20.md](../../docs/research/CDL_PUBLIC_TELEMETRY_DATASETS_AND_QUERY_BENCH_V1_2026-11-20.md): LO2v2 (115 runs, metrics/logs/traces) and the compact Alibaba microservice time/call-rate trace; licence, download/hash requirements, deterministic split protocol, query suite, comparator matrix and claim thresholds.
- [CDL-Bench-015 #12970](https://github.com/Luminous-Dynamics/symthaea/issues/12970): dataset ingestion and query-trace execution.

The environment used for this research could not resolve the Zenodo host, so dataset bytes were not acquired or hashed. Published record metadata/checksum values were inspected, but the archive acquisition remains explicitly pending. Do not describe the public-data benchmark as executed until actual source files, hashes and schema validation receipts exist.

The benchmark is now a workload-level test: include single-series, service-cohort, host-cohort, time-window, service+time, anomaly drill-down, cross-service and full-scan queries. Compare cold/warm-cache bytes read and decompressed, groups touched, unrelated keys decoded, rows replayed, p50/p95/p99 latency, throughput, CPU and peak RSS. ALP in Parquet/Arrow is mandatory for eligible decimal-like numeric columns; Pco/Pcodec and exact-bit Gorilla/Chimp/Elf variants remain matched baselines. OpenTelemetry Arrow is a transport/dataflow comparator, not a persisted-file codec. Zstandard seekable frames are prior art, so PZG2's contribution must be measured telemetry-aware grouping and verified query semantics rather than "chunks plus an index".
