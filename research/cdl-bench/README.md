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
- Exact IEEE-754 f64 raw/XOR-varint/reusable XOR-window framing, bit-preserving test fixtures, a fixed-chunk best-of-six size oracle, and a decoder-synchronous historical-cost online selector.
- SCM-specific typed-field, temporal-lag, oracle-parent, hybrid and
  shuffled-parent byte predictors, plus fixed-share and CUSUM change-point
  mixtures, all using the same arithmetic coder and exact decoder path.

## Exact IEEE-754 f64 reference track

The `float` command generates constant, smooth-ramp, oscillating,
high-precision and random-bit-pattern streams. It compares:
- raw IEEE-754 binary64 bit patterns;
- XOR with the previous u64 bit pattern, encoded as a canonical u64 varint;
- XOR with the previous u64 bit pattern, defining and reusing a leading/trailing significant-bit window;
- order-0 and order-1 byte CDF models;
- Zstandard on the original raw little-endian bytes, when installed.

The single-stream frames use a 32-byte typed header, inner CDL frame and
raw/inner CRC-32 checksums. Decoding reconstructs via `f64::from_bits`; it
does not perform floating-point subtraction or normalization. Tests compare
`to_bits()` for each sample, including NaN payloads, infinities, signed zero
and subnormals.

The command also emits two segmented rows at chunk sizes 64, 256 and 1,024:

- `segment_portfolio` is an **offline best-of-six size oracle**. For each
  complete chunk, the encoder tries Raw/Order0, Raw/Order1, XOR-varint/Order0,
  XOR-varint/Order1, XOR-window/Order0 and XOR-window/Order1,
  then keeps the smallest complete F64C child frame. Every child header and
  CRC, plus the outer F64S header, counts toward the total size. The decoder
  independently recomputes all six canonical child frames after
  reconstruction and rejects a child that is not the exact smallest choice.
  The selector sees the current chunk before choosing and is not a streaming
  algorithm, online predictive gate, or WZS reproduction.
- `online_historical_cost` chooses the current chunk from the average complete
  child-frame sizes for all six candidates measured on **previous** chunks only.
  The first chunk uses Raw/Order0. After each chunk, both encoder and decoder
  recompute all six candidate frame sizes from that chunk to update the same
  integer history.
  The selected transform/model is recorded in the child's F64C header and the
  decoder rejects a child whose IDs do not match its historical policy. There
  is no additional selector symbol or intervention information.

The online policy is a compute-heavy diagnostic: it makes six candidate
measurements per chunk on both sides. It is not yet optimized for throughput.
Both policies use the 32-byte F64S outer header and count each F64C child
header/CRC. They provide different reference points: an offline size oracle
and a decoder-reproducible past-only selection policy.

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

The strict-warning reference run covered 57,105 values and rejected seven
malformed marker/span/payload cases. Transform-only byte counts versus
XOR-varint were:
- edge values: 78 → 60 bytes (-23.08%);
- smooth bit increments: 2,329 → 6,155 (+164.28%);
- oscillating values: 20,479 → 18,430 (-10.01%);
- random patterns: 474,699 → 450,002 (-5.20%);
- repeated values: 3,016 → 3,006 (-0.33%).

These counts are before the byte entropy coder and all framing. The smooth
sequence is a significant regression, so this remains a selectable specialist,
not a universal default or a claim of Gorilla/Chimp parity.

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

## Plugging in a custom predictor

The library exposes:
- `ByteCdf = [u32; 257]`: cumulative integer frequencies for all 256 byte symbols.
- `BytePredictor`: a stable model ID, `cumulative_frequency()`, and `observe(decoded_symbol)`.
- `compress_with_predictor(input, &mut predictor)`.
- `decompress_with_predictor(frame, max_output_bytes, &mut fresh_predictor)`.

CDFs must start at zero, be monotonically nondecreasing, and have a positive final total. A symbol with zero width cannot be encoded. Custom model IDs 2–255 are reserved for caller-supplied predictors; IDs 0 and 1 remain the built-in order-0/order-1 models.

The encoder and decoder must start from identical predictor configurations and update only after each symbol is coded/decoded. The frame includes the model ID but does not contain the custom checkpoint, configuration or initial state. Those are caller-supplied side information and must be counted separately in cold-start/amortized accounting.

This makes it possible to compare HDC/CfC, context trees, transformer/SSM predictors or a causal residual model under the same arithmetic coder without changing the entropy-coding implementation. It does not by itself provide a Symthaea model adapter or prove a compression benefit.
