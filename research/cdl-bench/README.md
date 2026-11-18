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
- E2 five-variable fixed-point structural causal generator with serialized do-intervention manifests.
- E3 factorized SCM mechanism configurations with held-out combinations assembled from individually seen mechanism components; hidden configs remain evaluator-side.
- CLI file encode/decode and an E0 length comparison against installed
  Zstandard, Brotli, XZ and gzip commands.

## Run

```sh
cd research/cdl-bench
cargo test
cargo run --release -- e0 --bytes 65536 --seed 42
cargo run --release -- causal --steps 10000 --seed 42
cargo run --release -- numeric --values 10000 --seed 42
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

The command checks exact codec round-trips and reports framed lengths for the
order-0/order-1 models, plus Zstandard when installed. It is **not** yet a
causal predictor comparison: it tests generator reproducibility and coding
plumbing only. The finite-state latent labels remain evaluator-side; the
serialized SCM stream includes the intervention manifest and observed i16 values.

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

### E3 mechanism recombination

The same causal command also emits:
- Three training SCM configurations.
- Two held-out configurations that recombine mechanism component values seen in the training configurations, while keeping the full configuration tuple unseen.
- For each held-out configuration, a paired observational and `do(x1=20000)` trace with the same seed, so the intervention comparison is counterfactual.

The hidden mechanism configuration is retained in the evaluator-side ScmTrace metadata and is not serialized into the learner-visible observation stream. The CLI labels the paired held-out streams and verifies codec round trips; it does not yet train a mechanism learner or measure transfer. A generator unit test also checks that the upstream x0 remains invariant while descendants x2 and x4 respond to the x1 intervention in each held-out world. This is a dataset-generation fixture only.

## Plugging in a custom predictor

The library exposes:
- `ByteCdf = [u32; 257]`: cumulative integer frequencies for all 256 byte symbols.
- `BytePredictor`: a stable model ID, `cumulative_frequency()`, and `observe(decoded_symbol)`.
- `compress_with_predictor(input, &mut predictor)`.
- `decompress_with_predictor(frame, max_output_bytes, &mut fresh_predictor)`.

CDFs must start at zero, be monotonically nondecreasing, and have a positive final total. A symbol with zero width cannot be encoded. Custom model IDs 2–255 are reserved for caller-supplied predictors; IDs 0 and 1 remain the built-in order-0/order-1 models.

The encoder and decoder must start from identical predictor configurations and update only after each symbol is coded/decoded. The frame includes the model ID but does not contain the custom checkpoint, configuration or initial state. Those are caller-supplied side information and must be counted separately in cold-start/amortized accounting.

This makes it possible to compare HDC/CfC, context trees, transformer/SSM predictors or a causal residual model under the same arithmetic coder without changing the entropy-coding implementation. It does not by itself provide a Symthaea model adapter or prove a compression benefit.
