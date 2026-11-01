# symthaea-edge-core

A separate, bounded, allocation-free substrate for MCU-oriented Symthaea experiments.

This crate is intentionally **not** the full HDC-LTC runtime. It combines two compact representations:

- a small continuous liquid state in signed Q15 fixed point;
- packed binary HDC vectors for binding, permutation, bundling and associative exemplar recall.

The library is `no_std`, has no dependencies, forbids unsafe code, and uses fixed-size arrays. It contains no allocator-backed collection or runtime capacity growth.

## Primitives

### Packed binary HDC

`PackedHv<const WORDS: usize>` stores `WORDS × 64` bits in `[u64; WORDS]`.

- bind: XOR;
- similarity: signed bipolar Hamming similarity in Q15;
- permutation: cyclic left rotation over the entire bit string;
- bundle: per-bit majority, with exact ties resolving to zero.

A 16,384-bit vector (`WORDS = 256`) uses exactly 2,048 bytes, compared with 65,536 bytes for 16,384 `f32` values.

### Fixed-codebook scalar encoder

`ScalarLevelEncoder<LEVELS, FEATURES, WORDS>` accepts an explicit array of value-level hypervectors, fixed per-feature rotation offsets and an array of pre-quantized feature levels. It permutes the selected level vector for each feature and majority-bundles the results. Exact ties resolve to zero.

The encoder does not choose feature scaling/quantization, generate random hypervectors or hide codebook state. Those remain explicit benchmark artifacts and must be content-addressed. The fixed codebook and rotation array have their own payload estimator.

### Fixed-capacity associative memory

`PackedHdcMemory<CLASSES, SLOTS, WORDS>` allocates all exemplars and class metadata inline. Insertion returns `CapacityFull` when full. Round-robin replacement is available only through the explicitly named `insert_or_replace_oldest` method.

Prediction uses nearest Hamming exemplar. Equal-distance ties retain the first exemplar in the lowest class index.

### Q15 liquid state

`FixedPointLiquidState<DIM>` stores `DIM` signed 16-bit values. `evolve_q15` computes:

```text
state_next = round(((32768 - alpha) * state + alpha * equilibrium) / 32768)
```

The caller can pass `alpha_q15` directly or use `alpha_from_ratio_q16` / `alpha_from_dt_tau_us` from the fixed-point time-constant module. The Q16.16 ratio table spans [0,12] at 1/32 intervals and linearly interpolates precomputed Q15 samples; ratios above 12 saturate to one. Its conservative approximation budget is below 5 Q15 LSBs (4 interpolation + 0.5 table rounding + less than 0.5 ratio quantization). The microsecond convenience API uses u32 inputs and u64 integer division; the preferred tight-loop path is to supply a precomputed Q16.16 ratio. This is a base scalar tau policy, not a complete state-dependent or learned CfC tau model.

## Example composition

```rust
use symthaea_edge_core::{
    BoundedEdgeCore, PackedHv, ScalarLevelEncoder,
};

type EdgeCore = BoundedEdgeCore<128, 4, 16, 256>;

let mut core = EdgeCore::new().expect("non-zero fixed capacity");
let sensory_equilibrium = [0i16; 128];
core.update_liquid_state_from_dt_tau_us(&sensory_equilibrium, 10, 10).unwrap();

let codebook = [PackedHv::<256>::zero(); 2];
let encoder = ScalarLevelEncoder::<2, 3, 256>::new(codebook, [0, 4096, 8192])
    .expect("non-zero encoder dimensions");
let feature_levels = [0, 1, 0];
let exemplar = encoder.encode(&feature_levels).unwrap();
core.remember(0, exemplar).unwrap();
let result = core.classify_features(&encoder, &feature_levels).unwrap();
```

The example's memory is statically bounded. The 64 exemplar vectors occupy 128 KiB; liquid state occupies 256 bytes. Two per-class `usize` arrays add target-dependent metadata (32 bytes on a 32-bit MCU, 64 bytes on a 64-bit host). The two-level, three-feature codebook occupies another 4 KiB plus three `usize` rotation offsets. Struct padding and caller-owned inputs/stack are not part of the component payload count.

## What this does not establish

- It is not a complete CfC network or complete Symthaea runtime.
- It does not currently map elapsed time and tau to the Q15 interpolation coefficient.
- It does not include a sensor encoder, classifier training loop, drift detector or confidence calibration.
- It does not prove that any profile fits a complete MCU firmware image.
- It does not show an accuracy, latency, energy or SOTA advantage over TinyML.

The next validation sequence is: compile for host and a 32-bit no-std target; inspect the target linker map; add a tested dt/tau coefficient policy; then compare this profile against optimized static/adaptive TinyML under the exact same event stream and energy protocol.

## Build

```bash
cargo test -p symthaea-edge-core
cargo check -p symthaea-edge-core --target thumbv7em-none-eabihf
```

A target toolchain and target specification must be installed for the second command. Tests are authored but have not yet been compiled in the current authoring environment.
