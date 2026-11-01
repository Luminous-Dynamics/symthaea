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

### Fixed-capacity associative memory

`PackedHdcMemory<CLASSES, SLOTS, WORDS>` allocates all exemplars and class metadata inline. Insertion returns `CapacityFull` when full. Round-robin replacement is available only through the explicitly named `insert_or_replace_oldest` method.

Prediction uses nearest Hamming exemplar. Equal-distance ties retain the first exemplar in the lowest class index.

### Q15 liquid state

`FixedPointLiquidState<DIM>` stores `DIM` signed 16-bit values. `evolve_q15` computes:

```text
state_next = round(((32768 - alpha) * state + alpha * equilibrium) / 32768)
```

The caller passes `alpha_q15` as unsigned Q0.15 in [0, 32768]. This crate does **not** yet calculate `1 - exp(-dt/tau)`; selecting or approximating that coefficient, defining tau dynamics, and proving the error bound are separate work.

## Example composition

```rust
use symthaea_edge_core::{
    BoundedEdgeCore, PackedHv, Q15_ONE,
};

type EdgeCore = BoundedEdgeCore<128, 4, 16, 256>;

let mut core = EdgeCore::new().expect("non-zero fixed capacity");
let sensory_equilibrium = [0i16; 128];
core.update_liquid_state(&sensory_equilibrium, Q15_ONE)?;
let exemplar = PackedHv::<256>::zero();
core.remember(0, exemplar)?;
let result = core.classify(&exemplar);
# Ok::<(), symthaea_edge_core::MemoryError>(())
```

The example's memory is statically bounded. The 64 exemplar vectors occupy 128 KiB; liquid state occupies 256 bytes. Two per-class `usize` arrays add target-dependent metadata (32 bytes on a 32-bit MCU, 64 bytes on a 64-bit host). Struct padding and caller-owned inputs/stack are not part of the component payload count.

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
