# symthaea-probe-stream

Bridges a trained linear probe matrix **W** (offline distilled from a Foundation Model oracle)
with Symthaea's 16,384-dimensional [`ContinuousHV`] hypervector space, turning any embedding
source into a real-time hypervector stream ready for the **Holographic Liquid Brain** (HDC-LTC
cognitive loop).

## Architecture

```text
[Sensory Oracle]  ─►  [Embedding  Vec<f32>]
                              │
                        [ProbeMatrix W]   ← load_flat_f32() / from_zeros()
                              │  W @ e   (embedding_dim × hdc_dim dot products)
                              ▼
                       [ContinuousHV I(t)]  ← 16,384-dim bipolar hypervector
                              │
                   [TrajectoryRecorder]    ← optional: collect for distillation
                              │
                       [LTC Cognitive Loop]
```

## Quick Start

```rust,no_run
use symthaea_probe_stream::{ProbeMatrix, ProbeStreamAdapter};
use symthaea_probe_stream::backends::MockBackend;

// Zero probe (all +1.0 output) for testing, or load from file:
let probe = ProbeMatrix::from_zeros(768, 16_384);

let backend = MockBackend::new(768, 42);
let mut adapter = ProbeStreamAdapter::new(probe, backend);

// Pull the next hypervector
let hv = adapter.next_hv(0.0).unwrap();
assert_eq!(hv.dim(), 16_384);
```

## Loading a Trained Probe

```rust,no_run
use symthaea_probe_stream::ProbeMatrix;

// Raw flat f32 binary (row-major: hdc_dim rows × embedding_dim cols)
let probe = ProbeMatrix::load_flat_f32_file(
    "weights/probe_768_16384.bin",
    768,    // embedding_dim
    16_384, // hdc_dim
).unwrap();
```

## Backends

| Backend        | Description                                         |
|----------------|-----------------------------------------------------|
| `MockBackend`  | Deterministic xorshift64 RNG – unit tests / CI      |
| `OllamaBackend`| Calls Ollama HTTP API (`POST /api/embed`) – no async|

## Trajectory Recording

```rust,no_run
use symthaea_probe_stream::{ProbeMatrix, ProbeStreamAdapter, TrajectoryRecorder};
use symthaea_probe_stream::backends::MockBackend;

let probe   = ProbeMatrix::from_zeros(64, 256);
let backend = MockBackend::new(64, 1);
let adapter = ProbeStreamAdapter::new(probe, backend);
let rec     = TrajectoryRecorder::new(256, 1024);

let mut ra = adapter.with_recorder(rec);
for t in 0..100 {
    ra.next_hv(t as f64 * 0.01).unwrap();
}
let (_, recorder) = ra.finish();
recorder.save_to_file(std::path::Path::new("traj.bin")).unwrap();
```

## License

AGPL-3.0-or-later · Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
