// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-probe-stream
//!
//! Bridges a trained linear probe matrix W (offline distilled from a heavy
//! Foundation Model / Oracle) with Symthaea's 16,384-dimensional
//! `ContinuousHV` hypervector space.
//!
//! ## Architecture
//!
//! ```text
//! [Sensory Oracle] ─► [Embedding Vec<f32>]
//!                           │
//!                     [ProbeMatrix W]   ← load_flat_f32_file() or from_zeros()
//!                           │ W @ e  (hdc_dim × embedding_dim dot products)
//!                           ▼
//!                    [ContinuousHV I(t)]  ← 16,384-dim bipolar hypervector
//!                           │
//!                  [TrajectoryRecorder]  ← optional: collect for distillation
//!                           │
//!                    [LTC Cognitive Loop]
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use symthaea_probe_stream::{ProbeMatrix, ProbeStreamAdapter};
//! use symthaea_probe_stream::backends::MockBackend;
//!
//! // Load probe (flat f32 binary or from_zeros for testing)
//! let probe = ProbeMatrix::from_zeros(768, 16_384);
//!
//! // Wrap with a backend
//! let backend = MockBackend::new(768, 42);
//! let mut adapter = ProbeStreamAdapter::new(probe, backend);
//!
//! // Pull the next hypervector
//! let hv = adapter.next_hv(0.0).unwrap();
//! assert_eq!(hv.dim(), 16_384);
//! ```

pub mod adapter;
pub mod backends;
pub mod probe;
pub mod recorder;

pub use adapter::{ProbeStreamAdapter, RecordingAdapter};
pub use probe::ProbeMatrix;
pub use recorder::TrajectoryRecorder;
