// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-biometrics
//!
//! Behavioral biometrics for Symtropy: encodes mouse/keyboard telemetry into
//! HDC stress vectors using existing symthaea-core infrastructure.
//!
//! # Architecture
//!
//! ```text
//! Mouse (x, y, velocity)  ──→ GridEncoder + TemporalEncoder ──→ spatial HV
//! Keystroke timing        ──→ TemporalEncoder                ──→ temporal HV
//! Mouse velocity series   ──→ EchoStateNetwork               ──→ surprise (prediction error)
//!                                                               │
//!                              ┌────────────────────────────────┘
//!                              ▼
//!                         StressVector (arousal, valence, dominance)
//!                              │
//!                              ▼
//!                         PlayerStressModel (allostatic accumulation/decay)
//!                              │
//!                              ▼
//!                         SensorNudges (DA, NE, 5-HT, OT deltas)
//! ```
//!
//! # Reused Infrastructure
//!
//! | Component | Source | Purpose |
//! |-----------|--------|---------|
//! | `TemporalEncoder` | symthaea-core/hdc | Keystroke inter-arrival → phase vectors |
//! | `GridEncoder` | symthaea-core/hdc | Mouse position → spatial vectors |
//! | `EchoStateNetwork` | symthaea-core/hdc | Velocity surprise detection |

#![deny(unsafe_code)]

pub mod input_telemetry;
#[cfg(feature = "muse-bridge")]
pub mod muse_bridge;
pub mod stress_model;
