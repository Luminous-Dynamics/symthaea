// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![deny(unsafe_code)]

//! # Symthaea Telemetry Sink
//!
//! Wires the 7 canonical Time-Waterfall scalar metrics from live Symthaea
//! subsystems into a [`WaterfallBuffer`].
//!
//! ## The 7 Canonical Metrics
//!
//! | # | Key | Source |
//! |---|-----|--------|
//! | 1 | `phi` | `symthaea-phi-oracle` → `IntegrationReport::integration_index` |
//! | 2 | `fep_prediction_error` | `symthaea-fep` → `FreeEnergyComponents::prediction_error` |
//! | 3 | `workspace_activation` | `symthaea-workspace` → `GlobalWorkspace::current_focus.magnitude` |
//! | 4 | `hot_confidence` | `symthaea-fep` → `HiddenState::confidence()` |
//! | 5 | `anomaly_score` | Computed from phi + prediction_error divergence |
//! | 6 | `memory_pressure` | `symthaea-fep` → `HiddenState::total_uncertainty()` |
//! | 7 | `mip_instability` | `symthaea-phi-oracle` → `IntegrationReport::normalized_index` |
//!
//! ## Design
//!
//! The sink is intentionally thin. It reads from live subsystem state snapshots
//! and emits [`ProjectionFrame`]s into the buffer. It does not own or run the
//! subsystems — it observes them.
//!
//! ## Epistemic Discipline
//!
//! Confidence is computed honestly from the data:
//! - High anomaly_score → lower frame confidence
//! - Low phi + high prediction_error → anomaly tags
//! - Uniform metrics over time → `SuspiciousFalseGreen` warning

pub mod frame_builder;
pub mod metric_snapshot;
pub mod sink;

pub use frame_builder::FrameBuilder;
pub use metric_snapshot::MetricSnapshot;
pub use sink::TelemetrySink;
