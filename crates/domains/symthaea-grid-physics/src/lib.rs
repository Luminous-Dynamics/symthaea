// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Standalone electrical-grid physics for microgrid simulation.
//!
//! Structured like `mycelix-space/lib/orbital-mechanics`: a self-contained
//! physics library with its own tests, no dependency on the consciousness
//! stack (`symthaea-core`). Consumed as an optional simulator backend by
//! `symthaea-infrastructure`; see PLANETARY_ENERGY_COORDINATION_PLAN_2026-07-06.md
//! Phase 1 (physics + backend wiring) and Phase 2 (generation + scheduling).
//!
//! Modules:
//! - [`battery`] — battery SoC, round-trip efficiency, cycle-based degradation
//! - [`feeder`] — radial distribution power flow (linearized DistFlow)
//! - [`droop`] — P-f / Q-V droop control (grid-forming inverter behavior)
//! - [`trip_envelope`] — voltage ride-through / must-trip envelope (simulation only)
//! - [`islanding`] — passive islanding detection (ROCOF, threshold), incl. the
//!   real non-detection-zone limitation
//! - [`generation`] — PVWatts-style solar + IEC 61400-12-style wind generation
//! - [`scheduling`] — storage-scheduling scenario harness, scored on cost /
//!   unserved energy / battery cycles, with a naive baseline and a first
//!   reserve-aware advisor policy

pub mod battery;
pub mod droop;
pub mod feeder;
pub mod generation;
pub mod islanding;
pub mod scheduling;
pub mod trip_envelope;
