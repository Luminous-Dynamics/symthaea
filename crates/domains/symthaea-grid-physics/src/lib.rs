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
//! - [`battery`] — battery SoC, degradation and storage accounting
//! - [`feeder`] — radial distribution power flow (linearized DistFlow)
//! - [`droop`] — P-f / Q-V droop control (simulation only)
//! - [`trip_envelope`] — voltage ride-through / must-trip envelope
//! - [`islanding`] — passive islanding detection
//! - [`generation`] — PV/wind generation models
//! - [`scheduling`] — scenario runner and per-step dispatch receipts
//! - [`qualification`] — frozen scenario corpus and independent receipt checker
//! - [`study_binding`] — cross-project study identity on each step receipt

pub mod battery;
pub mod droop;
pub mod feeder;
pub mod generation;
pub mod islanding;
pub mod qualification;
pub mod scheduling;
pub mod study_binding;
pub mod trip_envelope;
