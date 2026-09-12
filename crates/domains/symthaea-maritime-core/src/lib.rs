// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared, mission-neutral maritime autonomy primitives.
//!
//! This crate intentionally stops below mission-specific applications. It defines
//! platform state, health, authority, degraded-operation, fleet-assurance,
//! operating-mode/limit evaluation, resident docking/service, independent resource
//! envelopes, wet/dry bay safety, maintenance/requalification, regenerative
//! industrial-closure/substitution/genome/flow-support qualification and
//! progressive-failure recording contracts reusable by AUVs, USVs, research
//! vessels, logistics craft and other maritime platforms.

pub mod authority;
pub mod bay;
pub mod degraded;
pub mod docking;
pub mod fleet;
pub mod flow_support;
pub mod genome;
pub mod genome_supported;
pub mod health;
pub mod lineage_evidence;
pub mod lineage_viability;
pub mod maintenance;
pub mod operating_mode;
pub mod prelude;
pub mod regenerative;
pub mod resilience;
pub mod resources;
pub mod state;
pub mod substitution;
pub mod supported_closure;
pub mod viability_calibration;

pub use authority::*;
pub use bay::*;
pub use degraded::*;
pub use docking::*;
pub use fleet::*;
pub use flow_support::*;
pub use genome::*;
pub use genome_supported::*;
pub use health::*;
pub use lineage_evidence::*;
pub use lineage_viability::*;
pub use maintenance::*;
pub use operating_mode::*;
pub use regenerative::*;
pub use resilience::*;
pub use resources::*;
pub use state::*;
pub use substitution::*;
pub use supported_closure::*;
pub use viability_calibration::*;
