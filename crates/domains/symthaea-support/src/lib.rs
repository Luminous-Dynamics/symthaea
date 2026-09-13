// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Support — IT support intelligence sub-crate
//!
//! Provides triage, diagnostics, knowledge management, privacy scrubbing,
//! action engine, predictive engine, and an evidence-bound live system-state graph
//! for universal IT support.

pub mod actions;
pub mod diagnostics;
pub mod federation;
pub mod knowledge;
pub mod predictive;
pub mod privacy;
pub mod scrubber;
pub mod system_state;
pub mod system_state_identity;
pub mod system_state_revision;
pub mod telemetry;
pub mod triage;
pub mod types;

pub use system_state::{
    CurrentnessStatusV1, EntityId, EntityKindV1, ObservationClockV1, ObservationId,
    ObservationProvenanceV1, ObservationSourceKindV1, RelationId, RelationKindV1, StateValueV1,
    SystemEntityV1, SystemObservationV1, SystemRelationV1, SystemStateGraphError,
    SystemStateGraphV1,
};
pub use system_state_identity::{
    SystemEntityRefV1, SystemEnvironmentIdV1, SystemObservationRefV1, SystemRelationRefV1,
    SystemStateGraphScopeV1, SystemStateIdentityErrorV1,
};
pub use system_state_revision::{
    SystemEntityRevisionRefV1, SystemGraphRevisionRefV1, SystemRelationRevisionRefV1,
    SystemStateRevisionErrorV1, SystemStateRevisionScopeExtV1,
};
