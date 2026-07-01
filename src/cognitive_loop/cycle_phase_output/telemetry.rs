// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Output-phase telemetry population.
//!
//! The submodules are split by ownership:
//! - `core`: base cycle metadata, consciousness, affect, attention, ethics.
//! - `modulation`: feedback-loop and adaptive modulation flags.
//! - `managers`: manager/drives/swarm/domain telemetry.
//! - `bridges`: external bridge and feature-gated telemetry.

mod bridges;
mod core;
mod managers;
mod modulation;

mod prelude {
    pub(super) use super::super::super::phase_results::{
        DynamicsPhaseResult, FeedbackPhaseResult, PerceptionPhaseResult,
    };
    pub(super) use super::super::super::thresholds::*;
    pub(super) use super::super::super::{CognitiveLoopService, CycleMetadata};
    pub(super) use std::mem;
}
