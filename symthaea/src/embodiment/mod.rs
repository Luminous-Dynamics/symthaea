// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Embodiment Module — Physical world interaction and robotics bridges.

pub mod detritivore_bridge;
pub mod helios_bridge;
pub mod metabolic_conductor;
pub mod robotics_bridge;

pub use detritivore_bridge::{DetritivoreEmbodiment, DetritivoreTelemetry};
pub use helios_bridge::{HeliosEmbodiment, HeliosTelemetry};
pub use metabolic_conductor::{MetabolicConductor, MetabolicHomeostasis};
pub use robotics_bridge::{RoboticAgent, RoboticStepResult};
