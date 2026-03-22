// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interoceptive tags for internal-state memory.

/// Canonical interoceptive tags used for memory topics and recall routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteroceptionTag {
    /// CPU/memory metabolic load — maps from cycle throughput and error trend
    ThermodynamicLoad,
    /// Infrastructure stress — maps from lock poisoning, task panics, DB failures
    InfrastructureStress,
}

impl InteroceptionTag {
    pub fn as_topic(self) -> &'static str {
        match self {
            InteroceptionTag::ThermodynamicLoad => "interoception:thermodynamic_load",
            InteroceptionTag::InfrastructureStress => "interoception:infrastructure_stress",
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            InteroceptionTag::ThermodynamicLoad => "ThermodynamicLoad",
            InteroceptionTag::InfrastructureStress => "InfrastructureStress",
        }
    }
}
