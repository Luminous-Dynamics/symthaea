// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Internationalization keys and labels for the 8D Sovereign Profile.
//!
//! Provides static English labels with i18n keys that frontend translation
//! systems (e.g., Praxis i18n.rs with 18 languages) can look up.

use crate::SovereignDimension;

/// Human-readable label for a dimension, with i18n key.
pub struct DimensionLabel {
    /// Machine key for translation lookup (e.g., "epistemic_integrity").
    pub key: &'static str,
    /// Short English name (e.g., "Epistemic Integrity").
    pub name_en: &'static str,
    /// Brief English description of what it measures.
    pub description_en: &'static str,
    /// Icon hint for UI rendering (emoji or icon name).
    pub icon: &'static str,
}

/// Labels for all 8 dimensions in canonical order.
pub const DIMENSION_LABELS: [DimensionLabel; 8] = [
    DimensionLabel {
        key: "epistemic_integrity",
        name_en: "Epistemic Integrity",
        description_en: "Your truth-telling track record",
        icon: "search",
    },
    DimensionLabel {
        key: "thermodynamic_yield",
        name_en: "Thermodynamic Yield",
        description_en: "Energy contributed to the commons",
        icon: "bolt",
    },
    DimensionLabel {
        key: "network_resilience",
        name_en: "Network Resilience",
        description_en: "Infrastructure uptime and reliability",
        icon: "wifi",
    },
    DimensionLabel {
        key: "economic_velocity",
        name_en: "Economic Velocity",
        description_en: "Healthy circulation of mutual credit",
        icon: "refresh",
    },
    DimensionLabel {
        key: "civic_participation",
        name_en: "Civic Participation",
        description_en: "Governance engagement and jury service",
        icon: "vote",
    },
    DimensionLabel {
        key: "stewardship_care",
        name_en: "Stewardship & Care",
        description_en: "Verified labor maintaining the commons",
        icon: "hands",
    },
    DimensionLabel {
        key: "semantic_resonance",
        name_en: "Semantic Resonance",
        description_en: "Alignment with community values",
        icon: "heart",
    },
    DimensionLabel {
        key: "domain_competence",
        name_en: "Domain Competence",
        description_en: "Peer-verified expertise in your field",
        icon: "star",
    },
];

/// Get the label for a specific dimension.
pub fn dimension_label(dim: SovereignDimension) -> &'static DimensionLabel {
    &DIMENSION_LABELS[dim.index()]
}

/// Tier labels with i18n keys.
pub const TIER_LABELS: [(&str, &str); 5] = [
    ("observer", "Observer"),
    ("participant", "Participant"),
    ("citizen", "Citizen"),
    ("steward", "Steward"),
    ("guardian", "Guardian"),
];
