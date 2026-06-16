// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neuromodulator-driven individual differences and dissociation analysis.
//!
//! Runs the benchmark battery under different neuromodulator profiles
//! to find double dissociations (e.g., High-DA enhances WCST but impairs CPT).
//!
//! Requires the `symthaea-backend` feature.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A neuromodulator profile (pharmacological manipulation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodProfile {
    /// Profile name (e.g., "Balanced", "High-DA").
    pub name: String,
    /// Dopamine clamp level (None = natural dynamics).
    pub da: Option<f32>,
    /// Noradrenaline clamp level.
    pub ne: Option<f32>,
    /// Serotonin clamp level.
    pub sht: Option<f32>,
    /// Acetylcholine clamp level.
    pub ach: Option<f32>,
    /// Human phenotype description.
    pub phenotype: String,
}

impl NeuromodProfile {
    /// The 5 standard profiles.
    pub fn standard_profiles() -> Vec<Self> {
        vec![
            Self {
                name: "Balanced".to_string(),
                da: None,
                ne: None,
                sht: None,
                ach: None,
                phenotype: "Typical healthy adult".to_string(),
            },
            Self {
                name: "High-DA".to_string(),
                da: Some(0.85),
                ne: None,
                sht: None,
                ach: None,
                phenotype: "Enhanced dopaminergic tone (reward sensitivity, flexibility)"
                    .to_string(),
            },
            Self {
                name: "High-NE".to_string(),
                da: None,
                ne: Some(0.85),
                sht: None,
                ach: None,
                phenotype: "Enhanced noradrenergic tone (arousal, alerting)".to_string(),
            },
            Self {
                name: "High-5HT".to_string(),
                da: None,
                ne: None,
                sht: Some(0.85),
                ach: None,
                phenotype: "Enhanced serotonergic tone (confidence, impulse control)".to_string(),
            },
            Self {
                name: "High-ACh".to_string(),
                da: None,
                ne: None,
                sht: None,
                ach: Some(0.85),
                phenotype: "Enhanced cholinergic tone (focused attention, memory encoding)"
                    .to_string(),
            },
        ]
    }
}

/// Performance on a benchmark under a specific profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePerformance {
    /// Profile name.
    pub profile: String,
    /// Benchmark name.
    pub benchmark: String,
    /// Absolute accuracy.
    pub accuracy: f64,
    /// Relative to balanced baseline (>1.0 = enhancement, <1.0 = impairment).
    pub relative_performance: f64,
}

/// A dissociation: one profile enhances benchmark A but impairs benchmark B.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dissociation {
    /// Profile causing the dissociation.
    pub profile: String,
    /// Enhanced benchmark.
    pub enhanced: String,
    /// Enhanced relative performance.
    pub enhanced_score: f64,
    /// Impaired benchmark.
    pub impaired: String,
    /// Impaired relative performance.
    pub impaired_score: f64,
}

/// Matrix of profiles × benchmarks → relative performance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DissociationMatrix {
    /// Performance entries: profiles × benchmarks.
    pub entries: Vec<ProfilePerformance>,
    /// Detected dissociations.
    pub dissociations: Vec<Dissociation>,
}

impl DissociationMatrix {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            dissociations: Vec::new(),
        }
    }

    /// Add a performance entry.
    pub fn add_entry(&mut self, entry: ProfilePerformance) {
        self.entries.push(entry);
    }

    /// Find dissociations: profile enhances A (>1.05) and impairs B (<0.95).
    pub fn find_dissociations(&mut self) {
        self.dissociations.clear();

        // Group by profile
        let mut by_profile: BTreeMap<String, Vec<&ProfilePerformance>> = BTreeMap::new();
        for entry in &self.entries {
            by_profile
                .entry(entry.profile.clone())
                .or_default()
                .push(entry);
        }

        for (profile, entries) in &by_profile {
            if profile == "Balanced" {
                continue; // Skip baseline
            }
            for enhanced in entries.iter().filter(|e| e.relative_performance > 1.05) {
                for impaired in entries.iter().filter(|e| e.relative_performance < 0.95) {
                    self.dissociations.push(Dissociation {
                        profile: profile.clone(),
                        enhanced: enhanced.benchmark.clone(),
                        enhanced_score: enhanced.relative_performance,
                        impaired: impaired.benchmark.clone(),
                        impaired_score: impaired.relative_performance,
                    });
                }
            }
        }
    }

    /// Format as markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        // Performance table
        out.push_str("## Profile × Benchmark Performance\n\n");
        out.push_str("| Profile | Benchmark | Accuracy | Relative |\n");
        out.push_str("|---------|-----------|----------|----------|\n");
        for e in &self.entries {
            out.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} |\n",
                e.profile, e.benchmark, e.accuracy, e.relative_performance
            ));
        }

        // Dissociations
        if !self.dissociations.is_empty() {
            out.push_str("\n## Dissociations\n\n");
            out.push_str("| Profile | Enhanced | Score | Impaired | Score |\n");
            out.push_str("|---------|----------|-------|----------|-------|\n");
            for d in &self.dissociations {
                out.push_str(&format!(
                    "| {} | {} | {:.3} | {} | {:.3} |\n",
                    d.profile, d.enhanced, d.enhanced_score, d.impaired, d.impaired_score
                ));
            }
        }

        out
    }
}

/// Expected dissociations for validation.
pub fn expected_dissociations() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // (profile, expected_enhancement, expected_impairment)
        ("High-DA", "Executive::WCST", "SustainedAttention::CPT"),
        ("High-NE", "SustainedAttention::PVT", "Motor::FittsLaw"),
        (
            "High-5HT",
            "Metacognition::Calibration",
            "Creativity::AlternateUses",
        ),
        (
            "High-ACh",
            "Attention::VisualSearch",
            "Creativity::RemoteAssociates",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_profiles_count() {
        let profiles = NeuromodProfile::standard_profiles();
        assert_eq!(profiles.len(), 5);
        assert!(profiles[0].da.is_none()); // Balanced
        assert_eq!(profiles[1].da, Some(0.85)); // High-DA
    }

    #[test]
    fn test_dissociation_detection() {
        let mut matrix = DissociationMatrix::new();
        // High-DA enhances WCST (1.15) but impairs CPT (0.85)
        matrix.add_entry(ProfilePerformance {
            profile: "High-DA".to_string(),
            benchmark: "WCST".to_string(),
            accuracy: 0.80,
            relative_performance: 1.15,
        });
        matrix.add_entry(ProfilePerformance {
            profile: "High-DA".to_string(),
            benchmark: "CPT".to_string(),
            accuracy: 0.70,
            relative_performance: 0.85,
        });
        matrix.find_dissociations();
        assert_eq!(matrix.dissociations.len(), 1);
        assert_eq!(matrix.dissociations[0].enhanced, "WCST");
        assert_eq!(matrix.dissociations[0].impaired, "CPT");
    }

    #[test]
    fn test_no_dissociation_for_balanced() {
        let mut matrix = DissociationMatrix::new();
        matrix.add_entry(ProfilePerformance {
            profile: "Balanced".to_string(),
            benchmark: "WCST".to_string(),
            accuracy: 0.80,
            relative_performance: 1.15,
        });
        matrix.find_dissociations();
        assert!(matrix.dissociations.is_empty());
    }

    #[test]
    fn test_markdown_output() {
        let mut matrix = DissociationMatrix::new();
        matrix.add_entry(ProfilePerformance {
            profile: "High-DA".to_string(),
            benchmark: "WCST".to_string(),
            accuracy: 0.80,
            relative_performance: 1.15,
        });
        let md = matrix.to_markdown();
        assert!(md.contains("High-DA"));
        assert!(md.contains("WCST"));
    }

    #[test]
    fn test_expected_dissociations_defined() {
        let expected = expected_dissociations();
        assert_eq!(expected.len(), 4);
        assert_eq!(expected[0].0, "High-DA");
    }
}
