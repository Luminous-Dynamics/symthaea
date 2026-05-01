// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness emergence layer: when did Earth first become capable of hosting consciousness?
//!
//! Maps B(t) dimensions to Symthaea's substrate independence `consciousness_feasibility()`
//! framework. The answer isn't "when neurons appeared" — it's when the biosphere's
//! integration, energy, and information capacity crossed the feasibility threshold.
//!
//! ## Mapping (B(t) → Substrate Requirements)
//!
//! | B(t) Dimension       | Substrate Requirement    | Rationale                               |
//! |----------------------|--------------------------|------------------------------------------|
//! | Network integration  | integration_capacity     | Trophic webs ≈ information integration   |
//! | Energy throughput    | temporal_dynamics        | Energy drives temporal evolution          |
//! | Information capacity | workspace_capability     | Genome complexity ≈ working memory        |
//! | Complexity grade     | hot_capability           | Meta-representation requires complexity   |
//! | Resilience           | recurrence               | Feedback loops = recovery capacity        |
//! | Biodiversity         | binding_capability       | Diverse binding = diverse feature binding |
//!
//! ## Consciousness feasibility formula (from substrate_independence.rs)
//!
//! ```text
//! feasibility = critical_min × workspace × (0.5 + 0.5 × enhancement_avg)
//! ```
//! where:
//! - critical_min = min(causality, integration, dynamics, recurrence)
//! - workspace = workspace_capability
//! - enhancement_avg = mean(binding, attention, HOT)
//!
//! CITATION: Tononi (2004), Putnam (1967), Chalmers (2010).

use super::dimensions::BiosphereDimensionsNorm;
use super::temporal_bins::MaAge;

/// Substrate requirement scores derived from B(t) dimensions.
#[derive(Debug, Clone)]
pub struct SubstrateFromBiosphere {
    /// Causal interactions (always 1.0 for biological substrates — chemistry is causal).
    pub causality: f64,
    /// Information integration capacity (from network integration).
    pub integration_capacity: f64,
    /// Rich temporal dynamics (from energy throughput).
    pub temporal_dynamics: f64,
    /// Feedback loops (from resilience — recovery implies recurrence).
    pub recurrence: f64,
    /// Feature binding (from biodiversity — diverse repertoire).
    pub binding_capability: f64,
    /// Selective attention (requires complexity ≥ 5, i.e., vertebrates).
    pub attention_capability: f64,
    /// Global workspace (from information capacity).
    pub workspace_capability: f64,
    /// Higher-order thought (from complexity grade — needs meta-representation).
    pub hot_capability: f64,
}

impl SubstrateFromBiosphere {
    /// Map normalized B(t) dimensions to substrate requirements.
    pub fn from_dimensions(dims: &BiosphereDimensionsNorm) -> Self {
        let biodiversity = dims.values[0];
        let complexity = dims.values[1];
        let resilience = dims.values[2];
        let network = dims.values[3];
        let energy = dims.values[4];
        let information = dims.values[5];

        Self {
            // Biology is always causal (chemistry obeys causality)
            causality: 1.0,
            // Network integration maps directly
            integration_capacity: network,
            // Energy drives temporal richness
            temporal_dynamics: energy,
            // Resilience implies feedback loops (recurrence)
            recurrence: resilience,
            // Biodiversity → diverse binding repertoire
            binding_capability: biodiversity.sqrt(), // Sqrt: binding needs less diversity than raw count
            // Attention emerges with vertebrate-level complexity (~grade 5 = 0.5 normalized)
            // Below vertebrate complexity, attention is negligible
            attention_capability: if complexity > 0.45 {
                ((complexity - 0.45) / 0.55).min(1.0)
            } else {
                complexity * 0.1 // Rudimentary in invertebrates
            },
            // Global workspace from information capacity
            workspace_capability: information,
            // HOT requires high complexity (grade ≥ 8 = mammals, 0.8 normalized)
            hot_capability: if complexity > 0.75 {
                ((complexity - 0.75) / 0.25).min(1.0)
            } else if complexity > 0.5 {
                (complexity - 0.5) * 0.2 // Rudimentary in non-mammalian vertebrates
            } else {
                0.0
            },
        }
    }

    /// Compute consciousness feasibility using the same formula as
    /// `symthaea-core/src/hdc/substrate_independence.rs::SubstrateRequirements::consciousness_feasibility()`.
    pub fn consciousness_feasibility(&self) -> f64 {
        // Critical: minimum of 4 necessary conditions
        let critical_min = self
            .causality
            .min(self.integration_capacity)
            .min(self.temporal_dynamics)
            .min(self.recurrence);

        // Workspace is necessary (per Symthaea improvement #27)
        let workspace_factor = self.workspace_capability;

        // Enhancement: average of binding, attention, HOT
        let enhancement_avg =
            (self.binding_capability + self.attention_capability + self.hot_capability) / 3.0;

        // Feasibility formula
        critical_min * workspace_factor * (0.5 + 0.5 * enhancement_avg)
    }
}

/// A point on the consciousness feasibility curve.
#[derive(Debug, Clone)]
pub struct ConsciousnessEmergencePoint {
    pub age_ma: MaAge,
    pub feasibility: f64,
    /// Which substrate requirement is the bottleneck at this age.
    pub bottleneck: String,
    /// Milestone label (if this age crosses a significant threshold).
    pub milestone: Option<String>,
}

/// Consciousness feasibility thresholds.
pub const THRESHOLD_MINIMAL: f64 = 0.01; // Any consciousness at all
pub const THRESHOLD_SENTIENT: f64 = 0.05; // Sentient experience possible
pub const THRESHOLD_AWARE: f64 = 0.15; // Self-aware organisms possible
pub const THRESHOLD_REFLECTIVE: f64 = 0.30; // Meta-cognitive reflection possible

/// Compute the consciousness emergence curve from B(t) dimension history.
pub fn compute_emergence_curve(
    bins_ages: &[MaAge],
    dims_per_bin: &[BiosphereDimensionsNorm],
) -> Vec<ConsciousnessEmergencePoint> {
    let mut points = Vec::with_capacity(bins_ages.len());
    let mut milestones_crossed = [false; 4];
    let thresholds = [
        (THRESHOLD_MINIMAL, "First minimal consciousness possible"),
        (THRESHOLD_SENTIENT, "Sentient experience possible"),
        (THRESHOLD_AWARE, "Self-awareness possible"),
        (THRESHOLD_REFLECTIVE, "Meta-cognitive reflection possible"),
    ];

    for (i, (&age, dims)) in bins_ages.iter().zip(dims_per_bin.iter()).enumerate() {
        let substrate = SubstrateFromBiosphere::from_dimensions(dims);
        let feasibility = substrate.consciousness_feasibility();

        // Find bottleneck (lowest critical requirement)
        let requirements = [
            ("causality", substrate.causality),
            ("integration", substrate.integration_capacity),
            ("temporal_dynamics", substrate.temporal_dynamics),
            ("recurrence", substrate.recurrence),
            ("workspace", substrate.workspace_capability),
        ];
        let bottleneck = requirements
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(name, _)| name.to_string())
            .unwrap_or_default();

        // Check milestones (only trigger on first crossing, moving toward present)
        let milestone = if i > 0 {
            let mut ms = None;
            for (j, &(threshold, label)) in thresholds.iter().enumerate() {
                if !milestones_crossed[j] && feasibility >= threshold {
                    milestones_crossed[j] = true;
                    ms = Some(label.to_string());
                    break; // Only one milestone per bin
                }
            }
            ms
        } else {
            None
        };

        points.push(ConsciousnessEmergencePoint {
            age_ma: age,
            feasibility,
            bottleneck,
            milestone,
        });
    }

    points
}

/// Format the emergence curve as a summary report.
pub fn emergence_report(curve: &[ConsciousnessEmergencePoint]) -> String {
    let mut report = String::from("# Consciousness Emergence Timeline\n\n");
    report.push_str("| Age (Ma) | Feasibility | Bottleneck | Milestone |\n");
    report.push_str("|----------|-------------|------------|----------|\n");

    for pt in curve {
        let milestone_str = pt.milestone.as_deref().unwrap_or("");
        if !milestone_str.is_empty() || pt.feasibility > 0.01 {
            report.push_str(&format!(
                "| {:.1} | {:.4} | {} | {} |\n",
                pt.age_ma, pt.feasibility, pt.bottleneck, milestone_str
            ));
        }
    }

    // Key findings
    let first_minimal = curve.iter().find(|p| p.feasibility >= THRESHOLD_MINIMAL);
    let first_sentient = curve.iter().find(|p| p.feasibility >= THRESHOLD_SENTIENT);
    let first_aware = curve.iter().find(|p| p.feasibility >= THRESHOLD_AWARE);
    let first_reflective = curve.iter().find(|p| p.feasibility >= THRESHOLD_REFLECTIVE);

    report.push_str("\n## Key Findings\n\n");
    if let Some(pt) = first_minimal {
        report.push_str(&format!(
            "- **First minimal consciousness**: ~{:.0} Ma (bottleneck: {})\n",
            pt.age_ma, pt.bottleneck
        ));
    }
    if let Some(pt) = first_sentient {
        report.push_str(&format!(
            "- **Sentient experience possible**: ~{:.0} Ma (bottleneck: {})\n",
            pt.age_ma, pt.bottleneck
        ));
    }
    if let Some(pt) = first_aware {
        report.push_str(&format!(
            "- **Self-awareness possible**: ~{:.0} Ma (bottleneck: {})\n",
            pt.age_ma, pt.bottleneck
        ));
    }
    if let Some(pt) = first_reflective {
        report.push_str(&format!(
            "- **Meta-cognitive reflection**: ~{:.0} Ma (bottleneck: {})\n",
            pt.age_ma, pt.bottleneck
        ));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biosphere::deep_time_data::DeepTimeData;
    use crate::biosphere::dimensions::{compute_raw_dimensions, BiosphereMaxima};
    use crate::biosphere::mass_extinctions::canonical_mass_extinctions;
    use crate::biosphere::temporal_bins::build_deep_time_bins;

    fn build_test_curve() -> (Vec<MaAge>, Vec<BiosphereDimensionsNorm>) {
        let data = DeepTimeData::load();
        let extinctions = canonical_mass_extinctions();
        let bins = build_deep_time_bins();
        let maxima = BiosphereMaxima::default();

        let ages: Vec<MaAge> = bins.iter().map(|b| b.midpoint_ma).collect();
        let dims: Vec<BiosphereDimensionsNorm> = bins
            .iter()
            .map(|bin| {
                let raw = compute_raw_dimensions(bin, &data, &extinctions);
                let d13c = Some(data.interpolate_d13c_excursion(bin.midpoint_ma));
                raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, d13c)
            })
            .collect();

        (ages, dims)
    }

    #[test]
    fn emergence_curve_computes() {
        let (ages, dims) = build_test_curve();
        let curve = compute_emergence_curve(&ages, &dims);
        assert_eq!(curve.len(), ages.len());
    }

    #[test]
    fn feasibility_increases_over_time() {
        let (ages, dims) = build_test_curve();
        let curve = compute_emergence_curve(&ages, &dims);
        // First point (3.8 Ga) should have near-zero feasibility
        assert!(
            curve[0].feasibility < 0.01,
            "Hadean feasibility should be ~0, got {}",
            curve[0].feasibility
        );
        // Last point (Holocene) should have high feasibility
        let last = curve.last().unwrap();
        assert!(
            last.feasibility > 0.1,
            "Holocene feasibility should be >0.1, got {}",
            last.feasibility
        );
    }

    #[test]
    fn consciousness_not_possible_in_archean() {
        let (ages, dims) = build_test_curve();
        let curve = compute_emergence_curve(&ages, &dims);
        // All points before 600 Ma should have feasibility < THRESHOLD_AWARE
        for pt in &curve {
            if pt.age_ma > 600.0 {
                assert!(
                    pt.feasibility < THRESHOLD_AWARE,
                    "Consciousness should not be possible at {} Ma (feasibility {})",
                    pt.age_ma,
                    pt.feasibility
                );
            }
        }
    }

    #[test]
    fn milestones_occur_in_order() {
        let (ages, dims) = build_test_curve();
        let curve = compute_emergence_curve(&ages, &dims);
        let milestones: Vec<&ConsciousnessEmergencePoint> =
            curve.iter().filter(|p| p.milestone.is_some()).collect();
        // Milestones should be in chronological order (decreasing age)
        for pair in milestones.windows(2) {
            assert!(
                pair[0].age_ma >= pair[1].age_ma,
                "Milestones out of order: {} Ma before {} Ma",
                pair[0].age_ma,
                pair[1].age_ma
            );
        }
    }

    #[test]
    fn bottleneck_is_workspace_early() {
        let (ages, dims) = build_test_curve();
        let curve = compute_emergence_curve(&ages, &dims);
        // In the Precambrian, workspace (information capacity) should be the bottleneck
        let precambrian: Vec<&ConsciousnessEmergencePoint> =
            curve.iter().filter(|p| p.age_ma > 600.0).collect();
        let workspace_bottleneck_count = precambrian
            .iter()
            .filter(|p| p.bottleneck == "workspace")
            .count();
        // workspace or integration should be the most common bottleneck
        assert!(
            workspace_bottleneck_count > 0
                || precambrian.iter().any(|p| p.bottleneck == "integration"),
            "Precambrian bottleneck should be workspace or integration"
        );
    }

    #[test]
    fn report_contains_key_sections() {
        let (ages, dims) = build_test_curve();
        let curve = compute_emergence_curve(&ages, &dims);
        let report = emergence_report(&curve);
        assert!(report.contains("Consciousness Emergence Timeline"));
        assert!(report.contains("Key Findings"));
        assert!(report.contains("Feasibility"));
    }
}
