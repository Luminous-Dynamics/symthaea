// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Discovery Bridge — Connecting Symthaea Physics to DeSci Claims
//!
//! When Symthaea's physics simulation produces a prediction, this module
//! packages it as a DeSci-compatible epistemic claim with LEM classification:
//!
//! - **E-axis**: Simulation confidence → E0 (speculation) through E4 (reproducible)
//! - **N-axis**: Based on equation catalog coverage → N0 (personal) through N3 (axiomatic)
//! - **M-axis**: Based on physical timescale → M0 (ephemeral) through M3 (foundational)
//!
//! This enables the Mycelix DeSci protocol to track, fact-check, and market
//! physics predictions alongside human scientific claims.

use crate::types::{PhysicsDomain, SearchResult};
use serde::{Deserialize, Serialize};

/// A physics prediction packaged as a DeSci-compatible claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsDiscovery {
    /// Human-readable description of the prediction
    pub description: String,
    /// The physics domain this prediction relates to
    pub domain: PhysicsDomain,
    /// Simulation confidence (0.0-1.0) — how certain is the prediction?
    pub simulation_confidence: f64,
    /// Nearest known equation (from catalog search)
    pub nearest_analog: Option<String>,
    /// Similarity to nearest known equation (0.0-1.0)
    pub analog_similarity: f64,
    /// LEM Cube classification
    pub lem: LEMClassification,
    /// Whether this prediction is novel (no close analog in catalog)
    pub is_novel: bool,
    /// Tags for knowledge graph indexing
    pub tags: Vec<String>,
}

/// LEM Cube epistemic classification (mirrors mycelix-desci EmpiricalAxis/NormativeAxis/MaterialityAxis).
///
/// Kept as simple u8 tiers to avoid cross-crate dependency on mycelix-desci
/// (which is a separate non-Symthaea workspace).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LEMClassification {
    /// E-axis: 0=Null, 1=Testimonial, 2=PrivatelyVerifiable, 3=CryptographicallyProven, 4=PubliclyReproducible
    pub empirical: u8,
    /// N-axis: 0=Personal, 1=Communal, 2=Network, 3=Axiomatic
    pub normative: u8,
    /// M-axis: 0=Ephemeral, 1=Temporal, 2=Persistent, 3=Foundational
    pub materiality: u8,
}

impl LEMClassification {
    /// Label for the E-axis tier.
    pub fn empirical_label(&self) -> &'static str {
        match self.empirical {
            0 => "E0: Null (unverifiable)",
            1 => "E1: Testimonial (simulation output)",
            2 => "E2: Privately Verifiable (reproducible simulation)",
            3 => "E3: Cryptographically Proven (hash-committed)",
            4 => "E4: Publicly Reproducible (open source + data)",
            _ => "Unknown",
        }
    }

    /// Label for the N-axis tier.
    pub fn normative_label(&self) -> &'static str {
        match self.normative {
            0 => "N0: Personal (single simulation)",
            1 => "N1: Communal (peer-reviewed simulation)",
            2 => "N2: Network (replicated by independent code)",
            3 => "N3: Axiomatic (mathematical proof)",
            _ => "Unknown",
        }
    }
}

/// Classify a physics prediction into a DeSci discovery claim.
///
/// Uses the catalog search results and simulation metadata to determine
/// the appropriate LEM Cube classification.
pub fn classify_prediction(
    description: &str,
    domain: PhysicsDomain,
    simulation_confidence: f64,
    catalog_results: &[SearchResult],
    is_open_source: bool,
) -> PhysicsDiscovery {
    let (nearest, similarity) = catalog_results
        .first()
        .map(|r| (Some(r.name.clone()), r.score as f64))
        .unwrap_or((None, 0.0));

    // E-axis: simulation confidence + reproducibility
    let empirical = if is_open_source && simulation_confidence > 0.8 {
        4 // E4: Publicly reproducible (open code + high confidence)
    } else if simulation_confidence > 0.6 {
        2 // E2: Privately verifiable (simulation reproducible with same code)
    } else if simulation_confidence > 0.3 {
        1 // E1: Testimonial (simulation ran, but low confidence)
    } else {
        0 // E0: Null (speculative)
    };

    // N-axis: based on catalog coverage (known physics = higher authority)
    let normative = if similarity > 0.9 {
        3 // N3: Axiomatic (matches known physics exactly)
    } else if similarity > 0.6 {
        2 // N2: Network (well-supported by known equations)
    } else if similarity > 0.3 {
        1 // N1: Communal (some structural analog exists)
    } else {
        0 // N0: Personal (no known analog — speculative)
    };

    // M-axis: physics domain determines permanence
    let materiality = match domain {
        PhysicsDomain::GeneralRelativity
        | PhysicsDomain::QuantumMechanics
        | PhysicsDomain::QuantumFieldTheory => 3, // Foundational
        PhysicsDomain::NuclearPhysics
        | PhysicsDomain::Electromagnetism
        | PhysicsDomain::StatisticalMechanics => 2, // Persistent
        PhysicsDomain::FluidDynamics | PhysicsDomain::Thermodynamics | PhysicsDomain::Optics => 1, // Temporal
        _ => 0, // Ephemeral (modified gravity, speculative)
    };

    let is_novel = similarity < 0.5;

    let mut tags = vec![format!("{:?}", domain)];
    if is_novel {
        tags.push("novel-prediction".to_string());
    }
    if let Some(ref analog) = nearest {
        tags.push(format!("analog:{}", analog));
    }

    PhysicsDiscovery {
        description: description.to_string(),
        domain,
        simulation_confidence,
        nearest_analog: nearest,
        analog_similarity: similarity,
        lem: LEMClassification {
            empirical,
            normative,
            materiality,
        },
        is_novel,
        tags,
    }
}

/// Classify the Lazar Gravity-A claim as a DeSci discovery.
///
/// This demonstrates the full pipeline: physics simulation → epistemic classification.
pub fn classify_lazar_gravity_a(catalog_results: &[SearchResult]) -> PhysicsDiscovery {
    classify_prediction(
        "Extended strong nuclear force (Gravity-A) via Element 115 proton bombardment \
         produces accessible gravity wave at 11.4 GHz. Claimed by Bob Lazar (1989). \
         Structural analysis shows moderate similarity to Yukawa potential (0.60) and \
         Schwarzschild metric (0.58). No exact analog in known physics.",
        PhysicsDomain::ModifiedGravity,
        0.0, // Zero simulation confidence — this is a claim, not a verified simulation
        catalog_results,
        false, // Not open source (classified program claim)
    )
}

/// Classify the Art's Parts THz waveguide claim as a DeSci discovery.
pub fn classify_arts_parts_waveguide(catalog_results: &[SearchResult]) -> PhysicsDiscovery {
    classify_prediction(
        "Bismuth-Magnesium(Zinc) layered metamaterial acts as terahertz waveguide \
         for anti-gravity propulsion. Structural analysis: 0.915 similarity to standard \
         waveguide dispersion relation. ORNL (2022): terrestrial isotopes, impure Bi \
         layers disrupt waveguide function. The physics is textbook; the sample fails.",
        PhysicsDomain::Optics,
        0.15, // Low confidence — ORNL showed sample doesn't meet specs
        catalog_results,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SimilarityBreakdown;

    fn mock_results(scores: &[(&str, f64, PhysicsDomain)]) -> Vec<SearchResult> {
        scores
            .iter()
            .map(|(name, score, domain)| SearchResult {
                name: name.to_string(),
                domain: *domain,
                score: *score as f32,
                breakdown: SimilarityBreakdown {
                    structural: *score as f32,
                    symmetry: 0.0,
                    dimensional: 0.0,
                    full: 0.0,
                },
            })
            .collect()
    }

    #[test]
    fn test_high_confidence_known_physics() {
        let results = mock_results(&[("Wave Equation", 0.95, PhysicsDomain::FluidDynamics)]);
        let d = classify_prediction(
            "Sound propagation in air",
            PhysicsDomain::FluidDynamics,
            0.9,
            &results,
            true,
        );
        assert_eq!(d.lem.empirical, 4); // E4: publicly reproducible
        assert_eq!(d.lem.normative, 3); // N3: axiomatic (exact match)
        assert!(!d.is_novel);
    }

    #[test]
    fn test_speculative_no_analog() {
        let results = mock_results(&[("Heat Equation", 0.2, PhysicsDomain::Thermodynamics)]);
        let d = classify_prediction(
            "Consciousness creates gravitational waves",
            PhysicsDomain::ModifiedGravity,
            0.05,
            &results,
            false,
        );
        assert_eq!(d.lem.empirical, 0); // E0: null
        assert_eq!(d.lem.normative, 0); // N0: personal (no analog)
        assert!(d.is_novel);
    }

    #[test]
    fn test_lazar_classification() {
        let results = mock_results(&[
            ("Yukawa Potential", 0.60, PhysicsDomain::NuclearPhysics),
            (
                "Schwarzschild Metric",
                0.58,
                PhysicsDomain::GeneralRelativity,
            ),
        ]);
        let d = classify_lazar_gravity_a(&results);
        assert_eq!(d.lem.empirical, 0); // E0: zero confidence
        assert_eq!(d.lem.normative, 2); // N2: 0.60 ≥ 0.6 threshold (structural analog in known physics)
        assert_eq!(d.lem.materiality, 0); // M0: modified gravity = ephemeral
        assert!(!d.is_novel); // 0.60 > 0.5 threshold
    }

    #[test]
    fn test_arts_parts_classification() {
        let results = mock_results(&[("Waveguide Dispersion", 0.915, PhysicsDomain::Optics)]);
        let d = classify_arts_parts_waveguide(&results);
        assert_eq!(d.lem.empirical, 0); // E0: 0.15 confidence < 0.3
        assert_eq!(d.lem.normative, 3); // N3: 0.915 ≈ axiomatic (it IS a waveguide)
        assert_eq!(d.lem.materiality, 1); // M1: optics = temporal
        assert!(!d.is_novel); // 0.915 >> 0.5
        // Key insight: the PHYSICS is known (N3), but the CLAIM is unverified (E0)
    }

    #[test]
    fn test_discovery_tags() {
        let results = mock_results(&[("Yukawa", 0.3, PhysicsDomain::NuclearPhysics)]);
        let d = classify_prediction("test", PhysicsDomain::NuclearPhysics, 0.5, &results, false);
        assert!(d.tags.contains(&"NuclearPhysics".to_string()));
        assert!(d.tags.contains(&"novel-prediction".to_string())); // 0.3 < 0.5
    }
}
