// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Discovery Bridge — Connecting Symthaea Physics to DeSci Claims
//!
//! When Symthaea's physics simulation produces a prediction, this module
//! packages it as a DeSci-compatible epistemic claim with LEM classification:
//!
//! - **E-axis**: A local simulation result supports at most E1 unless replay,
//!   integrity, or independent-reproduction evidence is supplied
//! - **N-axis**: Catalog similarity supports at most N1 structural analogy
//! - **M-axis**: Defaults to M0 because this API receives no durability evidence
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
    /// Sanitized simulation confidence (0.0-1.0), not an evidence tier.
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
    /// E-axis: 0=Null, 1=Testimonial, 2=PrivatelyVerifiable, 3=CryptographicallyVerifiable, 4=PubliclyReproduced
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
            2 => "E2: Privately Verifiable (replay evidence available)",
            3 => "E3: Cryptographically Verifiable (integrity-protected evidence)",
            4 => "E4: Publicly Reproduced (independent replication)",
            _ => "Unknown",
        }
    }

    /// Label for the N-axis tier.
    pub fn normative_label(&self) -> &'static str {
        match self.normative {
            0 => "N0: Personal (single simulation)",
            1 => "N1: Communal (catalogued structural analogue)",
            2 => "N2: Network (explicit network consensus)",
            3 => "N3: Axiomatic (formally derived)",
            _ => "Unknown",
        }
    }
}

/// Classify a physics prediction into a DeSci discovery claim.
///
/// This API has only a reported confidence score and equation-catalog search.
/// It deliberately does not infer replayability, cryptographic verification,
/// public reproduction, consensus, proof, or durability from those proxies.
pub fn classify_prediction(
    description: &str,
    domain: PhysicsDomain,
    simulation_confidence: f64,
    catalog_results: &[SearchResult],
    _is_open_source: bool,
) -> PhysicsDiscovery {
    let (nearest, raw_similarity) = catalog_results
        .first()
        .map(|r| (Some(r.name.clone()), r.score as f64))
        .unwrap_or((None, 0.0));
    let simulation_confidence = if simulation_confidence.is_finite() {
        simulation_confidence.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let similarity = if raw_similarity.is_finite() {
        raw_similarity.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // A local simulation report is testimonial. Open source availability and
    // a high self-reported score are not evidence that anyone replayed it.
    let empirical = if simulation_confidence > 0.3 {
        1 // E1: local simulation output
    } else {
        0 // E0: Null (speculative)
    };

    // Similarity to a catalogued expression is an analogy, not consensus or proof.
    let normative = if similarity > 0.3 {
        1 // N1: Communal (some structural analog exists)
    } else {
        0 // N0: Personal (no known analog — speculative)
    };

    // Domain identity does not say how durably this particular claim is recorded.
    let materiality = 0;

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
        assert_eq!(d.lem.empirical, 1); // E1: local simulation output
        assert_eq!(d.lem.normative, 1); // N1: catalogued structural analogy
        assert_eq!(d.lem.materiality, 0); // no durability evidence supplied
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
        assert_eq!(d.lem.normative, 1); // N1: structural analog in catalog
        assert_eq!(d.lem.materiality, 0); // M0: modified gravity = ephemeral
        assert!(!d.is_novel); // 0.60 > 0.5 threshold
    }

    #[test]
    fn test_arts_parts_classification() {
        let results = mock_results(&[("Waveguide Dispersion", 0.915, PhysicsDomain::Optics)]);
        let d = classify_arts_parts_waveguide(&results);
        assert_eq!(d.lem.empirical, 0); // E0: 0.15 confidence < 0.3
        assert_eq!(d.lem.normative, 1); // N1: strong structural analogy is still not proof
        assert_eq!(d.lem.materiality, 0); // no durability evidence supplied
        assert!(!d.is_novel); // 0.915 >> 0.5
        // The physics analogy can be strong while the sample claim remains unverified.
    }

    #[test]
    fn test_discovery_tags() {
        let results = mock_results(&[("Yukawa", 0.3, PhysicsDomain::NuclearPhysics)]);
        let d = classify_prediction("test", PhysicsDomain::NuclearPhysics, 0.5, &results, false);
        assert!(d.tags.contains(&"NuclearPhysics".to_string()));
        assert!(d.tags.contains(&"novel-prediction".to_string())); // 0.3 < 0.5
    }

    #[test]
    fn open_source_and_confidence_do_not_imply_reproduction() {
        let d = classify_prediction(
            "unreproduced local run",
            PhysicsDomain::Optics,
            1.0,
            &[],
            true,
        );

        assert_eq!(d.lem.empirical, 1);
        assert_eq!(d.lem.normative, 0);
        assert_eq!(d.lem.materiality, 0);
    }

    #[test]
    fn non_finite_scores_fail_closed() {
        let results = mock_results(&[("invalid", f64::NAN, PhysicsDomain::Optics)]);
        let d = classify_prediction(
            "invalid scores",
            PhysicsDomain::Optics,
            f64::NAN,
            &results,
            true,
        );

        assert_eq!(d.simulation_confidence, 0.0);
        assert_eq!(d.analog_similarity, 0.0);
        assert_eq!(d.lem.empirical, 0);
        assert_eq!(d.lem.normative, 0);
    }
}
