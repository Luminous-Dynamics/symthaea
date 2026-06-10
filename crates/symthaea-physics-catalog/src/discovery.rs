// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LEM epistemic classification for physics claims.

use crate::catalog::Domain;
use crate::search::SearchResult;
use serde::{Deserialize, Serialize};

/// LEM Cube classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LEMClassification {
    /// E-axis: 0=Null, 1=Testimonial, 2=Verifiable, 3=CryptographicallyProven, 4=PubliclyReproducible
    pub empirical: u8,
    /// N-axis: 0=Personal, 1=Communal, 2=Network, 3=Axiomatic
    pub normative: u8,
    /// M-axis: 0=Ephemeral, 1=Temporal, 2=Persistent, 3=Foundational
    pub materiality: u8,
}

/// A physics discovery claim with epistemic classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsDiscovery {
    pub description: String,
    pub domain: Domain,
    pub simulation_confidence: f64,
    pub nearest_analog: Option<String>,
    pub analog_similarity: f64,
    pub lem: LEMClassification,
    pub is_novel: bool,
    pub tags: Vec<String>,
}

/// Classify a physics claim via catalog search results.
pub fn classify_claim(
    description: &str,
    domain: Domain,
    simulation_confidence: f64,
    results: &[SearchResult],
    is_open_source: bool,
) -> PhysicsDiscovery {
    let (nearest, similarity) = results
        .first()
        .map(|r| (Some(r.name.clone()), r.score as f64))
        .unwrap_or((None, 0.0));

    let empirical = if is_open_source && simulation_confidence > 0.8 {
        4
    } else if simulation_confidence > 0.6 {
        2
    } else if simulation_confidence > 0.3 {
        1
    } else {
        0
    };

    let normative = if similarity > 0.9 {
        3
    } else if similarity > 0.6 {
        2
    } else if similarity > 0.3 {
        1
    } else {
        0
    };

    let materiality = match domain {
        Domain::GeneralRelativity | Domain::QuantumMechanics | Domain::QuantumFieldTheory => 3,
        Domain::NuclearPhysics | Domain::Electromagnetism | Domain::StatisticalMechanics => 2,
        Domain::FluidDynamics | Domain::Thermodynamics | Domain::Optics => 1,
        _ => 0,
    };

    let is_novel = similarity < 0.5;
    let mut tags = vec![format!("{:?}", domain)];
    if is_novel {
        tags.push("novel-prediction".to_string());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novel_classification() {
        let results = vec![SearchResult {
            name: "Heat Equation".to_string(),
            domain: Domain::Thermodynamics,
            latex: String::new(),
            score: 0.2,
        }];
        let d = classify_claim("test", Domain::ModifiedGravity, 0.0, &results, false);
        assert_eq!(d.lem.empirical, 0);
        assert_eq!(d.lem.normative, 0);
        assert!(d.is_novel);
    }

    #[test]
    fn test_known_physics() {
        let results = vec![SearchResult {
            name: "Wave Equation".to_string(),
            domain: Domain::FluidDynamics,
            latex: String::new(),
            score: 0.95,
        }];
        let d = classify_claim("sound propagation", Domain::Acoustics, 0.9, &results, true);
        assert_eq!(d.lem.empirical, 4);
        assert_eq!(d.lem.normative, 3);
        assert!(!d.is_novel);
    }
}
