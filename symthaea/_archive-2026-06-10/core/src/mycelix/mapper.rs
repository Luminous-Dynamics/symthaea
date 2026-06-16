// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ to Epistemic Level Mapper
//!
//! Maps Symthaea's consciousness metrics (Φ, workspace scope, importance)
//! to Mycelix's E/N/M classification system.

use crate::mycelix::types::*;

/// Configuration for Φ → E-Level mapping
#[derive(Debug, Clone)]
pub struct PhiMappingConfig {
    /// Φ threshold for E1 (Testimonial)
    pub e1_threshold: f32,
    /// Φ threshold for E2 (Privately Verifiable)
    pub e2_threshold: f32,
    /// Φ threshold for E3 (Cryptographically Verifiable)
    pub e3_threshold: f32,
    /// Φ threshold for E4 (Publicly Reproducible)
    pub e4_threshold: f32,
    /// Whether to require reproducibility for E4
    pub require_reproducibility_for_e4: bool,
}

impl Default for PhiMappingConfig {
    fn default() -> Self {
        Self {
            e1_threshold: 0.1,
            e2_threshold: 0.2,
            e3_threshold: 0.3,
            e4_threshold: 0.4,
            require_reproducibility_for_e4: true,
        }
    }
}

/// Maps Symthaea consciousness outputs to Mycelix epistemic classifications
pub struct PhiToEpistemicMapper {
    config: PhiMappingConfig,
}

impl PhiToEpistemicMapper {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: PhiMappingConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PhiMappingConfig) -> Self {
        Self { config }
    }

    /// Map Φ value to empirical level
    pub fn phi_to_empirical(&self, phi: f32, is_reproducible: bool) -> EmpiricalLevel {
        let can_reach_e4 = !self.config.require_reproducibility_for_e4 || is_reproducible;

        if phi >= self.config.e4_threshold && can_reach_e4 {
            EmpiricalLevel::PubliclyReproducible
        } else if phi >= self.config.e3_threshold {
            EmpiricalLevel::CryptographicallyVerifiable
        } else if phi >= self.config.e2_threshold {
            EmpiricalLevel::PrivatelyVerifiable
        } else if phi >= self.config.e1_threshold {
            EmpiricalLevel::Testimonial
        } else {
            EmpiricalLevel::Subjective
        }
    }

    /// Classify a conscious output
    ///
    /// Takes raw Symthaea output metrics and produces a full E/N/M classification.
    pub fn classify(
        &self,
        phi: f32,
        scope: WorkspaceScope,
        importance: f32,
        is_reproducible: bool,
    ) -> EpistemicClassification {
        EpistemicClassification {
            empirical: self.phi_to_empirical(phi, is_reproducible),
            normative: NormativeLevel::from_scope(scope),
            materiality: MaterialityLevel::from_importance(importance),
        }
    }

    /// Classify and wrap content into a ClassifiedConsciousOutput
    pub fn classify_output(
        &self,
        content: String,
        phi: f32,
        scope: WorkspaceScope,
        importance: f32,
        is_reproducible: bool,
    ) -> ClassifiedConsciousOutput {
        let mut output =
            ClassifiedConsciousOutput::new(content, phi, scope, importance, is_reproducible);
        output.classification = self.classify(phi, scope, importance, is_reproducible);
        output
    }

    /// Suggest classification improvements based on evidence
    pub fn suggest_improvements(
        &self,
        output: &ClassifiedConsciousOutput,
    ) -> Vec<ClassificationSuggestion> {
        let mut suggestions = Vec::new();

        // If low E-level but has strong evidence, suggest upgrade
        if output.classification.empirical < EmpiricalLevel::CryptographicallyVerifiable {
            let crypto_evidence_count = output
                .evidence
                .iter()
                .filter(|e| e.evidence_type == EvidenceType::CryptographicProof)
                .count();

            if crypto_evidence_count > 0 {
                suggestions.push(ClassificationSuggestion {
                    current: output.classification,
                    suggested: EpistemicClassification {
                        empirical: EmpiricalLevel::CryptographicallyVerifiable,
                        ..output.classification
                    },
                    reason: format!(
                        "Has {crypto_evidence_count} cryptographic proof(s), can upgrade to E3"
                    ),
                });
            }
        }

        // If high importance but low materiality, suggest upgrade
        if output.importance > 0.8
            && output.classification.materiality < MaterialityLevel::Permanent
        {
            suggestions.push(ClassificationSuggestion {
                current: output.classification,
                suggested: EpistemicClassification {
                    materiality: MaterialityLevel::Permanent,
                    ..output.classification
                },
                reason: format!(
                    "High importance ({:.2}) suggests permanent storage",
                    output.importance
                ),
            });
        }

        // If consensus evidence, suggest higher E-level
        let consensus_count = output
            .evidence
            .iter()
            .filter(|e| e.evidence_type == EvidenceType::Consensus)
            .count();

        if consensus_count >= 3
            && output.classification.empirical < EmpiricalLevel::PrivatelyVerifiable
        {
            suggestions.push(ClassificationSuggestion {
                current: output.classification,
                suggested: EpistemicClassification {
                    empirical: EmpiricalLevel::PrivatelyVerifiable,
                    ..output.classification
                },
                reason: format!("Consensus from {consensus_count} sources supports E2 minimum"),
            });
        }

        suggestions
    }
}

impl Default for PhiToEpistemicMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// A suggestion for improving classification
#[derive(Debug, Clone)]
pub struct ClassificationSuggestion {
    /// Current classification
    pub current: EpistemicClassification,
    /// Suggested new classification
    pub suggested: EpistemicClassification,
    /// Reason for the suggestion
    pub reason: String,
}

/// Trait for types that can be epistemically classified
pub trait EpistemicallyClassifiable {
    /// Get the Φ value for this item
    fn phi(&self) -> f32;

    /// Get the workspace scope
    fn scope(&self) -> WorkspaceScope;

    /// Get the importance
    fn importance(&self) -> f32;

    /// Check if reproducible
    fn is_reproducible(&self) -> bool;

    /// Classify using a mapper
    fn classify(&self, mapper: &PhiToEpistemicMapper) -> EpistemicClassification {
        mapper.classify(
            self.phi(),
            self.scope(),
            self.importance(),
            self.is_reproducible(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_mapping() {
        let mapper = PhiToEpistemicMapper::new();

        assert_eq!(
            mapper.phi_to_empirical(0.05, false),
            EmpiricalLevel::Subjective
        );
        assert_eq!(
            mapper.phi_to_empirical(0.15, false),
            EmpiricalLevel::Testimonial
        );
        assert_eq!(
            mapper.phi_to_empirical(0.25, false),
            EmpiricalLevel::PrivatelyVerifiable
        );
        assert_eq!(
            mapper.phi_to_empirical(0.35, false),
            EmpiricalLevel::CryptographicallyVerifiable
        );

        // E4 requires reproducibility
        assert_eq!(
            mapper.phi_to_empirical(0.45, false),
            EmpiricalLevel::CryptographicallyVerifiable
        );
        assert_eq!(
            mapper.phi_to_empirical(0.45, true),
            EmpiricalLevel::PubliclyReproducible
        );
    }

    #[test]
    fn test_full_classification() {
        let mapper = PhiToEpistemicMapper::new();

        let classification = mapper.classify(0.35, WorkspaceScope::Network, 0.6, false);

        assert_eq!(
            classification.empirical,
            EmpiricalLevel::CryptographicallyVerifiable
        );
        assert_eq!(classification.normative, NormativeLevel::Network);
        assert_eq!(classification.materiality, MaterialityLevel::MediumTerm);
        assert_eq!(classification.code(), "E3/N2/M2");
    }

    #[test]
    fn test_suggestions() {
        use std::time::SystemTime;

        let mapper = PhiToEpistemicMapper::new();

        let mut output = mapper.classify_output(
            "Test claim".to_string(),
            0.15, // Low Φ, gets E1
            WorkspaceScope::Network,
            0.9, // High importance
            false,
        );

        // Add cryptographic evidence
        output.evidence.push(Evidence {
            id: "ev1".to_string(),
            evidence_type: EvidenceType::CryptographicProof,
            content: "proof".to_string(),
            strength: 0.95,
            timestamp: SystemTime::now(),
            source: None,
        });

        let suggestions = mapper.suggest_improvements(&output);
        // Should suggest E3 upgrade (M3 already satisfied by importance 0.9)
        assert!(
            !suggestions.is_empty(),
            "Should have at least one suggestion"
        );
        assert!(
            suggestions
                .iter()
                .any(|s| s.suggested.empirical == EmpiricalLevel::CryptographicallyVerifiable),
            "Should suggest E3 upgrade based on crypto evidence"
        );
    }
}
