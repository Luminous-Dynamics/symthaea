// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Epistemic Classifier for Agent Outputs
//!
//! Classifies all AI agent outputs along the E-N-M-H dimensions:
//! - E (Empirical): How can this be verified? (E0-E4)
//! - N (Normative): What scope of agreement is claimed? (N0-N3)
//! - M (Materiality): How long is this relevant? (M0-M3)
//! - H (Harmonic): What harmonic impact? (H0-H4)
//!
//! K-Vector updates are weighted by epistemic level:
//! - E4 outputs (publicly reproducible) are worth more than E0 (unverifiable)

//! - High N-level outputs affect reputation more broadly
//! - High M-level outputs have longer-lasting trust implications

use crate::epistemic::{
    EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
    NormativeLevel,
};
use serde::{Deserialize, Serialize};

/// Agent output with mandatory epistemic classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// Unique output identifier
    pub output_id: String,
    /// Agent that produced this output
    pub agent_id: String,
    /// Output content (could be JSON, text, binary hash, etc.)
    pub content: OutputContent,
    /// Mandatory epistemic classification (E-N-M-H)
    pub classification: EpistemicClassificationExtended,
    /// Confidence in the classification (0.0-1.0)
    pub classification_confidence: f32,
    /// Timestamp of output
    pub timestamp: u64,
    /// Whether this output includes a proof
    pub has_proof: bool,
    /// Optional proof data (for E3+ outputs)
    pub proof_data: Option<Vec<u8>>,
    /// Context that informed this output
    pub context_references: Vec<String>,
}

/// Types of agent output content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputContent {
    /// Plain text response
    Text(String),
    /// Structured JSON data
    Json(String),
    /// Reference to stored data (hash)
    DataReference(String),
    /// Action result
    ActionResult {
        /// Type of action performed
        action_type: String,
        /// Whether the action succeeded
        success: bool,
        /// Additional details about the result
        details: String,
    },
    /// Analysis/recommendation
    Analysis {
        /// Subject being analyzed
        subject: String,
        /// Key findings from the analysis
        findings: Vec<String>,
        /// Suggested next steps
        recommendations: Vec<String>,
    },
}

/// Hints for automatic classification
#[derive(Debug, Clone, Default)]
pub struct ClassificationHints {
    /// Does output include cryptographic proof?
    pub has_crypto_proof: bool,
    /// Is output based on public data sources?
    pub uses_public_data: bool,
    /// Was output verified by third party?
    pub third_party_verified: bool,
    /// Is this a personal opinion?
    pub is_personal_opinion: bool,
    /// Scope of claimed agreement
    pub agreement_scope: Option<AgreementScope>,
    /// Expected relevance duration
    pub relevance_duration: Option<RelevanceDuration>,
    /// Number of harmonies affected
    pub affected_harmonies_count: usize,
    /// Is this civilizational in scope?
    pub civilizational_scope: bool,
}

/// Scope of claimed agreement
#[derive(Debug, Clone, Copy)]
pub enum AgreementScope {
    /// Only the agent agrees
    Self_,
    /// Local community agrees
    Community,
    /// Network-wide agreement
    Network,
    /// Mathematical/constitutional truth
    Axiomatic,
}

/// Expected relevance duration
#[derive(Debug, Clone, Copy)]
pub enum RelevanceDuration {
    /// Discard immediately
    Ephemeral,
    /// Valid for hours to days
    ShortTerm,
    /// Valid for weeks to months
    MediumTerm,
    /// Valid indefinitely
    LongTerm,
}

/// Classify agent output based on hints
///
/// This is the automatic classification function that determines E-N-M-H
/// based on output characteristics.
pub fn classify_output(hints: &ClassificationHints) -> EpistemicClassificationExtended {
    let empirical = classify_empirical(hints);
    let normative = classify_normative(hints);
    let materiality = classify_materiality(hints);
    let harmonic = classify_harmonic(hints);

    EpistemicClassificationExtended::new(empirical, normative, materiality, harmonic)
}

/// Determine empirical level (E0-E4)
fn classify_empirical(hints: &ClassificationHints) -> EmpiricalLevel {
    if hints.uses_public_data && hints.has_crypto_proof {
        // E4: Publicly reproducible with cryptographic proof
        EmpiricalLevel::E4PublicRepro
    } else if hints.has_crypto_proof {
        // E3: Has cryptographic proof but not fully public
        EmpiricalLevel::E3Cryptographic
    } else if hints.third_party_verified {
        // E2: Verified by trusted third party
        EmpiricalLevel::E2PrivateVerify
    } else if !hints.is_personal_opinion {
        // E1: Testimonial but not purely opinion
        EmpiricalLevel::E1Testimonial
    } else {
        // E0: Unverifiable personal opinion
        EmpiricalLevel::E0Null
    }
}

/// Determine normative level (N0-N3)
fn classify_normative(hints: &ClassificationHints) -> NormativeLevel {
    match hints.agreement_scope {
        Some(AgreementScope::Axiomatic) => NormativeLevel::N3Axiomatic,
        Some(AgreementScope::Network) => NormativeLevel::N2Network,
        Some(AgreementScope::Community) => NormativeLevel::N1Communal,
        Some(AgreementScope::Self_) | None => NormativeLevel::N0Personal,
    }
}

/// Determine materiality level (M0-M3)
fn classify_materiality(hints: &ClassificationHints) -> MaterialityLevel {
    match hints.relevance_duration {
        Some(RelevanceDuration::LongTerm) => MaterialityLevel::M3Foundational,
        Some(RelevanceDuration::MediumTerm) => MaterialityLevel::M2Persistent,
        Some(RelevanceDuration::ShortTerm) => MaterialityLevel::M1Temporal,
        Some(RelevanceDuration::Ephemeral) | None => MaterialityLevel::M0Ephemeral,
    }
}

/// Determine harmonic level (H0-H4)
fn classify_harmonic(hints: &ClassificationHints) -> HarmonicLevel {
    if hints.civilizational_scope {
        HarmonicLevel::H4Kosmic
    } else if hints.affected_harmonies_count >= 4 {
        HarmonicLevel::H3Civilizational
    } else if hints.affected_harmonies_count >= 2 {
        HarmonicLevel::H2Network
    } else if hints.affected_harmonies_count >= 1 {
        HarmonicLevel::H1Local
    } else {
        HarmonicLevel::H0None
    }
}

/// Calculate epistemic weight for K-Vector updates
///
/// Higher epistemic levels produce larger trust score impacts.
/// Formula: weight = E_factor * N_factor * M_factor * H_factor
pub fn calculate_epistemic_weight(classification: &EpistemicClassificationExtended) -> f32 {
    let e_factor = match classification.empirical {
        EmpiricalLevel::E0Null => 0.2,
        EmpiricalLevel::E1Testimonial => 0.4,
        EmpiricalLevel::E2PrivateVerify => 0.6,
        EmpiricalLevel::E3Cryptographic => 0.8,
        EmpiricalLevel::E4PublicRepro => 1.0,
    };

    let n_factor = match classification.normative {
        NormativeLevel::N0Personal => 0.5,
        NormativeLevel::N1Communal => 0.7,
        NormativeLevel::N2Network => 0.9,
        NormativeLevel::N3Axiomatic => 1.0,
    };

    let m_factor = match classification.materiality {
        MaterialityLevel::M0Ephemeral => 0.3,
        MaterialityLevel::M1Temporal => 0.5,
        MaterialityLevel::M2Persistent => 0.8,
        MaterialityLevel::M3Foundational => 1.0,
    };

    let h_factor = match classification.harmonic {
        HarmonicLevel::H0None => 0.5,
        HarmonicLevel::H1Local => 0.6,
        HarmonicLevel::H2Network => 0.8,
        HarmonicLevel::H3Civilizational => 0.9,
        HarmonicLevel::H4Kosmic => 1.0,
    };

    e_factor * n_factor * m_factor * h_factor
}

/// Create an agent output with automatic classification
pub fn create_classified_output(
    agent_id: &str,
    content: OutputContent,
    hints: &ClassificationHints,
    timestamp: u64,
) -> AgentOutput {
    let classification = classify_output(hints);

    // Calculate confidence based on how many hints were provided
    let hints_provided = [
        hints.has_crypto_proof,
        hints.uses_public_data,
        hints.third_party_verified,
        hints.is_personal_opinion,
        hints.agreement_scope.is_some(),
        hints.relevance_duration.is_some(),
        hints.affected_harmonies_count > 0,
        hints.civilizational_scope,
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    let classification_confidence = (hints_provided as f32 / 8.0).max(0.3);

    AgentOutput {
        output_id: format!("out-{}-{}", agent_id, timestamp),
        agent_id: agent_id.to_string(),
        content,
        classification,
        classification_confidence,
        timestamp,
        has_proof: hints.has_crypto_proof,
        proof_data: None,
        context_references: vec![],
    }
}

/// Builder for creating agent outputs with explicit classification
pub struct AgentOutputBuilder {
    agent_id: String,
    content: Option<OutputContent>,
    classification: Option<EpistemicClassificationExtended>,
    confidence: f32,
    timestamp: u64,
    has_proof: bool,
    proof_data: Option<Vec<u8>>,
    context_refs: Vec<String>,
}

impl AgentOutputBuilder {
    /// Create a new builder
    pub fn new(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            content: None,
            classification: None,
            confidence: 0.5,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            has_proof: false,
            proof_data: None,
            context_refs: vec![],
        }
    }

    /// Set the content
    pub fn content(mut self, content: OutputContent) -> Self {
        self.content = Some(content);
        self
    }

    /// Set explicit classification
    pub fn classification(
        mut self,
        e: EmpiricalLevel,
        n: NormativeLevel,
        m: MaterialityLevel,
        h: HarmonicLevel,
    ) -> Self {
        self.classification = Some(EpistemicClassificationExtended::new(e, n, m, h));
        self
    }

    /// Set classification confidence
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add proof data (automatically sets E3 minimum)
    pub fn with_proof(mut self, proof: Vec<u8>) -> Self {
        self.has_proof = true;
        self.proof_data = Some(proof);
        self
    }

    /// Add context reference
    pub fn context(mut self, reference: &str) -> Self {
        self.context_refs.push(reference.to_string());
        self
    }

    /// Build the output
    pub fn build(self) -> Result<AgentOutput, &'static str> {
        let content = self.content.ok_or("Content is required")?;
        let classification = self.classification.ok_or("Classification is required")?;

        // Validate: if has proof, E level must be at least E3
        if self.has_proof && classification.empirical < EmpiricalLevel::E3Cryptographic {
            return Err("Outputs with proof must have E3+ classification");
        }

        Ok(AgentOutput {
            output_id: format!("out-{}-{}", self.agent_id, self.timestamp),
            agent_id: self.agent_id,
            content,
            classification,
            classification_confidence: self.confidence,
            timestamp: self.timestamp,
            has_proof: self.has_proof,
            proof_data: self.proof_data,
            context_references: self.context_refs,
        })
    }
}

/// Aggregate epistemic statistics for an agent's outputs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpistemicStats {
    /// Count of outputs per E level
    pub empirical_distribution: [u32; 5],
    /// Count of outputs per N level
    pub normative_distribution: [u32; 4],
    /// Count of outputs per M level
    pub materiality_distribution: [u32; 4],
    /// Count of outputs per H level
    pub harmonic_distribution: [u32; 5],
    /// Average epistemic weight
    pub average_weight: f32,
    /// Total outputs analyzed
    pub total_outputs: u32,
}

impl EpistemicStats {
    /// Add an output classification to statistics
    pub fn add_output(&mut self, classification: &EpistemicClassificationExtended) {
        self.empirical_distribution[classification.empirical as usize] += 1;
        self.normative_distribution[classification.normative as usize] += 1;
        self.materiality_distribution[classification.materiality as usize] += 1;
        self.harmonic_distribution[classification.harmonic as usize] += 1;

        let weight = calculate_epistemic_weight(classification);
        let n = self.total_outputs as f32;
        self.average_weight = (self.average_weight * n + weight) / (n + 1.0);
        self.total_outputs += 1;
    }

    /// Get dominant empirical level
    pub fn dominant_empirical(&self) -> EmpiricalLevel {
        let max_idx = self
            .empirical_distribution
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match max_idx {
            0 => EmpiricalLevel::E0Null,
            1 => EmpiricalLevel::E1Testimonial,
            2 => EmpiricalLevel::E2PrivateVerify,
            3 => EmpiricalLevel::E3Cryptographic,
            _ => EmpiricalLevel::E4PublicRepro,
        }
    }

    /// Calculate epistemic quality score (0.0-1.0)
    /// Higher E and N levels = higher quality
    pub fn quality_score(&self) -> f32 {
        if self.total_outputs == 0 {
            return 0.5;
        }

        // Weight toward higher E and N levels
        let e_score: f32 = self
            .empirical_distribution
            .iter()
            .enumerate()
            .map(|(i, &count)| (i as f32 / 4.0) * count as f32)
            .sum::<f32>()
            / self.total_outputs as f32;

        let n_score: f32 = self
            .normative_distribution
            .iter()
            .enumerate()
            .map(|(i, &count)| (i as f32 / 3.0) * count as f32)
            .sum::<f32>()
            / self.total_outputs as f32;

        (e_score + n_score) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_high_epistemic() {
        let hints = ClassificationHints {
            has_crypto_proof: true,
            uses_public_data: true,
            third_party_verified: true,
            is_personal_opinion: false,
            agreement_scope: Some(AgreementScope::Network),
            relevance_duration: Some(RelevanceDuration::LongTerm),
            affected_harmonies_count: 3,
            civilizational_scope: false,
        };

        let class = classify_output(&hints);

        assert_eq!(class.empirical, EmpiricalLevel::E4PublicRepro);
        assert_eq!(class.normative, NormativeLevel::N2Network);
        assert_eq!(class.materiality, MaterialityLevel::M3Foundational);
        assert_eq!(class.harmonic, HarmonicLevel::H2Network);
    }

    #[test]
    fn test_classify_low_epistemic() {
        let hints = ClassificationHints {
            is_personal_opinion: true,
            ..Default::default()
        };

        let class = classify_output(&hints);

        assert_eq!(class.empirical, EmpiricalLevel::E0Null);
        assert_eq!(class.normative, NormativeLevel::N0Personal);
        assert_eq!(class.materiality, MaterialityLevel::M0Ephemeral);
        assert_eq!(class.harmonic, HarmonicLevel::H0None);
    }

    #[test]
    fn test_epistemic_weight_calculation() {
        // Maximum weight
        let max_class = EpistemicClassificationExtended::new(
            EmpiricalLevel::E4PublicRepro,
            NormativeLevel::N3Axiomatic,
            MaterialityLevel::M3Foundational,
            HarmonicLevel::H4Kosmic,
        );
        let max_weight = calculate_epistemic_weight(&max_class);
        assert!((max_weight - 1.0).abs() < 0.001);

        // Minimum weight
        let min_class = EpistemicClassificationExtended::new(
            EmpiricalLevel::E0Null,
            NormativeLevel::N0Personal,
            MaterialityLevel::M0Ephemeral,
            HarmonicLevel::H0None,
        );
        let min_weight = calculate_epistemic_weight(&min_class);
        assert!(min_weight < 0.05);
    }

    #[test]
    fn test_agent_output_builder() {
        let output = AgentOutputBuilder::new("agent-123")
            .content(OutputContent::Text("Hello world".to_string()))
            .classification(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
            )
            .confidence(0.8)
            .context("context-ref-1")
            .build()
            .unwrap();

        assert_eq!(output.agent_id, "agent-123");
        assert!((output.classification_confidence - 0.8).abs() < 0.001);
        assert!(!output.has_proof);
    }

    #[test]
    fn test_epistemic_stats() {
        let mut stats = EpistemicStats::default();

        // Add some outputs
        for _ in 0..5 {
            stats.add_output(&EpistemicClassificationExtended::new(
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M2Persistent,
                HarmonicLevel::H1Local,
            ));
        }

        for _ in 0..3 {
            stats.add_output(&EpistemicClassificationExtended::new(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
            ));
        }

        assert_eq!(stats.total_outputs, 8);
        assert_eq!(stats.dominant_empirical(), EmpiricalLevel::E3Cryptographic);
        assert!(stats.quality_score() > 0.3);
    }

    #[test]
    fn test_proof_requires_e3() {
        let result = AgentOutputBuilder::new("agent-123")
            .content(OutputContent::Text("Proven fact".to_string()))
            .classification(
                EmpiricalLevel::E1Testimonial, // Too low for proof
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
            )
            .with_proof(vec![1, 2, 3])
            .build();

        assert!(result.is_err());
    }
}
