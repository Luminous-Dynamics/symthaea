// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Civilizational Intention Framework
//!
//! Implementation of MIP-E-007: Civilizational Intention Framework
//!
//! Provides explicit, measurable goals that the Mycelix Protocol is designed
//! to achieve, transforming economics from mechanism to purpose.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique intention ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentionId(String);

impl IntentionId {
    /// Generate new intention ID
    pub fn generate() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self(format!("intent-{:x}", timestamp))
    }

    /// Create from string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Get domain from ID
    pub fn domain(&self) -> IntentionDomain {
        // In production, this would parse the domain from the ID structure
        IntentionDomain::UniversalBasicServices
    }
}

/// Major domains of civilizational concern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentionDomain {
    /// Basic human needs for all
    UniversalBasicServices,
    /// Planetary ecosystem health
    EcologicalRegeneration,
    /// Freely accessible knowledge
    KnowledgeCommons,
    /// Violence reduction and justice
    ConflictTransformation,
    /// Human consciousness development
    ConsciousnessEvolution,
    /// Democratic participation
    GovernanceEvolution,
    /// Infrastructure for all
    DigitalSovereignty,
    /// Cultural preservation and creation
    CulturalFlourishing,
    /// Scientific advancement
    ScientificFrontier,
    /// Joy and meaning
    CollectiveWellbeing,
}

impl IntentionDomain {
    /// Get default weight for domain (for incentive calculation)
    pub fn default_weight(&self) -> f64 {
        match self {
            IntentionDomain::UniversalBasicServices => 1.5,
            IntentionDomain::EcologicalRegeneration => 1.5,
            IntentionDomain::ConflictTransformation => 1.3,
            IntentionDomain::KnowledgeCommons => 1.2,
            IntentionDomain::DigitalSovereignty => 1.1,
            _ => 1.0,
        }
    }

    /// Get all domains
    pub fn all() -> Vec<IntentionDomain> {
        vec![
            IntentionDomain::UniversalBasicServices,
            IntentionDomain::EcologicalRegeneration,
            IntentionDomain::KnowledgeCommons,
            IntentionDomain::ConflictTransformation,
            IntentionDomain::ConsciousnessEvolution,
            IntentionDomain::GovernanceEvolution,
            IntentionDomain::DigitalSovereignty,
            IntentionDomain::CulturalFlourishing,
            IntentionDomain::ScientificFrontier,
            IntentionDomain::CollectiveWellbeing,
        ]
    }
}

/// An intention in the intention tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// Unique intention ID
    pub intention_id: IntentionId,
    /// Domain
    pub domain: IntentionDomain,
    /// Human-readable name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Quantitative target (optional)
    pub target: Option<QuantitativeTarget>,
    /// Target achievement date (optional)
    pub target_date: Option<u64>,
    /// How to measure progress
    pub metrics: Vec<ProgressMetric>,
    /// Parent intention (None if root)
    pub parent: Option<IntentionId>,
    /// Status
    pub status: IntentionStatus,
    /// Created timestamp
    pub created_at: u64,
}

/// Quantitative target for an intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantitativeTarget {
    /// Metric name
    pub metric_name: String,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Unit
    pub unit: String,
}

/// Metric for measuring intention progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressMetric {
    /// Metric name
    pub name: String,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Unit
    pub unit: String,
    /// Measurement source
    pub source: String,
    /// Last updated
    pub last_updated: u64,
}

/// Status of an intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentionStatus {
    /// Initial proposal
    Proposed {
        /// Timestamp when intention was proposed
        proposed_at: u64,
    },
    /// Under community review
    Review {
        /// Timestamp when review period started
        review_start: u64,
    },
    /// Ratified by Global DAO
    Ratified {
        /// Timestamp when intention was ratified
        ratified_at: u64,
    },
    /// Actively being pursued
    Active {
        /// Timestamp when intention became active
        activated_at: u64,
    },
    /// Successfully achieved
    Achieved {
        /// Timestamp when intention was achieved
        achieved_at: u64,
    },
    /// Replaced by better intention
    Superseded {
        /// Timestamp when intention was superseded
        superseded_at: u64,
        /// ID of the intention that supersedes this one
        successor: IntentionId,
    },
    /// Abandoned
    Abandoned {
        /// Timestamp when intention was abandoned
        abandoned_at: u64,
        /// Reason for abandonment
        reason: String,
    },
}

/// Progress toward an intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionProgress {
    /// Intention ID
    pub intention_id: IntentionId,
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    /// Progress delta since last measurement
    pub delta: f64,
    /// Current trajectory toward achievement
    pub trajectory: Trajectory,
    /// Estimated contribution from Mycelix protocol (0.0 to 1.0)
    pub mycelix_contribution: f64,
    /// Timestamp when this measurement was taken (ms since epoch)
    pub measured_at: u64,
    /// Confidence level of this measurement (0.0 to 1.0)
    pub confidence: f64,
}

/// Trajectory toward intention achievement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trajectory {
    /// Ahead of schedule
    Accelerating,
    /// On track
    OnTrack,
    /// Behind but recoverable
    Lagging,
    /// Significantly behind
    AtRisk,
    /// Moving backwards
    Regressing,
}

impl Trajectory {
    /// Get urgency multiplier for incentives
    pub fn urgency_multiplier(&self) -> f64 {
        match self {
            Trajectory::Regressing => 2.0,
            Trajectory::AtRisk => 1.5,
            Trajectory::Lagging => 1.2,
            Trajectory::OnTrack => 1.0,
            Trajectory::Accelerating => 0.8,
        }
    }
}

impl Intention {
    /// Create a new intention
    pub fn new(domain: IntentionDomain, name: String, description: String, timestamp: u64) -> Self {
        Self {
            intention_id: IntentionId::generate(),
            domain,
            name,
            description,
            target: None,
            target_date: None,
            metrics: vec![],
            parent: None,
            status: IntentionStatus::Proposed {
                proposed_at: timestamp,
            },
            created_at: timestamp,
        }
    }

    /// Set quantitative target
    pub fn with_target(mut self, target: QuantitativeTarget) -> Self {
        self.target = Some(target);
        self
    }

    /// Set target date
    pub fn with_target_date(mut self, date: u64) -> Self {
        self.target_date = Some(date);
        self
    }

    /// Add progress metric
    pub fn add_metric(&mut self, metric: ProgressMetric) {
        self.metrics.push(metric);
    }

    /// Calculate overall progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.metrics.is_empty() {
            if let Some(ref target) = self.target {
                return (target.current_value / target.target_value).clamp(0.0, 1.0);
            }
            return 0.0;
        }

        let total: f64 = self
            .metrics
            .iter()
            .map(|m| (m.current_value / m.target_value).clamp(0.0, 1.0))
            .sum();

        total / self.metrics.len() as f64
    }

    /// Check if intention is achieved
    pub fn is_achieved(&self) -> bool {
        matches!(self.status, IntentionStatus::Achieved { .. }) || self.progress() >= 1.0
    }
}

/// Contribution to an intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionContribution {
    /// Contributor DID
    pub contributor_did: String,
    /// Intention contributed to
    pub intention_id: IntentionId,
    /// Contribution type
    pub contribution_type: ContributionType,
    /// Estimated impact (0.0 to 1.0)
    pub estimated_impact: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Type of contribution to an intention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContributionType {
    /// Direct action taken toward the intention
    DirectAction {
        /// Description of the action taken
        description: String,
    },
    /// Financial contribution
    Funding {
        /// Amount of SAP contributed
        amount_sap: u64,
        /// Recipient of the funds
        recipient: String,
    },
    /// Infrastructure provided to support the intention
    Infrastructure {
        /// Description of the infrastructure
        description: String,
    },
    /// Knowledge or research contributed
    Knowledge {
        /// ID of the knowledge artifact
        artifact_id: String,
    },
    /// Coordination or governance work
    Coordination {
        /// Role performed
        role: String,
        /// Hours spent
        hours: f64,
    },
}

/// Calculate intention score for PoC
pub fn calculate_intention_score(
    contributions: &[IntentionContribution],
    progress_map: &HashMap<IntentionId, IntentionProgress>,
    domain_weights: &HashMap<IntentionDomain, f64>,
) -> f64 {
    if contributions.is_empty() {
        return 0.0;
    }

    let score: f64 = contributions
        .iter()
        .map(|c| {
            let progress = progress_map.get(&c.intention_id);
            let urgency = progress
                .map(|p| p.trajectory.urgency_multiplier())
                .unwrap_or(1.0);

            let domain = c.intention_id.domain();
            let domain_weight = domain_weights
                .get(&domain)
                .copied()
                .unwrap_or(domain.default_weight());

            c.estimated_impact * urgency * domain_weight
        })
        .sum();

    // Normalize to 0-1 range (assuming max reasonable score is ~5)
    (score / 5.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_intention() {
        let intention = Intention::new(
            IntentionDomain::UniversalBasicServices,
            "Clean Water For All".to_string(),
            "Universal access to safe drinking water".to_string(),
            1_000_000,
        );

        assert_eq!(intention.domain, IntentionDomain::UniversalBasicServices);
        assert!((intention.progress() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_intention_with_target() {
        let intention = Intention::new(
            IntentionDomain::EcologicalRegeneration,
            "Forest Restoration".to_string(),
            "Restore 1M hectares of forest".to_string(),
            1_000_000,
        )
        .with_target(QuantitativeTarget {
            metric_name: "Hectares restored".to_string(),
            current_value: 500_000.0,
            target_value: 1_000_000.0,
            unit: "hectares".to_string(),
        });

        assert!((intention.progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_trajectory_urgency() {
        assert!((Trajectory::Regressing.urgency_multiplier() - 2.0).abs() < 0.01);
        assert!((Trajectory::OnTrack.urgency_multiplier() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_domain_weights() {
        assert!((IntentionDomain::UniversalBasicServices.default_weight() - 1.5).abs() < 0.01);
        assert!((IntentionDomain::CulturalFlourishing.default_weight() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_intention_score() {
        let contributions = vec![IntentionContribution {
            contributor_did: "did:test:alice".to_string(),
            intention_id: IntentionId::from_string("intent-1".to_string()),
            contribution_type: ContributionType::DirectAction {
                description: "Built water well".to_string(),
            },
            estimated_impact: 0.5,
            timestamp: 1_000_000,
        }];

        let mut progress_map = HashMap::new();
        progress_map.insert(
            IntentionId::from_string("intent-1".to_string()),
            IntentionProgress {
                intention_id: IntentionId::from_string("intent-1".to_string()),
                progress: 0.5,
                delta: 0.1,
                trajectory: Trajectory::AtRisk, // 1.5x urgency
                mycelix_contribution: 0.1,
                measured_at: 1_000_000,
                confidence: 0.9,
            },
        );

        let domain_weights = HashMap::new(); // Use defaults

        let score = calculate_intention_score(&contributions, &progress_map, &domain_weights);

        // 0.5 impact × 1.5 urgency × 1.5 domain weight = 1.125 / 5.0 = 0.225
        assert!(score > 0.2);
        assert!(score < 0.3);
    }
}
