//! # Covenant Bonds
//!
//! Purpose-bound capital that survives beyond individuals.

use super::{CovenantId, CovenantType};
use serde::{Deserialize, Serialize};

/// A covenant binding capital to specific purposes and beneficiaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Covenant {
    /// Unique covenant ID
    pub covenant_id: CovenantId,
    /// Human-readable purpose
    pub purpose: String,
    /// Structured intention reference (MIP-E-007)
    pub intention_id: Option<String>,
    /// Beneficiary specification
    pub beneficiaries: BeneficiarySpec,
    /// Success metrics
    pub success_metrics: Vec<CovenantMetric>,
    /// Dissolution conditions
    pub dissolution_conditions: Vec<DissolutionCondition>,
    /// Successor covenant if this dissolves
    pub successor: Option<CovenantId>,
    /// Created timestamp
    pub created_at: u64,
    /// Covenant type (for multiplier calculation)
    pub covenant_type: CovenantType,
}

/// Beneficiary specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeneficiarySpec {
    /// Specific DIDs
    Named(
        /// List of beneficiary DIDs
        Vec<String>,
    ),
    /// Members of a Local DAO
    LocalDao {
        /// DAO identifier
        dao_id: String,
    },
    /// Members of a bioregion
    Bioregion {
        /// Bioregion identifier
        bioregion_id: String,
    },
    /// Future generations
    FutureGenerations {
        /// Born after this timestamp
        born_after: u64,
        /// Geographic scope
        scope: GeographicScope,
    },
    /// Ecological entity (river, forest, etc.)
    EcologicalEntity {
        /// Entity identifier
        entity_id: String,
        /// Legal guardian DAO
        guardian_dao: String,
    },
    /// Universal (all Mycelix members)
    Universal,
}

impl BeneficiarySpec {
    /// Get covenant type from beneficiary spec
    pub fn to_covenant_type(&self) -> CovenantType {
        match self {
            BeneficiarySpec::Named(_) => CovenantType::Named,
            BeneficiarySpec::LocalDao { .. } => CovenantType::LocalCommunity,
            BeneficiarySpec::Bioregion { .. } => CovenantType::LocalCommunity,
            BeneficiarySpec::FutureGenerations { .. } => CovenantType::FutureGenerations,
            BeneficiarySpec::EcologicalEntity { .. } => CovenantType::Ecological,
            BeneficiarySpec::Universal => CovenantType::Universal,
        }
    }
}

/// Geographic scope for beneficiary specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeographicScope {
    /// Global
    Global,
    /// Specific country
    Country(String),
    /// Specific region
    Region(String),
    /// Specific bioregion
    Bioregion(String),
    /// Specific local area
    Local {
        /// Latitude coordinate
        lat: f64,
        /// Longitude coordinate
        lon: f64,
        /// Radius in kilometers
        radius_km: f64,
    },
}

/// Metric for measuring covenant success
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovenantMetric {
    /// Metric name
    pub name: String,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Unit
    pub unit: String,
    /// Measurement method
    pub measurement_method: String,
}

/// Condition under which covenant may dissolve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DissolutionCondition {
    /// All metrics achieved
    AllMetricsAchieved,
    /// Specific date reached
    DateReached {
        /// Dissolution timestamp
        timestamp: u64,
    },
    /// Beneficiaries vote to dissolve
    BeneficiaryVote {
        /// Vote threshold fraction
        threshold: f64,
    },
    /// Guardian DAO decides
    GuardianDecision {
        /// Guardian DAO identifier
        dao_id: String,
    },
    /// Successor covenant created
    SuccessorCreated,
    /// Emergency dissolution (Karmic Council)
    Emergency,
}

impl Covenant {
    /// Create a new covenant
    pub fn new(purpose: String, beneficiaries: BeneficiarySpec, timestamp: u64) -> Self {
        let covenant_type = beneficiaries.to_covenant_type();

        Self {
            covenant_id: CovenantId::generate(),
            purpose,
            intention_id: None,
            beneficiaries,
            success_metrics: vec![],
            dissolution_conditions: vec![DissolutionCondition::AllMetricsAchieved],
            successor: None,
            created_at: timestamp,
            covenant_type,
        }
    }

    /// Add a success metric
    pub fn add_metric(&mut self, metric: CovenantMetric) {
        self.success_metrics.push(metric);
    }

    /// Add a dissolution condition
    pub fn add_dissolution_condition(&mut self, condition: DissolutionCondition) {
        self.dissolution_conditions.push(condition);
    }

    /// Link to civilizational intention (MIP-E-007)
    pub fn link_intention(&mut self, intention_id: String) {
        self.intention_id = Some(intention_id);
    }

    /// Set successor covenant
    pub fn set_successor(&mut self, successor_id: CovenantId) {
        self.successor = Some(successor_id);
    }

    /// Check if all metrics are achieved
    pub fn all_metrics_achieved(&self) -> bool {
        self.success_metrics
            .iter()
            .all(|m| m.current_value >= m.target_value)
    }

    /// Calculate overall progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.success_metrics.is_empty() {
            return 0.0;
        }

        let total_progress: f64 = self
            .success_metrics
            .iter()
            .map(|m| (m.current_value / m.target_value).min(1.0))
            .sum();

        total_progress / self.success_metrics.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_covenant() {
        let covenant = Covenant::new(
            "Restore the local watershed".to_string(),
            BeneficiarySpec::Bioregion {
                bioregion_id: "pacific-northwest".to_string(),
            },
            1_000_000,
        );

        assert_eq!(covenant.covenant_type, CovenantType::LocalCommunity);
        assert_eq!(covenant.progress(), 0.0);
    }

    #[test]
    fn test_beneficiary_to_type() {
        assert_eq!(
            BeneficiarySpec::Universal.to_covenant_type(),
            CovenantType::Universal
        );
        assert_eq!(
            BeneficiarySpec::FutureGenerations {
                born_after: 0,
                scope: GeographicScope::Global,
            }
            .to_covenant_type(),
            CovenantType::FutureGenerations
        );
    }

    #[test]
    fn test_progress_calculation() {
        let mut covenant = Covenant::new(
            "Test purpose".to_string(),
            BeneficiarySpec::Universal,
            1_000_000,
        );

        covenant.add_metric(CovenantMetric {
            name: "Metric 1".to_string(),
            current_value: 50.0,
            target_value: 100.0,
            unit: "units".to_string(),
            measurement_method: "observation".to_string(),
        });

        covenant.add_metric(CovenantMetric {
            name: "Metric 2".to_string(),
            current_value: 100.0,
            target_value: 100.0,
            unit: "units".to_string(),
            measurement_method: "observation".to_string(),
        });

        // (0.5 + 1.0) / 2 = 0.75
        assert!((covenant.progress() - 0.75).abs() < 0.01);
        assert!(!covenant.all_metrics_achieved());
    }
}
