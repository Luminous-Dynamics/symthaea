//! Identity-Gated Federated Learning Participation
//!
//! Provides identity verification requirements for FL participation,
//! preventing Sybil attacks on the learning process.

use crate::identity::{
    AssuranceLevel, FactorCategory, MycelixIdentity,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Requirements for participating in federated learning rounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLParticipationRequirements {
    /// Minimum assurance level required
    pub min_assurance_level: AssuranceLevel,

    /// Required factor categories (at least one from each)
    pub required_factor_categories: Vec<FactorCategory>,

    /// Minimum reputation score from MATL
    pub min_reputation_score: f32,

    /// Whether humanity proof is mandatory
    pub humanity_proof_required: bool,

    /// Minimum factor freshness (effective strength 0.0-1.0)
    pub min_factor_freshness: f32,

    /// Whether delegated participation is allowed
    pub allow_delegation: bool,

    /// Maximum rounds per identity per epoch
    pub max_rounds_per_epoch: Option<u64>,
}

impl FLParticipationRequirements {
    /// Standard training tasks (majority of FL)
    pub fn standard() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E2,
            required_factor_categories: vec![
                FactorCategory::Cryptographic,
                FactorCategory::ExternalVerification,
            ],
            min_reputation_score: 0.4,
            humanity_proof_required: false,
            min_factor_freshness: 0.5,
            allow_delegation: true,
            max_rounds_per_epoch: None,
        }
    }

    /// Sensitive model training (medical, financial)
    pub fn sensitive() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E3,
            required_factor_categories: vec![
                FactorCategory::Cryptographic,
                FactorCategory::Biometric,
                FactorCategory::ExternalVerification,
            ],
            min_reputation_score: 0.7,
            humanity_proof_required: true,
            min_factor_freshness: 0.7,
            allow_delegation: false,
            max_rounds_per_epoch: Some(100),
        }
    }

    /// Critical infrastructure models
    pub fn critical() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E4,
            required_factor_categories: vec![
                FactorCategory::Cryptographic,
                FactorCategory::Biometric,
                FactorCategory::SocialProof,
                FactorCategory::ExternalVerification,
            ],
            min_reputation_score: 0.85,
            humanity_proof_required: true,
            min_factor_freshness: 0.9,
            allow_delegation: false,
            max_rounds_per_epoch: Some(50),
        }
    }

    /// Open participation (development/testing)
    pub fn open() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E1,
            required_factor_categories: vec![FactorCategory::Cryptographic],
            min_reputation_score: 0.0,
            humanity_proof_required: false,
            min_factor_freshness: 0.0,
            allow_delegation: true,
            max_rounds_per_epoch: None,
        }
    }
}

/// Reasons for FL participation denial
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FLDenialReason {
    /// Assurance level too low
    InsufficientAssuranceLevel {
        required: AssuranceLevel,
        actual: AssuranceLevel,
    },

    /// Missing required factor category
    MissingFactorCategory(FactorCategory),

    /// Reputation score too low
    LowReputationScore { required: f32, actual: f32 },

    /// Factors are stale (need re-verification)
    StaleFactors { min_freshness: f32, actual: f32 },

    /// Humanity proof is required but missing
    HumanityProofRequired,

    /// Delegation not allowed for this task
    DelegationNotAllowed,

    /// Round limit exceeded for this epoch
    RoundLimitExceeded { limit: u64, used: u64 },

    /// Identity is in quarantine
    IdentityQuarantined,

    /// Byzantine behavior flagged
    ByzantineFlagged { detection_score: f32 },

    /// Identity not found
    IdentityNotFound,
}

impl std::fmt::Display for FLDenialReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FLDenialReason::InsufficientAssuranceLevel { required, actual } => {
                write!(
                    f,
                    "Assurance level too low: required {:?}, actual {:?}",
                    required, actual
                )
            }
            FLDenialReason::MissingFactorCategory(cat) => {
                write!(f, "Missing required factor category: {:?}", cat)
            }
            FLDenialReason::LowReputationScore { required, actual } => {
                write!(
                    f,
                    "Reputation score too low: required {}, actual {}",
                    required, actual
                )
            }
            FLDenialReason::StaleFactors {
                min_freshness,
                actual,
            } => {
                write!(
                    f,
                    "Factors need re-verification: min freshness {}, actual {}",
                    min_freshness, actual
                )
            }
            FLDenialReason::HumanityProofRequired => {
                write!(f, "Humanity proof is required for this task")
            }
            FLDenialReason::DelegationNotAllowed => {
                write!(f, "Delegated participation not allowed for this task")
            }
            FLDenialReason::RoundLimitExceeded { limit, used } => {
                write!(f, "Round limit exceeded: {} of {} used", used, limit)
            }
            FLDenialReason::IdentityQuarantined => {
                write!(f, "Identity is in quarantine")
            }
            FLDenialReason::ByzantineFlagged { detection_score } => {
                write!(f, "Byzantine behavior flagged: score {}", detection_score)
            }
            FLDenialReason::IdentityNotFound => {
                write!(f, "Identity not found")
            }
        }
    }
}

/// Capabilities granted upon admission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLCapabilities {
    /// Can submit gradients
    pub can_submit_gradients: bool,

    /// Maximum gradient size (bytes)
    pub max_gradient_size: usize,

    /// Eligible model types
    pub eligible_models: Vec<String>,

    /// Rate limit (submissions per minute)
    pub rate_limit: u32,

    /// Whether this participant can be an aggregator
    pub can_aggregate: bool,

    /// Whether this participant can validate others
    pub can_validate: bool,
}

impl FLCapabilities {
    /// No capabilities (denied)
    pub fn none() -> Self {
        Self {
            can_submit_gradients: false,
            max_gradient_size: 0,
            eligible_models: vec![],
            rate_limit: 0,
            can_aggregate: false,
            can_validate: false,
        }
    }

    /// Standard participant capabilities
    pub fn standard() -> Self {
        Self {
            can_submit_gradients: true,
            max_gradient_size: 100 * 1024 * 1024, // 100MB
            eligible_models: vec!["standard".to_string()],
            rate_limit: 60, // 1 per second
            can_aggregate: false,
            can_validate: false,
        }
    }

    /// Elevated capabilities for high-trust participants
    pub fn elevated() -> Self {
        Self {
            can_submit_gradients: true,
            max_gradient_size: 500 * 1024 * 1024, // 500MB
            eligible_models: vec!["standard".to_string(), "sensitive".to_string()],
            rate_limit: 120,
            can_aggregate: true,
            can_validate: true,
        }
    }

    /// Maximum capabilities
    pub fn maximum() -> Self {
        Self {
            can_submit_gradients: true,
            max_gradient_size: 1024 * 1024 * 1024, // 1GB
            eligible_models: vec![
                "standard".to_string(),
                "sensitive".to_string(),
                "critical".to_string(),
            ],
            rate_limit: 300,
            can_aggregate: true,
            can_validate: true,
        }
    }
}

/// Result of FL participation eligibility check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLAdmissionResult {
    /// Whether the identity is eligible
    pub eligible: bool,

    /// Identity DID
    pub identity_did: String,

    /// Effective assurance level (with freshness decay)
    pub effective_assurance: AssuranceLevel,

    /// Trust score from MATL
    pub trust_score: f32,

    /// Reasons for denial (if any)
    pub denial_reasons: Vec<FLDenialReason>,

    /// Granted capabilities
    pub capabilities: FLCapabilities,

    /// When the check was performed
    pub checked_at: DateTime<Utc>,
}

/// FL participation record for accountability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLParticipationRecord {
    /// Identity DID
    pub identity_did: String,

    /// Round number
    pub round_number: u64,

    /// Gradient hash
    pub gradient_hash: String,

    /// Submission timestamp
    pub submission_time: DateTime<Utc>,

    /// Post-Optimum Gradient Quality score
    pub pogq_score: f32,

    /// Byzantine classification (if flagged)
    pub byzantine_classification: Option<String>,
}

/// Check if an identity can participate in FL
pub fn check_fl_eligibility(
    identity: &MycelixIdentity,
    requirements: &FLParticipationRequirements,
    reputation_score: f32,
    rounds_used_this_epoch: Option<u64>,
    byzantine_score: Option<f32>,
    is_delegated: bool,
) -> FLAdmissionResult {
    let mut denial_reasons = Vec::new();

    // 1. Check assurance level
    let effective_level = identity.assurance_level();
    if effective_level < requirements.min_assurance_level {
        denial_reasons.push(FLDenialReason::InsufficientAssuranceLevel {
            required: requirements.min_assurance_level,
            actual: effective_level,
        });
    }

    // 2. Check required factor categories
    let available_categories = identity.factor_categories();
    for required in &requirements.required_factor_categories {
        if !available_categories.contains(required) {
            denial_reasons.push(FLDenialReason::MissingFactorCategory(*required));
        }
    }

    // 3. Check reputation score
    if reputation_score < requirements.min_reputation_score {
        denial_reasons.push(FLDenialReason::LowReputationScore {
            required: requirements.min_reputation_score,
            actual: reputation_score,
        });
    }

    // 4. Check humanity proof if required
    if requirements.humanity_proof_required && !identity.has_humanity_proof() {
        denial_reasons.push(FLDenialReason::HumanityProofRequired);
    }

    // 5. Check delegation
    if is_delegated && !requirements.allow_delegation {
        denial_reasons.push(FLDenialReason::DelegationNotAllowed);
    }

    // 6. Check round limits
    if let (Some(limit), Some(used)) = (requirements.max_rounds_per_epoch, rounds_used_this_epoch) {
        if used >= limit {
            denial_reasons.push(FLDenialReason::RoundLimitExceeded { limit, used });
        }
    }

    // 7. Check Byzantine flags
    if let Some(byz_score) = byzantine_score {
        if byz_score > 0.3 {
            denial_reasons.push(FLDenialReason::ByzantineFlagged {
                detection_score: byz_score,
            });
        }
    }

    let eligible = denial_reasons.is_empty();

    // Calculate capabilities based on assurance level
    let capabilities = if eligible {
        calculate_capabilities(effective_level, reputation_score)
    } else {
        FLCapabilities::none()
    };

    FLAdmissionResult {
        eligible,
        identity_did: identity.did.clone(),
        effective_assurance: effective_level,
        trust_score: reputation_score,
        denial_reasons,
        capabilities,
        checked_at: Utc::now(),
    }
}

/// Calculate capabilities based on assurance level and trust score
fn calculate_capabilities(level: AssuranceLevel, trust_score: f32) -> FLCapabilities {
    match (level, trust_score >= 0.8) {
        (AssuranceLevel::E4, _) => FLCapabilities::maximum(),
        (AssuranceLevel::E3, true) => FLCapabilities::elevated(),
        (AssuranceLevel::E3, false) => FLCapabilities::standard(),
        (AssuranceLevel::E2, _) => FLCapabilities::standard(),
        _ => FLCapabilities::none(),
    }
}

/// FL participation incentive calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLIncentive {
    pub action: String,
    pub credits_earned: u64,
    pub reputation_impact: f32,
}

impl FLIncentive {
    /// Incentives for FL actions
    pub fn for_action(action: &str, pogq_score: Option<f32>) -> Self {
        match action {
            "submit_gradient" => {
                let (credits, rep) = if let Some(score) = pogq_score {
                    if score > 0.9 {
                        (25, 0.005) // High quality
                    } else {
                        (10, 0.001) // Normal
                    }
                } else {
                    (10, 0.001)
                };
                FLIncentive {
                    action: action.to_string(),
                    credits_earned: credits,
                    reputation_impact: rep,
                }
            }
            "detect_byzantine" => FLIncentive {
                action: action.to_string(),
                credits_earned: 50,
                reputation_impact: 0.01,
            },
            "complete_round" => FLIncentive {
                action: action.to_string(),
                credits_earned: 5,
                reputation_impact: 0.0005,
            },
            "serve_aggregator" => FLIncentive {
                action: action.to_string(),
                credits_earned: 100,
                reputation_impact: 0.02,
            },
            _ => FLIncentive {
                action: action.to_string(),
                credits_earned: 0,
                reputation_impact: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{AgentType, CryptoKeyFactor, GitcoinPassportFactor, IdentityFactor};

    fn create_test_identity() -> MycelixIdentity {
        let mut identity = MycelixIdentity::new(
            "did:mycelix:test".to_string(),
            AgentType::HumanMember {
                biometric_hash: None,
                humanity_proofs: vec![],
            },
        );

        let mut crypto = CryptoKeyFactor::generate().unwrap();
        crypto.mark_verified();
        identity.add_factor(Box::new(crypto));

        let mut passport = GitcoinPassportFactor::new("0x123", 30.0).unwrap();
        passport.mark_verified();
        identity.add_factor(Box::new(passport));

        identity
    }

    #[test]
    fn test_standard_requirements_eligible() {
        let identity = create_test_identity();
        let requirements = FLParticipationRequirements::standard();

        let result = check_fl_eligibility(&identity, &requirements, 0.5, None, None, false);

        assert!(result.eligible);
        assert!(result.denial_reasons.is_empty());
        assert!(result.capabilities.can_submit_gradients);
    }

    #[test]
    fn test_sensitive_requirements_denied() {
        let identity = create_test_identity();
        let requirements = FLParticipationRequirements::sensitive();

        let result = check_fl_eligibility(&identity, &requirements, 0.5, None, None, false);

        // Should fail: no biometric factor, humanity proof required
        assert!(!result.eligible);
        assert!(!result.denial_reasons.is_empty());
    }

    #[test]
    fn test_low_reputation_denied() {
        let identity = create_test_identity();
        let requirements = FLParticipationRequirements::standard();

        let result = check_fl_eligibility(&identity, &requirements, 0.2, None, None, false);

        assert!(!result.eligible);
        assert!(result.denial_reasons.iter().any(|r| matches!(
            r,
            FLDenialReason::LowReputationScore { .. }
        )));
    }

    #[test]
    fn test_round_limit_exceeded() {
        let identity = create_test_identity();
        let mut requirements = FLParticipationRequirements::standard();
        requirements.max_rounds_per_epoch = Some(10);

        let result = check_fl_eligibility(&identity, &requirements, 0.5, Some(15), None, false);

        assert!(!result.eligible);
        assert!(result.denial_reasons.iter().any(|r| matches!(
            r,
            FLDenialReason::RoundLimitExceeded { .. }
        )));
    }

    #[test]
    fn test_byzantine_flagged() {
        let identity = create_test_identity();
        let requirements = FLParticipationRequirements::standard();

        let result = check_fl_eligibility(&identity, &requirements, 0.5, None, Some(0.5), false);

        assert!(!result.eligible);
        assert!(result.denial_reasons.iter().any(|r| matches!(
            r,
            FLDenialReason::ByzantineFlagged { .. }
        )));
    }

    #[test]
    fn test_delegation_not_allowed() {
        let identity = create_test_identity();
        let requirements = FLParticipationRequirements::sensitive();

        let result = check_fl_eligibility(&identity, &requirements, 0.8, None, None, true);

        assert!(!result.eligible);
        assert!(result
            .denial_reasons
            .contains(&FLDenialReason::DelegationNotAllowed));
    }

    #[test]
    fn test_incentive_calculations() {
        let submit = FLIncentive::for_action("submit_gradient", Some(0.95));
        assert_eq!(submit.credits_earned, 25);
        assert_eq!(submit.reputation_impact, 0.005);

        let normal_submit = FLIncentive::for_action("submit_gradient", Some(0.7));
        assert_eq!(normal_submit.credits_earned, 10);

        let aggregator = FLIncentive::for_action("serve_aggregator", None);
        assert_eq!(aggregator.credits_earned, 100);
    }

    #[test]
    fn test_capabilities_by_level() {
        let standard = calculate_capabilities(AssuranceLevel::E2, 0.5);
        assert!(standard.can_submit_gradients);
        assert!(!standard.can_aggregate);

        let elevated = calculate_capabilities(AssuranceLevel::E3, 0.85);
        assert!(elevated.can_aggregate);
        assert!(elevated.can_validate);

        let max = calculate_capabilities(AssuranceLevel::E4, 0.9);
        assert!(max.eligible_models.contains(&"critical".to_string()));
    }
}
