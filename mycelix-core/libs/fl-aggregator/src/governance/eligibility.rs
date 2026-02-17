//! Voter Eligibility
//!
//! Determines eligibility to participate in governance based on:
//! - Identity assurance level
//! - MATL trust score
//! - Stake/token holdings
//! - Participation history
//! - FL contribution record

use super::types::ProposalType;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// Eligibility requirements for different governance actions
#[derive(Debug, Clone)]
pub struct EligibilityRequirements {
    /// Minimum assurance level (E0-E4)
    pub min_assurance_level: u8,

    /// Minimum MATL score
    pub min_matl_score: f32,

    /// Minimum stake amount
    pub min_stake: f32,

    /// Minimum account age (days)
    pub min_account_age_days: u32,

    /// Required participation in recent votes
    pub min_recent_participation: f32,

    /// Whether humanity proof is required
    pub humanity_proof_required: bool,

    /// Whether FL participation is required
    pub fl_participation_required: bool,

    /// Minimum FL contributions
    pub min_fl_contributions: u32,
}

impl Default for EligibilityRequirements {
    fn default() -> Self {
        Self {
            min_assurance_level: 1, // E1 minimum
            min_matl_score: 0.3,
            min_stake: 0.0,
            min_account_age_days: 0,
            min_recent_participation: 0.0,
            humanity_proof_required: false,
            fl_participation_required: false,
            min_fl_contributions: 0,
        }
    }
}

impl EligibilityRequirements {
    /// Requirements for voting on standard proposals
    pub fn standard_voting() -> Self {
        Self {
            min_assurance_level: 1,
            min_matl_score: 0.3,
            min_stake: 0.0,
            min_account_age_days: 7,
            min_recent_participation: 0.0,
            humanity_proof_required: false,
            fl_participation_required: false,
            min_fl_contributions: 0,
        }
    }

    /// Requirements for voting on constitutional proposals
    pub fn constitutional_voting() -> Self {
        Self {
            min_assurance_level: 2,
            min_matl_score: 0.5,
            min_stake: 100.0,
            min_account_age_days: 30,
            min_recent_participation: 0.1,
            humanity_proof_required: true,
            fl_participation_required: false,
            min_fl_contributions: 0,
        }
    }

    /// Requirements for FL model governance
    pub fn fl_model_voting() -> Self {
        Self {
            min_assurance_level: 2,
            min_matl_score: 0.5,
            min_stake: 0.0,
            min_account_age_days: 14,
            min_recent_participation: 0.0,
            humanity_proof_required: false,
            fl_participation_required: true,
            min_fl_contributions: 10,
        }
    }

    /// Requirements for creating proposals
    pub fn proposal_creation() -> Self {
        Self {
            min_assurance_level: 2,
            min_matl_score: 0.5,
            min_stake: 100.0,
            min_account_age_days: 30,
            min_recent_participation: 0.2,
            humanity_proof_required: true,
            fl_participation_required: false,
            min_fl_contributions: 0,
        }
    }

    /// Get requirements for a proposal type
    pub fn for_proposal_type(proposal_type: ProposalType) -> Self {
        match proposal_type {
            ProposalType::Standard | ProposalType::Membership => Self::standard_voting(),
            ProposalType::Constitutional => Self::constitutional_voting(),
            ProposalType::ModelGovernance => Self::fl_model_voting(),
            ProposalType::Emergency => Self {
                min_assurance_level: 3,
                min_matl_score: 0.7,
                min_stake: 500.0,
                min_account_age_days: 90,
                min_recent_participation: 0.3,
                humanity_proof_required: true,
                fl_participation_required: false,
                min_fl_contributions: 0,
            },
            ProposalType::Treasury => Self {
                min_assurance_level: 2,
                min_matl_score: 0.6,
                min_stake: 200.0,
                min_account_age_days: 60,
                min_recent_participation: 0.15,
                humanity_proof_required: true,
                fl_participation_required: false,
                min_fl_contributions: 0,
            },
        }
    }
}

/// Voter profile for eligibility checking
#[derive(Debug, Clone)]
pub struct VoterProfile {
    pub did: String,
    pub assurance_level: u8,
    pub matl_score: f32,
    pub stake: f32,
    pub account_created: DateTime<Utc>,
    pub has_humanity_proof: bool,
    pub fl_contributions: u32,
    pub recent_votes: u32,
    pub recent_eligible_proposals: u32,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VoterProfile {
    /// Calculate account age in days
    pub fn account_age_days(&self) -> u32 {
        (Utc::now() - self.account_created).num_days().max(0) as u32
    }

    /// Calculate recent participation rate
    pub fn recent_participation_rate(&self) -> f32 {
        if self.recent_eligible_proposals > 0 {
            self.recent_votes as f32 / self.recent_eligible_proposals as f32
        } else {
            0.0
        }
    }
}

/// Result of eligibility check
#[derive(Debug, Clone)]
pub struct EligibilityResult {
    pub eligible: bool,
    pub voter_did: String,
    pub proposal_type: Option<ProposalType>,
    pub failures: Vec<EligibilityFailure>,
    pub warnings: Vec<String>,
    pub effective_weight: f32,
}

/// Specific eligibility failure
#[derive(Debug, Clone)]
pub struct EligibilityFailure {
    pub requirement: String,
    pub required: String,
    pub actual: String,
    pub severity: FailureSeverity,
}

/// Severity of eligibility failure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureSeverity {
    /// Hard failure - cannot participate
    Hard,
    /// Soft failure - reduced weight
    Soft,
    /// Warning - eligible but flagged
    Warning,
}

/// Eligibility checker
#[derive(Debug, Default)]
pub struct EligibilityChecker {
    /// Cached eligibility results
    cache: HashMap<(String, ProposalType), (EligibilityResult, DateTime<Utc>)>,

    /// Cache duration
    cache_duration: Duration,
}

impl EligibilityChecker {
    /// Create new eligibility checker
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            cache_duration: Duration::minutes(5),
        }
    }

    /// Check eligibility for voting
    pub fn check_voting_eligibility(
        &mut self,
        voter: &VoterProfile,
        proposal_type: ProposalType,
    ) -> EligibilityResult {
        // Check cache
        let cache_key = (voter.did.clone(), proposal_type);
        if let Some((result, cached_at)) = self.cache.get(&cache_key) {
            if Utc::now() - *cached_at < self.cache_duration {
                return result.clone();
            }
        }

        let requirements = EligibilityRequirements::for_proposal_type(proposal_type);
        let result = self.check_against_requirements(voter, &requirements, Some(proposal_type));

        // Cache result
        self.cache.insert(cache_key, (result.clone(), Utc::now()));

        result
    }

    /// Check eligibility for proposal creation
    pub fn check_proposal_eligibility(&self, voter: &VoterProfile) -> EligibilityResult {
        let requirements = EligibilityRequirements::proposal_creation();
        self.check_against_requirements(voter, &requirements, None)
    }

    /// Check voter against specific requirements
    fn check_against_requirements(
        &self,
        voter: &VoterProfile,
        requirements: &EligibilityRequirements,
        proposal_type: Option<ProposalType>,
    ) -> EligibilityResult {
        let mut failures = Vec::new();
        let mut warnings = Vec::new();
        let mut weight_multiplier = 1.0f32;

        // Check assurance level
        if voter.assurance_level < requirements.min_assurance_level {
            failures.push(EligibilityFailure {
                requirement: "assurance_level".to_string(),
                required: format!("E{}", requirements.min_assurance_level),
                actual: format!("E{}", voter.assurance_level),
                severity: FailureSeverity::Hard,
            });
        }

        // Check MATL score
        if voter.matl_score < requirements.min_matl_score {
            let gap = requirements.min_matl_score - voter.matl_score;
            if gap > 0.2 {
                failures.push(EligibilityFailure {
                    requirement: "matl_score".to_string(),
                    required: format!("{:.2}", requirements.min_matl_score),
                    actual: format!("{:.2}", voter.matl_score),
                    severity: FailureSeverity::Hard,
                });
            } else {
                // Soft failure - reduce weight
                weight_multiplier *= voter.matl_score / requirements.min_matl_score;
                warnings.push(format!(
                    "MATL score {:.2} below threshold {:.2}, weight reduced",
                    voter.matl_score, requirements.min_matl_score
                ));
            }
        }

        // Check stake
        if voter.stake < requirements.min_stake {
            if requirements.min_stake > 0.0 {
                failures.push(EligibilityFailure {
                    requirement: "stake".to_string(),
                    required: format!("{:.0}", requirements.min_stake),
                    actual: format!("{:.0}", voter.stake),
                    severity: FailureSeverity::Hard,
                });
            }
        }

        // Check account age
        let account_age = voter.account_age_days();
        if account_age < requirements.min_account_age_days {
            failures.push(EligibilityFailure {
                requirement: "account_age".to_string(),
                required: format!("{} days", requirements.min_account_age_days),
                actual: format!("{} days", account_age),
                severity: FailureSeverity::Hard,
            });
        }

        // Check participation
        let participation = voter.recent_participation_rate();
        if participation < requirements.min_recent_participation {
            if requirements.min_recent_participation > 0.0 {
                failures.push(EligibilityFailure {
                    requirement: "participation".to_string(),
                    required: format!("{:.0}%", requirements.min_recent_participation * 100.0),
                    actual: format!("{:.0}%", participation * 100.0),
                    severity: FailureSeverity::Soft,
                });
                weight_multiplier *= 0.5; // Reduce weight for low participation
            }
        }

        // Check humanity proof
        if requirements.humanity_proof_required && !voter.has_humanity_proof {
            failures.push(EligibilityFailure {
                requirement: "humanity_proof".to_string(),
                required: "true".to_string(),
                actual: "false".to_string(),
                severity: FailureSeverity::Hard,
            });
        }

        // Check FL participation
        if requirements.fl_participation_required {
            if voter.fl_contributions < requirements.min_fl_contributions {
                failures.push(EligibilityFailure {
                    requirement: "fl_contributions".to_string(),
                    required: format!("{}", requirements.min_fl_contributions),
                    actual: format!("{}", voter.fl_contributions),
                    severity: FailureSeverity::Hard,
                });
            }
        }

        // Determine overall eligibility
        let has_hard_failure = failures
            .iter()
            .any(|f| f.severity == FailureSeverity::Hard);

        let effective_weight = if has_hard_failure {
            0.0
        } else {
            voter.stake.max(1.0) * weight_multiplier
        };

        EligibilityResult {
            eligible: !has_hard_failure,
            voter_did: voter.did.clone(),
            proposal_type,
            failures,
            warnings,
            effective_weight,
        }
    }

    /// Get all eligible voters from a list
    pub fn filter_eligible(
        &mut self,
        voters: &[VoterProfile],
        proposal_type: ProposalType,
    ) -> Vec<(String, f32)> {
        voters
            .iter()
            .filter_map(|voter| {
                let result = self.check_voting_eligibility(voter, proposal_type);
                if result.eligible {
                    Some((voter.did.clone(), result.effective_weight))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate total eligible weight
    pub fn calculate_eligible_weight(
        &mut self,
        voters: &[VoterProfile],
        proposal_type: ProposalType,
    ) -> f32 {
        self.filter_eligible(voters, proposal_type)
            .iter()
            .map(|(_, weight)| weight)
            .sum()
    }

    /// Clear eligibility cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Check if a voter meets minimum requirements for any governance participation
pub fn meets_minimum_requirements(voter: &VoterProfile) -> bool {
    voter.assurance_level >= 1 && voter.matl_score >= 0.2 && voter.account_age_days() >= 1
}

/// Calculate eligibility score (0.0-1.0) for ranking voters
pub fn calculate_eligibility_score(voter: &VoterProfile) -> f32 {
    let assurance_component = (voter.assurance_level as f32 / 4.0).min(1.0);
    let matl_component = voter.matl_score;
    let participation_component = voter.recent_participation_rate();
    let age_component = (voter.account_age_days() as f32 / 365.0).min(1.0);

    // Weighted average
    let score = assurance_component * 0.3
        + matl_component * 0.3
        + participation_component * 0.2
        + age_component * 0.2;

    score.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_voter() -> VoterProfile {
        VoterProfile {
            did: "did:mycelix:test".to_string(),
            assurance_level: 2,
            matl_score: 0.7,
            stake: 500.0,
            account_created: Utc::now() - Duration::days(100),
            has_humanity_proof: true,
            fl_contributions: 20,
            recent_votes: 8,
            recent_eligible_proposals: 10,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_standard_voting_eligibility() {
        let mut checker = EligibilityChecker::new();
        let voter = test_voter();

        let result = checker.check_voting_eligibility(&voter, ProposalType::Standard);

        assert!(result.eligible);
        assert!(result.failures.is_empty());
        assert!(result.effective_weight > 0.0);
    }

    #[test]
    fn test_insufficient_assurance() {
        let mut checker = EligibilityChecker::new();
        let mut voter = test_voter();
        voter.assurance_level = 0;

        let result = checker.check_voting_eligibility(&voter, ProposalType::Standard);

        assert!(!result.eligible);
        assert!(result
            .failures
            .iter()
            .any(|f| f.requirement == "assurance_level"));
    }

    #[test]
    fn test_fl_model_voting_requires_contributions() {
        let mut checker = EligibilityChecker::new();
        let mut voter = test_voter();
        voter.fl_contributions = 5; // Below required 10

        let result = checker.check_voting_eligibility(&voter, ProposalType::ModelGovernance);

        assert!(!result.eligible);
        assert!(result
            .failures
            .iter()
            .any(|f| f.requirement == "fl_contributions"));
    }

    #[test]
    fn test_constitutional_requires_humanity_proof() {
        let mut checker = EligibilityChecker::new();
        let mut voter = test_voter();
        voter.has_humanity_proof = false;

        let result = checker.check_voting_eligibility(&voter, ProposalType::Constitutional);

        assert!(!result.eligible);
        assert!(result
            .failures
            .iter()
            .any(|f| f.requirement == "humanity_proof"));
    }

    #[test]
    fn test_eligibility_score_calculation() {
        let voter = test_voter();
        let score = calculate_eligibility_score(&voter);

        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_minimum_requirements() {
        let voter = test_voter();
        assert!(meets_minimum_requirements(&voter));

        let mut new_voter = voter.clone();
        new_voter.assurance_level = 0;
        assert!(!meets_minimum_requirements(&new_voter));
    }

    #[test]
    fn test_filter_eligible() {
        let mut checker = EligibilityChecker::new();
        let voters = vec![
            test_voter(),
            VoterProfile {
                did: "did:mycelix:ineligible".to_string(),
                assurance_level: 0,
                ..test_voter()
            },
        ];

        let eligible = checker.filter_eligible(&voters, ProposalType::Standard);

        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].0, "did:mycelix:test");
    }
}
