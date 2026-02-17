//! Assurance Level Calculation for Multi-Factor Identity
//! Aligned with Epistemic Charter v2.0 E-Axis
//!
//! Maps identity factor verification to graduated capabilities

use crate::identity::{AssuranceLevel, FactorCategory, FactorStatus, IdentityFactor};
use std::collections::{HashMap, HashSet};

/// Result of assurance level calculation
#[derive(Debug, Clone)]
pub struct AssuranceResult {
    /// Calculated assurance level
    pub level: AssuranceLevel,

    /// Numerical score (0.0-1.0)
    pub score: f32,

    /// Active factor IDs
    pub active_factors: Vec<String>,

    /// Factor categories present
    pub categories_present: HashSet<FactorCategory>,

    /// Capabilities granted at this level
    pub capabilities: HashSet<Capability>,

    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Capabilities available at different assurance levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    // E0 capabilities
    ReadPublicData,
    BrowseAnonymous,

    // E1 capabilities
    PostComments,
    SendMessages,
    CreateContent,

    // E2 capabilities
    ParticipateCommunity,
    SubmitProposals,
    ReceiveRewards,
    ParticipateFL,

    // E3 capabilities
    VoteGovernance,
    ValidateClaims,
    RunValidator,

    // E4 capabilities
    ProposeConstitutionalAmendment,
    JoinKnowledgeCouncil,
    ServeGuardian,
    ManageTreasury,
}

impl Capability {
    /// Get capabilities for each assurance level
    pub fn for_level(level: AssuranceLevel) -> HashSet<Capability> {
        use Capability::*;

        let mut caps = HashSet::new();

        // E0 - Anonymous
        caps.insert(ReadPublicData);
        caps.insert(BrowseAnonymous);

        if level >= AssuranceLevel::E1 {
            caps.insert(PostComments);
            caps.insert(SendMessages);
            caps.insert(CreateContent);
        }

        if level >= AssuranceLevel::E2 {
            caps.insert(ParticipateCommunity);
            caps.insert(SubmitProposals);
            caps.insert(ReceiveRewards);
            caps.insert(ParticipateFL);
        }

        if level >= AssuranceLevel::E3 {
            caps.insert(VoteGovernance);
            caps.insert(ValidateClaims);
            caps.insert(RunValidator);
        }

        if level >= AssuranceLevel::E4 {
            caps.insert(ProposeConstitutionalAmendment);
            caps.insert(JoinKnowledgeCouncil);
            caps.insert(ServeGuardian);
            caps.insert(ManageTreasury);
        }

        caps
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        use Capability::*;
        match self {
            ReadPublicData => "Read public data and content",
            BrowseAnonymous => "Browse anonymously without account",
            PostComments => "Post comments on discussions",
            SendMessages => "Send direct messages to other users",
            CreateContent => "Create and publish content",
            ParticipateCommunity => "Participate in community activities",
            SubmitProposals => "Submit proposals for consideration",
            ReceiveRewards => "Receive reputation and credit rewards",
            ParticipateFL => "Participate in federated learning rounds",
            VoteGovernance => "Vote on governance proposals",
            ValidateClaims => "Validate epistemic claims",
            RunValidator => "Run a network validator node",
            ProposeConstitutionalAmendment => "Propose constitutional amendments",
            JoinKnowledgeCouncil => "Join the Knowledge Council",
            ServeGuardian => "Serve as a recovery guardian",
            ManageTreasury => "Participate in treasury management",
        }
    }
}

/// Calculate assurance level from active identity factors
///
/// Algorithm:
/// 1. Sum contributions from all ACTIVE factors
/// 2. Apply bonus for factor diversity (multiple categories)
/// 3. Map score to assurance level
/// 4. Return capabilities and recommendations
pub fn calculate_assurance_level(factors: &[&dyn IdentityFactor]) -> AssuranceResult {
    // Filter to active factors
    let active_factors: Vec<_> = factors
        .iter()
        .filter(|f| f.status() == FactorStatus::Active)
        .collect();

    if active_factors.is_empty() {
        return AssuranceResult {
            level: AssuranceLevel::E0,
            score: 0.0,
            active_factors: vec![],
            categories_present: HashSet::new(),
            capabilities: Capability::for_level(AssuranceLevel::E0),
            recommendations: vec![
                "Create a DID with cryptographic key pair".to_string(),
                "Verify your Gitcoin Passport for community participation".to_string(),
                "Set up social recovery guardians".to_string(),
            ],
        };
    }

    // Calculate base score from factor contributions
    let base_score: f32 = active_factors.iter().map(|f| f.contribution()).sum();

    // Get unique categories
    let categories: HashSet<FactorCategory> = active_factors.iter().map(|f| f.category()).collect();

    // Apply diversity bonus: 5% per unique category
    let diversity_bonus = categories.len() as f32 * 0.05;

    // Final score (capped at 1.0)
    let final_score = (base_score + diversity_bonus).min(1.0);

    // Map score to level
    let level = AssuranceLevel::from_score(final_score);

    // Generate recommendations
    let recommendations = generate_recommendations(&active_factors, level);

    AssuranceResult {
        level,
        score: final_score,
        active_factors: active_factors.iter().map(|f| f.factor_id().to_string()).collect(),
        categories_present: categories,
        capabilities: Capability::for_level(level),
        recommendations,
    }
}

/// Generate recommendations for improving assurance level
fn generate_recommendations(
    active_factors: &[&&dyn IdentityFactor],
    current_level: AssuranceLevel,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Check what factor types are present
    let factor_types: HashSet<_> = active_factors.iter().map(|f| f.factor_type()).collect();

    // Missing factor recommendations
    if !factor_types.contains("CryptoKey") {
        recommendations.push("CRITICAL: Generate a primary cryptographic key pair".to_string());
    }

    if !factor_types.contains("GitcoinPassport") && current_level < AssuranceLevel::E3 {
        recommendations
            .push("Verify your Gitcoin Passport (required for governance voting)".to_string());
    }

    if !factor_types.contains("SocialRecovery") {
        recommendations
            .push("Set up social recovery with 5+ guardians for account security".to_string());
    }

    if !factor_types.contains("HardwareKey") && current_level >= AssuranceLevel::E2 {
        recommendations.push("Add a hardware security key for enhanced security".to_string());
    }

    if !factor_types.contains("Biometric") && current_level >= AssuranceLevel::E3 {
        recommendations.push("Consider adding biometric verification for convenience".to_string());
    }

    // Level-specific recommendations
    match current_level {
        AssuranceLevel::E0 => {
            recommendations.push("Get started: Create your first DID to join the network".to_string());
        }
        AssuranceLevel::E1 => {
            recommendations
                .push("Reach E2: Add 2+ additional factors for community participation".to_string());
        }
        AssuranceLevel::E2 => {
            recommendations
                .push("Reach E3: Verify Gitcoin Passport >= 20 for governance voting".to_string());
        }
        AssuranceLevel::E3 => {
            recommendations.push(
                "Reach E4: Add hardware key + biometric for constitutional proposals".to_string(),
            );
        }
        AssuranceLevel::E4 => {
            recommendations.push(
                "Maximum verification achieved! You can participate in all governance activities"
                    .to_string(),
            );
        }
    }

    recommendations
}

/// Get the minimum assurance level required for a capability
pub fn get_required_level(capability: Capability) -> AssuranceLevel {
    use Capability::*;

    match capability {
        ReadPublicData | BrowseAnonymous => AssuranceLevel::E0,
        PostComments | SendMessages | CreateContent => AssuranceLevel::E1,
        ParticipateCommunity | SubmitProposals | ReceiveRewards | ParticipateFL => {
            AssuranceLevel::E2
        }
        VoteGovernance | ValidateClaims | RunValidator => AssuranceLevel::E3,
        ProposeConstitutionalAmendment | JoinKnowledgeCouncil | ServeGuardian | ManageTreasury => {
            AssuranceLevel::E4
        }
    }
}

/// Check if an identity has a specific capability
pub fn has_capability(result: &AssuranceResult, capability: Capability) -> bool {
    result.capabilities.contains(&capability)
}

/// Get a summary of capabilities for display
pub fn get_capabilities_summary(result: &AssuranceResult) -> HashMap<String, serde_json::Value> {
    let mut summary = HashMap::new();

    summary.insert("level".to_string(), serde_json::json!(result.level.name()));
    summary.insert("score".to_string(), serde_json::json!(result.score));
    summary.insert(
        "active_factors".to_string(),
        serde_json::json!(result.active_factors.len()),
    );

    let caps = serde_json::json!({
        "read": has_capability(result, Capability::ReadPublicData),
        "write": has_capability(result, Capability::CreateContent),
        "community": has_capability(result, Capability::ParticipateCommunity),
        "fl_participation": has_capability(result, Capability::ParticipateFL),
        "governance": has_capability(result, Capability::VoteGovernance),
        "constitutional": has_capability(result, Capability::ProposeConstitutionalAmendment),
    });

    summary.insert("capabilities".to_string(), caps);

    if let Some(next) = result.level.next() {
        summary.insert("next_level".to_string(), serde_json::json!(next.name()));
    } else {
        summary.insert(
            "next_level".to_string(),
            serde_json::json!("Maximum level achieved"),
        );
    }

    summary.insert(
        "recommendations".to_string(),
        serde_json::json!(result.recommendations),
    );

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::factors::{CryptoKeyFactor, GitcoinPassportFactor};

    #[test]
    fn test_no_factors_e0() {
        let factors: Vec<&dyn IdentityFactor> = vec![];
        let result = calculate_assurance_level(&factors);

        assert_eq!(result.level, AssuranceLevel::E0);
        assert_eq!(result.score, 0.0);
        assert!(result.active_factors.is_empty());
    }

    #[test]
    fn test_single_crypto_key_e1() {
        let mut crypto = CryptoKeyFactor::generate().unwrap();
        crypto.mark_verified();

        let factors: Vec<&dyn IdentityFactor> = vec![&crypto];
        let result = calculate_assurance_level(&factors);

        // 0.5 contribution + 0.05 diversity = 0.55 -> E2
        assert!(result.level >= AssuranceLevel::E1);
        assert_eq!(result.active_factors.len(), 1);
    }

    #[test]
    fn test_multi_factor_e2() {
        let mut crypto = CryptoKeyFactor::generate().unwrap();
        crypto.mark_verified();

        let mut passport = GitcoinPassportFactor::new("0x123", 25.0).unwrap();
        passport.mark_verified();

        let factors: Vec<&dyn IdentityFactor> = vec![&crypto, &passport];
        let result = calculate_assurance_level(&factors);

        // 0.5 + 0.3 = 0.8 contribution + 0.10 diversity (2 categories) = 0.9 -> E4
        assert!(result.level >= AssuranceLevel::E2);
        assert_eq!(result.categories_present.len(), 2);
    }

    #[test]
    fn test_capabilities_for_level() {
        let e0_caps = Capability::for_level(AssuranceLevel::E0);
        assert!(e0_caps.contains(&Capability::ReadPublicData));
        assert!(!e0_caps.contains(&Capability::VoteGovernance));

        let e3_caps = Capability::for_level(AssuranceLevel::E3);
        assert!(e3_caps.contains(&Capability::VoteGovernance));
        assert!(e3_caps.contains(&Capability::ParticipateFL));
        assert!(!e3_caps.contains(&Capability::ProposeConstitutionalAmendment));

        let e4_caps = Capability::for_level(AssuranceLevel::E4);
        assert!(e4_caps.contains(&Capability::ProposeConstitutionalAmendment));
    }

    #[test]
    fn test_required_level_for_capability() {
        assert_eq!(
            get_required_level(Capability::ReadPublicData),
            AssuranceLevel::E0
        );
        assert_eq!(
            get_required_level(Capability::PostComments),
            AssuranceLevel::E1
        );
        assert_eq!(
            get_required_level(Capability::ParticipateFL),
            AssuranceLevel::E2
        );
        assert_eq!(
            get_required_level(Capability::VoteGovernance),
            AssuranceLevel::E3
        );
        assert_eq!(
            get_required_level(Capability::ProposeConstitutionalAmendment),
            AssuranceLevel::E4
        );
    }

    #[test]
    fn test_diversity_bonus() {
        // Single category: 1 * 0.05 = 0.05 bonus
        let mut crypto = CryptoKeyFactor::generate().unwrap();
        crypto.mark_verified();

        let factors1: Vec<&dyn IdentityFactor> = vec![&crypto];
        let result1 = calculate_assurance_level(&factors1);

        // Two categories: 2 * 0.05 = 0.10 bonus
        let mut passport = GitcoinPassportFactor::new("0x123", 25.0).unwrap();
        passport.mark_verified();

        let factors2: Vec<&dyn IdentityFactor> = vec![&crypto, &passport];
        let result2 = calculate_assurance_level(&factors2);

        // Check diversity bonus effect
        assert!(result2.score > result1.score);
        assert_eq!(result2.categories_present.len(), 2);
    }
}
