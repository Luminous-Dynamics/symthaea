//! MATL Integration for Multi-Factor Identity System
//!
//! Connects identity assurance to MATL trust scoring:
//! - Identity assurance -> Initial reputation boosting
//! - Factor diversity -> TCDM enhancement
//! - Verifiable credentials -> PoGQ attestations
//! - Guardian graph -> Cartel detection signals

use crate::identity::{
    AssuranceLevel, MycelixIdentity, VCType,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Risk level based on identity verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityRiskLevel {
    /// Anonymous, no verification
    Critical,
    /// Basic verification only
    High,
    /// Multi-factor, some diversity
    Medium,
    /// Strong verification, good diversity
    Low,
    /// Maximum verification achieved
    Minimal,
}

impl IdentityRiskLevel {
    /// Get numerical risk score (0.0 = minimal, 1.0 = critical)
    pub fn score(&self) -> f32 {
        match self {
            IdentityRiskLevel::Critical => 1.0,
            IdentityRiskLevel::High => 0.75,
            IdentityRiskLevel::Medium => 0.5,
            IdentityRiskLevel::Low => 0.25,
            IdentityRiskLevel::Minimal => 0.0,
        }
    }
}

/// Identity-derived trust signals for MATL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityTrustSignal {
    /// DID being evaluated
    pub did: String,

    /// Assurance level
    pub assurance_level: AssuranceLevel,

    /// Numerical assurance score (0.0-1.0)
    pub assurance_score: f32,

    /// Number of active factors
    pub active_factors: usize,

    /// Number of unique factor categories
    pub factor_categories: usize,

    /// Factor diversity score (0.0-1.0)
    pub factor_diversity_score: f32,

    /// Has VerifiedHuman credential
    pub verified_human: bool,

    /// Gitcoin Passport score (0-100)
    pub gitcoin_score: f32,

    /// Total credentials count
    pub credentials_count: usize,

    /// Valid credentials count
    pub credentials_valid: usize,

    /// Number of guardians configured
    pub guardian_count: usize,

    /// Guardian network diversity (0.0-1.0)
    pub guardian_graph_diversity: f32,

    /// Risk level assessment
    pub risk_level: IdentityRiskLevel,

    /// Sybil resistance score (0.0-1.0)
    pub sybil_resistance_score: f32,

    /// When signals were computed
    pub computed_at: DateTime<Utc>,

    /// Last verification timestamp
    pub last_verification: Option<DateTime<Utc>>,
}

/// MATL Composite Trust Score enhanced with identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMATLScore {
    /// Proof of Quality score (0.0-1.0)
    pub pogq_score: f32,

    /// Temporal/Community Diversity score (0.0-1.0)
    pub tcdm_score: f32,

    /// Behavioral entropy score (0.0-1.0)
    pub entropy_score: f32,

    /// Identity boost/penalty (-0.2 to +0.2)
    pub identity_boost: f32,

    /// Identity signals used
    pub identity_signals: IdentityTrustSignal,

    /// Original MATL score
    pub base_score: f32,

    /// Final enhanced score
    pub enhanced_score: f32,

    /// When computed
    pub computed_at: DateTime<Utc>,
}

/// Bridge between Multi-Factor Identity and MATL Trust Engine
pub struct IdentityMATLBridge {
    /// Cached signals by DID
    signal_cache: HashMap<String, IdentityTrustSignal>,
}

impl Default for IdentityMATLBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl IdentityMATLBridge {
    /// Create a new bridge
    pub fn new() -> Self {
        Self {
            signal_cache: HashMap::new(),
        }
    }

    /// Compute identity trust signals from an identity
    pub fn compute_identity_signals(&mut self, identity: &MycelixIdentity) -> IdentityTrustSignal {
        // Get active factors
        let active_factors = identity.active_factors();
        let factor_count = active_factors.len();

        // Calculate factor diversity
        let categories: HashSet<_> = active_factors.iter().map(|f| f.category()).collect();
        let category_count = categories.len();
        // Max possible categories = 5
        let factor_diversity = category_count as f32 / 5.0;

        // Calculate assurance score
        let assurance_score: f32 = active_factors.iter().map(|f| f.contribution()).sum();
        let total_score = (assurance_score + category_count as f32 * 0.05).min(1.0);
        let assurance_level = AssuranceLevel::from_score(total_score);

        // Check for verified human credential
        let verified_human = identity
            .credentials
            .iter()
            .any(|vc| vc.credential_type == VCType::VerifiedHumanity && vc.is_valid());

        // Get Gitcoin score from credentials
        let gitcoin_score = identity
            .credentials
            .iter()
            .find(|vc| vc.credential_type == VCType::GitcoinPassport && vc.is_valid())
            .and_then(|vc| vc.claims.get("passportScore"))
            .and_then(|v| v.as_f64())
            .map(|s| s as f32)
            .unwrap_or(0.0);

        // Count credentials
        let credentials_count = identity.credentials.len();
        let credentials_valid = identity.credentials.iter().filter(|vc| vc.is_valid()).count();

        // Guardian metrics from social recovery factor
        let guardian_count = identity
            .mfa_state
            .factors
            .iter()
            .filter(|f| f.factor_type() == "SocialRecovery")
            .filter_map(|f| {
                let json = f.to_json();
                json.get("guardian_count")?.as_u64().map(|c| c as usize)
            })
            .next()
            .unwrap_or(0);

        // Calculate Sybil resistance
        let sybil_resistance = self.calculate_sybil_resistance(
            total_score,
            gitcoin_score,
            verified_human,
            factor_diversity,
        );

        // Determine risk level
        let risk_level = self.determine_risk_level(assurance_level, sybil_resistance);

        // Get last verification time
        let last_verification = active_factors
            .iter()
            .filter_map(|f| f.last_verified())
            .max();

        let signal = IdentityTrustSignal {
            did: identity.did.clone(),
            assurance_level,
            assurance_score: total_score,
            active_factors: factor_count,
            factor_categories: category_count,
            factor_diversity_score: factor_diversity,
            verified_human,
            gitcoin_score,
            credentials_count,
            credentials_valid,
            guardian_count,
            guardian_graph_diversity: 0.5, // Default, requires graph analysis
            risk_level,
            sybil_resistance_score: sybil_resistance,
            computed_at: Utc::now(),
            last_verification,
        };

        // Cache signal
        self.signal_cache.insert(identity.did.clone(), signal.clone());

        signal
    }

    /// Enhance MATL trust score with identity verification
    pub fn enhance_matl_score(
        &self,
        did: &str,
        pogq_score: f32,
        tcdm_score: f32,
        entropy_score: f32,
        identity_signals: Option<&IdentityTrustSignal>,
    ) -> Result<EnhancedMATLScore, String> {
        // Get signals from cache or parameter
        let signals = identity_signals
            .cloned()
            .or_else(|| self.signal_cache.get(did).cloned())
            .ok_or_else(|| format!("No identity signals found for {}", did))?;

        // Calculate base MATL score: (PoGQ × 0.4) + (TCDM × 0.3) + (Entropy × 0.3)
        let base_score = (pogq_score * 0.4) + (tcdm_score * 0.3) + (entropy_score * 0.3);

        // Calculate identity boost (-0.2 to +0.2)
        let identity_boost = self.calculate_identity_boost(&signals);

        // Apply boost (clamped to [0.0, 1.0])
        let enhanced_score = (base_score + identity_boost).clamp(0.0, 1.0);

        Ok(EnhancedMATLScore {
            pogq_score,
            tcdm_score,
            entropy_score,
            identity_boost,
            identity_signals: signals,
            base_score,
            enhanced_score,
            computed_at: Utc::now(),
        })
    }

    /// Calculate identity boost/penalty for MATL score
    fn calculate_identity_boost(&self, signals: &IdentityTrustSignal) -> f32 {
        // Base boost from assurance level
        let assurance_boost = match signals.assurance_level {
            AssuranceLevel::E0 => -0.20,
            AssuranceLevel::E1 => -0.05,
            AssuranceLevel::E2 => 0.05,
            AssuranceLevel::E3 => 0.12,
            AssuranceLevel::E4 => 0.20,
        };

        // Gitcoin score boost
        let gitcoin_boost = if signals.gitcoin_score >= 50.0 {
            0.03
        } else if signals.gitcoin_score >= 20.0 {
            0.02
        } else {
            0.0
        };

        // Verified human boost
        let verified_human_boost = if signals.verified_human { 0.03 } else { 0.0 };

        // Guardian boost
        let guardian_boost = if signals.guardian_count >= 5 { 0.02 } else { 0.0 };

        // Total boost (clamped to -0.2 to +0.2)
        let total: f32 = assurance_boost + gitcoin_boost + verified_human_boost + guardian_boost;
        total.clamp(-0.2, 0.2)
    }

    /// Calculate Sybil resistance score
    fn calculate_sybil_resistance(
        &self,
        assurance_score: f32,
        gitcoin_score: f32,
        verified_human: bool,
        factor_diversity: f32,
    ) -> f32 {
        // Normalize Gitcoin score to 0.0-1.0
        let gitcoin_normalized = (gitcoin_score / 100.0).min(1.0);

        // Weighted combination
        let score = (assurance_score * 0.35)
            + (gitcoin_normalized * 0.30)
            + (if verified_human { 1.0 } else { 0.0 } * 0.20)
            + (factor_diversity * 0.15);

        score.min(1.0)
    }

    /// Determine identity risk level
    fn determine_risk_level(
        &self,
        assurance_level: AssuranceLevel,
        sybil_resistance: f32,
    ) -> IdentityRiskLevel {
        match assurance_level {
            AssuranceLevel::E0 => IdentityRiskLevel::Critical,
            AssuranceLevel::E1 => IdentityRiskLevel::High,
            AssuranceLevel::E2 => {
                if sybil_resistance >= 0.7 {
                    IdentityRiskLevel::Low
                } else {
                    IdentityRiskLevel::Medium
                }
            }
            AssuranceLevel::E3 => {
                if sybil_resistance >= 0.8 {
                    IdentityRiskLevel::Minimal
                } else {
                    IdentityRiskLevel::Low
                }
            }
            AssuranceLevel::E4 => IdentityRiskLevel::Minimal,
        }
    }

    /// Get initial MATL reputation based on identity verification
    pub fn get_initial_reputation(&self, did: &str) -> f32 {
        let signals = match self.signal_cache.get(did) {
            Some(s) => s,
            None => return 0.5, // Default neutral reputation
        };

        // Map assurance level to initial reputation
        let base_reputation = match signals.assurance_level {
            AssuranceLevel::E0 => 0.30,
            AssuranceLevel::E1 => 0.40,
            AssuranceLevel::E2 => 0.50,
            AssuranceLevel::E3 => 0.60,
            AssuranceLevel::E4 => 0.70,
        };

        // Boost for verified human
        let human_boost = if signals.verified_human { 0.05 } else { 0.0 };

        // Boost for high Gitcoin score
        let gitcoin_boost = if signals.gitcoin_score >= 50.0 {
            0.05
        } else {
            0.0
        };

        let total: f32 = base_reputation + human_boost + gitcoin_boost;
        total.min(0.7) // Cap at 0.7
    }

    /// Get cached signals for a DID
    pub fn get_cached_signals(&self, did: &str) -> Option<&IdentityTrustSignal> {
        self.signal_cache.get(did)
    }

    /// Clear cached signals
    pub fn clear_cache(&mut self) {
        self.signal_cache.clear();
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

        // Add crypto key factor
        let mut crypto = CryptoKeyFactor::generate().unwrap();
        crypto.mark_verified();
        identity.add_factor(Box::new(crypto));

        // Add Gitcoin passport factor
        let mut passport = GitcoinPassportFactor::new("0x123", 45.0).unwrap();
        passport.mark_verified();
        identity.add_factor(Box::new(passport));

        identity
    }

    #[test]
    fn test_compute_identity_signals() {
        let identity = create_test_identity();
        let mut bridge = IdentityMATLBridge::new();

        let signals = bridge.compute_identity_signals(&identity);

        assert_eq!(signals.did, "did:mycelix:test");
        assert_eq!(signals.active_factors, 2);
        assert_eq!(signals.factor_categories, 2);
        assert!(signals.assurance_score > 0.0);
        assert_eq!(signals.gitcoin_score, 0.0); // From factor, not credential
    }

    #[test]
    fn test_enhance_matl_score() {
        let identity = create_test_identity();
        let mut bridge = IdentityMATLBridge::new();

        let signals = bridge.compute_identity_signals(&identity);
        let enhanced = bridge
            .enhance_matl_score("did:mycelix:test", 0.75, 0.80, 0.70, Some(&signals))
            .unwrap();

        // Base: (0.75 * 0.4) + (0.80 * 0.3) + (0.70 * 0.3) = 0.30 + 0.24 + 0.21 = 0.75
        assert!((enhanced.base_score - 0.75).abs() < 0.01);

        // Enhanced should have identity boost applied
        assert!(enhanced.identity_boost != 0.0);
        assert!(enhanced.enhanced_score != enhanced.base_score);
    }

    #[test]
    fn test_identity_boost_calculation() {
        let bridge = IdentityMATLBridge::new();

        // E4 with high Gitcoin score and verified human
        let signals = IdentityTrustSignal {
            did: "test".to_string(),
            assurance_level: AssuranceLevel::E4,
            assurance_score: 0.9,
            active_factors: 5,
            factor_categories: 4,
            factor_diversity_score: 0.8,
            verified_human: true,
            gitcoin_score: 55.0,
            credentials_count: 3,
            credentials_valid: 3,
            guardian_count: 5,
            guardian_graph_diversity: 0.8,
            risk_level: IdentityRiskLevel::Minimal,
            sybil_resistance_score: 0.9,
            computed_at: Utc::now(),
            last_verification: Some(Utc::now()),
        };

        let boost = bridge.calculate_identity_boost(&signals);
        // E4: 0.20 + Gitcoin >= 50: 0.03 + Verified: 0.03 + Guardians >= 5: 0.02 = 0.28
        // Clamped to 0.2
        assert_eq!(boost, 0.2);
    }

    #[test]
    fn test_initial_reputation() {
        let identity = create_test_identity();
        let mut bridge = IdentityMATLBridge::new();

        bridge.compute_identity_signals(&identity);
        let rep = bridge.get_initial_reputation("did:mycelix:test");

        // Should be between 0.3 and 0.7
        assert!(rep >= 0.3 && rep <= 0.7);
    }

    #[test]
    fn test_risk_level() {
        let bridge = IdentityMATLBridge::new();

        assert_eq!(
            bridge.determine_risk_level(AssuranceLevel::E0, 0.0),
            IdentityRiskLevel::Critical
        );
        assert_eq!(
            bridge.determine_risk_level(AssuranceLevel::E1, 0.5),
            IdentityRiskLevel::High
        );
        assert_eq!(
            bridge.determine_risk_level(AssuranceLevel::E2, 0.8),
            IdentityRiskLevel::Low
        );
        assert_eq!(
            bridge.determine_risk_level(AssuranceLevel::E4, 0.9),
            IdentityRiskLevel::Minimal
        );
    }
}
