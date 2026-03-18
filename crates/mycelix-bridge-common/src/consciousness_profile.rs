//! Multi-dimensional consciousness profile for governance gating.
//!
//! Replaces the single MATL score (40% MFA + 60% reputation) with a
//! 4-dimensional `ConsciousnessProfile` that gates governance actions
//! with progressive vote weighting.
//!
//! ## Dimensions
//!
//! 1. **Identity** — MFA assurance level (Anonymous=0.0 → Critical=1.0)
//! 2. **Reputation** — Cross-hApp aggregated reputation with exponential decay
//! 3. **Community** — Peer trust attestations, weighted by attestor tier
//! 4. **Engagement** — Domain-specific participation, computed locally per bridge
//!
//! ## Usage
//!
//! Governance zomes call their local bridge's `get_consciousness_credential`,
//! then evaluate it with `evaluate_governance()` — a pure function with no
//! HDK dependency.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

use crate::consciousness_thresholds::{
    BOOTSTRAP_COMMUNITY_THRESHOLD, BOOTSTRAP_MIN_IDENTITY, BOOTSTRAP_TTL_US,
};

// ============================================================================
// Sigmoid authorization constants
// ============================================================================

/// Default sigmoid temperature for vote weight computation.
/// 0.05 provides a smooth transition over ~±0.1 around the threshold.
pub const VOTE_WEIGHT_TEMPERATURE: f64 = 0.05;

/// Maximum vote weight (basis points: 10000 = 100%).
pub const VOTE_WEIGHT_MAX_BP: f64 = 10000.0;

/// Hysteresis margin for tier transitions.
/// Promotion requires `threshold + margin`; demotion requires `threshold - margin`.
pub const TIER_HYSTERESIS_MARGIN: f64 = 0.05;

// ============================================================================
// Continuous sigmoid authorization
// ============================================================================

/// Compute continuous vote weight using sigmoid function.
///
/// Instead of hard tier thresholds, this provides a smooth gradient:
/// `W = W_max / (1 + e^(-(score - threshold) / temperature))`
///
/// - score < threshold: weight approaches 0 smoothly
/// - score = threshold: weight = W_max / 2
/// - score > threshold: weight approaches W_max smoothly
///
/// Temperature controls strictness:
/// - low temperature (0.02): sharp transition (like hard threshold)
/// - high temperature (0.10): gradual transition (noise-tolerant)
pub fn continuous_vote_weight(score: f64, threshold: f64, temperature: f64, max_weight: f64) -> f64 {
    if temperature <= 0.0 || !score.is_finite() {
        return 0.0;
    }
    let exponent = -((score - threshold) / temperature);
    // Clamp exponent to prevent overflow
    let exponent = exponent.clamp(-20.0, 20.0);
    max_weight / (1.0 + exponent.exp())
}

// ============================================================================
// Core types
// ============================================================================

/// 4-dimensional consciousness profile.
///
/// Each dimension is 0.0–1.0. Governance actions require different
/// minimum combinations. Vote weight scales with overall profile strength.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConsciousnessProfile {
    /// Identity verification strength (from MFA AssuranceLevel).
    /// Anonymous=0.0, Basic=0.25, Verified=0.5, HighlyAssured=0.75, Critical=1.0
    pub identity: f64,

    /// Cross-hApp reputation (from identity bridge aggregated reputation).
    /// Exponential decay with 30-day half-life, multi-source weighted average.
    pub reputation: f64,

    /// Community trust attestations (from aggregated peer trust credentials).
    /// Weighted by attestor's own tier — higher-consciousness peers count more.
    pub community: f64,

    /// Domain-specific engagement (computed locally by each cluster bridge).
    /// Based on event/query participation counts, decayed over time.
    pub engagement: f64,
}

impl ConsciousnessProfile {
    /// Combined score — weighted average of all 4 dimensions.
    ///
    /// Identity and community are weighted higher for governance:
    /// - identity:   25%
    /// - reputation:  25%
    /// - community:  30%
    /// - engagement: 20%
    pub fn combined_score(&self) -> f64 {
        self.identity * 0.25
            + self.reputation * 0.25
            + self.community * 0.30
            + self.engagement * 0.20
    }

    /// Derive the consciousness tier from this profile's combined score.
    pub fn tier(&self) -> ConsciousnessTier {
        ConsciousnessTier::from_score(self.combined_score())
    }

    /// Derive consciousness tier with hysteresis, given the current tier.
    ///
    /// Prevents rapid tier oscillation from measurement noise at boundaries.
    pub fn tier_with_hysteresis(&self, current_tier: ConsciousnessTier) -> ConsciousnessTier {
        ConsciousnessTier::from_score_with_hysteresis(self.combined_score(), current_tier)
    }

    /// Compute continuous vote weight for governance participation.
    ///
    /// Uses sigmoid function centered at Citizen threshold (0.4)
    /// with temperature 0.05 for noise-tolerant transitions.
    pub fn vote_weight_continuous(&self) -> f64 {
        continuous_vote_weight(
            self.combined_score(),
            0.4, // Citizen threshold
            VOTE_WEIGHT_TEMPERATURE,
            VOTE_WEIGHT_MAX_BP,
        )
    }

    /// Create a profile with all dimensions at zero (anonymous, no history).
    pub fn zero() -> Self {
        Self {
            identity: 0.0,
            reputation: 0.0,
            community: 0.0,
            engagement: 0.0,
        }
    }

    /// Sanitize a single f64 dimension: NaN/Infinity → 0.0, then clamp to [0, 1].
    #[inline]
    fn sanitize(v: f64) -> f64 {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Clamp all dimensions to 0.0–1.0, replacing NaN/Infinity with 0.0.
    pub fn clamped(&self) -> Self {
        Self {
            identity: Self::sanitize(self.identity),
            reputation: Self::sanitize(self.reputation),
            community: Self::sanitize(self.community),
            engagement: Self::sanitize(self.engagement),
        }
    }

    /// Returns true if all dimensions are finite (not NaN or Infinity).
    pub fn is_valid(&self) -> bool {
        self.identity.is_finite()
            && self.reputation.is_finite()
            && self.community.is_finite()
            && self.engagement.is_finite()
    }

    // ════════════════════════════════════════════════════════════════════════
    // MINIMAL VIABLE BRIDGE: Symthaea → Mycelix mapping
    // ════════════════════════════════════════════════════════════════════════

    /// Create a profile from a Symthaea unified consciousness score.
    ///
    /// This is the **Minimal Viable Bridge**: one Symthaea metric (C_unified)
    /// maps 1:1 to the engagement dimension. Other dimensions come from their
    /// respective sources (identity bridge, reputation history, peer attestations).
    ///
    /// # Arguments
    /// * `unified_consciousness` — C_unified from `ConsciousnessEngineOutput` \[0, 1\]
    /// * `identity` — from identity bridge (MFA assurance level) \[0, 1\]
    /// * `reputation` — from reputation bridge (30-day decayed history) \[0, 1\]
    /// * `community` — from community attestations (peer trust) \[0, 1\]
    ///
    /// All values are clamped to \[0, 1\].
    pub fn from_unified_consciousness(
        unified_consciousness: f64,
        identity: f64,
        reputation: f64,
        community: f64,
    ) -> Self {
        Self {
            identity: identity.clamp(0.0, 1.0),
            reputation: reputation.clamp(0.0, 1.0),
            community: community.clamp(0.0, 1.0),
            engagement: unified_consciousness.clamp(0.0, 1.0),
        }
    }
}

impl Default for ConsciousnessProfile {
    fn default() -> Self {
        Self::zero()
    }
}

/// Time-limited credential containing a `ConsciousnessProfile`.
///
/// Stored on the agent's source chain. Governance zomes validate locally
/// by checking issuer and expiry — no cross-cluster call needed at
/// governance time.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsciousnessCredential {
    /// Agent's DID (e.g., "did:mycelix:<pubkey>").
    pub did: String,
    /// The multi-dimensional profile.
    pub profile: ConsciousnessProfile,
    /// Derived tier at issuance time.
    pub tier: ConsciousnessTier,
    /// Issuance timestamp (microseconds since epoch).
    pub issued_at: u64,
    /// Expiry timestamp (default: issued_at + 24 hours).
    pub expires_at: u64,
    /// DID of the issuing bridge (e.g., "did:mycelix:<identity_bridge_pubkey>").
    pub issuer: String,
    /// BLAKE3 hash of the agent's behavioral trajectory (from TrajectoryAccumulator).
    #[serde(default)]
    pub trajectory_commitment: Option<[u8; 32]>,
    /// Extensible key-value store for future credential features.
    /// Avoids adding new `Option<T>` fields for every feature.
    #[serde(default)]
    pub extensions: std::collections::HashMap<String, Vec<u8>>,
}

impl ConsciousnessCredential {
    /// Default TTL for credentials: 24 hours in microseconds.
    pub const DEFAULT_TTL_US: u64 = 86_400_000_000;

    /// Check if the credential has expired relative to the given timestamp.
    pub fn is_expired(&self, now_us: u64) -> bool {
        now_us >= self.expires_at
    }

    /// Issue a credential from a unified consciousness score (Minimal Viable Bridge).
    ///
    /// Creates a `ConsciousnessProfile` using [`ConsciousnessProfile::from_unified_consciousness`],
    /// derives the tier, and wraps it in a 24h credential.
    ///
    /// # Arguments
    /// * `did` — Agent's DID string
    /// * `unified_consciousness` — C_unified from Symthaea's consciousness engine \[0, 1\]
    /// * `identity` / `reputation` / `community` — other profile dimensions \[0, 1\]
    /// * `issuer` — DID of the issuing bridge zome
    /// * `now_us` — current time in microseconds since epoch
    pub fn from_unified_consciousness(
        did: String,
        unified_consciousness: f64,
        identity: f64,
        reputation: f64,
        community: f64,
        issuer: String,
        now_us: u64,
    ) -> Self {
        let profile = ConsciousnessProfile::from_unified_consciousness(
            unified_consciousness,
            identity,
            reputation,
            community,
        );
        let tier = profile.clamped().tier();
        Self {
            did,
            profile,
            tier,
            issued_at: now_us,
            expires_at: now_us + Self::DEFAULT_TTL_US,
            issuer,
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        }
    }

    /// Set an extension value.
    pub fn set_extension(&mut self, key: impl Into<String>, value: Vec<u8>) {
        self.extensions.insert(key.into(), value);
    }

    /// Get an extension value.
    pub fn get_extension(&self, key: &str) -> Option<&Vec<u8>> {
        self.extensions.get(key)
    }

    /// Remove an extension value.
    pub fn remove_extension(&mut self, key: &str) -> Option<Vec<u8>> {
        self.extensions.remove(key)
    }

    /// Set the trajectory commitment from a TrajectoryAccumulator's output.
    pub fn with_trajectory_commitment(mut self, commitment: [u8; 32]) -> Self {
        self.trajectory_commitment = Some(commitment);
        self
    }

    /// Check if the credential has a trajectory binding.
    pub fn has_trajectory_binding(&self) -> bool {
        self.trajectory_commitment.is_some()
    }
}

// ============================================================================
// Tiers
// ============================================================================

/// Governance tiers derived from combined consciousness score.
///
/// Mirrors the TrustTier concept from the identity cluster, but defined
/// here as the canonical shared definition across all clusters.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConsciousnessTier {
    /// combined < 0.3 — can read, no governance participation
    Observer,
    /// combined >= 0.3 — basic proposals
    Participant,
    /// combined >= 0.4 — voting rights
    Citizen,
    /// combined >= 0.6 — constitutional actions
    Steward,
    /// combined >= 0.8 — emergency powers
    Guardian,
}

impl ConsciousnessTier {
    /// Derive a tier from a combined consciousness score.
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Guardian
        } else if score >= 0.6 {
            Self::Steward
        } else if score >= 0.4 {
            Self::Citizen
        } else if score >= 0.3 {
            Self::Participant
        } else {
            Self::Observer
        }
    }

    /// Minimum combined score required for this tier.
    pub fn min_score(&self) -> f64 {
        match self {
            Self::Observer => 0.0,
            Self::Participant => 0.3,
            Self::Citizen => 0.4,
            Self::Steward => 0.6,
            Self::Guardian => 0.8,
        }
    }

    /// Progressive vote weight in basis points (0–10000).
    ///
    /// Observers cannot vote. Weight increases with tier.
    pub fn vote_weight_bp(&self) -> u32 {
        match self {
            Self::Observer => 0,
            Self::Participant => 5000,
            Self::Citizen => 7500,
            Self::Steward => 10000,
            Self::Guardian => 10000,
        }
    }

    /// Tier transition with hysteresis to prevent oscillation at boundaries.
    ///
    /// Promotion requires crossing `threshold + TIER_HYSTERESIS_MARGIN`.
    /// Demotion requires dropping below `threshold - TIER_HYSTERESIS_MARGIN`.
    /// This prevents rapid tier flapping from measurement noise.
    pub fn from_score_with_hysteresis(score: f64, current_tier: ConsciousnessTier) -> ConsciousnessTier {
        let margin = TIER_HYSTERESIS_MARGIN;

        // Determine what tier we'd promote to (higher threshold required)
        let promoted = if score >= 0.8 + margin {
            ConsciousnessTier::Guardian
        } else if score >= 0.6 + margin {
            ConsciousnessTier::Steward
        } else if score >= 0.4 + margin {
            ConsciousnessTier::Citizen
        } else if score >= 0.3 + margin {
            ConsciousnessTier::Participant
        } else {
            ConsciousnessTier::Observer
        };

        // Determine what tier we'd demote to (lower threshold required)
        let demoted = if score < 0.3 - margin {
            ConsciousnessTier::Observer
        } else if score < 0.4 - margin {
            ConsciousnessTier::Participant
        } else if score < 0.6 - margin {
            ConsciousnessTier::Citizen
        } else if score < 0.8 - margin {
            ConsciousnessTier::Steward
        } else {
            ConsciousnessTier::Guardian
        };

        // Promote if we'd go higher, demote if we'd go lower, else stay
        if promoted > current_tier {
            promoted
        } else if demoted < current_tier {
            demoted
        } else {
            current_tier
        }
    }

    /// Degrade the tier by `levels` steps. Floors at Observer.
    pub fn degrade(self, levels: u32) -> Self {
        let tiers = [
            Self::Observer,
            Self::Participant,
            Self::Citizen,
            Self::Steward,
            Self::Guardian,
        ];
        let current_idx = tiers.iter().position(|t| *t == self).unwrap_or(0);
        let new_idx = current_idx.saturating_sub(levels as usize);
        tiers[new_idx]
    }

    /// Upgrade one level, capped at `max_tier`.
    pub fn upgrade_capped(self, max_tier: Self) -> Self {
        let tiers = [
            Self::Observer,
            Self::Participant,
            Self::Citizen,
            Self::Steward,
            Self::Guardian,
        ];
        let current_idx = tiers.iter().position(|t| *t == self).unwrap_or(0);
        let max_idx = tiers.iter().position(|t| *t == max_tier).unwrap_or(0);
        let new_idx = (current_idx + 1).min(max_idx);
        tiers[new_idx]
    }
}

// ============================================================================
// Governance requirements
// ============================================================================

/// What a governance action requires from the consciousness profile.
///
/// The `min_tier` is always checked. Optional per-dimension minimums
/// add additional requirements (e.g., constitutional changes require
/// minimum identity verification AND community trust).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GovernanceRequirement {
    /// Minimum consciousness tier required.
    pub min_tier: ConsciousnessTier,
    /// Optional minimum identity dimension (None = no minimum).
    pub min_identity: Option<f64>,
    /// Optional minimum community dimension (None = no minimum).
    pub min_community: Option<f64>,
}

/// Result of evaluating a profile against a governance requirement.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GovernanceEligibility {
    /// Whether the agent meets all requirements.
    pub eligible: bool,
    /// Progressive vote weight in basis points (0–10000).
    pub weight_bp: u32,
    /// The agent's derived consciousness tier.
    pub tier: ConsciousnessTier,
    /// The agent's consciousness profile.
    pub profile: ConsciousnessProfile,
    /// Why ineligible (empty if eligible).
    pub reasons: Vec<String>,
    /// Restoration progress (0.0–1.0). 1.0 for non-blacklisted agents.
    /// Only meaningful when used with `evaluate_governance_with_reputation`.
    #[serde(default = "default_restoration_progress")]
    pub restoration_progress: f64,
}

fn default_restoration_progress() -> f64 {
    1.0
}

// ============================================================================
// Gate audit input
// ============================================================================

/// Input for logging a governance gate decision via the bridge's
/// `log_governance_gate` extern.
///
/// Each gated coordinator constructs this after `evaluate_governance()`
/// and fires it as a best-effort cross-zome call to the cluster bridge.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GateAuditInput {
    /// The extern function that triggered the gate check.
    pub action_name: String,
    /// The zome that performed the check.
    pub zome_name: String,
    /// Whether the agent met all requirements.
    pub eligible: bool,
    /// The agent's derived consciousness tier (Debug-formatted).
    pub actual_tier: String,
    /// The minimum tier required by the governance action (Debug-formatted).
    pub required_tier: String,
    /// Progressive vote weight in basis points (0–10000).
    pub weight_bp: u32,
    /// Optional correlation ID for cross-cluster audit trail linkage.
    /// Format: "<agent_hex_prefix>:<timestamp_us>" — generated at action entry points.
    #[serde(default)]
    pub correlation_id: Option<String>,
}

// ============================================================================
// Evaluation (pure functions — no HDK dependency)
// ============================================================================

/// Grace period for recently-expired credentials: 30 minutes in microseconds.
///
/// During the grace period, basic/read operations (Participant tier or below)
/// are still allowed — giving the system time to proactively refresh.
pub const GRACE_PERIOD_US: u64 = 1_800_000_000;

/// Refresh window: proactively refresh credentials within 2 hours of expiry.
pub const REFRESH_WINDOW_US: u64 = 7_200_000_000;

/// Determine whether a gate decision should be logged to the audit trail.
///
/// Strategy: always log rejections and high-tier actions; sample 10% of
/// basic/proposal approvals to reduce DHT write load.
pub fn should_audit(
    requirement: &GovernanceRequirement,
    eligible: bool,
    agent_hash: &[u8],
    action_name: &str,
) -> bool {
    // Always log rejections
    if !eligible {
        return true;
    }
    match requirement.min_tier {
        // Always log constitutional and voting actions
        ConsciousnessTier::Steward | ConsciousnessTier::Guardian => true,
        ConsciousnessTier::Citizen => true,
        // Sample ~10% of basic/proposal approvals using action-salted hash
        _ => {
            let sample_byte = agent_hash.last().copied().unwrap_or(0);
            let salt: u8 = action_name.bytes().fold(0u8, |acc, b| acc.wrapping_add(b));
            sample_byte.wrapping_add(salt) < 26 // ~10% of 256
        }
    }
}

/// Check if a credential is within the refresh window (nearing expiry).
pub fn needs_refresh(credential: &ConsciousnessCredential, now_us: u64) -> bool {
    !credential.is_expired(now_us)
        && credential.expires_at > now_us
        && credential.expires_at - now_us < REFRESH_WINDOW_US
}

// ============================================================================
// Bootstrap (cold-start communities)
// ============================================================================

/// Check whether an agent is eligible for a bootstrap credential.
pub fn is_bootstrap_eligible(agent_count: u32, identity_score: f64) -> bool {
    agent_count < BOOTSTRAP_COMMUNITY_THRESHOLD && identity_score >= BOOTSTRAP_MIN_IDENTITY
}

/// Create a bootstrap credential for cold-start communities.
pub fn bootstrap_credential(
    did: String,
    identity_score: f64,
    now_us: u64,
) -> ConsciousnessCredential {
    let clamped_identity = identity_score.clamp(0.0, 1.0);
    ConsciousnessCredential {
        did,
        profile: ConsciousnessProfile {
            identity: clamped_identity,
            reputation: 0.0,
            community: 0.0,
            engagement: 0.0,
        },
        tier: ConsciousnessTier::Participant,
        issued_at: now_us,
        expires_at: now_us.saturating_add(BOOTSTRAP_TTL_US),
        issuer: "did:mycelix:bootstrap".to_string(),
        trajectory_commitment: None,
        extensions: std::collections::HashMap::new(),
    }
}

/// Evaluate a bootstrap credential against a governance requirement.
/// Capped at Participant tier — voting/constitutional/guardian always rejected.
pub fn evaluate_bootstrap_governance(
    credential: &ConsciousnessCredential,
    requirement: &GovernanceRequirement,
    now_us: u64,
) -> GovernanceEligibility {
    if credential.is_expired(now_us) {
        return GovernanceEligibility {
            eligible: false,
            weight_bp: 0,
            tier: ConsciousnessTier::Observer,
            profile: credential.profile.clone(),
            reasons: vec!["Bootstrap credential expired".into()],
            restoration_progress: 1.0,
        };
    }
    if requirement.min_tier > ConsciousnessTier::Participant {
        return GovernanceEligibility {
            eligible: false,
            weight_bp: 0,
            tier: credential.tier,
            profile: credential.profile.clone(),
            reasons: vec![format!(
                "Bootstrap credentials are capped at Participant; {:?} required",
                requirement.min_tier,
            )],
            restoration_progress: 1.0,
        };
    }
    if let Some(min_id) = requirement.min_identity {
        if credential.profile.identity < min_id {
            return GovernanceEligibility {
                eligible: false,
                weight_bp: 0,
                tier: credential.tier,
                profile: credential.profile.clone(),
                reasons: vec![format!(
                    "Identity {:.2} below required {:.2}",
                    credential.profile.identity, min_id,
                )],
                restoration_progress: 1.0,
            };
        }
    }
    GovernanceEligibility {
        eligible: true,
        weight_bp: 5_000,
        tier: ConsciousnessTier::Participant,
        profile: credential.profile.clone(),
        reasons: vec!["Bootstrap credential: temporary Participant access".into()],
        restoration_progress: 1.0,
    }
}

/// Evaluate a consciousness credential against a governance requirement.
///
/// This is the core gating function. It checks credential expiry first,
/// then evaluates the embedded profile against tier/dimension requirements.
/// Supports a 30-minute grace period for recently-expired credentials on
/// basic (Participant-tier) operations only.
/// Pure — no HDK calls, no side effects.
pub fn evaluate_governance(
    credential: &ConsciousnessCredential,
    requirement: &GovernanceRequirement,
    now_us: u64,
) -> GovernanceEligibility {
    if credential.is_expired(now_us) {
        // Check grace period: allow basic/read operations for 30 min after expiry
        let in_grace = now_us < credential.expires_at.saturating_add(GRACE_PERIOD_US);
        if in_grace && requirement.min_tier <= ConsciousnessTier::Participant {
            // Grace period — evaluate normally but add warning
            let clamped = credential.profile.clamped();
            let tier = clamped.tier();
            let mut reasons = Vec::new();

            if tier < requirement.min_tier {
                reasons.push(format!(
                    "Tier {:?} below required {:?} (score {:.3}, need >= {:.3})",
                    tier,
                    requirement.min_tier,
                    clamped.combined_score(),
                    requirement.min_tier.min_score(),
                ));
            }
            if let Some(min_id) = requirement.min_identity {
                if clamped.identity < min_id {
                    reasons.push(format!(
                        "Identity {:.3} below required {:.3}",
                        clamped.identity, min_id
                    ));
                }
            }
            if let Some(min_comm) = requirement.min_community {
                if clamped.community < min_comm {
                    reasons.push(format!(
                        "Community {:.3} below required {:.3}",
                        clamped.community, min_comm
                    ));
                }
            }

            let eligible = reasons.is_empty();
            let weight_bp = if eligible { tier.vote_weight_bp() } else { 0 };

            // Always add the grace period warning
            reasons.push("Credential in grace period — refresh recommended".into());

            return GovernanceEligibility {
                eligible,
                weight_bp,
                tier,
                profile: clamped,
                reasons,
                restoration_progress: 1.0,
            };
        }

        return GovernanceEligibility {
            eligible: false,
            weight_bp: 0,
            tier: ConsciousnessTier::Observer,
            profile: credential.profile.clone(),
            reasons: vec![format!(
                "Credential expired at {} (now {})",
                credential.expires_at, now_us
            )],
            restoration_progress: 1.0,
        };
    }
    let clamped = credential.profile.clamped();
    let tier = clamped.tier();
    let mut reasons = Vec::new();

    // Check tier
    if tier < requirement.min_tier {
        reasons.push(format!(
            "Tier {:?} below required {:?} (score {:.3}, need >= {:.3})",
            tier,
            requirement.min_tier,
            clamped.combined_score(),
            requirement.min_tier.min_score(),
        ));
    }

    // Check identity minimum
    if let Some(min_id) = requirement.min_identity {
        if clamped.identity < min_id {
            reasons.push(format!(
                "Identity {:.3} below required {:.3}",
                clamped.identity, min_id,
            ));
        }
    }

    // Check community minimum
    if let Some(min_comm) = requirement.min_community {
        if clamped.community < min_comm {
            reasons.push(format!(
                "Community {:.3} below required {:.3}",
                clamped.community, min_comm,
            ));
        }
    }

    let eligible = reasons.is_empty();
    let weight_bp = if eligible { tier.vote_weight_bp() } else { 0 };

    GovernanceEligibility {
        eligible,
        weight_bp,
        tier,
        profile: clamped,
        reasons,
        restoration_progress: 1.0,
    }
}

// ============================================================================
// Standard requirement presets
// ============================================================================

/// Requirement for basic governance participation (viewing proposals, commenting).
///
/// Participant tier (combined >= 0.3), no per-dimension minimums.
pub fn requirement_for_basic() -> GovernanceRequirement {
    GovernanceRequirement {
        min_tier: ConsciousnessTier::Participant,
        min_identity: None,
        min_community: None,
    }
}

/// Requirement for submitting proposals.
///
/// Participant tier + identity >= 0.25 (at least Basic MFA).
pub fn requirement_for_proposal() -> GovernanceRequirement {
    GovernanceRequirement {
        min_tier: ConsciousnessTier::Participant,
        min_identity: Some(0.25),
        min_community: None,
    }
}

/// Requirement for casting votes.
///
/// Citizen tier + identity >= 0.25. Weight scales with tier.
pub fn requirement_for_voting() -> GovernanceRequirement {
    GovernanceRequirement {
        min_tier: ConsciousnessTier::Citizen,
        min_identity: Some(0.25),
        min_community: None,
    }
}

/// Requirement for constitutional changes (bylaw amendments, etc.).
///
/// Steward tier + identity >= 0.5 + community >= 0.3.
pub fn requirement_for_constitutional() -> GovernanceRequirement {
    GovernanceRequirement {
        min_tier: ConsciousnessTier::Steward,
        min_identity: Some(0.5),
        min_community: Some(0.3),
    }
}

/// Requirement for guardian-level operations (system administration, etc.).
///
/// Guardian tier + identity >= 0.7 + community >= 0.5.
pub fn requirement_for_guardian() -> GovernanceRequirement {
    GovernanceRequirement {
        min_tier: ConsciousnessTier::Guardian,
        min_identity: Some(0.7),
        min_community: Some(0.5),
    }
}

// ============================================================================
// Shared consciousness gate (HDK-dependent)
// ============================================================================

/// Fetch the calling agent's consciousness credential via the specified bridge
/// zome and evaluate it against a governance requirement.
///
/// This is the shared implementation of `require_consciousness()` — every
/// coordinator keeps a thin 3-line wrapper that passes its cluster's bridge
/// zome name (e.g., `"commons_bridge"`, `"civic_bridge"`, `"hearth_bridge"`).
///
/// Steps:
/// 1. `agent_info()` → derive DID
/// 2. Cross-zome call to `<bridge_zome>::get_consciousness_credential`
/// 3. `evaluate_governance()` (pure)
/// 4. `should_audit()` → best-effort `log_governance_gate` if sampled
/// 5. Reject with `WasmError` if ineligible
pub fn gate_consciousness(
    bridge_zome: &str,
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    let agent = agent_info()?.agent_initial_pubkey;
    let did = format!("did:mycelix:{}", agent);

    let response = call(
        CallTargetCell::Local,
        ZomeName::new(bridge_zome),
        FunctionName::new("get_consciousness_credential"),
        None,
        did,
    )?;

    let credential: ConsciousnessCredential = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode consciousness credential: {}",
                e
            )))
        })?,
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Consciousness credential call failed: {:?}",
                other
            ))));
        }
    };

    let now_us = sys_time()?.as_micros() as u64;
    let eligibility = evaluate_governance(&credential, requirement, now_us);

    // Fire audit log (best-effort, rate-limited via should_audit)
    if should_audit(
        requirement,
        eligibility.eligible,
        agent.as_ref(),
        action_name,
    ) {
        let audit = GateAuditInput {
            action_name: action_name.to_string(),
            zome_name: zome_info()?.name.to_string(),
            eligible: eligibility.eligible,
            actual_tier: format!("{:?}", eligibility.tier),
            required_tier: format!("{:?}", requirement.min_tier),
            weight_bp: eligibility.weight_bp,
            correlation_id: None,
        };
        match call(
            CallTargetCell::Local,
            ZomeName::new(bridge_zome),
            FunctionName::new("log_governance_gate"),
            None,
            audit,
        ) {
            Ok(_) => {}
            Err(e) => {
                debug!("Audit log failed ({}): {:?}", bridge_zome, e);
            }
        }
    }

    // Best-effort refresh: if credential is nearing expiry, trigger a
    // background refresh call so the next gate check gets a fresh credential.
    // This does NOT block the current check — the credential is still valid.
    if needs_refresh(&credential, now_us) {
        debug!(
            "gate_consciousness: credential nearing expiry, triggering best-effort refresh via {}",
            bridge_zome
        );
        match call(
            CallTargetCell::Local,
            ZomeName::new(bridge_zome),
            FunctionName::new("refresh_consciousness_credential"),
            None,
            credential.did.clone(),
        ) {
            Ok(_) => {
                debug!(
                    "gate_consciousness: refresh triggered successfully via {}",
                    bridge_zome
                );
            }
            Err(e) => {
                debug!(
                    "gate_consciousness: refresh failed (non-fatal) via {}: {:?}",
                    bridge_zome, e
                );
            }
        }
    }

    if !eligibility.eligible {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Consciousness gate: tier {:?} insufficient. Reasons: {}",
            eligibility.tier,
            eligibility.reasons.join(", ")
        ))));
    }

    Ok(eligibility)
}

// ============================================================================
// Governance audit query types
// ============================================================================

/// Filter for querying governance gate audit events.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct GovernanceAuditFilter {
    /// Filter by the gated action name (e.g., "register_property").
    #[serde(default)]
    pub action_name: Option<String>,
    /// Filter by the zome that performed the check.
    #[serde(default)]
    pub zome_name: Option<String>,
    /// Filter by eligibility outcome.
    #[serde(default)]
    pub eligible: Option<bool>,
    /// Start of time range (inclusive), microseconds since epoch.
    #[serde(default)]
    pub from_us: Option<i64>,
    /// End of time range (inclusive), microseconds since epoch.
    #[serde(default)]
    pub to_us: Option<i64>,
}

/// Result of a governance audit query.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GovernanceAuditResult {
    /// Matched audit entries (deserialized from BridgeEventEntry payloads).
    pub entries: Vec<GateAuditInput>,
    /// Number of entries that matched the filter.
    pub total_matched: u32,
}

// ============================================================================
// Reputation Decay, Slashing, and Restoration
// ============================================================================

/// Reputation decay rate per day (multiplicative).
/// Half-life ~347 days: `0.998^347 ~ 0.500`.
///
/// Basis: Dunbar (2010) — social relationships require maintenance;
/// trust fades without sustained positive interaction.
pub const REPUTATION_DECAY_PER_DAY: f64 = 0.998;

/// Slashing factor for detected Byzantine behavior.
/// Applied as: `reputation *= (1.0 - SLASH_FACTOR)`.
///
/// Basis: Ostrom (1990) — graduated sanctions in commons governance.
/// First offense halves reputation; recovery is possible but slow.
pub const REPUTATION_SLASH_FACTOR: f64 = 0.5;

/// Reputation floor below which an agent is considered blacklisted.
/// At this level, the agent cannot participate in governance.
///
/// Basis: Axelrod (1984) — sustained defection warrants exclusion,
/// but the threshold must be low enough to allow recovery.
pub const REPUTATION_BLACKLIST_THRESHOLD: f64 = 0.05;

/// Minimum consecutive good interactions to lift a blacklist.
///
/// Basis: Ubuntu restorative justice — redemption through demonstrated
/// commitment to community norms. 100 interactions ~ weeks of good behavior.
pub const REPUTATION_RESTORATION_INTERACTIONS: u32 = 100;

/// Maximum slash events before permanent reputation cap at 0.5.
/// Prevents repeated gaming of slash/recovery cycles.
///
/// Basis: Three-strikes principle with proportional response.
pub const REPUTATION_MAX_SLASHES: u32 = 5;

/// Reputation state for an agent, tracking decay and sanctions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationState {
    /// Current reputation score (0.0-1.0).
    pub score: f64,
    /// Last update timestamp (microseconds since epoch).
    pub last_updated_us: u64,
    /// Number of consecutive good interactions since last slash.
    pub consecutive_good: u32,
    /// Total slash events in this agent's history.
    pub total_slashes: u32,
    /// Whether the agent is currently blacklisted.
    pub blacklisted: bool,
    /// Timestamp at which blacklist was applied (for duration tracking).
    pub blacklisted_since_us: Option<u64>,
}

impl Default for ReputationState {
    fn default() -> Self {
        Self {
            score: 0.0,
            last_updated_us: 0,
            consecutive_good: 0,
            total_slashes: 0,
            blacklisted: false,
            blacklisted_since_us: None,
        }
    }
}

impl ReputationState {
    /// Create a new reputation state with the given initial score and timestamp.
    pub fn new(initial_score: f64, now_us: u64) -> Self {
        let sanitized = if initial_score.is_finite() {
            initial_score.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            score: sanitized,
            last_updated_us: now_us,
            ..Default::default()
        }
    }

    /// Apply temporal decay based on elapsed time.
    ///
    /// `reputation *= DECAY^days_elapsed`
    ///
    /// Uses microsecond timestamps. Safe for NaN/Inf inputs (clamps to 0).
    pub fn apply_decay(&mut self, now_us: u64) {
        if now_us <= self.last_updated_us {
            return; // No time elapsed or clock skew
        }
        let elapsed_us = now_us - self.last_updated_us;
        let elapsed_days = elapsed_us as f64 / 86_400_000_000.0;

        if !elapsed_days.is_finite() || elapsed_days <= 0.0 {
            return;
        }

        let decay_factor = REPUTATION_DECAY_PER_DAY.powf(elapsed_days);
        if decay_factor.is_finite() {
            self.score = (self.score * decay_factor).clamp(0.0, 1.0);
        } else {
            self.score = 0.0; // Extreme elapsed time -> full decay
        }
        self.last_updated_us = now_us;

        // Check blacklist threshold after decay
        self.check_blacklist(now_us);
    }

    /// Record a positive interaction, incrementing the consecutive good count.
    ///
    /// If the agent is blacklisted and reaches RESTORATION_INTERACTIONS,
    /// the blacklist is lifted (Ubuntu restorative justice model).
    pub fn record_good_interaction(&mut self, reputation_boost: f64, now_us: u64) {
        self.apply_decay(now_us);
        self.consecutive_good = self.consecutive_good.saturating_add(1);

        // Apply reputation boost (clamped to prevent abuse)
        let effective_boost = reputation_boost.clamp(0.0, 0.1);
        let cap = if self.total_slashes >= REPUTATION_MAX_SLASHES {
            0.5 // Permanent cap after max slashes
        } else {
            1.0
        };
        self.score = (self.score + effective_boost).clamp(0.0, cap);

        // Check restoration
        if self.blacklisted && self.consecutive_good >= REPUTATION_RESTORATION_INTERACTIONS {
            self.blacklisted = false;
            self.blacklisted_since_us = None;
            // Restore to minimum Participant threshold
            self.score = self.score.max(0.1);
        }
    }

    /// Apply a reputation slash for detected Byzantine behavior.
    ///
    /// Resets consecutive good interactions. Checks blacklist after slashing.
    ///
    /// Returns the new reputation score.
    pub fn slash(&mut self, now_us: u64) -> f64 {
        self.apply_decay(now_us);
        self.score *= 1.0 - REPUTATION_SLASH_FACTOR;
        self.score = self.score.clamp(0.0, 1.0);
        self.consecutive_good = 0;
        self.total_slashes = self.total_slashes.saturating_add(1);
        self.check_blacklist(now_us);
        self.score
    }

    /// Apply a proportional slash with a custom factor.
    ///
    /// `factor` is clamped to [0.0, 1.0]. Applied as: `score *= (1.0 - factor)`.
    pub fn slash_proportional(&mut self, factor: f64, now_us: u64) -> f64 {
        self.apply_decay(now_us);
        let clamped_factor = factor.clamp(0.0, 1.0);
        self.score *= 1.0 - clamped_factor;
        self.score = self.score.clamp(0.0, 1.0);
        self.consecutive_good = 0;
        self.total_slashes = self.total_slashes.saturating_add(1);
        self.check_blacklist(now_us);
        self.score
    }

    /// Check and update blacklist status.
    fn check_blacklist(&mut self, now_us: u64) {
        if self.score < REPUTATION_BLACKLIST_THRESHOLD && !self.blacklisted {
            self.blacklisted = true;
            self.blacklisted_since_us = Some(now_us);
        }
    }

    /// Whether this agent can participate in governance.
    ///
    /// Blacklisted agents are excluded from all governance actions
    /// until they complete the restoration path.
    pub fn can_participate(&self) -> bool {
        !self.blacklisted
    }

    /// Restoration progress (0.0-1.0).
    ///
    /// Returns 1.0 for non-blacklisted agents.
    /// For blacklisted agents, tracks progress toward RESTORATION_INTERACTIONS.
    pub fn restoration_progress(&self) -> f64 {
        if !self.blacklisted {
            return 1.0;
        }
        (self.consecutive_good as f64 / REPUTATION_RESTORATION_INTERACTIONS as f64).clamp(0.0, 1.0)
    }
}

/// Apply reputation decay to a ConsciousnessProfile's reputation dimension.
///
/// Convenience function for use in credential issuance pipelines.
pub fn decay_reputation(profile: &ConsciousnessProfile, elapsed_days: f64) -> ConsciousnessProfile {
    if !elapsed_days.is_finite() || elapsed_days <= 0.0 {
        return profile.clone();
    }
    let decay_factor = REPUTATION_DECAY_PER_DAY.powf(elapsed_days);
    let decayed_rep = if decay_factor.is_finite() {
        (profile.reputation * decay_factor).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ConsciousnessProfile {
        identity: profile.identity,
        reputation: decayed_rep,
        community: profile.community,
        engagement: profile.engagement,
    }
}

/// Evaluate governance with reputation state integration.
///
/// Wraps `evaluate_governance` but also checks blacklist status
/// and applies restoration progress to vote weight.
///
/// # TOCTOU Warning
///
/// This function checks `reputation_state.blacklisted` at call time, but the
/// on-chain reputation may change between this check and the commit of the
/// governance action. Callers in coordinator zomes SHOULD re-check blacklist
/// status at commit time (e.g., in a `validate_*` callback) to close this
/// race window. In practice the risk is low (blacklisting is rare and requires
/// reputation < 0.05), but high-severity governance actions (constitutional
/// amendments, treasury operations) warrant the extra check.
pub fn evaluate_governance_with_reputation(
    credential: &ConsciousnessCredential,
    requirement: &GovernanceRequirement,
    reputation_state: &ReputationState,
    now_us: u64,
) -> GovernanceEligibility {
    // Blacklisted agents cannot participate
    if reputation_state.blacklisted {
        return GovernanceEligibility {
            eligible: false,
            weight_bp: 0,
            tier: ConsciousnessTier::Observer,
            profile: credential.profile.clone(),
            reasons: vec![format!(
                "Blacklisted: reputation {:.3} below threshold {:.3}. Restoration progress: {:.0}%",
                reputation_state.score,
                REPUTATION_BLACKLIST_THRESHOLD,
                reputation_state.restoration_progress() * 100.0,
            )],
            restoration_progress: reputation_state.restoration_progress(),
        };
    }

    // Evaluate normally
    let mut result = evaluate_governance(credential, requirement, now_us);

    // Scale vote weight by slash penalty if agent has been slashed before
    if reputation_state.total_slashes > 0 {
        let slash_penalty = 1.0 - (reputation_state.total_slashes as f64 * 0.05).min(0.25);
        result.weight_bp = (result.weight_bp as f64 * slash_penalty) as u32;
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: a non-expired timestamp 24 hours in the future.
    const NOW: u64 = 1_000_000_000_000;

    /// Build a fresh (non-expired) credential wrapping the given profile.
    fn fresh_credential(profile: ConsciousnessProfile) -> ConsciousnessCredential {
        let tier = profile.clamped().tier();
        ConsciousnessCredential {
            did: "did:test:alice".to_string(),
            profile,
            tier,
            issued_at: NOW - 60_000_000,
            expires_at: NOW + 86_400_000_000,
            issuer: "test".to_string(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        }
    }

    // -- ConsciousnessProfile --

    #[test]
    fn zero_profile_is_all_zeros() {
        let p = ConsciousnessProfile::zero();
        assert_eq!(p.identity, 0.0);
        assert_eq!(p.reputation, 0.0);
        assert_eq!(p.community, 0.0);
        assert_eq!(p.engagement, 0.0);
        assert_eq!(p.combined_score(), 0.0);
    }

    #[test]
    fn combined_score_weights_correct() {
        // All ones → 0.25 + 0.25 + 0.30 + 0.20 = 1.0
        let p = ConsciousnessProfile {
            identity: 1.0,
            reputation: 1.0,
            community: 1.0,
            engagement: 1.0,
        };
        assert!((p.combined_score() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn combined_score_weighted_average() {
        let p = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        assert!((p.combined_score() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn combined_score_identity_only() {
        let p = ConsciousnessProfile {
            identity: 1.0,
            reputation: 0.0,
            community: 0.0,
            engagement: 0.0,
        };
        assert!((p.combined_score() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn combined_score_community_only() {
        let p = ConsciousnessProfile {
            identity: 0.0,
            reputation: 0.0,
            community: 1.0,
            engagement: 0.0,
        };
        assert!((p.combined_score() - 0.30).abs() < 1e-10);
    }

    #[test]
    fn combined_score_engagement_only() {
        let p = ConsciousnessProfile {
            identity: 0.0,
            reputation: 0.0,
            community: 0.0,
            engagement: 1.0,
        };
        assert!((p.combined_score() - 0.20).abs() < 1e-10);
    }

    #[test]
    fn combined_score_reputation_only() {
        let p = ConsciousnessProfile {
            identity: 0.0,
            reputation: 1.0,
            community: 0.0,
            engagement: 0.0,
        };
        assert!((p.combined_score() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn clamped_clips_values() {
        let p = ConsciousnessProfile {
            identity: 1.5,
            reputation: -0.3,
            community: 2.0,
            engagement: -1.0,
        };
        let c = p.clamped();
        assert_eq!(c.identity, 1.0);
        assert_eq!(c.reputation, 0.0);
        assert_eq!(c.community, 1.0);
        assert_eq!(c.engagement, 0.0);
    }

    #[test]
    fn clamped_preserves_valid_values() {
        let p = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.7,
            community: 0.3,
            engagement: 0.9,
        };
        let c = p.clamped();
        assert_eq!(c, p);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(
            ConsciousnessProfile::default(),
            ConsciousnessProfile::zero()
        );
    }

    // -- ConsciousnessTier --

    #[test]
    fn tier_from_score_boundaries() {
        assert_eq!(
            ConsciousnessTier::from_score(0.0),
            ConsciousnessTier::Observer
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.29),
            ConsciousnessTier::Observer
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.3),
            ConsciousnessTier::Participant
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.39),
            ConsciousnessTier::Participant
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.4),
            ConsciousnessTier::Citizen
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.59),
            ConsciousnessTier::Citizen
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.6),
            ConsciousnessTier::Steward
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.79),
            ConsciousnessTier::Steward
        );
        assert_eq!(
            ConsciousnessTier::from_score(0.8),
            ConsciousnessTier::Guardian
        );
        assert_eq!(
            ConsciousnessTier::from_score(1.0),
            ConsciousnessTier::Guardian
        );
    }

    #[test]
    fn tier_min_scores_are_monotonic() {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        for i in 1..tiers.len() {
            assert!(
                tiers[i].min_score() > tiers[i - 1].min_score(),
                "{:?} min_score should be > {:?} min_score",
                tiers[i],
                tiers[i - 1]
            );
        }
    }

    #[test]
    fn tier_vote_weights_are_progressive() {
        assert_eq!(ConsciousnessTier::Observer.vote_weight_bp(), 0);
        assert!(ConsciousnessTier::Participant.vote_weight_bp() > 0);
        assert!(
            ConsciousnessTier::Citizen.vote_weight_bp()
                >= ConsciousnessTier::Participant.vote_weight_bp()
        );
        assert!(
            ConsciousnessTier::Steward.vote_weight_bp()
                >= ConsciousnessTier::Citizen.vote_weight_bp()
        );
        assert!(
            ConsciousnessTier::Guardian.vote_weight_bp()
                >= ConsciousnessTier::Steward.vote_weight_bp()
        );
    }

    #[test]
    fn tier_ordering() {
        assert!(ConsciousnessTier::Observer < ConsciousnessTier::Participant);
        assert!(ConsciousnessTier::Participant < ConsciousnessTier::Citizen);
        assert!(ConsciousnessTier::Citizen < ConsciousnessTier::Steward);
        assert!(ConsciousnessTier::Steward < ConsciousnessTier::Guardian);
    }

    // -- Profile → Tier integration --

    #[test]
    fn profile_tier_derivation() {
        let observer = ConsciousnessProfile::zero();
        assert_eq!(observer.tier(), ConsciousnessTier::Observer);

        let participant = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.3,
            community: 0.3,
            engagement: 0.3,
        };
        assert_eq!(participant.tier(), ConsciousnessTier::Participant);

        let citizen = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.4,
            engagement: 0.2,
        };
        // 0.5*0.25 + 0.5*0.25 + 0.4*0.30 + 0.2*0.20 = 0.125+0.125+0.12+0.04 = 0.41
        assert_eq!(citizen.tier(), ConsciousnessTier::Citizen);

        let steward = ConsciousnessProfile {
            identity: 0.75,
            reputation: 0.7,
            community: 0.6,
            engagement: 0.5,
        };
        // 0.75*0.25 + 0.7*0.25 + 0.6*0.30 + 0.5*0.20 = 0.1875+0.175+0.18+0.10 = 0.6425
        assert_eq!(steward.tier(), ConsciousnessTier::Steward);

        let guardian = ConsciousnessProfile {
            identity: 1.0,
            reputation: 0.9,
            community: 0.8,
            engagement: 0.7,
        };
        // 1.0*0.25 + 0.9*0.25 + 0.8*0.30 + 0.7*0.20 = 0.25+0.225+0.24+0.14 = 0.855
        assert_eq!(guardian.tier(), ConsciousnessTier::Guardian);
    }

    // -- evaluate_governance --

    #[test]
    fn evaluate_observer_rejected_for_basic() {
        let profile = ConsciousnessProfile::zero();
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_basic(),
            NOW,
        );
        assert!(!result.eligible);
        assert_eq!(result.weight_bp, 0);
        assert_eq!(result.tier, ConsciousnessTier::Observer);
        assert!(!result.reasons.is_empty());
    }

    #[test]
    fn evaluate_participant_passes_basic() {
        let profile = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.3,
            community: 0.3,
            engagement: 0.3,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_basic(),
            NOW,
        );
        assert!(result.eligible);
        assert_eq!(result.weight_bp, 5000);
        assert_eq!(result.tier, ConsciousnessTier::Participant);
        assert!(result.reasons.is_empty());
    }

    #[test]
    fn evaluate_participant_rejected_for_voting() {
        let profile = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.3,
            community: 0.3,
            engagement: 0.3,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_voting(),
            NOW,
        );
        assert!(!result.eligible);
        assert!(!result.reasons.is_empty());
    }

    #[test]
    fn evaluate_citizen_passes_voting() {
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.4,
            engagement: 0.2,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_voting(),
            NOW,
        );
        assert!(result.eligible);
        assert_eq!(result.weight_bp, 7500); // Citizen weight
        assert_eq!(result.tier, ConsciousnessTier::Citizen);
    }

    #[test]
    fn evaluate_proposal_requires_identity() {
        // High combined score but zero identity
        let profile = ConsciousnessProfile {
            identity: 0.0,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_proposal(),
            NOW,
        );
        assert!(!result.eligible);
        assert!(result.reasons.iter().any(|r| r.contains("Identity")));
    }

    #[test]
    fn evaluate_constitutional_requires_all() {
        // Steward tier but low identity
        let profile = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.8,
            community: 0.8,
            engagement: 0.8,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_constitutional(),
            NOW,
        );
        assert!(!result.eligible);
        assert!(result.reasons.iter().any(|r| r.contains("Identity")));
    }

    #[test]
    fn evaluate_constitutional_requires_community() {
        // Steward tier, good identity, but low community
        let profile = ConsciousnessProfile {
            identity: 0.75,
            reputation: 0.7,
            community: 0.1,
            engagement: 0.8,
        };
        // combined = 0.1875+0.175+0.03+0.16 = 0.5525 → Citizen (not Steward!)
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_constitutional(),
            NOW,
        );
        assert!(!result.eligible);
        // Should fail on tier AND community
        assert!(result
            .reasons
            .iter()
            .any(|r| r.contains("Community") || r.contains("Tier")));
    }

    #[test]
    fn evaluate_guardian_passes_constitutional() {
        let profile = ConsciousnessProfile {
            identity: 1.0,
            reputation: 0.9,
            community: 0.8,
            engagement: 0.7,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_constitutional(),
            NOW,
        );
        assert!(result.eligible);
        assert_eq!(result.weight_bp, 10000);
        assert_eq!(result.tier, ConsciousnessTier::Guardian);
    }

    #[test]
    fn evaluate_clamps_out_of_range_values() {
        let profile = ConsciousnessProfile {
            identity: 2.0,
            reputation: 2.0,
            community: 2.0,
            engagement: 2.0,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_constitutional(),
            NOW,
        );
        assert!(result.eligible);
        // Clamped to 1.0 each → combined = 1.0 → Guardian
        assert_eq!(result.tier, ConsciousnessTier::Guardian);
        assert_eq!(result.profile.identity, 1.0);
    }

    #[test]
    fn evaluate_multiple_failure_reasons() {
        let profile = ConsciousnessProfile::zero();
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_constitutional(),
            NOW,
        );
        assert!(!result.eligible);
        // Should fail on tier, identity, and community
        assert!(
            result.reasons.len() >= 3,
            "Expected 3+ reasons, got: {:?}",
            result.reasons
        );
    }

    // -- Progressive weight composition --

    #[test]
    fn progressive_weight_composition_with_role() {
        // Simulates hearth-decisions: final_weight = role_bp * consciousness_bp / 10000
        let citizen_bp: u64 = ConsciousnessTier::Citizen.vote_weight_bp() as u64;
        let adult_role_bp: u64 = 10000; // Adult
        let youth_role_bp: u64 = 5000; // Youth

        let adult_final = (adult_role_bp * citizen_bp / 10000) as u32;
        let youth_final = (youth_role_bp * citizen_bp / 10000) as u32;

        assert_eq!(adult_final, 7500);
        assert_eq!(youth_final, 3750);
    }

    // -- ConsciousnessCredential --

    #[test]
    fn credential_not_expired_when_fresh() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 1_000_000,
            expires_at: 1_000_000 + ConsciousnessCredential::DEFAULT_TTL_US,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        assert!(!cred.is_expired(1_000_000));
        assert!(!cred.is_expired(1_000_000 + ConsciousnessCredential::DEFAULT_TTL_US - 1));
    }

    #[test]
    fn credential_expired_at_boundary() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 1_000_000,
            expires_at: 1_000_000 + ConsciousnessCredential::DEFAULT_TTL_US,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        assert!(cred.is_expired(cred.expires_at));
        assert!(cred.is_expired(cred.expires_at + 1));
    }

    #[test]
    fn default_ttl_is_24_hours() {
        assert_eq!(ConsciousnessCredential::DEFAULT_TTL_US, 86_400_000_000);
    }

    // -- Serde roundtrips --

    #[test]
    fn profile_serde_roundtrip() {
        let p = ConsciousnessProfile {
            identity: 0.75,
            reputation: 0.5,
            community: 0.6,
            engagement: 0.3,
        };
        let json = serde_json::to_string(&p).unwrap();
        let p2: ConsciousnessProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn tier_serde_roundtrip() {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        for tier in &tiers {
            let json = serde_json::to_string(tier).unwrap();
            let t2: ConsciousnessTier = serde_json::from_str(&json).unwrap();
            assert_eq!(*tier, t2);
        }
    }

    #[test]
    fn credential_serde_roundtrip() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:abc123".into(),
            profile: ConsciousnessProfile {
                identity: 0.5,
                reputation: 0.6,
                community: 0.7,
                engagement: 0.4,
            },
            tier: ConsciousnessTier::Steward,
            issued_at: 1_700_000_000_000_000,
            expires_at: 1_700_000_000_000_000 + ConsciousnessCredential::DEFAULT_TTL_US,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&cred).unwrap();
        let c2: ConsciousnessCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.did, "did:mycelix:abc123");
        assert_eq!(c2.tier, ConsciousnessTier::Steward);
        assert_eq!(c2.profile.identity, 0.5);
    }

    #[test]
    fn governance_requirement_serde_roundtrip() {
        let req = requirement_for_constitutional();
        let json = serde_json::to_string(&req).unwrap();
        let r2: GovernanceRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.min_tier, ConsciousnessTier::Steward);
        assert_eq!(r2.min_identity, Some(0.5));
        assert_eq!(r2.min_community, Some(0.3));
    }

    #[test]
    fn governance_eligibility_serde_roundtrip() {
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.4,
            engagement: 0.2,
        };
        let eligibility = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_voting(),
            NOW,
        );
        let json = serde_json::to_string(&eligibility).unwrap();
        let e2: GovernanceEligibility = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.eligible, eligibility.eligible);
        assert_eq!(e2.weight_bp, eligibility.weight_bp);
        assert_eq!(e2.tier, eligibility.tier);
    }

    // -- Edge cases --

    #[test]
    fn negative_values_clamped_to_zero() {
        let profile = ConsciousnessProfile {
            identity: -0.5,
            reputation: -1.0,
            community: -0.1,
            engagement: -999.0,
        };
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_basic(),
            NOW,
        );
        assert!(!result.eligible);
        assert_eq!(result.profile.identity, 0.0);
        assert_eq!(result.profile.reputation, 0.0);
        assert_eq!(result.profile.community, 0.0);
        assert_eq!(result.profile.engagement, 0.0);
    }

    #[test]
    fn exact_threshold_boundary_participant() {
        // combined = exactly 0.3
        let profile = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.3,
            community: 0.3,
            engagement: 0.3,
        };
        assert_eq!(profile.tier(), ConsciousnessTier::Participant);
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_basic(),
            NOW,
        );
        assert!(result.eligible);
    }

    #[test]
    fn just_below_participant_threshold() {
        // combined just under 0.3
        let profile = ConsciousnessProfile {
            identity: 0.29,
            reputation: 0.29,
            community: 0.29,
            engagement: 0.29,
        };
        // 0.29 * (0.25+0.25+0.30+0.20) = 0.29 * 1.0 = 0.29
        assert_eq!(profile.tier(), ConsciousnessTier::Observer);
        let result = evaluate_governance(
            &fresh_credential(profile.clone()),
            &requirement_for_basic(),
            NOW,
        );
        assert!(!result.eligible);
    }

    // -- Requirement presets --

    #[test]
    fn requirement_presets_are_ordered() {
        let basic = requirement_for_basic();
        let proposal = requirement_for_proposal();
        let voting = requirement_for_voting();
        let constitutional = requirement_for_constitutional();

        assert!(basic.min_tier <= proposal.min_tier);
        assert!(proposal.min_tier <= voting.min_tier);
        assert!(voting.min_tier <= constitutional.min_tier);
    }

    #[test]
    fn requirement_identity_thresholds_ordered() {
        let proposal_id = requirement_for_proposal().min_identity.unwrap_or(0.0);
        let voting_id = requirement_for_voting().min_identity.unwrap_or(0.0);
        let const_id = requirement_for_constitutional().min_identity.unwrap_or(0.0);

        assert!(proposal_id <= voting_id || proposal_id <= const_id);
        assert!(voting_id <= const_id);
    }

    // -- Credential expiry edge cases --

    #[test]
    fn credential_not_expired_one_microsecond_before() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 1_000_000,
            expires_at: 2_000_000,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        assert!(!cred.is_expired(1_999_999));
    }

    #[test]
    fn credential_expired_exactly_at_boundary() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 1_000_000,
            expires_at: 2_000_000,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        // >= means expired at exact boundary
        assert!(cred.is_expired(2_000_000));
    }

    #[test]
    fn credential_expired_one_microsecond_after() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 1_000_000,
            expires_at: 2_000_000,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        assert!(cred.is_expired(2_000_001));
    }

    #[test]
    fn credential_zero_ttl_always_expired() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 1_000_000,
            expires_at: 1_000_000, // zero TTL
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        assert!(cred.is_expired(1_000_000));
    }

    #[test]
    fn credential_u64_max_expires_at_not_expired() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 0,
            expires_at: u64::MAX,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        // Any reasonable timestamp is before u64::MAX
        assert!(!cred.is_expired(1_700_000_000_000_000));
    }

    #[test]
    fn credential_u64_max_expires_at_expired_at_max() {
        let cred = ConsciousnessCredential {
            did: "did:mycelix:test".into(),
            profile: ConsciousnessProfile::zero(),
            tier: ConsciousnessTier::Observer,
            issued_at: 0,
            expires_at: u64::MAX,
            issuer: "did:mycelix:issuer".into(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        assert!(cred.is_expired(u64::MAX));
    }

    // -- evaluate_governance expiry integration --

    #[test]
    fn evaluate_governance_rejects_expired_credential_past_grace() {
        let profile = ConsciousnessProfile {
            identity: 1.0,
            reputation: 1.0,
            community: 1.0,
            engagement: 1.0,
        };
        let mut cred = fresh_credential(profile);
        // Expired well past the 30-min grace period
        cred.expires_at = NOW - GRACE_PERIOD_US - 1;
        let result = evaluate_governance(&cred, &requirement_for_basic(), NOW);
        assert!(!result.eligible);
        assert!(result.reasons[0].contains("expired"));
    }

    #[test]
    fn evaluate_governance_rejects_expired_credential_for_voting() {
        let profile = ConsciousnessProfile {
            identity: 1.0,
            reputation: 1.0,
            community: 1.0,
            engagement: 1.0,
        };
        let mut cred = fresh_credential(profile);
        cred.expires_at = NOW - 1; // recently expired
                                   // Grace period does NOT apply to voting-tier operations
        let result = evaluate_governance(&cred, &requirement_for_voting(), NOW);
        assert!(!result.eligible);
        assert!(result.reasons[0].contains("expired"));
    }

    #[test]
    fn evaluate_governance_accepts_fresh_credential() {
        let profile = ConsciousnessProfile {
            identity: 1.0,
            reputation: 1.0,
            community: 1.0,
            engagement: 1.0,
        };
        let cred = fresh_credential(profile);
        let result = evaluate_governance(&cred, &requirement_for_constitutional(), NOW);
        assert!(result.eligible);
    }

    #[test]
    fn evaluate_governance_rejects_at_expiry_boundary_for_voting() {
        let profile = ConsciousnessProfile {
            identity: 1.0,
            reputation: 1.0,
            community: 1.0,
            engagement: 1.0,
        };
        let mut cred = fresh_credential(profile);
        cred.expires_at = NOW; // expires exactly at NOW
                               // Grace period does NOT apply to voting-tier operations
        let result = evaluate_governance(&cred, &requirement_for_voting(), NOW);
        assert!(!result.eligible);
        assert!(result.reasons[0].contains("expired"));
    }

    // -- Progressive weight composition edge cases --

    #[test]
    fn weight_composition_no_overflow_at_max() {
        // (10000 * 10000) / 10000 = 10000 — no overflow in u64 intermediate
        let role_bp: u64 = 10000;
        let consciousness_bp: u64 = 10000;
        let result = (role_bp * consciousness_bp / 10000) as u32;
        assert_eq!(result, 10000);
    }

    #[test]
    fn weight_composition_observer_always_zero() {
        let observer_bp = ConsciousnessTier::Observer.vote_weight_bp() as u64;
        for role_bp in [0u64, 5000, 10000, u32::MAX as u64] {
            let result = (role_bp * observer_bp / 10000) as u32;
            assert_eq!(result, 0, "Observer * role {} should be 0", role_bp);
        }
    }

    #[test]
    fn weight_composition_zero_role_always_zero() {
        for tier in [
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ] {
            let tier_bp = tier.vote_weight_bp() as u64;
            let result = (0u64 * tier_bp / 10000) as u32;
            assert_eq!(result, 0, "Role 0 * {:?} should be 0", tier);
        }
    }

    #[test]
    fn weight_composition_all_tiers_all_standard_roles() {
        // Standard Mycelix role weights: Youth=5000, Adult=10000, Elder=10000
        let roles = [(5000u64, "Youth"), (10000u64, "Adult"), (10000u64, "Elder")];
        let tiers = [
            (ConsciousnessTier::Observer, 0u32),
            (ConsciousnessTier::Participant, 5000),
            (ConsciousnessTier::Citizen, 7500),
            (ConsciousnessTier::Steward, 10000),
            (ConsciousnessTier::Guardian, 10000),
        ];
        for (role_bp, role_name) in &roles {
            for (tier, tier_bp) in &tiers {
                let expected = (*role_bp * *tier_bp as u64 / 10000) as u32;
                let actual = (*role_bp * tier.vote_weight_bp() as u64 / 10000) as u32;
                assert_eq!(
                    actual, expected,
                    "{} ({}) x {:?} ({}): expected {}, got {}",
                    role_name, role_bp, tier, tier_bp, expected, actual
                );
            }
        }
    }

    // -- Tier from_score edge cases --

    #[test]
    fn tier_from_score_negative_is_observer() {
        assert_eq!(
            ConsciousnessTier::from_score(-1.0),
            ConsciousnessTier::Observer
        );
        assert_eq!(
            ConsciousnessTier::from_score(-0.001),
            ConsciousnessTier::Observer
        );
    }

    #[test]
    fn tier_from_score_above_one_is_guardian() {
        assert_eq!(
            ConsciousnessTier::from_score(1.5),
            ConsciousnessTier::Guardian
        );
        assert_eq!(
            ConsciousnessTier::from_score(100.0),
            ConsciousnessTier::Guardian
        );
    }

    // -- Grace period --

    #[test]
    fn grace_period_allows_basic_operations() {
        let profile = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.3,
            community: 0.3,
            engagement: 0.3,
        };
        let mut cred = fresh_credential(profile);
        // Expired 10 minutes ago (within 30-min grace)
        cred.expires_at = NOW - 600_000_000;
        let result = evaluate_governance(&cred, &requirement_for_basic(), NOW);
        assert!(result.eligible);
        assert!(result.reasons.iter().any(|r| r.contains("grace period")));
    }

    #[test]
    fn grace_period_rejects_voting_operations() {
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.4,
            engagement: 0.2,
        };
        let mut cred = fresh_credential(profile);
        cred.expires_at = NOW - 600_000_000; // within grace
        let result = evaluate_governance(&cred, &requirement_for_voting(), NOW);
        assert!(!result.eligible);
        assert!(result.reasons[0].contains("expired"));
    }

    #[test]
    fn grace_period_expired_past_window() {
        let profile = ConsciousnessProfile {
            identity: 0.3,
            reputation: 0.3,
            community: 0.3,
            engagement: 0.3,
        };
        let mut cred = fresh_credential(profile);
        // Expired 2 hours ago (past 30-min grace)
        cred.expires_at = NOW - 7_200_000_000;
        let result = evaluate_governance(&cred, &requirement_for_basic(), NOW);
        assert!(!result.eligible);
    }

    // -- should_audit --

    #[test]
    fn should_audit_always_logs_rejections() {
        assert!(should_audit(&requirement_for_basic(), false, &[0u8], "any"));
        assert!(should_audit(
            &requirement_for_basic(),
            false,
            &[255u8],
            "any"
        ));
    }

    #[test]
    fn should_audit_always_logs_constitutional() {
        assert!(should_audit(
            &requirement_for_constitutional(),
            true,
            &[0u8],
            "amend"
        ));
        assert!(should_audit(
            &requirement_for_constitutional(),
            true,
            &[255u8],
            "amend"
        ));
    }

    #[test]
    fn should_audit_always_logs_voting() {
        assert!(should_audit(
            &requirement_for_voting(),
            true,
            &[0u8],
            "vote"
        ));
        assert!(should_audit(
            &requirement_for_voting(),
            true,
            &[255u8],
            "vote"
        ));
    }

    #[test]
    fn should_audit_samples_basic_approvals() {
        let z = ""; // zero salt
        assert!(should_audit(&requirement_for_basic(), true, &[0u8], z));
        assert!(should_audit(&requirement_for_basic(), true, &[25u8], z));
        assert!(!should_audit(&requirement_for_basic(), true, &[26u8], z));
        assert!(!should_audit(&requirement_for_basic(), true, &[255u8], z));
    }

    #[test]
    fn should_audit_salt_varies_by_action() {
        let agent = &[30u8];
        assert!(!should_audit(&requirement_for_basic(), true, agent, ""));
        // "qq" salt = 226. 30 + 226 = 256 → 0 mod 256 → sampled
        assert!(should_audit(&requirement_for_basic(), true, agent, "qq"));
    }

    // -- needs_refresh --

    #[test]
    fn needs_refresh_within_window() {
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        let mut cred = fresh_credential(profile);
        // Expires in 1 hour (within 2-hour window)
        cred.expires_at = NOW + 3_600_000_000;
        assert!(needs_refresh(&cred, NOW));
    }

    #[test]
    fn needs_refresh_not_within_window() {
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        let cred = fresh_credential(profile);
        // Expires in 24 hours (outside 2-hour window)
        assert!(!needs_refresh(&cred, NOW));
    }

    #[test]
    fn needs_refresh_expired_credential() {
        let profile = ConsciousnessProfile::zero();
        let mut cred = fresh_credential(profile);
        cred.expires_at = NOW - 1;
        assert!(!needs_refresh(&cred, NOW));
    }

    #[test]
    fn needs_refresh_exactly_at_boundary() {
        // Credential expires exactly 2 hours from now — just inside the window
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        let mut cred = fresh_credential(profile.clone());
        // Exactly at REFRESH_WINDOW_US boundary (2 hours = 7_200_000_000 us)
        cred.expires_at = NOW + REFRESH_WINDOW_US;
        // At exactly the boundary, expires_at - now == REFRESH_WINDOW_US,
        // so the condition (expires_at - now < REFRESH_WINDOW_US) is false
        assert!(!needs_refresh(&cred, NOW));

        // One microsecond inside the window
        cred.expires_at = NOW + REFRESH_WINDOW_US - 1;
        assert!(needs_refresh(&cred, NOW));
    }

    #[test]
    fn needs_refresh_just_expired_not_refreshable() {
        // Credential expired 1 microsecond ago — should NOT trigger refresh
        let profile = ConsciousnessProfile {
            identity: 0.8,
            reputation: 0.7,
            community: 0.6,
            engagement: 0.5,
        };
        let mut cred = fresh_credential(profile);
        cred.expires_at = NOW; // expires exactly at NOW
        assert!(!needs_refresh(&cred, NOW));
    }

    #[test]
    fn needs_refresh_far_future_not_refreshable() {
        // Credential expires 23 hours from now — well outside 2-hour window
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        let mut cred = fresh_credential(profile);
        cred.expires_at = NOW + 82_800_000_000; // 23 hours
        assert!(!needs_refresh(&cred, NOW));
    }

    #[test]
    fn refresh_on_gate_check_flow() {
        // Simulate the gate_consciousness flow: credential nearing expiry
        // should still pass the gate check (it's valid) but needs_refresh
        // returns true so refresh would be triggered.
        let profile = ConsciousnessProfile {
            identity: 0.8,
            reputation: 0.7,
            community: 0.6,
            engagement: 0.5,
        };
        let mut cred = fresh_credential(profile);
        // Expires in 30 minutes — within 2-hour refresh window
        cred.expires_at = NOW + 1_800_000_000;

        // Step 1: Gate check — credential is still valid
        assert!(!cred.is_expired(NOW), "credential should NOT be expired");

        // Step 2: Evaluate governance — should be eligible
        let requirement = GovernanceRequirement {
            min_tier: ConsciousnessTier::Participant,
            min_identity: None,
            min_community: None,
        };
        let eligibility = evaluate_governance(&cred, &requirement, NOW);
        assert!(
            eligibility.eligible,
            "nearing-expiry credential should still pass gate"
        );

        // Step 3: needs_refresh should be true — triggering background refresh
        assert!(
            needs_refresh(&cred, NOW),
            "credential nearing expiry should trigger refresh"
        );
    }

    #[test]
    fn refresh_on_gate_check_fresh_credential_no_refresh() {
        // Fresh credential: gate passes and NO refresh triggered
        let profile = ConsciousnessProfile {
            identity: 0.8,
            reputation: 0.7,
            community: 0.6,
            engagement: 0.5,
        };
        let cred = fresh_credential(profile); // expires in 24 hours

        // Gate check passes
        let requirement = GovernanceRequirement {
            min_tier: ConsciousnessTier::Participant,
            min_identity: None,
            min_community: None,
        };
        let eligibility = evaluate_governance(&cred, &requirement, NOW);
        assert!(eligibility.eligible);

        // No refresh needed — credential is fresh
        assert!(
            !needs_refresh(&cred, NOW),
            "fresh credential should NOT trigger refresh"
        );
    }

    #[test]
    fn refresh_on_gate_check_expired_credential_no_refresh() {
        // Expired credential past grace period: gate FAILS and no refresh triggered
        // (refresh is pointless for already-expired credentials)
        let profile = ConsciousnessProfile {
            identity: 0.8,
            reputation: 0.7,
            community: 0.6,
            engagement: 0.5,
        };
        let mut cred = fresh_credential(profile);
        // Expired well past the 30-minute grace period
        cred.expires_at = NOW - GRACE_PERIOD_US - 1;

        // Gate check fails (expired past grace)
        let requirement = GovernanceRequirement {
            min_tier: ConsciousnessTier::Participant,
            min_identity: None,
            min_community: None,
        };
        let eligibility = evaluate_governance(&cred, &requirement, NOW);
        assert!(!eligibility.eligible, "expired credential should fail gate");

        // No refresh for expired credentials
        assert!(
            !needs_refresh(&cred, NOW),
            "expired credential should NOT trigger refresh"
        );
    }

    // -- GateAuditInput with correlation_id --

    #[test]
    fn gate_audit_input_serde_with_correlation_id() {
        let audit = GateAuditInput {
            action_name: "test".into(),
            zome_name: "test_zome".into(),
            eligible: true,
            actual_tier: "Citizen".into(),
            required_tier: "Participant".into(),
            weight_bp: 7500,
            correlation_id: Some("abcdef01:1700000000000000".into()),
        };
        let json = serde_json::to_string(&audit).unwrap();
        let a2: GateAuditInput = serde_json::from_str(&json).unwrap();
        assert_eq!(a2.correlation_id, Some("abcdef01:1700000000000000".into()));
    }

    #[test]
    fn gate_audit_input_serde_without_correlation_id() {
        // Backward compat: old audit inputs without correlation_id
        let json = r#"{"action_name":"test","zome_name":"z","eligible":true,"actual_tier":"Citizen","required_tier":"Participant","weight_bp":7500}"#;
        let audit: GateAuditInput = serde_json::from_str(json).unwrap();
        assert_eq!(audit.correlation_id, None);
    }

    // ════════════════════════════════════════════════════════════════════════
    // MINIMAL VIABLE BRIDGE: End-to-end tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn mvb_profile_from_unified_consciousness() {
        let profile = ConsciousnessProfile::from_unified_consciousness(
            0.65, // C_unified from Symthaea
            0.80, // identity (verified MFA)
            0.50, // reputation (moderate history)
            0.40, // community (some attestations)
        );
        assert_eq!(profile.engagement, 0.65);
        assert_eq!(profile.identity, 0.80);
        assert_eq!(profile.reputation, 0.50);
        assert_eq!(profile.community, 0.40);
        // combined = 0.80*0.25 + 0.50*0.25 + 0.40*0.30 + 0.65*0.20
        //         = 0.20 + 0.125 + 0.12 + 0.13 = 0.575
        let expected = 0.80 * 0.25 + 0.50 * 0.25 + 0.40 * 0.30 + 0.65 * 0.20;
        assert!((profile.combined_score() - expected).abs() < 1e-10);
    }

    #[test]
    fn mvb_profile_clamps_out_of_range() {
        let profile = ConsciousnessProfile::from_unified_consciousness(1.5, -0.2, 0.5, 0.5);
        assert_eq!(profile.engagement, 1.0);
        assert_eq!(profile.identity, 0.0);
    }

    #[test]
    fn mvb_credential_from_unified_consciousness() {
        let cred = ConsciousnessCredential::from_unified_consciousness(
            "did:mycelix:agent123".into(),
            0.65, // C_unified
            0.80,
            0.50,
            0.40,
            "did:mycelix:bridge".into(),
            NOW,
        );
        assert_eq!(cred.did, "did:mycelix:agent123");
        assert_eq!(cred.profile.engagement, 0.65);
        assert_eq!(cred.issued_at, NOW);
        assert_eq!(
            cred.expires_at,
            NOW + ConsciousnessCredential::DEFAULT_TTL_US
        );
        assert!(!cred.is_expired(NOW));
        // Tier should be Citizen (combined ≈ 0.575 → >= 0.4)
        assert!(cred.tier >= ConsciousnessTier::Citizen);
    }

    #[test]
    fn mvb_end_to_end_high_consciousness_proposal_eligible() {
        // Scenario: Agent with high C_unified submits a proposal
        // Expected: Eligible (Citizen tier, proposal requires Participant)

        // Step 1: Map C_unified → profile
        let cred = ConsciousnessCredential::from_unified_consciousness(
            "did:mycelix:agent_high".into(),
            0.70, // high consciousness
            0.80, // verified identity
            0.60, // good reputation
            0.50, // moderate community
            "did:mycelix:bridge".into(),
            NOW,
        );

        // Step 2: Evaluate against proposal requirement
        let result = evaluate_governance(&cred, &requirement_for_proposal(), NOW);

        // Step 3: Verify decision
        assert!(
            result.eligible,
            "High-consciousness agent should be proposal-eligible"
        );
        assert!(result.tier >= ConsciousnessTier::Participant);
        assert!(result.weight_bp > 0);

        // Step 4: Audit logging (verify audit input can be constructed)
        let should_log = should_audit(
            &requirement_for_proposal(),
            result.eligible,
            b"agent_high",
            "submit_proposal",
        );
        // 100% of basic approvals sampled at 10%, but we just verify the function works
        let _audit = GateAuditInput {
            action_name: "submit_proposal".into(),
            zome_name: "commons_bridge".into(),
            eligible: result.eligible,
            actual_tier: format!("{:?}", result.tier),
            required_tier: format!("{:?}", ConsciousnessTier::Participant),
            weight_bp: result.weight_bp,
            correlation_id: Some(format!("mvb:{}", NOW)),
        };
        // Verify it serializes (real bridge would store on source chain)
        let json = serde_json::to_string(&_audit).unwrap();
        assert!(json.contains("submit_proposal"));
        let _ = should_log; // used
    }

    #[test]
    fn mvb_end_to_end_low_consciousness_proposal_rejected() {
        // Scenario: Agent with low C_unified attempts a proposal
        // Expected: Rejected (Observer tier, proposal requires Participant)

        let cred = ConsciousnessCredential::from_unified_consciousness(
            "did:mycelix:agent_low".into(),
            0.10, // low consciousness
            0.20, // minimal identity
            0.10, // low reputation
            0.15, // minimal community
            "did:mycelix:bridge".into(),
            NOW,
        );

        let result = evaluate_governance(&cred, &requirement_for_proposal(), NOW);

        assert!(
            !result.eligible,
            "Low-consciousness agent should NOT be proposal-eligible"
        );
        assert_eq!(result.tier, ConsciousnessTier::Observer);
        assert_eq!(result.weight_bp, 0);

        // Rejections are always logged (100% audit rate)
        assert!(should_audit(
            &requirement_for_proposal(),
            false,
            b"agent_low",
            "submit_proposal"
        ));
    }

    #[test]
    fn mvb_end_to_end_read_always_allowed() {
        // Scenario: Any agent can read, regardless of consciousness level
        // The governance system gates blast radius, not voice.

        let cred = ConsciousnessCredential::from_unified_consciousness(
            "did:mycelix:agent_zero".into(),
            0.0, // zero consciousness
            0.0,
            0.0,
            0.0,
            "did:mycelix:bridge".into(),
            NOW,
        );

        // basic requirement is Participant (0.3), but read ops are UNGATED
        // In production, read ops don't call gate_consciousness at all.
        // This test verifies the profile correctly reflects zero state.
        assert_eq!(cred.tier, ConsciousnessTier::Observer);
        assert_eq!(cred.profile.combined_score(), 0.0);
    }

    // -- Bootstrap tests --

    #[test]
    fn bootstrap_eligible_small_community() {
        assert!(is_bootstrap_eligible(0, 0.25));
        assert!(is_bootstrap_eligible(4, 0.50));
    }

    #[test]
    fn bootstrap_ineligible_large_community() {
        assert!(!is_bootstrap_eligible(5, 0.50));
        assert!(!is_bootstrap_eligible(100, 1.0));
    }

    #[test]
    fn bootstrap_ineligible_low_identity() {
        assert!(!is_bootstrap_eligible(2, 0.24));
        assert!(!is_bootstrap_eligible(0, 0.0));
    }

    #[test]
    fn bootstrap_boundary() {
        assert!(!is_bootstrap_eligible(BOOTSTRAP_COMMUNITY_THRESHOLD, 0.5));
        assert!(is_bootstrap_eligible(
            BOOTSTRAP_COMMUNITY_THRESHOLD - 1,
            0.5
        ));
        assert!(is_bootstrap_eligible(2, BOOTSTRAP_MIN_IDENTITY));
    }

    #[test]
    fn bootstrap_credential_properties() {
        let cred = bootstrap_credential("did:mycelix:first".into(), 0.5, NOW);
        assert_eq!(cred.tier, ConsciousnessTier::Participant);
        assert_eq!(cred.profile.identity, 0.5);
        assert_eq!(cred.profile.reputation, 0.0);
        assert_eq!(cred.issuer, "did:mycelix:bootstrap");
        assert_eq!(cred.expires_at, NOW + BOOTSTRAP_TTL_US);
    }

    #[test]
    fn bootstrap_governance_basic_eligible() {
        let cred = bootstrap_credential("did:mycelix:first".into(), 0.5, NOW);
        let result = evaluate_bootstrap_governance(&cred, &requirement_for_basic(), NOW);
        assert!(result.eligible);
        assert_eq!(result.weight_bp, 5_000);
    }

    #[test]
    fn bootstrap_governance_voting_rejected() {
        let cred = bootstrap_credential("did:mycelix:first".into(), 0.5, NOW);
        let result = evaluate_bootstrap_governance(&cred, &requirement_for_voting(), NOW);
        assert!(!result.eligible);
        assert!(result.reasons[0].contains("capped at Participant"));
    }

    #[test]
    fn bootstrap_governance_expired_rejected() {
        let cred = bootstrap_credential("did:mycelix:first".into(), 0.5, NOW);
        let result = evaluate_bootstrap_governance(
            &cred,
            &requirement_for_basic(),
            NOW + BOOTSTRAP_TTL_US + 1,
        );
        assert!(!result.eligible);
    }

    #[test]
    fn bootstrap_serde_roundtrip() {
        let cred = bootstrap_credential("did:mycelix:first".into(), 0.5, NOW);
        let json = serde_json::to_string(&cred).unwrap();
        let cred2: ConsciousnessCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(cred.did, cred2.did);
        assert_eq!(cred.tier, cred2.tier);
    }

    // -- NaN/Infinity defense --

    #[test]
    fn clamped_sanitizes_nan_to_zero() {
        let p = ConsciousnessProfile {
            identity: f64::NAN,
            reputation: f64::NAN,
            community: f64::NAN,
            engagement: f64::NAN,
        };
        let c = p.clamped();
        assert_eq!(c.identity, 0.0);
        assert_eq!(c.reputation, 0.0);
        assert_eq!(c.community, 0.0);
        assert_eq!(c.engagement, 0.0);
        assert_eq!(c.tier(), ConsciousnessTier::Observer);
    }

    #[test]
    fn clamped_sanitizes_infinity_to_bounds() {
        let p = ConsciousnessProfile {
            identity: f64::INFINITY,
            reputation: f64::NEG_INFINITY,
            community: f64::INFINITY,
            engagement: f64::NEG_INFINITY,
        };
        let c = p.clamped();
        // Infinity → 0.0 (not 1.0), because non-finite values are untrusted
        assert_eq!(c.identity, 0.0);
        assert_eq!(c.reputation, 0.0);
        assert_eq!(c.community, 0.0);
        assert_eq!(c.engagement, 0.0);
    }

    #[test]
    fn clamped_mixed_nan_and_valid() {
        let p = ConsciousnessProfile {
            identity: 0.75,
            reputation: f64::NAN,
            community: 0.50,
            engagement: f64::INFINITY,
        };
        let c = p.clamped();
        assert_eq!(c.identity, 0.75);
        assert_eq!(c.reputation, 0.0);
        assert_eq!(c.community, 0.50);
        assert_eq!(c.engagement, 0.0);
    }

    #[test]
    fn is_valid_detects_nan() {
        let valid = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        assert!(valid.is_valid());

        let nan_id = ConsciousnessProfile {
            identity: f64::NAN,
            ..valid.clone()
        };
        assert!(!nan_id.is_valid());

        let inf_rep = ConsciousnessProfile {
            reputation: f64::INFINITY,
            ..valid.clone()
        };
        assert!(!inf_rep.is_valid());

        let neg_inf = ConsciousnessProfile {
            community: f64::NEG_INFINITY,
            ..valid
        };
        assert!(!neg_inf.is_valid());
    }

    #[test]
    fn nan_credential_evaluates_to_observer() {
        let cred = fresh_credential(ConsciousnessProfile {
            identity: f64::NAN,
            reputation: f64::NAN,
            community: f64::NAN,
            engagement: f64::NAN,
        });
        let result = evaluate_governance(&cred, &requirement_for_basic(), NOW);
        // NaN dimensions → clamped to 0.0 → Observer tier → rejected for Participant requirement
        assert!(!result.eligible);
        assert_eq!(result.tier, ConsciousnessTier::Observer);
    }

    #[test]
    fn nan_does_not_bypass_gate() {
        // Ensure NaN can never produce a passing gate check
        let cred = fresh_credential(ConsciousnessProfile {
            identity: f64::NAN,
            reputation: 0.9,
            community: 0.9,
            engagement: 0.9,
        });
        let result = evaluate_governance(&cred, &requirement_for_voting(), NOW);
        // Even with high rep/community/engagement, NaN identity → 0.0 identity
        // combined = 0.0*0.25 + 0.9*0.25 + 0.9*0.30 + 0.9*0.20 = 0.675 → Steward
        // BUT min_identity is Some(0.25), and clamped identity is 0.0 → rejected
        assert!(!result.eligible);
        assert!(result.reasons.iter().any(|r| r.contains("Identity")));
    }

    // ========================================================================
    // Property-based tests (proptest)
    // ========================================================================

    use proptest::prelude::*;

    /// Strategy for generating valid profile dimension values in [0.0, 1.0].
    fn dimension() -> impl Strategy<Value = f64> {
        0.0..=1.0_f64
    }

    /// Strategy for generating a valid ConsciousnessProfile.
    fn profile_strategy() -> impl Strategy<Value = ConsciousnessProfile> {
        (dimension(), dimension(), dimension(), dimension()).prop_map(
            |(identity, reputation, community, engagement)| ConsciousnessProfile {
                identity,
                reputation,
                community,
                engagement,
            },
        )
    }

    proptest! {
        /// Higher profile scores always produce equal or higher tiers.
        ///
        /// If every dimension of profile B is >= the corresponding dimension
        /// of profile A, then B's tier must be >= A's tier.
        #[test]
        fn test_tier_progression_monotonic(
            id_lo in dimension(),
            rep_lo in dimension(),
            com_lo in dimension(),
            eng_lo in dimension(),
            id_delta in 0.0..=1.0_f64,
            rep_delta in 0.0..=1.0_f64,
            com_delta in 0.0..=1.0_f64,
            eng_delta in 0.0..=1.0_f64,
        ) {
            let lo = ConsciousnessProfile {
                identity: id_lo,
                reputation: rep_lo,
                community: com_lo,
                engagement: eng_lo,
            };
            let hi = ConsciousnessProfile {
                identity: (id_lo + id_delta).min(1.0),
                reputation: (rep_lo + rep_delta).min(1.0),
                community: (com_lo + com_delta).min(1.0),
                engagement: (eng_lo + eng_delta).min(1.0),
            };
            prop_assert!(hi.tier() >= lo.tier(),
                "Higher profile {:?} (tier {:?}) should be >= lower profile {:?} (tier {:?})",
                hi, hi.tier(), lo, lo.tier());
        }

        /// Higher tiers always have equal or higher vote weights.
        #[test]
        fn test_vote_weight_monotonic_with_tier(
            a in profile_strategy(),
            b in profile_strategy(),
        ) {
            let tier_a = a.tier();
            let tier_b = b.tier();
            if tier_a <= tier_b {
                prop_assert!(tier_a.vote_weight_bp() <= tier_b.vote_weight_bp(),
                    "Tier {:?} (weight {}) should have <= weight than {:?} (weight {})",
                    tier_a, tier_a.vote_weight_bp(), tier_b, tier_b.vote_weight_bp());
            }
        }

        /// A profile with all zeros is always Observer tier.
        #[test]
        fn test_all_zero_profile_is_observer(_seed in 0u32..100) {
            let p = ConsciousnessProfile::zero();
            prop_assert_eq!(p.tier(), ConsciousnessTier::Observer);
            prop_assert_eq!(p.combined_score(), 0.0);
            prop_assert_eq!(p.tier().vote_weight_bp(), 0);
        }

        /// Max values (all 1.0) never panic and produce Guardian tier.
        #[test]
        fn test_all_max_profile_no_panic(_seed in 0u32..100) {
            let p = ConsciousnessProfile {
                identity: 1.0,
                reputation: 1.0,
                community: 1.0,
                engagement: 1.0,
            };
            let tier = p.tier();
            let score = p.combined_score();
            let weight = tier.vote_weight_bp();
            prop_assert_eq!(tier, ConsciousnessTier::Guardian);
            prop_assert!((score - 1.0).abs() < 1e-10);
            prop_assert_eq!(weight, 10000);
        }

        /// Any credential created via from_unified_consciousness has
        /// expires_at > issued_at (expiry is always in the future relative
        /// to issuance).
        #[test]
        fn test_credential_expiry_in_future(
            c_unified in dimension(),
            identity in dimension(),
            reputation in dimension(),
            community in dimension(),
            now_us in 0u64..=u64::MAX / 2,
        ) {
            let cred = ConsciousnessCredential::from_unified_consciousness(
                "did:test:prop".into(),
                c_unified,
                identity,
                reputation,
                community,
                "did:test:issuer".into(),
                now_us,
            );
            prop_assert!(cred.expires_at > cred.issued_at,
                "expires_at ({}) must be > issued_at ({})",
                cred.expires_at, cred.issued_at);
            prop_assert_eq!(cred.expires_at, now_us + ConsciousnessCredential::DEFAULT_TTL_US);
        }

        /// Computed combined_score is always in [0.0, 1.0] for any
        /// valid (clamped) profile.
        #[test]
        fn test_profile_scores_bounded(
            identity in -10.0..=10.0_f64,
            reputation in -10.0..=10.0_f64,
            community in -10.0..=10.0_f64,
            engagement in -10.0..=10.0_f64,
        ) {
            let raw = ConsciousnessProfile { identity, reputation, community, engagement };
            let clamped = raw.clamped();

            // All clamped dimensions in [0, 1]
            prop_assert!(clamped.identity >= 0.0 && clamped.identity <= 1.0);
            prop_assert!(clamped.reputation >= 0.0 && clamped.reputation <= 1.0);
            prop_assert!(clamped.community >= 0.0 && clamped.community <= 1.0);
            prop_assert!(clamped.engagement >= 0.0 && clamped.engagement <= 1.0);

            // Combined score of clamped profile in [0, 1]
            let score = clamped.combined_score();
            prop_assert!(score >= 0.0 && score <= 1.0,
                "Combined score {} out of bounds for clamped profile {:?}",
                score, clamped);

            // Tier is always valid (no panic)
            let _tier = clamped.tier();
        }
    }

    // ========================================================================
    // Reputation decay, slashing, and restoration tests
    // ========================================================================

    #[test]
    fn test_reputation_decay_one_day() {
        let mut state = ReputationState::new(1.0, 0);
        let one_day_us = 86_400_000_000u64;
        state.apply_decay(one_day_us);
        assert!(
            (state.score - REPUTATION_DECAY_PER_DAY).abs() < 1e-6,
            "After 1 day, score should be {}, got {}",
            REPUTATION_DECAY_PER_DAY,
            state.score
        );
    }

    #[test]
    fn test_reputation_decay_half_life() {
        let mut state = ReputationState::new(1.0, 0);
        let days_347_us = 347 * 86_400_000_000u64;
        state.apply_decay(days_347_us);
        assert!(
            state.score > 0.49 && state.score < 0.51,
            "After ~347 days, score should be ~0.5, got {}",
            state.score
        );
    }

    #[test]
    fn test_reputation_no_decay_on_zero_elapsed() {
        let mut state = ReputationState::new(0.8, 1000);
        state.apply_decay(1000); // same timestamp
        assert!((state.score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_reputation_slash() {
        let mut state = ReputationState::new(1.0, 0);
        let new_score = state.slash(1000);
        assert!(
            (new_score - 0.5).abs() < 1e-6,
            "After slash, score should be 0.5, got {}",
            new_score
        );
        assert_eq!(state.total_slashes, 1);
        assert_eq!(state.consecutive_good, 0);
    }

    #[test]
    fn test_reputation_slash_below_blacklist() {
        let mut state = ReputationState::new(0.08, 0);
        state.slash(1000);
        // 0.08 * 0.5 = 0.04 < 0.05 threshold
        assert!(state.blacklisted);
        assert!(!state.can_participate());
    }

    #[test]
    fn test_reputation_restoration_path() {
        let mut state = ReputationState::new(0.03, 0);
        state.blacklisted = true;
        state.blacklisted_since_us = Some(0);

        // Progress through restoration
        for i in 0..REPUTATION_RESTORATION_INTERACTIONS {
            assert!(
                state.blacklisted,
                "Should still be blacklisted at interaction {}",
                i
            );
            state.record_good_interaction(0.001, 1000 + i as u64 * 1000);
        }

        assert!(
            !state.blacklisted,
            "Should be restored after {} good interactions",
            REPUTATION_RESTORATION_INTERACTIONS
        );
        assert!(
            state.score >= 0.1,
            "Score should be at least 0.1 after restoration"
        );
    }

    #[test]
    fn test_reputation_max_slashes_cap() {
        let mut state = ReputationState::new(1.0, 0);

        // Slash more than MAX_SLASHES times
        for i in 0..(REPUTATION_MAX_SLASHES + 2) {
            state.score = 0.8; // Reset score to test cap
            state.slash(i as u64 * 1000);
        }

        // Now good interactions should be capped at 0.5
        state.score = 0.3;
        state.blacklisted = false;
        for i in 0..200 {
            state.record_good_interaction(0.01, 1_000_000 + i as u64 * 1000);
        }
        assert!(
            state.score <= 0.5 + 1e-6,
            "Score should be capped at 0.5 after {} slashes, got {}",
            REPUTATION_MAX_SLASHES,
            state.score
        );
    }

    #[test]
    fn test_reputation_proportional_slash() {
        let mut state = ReputationState::new(1.0, 0);
        state.slash_proportional(0.3, 1000);
        assert!((state.score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_reputation_restoration_progress() {
        let mut state = ReputationState::new(0.01, 0);
        state.blacklisted = true;
        state.blacklisted_since_us = Some(0);
        state.consecutive_good = 50;

        let progress = state.restoration_progress();
        assert!(
            (progress - 0.5).abs() < 1e-6,
            "50/{} should be 50% progress, got {}",
            REPUTATION_RESTORATION_INTERACTIONS,
            progress
        );
    }

    #[test]
    fn test_reputation_non_blacklisted_full_progress() {
        let state = ReputationState::new(0.8, 0);
        assert!((state.restoration_progress() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_reputation_profile() {
        let profile = ConsciousnessProfile {
            identity: 0.9,
            reputation: 1.0,
            community: 0.7,
            engagement: 0.5,
        };
        let decayed = decay_reputation(&profile, 30.0);
        assert!(
            (decayed.identity - 0.9).abs() < 1e-10,
            "Identity should not decay"
        );
        assert!(decayed.reputation < 1.0, "Reputation should decay");
        assert!(
            (decayed.community - 0.7).abs() < 1e-10,
            "Community should not decay"
        );
    }

    #[test]
    fn test_evaluate_governance_blacklisted() {
        let credential = ConsciousnessCredential {
            did: "did:mycelix:test".to_string(),
            profile: ConsciousnessProfile {
                identity: 0.9,
                reputation: 0.01,
                community: 0.8,
                engagement: 0.7,
            },
            tier: ConsciousnessTier::Citizen,
            issued_at: 0,
            expires_at: 100_000_000_000,
            issuer: "did:mycelix:bridge".to_string(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        let rep_state = ReputationState {
            score: 0.01,
            blacklisted: true,
            blacklisted_since_us: Some(0),
            consecutive_good: 20,
            total_slashes: 2,
            last_updated_us: 0,
        };
        let requirement = GovernanceRequirement {
            min_tier: ConsciousnessTier::Participant,
            min_identity: None,
            min_community: None,
        };

        let result = evaluate_governance_with_reputation(
            &credential,
            &requirement,
            &rep_state,
            50_000_000_000,
        );
        assert!(!result.eligible, "Blacklisted agent should not be eligible");
        assert_eq!(result.weight_bp, 0);
    }

    #[test]
    fn test_evaluate_governance_slash_penalty() {
        let credential = ConsciousnessCredential {
            did: "did:mycelix:test".to_string(),
            profile: ConsciousnessProfile {
                identity: 0.9,
                reputation: 0.8,
                community: 0.8,
                engagement: 0.7,
            },
            tier: ConsciousnessTier::Steward,
            issued_at: 0,
            expires_at: 100_000_000_000,
            issuer: "did:mycelix:bridge".to_string(),
            trajectory_commitment: None,
            extensions: std::collections::HashMap::new(),
        };
        let rep_state = ReputationState {
            score: 0.8,
            blacklisted: false,
            blacklisted_since_us: None,
            consecutive_good: 50,
            total_slashes: 2, // 2 slashes -> 10% weight reduction
            last_updated_us: 0,
        };
        let requirement = GovernanceRequirement {
            min_tier: ConsciousnessTier::Participant,
            min_identity: None,
            min_community: None,
        };

        let result = evaluate_governance_with_reputation(
            &credential,
            &requirement,
            &rep_state,
            50_000_000_000,
        );
        assert!(result.eligible);
        // Weight should be reduced by slash penalty
        let base_weight = credential.profile.clamped().tier().vote_weight_bp();
        assert!(
            result.weight_bp < base_weight,
            "Weight {} should be less than base {} due to slash penalty",
            result.weight_bp,
            base_weight
        );
    }

    #[test]
    fn test_reputation_nan_safety() {
        let state = ReputationState::new(f64::NAN, 0);
        assert!(
            (state.score - 0.0).abs() < 1e-10,
            "NaN should clamp to 0"
        );

        let mut state2 = ReputationState::new(0.5, 0);
        state2.apply_decay(u64::MAX); // extreme elapsed time
        assert!(state2.score.is_finite());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Continuous sigmoid authorization tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sigmoid_at_threshold_returns_half_max() {
        let w = continuous_vote_weight(0.4, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
        assert!(
            (w - VOTE_WEIGHT_MAX_BP / 2.0).abs() < 1e-6,
            "At threshold, weight should be max/2, got {w}"
        );
    }

    #[test]
    fn test_sigmoid_below_threshold_near_zero() {
        let w = continuous_vote_weight(0.1, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
        assert!(
            w < 100.0,
            "Well below threshold, weight should be near zero, got {w}"
        );
    }

    #[test]
    fn test_sigmoid_above_threshold_near_max() {
        let w = continuous_vote_weight(0.7, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
        assert!(
            w > 9900.0,
            "Well above threshold, weight should be near max, got {w}"
        );
    }

    #[test]
    fn test_sigmoid_zero_temperature_returns_zero() {
        let w = continuous_vote_weight(0.5, 0.4, 0.0, VOTE_WEIGHT_MAX_BP);
        assert!(
            (w - 0.0).abs() < 1e-10,
            "Zero temperature should return 0.0, got {w}"
        );
    }

    #[test]
    fn test_sigmoid_nan_score_returns_zero() {
        let w = continuous_vote_weight(f64::NAN, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
        assert!(
            (w - 0.0).abs() < 1e-10,
            "NaN score should return 0.0, got {w}"
        );
    }

    #[test]
    fn test_sigmoid_infinity_score_returns_zero() {
        let w = continuous_vote_weight(f64::INFINITY, 0.4, VOTE_WEIGHT_TEMPERATURE, VOTE_WEIGHT_MAX_BP);
        assert!(
            (w - 0.0).abs() < 1e-10,
            "Infinity score should return 0.0, got {w}"
        );
    }

    #[test]
    fn test_vote_weight_continuous_on_profile() {
        let profile = ConsciousnessProfile {
            identity: 0.5,
            reputation: 0.5,
            community: 0.5,
            engagement: 0.5,
        };
        // combined_score = 0.5, which is above 0.4 threshold
        let w = profile.vote_weight_continuous();
        assert!(w > VOTE_WEIGHT_MAX_BP / 2.0, "Score 0.5 > threshold 0.4 should give > half max");
        assert!(w < VOTE_WEIGHT_MAX_BP, "Should not exceed max");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Hysteresis tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_hysteresis_no_promote_in_deadband() {
        // Score 0.42 — above Citizen threshold (0.4) but below promote threshold (0.45)
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.42, ConsciousnessTier::Participant);
        assert_eq!(
            tier,
            ConsciousnessTier::Participant,
            "0.42 should NOT promote from Participant (need 0.45)"
        );
    }

    #[test]
    fn test_hysteresis_no_demote_in_deadband() {
        // Score 0.38 — below Citizen threshold (0.4) but above demote threshold (0.35)
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.38, ConsciousnessTier::Citizen);
        assert_eq!(
            tier,
            ConsciousnessTier::Citizen,
            "0.38 should NOT demote from Citizen (need < 0.35)"
        );
    }

    #[test]
    fn test_hysteresis_demotes_below_lower_threshold() {
        // Score 0.34 — below demote threshold (0.35)
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.34, ConsciousnessTier::Citizen);
        assert_eq!(
            tier,
            ConsciousnessTier::Participant,
            "0.34 should demote from Citizen (below 0.35)"
        );
    }

    #[test]
    fn test_hysteresis_promotes_above_upper_threshold() {
        // Score 0.46 — above promote threshold (0.45)
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.46, ConsciousnessTier::Participant);
        assert_eq!(
            tier,
            ConsciousnessTier::Citizen,
            "0.46 should promote from Participant to Citizen (above 0.45)"
        );
    }

    #[test]
    fn test_hysteresis_guardian_boundary() {
        // 0.83 — above 0.8 but below promote threshold 0.85
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.83, ConsciousnessTier::Steward);
        assert_eq!(tier, ConsciousnessTier::Steward, "0.83 should NOT promote to Guardian (need 0.85)");

        // 0.86 — above promote threshold 0.85
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.86, ConsciousnessTier::Steward);
        assert_eq!(tier, ConsciousnessTier::Guardian, "0.86 should promote to Guardian");

        // 0.76 — below demote threshold 0.75
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.74, ConsciousnessTier::Guardian);
        assert_eq!(tier, ConsciousnessTier::Steward, "0.74 should demote from Guardian");
    }

    #[test]
    fn test_hysteresis_observer_boundary() {
        // 0.33 — above 0.3 but below promote threshold 0.35
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.33, ConsciousnessTier::Observer);
        assert_eq!(tier, ConsciousnessTier::Observer, "0.33 should NOT promote from Observer (need 0.35)");

        // 0.36 — above promote threshold 0.35
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.36, ConsciousnessTier::Observer);
        assert_eq!(tier, ConsciousnessTier::Participant, "0.36 should promote to Participant");

        // 0.24 — below demote threshold 0.25
        let tier = ConsciousnessTier::from_score_with_hysteresis(0.24, ConsciousnessTier::Participant);
        assert_eq!(tier, ConsciousnessTier::Observer, "0.24 should demote to Observer");
    }

    #[test]
    fn extensions_default_empty() {
        let cred = ConsciousnessCredential::from_unified_consciousness(
            "did:test".into(), 0.5, 0.5, 0.5, 0.5, "issuer".into(), 1000,
        );
        assert!(cred.extensions.is_empty());
        assert!(cred.trajectory_commitment.is_none());
    }

    #[test]
    fn extensions_set_get_roundtrip() {
        let mut cred = ConsciousnessCredential::from_unified_consciousness(
            "did:test".into(), 0.5, 0.5, 0.5, 0.5, "issuer".into(), 1000,
        );
        cred.set_extension("foo", vec![1, 2, 3]);
        assert_eq!(cred.get_extension("foo"), Some(&vec![1, 2, 3]));
        assert_eq!(cred.get_extension("bar"), None);
    }

    #[test]
    fn extensions_remove() {
        let mut cred = ConsciousnessCredential::from_unified_consciousness(
            "did:test".into(), 0.5, 0.5, 0.5, 0.5, "issuer".into(), 1000,
        );
        cred.set_extension("key", vec![42]);
        let removed = cred.remove_extension("key");
        assert_eq!(removed, Some(vec![42]));
        assert!(cred.get_extension("key").is_none());
    }
}
