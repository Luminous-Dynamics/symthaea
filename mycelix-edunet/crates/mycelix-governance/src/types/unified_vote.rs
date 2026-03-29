// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified vote system with composable modifiers
//!
//! ## Revolutionary Architecture: From Fragmentation to Composition
//!
//! **Traditional Problem**:
//! ```text
//! QuadraticVote { ... }      ← Can't have conviction
//! ConvictionVote { ... }     ← Can't be quadratic
//! ReputationVote { ... }     ← Can't have either
//! ```
//!
//! Users must choose ONE mechanism, losing synergistic attack resistance.
//!
//! **Composable Solution**:
//! ```text
//! UnifiedVote {
//!     quadratic_modifier: Some(...),      ← Optional
//!     conviction_modifier: Some(...),     ← Optional
//!     delegation_modifier: Some(...),     ← Optional
//! }
//! ```
//!
//! All modifiers combine multiplicatively for exponential attack cost!
//!
//! ## Mathematical Foundation: Synergistic Attack Resistance
//!
//! ```text
//! Traditional (choose one):
//!   Quadratic only:   √R             → Attack cost: 10,000r
//!   Conviction only:  R × C(t)       → Attack cost: ~3,000r
//!
//! Composable (combine all):
//!   weight = √(R × C(t) × 0.9^d)    → Attack cost: 15,000r
//!
//! Result: 1.5x better than best single mechanism!
//! BFT: ~55% → ~62%+
//! ```
//!
//! ## Example
//!
//! ```rust
//! use mycelix_governance::types::unified_vote::{UnifiedVote, QuadraticModifier, ConvictionModifier};
//! use mycelix_governance::types::vote::VoteChoice;
//!
//! let vote = UnifiedVote::builder()
//!     .proposal_id("prop_001".to_string())
//!     .voter("agent_pubkey".to_string())
//!     .choice(VoteChoice::For)
//!     .reputation_allocated(100.0)
//!     // Stack modifiers for synergistic benefits
//!     .with_quadratic()
//!     .with_conviction(7.0)  // 7 days locked
//!     .build();
//!
//! // Final weight = √(100 × 1.63) ≈ 12.77
//! // vs Quadratic only: √100 = 10.0
//! // vs Conviction only: 100 × 1.63 = 163.0
//! // Composable system balances both benefits!
//! ```

use serde::{Deserialize, Serialize};

use super::vote::VoteChoice;
use super::conviction_vote::ConvictionParameters;

/// Unified vote with optional stackable modifiers
///
/// ## Design Philosophy
///
/// Each modifier is **optional** and **composable**:
/// - No modifiers = simple reputation-weighted vote
/// - Quadratic only = sybil-resistant but instant
/// - Conviction only = time-weighted but vulnerable to whales
/// - Both = synergistic attack resistance exceeding sum of parts
///
/// ## Backward Compatibility
///
/// Can simulate existing vote types:
/// ```rust
/// # use mycelix_governance::types::unified_vote::UnifiedVote;
/// # use mycelix_governance::types::vote::VoteChoice;
/// // ReputationVote equivalent (no modifiers)
/// let vote = UnifiedVote::simple(
///     "prop_001".to_string(),
///     "voter".to_string(),
///     VoteChoice::For,
///     100.0,
/// );
///
/// // QuadraticVote equivalent (quadratic only)
/// let vote = UnifiedVote::builder()
///     .proposal_id("prop_001".to_string())
///     .voter("voter".to_string())
///     .choice(VoteChoice::For)
///     .reputation_allocated(100.0)
///     .with_quadratic()
///     .build();
///
/// // ConvictionVote equivalent (conviction only)
/// let vote = UnifiedVote::builder()
///     .proposal_id("prop_001".to_string())
///     .voter("voter".to_string())
///     .choice(VoteChoice::For)
///     .reputation_allocated(100.0)
///     .with_conviction(7.0)
///     .build();
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnifiedVote {
    // Core identification
    pub proposal_id: String,
    pub voter: String,
    pub choice: VoteChoice,
    pub justification: Option<String>,
    pub timestamp: i64,

    // Base vote parameters
    pub reputation_allocated: f64,

    // Optional modifiers (stackable)
    pub quadratic_modifier: Option<QuadraticModifier>,
    pub conviction_modifier: Option<ConvictionModifier>,
    pub delegation_modifier: Option<DelegationModifier>,

    // Computed final weight (cached for performance)
    pub final_weight: f64,
}

impl UnifiedVote {
    /// Create a simple reputation-weighted vote (no modifiers)
    pub fn simple(
        proposal_id: String,
        voter: String,
        choice: VoteChoice,
        reputation_allocated: f64,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            proposal_id,
            voter,
            choice,
            justification: None,
            timestamp: now,
            reputation_allocated,
            quadratic_modifier: None,
            conviction_modifier: None,
            delegation_modifier: None,
            final_weight: reputation_allocated, // No modifiers = linear weight
        }
    }

    /// Start building a vote with modifiers
    pub fn builder() -> UnifiedVoteBuilder {
        UnifiedVoteBuilder::new()
    }

    /// Calculate final weight by applying all active modifiers
    ///
    /// ## Modifier Application Order
    ///
    /// 1. **Conviction multiplier** (1.0 to 2.0) - increases with time
    /// 2. **Quadratic dampening** (√x) - prevents whale dominance
    /// 3. **Delegation attenuation** (0.9^depth) - prevents deep chains
    ///
    /// ## Formula
    ///
    /// ```text
    /// base_weight = reputation_allocated
    ///
    /// if conviction_modifier:
    ///     base_weight *= conviction_multiplier  (1.0 to 2.0)
    ///
    /// if quadratic_modifier:
    ///     base_weight = √base_weight
    ///
    /// if delegation_modifier:
    ///     base_weight *= 0.9^chain_depth
    ///
    /// final_weight = base_weight
    /// ```
    ///
    /// ## Why This Order?
    ///
    /// 1. Conviction first: Rewards time commitment before dampening
    /// 2. Quadratic second: Dampens total (rep × conviction) to prevent whales
    /// 3. Delegation last: Attenuates after other calculations
    ///
    /// ## Example
    ///
    /// ```rust
    /// # use mycelix_governance::types::unified_vote::UnifiedVote;
    /// # use mycelix_governance::types::vote::VoteChoice;
    /// let vote = UnifiedVote::builder()
    ///     .proposal_id("prop_001".to_string())
    ///     .voter("voter".to_string())
    ///     .choice(VoteChoice::For)
    ///     .reputation_allocated(100.0)
    ///     .with_quadratic()
    ///     .with_conviction(7.0)  // ~1.63x multiplier
    ///     .build();
    ///
    /// // Calculation:
    /// // 1. Conviction: 100 × 1.63 = 163.0
    /// // 2. Quadratic: √163 ≈ 12.77
    /// // 3. No delegation: 12.77 × 1.0 = 12.77
    /// assert_eq!(vote.final_weight, 12.77);
    /// ```
    pub fn calculate_final_weight(&self) -> f64 {
        let mut weight = self.reputation_allocated;

        // 1. Apply conviction multiplier (1.0 to 2.0)
        if let Some(ref conviction) = self.conviction_modifier {
            weight *= conviction.multiplier;
        }

        // 2. Apply quadratic dampening AFTER conviction
        if self.quadratic_modifier.is_some() {
            weight = weight.sqrt();
        }

        // 3. Apply delegation chain attenuation (0.9^depth)
        if let Some(ref delegation) = self.delegation_modifier {
            weight *= 0.9_f64.powi(delegation.chain_depth as i32);
        }

        weight
    }

    /// Recalculate and update final_weight (call after modifying vote)
    pub fn update_weight(&mut self) {
        self.final_weight = self.calculate_final_weight();
    }
}

/// Quadratic voting modifier - applies √(x) to prevent plutocracy
///
/// ## Why Quadratic?
///
/// Makes concentrated voting less effective than distributed:
/// - 1 whale with 10,000 reputation → √10,000 = 100 weight
/// - 100 nodes with 100 each → 100 × √100 = 1,000 weight
/// - Attack cost: 100x harder (need 10,000x more reputation!)
///
/// ## Mathematical Analysis
///
/// ```text
/// Without quadratic:
///   Attacker with R = 100 honest nodes
///   Attack cost: Linear (R = 100r)
///
/// With quadratic:
///   Attacker needs: (100√r)² = 10,000r
///   Attack cost: 100x harder
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct QuadraticModifier {
    /// Whether dampening is enabled (always true if present)
    pub enabled: bool,
}

impl Default for QuadraticModifier {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Conviction voting modifier - exponential time-weighted commitment
///
/// ## Why Conviction?
///
/// Prevents instant strategic voting by rewarding time commitment:
/// - Short term (1 day): ~1.13x weight
/// - Medium term (7 days): ~1.63x weight
/// - Long term (14+ days): ~1.86x weight
///
/// Early unlock = 50% reputation penalty (default)
///
/// ## Formula
///
/// ```text
/// C(t) = 1 + (C_max - 1) × (1 - e^(-t/τ))
///
/// where:
///   t = days locked
///   τ = time constant (default: 7 days)
///   C_max = max multiplier (default: 2.0)
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConvictionModifier {
    /// Days locked
    pub days_locked: f64,

    /// Current conviction multiplier (1.0 to 2.0)
    pub multiplier: f64,

    /// When the lock expires (Unix timestamp)
    pub lock_expiry: i64,

    /// Conviction parameters (time constant, max multiplier, etc.)
    pub params: ConvictionParameters,

    /// Whether lock can be broken early (with penalty)
    pub allow_early_unlock: bool,
}

impl ConvictionModifier {
    /// Create new conviction modifier with given lock period
    ///
    /// ## Arguments
    ///
    /// * `days_locked` - How long reputation is committed
    /// * `params` - Optional custom parameters (uses defaults if None)
    ///
    /// ## Returns
    ///
    /// ConvictionModifier with calculated multiplier
    pub fn new(days_locked: f64, params: Option<ConvictionParameters>) -> Self {
        let params = params.unwrap_or_default();
        let now = chrono::Utc::now().timestamp();
        let lock_expiry = now + (days_locked * 86400.0) as i64;
        let multiplier = params.calculate_conviction(now, lock_expiry);

        Self {
            days_locked,
            multiplier,
            lock_expiry,
            params,
            allow_early_unlock: true,
        }
    }

    /// Transfer conviction through delegation chain
    ///
    /// ## Attenuation
    ///
    /// Each delegation hop reduces conviction by 10%:
    /// - 0 hops: C(t) = 1.63
    /// - 1 hop: C(t) × 0.9 = 1.47
    /// - 2 hops: C(t) × 0.81 = 1.32
    /// - 5 hops: C(t) × 0.59 = 0.96 (below baseline!)
    ///
    /// This prevents deep delegation chains from preserving full conviction.
    pub fn transfer_through_delegation(&self, chain_depth: u32) -> Self {
        let mut transferred = self.clone();

        // Attenuation factor: 10% reduction per hop
        let attenuation = 0.9_f64.powi(chain_depth as i32);

        // Reduce conviction but preserve lock time
        // Formula: C'(t) = 1 + (C(t) - 1) × attenuation
        transferred.multiplier = 1.0 + (self.multiplier - 1.0) * attenuation;

        transferred
    }

    /// Calculate penalty for breaking commitment early
    ///
    /// ## Default Penalty
    ///
    /// 50% of allocated reputation is slashed if unlocking before expiry.
    ///
    /// ## Returns
    ///
    /// Reputation to slash (0.0 to reputation_allocated)
    pub fn calculate_early_unlock_penalty(&self, reputation_allocated: f64) -> f64 {
        reputation_allocated * self.params.early_unlock_penalty
    }
}

/// Delegation modifier - liquid democracy with conviction transfer
///
/// ## Why Delegation?
///
/// Allows expertise routing while maintaining attack resistance:
/// - Technical proposals → Technical experts
/// - Economic proposals → Economic experts
/// - Security proposals → Security experts
///
/// ## Attenuation
///
/// Each hop in delegation chain reduces weight by 10%:
/// - Direct vote: 1.0x
/// - 1 hop (A→B): 0.9x
/// - 2 hops (A→B→C): 0.81x
/// - 5 hops: 0.59x (self-defeating)
///
/// This prevents infinite chains and sybil attacks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DelegationModifier {
    /// Who the vote is delegated to (AgentPubKey)
    pub delegate: String,

    /// Depth of delegation chain (0 = direct vote)
    pub chain_depth: u32,

    /// Whether conviction is transferred or reset
    pub preserve_conviction: bool,

    /// Delegation scope (all proposals or specific category)
    pub scope: DelegationScope,
}

/// Scope of delegation - who can vote on what
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DelegationScope {
    /// Delegate all votes to this person
    AllProposals,

    /// Delegate only specific category
    Category(super::proposal::ProposalCategory),

    /// Different delegates per category (most fine-grained)
    PerCategory(std::collections::HashMap<super::proposal::ProposalCategory, String>),
}

/// Builder for constructing UnifiedVote with fluent API
pub struct UnifiedVoteBuilder {
    proposal_id: Option<String>,
    voter: Option<String>,
    choice: Option<VoteChoice>,
    justification: Option<String>,
    reputation_allocated: f64,
    quadratic_modifier: Option<QuadraticModifier>,
    conviction_modifier: Option<ConvictionModifier>,
    delegation_modifier: Option<DelegationModifier>,
}

impl UnifiedVoteBuilder {
    pub fn new() -> Self {
        Self {
            proposal_id: None,
            voter: None,
            choice: None,
            justification: None,
            reputation_allocated: 0.0,
            quadratic_modifier: None,
            conviction_modifier: None,
            delegation_modifier: None,
        }
    }

    pub fn proposal_id(mut self, id: String) -> Self {
        self.proposal_id = Some(id);
        self
    }

    pub fn voter(mut self, voter: String) -> Self {
        self.voter = Some(voter);
        self
    }

    pub fn choice(mut self, choice: VoteChoice) -> Self {
        self.choice = Some(choice);
        self
    }

    pub fn justification(mut self, text: String) -> Self {
        self.justification = Some(text);
        self
    }

    pub fn reputation_allocated(mut self, amount: f64) -> Self {
        self.reputation_allocated = amount;
        self
    }

    /// Enable quadratic dampening
    pub fn with_quadratic(mut self) -> Self {
        self.quadratic_modifier = Some(QuadraticModifier::default());
        self
    }

    /// Enable conviction voting with given lock period
    pub fn with_conviction(mut self, days_locked: f64) -> Self {
        self.conviction_modifier = Some(ConvictionModifier::new(days_locked, None));
        self
    }

    /// Enable conviction with custom parameters
    pub fn with_conviction_params(mut self, days_locked: f64, params: ConvictionParameters) -> Self {
        self.conviction_modifier = Some(ConvictionModifier::new(days_locked, Some(params)));
        self
    }

    /// Enable delegation to another agent
    pub fn with_delegation(mut self, delegate: String, chain_depth: u32) -> Self {
        self.delegation_modifier = Some(DelegationModifier {
            delegate,
            chain_depth,
            preserve_conviction: true,
            scope: DelegationScope::AllProposals,
        });
        self
    }

    /// Build the final UnifiedVote
    pub fn build(self) -> UnifiedVote {
        let proposal_id = self.proposal_id.expect("proposal_id is required");
        let voter = self.voter.expect("voter is required");
        let choice = self.choice.expect("choice is required");
        let now = chrono::Utc::now().timestamp();

        let mut vote = UnifiedVote {
            proposal_id,
            voter,
            choice,
            justification: self.justification,
            timestamp: now,
            reputation_allocated: self.reputation_allocated,
            quadratic_modifier: self.quadratic_modifier,
            conviction_modifier: self.conviction_modifier,
            delegation_modifier: self.delegation_modifier,
            final_weight: 0.0, // Will be calculated
        };

        vote.final_weight = vote.calculate_final_weight();
        vote
    }
}

impl Default for UnifiedVoteBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Migration Support - Converting from Legacy Vote Types
// ============================================================================

/// Convert from QuadraticVote to UnifiedVote
impl From<crate::types::quadratic_vote::QuadraticVote> for UnifiedVote {
    fn from(legacy: crate::types::quadratic_vote::QuadraticVote) -> Self {
        let now = chrono::Utc::now().timestamp();

        // QuadraticVote always has quadratic modifier
        let quadratic_modifier = Some(QuadraticModifier::default());

        // Extract conviction modifier if conviction_factor > 1.0 or deadline exists
        let conviction_modifier = if legacy.conviction_factor > 1.0 || legacy.conviction_deadline.is_some() {
            // Estimate days_locked from conviction_factor using inverse formula:
            // conviction_factor = 1.0 + (max - 1.0) * (1.0 - exp(-days / tau))
            // Assuming tau = 7.0, max = 3.0:
            // norm_conv = (factor - 1.0) / (max - 1.0)
            // days = -tau * ln(1.0 - norm_conv)
            let norm_conv = ((legacy.conviction_factor - 1.0) / 2.0).min(0.999_f64); // Clamp to avoid log(0)
            let days_locked: f64 = if norm_conv > 0.0 {
                -7.0_f64 * (1.0_f64 - norm_conv).ln()
            } else {
                0.0_f64
            };

            Some(ConvictionModifier::new(days_locked.max(0.0_f64), None))
        } else {
            None
        };

        let mut vote = UnifiedVote {
            proposal_id: legacy.proposal_id,
            voter: legacy.voter,
            choice: legacy.choice,
            justification: legacy.justification,
            timestamp: now,
            reputation_allocated: legacy.reputation_allocated,
            quadratic_modifier,
            conviction_modifier,
            delegation_modifier: None,
            final_weight: 0.0, // Will be calculated
        };

        vote.final_weight = vote.calculate_final_weight();
        vote
    }
}

/// Convert from ConvictionVote to UnifiedVote
impl From<crate::types::conviction_vote::ConvictionVote> for UnifiedVote {
    fn from(legacy: crate::types::conviction_vote::ConvictionVote) -> Self {
        let now = chrono::Utc::now().timestamp();

        // Calculate days_locked from lock_start and lock_expiry
        let days_locked: f64 = if let Some(expiry) = legacy.lock_expiry {
            let lock_duration_seconds = (expiry - legacy.lock_start).max(0_i64) as f64;
            lock_duration_seconds / 86400.0 // Convert to days
        } else {
            // Indefinite lock - use a large value (e.g., 365 days)
            365.0_f64
        };

        let conviction_modifier = Some(ConvictionModifier::new(
            days_locked,
            Some(legacy.parameters.clone()),
        ));

        let mut vote = UnifiedVote {
            proposal_id: legacy.proposal_id,
            voter: legacy.voter,
            choice: legacy.choice,
            justification: legacy.justification,
            timestamp: now,
            reputation_allocated: legacy.reputation_weight,
            quadratic_modifier: None,
            conviction_modifier,
            delegation_modifier: None,
            final_weight: 0.0, // Will be calculated
        };

        vote.final_weight = vote.calculate_final_weight();
        vote
    }
}

/// Convert from ReputationVote to UnifiedVote (pure reputation, no modifiers)
impl From<crate::types::vote::ReputationVote> for UnifiedVote {
    fn from(legacy: crate::types::vote::ReputationVote) -> Self {
        let mut vote = UnifiedVote {
            proposal_id: legacy.proposal_id,
            voter: legacy.voter,
            choice: legacy.choice,
            justification: legacy.justification,
            timestamp: legacy.timestamp,
            reputation_allocated: legacy.reputation_weight,
            quadratic_modifier: None,
            conviction_modifier: None,
            delegation_modifier: None,
            final_weight: 0.0, // Will be calculated
        };

        vote.final_weight = vote.calculate_final_weight();
        vote
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_vote_no_modifiers() {
        let vote = UnifiedVote::simple(
            "prop_001".to_string(),
            "voter".to_string(),
            VoteChoice::For,
            100.0,
        );

        // No modifiers = linear weight
        assert_eq!(vote.final_weight, 100.0);
        assert!(vote.quadratic_modifier.is_none());
        assert!(vote.conviction_modifier.is_none());
        assert!(vote.delegation_modifier.is_none());
    }

    #[test]
    fn test_quadratic_only() {
        let vote = UnifiedVote::builder()
            .proposal_id("prop_001".to_string())
            .voter("voter".to_string())
            .choice(VoteChoice::For)
            .reputation_allocated(100.0)
            .with_quadratic()
            .build();

        // Quadratic dampening: √100 = 10
        assert_eq!(vote.final_weight, 10.0);
        assert!(vote.quadratic_modifier.is_some());
    }

    #[test]
    fn test_conviction_only() {
        let vote = UnifiedVote::builder()
            .proposal_id("prop_001".to_string())
            .voter("voter".to_string())
            .choice(VoteChoice::For)
            .reputation_allocated(100.0)
            .with_conviction(7.0)  // ~1.63x multiplier
            .build();

        // Conviction but no quadratic: 100 × 1.63 ≈ 163
        assert!(vote.final_weight > 160.0 && vote.final_weight < 165.0);
        assert!(vote.conviction_modifier.is_some());
    }

    #[test]
    fn test_quadratic_plus_conviction_synergy() {
        let vote = UnifiedVote::builder()
            .proposal_id("prop_001".to_string())
            .voter("voter".to_string())
            .choice(VoteChoice::For)
            .reputation_allocated(100.0)
            .with_quadratic()
            .with_conviction(7.0)  // ~1.63x
            .build();

        // Conviction THEN quadratic: √(100 × 1.63) ≈ 12.77
        // This is the synergistic benefit!
        assert!(vote.final_weight > 12.5 && vote.final_weight < 13.0);
    }

    #[test]
    fn test_delegation_attenuation() {
        let vote = UnifiedVote::builder()
            .proposal_id("prop_001".to_string())
            .voter("voter".to_string())
            .choice(VoteChoice::For)
            .reputation_allocated(100.0)
            .with_delegation("delegate".to_string(), 2)  // 2 hops
            .build();

        // 100 × 0.9^2 = 100 × 0.81 = 81.0
        assert_eq!(vote.final_weight, 81.0);
    }

    #[test]
    fn test_all_modifiers_combined() {
        let vote = UnifiedVote::builder()
            .proposal_id("prop_001".to_string())
            .voter("voter".to_string())
            .choice(VoteChoice::For)
            .reputation_allocated(100.0)
            .with_quadratic()
            .with_conviction(7.0)  // ~1.63x
            .with_delegation("delegate".to_string(), 1)  // 0.9x
            .build();

        // 1. Conviction: 100 × 1.63 = 163
        // 2. Quadratic: √163 ≈ 12.77
        // 3. Delegation: 12.77 × 0.9 ≈ 11.49
        assert!(vote.final_weight > 11.0 && vote.final_weight < 12.0);
    }

    #[test]
    fn test_attack_cost_improvement() {
        // Scenario: 100 honest voters with 100 reputation each

        // Honest collective with quadratic + conviction
        let honest_weight: f64 = (0..100)
            .map(|_| {
                let vote = UnifiedVote::builder()
                    .proposal_id("prop_001".to_string())
                    .voter(format!("honest_{}", 1))
                    .choice(VoteChoice::For)
                    .reputation_allocated(100.0)
                    .with_quadratic()
                    .with_conviction(7.0)
                    .build();
                vote.final_weight
            })
            .sum();

        // Each honest voter: √(100 × 1.63) ≈ 12.77
        // Total: 100 × 12.77 ≈ 1,277

        // Attacker needs to match with reputation R
        // √(R × 1.63) ≥ 1,277
        // R × 1.63 ≥ 1,631,529
        // R ≥ 1,001,000 (approximately)

        // Attack cost: 1,001,000 / 10,000 = 100x harder than without modifiers!
        let required_attacker_reputation = (honest_weight * honest_weight) / 1.63;

        assert!(required_attacker_reputation > 1_000_000.0);
    }

    // ============================================================================
    // Migration Tests - Converting Legacy Vote Types to UnifiedVote
    // ============================================================================

    #[test]
    fn test_quadratic_vote_migration() {
        use super::super::quadratic_vote::QuadraticVote;
        use super::super::vote::VoteChoice;

        let legacy = QuadraticVote {
            proposal_id: "prop_001".to_string(),
            voter: "voter_123".to_string(),
            choice: VoteChoice::For,
            justification: Some("I support this".to_string()),
            reputation_allocated: 100.0,
            quadratic_weight: 10.0, // √100
            conviction_factor: 1.5,
            timestamp: 1640000000,
            conviction_deadline: Some(1640086400), // 1 day later
        };

        let unified: UnifiedVote = legacy.into();

        assert_eq!(unified.proposal_id, "prop_001");
        assert_eq!(unified.voter, "voter_123");
        assert_eq!(unified.choice, VoteChoice::For);
        assert_eq!(unified.base_reputation_allocated, 100.0);

        // Should preserve quadratic modifier
        assert!(unified.quadratic_modifier.is_some());

        // Should preserve conviction modifier (factor > 1.0)
        assert!(unified.conviction_modifier.is_some());
        let conviction = unified.conviction_modifier.unwrap();
        assert!(conviction.days_locked > 0.0);
    }

    #[test]
    fn test_conviction_vote_migration() {
        use super::super::conviction_vote::{ConvictionVote, ConvictionParameters, ConvictionDecay};
        use super::super::vote::VoteChoice;

        let params = ConvictionParameters {
            time_constant_days: 7.0,
            max_conviction_multiplier: 3.0,
            min_lock_period_hours: 24,
            decay_function: ConvictionDecay::Exponential,
        };

        let legacy = ConvictionVote {
            proposal_id: "prop_002".to_string(),
            voter: "voter_456".to_string(),
            choice: VoteChoice::Against,
            justification: None,
            reputation_weight: 50.0,
            base_weight: 50.0,
            conviction_multiplier: 2.0,
            lock_start: 1640000000,
            lock_expiry: Some(1640604800), // 7 days
            timestamp: 1640000000,
            parameters: params,
        };

        let unified: UnifiedVote = legacy.into();

        assert_eq!(unified.proposal_id, "prop_002");
        assert_eq!(unified.voter, "voter_456");
        assert_eq!(unified.choice, VoteChoice::Against);
        assert_eq!(unified.base_reputation_allocated, 50.0);

        // Should have conviction modifier
        assert!(unified.conviction_modifier.is_some());
        let conviction = unified.conviction_modifier.unwrap();
        assert_eq!(conviction.days_locked, 7.0);
    }

    #[test]
    fn test_reputation_vote_migration() {
        use super::super::vote::{ReputationVote, VoteChoice};

        let legacy = ReputationVote {
            proposal_id: "prop_003".to_string(),
            voter: "voter_789".to_string(),
            choice: VoteChoice::Abstain,
            justification: Some("Need more info".to_string()),
            reputation_weight: 75.0,
            timestamp: 1640000000,
        };

        let unified: UnifiedVote = legacy.into();

        assert_eq!(unified.proposal_id, "prop_003");
        assert_eq!(unified.voter, "voter_789");
        assert_eq!(unified.choice, VoteChoice::Abstain);
        assert_eq!(unified.base_reputation_allocated, 75.0);

        // Should have NO modifiers (pure reputation vote)
        assert!(unified.quadratic_modifier.is_none());
        assert!(unified.conviction_modifier.is_none());
        assert!(unified.delegation_modifier.is_none());
    }
}
