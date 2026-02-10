//! # Economic Equilibrium Mechanisms
//!
//! Game-theoretic balance for the KREDIT system.
//!
//! ## Features
//!
//! - **Slashing**: Penalty for malicious behavior
//! - **Rewards**: Incentives for honest participation
//! - **Bonding Curves**: Trust accumulation with diminishing returns
//! - **Commit-Reveal Voting**: Anti-gaming mechanism
//!
//! ## Philosophy
//!
//! Economic incentives must align individual agent behavior with collective good.
//! Gaming should be more expensive than honest participation, and honest
//! participation should be more rewarding than defection.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Slashing
// ============================================================================

/// Slashing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingConfig {
    /// Enable slashing
    pub enabled: bool,
    /// Slash percentage for minor violations (0.0-1.0)
    pub minor_violation_rate: f64,
    /// Slash percentage for major violations (0.0-1.0)
    pub major_violation_rate: f64,
    /// Slash percentage for critical violations (0.0-1.0)
    pub critical_violation_rate: f64,
    /// Cooldown before KREDIT can be used after slashing (ms)
    pub slash_cooldown_ms: u64,
    /// Maximum cumulative slash before suspension
    pub max_cumulative_slash: f64,
}

impl Default for SlashingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            minor_violation_rate: 0.05,    // 5%
            major_violation_rate: 0.20,    // 20%
            critical_violation_rate: 0.50, // 50%
            slash_cooldown_ms: 3600_000,   // 1 hour
            max_cumulative_slash: 0.9,     // 90% max
        }
    }
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Minor rule violation
    Minor,
    /// Significant violation
    Major,
    /// Critical/malicious violation
    Critical,
}

/// Slashing event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashEvent {
    /// Agent ID
    pub agent_id: String,
    /// Violation type
    pub violation: ViolationType,
    /// Severity
    pub severity: ViolationSeverity,
    /// KREDIT slashed
    pub amount_slashed: u64,
    /// Percentage slashed
    pub slash_rate: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Evidence/reason
    pub evidence: String,
}

/// Types of violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Attempted gaming of trust score
    TrustGaming,
    /// Sybil attack detected
    SybilAttack,
    /// Collusion detected
    Collusion,
    /// Invalid proof submitted
    InvalidProof,
    /// Vote manipulation
    VoteManipulation,
    /// Exceeded rate limits
    RateLimitViolation,
    /// Protocol violation
    ProtocolViolation,
    /// Custom violation
    Custom(String),
}

/// Slashing engine
pub struct SlashingEngine {
    /// Configuration
    config: SlashingConfig,
    /// Slash history per agent
    history: HashMap<String, VecDeque<SlashEvent>>,
    /// Cumulative slash per agent
    cumulative_slash: HashMap<String, f64>,
    /// Agents in cooldown
    cooldowns: HashMap<String, u64>,
}

impl SlashingEngine {
    /// Create a new slashing engine
    pub fn new(config: SlashingConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            cumulative_slash: HashMap::new(),
            cooldowns: HashMap::new(),
        }
    }

    /// Execute a slash
    pub fn slash(
        &mut self,
        agent_id: &str,
        violation: ViolationType,
        severity: ViolationSeverity,
        current_kredit: u64,
        evidence: &str,
    ) -> SlashResult {
        if !self.config.enabled {
            return SlashResult::Disabled;
        }

        let slash_rate = match severity {
            ViolationSeverity::Minor => self.config.minor_violation_rate,
            ViolationSeverity::Major => self.config.major_violation_rate,
            ViolationSeverity::Critical => self.config.critical_violation_rate,
        };

        let amount_slashed = (current_kredit as f64 * slash_rate) as u64;

        let now = Self::now();

        let event = SlashEvent {
            agent_id: agent_id.to_string(),
            violation,
            severity,
            amount_slashed,
            slash_rate,
            timestamp: now,
            evidence: evidence.to_string(),
        };

        // Record history
        self.history
            .entry(agent_id.to_string())
            .or_default()
            .push_back(event.clone());

        // Update cumulative slash
        let cumulative = self
            .cumulative_slash
            .entry(agent_id.to_string())
            .or_insert(0.0);
        *cumulative += slash_rate;

        // Set cooldown
        self.cooldowns
            .insert(agent_id.to_string(), now + self.config.slash_cooldown_ms);

        // Check for suspension
        if *cumulative >= self.config.max_cumulative_slash {
            return SlashResult::Suspended {
                event,
                cumulative_slash: *cumulative,
            };
        }

        SlashResult::Slashed {
            event,
            cumulative_slash: *cumulative,
        }
    }

    /// Check if agent is in cooldown
    pub fn in_cooldown(&self, agent_id: &str) -> bool {
        self.cooldowns
            .get(agent_id)
            .map(|&expiry| Self::now() < expiry)
            .unwrap_or(false)
    }

    /// Get cooldown remaining (ms)
    pub fn cooldown_remaining(&self, agent_id: &str) -> u64 {
        self.cooldowns
            .get(agent_id)
            .map(|&expiry| expiry.saturating_sub(Self::now()))
            .unwrap_or(0)
    }

    /// Get slash history
    pub fn get_history(&self, agent_id: &str) -> Vec<SlashEvent> {
        self.history
            .get(agent_id)
            .map(|h| h.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get cumulative slash
    pub fn get_cumulative_slash(&self, agent_id: &str) -> f64 {
        self.cumulative_slash.get(agent_id).copied().unwrap_or(0.0)
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Result of slashing
#[derive(Debug, Clone)]
pub enum SlashResult {
    /// Slashing is disabled
    Disabled,
    /// Successfully slashed.
    Slashed {
        /// Details of the slash event.
        event: SlashEvent,
        /// Cumulative slash fraction for this agent.
        cumulative_slash: f64,
    },
    /// Agent suspended due to excessive violations.
    Suspended {
        /// Details of the slash event.
        event: SlashEvent,
        /// Cumulative slash fraction for this agent.
        cumulative_slash: f64,
    },
}

// ============================================================================
// Rewards
// ============================================================================

/// Reward configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Enable rewards
    pub enabled: bool,
    /// Base reward for participation
    pub base_participation_reward: u64,
    /// Bonus multiplier for consistent participation
    pub consistency_multiplier: f64,
    /// Trust-weighted reward scaling
    pub trust_weight: f64,
    /// Maximum reward per period
    pub max_reward_per_period: u64,
    /// Reward period (ms)
    pub period_ms: u64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_participation_reward: 10,
            consistency_multiplier: 1.5,
            trust_weight: 0.5,
            max_reward_per_period: 1000,
            period_ms: 24 * 3600_000, // 24 hours
        }
    }
}

/// Reward event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEvent {
    /// Agent ID
    pub agent_id: String,
    /// Reward type
    pub reward_type: RewardType,
    /// Amount rewarded
    pub amount: u64,
    /// Trust at time of reward
    pub trust: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Types of rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardType {
    /// Participation in voting
    Participation,
    /// Honest behavior (verified by outcome)
    HonestBehavior,
    /// Consensus contribution
    ConsensusContribution,
    /// Long-term reliability
    Reliability,
    /// Referral/onboarding
    Referral,
    /// Custom reward
    Custom(String),
}

/// Reward engine
pub struct RewardEngine {
    /// Configuration
    config: RewardConfig,
    /// Rewards per period per agent
    period_rewards: HashMap<String, PeriodRewards>,
    /// Reward history
    history: HashMap<String, VecDeque<RewardEvent>>,
}

/// Rewards within a period
#[derive(Debug, Clone, Default)]
struct PeriodRewards {
    /// Period start
    period_start: u64,
    /// Total rewards this period
    total_rewards: u64,
    /// Participation count
    participation_count: u32,
}

impl RewardEngine {
    /// Create a new reward engine
    pub fn new(config: RewardConfig) -> Self {
        Self {
            config,
            period_rewards: HashMap::new(),
            history: HashMap::new(),
        }
    }

    /// Calculate reward for participation
    pub fn calculate_participation_reward(&self, agent_id: &str, trust: f64) -> u64 {
        if !self.config.enabled {
            return 0;
        }

        let base = self.config.base_participation_reward;
        let trust_bonus = (base as f64 * trust * self.config.trust_weight) as u64;

        // Check consistency bonus
        let consistency_bonus = self
            .period_rewards
            .get(agent_id)
            .map(|p| {
                if p.participation_count > 5 {
                    (base as f64 * (self.config.consistency_multiplier - 1.0)) as u64
                } else {
                    0
                }
            })
            .unwrap_or(0);

        base + trust_bonus + consistency_bonus
    }

    /// Grant a reward
    pub fn grant_reward(
        &mut self,
        agent_id: &str,
        reward_type: RewardType,
        trust: f64,
    ) -> Option<RewardEvent> {
        if !self.config.enabled {
            return None;
        }

        let now = Self::now();

        // Pre-calculate participation reward before borrowing period_rewards mutably
        let participation_reward = self.calculate_participation_reward(agent_id, trust);

        // Get or create period rewards
        let period = self
            .period_rewards
            .entry(agent_id.to_string())
            .or_insert_with(|| PeriodRewards {
                period_start: now,
                total_rewards: 0,
                participation_count: 0,
            });

        // Check if new period
        if now.saturating_sub(period.period_start) >= self.config.period_ms {
            *period = PeriodRewards {
                period_start: now,
                total_rewards: 0,
                participation_count: 0,
            };
        }

        // Check period limit
        if period.total_rewards >= self.config.max_reward_per_period {
            return None;
        }

        // Calculate reward
        let amount = match &reward_type {
            RewardType::Participation => participation_reward,
            RewardType::HonestBehavior => {
                (self.config.base_participation_reward as f64 * 2.0 * trust) as u64
            }
            RewardType::ConsensusContribution => self.config.base_participation_reward * 3,
            RewardType::Reliability => {
                (self.config.base_participation_reward as f64 * 5.0 * trust) as u64
            }
            RewardType::Referral => self.config.base_participation_reward * 10,
            RewardType::Custom(_) => self.config.base_participation_reward,
        };

        // Apply period limit
        let remaining = self
            .config
            .max_reward_per_period
            .saturating_sub(period.total_rewards);
        let actual_amount = amount.min(remaining);

        if actual_amount == 0 {
            return None;
        }

        // Update tracking
        period.total_rewards += actual_amount;
        period.participation_count += 1;

        let event = RewardEvent {
            agent_id: agent_id.to_string(),
            reward_type,
            amount: actual_amount,
            trust,
            timestamp: now,
        };

        // Record history
        self.history
            .entry(agent_id.to_string())
            .or_default()
            .push_back(event.clone());

        Some(event)
    }

    /// Get reward history
    pub fn get_history(&self, agent_id: &str) -> Vec<RewardEvent> {
        self.history
            .get(agent_id)
            .map(|h| h.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ============================================================================
// Bonding Curves
// ============================================================================

/// Bonding curve for trust accumulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondingCurve {
    /// Curve type
    pub curve_type: BondingCurveType,
    /// Maximum trust achievable
    pub max_trust: f64,
    /// Steepness parameter
    pub steepness: f64,
    /// Midpoint (inflection point)
    pub midpoint: f64,
}

/// Types of bonding curves
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BondingCurveType {
    /// Linear: trust = min(actions * rate, max)
    Linear,
    /// Logarithmic: trust = log(1 + actions) * rate
    Logarithmic,
    /// Sigmoid: trust = max / (1 + e^(-steepness*(actions-midpoint)))
    Sigmoid,
    /// Square root: trust = sqrt(actions) * rate
    SquareRoot,
}

impl Default for BondingCurve {
    fn default() -> Self {
        Self {
            curve_type: BondingCurveType::Sigmoid,
            max_trust: 1.0,
            steepness: 0.1,
            midpoint: 50.0,
        }
    }
}

impl BondingCurve {
    /// Calculate trust from action count
    pub fn calculate_trust(&self, successful_actions: u64) -> f64 {
        let x = successful_actions as f64;

        let trust = match self.curve_type {
            BondingCurveType::Linear => (x * 0.01).min(self.max_trust),
            BondingCurveType::Logarithmic => ((1.0 + x).ln() * 0.2).min(self.max_trust),
            BondingCurveType::Sigmoid => {
                self.max_trust / (1.0 + (-self.steepness * (x - self.midpoint)).exp())
            }
            BondingCurveType::SquareRoot => (x.sqrt() * 0.1).min(self.max_trust),
        };

        trust.clamp(0.0, self.max_trust)
    }

    /// Calculate marginal trust gain for next action
    pub fn marginal_gain(&self, current_actions: u64) -> f64 {
        let current = self.calculate_trust(current_actions);
        let next = self.calculate_trust(current_actions + 1);
        next - current
    }

    /// Calculate actions needed to reach target trust
    pub fn actions_for_trust(&self, target_trust: f64) -> Option<u64> {
        if target_trust >= self.max_trust {
            return None; // Can never reach
        }

        // Binary search for sigmoid
        match self.curve_type {
            BondingCurveType::Sigmoid => {
                // Solve: target = max / (1 + e^(-k*(x-m)))
                // x = m - ln(max/target - 1) / k
                let ratio = self.max_trust / target_trust - 1.0;
                if ratio <= 0.0 {
                    return None;
                }
                let x = self.midpoint - ratio.ln() / self.steepness;
                Some(x.max(0.0).ceil() as u64)
            }
            _ => {
                // Binary search for other curves
                let mut low = 0u64;
                let mut high = 10000u64;

                while low < high {
                    let mid = (low + high) / 2;
                    if self.calculate_trust(mid) < target_trust {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }

                Some(low)
            }
        }
    }
}

// ============================================================================
// Commit-Reveal Voting
// ============================================================================

/// Commit-reveal voting to prevent vote manipulation
#[allow(dead_code)]
pub struct CommitRevealVoting {
    /// Commits (agent_id -> (proposal_id -> commitment))
    commits: HashMap<String, HashMap<String, VoteCommitment>>,
    /// Reveals
    reveals: HashMap<String, HashMap<String, VoteReveal>>,
    /// Commit phase duration (ms)
    commit_phase_ms: u64,
    /// Reveal phase duration (ms)
    reveal_phase_ms: u64,
}

/// A vote commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteCommitment {
    /// Hash of (vote || salt)
    pub commitment_hash: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
}

/// A revealed vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteReveal {
    /// The vote
    pub vote: CommitRevealVote,
    /// Salt used
    pub salt: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
}

/// Vote for commit-reveal
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CommitRevealVote {
    /// Yes/Approve
    Yes,
    /// No/Reject
    No,
    /// Abstain
    Abstain,
}

impl CommitRevealVoting {
    /// Create a new commit-reveal voting system
    pub fn new(commit_phase_ms: u64, reveal_phase_ms: u64) -> Self {
        Self {
            commits: HashMap::new(),
            reveals: HashMap::new(),
            commit_phase_ms,
            reveal_phase_ms,
        }
    }

    /// Create a commitment for a vote
    pub fn create_commitment(vote: CommitRevealVote, salt: &[u8; 32]) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(b"vote-commitment-v1");
        hasher.update([vote as u8]);
        hasher.update(salt);
        hasher.finalize().into()
    }

    /// Submit a commitment
    pub fn commit(
        &mut self,
        agent_id: &str,
        proposal_id: &str,
        commitment: VoteCommitment,
    ) -> bool {
        let agent_commits = self.commits.entry(agent_id.to_string()).or_default();

        if agent_commits.contains_key(proposal_id) {
            return false; // Already committed
        }

        agent_commits.insert(proposal_id.to_string(), commitment);
        true
    }

    /// Reveal a vote
    pub fn reveal(
        &mut self,
        agent_id: &str,
        proposal_id: &str,
        vote: CommitRevealVote,
        salt: &[u8; 32],
    ) -> Result<(), CommitRevealError> {
        // Get commitment
        let commitment = self
            .commits
            .get(agent_id)
            .and_then(|c| c.get(proposal_id))
            .ok_or(CommitRevealError::NoCommitment)?;

        // Verify commitment matches
        let expected_hash = Self::create_commitment(vote, salt);
        if commitment.commitment_hash != expected_hash {
            return Err(CommitRevealError::CommitmentMismatch);
        }

        // Store reveal
        let now = Self::now();
        let reveal = VoteReveal {
            vote,
            salt: *salt,
            timestamp: now,
        };

        self.reveals
            .entry(agent_id.to_string())
            .or_default()
            .insert(proposal_id.to_string(), reveal);

        Ok(())
    }

    /// Get revealed votes for a proposal
    pub fn get_revealed_votes(&self, proposal_id: &str) -> Vec<(String, CommitRevealVote)> {
        let mut votes = Vec::new();

        for (agent_id, agent_reveals) in &self.reveals {
            if let Some(reveal) = agent_reveals.get(proposal_id) {
                votes.push((agent_id.clone(), reveal.vote));
            }
        }

        votes
    }

    /// Check if agent has committed for proposal
    pub fn has_committed(&self, agent_id: &str, proposal_id: &str) -> bool {
        self.commits
            .get(agent_id)
            .map(|c| c.contains_key(proposal_id))
            .unwrap_or(false)
    }

    /// Check if agent has revealed for proposal
    pub fn has_revealed(&self, agent_id: &str, proposal_id: &str) -> bool {
        self.reveals
            .get(agent_id)
            .map(|r| r.contains_key(proposal_id))
            .unwrap_or(false)
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Commit-reveal errors
#[derive(Debug, Clone)]
pub enum CommitRevealError {
    /// No commitment found
    NoCommitment,
    /// Commitment hash doesn't match reveal
    CommitmentMismatch,
    /// Commit phase ended
    CommitPhaseEnded,
    /// Reveal phase ended
    RevealPhaseEnded,
}

impl std::fmt::Display for CommitRevealError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCommitment => write!(f, "No commitment found"),
            Self::CommitmentMismatch => write!(f, "Commitment hash doesn't match reveal"),
            Self::CommitPhaseEnded => write!(f, "Commit phase has ended"),
            Self::RevealPhaseEnded => write!(f, "Reveal phase has ended"),
        }
    }
}

impl std::error::Error for CommitRevealError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slashing() {
        let mut engine = SlashingEngine::new(SlashingConfig::default());

        let result = engine.slash(
            "agent-1",
            ViolationType::TrustGaming,
            ViolationSeverity::Minor,
            1000,
            "Detected gaming attempt",
        );

        match result {
            SlashResult::Slashed { event, .. } => {
                assert_eq!(event.amount_slashed, 50); // 5% of 1000
            }
            _ => panic!("Expected Slashed result"),
        }
    }

    #[test]
    fn test_rewards() {
        let mut engine = RewardEngine::new(RewardConfig::default());

        let event = engine.grant_reward("agent-1", RewardType::Participation, 0.8);
        assert!(event.is_some());

        let event = event.unwrap();
        assert!(event.amount > 0);
    }

    #[test]
    fn test_bonding_curve_sigmoid() {
        let curve = BondingCurve::default();

        // At midpoint, should be max/2
        let mid_trust = curve.calculate_trust(50);
        assert!((mid_trust - 0.5).abs() < 0.01);

        // Far past midpoint, should approach max
        let high_trust = curve.calculate_trust(200);
        assert!(high_trust > 0.9);

        // At 0, should be low
        let low_trust = curve.calculate_trust(0);
        assert!(low_trust < 0.1);
    }

    #[test]
    fn test_commit_reveal() {
        let mut voting = CommitRevealVoting::new(3600_000, 3600_000);

        let salt = [42u8; 32];
        let commitment_hash = CommitRevealVoting::create_commitment(CommitRevealVote::Yes, &salt);

        let commitment = VoteCommitment {
            commitment_hash,
            timestamp: 0,
        };

        // Commit
        assert!(voting.commit("agent-1", "prop-1", commitment));
        assert!(voting.has_committed("agent-1", "prop-1"));

        // Reveal
        voting
            .reveal("agent-1", "prop-1", CommitRevealVote::Yes, &salt)
            .unwrap();
        assert!(voting.has_revealed("agent-1", "prop-1"));

        // Get votes
        let votes = voting.get_revealed_votes("prop-1");
        assert_eq!(votes.len(), 1);
        assert!(matches!(votes[0].1, CommitRevealVote::Yes));
    }

    #[test]
    fn test_commit_reveal_mismatch() {
        let mut voting = CommitRevealVoting::new(3600_000, 3600_000);

        let salt = [42u8; 32];
        let wrong_salt = [99u8; 32];
        let commitment_hash = CommitRevealVoting::create_commitment(CommitRevealVote::Yes, &salt);

        let commitment = VoteCommitment {
            commitment_hash,
            timestamp: 0,
        };

        voting.commit("agent-1", "prop-1", commitment);

        // Try to reveal with wrong salt
        let result = voting.reveal("agent-1", "prop-1", CommitRevealVote::Yes, &wrong_salt);
        assert!(matches!(result, Err(CommitRevealError::CommitmentMismatch)));

        // Try to reveal wrong vote
        let result = voting.reveal("agent-1", "prop-1", CommitRevealVote::No, &salt);
        assert!(matches!(result, Err(CommitRevealError::CommitmentMismatch)));
    }
}
