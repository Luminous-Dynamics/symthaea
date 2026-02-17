//! Reputation-weighted voting types

use serde::{Deserialize, Serialize};

/// Reputation-weighted vote with composite trust scoring
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReputationVote {
    pub proposal_id: String,
    pub voter: String, // AgentPubKey as string
    pub choice: VoteChoice,
    pub justification: Option<String>,

    // Reputation weighting
    pub reputation_weight: f64, // Composite PoGQ + TCDM + Entropy + Stake

    pub timestamp: i64,
}

/// Vote choice enumeration
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}
