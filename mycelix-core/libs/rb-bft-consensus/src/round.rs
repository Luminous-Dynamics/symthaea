// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Round management for RB-BFT consensus

use serde::{Deserialize, Serialize};

use crate::proposal::{Proposal, ProposalStatus, ProposalVotingState};
use crate::vote::VoteCollection;
use crate::validator::ValidatorSet;
use crate::error::{ConsensusError, ConsensusResult};

/// State of a consensus round
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Round not yet started
    NotStarted,
    /// Waiting for leader to propose
    WaitingForProposal,
    /// Proposal received, gathering votes
    Voting,
    /// Consensus reached
    Committed,
    /// Round failed (timeout or rejection)
    Failed,
    /// Round skipped (e.g., leader offline)
    Skipped,
}

/// A single consensus round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRound {
    /// Round number
    pub number: u64,
    /// Current state
    pub state: RoundState,
    /// Leader for this round
    pub leader: String,
    /// Proposal (if any)
    pub proposal: Option<Proposal>,
    /// Voting state
    pub voting_state: Option<ProposalVotingState>,
    /// Collected votes
    pub votes: VoteCollection,
    /// When the round started
    pub started_at: i64,
    /// When the round ended (0 if ongoing)
    pub ended_at: i64,
    /// Timeout duration in milliseconds
    pub timeout_ms: u64,
    /// Result hash (if committed)
    pub result_hash: Option<String>,
}

impl ConsensusRound {
    /// Create a new round
    pub fn new(number: u64, leader: String, timeout_ms: u64) -> Self {
        Self {
            number,
            state: RoundState::NotStarted,
            leader,
            proposal: None,
            voting_state: None,
            votes: VoteCollection::new(),
            started_at: 0,
            ended_at: 0,
            timeout_ms,
            result_hash: None,
        }
    }

    /// Start the round
    pub fn start(&mut self) {
        self.started_at = current_timestamp();
        self.state = RoundState::WaitingForProposal;
    }

    /// Submit a proposal for this round
    pub fn submit_proposal(&mut self, proposal: Proposal, threshold: f32) -> ConsensusResult<()> {
        if self.state != RoundState::WaitingForProposal {
            return Err(ConsensusError::InvalidRoundState {
                expected: "WaitingForProposal".to_string(),
                actual: format!("{:?}", self.state),
            });
        }

        if proposal.round != self.number {
            return Err(ConsensusError::InvalidProposal {
                reason: format!("Wrong round: expected {}, got {}", self.number, proposal.round),
            });
        }

        if proposal.proposer != self.leader {
            return Err(ConsensusError::NotLeader {
                round: self.number,
                leader: self.leader.clone(),
            });
        }

        if !proposal.is_valid() {
            return Err(ConsensusError::InvalidProposal {
                reason: "Proposal validation failed".to_string(),
            });
        }

        self.voting_state = Some(ProposalVotingState::new(proposal.id.clone(), threshold));
        self.proposal = Some(proposal);
        self.state = RoundState::Voting;

        Ok(())
    }

    /// Check if the round has timed out
    pub fn is_timed_out(&self) -> bool {
        if self.started_at == 0 {
            return false;
        }
        let elapsed = (current_timestamp() - self.started_at) as u64 * 1000; // Convert to ms
        elapsed > self.timeout_ms
    }

    /// Check if the round is complete (committed or failed)
    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            RoundState::Committed | RoundState::Failed | RoundState::Skipped
        )
    }

    /// Commit the round with a result
    pub fn commit(&mut self, result_hash: String) {
        self.state = RoundState::Committed;
        self.result_hash = Some(result_hash);
        self.ended_at = current_timestamp();

        if let Some(ref mut vs) = self.voting_state {
            vs.finalize(true);
        }
    }

    /// Fail the round
    pub fn fail(&mut self, _reason: &str) {
        self.state = RoundState::Failed;
        self.ended_at = current_timestamp();

        if let Some(ref mut vs) = self.voting_state {
            vs.status = ProposalStatus::Rejected;
            vs.voting_ended = current_timestamp();
        }
    }

    /// Skip the round
    pub fn skip(&mut self) {
        self.state = RoundState::Skipped;
        self.ended_at = current_timestamp();
    }

    /// Get round duration in seconds
    pub fn duration_secs(&self) -> i64 {
        if self.ended_at > 0 {
            self.ended_at - self.started_at
        } else if self.started_at > 0 {
            current_timestamp() - self.started_at
        } else {
            0
        }
    }
}

/// Manager for tracking multiple rounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundManager {
    /// Current round number
    current_round: u64,
    /// Active round (if any)
    active_round: Option<ConsensusRound>,
    /// History of completed rounds
    completed_rounds: Vec<RoundSummary>,
    /// Default timeout for rounds
    default_timeout_ms: u64,
    /// Maximum rounds to keep in history
    max_history: usize,
}

impl RoundManager {
    /// Create a new round manager
    pub fn new(default_timeout_ms: u64) -> Self {
        Self {
            current_round: 0,
            active_round: None,
            completed_rounds: Vec::new(),
            default_timeout_ms,
            max_history: 100,
        }
    }

    /// Get current round number
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Get the active round (if any)
    pub fn active_round(&self) -> Option<&ConsensusRound> {
        self.active_round.as_ref()
    }

    /// Get mutable reference to active round
    pub fn active_round_mut(&mut self) -> Option<&mut ConsensusRound> {
        self.active_round.as_mut()
    }

    /// Start a new round
    pub fn start_new_round(&mut self, validators: &ValidatorSet) -> ConsensusResult<&ConsensusRound> {
        // Complete any active round first
        if let Some(ref round) = self.active_round {
            if !round.is_complete() {
                return Err(ConsensusError::InvalidRoundState {
                    expected: "no active round or completed round".to_string(),
                    actual: format!("round {} in state {:?}", round.number, round.state),
                });
            }
        }

        // Archive the completed round
        if let Some(round) = self.active_round.take() {
            self.archive_round(&round);
        }

        // Increment round number
        self.current_round += 1;

        // Select leader for new round
        let leader = validators
            .select_leader(self.current_round)
            .ok_or(ConsensusError::InsufficientValidators {
                have: validators.active_count(),
                need: 1,
            })?;

        // Create and start new round
        let mut round = ConsensusRound::new(
            self.current_round,
            leader.id.clone(),
            self.default_timeout_ms,
        );
        round.start();

        self.active_round = Some(round);
        Ok(self.active_round.as_ref().expect("active_round set to Some on line above"))
    }

    /// Archive a completed round
    fn archive_round(&mut self, round: &ConsensusRound) {
        let summary = RoundSummary {
            number: round.number,
            state: round.state,
            leader: round.leader.clone(),
            proposal_id: round.proposal.as_ref().map(|p| p.id.clone()),
            vote_count: round.votes.vote_count(),
            duration_secs: round.duration_secs(),
            result_hash: round.result_hash.clone(),
        };

        self.completed_rounds.push(summary);

        // Trim history if needed
        while self.completed_rounds.len() > self.max_history {
            self.completed_rounds.remove(0);
        }
    }

    /// Get round history
    pub fn history(&self) -> &[RoundSummary] {
        &self.completed_rounds
    }

    /// Get statistics
    pub fn stats(&self) -> RoundStats {
        let total = self.completed_rounds.len();
        let committed = self.completed_rounds.iter()
            .filter(|r| r.state == RoundState::Committed)
            .count();
        let failed = self.completed_rounds.iter()
            .filter(|r| r.state == RoundState::Failed)
            .count();
        let skipped = self.completed_rounds.iter()
            .filter(|r| r.state == RoundState::Skipped)
            .count();

        let avg_duration = if total > 0 {
            self.completed_rounds.iter().map(|r| r.duration_secs).sum::<i64>() as f64 / total as f64
        } else {
            0.0
        };

        RoundStats {
            total_rounds: total,
            committed_rounds: committed,
            failed_rounds: failed,
            skipped_rounds: skipped,
            success_rate: if total > 0 { committed as f64 / total as f64 } else { 0.0 },
            avg_duration_secs: avg_duration,
        }
    }
}

/// Summary of a completed round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundSummary {
    pub number: u64,
    pub state: RoundState,
    pub leader: String,
    pub proposal_id: Option<String>,
    pub vote_count: usize,
    pub duration_secs: i64,
    pub result_hash: Option<String>,
}

/// Statistics about rounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundStats {
    pub total_rounds: usize,
    pub committed_rounds: usize,
    pub failed_rounds: usize,
    pub skipped_rounds: usize,
    pub success_rate: f64,
    pub avg_duration_secs: f64,
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_lifecycle() {
        let mut round = ConsensusRound::new(1, "leader-1".to_string(), 30000);

        assert_eq!(round.state, RoundState::NotStarted);

        round.start();
        assert_eq!(round.state, RoundState::WaitingForProposal);

        round.commit("hash-123".to_string());
        assert_eq!(round.state, RoundState::Committed);
        assert!(round.is_complete());
    }

    #[test]
    fn test_round_timeout() {
        let mut round = ConsensusRound::new(1, "leader-1".to_string(), 100); // 100ms timeout
        round.start();

        // Simulate time passing by manually adjusting started_at
        round.started_at -= 1; // Subtract 1 second (1000ms > 100ms timeout)
        assert!(round.is_timed_out());
    }
}
