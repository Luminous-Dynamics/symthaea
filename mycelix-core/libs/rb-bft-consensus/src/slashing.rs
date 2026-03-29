// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Slashing conditions and evidence for RB-BFT consensus

use serde::{Deserialize, Serialize};

use crate::vote::{Vote, DoubleVoteEvidence};
use crate::validator::SlashingSeverity;

/// Types of slashable offenses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlashableOffense {
    /// Voting twice in the same round with different decisions
    DoubleVoting(Box<DoubleVoteEvidence>),

    /// Proposing multiple different blocks in the same round
    DoubleProposing {
        round: u64,
        proposer: String,
        proposal_hash_1: String,
        proposal_hash_2: String,
    },

    /// Signing conflicting commits
    ConflictingCommit {
        round: u64,
        validator: String,
        commit_hash_1: String,
        commit_hash_2: String,
    },

    /// Consistently voting against valid proposals without justification
    MaliciousRejection {
        validator: String,
        rejection_count: u32,
        rounds_observed: u32,
    },

    /// Not participating in consensus when required
    Inactivity {
        validator: String,
        missed_rounds: u32,
        total_rounds: u32,
    },

    /// Submitting invalid proposals repeatedly
    InvalidProposals {
        validator: String,
        invalid_count: u32,
        reasons: Vec<String>,
    },

    /// Attempting to manipulate reputation scores
    ReputationManipulation {
        validator: String,
        evidence: String,
    },

    /// H-02 remediation: Suspicious abstention pattern
    /// Coordinated abstention can manipulate consensus outcomes
    SuspiciousAbstention {
        validator: String,
        abstention_rate: f32,
        window_size: u64,
        rounds_abstained: Vec<u64>,
    },
}

impl SlashableOffense {
    /// Get the severity of this offense
    pub fn severity(&self) -> SlashingSeverity {
        match self {
            SlashableOffense::DoubleVoting(_) => SlashingSeverity::Severe,
            SlashableOffense::DoubleProposing { .. } => SlashingSeverity::Severe,
            SlashableOffense::ConflictingCommit { .. } => SlashingSeverity::Critical,
            SlashableOffense::MaliciousRejection { rejection_count, .. } => {
                if *rejection_count > 10 {
                    SlashingSeverity::Moderate
                } else {
                    SlashingSeverity::Minor
                }
            }
            SlashableOffense::Inactivity { missed_rounds, total_rounds, .. } => {
                let rate = *missed_rounds as f32 / *total_rounds as f32;
                if rate > 0.5 {
                    SlashingSeverity::Moderate
                } else {
                    SlashingSeverity::Minor
                }
            }
            SlashableOffense::InvalidProposals { invalid_count, .. } => {
                if *invalid_count > 5 {
                    SlashingSeverity::Moderate
                } else {
                    SlashingSeverity::Minor
                }
            }
            SlashableOffense::ReputationManipulation { .. } => SlashingSeverity::Critical,
            // H-02 remediation: Suspicious abstention severity based on rate
            SlashableOffense::SuspiciousAbstention { abstention_rate, .. } => {
                if *abstention_rate > 0.5 {
                    SlashingSeverity::Moderate
                } else {
                    SlashingSeverity::Minor
                }
            }
        }
    }

    /// Get the validator who committed this offense
    pub fn offending_validator(&self) -> &str {
        match self {
            SlashableOffense::DoubleVoting(evidence) => evidence.byzantine_validator(),
            SlashableOffense::DoubleProposing { proposer, .. } => proposer,
            SlashableOffense::ConflictingCommit { validator, .. } => validator,
            SlashableOffense::MaliciousRejection { validator, .. } => validator,
            SlashableOffense::Inactivity { validator, .. } => validator,
            SlashableOffense::InvalidProposals { validator, .. } => validator,
            SlashableOffense::ReputationManipulation { validator, .. } => validator,
            SlashableOffense::SuspiciousAbstention { validator, .. } => validator,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            SlashableOffense::DoubleVoting(_) => "Double voting in same round".to_string(),
            SlashableOffense::DoubleProposing { round, .. } => {
                format!("Double proposing in round {}", round)
            }
            SlashableOffense::ConflictingCommit { round, .. } => {
                format!("Conflicting commits in round {}", round)
            }
            SlashableOffense::MaliciousRejection { rejection_count, .. } => {
                format!("Malicious rejection ({} times)", rejection_count)
            }
            SlashableOffense::Inactivity { missed_rounds, total_rounds, .. } => {
                format!("Inactivity: missed {} of {} rounds", missed_rounds, total_rounds)
            }
            SlashableOffense::InvalidProposals { invalid_count, .. } => {
                format!("Invalid proposals ({} times)", invalid_count)
            }
            SlashableOffense::ReputationManipulation { .. } => {
                "Reputation manipulation attempt".to_string()
            }
            SlashableOffense::SuspiciousAbstention { abstention_rate, window_size, .. } => {
                format!(
                    "Suspicious abstention pattern: {:.1}% over {} rounds",
                    abstention_rate * 100.0,
                    window_size
                )
            }
        }
    }
}

/// Default confirmation deadline in seconds (24 hours)
pub const DEFAULT_CONFIRMATION_DEADLINE_SECS: i64 = 24 * 60 * 60;

/// A slashing event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    /// Unique event ID
    pub id: String,
    /// The offense that triggered slashing
    pub offense: SlashableOffense,
    /// Severity assigned
    pub severity: SlashingSeverity,
    /// Penalty applied to reputation (0.0-1.0)
    pub penalty: f32,
    /// When the slashing occurred
    pub timestamp: i64,
    /// M-03 remediation: Deadline for confirmations (Unix timestamp)
    pub confirmation_deadline: i64,
    /// Validator who reported the offense
    pub reporter: String,
    /// Whether this slashing has been confirmed by consensus
    pub confirmed: bool,
    /// Validators who confirmed this slashing
    pub confirmations: Vec<String>,
}

impl SlashingEvent {
    /// Create a new slashing event
    pub fn new(offense: SlashableOffense, reporter: String) -> Self {
        Self::with_deadline(offense, reporter, DEFAULT_CONFIRMATION_DEADLINE_SECS)
    }

    /// Create a new slashing event with custom deadline
    pub fn with_deadline(offense: SlashableOffense, reporter: String, deadline_secs: i64) -> Self {
        let severity = offense.severity();
        let penalty = match severity {
            SlashingSeverity::Minor => 0.1,
            SlashingSeverity::Moderate => 0.3,
            SlashingSeverity::Severe => 0.5,
            SlashingSeverity::Critical => 0.8,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        Self {
            id: format!(
                "slash-{}-{}",
                offense.offending_validator(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0)
            ),
            offense,
            severity,
            penalty,
            timestamp: now,
            // M-03 remediation: Set confirmation deadline
            confirmation_deadline: now + deadline_secs,
            reporter,
            confirmed: false,
            confirmations: Vec::new(),
        }
    }

    /// Add a confirmation
    pub fn add_confirmation(&mut self, validator: String) {
        if !self.confirmations.contains(&validator) {
            self.confirmations.push(validator);
        }
    }

    /// M-03 remediation: Check if the confirmation window has expired
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        now > self.confirmation_deadline
    }

    /// Get remaining time until deadline (in seconds)
    pub fn time_remaining(&self) -> i64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        (self.confirmation_deadline - now).max(0)
    }

    /// Check if slashing is confirmed (requires min confirmations AND not expired)
    ///
    /// M-03 remediation: Confirmations must arrive before deadline
    pub fn is_confirmed(&self, min_confirmations: usize) -> bool {
        !self.is_expired() && self.confirmations.len() >= min_confirmations
    }

    /// Confirm the slashing
    pub fn confirm(&mut self) {
        self.confirmed = true;
    }
}

/// Manager for tracking slashing events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlashingManager {
    /// Pending slashing events (awaiting confirmation)
    pending: Vec<SlashingEvent>,
    /// Confirmed slashing events
    confirmed: Vec<SlashingEvent>,
    /// Slashing events by validator
    by_validator: std::collections::HashMap<String, Vec<String>>,
    /// Minimum confirmations required
    min_confirmations: usize,
}

impl SlashingManager {
    /// Create a new slashing manager
    pub fn new(min_confirmations: usize) -> Self {
        Self {
            pending: Vec::new(),
            confirmed: Vec::new(),
            by_validator: std::collections::HashMap::new(),
            min_confirmations,
        }
    }

    /// Report an offense
    pub fn report_offense(&mut self, offense: SlashableOffense, reporter: String) -> &SlashingEvent {
        let event = SlashingEvent::new(offense, reporter);
        self.pending.push(event);
        self.pending.last().expect("event just pushed above")
    }

    /// Add confirmation to a pending event
    pub fn confirm_event(&mut self, event_id: &str, confirmer: String) -> bool {
        if let Some(event) = self.pending.iter_mut().find(|e| e.id == event_id) {
            event.add_confirmation(confirmer);

            if event.is_confirmed(self.min_confirmations) {
                event.confirm();

                // Track by validator
                let validator = event.offense.offending_validator().to_string();
                self.by_validator
                    .entry(validator)
                    .or_default()
                    .push(event.id.clone());

                return true;
            }
        }
        false
    }

    /// Move confirmed events from pending to confirmed, and remove expired events
    ///
    /// M-03 remediation: Expired events are discarded, not confirmed
    pub fn process_confirmations(&mut self) {
        let mut newly_confirmed = Vec::new();
        let mut still_pending = Vec::new();

        for event in self.pending.drain(..) {
            if event.confirmed {
                newly_confirmed.push(event);
            } else if event.is_expired() {
                // M-03: Discard expired events - they can no longer be confirmed
                // Optionally could log or track these for monitoring
            } else {
                still_pending.push(event);
            }
        }

        self.pending = still_pending;
        self.confirmed.extend(newly_confirmed);
    }

    /// Get count of expired (discarded) events since last check
    /// Useful for monitoring slashing effectiveness
    pub fn count_expired(&self) -> usize {
        self.pending.iter().filter(|e| e.is_expired()).count()
    }

    /// Get slashing events for a validator
    pub fn get_for_validator(&self, validator: &str) -> Vec<&SlashingEvent> {
        self.by_validator
            .get(validator)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.confirmed.iter().find(|e| &e.id == id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get total penalty for a validator
    pub fn total_penalty(&self, validator: &str) -> f32 {
        self.get_for_validator(validator)
            .iter()
            .map(|e| e.penalty)
            .sum()
    }

    /// Get pending events
    pub fn pending_events(&self) -> &[SlashingEvent] {
        &self.pending
    }

    /// Get confirmed events
    pub fn confirmed_events(&self) -> &[SlashingEvent] {
        &self.confirmed
    }
}

/// Detect double voting from a collection of votes
pub fn detect_double_voting(votes: &[Vote]) -> Vec<DoubleVoteEvidence> {
    let mut evidence = Vec::new();
    let mut seen: std::collections::HashMap<(&str, u64), &Vote> = std::collections::HashMap::new();

    for vote in votes {
        let key = (vote.voter.as_str(), vote.round);

        if let Some(prev_vote) = seen.get(&key) {
            if prev_vote.decision != vote.decision {
                if let Some(e) = DoubleVoteEvidence::new((*prev_vote).clone(), vote.clone()) {
                    evidence.push(e);
                }
            }
        } else {
            seen.insert(key, vote);
        }
    }

    evidence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vote::VoteDecision;

    #[test]
    fn test_double_vote_detection() {
        let vote1 = Vote::new(
            "prop-1".to_string(),
            1,
            "voter-1".to_string(),
            VoteDecision::Approve,
            0.8,
        );
        let vote2 = Vote::new(
            "prop-1".to_string(),
            1,
            "voter-1".to_string(),
            VoteDecision::Reject,
            0.8,
        );

        let evidence = detect_double_voting(&[vote1, vote2]);
        assert_eq!(evidence.len(), 1);
    }

    #[test]
    fn test_slashing_severity() {
        let offense = SlashableOffense::Inactivity {
            validator: "test".to_string(),
            missed_rounds: 3,
            total_rounds: 10,
        };
        assert_eq!(offense.severity(), SlashingSeverity::Minor);

        let offense2 = SlashableOffense::Inactivity {
            validator: "test".to_string(),
            missed_rounds: 8,
            total_rounds: 10,
        };
        assert_eq!(offense2.severity(), SlashingSeverity::Moderate);
    }

    #[test]
    fn test_slashing_manager() {
        let mut manager = SlashingManager::new(2);

        let offense = SlashableOffense::InvalidProposals {
            validator: "bad-actor".to_string(),
            invalid_count: 3,
            reasons: vec!["invalid hash".to_string()],
        };

        manager.report_offense(offense, "reporter-1".to_string());

        // First confirmation
        assert!(!manager.confirm_event(&manager.pending[0].id.clone(), "confirmer-1".to_string()));

        // Second confirmation (should confirm)
        let event_id = manager.pending[0].id.clone();
        assert!(manager.confirm_event(&event_id, "confirmer-2".to_string()));
    }
}
