// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Round state machine and gradient collection.
//!
//! Each FL round transitions through: Pending -> Collecting -> Aggregating -> Complete/Failed/TimedOut.

use std::collections::HashSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::FlError;
use crate::types::{AggregationResult, Gradient};

/// Status of an FL round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RoundStatus {
    /// Round created but not yet started.
    Pending,
    /// Actively collecting gradients from nodes.
    Collecting,
    /// Running aggregation on collected gradients.
    Aggregating,
    /// Round completed successfully.
    Complete,
    /// Round failed with an error message.
    Failed(String),
    /// Round timed out before enough gradients were collected.
    TimedOut,
}

impl std::fmt::Display for RoundStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoundStatus::Pending => write!(f, "Pending"),
            RoundStatus::Collecting => write!(f, "Collecting"),
            RoundStatus::Aggregating => write!(f, "Aggregating"),
            RoundStatus::Complete => write!(f, "Complete"),
            RoundStatus::Failed(msg) => write!(f, "Failed({})", msg),
            RoundStatus::TimedOut => write!(f, "TimedOut"),
        }
    }
}

/// A single FL round, collecting gradients and tracking state.
pub struct Round {
    /// Round number (1-indexed).
    pub round_number: u64,

    /// Current status of this round.
    pub status: RoundStatus,

    /// Collected gradients from nodes.
    pub gradients: Vec<Gradient>,

    /// Set of node IDs that have submitted this round.
    pub submitted_nodes: HashSet<String>,

    /// When the round started collecting gradients.
    pub started_at: Option<Instant>,

    /// When the round completed (or failed/timed out).
    pub completed_at: Option<Instant>,

    /// Aggregation result, if the round completed successfully.
    pub result: Option<AggregationResult>,
}

impl Round {
    /// Create a new round in Pending state.
    pub fn new(round_number: u64) -> Self {
        Self {
            round_number,
            status: RoundStatus::Pending,
            gradients: Vec::new(),
            submitted_nodes: HashSet::new(),
            started_at: None,
            completed_at: None,
            result: None,
        }
    }

    /// Transition to the Collecting state.
    pub fn start(&mut self) {
        self.status = RoundStatus::Collecting;
        self.started_at = Some(Instant::now());
    }

    /// Submit a gradient from a node.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::InvalidRoundState`] if the round is not in Collecting state.
    /// Returns [`FlError::DuplicateSubmission`] if the node already submitted this round.
    pub fn submit_gradient(&mut self, gradient: Gradient) -> Result<(), FlError> {
        if self.status != RoundStatus::Collecting {
            return Err(FlError::InvalidRoundState {
                expected: "Collecting".into(),
                got: self.status.to_string(),
            });
        }

        if self.submitted_nodes.contains(&gradient.node_id) {
            return Err(FlError::DuplicateSubmission(
                gradient.node_id.clone(),
                self.round_number,
            ));
        }

        self.submitted_nodes.insert(gradient.node_id.clone());
        self.gradients.push(gradient);
        Ok(())
    }

    /// Check if enough nodes have submitted gradients.
    pub fn has_enough_nodes(&self, min_nodes: usize) -> bool {
        self.gradients.len() >= min_nodes
    }

    /// Mark the round as complete with an aggregation result.
    pub fn complete(&mut self, result: AggregationResult) {
        self.status = RoundStatus::Complete;
        self.completed_at = Some(Instant::now());
        self.result = Some(result);
    }

    /// Mark the round as failed with a reason.
    pub fn fail(&mut self, reason: String) {
        self.status = RoundStatus::Failed(reason);
        self.completed_at = Some(Instant::now());
    }

    /// Mark the round as timed out.
    pub fn timeout(&mut self) {
        self.status = RoundStatus::TimedOut;
        self.completed_at = Some(Instant::now());
    }

    /// Duration of this round in milliseconds, if both start and end times are available.
    pub fn duration_ms(&self) -> Option<u64> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start).as_millis() as u64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient(node_id: &str, round: u64) -> Gradient {
        Gradient::new(node_id, vec![1.0, 2.0, 3.0], round)
    }

    #[test]
    fn test_round_new_is_pending() {
        let round = Round::new(1);
        assert_eq!(round.status, RoundStatus::Pending);
        assert_eq!(round.round_number, 1);
        assert!(round.gradients.is_empty());
        assert!(round.started_at.is_none());
    }

    #[test]
    fn test_round_start_transitions_to_collecting() {
        let mut round = Round::new(1);
        round.start();
        assert_eq!(round.status, RoundStatus::Collecting);
        assert!(round.started_at.is_some());
    }

    #[test]
    fn test_submit_gradient_stored() {
        let mut round = Round::new(1);
        round.start();
        round
            .submit_gradient(make_gradient("node-1", 1))
            .unwrap();
        assert_eq!(round.gradients.len(), 1);
        assert!(round.submitted_nodes.contains("node-1"));
    }

    #[test]
    fn test_submit_gradient_pending_rejected() {
        let mut round = Round::new(1);
        // Round is still Pending, not Collecting
        let result = round.submit_gradient(make_gradient("node-1", 1));
        assert!(matches!(result, Err(FlError::InvalidRoundState { .. })));
    }

    #[test]
    fn test_duplicate_submission_rejected() {
        let mut round = Round::new(1);
        round.start();
        round
            .submit_gradient(make_gradient("node-1", 1))
            .unwrap();
        let result = round.submit_gradient(make_gradient("node-1", 1));
        assert!(matches!(result, Err(FlError::DuplicateSubmission(_, _))));
    }

    #[test]
    fn test_has_enough_nodes() {
        let mut round = Round::new(1);
        round.start();
        assert!(!round.has_enough_nodes(3));

        round
            .submit_gradient(make_gradient("a", 1))
            .unwrap();
        round
            .submit_gradient(make_gradient("b", 1))
            .unwrap();
        assert!(!round.has_enough_nodes(3));

        round
            .submit_gradient(make_gradient("c", 1))
            .unwrap();
        assert!(round.has_enough_nodes(3));
    }

    #[test]
    fn test_round_complete() {
        let mut round = Round::new(1);
        round.start();
        let result = AggregationResult {
            gradient: vec![1.0, 2.0],
            included_nodes: vec!["a".into()],
            excluded_nodes: vec![],
            scores: vec![],
        };
        round.complete(result);
        assert_eq!(round.status, RoundStatus::Complete);
        assert!(round.completed_at.is_some());
        assert!(round.result.is_some());
    }

    #[test]
    fn test_round_fail() {
        let mut round = Round::new(1);
        round.start();
        round.fail("not enough nodes".into());
        assert!(matches!(round.status, RoundStatus::Failed(_)));
        assert!(round.completed_at.is_some());
    }

    #[test]
    fn test_round_timeout() {
        let mut round = Round::new(1);
        round.start();
        round.timeout();
        assert_eq!(round.status, RoundStatus::TimedOut);
        assert!(round.completed_at.is_some());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(RoundStatus::Pending.to_string(), "Pending");
        assert_eq!(RoundStatus::Collecting.to_string(), "Collecting");
        assert_eq!(RoundStatus::Complete.to_string(), "Complete");
        assert_eq!(RoundStatus::TimedOut.to_string(), "TimedOut");
    }

    #[test]
    fn test_status_serde_roundtrip() {
        let statuses = vec![
            RoundStatus::Pending,
            RoundStatus::Collecting,
            RoundStatus::Aggregating,
            RoundStatus::Complete,
            RoundStatus::Failed("err".into()),
            RoundStatus::TimedOut,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let s2: RoundStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, &s2);
        }
    }
}
