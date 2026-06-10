// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FL Coordinator
//!
//! Manages federated learning rounds and participant interactions.

use std::collections::HashMap;
use thiserror::Error;

use super::aggregation::{
    calculate_update_quality, coordinate_median, fedavg, krum, trimmed_mean,
    trust_weighted_aggregation, AggregationError,
};
use super::types::{
    AggregatedGradient, AggregationMethod, FLConfig, FLRound, GradientUpdate, Participant,
    RoundStatus,
};
use crate::matl::ProofOfGradientQuality;

/// FL Coordinator errors
#[derive(Debug, Error)]
pub enum CoordinatorError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Previous round not completed (status: {status:?}, round: {round_id})")]
    RoundNotCompleted { status: RoundStatus, round_id: u64 },
    #[error("No active round")]
    NoActiveRound,
    #[error("Not enough participants: need {required}, have {available}")]
    NotEnoughParticipants { required: usize, available: usize },
    #[error("Round timeout exceeded")]
    RoundTimeout,
    #[error("Aggregation error: {0}")]
    AggregationError(#[from] AggregationError),
    #[error("Participant not registered: {0}")]
    ParticipantNotRegistered(String),
    #[error("Participant below trust threshold: {0}")]
    ParticipantBelowThreshold(String),
    #[error("Model version mismatch: expected {expected}, got {actual}")]
    ModelVersionMismatch { expected: u64, actual: u64 },
}

/// Federated Learning Coordinator
///
/// Manages FL rounds, participant registration, and gradient aggregation.
pub struct FLCoordinator {
    /// Configuration
    config: FLConfig,
    /// Current active round
    current_round: Option<FLRound>,
    /// Completed round history
    round_history: Vec<FLRound>,
    /// Registered participants
    participants: HashMap<String, Participant>,
    /// Current model version
    model_version: u64,
}

impl FLCoordinator {
    /// Create a new FL coordinator
    pub fn new(config: FLConfig) -> Self {
        Self {
            config,
            current_round: None,
            round_history: Vec::new(),
            participants: HashMap::new(),
            model_version: 0,
        }
    }

    /// Create with validated config
    pub fn with_validated_config(config: FLConfig) -> Result<Self, CoordinatorError> {
        config
            .validate()
            .map_err(|e| CoordinatorError::ConfigError(e.to_string()))?;
        Ok(Self::new(config))
    }

    /// Register a participant
    pub fn register_participant(&mut self, id: String) -> &Participant {
        self.participants
            .entry(id.clone())
            .or_insert_with(|| Participant::new(id))
    }

    /// Unregister a participant
    pub fn unregister_participant(&mut self, id: &str) -> Option<Participant> {
        self.participants.remove(id)
    }

    /// Get participant
    pub fn get_participant(&self, id: &str) -> Option<&Participant> {
        self.participants.get(id)
    }

    /// Get mutable participant
    pub fn get_participant_mut(&mut self, id: &str) -> Option<&mut Participant> {
        self.participants.get_mut(id)
    }

    /// Start a new FL round
    pub fn start_round(&mut self) -> Result<&FLRound, CoordinatorError> {
        // Check if previous round is completed
        if let Some(ref round) = self.current_round {
            if round.status != RoundStatus::Completed && round.status != RoundStatus::Failed {
                return Err(CoordinatorError::RoundNotCompleted {
                    status: round.status,
                    round_id: round.round_id,
                });
            }
        }

        self.model_version += 1;

        let round = FLRound::new(
            self.round_history.len() as u64 + 1,
            self.model_version,
            self.participants.clone(),
        );

        self.current_round = Some(round);
        // Safe: we just set current_round to Some above
        Ok(self
            .current_round
            .as_ref()
            .expect("current_round should be Some after setting it"))
    }

    /// Submit a gradient update
    pub fn submit_update(&mut self, update: GradientUpdate) -> bool {
        let round = match &mut self.current_round {
            Some(r) if r.status == RoundStatus::Collecting => r,
            _ => return false,
        };

        // Validate model version
        if update.model_version != round.model_version {
            return false;
        }

        // Validate update
        if !update.is_valid() {
            return false;
        }

        // Validate participant
        let participant = match self.participants.get_mut(&update.participant_id) {
            Some(p) => p,
            None => return false,
        };

        // Check trust threshold
        if participant.trust_score() < self.config.trust_threshold {
            return false;
        }

        // Validate gradient consistency with existing updates
        if !round.updates.is_empty() {
            let expected_size = round.updates[0].gradients.len();
            if update.gradients.len() != expected_size {
                return false;
            }
        }

        // Create PoGQ for this update
        let quality = calculate_update_quality(&update);
        participant.pogq = Some(ProofOfGradientQuality::new(quality, 0.8, 0.1));
        participant.last_contribution = Some(update.clone());

        round.updates.push(update);

        // Check if we have max participants
        if round.updates.len() >= self.config.max_participants {
            // Auto-aggregate when max reached
            let _ = self.aggregate_round_internal();
        }

        true
    }

    /// Aggregate the current round
    pub fn aggregate_round(&mut self) -> Result<AggregatedGradient, CoordinatorError> {
        self.aggregate_round_internal()
    }

    fn aggregate_round_internal(&mut self) -> Result<AggregatedGradient, CoordinatorError> {
        // First, gather the information we need before borrowing mutably
        let (round_updates, model_version, update_count, participant_ids) = {
            let round = match &self.current_round {
                Some(r) if r.status == RoundStatus::Collecting => r,
                Some(r) if r.status == RoundStatus::Completed => {
                    return r
                        .aggregated_result
                        .clone()
                        .ok_or(CoordinatorError::NoActiveRound);
                }
                _ => return Err(CoordinatorError::NoActiveRound),
            };

            // Check minimum participants
            if round.updates.len() < self.config.min_participants {
                return Err(CoordinatorError::NotEnoughParticipants {
                    required: self.config.min_participants,
                    available: round.updates.len(),
                });
            }

            (
                round.updates.clone(),
                round.model_version,
                round.updates.len(),
                round
                    .updates
                    .iter()
                    .map(|u| u.participant_id.clone())
                    .collect::<Vec<_>>(),
            )
        };

        // Update round status
        if let Some(ref mut round) = self.current_round {
            round.status = RoundStatus::Aggregating;
        }

        // Perform aggregation based on configured method
        let aggregated_result = match self.config.aggregation_method {
            AggregationMethod::FedAvg => {
                let gradients = fedavg(&round_updates)?;
                AggregatedGradient::new(
                    gradients,
                    model_version,
                    update_count,
                    0,
                    AggregationMethod::FedAvg,
                )
            }
            AggregationMethod::TrimmedMean => {
                let gradients = trimmed_mean(&round_updates, self.config.byzantine_tolerance)?;
                AggregatedGradient::new(
                    gradients,
                    model_version,
                    update_count,
                    0,
                    AggregationMethod::TrimmedMean,
                )
            }
            AggregationMethod::Median => {
                let gradients = coordinate_median(&round_updates)?;
                AggregatedGradient::new(
                    gradients,
                    model_version,
                    update_count,
                    0,
                    AggregationMethod::Median,
                )
            }
            AggregationMethod::Krum => {
                let gradients = krum(&round_updates, 1)?;
                AggregatedGradient::new(
                    gradients,
                    model_version,
                    update_count,
                    0,
                    AggregationMethod::Krum,
                )
            }
            AggregationMethod::TrustWeighted => trust_weighted_aggregation(
                &round_updates,
                &self.participants,
                self.config.trust_threshold,
            )?,
        };

        // Update participant reputations
        for participant_id in &participant_ids {
            if let Some(participant) = self.participants.get_mut(participant_id) {
                participant.record_positive();
            }
        }

        // Complete the round
        if let Some(ref mut round) = self.current_round {
            round.aggregated_result = Some(aggregated_result.clone());
            round.status = RoundStatus::Completed;
            round.end_time = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            );
        }

        // Move to history
        if let Some(completed_round) = self.current_round.take() {
            self.round_history.push(completed_round);
        }

        Ok(aggregated_result)
    }

    /// Cancel the current round
    pub fn cancel_round(&mut self) {
        if let Some(ref mut round) = self.current_round {
            round.status = RoundStatus::Failed;
            round.end_time = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            );
        }
    }

    /// Get round statistics
    pub fn get_round_stats(&self) -> RoundStats {
        let total_participation: usize = self.round_history.iter().map(|r| r.updates.len()).sum();

        RoundStats {
            total_rounds: self.round_history.len(),
            current_round: self.current_round.as_ref().map(|r| r.round_id),
            participant_count: self.participants.len(),
            average_participation: if self.round_history.is_empty() {
                0.0
            } else {
                total_participation as f64 / self.round_history.len() as f64
            },
            model_version: self.model_version,
        }
    }

    /// Get current round
    pub fn current_round(&self) -> Option<&FLRound> {
        self.current_round.as_ref()
    }

    /// Get round history
    pub fn round_history(&self) -> &[FLRound] {
        &self.round_history
    }

    /// Get configuration
    pub fn config(&self) -> &FLConfig {
        &self.config
    }

    /// Get all participants
    pub fn participants(&self) -> &HashMap<String, Participant> {
        &self.participants
    }

    /// Get current model version
    pub fn model_version(&self) -> u64 {
        self.model_version
    }
}

/// Round statistics
#[derive(Debug, Clone)]
pub struct RoundStats {
    /// Total completed rounds
    pub total_rounds: usize,
    /// Current round ID (if active)
    pub current_round: Option<u64>,
    /// Number of registered participants
    pub participant_count: usize,
    /// Average updates per round
    pub average_participation: f64,
    /// Current model version
    pub model_version: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_coordinator() -> FLCoordinator {
        let config = FLConfig {
            min_participants: 2,
            max_participants: 10,
            round_timeout_ms: 60_000,
            byzantine_tolerance: 0.33,
            aggregation_method: AggregationMethod::FedAvg,
            trust_threshold: 0.3,
        };

        FLCoordinator::new(config)
    }

    #[test]
    fn test_participant_registration() {
        let mut coordinator = create_test_coordinator();

        coordinator.register_participant("p1".to_string());
        coordinator.register_participant("p2".to_string());

        assert_eq!(coordinator.participants().len(), 2);
        assert!(coordinator.get_participant("p1").is_some());
        assert!(coordinator.get_participant("p3").is_none());
    }

    #[test]
    fn test_start_round() {
        let mut coordinator = create_test_coordinator();
        coordinator.register_participant("p1".to_string());

        let result = coordinator.start_round();
        assert!(result.is_ok());
        assert!(coordinator.current_round().is_some());
        assert_eq!(coordinator.model_version(), 1);
    }

    #[test]
    fn test_submit_update() {
        let mut coordinator = create_test_coordinator();
        coordinator.register_participant("p1".to_string());
        coordinator.start_round().expect("Failed to start round");

        let update = GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2], 100, 0.5);
        assert!(coordinator.submit_update(update));
    }

    #[test]
    fn test_submit_update_unregistered() {
        let mut coordinator = create_test_coordinator();
        coordinator.register_participant("p1".to_string());
        coordinator.start_round().expect("Failed to start round");

        // Unregistered participant
        let update = GradientUpdate::new("unknown".to_string(), 1, vec![0.1, 0.2], 100, 0.5);
        assert!(!coordinator.submit_update(update));
    }

    #[test]
    fn test_full_round() {
        let mut coordinator = create_test_coordinator();
        coordinator.register_participant("p1".to_string());
        coordinator.register_participant("p2".to_string());

        coordinator.start_round().expect("Failed to start round");

        let update1 = GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2], 100, 0.5);
        let update2 = GradientUpdate::new("p2".to_string(), 1, vec![0.2, 0.3], 100, 0.4);

        assert!(coordinator.submit_update(update1));
        assert!(coordinator.submit_update(update2));

        let result = coordinator.aggregate_round();
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert_eq!(aggregated.participant_count, 2);
    }

    #[test]
    fn test_round_stats() {
        let mut coordinator = create_test_coordinator();
        coordinator.register_participant("p1".to_string());
        coordinator.register_participant("p2".to_string());

        // Complete one round
        coordinator.start_round().unwrap();
        coordinator.submit_update(GradientUpdate::new(
            "p1".to_string(),
            1,
            vec![0.1],
            100,
            0.5,
        ));
        coordinator.submit_update(GradientUpdate::new(
            "p2".to_string(),
            1,
            vec![0.2],
            100,
            0.4,
        ));
        coordinator.aggregate_round().unwrap();

        let stats = coordinator.get_round_stats();
        assert_eq!(stats.total_rounds, 1);
        assert_eq!(stats.participant_count, 2);
        assert!((stats.average_participation - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_different_aggregation_methods() {
        let methods = vec![
            AggregationMethod::FedAvg,
            AggregationMethod::TrimmedMean,
            AggregationMethod::Median,
            AggregationMethod::TrustWeighted,
        ];

        for method in methods {
            let config = FLConfig {
                min_participants: 2,
                aggregation_method: method,
                ..Default::default()
            };

            let mut coordinator = FLCoordinator::new(config);
            coordinator.register_participant("p1".to_string());
            coordinator.register_participant("p2".to_string());
            coordinator.start_round().unwrap();

            coordinator.submit_update(GradientUpdate::new(
                "p1".to_string(),
                1,
                vec![0.1, 0.2],
                100,
                0.5,
            ));
            coordinator.submit_update(GradientUpdate::new(
                "p2".to_string(),
                1,
                vec![0.2, 0.3],
                100,
                0.4,
            ));

            let result = coordinator.aggregate_round();
            assert!(result.is_ok(), "Aggregation failed for method {:?}", method);
        }
    }
}
