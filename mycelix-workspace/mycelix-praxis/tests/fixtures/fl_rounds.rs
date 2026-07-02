// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Federated Learning round test fixtures
//
// Generators for FL rounds, updates, and related data structures

use super::{random_hex, seeded_rng, DEFAULT_SEED};
use praxis_core::types::{ModelHash, RoundId, RoundState};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlRound {
    pub round_id: RoundId,
    pub model_id: String,
    pub state: RoundState,
    pub current_participants: u32,
    pub min_participants: u32,
    pub max_participants: u32,
    pub aggregation_method: String,
    pub clip_norm: f32,
    pub current_model_hash: Option<ModelHash>,
    pub aggregated_model_hash: Option<ModelHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlUpdate {
    pub round_id: RoundId,
    pub model_id: String,
    pub parent_model_hash: ModelHash,
    pub grad_commitment: Vec<u8>,
    pub clipped_l2_norm: f32,
    pub local_val_loss: f32,
    pub sample_count: u32,
    pub timestamp: i64,
}

/// FL round builder for flexible test data creation
pub struct FlRoundBuilder {
    seed: u64,
    participant_count: u32,
    state: RoundState,
    aggregation_method: String,
    clip_norm: f32,
}

impl Default for FlRoundBuilder {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            participant_count: 10,
            state: RoundState::Join,
            aggregation_method: "trimmed_mean".to_string(),
            clip_norm: 1.0,
        }
    }
}

impl FlRoundBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn participants(mut self, count: u32) -> Self {
        self.participant_count = count;
        self
    }

    pub fn state(mut self, state: RoundState) -> Self {
        self.state = state;
        self
    }

    pub fn aggregation_method(mut self, method: impl Into<String>) -> Self {
        self.aggregation_method = method.into();
        self
    }

    pub fn clip_norm(mut self, norm: f32) -> Self {
        self.clip_norm = norm;
        self
    }

    pub fn build(self) -> FlRound {
        let mut rng = seeded_rng(self.seed);

        let round_id = RoundId(format!("fl-round-{}", random_hex(&mut rng, 8)));
        let model_id = format!("model-{}", random_hex(&mut rng, 8));

        let (current_model_hash, aggregated_model_hash) = match self.state {
            RoundState::Discover | RoundState::Join => (None, None),
            RoundState::Assign | RoundState::Update | RoundState::Aggregate => {
                (Some(ModelHash(random_hex(&mut rng, 32))), None)
            }
            RoundState::Release | RoundState::Completed => (
                Some(ModelHash(random_hex(&mut rng, 32))),
                Some(ModelHash(random_hex(&mut rng, 32))),
            ),
            RoundState::Failed => (Some(ModelHash(random_hex(&mut rng, 32))), None),
        };

        FlRound {
            round_id,
            model_id,
            state: self.state,
            current_participants: self.participant_count,
            min_participants: 5,
            max_participants: 100,
            aggregation_method: self.aggregation_method,
            clip_norm: self.clip_norm,
            current_model_hash,
            aggregated_model_hash,
        }
    }
}

/// FL update builder for flexible test data creation
pub struct FlUpdateBuilder {
    seed: u64,
    round_id: Option<RoundId>,
    local_val_loss: f32,
    sample_count: u32,
    clipped_l2_norm: f32,
}

impl Default for FlUpdateBuilder {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            round_id: None,
            local_val_loss: 0.5,
            sample_count: 100,
            clipped_l2_norm: 1.0,
        }
    }
}

impl FlUpdateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn round_id(mut self, round_id: RoundId) -> Self {
        self.round_id = Some(round_id);
        self
    }

    pub fn local_val_loss(mut self, loss: f32) -> Self {
        self.local_val_loss = loss;
        self
    }

    pub fn sample_count(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    pub fn clipped_l2_norm(mut self, norm: f32) -> Self {
        self.clipped_l2_norm = norm;
        self
    }

    pub fn build(self) -> FlUpdate {
        let mut rng = seeded_rng(self.seed);

        let round_id = self
            .round_id
            .unwrap_or_else(|| RoundId(format!("fl-round-{}", random_hex(&mut rng, 8))));
        let model_id = format!("model-{}", random_hex(&mut rng, 8));
        let parent_model_hash = ModelHash(random_hex(&mut rng, 32));

        // Generate random commitment (32 bytes)
        let grad_commitment: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();

        FlUpdate {
            round_id,
            model_id,
            parent_model_hash,
            grad_commitment,
            clipped_l2_norm: self.clipped_l2_norm,
            local_val_loss: self.local_val_loss,
            sample_count: self.sample_count,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}

/// Generate a random FL round with default settings
pub fn random() -> FlRound {
    FlRoundBuilder::new().build()
}

/// Generate an FL round with specific seed
pub fn with_seed(seed: u64) -> FlRound {
    FlRoundBuilder::new().seed(seed).build()
}

/// Generate an FL round with specific number of participants
pub fn with_participants(count: u32) -> FlRound {
    FlRoundBuilder::new().participants(count).build()
}

/// Generate an FL round in a specific state
pub fn in_state(state: RoundState) -> FlRound {
    FlRoundBuilder::new().state(state).build()
}

/// Generate a completed FL round
pub fn completed() -> FlRound {
    FlRoundBuilder::new()
        .state(RoundState::Completed)
        .participants(37)
        .build()
}

/// Generate an active FL round (in UPDATE phase)
pub fn active() -> FlRound {
    FlRoundBuilder::new()
        .state(RoundState::Update)
        .participants(25)
        .build()
}

/// Generate a random FL update
pub fn random_update() -> FlUpdate {
    FlUpdateBuilder::new().build()
}

/// Generate an FL update for a specific round
pub fn update_for_round(round_id: RoundId) -> FlUpdate {
    FlUpdateBuilder::new().round_id(round_id).build()
}

/// Generate multiple FL updates for a round
pub fn updates_for_round(round_id: RoundId, count: usize) -> Vec<FlUpdate> {
    (0..count)
        .map(|i| {
            FlUpdateBuilder::new()
                .round_id(round_id.clone())
                .seed(DEFAULT_SEED + i as u64)
                .build()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_round() {
        let round = random();
        assert!(!round.round_id.0.is_empty());
        assert!(round.current_participants > 0);
    }

    #[test]
    fn test_round_builder() {
        let round = FlRoundBuilder::new()
            .participants(50)
            .state(RoundState::Update)
            .aggregation_method("median")
            .clip_norm(2.0)
            .build();

        assert_eq!(round.current_participants, 50);
        assert_eq!(round.state, RoundState::Update);
        assert_eq!(round.aggregation_method, "median");
        assert_eq!(round.clip_norm, 2.0);
    }

    #[test]
    fn test_completed_round_has_hashes() {
        let round = completed();
        assert_eq!(round.state, RoundState::Completed);
        assert!(round.current_model_hash.is_some());
        assert!(round.aggregated_model_hash.is_some());
    }

    #[test]
    fn test_random_update() {
        let update = random_update();
        assert!(!update.round_id.0.is_empty());
        assert!(!update.grad_commitment.is_empty());
        assert!(update.sample_count > 0);
    }

    #[test]
    fn test_update_for_round() {
        let round_id = RoundId("test-round".to_string());
        let update = update_for_round(round_id.clone());

        assert_eq!(update.round_id, round_id);
    }

    #[test]
    fn test_multiple_updates() {
        let round_id = RoundId("test-round".to_string());
        let updates = updates_for_round(round_id.clone(), 10);

        assert_eq!(updates.len(), 10);
        assert!(updates.iter().all(|u| u.round_id == round_id));
        // Check they're different (different seeds)
        assert_ne!(updates[0].grad_commitment, updates[1].grad_commitment);
    }

    #[test]
    fn test_deterministic_generation() {
        let round1 = with_seed(42);
        let round2 = with_seed(42);

        assert_eq!(round1.round_id, round2.round_id);
        assert_eq!(round1.model_id, round2.model_id);
    }
}
