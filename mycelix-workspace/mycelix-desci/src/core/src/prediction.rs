// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prediction Markets for Epistemic Claims
//!
//! Implements futarchy-style markets for claim verification.
//! Allows reputation-backed predictions on claim outcomes,
//! creating economic incentives for epistemic accuracy.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Type of prediction market
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketType {
    /// Binary outcome (true/false)
    Binary,
    /// Scalar value (confidence 0-100%)
    Scalar,
    /// Categorical (multiple discrete outcomes)
    Categorical,
    /// Conditional (outcome depends on another claim)
    Conditional,
}

/// State of a prediction market
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    /// Market is open for predictions
    Open,
    /// Predictions locked, awaiting resolution
    Locked,
    /// Market resolved with outcome
    Resolved,
    /// Market cancelled (claim withdrawn, etc.)
    Cancelled,
    /// Disputed resolution
    Disputed,
}

/// A position in a prediction market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Unique position ID
    pub id: Uuid,
    /// Participant who holds this position
    pub participant_id: String,
    /// The prediction (0.0-1.0 for probability)
    pub prediction: f64,
    /// Amount of reputation staked
    pub stake: f64,
    /// When the position was taken
    pub created_at: DateTime<Utc>,
    /// Whether position has been settled
    pub settled: bool,
    /// Payout received (after settlement)
    pub payout: Option<f64>,
}

/// Resolution details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    /// The actual outcome (0.0-1.0)
    pub outcome: f64,
    /// Who resolved the market
    pub resolver_id: String,
    /// Resolution method
    pub method: ResolutionMethod,
    /// Evidence supporting resolution
    pub evidence_refs: Vec<String>,
    /// When resolved
    pub resolved_at: DateTime<Utc>,
}

/// How a market was resolved
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionMethod {
    /// Resolved by oracle (trusted third party)
    Oracle,
    /// Resolved by consensus (multiple verifiers)
    Consensus,
    /// Resolved by smart contract condition
    Automatic,
    /// Resolved by dispute resolution
    DisputeResolution,
    /// Time-based resolution (prediction deadline)
    TimeBased,
}

/// A prediction market for a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMarket {
    /// Unique market ID
    pub id: Uuid,
    /// ID of the claim this market is about
    pub claim_id: Uuid,
    /// Market type
    pub market_type: MarketType,
    /// Current market state
    pub state: MarketState,
    /// Human-readable question
    pub question: String,
    /// When the market closes for new predictions
    pub close_time: DateTime<Utc>,
    /// When the market should be resolved
    pub resolution_time: DateTime<Utc>,
    /// All positions in the market
    pub positions: Vec<Position>,
    /// Current aggregate probability (market price)
    pub current_probability: f64,
    /// Total reputation staked
    pub total_stake: f64,
    /// Liquidity parameter for LMSR
    pub liquidity: f64,
    /// Resolution (if resolved)
    pub resolution: Option<Resolution>,
    /// Minimum stake amount
    pub min_stake: f64,
    /// Maximum stake per participant
    pub max_stake_per_participant: f64,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Creator ID
    pub creator_id: String,
}

impl PredictionMarket {
    /// Create a new binary prediction market
    pub fn binary(
        claim_id: Uuid,
        question: String,
        close_time: DateTime<Utc>,
        resolution_time: DateTime<Utc>,
        creator_id: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            claim_id,
            market_type: MarketType::Binary,
            state: MarketState::Open,
            question,
            close_time,
            resolution_time,
            positions: Vec::new(),
            current_probability: 0.5, // Start at 50%
            total_stake: 0.0,
            liquidity: 100.0, // Default liquidity
            resolution: None,
            min_stake: 1.0,
            max_stake_per_participant: 1000.0,
            created_at: Utc::now(),
            creator_id,
        }
    }

    /// Create a short-term verification market
    pub fn verification_market(claim_id: Uuid, creator_id: String) -> Self {
        let now = Utc::now();
        Self::binary(
            claim_id,
            "Will this claim be verified as accurate?".to_string(),
            now + Duration::days(7),
            now + Duration::days(30),
            creator_id,
        )
    }

    /// Place a prediction (buy shares)
    pub fn place_prediction(
        &mut self,
        participant_id: String,
        prediction: f64,
        stake: f64,
    ) -> Result<Position, MarketError> {
        // Validate market state
        if self.state != MarketState::Open {
            return Err(MarketError::MarketNotOpen);
        }

        if Utc::now() > self.close_time {
            return Err(MarketError::MarketClosed);
        }

        // Validate prediction
        if prediction < 0.0 || prediction > 1.0 {
            return Err(MarketError::InvalidPrediction);
        }

        // Validate stake
        if stake < self.min_stake {
            return Err(MarketError::StakeTooLow);
        }

        let participant_total: f64 = self
            .positions
            .iter()
            .filter(|p| p.participant_id == participant_id)
            .map(|p| p.stake)
            .sum();

        if participant_total + stake > self.max_stake_per_participant {
            return Err(MarketError::StakeTooHigh);
        }

        // Create position
        let position = Position {
            id: Uuid::new_v4(),
            participant_id,
            prediction,
            stake,
            created_at: Utc::now(),
            settled: false,
            payout: None,
        };

        // Update market probability using stake-weighted average
        let old_weight = self.total_stake;
        let new_weight = old_weight + stake;

        if new_weight > 0.0 {
            self.current_probability =
                (self.current_probability * old_weight + prediction * stake) / new_weight;
        }

        self.total_stake = new_weight;
        self.positions.push(position.clone());

        Ok(position)
    }

    /// Lock the market (no more predictions)
    pub fn lock(&mut self) -> Result<(), MarketError> {
        if self.state != MarketState::Open {
            return Err(MarketError::InvalidStateTransition);
        }
        self.state = MarketState::Locked;
        Ok(())
    }

    /// Resolve the market with an outcome
    pub fn resolve(
        &mut self,
        outcome: f64,
        resolver_id: String,
        method: ResolutionMethod,
        evidence_refs: Vec<String>,
    ) -> Result<Vec<Settlement>, MarketError> {
        if self.state != MarketState::Locked && self.state != MarketState::Open {
            return Err(MarketError::InvalidStateTransition);
        }

        if outcome < 0.0 || outcome > 1.0 {
            return Err(MarketError::InvalidOutcome);
        }

        self.resolution = Some(Resolution {
            outcome,
            resolver_id,
            method,
            evidence_refs,
            resolved_at: Utc::now(),
        });

        self.state = MarketState::Resolved;

        // Calculate settlements
        let settlements = self.calculate_settlements(outcome);

        // Mark positions as settled
        for position in &mut self.positions {
            position.settled = true;
            position.payout = settlements
                .iter()
                .find(|s| s.position_id == position.id)
                .map(|s| s.payout);
        }

        Ok(settlements)
    }

    /// Calculate payouts using Brier scoring rule
    fn calculate_settlements(&self, outcome: f64) -> Vec<Settlement> {
        let mut settlements = Vec::new();

        for position in &self.positions {
            // Brier score: lower is better, we invert for payout
            // Score = 1 - (prediction - outcome)^2
            let accuracy = 1.0 - (position.prediction - outcome).powi(2);

            // Payout = stake * (1 + accuracy) / 2
            // This ranges from 0x to 1x stake depending on accuracy
            let payout = position.stake * accuracy;

            settlements.push(Settlement {
                position_id: position.id,
                participant_id: position.participant_id.clone(),
                stake: position.stake,
                prediction: position.prediction,
                outcome,
                accuracy,
                payout,
            });
        }

        settlements
    }

    /// Get current market price (aggregate probability)
    pub fn market_price(&self) -> f64 {
        self.current_probability
    }

    /// Get participant's total stake
    pub fn participant_stake(&self, participant_id: &str) -> f64 {
        self.positions
            .iter()
            .filter(|p| p.participant_id == participant_id)
            .map(|p| p.stake)
            .sum()
    }

    /// Check if market should auto-lock
    pub fn should_auto_lock(&self) -> bool {
        self.state == MarketState::Open && Utc::now() > self.close_time
    }

    /// Get summary statistics
    pub fn summary(&self) -> MarketSummary {
        MarketSummary {
            market_id: self.id,
            claim_id: self.claim_id,
            state: self.state,
            current_probability: self.current_probability,
            total_stake: self.total_stake,
            num_participants: self.positions.iter()
                .map(|p| &p.participant_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            num_positions: self.positions.len(),
            time_remaining: if self.close_time > Utc::now() {
                Some(self.close_time - Utc::now())
            } else {
                None
            },
        }
    }
}

/// Settlement result for a position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settlement {
    pub position_id: Uuid,
    pub participant_id: String,
    pub stake: f64,
    pub prediction: f64,
    pub outcome: f64,
    pub accuracy: f64,
    pub payout: f64,
}

/// Market summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSummary {
    pub market_id: Uuid,
    pub claim_id: Uuid,
    pub state: MarketState,
    pub current_probability: f64,
    pub total_stake: f64,
    pub num_participants: usize,
    pub num_positions: usize,
    pub time_remaining: Option<Duration>,
}

/// Market operation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketError {
    MarketNotOpen,
    MarketClosed,
    InvalidPrediction,
    StakeTooLow,
    StakeTooHigh,
    InvalidStateTransition,
    InvalidOutcome,
    NotFound,
}

impl std::fmt::Display for MarketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MarketNotOpen => write!(f, "Market is not open for predictions"),
            Self::MarketClosed => write!(f, "Market has closed"),
            Self::InvalidPrediction => write!(f, "Prediction must be between 0 and 1"),
            Self::StakeTooLow => write!(f, "Stake is below minimum"),
            Self::StakeTooHigh => write!(f, "Stake exceeds maximum per participant"),
            Self::InvalidStateTransition => write!(f, "Invalid state transition"),
            Self::InvalidOutcome => write!(f, "Outcome must be between 0 and 1"),
            Self::NotFound => write!(f, "Market not found"),
        }
    }
}

impl std::error::Error for MarketError {}

/// Registry of prediction markets
#[derive(Debug, Clone, Default)]
pub struct PredictionMarketRegistry {
    markets: HashMap<Uuid, PredictionMarket>,
    claim_markets: HashMap<Uuid, Vec<Uuid>>,
}

impl PredictionMarketRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create and register a new market
    pub fn create_market(&mut self, market: PredictionMarket) -> Uuid {
        let market_id = market.id;
        let claim_id = market.claim_id;

        self.markets.insert(market_id, market);
        self.claim_markets
            .entry(claim_id)
            .or_default()
            .push(market_id);

        market_id
    }

    /// Get a market by ID
    pub fn get(&self, market_id: Uuid) -> Option<&PredictionMarket> {
        self.markets.get(&market_id)
    }

    /// Get mutable market reference
    pub fn get_mut(&mut self, market_id: Uuid) -> Option<&mut PredictionMarket> {
        self.markets.get_mut(&market_id)
    }

    /// Get all markets for a claim
    pub fn markets_for_claim(&self, claim_id: Uuid) -> Vec<&PredictionMarket> {
        self.claim_markets
            .get(&claim_id)
            .map(|ids| ids.iter().filter_map(|id| self.markets.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all open markets
    pub fn open_markets(&self) -> Vec<&PredictionMarket> {
        self.markets
            .values()
            .filter(|m| m.state == MarketState::Open)
            .collect()
    }

    /// Auto-lock expired markets
    pub fn auto_lock_expired(&mut self) {
        let to_lock: Vec<Uuid> = self
            .markets
            .values()
            .filter(|m| m.should_auto_lock())
            .map(|m| m.id)
            .collect();

        for id in to_lock {
            if let Some(market) = self.markets.get_mut(&id) {
                let _ = market.lock();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_market() {
        let claim_id = Uuid::new_v4();
        let market = PredictionMarket::verification_market(claim_id, "creator@test.com".to_string());

        assert_eq!(market.claim_id, claim_id);
        assert_eq!(market.state, MarketState::Open);
        assert_eq!(market.current_probability, 0.5);
    }

    #[test]
    fn test_place_prediction() {
        let claim_id = Uuid::new_v4();
        let mut market = PredictionMarket::verification_market(claim_id, "creator@test.com".to_string());

        let result = market.place_prediction("alice@test.com".to_string(), 0.8, 10.0);
        assert!(result.is_ok());

        let position = result.unwrap();
        assert_eq!(position.prediction, 0.8);
        assert_eq!(position.stake, 10.0);

        // Market price should move toward prediction
        assert!(market.current_probability > 0.5);
    }

    #[test]
    fn test_resolve_market() {
        let claim_id = Uuid::new_v4();
        let mut market = PredictionMarket::verification_market(claim_id, "creator@test.com".to_string());

        // Place predictions
        market.place_prediction("alice@test.com".to_string(), 0.9, 10.0).unwrap();
        market.place_prediction("bob@test.com".to_string(), 0.3, 10.0).unwrap();

        // Lock and resolve
        market.lock().unwrap();
        let settlements = market.resolve(
            1.0, // Claim was verified as true
            "oracle@test.com".to_string(),
            ResolutionMethod::Oracle,
            vec!["evidence.pdf".to_string()],
        ).unwrap();

        assert_eq!(settlements.len(), 2);
        assert_eq!(market.state, MarketState::Resolved);

        // Alice (predicted 0.9) should do better than Bob (predicted 0.3)
        let alice_settlement = settlements.iter().find(|s| s.participant_id == "alice@test.com").unwrap();
        let bob_settlement = settlements.iter().find(|s| s.participant_id == "bob@test.com").unwrap();

        assert!(alice_settlement.payout > bob_settlement.payout);
    }

    #[test]
    fn test_brier_scoring() {
        // Perfect prediction
        let perfect_accuracy = 1.0 - (1.0 - 1.0_f64).powi(2);
        assert_eq!(perfect_accuracy, 1.0);

        // Worst prediction
        let worst_accuracy = 1.0 - (0.0 - 1.0_f64).powi(2);
        assert_eq!(worst_accuracy, 0.0);

        // Neutral prediction
        let neutral_accuracy = 1.0 - (0.5 - 1.0_f64).powi(2);
        assert_eq!(neutral_accuracy, 0.75);
    }

    #[test]
    fn test_market_registry() {
        let mut registry = PredictionMarketRegistry::new();
        let claim_id = Uuid::new_v4();

        let market = PredictionMarket::verification_market(claim_id, "creator@test.com".to_string());
        let market_id = registry.create_market(market);

        assert!(registry.get(market_id).is_some());
        assert_eq!(registry.markets_for_claim(claim_id).len(), 1);
    }

    #[test]
    fn test_stake_limits() {
        let claim_id = Uuid::new_v4();
        let mut market = PredictionMarket::verification_market(claim_id, "creator@test.com".to_string());

        // Stake too low
        let result = market.place_prediction("alice@test.com".to_string(), 0.5, 0.1);
        assert_eq!(result.unwrap_err(), MarketError::StakeTooLow);

        // Stake too high
        let result = market.place_prediction("bob@test.com".to_string(), 0.5, 10000.0);
        assert_eq!(result.unwrap_err(), MarketError::StakeTooHigh);
    }
}
