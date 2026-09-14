// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded-rational policy primitives over declared expected action utilities.
//!
//! A local logit choice rule is intentionally **not** called a quantal-response
//! equilibrium. QRE requires a fixed point between all players' mixed strategies
//! and their quantal responses. This module provides the lower-level validated
//! distributions, independent product beliefs, expected action values, exact
//! best-response policy and one-step logit choice policy.

use crate::NPlayerGame;

/// Absolute tolerance accepted for probability-mass normalization checks.
///
/// Inputs are not silently renormalized: their sum must already be within this
/// tolerance of one.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-12;

/// Validated finite probability distribution over one player's actions.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionDistribution {
    probabilities: Vec<f64>,
}

impl ActionDistribution {
    /// Validate a probability vector without silently renormalizing it.
    pub fn new(probabilities: Vec<f64>) -> Result<Self, String> {
        if probabilities.is_empty() {
            return Err("action distribution must be non-empty".to_string());
        }
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        {
            return Err("action probabilities must be finite and non-negative".to_string());
        }

        let sum = probabilities.iter().sum::<f64>();
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err("action probabilities must sum to one within declared tolerance".to_string());
        }

        Ok(Self { probabilities })
    }

    /// Uniform distribution over `actions` actions.
    pub fn uniform(actions: usize) -> Result<Self, String> {
        if actions == 0 {
            return Err("uniform distribution requires at least one action".to_string());
        }
        let probability = 1.0 / actions as f64;
        Self::new(vec![probability; actions])
    }

    /// Probability mass in action-index order.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Number of actions in the distribution.
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    /// Whether this distribution has no actions. Always false for validated values.
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }
}

/// Independent mixed-strategy beliefs, one validated distribution per player.
///
/// This is explicitly a product-distribution model. Correlated strategies need
/// a distinct future type rather than overloading this representation.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedStrategyProfile {
    strategy_counts: Vec<usize>,
    distributions: Vec<ActionDistribution>,
}

impl MixedStrategyProfile {
    /// Bind one action distribution to every player in a declared game shape.
    pub fn new(
        game: &NPlayerGame,
        distributions: Vec<ActionDistribution>,
    ) -> Result<Self, String> {
        if distributions.len() != game.players() {
            return Err("mixed-strategy profile must contain one distribution per player".to_string());
        }
        for (player, distribution) in distributions.iter().enumerate() {
            if distribution.len() != game.strategy_counts()[player] {
                return Err("mixed-strategy action count does not match game strategy count".to_string());
            }
        }

        Ok(Self {
            strategy_counts: game.strategy_counts().to_vec(),
            distributions,
        })
    }

    /// Player distributions in player-index order.
    pub fn distributions(&self) -> &[ActionDistribution] {
        &self.distributions
    }

    fn validate_for_game(&self, game: &NPlayerGame) -> Result<(), String> {
        if self.strategy_counts != game.strategy_counts() {
            return Err("mixed-strategy profile shape does not match game".to_string());
        }
        Ok(())
    }
}

/// Behavior policy that maps declared expected action utilities to a choice distribution.
pub trait BehaviorPolicy {
    fn action_distribution(&self, action_values: &[f64]) -> Result<ActionDistribution, String>;
}

/// Exact best response with explicit tie tolerance and equal mass over ties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestResponse {
    tie_tolerance: f64,
}

impl BestResponse {
    pub fn new(tie_tolerance: f64) -> Result<Self, String> {
        if !tie_tolerance.is_finite() || tie_tolerance < 0.0 {
            return Err("best-response tie tolerance must be finite and non-negative".to_string());
        }
        Ok(Self { tie_tolerance })
    }

    pub fn tie_tolerance(&self) -> f64 {
        self.tie_tolerance
    }
}

impl BehaviorPolicy for BestResponse {
    fn action_distribution(&self, action_values: &[f64]) -> Result<ActionDistribution, String> {
        validate_action_values(action_values)?;
        let maximum = action_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut tied = Vec::new();
        for (action, value) in action_values.iter().copied().enumerate() {
            let gap = maximum - value;
            if value == maximum || (gap.is_finite() && gap <= self.tie_tolerance) {
                tied.push(action);
            }
        }

        let tie_probability = 1.0 / tied.len() as f64;
        let mut probabilities = vec![0.0; action_values.len()];
        for action in tied {
            probabilities[action] = tie_probability;
        }
        ActionDistribution::new(probabilities)
    }
}

/// One-step logit/quantal choice rule over expected action utilities.
///
/// This type is deliberately not named QRE: it does not solve a strategic
/// fixed point. `precision = 0` yields a uniform distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitChoice {
    precision: f64,
}

impl LogitChoice {
    pub fn new(precision: f64) -> Result<Self, String> {
        if !precision.is_finite() || precision < 0.0 {
            return Err("logit precision must be finite and non-negative".to_string());
        }
        Ok(Self { precision })
    }

    pub fn precision(&self) -> f64 {
        self.precision
    }
}

impl BehaviorPolicy for LogitChoice {
    fn action_distribution(&self, action_values: &[f64]) -> Result<ActionDistribution, String> {
        validate_action_values(action_values)?;
        if self.precision == 0.0 {
            return ActionDistribution::uniform(action_values.len());
        }

        let maximum = action_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights = Vec::with_capacity(action_values.len());
        let mut weight_sum = 0.0f64;

        for value in action_values {
            let shifted = *value - maximum;
            let exponent = self.precision * shifted;
            let weight = exponent.exp();
            if !weight.is_finite() || weight < 0.0 {
                return Err("logit weight became non-finite".to_string());
            }
            weights.push(weight);
            weight_sum += weight;
        }

        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err("logit normalization mass is invalid".to_string());
        }

        let probabilities = weights
            .into_iter()
            .map(|weight| weight / weight_sum)
            .collect();
        ActionDistribution::new(probabilities)
    }
}

impl NPlayerGame {
    /// Expected payoff of each pure action for `player` against independent
    /// product beliefs over every opponent.
    ///
    /// The target player's own distribution in `beliefs` is shape-checked but
    /// does not influence these conditional action values.
    pub fn expected_action_values(
        &self,
        player: usize,
        beliefs: &MixedStrategyProfile,
    ) -> Result<Vec<f64>, String> {
        if player >= self.players() {
            return Err("player index out of range".to_string());
        }
        beliefs.validate_for_game(self)?;

        let mut values = vec![0.0f64; self.strategy_counts()[player]];
        for profile_index in 0..self.profile_count() {
            let profile = self.decode_profile(profile_index)?;
            let mut opponent_probability = 1.0f64;
            for opponent in 0..self.players() {
                if opponent == player {
                    continue;
                }
                opponent_probability *=
                    beliefs.distributions[opponent].probabilities[profile[opponent]];
            }

            let action = profile[player];
            let contribution = opponent_probability * self.payoff(player, &profile)?;
            let updated = values[action] + contribution;
            if !updated.is_finite() {
                return Err("expected action payoff overflowed finite f64 range".to_string());
            }
            values[action] = updated;
        }
        Ok(values)
    }
}

fn validate_action_values(action_values: &[f64]) -> Result<(), String> {
    if action_values.is_empty() {
        return Err("behavior policy requires at least one action value".to_string());
    }
    if action_values.iter().any(|value| !value.is_finite()) {
        return Err("action values must be finite".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prisoners_dilemma() -> NPlayerGame {
        NPlayerGame::new(
            vec![2, 2],
            vec![vec![3.0, 0.0, 5.0, 1.0], vec![3.0, 5.0, 0.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn action_distribution_rejects_invalid_probability_mass() {
        assert!(ActionDistribution::new(vec![]).is_err());
        assert!(ActionDistribution::new(vec![0.5, -0.5, 1.0]).is_err());
        assert!(ActionDistribution::new(vec![0.5, f64::NAN]).is_err());
        assert!(ActionDistribution::new(vec![0.4, 0.4]).is_err());
        assert!(ActionDistribution::new(vec![0.5, 0.5]).is_ok());
    }

    #[test]
    fn mixed_strategy_profile_requires_exact_game_shape() {
        let game = prisoners_dilemma();
        let uniform_two = ActionDistribution::uniform(2).unwrap();
        let uniform_three = ActionDistribution::uniform(3).unwrap();

        assert!(
            MixedStrategyProfile::new(&game, vec![uniform_two.clone()]).is_err()
        );
        assert!(
            MixedStrategyProfile::new(&game, vec![uniform_two, uniform_three]).is_err()
        );
    }

    #[test]
    fn expected_action_values_match_hand_computed_prisoners_dilemma() {
        let game = prisoners_dilemma();
        let beliefs = MixedStrategyProfile::new(
            &game,
            vec![
                ActionDistribution::uniform(2).unwrap(),
                ActionDistribution::uniform(2).unwrap(),
            ],
        )
        .unwrap();

        let row_values = game.expected_action_values(0, &beliefs).unwrap();
        assert!((row_values[0] - 1.5).abs() < 1e-12);
        assert!((row_values[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn target_players_own_belief_does_not_change_conditional_action_values() {
        let game = prisoners_dilemma();
        let opponent = ActionDistribution::uniform(2).unwrap();
        let beliefs_a = MixedStrategyProfile::new(
            &game,
            vec![ActionDistribution::new(vec![1.0, 0.0]).unwrap(), opponent.clone()],
        )
        .unwrap();
        let beliefs_b = MixedStrategyProfile::new(
            &game,
            vec![ActionDistribution::new(vec![0.0, 1.0]).unwrap(), opponent],
        )
        .unwrap();

        assert_eq!(
            game.expected_action_values(0, &beliefs_a).unwrap(),
            game.expected_action_values(0, &beliefs_b).unwrap()
        );
    }

    #[test]
    fn best_response_splits_declared_ties() {
        let policy = BestResponse::new(0.0).unwrap();
        let distribution = policy.action_distribution(&[2.0, 2.0, 1.0]).unwrap();
        assert_eq!(distribution.probabilities(), &[0.5, 0.5, 0.0]);

        let tolerant = BestResponse::new(0.01).unwrap();
        let distribution = tolerant.action_distribution(&[1.0, 0.995, 0.0]).unwrap();
        assert_eq!(distribution.probabilities(), &[0.5, 0.5, 0.0]);
    }

    #[test]
    fn logit_zero_precision_is_uniform() {
        let policy = LogitChoice::new(0.0).unwrap();
        let distribution = policy.action_distribution(&[-100.0, 0.0, 100.0]).unwrap();
        for probability in distribution.probabilities() {
            assert!((*probability - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn high_logit_precision_concentrates_on_unique_best_action_without_overflow() {
        let policy = LogitChoice::new(1000.0).unwrap();
        let distribution = policy.action_distribution(&[-1_000_000.0, 0.0, 1.0]).unwrap();
        assert!(distribution.probabilities()[2] > 0.999_999);
        assert!(distribution.probabilities().iter().all(|p| p.is_finite()));
    }

    #[test]
    fn logit_choice_is_invariant_to_additive_utility_shift() {
        let policy = LogitChoice::new(1.7).unwrap();
        let a = policy.action_distribution(&[1.0, 2.0, 3.0]).unwrap();
        let b = policy.action_distribution(&[101.0, 102.0, 103.0]).unwrap();
        for (left, right) in a.probabilities().iter().zip(b.probabilities()) {
            assert!((*left - *right).abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_behavior_parameters_fail_closed() {
        assert!(BestResponse::new(-1.0).is_err());
        assert!(BestResponse::new(f64::NAN).is_err());
        assert!(LogitChoice::new(-1.0).is_err());
        assert!(LogitChoice::new(f64::INFINITY).is_err());
        assert!(
            BestResponse::new(0.0)
                .unwrap()
                .action_distribution(&[f64::NAN])
                .is_err()
        );
    }
}
