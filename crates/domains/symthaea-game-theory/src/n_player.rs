// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated finite N-player normal-form games and unilateral-deviation analysis.
//!
//! This module intentionally stays dependency-free and formal. It does not model
//! bounded rationality, coalitions, false-name identities, evidence, governance
//! legitimacy, or constitutional authority. Those belong to later IG tranches.

/// Exact witness for one profitable unilateral deviation from a declared profile.
#[derive(Debug, Clone, PartialEq)]
pub struct UnilateralDeviation {
    /// Player whose strategy changes.
    pub player: usize,
    /// Strategy used in the baseline profile.
    pub from_strategy: usize,
    /// Alternative strategy used in the deviated profile.
    pub to_strategy: usize,
    /// Payoff at the baseline profile.
    pub baseline_payoff: f64,
    /// Payoff after this player's unilateral deviation.
    pub deviated_payoff: f64,
    /// `deviated_payoff - baseline_payoff`, strictly positive for witnesses.
    pub gain: f64,
    /// Complete strategy profile after the deviation.
    pub deviated_profile: Vec<usize>,
}

/// A validated finite N-player normal-form game.
///
/// `payoffs[player][profile_index]` stores the payoff to `player` at the
/// mixed-radix encoded pure-strategy profile. Construction rejects empty games,
/// zero strategy counts, profile-count overflow, payoff-shape mismatches and
/// non-finite payoffs.
#[derive(Debug, Clone, PartialEq)]
pub struct NPlayerGame {
    strategy_counts: Vec<usize>,
    payoffs: Vec<Vec<f64>>,
    profile_count: usize,
}

impl NPlayerGame {
    /// Construct a validated finite normal-form game.
    pub fn new(strategy_counts: Vec<usize>, payoffs: Vec<Vec<f64>>) -> Result<Self, String> {
        if strategy_counts.is_empty() {
            return Err("normal-form game must contain at least one player".to_string());
        }
        if strategy_counts.contains(&0) {
            return Err("every player must have at least one strategy".to_string());
        }

        let mut profile_count = 1usize;
        for count in &strategy_counts {
            profile_count = profile_count
                .checked_mul(*count)
                .ok_or_else(|| "strategy profile count overflows usize".to_string())?;
        }

        if payoffs.len() != strategy_counts.len() {
            return Err("payoff table count must equal player count".to_string());
        }
        for player_payoffs in &payoffs {
            if player_payoffs.len() != profile_count {
                return Err(
                    "each player payoff table must contain one value per strategy profile"
                        .to_string(),
                );
            }
            if player_payoffs.iter().any(|value| !value.is_finite()) {
                return Err("all payoffs must be finite".to_string());
            }
        }

        Ok(Self {
            strategy_counts,
            payoffs,
            profile_count,
        })
    }

    /// Number of players.
    pub fn players(&self) -> usize {
        self.strategy_counts.len()
    }

    /// Number of available pure strategies for each player.
    pub fn strategy_counts(&self) -> &[usize] {
        &self.strategy_counts
    }

    /// Total number of pure-strategy profiles.
    pub fn profile_count(&self) -> usize {
        self.profile_count
    }

    /// Encode a complete pure-strategy profile using canonical mixed-radix order.
    pub fn profile_index(&self, profile: &[usize]) -> Result<usize, String> {
        self.validate_profile(profile)?;

        let mut index = 0usize;
        for (strategy, count) in profile.iter().zip(&self.strategy_counts) {
            index = index
                .checked_mul(*count)
                .and_then(|value| value.checked_add(*strategy))
                .ok_or_else(|| "strategy profile index overflows usize".to_string())?;
        }
        Ok(index)
    }

    /// Decode a canonical mixed-radix profile index.
    pub fn decode_profile(&self, mut index: usize) -> Result<Vec<usize>, String> {
        if index >= self.profile_count {
            return Err("strategy profile index out of range".to_string());
        }

        let mut profile = vec![0usize; self.players()];
        for player in (0..self.players()).rev() {
            let count = self.strategy_counts[player];
            profile[player] = index % count;
            index /= count;
        }
        Ok(profile)
    }

    /// Payoff to one player at a complete pure-strategy profile.
    pub fn payoff(&self, player: usize, profile: &[usize]) -> Result<f64, String> {
        if player >= self.players() {
            return Err("player index out of range".to_string());
        }
        let index = self.profile_index(profile)?;
        Ok(self.payoffs[player][index])
    }

    /// Profitable unilateral deviations available to one player.
    ///
    /// Returned witnesses are ordered by ascending alternative strategy index.
    pub fn unilateral_deviations(
        &self,
        profile: &[usize],
        player: usize,
    ) -> Result<Vec<UnilateralDeviation>, String> {
        self.validate_profile(profile)?;
        if player >= self.players() {
            return Err("player index out of range".to_string());
        }

        let baseline_payoff = self.payoff(player, profile)?;
        let from_strategy = profile[player];
        let mut witnesses = Vec::new();

        for to_strategy in 0..self.strategy_counts[player] {
            if to_strategy == from_strategy {
                continue;
            }
            let mut deviated_profile = profile.to_vec();
            deviated_profile[player] = to_strategy;
            let deviated_payoff = self.payoff(player, &deviated_profile)?;
            let gain = deviated_payoff - baseline_payoff;
            if gain > 0.0 {
                witnesses.push(UnilateralDeviation {
                    player,
                    from_strategy,
                    to_strategy,
                    baseline_payoff,
                    deviated_payoff,
                    gain,
                    deviated_profile,
                });
            }
        }

        Ok(witnesses)
    }

    /// All profitable unilateral deviations from a profile, grouped by player order.
    pub fn profitable_unilateral_deviations(
        &self,
        profile: &[usize],
    ) -> Result<Vec<UnilateralDeviation>, String> {
        self.validate_profile(profile)?;
        let mut witnesses = Vec::new();
        for player in 0..self.players() {
            witnesses.extend(self.unilateral_deviations(profile, player)?);
        }
        Ok(witnesses)
    }

    /// Maximum profitable unilateral gain available to one player.
    ///
    /// Returns `0.0` when the player has no profitable unilateral deviation.
    pub fn player_regret(&self, profile: &[usize], player: usize) -> Result<f64, String> {
        let maximum = self
            .unilateral_deviations(profile, player)?
            .into_iter()
            .map(|witness| witness.gain)
            .fold(0.0f64, f64::max);
        Ok(maximum)
    }

    /// Per-player unilateral regret at a profile.
    pub fn regrets(&self, profile: &[usize]) -> Result<Vec<f64>, String> {
        self.validate_profile(profile)?;
        (0..self.players())
            .map(|player| self.player_regret(profile, player))
            .collect()
    }

    /// Largest profitable unilateral gain available to any player.
    pub fn max_regret(&self, profile: &[usize]) -> Result<f64, String> {
        Ok(self.regrets(profile)?.into_iter().fold(0.0f64, f64::max))
    }

    /// Whether the profile is an epsilon-Nash equilibrium for unilateral pure deviations.
    ///
    /// `epsilon` must be finite and non-negative. `epsilon = 0` is exact pure Nash.
    pub fn is_epsilon_nash(&self, profile: &[usize], epsilon: f64) -> Result<bool, String> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err("epsilon must be finite and non-negative".to_string());
        }
        Ok(self.max_regret(profile)? <= epsilon)
    }

    /// Whether the profile is an exact pure-strategy Nash equilibrium.
    pub fn is_pure_nash(&self, profile: &[usize]) -> Result<bool, String> {
        self.is_epsilon_nash(profile, 0.0)
    }

    /// Enumerate every exact pure-strategy Nash equilibrium.
    pub fn pure_nash_equilibria(&self) -> Vec<Vec<usize>> {
        let mut equilibria = Vec::new();
        for index in 0..self.profile_count {
            // Every index in this loop is valid by construction.
            let profile = self
                .decode_profile(index)
                .expect("validated profile index must decode");
            if self
                .is_pure_nash(&profile)
                .expect("decoded profile must be valid")
            {
                equilibria.push(profile);
            }
        }
        equilibria
    }

    fn validate_profile(&self, profile: &[usize]) -> Result<(), String> {
        if profile.len() != self.players() {
            return Err("strategy profile length must equal player count".to_string());
        }
        for (player, strategy) in profile.iter().enumerate() {
            if *strategy >= self.strategy_counts[player] {
                return Err("strategy index out of range".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Game;

    fn prisoners_dilemma_n_player() -> NPlayerGame {
        NPlayerGame::new(
            vec![2, 2],
            vec![vec![3.0, 0.0, 5.0, 1.0], vec![3.0, 5.0, 0.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn construction_rejects_malformed_games() {
        assert!(NPlayerGame::new(vec![], vec![]).is_err());
        assert!(NPlayerGame::new(vec![2, 0], vec![vec![], vec![]]).is_err());
        assert!(NPlayerGame::new(vec![2, 2], vec![vec![0.0; 4]]).is_err());
        assert!(
            NPlayerGame::new(vec![2, 2], vec![vec![0.0; 3], vec![0.0; 4]]).is_err()
        );
        assert!(
            NPlayerGame::new(
                vec![2, 2],
                vec![vec![0.0, f64::NAN, 0.0, 0.0], vec![0.0; 4]],
            )
            .is_err()
        );
        assert!(NPlayerGame::new(vec![usize::MAX, 2], vec![vec![], vec![]]).is_err());
    }

    #[test]
    fn heterogeneous_profile_index_round_trips() {
        let game = NPlayerGame::new(vec![2, 3, 2], vec![vec![0.0; 12]; 3]).unwrap();
        for index in 0..game.profile_count() {
            let profile = game.decode_profile(index).unwrap();
            assert_eq!(game.profile_index(&profile).unwrap(), index);
        }
        assert_eq!(game.profile_index(&[1, 2, 1]).unwrap(), 11);
        assert!(game.profile_index(&[1, 3, 0]).is_err());
        assert!(game.decode_profile(12).is_err());
    }

    #[test]
    fn deviation_witness_exposes_exact_profitable_change() {
        let game = prisoners_dilemma_n_player();
        let deviations = game.profitable_unilateral_deviations(&[0, 0]).unwrap();
        assert_eq!(deviations.len(), 2);

        assert_eq!(
            deviations[0],
            UnilateralDeviation {
                player: 0,
                from_strategy: 0,
                to_strategy: 1,
                baseline_payoff: 3.0,
                deviated_payoff: 5.0,
                gain: 2.0,
                deviated_profile: vec![1, 0],
            }
        );
        assert_eq!(
            deviations[1],
            UnilateralDeviation {
                player: 1,
                from_strategy: 0,
                to_strategy: 1,
                baseline_payoff: 3.0,
                deviated_payoff: 5.0,
                gain: 2.0,
                deviated_profile: vec![0, 1],
            }
        );
    }

    #[test]
    fn regret_and_epsilon_nash_are_explicit() {
        let game = prisoners_dilemma_n_player();
        assert_eq!(game.regrets(&[0, 0]).unwrap(), vec![2.0, 2.0]);
        assert_eq!(game.max_regret(&[0, 0]).unwrap(), 2.0);
        assert!(!game.is_epsilon_nash(&[0, 0], 1.999).unwrap());
        assert!(game.is_epsilon_nash(&[0, 0], 2.0).unwrap());
        assert!(game.is_pure_nash(&[1, 1]).unwrap());
        assert_eq!(game.regrets(&[1, 1]).unwrap(), vec![0.0, 0.0]);
        assert!(game.is_epsilon_nash(&[1, 1], f64::NAN).is_err());
        assert!(game.is_epsilon_nash(&[1, 1], -0.1).is_err());
    }

    #[test]
    fn three_player_coordination_has_two_pure_equilibria() {
        // Every player receives 1 iff all three choose the same action, else 0.
        let common = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let game = NPlayerGame::new(vec![2, 2, 2], vec![common.clone(), common.clone(), common])
            .unwrap();
        assert_eq!(
            game.pure_nash_equilibria(),
            vec![vec![0, 0, 0], vec![1, 1, 1]]
        );
    }

    #[test]
    fn two_player_nash_matches_existing_game_fixture() {
        let legacy = Game::new(
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let game = prisoners_dilemma_n_player();
        let legacy_equilibria: Vec<Vec<usize>> = legacy
            .pure_nash_equilibria()
            .into_iter()
            .map(|(row, column)| vec![row, column])
            .collect();
        assert_eq!(game.pure_nash_equilibria(), legacy_equilibria);
    }
}
