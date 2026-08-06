// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auditable 2×2 normal-form games.

use crate::error::{EconomicsError, Result, ensure_finite};

/// An interior mixed-strategy Nash equilibrium.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedNash {
    /// Probability that the row player chooses strategy 0.
    pub row_strategy_0_probability: f64,
    /// Probability that the column player chooses strategy 0.
    pub column_strategy_0_probability: f64,
    pub row_expected_payoff: f64,
    pub column_expected_payoff: f64,
}

/// A two-player 2×2 game. `payoffs[row][column] = (row_player, column_player)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Game2x2 {
    payoffs: [[(f64, f64); 2]; 2],
}

impl Game2x2 {
    pub fn new(payoffs: [[(f64, f64); 2]; 2]) -> Result<Self> {
        let finite = payoffs
            .iter()
            .flatten()
            .all(|(row, column)| row.is_finite() && column.is_finite());
        if finite {
            Ok(Self { payoffs })
        } else {
            Err(EconomicsError::NonFiniteInput {
                context: "game payoffs",
            })
        }
    }

    /// Canonical Prisoner's Dilemma (`0=cooperate`, `1=defect`).
    pub fn prisoners_dilemma() -> Self {
        Self {
            payoffs: [[(-1.0, -1.0), (-3.0, 0.0)], [(0.0, -3.0), (-2.0, -2.0)]],
        }
    }

    pub fn payoffs(self) -> [[(f64, f64); 2]; 2] {
        self.payoffs
    }

    pub fn payoff(self, row: usize, column: usize) -> Option<(f64, f64)> {
        self.payoffs
            .get(row)
            .and_then(|line| line.get(column))
            .copied()
    }

    /// Row-player best responses to a fixed column strategy.
    pub fn row_best_responses(&self, column: usize) -> Result<Vec<usize>> {
        if column >= 2 {
            return Err(EconomicsError::InvalidParameter {
                context: "column strategy index must be 0 or 1",
            });
        }
        let first = self.payoffs[0][column].0;
        let second = self.payoffs[1][column].0;
        Ok(match first.total_cmp(&second) {
            core::cmp::Ordering::Greater => vec![0],
            core::cmp::Ordering::Less => vec![1],
            core::cmp::Ordering::Equal => vec![0, 1],
        })
    }

    /// Column-player best responses to a fixed row strategy.
    pub fn column_best_responses(&self, row: usize) -> Result<Vec<usize>> {
        if row >= 2 {
            return Err(EconomicsError::InvalidParameter {
                context: "row strategy index must be 0 or 1",
            });
        }
        let first = self.payoffs[row][0].1;
        let second = self.payoffs[row][1].1;
        Ok(match first.total_cmp(&second) {
            core::cmp::Ordering::Greater => vec![0],
            core::cmp::Ordering::Less => vec![1],
            core::cmp::Ordering::Equal => vec![0, 1],
        })
    }

    /// Pure-strategy Nash equilibria as `(row, column)` index pairs.
    pub fn pure_nash_equilibria(&self) -> Vec<(usize, usize)> {
        let mut equilibria = Vec::new();
        for row in 0..2 {
            for column in 0..2 {
                let row_is_best = self.payoffs[row][column].0 >= self.payoffs[1 - row][column].0;
                let column_is_best = self.payoffs[row][column].1 >= self.payoffs[row][1 - column].1;
                if row_is_best && column_is_best {
                    equilibria.push((row, column));
                }
            }
        }
        equilibria
    }

    /// The row player's strictly dominant strategy index, if one exists.
    pub fn row_dominant_strategy(&self) -> Option<usize> {
        if (0..2).all(|column| self.payoffs[0][column].0 > self.payoffs[1][column].0) {
            Some(0)
        } else if (0..2).all(|column| self.payoffs[1][column].0 > self.payoffs[0][column].0) {
            Some(1)
        } else {
            None
        }
    }

    /// The column player's strictly dominant strategy index, if one exists.
    pub fn column_dominant_strategy(&self) -> Option<usize> {
        if (0..2).all(|row| self.payoffs[row][0].1 > self.payoffs[row][1].1) {
            Some(0)
        } else if (0..2).all(|row| self.payoffs[row][1].1 > self.payoffs[row][0].1) {
            Some(1)
        } else {
            None
        }
    }

    /// Row strategies that weakly dominate the alternative, with at least one
    /// strict improvement.
    pub fn row_weakly_dominant_strategies(&self) -> Vec<usize> {
        (0..2)
            .filter(|&candidate| {
                let alternative = 1 - candidate;
                (0..2).all(|column| {
                    self.payoffs[candidate][column].0 >= self.payoffs[alternative][column].0
                }) && (0..2).any(|column| {
                    self.payoffs[candidate][column].0 > self.payoffs[alternative][column].0
                })
            })
            .collect()
    }

    /// Column strategies that weakly dominate the alternative, with at least
    /// one strict improvement.
    pub fn column_weakly_dominant_strategies(&self) -> Vec<usize> {
        (0..2)
            .filter(|&candidate| {
                let alternative = 1 - candidate;
                (0..2).all(|row| self.payoffs[row][candidate].1 >= self.payoffs[row][alternative].1)
                    && (0..2).any(|row| {
                        self.payoffs[row][candidate].1 > self.payoffs[row][alternative].1
                    })
            })
            .collect()
    }

    /// Unique interior mixed-strategy Nash equilibrium, if it exists.
    ///
    /// Boundary probabilities are pure strategies and are therefore omitted.
    /// Degenerate games with a continuum of mixtures return `None`.
    pub fn interior_mixed_nash(&self) -> Option<MixedNash> {
        let a00 = self.payoffs[0][0].0;
        let a01 = self.payoffs[0][1].0;
        let a10 = self.payoffs[1][0].0;
        let a11 = self.payoffs[1][1].0;
        let b00 = self.payoffs[0][0].1;
        let b01 = self.payoffs[0][1].1;
        let b10 = self.payoffs[1][0].1;
        let b11 = self.payoffs[1][1].1;

        let row_denominator = b00 - b01 - b10 + b11;
        let column_denominator = a00 - a01 - a10 + a11;
        if row_denominator == 0.0 || column_denominator == 0.0 {
            return None;
        }

        let row_probability = (b11 - b10) / row_denominator;
        let column_probability = (a11 - a01) / column_denominator;
        if !(0.0 < row_probability
            && row_probability < 1.0
            && 0.0 < column_probability
            && column_probability < 1.0)
        {
            return None;
        }

        let row_expected_payoff = column_probability * a00 + (1.0 - column_probability) * a01;
        let column_expected_payoff = row_probability * b00 + (1.0 - row_probability) * b10;
        Some(MixedNash {
            row_strategy_0_probability: row_probability,
            column_strategy_0_probability: column_probability,
            row_expected_payoff,
            column_expected_payoff,
        })
    }

    /// Outcomes not Pareto-dominated by another pure outcome.
    pub fn pareto_efficient_outcomes(&self) -> Vec<(usize, usize)> {
        let mut efficient = Vec::new();
        for row in 0..2 {
            for column in 0..2 {
                let candidate = self.payoffs[row][column];
                let dominated = (0..2).any(|other_row| {
                    (0..2).any(|other_column| {
                        if other_row == row && other_column == column {
                            return false;
                        }
                        let other = self.payoffs[other_row][other_column];
                        other.0 >= candidate.0
                            && other.1 >= candidate.1
                            && (other.0 > candidate.0 || other.1 > candidate.1)
                    })
                });
                if !dominated {
                    efficient.push((row, column));
                }
            }
        }
        efficient
    }

    /// Pure outcomes maximizing the sum of both players' payoffs.
    pub fn social_welfare_maximizers(&self) -> Vec<(usize, usize)> {
        let maximum = self
            .payoffs
            .iter()
            .flatten()
            .map(|(row, column)| row + column)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut outcomes = Vec::new();
        for row in 0..2 {
            for column in 0..2 {
                let payoff = self.payoffs[row][column];
                if payoff.0 + payoff.1 == maximum {
                    outcomes.push((row, column));
                }
            }
        }
        outcomes
    }

    /// True when every outcome has the same total payoff within `tolerance`.
    pub fn is_constant_sum(&self, tolerance: f64) -> Result<bool> {
        ensure_finite(tolerance, "constant-sum tolerance")?;
        if tolerance < 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "constant-sum tolerance must be non-negative",
            });
        }
        let reference = self.payoffs[0][0].0 + self.payoffs[0][0].1;
        Ok(self
            .payoffs
            .iter()
            .flatten()
            .all(|payoff| (payoff.0 + payoff.1 - reference).abs() <= tolerance))
    }

    /// Swap players and strategy axes.
    pub fn transpose(self) -> Self {
        let mut transposed = [[(0.0, 0.0); 2]; 2];
        for row in 0..2 {
            for column in 0..2 {
                let payoff = self.payoffs[row][column];
                transposed[column][row] = (payoff.1, payoff.0);
            }
        }
        Self {
            payoffs: transposed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prisoners_dilemma_truth_is_explicit() {
        let game = Game2x2::prisoners_dilemma();
        assert_eq!(game.pure_nash_equilibria(), vec![(1, 1)]);
        assert_eq!(game.row_dominant_strategy(), Some(1));
        assert_eq!(game.column_dominant_strategy(), Some(1));
        assert!(game.pareto_efficient_outcomes().contains(&(0, 0)));
        assert_eq!(game.social_welfare_maximizers(), vec![(0, 0)]);
    }

    #[test]
    fn matching_pennies_has_interior_mixed_nash() {
        let game = Game2x2::new([[(1.0, -1.0), (-1.0, 1.0)], [(-1.0, 1.0), (1.0, -1.0)]]).unwrap();
        assert!(game.pure_nash_equilibria().is_empty());
        let mixed = game.interior_mixed_nash().unwrap();
        assert!((mixed.row_strategy_0_probability - 0.5).abs() < 1e-12);
        assert!((mixed.column_strategy_0_probability - 0.5).abs() < 1e-12);
        assert!(mixed.row_expected_payoff.abs() < 1e-12);
        assert!(mixed.column_expected_payoff.abs() < 1e-12);
        assert!(game.is_constant_sum(1e-12).unwrap());
    }

    #[test]
    fn transpose_swaps_player_analysis() {
        let game = Game2x2::prisoners_dilemma();
        let transposed = game.transpose();
        assert_eq!(
            game.column_dominant_strategy(),
            transposed.row_dominant_strategy()
        );
        assert_eq!(
            game.pure_nash_equilibria(),
            transposed.pure_nash_equilibria()
        );
    }

    #[test]
    fn weak_dominance_allows_ties() {
        let game = Game2x2::new([[(2.0, 1.0), (1.0, 2.0)], [(2.0, 0.0), (0.0, 2.0)]]).unwrap();
        assert_eq!(game.row_dominant_strategy(), None);
        assert_eq!(game.row_weakly_dominant_strategies(), vec![0]);
    }

    #[test]
    fn non_finite_payoffs_are_rejected() {
        assert!(Game2x2::new([[(f64::NAN, 0.0); 2]; 2]).is_err());
    }
}
