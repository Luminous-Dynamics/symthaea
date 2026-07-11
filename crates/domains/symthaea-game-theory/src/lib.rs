// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-game-theory
//!
//! Two-player normal-form (matrix) games — best responses, **pure-strategy Nash
//! equilibria**, strict dominance, and **mixed-strategy Nash** for 2×2 games.
//! Complements `symthaea-economics` (which has only 2×2 payoff helpers) and
//! `symthaea-social-choice`.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. A [`Game`] holds two
//! payoff matrices — `row_payoff[i][j]` and `col_payoff[i][j]` — for the row
//! player choosing strategy `i` and the column player choosing `j`.
//!
//! ## Example
//!
//! ```
//! use symthaea_game_theory::Game;
//! // Prisoner's dilemma (0 = cooperate, 1 = defect): unique pure Nash is
//! // mutual defection (1, 1).
//! let g = Game::new(
//!     vec![vec![3.0, 0.0], vec![5.0, 1.0]], // row payoffs
//!     vec![vec![3.0, 5.0], vec![0.0, 1.0]], // col payoffs
//! ).unwrap();
//! assert_eq!(g.pure_nash_equilibria(), vec![(1, 1)]);
//! ```

/// A two-player normal-form game.
#[derive(Debug, Clone)]
pub struct Game {
    row_payoff: Vec<Vec<f64>>,
    col_payoff: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl Game {
    /// Build from the row and column payoff matrices (same dimensions).
    pub fn new(row_payoff: Vec<Vec<f64>>, col_payoff: Vec<Vec<f64>>) -> Result<Game, String> {
        let rows = row_payoff.len();
        if rows == 0 || col_payoff.len() != rows {
            return Err("payoff matrices must be non-empty and same height".to_string());
        }
        let cols = row_payoff[0].len();
        if cols == 0
            || row_payoff.iter().any(|r| r.len() != cols)
            || col_payoff.iter().any(|r| r.len() != cols)
        {
            return Err("payoff matrices must be rectangular and same width".to_string());
        }
        Ok(Game {
            row_payoff,
            col_payoff,
            rows,
            cols,
        })
    }

    /// (rows, cols) — the strategy-count for each player.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The row player's best-response strategies to column strategy `j`.
    pub fn row_best_responses(&self, j: usize) -> Vec<usize> {
        let best = (0..self.rows)
            .map(|i| self.row_payoff[i][j])
            .fold(f64::NEG_INFINITY, f64::max);
        (0..self.rows)
            .filter(|&i| (self.row_payoff[i][j] - best).abs() < 1e-12)
            .collect()
    }

    /// The column player's best-response strategies to row strategy `i`.
    pub fn col_best_responses(&self, i: usize) -> Vec<usize> {
        let best = (0..self.cols)
            .map(|j| self.col_payoff[i][j])
            .fold(f64::NEG_INFINITY, f64::max);
        (0..self.cols)
            .filter(|&j| (self.col_payoff[i][j] - best).abs() < 1e-12)
            .collect()
    }

    /// All pure-strategy Nash equilibria as `(row, col)` strategy pairs: cells
    /// that are simultaneously a best response for both players.
    pub fn pure_nash_equilibria(&self) -> Vec<(usize, usize)> {
        let mut eq = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                let row_ok = (0..self.rows).all(|k| self.row_payoff[i][j] >= self.row_payoff[k][j]);
                let col_ok = (0..self.cols).all(|l| self.col_payoff[i][j] >= self.col_payoff[i][l]);
                if row_ok && col_ok {
                    eq.push((i, j));
                }
            }
        }
        eq
    }

    /// Row strategies that are strictly dominated by another *pure* row strategy
    /// (some `k` beats `i` against every column strategy).
    pub fn strictly_dominated_rows(&self) -> Vec<usize> {
        (0..self.rows)
            .filter(|&i| {
                (0..self.rows).any(|k| {
                    k != i && (0..self.cols).all(|j| self.row_payoff[k][j] > self.row_payoff[i][j])
                })
            })
            .collect()
    }

    /// Column strategies that are strictly dominated by another pure column
    /// strategy.
    pub fn strictly_dominated_cols(&self) -> Vec<usize> {
        (0..self.cols)
            .filter(|&j| {
                (0..self.cols).any(|l| {
                    l != j && (0..self.rows).all(|i| self.col_payoff[i][l] > self.col_payoff[i][j])
                })
            })
            .collect()
    }

    /// The interior mixed-strategy Nash equilibrium of a 2×2 game, if one
    /// exists. Returns `((p, 1−p), (q, 1−q))` where `p` is the row player's
    /// probability of strategy 0 (making the column player indifferent) and `q`
    /// the column player's probability of strategy 0. `None` if the game is not
    /// 2×2, or the mixing probabilities fall outside `(0, 1)` (no interior
    /// mixed equilibrium — the game is dominance-solvable).
    pub fn mixed_nash_2x2(&self) -> Option<((f64, f64), (f64, f64))> {
        if (self.rows, self.cols) != (2, 2) {
            return None;
        }
        let b = &self.col_payoff;
        let a = &self.row_payoff;
        // Row mixes p so column is indifferent.
        let (ca, cb) = (b[0][0] - b[0][1], b[1][0] - b[1][1]);
        // Column mixes q so row is indifferent.
        let (ra, rb) = (a[0][0] - a[1][0], a[0][1] - a[1][1]);
        if (cb - ca).abs() < 1e-12 || (rb - ra).abs() < 1e-12 {
            return None;
        }
        let p = cb / (cb - ca);
        let q = rb / (rb - ra);
        if (0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&q) {
            Some(((p, 1.0 - p), (q, 1.0 - q)))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prisoners_dilemma() -> Game {
        Game::new(
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
            vec![vec![3.0, 5.0], vec![0.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn prisoners_dilemma_defect_is_dominant() {
        let g = prisoners_dilemma();
        assert_eq!(g.pure_nash_equilibria(), vec![(1, 1)]);
        // Cooperate (row 0, col 0) is strictly dominated by defect.
        assert_eq!(g.strictly_dominated_rows(), vec![0]);
        assert_eq!(g.strictly_dominated_cols(), vec![0]);
        // No interior mixed equilibrium (dominance-solvable).
        assert!(g.mixed_nash_2x2().is_none());
    }

    #[test]
    fn matching_pennies_mixed_only() {
        // Row wants to match, column wants to mismatch: no pure Nash, unique
        // mixed Nash at (1/2, 1/2) for both.
        let g = Game::new(
            vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
            vec![vec![-1.0, 1.0], vec![1.0, -1.0]],
        )
        .unwrap();
        assert!(g.pure_nash_equilibria().is_empty());
        let ((p, _), (q, _)) = g.mixed_nash_2x2().unwrap();
        assert!((p - 0.5).abs() < 1e-12, "p={p}");
        assert!((q - 0.5).abs() < 1e-12, "q={q}");
    }

    #[test]
    fn coordination_game_two_pure_and_one_mixed() {
        // Both prefer to coordinate; (0,0) and (1,1) are pure Nash, plus a mixed.
        let g = Game::new(
            vec![vec![2.0, 0.0], vec![0.0, 1.0]],
            vec![vec![2.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let eq = g.pure_nash_equilibria();
        assert!(eq.contains(&(0, 0)) && eq.contains(&(1, 1)), "{eq:?}");
        // Mixed: row plays strat 0 with p = 1/3 (makes column indifferent).
        let ((p, _), _) = g.mixed_nash_2x2().unwrap();
        assert!((p - 1.0 / 3.0).abs() < 1e-12, "p={p}");
    }

    #[test]
    fn best_responses() {
        let g = prisoners_dilemma();
        // Whatever the column does, defect (row 1) is the row's best response.
        assert_eq!(g.row_best_responses(0), vec![1]);
        assert_eq!(g.row_best_responses(1), vec![1]);
    }
}
