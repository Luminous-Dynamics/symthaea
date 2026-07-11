// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 2×2 normal-form games: pure-strategy Nash equilibria and dominant strategies.

/// A two-player 2×2 game. `payoffs[row][col] = (row_player, col_player)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Game2x2 {
    pub payoffs: [[(f64, f64); 2]; 2],
}

impl Game2x2 {
    /// The canonical Prisoner's Dilemma (0=cooperate, 1=defect); (defect,defect)
    /// is the unique Nash equilibrium though (cooperate,cooperate) Pareto-dominates.
    pub fn prisoners_dilemma() -> Game2x2 {
        Game2x2 {
            payoffs: [
                [(-1.0, -1.0), (-3.0, 0.0)], // C: vs C, vs D
                [(0.0, -3.0), (-2.0, -2.0)], // D: vs C, vs D
            ],
        }
    }

    /// Pure-strategy Nash equilibria as `(row, col)` index pairs.
    ///
    /// `(i,j)` is a Nash equilibrium iff row `i` is a best response to col `j`
    /// (maximizes the row player's payoff given `j`) and vice versa.
    pub fn pure_nash_equilibria(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let row_ok = self.payoffs[i][j].0 >= self.payoffs[1 - i][j].0;
                let col_ok = self.payoffs[i][j].1 >= self.payoffs[i][1 - j].1;
                if row_ok && col_ok {
                    out.push((i, j));
                }
            }
        }
        out
    }

    /// The row player's strictly dominant strategy index, if one exists.
    pub fn row_dominant_strategy(&self) -> Option<usize> {
        // Row 0 dominates if it beats row 1 against every column.
        if (0..2).all(|j| self.payoffs[0][j].0 > self.payoffs[1][j].0) {
            Some(0)
        } else if (0..2).all(|j| self.payoffs[1][j].0 > self.payoffs[0][j].0) {
            Some(1)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prisoners_dilemma_nash_is_defect_defect() {
        let g = Game2x2::prisoners_dilemma();
        assert_eq!(g.pure_nash_equilibria(), vec![(1, 1)]);
        assert_eq!(g.row_dominant_strategy(), Some(1)); // defect dominates
    }

    #[test]
    fn coordination_game_has_two_equilibria() {
        // Both pick the same → reward; mismatched → 0.
        let g = Game2x2 {
            payoffs: [[(2.0, 2.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]],
        };
        let ne = g.pure_nash_equilibria();
        assert!(ne.contains(&(0, 0)) && ne.contains(&(1, 1)));
        assert_eq!(ne.len(), 2);
        assert_eq!(g.row_dominant_strategy(), None); // no dominant strategy
    }
}
