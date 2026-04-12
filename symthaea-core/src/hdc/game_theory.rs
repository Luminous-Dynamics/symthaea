// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Game Theory module: normal-form games, zero-sum minimax, cooperative games,
//! Sprague-Grundy theory, mechanism design, and auctions.

use std::collections::HashMap;

// ─── Normal Form Games ───────────────────────────────────────────────────────

/// A normal-form game with `players` players.
/// `strategy_counts[i]` is the number of pure strategies for player i.
/// `payoffs[player][strategy_profile_idx]` gives the payoff.
/// Strategy profile index is computed via mixed-radix encoding.
#[derive(Clone, Debug)]
pub struct NormalFormGame {
    pub players: usize,
    pub strategy_counts: Vec<usize>,
    pub payoffs: Vec<Vec<f64>>,
}

/// Compute the mixed-radix index of a strategy profile.
/// `strategies[i]` is the strategy chosen by player i.
pub fn strategy_profile_index(strategies: &[usize], counts: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut multiplier = 1usize;
    for i in (0..strategies.len()).rev() {
        idx += strategies[i] * multiplier;
        multiplier *= counts[i];
    }
    idx
}

/// Decode a profile index back to a strategy vector.
fn decode_profile(mut idx: usize, counts: &[usize]) -> Vec<usize> {
    let n = counts.len();
    let mut strategies = vec![0usize; n];
    for i in (0..n).rev() {
        strategies[i] = idx % counts[i];
        idx /= counts[i];
    }
    strategies
}

/// Total number of strategy profiles.
fn num_profiles(counts: &[usize]) -> usize {
    counts.iter().product()
}

impl NormalFormGame {
    /// Create a new game. `payoffs[player]` must have length equal to the total
    /// number of strategy profiles.
    pub fn new(players: usize, strategy_counts: Vec<usize>, payoffs: Vec<Vec<f64>>) -> Self {
        NormalFormGame { players, strategy_counts, payoffs }
    }

    /// Get payoff for `player` at strategy profile `profile`.
    pub fn get_payoff(&self, player: usize, profile: &[usize]) -> f64 {
        let idx = strategy_profile_index(profile, &self.strategy_counts);
        self.payoffs[player][idx]
    }

    /// Check whether `profile` is a pure Nash equilibrium.
    /// No player can improve by unilateral deviation.
    pub fn is_pure_nash(&self, profile: &[usize]) -> bool {
        for p in 0..self.players {
            let current = self.get_payoff(p, profile);
            // Try every alternative strategy for player p
            for alt in 0..self.strategy_counts[p] {
                if alt == profile[p] {
                    continue;
                }
                let mut dev = profile.to_vec();
                dev[p] = alt;
                if self.get_payoff(p, &dev) > current {
                    return false;
                }
            }
        }
        true
    }

    /// Enumerate all pure Nash equilibria.
    pub fn find_pure_nash(&self) -> Vec<Vec<usize>> {
        let total = num_profiles(&self.strategy_counts);
        let mut result = Vec::new();
        for idx in 0..total {
            let profile = decode_profile(idx, &self.strategy_counts);
            if self.is_pure_nash(&profile) {
                result.push(profile);
            }
        }
        result
    }
}

// ─── Iterated Dominance ───────────────────────────────────────────────────────

impl NormalFormGame {
    /// Return the indices of strictly dominated strategies for `player`.
    /// Strategy s is strictly dominated if there exists s' such that
    /// u(s', σ_{-i}) > u(s, σ_{-i}) for ALL opponent strategy profiles.
    pub fn strictly_dominated_strategies(&self, player: usize) -> Vec<usize> {
        let n_s = self.strategy_counts[player];
        // Build all opponent profiles (Cartesian product of others' strategies)
        let opponent_counts: Vec<usize> = self
            .strategy_counts
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != player)
            .map(|(_, &c)| c)
            .collect();
        let n_opp = opponent_counts.iter().product::<usize>().max(1);

        let mut dominated = Vec::new();

        'outer: for s in 0..n_s {
            // Try to find a dominating strategy s_prime
            for s_prime in 0..n_s {
                if s_prime == s {
                    continue;
                }
                // Check dominance over all opponent profiles
                let mut dominates = true;
                for opp_idx in 0..n_opp {
                    let opp_profile = if opponent_counts.is_empty() {
                        vec![]
                    } else {
                        decode_profile(opp_idx, &opponent_counts)
                    };
                    // Build full profiles
                    let mut profile_s = vec![0usize; self.players];
                    let mut profile_sp = vec![0usize; self.players];
                    let mut opp_iter = opp_profile.iter();
                    for i in 0..self.players {
                        if i == player {
                            profile_s[i] = s;
                            profile_sp[i] = s_prime;
                        } else {
                            let v = *opp_iter.next().unwrap_or(&0);
                            profile_s[i] = v;
                            profile_sp[i] = v;
                        }
                    }
                    if self.get_payoff(player, &profile_sp) <= self.get_payoff(player, &profile_s) {
                        dominates = false;
                        break;
                    }
                }
                if dominates {
                    dominated.push(s);
                    continue 'outer;
                }
            }
        }
        dominated
    }

    /// Iteratively eliminate strictly dominated strategies until a fixed point.
    pub fn iterated_elimination_strict(&self) -> Self {
        let mut current = self.clone();
        loop {
            let mut eliminated = false;
            // Track which strategies remain for each player
            let mut kept: Vec<Vec<usize>> = (0..current.players)
                .map(|p| (0..current.strategy_counts[p]).collect())
                .collect();

            for p in 0..current.players {
                let dom = current.strictly_dominated_strategies(p);
                if !dom.is_empty() {
                    eliminated = true;
                    kept[p].retain(|s| !dom.contains(s));
                }
            }

            if !eliminated {
                break;
            }

            // Rebuild game restricted to kept strategies
            let new_counts: Vec<usize> = kept.iter().map(|v| v.len()).collect();
            let total_new = num_profiles(&new_counts);
            let mut new_payoffs: Vec<Vec<f64>> = vec![vec![0.0; total_new]; current.players];

            for new_idx in 0..total_new {
                let new_profile = decode_profile(new_idx, &new_counts);
                // Map back to original indices
                let orig_profile: Vec<usize> =
                    new_profile.iter().enumerate().map(|(p, &s)| kept[p][s]).collect();
                for pl in 0..current.players {
                    new_payoffs[pl][new_idx] = current.get_payoff(pl, &orig_profile);
                }
            }

            current = NormalFormGame::new(current.players, new_counts, new_payoffs);
        }
        current
    }
}

// ─── Zero-Sum Games ───────────────────────────────────────────────────────────

/// A two-player zero-sum game. Row player maximizes, column player minimizes.
#[derive(Clone, Debug)]
pub struct ZeroSumGame {
    pub rows: usize,
    pub cols: usize,
    /// matrix[i][j] is the payoff to the row player.
    pub matrix: Vec<Vec<f64>>,
}

impl ZeroSumGame {
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        ZeroSumGame { rows, cols, matrix }
    }

    /// Maximin value: max over rows of (min over cols).
    pub fn maximin_value(&self) -> f64 {
        self.matrix
            .iter()
            .map(|row| row.iter().cloned().fold(f64::INFINITY, f64::min))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimax value: min over cols of (max over rows).
    pub fn minimax_value(&self) -> f64 {
        (0..self.cols)
            .map(|j| {
                (0..self.rows)
                    .map(|i| self.matrix[i][j])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Find a saddle point: cell where row_max == col_min (pure strategy equilibrium).
    pub fn has_saddle_point(&self) -> Option<(usize, usize)> {
        for i in 0..self.rows {
            let row_min = self.matrix[i].iter().cloned().fold(f64::INFINITY, f64::min);
            for j in 0..self.cols {
                let col_max = (0..self.rows)
                    .map(|r| self.matrix[r][j])
                    .fold(f64::NEG_INFINITY, f64::max);
                if (self.matrix[i][j] - row_min).abs() < 1e-12
                    && (self.matrix[i][j] - col_max).abs() < 1e-12
                {
                    return Some((i, j));
                }
            }
        }
        None
    }

    /// Compute mixed strategy Nash equilibrium via fictitious play.
    /// Returns (game_value, row_mixed_strategy, col_mixed_strategy).
    pub fn mixed_strategy_value_lp(&self) -> (f64, Vec<f64>, Vec<f64>) {
        let r = self.rows;
        let c = self.cols;
        let iters = 1000usize;

        // Fictitious play: each player best-responds to empirical distribution of other.
        let mut row_counts = vec![0usize; r];
        let mut col_counts = vec![0usize; c];

        // Initialise
        row_counts[0] = 1;
        col_counts[0] = 1;

        for t in 1..iters {
            // Column player best response: minimize against empirical row distribution
            let col_br = (0..c)
                .min_by(|&j1, &j2| {
                    let v1: f64 = (0..r)
                        .map(|i| self.matrix[i][j1] * row_counts[i] as f64)
                        .sum();
                    let v2: f64 = (0..r)
                        .map(|i| self.matrix[i][j2] * row_counts[i] as f64)
                        .sum();
                    v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            // Row player best response: maximize against empirical col distribution
            let row_br = (0..r)
                .max_by(|&i1, &i2| {
                    let v1: f64 = (0..c)
                        .map(|j| self.matrix[i1][j] * col_counts[j] as f64)
                        .sum();
                    let v2: f64 = (0..c)
                        .map(|j| self.matrix[i2][j] * col_counts[j] as f64)
                        .sum();
                    v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            col_counts[col_br] += 1;
            row_counts[row_br] += 1;
            let _ = t;
        }

        let total_row = row_counts.iter().sum::<usize>() as f64;
        let total_col = col_counts.iter().sum::<usize>() as f64;
        let row_probs: Vec<f64> = row_counts.iter().map(|&c| c as f64 / total_row).collect();
        let col_probs: Vec<f64> = col_counts.iter().map(|&c| c as f64 / total_col).collect();

        // Compute game value as row_probs^T * matrix * col_probs
        let value: f64 = (0..r)
            .map(|i| (0..c).map(|j| row_probs[i] * self.matrix[i][j] * col_probs[j]).sum::<f64>())
            .sum();

        (value, row_probs, col_probs)
    }
}

// ─── Cooperative Game Theory ──────────────────────────────────────────────────

/// A cooperative game with n players.
/// `v` maps coalition bitmask (bit i set ⟺ player i is in coalition) → value.
#[derive(Clone, Debug)]
pub struct CooperativeGame {
    pub n: usize,
    pub v: HashMap<u64, f64>,
}

impl CooperativeGame {
    pub fn new(n: usize, v: HashMap<u64, f64>) -> Self {
        CooperativeGame { n, v }
    }

    fn coalition_value(&self, mask: u64) -> f64 {
        *self.v.get(&mask).unwrap_or(&0.0)
    }

    /// Compute Shapley values.
    /// φᵢ = Σ_{S ⊆ N\{i}} [|S|!(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]
    pub fn shapley_value(&self) -> Vec<f64> {
        let n = self.n;
        let mut phi = vec![0.0f64; n];
        let n_fact = factorial(n) as f64;

        for i in 0..n {
            let mut value = 0.0f64;
            // Enumerate all subsets S not containing i
            for mask in 0u64..(1u64 << n) {
                if mask & (1u64 << i) != 0 {
                    continue; // S contains i, skip
                }
                let s_size = mask.count_ones() as usize;
                let weight =
                    (factorial(s_size) * factorial(n - s_size - 1)) as f64 / n_fact;
                let v_with = self.coalition_value(mask | (1u64 << i));
                let v_without = self.coalition_value(mask);
                value += weight * (v_with - v_without);
            }
            phi[i] = value;
        }
        phi
    }

    /// Check superadditivity: v(S∪T) ≥ v(S) + v(T) for all disjoint S, T.
    pub fn is_superadditive(&self) -> bool {
        let n = self.n;
        for s in 0u64..(1u64 << n) {
            for t in 0u64..(1u64 << n) {
                if s & t != 0 {
                    continue; // Not disjoint
                }
                if self.coalition_value(s | t) < self.coalition_value(s) + self.coalition_value(t) - 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Simplified core non-emptiness check:
    /// The core is non-empty if there exist allocations x_i such that
    /// sum(x_i for i in S) >= v(S) for all S, and sum(x_i) = v(N).
    /// We use a necessary condition: grand coalition value >= max sub-coalition values.
    pub fn core_nonempty(&self) -> bool {
        let grand = self.coalition_value((1u64 << self.n) - 1);
        // Check: for every coalition S, v(S) + v(complement) <= v(N)
        for s in 1u64..(1u64 << self.n) {
            let complement = ((1u64 << self.n) - 1) & !s;
            if self.coalition_value(s) + self.coalition_value(complement) > grand + 1e-10 {
                return false;
            }
        }
        true
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

// ─── Combinatorial Game Theory (Nim / Sprague-Grundy) ────────────────────────

/// Nim game value (XOR of pile sizes).
pub fn nim_value(piles: &[usize]) -> usize {
    piles.iter().fold(0, |acc, &p| acc ^ p)
}

/// True if the current position is a losing position (nim value == 0).
pub fn nim_is_losing_position(piles: &[usize]) -> bool {
    nim_value(piles) == 0
}

/// Compute the Grundy (nimber) value for position n with given move set.
/// moves: set of positive integers you may subtract.
pub fn grundy_value(n: usize, moves: &[usize]) -> usize {
    if n == 0 {
        return 0;
    }
    let reachable: Vec<usize> = moves
        .iter()
        .filter(|&&m| m <= n)
        .map(|&m| grundy_value(n - m, moves))
        .collect();
    mex(&reachable)
}

/// Minimum excludant: smallest non-negative integer not in `values`.
pub fn mex(values: &[usize]) -> usize {
    let mut i = 0;
    loop {
        if !values.contains(&i) {
            return i;
        }
        i += 1;
    }
}

// ─── Mechanism Design ─────────────────────────────────────────────────────────

/// VCG payments for a single-item allocation.
/// `valuations[i]` is agent i's value.
/// `chosen_idx` is the winner (assumed to be argmax of social welfare).
/// Payment_i = sum_{j≠i} v_j(outcome without i) - sum_{j≠i} v_j(outcome with i).
/// For single-item: outcome without i → second-best agent wins; outcome with i → chosen wins.
pub fn vcg_payment(valuations: &[f64], chosen_idx: usize) -> Vec<f64> {
    let n = valuations.len();
    let mut payments = vec![0.0f64; n];

    for i in 0..n {
        if i == chosen_idx {
            // Winner pays: highest value among others (second-price)
            let others_best = valuations
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v)
                .fold(f64::NEG_INFINITY, f64::max);
            payments[i] = if others_best == f64::NEG_INFINITY { 0.0 } else { others_best };
        } else {
            // Losers pay 0
            payments[i] = 0.0;
        }
    }
    payments
}

/// Check dominant strategy truthfulness for a given mechanism on a small example
/// (3 agents, 3 types: low=0.1, med=0.5, high=0.9).
/// A mechanism is dominant-strategy truthful if for every agent and every type profile,
/// reporting truthfully is a best response regardless of others' reports.
pub fn is_dominant_strategy_truthful(mechanism: &dyn Fn(&[f64]) -> usize) -> bool {
    let types = [0.1f64, 0.5, 0.9];
    // For each agent, for each true type and each possible misreport, check
    // that truthful reporting weakly dominates.
    for agent in 0..3 {
        for true_type_idx in 0..3 {
            let true_val = types[true_type_idx];
            for misreport_idx in 0..3 {
                if misreport_idx == true_type_idx {
                    continue;
                }
                // Try over all opponent profiles
                for o1 in 0..3 {
                    for o2 in 0..3 {
                        let mut truthful_report = vec![0.0f64; 3];
                        let mut lie_report = vec![0.0f64; 3];
                        // Fill opponents
                        let opponents: Vec<usize> = (0..3).filter(|&x| x != agent).collect();
                        truthful_report[opponents[0]] = types[o1];
                        truthful_report[opponents[1]] = types[o2];
                        lie_report[opponents[0]] = types[o1];
                        lie_report[opponents[1]] = types[o2];

                        truthful_report[agent] = true_val;
                        lie_report[agent] = types[misreport_idx];

                        let truthful_winner = mechanism(&truthful_report);
                        let lie_winner = mechanism(&lie_report);

                        // Utility = value if you win, 0 otherwise (quasi-linear, no payment for simplicity)
                        let u_truth = if truthful_winner == agent { true_val } else { 0.0 };
                        let u_lie = if lie_winner == agent { true_val } else { 0.0 };

                        if u_lie > u_truth + 1e-10 {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

// ─── Auctions ─────────────────────────────────────────────────────────────────

/// Vickrey (second-price sealed-bid) auction.
/// Returns (winner_index, price_paid).
pub fn vickrey_winner(bids: &[f64]) -> (usize, f64) {
    assert!(!bids.is_empty(), "No bids");
    let winner = bids
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();

    let second_highest = bids
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != winner)
        .map(|(_, &v)| v)
        .fold(f64::NEG_INFINITY, f64::max);

    let price = if second_highest == f64::NEG_INFINITY { 0.0 } else { second_highest };
    (winner, price)
}

/// Symmetric Bayesian Nash Equilibrium bid function for first-price auction.
/// Values are i.i.d. Uniform[0, value_upper].
/// b(v) = v * (n-1) / n
pub fn first_price_ne_symmetric(n: usize, value_upper: f64) -> Box<dyn Fn(f64) -> f64> {
    let n_f = n as f64;
    let _value_upper = value_upper;
    Box::new(move |v: f64| v * (n_f - 1.0) / n_f)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn prisoner_dilemma() -> NormalFormGame {
        // 2 players, 2 strategies each: 0=Cooperate, 1=Defect
        // Payoff matrix:
        //         C    D
        //   C  (3,3) (0,5)
        //   D  (5,0) (1,1)
        let payoffs = vec![
            vec![3.0, 0.0, 5.0, 1.0], // player 0 payoffs
            vec![3.0, 5.0, 0.0, 1.0], // player 1 payoffs
        ];
        NormalFormGame::new(2, vec![2, 2], payoffs)
    }

    #[test]
    fn test_prisoner_dilemma_unique_pure_ne() {
        let g = prisoner_dilemma();
        let ne = g.find_pure_nash();
        assert_eq!(ne.len(), 1, "Prisoner's dilemma has exactly one pure NE");
        assert_eq!(ne[0], vec![1, 1], "The unique NE is (Defect, Defect)");
    }

    #[test]
    fn test_strategy_profile_index_roundtrip() {
        let counts = vec![3, 4, 2];
        for idx in 0..24 {
            let profile = decode_profile(idx, &counts);
            let back = strategy_profile_index(&profile, &counts);
            assert_eq!(back, idx);
        }
    }

    #[test]
    fn test_nim_xor() {
        assert_eq!(nim_value(&[3, 5, 6]), 3 ^ 5 ^ 6);
        assert_eq!(nim_value(&[1, 2, 3]), 0); // losing position
        assert!(nim_is_losing_position(&[1, 2, 3]));
        assert!(!nim_is_losing_position(&[1, 2, 4]));
    }

    #[test]
    fn test_nim_single_pile() {
        // Single pile of size k: winning iff k > 0
        assert!(!nim_is_losing_position(&[5]));
        assert!(nim_is_losing_position(&[0]));
    }

    #[test]
    fn test_grundy_subtraction_game() {
        // Subtraction game with moves {1,2}: Grundy values cycle 0,1,2,0,1,2,...
        let moves = vec![1, 2];
        assert_eq!(grundy_value(0, &moves), 0);
        assert_eq!(grundy_value(1, &moves), 1);
        assert_eq!(grundy_value(2, &moves), 2);
        assert_eq!(grundy_value(3, &moves), 0);
        assert_eq!(grundy_value(4, &moves), 1);
    }

    #[test]
    fn test_mex() {
        assert_eq!(mex(&[1, 2, 3]), 0);
        assert_eq!(mex(&[0, 1, 3]), 2);
        assert_eq!(mex(&[]), 0);
        assert_eq!(mex(&[0, 1, 2, 3]), 4);
    }

    #[test]
    fn test_shapley_value_three_player_majority() {
        // 3-player majority game: v(S) = 1 if |S| >= 2, else 0
        let mut v = HashMap::new();
        for mask in 0u64..8 {
            let size = mask.count_ones() as usize;
            v.insert(mask, if size >= 2 { 1.0 } else { 0.0 });
        }
        let game = CooperativeGame::new(3, v);
        let phi = game.shapley_value();
        // By symmetry, each player gets 1/3
        for p in &phi {
            assert!((p - 1.0 / 3.0).abs() < 1e-10, "Shapley value: expected 1/3, got {}", p);
        }
    }

    #[test]
    fn test_shapley_value_sum_equals_grand() {
        let mut v = HashMap::new();
        v.insert(0b001u64, 2.0);
        v.insert(0b010u64, 3.0);
        v.insert(0b100u64, 1.0);
        v.insert(0b011u64, 6.0);
        v.insert(0b101u64, 4.0);
        v.insert(0b110u64, 5.0);
        v.insert(0b111u64, 8.0);
        let game = CooperativeGame::new(3, v);
        let phi = game.shapley_value();
        let sum: f64 = phi.iter().sum();
        assert!((sum - 8.0).abs() < 1e-10, "Shapley values sum to grand coalition value");
    }

    #[test]
    fn test_vickrey_auction() {
        let bids = vec![1.0, 3.0, 2.5, 0.5];
        let (winner, price) = vickrey_winner(&bids);
        assert_eq!(winner, 1, "Highest bidder wins");
        assert!((price - 2.5).abs() < 1e-10, "Price is second-highest bid");
    }

    #[test]
    fn test_vickrey_truth_telling() {
        // In Vickrey, truth-telling is dominant strategy.
        // Lowering bid only loses or ties, never gains.
        let bids = vec![5.0, 3.0, 2.0];
        let (w1, p1) = vickrey_winner(&bids);
        assert_eq!(w1, 0);
        assert!((p1 - 3.0).abs() < 1e-10);

        // If winner misreports lower than second-highest → loses
        let mut bids2 = bids.clone();
        bids2[0] = 2.5;
        let (w2, _) = vickrey_winner(&bids2);
        assert_ne!(w2, 0, "Under-bidding below second-highest loses");
    }

    #[test]
    fn test_saddle_point_detection() {
        // Matrix with saddle point at (1,1) value 2
        let m = ZeroSumGame::new(vec![vec![3.0, 2.0], vec![1.0, 0.0]]);
        // row-mins: [2, 0], col-maxes: [3, 2]
        // maximin = 2, minimax = 2 → saddle at (0,1)
        let sp = m.has_saddle_point();
        assert!(sp.is_some(), "Should have saddle point");
        let (r, c) = sp.unwrap();
        assert!((m.matrix[r][c] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_saddle_point_matching_pennies() {
        // Matching pennies: [[1,-1],[-1,1]] — no pure saddle
        let m = ZeroSumGame::new(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]);
        assert!(m.has_saddle_point().is_none());
    }

    #[test]
    fn test_first_price_ne_bid_below_value() {
        let bid_fn = first_price_ne_symmetric(2, 1.0);
        let v = 0.6f64;
        let b = bid_fn(v);
        assert!(b < v, "BNE bid is below true value");
        assert!((b - 0.3).abs() < 1e-10, "2-player: bid = v/2");
    }

    #[test]
    fn test_superadditivity() {
        // Additive game: v(S) = sum of individual values → superadditive
        let mut v = HashMap::new();
        let vals = [1.0, 2.0, 3.0];
        for mask in 0u64..8 {
            let val: f64 =
                (0..3).filter(|&i| mask & (1 << i) != 0).map(|i| vals[i]).sum();
            v.insert(mask, val);
        }
        let game = CooperativeGame::new(3, v);
        assert!(game.is_superadditive());
    }

    #[test]
    fn test_strictly_dominated() {
        // Rock-Paper-Scissors has no strictly dominated strategies
        let payoffs = vec![
            vec![0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0],
            vec![0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0],
        ];
        let g = NormalFormGame::new(2, vec![3, 3], payoffs);
        for p in 0..2 {
            let dom = g.strictly_dominated_strategies(p);
            assert!(dom.is_empty(), "RPS has no strictly dominated strategies");
        }
    }

    #[test]
    fn test_fictitious_play_matching_pennies_value() {
        // Matching pennies value should be ≈ 0
        let m = ZeroSumGame::new(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]);
        let (value, _rp, _cp) = m.mixed_strategy_value_lp();
        assert!(value.abs() < 0.1, "Matching pennies value ≈ 0, got {}", value);
    }
}
