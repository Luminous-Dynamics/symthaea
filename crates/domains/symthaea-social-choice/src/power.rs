// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Voting-power indices for weighted voting games — how much *influence* a
//! player really has, which can differ sharply from their raw weight.
//!
//! A game is a list of integer `weights` and a `quota`: a coalition wins iff its
//! total weight ≥ quota. Both indices enumerate over subsets, so keep the number
//! of players modest (≲ 20).

/// n! as an f64 (exact for n ≤ 18, fine as a ratio denominator beyond that).
fn factorial(n: usize) -> f64 {
    (1..=n).map(|x| x as f64).product()
}

/// The **normalized Banzhaf index**: each player's share of the coalitions in
/// which they are *critical* (their departure flips a win to a loss).
pub fn banzhaf(weights: &[u64], quota: u64) -> Vec<f64> {
    let n = weights.len();
    let mut critical = vec![0u64; n];
    for mask in 0u32..(1u32 << n) {
        let sum: u64 = (0..n)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| weights[i])
            .sum();
        if sum >= quota {
            for i in 0..n {
                if mask & (1 << i) != 0 && sum - weights[i] < quota {
                    critical[i] += 1;
                }
            }
        }
    }
    let total: u64 = critical.iter().sum();
    if total == 0 {
        return vec![0.0; n];
    }
    critical.iter().map(|&c| c as f64 / total as f64).collect()
}

/// The **Shapley-Shubik index**: the fraction of orderings in which a player is
/// *pivotal* (the voter whose arrival first makes a growing coalition win).
/// Computed via the equivalent subset formula, so no n! enumeration is needed.
pub fn shapley_shubik(weights: &[u64], quota: u64) -> Vec<f64> {
    let n = weights.len();
    let nf = factorial(n);
    let mut idx = vec![0.0; n];
    for i in 0..n {
        let others: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        let k = others.len();
        for mask in 0u32..(1u32 << k) {
            let mut sum = 0u64;
            let mut size = 0usize;
            for (b, &j) in others.iter().enumerate() {
                if mask & (1 << b) != 0 {
                    sum += weights[j];
                    size += 1;
                }
            }
            // Coalition S (without i) loses, but S ∪ {i} wins ⇒ i pivots for
            // the |S|!·(n−|S|−1)! orderings with S before i and the rest after.
            if sum < quota && sum + weights[i] >= quota {
                idx[i] += factorial(size) * factorial(n - size - 1);
            }
        }
        idx[i] /= nf;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-9, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn banzhaf_weighted_triple() {
        // Weights [3,2,1], quota 4. Player 0 is critical in every winning
        // coalition → 0.6, others 0.2 each.
        close(&banzhaf(&[3, 2, 1], 4), &[0.6, 0.2, 0.2]);
    }

    #[test]
    fn shapley_weighted_triple() {
        // Same game: player 0 pivots in 4 of 6 orderings → 2/3, others 1/6.
        close(
            &shapley_shubik(&[3, 2, 1], 4),
            &[4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        );
    }

    #[test]
    fn indices_sum_to_one() {
        for &(w, q) in &[
            (&[4u64, 3, 2, 1][..], 6u64),
            (&[10, 10, 10, 1][..], 16),
            (&[5, 4, 3, 2, 1][..], 8),
        ] {
            let bz: f64 = banzhaf(w, q).iter().sum();
            let ss: f64 = shapley_shubik(w, q).iter().sum();
            assert!((bz - 1.0).abs() < 1e-9, "banzhaf sum {bz}");
            assert!((ss - 1.0).abs() < 1e-9, "shapley sum {ss}");
        }
    }

    #[test]
    fn dummy_player_has_zero_power() {
        // In [3,1] with quota 3, player 1 never matters.
        let bz = banzhaf(&[3, 1], 3);
        assert!((bz[1]).abs() < 1e-12);
        let ss = shapley_shubik(&[3, 1], 3);
        assert!((ss[1]).abs() < 1e-12);
    }
}
