// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discrete-time Markov chains: state distributions over time and the
//! stationary distribution.

/// A discrete-time Markov chain given by a row-stochastic transition matrix
/// `p[i][j] = P(next = j | current = i)`.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    p: Vec<Vec<f64>>,
    n: usize,
}

impl MarkovChain {
    /// Construct from a transition matrix. Returns `Err` if it is not square or
    /// a row does not sum to 1 (within 1e-9).
    pub fn new(p: Vec<Vec<f64>>) -> Result<MarkovChain, String> {
        let n = p.len();
        if n == 0 {
            return Err("empty transition matrix".to_string());
        }
        for (i, row) in p.iter().enumerate() {
            if row.len() != n {
                return Err(format!("row {i} has length {} != {n}", row.len()));
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-9 {
                return Err(format!("row {i} sums to {sum}, not 1"));
            }
            if row.iter().any(|&x| x < -1e-12) {
                return Err(format!("row {i} has a negative probability"));
            }
        }
        Ok(MarkovChain { p, n })
    }

    /// Number of states.
    pub fn size(&self) -> usize {
        self.n
    }

    /// The transition matrix.
    pub fn transition(&self) -> &[Vec<f64>] {
        &self.p
    }

    /// Advance a distribution one step: `next = dist · P`.
    pub fn step(&self, dist: &[f64]) -> Vec<f64> {
        let mut next = vec![0.0; self.n];
        for i in 0..self.n {
            if dist[i] == 0.0 {
                continue;
            }
            for j in 0..self.n {
                next[j] += dist[i] * self.p[i][j];
            }
        }
        next
    }

    /// The distribution after `steps` transitions from `initial`.
    pub fn distribution_after(&self, initial: &[f64], steps: usize) -> Vec<f64> {
        let mut d = initial.to_vec();
        for _ in 0..steps {
            d = self.step(&d);
        }
        d
    }

    /// The stationary distribution π (πP = π), found by power iteration from a
    /// uniform start. For an irreducible aperiodic chain this converges; for a
    /// chain with several recurrent classes it returns the limit of the uniform
    /// start.
    pub fn stationary_distribution(&self, iterations: usize) -> Vec<f64> {
        let mut d = vec![1.0 / self.n as f64; self.n];
        for _ in 0..iterations {
            let next = self.step(&d);
            let delta: f64 = next.iter().zip(&d).map(|(a, b)| (a - b).abs()).sum();
            d = next;
            if delta < 1e-15 {
                break;
            }
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_matrix() {
        assert!(MarkovChain::new(vec![vec![0.5, 0.4]]).is_err()); // row sums to 0.9
        assert!(MarkovChain::new(vec![vec![1.0, 0.0], vec![0.3, 0.3]]).is_err());
    }

    #[test]
    fn two_state_stationary() {
        // P = [[0.9,0.1],[0.5,0.5]]. Stationary π = (5/6, 1/6).
        let c = MarkovChain::new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
        let pi = c.stationary_distribution(1000);
        assert!((pi[0] - 5.0 / 6.0).abs() < 1e-9, "{pi:?}");
        assert!((pi[1] - 1.0 / 6.0).abs() < 1e-9);
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn distribution_converges_to_stationary() {
        let c = MarkovChain::new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
        let d = c.distribution_after(&[1.0, 0.0], 200);
        assert!((d[0] - 5.0 / 6.0).abs() < 1e-6, "{d:?}");
    }
}
