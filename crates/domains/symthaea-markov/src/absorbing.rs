// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Absorbing Markov chains: expected steps to absorption and absorption
//! probabilities, from the fundamental matrix N = (I − Q)⁻¹.
//!
//! A state `i` is **absorbing** when `P[i][i] = 1` (once entered, never left).
//! The remaining **transient** states are analysed by splitting the reordered
//! transition matrix into `Q` (transient→transient) and `R` (transient→
//! absorbing). We avoid forming N explicitly by solving the relevant linear
//! systems: expected steps solve `(I − Q) t = 1`, and absorption probabilities
//! solve `(I − Q) B = R`. The linear algebra is delegated to `symthaea-linalg`
//! (shared LU solver) rather than a private Gaussian elimination.

use symthaea_linalg::Matrix;
use symthaea_linalg::decomp::lu_decompose;

/// A transition matrix split for absorbing-chain analysis.
pub struct Absorbing {
    /// Indices of absorbing states, ascending.
    pub absorbing: Vec<usize>,
    /// Indices of transient states, ascending (the order of `t` / rows of `b`).
    pub transient: Vec<usize>,
}

/// Classify the states of a transition matrix. A row `i` is absorbing iff
/// `p[i][i] == 1`.
pub fn classify(p: &[Vec<f64>]) -> Absorbing {
    let n = p.len();
    let mut absorbing = Vec::new();
    let mut transient = Vec::new();
    for i in 0..n {
        if (p[i][i] - 1.0).abs() < 1e-12 {
            absorbing.push(i);
        } else {
            transient.push(i);
        }
    }
    Absorbing {
        absorbing,
        transient,
    }
}

/// Build `I − Q` for the transient states.
fn i_minus_q(p: &[Vec<f64>], transient: &[usize]) -> Vec<Vec<f64>> {
    let t = transient.len();
    let mut m = vec![vec![0.0; t]; t];
    for (r, &i) in transient.iter().enumerate() {
        for (c, &j) in transient.iter().enumerate() {
            m[r][c] = if r == c { 1.0 } else { 0.0 } - p[i][j];
        }
    }
    m
}

/// Expected number of steps until absorption, starting from each transient
/// state (aligned to `classify().transient`). `None` if the system is singular
/// (e.g. a transient state that cannot actually reach an absorbing one).
pub fn expected_steps_to_absorption(p: &[Vec<f64>]) -> Option<Vec<f64>> {
    let cls = classify(p);
    if cls.transient.is_empty() {
        return Some(Vec::new());
    }
    let a = Matrix::from_rows(i_minus_q(p, &cls.transient)).ok()?;
    let ones = vec![1.0; cls.transient.len()];
    symthaea_linalg::solve(&a, &ones)
}

/// Absorption probabilities `b[k][a]` = probability of ending in the `a`-th
/// absorbing state (aligned to `classify().absorbing`) starting from the `k`-th
/// transient state. `None` if singular.
pub fn absorption_probabilities(p: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let cls = classify(p);
    let (t, m) = (cls.transient.len(), cls.absorbing.len());
    if t == 0 {
        return Some(Vec::new());
    }
    let iq = Matrix::from_rows(i_minus_q(p, &cls.transient)).ok()?;
    // Factor (I − Q) once and reuse it for every absorbing column of R.
    let lu = lu_decompose(&iq)?;
    let mut result = vec![vec![0.0; m]; t];
    for (ac, &a_state) in cls.absorbing.iter().enumerate() {
        let r_col: Vec<f64> = cls.transient.iter().map(|&i| p[i][a_state]).collect();
        let x = lu.solve(&r_col)?;
        for (k, &val) in x.iter().enumerate() {
            result[k][ac] = val;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symmetric random walk on {0,1,2,3,4} with 0 and 4 absorbing — gambler's
    /// ruin, N = 4. Transient {1,2,3} each step ±1 with prob 0.5.
    fn gamblers_ruin() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0], // 0 absorbing
            vec![0.5, 0.0, 0.5, 0.0, 0.0], // 1
            vec![0.0, 0.5, 0.0, 0.5, 0.0], // 2
            vec![0.0, 0.0, 0.5, 0.0, 0.5], // 3
            vec![0.0, 0.0, 0.0, 0.0, 1.0], // 4 absorbing
        ]
    }

    #[test]
    fn classify_absorbing_and_transient() {
        let cls = classify(&gamblers_ruin());
        assert_eq!(cls.absorbing, vec![0, 4]);
        assert_eq!(cls.transient, vec![1, 2, 3]);
    }

    #[test]
    fn expected_steps_closed_form() {
        // For symmetric gambler's ruin on N steps, E[steps from i] = i·(N − i).
        // N=4: state1→3, state2→4, state3→3.
        let t = expected_steps_to_absorption(&gamblers_ruin()).unwrap();
        assert!((t[0] - 3.0).abs() < 1e-9, "{t:?}"); // from state 1
        assert!((t[1] - 4.0).abs() < 1e-9); // from state 2
        assert!((t[2] - 3.0).abs() < 1e-9); // from state 3
    }

    #[test]
    fn absorption_probabilities_closed_form() {
        // P(reach right absorber 4 | start i) = i/N. Columns are [state0, state4].
        let b = absorption_probabilities(&gamblers_ruin()).unwrap();
        // From state 1: 3/4 to state 0, 1/4 to state 4.
        assert!((b[0][0] - 0.75).abs() < 1e-9, "{b:?}");
        assert!((b[0][1] - 0.25).abs() < 1e-9);
        // From state 2: 1/2 each.
        assert!((b[1][0] - 0.5).abs() < 1e-9);
        // From state 3: 1/4 to state 0, 3/4 to state 4.
        assert!((b[2][1] - 0.75).abs() < 1e-9);
    }
}
