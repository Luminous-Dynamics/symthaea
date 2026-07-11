// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Entropy and the information measures derived from it, all in **bits**
//! (base-2 logarithm). Convention `0·log 0 = 0`.

fn plogp(p: f64) -> f64 {
    if p <= 0.0 { 0.0 } else { -p * p.log2() }
}

/// Shannon entropy `H(X) = −Σ p·log₂ p` of a probability distribution (bits).
pub fn entropy(probs: &[f64]) -> f64 {
    probs.iter().map(|&p| plogp(p)).sum()
}

/// The binary entropy function `H(p) = −p·log p − (1−p)·log(1−p)`.
pub fn binary_entropy(p: f64) -> f64 {
    plogp(p) + plogp(1.0 - p)
}

/// Joint entropy `H(X, Y)` of a joint distribution `joint[x][y]`.
pub fn joint_entropy(joint: &[Vec<f64>]) -> f64 {
    joint.iter().flatten().map(|&p| plogp(p)).sum()
}

/// The marginal distribution of `X` (row sums).
pub fn marginal_x(joint: &[Vec<f64>]) -> Vec<f64> {
    joint.iter().map(|row| row.iter().sum()).collect()
}

/// The marginal distribution of `Y` (column sums).
pub fn marginal_y(joint: &[Vec<f64>]) -> Vec<f64> {
    if joint.is_empty() {
        return Vec::new();
    }
    let cols = joint[0].len();
    (0..cols)
        .map(|j| joint.iter().map(|row| row[j]).sum())
        .collect()
}

/// Conditional entropy `H(Y | X) = H(X, Y) − H(X)`.
pub fn conditional_entropy_y_given_x(joint: &[Vec<f64>]) -> f64 {
    joint_entropy(joint) - entropy(&marginal_x(joint))
}

/// Mutual information `I(X; Y) = H(X) + H(Y) − H(X, Y)` (bits, ≥ 0).
pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    entropy(&marginal_x(joint)) + entropy(&marginal_y(joint)) - joint_entropy(joint)
}

/// Kullback-Leibler divergence `D(p ‖ q) = Σ p·log(p/q)` (bits). `q` must have
/// support wherever `p` does; `None` otherwise.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    let mut sum = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return None; // p not absolutely continuous w.r.t. q
            }
            sum += pi * (pi / qi).log2();
        }
    }
    Some(sum)
}

/// Cross-entropy `H(p, q) = −Σ p·log q` (bits).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.len() != q.len() {
        return None;
    }
    let mut sum = 0.0;
    for (&pi, &qi) in p.iter().zip(q) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return None;
            }
            sum -= pi * qi.log2();
        }
    }
    Some(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_landmarks() {
        assert!((entropy(&[0.5, 0.5]) - 1.0).abs() < 1e-12); // fair coin = 1 bit
        assert!(entropy(&[1.0, 0.0]).abs() < 1e-12); // certainty = 0
        assert!((entropy(&[0.25; 4]) - 2.0).abs() < 1e-12); // uniform over 4 = 2 bits
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-12);
        assert!(binary_entropy(0.0).abs() < 1e-12);
    }

    #[test]
    fn mutual_information_extremes() {
        // Independent X, Y → I = 0.
        let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        assert!(mutual_information(&indep).abs() < 1e-12);
        // Perfectly correlated → I = H(X) = 1 bit.
        let corr = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        assert!((mutual_information(&corr) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mutual_information_identities() {
        let joint = vec![vec![0.3, 0.1], vec![0.2, 0.4]];
        let hx = entropy(&marginal_x(&joint));
        let hy_given_x = conditional_entropy_y_given_x(&joint);
        let hy = entropy(&marginal_y(&joint));
        // I(X;Y) = H(Y) − H(Y|X).
        assert!((mutual_information(&joint) - (hy - hy_given_x)).abs() < 1e-12);
        // Chain rule: H(X,Y) = H(X) + H(Y|X).
        assert!((joint_entropy(&joint) - (hx + hy_given_x)).abs() < 1e-12);
    }

    #[test]
    fn kl_and_cross_entropy() {
        let p = [0.5, 0.5];
        assert!(kl_divergence(&p, &p).unwrap().abs() < 1e-12); // D(p‖p) = 0
        // Cross-entropy H(p,q) = H(p) + D(p‖q).
        let q = [0.25, 0.75];
        let lhs = cross_entropy(&p, &q).unwrap();
        let rhs = entropy(&p) + kl_divergence(&p, &q).unwrap();
        assert!((lhs - rhs).abs() < 1e-12);
        // KL is non-negative.
        assert!(kl_divergence(&p, &q).unwrap() >= 0.0);
        assert!(kl_divergence(&p, &[1.0, 0.0]).is_none()); // q lacks support
    }
}
