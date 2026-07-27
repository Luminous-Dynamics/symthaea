// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diophantine equation solvers.
//!
//! Phase 1 of the IMO roadmap. Provides:
//! - Pell equation x² − D·y² = 1 via continued fraction expansion of √D
//! - General solution recurrence from the fundamental solution
//!
//! The continued-fraction algorithm is the classical one (see Hardy & Wright,
//! "An Introduction to the Theory of Numbers", Ch. 10): expand √D as
//! [a₀; a₁, a₂, …] and walk the convergents h_k/k_k until h_k² − D·k_k² = 1
//! (equivalently, until the period of the expansion ends on an even index).

/// Result of solving x² − D·y² = 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PellSolution {
    /// D in x² − Dy² = 1.
    pub d: i64,
    /// Fundamental solution (x₁, y₁): smallest positive (x,y) with y > 0.
    pub fundamental: (i128, i128),
}

impl PellSolution {
    /// Returns the n-th positive solution (x_n, y_n) generated from the
    /// recurrence
    ///
    /// ```text
    /// x_{n+1} = x₁·x_n + D·y₁·y_n
    /// y_{n+1} = x₁·y_n + y₁·x_n
    /// ```
    ///
    /// with (x₁, y₁) the fundamental solution. `n ≥ 1`.
    pub fn nth(&self, n: usize) -> (i128, i128) {
        assert!(n >= 1, "Pell solutions are 1-indexed");
        let (x1, y1) = self.fundamental;
        let d = self.d as i128;
        let mut xn = x1;
        let mut yn = y1;
        for _ in 1..n {
            let xn1 = x1 * xn + d * y1 * yn;
            let yn1 = x1 * yn + y1 * xn;
            xn = xn1;
            yn = yn1;
        }
        (xn, yn)
    }

    /// Verify a claimed solution: returns true iff x² − D·y² = 1.
    pub fn verify(&self, x: i128, y: i128) -> bool {
        x * x - (self.d as i128) * y * y == 1
    }
}

/// Solve the Pell equation x² − D·y² = 1 for positive squarefree-ish D.
///
/// Returns `Some(PellSolution)` with the fundamental solution, or `None` if
/// D ≤ 0 or D is a perfect square (in which case the equation has only the
/// trivial solution (±1, 0)).
///
/// Uses continued-fraction expansion of √D. For every D that is not a perfect
/// square, √D has a purely periodic continued fraction after the initial a₀,
/// and the fundamental solution appears at the first convergent whose index
/// completes an even multiple of the period.
pub fn pell_equation(d: i64) -> Option<PellSolution> {
    if d <= 0 {
        return None;
    }
    let sqrt_d = (d as f64).sqrt() as i64;
    if sqrt_d * sqrt_d == d {
        return None; // perfect square ⇒ no nontrivial solution
    }

    // Continued fraction of √D: a_k = ⌊(m_k + √D) / d_k⌋
    // with recurrences
    //   m_{k+1} = d_k·a_k − m_k
    //   d_{k+1} = (D − m_{k+1}²) / d_k
    //   a_{k+1} = ⌊(a₀ + m_{k+1}) / d_{k+1}⌋
    let mut m: i64 = 0;
    let mut den: i64 = 1;
    let a0 = sqrt_d;
    let mut a = a0;

    // Convergents h_k / k_k from recurrence
    //   h_k = a_k·h_{k-1} + h_{k-2},  h_{-1}=1, h_{-2}=0
    //   k_k = a_k·k_{k-1} + k_{k-2},  k_{-1}=0, k_{-2}=1
    let mut h_prev2: i128 = 0;
    let mut h_prev1: i128 = 1;
    let mut k_prev2: i128 = 1;
    let mut k_prev1: i128 = 0;

    // Track whether we've taken an even number of a-steps past a₀.
    // The fundamental solution of x²−Dy²=1 appears at:
    //   period even → index (period − 1) of the a_1, a_2, ... sequence
    //   period odd  → index (2·period − 1)
    // We don't need to know the period up front — we just check each convergent.
    for _ in 0..200_000 {
        // Compute next convergent with overflow protection. If any step
        // overflows i128, we've exceeded the solver's numeric range and
        // return None rather than panic.
        let h = (a as i128)
            .checked_mul(h_prev1)
            .and_then(|v| v.checked_add(h_prev2))?;
        let k = (a as i128)
            .checked_mul(k_prev1)
            .and_then(|v| v.checked_add(k_prev2))?;
        let h2 = h.checked_mul(h)?;
        let k2 = k.checked_mul(k)?;
        let dk2 = (d as i128).checked_mul(k2)?;
        if h2 - dk2 == 1 && k > 0 {
            return Some(PellSolution {
                d,
                fundamental: (h, k),
            });
        }
        h_prev2 = h_prev1;
        h_prev1 = h;
        k_prev2 = k_prev1;
        k_prev1 = k;

        // Next continued-fraction term
        m = den * a - m;
        den = (d - m * m) / den;
        if den == 0 {
            break;
        }
        a = (a0 + m) / den;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pell_d2() {
        // x² − 2y² = 1  → fundamental (3, 2)
        let sol = pell_equation(2).unwrap();
        assert_eq!(sol.fundamental, (3, 2));
        assert!(sol.verify(3, 2));
    }

    #[test]
    fn test_pell_d3() {
        // x² − 3y² = 1  → fundamental (2, 1)
        let sol = pell_equation(3).unwrap();
        assert_eq!(sol.fundamental, (2, 1));
    }

    #[test]
    fn test_pell_d5() {
        // x² − 5y² = 1  → fundamental (9, 4)
        let sol = pell_equation(5).unwrap();
        assert_eq!(sol.fundamental, (9, 4));
    }

    #[test]
    fn test_pell_d7() {
        // x² − 7y² = 1  → fundamental (8, 3)
        let sol = pell_equation(7).unwrap();
        assert_eq!(sol.fundamental, (8, 3));
    }

    #[test]
    fn test_pell_d13() {
        // x² − 13y² = 1  → fundamental (649, 180)
        let sol = pell_equation(13).unwrap();
        assert_eq!(sol.fundamental, (649, 180));
    }

    #[test]
    fn test_pell_d61_famously_large() {
        // Fermat's challenge: x² − 61y² = 1 has fundamental
        // (1766319049, 226153980). A classical test of any Pell implementation.
        let sol = pell_equation(61).unwrap();
        assert_eq!(sol.fundamental, (1_766_319_049, 226_153_980));
        assert!(sol.verify(1_766_319_049, 226_153_980));
    }

    #[test]
    fn test_pell_nth_solution() {
        let sol = pell_equation(2).unwrap();
        // (3,2) → (17,12) → (99,70) → ...
        assert_eq!(sol.nth(1), (3, 2));
        assert_eq!(sol.nth(2), (17, 12));
        assert_eq!(sol.nth(3), (99, 70));
        for n in 1..=6 {
            let (x, y) = sol.nth(n);
            assert!(sol.verify(x, y));
        }
    }

    #[test]
    fn test_pell_perfect_square() {
        // D = 4: no nontrivial solutions
        assert!(pell_equation(4).is_none());
        assert!(pell_equation(9).is_none());
        assert!(pell_equation(25).is_none());
    }

    #[test]
    fn test_pell_nonpositive() {
        assert!(pell_equation(0).is_none());
        assert!(pell_equation(-3).is_none());
    }
}
