// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-combinatorics
//!
//! Foundational counting: factorials, permutations & combinations, Catalan and
//! Stirling numbers, integer partitions, and derangements. Complements
//! `symthaea-analytic-number-theory` (which is about primes/divisibility) and
//! the discrete side of `symthaea-graph-theory`.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Counts return `u128`
//! (exact). `factorial` overflows past 34!; `combinations` uses an
//! overflow-resistant multiplicative form and stays exact far longer. Every
//! function is checked against a known value.
//!
//! ## Example
//!
//! ```
//! use symthaea_combinatorics::{combinations, catalan};
//! assert_eq!(combinations(5, 2), 10);  // "5 choose 2"
//! assert_eq!(catalan(4), 14);          // Catalan number C₄
//! ```

/// `n!` — the factorial. Overflows `u128` past 34!.
pub fn factorial(n: u64) -> u128 {
    (1..=n as u128).product()
}

/// Number of ordered arrangements `P(n, k) = n·(n−1)···(n−k+1)`. `0` if `k > n`.
pub fn permutations(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    (0..k).map(|i| (n - i) as u128).product()
}

/// The binomial coefficient `C(n, k)` ("n choose k") via an exact multiplicative
/// form that avoids computing large factorials. `0` if `k > n`.
pub fn combinations(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k); // symmetry: C(n,k) = C(n,n-k)
    let mut result: u128 = 1;
    for i in 0..k {
        result = result * (n - i) as u128 / (i as u128 + 1);
    }
    result
}

/// The multinomial coefficient `(Σkᵢ)! / ∏kᵢ!` — the number of distinct
/// arrangements of a multiset with the given group sizes.
pub fn multinomial(groups: &[u64]) -> u128 {
    // Build up as a product of binomials to stay exact: C(n, k1)·C(n-k1, k2)···
    let mut remaining: u64 = groups.iter().sum();
    let mut result: u128 = 1;
    for &g in groups {
        result *= combinations(remaining, g);
        remaining -= g;
    }
    result
}

/// The `n`-th Catalan number `Cₙ = C(2n, n)/(n+1)`.
pub fn catalan(n: u64) -> u128 {
    combinations(2 * n, n) / (n as u128 + 1)
}

/// Stirling number of the second kind `S(n, k)` — the number of ways to
/// partition an `n`-element set into `k` non-empty unlabeled subsets.
pub fn stirling_second(n: u64, k: u64) -> u128 {
    let (n, k) = (n as usize, k as usize);
    if k == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if k > n {
        return 0;
    }
    // DP over the recurrence S(n,k) = k·S(n−1,k) + S(n−1,k−1).
    let mut prev = vec![0u128; k + 1];
    prev[0] = 1;
    for i in 1..=n {
        let mut cur = vec![0u128; k + 1];
        for j in 1..=k.min(i) {
            cur[j] = j as u128 * prev[j] + prev[j - 1];
        }
        prev = cur;
    }
    prev[k]
}

/// The number of integer partitions of `n` (unordered sums of positive
/// integers). `partitions(0) = 1`.
pub fn partitions(n: u64) -> u128 {
    let n = n as usize;
    let mut ways = vec![0u128; n + 1];
    ways[0] = 1;
    for part in 1..=n {
        for j in part..=n {
            ways[j] += ways[j - part];
        }
    }
    ways[n]
}

/// The number of derangements `!n` — permutations with no fixed point.
pub fn derangements(n: u64) -> u128 {
    match n {
        0 => 1,
        1 => 0,
        _ => {
            let (mut a, mut b) = (1u128, 0u128); // D(0), D(1)
            for i in 2..=n as u128 {
                let d = (i - 1) * (b + a);
                a = b;
                b = d;
            }
            b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorials_and_arrangements() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(permutations(5, 2), 20);
        assert_eq!(permutations(5, 0), 1);
        assert_eq!(permutations(3, 5), 0);
    }

    #[test]
    fn binomials() {
        assert_eq!(combinations(5, 2), 10);
        assert_eq!(combinations(10, 5), 252);
        assert_eq!(combinations(52, 5), 2_598_960); // poker hands
        assert_eq!(combinations(5, 0), 1);
        assert_eq!(combinations(3, 5), 0);
        // Pascal: C(n,k) = C(n-1,k-1) + C(n-1,k).
        assert_eq!(combinations(9, 4), combinations(8, 3) + combinations(8, 4));
    }

    #[test]
    fn multinomial_mississippi() {
        // Arrangements of MISSISSIPPI: 11! / (1!·4!·4!·2!) = 34650.
        assert_eq!(multinomial(&[1, 4, 4, 2]), 34650);
    }

    #[test]
    fn catalan_stirling_partitions_derangements() {
        // Catalan: 1,1,2,5,14,42.
        assert_eq!(catalan(0), 1);
        assert_eq!(catalan(4), 14);
        assert_eq!(catalan(5), 42);
        // Stirling 2nd: S(4,2)=7, S(n,n)=1, S(n,1)=1.
        assert_eq!(stirling_second(4, 2), 7);
        assert_eq!(stirling_second(5, 5), 1);
        assert_eq!(stirling_second(6, 1), 1);
        // Integer partitions: p(5)=7, p(0)=1.
        assert_eq!(partitions(0), 1);
        assert_eq!(partitions(5), 7);
        assert_eq!(partitions(10), 42);
        // Derangements: !4 = 9, !5 = 44.
        assert_eq!(derangements(4), 9);
        assert_eq!(derangements(5), 44);
    }
}
