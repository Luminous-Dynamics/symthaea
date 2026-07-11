// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-order-theory
//!
//! Finite partial orders and lattices — the mathematics under abstract
//! interpretation and dataflow analysis (relevant to Symthaea's own code
//! reasoning), and a clean home for meet/join and fixed-point theory.
//!
//! A [`Poset`] on elements `0..n` is built from its **cover relations** (the
//! Hasse diagram edges `a ⋖ b`); the reflexive-transitive closure gives `≤`.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against known
//! lattices (divisibility, powerset).
//!
//! ## Example
//!
//! ```
//! use symthaea_order_theory::Poset;
//! // Divisors of 12 under divisibility: meet = gcd, join = lcm.
//! let p = Poset::divisibility(12);
//! let (four, six) = (p.index_of(4).unwrap(), p.index_of(6).unwrap());
//! assert_eq!(p.value(p.meet(four, six).unwrap()), 2); // gcd(4,6)
//! assert_eq!(p.value(p.join(four, six).unwrap()), 12); // lcm(4,6)
//! assert!(p.is_lattice());
//! ```

/// A finite partial order on elements `0..n`. Optionally each element carries an
/// integer `label` (used by the divisibility constructor so callers can talk in
/// terms of the actual divisors).
#[derive(Debug, Clone)]
pub struct Poset {
    n: usize,
    leq: Vec<Vec<bool>>,
    labels: Vec<i64>,
}

impl Poset {
    /// Build from cover relations `a ⋖ b` on `n` elements. The reflexive and
    /// transitive closure is computed automatically. Labels default to the
    /// index.
    pub fn from_covers(n: usize, covers: &[(usize, usize)]) -> Poset {
        let mut leq = vec![vec![false; n]; n];
        for i in 0..n {
            leq[i][i] = true;
        }
        for &(a, b) in covers {
            leq[a][b] = true;
        }
        // Warshall transitive closure.
        for k in 0..n {
            for i in 0..n {
                if leq[i][k] {
                    for j in 0..n {
                        if leq[k][j] {
                            leq[i][j] = true;
                        }
                    }
                }
            }
        }
        Poset {
            n,
            leq,
            labels: (0..n as i64).collect(),
        }
    }

    /// The divisibility lattice of the divisors of `m`: element per divisor,
    /// `a ≤ b` iff `a | b`.
    pub fn divisibility(m: i64) -> Poset {
        let divisors: Vec<i64> = (1..=m).filter(|d| m % d == 0).collect();
        let n = divisors.len();
        let mut leq = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                leq[i][j] = divisors[j] % divisors[i] == 0;
            }
        }
        Poset {
            n,
            leq,
            labels: divisors,
        }
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Whether `a ≤ b`.
    pub fn leq(&self, a: usize, b: usize) -> bool {
        self.leq[a][b]
    }

    /// The integer label of an element.
    pub fn value(&self, a: usize) -> i64 {
        self.labels[a]
    }

    /// The element index whose label is `v`, if any.
    pub fn index_of(&self, v: i64) -> Option<usize> {
        self.labels.iter().position(|&x| x == v)
    }

    /// The meet (greatest lower bound) of `a` and `b`, if it exists and is
    /// unique.
    pub fn meet(&self, a: usize, b: usize) -> Option<usize> {
        // Common lower bounds.
        let lowers: Vec<usize> = (0..self.n)
            .filter(|&x| self.leq[x][a] && self.leq[x][b])
            .collect();
        // The glb is a lower bound above every other lower bound.
        lowers
            .iter()
            .copied()
            .find(|&g| lowers.iter().all(|&x| self.leq[x][g]))
    }

    /// The join (least upper bound) of `a` and `b`, if it exists and is unique.
    pub fn join(&self, a: usize, b: usize) -> Option<usize> {
        let uppers: Vec<usize> = (0..self.n)
            .filter(|&x| self.leq[a][x] && self.leq[b][x])
            .collect();
        uppers
            .iter()
            .copied()
            .find(|&l| uppers.iter().all(|&x| self.leq[l][x]))
    }

    /// The bottom element (`⊥ ≤` everything), if any.
    pub fn bottom(&self) -> Option<usize> {
        (0..self.n).find(|&x| (0..self.n).all(|y| self.leq[x][y]))
    }

    /// The top element (everything `≤ ⊤`), if any.
    pub fn top(&self) -> Option<usize> {
        (0..self.n).find(|&x| (0..self.n).all(|y| self.leq[y][x]))
    }

    /// Whether every pair has both a meet and a join (i.e. this is a lattice).
    pub fn is_lattice(&self) -> bool {
        (0..self.n)
            .all(|a| (0..self.n).all(|b| self.meet(a, b).is_some() && self.join(a, b).is_some()))
    }

    /// Whether the lattice is distributive:
    /// `a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)` for all triples.
    pub fn is_distributive(&self) -> bool {
        if !self.is_lattice() {
            return false;
        }
        for a in 0..self.n {
            for b in 0..self.n {
                for c in 0..self.n {
                    let lhs = self.meet(a, self.join(b, c).unwrap());
                    let rhs = self.join(self.meet(a, b).unwrap(), self.meet(a, c).unwrap());
                    if lhs != rhs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// The least fixed point of a monotone map `f` (Knaster-Tarski), by ascending
    /// Kleene iteration from `⊥`. `f` must be monotone; `None` if there is no
    /// bottom element.
    pub fn least_fixed_point(&self, f: impl Fn(usize) -> usize) -> Option<usize> {
        let mut x = self.bottom()?;
        loop {
            let next = f(x);
            if next == x {
                return Some(x);
            }
            x = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisibility_lattice_of_12() {
        let p = Poset::divisibility(12); // {1,2,3,4,6,12}
        assert_eq!(p.size(), 6);
        let idx = |v| p.index_of(v).unwrap();
        // meet = gcd, join = lcm.
        assert_eq!(p.value(p.meet(idx(4), idx(6)).unwrap()), 2);
        assert_eq!(p.value(p.join(idx(4), idx(6)).unwrap()), 12);
        assert_eq!(p.value(p.meet(idx(2), idx(3)).unwrap()), 1);
        assert_eq!(p.value(p.bottom().unwrap()), 1);
        assert_eq!(p.value(p.top().unwrap()), 12);
        assert!(p.is_lattice());
        assert!(p.is_distributive()); // divisibility lattices are distributive
    }

    #[test]
    fn diamond_m3_is_a_nondistributive_lattice() {
        // M₃: ⊥(0) < a(1),b(2),c(3) < ⊤(4). A lattice, but not distributive.
        let p = Poset::from_covers(5, &[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]);
        assert!(p.is_lattice());
        assert!(!p.is_distributive());
        // a ∧ b = ⊥, a ∨ b = ⊤.
        assert_eq!(p.meet(1, 2), Some(0));
        assert_eq!(p.join(1, 2), Some(4));
    }

    #[test]
    fn non_lattice_has_no_unique_bounds() {
        // Two incomparable minimal elements below two incomparable maximal ones
        // (the "N"/2x2 antichain-ish) — some pair lacks a unique join.
        let p = Poset::from_covers(4, &[(0, 2), (0, 3), (1, 2), (1, 3)]);
        // 0 and 1 have two upper bounds (2,3) with no least → no join.
        assert_eq!(p.join(0, 1), None);
        assert!(!p.is_lattice());
    }

    #[test]
    fn knaster_tarski_least_fixed_point() {
        // Chain ⊥=0 < 1 < 2 < 3=⊤ (divisors of 8: 1,2,4,8). f(x)=min(x+1, top)
        // is monotone; its least fixed point is the top.
        let p = Poset::divisibility(8); // {1,2,4,8} → indices 0<1<2<3
        let top = p.top().unwrap();
        let lfp = p.least_fixed_point(|x| (x + 1).min(top)).unwrap();
        assert_eq!(lfp, top);
        // The identity map's least fixed point is ⊥.
        assert_eq!(p.least_fixed_point(|x| x).unwrap(), p.bottom().unwrap());
    }
}
