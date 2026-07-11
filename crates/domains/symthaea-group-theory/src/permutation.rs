// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Permutations of `{0, …, n−1}` — the elements of the symmetric group `Sₙ`.

/// A permutation in one-line notation: `map[i]` is the image of `i`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permutation {
    map: Vec<usize>,
}

impl Permutation {
    /// The identity permutation on `n` points.
    pub fn identity(n: usize) -> Permutation {
        Permutation {
            map: (0..n).collect(),
        }
    }

    /// Build from one-line notation; `Err` if it is not a bijection of `0..n`.
    pub fn from_one_line(map: Vec<usize>) -> Result<Permutation, String> {
        let n = map.len();
        let mut seen = vec![false; n];
        for &v in &map {
            if v >= n || seen[v] {
                return Err("not a permutation of 0..n".to_string());
            }
            seen[v] = true;
        }
        Ok(Permutation { map })
    }

    /// The number of points.
    pub fn degree(&self) -> usize {
        self.map.len()
    }

    /// The image of `i`.
    pub fn apply(&self, i: usize) -> usize {
        self.map[i]
    }

    /// Composition `self ∘ other`: apply `other` first, then `self`.
    pub fn compose(&self, other: &Permutation) -> Permutation {
        Permutation {
            map: (0..self.degree()).map(|i| self.map[other.map[i]]).collect(),
        }
    }

    /// The inverse permutation.
    pub fn inverse(&self) -> Permutation {
        let mut inv = vec![0; self.degree()];
        for (i, &v) in self.map.iter().enumerate() {
            inv[v] = i;
        }
        Permutation { map: inv }
    }

    /// The disjoint cycle decomposition (each cycle as a list of points; fixed
    /// points are omitted).
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let n = self.degree();
        let mut seen = vec![false; n];
        let mut cycles = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut x = start;
            while !seen[x] {
                seen[x] = true;
                cycle.push(x);
                x = self.map[x];
            }
            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }
        cycles
    }

    /// The order: the least `k ≥ 1` with `self^k = identity` (the lcm of the
    /// cycle lengths).
    pub fn order(&self) -> usize {
        self.cycles().iter().map(|c| c.len()).fold(1, lcm)
    }

    /// The sign (parity): `+1` for an even permutation, `−1` for an odd one.
    /// A cycle of length `ℓ` contributes parity `(−1)^{ℓ−1}`.
    pub fn sign(&self) -> i32 {
        let odd_cycles = self.cycles().iter().filter(|c| c.len() % 2 == 0).count();
        if odd_cycles % 2 == 0 { 1 } else { -1 }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_and_inverse() {
        let p = Permutation::from_one_line(vec![1, 2, 0]).unwrap(); // 3-cycle (0 1 2)
        let id = Permutation::identity(3);
        assert_eq!(p.compose(&p.inverse()), id);
        // p∘p∘p = identity (order 3).
        assert_eq!(p.compose(&p).compose(&p), id);
    }

    #[test]
    fn cycles_and_order() {
        // (0 1)(2 3 4): a transposition and a 3-cycle → order lcm(2,3)=6.
        let p = Permutation::from_one_line(vec![1, 0, 3, 4, 2]).unwrap();
        assert_eq!(p.order(), 6);
        let cyc = p.cycles();
        assert_eq!(cyc.len(), 2);
    }

    #[test]
    fn sign_of_transposition_and_three_cycle() {
        // A transposition is odd.
        let t = Permutation::from_one_line(vec![1, 0, 2]).unwrap();
        assert_eq!(t.sign(), -1);
        // A 3-cycle is even.
        let c = Permutation::from_one_line(vec![1, 2, 0]).unwrap();
        assert_eq!(c.sign(), 1);
        assert_eq!(Permutation::identity(5).sign(), 1);
    }

    #[test]
    fn rejects_non_permutation() {
        assert!(Permutation::from_one_line(vec![0, 0, 1]).is_err());
        assert!(Permutation::from_one_line(vec![0, 1, 3]).is_err());
    }
}
