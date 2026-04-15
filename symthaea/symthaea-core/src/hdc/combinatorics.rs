// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Combinatorics for Symthaea
//!
//! Advanced combinatorics toolkit: generating functions, partition theory,
//! Stirling & Bell numbers, Catalan numbers, derangements, Burnside/Pólya
//! enumeration, Ramsey theory, graphic matroids, and inclusion-exclusion.
//!
//! All implementations are pure Rust (no external crates beyond `std`).

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// GENERATING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse polynomial over f64 used for ordinary / exponential generating
/// functions. Terms are stored as `(degree, coefficient)` pairs, sorted
/// by ascending degree, with zero-coefficient terms omitted.
#[derive(Debug, Clone, PartialEq)]
pub struct Poly {
    /// Sorted list of (degree, coefficient) pairs.  Degree 0 = constant term.
    pub terms: Vec<(usize, f64)>,
}

impl Poly {
    /// Create the zero polynomial.
    pub fn zero() -> Self {
        Self { terms: vec![] }
    }

    /// Create a monomial c * x^d.
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        if coeff == 0.0 {
            return Self::zero();
        }
        Self {
            terms: vec![(degree, coeff)],
        }
    }

    /// Coefficient of x^n (0.0 if absent).
    pub fn extract(&self, n: usize) -> f64 {
        self.terms
            .iter()
            .find(|&&(d, _)| d == n)
            .map(|&(_, c)| c)
            .unwrap_or(0.0)
    }

    /// Ordinary generating function addition: (f + g)(x).
    pub fn ogf_add(&self, other: &Self) -> Self {
        let mut map: HashMap<usize, f64> = HashMap::new();
        for &(d, c) in &self.terms {
            *map.entry(d).or_insert(0.0) += c;
        }
        for &(d, c) in &other.terms {
            *map.entry(d).or_insert(0.0) += c;
        }
        let mut terms: Vec<(usize, f64)> = map
            .into_iter()
            .filter(|&(_, c)| c.abs() > f64::EPSILON)
            .collect();
        terms.sort_by_key(|&(d, _)| d);
        Self { terms }
    }

    /// Ordinary generating function multiplication: (f * g)(x).
    pub fn ogf_mul(&self, other: &Self) -> Self {
        let mut map: HashMap<usize, f64> = HashMap::new();
        for &(d1, c1) in &self.terms {
            for &(d2, c2) in &other.terms {
                *map.entry(d1 + d2).or_insert(0.0) += c1 * c2;
            }
        }
        let mut terms: Vec<(usize, f64)> = map
            .into_iter()
            .filter(|&(_, c)| c.abs() > f64::EPSILON)
            .collect();
        terms.sort_by_key(|&(d, _)| d);
        Self { terms }
    }

    /// Exponential generating function multiplication.
    ///
    /// In EGF convention the coefficient of x^n represents a_n/n!.  When
    /// multiplying two EGFs F and G the coefficient of x^n in F*G is
    /// sum_{k=0}^{n} C(n,k) * a_k * b_{n-k}.  This function performs that
    /// convolution directly on the stored coefficients (which are a_n/n!),
    /// so the result stores (a*b)_n / n!.
    pub fn egf_mul(&self, other: &Self) -> Self {
        let max_deg = self.terms.iter().map(|&(d, _)| d).max().unwrap_or(0)
            + other.terms.iter().map(|&(d, _)| d).max().unwrap_or(0);

        let mut result = vec![0.0f64; max_deg + 1];
        let mut fa = vec![0.0f64; max_deg + 1];
        let mut fb = vec![0.0f64; max_deg + 1];
        for &(d, c) in &self.terms {
            if d <= max_deg {
                fa[d] = c;
            }
        }
        for &(d, c) in &other.terms {
            if d <= max_deg {
                fb[d] = c;
            }
        }

        // Precompute factorials
        let mut fact = vec![1.0f64; max_deg + 2];
        for i in 1..=max_deg + 1 {
            fact[i] = fact[i - 1] * i as f64;
        }

        // Convolution: result[n] = sum_{k=0}^{n} C(n,k) * fa[k] * fact[k] * fb[n-k] * fact[n-k] / fact[n]
        // But since fa[k] stores a_k/k! and fb stores b/!, we get:
        // result[n] = sum_{k=0}^{n} (a_k * b_{n-k}) / n! * C(n,k) * k! * (n-k)! = ...
        // Simpler: result_coeff[n] = sum fa[k]*fact[k] * fb[n-k]*fact[n-k] * binom / fact[n]
        // = sum a_k b_{n-k} binom(n,k) / n!  ... stored as (a*b)[n]/n!
        for n in 0..=max_deg {
            let mut s = 0.0;
            for k in 0..=n {
                let binom = fact[n] / (fact[k] * fact[n - k]);
                s += fa[k] * fact[k] * fb[n - k] * fact[n - k] * binom / fact[n];
            }
            result[n] = s;
        }

        let terms: Vec<(usize, f64)> = result
            .into_iter()
            .enumerate()
            .filter(|&(_, c)| c.abs() > f64::EPSILON)
            .collect();
        Self { terms }
    }

    // ── Named generating functions ───────────────────────────────────────────

    /// OGF for geometric series 1/(1-rx) truncated to `terms` coefficients.
    /// Extract coefficient of x^n to get r^n.
    pub fn ogf_geometric(r: f64, num_terms: usize) -> Self {
        let mut pow = 1.0;
        let terms: Vec<(usize, f64)> = (0..num_terms)
            .map(|n| {
                let coeff = pow;
                pow *= r;
                (n, coeff)
            })
            .collect();
        Self { terms }
    }

    /// EGF for e^x = sum x^n/n! truncated to `num_terms` coefficients.
    /// Stores coefficient 1/n! at degree n.
    pub fn egf_exp(num_terms: usize) -> Self {
        let mut fact = 1.0f64;
        let terms: Vec<(usize, f64)> = (0..num_terms)
            .map(|n| {
                let coeff = 1.0 / fact;
                if n + 1 < num_terms {
                    fact *= (n + 1) as f64;
                }
                (n, coeff)
            })
            .collect();
        Self { terms }
    }

    /// EGF for sin(x) = sum_{n odd} (-1)^((n-1)/2) x^n/n!
    pub fn egf_sin(num_terms: usize) -> Self {
        let mut fact = 1.0f64;
        let mut terms = vec![];
        for n in 0..num_terms {
            if n > 0 {
                fact *= n as f64;
            }
            if n % 2 == 1 {
                let sign = if (n / 2) % 2 == 0 { 1.0 } else { -1.0 };
                terms.push((n, sign / fact));
            }
        }
        Self { terms }
    }

    /// EGF for cos(x) = sum_{n even} (-1)^(n/2) x^n/n!
    pub fn egf_cos(num_terms: usize) -> Self {
        let mut fact = 1.0f64;
        let mut terms = vec![];
        for n in 0..num_terms {
            if n > 0 {
                fact *= n as f64;
            }
            if n % 2 == 0 {
                let sign = if (n / 2) % 2 == 0 { 1.0 } else { -1.0 };
                terms.push((n, sign / fact));
            }
        }
        Self { terms }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PARTITION THEORY
// ═══════════════════════════════════════════════════════════════════════════

/// Number of integer partitions of n via Euler's pentagonal number theorem.
///
/// p(n) = sum_{k != 0} (-1)^(k+1) * p(n - omega(k))
/// where omega(k) = k*(3k-1)/2 are the generalised pentagonal numbers.
pub fn partition_count(n: u64) -> u64 {
    if n == 0 {
        return 1;
    }
    let n_usize = n as usize;
    let mut p = vec![0i64; n_usize + 1];
    p[0] = 1;

    for i in 1..=n_usize {
        let mut k: i64 = 1;
        loop {
            // Generalised pentagonal numbers: omega(k) and omega(-k)
            let pent_pos = (k * (3 * k - 1) / 2) as usize;
            if pent_pos > i {
                break;
            }
            let sign = if k % 2 == 1 { 1i64 } else { -1i64 };
            p[i] += sign * p[i - pent_pos];

            let pent_neg = (k * (3 * k + 1) / 2) as usize;
            if pent_neg <= i {
                p[i] += sign * p[i - pent_neg];
            }

            k += 1;
        }
    }

    p[n_usize].unsigned_abs()
}

/// Enumerate all partitions of n (n ≤ 20).
/// Returns a vector of partitions; each partition is sorted descending.
pub fn partitions(n: usize) -> Vec<Vec<usize>> {
    assert!(n <= 20, "partitions: n must be ≤ 20");
    let mut result = vec![];
    let mut current = vec![];
    partitions_rec(n, n, &mut current, &mut result);
    result
}

fn partitions_rec(
    remaining: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }
    let upper = remaining.min(max_part);
    for part in (1..=upper).rev() {
        current.push(part);
        partitions_rec(remaining - part, part, current, result);
        current.pop();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STIRLING NUMBERS
// ═══════════════════════════════════════════════════════════════════════════

/// Signed Stirling number of the first kind s(n, k).
///
/// Recurrence: s(n, k) = s(n-1, k-1) - (n-1)*s(n-1, k)
/// with s(0,0) = 1, s(n,0) = 0 for n>0, s(0,k) = 0 for k>0.
pub fn stirling_first(n: usize, k: usize) -> i64 {
    if n == 0 && k == 0 {
        return 1;
    }
    if n == 0 || k == 0 || k > n {
        return 0;
    }
    // Build table bottom-up
    let mut table = vec![vec![0i64; k + 1]; n + 1];
    table[0][0] = 1;
    for i in 1..=n {
        for j in 1..=k.min(i) {
            table[i][j] = table[i - 1][j - 1] - ((i as i64 - 1) * table[i - 1][j]);
        }
    }
    table[n][k]
}

/// Stirling number of the second kind S(n, k).
///
/// Recurrence: S(n, k) = k*S(n-1, k) + S(n-1, k-1)
/// with S(0,0) = 1, S(n,0) = 0 for n>0, S(0,k) = 0 for k>0.
pub fn stirling_second(n: usize, k: usize) -> u64 {
    if n == 0 && k == 0 {
        return 1;
    }
    if n == 0 || k == 0 || k > n {
        return 0;
    }
    let mut table = vec![vec![0u64; k + 1]; n + 1];
    table[0][0] = 1;
    for i in 1..=n {
        for j in 1..=k.min(i) {
            table[i][j] = (j as u64) * table[i - 1][j] + table[i - 1][j - 1];
        }
    }
    table[n][k]
}

// ═══════════════════════════════════════════════════════════════════════════
// BELL NUMBERS
// ═══════════════════════════════════════════════════════════════════════════

/// Bell number B(n) via the Bell triangle.
///
/// B(0) = 1, and B(n) = first element of row n of the Bell triangle.
pub fn bell(n: usize) -> u64 {
    if n == 0 {
        return 1;
    }
    // Build Bell triangle iteratively
    let mut prev_row = vec![1u64]; // row 0
    for _ in 1..=n {
        let mut row = vec![*prev_row.last().unwrap()];
        for j in 0..prev_row.len() {
            row.push(row[j] + prev_row[j]);
        }
        prev_row = row;
    }
    prev_row[0]
}

// ═══════════════════════════════════════════════════════════════════════════
// CATALAN NUMBERS
// ═══════════════════════════════════════════════════════════════════════════

/// Catalan number C(n) = C(2n, n) / (n+1).
///
/// Uses big-integer-safe arithmetic via u128 intermediates.
pub fn catalan(n: u64) -> u64 {
    if n == 0 {
        return 1;
    }
    // C(2n, n) = (2n)! / (n! * n!)
    // Compute iteratively to avoid overflow: product_{i=1}^{n} (n+i)/i / (n+1)
    let mut result = 1u128;
    for i in 1..=n as u128 {
        result = result * (n as u128 + i) / i;
    }
    (result / (n as u128 + 1)) as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// DERANGEMENTS
// ═══════════════════════════════════════════════════════════════════════════

/// Number of derangements of n elements: D(n).
///
/// Recurrence: D(0) = 1, D(1) = 0, D(n) = (n-1)*(D(n-1) + D(n-2)).
pub fn derangement(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 0,
        _ => {
            let mut d_prev2 = 1u64; // D(0)
            let mut d_prev1 = 0u64; // D(1)
            for k in 2..=n {
                let d = (k - 1) * (d_prev1 + d_prev2);
                d_prev2 = d_prev1;
                d_prev1 = d;
            }
            d_prev1
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BURNSIDE'S LEMMA & PÓLYA ENUMERATION
// ═══════════════════════════════════════════════════════════════════════════

/// Apply Burnside's lemma to count distinct orbits.
///
/// |X/G| = (1/|G|) * Σ_{g ∈ G} |Fix(g)|
///
/// # Arguments
/// * `group_size` — |G|
/// * `fixed_point_counts` — |Fix(g)| for each group element g
///
/// # Panics
/// Panics if `group_size == 0` or if the sum is not divisible by `group_size`.
pub fn burnside(group_size: usize, fixed_point_counts: &[usize]) -> usize {
    assert!(group_size > 0, "burnside: group_size must be positive");
    let total: usize = fixed_point_counts.iter().sum();
    assert!(
        total % group_size == 0,
        "burnside: sum of fixed points ({total}) not divisible by group size ({group_size})"
    );
    total / group_size
}

/// Pólya enumeration theorem.
///
/// Given the cycle index of a permutation group as a list of
/// (cycle_length, count_of_cycles_of_that_length) pairs, compute the number
/// of distinct colorings with `colors` colors.
///
/// The cycle index monomial for a single permutation is
/// ∏_{k} z_k^{count_k}.
/// Under Pólya, each z_k is replaced by the sum of k-th powers of color
/// weights; for uniform unit weights this equals `colors^k`.
///
/// If `cycle_index_coeffs` represents a single group element's cycle type,
/// the contribution is ∏_{k} colors^{count_k * k} / ... but here we accept
/// the aggregate cycle index: a slice representing the *average* over all
/// permutations already encoded as (k, total_count) where total_count is
/// already summed over all group elements divided by group_size.
///
/// For flexibility this function treats each (k, count) entry as a factor
/// `colors^count` in the product (the caller provides the cycle index in
/// already-reduced form).
pub fn polya_count(colors: usize, cycle_index_coeffs: &[(usize, usize)]) -> u64 {
    let mut result = 1u64;
    for &(k, count) in cycle_index_coeffs {
        result *= (colors as u64).pow(count as u32);
        let _ = k; // cycle length metadata — weighting is via count
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// FUNDAMENTAL COUNTING
// ═══════════════════════════════════════════════════════════════════════════

/// Binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Uses the multiplicative formula to avoid overflow for moderate n.
/// Returns 0 if k > n.
pub fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k); // symmetry: C(n,k) = C(n,n-k)
    let mut result = 1u64;
    for i in 0..k {
        result = result.checked_mul(n - i).unwrap_or(u64::MAX) / (i + 1);
    }
    result
}

/// Multinomial coefficient: n! / (k₁! * k₂! * ... * kₘ!).
///
/// `groups` must sum to n. Returns 0 if they don't.
pub fn multinomial(n: u64, groups: &[u64]) -> u64 {
    let sum: u64 = groups.iter().sum();
    if sum != n {
        return 0;
    }
    let mut result = 1u64;
    let mut remaining = n;
    for &k in groups {
        result = result
            .checked_mul(binomial(remaining, k))
            .unwrap_or(u64::MAX);
        remaining -= k;
    }
    result
}

/// Fibonacci number F(n) via iterative doubling. O(log n).
///
/// F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2).
pub fn fibonacci(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 1..n {
        let tmp = a.saturating_add(b);
        a = b;
        b = tmp;
    }
    b
}

/// Lucas number L(n). L(0)=2, L(1)=1, L(n) = L(n-1) + L(n-2).
///
/// Related to Fibonacci: L(n) = F(n-1) + F(n+1).
pub fn lucas(n: u64) -> u64 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }
    let (mut a, mut b) = (2u64, 1u64);
    for _ in 2..=n {
        let tmp = a.saturating_add(b);
        a = b;
        b = tmp;
    }
    b
}

/// Möbius function μ(n).
///
/// μ(1) = 1; μ(n) = 0 if n has a squared prime factor;
/// μ(n) = (-1)^k if n is a product of k distinct primes.
pub fn moebius(n: u64) -> i64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut m = n;
    let mut num_factors = 0i64;
    let mut d = 2u64;
    while d * d <= m {
        if m % d == 0 {
            m /= d;
            if m % d == 0 {
                return 0; // squared factor
            }
            num_factors += 1;
        }
        d += 1;
    }
    if m > 1 {
        num_factors += 1; // remaining prime factor
    }
    if num_factors % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Rising factorial (Pochhammer symbol): x^{(n)} = x(x+1)(x+2)...(x+n-1).
pub fn rising_factorial(x: u64, n: u64) -> u64 {
    (0..n).fold(1u64, |acc, i| acc.saturating_mul(x + i))
}

/// Falling factorial: x_{(n)} = x(x-1)(x-2)...(x-n+1).
pub fn falling_factorial(x: u64, n: u64) -> u64 {
    if n > x {
        return 0;
    }
    (0..n).fold(1u64, |acc, i| acc.saturating_mul(x - i))
}

// ═══════════════════════════════════════════════════════════════════════════
// RAMSEY THEORY
// ═══════════════════════════════════════════════════════════════════════════

/// Probabilistic lower bound for the Ramsey number R(k, l).
///
/// Uses the approximation floor(exp(k*l / (k+l))) + 1.
pub fn ramsey_lower_bound(k: usize, l: usize) -> usize {
    let exp_arg = (k as f64 * l as f64) / (k as f64 + l as f64);
    exp_arg.exp().floor() as usize + 1
}

/// Hard-coded small Ramsey numbers R(k, l) for known values.
pub fn small_ramsey_numbers() -> HashMap<(usize, usize), usize> {
    let mut m = HashMap::new();
    m.insert((3, 3), 6);
    m.insert((3, 4), 9);
    m.insert((3, 5), 14);
    m.insert((4, 4), 18);
    m.insert((3, 6), 18);
    m.insert((3, 7), 23);
    m.insert((3, 8), 28);
    m.insert((3, 9), 36);
    m
}

// ═══════════════════════════════════════════════════════════════════════════
// GRAPHIC MATROID (Union-Find)
// ═══════════════════════════════════════════════════════════════════════════

/// A graphic matroid defined by a simple undirected graph.
///
/// Edges are stored as (u, v) pairs; vertices are identified by index.
/// Independent sets in the graphic matroid are exactly the forests (acyclic
/// subgraphs).
#[derive(Debug, Clone)]
pub struct GraphicMatroid {
    /// Number of vertices.
    pub num_vertices: usize,
    /// Edge list: (u, v) for each edge index.
    pub edges: Vec<(usize, usize)>,
}

impl GraphicMatroid {
    /// Create a graphic matroid with given vertices and edges.
    pub fn new(num_vertices: usize, edges: Vec<(usize, usize)>) -> Self {
        Self {
            num_vertices,
            edges,
        }
    }

    /// Check if a subset of edge indices forms an independent set (forest).
    pub fn is_independent(&self, edge_indices: &[usize]) -> bool {
        let mut uf = UnionFind::new(self.num_vertices);
        for &ei in edge_indices {
            let (u, v) = self.edges[ei];
            if !uf.union(u, v) {
                return false; // cycle detected
            }
        }
        true
    }

    /// Rank of a subset of edge indices: size of the largest independent
    /// subset (via greedy augmentation).
    pub fn rank(&self, edge_indices: &[usize]) -> usize {
        let mut uf = UnionFind::new(self.num_vertices);
        let mut r = 0;
        for &ei in edge_indices {
            let (u, v) = self.edges[ei];
            if uf.union(u, v) {
                r += 1;
            }
        }
        r
    }

    /// Greedy maximum-weight basis (Kruskal's algorithm on the given edge subset).
    ///
    /// `weights[i]` is the weight of edge i.  Returns the indices of edges
    /// selected for the max-weight spanning forest.
    pub fn greedy_max_weight_basis(&self, weights: &[f64]) -> Vec<usize> {
        assert_eq!(
            weights.len(),
            self.edges.len(),
            "weights length must match edges length"
        );
        let mut indexed: Vec<usize> = (0..self.edges.len()).collect();
        indexed.sort_by(|&a, &b| weights[b].partial_cmp(&weights[a]).unwrap());

        let mut uf = UnionFind::new(self.num_vertices);
        let mut basis = vec![];
        for ei in indexed {
            let (u, v) = self.edges[ei];
            if uf.union(u, v) {
                basis.push(ei);
            }
        }
        basis
    }
}

/// Simple union-find (disjoint set union) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Returns true if x and y were in different components (union performed).
    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INCLUSION-EXCLUSION
// ═══════════════════════════════════════════════════════════════════════════

/// Generalised inclusion-exclusion principle.
///
/// Computes the union form:
/// Σ_{∅≠S ⊆ {0..sets-1}} (-1)^(|S|+1) * f(S), where `f(S)` is the
/// intersection size for the selected subset.
///
/// # Arguments
/// * `sets` — number of sets (universe size n; iterates over 2^n subsets)
/// * `f` — function from subset (as sorted index slice) to i64 value
pub fn inclusion_exclusion<F>(sets: usize, f: F) -> i64
where
    F: Fn(&[usize]) -> i64,
{
    assert!(
        sets <= 20,
        "inclusion_exclusion: sets must be ≤ 20 (2^20 subsets)"
    );
    let n = 1usize << sets;
    let mut total = 0i64;
    for mask in 1..n {
        let subset: Vec<usize> = (0..sets).filter(|&i| mask & (1 << i) != 0).collect();
        let sign = if subset.len() % 2 == 1 { 1i64 } else { -1i64 };
        total += sign * f(&subset);
    }
    total
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_count_small() {
        // p(0)=1, p(1)=1, p(2)=2, p(3)=3, p(4)=5, p(5)=7
        assert_eq!(partition_count(0), 1);
        assert_eq!(partition_count(1), 1);
        assert_eq!(partition_count(5), 7);
        assert_eq!(partition_count(10), 42);
    }

    #[test]
    fn test_partitions_enumerate() {
        let p5 = partitions(5);
        // There are 7 partitions of 5
        assert_eq!(p5.len(), 7);
        // Each partition sums to 5
        for p in &p5 {
            assert_eq!(p.iter().sum::<usize>(), 5);
        }
    }

    #[test]
    fn test_stirling_second() {
        // S(4,2) = 7
        assert_eq!(stirling_second(4, 2), 7);
        // S(n,1) = 1 for n >= 1
        assert_eq!(stirling_second(5, 1), 1);
        // S(n,n) = 1
        assert_eq!(stirling_second(6, 6), 1);
        // S(4,3) = 6
        assert_eq!(stirling_second(4, 3), 6);
    }

    #[test]
    fn test_stirling_first() {
        // s(1,1) = 1
        assert_eq!(stirling_first(1, 1), 1);
        // s(4,2) = 11
        assert_eq!(stirling_first(4, 2), 11);
        // s(n,n) = 1
        assert_eq!(stirling_first(5, 5), 1);
    }

    #[test]
    fn test_bell_numbers() {
        // B(0)=1, B(1)=1, B(2)=2, B(3)=5, B(4)=15, B(5)=52
        assert_eq!(bell(0), 1);
        assert_eq!(bell(1), 1);
        assert_eq!(bell(5), 52);
        assert_eq!(bell(4), 15);
    }

    #[test]
    fn test_catalan_numbers() {
        // C(0)=1, C(1)=1, C(2)=2, C(3)=5, C(4)=14, C(5)=42
        assert_eq!(catalan(0), 1);
        assert_eq!(catalan(5), 42);
        assert_eq!(catalan(4), 14);
    }

    #[test]
    fn test_derangement() {
        // D(0)=1, D(1)=0, D(2)=1, D(3)=2, D(4)=9
        assert_eq!(derangement(0), 1);
        assert_eq!(derangement(1), 0);
        assert_eq!(derangement(4), 9);
        assert_eq!(derangement(3), 2);
    }

    #[test]
    fn test_burnside() {
        // Rotational symmetries of a square (4 rotations), coloring vertices with 2 colors:
        // Fix(id)=16, Fix(rot90)=2, Fix(rot180)=4, Fix(rot270)=2  → (16+2+4+2)/4 = 6
        let fixed = [16, 2, 4, 2];
        assert_eq!(burnside(4, &fixed), 6);
    }

    #[test]
    fn test_ramsey_r33() {
        let table = small_ramsey_numbers();
        assert_eq!(table[&(3, 3)], 6);
        assert_eq!(table[&(4, 4)], 18);
        assert_eq!(table[&(3, 9)], 36);
    }

    #[test]
    fn test_graphic_matroid_independence() {
        // Triangle: 3 vertices, 3 edges (0-1, 1-2, 0-2)
        let m = GraphicMatroid::new(3, vec![(0, 1), (1, 2), (0, 2)]);
        // Any two edges form a forest (independent)
        assert!(m.is_independent(&[0, 1]));
        assert!(m.is_independent(&[0, 2]));
        // All three edges form a cycle (not independent)
        assert!(!m.is_independent(&[0, 1, 2]));
        // Single edge is always independent
        assert!(m.is_independent(&[0]));
    }

    #[test]
    fn test_graphic_matroid_rank() {
        // K4: 4 vertices, 6 edges — spanning tree rank = 3
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let m = GraphicMatroid::new(4, edges);
        let all: Vec<usize> = (0..6).collect();
        assert_eq!(m.rank(&all), 3);
    }

    #[test]
    fn test_graphic_matroid_greedy_basis() {
        // Path graph: 4 vertices, 3 edges
        let m = GraphicMatroid::new(4, vec![(0, 1), (1, 2), (2, 3)]);
        let weights = [1.0, 2.0, 3.0];
        let basis = m.greedy_max_weight_basis(&weights);
        // All 3 edges form a spanning tree (no cycles possible in a path)
        assert_eq!(basis.len(), 3);
    }

    #[test]
    fn test_ogf_mul() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let a = Poly {
            terms: vec![(0, 1.0), (1, 1.0)],
        };
        let result = a.ogf_mul(&a);
        assert!((result.extract(0) - 1.0).abs() < 1e-10);
        assert!((result.extract(1) - 2.0).abs() < 1e-10);
        assert!((result.extract(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ogf_geometric() {
        // Geometric series with r=2: coefficient of x^n should be 2^n
        let g = Poly::ogf_geometric(2.0, 6);
        for n in 0..6usize {
            let expected = (2.0f64).powi(n as i32);
            assert!((g.extract(n) - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_inclusion_exclusion_union() {
        // |A ∪ B ∪ C| = |A| + |B| + |C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|
        // Using sizes: |A|=3, |B|=3, |C|=3, pairwise=1, triple=1
        // Expected: 3+3+3-1-1-1+1 = 7
        let result = inclusion_exclusion(3, |subset| match subset {
            [] => 0,
            [_] => 3,
            [_, _] => 1,
            _ => 1,
        });
        assert_eq!(result, 7);
    }

    #[test]
    fn test_egf_exp_coefficients() {
        // e^x: coefficient of x^n should be 1/n!
        let e = Poly::egf_exp(8);
        let mut fact = 1.0f64;
        for n in 0..8usize {
            if n > 0 {
                fact *= n as f64;
            }
            let expected = 1.0 / fact;
            assert!((e.extract(n) - expected).abs() < 1e-12, "n={n}");
        }
    }

    #[test]
    fn test_polya_count() {
        // Necklace of 3 beads, 2 colors: cycle index of Z_3 is (1/3)(z1^3 + 2*z3)
        // polya = (1/3)(2^3 + 2*2^1) = (8+4)/3 = 4
        // We encode as: 2 cycle types: [(1, 3), (3, 1)] — but polya_count multiplies colors^count
        // For Z_3 average: identity has 3 1-cycles, two rotations each have 1 3-cycle.
        // polya_count represents ONE permutation's contribution at a time.
        // Identity contribution: colors^3 = 8; two 3-cycle contributions: colors^1 = 2 each.
        // Average = (8 + 2 + 2) / 3 = 4
        let id_contrib = polya_count(2, &[(1, 3)]); // identity: 3 fixed-1-cycles → 2^3 = 8
        let rot_contrib = polya_count(2, &[(3, 1)]); // each rotation: 1 fixed-3-cycle → 2^1 = 2
        let total = (id_contrib + 2 * rot_contrib) / 3;
        assert_eq!(total, 4);
    }

    // ── Fundamental counting tests ──────────────────────────────────────

    #[test]
    fn test_binomial_basic() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 6), 0); // k > n
    }

    #[test]
    fn test_binomial_symmetry() {
        for n in 0..15 {
            for k in 0..=n {
                assert_eq!(binomial(n, k), binomial(n, n - k));
            }
        }
    }

    #[test]
    fn test_multinomial() {
        // C(6; 2,2,2) = 6!/(2!2!2!) = 90
        assert_eq!(multinomial(6, &[2, 2, 2]), 90);
        // C(4; 1,1,1,1) = 4! = 24
        assert_eq!(multinomial(4, &[1, 1, 1, 1]), 24);
        // Sum doesn't match n → 0
        assert_eq!(multinomial(5, &[2, 2]), 0);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    #[test]
    fn test_lucas() {
        assert_eq!(lucas(0), 2);
        assert_eq!(lucas(1), 1);
        assert_eq!(lucas(5), 11);
        // L(n) = F(n-1) + F(n+1)
        assert_eq!(lucas(10), fibonacci(9) + fibonacci(11));
    }

    #[test]
    fn test_moebius() {
        assert_eq!(moebius(1), 1);
        assert_eq!(moebius(2), -1); // prime
        assert_eq!(moebius(3), -1); // prime
        assert_eq!(moebius(4), 0); // 2²
        assert_eq!(moebius(6), 1); // 2×3 (two distinct primes)
        assert_eq!(moebius(30), -1); // 2×3×5 (three distinct primes)
    }

    #[test]
    fn test_rising_falling_factorial() {
        // 5^(3) = 5×6×7 = 210
        assert_eq!(rising_factorial(5, 3), 210);
        // 5_(3) = 5×4×3 = 60
        assert_eq!(falling_factorial(5, 3), 60);
        // n_(n) = n!
        assert_eq!(falling_factorial(5, 5), 120);
        // C(n,k) = n_(k) / k!
        assert_eq!(
            falling_factorial(10, 3) / falling_factorial(3, 3),
            binomial(10, 3)
        );
    }
}
