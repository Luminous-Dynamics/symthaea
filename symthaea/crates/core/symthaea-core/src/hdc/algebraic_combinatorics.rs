// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Algebraic Combinatorics
//!
//! Symmetric functions, Schur polynomials, Young tableaux (RSK correspondence,
//! hook-length formula), Littlewood-Richardson coefficients, and Kazhdan-Lusztig
//! polynomials for S₃ — all in pure Rust (no external crates).
//!
//! ## Capabilities
//!
//! - **Symmetric functions** — elementary eₖ, complete homogeneous hₖ, power sum pₖ,
//!   Newton's identities
//! - **Schur polynomials** — bialternant / Jacobi-Trudi determinant formula
//! - **Young tableaux** — standard/semistandard check, row bumping (RSK insertion),
//!   hook-length formula, RSK correspondence
//! - **Littlewood-Richardson rule** — LR coefficient c^λ_{μν} for small partitions
//! - **Kazhdan-Lusztig polynomials** — explicit values for S₃
//!
//! ## References
//!
//! - Stanley (1999) — *Enumerative Combinatorics Vol. 2*, Cambridge
//! - Sagan (2001) — *The Symmetric Group*, Springer
//! - Fulton (1997) — *Young Tableaux*, Cambridge
//! - Kazhdan & Lusztig (1979) — Representations of Coxeter groups and Hecke algebras

// ═══════════════════════════════════════════════════════════════════════════
// SYMMETRIC FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Elementary symmetric polynomial eₖ(x₁,…,xₙ) = Σ_{i₁ < … < iₖ} xᵢ₁ · … · xᵢₖ.
pub fn elementary_symmetric(vars: &[f64], k: usize) -> f64 {
    let n = vars.len();
    if k > n {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    // Iterate over k-subsets of {0..n-1}
    let mut total = 0.0;
    subset_sum(vars, k, 0, 1.0, &mut total);
    total
}

fn subset_sum(vars: &[f64], remaining: usize, start: usize, current: f64, total: &mut f64) {
    if remaining == 0 {
        *total += current;
        return;
    }
    let n = vars.len();
    if start + remaining > n {
        return;
    }
    for i in start..=(n - remaining) {
        subset_sum(vars, remaining - 1, i + 1, current * vars[i], total);
    }
}

/// Complete homogeneous symmetric polynomial hₖ = Σ_{i₁ ≤ … ≤ iₖ} xᵢ₁ · … · xᵢₖ.
pub fn complete_homogeneous(vars: &[f64], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let n = vars.len();
    if n == 0 {
        return 0.0;
    }
    // Dynamic programming: h[j] = h_j(x₁,...,xₙ)
    let mut h = vec![0.0; k + 1];
    h[0] = 1.0;
    for &x in vars {
        // Contribution from x: h_k += x * h_{k-1} (at each step, x can appear multiple times)
        for j in (1..=k).rev() {
            // Actually use inclusion: accumulate by adding x to each degree
        }
        // Correct DP: for each variable update h from left to allow repetition
        let mut new_h = h.clone();
        for j in 1..=k {
            // h_j(x₁,...,xᵢ) = h_j(x₁,...,xᵢ₋₁) + xᵢ * h_{j-1}(x₁,...,xᵢ)
            // This is the "unbounded knapsack" recurrence
            new_h[j] = h[j] + x * new_h[j - 1];
        }
        h = new_h;
    }
    h[k]
}

/// Power sum symmetric polynomial pₖ = Σ xᵢᵏ.
pub fn power_sum(vars: &[f64], k: usize) -> f64 {
    vars.iter().map(|&x| x.powi(k as i32)).sum()
}

/// Verify Newton's identity: k · eₖ = Σᵢ₌₁ᵏ (-1)^{i-1} eₖ₋ᵢ · pᵢ.
pub fn newton_identity_check(vars: &[f64], k: usize) -> bool {
    let lhs = k as f64 * elementary_symmetric(vars, k);
    let mut rhs = 0.0;
    for i in 1..=k {
        let sign = if (i - 1) % 2 == 0 { 1.0 } else { -1.0 };
        rhs += sign * elementary_symmetric(vars, k - i) * power_sum(vars, i);
    }
    (lhs - rhs).abs() < 1e-9
}

// ═══════════════════════════════════════════════════════════════════════════
// PARTITIONS
// ═══════════════════════════════════════════════════════════════════════════

/// A partition λ: non-increasing sequence of positive integers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    /// Non-increasing positive integers.
    pub parts: Vec<usize>,
}

impl Partition {
    /// Create a partition, stripping zeros and verifying non-increasing order.
    pub fn new(mut parts: Vec<usize>) -> Self {
        parts.retain(|&x| x > 0);
        // Verify non-increasing
        for i in 1..parts.len() {
            assert!(
                parts[i] <= parts[i - 1],
                "partition parts must be non-increasing"
            );
        }
        Self { parts }
    }

    /// Sum of all parts.
    pub fn size(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Number of parts.
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Conjugate (transpose) partition.
    pub fn conjugate(&self) -> Self {
        if self.parts.is_empty() {
            return Self { parts: vec![] };
        }
        let max_part = self.parts[0];
        let mut conj = Vec::with_capacity(max_part);
        for j in 1..=max_part {
            let count = self.parts.iter().filter(|&&p| p >= j).count();
            if count > 0 {
                conj.push(count);
            }
        }
        Self { parts: conj }
    }

    /// Dominance partial order: λ ≥ μ if Σᵢ₌₁ᵏ λᵢ ≥ Σᵢ₌₁ᵏ μᵢ for all k.
    pub fn dominates(&self, other: &Self) -> bool {
        assert_eq!(
            self.size(),
            other.size(),
            "dominance order requires equal size partitions"
        );
        let n = self.size();
        let mut sum_self = 0;
        let mut sum_other = 0;
        for k in 0..n {
            sum_self += self.parts.get(k).copied().unwrap_or(0);
            sum_other += other.parts.get(k).copied().unwrap_or(0);
            if sum_self < sum_other {
                return false;
            }
        }
        true
    }

    /// Is the partition valid (non-increasing, positive parts)?
    pub fn is_valid(&self) -> bool {
        for i in 1..self.parts.len() {
            if self.parts[i] > self.parts[i - 1] {
                return false;
            }
        }
        self.parts.iter().all(|&p| p > 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCHUR POLYNOMIALS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Schur polynomial sλ(x₁,…,xₙ) via the Jacobi-Trudi formula:
///   sλ = det(h_{λᵢ - i + j})_{1 ≤ i,j ≤ ℓ(λ)}
/// where hₖ = 0 for k < 0, h₀ = 1.
pub fn schur_polynomial(partition: &Partition, vars: &[f64]) -> f64 {
    jacobi_trudi_h(partition, vars)
}

/// Jacobi-Trudi determinant: det(h_{λᵢ - i + j}).
pub fn jacobi_trudi_h(partition: &Partition, vars: &[f64]) -> f64 {
    let l = partition.length();
    if l == 0 {
        return 1.0;
    }
    // Build the matrix M where M[i][j] = h_{λᵢ - i + j} (1-indexed → 0-indexed: i,j ∈ 0..l)
    let mut mat = vec![vec![0.0; l]; l];
    for i in 0..l {
        for j in 0..l {
            let idx = partition.parts[i] as i32 - (i as i32) + (j as i32);
            mat[i][j] = if idx < 0 {
                0.0
            } else if idx == 0 {
                1.0
            } else {
                complete_homogeneous(vars, idx as usize)
            };
        }
    }
    det_nxn(&mat)
}

/// Compute the determinant of an n×n matrix via Gaussian elimination.
fn det_nxn(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    if n == 0 {
        return 1.0;
    }
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut det = 1.0;
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs()));
        if let Some(pr) = pivot_row {
            if a[pr][col].abs() < 1e-14 {
                return 0.0;
            }
            if pr != col {
                a.swap(pr, col);
                det = -det;
            }
        }
        det *= a[col][col];
        let piv = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / piv;
            for j in col..n {
                let v = a[col][j] * factor;
                a[row][j] -= v;
            }
        }
    }
    det
}

// ═══════════════════════════════════════════════════════════════════════════
// YOUNG TABLEAUX
// ═══════════════════════════════════════════════════════════════════════════

/// A filling of a Young diagram shape with positive integers.
#[derive(Debug, Clone)]
pub struct YoungTableau {
    pub shape: Partition,
    /// `entries[i][j]` = entry in row i, column j (0-indexed).
    pub entries: Vec<Vec<usize>>,
}

impl YoungTableau {
    /// Create an empty tableau of the given shape (entries set to 0).
    pub fn empty(shape: Partition) -> Self {
        let entries = shape.parts.iter().map(|&len| vec![0; len]).collect();
        Self { shape, entries }
    }

    /// Is this a Standard Young Tableau?
    ///   - Entries are exactly {1, …, n} each appearing once.
    ///   - Rows strictly increase left to right.
    ///   - Columns strictly increase top to bottom.
    pub fn is_standard(&self) -> bool {
        let n: usize = self.entries.iter().map(|r| r.len()).sum();
        // Collect all entries
        let mut all: Vec<usize> = self
            .entries
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        all.sort_unstable();
        if all != (1..=n).collect::<Vec<_>>() {
            return false;
        }
        // Rows strictly increasing
        for row in &self.entries {
            for j in 1..row.len() {
                if row[j] <= row[j - 1] {
                    return false;
                }
            }
        }
        // Columns strictly increasing
        let num_rows = self.entries.len();
        for row in 1..num_rows {
            let prev_len = self.entries[row - 1].len();
            let curr_len = self.entries[row].len();
            for col in 0..curr_len.min(prev_len) {
                if self.entries[row][col] <= self.entries[row - 1][col] {
                    return false;
                }
            }
        }
        true
    }

    /// Is this a Semistandard Young Tableau?
    ///   - Rows weakly increase left to right.
    ///   - Columns strictly increase top to bottom.
    ///   - Entries are positive integers.
    pub fn is_semistandard(&self) -> bool {
        if self.entries.iter().flat_map(|r| r.iter()).any(|&x| x == 0) {
            return false;
        }
        for row in &self.entries {
            for j in 1..row.len() {
                if row[j] < row[j - 1] {
                    return false;
                }
            }
        }
        let num_rows = self.entries.len();
        for row in 1..num_rows {
            let prev_len = self.entries[row - 1].len();
            let curr_len = self.entries[row].len();
            for col in 0..curr_len.min(prev_len) {
                if self.entries[row][col] <= self.entries[row - 1][col] {
                    return false;
                }
            }
        }
        true
    }

    /// Reading word: read rows from bottom to top, left to right.
    pub fn reading_word(&self) -> Vec<usize> {
        self.entries
            .iter()
            .rev()
            .flat_map(|r| r.iter().copied())
            .collect()
    }

    /// RSK row insertion of x into the tableau.
    ///
    /// Inserts x into row 0. If x bumps an element, that element is inserted into row 1, etc.
    /// Returns `Some(bumped)` if a new row needs to be added (bumped from last row),
    /// or `None` if x was appended to an existing row without bumping anything out.
    pub fn row_bumping(&mut self, x: usize) -> Option<usize> {
        let mut to_insert = x;
        for row in &mut self.entries {
            // Find leftmost element > to_insert
            if let Some(pos) = row.iter().position(|&e| e > to_insert) {
                std::mem::swap(&mut row[pos], &mut to_insert);
                // Continue to next row
            } else {
                // No element > to_insert: append to end of this row
                row.push(to_insert);
                return None; // no bump out
            }
        }
        // to_insert was bumped out of all rows: need a new row
        Some(to_insert)
    }

    /// Hook-length formula: number of SYT of the given shape.
    ///   f^λ = n! / Π_{(i,j) ∈ λ} hook(i,j)
    pub fn num_standard_tableaux(partition: &Partition) -> usize {
        hook_length_formula(partition)
    }
}

/// Hook length at cell (i, j) (0-indexed) in partition λ:
///   hook(i,j) = λᵢ - j + #{rows k > i : λₖ > j} - 1 + 1
///             = (arm length) + (leg length) + 1
fn hook_length(partition: &Partition, i: usize, j: usize) -> usize {
    let arm = partition.parts[i] - j - 1; // cells to the right in same row
    let leg = partition
        .parts
        .iter()
        .skip(i + 1)
        .filter(|&&p| p > j)
        .count();
    arm + leg + 1
}

/// Hook-length formula: f^λ = n! / Π_{cells} hook(i,j).
pub fn hook_length_formula(partition: &Partition) -> usize {
    let n = partition.size();
    let num = factorial(n);
    let mut denom: usize = 1;
    for (i, &row_len) in partition.parts.iter().enumerate() {
        for j in 0..row_len {
            denom *= hook_length(partition, i, j);
        }
    }
    num / denom
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// RSK correspondence: bijection from permutations to pairs of SYT of the same shape.
///
/// Returns (insertion_tableau P, recording_tableau Q).
pub fn rsk_correspondence(permutation: &[usize]) -> (YoungTableau, YoungTableau) {
    let n = permutation.len();
    // Build P via row insertion
    let mut p = YoungTableau {
        shape: Partition { parts: vec![] },
        entries: vec![],
    };
    let mut q = YoungTableau {
        shape: Partition { parts: vec![] },
        entries: vec![],
    };

    for (step, &sigma_i) in permutation.iter().enumerate() {
        let label = step + 1; // recording label (1-indexed)
        let to_insert = sigma_i;
        let mut val = to_insert;

        // Row-bump into P, recording where new cell appears in Q
        let mut new_row_idx = p.entries.len(); // default: new row at bottom
        for (row_idx, row) in p.entries.iter_mut().enumerate() {
            if let Some(pos) = row.iter().position(|&e| e > val) {
                std::mem::swap(&mut row[pos], &mut val);
            } else {
                row.push(val);
                new_row_idx = row_idx;
                val = 0; // signal: no new row needed
                break;
            }
        }

        if val != 0 {
            // Need a new row for P and Q
            p.entries.push(vec![val]);
            q.entries.push(vec![label]);
        } else {
            // Record in Q at the same position as where P grew
            let col_idx = p.entries[new_row_idx].len() - 1;
            if new_row_idx < q.entries.len() {
                q.entries[new_row_idx].push(label);
            } else {
                q.entries.push(vec![label]);
            }
        }
    }

    // Rebuild shapes
    let p_shape = Partition {
        parts: p
            .entries
            .iter()
            .map(|r| r.len())
            .filter(|&l| l > 0)
            .collect(),
    };
    let q_shape = Partition {
        parts: q
            .entries
            .iter()
            .map(|r| r.len())
            .filter(|&l| l > 0)
            .collect(),
    };
    p.shape = p_shape;
    q.shape = q_shape;

    (p, q)
}

// ═══════════════════════════════════════════════════════════════════════════
// LITTLEWOOD-RICHARDSON RULE
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Littlewood-Richardson coefficient c^λ_{μν}: the number of
/// semistandard skew tableaux of shape λ/μ with content ν whose reading word
/// is a ballot sequence (lattice word).
///
/// Implemented for partitions with n ≤ 6.
pub fn littlewood_richardson_coeff(mu: &Partition, nu: &Partition, lambda: &Partition) -> usize {
    // Validate: |μ| + |ν| = |λ|, λ ⊇ μ
    if mu.size() + nu.size() != lambda.size() {
        return 0;
    }
    if !contains(lambda, mu) {
        return 0;
    }
    if lambda.size() > 6 {
        return 0;
    } // implementation limit

    // Count SSYT of skew shape λ/μ with content ν (weight ν) whose reading word is a ballot sequence.
    let nu_content = nu.parts.clone();
    let max_entry = nu.length();
    if max_entry == 0 {
        return if mu == lambda { 1 } else { 0 };
    }

    // Enumerate all fillings of the skew shape λ/μ with entries 1..=max_entry
    // such that the content equals ν and the tableau is semistandard.
    let skew_cells = skew_shape_cells(lambda, mu);
    if skew_cells.is_empty() {
        return if nu.size() == 0 { 1 } else { 0 };
    }

    count_lr_tableaux(&skew_cells, &nu_content, max_entry, lambda)
}

/// Returns whether λ ⊇ μ (μ fits inside λ).
fn contains(lambda: &Partition, mu: &Partition) -> bool {
    for (i, &mu_i) in mu.parts.iter().enumerate() {
        if lambda.parts.get(i).copied().unwrap_or(0) < mu_i {
            return false;
        }
    }
    true
}

/// Returns the cells (row, col) in the skew shape λ/μ (0-indexed).
fn skew_shape_cells(lambda: &Partition, mu: &Partition) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    for (i, &lam_i) in lambda.parts.iter().enumerate() {
        let mu_i = mu.parts.get(i).copied().unwrap_or(0);
        for j in mu_i..lam_i {
            cells.push((i, j));
        }
    }
    cells
}

/// Reading word: bottom-to-top, left-to-right.
fn reading_word_from_filling(
    cells: &[(usize, usize)],
    filling: &[usize],
    num_rows: usize,
) -> Vec<usize> {
    // Build row groups
    let max_row = cells.iter().map(|&(r, _)| r).max().unwrap_or(0);
    let mut rows: Vec<Vec<(usize, usize)>> = vec![vec![]; max_row + 1];
    for (&(r, c), &v) in cells.iter().zip(filling.iter()) {
        rows[r].push((c, v));
    }
    let mut word = Vec::new();
    for row in rows.iter().rev() {
        let mut sorted = row.clone();
        sorted.sort_by_key(|&(c, _)| c);
        for (_, v) in sorted {
            word.push(v);
        }
    }
    word
}

/// Check if a word is a ballot (lattice) sequence:
///   In every prefix, for all i < j: count(i) ≥ count(j).
fn is_ballot_sequence(word: &[usize], max_entry: usize) -> bool {
    let mut counts = vec![0usize; max_entry + 1];
    for &x in word {
        if x == 0 || x > max_entry {
            return false;
        }
        counts[x] += 1;
        for k in 1..max_entry {
            if counts[k] < counts[k + 1] {
                return false;
            }
        }
    }
    true
}

/// Enumerate all SSYT fillings of skew cells with content ν and ballot reading word.
fn count_lr_tableaux(
    cells: &[(usize, usize)],
    nu_content: &[usize],
    max_entry: usize,
    lambda: &Partition,
) -> usize {
    let n = cells.len();
    let mut filling = vec![0usize; n];
    let mut count = 0;

    // Build a sorted order for checking SSYT constraints
    // cells are given in row-major order; we need to check left-neighbor and top-neighbor
    let num_rows = lambda.parts.len();

    // Build index maps for quick lookup
    let mut cell_to_idx: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for (idx, &cell) in cells.iter().enumerate() {
        cell_to_idx.insert(cell, idx);
    }

    enumerate_filling(
        cells,
        &cell_to_idx,
        &mut filling,
        0,
        nu_content,
        max_entry,
        lambda,
        &mut count,
    );
    count
}

fn enumerate_filling(
    cells: &[(usize, usize)],
    cell_to_idx: &std::collections::HashMap<(usize, usize), usize>,
    filling: &mut Vec<usize>,
    pos: usize,
    nu_content: &[usize],
    max_entry: usize,
    lambda: &Partition,
    count: &mut usize,
) {
    if pos == cells.len() {
        // Check content equals ν
        let mut actual_content = vec![0usize; max_entry + 1];
        for &v in filling.iter() {
            actual_content[v] += 1;
        }
        for (k, &nu_k) in nu_content.iter().enumerate() {
            if actual_content.get(k + 1).copied().unwrap_or(0) != nu_k {
                return;
            }
        }
        // Check ballot sequence
        let word = reading_word_from_filling(cells, filling, lambda.parts.len());
        if is_ballot_sequence(&word, max_entry) {
            *count += 1;
        }
        return;
    }

    let (row, col) = cells[pos];

    for entry in 1..=max_entry {
        // SSYT: rows weakly increase (left neighbor must be ≤ entry)
        if col > 0 {
            // Left neighbor could be in μ (fixed) or in the skew
            if let Some(&left_idx) = cell_to_idx.get(&(row, col - 1))
                && filling[left_idx] > entry
            {
                continue;
            }
            // If left neighbor is in μ (not a skew cell), the constraint is automatically satisfied
            // since μ cells are not filled (they are 0) — we treat them as boundary
        }
        // SSYT: columns strictly increase (top neighbor must be < entry)
        if row > 0
            && let Some(&top_idx) = cell_to_idx.get(&(row - 1, col))
            && filling[top_idx] >= entry
        {
            continue;
        }

        filling[pos] = entry;
        enumerate_filling(
            cells,
            cell_to_idx,
            filling,
            pos + 1,
            nu_content,
            max_entry,
            lambda,
            count,
        );
        filling[pos] = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KAZHDAN-LUSZTIG POLYNOMIALS FOR S₃
// ═══════════════════════════════════════════════════════════════════════════

/// Bruhat order on S₃.
///
/// Permutations in one-line notation (sorted lexicographically):
///   0: [1,2,3], 1: [1,3,2], 2: [2,1,3], 3: [2,3,1], 4: [3,1,2], 5: [3,2,1]
///
/// Returns a 6×6 boolean matrix where `result[i][j] = true` iff σᵢ ≤ σⱼ in Bruhat order.
pub fn bruhat_order_s3() -> Vec<Vec<bool>> {
    // Bruhat order on S₃ (by inversion table comparison):
    // [1,2,3] ≤ everything
    // [3,2,1] ≥ everything
    // The full Bruhat order for S₃ (6 elements, 0-indexed):
    // 0=[123]<1=[132],2=[213],<3=[231],4=[312],<5=[321]
    // Additional: 1<4, 2<3, 3<5, 4<5, 1<5 via transitivity
    // Hasse diagram edges: 0→1, 0→2, 1→3, 1→4, 2→3, 3→5, 4→5, 2→4? NO
    // Let's compute via inversions:
    // Bruhat: x ≤ w iff every inversion table value of x is ≤ that of w? No, use length+cover.
    // Actually use: x ≤ w iff for every k: #{i≤k : x(i) ≤ j} ≥ #{i≤k : w(i) ≤ j} for all j.
    // Simplified: compute via direct cover relations for S₃
    // Covers: 0<1, 0<2, 1<3, 1<4, 2<3, 2<4, 3<5, 4<5
    // Let's compute transitive closure:
    let perms: Vec<[usize; 3]> = vec![
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ];
    let n = 6;
    let mut order = vec![vec![false; n]; n];
    // x ≤ w in Bruhat iff for all 1≤i≤3, 1≤j≤3: #{k≤i : x[k]≤j} ≥ #{k≤i : w[k]≤j}
    for xi in 0..n {
        for wi in 0..n {
            let x = &perms[xi];
            let w = &perms[wi];
            let mut ok = true;
            'outer: for i in 1..=3 {
                for j in 1..=3 {
                    let cx = x[..i].iter().filter(|&&v| v <= j).count();
                    let cw = w[..i].iter().filter(|&&v| v <= j).count();
                    if cx < cw {
                        ok = false;
                        break 'outer;
                    }
                }
            }
            order[xi][wi] = ok;
        }
    }
    order
}

/// Kazhdan-Lusztig polynomial P_{x,w}(q) for S₃ as a coefficient vector [a₀, a₁, …].
///
/// The permutation indices correspond to the lexicographic ordering:
///   0=[1,2,3], 1=[1,3,2], 2=[2,1,3], 3=[2,3,1], 4=[3,1,2], 5=[3,2,1].
///
/// For S₃ all KL polynomials are either 0 or 1 (the group is too small for higher terms).
pub fn kl_polynomial_s3(x_idx: usize, w_idx: usize) -> Vec<i32> {
    let order = bruhat_order_s3();
    assert!(x_idx < 6 && w_idx < 6);
    // P_{w,w} = 1
    if x_idx == w_idx {
        return vec![1];
    }
    // P_{x,w} = 0 if x > w
    if !order[x_idx][w_idx] {
        return vec![0];
    }
    // For S₃: all P_{x,w} = 1 when x < w (S₃ is too small for q-terms)
    vec![1]
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ─── Symmetric functions ─────────────────────────────────────────────────

    #[test]
    fn test_elementary_symmetric_k0() {
        assert!(close(elementary_symmetric(&[1.0, 2.0, 3.0], 0), 1.0, 1e-12));
    }

    #[test]
    fn test_elementary_symmetric_k1() {
        // e₁ = x₁ + x₂ + x₃ = 6
        assert!(close(elementary_symmetric(&[1.0, 2.0, 3.0], 1), 6.0, 1e-12));
    }

    #[test]
    fn test_elementary_symmetric_k2() {
        // e₂([1,2,3]) = 1*2 + 1*3 + 2*3 = 2 + 3 + 6 = 11
        let result = elementary_symmetric(&[1.0, 2.0, 3.0], 2);
        assert!(close(result, 11.0, 1e-12), "e₂ = {}", result);
    }

    #[test]
    fn test_elementary_symmetric_k3() {
        // e₃([1,2,3]) = 1*2*3 = 6
        assert!(close(elementary_symmetric(&[1.0, 2.0, 3.0], 3), 6.0, 1e-12));
    }

    #[test]
    fn test_complete_homogeneous_k0() {
        assert!(close(complete_homogeneous(&[1.0, 2.0], 0), 1.0, 1e-12));
    }

    #[test]
    fn test_complete_homogeneous_k1() {
        // h₁ = e₁ = 1 + 2 = 3
        assert!(close(complete_homogeneous(&[1.0, 2.0], 1), 3.0, 1e-12));
    }

    #[test]
    fn test_complete_homogeneous_k2() {
        // h₂([1,2]) = 1² + 1*2 + 2² = 1 + 2 + 4 = 7
        let result = complete_homogeneous(&[1.0, 2.0], 2);
        assert!(close(result, 7.0, 1e-12), "h₂([1,2]) = {}", result);
    }

    #[test]
    fn test_power_sum() {
        // p₂([1,2,3]) = 1 + 4 + 9 = 14
        assert!(close(power_sum(&[1.0, 2.0, 3.0], 2), 14.0, 1e-12));
    }

    #[test]
    fn test_newton_identity() {
        let vars = vec![1.0, 2.0, 3.0];
        assert!(newton_identity_check(&vars, 1));
        assert!(newton_identity_check(&vars, 2));
        assert!(newton_identity_check(&vars, 3));
    }

    // ─── Partitions ──────────────────────────────────────────────────────────

    #[test]
    fn test_partition_conjugate() {
        // λ = (3,2,1) → conjugate = (3,2,1)
        let lam = Partition::new(vec![3, 2, 1]);
        let conj = lam.conjugate();
        assert_eq!(conj.parts, vec![3, 2, 1]);
    }

    #[test]
    fn test_partition_conjugate_rectangular() {
        // λ = (3,3) → conjugate = (2,2,2)
        let lam = Partition::new(vec![3, 3]);
        let conj = lam.conjugate();
        assert_eq!(conj.parts, vec![2, 2, 2]);
    }

    #[test]
    fn test_partition_dominance() {
        // (3,1) dominates (2,2) since partial sums: 3 ≥ 2 and 4 ≥ 4
        let a = Partition::new(vec![3, 1]);
        let b = Partition::new(vec![2, 2]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    // ─── Schur polynomials ───────────────────────────────────────────────────

    #[test]
    fn test_schur_trivial_partition() {
        // λ = (1): s_(1)(x,y) = x + y = e₁
        let lam = Partition::new(vec![1]);
        let vars = vec![2.0, 3.0];
        let s = schur_polynomial(&lam, &vars);
        assert!(close(s, 5.0, 1e-10), "s_(1)(2,3) = {}", s);
    }

    #[test]
    fn test_schur_partition_2() {
        // λ = (2): s_(2)(x,y) = x² + xy + y² = h₂(x,y)
        let lam = Partition::new(vec![2]);
        let vars = vec![1.0, 2.0];
        let s = schur_polynomial(&lam, &vars);
        // h₂(1,2) = 1 + 2 + 4 = 7
        assert!(close(s, 7.0, 1e-10), "s_(2)(1,2) = {}", s);
    }

    #[test]
    fn test_schur_partition_2_1_in_2vars() {
        // λ=(2,1) in 2 vars: s_{(2,1)}(x,y) = x²y + xy²  = xy(x+y)
        let lam = Partition::new(vec![2, 1]);
        let vars = vec![2.0, 3.0];
        let s = schur_polynomial(&lam, &vars);
        // 2²·3 + 2·3² = 12 + 18 = 30
        assert!(close(s, 30.0, 1e-9), "s_(2,1)(2,3) = {}", s);
    }

    // ─── Hook-length formula ─────────────────────────────────────────────────

    #[test]
    fn test_hook_length_staircase_321() {
        // λ = (3,2,1), n = 6: f^λ = 6!/(5·3·1·3·1·1) = 720/45 = 16
        let lam = Partition::new(vec![3, 2, 1]);
        let f = hook_length_formula(&lam);
        assert_eq!(f, 16, "f^(3,2,1) = {}", f);
    }

    #[test]
    fn test_hook_length_single_row() {
        // λ = (n): only 1 SYT
        let lam = Partition::new(vec![4]);
        assert_eq!(hook_length_formula(&lam), 1);
    }

    #[test]
    fn test_hook_length_column() {
        // λ = (1,1,1,1): column of 4, only 1 SYT
        let lam = Partition::new(vec![1, 1, 1, 1]);
        assert_eq!(hook_length_formula(&lam), 1);
    }

    // ─── Young tableaux ───────────────────────────────────────────────────────

    #[test]
    fn test_is_standard_valid() {
        let shape = Partition::new(vec![3, 2]);
        let mut t = YoungTableau::empty(shape);
        t.entries = vec![vec![1, 2, 5], vec![3, 4]];
        assert!(t.is_standard());
    }

    #[test]
    fn test_is_standard_invalid_row() {
        let shape = Partition::new(vec![3]);
        let mut t = YoungTableau::empty(shape);
        t.entries = vec![vec![1, 3, 2]]; // row not strictly increasing
        assert!(!t.is_standard());
    }

    // ─── RSK correspondence ───────────────────────────────────────────────────

    #[test]
    fn test_rsk_identity_permutation() {
        // [1,2,3] → P = Q = standard staircase: rows [1],[2],[3] or row [1,2,3]
        let (p, q) = rsk_correspondence(&[1, 2, 3]);
        // For identity permutation, RSK gives P = Q = [[1,2,3]]
        assert_eq!(p.shape, q.shape, "P and Q must have the same shape");
        assert_eq!(p.entries[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_rsk_reversal_permutation() {
        // [3,2,1] is the longest permutation; RSK gives a 1-column tableau
        let (p, q) = rsk_correspondence(&[3, 2, 1]);
        assert_eq!(p.shape, q.shape);
        // P should be the column [1,2,3] (each element bumps to new row)
        assert_eq!(p.entries.len(), 3, "staircase shape expected");
    }

    // ─── Littlewood-Richardson ────────────────────────────────────────────────

    #[test]
    fn test_lr_trivial_empty_mu() {
        // c^ν_{∅,ν} = 1 (sλ = sμ * sν trivially when μ = ∅)
        let mu = Partition { parts: vec![] };
        let nu = Partition::new(vec![2]);
        let lambda = Partition::new(vec![2]);
        let c = littlewood_richardson_coeff(&mu, &nu, &lambda);
        assert_eq!(c, 1, "c^(2)_{{∅,(2)}} = 1");
    }

    #[test]
    fn test_lr_adjacent_rows() {
        // c^(2,1)_{(1),(1)} — adding one box to a single box
        let mu = Partition::new(vec![1]);
        let nu = Partition::new(vec![1]);
        let lambda = Partition::new(vec![2]);
        // s_(1) * s_(1) = s_(2) + s_(1,1); so c^(2)_{(1),(1)} = 1
        let c = littlewood_richardson_coeff(&mu, &nu, &lambda);
        assert_eq!(c, 1, "c^(2)_{{(1),(1)}} = {}", c);
    }

    // ─── Kazhdan-Lusztig ─────────────────────────────────────────────────────

    #[test]
    fn test_kl_identity() {
        // P_{w,w} = 1 for all w
        for i in 0..6 {
            let p = kl_polynomial_s3(i, i);
            assert_eq!(p, vec![1], "P_{{w,w}} = 1 for index {}", i);
        }
    }

    #[test]
    fn test_kl_not_comparable() {
        // x > w → P_{x,w} = 0
        let order = bruhat_order_s3();
        for xi in 0..6 {
            for wi in 0..6 {
                if !order[xi][wi] {
                    let p = kl_polynomial_s3(xi, wi);
                    assert_eq!(p, vec![0], "P_{{x,w}} = 0 for x > w (idx {} > {})", xi, wi);
                }
            }
        }
    }

    #[test]
    fn test_bruhat_order_s3_identity_is_minimum() {
        let order = bruhat_order_s3();
        // The identity [1,2,3] (index 0) is below everything
        for w in 0..6 {
            assert!(order[0][w], "identity should be below everything (w={})", w);
        }
    }

    #[test]
    fn test_bruhat_order_s3_longest_is_maximum() {
        let order = bruhat_order_s3();
        // The longest element [3,2,1] (index 5) is above everything
        for x in 0..6 {
            assert!(
                order[x][5],
                "longest element should be above everything (x={})",
                x
            );
        }
    }
}
