// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Combinatorial primitives (Phase 4 scoped — IMO roadmap)
//!
//! Three core tools from the IMO combinatorics toolkit:
//!
//! 1. **Pigeonhole principle** — given a partition of n items into k
//!    buckets, determine the minimum guaranteed max bucket size, and apply
//!    it to concrete item sets with a partition function.
//!
//! 2. **Linear invariant search** — given a trajectory of states (snapshots
//!    of a discrete transition system), find a linear combination
//!    c₁·s₁ + c₂·s₂ + ... + cₙ·sₙ that is approximately constant across
//!    all transitions. Used to prove conservation laws in combinatorial
//!    game-theory problems (chip-firing, token games, etc.).
//!
//! 3. **Monovariant search** — find a linear function that strictly
//!    decreases (or increases) across every transition. Used in termination
//!    proofs via infinite descent.
//!
//! ## Scope limits honestly acknowledged
//!
//! - Only *linear* invariants/monovariants. Polynomial invariants (Cassini
//!   identity, Fibonacci-like) require symbolic manipulation deferred to
//!   Phase 3B.
//! - Pigeonhole is a *concrete* applicator, not a symbolic prover. The
//!   SMT-encoding path (pigeonhole_smt from the plan) is deferred until
//!   the Phase 4 full refactor (Tactic::Context).
//! - Monovariant search uses a simple LP-free heuristic — full LP-based
//!   solver can come later if needed.

use std::collections::HashMap;
use std::hash::Hash;

// ─── Pigeonhole principle ────────────────────────────────────────────────────

/// Minimum max-bucket size guaranteed by pigeonhole: if you distribute
/// `items` items across `boxes` buckets, some bucket must contain at
/// least ⌈items / boxes⌉ items.
///
/// Returns 0 if `boxes == 0` (vacuously).
pub fn pigeonhole_min_max_bucket(items: usize, boxes: usize) -> usize {
    if boxes == 0 {
        return 0;
    }
    items.div_ceil(boxes) // ceiling division
}

/// Apply the pigeonhole principle to a concrete set. Given a slice of
/// items and a partition function that maps each item to a bucket key,
/// compute the size of the largest bucket. Return `Some(max_bucket)` if
/// it is ≥ `min_collision` (pigeonhole forced a collision), otherwise
/// `None`.
pub fn pigeonhole_apply<T, K, F>(items: &[T], partition: F, min_collision: usize) -> Option<usize>
where
    K: Hash + Eq,
    F: Fn(&T) -> K,
{
    let mut buckets: HashMap<K, usize> = HashMap::new();
    for it in items {
        *buckets.entry(partition(it)).or_insert(0) += 1;
    }
    let max_bucket = buckets.values().copied().max().unwrap_or(0);
    if max_bucket >= min_collision {
        Some(max_bucket)
    } else {
        None
    }
}

/// Convenience: witness of a pigeonhole collision. Given items and a
/// partition, return `Some((bucket_key, collision_members))` for the
/// bucket with the most members if size ≥ 2, else `None`. Useful for
/// constructive IMO proofs that need to name the colliding items.
pub fn pigeonhole_witness<T, K, F>(items: &[T], partition: F) -> Option<(K, Vec<usize>)>
where
    K: Hash + Eq + Clone,
    F: Fn(&T) -> K,
{
    let mut buckets: HashMap<K, Vec<usize>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        buckets
            .entry(partition(it))
            .or_insert_with(Vec::new)
            .push(i);
    }
    buckets
        .into_iter()
        .filter(|(_, v)| v.len() >= 2)
        .max_by_key(|(_, v)| v.len())
}

// ─── Linear invariant search ────────────────────────────────────────────────

/// Find a linear invariant of a discrete transition system given a
/// sampled trajectory. Returns coefficients `(c_1, ..., c_n)` such that
/// `c · s` is approximately constant for every state `s` on the
/// trajectory.
///
/// Algorithm: compute the consecutive-difference matrix D where row i is
/// `trajectory[i+1] - trajectory[i]`. An invariant corresponds to a
/// vector in the *right* null space of D (D · c = 0). We extract such
/// a vector via Gram–Schmidt orthogonalization of D's rows plus residual
/// projection of standard basis vectors.
///
/// Returns `Some((coefficients, residual))` where residual is the
/// maximum |c · Δ_i| across all consecutive differences. Returns `None`
/// if the trajectory has fewer than two samples, if all samples are
/// identical (trivial — everything is invariant), or if no non-trivial
/// invariant exists.
pub fn find_linear_invariant(trajectory: &[Vec<f64>]) -> Option<(Vec<f64>, f64)> {
    if trajectory.len() < 2 {
        return None;
    }
    let n = trajectory[0].len();
    if n == 0 {
        return None;
    }
    let deltas: Vec<Vec<f64>> = trajectory
        .windows(2)
        .map(|w| w[1].iter().zip(w[0].iter()).map(|(b, a)| b - a).collect())
        .collect();
    // Degenerate: trajectory is constant.
    if deltas.iter().all(|d| d.iter().all(|x| x.abs() < 1e-12)) {
        return None;
    }
    // Gram–Schmidt orthonormal basis of the row space of D.
    let tol = 1e-9;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for row in &deltas {
        let mut r: Vec<f64> = row.clone();
        for b in &basis {
            let dot: f64 = r.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            for (ri, bi) in r.iter_mut().zip(b.iter()) {
                *ri -= dot * bi;
            }
        }
        let norm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            for ri in r.iter_mut() {
                *ri /= norm;
            }
            basis.push(r);
        }
    }
    // Full row-rank → no invariant exists.
    if basis.len() >= n {
        return None;
    }
    // Extract a null-space vector: for each standard basis vector e_i,
    // subtract its projection onto the row-space basis. If the residual
    // has non-zero norm, it lies in the null space of D and is an
    // invariant of the trajectory.
    for pivot in 0..n {
        let mut c = vec![0.0f64; n];
        c[pivot] = 1.0;
        for b in &basis {
            let dot: f64 = c.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            for (ci, bi) in c.iter_mut().zip(b.iter()) {
                *ci -= dot * bi;
            }
        }
        let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            for ci in c.iter_mut() {
                *ci /= norm;
            }
            // Verify residual against the raw deltas
            let mut max_res = 0.0f64;
            for row in &deltas {
                let dot: f64 = c.iter().zip(row.iter()).map(|(ci, ri)| ci * ri).sum();
                if dot.abs() > max_res {
                    max_res = dot.abs();
                }
            }
            if max_res < 1e-6 {
                return Some((c, max_res));
            }
        }
    }
    None
}

// ─── Monovariant search ─────────────────────────────────────────────────────

/// Find a linear monovariant: coefficients `c` such that `c · s` is
/// *strictly* decreasing across every trajectory transition (or strictly
/// increasing if `seek_decreasing == false`). Used to prove termination
/// in combinatorial processes.
///
/// Algorithm (simple, non-LP): for each state dimension j, test whether
/// ±e_j (the j-th standard basis vector) is monotone. If so, return.
/// Otherwise, try the negative-mean-delta direction: c = -Δ̄ where Δ̄ is
/// the mean consecutive difference. Test that and return if it works.
/// Returns None otherwise.
///
/// This catches all obvious monovariants without a full LP solve. A more
/// sophisticated search would use linear programming to find any feasible
/// monovariant; that's deferred.
pub fn find_linear_monovariant(trajectory: &[Vec<f64>], seek_decreasing: bool) -> Option<Vec<f64>> {
    if trajectory.len() < 2 {
        return None;
    }
    let n = trajectory[0].len();
    let deltas: Vec<Vec<f64>> = trajectory
        .windows(2)
        .map(|w| w[1].iter().zip(w[0].iter()).map(|(b, a)| b - a).collect())
        .collect();
    let strict_tol = 1e-9;
    let check = |c: &[f64]| -> bool {
        deltas.iter().all(|d| {
            let dot: f64 = c.iter().zip(d.iter()).map(|(ci, di)| ci * di).sum();
            if seek_decreasing {
                dot < -strict_tol
            } else {
                dot > strict_tol
            }
        })
    };
    // Try each ±e_j.
    for j in 0..n {
        let mut pos = vec![0.0f64; n];
        pos[j] = 1.0;
        if check(&pos) {
            return Some(pos);
        }
        let mut neg = vec![0.0f64; n];
        neg[j] = -1.0;
        if check(&neg) {
            return Some(neg);
        }
    }
    // Try mean-delta direction.
    let mut mean_delta = vec![0.0f64; n];
    for d in &deltas {
        for (i, v) in d.iter().enumerate() {
            mean_delta[i] += *v;
        }
    }
    for v in mean_delta.iter_mut() {
        *v /= deltas.len() as f64;
    }
    // If seek_decreasing, we want c · d < 0 for all d. A good candidate
    // is c = -mean_delta.
    let sign = if seek_decreasing { -1.0 } else { 1.0 };
    let candidate: Vec<f64> = mean_delta.iter().map(|v| sign * v).collect();
    if check(&candidate) {
        return Some(candidate);
    }
    None
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pigeonhole ─────────────────────────────────────────────────────

    #[test]
    fn test_pigeonhole_min_max_bucket() {
        assert_eq!(pigeonhole_min_max_bucket(7, 6), 2); // 7 items, 6 boxes → ⌈7/6⌉ = 2
        assert_eq!(pigeonhole_min_max_bucket(10, 3), 4); // ⌈10/3⌉ = 4
        assert_eq!(pigeonhole_min_max_bucket(5, 5), 1); // not forced
        assert_eq!(pigeonhole_min_max_bucket(0, 5), 0);
        assert_eq!(pigeonhole_min_max_bucket(5, 0), 0); // degenerate
    }

    #[test]
    fn test_pigeonhole_apply_mod_6() {
        // Classic: 7 integers mod 6 must have at least one collision
        // (pigeonhole: 7 items in 6 boxes).
        let ints = vec![3, 14, 27, 100, 5, 18, 71];
        let result = pigeonhole_apply(&ints, |&n: &i32| (n % 6 + 6) % 6, 2);
        assert!(result.is_some(), "pigeonhole must force a collision");
        let max_bucket = result.unwrap();
        assert!(max_bucket >= 2);
    }

    #[test]
    fn test_pigeonhole_apply_no_forced_collision() {
        // 3 items in 3 boxes — pigeonhole doesn't force anything.
        let ints = vec![1, 2, 3];
        let result = pigeonhole_apply(&ints, |&n: &i32| n, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_pigeonhole_witness_returns_collision_members() {
        // 7 integers: 3, 9, 15 are all ≡ 3 (mod 6). Expect a bucket
        // containing at least these three.
        let ints = vec![3, 9, 15, 4, 5, 7, 10];
        let result = pigeonhole_witness(&ints, |&n: &i32| (n % 6 + 6) % 6);
        let (_, members) = result.expect("collision must exist");
        assert!(members.len() >= 3);
    }

    #[test]
    fn test_pigeonhole_birthday_14_people() {
        // 14 people, 12 months → pigeonhole forces some month ≥ 2.
        let people: Vec<usize> = (0..14).collect();
        let months = [1, 4, 7, 2, 8, 12, 3, 5, 11, 6, 9, 10, 4, 7]; // birthdays
        let result = pigeonhole_apply(&people, |&i| months[i], 2);
        assert!(result.is_some());
    }

    // ── Linear invariant search ────────────────────────────────────────

    #[test]
    fn test_linear_invariant_sum_conserved() {
        // (a, b) → (a-1, b+1): the sum a+b is invariant.
        let trajectory = vec![
            vec![5.0, 3.0],
            vec![4.0, 4.0],
            vec![3.0, 5.0],
            vec![2.0, 6.0],
        ];
        let result = find_linear_invariant(&trajectory);
        let (c, res) = result.expect("should find a linear invariant");
        assert!(res < 1e-9, "residual too high: {}", res);
        // c should be parallel to (1, 1) — i.e., c[0] ≈ c[1]
        let ratio = c[0] / c[1];
        assert!((ratio - 1.0).abs() < 1e-6, "invariant not a+b: c={:?}", c);
    }

    #[test]
    fn test_linear_invariant_weighted_sum() {
        // (a, b) → (a+1, b+2): invariant is 2a - b.
        let trajectory = vec![
            vec![0.0, 0.0],
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ];
        let (c, res) = find_linear_invariant(&trajectory).expect("invariant exists");
        assert!(res < 1e-9);
        // c parallel to (2, -1): c[0] / c[1] ≈ -2
        let ratio = c[0] / c[1];
        assert!((ratio + 2.0).abs() < 1e-6, "expected 2a-b, got c={:?}", c);
    }

    #[test]
    fn test_linear_invariant_rejects_no_invariant() {
        // (a, b) → (2a, b+1): no linear invariant.
        let trajectory = vec![
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            vec![4.0, 2.0],
            vec![8.0, 3.0],
        ];
        let result = find_linear_invariant(&trajectory);
        assert!(
            result.is_none(),
            "should find no invariant, got {:?}",
            result
        );
    }

    #[test]
    fn test_linear_invariant_3d() {
        // State (x, y, z). Transitions (x,y,z) → (x+1, y+1, z+2).
        // Invariant: 2x - z (since dx=1, dz=2, dy=1, 2·1 - 2 = 0).
        let trajectory = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 2.0],
            vec![2.0, 2.0, 4.0],
            vec![3.0, 3.0, 6.0],
        ];
        let (c, res) = find_linear_invariant(&trajectory).expect("invariant exists");
        assert!(res < 1e-9);
        // Any linear combo orthogonal to (1, 1, 2). E.g., (1, 0, -0.5),
        // (0, 1, -0.5), (1, -1, 0), etc. Verify by checking the found
        // vector is orthogonal to the delta (1, 1, 2).
        let dot = c[0] * 1.0 + c[1] * 1.0 + c[2] * 2.0;
        assert!(dot.abs() < 1e-9, "c={:?} not orthogonal to delta", c);
    }

    // ── Monovariant search ─────────────────────────────────────────────

    #[test]
    fn test_monovariant_basic_counter() {
        // Simple counter: x decreases at every step.
        let trajectory = vec![vec![10.0], vec![9.0], vec![8.0], vec![7.0], vec![6.0]];
        let c = find_linear_monovariant(&trajectory, true).expect("x is decreasing");
        assert!(c[0] > 0.0, "coefficient on x should be positive: {:?}", c);
    }

    #[test]
    fn test_monovariant_two_var_always_sum_decreasing() {
        // (a, b) → (a-1, b-1): sum a+b monotonically decreases.
        let trajectory = vec![
            vec![10.0, 5.0],
            vec![9.0, 4.0],
            vec![8.0, 3.0],
            vec![7.0, 2.0],
        ];
        let c = find_linear_monovariant(&trajectory, true).expect("sum decreases");
        // Either e_0, e_1, or (1, 1)-direction is valid
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_monovariant_rejects_oscillation() {
        // State oscillates — no monovariant exists.
        let trajectory = vec![vec![1.0], vec![2.0], vec![1.0], vec![2.0]];
        assert!(find_linear_monovariant(&trajectory, true).is_none());
        assert!(find_linear_monovariant(&trajectory, false).is_none());
    }

    #[test]
    fn test_monovariant_strictly_increasing_2d() {
        // (a, b) → (a+1, b+1): sum strictly increases.
        let trajectory = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let c = find_linear_monovariant(&trajectory, false).expect("sum increases");
        assert!(c.iter().any(|v| *v > 0.0));
    }
}
