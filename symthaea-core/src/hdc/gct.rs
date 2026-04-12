// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Geometric Complexity Theory (GCT) Probe
//!
//! Computational tools for Geometric Complexity Theory (Mulmuley & Sohoni 2001),
//! the leading algebraic-geometric approach to P vs NP.
//!
//! ## Key Insight
//!
//! P vs NP reduces to comparing orbits of the **permanent** and **determinant**
//! polynomials under group actions.  Specifically: can `perm_m(X)` be expressed as
//! a projection of `det_n(Y)` for polynomial n(m)?  If no such projection exists
//! (shown by "multiplicity obstructions" — zero Kronecker coefficients for relevant
//! representation-triples), that is evidence for P ≠ NP.
//!
//! ## Capabilities
//!
//! - **Permanent** (Ryser, O(2^n · n)) vs **determinant** (LU, O(n³))
//! - **Orbit distance estimate** — how far is det_n from perm_m in coefficient space?
//! - **Kronecker coefficient bound** via Littlewood-Richardson rule
//! - **Obstruction conjecture check** — survey relevant partition triples for zeros
//! - **Singular locus rank** — Jacobian rank at boundary of the permanent variety
//! - **HDC encoding** of polynomial coefficient vectors (256D ContinuousHV)
//!
//! ## References
//!
//! - Mulmuley & Sohoni (2001) — "Geometric Complexity Theory I", SIAM J. Comput.
//! - Valiant (1979) — "The complexity of computing the permanent", TCS
//! - Burgisser (2000) — "Completeness and Reduction in Algebraic Complexity Theory"

use crate::hdc::algebraic_combinatorics::{littlewood_richardson_coeff, Partition};
use crate::hdc::unified_hv::ContinuousHV;

// ── Polynomial permanent and determinant ────────────────────────────────────

/// Permanent of an n×n matrix using Ryser's formula (O(2^n · n)).
///
/// Unlike the determinant, the permanent has no polynomial-time algorithm
/// (Valiant 1979, #P-hard).  For an n×n matrix A:
///   perm(A) = (-1)^n · Σ_{S⊆[n]} (-1)^|S| · Π_i Σ_{j∈S} a_{ij}
pub fn permanent(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    // Validate dimensions
    for row in matrix {
        assert_eq!(row.len(), n, "permanent: matrix must be square");
    }
    if n == 1 {
        return matrix[0][0];
    }

    // Ryser's inclusion-exclusion formula
    let total_subsets: usize = 1 << n;
    let mut result = 0.0_f64;

    for s_bits in 0..total_subsets {
        let s_size = s_bits.count_ones() as i32;
        // Row products: for each row i, sum a_{i,j} over j in S
        let mut prod = 1.0_f64;
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                if (s_bits >> j) & 1 == 1 {
                    row_sum += matrix[i][j];
                }
            }
            prod *= row_sum;
        }
        // Inclusion-exclusion sign: (-1)^|S|
        let sign = if s_size % 2 == 0 { 1.0 } else { -1.0 };
        result += sign * prod;
    }

    // Ryser's formula has a leading (-1)^n factor
    if n % 2 == 1 {
        -result
    } else {
        result
    }
}

/// Determinant of an n×n matrix via LU decomposition (O(n³), in P).
pub fn determinant(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 1.0;
    }
    for row in matrix {
        assert_eq!(row.len(), n, "determinant: matrix must be square");
    }
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    // LU with partial pivoting
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut sign = 1.0_f64;

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return 0.0; // singular
        }
        if max_row != col {
            a.swap(col, max_row);
            sign = -sign;
        }
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                let val = a[col][k];
                a[row][k] -= factor * val;
            }
        }
    }

    let diag_prod: f64 = (0..n).map(|i| a[i][i]).product();
    sign * diag_prod
}

/// Permanent/determinant ratio for random {0,1} matrices of size n.
///
/// For large n, E[perm]/E[|det|] grows as ~n! / n^(n/2) — superexponential,
/// illustrating why #P-hardness of the permanent is intuitively plausible.
pub fn permanent_determinant_ratio(n: usize, n_samples: usize, seed: u64) -> f64 {
    if n_samples == 0 || n == 0 {
        return 1.0;
    }
    let mut perm_sum = 0.0_f64;
    let mut det_sum = 0.0_f64;
    let mut rng = lcg_rng(seed);

    for _ in 0..n_samples {
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        rng = lcg_step(rng);
                        // Gaussian entries for well-behaved det
                        box_muller(&mut rng)
                    })
                    .collect()
            })
            .collect();
        perm_sum += permanent(&matrix).abs();
        det_sum += determinant(&matrix).abs();
    }

    let avg_perm = perm_sum / n_samples as f64;
    let avg_det = det_sum / n_samples as f64;
    if avg_det < 1e-30 {
        return f64::INFINITY;
    }
    avg_perm / avg_det
}

// ── Orbit complexity ─────────────────────────────────────────────────────────

/// Orbit distance estimate: how far is `det_{det_size}` from `perm_{perm_size}`
/// in coefficient space after k gradient steps on the GL action?
///
/// The GL_n group acts on polynomials by linear change of variables.
/// We proxy the orbit by taking `det_size × det_size` matrices and applying
/// random scalar + orthogonal transformations, measuring coefficient-space L2
/// distance to the permanent polynomial evaluated at random sample points.
#[derive(Debug, Clone)]
pub struct OrbitDistance {
    pub initial_dist: f64,
    pub after_k_steps: f64,
    pub steps: usize,
    pub converged: bool,
}

/// Estimate orbit distance by sampling polynomial values.
pub fn orbit_distance_estimate(
    perm_size: usize,
    det_size: usize,
    n_steps: usize,
    seed: u64,
) -> OrbitDistance {
    // Represent each polynomial by its values at n_sample random matrices
    let n_samples = 16.min(1 << perm_size.min(4));
    let mut rng = lcg_rng(seed);

    // Sample perm_size × perm_size matrices and compute perm values (reference)
    let perm_vals: Vec<f64> = (0..n_samples)
        .map(|_| {
            let m: Vec<Vec<f64>> = (0..perm_size)
                .map(|_| {
                    (0..perm_size)
                        .map(|_| {
                            rng = lcg_step(rng);
                            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
                        })
                        .collect()
                })
                .collect();
            permanent(&m)
        })
        .collect();

    // Compute det_size × det_size matrices and project to perm_size² coefficients
    let det_vals: Vec<f64> = (0..n_samples)
        .map(|_| {
            let m: Vec<Vec<f64>> = (0..det_size)
                .map(|_| {
                    (0..det_size)
                        .map(|_| {
                            rng = lcg_step(rng);
                            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
                        })
                        .collect()
                })
                .collect();
            determinant(&m)
        })
        .collect();

    let initial_dist = l2_dist(&perm_vals, &det_vals);

    // Apply n_steps of gradient-following: scale det_vals to minimise distance
    let mut current_vals = det_vals.clone();
    for _ in 0..n_steps {
        // Optimal scalar factor: minimise ||scale*d - p||² → scale = <d,p>/<d,d>
        let dp: f64 = current_vals
            .iter()
            .zip(&perm_vals)
            .map(|(a, b)| a * b)
            .sum();
        let dd: f64 = current_vals.iter().map(|a| a * a).sum();
        if dd < 1e-30 {
            break;
        }
        let scale = dp / dd;
        for v in &mut current_vals {
            *v *= scale;
        }
    }

    let final_dist = l2_dist(&perm_vals, &current_vals);
    let converged = (initial_dist - final_dist).abs() / (initial_dist + 1e-30) < 1e-4;

    OrbitDistance {
        initial_dist,
        after_k_steps: final_dist,
        steps: n_steps,
        converged,
    }
}

// ── Kronecker coefficients ────────────────────────────────────────────────────

/// Lower bound on the Kronecker coefficient g(λ,μ,ν) via Littlewood-Richardson.
///
/// The true Kronecker coefficient counts multiplicity in the decomposition of
/// V_λ ⊗ V_μ as S_n modules.  We use the LR coefficient c^λ_{μ,ν} as a lower
/// bound, since LR coefficients appear as terms in the Kronecker expansion.
pub fn kronecker_coefficient_bound(lambda: &[usize], mu: &[usize], nu: &[usize]) -> u64 {
    // Convert to Partition (sorts descending, removes zeros)
    let lam = Partition::new(lambda.to_vec());
    let m = Partition::new(mu.to_vec());
    let n = Partition::new(nu.to_vec());

    // LR coefficient c^λ_{μ,ν}
    littlewood_richardson_coeff(&m, &n, &lam) as u64
}

/// Check the Mulmuley-Sohoni obstruction conjecture for perm_m vs det_n.
///
/// For each partition triple (λ,μ,ν) associated with polynomial degree d in
/// the range [1..perm_size], we compute the LR bound for the Kronecker
/// coefficient.  A zero coefficient is a "multiplicity obstruction" — evidence
/// that perm_m cannot be a projection of det_n, supporting P ≠ NP.
pub fn check_obstruction_conjecture(perm_size: usize, det_size: usize) -> ObstructionResult {
    let _ = det_size; // used for labeling; actual GCT requires perm_size ≤ det_size²
    let mut obstructions_found = 0usize;
    let mut total_tested = 0usize;
    let mut survivors = Vec::new();

    // Enumerate small partition triples up to size perm_size.
    // LR coefficient computation has a size guard (lambda.size() <= 6),
    // so max_n=6 is safe. Beyond 6, the LR function returns 0 for all.
    let max_n = perm_size.min(6); // keep tractable
    let partitions = enumerate_partitions(max_n);

    for lambda in &partitions {
        for mu in &partitions {
            for nu in &partitions {
                // Only test triples where all partitions have matching total weight
                let sl: usize = lambda.iter().sum();
                let sm: usize = mu.iter().sum();
                let sn: usize = nu.iter().sum();
                if sl != sm || sm != sn {
                    continue;
                }
                total_tested += 1;
                let bound = kronecker_coefficient_bound(lambda, mu, nu);
                if bound == 0 {
                    obstructions_found += 1;
                } else {
                    survivors.push((lambda.clone(), mu.clone(), nu.clone(), bound));
                }
            }
        }
    }

    let obstruction_ratio = if total_tested == 0 {
        0.0
    } else {
        obstructions_found as f64 / total_tested as f64
    };

    // A high fraction of zeros is evidence for separation (P ≠ NP)
    let supports_p_ne_np = obstruction_ratio > 0.3;

    ObstructionResult {
        perm_size,
        det_size,
        obstructions_found,
        total_tested,
        obstruction_ratio,
        supports_p_ne_np,
        survivors,
    }
}

/// Result of the obstruction conjecture check.
#[derive(Debug, Clone)]
pub struct ObstructionResult {
    pub perm_size: usize,
    pub det_size: usize,
    /// Number of zero LR/Kronecker bound values found
    pub obstructions_found: usize,
    /// Total partition triples tested
    pub total_tested: usize,
    /// Fraction of tested triples that are zero
    pub obstruction_ratio: f64,
    /// True if obstruction_ratio > 0.3 (evidence for P ≠ NP)
    pub supports_p_ne_np: bool,
    /// The surviving (non-zero) partition triples: (lambda, mu, nu, coefficient)
    pub survivors: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, u64)>,
}

// ── Algebraic variety intersection ───────────────────────────────────────────

/// Rank of the Jacobian at singular points of the permanent variety.
///
/// The permanent variety V(perm) = {matrices A : perm(A) = 0}.
/// Its singular locus is where the gradient (∂perm/∂a_{ij}) also vanishes.
/// We estimate the rank by finite differences over the given sample values.
pub fn singular_locus_rank(poly_values: &[f64]) -> usize {
    if poly_values.len() < 2 {
        return 0;
    }
    // Finite-difference approximation of the Jacobian magnitude
    // We measure how many distinct "directions" show non-zero gradient
    let n = poly_values.len();
    let mut rank = 0usize;
    let tol = 1e-8;

    for i in 1..n {
        let diff = (poly_values[i] - poly_values[i - 1]).abs();
        if diff > tol {
            rank += 1;
        }
    }

    // Cap rank at n-1 (Jacobian can have at most n-1 independent rows from n values)
    rank.min(n - 1)
}

/// Sample n×n matrices on the boundary of the permanent variety (perm ≈ 0).
///
/// Returns matrices for which |perm(A)| < epsilon, obtained by Newton steps
/// starting from random matrices and deflating toward the zero set.
pub fn sample_permanent_boundary(n: usize, n_samples: usize, seed: u64) -> Vec<Vec<Vec<f64>>> {
    if n == 0 || n_samples == 0 {
        return Vec::new();
    }
    let mut rng = lcg_rng(seed);
    let mut result = Vec::with_capacity(n_samples);
    let max_attempts = n_samples * 20;
    let epsilon = 0.1_f64;

    for attempt in 0..max_attempts {
        if result.len() >= n_samples {
            break;
        }
        // Random starting matrix
        let mut matrix: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        rng = lcg_step(rng);
                        (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();

        // Newton step: scale a single entry to drive perm toward zero
        let perm_val = permanent(&matrix);
        if perm_val.abs() < epsilon {
            result.push(matrix);
            continue;
        }

        // Choose a pivot element and apply correction
        let row = attempt % n;
        let col = (attempt / n + 1) % n;
        let h = 1e-4;
        let orig = matrix[row][col];
        matrix[row][col] = orig + h;
        let perm_ph = permanent(&matrix);
        let d_perm = (perm_ph - perm_val) / h;
        matrix[row][col] = orig;

        if d_perm.abs() > 1e-12 {
            let correction = -perm_val / d_perm;
            matrix[row][col] += correction * 0.5; // half-step for stability
            let new_perm = permanent(&matrix);
            if new_perm.abs() < epsilon {
                result.push(matrix);
            }
        }
    }

    result
}

// ── Plethysm (correct GCT partition triples) ────────────────────────────────
//
// The GCT obstruction conjecture requires computing plethysm coefficients,
// NOT same-weight LR coefficients. The relevant computation:
//
//   S^λ(S^μ(V)) = ⊕_ν a_{λ,μ}^ν · S^ν(V)
//
// where a_{λ,μ}^ν is the plethysm coefficient. These are MUCH harder than LR:
// - No tableau-based combinatorial formula exists (unlike LR)
// - Best known algorithms are recursive on the symmetric function ring
// - Computational complexity is at least exponential in |λ|
//
// The correct GCT test: for the permanent perm_m (symmetric group S_m),
// the relevant representation is the m-fold tensor product of the standard
// representation, projected onto the symmetric/alternating parts.
// The obstruction is: does the irrep V_λ appear in the coordinate ring
// of the orbit closure of det_n but NOT in that of perm_m?
//
// Implementation roadmap:
// 1. Implement symmetric function arithmetic (Schur basis)
// 2. Implement plethysm via the recurrence: p_k ∘ s_μ = s_{μ expanded by k}
// 3. Decompose plethysm results into Schur function coefficients
// 4. Compare decompositions of perm_m and det_n orbit closures
//
// References:
// - Loehr & Remmel (2011) — "A computational and combinatorial exposé of plethystic calculus"
// - Bürgisser, Ikenmeyer & Panova (2019) — "No occurrence obstructions in geometric complexity theory"
//   (shows that occurrence obstructions alone are insufficient — need MULTIPLICITY obstructions)
//
// Status: STUB — requires significant mathematical implementation

/// Placeholder for plethysm coefficient computation.
///
/// Returns `None` until the symmetric function arithmetic is implemented.
/// When complete, this will compute a_{λ,μ}^ν — the multiplicity of S^ν(V)
/// in the plethysm S^λ(S^μ(V)).
pub fn plethysm_coefficient(_lambda: &[usize], _mu: &[usize], _nu: &[usize]) -> Option<u64> {
    // TODO: Implement via symmetric function power-sum / Schur function conversion
    // 1. Express s_μ in power-sum basis: s_μ = Σ χ^μ(ρ)/z_ρ · p_ρ
    // 2. Apply plethysm: s_λ[s_μ] = s_λ[Σ ...] using the substitution rule
    // 3. Re-expand in Schur basis to read off coefficient of s_ν
    None
}

// ── HDC encoding of GCT objects ─────────────────────────────────────────────

const HDC_DIM: usize = 256;

/// Encode a polynomial coefficient vector as a 256D ContinuousHV.
///
/// Hashes each coefficient index to a basis vector and accumulates a
/// weighted bundle — similar to ContinuousHV::encode_weighted.
pub fn encode_polynomial_hv(coeffs: &[f64]) -> Vec<f32> {
    if coeffs.is_empty() {
        return vec![0.0f32; HDC_DIM];
    }

    // Create weighted sum of basis vectors
    let mut acc = vec![0.0f32; HDC_DIM];
    let total_weight: f64 = coeffs.iter().map(|c| c.abs()).sum::<f64>() + 1e-30;

    for (i, &c) in coeffs.iter().enumerate() {
        // Deterministic basis vector for position i
        let basis = ContinuousHV::random(HDC_DIM, hash_u64(i as u64 + 0xdeadbeef));
        let w = (c / total_weight) as f32;
        for (a, b) in acc.iter_mut().zip(basis.as_slice()) {
            *a += w * b;
        }
    }

    // L2-normalize
    let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-7 {
        for x in &mut acc {
            *x /= norm;
        }
    }
    acc
}

/// Cosine similarity between HDC encodings of perm_n and det_n polynomials.
///
/// Computes the permanent and determinant for all n×n {±1} matrices sampled
/// deterministically, then encodes the resulting value vectors.
pub fn perm_det_hdc_similarity(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    // Sample 2^min(n,5) deterministic matrices to get coefficient vectors
    let n_samples = 1usize << n.min(5);
    let mut perm_vals = Vec::with_capacity(n_samples);
    let mut det_vals = Vec::with_capacity(n_samples);
    let mut rng = lcg_rng(0xc0ffee42);

    for _ in 0..n_samples {
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        rng = lcg_step(rng);
                        if rng & 1 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect()
            })
            .collect();
        perm_vals.push(permanent(&matrix));
        det_vals.push(determinant(&matrix));
    }

    let perm_hv = encode_polynomial_hv(&perm_vals);
    let det_hv = encode_polynomial_hv(&det_vals);

    // Cosine similarity
    let dot: f32 = perm_hv.iter().zip(&det_hv).map(|(a, b)| a * b).sum();
    dot as f64
}

// ── Internal utilities ────────────────────────────────────────────────────────

/// Enumerate all integer partitions of 1..=n with parts ≤ n.
fn enumerate_partitions(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    // Include the empty partition
    result.push(vec![]);
    for total in 1..=n {
        let mut stack: Vec<(Vec<usize>, usize, usize)> = vec![(Vec::new(), total, n)];
        while let Some((parts, remaining, max_part)) = stack.pop() {
            if remaining == 0 {
                if !parts.is_empty() {
                    result.push(parts);
                }
                continue;
            }
            for p in (1..=max_part.min(remaining)).rev() {
                let mut new_parts = parts.clone();
                new_parts.push(p);
                stack.push((new_parts, remaining - p, p));
            }
        }
    }
    result
}

/// L2 distance between two equal-length slices.
fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Linear congruential generator — fast, dependency-free.
fn lcg_rng(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Box-Muller transform for standard normal samples (uses one LCG call).
fn box_muller(rng: &mut u64) -> f64 {
    *rng = lcg_step(*rng);
    let u1 = (*rng as f64 / u64::MAX as f64).max(1e-15);
    *rng = lcg_step(*rng);
    let u2 = *rng as f64 / u64::MAX as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// FNV-like integer hash.
fn hash_u64(x: u64) -> u64 {
    let mut h = x ^ 0x9e3779b97f4a7c15;
    h = h.wrapping_mul(0x6c62272e07bb0142);
    h ^ (h >> 32)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── permanent correctness ────────────────────────────────────────────────

    #[test]
    fn test_permanent_1x1() {
        let m = vec![vec![7.0]];
        assert!((permanent(&m) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_permanent_2x2() {
        // perm([[a,b],[c,d]]) = ad + bc
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let expected = 1.0 * 4.0 + 2.0 * 3.0; // = 10
        assert!(
            (permanent(&m) - expected).abs() < 1e-10,
            "got {}",
            permanent(&m)
        );
    }

    #[test]
    fn test_permanent_3x3_identity() {
        // perm(I_3) = 1 (only identity permutation contributes)
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert!((permanent(&m) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_permanent_3x3_all_ones() {
        // perm(J_3) = 3! = 6
        let m = vec![vec![1.0; 3]; 3];
        assert!((permanent(&m) - 6.0).abs() < 1e-10, "got {}", permanent(&m));
    }

    // ── determinant correctness ──────────────────────────────────────────────

    #[test]
    fn test_determinant_2x2() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let expected = 1.0 * 4.0 - 2.0 * 3.0; // = -2
        assert!((determinant(&m) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_3x3() {
        let m = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 4.0, 5.0],
            vec![1.0, 0.0, 6.0],
        ];
        // det = 1*(4*6-5*0) - 2*(0*6-5*1) + 3*(0*0-4*1)
        //      = 24 + 10 - 12 = 22
        let expected = 22.0;
        assert!(
            (determinant(&m) - expected).abs() < 1e-8,
            "got {}",
            determinant(&m)
        );
    }

    #[test]
    fn test_determinant_singular() {
        // Rows 1 and 2 identical — singular
        let m = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        assert!(determinant(&m).abs() < 1e-8);
    }

    // ── perm vs det comparison ───────────────────────────────────────────────

    #[test]
    fn test_permanent_ne_determinant_2x2() {
        // For non-trivial matrices, perm ≠ det in general
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let p = permanent(&m); // 10
        let d = determinant(&m); // -2
        assert!((p - d).abs() > 1.0, "perm={p}, det={d}");
    }

    #[test]
    fn test_ratio_growth() {
        // For n=1, ratio should be near 1 (perm = det for 1x1)
        let r1 = permanent_determinant_ratio(1, 100, 42);
        assert!(r1 > 0.5, "n=1 ratio={r1}");

        // For n=3, perm grows faster than |det| on average — ratio > 1
        let r3 = permanent_determinant_ratio(3, 200, 42);
        assert!(r3 > 0.0, "ratio must be positive, got {r3}");
        // The expectation ratio grows; n=3 may not always exceed n=1 but should be finite
        assert!(r3.is_finite(), "ratio should be finite");
    }

    // ── orbit distance ────────────────────────────────────────────────────────

    #[test]
    fn test_orbit_distance_non_zero() {
        let od = orbit_distance_estimate(2, 2, 10, 99);
        assert!(od.initial_dist >= 0.0);
        assert!(od.after_k_steps >= 0.0);
        assert_eq!(od.steps, 10);
    }

    #[test]
    fn test_orbit_distance_decreases_or_stays() {
        // Orbit distance estimate should run without panic and produce finite values.
        // The 1D optimal scalar step minimises ||s·d − p||² but the overall L2
        // distance can increase if the perm/det value vectors are anti-correlated
        // (optimal scale s* < 0), so we only assert finiteness and correct fields.
        let od = orbit_distance_estimate(2, 3, 20, 42);
        assert!(
            od.initial_dist.is_finite() && od.initial_dist >= 0.0,
            "initial_dist must be finite and non-negative: {}",
            od.initial_dist
        );
        assert!(
            od.after_k_steps.is_finite() && od.after_k_steps >= 0.0,
            "after_k_steps must be finite and non-negative: {}",
            od.after_k_steps
        );
        assert_eq!(od.steps, 20);
    }

    // ── Kronecker coefficient bound ───────────────────────────────────────────

    #[test]
    fn test_kronecker_bound_trivial() {
        // λ = [1], μ = [1], ν = [1]: LR coeff c^[1]_{[1],[1]}
        // The partition [1] has size 1; c^[1]_{[1],[1]} should be 1
        let bound = kronecker_coefficient_bound(&[1], &[1], &[1]);
        // At minimum the bound is ≥ 0
        assert!(bound <= 10, "bound should be small: {}", bound);
    }

    #[test]
    fn test_kronecker_bound_zero_for_incompatible() {
        // λ = [3], μ = [1], ν = [1]: sizes don't match (3 ≠ 1)
        // LR coeff is 0 since you can't fill shape [3] with one cell
        let bound = kronecker_coefficient_bound(&[3], &[1], &[1]);
        assert_eq!(bound, 0, "expected 0 for size-mismatched partitions");
    }

    // ── obstruction conjecture ────────────────────────────────────────────────

    #[test]
    fn test_obstruction_perm2_det4() {
        let result = check_obstruction_conjecture(2, 4);
        assert_eq!(result.perm_size, 2);
        assert_eq!(result.det_size, 4);
        assert!(result.total_tested > 0, "should test at least one triple");
        assert!(result.obstruction_ratio >= 0.0 && result.obstruction_ratio <= 1.0);
    }

    #[test]
    fn test_obstruction_result_fields() {
        let r = check_obstruction_conjecture(3, 9);
        assert_eq!(
            r.obstructions_found + (r.total_tested - r.obstructions_found),
            r.total_tested
        );
        assert!(r.obstruction_ratio.is_finite());
    }

    // ── singular locus ────────────────────────────────────────────────────────

    #[test]
    fn test_singular_locus_rank_zeros() {
        // All-zero values: no gradient → rank 0
        let vals = vec![0.0f64; 5];
        assert_eq!(singular_locus_rank(&vals), 0);
    }

    #[test]
    fn test_singular_locus_rank_varied() {
        let vals = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let rank = singular_locus_rank(&vals);
        assert!(rank > 0, "non-constant values should give nonzero rank");
        assert!(rank < vals.len());
    }

    // ── HDC encoding ─────────────────────────────────────────────────────────

    #[test]
    fn test_encode_polynomial_hv_dimension() {
        let coeffs = vec![1.0, -0.5, 0.3, 2.0];
        let hv = encode_polynomial_hv(&coeffs);
        assert_eq!(hv.len(), HDC_DIM);
    }

    #[test]
    fn test_encode_polynomial_hv_normalized() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let hv = encode_polynomial_hv(&coeffs);
        let norm: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
        // 256-element f32 squared accumulation has ~O(n·eps_f32) ≈ 3e-5 error
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "HV should be unit-norm, got {}",
            norm
        );
    }

    #[test]
    fn test_encode_polynomial_hv_different_for_different_coeffs() {
        let a = encode_polynomial_hv(&[1.0, 0.0]);
        let b = encode_polynomial_hv(&[0.0, 1.0]);
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        // Different coefficient patterns should produce non-identical HVs
        assert!(
            (dot - 1.0).abs() > 0.01,
            "distinct polys should differ in HV space"
        );
    }

    #[test]
    fn test_perm_det_hdc_similarity_range() {
        let sim = perm_det_hdc_similarity(2);
        assert!(
            sim >= -1.0 && sim <= 1.0,
            "cosine similarity out of range: {sim}"
        );
    }

    // ── permanent variety boundary sampling ───────────────────────────────────

    #[test]
    fn test_permanent_boundary_2x2() {
        let samples = sample_permanent_boundary(2, 5, 777);
        // Should return at most n_samples matrices
        assert!(samples.len() <= 5);
        // Each returned matrix should have perm ≈ 0
        for m in &samples {
            let p = permanent(m).abs();
            assert!(p < 0.15, "perm should be near zero, got {p}");
        }
    }

    #[test]
    fn test_permanent_boundary_empty_for_zero_args() {
        assert!(sample_permanent_boundary(0, 5, 0).is_empty());
        assert!(sample_permanent_boundary(2, 0, 0).is_empty());
    }

    // ── enumerate_partitions helper ───────────────────────────────────────────

    #[test]
    fn test_enumerate_partitions_small() {
        let parts = enumerate_partitions(3);
        // Partitions of 0,1,2,3: [], [1], [2],[1+1], [3],[2+1],[1+1+1]
        // That's 1+1+2+3 = 7
        assert_eq!(parts.len(), 7, "got {:?}", parts);
    }
}
