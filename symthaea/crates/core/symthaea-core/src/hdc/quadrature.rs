// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Numerical Quadrature Engine
//!
//! Methods for computing definite integrals numerically, with consciousness-coupled
//! HDC encoding and fundamental theorem verification.
//!
//! ## Methods
//!
//! - **Composite Simpson's rule**: O(h^4) convergence
//! - **Gauss-Legendre quadrature**: High accuracy with few evaluations (up to 10-point)
//! - **Adaptive Simpson**: Recursive subdivision until tolerance met
//!
//! ## Fundamental Theorem Verification
//!
//! When a symbolic antiderivative is available, the engine can verify the fundamental
//! theorem of calculus: the numerical integral should equal F(b) - F(a).
//! Agreement increases Phi (multi-path verification).

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_TOLERANCE: f64 = 1e-10;
const MAX_RECURSION_DEPTH: usize = 50;
const DEFAULT_INTERVALS: usize = 100;

// ─── Types ───────────────────────────────────────────────────────────────────

/// Method used for numerical integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuadratureMethod {
    Simpson,
    GaussLegendre,
    AdaptiveSimpson,
}

impl std::fmt::Display for QuadratureMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuadratureMethod::Simpson => write!(f, "Simpson"),
            QuadratureMethod::GaussLegendre => write!(f, "Gauss-Legendre"),
            QuadratureMethod::AdaptiveSimpson => write!(f, "Adaptive Simpson"),
        }
    }
}

/// Result of a quadrature operation
#[derive(Debug, Clone)]
pub struct QuadratureResult {
    /// Computed integral value
    pub value: f64,
    /// Integration bounds [a, b]
    pub bounds: (f64, f64),
    /// Method used
    pub method: QuadratureMethod,
    /// Number of function evaluations
    pub evaluations: usize,
    /// Estimated error (for adaptive methods)
    pub error_estimate: Option<f64>,
    /// Phi from the computation
    pub phi: f64,
    /// HDC encoding
    pub encoding: BinaryHV,
}

/// Result of fundamental theorem verification
#[derive(Debug, Clone)]
pub struct FundamentalTheoremResult {
    /// Numerical integral value
    pub numerical: f64,
    /// Symbolic antiderivative evaluation F(b) - F(a)
    pub symbolic: f64,
    /// Absolute difference
    pub difference: f64,
    /// Whether they agree within tolerance
    pub verified: bool,
    /// Combined Phi (with agreement bonus)
    pub phi: f64,
}

// ─── Gauss-Legendre Nodes and Weights ────────────────────────────────────────

/// Pre-computed Gauss-Legendre nodes and weights for [-1, 1]
fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.5773502691896258, 0.5773502691896258],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        ),
        4 => (
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.538_469_310_105_683,
                0.0,
                0.538_469_310_105_683,
                0.906_179_845_938_664,
            ],
            vec![
                0.236_926_885_056_189_1,
                0.478_628_670_499_366_5,
                0.568_888_888_888_889,
                0.478_628_670_499_366_5,
                0.236_926_885_056_189_1,
            ],
        ),
        7 => (
            vec![
                -0.949_107_912_342_759,
                -0.741_531_185_599_394,
                -0.405_845_151_377_397,
                0.0,
                0.405_845_151_377_397,
                0.741_531_185_599_394,
                0.949_107_912_342_759,
            ],
            vec![
                0.129_484_966_168_870,
                0.279_705_391_489_277,
                0.381_830_050_505_119,
                0.417_959_183_673_469,
                0.381_830_050_505_119,
                0.279_705_391_489_277,
                0.129_484_966_168_870,
            ],
        ),
        10 => (
            vec![
                -0.973_906_528_517_172,
                -0.865_063_366_688_985,
                -0.679_409_568_299_024,
                -0.433_395_394_129_247,
                -0.148_874_338_981_631,
                0.148_874_338_981_631,
                0.433_395_394_129_247,
                0.679_409_568_299_024,
                0.865_063_366_688_985,
                0.973_906_528_517_172,
            ],
            vec![
                0.066_671_344_308_688,
                0.149_451_349_150_581,
                0.219_086_362_515_982,
                0.269_266_719_309_996,
                0.295_524_224_714_753,
                0.295_524_224_714_753,
                0.269_266_719_309_996,
                0.219_086_362_515_982,
                0.149_451_349_150_581,
                0.066_671_344_308_688,
            ],
        ),
        _ => panic!(
            "Gauss-Legendre nodes/weights not available for n={n}. Supported: 1,2,3,4,5,7,10"
        ),
    }
}

/// Return pre-computed Gauss-Legendre nodes and weights on `[-1, 1]`.
///
/// Supported orders: 1, 2, 3, 4, 5, 7, 10. Panics on unsupported `n`.
/// This is a public wrapper around the internal `gauss_legendre_nodes` function.
pub fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    gauss_legendre_nodes(n)
}

// ─── Free-standing Functions ─────────────────────────────────────────────────
// Pure numerical routines with no HDC coupling.

/// Lightweight result for the free-standing quadrature functions.
#[derive(Debug, Clone, Copy)]
pub struct QuadratureResultBasic {
    /// Approximate value of the integral.
    pub value: f64,
    /// Estimated absolute error (0.0 when no estimate is available).
    pub error_estimate: f64,
    /// Number of function evaluations used.
    pub evaluations: usize,
}

/// Composite trapezoidal rule for `∫[a,b] f(x) dx` with `n` sub-intervals.
///
/// Error is O(h^2) where h = (b-a)/n.
pub fn trapezoid<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> QuadratureResultBasic {
    if n == 0 || (b - a).abs() < f64::EPSILON {
        return QuadratureResultBasic {
            value: 0.0,
            error_estimate: 0.0,
            evaluations: if n == 0 { 0 } else { 2 },
        };
    }
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    QuadratureResultBasic {
        value: sum * h,
        error_estimate: 0.0,
        evaluations: n + 1,
    }
}

/// Composite Simpson's 1/3 rule (free-standing).
///
/// `n` must be even; it is rounded up if odd. Error is O(h^4).
pub fn simpson_free<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> QuadratureResultBasic {
    let n = if n < 2 {
        2
    } else if n % 2 != 0 {
        n + 1
    } else {
        n
    };
    if (b - a).abs() < f64::EPSILON {
        return QuadratureResultBasic {
            value: 0.0,
            error_estimate: 0.0,
            evaluations: 2,
        };
    }
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * f(a + i as f64 * h);
    }
    QuadratureResultBasic {
        value: sum * h / 3.0,
        error_estimate: 0.0,
        evaluations: n + 1,
    }
}

/// Gauss-Legendre quadrature on `[a, b]` (free-standing).
pub fn gauss_legendre_basic<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    n_points: usize,
) -> QuadratureResultBasic {
    if (b - a).abs() < f64::EPSILON {
        return QuadratureResultBasic {
            value: 0.0,
            error_estimate: 0.0,
            evaluations: 0,
        };
    }
    let (nodes, weights) = gauss_legendre_nodes_weights(n_points);
    let half_len = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let mut sum = 0.0;
    for (xi, wi) in nodes.iter().zip(weights.iter()) {
        sum += wi * f(mid + half_len * xi);
    }
    QuadratureResultBasic {
        value: sum * half_len,
        error_estimate: 0.0,
        evaluations: n_points,
    }
}

/// Adaptive Simpson's rule (free-standing) with recursive subdivision.
pub fn adaptive_simpson_basic<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> QuadratureResultBasic {
    let mut evals = 0usize;
    let value = adaptive_simpson_recurse(f, a, b, tol, max_depth, &mut evals);
    QuadratureResultBasic {
        value,
        error_estimate: tol,
        evaluations: evals,
    }
}

fn adaptive_simpson_recurse<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    depth: usize,
    evals: &mut usize,
) -> f64 {
    let mid = 0.5 * (a + b);
    let h_whole = (b - a) / 6.0;
    let fa = f(a);
    let fm = f(mid);
    let fb = f(b);
    *evals += 3;
    let whole = h_whole * (fa + 4.0 * fm + fb);

    let lm = 0.5 * (a + mid);
    let rm = 0.5 * (mid + b);
    let flm = f(lm);
    let frm = f(rm);
    *evals += 2;

    let left = (mid - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - mid) / 6.0 * (fm + 4.0 * frm + fb);
    let refined = left + right;
    let err = (refined - whole) / 15.0;

    if depth == 0 || err.abs() < tol {
        refined + err
    } else {
        adaptive_simpson_recurse(f, a, mid, tol * 0.5, depth - 1, evals)
            + adaptive_simpson_recurse(f, mid, b, tol * 0.5, depth - 1, evals)
    }
}

/// Romberg integration (Richardson extrapolation on the trapezoid rule).
///
/// Builds a triangular table up to `max_order` rows. Stops early if the
/// difference between successive diagonal entries is below `tol`.
pub fn romberg<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    max_order: usize,
    tol: f64,
) -> QuadratureResultBasic {
    if (b - a).abs() < f64::EPSILON {
        return QuadratureResultBasic {
            value: 0.0,
            error_estimate: 0.0,
            evaluations: 0,
        };
    }
    let max_order = max_order.max(1);
    let mut r = vec![vec![0.0f64; max_order]; max_order];
    let mut total_evals = 2usize;

    // R[0][0] = single-interval trapezoid
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));

    for i in 1..max_order {
        let n = 1usize << i; // 2^i sub-intervals
        let h = (b - a) / n as f64;

        // Add new midpoints
        let mut new_sum = 0.0;
        for k in 0..(1usize << (i - 1)) {
            new_sum += f(a + (2 * k + 1) as f64 * h);
        }
        total_evals += 1usize << (i - 1);
        r[i][0] = 0.5 * r[i - 1][0] + h * new_sum;

        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }

        let err = (r[i][i] - r[i - 1][i - 1]).abs();
        if err < tol {
            return QuadratureResultBasic {
                value: r[i][i],
                error_estimate: err,
                evaluations: total_evals,
            };
        }
    }

    let last = max_order - 1;
    let err = if max_order >= 2 {
        (r[last][last] - r[last - 1][last - 1]).abs()
    } else {
        0.0
    };
    QuadratureResultBasic {
        value: r[last][last],
        error_estimate: err,
        evaluations: total_evals,
    }
}

// ─── Quadrature Engine ───────────────────────────────────────────────────────

/// The Hyperdimensional Quadrature Engine
pub struct QuadratureEngine;

impl QuadratureEngine {
    /// Composite Simpson's rule for ∫[a,b] f(x) dx.
    ///
    /// Uses n intervals (must be even). Error is O(h^4).
    pub fn simpson<F>(f: &F, a: f64, b: f64, n: usize) -> QuadratureResult
    where
        F: Fn(f64) -> f64,
    {
        let n = if n % 2 == 1 { n + 1 } else { n.max(2) };
        let h = (b - a) / n as f64;

        let mut sum = f(a) + f(b);
        for i in 1..n {
            let x = a + i as f64 * h;
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * f(x);
        }
        let value = sum * h / 3.0;

        QuadratureResult {
            value,
            bounds: (a, b),
            method: QuadratureMethod::Simpson,
            evaluations: n + 1,
            error_estimate: None,
            phi: Self::compute_phi(n + 1, QuadratureMethod::Simpson),
            encoding: Self::encode_integral(value, a, b),
        }
    }

    /// Gauss-Legendre quadrature for ∫[a,b] f(x) dx.
    ///
    /// High accuracy with few function evaluations. Supports n = 1..10 points.
    pub fn gauss_legendre<F>(f: &F, a: f64, b: f64, n_points: usize) -> QuadratureResult
    where
        F: Fn(f64) -> f64,
    {
        let (nodes, weights) = gauss_legendre_nodes(n_points);

        // Transform from [-1,1] to [a,b]: x = (b-a)/2 * t + (a+b)/2
        let half_range = (b - a) / 2.0;
        let mid = (a + b) / 2.0;

        let value: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&t, &w)| {
                let x = half_range * t + mid;
                w * f(x)
            })
            .sum::<f64>()
            * half_range;

        QuadratureResult {
            value,
            bounds: (a, b),
            method: QuadratureMethod::GaussLegendre,
            evaluations: n_points,
            error_estimate: None,
            phi: Self::compute_phi(n_points, QuadratureMethod::GaussLegendre),
            encoding: Self::encode_integral(value, a, b),
        }
    }

    /// Adaptive Simpson's rule with automatic error control.
    ///
    /// Recursively subdivides the interval until the estimated error
    /// is below the tolerance.
    pub fn adaptive_simpson<F>(f: &F, a: f64, b: f64, tol: f64) -> QuadratureResult
    where
        F: Fn(f64) -> f64,
    {
        let tol = if tol <= 0.0 { DEFAULT_TOLERANCE } else { tol };
        let mut eval_count = 0;

        let value = Self::adaptive_simpson_recursive(f, a, b, tol, 0, &mut eval_count);

        QuadratureResult {
            value,
            bounds: (a, b),
            method: QuadratureMethod::AdaptiveSimpson,
            evaluations: eval_count,
            error_estimate: Some(tol),
            phi: Self::compute_phi(eval_count, QuadratureMethod::AdaptiveSimpson),
            encoding: Self::encode_integral(value, a, b),
        }
    }

    fn adaptive_simpson_recursive<F>(
        f: &F,
        a: f64,
        b: f64,
        tol: f64,
        depth: usize,
        eval_count: &mut usize,
    ) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mid = (a + b) / 2.0;
        let h = (b - a) / 6.0;

        let fa = f(a);
        let fm = f(mid);
        let fb = f(b);
        *eval_count += 3;

        let whole = h * (fa + 4.0 * fm + fb);

        let left_mid = (a + mid) / 2.0;
        let right_mid = (mid + b) / 2.0;
        let flm = f(left_mid);
        let frm = f(right_mid);
        *eval_count += 2;

        let left = (mid - a) / 6.0 * (fa + 4.0 * flm + fm);
        let right = (b - mid) / 6.0 * (fm + 4.0 * frm + fb);
        let two_halves = left + right;

        let error = (two_halves - whole) / 15.0;

        if depth >= MAX_RECURSION_DEPTH || error.abs() < tol {
            two_halves + error
        } else {
            Self::adaptive_simpson_recursive(f, a, mid, tol / 2.0, depth + 1, eval_count)
                + Self::adaptive_simpson_recursive(f, mid, b, tol / 2.0, depth + 1, eval_count)
        }
    }

    /// Integrate with default settings (composite Simpson, 100 intervals)
    pub fn integrate<F>(f: &F, a: f64, b: f64) -> QuadratureResult
    where
        F: Fn(f64) -> f64,
    {
        Self::simpson(f, a, b, DEFAULT_INTERVALS)
    }

    /// Verify the fundamental theorem of calculus.
    ///
    /// Compares numerical integral of f over [a,b] with antiderivative F(b) - F(a).
    /// When they agree, Phi increases substantially.
    pub fn verify_fundamental_theorem<F, G>(
        f: &F,
        antiderivative: &G,
        a: f64,
        b: f64,
        tol: f64,
    ) -> FundamentalTheoremResult
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let numerical = Self::adaptive_simpson(f, a, b, tol).value;
        let symbolic = antiderivative(b) - antiderivative(a);
        let difference = (numerical - symbolic).abs();
        let verified = difference < tol.sqrt();

        let phi = if verified {
            // Strong agreement: multi-path verification success
            0.8
        } else {
            0.2
        };

        FundamentalTheoremResult {
            numerical,
            symbolic,
            difference,
            verified,
            phi,
        }
    }

    /// Multi-method integration: compute with Simpson and Gauss-Legendre,
    /// compare results for confidence.
    pub fn integrate_multipath<F>(f: &F, a: f64, b: f64, tol: f64) -> QuadratureResult
    where
        F: Fn(f64) -> f64,
    {
        let simpson_result = Self::simpson(f, a, b, DEFAULT_INTERVALS);
        let gauss_result = Self::gauss_legendre(f, a, b, 10);
        let adaptive_result = Self::adaptive_simpson(f, a, b, tol);

        // Check agreement
        let max_diff = (simpson_result.value - gauss_result.value)
            .abs()
            .max((simpson_result.value - adaptive_result.value).abs())
            .max((gauss_result.value - adaptive_result.value).abs());

        let agreement_bonus = if max_diff < tol.sqrt() { 0.5 } else { 0.0 };

        // Use adaptive result as best estimate
        QuadratureResult {
            value: adaptive_result.value,
            bounds: (a, b),
            method: QuadratureMethod::AdaptiveSimpson,
            evaluations: simpson_result.evaluations
                + gauss_result.evaluations
                + adaptive_result.evaluations,
            error_estimate: Some(max_diff),
            phi: adaptive_result.phi + agreement_bonus,
            encoding: adaptive_result.encoding,
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn compute_phi(evaluations: usize, method: QuadratureMethod) -> f64 {
        let efficiency = 1.0 / (1.0 + evaluations as f64 / 50.0);
        let method_bonus = match method {
            QuadratureMethod::Simpson => 0.0,
            QuadratureMethod::GaussLegendre => 0.15,
            QuadratureMethod::AdaptiveSimpson => 0.1,
        };
        efficiency + method_bonus
    }

    fn encode_integral(value: f64, a: f64, b: f64) -> BinaryHV {
        let int_prim = BinaryHV::random(seed_from_name("INTEGRAL_NUMERIC"));
        let val_hv = BinaryHV::random(seed_from_name(&format!("INT_VAL_{}", value.to_bits())));
        let bounds_hv = BinaryHV::random(seed_from_name(&format!(
            "BOUNDS_{}_{}",
            a.to_bits(),
            b.to_bits()
        )));
        int_prim.bind(&val_hv).bind(&bounds_hv)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GAUSS-KRONROD G7-K15 ADAPTIVE QUADRATURE
// ═══════════════════════════════════════════════════════════════════════════

/// Gauss-Kronrod G7-K15 quadrature result.
#[derive(Debug, Clone)]
pub struct GaussKronrodResult {
    /// Integral estimate (K15 value)
    pub value: f64,
    /// Error estimate |K15 - G7|
    pub error: f64,
    /// Total function evaluations
    pub evaluations: usize,
    /// Number of subintervals used
    pub intervals: usize,
}

/// Gauss-Kronrod 7-15 nodes and weights (from QUADPACK).
/// K15 uses 15 nodes; G7 uses the 7 odd-indexed nodes.
/// This is the standard adaptive quadrature workhorse.
fn gk15_nodes_weights() -> (&'static [f64], &'static [f64], &'static [f64]) {
    // K15 nodes (symmetric, positive half only — negate for other half)
    static NODES: [f64; 8] = [
        0.0,
        0.207_784_955_007_898_47,
        0.405_845_151_377_397_2,
        0.586_087_235_467_691_1,
        0.741_531_185_599_394_5,
        0.864_864_423_359_769_1,
        0.949_107_912_342_758_5,
        0.991_455_371_120_812_6,
    ];

    // K15 weights (15-point Kronrod)
    static WGK: [f64; 8] = [
        0.209_482_141_084_727_83,
        0.204_432_940_075_298_89,
        0.190_350_578_064_785_41,
        0.169_004_726_639_267,
        0.140_653_259_715_525_92,
        0.104_790_010_322_250_18,
        0.063_092_092_629_978_55,
        0.022_935_322_010_529_22,
    ];

    // G7 weights (7-point Gauss, only at odd-indexed K15 nodes)
    static WG: [f64; 4] = [
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.279_705_391_489_276_67,
        0.129_484_966_168_869_7,
    ];

    (&NODES, &WGK, &WG)
}

/// Single-interval Gauss-Kronrod G7-K15 quadrature.
/// Returns (k15_value, g7_value, error_estimate).
fn gk15_single<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64, f64) {
    let (nodes, wgk, wg) = gk15_nodes_weights();
    let center = 0.5 * (a + b);
    let half_length = 0.5 * (b - a);

    let f_center = f(center);
    let mut result_kronrod = f_center * wgk[0];
    let mut result_gauss = f_center * wg[0]; // center node is shared with G7

    // G7 uses nodes at K15 indices {0, 2, 4, 6} (even indices = the Gauss points)
    // K15 adds nodes at indices {1, 3, 5, 7} (odd indices = Kronrod extensions)
    let mut g_idx = 1usize; // next G7 weight index (0 was used for center)

    for i in 1..8 {
        let x = half_length * nodes[i];
        let fval1 = f(center - x);
        let fval2 = f(center + x);
        let fsum = fval1 + fval2;

        result_kronrod += fsum * wgk[i];

        // Even K15 indices (2, 4, 6) correspond to G7 nodes
        if i % 2 == 0 && g_idx < wg.len() {
            result_gauss += fsum * wg[g_idx];
            g_idx += 1;
        }
    }

    let k15 = result_kronrod * half_length;
    let g7 = result_gauss * half_length;
    let err = (k15 - g7).abs();

    (k15, g7, err)
}

/// Adaptive Gauss-Kronrod G7-K15 quadrature (QUADPACK-style).
///
/// Recursively subdivides intervals where the G7-K15 error estimate exceeds
/// the local tolerance. This is the standard algorithm used by QUADPACK's
/// DQAG routine, scipy.integrate.quad, and Mathematica's NIntegrate.
///
/// Parameters:
/// - `f`: integrand
/// - `a`, `b`: integration bounds
/// - `abs_tol`: absolute tolerance
/// - `rel_tol`: relative tolerance
/// - `max_subdivisions`: maximum number of interval subdivisions
pub fn gauss_kronrod_adaptive<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    abs_tol: f64,
    rel_tol: f64,
    max_subdivisions: usize,
) -> GaussKronrodResult {
    let mut total_value = 0.0;
    let mut total_error = 0.0;
    let mut total_evals = 0;
    let mut intervals = 0;

    // Priority queue of intervals: (error, a, b, value)
    let mut heap: Vec<(f64, f64, f64, f64)> = Vec::new();

    // Initial evaluation
    let (val, _, err) = gk15_single(f, a, b);
    total_value = val;
    total_error = err;
    total_evals = 15;
    intervals = 1;

    if err <= abs_tol.max(rel_tol * val.abs()) || max_subdivisions == 0 {
        return GaussKronrodResult {
            value: total_value,
            error: total_error,
            evaluations: total_evals,
            intervals,
        };
    }

    // Push initial interval for subdivision
    heap.push((err, a, b, val));

    for _ in 0..max_subdivisions {
        if heap.is_empty() {
            break;
        }

        // Find interval with largest error
        let max_idx = heap
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (old_err, ia, ib, old_val) = heap.swap_remove(max_idx);
        let mid = 0.5 * (ia + ib);

        // Subdivide
        let (val_left, _, err_left) = gk15_single(f, ia, mid);
        let (val_right, _, err_right) = gk15_single(f, mid, ib);
        total_evals += 30; // 15 per sub-interval

        // Update totals
        total_value += (val_left + val_right) - old_val;
        total_error += (err_left + err_right) - old_err;
        intervals += 1;

        // Push sub-intervals if they still need subdivision
        let local_tol = abs_tol * (ib - ia) / (b - a); // distribute tolerance proportionally
        if err_left > local_tol.max(rel_tol * val_left.abs()) {
            heap.push((err_left, ia, mid, val_left));
        }
        if err_right > local_tol.max(rel_tol * val_right.abs()) {
            heap.push((err_right, mid, ib, val_right));
        }

        // Check global convergence
        if total_error.abs() <= abs_tol.max(rel_tol * total_value.abs()) {
            break;
        }
    }

    GaussKronrodResult {
        value: total_value,
        error: total_error.abs(),
        evaluations: total_evals,
        intervals,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    // ── Simpson's Rule ───────────────────────────────────────────────────

    #[test]
    fn test_simpson_constant() {
        // ∫[0,1] 5 dx = 5
        let result = QuadratureEngine::simpson(&|_x| 5.0, 0.0, 1.0, 10);
        assert!((result.value - 5.0).abs() < TOL);
    }

    #[test]
    fn test_simpson_linear() {
        // ∫[0,1] x dx = 0.5
        let result = QuadratureEngine::simpson(&|x| x, 0.0, 1.0, 10);
        assert!((result.value - 0.5).abs() < TOL);
    }

    #[test]
    fn test_simpson_quadratic() {
        // ∫[0,1] x² dx = 1/3
        let result = QuadratureEngine::simpson(&|x| x * x, 0.0, 1.0, 10);
        assert!((result.value - 1.0 / 3.0).abs() < TOL);
    }

    #[test]
    fn test_simpson_cubic() {
        // ∫[0,1] x³ dx = 1/4
        let result = QuadratureEngine::simpson(&|x| x * x * x, 0.0, 1.0, 100);
        assert!((result.value - 0.25).abs() < TOL);
    }

    #[test]
    fn test_simpson_sin() {
        // ∫[0,π] sin(x) dx = 2
        let result = QuadratureEngine::simpson(&|x| x.sin(), 0.0, std::f64::consts::PI, 100);
        assert!(
            (result.value - 2.0).abs() < 1e-6,
            "Simpson sin: got {}, expected 2.0, err={}",
            result.value,
            (result.value - 2.0).abs()
        );
    }

    // ── Gauss-Legendre ───────────────────────────────────────────────────

    #[test]
    fn test_gauss_polynomial_exact() {
        // Gauss-Legendre with n points is exact for polynomials up to degree 2n-1
        // 5-point should be exact for degree 9

        // ∫[-1,1] x^4 dx = 2/5
        let result = QuadratureEngine::gauss_legendre(&|x| x.powi(4), -1.0, 1.0, 5);
        assert!((result.value - 0.4).abs() < TOL);
    }

    #[test]
    fn test_gauss_exp() {
        // ∫[0,1] e^x dx = e - 1
        let expected = std::f64::consts::E - 1.0;
        let result = QuadratureEngine::gauss_legendre(&|x| x.exp(), 0.0, 1.0, 10);
        assert!(
            (result.value - expected).abs() < TOL,
            "Got {}, expected {}",
            result.value,
            expected
        );
    }

    #[test]
    fn test_gauss_trig() {
        // ∫[0,π/2] cos(x) dx = 1
        let result =
            QuadratureEngine::gauss_legendre(&|x| x.cos(), 0.0, std::f64::consts::FRAC_PI_2, 5);
        assert!((result.value - 1.0).abs() < TOL);
    }

    // ── Adaptive Simpson ─────────────────────────────────────────────────

    #[test]
    fn test_adaptive_gaussian() {
        // ∫[-∞,∞] e^(-x²) dx = √π, but we approximate with [-5,5]
        let result = QuadratureEngine::adaptive_simpson(&|x| (-x * x).exp(), -5.0, 5.0, 1e-10);
        let expected = std::f64::consts::PI.sqrt();
        assert!(
            (result.value - expected).abs() < 1e-8,
            "Got {}, expected {}",
            result.value,
            expected
        );
    }

    #[test]
    fn test_adaptive_oscillatory() {
        // ∫[0,2π] sin(x) dx = 0
        let result = QuadratureEngine::adaptive_simpson(
            &|x| x.sin(),
            0.0,
            2.0 * std::f64::consts::PI,
            1e-10,
        );
        assert!(result.value.abs() < 1e-8);
    }

    #[test]
    fn test_adaptive_sharp_peak() {
        // ∫[0,1] 1/(1+100x²) dx — sharp peak at 0
        let result =
            QuadratureEngine::adaptive_simpson(&|x| 1.0 / (1.0 + 100.0 * x * x), 0.0, 1.0, 1e-8);
        // Expected: (1/10) * arctan(10) ≈ 0.14711
        let expected = (10.0_f64).atan() / 10.0;
        assert!(
            (result.value - expected).abs() < 1e-6,
            "Got {}, expected {}",
            result.value,
            expected
        );
    }

    // ── Fundamental Theorem Verification ─────────────────────────────────

    #[test]
    fn test_fundamental_theorem_polynomial() {
        // f(x) = 3x², F(x) = x³
        let result = QuadratureEngine::verify_fundamental_theorem(
            &|x| 3.0 * x * x,
            &|x| x * x * x,
            0.0,
            2.0,
            1e-10,
        );
        assert!(result.verified);
        assert!((result.numerical - 8.0).abs() < 1e-6);
        assert!((result.symbolic - 8.0).abs() < 1e-15);
        assert!(result.phi > 0.5);
    }

    #[test]
    fn test_fundamental_theorem_trig() {
        // f(x) = cos(x), F(x) = sin(x)
        let result = QuadratureEngine::verify_fundamental_theorem(
            &|x| x.cos(),
            &|x| x.sin(),
            0.0,
            std::f64::consts::PI,
            1e-10,
        );
        assert!(result.verified);
        // sin(π) - sin(0) = 0
        assert!(result.symbolic.abs() < 1e-15);
    }

    #[test]
    fn test_fundamental_theorem_exp() {
        // f(x) = e^x, F(x) = e^x
        let result = QuadratureEngine::verify_fundamental_theorem(
            &|x| x.exp(),
            &|x| x.exp(),
            0.0,
            1.0,
            1e-10,
        );
        assert!(result.verified);
        let expected = std::f64::consts::E - 1.0;
        assert!((result.symbolic - expected).abs() < 1e-15);
    }

    // ── Multi-path ───────────────────────────────────────────────────────

    #[test]
    fn test_multipath_integration() {
        let result = QuadratureEngine::integrate_multipath(&|x| x * x, 0.0, 1.0, 1e-10);
        assert!((result.value - 1.0 / 3.0).abs() < 1e-8);
        // Should have agreement bonus
        assert!(result.phi > 0.5);
    }

    // ── Default integrate ────────────────────────────────────────────────

    #[test]
    fn test_default_integrate() {
        let result = QuadratureEngine::integrate(&|x| x.exp(), 0.0, 1.0);
        let expected = std::f64::consts::E - 1.0;
        assert!((result.value - expected).abs() < 1e-8);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 1c: Free-standing function tests
    // ══════════════════════════════════════════════════════════════════════

    use std::f64::consts::{E, PI};

    // ── Known integrals ───────────────────────────────────────────────────

    #[test]
    fn trapezoid_x_squared() {
        let r = trapezoid(|x| x * x, 0.0, 1.0, 1000);
        assert!((r.value - 1.0 / 3.0).abs() < 1e-6, "got {}", r.value);
        assert_eq!(r.evaluations, 1001);
    }

    #[test]
    fn simpson_free_x_squared_exact() {
        let r = simpson_free(|x| x * x, 0.0, 1.0, 2);
        assert!((r.value - 1.0 / 3.0).abs() < 1e-14, "got {}", r.value);
    }

    #[test]
    fn simpson_free_sin_integral() {
        let r = simpson_free(|x| x.sin(), 0.0, PI, 100);
        assert!((r.value - 2.0).abs() < 1e-7, "got {}", r.value);
    }

    #[test]
    fn trapezoid_exp_integral() {
        let expected = E - 1.0;
        let r = trapezoid(|x| x.exp(), 0.0, 1.0, 10_000);
        assert!((r.value - expected).abs() < 1e-7, "got {}", r.value);
    }

    #[test]
    fn gauss_legendre_basic_exp_integral() {
        let expected = E - 1.0;
        let r = gauss_legendre_basic(|x| x.exp(), 0.0, 1.0, 10);
        assert!((r.value - expected).abs() < 1e-12, "got {}", r.value);
        assert_eq!(r.evaluations, 10);
    }

    // ── Gaussian integral ─────────────────────────────────────────────────

    #[test]
    fn gaussian_integral_test() {
        // ∫₋₃³ e^{-x²} dx ≈ √π × erf(3) ≈ 1.7724...
        let expected = PI.sqrt() * 0.9999779095030014;
        // Adaptive Simpson handles the Gaussian well
        let r = adaptive_simpson_basic(&|x: f64| (-x * x).exp(), -3.0, 3.0, 1e-10, 20);
        assert!((r.value - expected).abs() < 1e-8, "got {}", r.value);
    }

    // ── Polynomial exactness of Gauss-Legendre ────────────────────────────

    #[test]
    fn gauss_legendre_exact_degree_2n_minus_1() {
        // n=3 integrates degree 5 exactly
        let r = gauss_legendre_basic(|x| x.powi(5), -1.0, 1.0, 3);
        assert!(r.value.abs() < 1e-14, "x^5 got {}", r.value);

        let expected = 2.0 / 5.0 - 2.0 / 3.0 + 4.0;
        let r = gauss_legendre_basic(|x| x.powi(4) + x.powi(3) - x * x + 2.0, -1.0, 1.0, 3);
        assert!((r.value - expected).abs() < 1e-13, "poly got {}", r.value);
    }

    #[test]
    fn gauss_legendre_n2_exact_cubic() {
        let r = gauss_legendre_basic(|x| x.powi(3), -1.0, 1.0, 2);
        assert!(r.value.abs() < 1e-14, "x^3 got {}", r.value);

        let r = gauss_legendre_basic(|x| x * x, -1.0, 1.0, 2);
        assert!((r.value - 2.0 / 3.0).abs() < 1e-14, "x^2 got {}", r.value);
    }

    // ── Adaptive Simpson (free-standing) ──────────────────────────────────

    #[test]
    fn adaptive_basic_sin() {
        let r = adaptive_simpson_basic(&|x: f64| x.sin(), 0.0, PI, 1e-12, 20);
        assert!((r.value - 2.0).abs() < 1e-12, "got {}", r.value);
    }

    #[test]
    fn adaptive_tighter_tolerance_more_accurate() {
        let loose = adaptive_simpson_basic(&|x: f64| x.sin(), 0.0, PI, 1e-4, 20);
        let tight = adaptive_simpson_basic(&|x: f64| x.sin(), 0.0, PI, 1e-12, 20);
        let exact = 2.0;
        assert!((tight.value - exact).abs() <= (loose.value - exact).abs() + 1e-15);
        assert!(tight.evaluations >= loose.evaluations);
    }

    // ── Romberg ───────────────────────────────────────────────────────────

    #[test]
    fn romberg_sin_test() {
        let r = romberg(|x| x.sin(), 0.0, PI, 10, 1e-12);
        assert!(
            (r.value - 2.0).abs() < 1e-12,
            "got {} err {}",
            r.value,
            r.error_estimate
        );
    }

    #[test]
    fn romberg_exp_test() {
        let expected = E - 1.0;
        let r = romberg(|x| x.exp(), 0.0, 1.0, 10, 1e-12);
        assert!((r.value - expected).abs() < 1e-12, "got {}", r.value);
    }

    #[test]
    fn romberg_polynomial_test() {
        let r = romberg(|x| 3.0 * x.powi(3) - x + 1.0, 0.0, 2.0, 8, 1e-14);
        assert!((r.value - 12.0).abs() < 1e-12, "got {}", r.value);
    }

    // ── Method comparison ─────────────────────────────────────────────────

    #[test]
    fn higher_order_needs_fewer_points() {
        let exact = 2.0;
        let target_err = 1e-6;

        let t = trapezoid(|x| x.sin(), 0.0, PI, 2000);
        assert!((t.value - exact).abs() < target_err);

        let s = simpson_free(|x| x.sin(), 0.0, PI, 50);
        assert!((s.value - exact).abs() < target_err);

        let g = gauss_legendre_basic(|x| x.sin(), 0.0, PI, 7);
        assert!((g.value - exact).abs() < target_err);

        assert!(g.evaluations < s.evaluations);
        assert!(s.evaluations < t.evaluations);
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn zero_width_interval_all_methods() {
        assert_eq!(trapezoid(|x| x.sin(), 1.0, 1.0, 100).value, 0.0);
        assert_eq!(simpson_free(|x| x.sin(), 1.0, 1.0, 100).value, 0.0);
        assert_eq!(gauss_legendre_basic(|x| x.sin(), 1.0, 1.0, 5).value, 0.0);
        assert_eq!(romberg(|x| x.sin(), 1.0, 1.0, 5, 1e-10).value, 0.0);
    }

    #[test]
    fn constant_function_all_methods() {
        let c = 7.0;
        let (a, b) = (-2.0, 5.0);
        let expected = c * (b - a);

        assert!((trapezoid(|_| c, a, b, 1).value - expected).abs() < 1e-10);
        assert!((simpson_free(|_| c, a, b, 2).value - expected).abs() < 1e-10);
        assert!((gauss_legendre_basic(|_| c, a, b, 2).value - expected).abs() < 1e-10);
        assert!((romberg(|_| c, a, b, 3, 1e-10).value - expected).abs() < 1e-10);
    }

    #[test]
    fn simpson_free_rounds_n_to_even() {
        let r = simpson_free(|x| x * x, 0.0, 1.0, 3);
        assert!((r.value - 1.0 / 3.0).abs() < 1e-14, "got {}", r.value);
        assert_eq!(r.evaluations, 5);
    }

    #[test]
    fn gauss_legendre_weights_sum_to_two() {
        for n in [1, 2, 3, 4, 5, 7, 10] {
            let (_, weights) = gauss_legendre_nodes_weights(n);
            let sum: f64 = weights.iter().sum();
            assert!((sum - 2.0).abs() < 1e-14, "n={n} weight sum={sum}");
        }
    }

    #[test]
    #[should_panic(expected = "not available for n=6")]
    fn gauss_legendre_unsupported_n_panics() {
        gauss_legendre_nodes_weights(6);
    }

    // ── Gauss-Kronrod G7-K15 tests ──────────────────────────────────────

    #[test]
    fn test_gk15_polynomial_exact() {
        // G-K 15-point should exactly integrate polynomials up to degree 29
        // Test with x^4 on [0,1]: exact = 1/5 = 0.2
        let result = gauss_kronrod_adaptive(&|x: f64| x.powi(4), 0.0, 1.0, 1e-14, 1e-14, 0);
        assert!(
            (result.value - 0.2).abs() < 1e-10,
            "G-K 15 should exactly integrate x^4: got {}",
            result.value
        );
    }

    #[test]
    fn test_gk15_exp_integral() {
        // ∫₀¹ exp(x) dx = e - 1 ≈ 1.71828
        let expected = std::f64::consts::E - 1.0;
        let result = gauss_kronrod_adaptive(&|x: f64| x.exp(), 0.0, 1.0, 1e-12, 1e-12, 10);
        assert!(
            (result.value - expected).abs() < 1e-10,
            "∫exp(x) from 0 to 1 = e-1: got {}, error {}",
            result.value,
            result.error
        );
    }

    #[test]
    fn test_gk15_sin_integral() {
        // ∫₀^π sin(x) dx = 2
        let result = gauss_kronrod_adaptive(
            &|x: f64| x.sin(),
            0.0,
            std::f64::consts::PI,
            1e-12,
            1e-12,
            10,
        );
        assert!(
            (result.value - 2.0).abs() < 1e-8,
            "∫sin(x) from 0 to π = 2: got {}",
            result.value
        );
    }

    #[test]
    fn test_gk15_gaussian_integral() {
        // ∫₋₅⁵ exp(-x²) dx ≈ √π ≈ 1.7724538509
        let expected = std::f64::consts::PI.sqrt();
        let result = gauss_kronrod_adaptive(&|x: f64| (-x * x).exp(), -5.0, 5.0, 1e-10, 1e-10, 50);
        assert!(
            (result.value - expected).abs() < 1e-6,
            "∫exp(-x²) ≈ √π: got {}, expected {}",
            result.value,
            expected
        );
    }

    #[test]
    fn test_gk15_oscillatory() {
        // ∫₀^{2π} sin(10x) dx = 0 (by symmetry)
        let result = gauss_kronrod_adaptive(
            &|x: f64| (10.0 * x).sin(),
            0.0,
            2.0 * std::f64::consts::PI,
            1e-10,
            1e-10,
            100,
        );
        assert!(
            result.value.abs() < 1e-6,
            "∫sin(10x) over full period = 0: got {}",
            result.value
        );
    }

    // ── Physics validation tests (known constants) ──────────────────────

    /// Hydrogen atom ground state energy from Bohr model.
    /// E₁ = -m_e * e⁴ / (2ℏ²) = -13.6 eV
    #[test]
    fn test_physics_hydrogen_ground_state() {
        let m_e: f64 = 9.109_383_7015e-31; // kg
        let e_charge: f64 = 1.602_176_634e-19; // C
        let hbar: f64 = 1.054_571_817e-34; // J·s
        let eps0: f64 = 8.854_187_8128e-12; // F/m
        let ev: f64 = 1.602_176_634e-19; // J per eV

        // E₁ = -m_e * e⁴ / (2 * (4πε₀)² * ℏ²)
        let e1_joules = -m_e * e_charge.powi(4)
            / (2.0 * (4.0 * std::f64::consts::PI * eps0).powi(2) * hbar.powi(2));
        let e1_ev = e1_joules / ev;

        assert!(
            (e1_ev - (-13.6)).abs() < 0.1,
            "Hydrogen ground state should be -13.6 eV, got {:.1} eV",
            e1_ev
        );
    }

    /// Bohr radius: a₀ = 4πε₀ℏ² / (m_e * e²) ≈ 0.529 Å
    #[test]
    fn test_physics_bohr_radius() {
        let m_e: f64 = 9.109_383_7015e-31;
        let e_charge: f64 = 1.602_176_634e-19;
        let hbar: f64 = 1.054_571_817e-34;
        let eps0: f64 = 8.854_187_8128e-12;

        let a0 = 4.0 * std::f64::consts::PI * eps0 * hbar.powi(2) / (m_e * e_charge.powi(2));
        let a0_angstrom = a0 * 1e10;

        assert!(
            (a0_angstrom - 0.529).abs() < 0.001,
            "Bohr radius should be 0.529 Å, got {:.4} Å",
            a0_angstrom
        );
    }

    /// Fine structure constant: α = e²/(4πε₀ℏc) ≈ 1/137.036
    #[test]
    fn test_physics_fine_structure_constant() {
        let e_charge: f64 = 1.602_176_634e-19;
        let hbar: f64 = 1.054_571_817e-34;
        let eps0: f64 = 8.854_187_8128e-12;
        let c: f64 = 2.997_924_58e8;

        let alpha = e_charge.powi(2) / (4.0 * std::f64::consts::PI * eps0 * hbar * c);
        let alpha_inv = 1.0 / alpha;

        assert!(
            (alpha_inv - 137.036).abs() < 0.01,
            "1/α should be 137.036, got {:.3}",
            alpha_inv
        );
    }

    /// Wien's displacement law: λ_max * T = b ≈ 2.898 × 10⁻³ m·K
    #[test]
    fn test_physics_wien_displacement() {
        let h: f64 = 6.626_070_15e-34;
        let c: f64 = 2.997_924_58e8;
        let k_b: f64 = 1.380_649e-23;

        // Wien's constant b = hc/(k_B * x) where x ≈ 4.965 is the root of
        // x*e^x/(e^x - 1) = 5 → x ≈ 4.96511
        let x_wien = 4.965_114_231_74;
        let b = h * c / (k_b * x_wien);

        assert!(
            (b * 1e3 - 2.898).abs() < 0.001,
            "Wien constant should be 2.898e-3 m·K, got {:.12}",
            b
        );
    }

    /// Harmonic oscillator: ω = √(k/m), zero-point energy = ℏω/2
    #[test]
    fn test_physics_harmonic_oscillator() {
        let hbar: f64 = 1.054_571_817e-34;
        let k_spring: f64 = 100.0; // N/m
        let mass: f64 = 1.0; // kg

        let omega = (k_spring / mass).sqrt();
        let e_zero = 0.5 * hbar * omega;

        // ω = 10 rad/s, E₀ = 5.27e-34 J
        assert!(
            (omega - 10.0).abs() < 1e-10,
            "ω = √(k/m) = 10, got {}",
            omega
        );
        assert!(
            (e_zero - 5.272_859_085e-34).abs() < 1e-37,
            "Zero-point energy ℏω/2 = 5.27e-34 J, got {:.3e}",
            e_zero
        );
    }

    /// Stefan-Boltzmann: P = σT⁴ where σ = 2π⁵k⁴/(15h³c²)
    #[test]
    fn test_physics_stefan_boltzmann() {
        let h: f64 = 6.626_070_15e-34;
        let c: f64 = 2.997_924_58e8;
        let k_b: f64 = 1.380_649e-23;
        let pi = std::f64::consts::PI;

        let sigma = 2.0 * pi.powi(5) * k_b.powi(4) / (15.0 * h.powi(3) * c.powi(2));

        // σ ≈ 5.670 × 10⁻⁸ W/(m²·K⁴)
        assert!(
            (sigma * 1e8 - 5.670).abs() < 0.01,
            "Stefan-Boltzmann constant should be 5.670e-8, got {:.4e}",
            sigma
        );
    }
}
