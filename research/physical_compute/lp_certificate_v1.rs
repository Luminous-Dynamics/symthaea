//! Independent certificate checks for standard-form linear programs.
//!
//! Problem form:
//!   minimize c^T x
//!   subject to A x = b, x >= 0
//!
//! This module is a small research reference, not a production LP solver.
//! It checks certificates against the original, caller-supplied problem data.
//! It does not trust solver status strings or use hidden evaluator truth.
//!
//! Numerical checks use explicitly supplied tolerances and scaled residuals.
//! A successful floating-point check is an evidence-qualified numerical
//! certificate under those tolerances, not a formal exact-arithmetic proof.

use core::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct LpProblem {
    pub rows: usize,
    pub cols: usize,
    /// Row-major matrix with exactly rows * cols entries.
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    pub primal_feasibility: f64,
    pub dual_feasibility: f64,
    pub relative_gap: f64,
    pub complementarity: f64,
    pub certificate_margin: f64,
}

impl Tolerances {
    pub const fn strict() -> Self {
        Self {
            primal_feasibility: 1.0e-9,
            dual_feasibility: 1.0e-9,
            relative_gap: 1.0e-9,
            complementarity: 1.0e-9,
            certificate_margin: 1.0e-9,
        }
    }

    fn validate(self) -> Result<(), VerifyError> {
        let values = [
            self.primal_feasibility,
            self.dual_feasibility,
            self.relative_gap,
            self.complementarity,
            self.certificate_margin,
        ];
        if values.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0) {
            return Err(VerifyError::InvalidTolerance);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OptimalCertificate {
    pub primal: Vec<f64>,
    /// Equality-constraint dual multipliers. They are unrestricted in sign.
    pub dual: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FarkasCertificate {
    /// For this standard form, A^T y >= 0 and b^T y < 0 certify infeasibility.
    pub y: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UnboundedCertificate {
    /// A feasible base point.
    pub feasible_point: Vec<f64>,
    /// A nonnegative null-space direction with strictly negative objective slope.
    pub improving_ray: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimalMetrics {
    pub primal_objective: f64,
    pub dual_objective: f64,
    pub primal_residual_inf: f64,
    pub primal_nonnegativity_violation: f64,
    pub dual_nonnegativity_violation: f64,
    pub relative_duality_gap: f64,
    pub relative_complementarity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarkasMetrics {
    pub dual_cone_violation: f64,
    pub contradiction_margin: f64,
    /// b^T y after infinity-normalizing the homogeneous Farkas vector.
    pub b_transpose_y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnboundedMetrics {
    pub base_point_residual_inf: f64,
    pub ray_residual_inf: f64,
    pub ray_nonnegativity_violation: f64,
    /// c^T d after infinity-normalizing the homogeneous improving ray.
    pub objective_slope: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerifyError {
    InvalidDimensions,
    NonFiniteInput,
    NonFiniteArithmetic,
    InvalidTolerance,
    PrimalInfeasible,
    PrimalNonnegativeViolation,
    DualInfeasible,
    DualityGapTooLarge,
    ComplementarityTooLarge,
    FarkasConeViolation,
    NoStrictFarkasContradiction,
    RayNotFeasible,
    RayNotNonnegative,
    RayNotImproving,
}

impl LpProblem {
    pub fn validate(&self) -> Result<(), VerifyError> {
        let expected = self
            .rows
            .checked_mul(self.cols)
            .ok_or(VerifyError::InvalidDimensions)?;
        if self.rows == 0
            || self.cols == 0
            || self.a.len() != expected
            || self.b.len() != self.rows
            || self.c.len() != self.cols
        {
            return Err(VerifyError::InvalidDimensions);
        }
        if !all_finite(&self.a) || !all_finite(&self.b) || !all_finite(&self.c) {
            return Err(VerifyError::NonFiniteInput);
        }
        Ok(())
    }
}

pub fn verify_optimal(
    problem: &LpProblem,
    certificate: &OptimalCertificate,
    tolerances: Tolerances,
) -> Result<OptimalMetrics, VerifyError> {
    problem.validate()?;
    tolerances.validate()?;

    if certificate.primal.len() != problem.cols || certificate.dual.len() != problem.rows {
        return Err(VerifyError::InvalidDimensions);
    }
    if !all_finite(&certificate.primal) || !all_finite(&certificate.dual) {
        return Err(VerifyError::NonFiniteInput);
    }

    let ax = mat_vec(problem, &certificate.primal)?;
    let aty = transpose_mat_vec(problem, &certificate.dual)?;
    let mut slack = Vec::with_capacity(problem.cols);
    for (c_i, aty_i) in problem.c.iter().zip(aty.iter()) {
        slack.push(c_i - aty_i);
    }
    if !all_finite(&slack) {
        return Err(VerifyError::NonFiniteArithmetic);
    }

    let x_inf = norm_inf(&certificate.primal);
    // These are dimensionless backward/relative errors. Do not floor their
    // denominators at 1: doing so makes a uniformly tiny but inconsistent LP
    // appear feasible, which is precisely the failure mode this checker must
    // catch after solver scaling is mapped back to the original model.
    // Use componentwise backward errors: a large row/column must not hide
    // a violated small row/column in a badly scaled original model.
    let mut primal_residual_inf = 0.0_f64;
    for row in 0..problem.rows {
        let mut row_scale = problem.b[row].abs();
        for col in 0..problem.cols {
            let product = problem.a[row * problem.cols + col] * certificate.primal[col];
            if !product.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
            row_scale = checked_sum(row_scale, product.abs())?;
        }
        let residual = (ax[row] - problem.b[row]).abs();
        primal_residual_inf =
            primal_residual_inf.max(normalized_ratio(residual, row_scale));
    }

    let primal_nonnegativity_violation =
        normalized_ratio(max_negative_part(&certificate.primal), x_inf);

    let mut dual_nonnegativity_violation = 0.0_f64;
    for col in 0..problem.cols {
        let mut column_scale = problem.c[col].abs();
        for row in 0..problem.rows {
            let product = problem.a[row * problem.cols + col] * certificate.dual[row];
            if !product.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
            column_scale = checked_sum(column_scale, product.abs())?;
        }
        let violation = (-slack[col]).max(0.0);
        dual_nonnegativity_violation =
            dual_nonnegativity_violation.max(normalized_ratio(violation, column_scale));
    }

    let primal_objective = dot(&problem.c, &certificate.primal)?;
    let dual_objective = dot(&problem.b, &certificate.dual)?;
    let objective_scale = checked_sum(primal_objective.abs(), dual_objective.abs())?;
    let relative_duality_gap = normalized_ratio(
        (primal_objective - dual_objective).abs(),
        objective_scale,
    );

    let slack_inf = norm_inf(&slack);
    let mut relative_complementarity = 0.0_f64;
    if x_inf > 0.0 && slack_inf > 0.0 {
        for (x_i, s_i) in certificate.primal.iter().zip(slack.iter()) {
            let normalized_product = (x_i / x_inf) * (s_i / slack_inf);
            relative_complementarity =
                relative_complementarity.max(normalized_product.abs());
        }
    }

    if ![
        primal_residual_inf,
        primal_nonnegativity_violation,
        dual_nonnegativity_violation,
        relative_duality_gap,
        relative_complementarity,
        primal_objective,
        dual_objective,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return Err(VerifyError::NonFiniteArithmetic);
    }

    if primal_residual_inf > tolerances.primal_feasibility {
        return Err(VerifyError::PrimalInfeasible);
    }
    if primal_nonnegativity_violation > tolerances.primal_feasibility {
        return Err(VerifyError::PrimalNonnegativeViolation);
    }
    if dual_nonnegativity_violation > tolerances.dual_feasibility {
        return Err(VerifyError::DualInfeasible);
    }
    if relative_duality_gap > tolerances.relative_gap {
        return Err(VerifyError::DualityGapTooLarge);
    }
    if relative_complementarity > tolerances.complementarity {
        return Err(VerifyError::ComplementarityTooLarge);
    }

    Ok(OptimalMetrics {
        primal_objective,
        dual_objective,
        primal_residual_inf,
        primal_nonnegativity_violation,
        dual_nonnegativity_violation,
        relative_duality_gap,
        relative_complementarity,
    })
}

pub fn verify_infeasible(
    problem: &LpProblem,
    certificate: &FarkasCertificate,
    tolerances: Tolerances,
) -> Result<FarkasMetrics, VerifyError> {
    problem.validate()?;
    tolerances.validate()?;

    if certificate.y.len() != problem.rows {
        return Err(VerifyError::InvalidDimensions);
    }
    if !all_finite(&certificate.y) {
        return Err(VerifyError::NonFiniteInput);
    }

    // Farkas multipliers are homogeneous: y and alpha*y (alpha > 0)
    // represent the same certificate. Normalize before applying tolerances so
    // a harmless change of certificate scale cannot turn PASS into REJECT.
    let y_inf = norm_inf(&certificate.y);
    if y_inf == 0.0 {
        return Err(VerifyError::NoStrictFarkasContradiction);
    }
    let normalized_y: Vec<f64> = certificate.y.iter().map(|value| value / y_inf).collect();
    let aty = transpose_mat_vec(problem, &normalized_y)?;

    // Check each dual-cone component against its actual absolute
    // contributions. A small multiplier must not hide a violated cone
    // component behind the unweighted norm of its matrix column.
    let mut dual_cone_violation = 0.0_f64;
    for col in 0..problem.cols {
        let mut column_scale = 0.0_f64;
        for row in 0..problem.rows {
            let product = problem.a[row * problem.cols + col] * normalized_y[row];
            if !product.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
            column_scale = checked_sum(column_scale, product.abs())?;
        }
        dual_cone_violation = dual_cone_violation.max(normalized_ratio(
            (-aty[col]).max(0.0),
            column_scale,
        ));
    }

    let b_transpose_y = dot(&problem.b, &normalized_y)?;
    let mut contradiction_scale = 0.0_f64;
    for row in 0..problem.rows {
        let product = problem.b[row] * normalized_y[row];
        if !product.is_finite() {
            return Err(VerifyError::NonFiniteArithmetic);
        }
        contradiction_scale = checked_sum(contradiction_scale, product.abs())?;
    }
    let contradiction_margin = if contradiction_scale > 0.0 {
        -b_transpose_y / contradiction_scale
    } else {
        0.0
    };

    if !dual_cone_violation.is_finite()
        || !contradiction_margin.is_finite()
        || !b_transpose_y.is_finite()
    {
        return Err(VerifyError::NonFiniteArithmetic);
    }
    if dual_cone_violation > tolerances.dual_feasibility {
        return Err(VerifyError::FarkasConeViolation);
    }
    if contradiction_margin < tolerances.certificate_margin {
        return Err(VerifyError::NoStrictFarkasContradiction);
    }

    Ok(FarkasMetrics {
        dual_cone_violation,
        contradiction_margin,
        b_transpose_y,
    })
}

pub fn verify_unbounded(
    problem: &LpProblem,
    certificate: &UnboundedCertificate,
    tolerances: Tolerances,
) -> Result<UnboundedMetrics, VerifyError> {
    problem.validate()?;
    tolerances.validate()?;

    if certificate.feasible_point.len() != problem.cols
        || certificate.improving_ray.len() != problem.cols
    {
        return Err(VerifyError::InvalidDimensions);
    }
    if !all_finite(&certificate.feasible_point) || !all_finite(&certificate.improving_ray) {
        return Err(VerifyError::NonFiniteInput);
    }

    let base_ax = mat_vec(problem, &certificate.feasible_point)?;
    let x_inf = norm_inf(&certificate.feasible_point);
    let mut base_point_residual_inf = 0.0_f64;
    for row in 0..problem.rows {
        let mut row_scale = problem.b[row].abs();
        for col in 0..problem.cols {
            let product =
                problem.a[row * problem.cols + col] * certificate.feasible_point[col];
            if !product.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
            row_scale = checked_sum(row_scale, product.abs())?;
        }
        let residual = (base_ax[row] - problem.b[row]).abs();
        base_point_residual_inf =
            base_point_residual_inf.max(normalized_ratio(residual, row_scale));
    }

    // An improving ray is homogeneous under positive scaling. Normalize it
    // before checking A*d = 0, d >= 0 and c^T*d < 0; otherwise a tiny copy of
    // a non-ray can appear feasible merely because the absolute scale is tiny.
    let d_inf = norm_inf(&certificate.improving_ray);
    if d_inf == 0.0 {
        return Err(VerifyError::RayNotImproving);
    }
    let normalized_ray: Vec<f64> = certificate
        .improving_ray
        .iter()
        .map(|value| value / d_inf)
        .collect();
    let ray_ad = mat_vec(problem, &normalized_ray)?;
    let mut ray_residual_inf = 0.0_f64;
    for row in 0..problem.rows {
        let mut row_scale = 0.0_f64;
        for col in 0..problem.cols {
            let product = problem.a[row * problem.cols + col] * normalized_ray[col];
            if !product.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
            row_scale = checked_sum(row_scale, product.abs())?;
        }
        ray_residual_inf =
            ray_residual_inf.max(normalized_ratio(ray_ad[row].abs(), row_scale));
    }
    let ray_nonnegativity_violation = max_negative_part(&normalized_ray);

    // Normalize objective descent by the absolute contributions to c^T d.
    let objective_slope = dot(&problem.c, &normalized_ray)?;
    let mut slope_scale = 0.0_f64;
    for col in 0..problem.cols {
        let product = problem.c[col] * normalized_ray[col];
        if !product.is_finite() {
            return Err(VerifyError::NonFiniteArithmetic);
        }
        slope_scale = checked_sum(slope_scale, product.abs())?;
    }
    let improving_margin = normalized_ratio(-objective_slope, slope_scale);

    if ![
        base_point_residual_inf,
        ray_residual_inf,
        ray_nonnegativity_violation,
        objective_slope,
        improving_margin,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return Err(VerifyError::NonFiniteArithmetic);
    }
    if base_point_residual_inf > tolerances.primal_feasibility {
        return Err(VerifyError::RayNotFeasible);
    }
    if normalized_ratio(max_negative_part(&certificate.feasible_point), x_inf)
        > tolerances.primal_feasibility
    {
        return Err(VerifyError::RayNotFeasible);
    }
    if ray_residual_inf > tolerances.primal_feasibility {
        return Err(VerifyError::RayNotFeasible);
    }
    if ray_nonnegativity_violation > tolerances.primal_feasibility {
        return Err(VerifyError::RayNotNonnegative);
    }
    if improving_margin < tolerances.certificate_margin {
        return Err(VerifyError::RayNotImproving);
    }

    Ok(UnboundedMetrics {
        base_point_residual_inf,
        ray_residual_inf,
        ray_nonnegativity_violation,
        objective_slope,
    })
}

fn mat_vec(problem: &LpProblem, x: &[f64]) -> Result<Vec<f64>, VerifyError> {
    let mut out = Vec::with_capacity(problem.rows);
    for row in 0..problem.rows {
        let start = row * problem.cols;
        let end = start + problem.cols;
        out.push(dot(&problem.a[start..end], x)?);
    }
    Ok(out)
}

fn transpose_mat_vec(problem: &LpProblem, y: &[f64]) -> Result<Vec<f64>, VerifyError> {
    let mut out = Vec::with_capacity(problem.cols);
    for col in 0..problem.cols {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for row in 0..problem.rows {
            let product = problem.a[row * problem.cols + col] * y[row];
            if !product.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
            let adjusted = product - correction;
            let next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        if !sum.is_finite() {
            return Err(VerifyError::NonFiniteArithmetic);
        }
        out.push(sum);
    }
    Ok(out)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, VerifyError> {
    if left.len() != right.len() {
        return Err(VerifyError::InvalidDimensions);
    }
    let mut sum = 0.0;
    let mut correction = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        let product = a * b;
        if !product.is_finite() {
            return Err(VerifyError::NonFiniteArithmetic);
        }
        let adjusted = product - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    if !sum.is_finite() {
        return Err(VerifyError::NonFiniteArithmetic);
    }
    Ok(sum)
}

fn norm_inf(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, value| m.max(value.abs()))
}

fn max_negative_part(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |m, value| m.max((-value).max(0.0)))
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn checked_sum(a: f64, b: f64) -> Result<f64, VerifyError> {
    let value = a + b;
    if !value.is_finite() {
        return Err(VerifyError::NonFiniteArithmetic);
    }
    Ok(value)
}

/// Relative error with explicit zero-scale semantics. A nonzero residual
/// against a zero reference scale is a rejection, not an absolute-tolerance pass.
fn normalized_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        let ratio = numerator / denominator;
        if ratio.is_finite() { ratio } else { f64::MAX }
    } else if numerator == 0.0 {
        0.0
    } else {
        f64::MAX
    }
}


// --- Exact certificate checks for binary64 input values --------------------
//
// Every finite f64 is an exact dyadic rational. A product of two f64 values
// therefore has an integer significand of at most 106 bits and a power-of-two
// exponent in [-2148, 1942]. The accumulator below sums those products exactly
// using separate positive/negative fixed-width unsigned superaccumulators.
// 68 limbs cover the complete binary64 product exponent span plus the maximum
// possible usize-sized term count on 64-bit targets.
//
// These routines certify algebraic signs/equalities of the exact binary64
// inputs. They do not certify decimal values that were rounded before input,
// nor do they turn an approximate solver iterate into an exact certificate.

const EXACT_ACCUMULATOR_LIMBS: usize = 68;
const EXACT_PRODUCT_EXPONENT_OFFSET: i32 = 2148;

fn add_word(
    words: &mut [u64; EXACT_ACCUMULATOR_LIMBS],
    mut index: usize,
    mut value: u64,
) -> Result<(), VerifyError> {
    while value != 0 {
        if index >= EXACT_ACCUMULATOR_LIMBS {
            return Err(VerifyError::NonFiniteArithmetic);
        }
        let sum = words[index] as u128 + value as u128;
        words[index] = sum as u64;
        value = (sum >> 64) as u64;
        index += 1;
    }
    Ok(())
}

fn add_shifted_product(
    words: &mut [u64; EXACT_ACCUMULATOR_LIMBS],
    product: u128,
    shift: usize,
) -> Result<(), VerifyError> {
    let word_index = shift / 64;
    let bit_shift = shift % 64;
    let low = product as u64;
    let high = (product >> 64) as u64;

    if bit_shift == 0 {
        add_word(words, word_index, low)?;
        add_word(words, word_index + 1, high)?;
    } else {
        let low_part = low << bit_shift;
        let middle_part =
            (low >> (64 - bit_shift)) | (high << bit_shift);
        let high_part = high >> (64 - bit_shift);
        add_word(words, word_index, low_part)?;
        add_word(words, word_index + 1, middle_part)?;
        add_word(words, word_index + 2, high_part)?;
    }
    Ok(())
}

fn exact_sum_products(terms: &[(f64, f64, bool)]) -> Result<Ordering, VerifyError> {
    let mut positive = [0_u64; EXACT_ACCUMULATOR_LIMBS];
    let mut negative = [0_u64; EXACT_ACCUMULATOR_LIMBS];

    for &(left, right, subtract) in terms {
        if !left.is_finite() || !right.is_finite() {
            return Err(VerifyError::NonFiniteInput);
        }

        let left_bits = left.to_bits();
        let right_bits = right.to_bits();
        let left_exp = ((left_bits >> 52) & 0x7ff) as i32;
        let right_exp = ((right_bits >> 52) & 0x7ff) as i32;
        let fraction_mask = (1_u64 << 52) - 1;
        let left_fraction = left_bits & fraction_mask;
        let right_fraction = right_bits & fraction_mask;

        let (left_mantissa, left_exponent) = if left_exp == 0 {
            (left_fraction, -1074)
        } else {
            ((1_u64 << 52) | left_fraction, left_exp - 1023 - 52)
        };
        let (right_mantissa, right_exponent) = if right_exp == 0 {
            (right_fraction, -1074)
        } else {
            ((1_u64 << 52) | right_fraction, right_exp - 1023 - 52)
        };

        if left_mantissa == 0 || right_mantissa == 0 {
            continue;
        }

        let product = (left_mantissa as u128) * (right_mantissa as u128);
        let product_exponent = left_exponent + right_exponent;
        let shift = product_exponent + EXACT_PRODUCT_EXPONENT_OFFSET;
        if shift < 0 {
            return Err(VerifyError::NonFiniteArithmetic);
        }

        let is_negative =
            (left_bits >> 63 != 0) ^ (right_bits >> 63 != 0) ^ subtract;
        if is_negative {
            add_shifted_product(&mut negative, product, shift as usize)?;
        } else {
            add_shifted_product(&mut positive, product, shift as usize)?;
        }
    }

    Ok(positive.iter().rev().cmp(negative.iter().rev()))
}

fn exact_dot_sign(left: &[f64], right: &[f64]) -> Result<Ordering, VerifyError> {
    if left.len() != right.len() {
        return Err(VerifyError::InvalidDimensions);
    }
    let terms: Vec<(f64, f64, bool)> = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(a, b)| (a, b, false))
        .collect();
    exact_sum_products(&terms)
}

/// Exact sign of constant - dot(left, right).
fn exact_affine_sign(
    constant: f64,
    left: &[f64],
    right: &[f64],
) -> Result<Ordering, VerifyError> {
    if left.len() != right.len() {
        return Err(VerifyError::InvalidDimensions);
    }
    let mut terms = Vec::with_capacity(left.len() + 1);
    terms.push((constant, 1.0, false));
    terms.extend(
        left.iter()
            .copied()
            .zip(right.iter().copied())
            .map(|(a, b)| (a, b, true)),
    );
    exact_sum_products(&terms)
}

fn exact_difference_of_dots_sign(
    left_a: &[f64],
    left_b: &[f64],
    right_a: &[f64],
    right_b: &[f64],
) -> Result<Ordering, VerifyError> {
    if left_a.len() != left_b.len() || right_a.len() != right_b.len() {
        return Err(VerifyError::InvalidDimensions);
    }
    let mut terms = Vec::with_capacity(left_a.len() + right_a.len());
    terms.extend(
        left_a
            .iter()
            .copied()
            .zip(left_b.iter().copied())
            .map(|(a, b)| (a, b, false)),
    );
    terms.extend(
        right_a
            .iter()
            .copied()
            .zip(right_b.iter().copied())
            .map(|(a, b)| (a, b, true)),
    );
    exact_sum_products(&terms)
}

/// Exact optimality check over the finite binary64 values in the original model.
///
/// This requires exact primal equalities, nonnegative primal values, exact dual
/// inequalities and exact primal/dual objective equality. Approximate solver
/// outputs should use verify_optimal() instead and retain its tolerance label.
pub fn verify_optimal_exact(
    problem: &LpProblem,
    certificate: &OptimalCertificate,
) -> Result<(), VerifyError> {
    problem.validate()?;
    if certificate.primal.len() != problem.cols || certificate.dual.len() != problem.rows {
        return Err(VerifyError::InvalidDimensions);
    }
    if !all_finite(&certificate.primal) || !all_finite(&certificate.dual) {
        return Err(VerifyError::NonFiniteInput);
    }
    if certificate.primal.iter().any(|value| *value < 0.0) {
        return Err(VerifyError::PrimalNonnegativeViolation);
    }

    for row in 0..problem.rows {
        let start = row * problem.cols;
        let end = start + problem.cols;
        if exact_affine_sign(
            problem.b[row],
            &problem.a[start..end],
            &certificate.primal,
        )? != Ordering::Equal
        {
            return Err(VerifyError::PrimalInfeasible);
        }
    }

    for col in 0..problem.cols {
        let column: Vec<f64> = (0..problem.rows)
            .map(|row| problem.a[row * problem.cols + col])
            .collect();
        if exact_affine_sign(problem.c[col], &column, &certificate.dual)?
            == Ordering::Less
        {
            return Err(VerifyError::DualInfeasible);
        }
    }

    if exact_difference_of_dots_sign(
        &problem.c,
        &certificate.primal,
        &problem.b,
        &certificate.dual,
    )? != Ordering::Equal
    {
        return Err(VerifyError::DualityGapTooLarge);
    }
    Ok(())
}

/// Exact Farkas check over finite binary64 input values. Unlike the
/// tolerance-based verifier, this requires A^T y >= 0 and b^T y < 0 exactly.
pub fn verify_infeasible_exact(
    problem: &LpProblem,
    certificate: &FarkasCertificate,
) -> Result<(), VerifyError> {
    problem.validate()?;
    if certificate.y.len() != problem.rows {
        return Err(VerifyError::InvalidDimensions);
    }
    if !all_finite(&certificate.y) {
        return Err(VerifyError::NonFiniteInput);
    }

    for col in 0..problem.cols {
        let column: Vec<f64> = (0..problem.rows)
            .map(|row| problem.a[row * problem.cols + col])
            .collect();
        if exact_dot_sign(&column, &certificate.y)? == Ordering::Less {
            return Err(VerifyError::FarkasConeViolation);
        }
    }
    if exact_dot_sign(&problem.b, &certificate.y)? != Ordering::Less {
        return Err(VerifyError::NoStrictFarkasContradiction);
    }
    Ok(())
}

/// Exact unboundedness check over finite binary64 input values.
pub fn verify_unbounded_exact(
    problem: &LpProblem,
    certificate: &UnboundedCertificate,
) -> Result<(), VerifyError> {
    problem.validate()?;
    if certificate.feasible_point.len() != problem.cols
        || certificate.improving_ray.len() != problem.cols
    {
        return Err(VerifyError::InvalidDimensions);
    }
    if !all_finite(&certificate.feasible_point) || !all_finite(&certificate.improving_ray) {
        return Err(VerifyError::NonFiniteInput);
    }
    if certificate.feasible_point.iter().any(|value| *value < 0.0) {
        return Err(VerifyError::RayNotFeasible);
    }
    if certificate.improving_ray.iter().any(|value| *value < 0.0) {
        return Err(VerifyError::RayNotNonnegative);
    }

    for row in 0..problem.rows {
        let start = row * problem.cols;
        let end = start + problem.cols;
        let matrix_row = &problem.a[start..end];
        if exact_affine_sign(
            problem.b[row],
            matrix_row,
            &certificate.feasible_point,
        )? != Ordering::Equal
        {
            return Err(VerifyError::RayNotFeasible);
        }
        if exact_dot_sign(matrix_row, &certificate.improving_ray)? != Ordering::Equal {
            return Err(VerifyError::RayNotFeasible);
        }
    }

    if exact_dot_sign(&problem.c, &certificate.improving_ray)? != Ordering::Less {
        return Err(VerifyError::RayNotImproving);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerances() -> Tolerances {
        Tolerances::strict()
    }


    #[test]
    fn exact_optimality_accepts_binary64_exact_fixture() {
        let problem = LpProblem {
            rows: 1,
            cols: 3,
            a: vec![1.0, 1.0, 1.0],
            b: vec![1.0],
            c: vec![-3.0, -2.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0, 0.0, 0.0],
            dual: vec![-3.0],
        };
        assert_eq!(verify_optimal_exact(&problem, &certificate), Ok(()));
    }

    #[test]
    fn exact_optimality_rejects_tolerance_only_primal_point() {
        let problem = LpProblem {
            rows: 1,
            cols: 2,
            a: vec![1.0, 1.0],
            b: vec![1.0],
            c: vec![-1.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![0.5, 0.5000000000001],
            dual: vec![-1.0],
        };
        assert!(verify_optimal(&problem, &certificate, tolerances()).is_ok());
        assert_eq!(
            verify_optimal_exact(&problem, &certificate),
            Err(VerifyError::PrimalInfeasible)
        );
    }

    #[test]
    fn exact_farkas_rejects_tiny_negative_cone_component() {
        let problem = LpProblem {
            rows: 2,
            cols: 2,
            a: vec![1.0e12, 0.0, 0.0, -1.0],
            b: vec![-1.0, 0.0],
            c: vec![0.0, 0.0],
        };
        let certificate = FarkasCertificate { y: vec![1.0, 1.0e-12] };
        assert_eq!(
            verify_infeasible(&problem, &certificate, tolerances()),
            Ok(FarkasMetrics {
                dual_cone_violation: 1.0e-12,
                contradiction_margin: 1.0,
                b_transpose_y: -1.0,
            })
        );
        assert_eq!(
            verify_infeasible_exact(&problem, &certificate),
            Err(VerifyError::FarkasConeViolation)
        );
    }

    #[test]
    fn exact_farkas_accepts_valid_certificate() {
        let problem = LpProblem {
            rows: 1,
            cols: 2,
            a: vec![1.0, 1.0],
            b: vec![-1.0],
            c: vec![0.0, 0.0],
        };
        let certificate = FarkasCertificate { y: vec![1.0] };
        assert_eq!(verify_infeasible_exact(&problem, &certificate), Ok(()));
    }

    #[test]
    fn exact_unboundedness_accepts_valid_ray() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![0.0],
            b: vec![0.0],
            c: vec![-1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![0.0],
            improving_ray: vec![1.0],
        };
        assert_eq!(verify_unbounded_exact(&problem, &certificate), Ok(()));
    }

    #[test]
    fn exact_unboundedness_rejects_tiny_nonzero_nullspace_residual() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![1.0],
            b: vec![0.0],
            c: vec![-1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![0.0],
            improving_ray: vec![1.0e-200],
        };
        assert_eq!(
            verify_unbounded_exact(&problem, &certificate),
            Err(VerifyError::RayNotFeasible)
        );
    }

    #[test]
    fn exact_dot_sign_survives_cancellation_and_subnormal_term() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            exact_dot_sign(
                &[1.0e308, 1.0e308, 1.0],
                &[1.0e-308, -1.0e-308, -tiny],
            ),
            Ok(Ordering::Less)
        );
    }

    #[test]
    fn exact_dot_sign_handles_large_exponent_separation() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            exact_dot_sign(&[1.0, tiny], &[1.0, 1.0]),
            Ok(Ordering::Greater)
        );
    }
    #[test]
    fn accepts_primal_dual_optimality_certificate() {
        // min -3x0 -2x1 subject to x0 + x1 + x2 = 1, x >= 0.
        // x=(1,0,0), y=-3, slack=(0,1,3).
        let problem = LpProblem {
            rows: 1,
            cols: 3,
            a: vec![1.0, 1.0, 1.0],
            b: vec![1.0],
            c: vec![-3.0, -2.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0, 0.0, 0.0],
            dual: vec![-3.0],
        };
        let metrics = verify_optimal(&problem, &certificate, tolerances()).unwrap();
        assert!((metrics.primal_objective + 3.0).abs() < 1.0e-12);
        assert!((metrics.dual_objective + 3.0).abs() < 1.0e-12);
        assert!(metrics.relative_duality_gap < 1.0e-12);
    }

    #[test]
    fn rejects_primal_infeasible_candidate() {
        let problem = LpProblem {
            rows: 1,
            cols: 3,
            a: vec![1.0, 1.0, 1.0],
            b: vec![1.0],
            c: vec![-3.0, -2.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![0.0, 0.0, 0.0],
            dual: vec![-3.0],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::PrimalInfeasible)
        );
    }

    #[test]
    fn rejects_dual_infeasible_candidate() {
        let problem = LpProblem {
            rows: 1,
            cols: 3,
            a: vec![1.0, 1.0, 1.0],
            b: vec![1.0],
            c: vec![-3.0, -2.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0, 0.0, 0.0],
            dual: vec![-2.0],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::DualInfeasible)
        );
    }

    #[test]
    fn rejects_large_duality_gap() {
        let problem = LpProblem {
            rows: 1,
            cols: 3,
            a: vec![1.0, 1.0, 1.0],
            b: vec![1.0],
            c: vec![-3.0, -2.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![0.0, 1.0, 0.0],
            dual: vec![-3.0],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::DualityGapTooLarge)
        );
    }

    #[test]
    fn accepts_farkas_infeasibility_certificate() {
        // x0 + x1 = -1, x >= 0 is infeasible.
        // y=1 gives A^T y=(1,1)>=0 and b^T y=-1<0.
        let problem = LpProblem {
            rows: 1,
            cols: 2,
            a: vec![1.0, 1.0],
            b: vec![-1.0],
            c: vec![0.0, 0.0],
        };
        let certificate = FarkasCertificate { y: vec![1.0] };
        assert!(verify_infeasible(&problem, &certificate, tolerances()).is_ok());
    }

    #[test]
    fn rejects_invalid_farkas_sign() {
        let problem = LpProblem {
            rows: 1,
            cols: 2,
            a: vec![1.0, 1.0],
            b: vec![-1.0],
            c: vec![0.0, 0.0],
        };
        let certificate = FarkasCertificate { y: vec![-1.0] };
        assert_eq!(
            verify_infeasible(&problem, &certificate, tolerances()),
            Err(VerifyError::FarkasConeViolation)
        );
    }

    #[test]
    fn accepts_unbounded_improving_ray() {
        // min -x subject to 0*x=0, x>=0 is unbounded below.
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![0.0],
            b: vec![0.0],
            c: vec![-1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![0.0],
            improving_ray: vec![1.0],
        };
        assert!(verify_unbounded(&problem, &certificate, tolerances()).is_ok());
    }

    #[test]
    fn rejects_non_improving_ray() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![0.0],
            b: vec![0.0],
            c: vec![1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![0.0],
            improving_ray: vec![1.0],
        };
        assert_eq!(
            verify_unbounded(&problem, &certificate, tolerances()),
            Err(VerifyError::RayNotImproving)
        );
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![1.0],
            b: vec![1.0],
            c: vec![-1.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0, 2.0],
            dual: vec![-1.0],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::InvalidDimensions)
        );
    }

    #[test]
    fn rejects_non_finite_problem_input() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![f64::NAN],
            b: vec![1.0],
            c: vec![-1.0],
        };
        let certificate = FarkasCertificate { y: vec![1.0] };
        assert_eq!(
            verify_infeasible(&problem, &certificate, tolerances()),
            Err(VerifyError::NonFiniteInput)
        );
    }

    #[test]
    fn rejects_tolerances_above_one() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![1.0],
            b: vec![1.0],
            c: vec![-1.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0],
            dual: vec![-1.0],
        };
        let invalid = Tolerances {
            relative_gap: 1.01,
            ..tolerances()
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, invalid),
            Err(VerifyError::InvalidTolerance)
        );
    }


    #[test]
    fn rejects_infeasible_small_row_hidden_by_large_row() {
        let problem = LpProblem {
            rows: 2,
            cols: 2,
            a: vec![1.0e12, 0.0, 0.0, 0.0],
            b: vec![1.0e12, 1.0],
            c: vec![0.0, 0.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0, 0.0],
            dual: vec![0.0, 0.0],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::PrimalInfeasible)
        );
    }

    #[test]
    fn rejects_dual_violation_in_small_column() {
        let problem = LpProblem {
            rows: 1,
            cols: 2,
            a: vec![1.0e12, 1.0e-12],
            b: vec![1.0e12],
            c: vec![1.0e12, -1.0e-12],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0, 0.0],
            dual: vec![0.0],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::DualInfeasible)
        );
    }

    #[test]
    fn rejects_farkas_cone_violation_from_small_multiplier_component() {
        let problem = LpProblem {
            rows: 2,
            cols: 2,
            a: vec![1.0e12, 0.0, 0.0, -1.0],
            b: vec![-1.0, 0.0],
            c: vec![0.0, 0.0],
        };
        let certificate = FarkasCertificate { y: vec![1.0, 1.0e-12] };
        assert_eq!(
            verify_infeasible(&problem, &certificate, tolerances()),
            Err(VerifyError::FarkasConeViolation)
        );
    }

    #[test]
    fn rejects_unbounded_ray_with_small_row_nullspace_violation() {
        let problem = LpProblem {
            rows: 2,
            cols: 2,
            a: vec![1.0e12, 0.0, 0.0, 1.0],
            b: vec![0.0, 0.0],
            c: vec![0.0, -1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![0.0, 0.0],
            improving_ray: vec![1.0e-12, 1.0],
        };
        assert_eq!(
            verify_unbounded(&problem, &certificate, tolerances()),
            Err(VerifyError::RayNotFeasible)
        );
    }

    #[test]
    fn rejects_unbounded_base_point_with_small_row_residual() {
        let problem = LpProblem {
            rows: 2,
            cols: 2,
            a: vec![1.0e12, 0.0, 0.0, 0.0],
            b: vec![1.0e12, 1.0],
            c: vec![0.0, -1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![1.0, 0.0],
            improving_ray: vec![0.0, 1.0],
        };
        assert_eq!(
            verify_unbounded(&problem, &certificate, tolerances()),
            Err(VerifyError::RayNotFeasible)
        );
    }
    #[test]
    fn rejects_infeasible_certificate_after_uniform_model_scaling() {
        // Scaling the whole LP by 1e-12 must not make x=0 satisfy Ax=b.
        // The previous max(1, scale) denominator accepted this false certificate.
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![1.0e-12],
            b: vec![1.0e-12],
            c: vec![-1.0e-12],
        };
        let certificate = OptimalCertificate {
            primal: vec![0.0],
            dual: vec![-1.0e-12],
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, tolerances()),
            Err(VerifyError::PrimalInfeasible)
        );
    }

    #[test]
    fn farkas_certificate_is_invariant_to_positive_scaling() {
        let problem = LpProblem {
            rows: 1,
            cols: 2,
            a: vec![1.0, 1.0],
            b: vec![-1.0],
            c: vec![0.0, 0.0],
        };
        for scale in [1.0e-200, 1.0, 1.0e200] {
            let certificate = FarkasCertificate { y: vec![scale] };
            assert!(
                verify_infeasible(&problem, &certificate, tolerances()).is_ok(),
                "valid Farkas certificate rejected at scale {scale:e}"
            );
        }
    }

    #[test]
    fn unbounded_ray_certificate_is_invariant_to_positive_scaling() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![0.0],
            b: vec![0.0],
            c: vec![-1.0],
        };
        for scale in [1.0e-200, 1.0, 1.0e200] {
            let certificate = UnboundedCertificate {
                feasible_point: vec![0.0],
                improving_ray: vec![scale],
            };
            assert!(
                verify_unbounded(&problem, &certificate, tolerances()).is_ok(),
                "valid improving ray rejected at scale {scale:e}"
            );
        }
    }

    #[test]
    fn rejects_tiny_scaled_non_nullspace_ray() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![1.0],
            b: vec![0.0],
            c: vec![-1.0],
        };
        let certificate = UnboundedCertificate {
            feasible_point: vec![0.0],
            improving_ray: vec![1.0e-200],
        };
        assert_eq!(
            verify_unbounded(&problem, &certificate, tolerances()),
            Err(VerifyError::RayNotFeasible)
        );
    }

    #[test]
    fn rejects_invalid_tolerances() {
        let problem = LpProblem {
            rows: 1,
            cols: 1,
            a: vec![1.0],
            b: vec![1.0],
            c: vec![-1.0],
        };
        let certificate = OptimalCertificate {
            primal: vec![1.0],
            dual: vec![-1.0],
        };
        let invalid = Tolerances {
            primal_feasibility: 0.0,
            ..tolerances()
        };
        assert_eq!(
            verify_optimal(&problem, &certificate, invalid),
            Err(VerifyError::InvalidTolerance)
        );
    }
}
