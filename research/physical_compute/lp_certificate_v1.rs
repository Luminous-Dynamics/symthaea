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
    let b_inf = norm_inf(&problem.b);
    let a_inf = matrix_inf_norm(problem)?;
    let primal_scale = checked_max3(1.0, b_inf, a_inf * x_inf)?;
    let mut primal_residual_inf = 0.0_f64;
    for (ax_i, b_i) in ax.iter().zip(problem.b.iter()) {
        primal_residual_inf = primal_residual_inf.max((ax_i - b_i).abs());
    }
    let primal_residual_inf = primal_residual_inf / primal_scale;

    let primal_nonnegativity_violation =
        max_negative_part(&certificate.primal) / checked_max2(1.0, x_inf)?;

    let aty_inf = norm_inf(&aty);
    let c_inf = norm_inf(&problem.c);
    let dual_scale = checked_max3(1.0, c_inf, aty_inf)?;
    let dual_nonnegativity_violation = max_negative_part(&slack) / dual_scale;

    let primal_objective = dot(&problem.c, &certificate.primal)?;
    let dual_objective = dot(&problem.b, &certificate.dual)?;
    let objective_scale = checked_max3(
        1.0,
        primal_objective.abs(),
        dual_objective.abs(),
    )?;
    let relative_duality_gap = (primal_objective - dual_objective).abs() / objective_scale;

    let slack_inf = norm_inf(&slack);
    let complementarity_scale = checked_max2(1.0, x_inf * slack_inf)?;
    let mut complementarity_inf = 0.0_f64;
    for (x_i, s_i) in certificate.primal.iter().zip(slack.iter()) {
        complementarity_inf = complementarity_inf.max((x_i * s_i).abs());
    }
    let relative_complementarity = complementarity_inf / complementarity_scale;

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

    // Check each dual-cone component against its own column scale. A global
    // matrix norm can hide a violation in a small column when another column
    // has much larger coefficients.
    let mut dual_cone_violation = 0.0_f64;
    for col in 0..problem.cols {
        let mut column_scale = 0.0_f64;
        for row in 0..problem.rows {
            column_scale += problem.a[row * problem.cols + col].abs();
            if !column_scale.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
        }
        let component_violation = if column_scale > 0.0 {
            (-aty[col]).max(0.0) / column_scale
        } else if aty[col] < 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        dual_cone_violation = dual_cone_violation.max(component_violation);
    }

    let b_transpose_y = dot(&problem.b, &normalized_y)?;
    let b_inf = norm_inf(&problem.b);
    let y_l1 = sum_abs(&normalized_y)?;
    let contradiction_scale = b_inf * y_l1;
    if !contradiction_scale.is_finite() {
        return Err(VerifyError::NonFiniteArithmetic);
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
    let a_inf = matrix_inf_norm(problem)?;
    let b_inf = norm_inf(&problem.b);
    let x_inf = norm_inf(&certificate.feasible_point);
    let base_scale = checked_max3(1.0, b_inf, a_inf * x_inf)?;
    let mut base_residual = 0.0_f64;
    for (ax_i, b_i) in base_ax.iter().zip(problem.b.iter()) {
        base_residual = base_residual.max((ax_i - b_i).abs());
    }
    let base_point_residual_inf = base_residual / base_scale;

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
    let ray_residual_raw = norm_inf(&ray_ad);
    let ray_residual_inf = if a_inf > 0.0 {
        ray_residual_raw / a_inf
    } else if ray_residual_raw == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let ray_nonnegativity_violation = max_negative_part(&normalized_ray);

    // The reported slope uses the normalized ray too: its magnitude depends
    // on arbitrary certificate scaling, but its sign and normalized margin do not.
    let objective_slope = dot(&problem.c, &normalized_ray)?;
    let c_inf = norm_inf(&problem.c);
    let d_l1 = sum_abs(&normalized_ray)?;
    let slope_scale = c_inf * d_l1;
    if !slope_scale.is_finite() {
        return Err(VerifyError::NonFiniteArithmetic);
    }
    let improving_margin = if slope_scale > 0.0 {
        -objective_slope / slope_scale
    } else {
        0.0
    };

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
    if max_negative_part(&certificate.feasible_point) / checked_max2(1.0, x_inf)?
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

fn matrix_inf_norm(problem: &LpProblem) -> Result<f64, VerifyError> {
    let mut maximum = 0.0_f64;
    for row in 0..problem.rows {
        let mut row_sum = 0.0;
        for col in 0..problem.cols {
            row_sum += problem.a[row * problem.cols + col].abs();
            if !row_sum.is_finite() {
                return Err(VerifyError::NonFiniteArithmetic);
            }
        }
        maximum = maximum.max(row_sum);
    }
    Ok(maximum)
}

fn norm_inf(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, value| m.max(value.abs()))
}

fn max_negative_part(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |m, value| m.max((-value).max(0.0)))
}

fn sum_abs(values: &[f64]) -> Result<f64, VerifyError> {
    let mut sum = 0.0;
    for value in values {
        sum += value.abs();
        if !sum.is_finite() {
            return Err(VerifyError::NonFiniteArithmetic);
        }
    }
    Ok(sum)
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn checked_max2(a: f64, b: f64) -> Result<f64, VerifyError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(VerifyError::NonFiniteArithmetic);
    }
    Ok(a.max(b))
}

fn checked_max3(a: f64, b: f64, c: f64) -> Result<f64, VerifyError> {
    checked_max2(checked_max2(a, b)?, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerances() -> Tolerances {
        Tolerances::strict()
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
