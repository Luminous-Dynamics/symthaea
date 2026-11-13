//! Robust bounded inverse-control proposals for calibrated vocal controls.
//!
//! The solver consumes an empirical interval-valued response matrix and
//! no-op-calibrated metric acceptance bands. It computes a regularized,
//! box-constrained nominal least-squares proposal, then applies conservative
//! interval arithmetic to every target and collateral metric.
//!
//! It never executes controls, renders audio, weakens tolerances or authorizes
//! a dependent mutation. A proposal is returned only when every predicted
//! metric interval is wholly inside its frozen acceptance band.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub const ROBUST_VOCAL_CONTROL_SCHEMA_V1: &str = "robust-vocal-control/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlSolveErrorKindV1 {
    InvalidInput,
    NotRobustlyFeasible,
    DidNotConverge,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricPredictionV1 {
    pub metric_id: String,
    pub unit: String,
    pub role: MetricRoleV1,
    pub nominal_delta: f64,
    pub lower_delta: f64,
    pub upper_delta: f64,
    pub accepted_min: f64,
    pub accepted_max: f64,
    pub robustly_inside_band: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlSolveErrorV1 {
    pub kind: ControlSolveErrorKindV1,
    pub message: String,
    pub predictions: Vec<MetricPredictionV1>,
}

impl ControlSolveErrorV1 {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ControlSolveErrorKindV1::InvalidInput,
            message: message.into(),
            predictions: Vec::new(),
        }
    }

    fn infeasible(message: impl Into<String>, predictions: Vec<MetricPredictionV1>) -> Self {
        Self {
            kind: ControlSolveErrorKindV1::NotRobustlyFeasible,
            message: message.into(),
            predictions,
        }
    }

    fn non_convergent(message: impl Into<String>) -> Self {
        Self {
            kind: ControlSolveErrorKindV1::DidNotConverge,
            message: message.into(),
            predictions: Vec::new(),
        }
    }
}

impl Display for ControlSolveErrorV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl Error for ControlSolveErrorV1 {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricRoleV1 {
    Target,
    Collateral,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricDefinitionV1 {
    pub metric_id: String,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlDefinitionV1 {
    pub control_id: String,
    pub unit: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalV1 {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponseMatrixV1 {
    pub response_model_sha256: String,
    pub metrics: Vec<MetricDefinitionV1>,
    pub controls: Vec<ControlDefinitionV1>,
    /// Matrix rows correspond to metrics, columns to controls.
    pub jacobian: Vec<Vec<IntervalV1>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricConstraintV1 {
    pub metric_id: String,
    pub unit: String,
    pub role: MetricRoleV1,
    /// For Target, this is the requested response band.
    /// For Collateral, this is the calibrated no-op preservation band.
    pub accepted_delta_min: f64,
    pub accepted_delta_max: f64,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlStateV1 {
    pub control_id: String,
    pub unit: String,
    pub current_value: f64,
    pub minimum_value: f64,
    pub maximum_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolverConfigV1 {
    /// Strictly positive Tikhonov damping.
    pub regularization: f64,
    /// Maximum absolute control change for this one step.
    pub maximum_control_step: f64,
    /// Projected-gradient infinity-norm convergence threshold.
    pub convergence_tolerance: f64,
    pub maximum_iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VocalControlRequestV1 {
    pub response_model_sha256: String,
    pub no_op_calibration_receipt_sha256: String,
    pub metric_constraints: Vec<MetricConstraintV1>,
    pub control_states: Vec<ControlStateV1>,
    pub solver: SolverConfigV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlDeltaV1 {
    pub control_id: String,
    pub unit: String,
    pub current_value: f64,
    pub proposed_delta: f64,
    pub proposed_value: f64,
    pub minimum_value: f64,
    pub maximum_value: f64,
    pub maximum_step: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VocalControlProposalV1 {
    pub schema: String,
    pub response_model_sha256: String,
    pub no_op_calibration_receipt_sha256: String,
    pub control_deltas: Vec<ControlDeltaV1>,
    pub metric_predictions: Vec<MetricPredictionV1>,
    pub objective_value: f64,
    pub iterations: usize,
}

/// Solve a regularized, box-constrained local inverse-control problem.
///
/// The optimizer uses Jacobian interval midpoints for the nominal quadratic.
/// It then applies interval arithmetic to the entire coefficient intervals.
/// A proposal is returned only if every target and collateral prediction is
/// wholly inside its accepted band. Otherwise the function returns the full
/// out-of-band predictions and no control proposal.
pub fn solve_robust_vocal_control(
    matrix: &ResponseMatrixV1,
    request: &VocalControlRequestV1,
) -> Result<VocalControlProposalV1, ControlSolveErrorV1> {
    let (metrics, controls) = validate_and_align(matrix, request)?;
    let rows = metrics.len();
    let columns = controls.len();

    let mut lower_control_deltas = Vec::with_capacity(columns);
    let mut upper_control_deltas = Vec::with_capacity(columns);
    for control in &controls {
        let actuator_lower = control.minimum_value - control.current_value;
        let actuator_upper = control.maximum_value - control.current_value;
        let lower = actuator_lower.max(-request.solver.maximum_control_step);
        let upper = actuator_upper.min(request.solver.maximum_control_step);
        if lower > upper || !lower.is_finite() || !upper.is_finite() {
            return Err(ControlSolveErrorV1::infeasible(
                format!(
                    "control {} has no feasible delta inside actuator and trust-region bounds",
                    control.control_id
                ),
                Vec::new(),
            ));
        }
        lower_control_deltas.push(lower);
        upper_control_deltas.push(upper);
    }

    let mut nominal_jacobian = vec![vec![0.0f64; columns]; rows];
    let mut goals = vec![0.0f64; rows];
    let mut lipschitz_bound = 2.0 * request.solver.regularization;

    for row in 0..rows {
        goals[row] = match metrics[row].role {
            MetricRoleV1::Target => {
                (metrics[row].accepted_delta_min + metrics[row].accepted_delta_max) / 2.0
            }
            MetricRoleV1::Collateral => 0.0,
        };
        for column in 0..columns {
            let interval = matrix.jacobian[row][column];
            let midpoint = interval.lower + (interval.upper - interval.lower) / 2.0;
            if !midpoint.is_finite() {
                return Err(ControlSolveErrorV1::invalid(
                    "Jacobian midpoint is not finite",
                ));
            }
            nominal_jacobian[row][column] = midpoint;
            lipschitz_bound +=
                2.0 * metrics[row].weight * midpoint * midpoint;
        }
    }

    if !lipschitz_bound.is_finite() || lipschitz_bound <= 0.0 {
        return Err(ControlSolveErrorV1::invalid(
            "computed projected-gradient Lipschitz bound is invalid",
        ));
    }
    let learning_rate = 1.0 / lipschitz_bound;
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(ControlSolveErrorV1::invalid(
            "computed projected-gradient step is invalid",
        ));
    }

    let mut delta = vec![0.0f64; columns];
    for column in 0..columns {
        delta[column] = clamp(
            0.0,
            lower_control_deltas[column],
            upper_control_deltas[column],
        );
    }

    let mut converged = false;
    let mut used_iterations = 0usize;
    for iteration in 1..=request.solver.maximum_iterations {
        let mut predictions = vec![0.0f64; rows];
        for row in 0..rows {
            for column in 0..columns {
                predictions[row] += nominal_jacobian[row][column] * delta[column];
            }
        }

        let mut gradient = vec![0.0f64; columns];
        for column in 0..columns {
            gradient[column] = 2.0 * request.solver.regularization * delta[column];
            for row in 0..rows {
                let residual = predictions[row] - goals[row];
                gradient[column] += 2.0
                    * metrics[row].weight
                    * nominal_jacobian[row][column]
                    * residual;
            }
        }

        let mut maximum_change = 0.0f64;
        let mut next_delta = delta.clone();
        for column in 0..columns {
            let candidate = delta[column] - learning_rate * gradient[column];
            next_delta[column] = clamp(
                candidate,
                lower_control_deltas[column],
                upper_control_deltas[column],
            );
            maximum_change =
                maximum_change.max((next_delta[column] - delta[column]).abs());
        }
        delta = next_delta;
        used_iterations = iteration;

        if maximum_change <= request.solver.convergence_tolerance {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(ControlSolveErrorV1::non_convergent(format!(
            "projected-gradient solver did not converge in {} iterations",
            request.solver.maximum_iterations
        )));
    }

    let mut metric_predictions = Vec::with_capacity(rows);
    let mut robust_failure_count = 0usize;
    for row in 0..rows {
        let mut nominal_delta = 0.0f64;
        let mut lower_delta = 0.0f64;
        let mut upper_delta = 0.0f64;

        for column in 0..columns {
            let control_delta = delta[column];
            let coefficient = matrix.jacobian[row][column];
            let midpoint = coefficient.lower + (coefficient.upper - coefficient.lower) / 2.0;
            nominal_delta += midpoint * control_delta;

            let (term_lower, term_upper) = if control_delta >= 0.0 {
                (
                    coefficient.lower * control_delta,
                    coefficient.upper * control_delta,
                )
            } else {
                (
                    coefficient.upper * control_delta,
                    coefficient.lower * control_delta,
                )
            };
            lower_delta += term_lower;
            upper_delta += term_upper;
        }

        let robustly_inside_band = lower_delta >= metrics[row].accepted_delta_min
            && upper_delta <= metrics[row].accepted_delta_max;
        if !robustly_inside_band {
            robust_failure_count += 1;
        }
        metric_predictions.push(MetricPredictionV1 {
            metric_id: metrics[row].metric_id.clone(),
            unit: metrics[row].unit.clone(),
            role: metrics[row].role,
            nominal_delta,
            lower_delta,
            upper_delta,
            accepted_min: metrics[row].accepted_delta_min,
            accepted_max: metrics[row].accepted_delta_max,
            robustly_inside_band,
        });
    }

    if robust_failure_count > 0 {
        return Err(ControlSolveErrorV1::infeasible(
            format!(
                "{robust_failure_count} target/collateral metric intervals do not fit their frozen acceptance bands"
            ),
            metric_predictions,
        ));
    }

    let mut objective_value = 0.0f64;
    for row in 0..rows {
        let residual = metric_predictions[row].nominal_delta - goals[row];
        objective_value += metrics[row].weight * residual * residual;
    }
    for value in &delta {
        objective_value += request.solver.regularization * value * value;
    }
    if !objective_value.is_finite() {
        return Err(ControlSolveErrorV1::invalid(
            "computed objective value is not finite",
        ));
    }

    let control_deltas = controls
        .iter()
        .enumerate()
        .map(|(index, control)| ControlDeltaV1 {
            control_id: control.control_id.clone(),
            unit: control.unit.clone(),
            current_value: control.current_value,
            proposed_delta: delta[index],
            proposed_value: control.current_value + delta[index],
            minimum_value: control.minimum_value,
            maximum_value: control.maximum_value,
            maximum_step: request.solver.maximum_control_step,
        })
        .collect();

    Ok(VocalControlProposalV1 {
        schema: ROBUST_VOCAL_CONTROL_SCHEMA_V1.to_owned(),
        response_model_sha256: matrix.response_model_sha256.clone(),
        no_op_calibration_receipt_sha256: request.no_op_calibration_receipt_sha256.clone(),
        control_deltas,
        metric_predictions,
        objective_value,
        iterations: used_iterations,
    })
}

fn validate_and_align<'a>(
    matrix: &'a ResponseMatrixV1,
    request: &'a VocalControlRequestV1,
) -> Result<(Vec<&'a MetricConstraintV1>, Vec<&'a ControlStateV1>), ControlSolveErrorV1> {
    validate_digest(
        &matrix.response_model_sha256,
        "response_model_sha256",
    )?;
    validate_digest(
        &request.response_model_sha256,
        "request response_model_sha256",
    )?;
    validate_digest(
        &request.no_op_calibration_receipt_sha256,
        "no_op_calibration_receipt_sha256",
    )?;
    if matrix.response_model_sha256 != request.response_model_sha256 {
        return Err(ControlSolveErrorV1::invalid(
            "request response model digest does not match the supplied matrix",
        ));
    }

    if matrix.metrics.is_empty() || matrix.controls.is_empty() {
        return Err(ControlSolveErrorV1::invalid(
            "response matrix must contain at least one metric and one control",
        ));
    }
    if matrix.jacobian.len() != matrix.metrics.len()
        || matrix.jacobian.iter().any(|row| row.len() != matrix.controls.len())
    {
        return Err(ControlSolveErrorV1::invalid(
            "response matrix dimensions do not match metric/control definitions",
        ));
    }

    let mut matrix_metric_ids = Vec::new();
    for metric in &matrix.metrics {
        require_non_empty(&metric.metric_id, "matrix metric_id")?;
        require_non_empty(&metric.unit, "matrix metric unit")?;
        if matrix_metric_ids.contains(&metric.metric_id) {
            return Err(ControlSolveErrorV1::invalid(format!(
                "duplicate matrix metric ID {}",
                metric.metric_id
            )));
        }
        matrix_metric_ids.push(metric.metric_id.clone());
    }

    let mut matrix_control_ids = Vec::new();
    for control in &matrix.controls {
        require_non_empty(&control.control_id, "matrix control_id")?;
        require_non_empty(&control.unit, "matrix control unit")?;
        if matrix_control_ids.contains(&control.control_id) {
            return Err(ControlSolveErrorV1::invalid(format!(
                "duplicate matrix control ID {}",
                control.control_id
            )));
        }
        matrix_control_ids.push(control.control_id.clone());
    }

    for row in &matrix.jacobian {
        for cell in row {
            if !cell.lower.is_finite() || !cell.upper.is_finite() || cell.lower > cell.upper {
                return Err(ControlSolveErrorV1::invalid(
                    "Jacobian intervals must be finite and ordered",
                ));
            }
        }
    }

    if request.metric_constraints.len() != matrix.metrics.len() {
        return Err(ControlSolveErrorV1::invalid(
            "metric constraints must cover every response-matrix metric exactly once",
        ));
    }
    if request.control_states.len() != matrix.controls.len() {
        return Err(ControlSolveErrorV1::invalid(
            "control states must cover every response-matrix control exactly once",
        ));
    }

    let mut metric_by_id = std::collections::BTreeMap::new();
    let mut target_count = 0usize;
    let mut collateral_count = 0usize;
    for metric in &request.metric_constraints {
        require_non_empty(&metric.metric_id, "constraint metric_id")?;
        require_non_empty(&metric.unit, "constraint metric unit")?;
        if !metric.accepted_delta_min.is_finite()
            || !metric.accepted_delta_max.is_finite()
            || metric.accepted_delta_min >= metric.accepted_delta_max
            || !metric.weight.is_finite()
            || metric.weight <= 0.0
        {
            return Err(ControlSolveErrorV1::invalid(format!(
                "metric {} has invalid acceptance band or weight",
                metric.metric_id
            )));
        }
        match metric.role {
            MetricRoleV1::Target => {
                target_count += 1;
                if metric.accepted_delta_min <= 0.0 && metric.accepted_delta_max >= 0.0 {
                    return Err(ControlSolveErrorV1::invalid(format!(
                        "target metric {} must require a non-zero response band",
                        metric.metric_id
                    )));
                }
            }
            MetricRoleV1::Collateral => {
                collateral_count += 1;
                if metric.accepted_delta_min > 0.0 || metric.accepted_delta_max < 0.0 {
                    return Err(ControlSolveErrorV1::invalid(format!(
                        "collateral metric {} acceptance band must contain zero",
                        metric.metric_id
                    )));
                }
            }
        }
        if metric_by_id
            .insert(metric.metric_id.as_str(), metric)
            .is_some()
        {
            return Err(ControlSolveErrorV1::invalid(format!(
                "duplicate metric constraint {}",
                metric.metric_id
            )));
        }
    }

    if target_count == 0 || collateral_count == 0 {
        return Err(ControlSolveErrorV1::invalid(
            "at least one target and one collateral metric are required",
        ));
    }

    let mut aligned_metrics = Vec::with_capacity(matrix.metrics.len());
    for definition in &matrix.metrics {
        let constraint = metric_by_id
            .get(definition.metric_id.as_str())
            .copied()
            .ok_or_else(|| {
                ControlSolveErrorV1::invalid(format!(
                    "missing metric constraint for {}",
                    definition.metric_id
                ))
            })?;
        if constraint.unit != definition.unit {
            return Err(ControlSolveErrorV1::invalid(format!(
                "unit mismatch for metric {}",
                definition.metric_id
            )));
        }
        aligned_metrics.push(constraint);
    }

    let mut control_by_id = std::collections::BTreeMap::new();
    for control in &request.control_states {
        require_non_empty(&control.control_id, "control state control_id")?;
        require_non_empty(&control.unit, "control state unit")?;
        if !control.current_value.is_finite()
            || !control.minimum_value.is_finite()
            || !control.maximum_value.is_finite()
            || control.minimum_value > control.current_value
            || control.current_value > control.maximum_value
        {
            return Err(ControlSolveErrorV1::invalid(format!(
                "control {} has invalid current/min/max values",
                control.control_id
            )));
        }
        if control_by_id
            .insert(control.control_id.as_str(), control)
            .is_some()
        {
            return Err(ControlSolveErrorV1::invalid(format!(
                "duplicate control state {}",
                control.control_id
            )));
        }
    }

    let mut aligned_controls = Vec::with_capacity(matrix.controls.len());
    for definition in &matrix.controls {
        let state = control_by_id
            .get(definition.control_id.as_str())
            .copied()
            .ok_or_else(|| {
                ControlSolveErrorV1::invalid(format!(
                    "missing control state for {}",
                    definition.control_id
                ))
            })?;
        if state.unit != definition.unit {
            return Err(ControlSolveErrorV1::invalid(format!(
                "unit mismatch for control {}",
                definition.control_id
            )));
        }
        aligned_controls.push(state);
    }

    let config = &request.solver;
    validate_positive_finite(config.regularization, "regularization")?;
    validate_positive_finite(config.maximum_control_step, "maximum_control_step")?;
    validate_positive_finite(
        config.convergence_tolerance,
        "convergence_tolerance",
    )?;
    if config.maximum_iterations == 0 {
        return Err(ControlSolveErrorV1::invalid(
            "maximum_iterations must be greater than zero",
        ));
    }

    Ok((aligned_metrics, aligned_controls))
}

fn clamp(value: f64, lower: f64, upper: f64) -> f64 {
    value.max(lower).min(upper)
}

fn validate_digest(value: &str, field: &str) -> Result<(), ControlSolveErrorV1> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
    });
    if !valid {
        return Err(ControlSolveErrorV1::invalid(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), ControlSolveErrorV1> {
    if value.trim().is_empty() {
        Err(ControlSolveErrorV1::invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_positive_finite(value: f64, field: &str) -> Result<(), ControlSolveErrorV1> {
    if !value.is_finite() || value <= 0.0 {
        Err(ControlSolveErrorV1::invalid(format!(
            "{field} must be finite and strictly positive"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn matrix() -> ResponseMatrixV1 {
        ResponseMatrixV1 {
            response_model_sha256: digest('a'),
            metrics: vec![
                MetricDefinitionV1 {
                    metric_id: "pitch-shift".to_owned(),
                    unit: "semitone".to_owned(),
                },
                MetricDefinitionV1 {
                    metric_id: "phrase-energy".to_owned(),
                    unit: "db".to_owned(),
                },
            ],
            controls: vec![ControlDefinitionV1 {
                control_id: "pitch-control".to_owned(),
                unit: "semitone".to_owned(),
            }],
            jacobian: vec![
                vec![IntervalV1 {
                    lower: 0.99,
                    upper: 1.01,
                }],
                vec![IntervalV1 {
                    lower: -0.02,
                    upper: 0.02,
                }],
            ],
        }
    }

    fn request() -> VocalControlRequestV1 {
        VocalControlRequestV1 {
            response_model_sha256: digest('a'),
            no_op_calibration_receipt_sha256: digest('b'),
            metric_constraints: vec![
                MetricConstraintV1 {
                    metric_id: "pitch-shift".to_owned(),
                    unit: "semitone".to_owned(),
                    role: MetricRoleV1::Target,
                    accepted_delta_min: 0.09,
                    accepted_delta_max: 0.11,
                    weight: 10.0,
                },
                MetricConstraintV1 {
                    metric_id: "phrase-energy".to_owned(),
                    unit: "db".to_owned(),
                    role: MetricRoleV1::Collateral,
                    accepted_delta_min: -0.005,
                    accepted_delta_max: 0.005,
                    weight: 10.0,
                },
            ],
            control_states: vec![ControlStateV1 {
                control_id: "pitch-control".to_owned(),
                unit: "semitone".to_owned(),
                current_value: 0.0,
                minimum_value: -1.0,
                maximum_value: 1.0,
            }],
            solver: SolverConfigV1 {
                regularization: 0.01,
                maximum_control_step: 0.2,
                convergence_tolerance: 1e-10,
                maximum_iterations: 10_000,
            },
        }
    }

    #[test]
    fn proposes_target_step_with_robust_collateral_preservation() {
        let proposal = solve_robust_vocal_control(&matrix(), &request()).unwrap();
        assert_eq!(proposal.schema, ROBUST_VOCAL_CONTROL_SCHEMA_V1);
        assert_eq!(proposal.control_deltas.len(), 1);
        assert!((proposal.control_deltas[0].proposed_delta - 0.1).abs() < 1e-4);
        assert!(proposal.metric_predictions.iter().all(|metric| metric.robustly_inside_band));
        assert_eq!(proposal.response_model_sha256, digest('a'));
        assert_eq!(proposal.no_op_calibration_receipt_sha256, digest('b'));
    }

    #[test]
    fn rejects_nominally_good_step_with_uncertain_collateral_coupling() {
        let mut matrix = matrix();
        matrix.jacobian[1][0] = IntervalV1 {
            lower: 0.15,
            upper: 0.30,
        };
        let error = solve_robust_vocal_control(&matrix, &request()).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::NotRobustlyFeasible);
        assert!(!error.predictions[1].robustly_inside_band);
    }

    #[test]
    fn rejects_target_when_jacobian_uncertainty_is_too_wide() {
        let mut matrix = matrix();
        matrix.jacobian[0][0] = IntervalV1 {
            lower: 0.5,
            upper: 1.5,
        };
        let error = solve_robust_vocal_control(&matrix, &request()).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::NotRobustlyFeasible);
        assert!(!error.predictions[0].robustly_inside_band);
    }

    #[test]
    fn enforces_actuator_and_per_step_bounds() {
        let mut request = request();
        request.control_states[0].maximum_value = 0.04;
        let error = solve_robust_vocal_control(&matrix(), &request).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::NotRobustlyFeasible);
    }

    #[test]
    fn zero_response_is_not_accepted_as_a_target_success() {
        let mut matrix = matrix();
        matrix.jacobian[0][0] = IntervalV1 {
            lower: 0.0,
            upper: 0.0,
        };
        let error = solve_robust_vocal_control(&matrix, &request()).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::NotRobustlyFeasible);
    }

    #[test]
    fn coupled_controls_can_cancel_collateral_response() {
        let mut matrix = matrix();
        matrix.controls.push(ControlDefinitionV1 {
            control_id: "expression-control".to_owned(),
            unit: "unitless".to_owned(),
        });
        matrix.jacobian = vec![
            vec![
                IntervalV1 { lower: 0.99, upper: 1.01 },
                IntervalV1 { lower: 0.99, upper: 1.01 },
            ],
            vec![
                IntervalV1 { lower: 0.19, upper: 0.21 },
                IntervalV1 { lower: -0.21, upper: -0.19 },
            ],
        ];
        let mut request = request();
        request.control_states.push(ControlStateV1 {
            control_id: "expression-control".to_owned(),
            unit: "unitless".to_owned(),
            current_value: 0.0,
            minimum_value: -0.5,
            maximum_value: 0.5,
        });
        let proposal = solve_robust_vocal_control(&matrix, &request).unwrap();
        assert!(proposal.control_deltas[0].proposed_delta > 0.0);
        assert!(proposal.control_deltas[1].proposed_delta > 0.0);
        assert!(proposal.metric_predictions.iter().all(|metric| metric.robustly_inside_band));
    }

    #[test]
    fn rejects_response_model_digest_mismatch() {
        let mut request = request();
        request.response_model_sha256 = digest('c');
        let error = solve_robust_vocal_control(&matrix(), &request).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::InvalidInput);
    }

    #[test]
    fn rejects_missing_no_op_calibration_receipt() {
        let mut request = request();
        request.no_op_calibration_receipt_sha256 = "missing".to_owned();
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());
    }

    #[test]
    fn rejects_malformed_jacobian_intervals() {
        let mut matrix = matrix();
        matrix.jacobian[0][0] = IntervalV1 {
            lower: f64::NAN,
            upper: 1.0,
        };
        assert!(solve_robust_vocal_control(&matrix, &request()).is_err());

        let mut matrix = matrix();
        matrix.jacobian[0][0] = IntervalV1 {
            lower: 1.2,
            upper: 0.8,
        };
        assert!(solve_robust_vocal_control(&matrix, &request()).is_err());
    }

    #[test]
    fn rejects_metric_or_control_unit_mismatch() {
        let mut request = request();
        request.metric_constraints[0].unit = "hz".to_owned();
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());

        let mut request = request();
        request.control_states[0].unit = "hz".to_owned();
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());
    }

    #[test]
    fn rejects_control_current_value_outside_bounds() {
        let mut request = request();
        request.control_states[0].current_value = 3.0;
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());
    }

    #[test]
    fn target_band_must_exclude_no_op_and_collateral_band_must_include_it() {
        let mut target_no_op = request();
        target_no_op.metric_constraints[0].accepted_delta_min = -0.1;
        target_no_op.metric_constraints[0].accepted_delta_max = 0.1;
        assert!(solve_robust_vocal_control(&matrix(), &target_no_op).is_err());

        let mut collateral_excludes_no_op = request();
        collateral_excludes_no_op.metric_constraints[1].accepted_delta_min = 0.01;
        collateral_excludes_no_op.metric_constraints[1].accepted_delta_max = 0.02;
        assert!(solve_robust_vocal_control(&matrix(), &collateral_excludes_no_op).is_err());
    }

    #[test]
    fn rejects_missing_or_extra_metric_and_control_coverage() {
        let mut missing_metric = request();
        missing_metric.metric_constraints.pop();
        assert!(solve_robust_vocal_control(&matrix(), &missing_metric).is_err());

        let mut missing_control = request();
        missing_control.control_states.clear();
        assert!(solve_robust_vocal_control(&matrix(), &missing_control).is_err());
    }

    #[test]
    fn observer_order_in_request_does_not_change_matrix_row_alignment() {
        let matrix = matrix();
        let mut request = request();
        request.metric_constraints.swap(0, 1);
        let proposal = solve_robust_vocal_control(&matrix, &request).unwrap();
        assert!(proposal.metric_predictions[0].metric_id == "pitch-shift");
        assert!(proposal.metric_predictions[1].metric_id == "phrase-energy");
    }

    #[test]
    fn impossible_collateral_guard_returns_predictions_and_no_proposal() {
        let mut matrix = matrix();
        matrix.jacobian[1][0] = IntervalV1 {
            lower: 0.2,
            upper: 0.4,
        };
        let error = solve_robust_vocal_control(&matrix, &request()).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::NotRobustlyFeasible);
        assert_eq!(error.predictions.len(), 2);
    }

    #[test]
    fn deterministic_input_produces_identical_control_proposal() {
        let matrix = matrix();
        let request = request();
        let first = solve_robust_vocal_control(&matrix, &request).unwrap();
        let second = solve_robust_vocal_control(&matrix, &request).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn requires_target_and_collateral_metrics() {
        let mut request = request();
        request.metric_constraints[1].role = MetricRoleV1::Target;
        request.metric_constraints[1].accepted_delta_min = 0.01;
        request.metric_constraints[1].accepted_delta_max = 0.02;
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());
    }

    #[test]
    fn does_not_return_partial_delta_when_optimizer_fails_to_converge() {
        let mut request = request();
        request.solver.maximum_iterations = 1;
        request.solver.convergence_tolerance = 1e-20;
        let error = solve_robust_vocal_control(&matrix(), &request).unwrap_err();
        assert_eq!(error.kind, ControlSolveErrorKindV1::DidNotConverge);
        assert!(error.predictions.is_empty());
    }

    #[test]
    fn rejects_zero_iteration_budget_and_invalid_regularization() {
        let mut request = request();
        request.solver.maximum_iterations = 0;
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());

        let mut request = request();
        request.solver.regularization = 0.0;
        assert!(solve_robust_vocal_control(&matrix(), &request).is_err());
    }
}
