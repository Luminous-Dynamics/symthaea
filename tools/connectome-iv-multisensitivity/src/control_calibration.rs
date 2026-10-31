//! Negative-control calibration of a sample-RMS direct-instrument-effect bound.
//!
//! The control model is Y0 = intercept + Z * gamma0 + error. Under the
//! conditional Gaussian, fixed-design, homoskedastic, independent-row model,
//! the OLS confidence ellipsoid for gamma0 has radius
//! sqrt(r * sigma_hat^2 * F_critical) in the K=Z'Z metric.
//! The triangle inequality therefore gives the one-sided upper bound
//!
//! sqrt(gamma_hat' K gamma_hat / n)
//!     + sqrt(r * sigma_hat^2 * F_critical / n).
//!
//! Control and target designs must have matching normalized Gram matrices
//! K/n because the pAR support is sample-design-dependent. A transfer
//! multiplier is reported separately as a sensitivity assumption; it is
//! never claimed to be validated by the negative-control regression.

use super::{cholesky, compensated_dot, compensated_sum, quadratic_form, solve_cholesky};
use core::fmt;

const GRAM_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeControlType {
    OpsinFreeLightOn,
    FluorophoreOnlyLightOn,
    OpsinExpressingLightOff,
    ShamNoLight,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlCalibrationProvenance {
    pub control_cohort_id: String,
    pub control_type: NegativeControlType,
    pub artifact_sha256: String,
    pub instrument_design_sha256: String,
    pub prespecified_before_unblind: bool,
    /// Design assertion; the function cannot prove the control is causally negative.
    pub outcome_is_negative_control: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ControlCalibrationObservation {
    pub instruments: Vec<f64>,
    pub direct_control_outcome: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectEffectBoundCalibration {
    pub provenance: ControlCalibrationProvenance,
    pub control_count: usize,
    pub instrument_count: usize,
    pub residual_degrees_of_freedom: usize,
    pub instrument_gram: Vec<Vec<f64>>,
    pub normalized_instrument_gram: Vec<Vec<f64>>,
    pub direct_effect_coefficients: Vec<f64>,
    pub residual_sigma: f64,
    pub f_critical_value: f64,
    pub center_rms: f64,
    pub confidence_radius_rms: f64,
    pub control_only_upper_bound_rms: f64,
    pub transport_multiplier: f64,
    pub applied_sensitivity_bound_rms: f64,
    /// Always false: matching design metrics does not prove control-to-target transport.
    pub transportability_proven: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationError {
    EmptyControlSample,
    EmptyTargetDesign,
    EmptyCohortId,
    InvalidArtifactDigest,
    InvalidDesignDigest,
    NotPrespecified,
    NotNegativeControl,
    WrongControlType,
    TooFewObservations { minimum: usize, actual: usize, instruments: usize },
    InstrumentDimensionMismatch { row: usize, expected: usize, actual: usize },
    TargetDimensionMismatch { row: usize, expected: usize, actual: usize },
    NonFiniteValue { row: usize, field: &'static str },
    RankDeficient { pivot: usize, value: f64 },
    GramMismatch { row: usize, column: usize, relative_difference: f64 },
    InvalidFCritical { value: f64 },
    InvalidTransportMultiplier { value: f64 },
    DegenerateResidual,
    NumericalOverflow { field: &'static str },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyControlSample => write!(f, "control sample is empty"),
            Self::EmptyTargetDesign => write!(f, "target instrument design is empty"),
            Self::EmptyCohortId => write!(f, "control cohort id is empty"),
            Self::InvalidArtifactDigest => write!(f, "artifact digest must be lowercase SHA-256"),
            Self::InvalidDesignDigest => write!(f, "instrument design digest must be lowercase SHA-256"),
            Self::NotPrespecified => write!(f, "control bound was not prespecified before unblinding"),
            Self::NotNegativeControl => write!(f, "outcome is not declared negative under the design"),
            Self::WrongControlType => write!(f, "control type cannot calibrate a direct light effect"),
            Self::TooFewObservations { minimum, actual, instruments } => write!(
                f, "at least {minimum} controls are required for {instruments} instruments; received {actual}"
            ),
            Self::InstrumentDimensionMismatch { row, expected, actual } => write!(
                f, "control row {row} has {actual} instruments; expected {expected}"
            ),
            Self::TargetDimensionMismatch { row, expected, actual } => write!(
                f, "target row {row} has {actual} instruments; expected {expected}"
            ),
            Self::NonFiniteValue { row, field } => write!(f, "row {row} has non-finite {field}"),
            Self::RankDeficient { pivot, value } => write!(f, "instrument design is rank-deficient at pivot {pivot}: {value}"),
            Self::GramMismatch { row, column, relative_difference } => write!(
                f, "normalized control/target Gram mismatch at ({row},{column}): {relative_difference}"
            ),
            Self::InvalidFCritical { value } => write!(f, "F critical value is invalid: {value}"),
            Self::InvalidTransportMultiplier { value } => write!(f, "transport multiplier must be at least one: {value}"),
            Self::DegenerateResidual => write!(f, "control regression residual variance is zero"),
            Self::NumericalOverflow { field } => write!(f, "numerical overflow computing {field}"),
        }
    }
}
impl std::error::Error for CalibrationError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn centered_design(
    rows: &[Vec<f64>],
    columns: usize,
    target: bool,
) -> Result<Vec<Vec<f64>>, CalibrationError> {
    if rows.is_empty() {
        return Err(if target { CalibrationError::EmptyTargetDesign } else { CalibrationError::EmptyControlSample });
    }
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != columns {
            return Err(if target {
                CalibrationError::TargetDimensionMismatch { row: row_index, expected: columns, actual: row.len() }
            } else {
                CalibrationError::InstrumentDimensionMismatch { row: row_index, expected: columns, actual: row.len() }
            });
        }
        for value in row {
            if !value.is_finite() {
                return Err(CalibrationError::NonFiniteValue { row: row_index, field: "instrument" });
            }
        }
    }
    let means: Vec<f64> = (0..columns)
        .map(|column| compensated_sum(rows.iter().map(|row| row[column])) / rows.len() as f64)
        .collect();
    Ok(rows.iter().map(|row| (0..columns).map(|column| row[column] - means[column]).collect()).collect())
}

fn gram(rows: &[Vec<f64>], columns: usize) -> Result<Vec<Vec<f64>>, CalibrationError> {
    let mut result = vec![vec![0.0; columns]; columns];
    for i in 0..columns {
        for j in i..columns {
            let value = compensated_sum(rows.iter().map(|row| row[i] * row[j]));
            if !value.is_finite() {
                return Err(CalibrationError::NumericalOverflow { field: "instrument Gram matrix" });
            }
            result[i][j] = value;
            result[j][i] = value;
        }
    }
    Ok(result)
}

fn map_rank_error(error: super::MultiIvError) -> CalibrationError {
    match error {
        super::MultiIvError::RankDeficientInstrumentSet { pivot_index, pivot_value } => {
            CalibrationError::RankDeficient { pivot: pivot_index, value: pivot_value }
        }
        _ => CalibrationError::NumericalOverflow { field: "instrument Gram factorization" },
    }
}

fn validate_same_normalized_gram(
    control: &[Vec<f64>],
    control_n: usize,
    target: &[Vec<f64>],
    target_n: usize,
) -> Result<(), CalibrationError> {
    let columns = control.len();
    let control_w: Vec<Vec<f64>> = control.iter()
        .map(|row| row.iter().map(|value| value / control_n as f64).collect())
        .collect();
    let target_w: Vec<Vec<f64>> = target.iter()
        .map(|row| row.iter().map(|value| value / target_n as f64).collect())
        .collect();

    for i in 0..columns {
        for j in 0..columns {
            let denominator = (control_w[i][i] * control_w[j][j])
                .abs().sqrt()
                .max((target_w[i][i] * target_w[j][j]).abs().sqrt())
                .max(f64::MIN_POSITIVE);
            let difference = (control_w[i][j] - target_w[i][j]).abs() / denominator;
            if !difference.is_finite() || difference > GRAM_TOLERANCE {
                return Err(CalibrationError::GramMismatch {
                    row: i,
                    column: j,
                    relative_difference: difference,
                });
            }
        }
    }
    Ok(())
}

/// Estimate a conservative control-only upper confidence bound on the
/// sample-RMS direct effect, then apply a separate prespecified transfer
/// multiplier. The control and target instrument designs must have the same
/// normalized Gram matrix K/n, within a fixed 1e-8 relative tolerance.
///
/// The F critical value must correspond to F(r, n_control - 1 - r) at the
/// registered confidence level. Under the stated Gaussian fixed-design
/// homoskedastic model, the confidence ellipsoid radius in the K metric is
/// sqrt(r * sigma_hat^2 * Fcrit). The returned bound adds its RMS-scaled
/// radius to the fitted center norm.
///
/// Only opsin-free/light-on and fluorophore-only/light-on controls are accepted.
/// Opsin-expressing/light-off controls are complementary but do not calibrate
/// light-driven direct paths. The multiplier is an explicit sensitivity
/// assumption, not an estimated transportability guarantee.
pub fn calibrate_direct_effect_rms_bound(
    controls: &[ControlCalibrationObservation],
    target_instrument_rows: &[Vec<f64>],
    provenance: ControlCalibrationProvenance,
    f_critical_value: f64,
    transport_multiplier: f64,
) -> Result<DirectEffectBoundCalibration, CalibrationError> {
    if controls.is_empty() {
        return Err(CalibrationError::EmptyControlSample);
    }
    if target_instrument_rows.is_empty() {
        return Err(CalibrationError::EmptyTargetDesign);
    }
    if provenance.control_cohort_id.trim().is_empty() {
        return Err(CalibrationError::EmptyCohortId);
    }
    if !valid_sha256(&provenance.artifact_sha256) {
        return Err(CalibrationError::InvalidArtifactDigest);
    }
    if !valid_sha256(&provenance.instrument_design_sha256) {
        return Err(CalibrationError::InvalidDesignDigest);
    }
    if !provenance.prespecified_before_unblind {
        return Err(CalibrationError::NotPrespecified);
    }
    if !provenance.outcome_is_negative_control {
        return Err(CalibrationError::NotNegativeControl);
    }
    if !matches!(provenance.control_type, NegativeControlType::OpsinFreeLightOn | NegativeControlType::FluorophoreOnlyLightOn) {
        return Err(CalibrationError::WrongControlType);
    }
    if !f_critical_value.is_finite() || f_critical_value <= 0.0 {
        return Err(CalibrationError::InvalidFCritical { value: f_critical_value });
    }
    if !transport_multiplier.is_finite() || transport_multiplier < 1.0 {
        return Err(CalibrationError::InvalidTransportMultiplier { value: transport_multiplier });
    }

    let columns = controls[0].instruments.len();
    if columns == 0 {
        return Err(CalibrationError::EmptyControlSample);
    }
    let n = controls.len();
    let minimum = columns + 2;
    if n < minimum {
        return Err(CalibrationError::TooFewObservations { minimum, actual: n, instruments: columns });
    }

    let mut control_rows = Vec::with_capacity(n);
    let mut outcomes = Vec::with_capacity(n);
    for (index, control) in controls.iter().enumerate() {
        if control.instruments.len() != columns {
            return Err(CalibrationError::InstrumentDimensionMismatch {
                row: index, expected: columns, actual: control.instruments.len()
            });
        }
        if !control.direct_control_outcome.is_finite() {
            return Err(CalibrationError::NonFiniteValue { row: index, field: "direct_control_outcome" });
        }
        control_rows.push(control.instruments.clone());
        outcomes.push(control.direct_control_outcome);
    }

    let control_z = centered_design(&control_rows, columns, false)?;
    let target_z = centered_design(target_instrument_rows, columns, true)?;
    let control_gram = gram(&control_z, columns)?;
    let target_gram = gram(&target_z, columns)?;
    let factor = cholesky(&control_gram).map_err(map_rank_error)?;
    let _target_factor = cholesky(&target_gram).map_err(map_rank_error)?;
    validate_same_normalized_gram(&control_gram, n, &target_gram, target_instrument_rows.len())?;

    let outcome_mean = compensated_sum(outcomes.iter().copied()) / n as f64;
    let centered_outcome: Vec<f64> = outcomes.iter().map(|value| value - outcome_mean).collect();
    let cross: Vec<f64> = (0..columns)
        .map(|column| compensated_sum(control_z.iter().zip(&centered_outcome).map(|(row, y)| row[column] * y)))
        .collect();
    let gamma = solve_cholesky(&factor, &cross).map_err(map_rank_error)?;

    let residuals: Vec<f64> = control_z.iter().zip(&centered_outcome)
        .map(|(row, y)| y - compensated_dot(row, &gamma))
        .collect();
    let residual_sse = compensated_sum(residuals.iter().map(|value| value * value));
    if !residual_sse.is_finite() {
        return Err(CalibrationError::NumericalOverflow { field: "control residual sum of squares" });
    }
    if residual_sse <= 0.0 {
        return Err(CalibrationError::DegenerateResidual);
    }

    let df = n - 1 - columns;
    let sigma = (residual_sse / df as f64).sqrt();
    let center_q = quadratic_form(&gamma, &control_gram).map_err(map_rank_error)?;
    if center_q < 0.0 {
        return Err(CalibrationError::NumericalOverflow { field: "negative center quadratic form" });
    }
    let center_rms = (center_q / n as f64).sqrt();
    let radius_rms = (
        columns as f64 * sigma * sigma * f_critical_value / n as f64
    ).sqrt();
    let upper = center_rms + radius_rms;
    let applied = upper * transport_multiplier;

    for (field, value) in [
        ("residual sigma", sigma),
        ("center RMS", center_rms),
        ("confidence radius", radius_rms),
        ("control-only upper bound", upper),
        ("applied sensitivity bound", applied),
    ] {
        if !value.is_finite() {
            return Err(CalibrationError::NumericalOverflow { field });
        }
    }

    let normalized_gram = control_gram.iter()
        .map(|row| row.iter().map(|value| value / n as f64).collect())
        .collect();

    Ok(DirectEffectBoundCalibration {
        provenance,
        control_count: n,
        instrument_count: columns,
        residual_degrees_of_freedom: df,
        instrument_gram: control_gram,
        normalized_instrument_gram: normalized_gram,
        direct_effect_coefficients: gamma,
        residual_sigma: sigma,
        f_critical_value,
        center_rms,
        confidence_radius_rms: radius_rms,
        control_only_upper_bound_rms: upper,
        transport_multiplier,
        applied_sensitivity_bound_rms: applied,
        transportability_proven: false,
    })
}
