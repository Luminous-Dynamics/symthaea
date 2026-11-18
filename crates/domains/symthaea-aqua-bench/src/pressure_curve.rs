//! Exact-point pressure/flow run admission for Luminous Aqua.
//!
//! This module validates the identity, units, pressure datum and measurement-quality
//! contract of a pressure/flow observation. It does not infer a flow curve, interpolate
//! untested pressures, qualify task quality, certify a product, or replace Mycelix
//! metrology admission / Symtropy hydraulic authority.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// A source-referenced pressure envelope for the complete test article / rig assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedPressureEnvelope {
    pub minimum_kpa: f64,
    pub maximum_kpa: f64,
    /// Digest of the external assembly rating / test-authorization evidence.
    pub evidence_sha256: String,
}

/// Optional manufacturer-declared operating range for an exact product variant.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredProductPressureRange {
    pub minimum_kpa: f64,
    pub maximum_kpa: f64,
    /// Digest binding the exact SKU / market variant datasheet revision.
    pub source_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOutcome {
    NotAssessed,
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardMethodOutcome {
    NotAssessed,
    Pass,
    Fail,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductPressureRangeStatus {
    WithinDeclaredRange,
    OutsideDeclaredRange,
    NotDeclared,
}

/// One exact pressure-point observation. Static and upstream pressures are contextual;
/// dynamic pressure at the terminal fitting inlet is the primary test pressure.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureFlowObservation {
    pub run_id: String,
    pub specimen_id: String,
    pub market_variant_id: String,
    pub mode_id: String,
    pub geometry_sha256: String,
    /// Frozen digest of task, inlet water, target pressure, room, mounting and test setup.
    pub test_context_sha256: String,
    pub protocol_sha256: String,
    pub pressure_setpoint_kpa: f64,
    /// Preregistered maximum deviation from setpoint, in kPa.
    pub setpoint_tolerance_kpa: f64,
    /// Measured while flowing at the terminal fitting inlet, in kPa.
    pub dynamic_fitting_inlet_pressure_kpa: f64,
    /// Expanded or otherwise declared pressure uncertainty, in kPa.
    pub dynamic_pressure_uncertainty_kpa: f64,
    /// Optional upstream pressure channels; never substitutes for fitting-inlet pressure.
    pub static_supply_pressure_kpa: Option<f64>,
    pub upstream_residual_pressure_kpa: Option<f64>,
    /// Measured flow at the declared fitting/mode, in litres per minute.
    pub flow_lpm: f64,
    pub flow_uncertainty_lpm: f64,
    pub pressure_envelope: QualifiedPressureEnvelope,
    /// None means the exact product's published range has not been established.
    pub product_declared_range: Option<DeclaredProductPressureRange>,
    /// Independent endpoint status; the module does not decide the endpoint.
    pub task_outcome: TaskOutcome,
    pub standard_method_outcome: StandardMethodOutcome,
    /// Stable instrument identity or metrology receipt key for the pressure channel.
    pub pressure_instrument_id: String,
    /// Stable instrument identity or metrology receipt key for the flow channel.
    pub flow_instrument_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PressureRunError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    InvalidRange { field: &'static str },
    InvalidSetpoint,
    InvalidDynamicPressure,
    InvalidSetpointTolerance,
    InvalidComparisonTolerance,
    PressureOutsideQualifiedEnvelope,
    PressureMissesSetpointTolerance,
    InvalidPressureUncertainty,
    InvalidOptionalPressure { field: &'static str },
    InvalidFlow,
    InvalidFlowUncertainty,
    SameRunCompared,
    ContextMismatch { field: &'static str },
    PressureTargetMismatch,
    PressureToleranceMismatch,
    PressureSeparationTooLarge,
}

impl Display for PressureRunError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => {
                write!(f, "missing required pressure-run field: {field}")
            }
            Self::InvalidDigest { field } => write!(f, "invalid SHA-256 digest for: {field}"),
            Self::InvalidRange { field } => write!(f, "invalid qualified pressure range: {field}"),
            Self::InvalidSetpoint => write!(f, "pressure setpoint must be finite and positive"),
            Self::InvalidDynamicPressure => {
                write!(f, "dynamic fitting-inlet pressure must be finite and positive")
            }
            Self::InvalidSetpointTolerance => {
                write!(f, "pressure setpoint tolerance must be finite and positive")
            }
            Self::InvalidComparisonTolerance => {
                write!(f, "maximum matched-pressure separation must be finite and positive")
            }
            Self::PressureOutsideQualifiedEnvelope => {
                write!(f, "measured pressure uncertainty interval exceeds qualified test envelope")
            }
            Self::PressureMissesSetpointTolerance => {
                write!(
                    f,
                    "measured pressure plus uncertainty misses preregistered setpoint tolerance"
                )
            }
            Self::InvalidPressureUncertainty => {
                write!(f, "pressure uncertainty must be finite and non-negative")
            }
            Self::InvalidOptionalPressure { field } => {
                write!(f, "optional pressure channel is invalid: {field}")
            }
            Self::InvalidFlow => write!(f, "flow must be finite and non-negative"),
            Self::InvalidFlowUncertainty => {
                write!(f, "flow uncertainty must be finite and non-negative")
            }
            Self::SameRunCompared => write!(f, "a pressure run cannot be compared with itself"),
            Self::ContextMismatch { field } => {
                write!(f, "matched pressure-run context differs at: {field}")
            }
            Self::PressureTargetMismatch => {
                write!(f, "matched pressure runs use different setpoints")
            }
            Self::PressureToleranceMismatch => {
                write!(f, "matched pressure runs use different setpoint tolerances")
            }
            Self::PressureSeparationTooLarge => {
                write!(
                    f,
                    "measured dynamic pressure separation exceeds the preregistered comparison tolerance"
                )
            }
        }
    }
}

impl Error for PressureRunError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), PressureRunError> {
    if value.trim().is_empty() {
        return Err(PressureRunError::MissingField { field });
    }
    Ok(())
}

fn validate_range(
    minimum_kpa: f64,
    maximum_kpa: f64,
    field: &'static str,
) -> Result<(), PressureRunError> {
    if !minimum_kpa.is_finite()
        || !maximum_kpa.is_finite()
        || minimum_kpa <= 0.0
        || maximum_kpa < minimum_kpa
    {
        return Err(PressureRunError::InvalidRange { field });
    }
    Ok(())
}

impl PressureFlowObservation {
    pub fn validate(&self) -> Result<(), PressureRunError> {
        require_nonempty("run_id", &self.run_id)?;
        require_nonempty("specimen_id", &self.specimen_id)?;
        require_nonempty("market_variant_id", &self.market_variant_id)?;
        require_nonempty("mode_id", &self.mode_id)?;
        require_nonempty("pressure_instrument_id", &self.pressure_instrument_id)?;
        require_nonempty("flow_instrument_id", &self.flow_instrument_id)?;

        for (field, digest) in [
            ("geometry_sha256", self.geometry_sha256.as_str()),
            ("test_context_sha256", self.test_context_sha256.as_str()),
            ("protocol_sha256", self.protocol_sha256.as_str()),
            ("pressure_envelope.evidence_sha256", self.pressure_envelope.evidence_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                return Err(PressureRunError::InvalidDigest { field });
            }
        }

        validate_range(
            self.pressure_envelope.minimum_kpa,
            self.pressure_envelope.maximum_kpa,
            "test_article",
        )?;

        if !self.pressure_setpoint_kpa.is_finite() || self.pressure_setpoint_kpa <= 0.0 {
            return Err(PressureRunError::InvalidSetpoint);
        }
        if !self.setpoint_tolerance_kpa.is_finite() || self.setpoint_tolerance_kpa <= 0.0 {
            return Err(PressureRunError::InvalidSetpointTolerance);
        }
        if !self.dynamic_pressure_uncertainty_kpa.is_finite()
            || self.dynamic_pressure_uncertainty_kpa < 0.0
        {
            return Err(PressureRunError::InvalidPressureUncertainty);
        }
        if !self.dynamic_fitting_inlet_pressure_kpa.is_finite()
            || self.dynamic_fitting_inlet_pressure_kpa <= 0.0
        {
            return Err(PressureRunError::InvalidDynamicPressure);
        }

        let pressure_lower =
            self.dynamic_fitting_inlet_pressure_kpa - self.dynamic_pressure_uncertainty_kpa;
        let pressure_upper =
            self.dynamic_fitting_inlet_pressure_kpa + self.dynamic_pressure_uncertainty_kpa;

        if pressure_lower < self.pressure_envelope.minimum_kpa
            || pressure_upper > self.pressure_envelope.maximum_kpa
        {
            return Err(PressureRunError::PressureOutsideQualifiedEnvelope);
        }

        let setpoint_deviation = (self.dynamic_fitting_inlet_pressure_kpa
            - self.pressure_setpoint_kpa)
            .abs()
            + self.dynamic_pressure_uncertainty_kpa;
        if setpoint_deviation > self.setpoint_tolerance_kpa {
            return Err(PressureRunError::PressureMissesSetpointTolerance);
        }

        for (field, value) in [
            ("static_supply_pressure_kpa", self.static_supply_pressure_kpa),
            ("upstream_residual_pressure_kpa", self.upstream_residual_pressure_kpa),
        ] {
            if value.is_some_and(|pressure| !pressure.is_finite() || pressure < 0.0) {
                return Err(PressureRunError::InvalidOptionalPressure { field });
            }
        }

        if !self.flow_lpm.is_finite() || self.flow_lpm < 0.0 {
            return Err(PressureRunError::InvalidFlow);
        }
        if !self.flow_uncertainty_lpm.is_finite() || self.flow_uncertainty_lpm < 0.0 {
            return Err(PressureRunError::InvalidFlowUncertainty);
        }

        if let Some(range) = &self.product_declared_range {
            validate_range(
                range.minimum_kpa,
                range.maximum_kpa,
                "product_declared",
            )?;
            if !is_sha256(&range.source_sha256) {
                return Err(PressureRunError::InvalidDigest {
                    field: "product_declared_range.source_sha256",
                });
            }
        }

        Ok(())
    }

    /// Reports whether this point is inside the exact product's declared range.
    ///
    /// This is a manufacturer-document status only. It does not determine whether the
    /// task succeeded or whether the product is independently qualified.
    pub fn product_pressure_range_status(&self) -> ProductPressureRangeStatus {
        match &self.product_declared_range {
            None => ProductPressureRangeStatus::NotDeclared,
            Some(range) => {
                let lower =
                    self.dynamic_fitting_inlet_pressure_kpa - self.dynamic_pressure_uncertainty_kpa;
                let upper =
                    self.dynamic_fitting_inlet_pressure_kpa + self.dynamic_pressure_uncertainty_kpa;
                if lower >= range.minimum_kpa && upper <= range.maximum_kpa {
                    ProductPressureRangeStatus::WithinDeclaredRange
                } else {
                    ProductPressureRangeStatus::OutsideDeclaredRange
                }
            }
        }
    }
}

/// Validate that two observations are suitable for an exact-point pressure comparison.
///
/// The maximum permissible difference between dynamic fitting-inlet pressure readings is
/// supplied by the preregistered protocol, not chosen after seeing the flows. The pressure
/// uncertainty intervals are added conservatively. No interpolation between pressure
/// points is performed.
pub fn validate_matched_pressure_comparison(
    control: &PressureFlowObservation,
    candidate: &PressureFlowObservation,
    maximum_pressure_separation_kpa: f64,
) -> Result<(), PressureRunError> {
    control.validate()?;
    candidate.validate()?;

    if control.run_id == candidate.run_id {
        return Err(PressureRunError::SameRunCompared);
    }
    if !maximum_pressure_separation_kpa.is_finite() || maximum_pressure_separation_kpa <= 0.0 {
        return Err(PressureRunError::InvalidComparisonTolerance);
    }

    for (field, left, right) in [
        (
            "test_context_sha256",
            control.test_context_sha256.as_str(),
            candidate.test_context_sha256.as_str(),
        ),
        (
            "protocol_sha256",
            control.protocol_sha256.as_str(),
            candidate.protocol_sha256.as_str(),
        ),
    ] {
        if left != right {
            return Err(PressureRunError::ContextMismatch { field });
        }
    }
    if control.pressure_setpoint_kpa != candidate.pressure_setpoint_kpa {
        return Err(PressureRunError::PressureTargetMismatch);
    }
    if control.setpoint_tolerance_kpa != candidate.setpoint_tolerance_kpa {
        return Err(PressureRunError::PressureToleranceMismatch);
    }

    let conservative_separation = (control.dynamic_fitting_inlet_pressure_kpa
        - candidate.dynamic_fitting_inlet_pressure_kpa)
        .abs()
        + control.dynamic_pressure_uncertainty_kpa
        + candidate.dynamic_pressure_uncertainty_kpa;

    if conservative_separation > maximum_pressure_separation_kpa {
        return Err(PressureRunError::PressureSeparationTooLarge);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn observation(run_id: &str, product: &str, pressure_kpa: f64) -> PressureFlowObservation {
        PressureFlowObservation {
            run_id: run_id.to_owned(),
            specimen_id: format!("{product}-specimen-01"),
            market_variant_id: product.to_owned(),
            mode_id: "RAIN".to_owned(),
            geometry_sha256: digest('a'),
            test_context_sha256: digest('b'),
            protocol_sha256: digest('c'),
            pressure_setpoint_kpa: 100.0,
            setpoint_tolerance_kpa: 5.0,
            dynamic_fitting_inlet_pressure_kpa: pressure_kpa,
            dynamic_pressure_uncertainty_kpa: 1.0,
            static_supply_pressure_kpa: Some(180.0),
            upstream_residual_pressure_kpa: Some(130.0),
            flow_lpm: 5.0,
            flow_uncertainty_lpm: 0.1,
            pressure_envelope: QualifiedPressureEnvelope {
                minimum_kpa: 50.0,
                maximum_kpa: 300.0,
                evidence_sha256: digest('d'),
            },
            product_declared_range: None,
            task_outcome: TaskOutcome::NotAssessed,
            standard_method_outcome: StandardMethodOutcome::NotAssessed,
            pressure_instrument_id: "pressure-channel-01".to_owned(),
            flow_instrument_id: "flow-channel-01".to_owned(),
        }
    }

    #[test]
    fn accepts_exact_point_within_qualified_envelope() {
        let run = observation("run-01", "aqua-article", 100.0);
        assert_eq!(run.validate(), Ok(()));
    }

    #[test]
    fn uses_dynamic_fitting_inlet_pressure_not_static_or_upstream_pressure() {
        let mut run = observation("run-01", "aqua-article", 140.0);
        run.pressure_setpoint_kpa = 100.0;
        run.static_supply_pressure_kpa = Some(100.0);
        run.upstream_residual_pressure_kpa = Some(100.0);
        assert_eq!(
            run.validate(),
            Err(PressureRunError::PressureMissesSetpointTolerance)
        );
    }

    #[test]
    fn pressure_uncertainty_must_fit_inside_setpoint_tolerance() {
        let mut run = observation("run-01", "aqua-article", 104.0);
        run.dynamic_pressure_uncertainty_kpa = 2.0;
        assert_eq!(
            run.validate(),
            Err(PressureRunError::PressureMissesSetpointTolerance)
        );
    }

    #[test]
    fn pressure_interval_must_remain_inside_test_article_rating() {
        let mut run = observation("run-01", "aqua-article", 100.0);
        run.pressure_envelope.maximum_kpa = 100.5;
        run.dynamic_pressure_uncertainty_kpa = 1.0;
        assert_eq!(
            run.validate(),
            Err(PressureRunError::PressureOutsideQualifiedEnvelope)
        );
    }

    #[test]
    fn manufacturer_range_is_reported_separately_from_rig_admission() {
        let mut run = observation("run-01", "brondell-nebia-corre", 100.0);
        run.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: 138.0,
            maximum_kpa: 552.0,
            source_sha256: digest('e'),
        });
        assert_eq!(run.validate(), Ok(()));
        assert_eq!(
            run.product_pressure_range_status(),
            ProductPressureRangeStatus::OutsideDeclaredRange
        );
    }

    #[test]
    fn missing_manufacturer_range_is_not_inferred() {
        let run = observation("run-01", "aqua-article", 100.0);
        assert_eq!(
            run.product_pressure_range_status(),
            ProductPressureRangeStatus::NotDeclared
        );
    }

    #[test]
    fn reports_pressure_inside_manufacturer_range() {
        let mut run = observation("run-01", "hansgrohe-activera", 100.0);
        run.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: 50.0,
            maximum_kpa: 500.0,
            source_sha256: digest('e'),
        });
        assert_eq!(
            run.product_pressure_range_status(),
            ProductPressureRangeStatus::WithinDeclaredRange
        );
    }

    #[test]
    fn rejects_invalid_range_evidence_digest() {
        let mut run = observation("run-01", "aqua-article", 100.0);
        run.pressure_envelope.evidence_sha256 = "bad-digest".to_owned();
        assert_eq!(
            run.validate(),
            Err(PressureRunError::InvalidDigest {
                field: "pressure_envelope.evidence_sha256"
            })
        );
    }

    #[test]
    fn matched_comparison_accepts_same_context_and_pressure_point() {
        let control = observation("control-01", "hansgrohe-activera", 100.0);
        let candidate = observation("candidate-01", "aqua-article", 101.0);
        assert_eq!(
            validate_matched_pressure_comparison(&control, &candidate, 3.0),
            Ok(())
        );
    }

    #[test]
    fn matched_comparison_rejects_different_test_context() {
        let control = observation("control-01", "hansgrohe-activera", 100.0);
        let mut candidate = observation("candidate-01", "aqua-article", 100.0);
        candidate.test_context_sha256 = digest('f');
        assert_eq!(
            validate_matched_pressure_comparison(&control, &candidate, 3.0),
            Err(PressureRunError::ContextMismatch {
                field: "test_context_sha256"
            })
        );
    }

    #[test]
    fn matched_comparison_rejects_different_pressure_setpoints() {
        let control = observation("control-01", "hansgrohe-activera", 100.0);
        let mut candidate = observation("candidate-01", "aqua-article", 100.0);
        candidate.pressure_setpoint_kpa = 150.0;
        assert_eq!(
            validate_matched_pressure_comparison(&control, &candidate, 3.0),
            Err(PressureRunError::PressureTargetMismatch)
        );
    }

    #[test]
    fn matched_comparison_rejects_excessive_actual_pressure_separation() {
        let control = observation("control-01", "hansgrohe-activera", 100.0);
        let candidate = observation("candidate-01", "aqua-article", 104.0);
        assert_eq!(
            validate_matched_pressure_comparison(&control, &candidate, 3.0),
            Err(PressureRunError::PressureSeparationTooLarge)
        );
    }

    #[test]
    fn matched_comparison_rejects_same_run() {
        let run = observation("same-run", "hansgrohe-activera", 100.0);
        assert_eq!(
            validate_matched_pressure_comparison(&run, &run, 3.0),
            Err(PressureRunError::SameRunCompared)
        );
    }

    #[test]
    fn task_failure_remains_data_not_a_pressure_metadata_error() {
        let mut run = observation("run-01", "aqua-article", 100.0);
        run.task_outcome = TaskOutcome::Fail;
        run.standard_method_outcome = StandardMethodOutcome::Fail;
        assert_eq!(run.validate(), Ok(()));
    }

    #[test]
    fn rejects_negative_or_non_finite_flow() {
        let mut run = observation("run-01", "aqua-article", 100.0);
        run.flow_lpm = -0.1;
        assert_eq!(run.validate(), Err(PressureRunError::InvalidFlow));
    }
}


#[cfg(test)]
mod tolerance_tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn observation(run_id: &str, product: &str) -> PressureFlowObservation {
        PressureFlowObservation {
            run_id: run_id.to_owned(),
            specimen_id: format!("{product}-specimen-01"),
            market_variant_id: product.to_owned(),
            mode_id: "RAIN".to_owned(),
            geometry_sha256: digest('a'),
            test_context_sha256: digest('b'),
            protocol_sha256: digest('c'),
            pressure_setpoint_kpa: 100.0,
            setpoint_tolerance_kpa: 5.0,
            dynamic_fitting_inlet_pressure_kpa: 100.0,
            dynamic_pressure_uncertainty_kpa: 1.0,
            static_supply_pressure_kpa: Some(180.0),
            upstream_residual_pressure_kpa: Some(130.0),
            flow_lpm: 5.0,
            flow_uncertainty_lpm: 0.1,
            pressure_envelope: QualifiedPressureEnvelope {
                minimum_kpa: 50.0,
                maximum_kpa: 300.0,
                evidence_sha256: digest('d'),
            },
            product_declared_range: None,
            task_outcome: TaskOutcome::NotAssessed,
            standard_method_outcome: StandardMethodOutcome::NotAssessed,
            pressure_instrument_id: "pressure-channel-01".to_owned(),
            flow_instrument_id: "flow-channel-01".to_owned(),
        }
    }

    #[test]
    fn rejects_invalid_dynamic_pressure_with_specific_error() {
        let mut run = observation("run-01", "aqua-article");
        run.dynamic_fitting_inlet_pressure_kpa = f64::NAN;
        assert_eq!(run.validate(), Err(PressureRunError::InvalidDynamicPressure));
    }

    #[test]
    fn rejects_invalid_matched_pressure_separation_tolerance() {
        let control = observation("control-01", "incumbent");
        let candidate = observation("candidate-01", "aqua-article");
        assert_eq!(
            validate_matched_pressure_comparison(&control, &candidate, 0.0),
            Err(PressureRunError::InvalidComparisonTolerance)
        );
    }
}
