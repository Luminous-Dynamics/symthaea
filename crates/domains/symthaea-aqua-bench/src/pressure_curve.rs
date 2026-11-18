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
    /// Some products publish a minimum without publishing a maximum.
    pub minimum_kpa: Option<f64>,
    /// Some products publish a maximum without publishing a minimum.
    pub maximum_kpa: Option<f64>,
    /// Pressure datum used in the manufacturer's declaration, if the source states it.
    /// Unknown datum is preserved and cannot be treated as a matched pressure comparison.
    pub pressure_datum: Option<PressureMeasurementDatum>,
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

/// Exact location at which the dynamic pressure channel is measured.
///
/// UWL's simultaneous pressure applied to each hot/cold inlet and ETL's faceplate-centre
/// P2 flow-test requirement are method-specific conditions. The single-channel generic
/// adapter does not claim to reproduce those multi-datum UWL/ETL procedures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureMeasurementDatum {
    /// Dynamic pressure measured immediately upstream of the terminal fitting.
    TerminalFittingInlet,
    /// Dynamic pressure measured at the point of discharge, as defined by the method.
    PointOfDischarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductPressureRangeStatus {
    WithinFullyDeclaredRange,
    MeetsDeclaredMinimumOnly,
    MeetsDeclaredMaximumOnly,
    OutsideDeclaredRange,
    NotDeclared,
    DeclaredDatumNotSpecified,
    DeclaredDatumMismatch,
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
    /// Frozen digest of the exact run, including its pressure target and recorded conditions.
    pub test_context_sha256: String,
    /// Digest of pressure-invariant service conditions shared across a pressure sweep:
    /// task, mode, inlet-water/temperature profile, room, mounting, specimen state and
    /// measurement setup. It deliberately excludes pressure target and pressure result.
    pub pressure_invariant_context_sha256: String,
    pub protocol_sha256: String,
    pub pressure_setpoint_kpa: f64,
    /// Preregistered maximum deviation from setpoint, in kPa.
    pub setpoint_tolerance_kpa: f64,
    /// Measured while flowing at pressure_datum, in kPa.
    pub dynamic_pressure_kpa: f64,
    /// Datum for the dynamic pressure channel. Comparisons must use the same datum.
    pub pressure_datum: PressureMeasurementDatum,
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
    PressureDatumMismatch,
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
            Self::PressureDatumMismatch => {
                write!(f, "matched pressure runs use different measurement datums")
            }
            Self::PressureSeparationTooLarge => {
                write!(
                    f,
                    concat!(
                        "measured dynamic pressure separation exceeds the ",
                        "preregistered comparison tolerance"
                    )
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
            (
                "pressure_invariant_context_sha256",
                self.pressure_invariant_context_sha256.as_str(),
            ),
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
        if !self.dynamic_pressure_kpa.is_finite()
            || self.dynamic_pressure_kpa <= 0.0
        {
            return Err(PressureRunError::InvalidDynamicPressure);
        }

        let pressure_lower =
            self.dynamic_pressure_kpa - self.dynamic_pressure_uncertainty_kpa;
        let pressure_upper =
            self.dynamic_pressure_kpa + self.dynamic_pressure_uncertainty_kpa;

        if pressure_lower < self.pressure_envelope.minimum_kpa
            || pressure_upper > self.pressure_envelope.maximum_kpa
        {
            return Err(PressureRunError::PressureOutsideQualifiedEnvelope);
        }

        let setpoint_deviation = (self.dynamic_pressure_kpa
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
            if range.minimum_kpa.is_none() && range.maximum_kpa.is_none() {
                return Err(PressureRunError::InvalidRange {
                    field: "product_declared",
                });
            }
            for (field, pressure) in [
                ("product_declared.minimum_kpa", range.minimum_kpa),
                ("product_declared.maximum_kpa", range.maximum_kpa),
            ] {
                if pressure.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                    return Err(PressureRunError::InvalidRange { field });
                }
            }
            if let (Some(minimum), Some(maximum)) =
                (range.minimum_kpa, range.maximum_kpa)
            {
                if maximum < minimum {
                    return Err(PressureRunError::InvalidRange {
                        field: "product_declared",
                    });
                }
            }
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
    pub fn product_pressure_range_status(
        &self,
    ) -> Result<ProductPressureRangeStatus, PressureRunError> {
        self.validate()?;
        Ok(match &self.product_declared_range {
            None => ProductPressureRangeStatus::NotDeclared,
            Some(range) => {
                let lower =
                    self.dynamic_pressure_kpa - self.dynamic_pressure_uncertainty_kpa;
                let upper =
                    self.dynamic_pressure_kpa + self.dynamic_pressure_uncertainty_kpa;
                if range.minimum_kpa.is_some_and(|minimum| lower < minimum)
                    || range.maximum_kpa.is_some_and(|maximum| upper > maximum)
                {
                    ProductPressureRangeStatus::OutsideDeclaredRange
                } else {
                    match (range.minimum_kpa, range.maximum_kpa) {
                        (Some(_), Some(_)) => {
                            ProductPressureRangeStatus::WithinFullyDeclaredRange
                        }
                        (Some(_), None) => {
                            ProductPressureRangeStatus::MeetsDeclaredMinimumOnly
                        }
                        (None, Some(_)) => {
                            ProductPressureRangeStatus::MeetsDeclaredMaximumOnly
                        }
                        (None, None) => ProductPressureRangeStatus::NotDeclared,
                    }
                }
            }
        })
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

    let conservative_separation = (control.dynamic_pressure_kpa
        - candidate.dynamic_pressure_kpa)
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
            pressure_invariant_context_sha256: digest('f'),
            protocol_sha256: digest('c'),
            pressure_setpoint_kpa: 100.0,
            setpoint_tolerance_kpa: 5.0,
            dynamic_pressure_kpa: pressure_kpa,
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
            minimum_kpa: Some(138.0),
            maximum_kpa: Some(552.0),
            source_sha256: digest('e'),
        });
        assert_eq!(run.validate(), Ok(()));
        assert_eq!(
            run.product_pressure_range_status(),
            Ok(ProductPressureRangeStatus::OutsideDeclaredRange)
        );
    }

    #[test]
    fn missing_manufacturer_range_is_not_inferred() {
        let run = observation("run-01", "aqua-article", 100.0);
        assert_eq!(
            run.product_pressure_range_status(),
            Ok(ProductPressureRangeStatus::NotDeclared)
        );
    }

    #[test]
    fn reports_pressure_inside_manufacturer_range() {
        let mut run = observation("run-01", "hansgrohe-activera", 100.0);
        run.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: Some(50.0),
            maximum_kpa: Some(500.0),
            source_sha256: digest('e'),
        });
        assert_eq!(
            run.product_pressure_range_status(),
            Ok(ProductPressureRangeStatus::WithinFullyDeclaredRange)
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
    fn minimum_only_product_declaration_is_not_promoted_to_full_range() {
        let mut run = observation("run-01", "hansgrohe-activera", 100.0);
        run.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: Some(50.0),
            maximum_kpa: None,
            source_sha256: digest('e'),
        });

        assert_eq!(run.validate(), Ok(()));
        assert_eq!(
            run.product_pressure_range_status(),
            Ok(ProductPressureRangeStatus::MeetsDeclaredMinimumOnly)
        );
    }

    #[test]
    fn maximum_only_product_declaration_remains_one_sided() {
        let mut run = observation("run-01", "test-fixture", 100.0);
        run.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: None,
            maximum_kpa: Some(150.0),
            source_sha256: digest('e'),
        });

        assert_eq!(run.validate(), Ok(()));
        assert_eq!(
            run.product_pressure_range_status(),
            Ok(ProductPressureRangeStatus::MeetsDeclaredMaximumOnly)
        );
    }

    #[test]
    fn rejects_product_range_with_no_declared_bound() {
        let mut run = observation("run-01", "test-fixture", 100.0);
        run.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: None,
            maximum_kpa: None,
            source_sha256: digest('e'),
        });

        assert_eq!(
            run.validate(),
            Err(PressureRunError::InvalidRange {
                field: "product_declared"
            })
        );
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
            pressure_invariant_context_sha256: digest('f'),
            protocol_sha256: digest('c'),
            pressure_setpoint_kpa: 100.0,
            setpoint_tolerance_kpa: 5.0,
            dynamic_pressure_kpa: 100.0,
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
        run.dynamic_pressure_kpa = f64::NAN;
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


/// Classification of a preregistered minimum flow-retention threshold.
///
/// This is a measurement-uncertainty classification, not certification or a task-quality
/// result. The lower and upper values are conservative bounds from the declared flow
/// measurement uncertainties, not statistical confidence intervals across specimens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionThresholdStatus {
    MeetsOnConservativeMeasurementBound,
    FailsEvenOnOptimisticMeasurementBound,
    UnresolvedByMeasurementUncertainty,
}

/// Errors when comparing a single specimen at two exact pressure points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PressureRetentionError {
    InvalidRequiredRetentionFraction,
    SameRunCompared,
    PressurePointsNotStrictlyOrdered,
    PressureIntervalsNotSeparated,
    SpecimenMismatch,
    ProductVariantMismatch,
    ModeMismatch,
    GeometryMismatch,
    PressureInvariantContextMismatch,
    ProtocolMismatch,
    PressureEnvelopeMismatch,
    ProductDeclarationMismatch,
    InstrumentMismatch,
    ReferenceFlowMustBePositive,
    ReferenceFlowUncertaintyDominates,
    InvalidObservation { side: &'static str },
    NonFiniteRetentionResult,
}

impl Display for PressureRetentionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequiredRetentionFraction => {
                write!(f, "required flow-retention fraction must be finite and in (0, 1]")
            }
            Self::SameRunCompared => {
                write!(f, "a pressure-response run cannot be compared with itself")
            }
            Self::PressurePointsNotStrictlyOrdered => {
                write!(f, "low-pressure setpoint must be lower than high-pressure setpoint")
            }
            Self::PressureIntervalsNotSeparated => {
                write!(
                    f,
                    "measured pressure uncertainty intervals must not overlap"
                )
            }
            Self::SpecimenMismatch => {
                write!(f, "pressure-response observations must use the same specimen")
            }
            Self::ProductVariantMismatch => {
                write!(f, "pressure-response observations must use the same exact product variant")
            }
            Self::ModeMismatch => {
                write!(f, "pressure-response observations must use the same fixture mode")
            }
            Self::GeometryMismatch => {
                write!(f, "pressure-response observations must use the same geometry revision")
            }
            Self::PressureInvariantContextMismatch => {
                write!(f, "pressure-invariant service conditions differ across pressure points")
            }
            Self::ProtocolMismatch => {
                write!(f, "pressure-response observations use different protocol revisions")
            }
            Self::PressureEnvelopeMismatch => {
                write!(
                    f,
                    "pressure-response observations use different test-article rating evidence"
                )
            }
            Self::ProductDeclarationMismatch => {
                write!(
                    f,
                    "pressure-response observations use different product pressure declarations"
                )
            }
            Self::InstrumentMismatch => {
                write!(f, "pressure-response observations use different instrument identities")
            }
            Self::ReferenceFlowMustBePositive => {
                write!(f, "high-pressure reference flow must be greater than zero")
            }
            Self::ReferenceFlowUncertaintyDominates => {
                write!(f, "high-pressure flow uncertainty reaches or exceeds the reference flow")
            }
            Self::InvalidObservation { side } => {
                write!(f, "{side} pressure observation failed exact-point validation")
            }
            Self::NonFiniteRetentionResult => {
                write!(f, "flow-retention result is non-finite")
            }
        }
    }
}

impl Error for PressureRetentionError {}

/// Exact two-point pressure response for the same physical specimen and mode.
///
/// No curve fitting, interpolation or extrapolation occurs. Both points must be measured
/// observations, not requested targets alone.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowRetentionComparison {
    pub low_pressure_setpoint_kpa: f64,
    pub high_pressure_setpoint_kpa: f64,
    pub low_pressure_measured_kpa: f64,
    pub high_pressure_measured_kpa: f64,
    pub low_pressure_flow_lpm: f64,
    pub high_pressure_flow_lpm: f64,
    /// Central measured low/high flow ratio.
    pub flow_retention_fraction: f64,
    /// Conservative lower bound: (low flow - uncertainty) / (high flow + uncertainty).
    pub retention_lower_bound: f64,
    /// Conservative upper bound: (low flow + uncertainty) / (high flow - uncertainty).
    pub retention_upper_bound: f64,
    pub required_retention_fraction: f64,
    pub threshold_status: RetentionThresholdStatus,
    /// Kept separate: flow retention cannot establish task completion.
    pub low_pressure_task_outcome: TaskOutcome,
    pub high_pressure_task_outcome: TaskOutcome,
    pub low_pressure_standard_outcome: StandardMethodOutcome,
    pub high_pressure_standard_outcome: StandardMethodOutcome,
}

/// Compare exact low- and high-pressure observations from the same specimen.
///
/// The invariant context digest must bind every service/test condition that should remain
/// constant across pressure points, while excluding the pressure setpoint and measured
/// pressure. Run-specific context digests are allowed to differ because pressure changes.
/// Flow uncertainty bounds are measurement bounds, not confidence intervals across
/// independent specimens.
///
/// A UK ETL-style 1.5 bar / 3 bar comparison may use a 0.60 required fraction only when
/// its product category, reference-flow and Unified Water Label method conditions apply.
/// This helper does not determine ETL eligibility or certification.
pub fn compare_flow_retention(
    low_pressure: &PressureFlowObservation,
    high_pressure: &PressureFlowObservation,
    required_retention_fraction: f64,
) -> Result<FlowRetentionComparison, PressureRetentionError> {
    low_pressure
        .validate()
        .map_err(|_| PressureRetentionError::InvalidObservation { side: "low" })?;
    high_pressure
        .validate()
        .map_err(|_| PressureRetentionError::InvalidObservation { side: "high" })?;

    if !required_retention_fraction.is_finite()
        || required_retention_fraction <= 0.0
        || required_retention_fraction > 1.0
    {
        return Err(PressureRetentionError::InvalidRequiredRetentionFraction);
    }
    if low_pressure.run_id == high_pressure.run_id {
        return Err(PressureRetentionError::SameRunCompared);
    }
    if low_pressure.pressure_setpoint_kpa >= high_pressure.pressure_setpoint_kpa {
        return Err(PressureRetentionError::PressurePointsNotStrictlyOrdered);
    }
    if low_pressure.specimen_id != high_pressure.specimen_id {
        return Err(PressureRetentionError::SpecimenMismatch);
    }
    if low_pressure.market_variant_id != high_pressure.market_variant_id {
        return Err(PressureRetentionError::ProductVariantMismatch);
    }
    if low_pressure.mode_id != high_pressure.mode_id {
        return Err(PressureRetentionError::ModeMismatch);
    }
    if low_pressure.geometry_sha256 != high_pressure.geometry_sha256 {
        return Err(PressureRetentionError::GeometryMismatch);
    }
    if low_pressure.pressure_invariant_context_sha256
        != high_pressure.pressure_invariant_context_sha256
    {
        return Err(PressureRetentionError::PressureInvariantContextMismatch);
    }
    if low_pressure.protocol_sha256 != high_pressure.protocol_sha256 {
        return Err(PressureRetentionError::ProtocolMismatch);
    }
    if low_pressure.pressure_envelope.minimum_kpa != high_pressure.pressure_envelope.minimum_kpa
        || low_pressure.pressure_envelope.maximum_kpa != high_pressure.pressure_envelope.maximum_kpa
        || low_pressure.pressure_envelope.evidence_sha256
            != high_pressure.pressure_envelope.evidence_sha256
    {
        return Err(PressureRetentionError::PressureEnvelopeMismatch);
    }
    if low_pressure.product_declared_range != high_pressure.product_declared_range {
        return Err(PressureRetentionError::ProductDeclarationMismatch);
    }
    if low_pressure.pressure_instrument_id != high_pressure.pressure_instrument_id
        || low_pressure.flow_instrument_id != high_pressure.flow_instrument_id
    {
        return Err(PressureRetentionError::InstrumentMismatch);
    }
    let low_pressure_upper_kpa = low_pressure.dynamic_pressure_kpa
        + low_pressure.dynamic_pressure_uncertainty_kpa;
    let high_pressure_lower_kpa = high_pressure.dynamic_pressure_kpa
        - high_pressure.dynamic_pressure_uncertainty_kpa;
    if low_pressure_upper_kpa >= high_pressure_lower_kpa {
        return Err(PressureRetentionError::PressureIntervalsNotSeparated);
    }
    if high_pressure.flow_lpm <= 0.0 {
        return Err(PressureRetentionError::ReferenceFlowMustBePositive);
    }

    let high_flow_lower = high_pressure.flow_lpm - high_pressure.flow_uncertainty_lpm;
    if high_flow_lower <= 0.0 {
        return Err(PressureRetentionError::ReferenceFlowUncertaintyDominates);
    }

    let central = low_pressure.flow_lpm / high_pressure.flow_lpm;
    let lower = (low_pressure.flow_lpm - low_pressure.flow_uncertainty_lpm).max(0.0)
        / (high_pressure.flow_lpm + high_pressure.flow_uncertainty_lpm);
    let upper = (low_pressure.flow_lpm + low_pressure.flow_uncertainty_lpm) / high_flow_lower;

    if !central.is_finite() || !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(PressureRetentionError::NonFiniteRetentionResult);
    }

    let threshold_status = if lower >= required_retention_fraction {
        RetentionThresholdStatus::MeetsOnConservativeMeasurementBound
    } else if upper < required_retention_fraction {
        RetentionThresholdStatus::FailsEvenOnOptimisticMeasurementBound
    } else {
        RetentionThresholdStatus::UnresolvedByMeasurementUncertainty
    };

    Ok(FlowRetentionComparison {
        low_pressure_setpoint_kpa: low_pressure.pressure_setpoint_kpa,
        high_pressure_setpoint_kpa: high_pressure.pressure_setpoint_kpa,
        low_pressure_measured_kpa: low_pressure.dynamic_pressure_kpa,
        high_pressure_measured_kpa: high_pressure.dynamic_pressure_kpa,
        low_pressure_flow_lpm: low_pressure.flow_lpm,
        high_pressure_flow_lpm: high_pressure.flow_lpm,
        flow_retention_fraction: central,
        retention_lower_bound: lower,
        retention_upper_bound: upper,
        required_retention_fraction,
        threshold_status,
        low_pressure_task_outcome: low_pressure.task_outcome,
        high_pressure_task_outcome: high_pressure.task_outcome,
        low_pressure_standard_outcome: low_pressure.standard_method_outcome,
        high_pressure_standard_outcome: high_pressure.standard_method_outcome,
    })
}

#[cfg(test)]
mod retention_tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn observation(run_id: &str, pressure_kpa: f64, flow_lpm: f64) -> PressureFlowObservation {
        PressureFlowObservation {
            run_id: run_id.to_owned(),
            specimen_id: "specimen-01".to_owned(),
            market_variant_id: "exact-market-sku-01".to_owned(),
            mode_id: "RAIN".to_owned(),
            geometry_sha256: digest('a'),
            test_context_sha256: digest(if pressure_kpa < 200.0 { 'b' } else { 'e' }),
            pressure_invariant_context_sha256: digest('f'),
            protocol_sha256: digest('c'),
            pressure_setpoint_kpa: pressure_kpa,
            setpoint_tolerance_kpa: 5.0,
            dynamic_pressure_kpa: pressure_kpa,
            dynamic_pressure_uncertainty_kpa: 1.0,
            static_supply_pressure_kpa: Some(350.0),
            upstream_residual_pressure_kpa: Some(pressure_kpa + 15.0),
            flow_lpm,
            flow_uncertainty_lpm: 0.1,
            pressure_envelope: QualifiedPressureEnvelope {
                minimum_kpa: 50.0,
                maximum_kpa: 550.0,
                evidence_sha256: digest('d'),
            },
            product_declared_range: None,
            task_outcome: TaskOutcome::NotAssessed,
            standard_method_outcome: StandardMethodOutcome::NotAssessed,
            pressure_instrument_id: "pressure-01".to_owned(),
            flow_instrument_id: "flow-01".to_owned(),
        }
    }

    #[test]
    fn computes_exact_two_point_flow_retention_and_uncertainty_bounds() {
        let low = observation("run-150", 150.0, 4.0);
        let high = observation("run-300", 300.0, 5.0);
        let result = compare_flow_retention(&low, &high, 0.60).expect("valid paired runs");

        assert!((result.flow_retention_fraction - 0.8).abs() < 1e-12);
        assert!((result.retention_lower_bound - 3.9 / 5.1).abs() < 1e-12);
        assert!((result.retention_upper_bound - 4.1 / 4.9).abs() < 1e-12);
        assert_eq!(
            result.threshold_status,
            RetentionThresholdStatus::MeetsOnConservativeMeasurementBound
        );
    }

    #[test]
    fn distinguishes_unresolved_measurement_uncertainty_from_pass() {
        let low = observation("run-150", 150.0, 4.0);
        let high = observation("run-300", 300.0, 5.0);
        let result = compare_flow_retention(&low, &high, 0.80).expect("valid paired runs");
        assert_eq!(
            result.threshold_status,
            RetentionThresholdStatus::UnresolvedByMeasurementUncertainty
        );
    }

    #[test]
    fn detects_threshold_failure_even_at_optimistic_bound() {
        let low = observation("run-150", 150.0, 3.0);
        let high = observation("run-300", 300.0, 5.0);
        let result = compare_flow_retention(&low, &high, 0.70).expect("valid paired runs");
        assert_eq!(
            result.threshold_status,
            RetentionThresholdStatus::FailsEvenOnOptimisticMeasurementBound
        );
    }

    #[test]
    fn allows_pressure_specific_run_context_but_requires_shared_service_context() {
        let low = observation("run-150", 150.0, 4.0);
        let high = observation("run-300", 300.0, 5.0);
        assert_ne!(low.test_context_sha256, high.test_context_sha256);
        assert_eq!(compare_flow_retention(&low, &high, 0.60).map(|_| ()), Ok(()));
    }

    #[test]
    fn rejects_different_invariant_service_context() {
        let low = observation("run-150", 150.0, 4.0);
        let mut high = observation("run-300", 300.0, 5.0);
        high.pressure_invariant_context_sha256 = digest('9');
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::PressureInvariantContextMismatch)
        );
    }

    #[test]
    fn rejects_different_specimen_or_mode() {
        let low = observation("run-150", 150.0, 4.0);
        let mut high = observation("run-300", 300.0, 5.0);
        high.specimen_id = "specimen-02".to_owned();
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::SpecimenMismatch)
        );
        high.specimen_id = low.specimen_id.clone();
        high.mode_id = "MASSAGE".to_owned();
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::ModeMismatch)
        );
    }

    #[test]
    fn rejects_non_ordered_pressure_points_and_invalid_threshold() {
        let low = observation("run-300-a", 300.0, 4.0);
        let high = observation("run-300-b", 300.0, 5.0);
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::PressurePointsNotStrictlyOrdered)
        );
        let low = observation("run-150", 150.0, 4.0);
        let high = observation("run-300", 300.0, 5.0);
        assert_eq!(
            compare_flow_retention(&low, &high, 0.0),
            Err(PressureRetentionError::InvalidRequiredRetentionFraction)
        );
    }

    #[test]
    fn rejects_zero_reference_flow_and_uncertainty_dominance() {
        let low = observation("run-150", 150.0, 0.0);
        let high = observation("run-300", 300.0, 0.0);
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::ReferenceFlowMustBePositive)
        );
        let low = observation("run-150", 150.0, 4.0);
        let mut high = observation("run-300", 300.0, 5.0);
        high.flow_uncertainty_lpm = 5.0;
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::ReferenceFlowUncertaintyDominates)
        );
    }

    #[test]
    fn rejects_different_product_variant_and_geometry() {
        let low = observation("run-150", 150.0, 4.0);
        let mut high = observation("run-300", 300.0, 5.0);
        high.market_variant_id = "different-sku".to_owned();
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::ProductVariantMismatch)
        );

        high.market_variant_id = low.market_variant_id.clone();
        high.geometry_sha256 = digest('9');
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::GeometryMismatch)
        );
    }

    #[test]
    fn rejects_protocol_rating_and_instrument_mismatches() {
        let low = observation("run-150", 150.0, 4.0);
        let mut high = observation("run-300", 300.0, 5.0);
        high.protocol_sha256 = digest('8');
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::ProtocolMismatch)
        );

        high = observation("run-300", 300.0, 5.0);
        high.pressure_envelope.maximum_kpa = 500.0;
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::PressureEnvelopeMismatch)
        );

        high = observation("run-300", 300.0, 5.0);
        high.flow_instrument_id = "flow-02".to_owned();
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::InstrumentMismatch)
        );
    }

    #[test]
    fn rejects_different_manufacturer_pressure_declarations() {
        let low = observation("run-150", 150.0, 4.0);
        let mut high = observation("run-300", 300.0, 5.0);
        high.product_declared_range = Some(DeclaredProductPressureRange {
            minimum_kpa: Some(50.0),
            maximum_kpa: Some(500.0),
            source_sha256: digest('e'),
        });
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::ProductDeclarationMismatch)
        );
    }

    #[test]
    fn preserves_task_and_standard_outcomes_separately_from_flow_retention() {
        let mut low = observation("run-150", 150.0, 4.0);
        low.task_outcome = TaskOutcome::Fail;
        low.standard_method_outcome = StandardMethodOutcome::Fail;
        let high = observation("run-300", 300.0, 5.0);

        let result = compare_flow_retention(&low, &high, 0.60).expect("valid paired runs");
        assert_eq!(
            result.threshold_status,
            RetentionThresholdStatus::MeetsOnConservativeMeasurementBound
        );
        assert_eq!(result.low_pressure_task_outcome, TaskOutcome::Fail);
        assert_eq!(result.low_pressure_standard_outcome, StandardMethodOutcome::Fail);
    }

    #[test]
    fn missing_pressure_invariant_context_is_rejected_before_pairing() {
        let mut low = observation("run-150", 150.0, 4.0);
        low.pressure_invariant_context_sha256 = "missing".to_owned();
        let high = observation("run-300", 300.0, 5.0);
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::InvalidObservation { side: "low" })
        );
    }


    #[test]
    fn rejects_reused_run_id() {
        let low = observation("same-run", 150.0, 4.0);
        let high = observation("same-run", 300.0, 5.0);
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::SameRunCompared)
        );
    }

    #[test]
    fn rejects_overlapping_measured_pressure_uncertainty_intervals() {
        let mut low = observation("run-low", 150.0, 4.0);
        let mut high = observation("run-high", 160.0, 4.2);
        low.setpoint_tolerance_kpa = 10.0;
        high.setpoint_tolerance_kpa = 10.0;
        low.dynamic_pressure_kpa = 155.0;
        high.dynamic_pressure_kpa = 156.0;
        assert_eq!(
            compare_flow_retention(&low, &high, 0.60),
            Err(PressureRetentionError::PressureIntervalsNotSeparated)
        );
    }

}
