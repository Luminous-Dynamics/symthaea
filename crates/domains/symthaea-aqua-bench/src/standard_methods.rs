//! Method-specific pressure evidence contracts for UWL and UK ETL shower tests.
//!
//! These records preserve method-specific measurement channels and tolerance boundaries.
//! They do not implement the complete standards procedures, certify a product, or replace
//! Mycelix calibration/uncertainty admission or an independent test laboratory.

use std::error::Error;
use std::fmt::{Display, Formatter};

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_text(field: &'static str, value: &str) -> Result<(), MethodPressureError> {
    if value.trim().is_empty() {
        return Err(MethodPressureError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), MethodPressureError> {
    if !is_sha256(value) {
        return Err(MethodPressureError::InvalidDigest { field });
    }
    Ok(())
}

fn validate_pressure_interval(
    measured_kpa: f64,
    uncertainty_kpa: f64,
    minimum_kpa: f64,
    maximum_kpa: f64,
) -> Result<(), MethodPressureError> {
    if !measured_kpa.is_finite() || measured_kpa <= 0.0 {
        return Err(MethodPressureError::InvalidPressure);
    }
    if !uncertainty_kpa.is_finite() || uncertainty_kpa < 0.0 {
        return Err(MethodPressureError::InvalidPressureUncertainty);
    }
    let lower = measured_kpa - uncertainty_kpa;
    let upper = measured_kpa + uncertainty_kpa;
    if lower < minimum_kpa || upper > maximum_kpa {
        return Err(MethodPressureError::PressureIntervalOutsideMethodTolerance);
    }
    Ok(())
}

/// UWL pressure points from January 2023 Table 1.
///
/// The intervals encode the published asymmetric tolerances as measurement intervals,
/// not symmetric plus/minus tolerances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UwlPressurePoint {
    /// Type 1 P1: 0.15 MPa +0.01/-0 MPa, applied to each inlet.
    Type1P1,
    /// Type 1 P2: 0.30 MPa +0.02/-0 MPa, applied to each inlet.
    Type1P2,
    /// Type 2 P3: 0.01 MPa +0.002/-0 MPa, applied to each inlet.
    Type2P3,
}

impl UwlPressurePoint {
    pub fn allowed_interval_kpa(self) -> (f64, f64) {
        match self {
            Self::Type1P1 => (150.0, 160.0),
            Self::Type1P2 => (300.0, 320.0),
            Self::Type2P3 => (10.0, 12.0),
        }
    }
}

/// Identity of one independently supplied and measured UWL inlet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UwlInlet {
    Hot,
    Cold,
}

/// One pressure-channel record for one UWL inlet.
#[derive(Debug, Clone, PartialEq)]
pub struct UwlInletPressureChannel {
    pub inlet: UwlInlet,
    pub measured_dynamic_pressure_kpa: f64,
    pub uncertainty_kpa: f64,
    pub instrument_id: String,
    pub calibration_receipt_sha256: String,
}

/// A UWL pressure-point observation with distinct hot and cold inlet channels.
///
/// Both channels are retained independently; they are never averaged into one scalar.
#[derive(Debug, Clone, PartialEq)]
pub struct UwlPressureRun {
    pub run_id: String,
    pub specimen_id: String,
    pub market_variant_id: String,
    pub mode_id: String,
    pub hardware_configuration_sha256: String,
    pub test_context_sha256: String,
    pub protocol_sha256: String,
    pub point: UwlPressurePoint,
    pub hot_inlet: UwlInletPressureChannel,
    pub cold_inlet: UwlInletPressureChannel,
}

impl UwlPressureRun {
    pub fn validate(&self) -> Result<(), MethodPressureError> {
        require_text("run_id", &self.run_id)?;
        require_text("specimen_id", &self.specimen_id)?;
        require_text("market_variant_id", &self.market_variant_id)?;
        require_text("mode_id", &self.mode_id)?;
        require_digest(
            "hardware_configuration_sha256",
            &self.hardware_configuration_sha256,
        )?;
        require_digest("test_context_sha256", &self.test_context_sha256)?;
        require_digest("protocol_sha256", &self.protocol_sha256)?;

        if self.hot_inlet.inlet != UwlInlet::Hot {
            return Err(MethodPressureError::InletChannelIdentityMismatch {
                field: "hot_inlet",
            });
        }
        if self.cold_inlet.inlet != UwlInlet::Cold {
            return Err(MethodPressureError::InletChannelIdentityMismatch {
                field: "cold_inlet",
            });
        }

        for (field, channel) in [
            ("hot_inlet", &self.hot_inlet),
            ("cold_inlet", &self.cold_inlet),
        ] {
            require_text(
                if field == "hot_inlet" {
                    "hot_inlet.instrument_id"
                } else {
                    "cold_inlet.instrument_id"
                },
                &channel.instrument_id,
            )?;
            require_digest(
                if field == "hot_inlet" {
                    "hot_inlet.calibration_receipt_sha256"
                } else {
                    "cold_inlet.calibration_receipt_sha256"
                },
                &channel.calibration_receipt_sha256,
            )?;
            let (minimum, maximum) = self.point.allowed_interval_kpa();
            validate_pressure_interval(
                channel.measured_dynamic_pressure_kpa,
                channel.uncertainty_kpa,
                minimum,
                maximum,
            )?;
        }

        Ok(())
    }
}

/// ETL maximum-flow pressure datum: P2 at the centre of the shower faceplate.
#[derive(Debug, Clone, PartialEq)]
pub struct EtlFaceplateFlowObservation {
    pub run_id: String,
    pub specimen_id: String,
    pub market_variant_id: String,
    pub mode_id: String,
    pub geometry_sha256: String,
    pub test_context_sha256: String,
    pub protocol_sha256: String,
    pub faceplate_center_pressure_kpa: f64,
    pub faceplate_pressure_uncertainty_kpa: f64,
    pub pressure_instrument_id: String,
    pub pressure_calibration_receipt_sha256: String,
    pub stabilized_flow_lpm: f64,
    pub flow_stabilized: bool,
    pub stabilization_receipt_sha256: String,
    pub flow_uncertainty_lpm: f64,
    pub flow_instrument_id: String,
    pub flow_calibration_receipt_sha256: String,
}

impl EtlFaceplateFlowObservation {
    /// Validate the ETL P2 faceplate-centre pressure and stabilized-flow evidence shape.
    ///
    /// ETL links P2 to the UWL 2023 Type 1 P2 value of 300 kPa +20/-0. This validates
    /// the channel's identity and numeric interval only; it is not an ETL conformity test.
    pub fn validate(&self) -> Result<(), MethodPressureError> {
        for (field, value) in [
            ("run_id", self.run_id.as_str()),
            ("specimen_id", self.specimen_id.as_str()),
            ("market_variant_id", self.market_variant_id.as_str()),
            ("mode_id", self.mode_id.as_str()),
            ("pressure_instrument_id", self.pressure_instrument_id.as_str()),
            ("flow_instrument_id", self.flow_instrument_id.as_str()),
        ] {
            require_text(field, value)?;
        }

        for (field, digest) in [
            ("geometry_sha256", self.geometry_sha256.as_str()),
            ("test_context_sha256", self.test_context_sha256.as_str()),
            ("protocol_sha256", self.protocol_sha256.as_str()),
            (
                "pressure_calibration_receipt_sha256",
                self.pressure_calibration_receipt_sha256.as_str(),
            ),
            (
                "flow_calibration_receipt_sha256",
                self.flow_calibration_receipt_sha256.as_str(),
            ),
        ] {
            require_digest(field, digest)?;
        }

        validate_pressure_interval(
            self.faceplate_center_pressure_kpa,
            self.faceplate_pressure_uncertainty_kpa,
            300.0,
            320.0,
        )?;

        if !self.flow_stabilized {
            return Err(MethodPressureError::FlowNotStabilized);
        }
        require_digest(
            "stabilization_receipt_sha256",
            &self.stabilization_receipt_sha256,
        )?;

        if !self.stabilized_flow_lpm.is_finite() || self.stabilized_flow_lpm <= 0.0 {
            return Err(MethodPressureError::InvalidFlow);
        }
        if !self.flow_uncertainty_lpm.is_finite() || self.flow_uncertainty_lpm < 0.0 {
            return Err(MethodPressureError::InvalidFlowUncertainty);
        }

        Ok(())
    }
}


/// UWL 2023 functional-test observations for a temporary boost mode.
///
/// The flow-control-close / reopen step is explicit. The post-close flow is measured
/// without actively using boost and must be indistinguishable from regular flow within
/// the combined declared measurement uncertainty. The boosted-flow upper bound must be
/// strictly below twice the regular-flow lower bound, following UWL 12.1.2.
#[derive(Debug, Clone, PartialEq)]
pub struct UwlBoostFunctionObservation {
    pub run_id: String,
    pub specimen_id: String,
    pub market_variant_id: String,
    pub mode_id: String,
    pub hardware_configuration_sha256: String,
    pub test_context_sha256: String,
    pub protocol_sha256: String,
    pub pressure: UwlPressureRun,
    pub regular_flow_lpm: f64,
    pub regular_flow_uncertainty_lpm: f64,
    pub boost_flow_lpm: f64,
    pub boost_flow_uncertainty_lpm: f64,
    pub post_close_flow_lpm: f64,
    pub post_close_flow_uncertainty_lpm: f64,
    pub boost_activated_receipt_sha256: String,
    pub flow_close_reopen_receipt_sha256: String,
    pub boost_reset_observed_after_close: bool,
    pub flow_instrument_id: String,
    pub flow_calibration_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UwlBoostFunctionResult {
    pub regular_flow_lpm: f64,
    pub boost_flow_lpm: f64,
    pub post_close_flow_lpm: f64,
    pub boost_to_regular_ratio: f64,
    pub post_close_difference_lpm: f64,
    pub post_close_equivalent_within_measurement_uncertainty: bool,
    pub boost_below_two_times_regular_on_conservative_bounds: bool,
    /// Method result only. This does not qualify task quality, energy savings, comfort,
    /// hygiene, a device's safety or complete UWL scheme conformity.
    pub method_conditions_supported: bool,
}

impl UwlBoostFunctionObservation {
    /// Validate the UWL 2023 boost-function sequence at Type 1 P2.
    ///
    /// UWL 12.1 defines boost as a temporary increase for a specific user action. Its
    /// procedure records Qreg, activates boost and records Qboost, closes the flow control,
    /// reopens without boost and records Qmax. The requirement is Qmax = Qreg and
    /// Qboost < 2 Qreg. The method does not define a universal absolute L/min equality
    /// tolerance here, so equality is assessed only within the declared combined flow
    /// measurement uncertainty; the laboratory must preserve that uncertainty evidence.
    pub fn validate(&self) -> Result<UwlBoostFunctionResult, MethodPressureError> {
        for (field, value) in [
            ("run_id", self.run_id.as_str()),
            ("specimen_id", self.specimen_id.as_str()),
            ("market_variant_id", self.market_variant_id.as_str()),
            ("mode_id", self.mode_id.as_str()),
            ("flow_instrument_id", self.flow_instrument_id.as_str()),
        ] {
            require_text(field, value)?;
        }
        for (field, digest) in [
            ("hardware_configuration_sha256", self.hardware_configuration_sha256.as_str()),
            ("test_context_sha256", self.test_context_sha256.as_str()),
            ("protocol_sha256", self.protocol_sha256.as_str()),
            ("boost_activated_receipt_sha256", self.boost_activated_receipt_sha256.as_str()),
            ("flow_close_reopen_receipt_sha256", self.flow_close_reopen_receipt_sha256.as_str()),
            ("flow_calibration_receipt_sha256", self.flow_calibration_receipt_sha256.as_str()),
        ] {
            require_digest(field, digest)?;
        }

        self.pressure.validate()?;
        if self.pressure.point != UwlPressurePoint::Type1P2 {
            return Err(MethodPressureError::BoostRequiresType1P2);
        }
        if self.pressure.run_id != self.run_id
            || self.pressure.specimen_id != self.specimen_id
            || self.pressure.market_variant_id != self.market_variant_id
            || self.pressure.mode_id != self.mode_id
            || self.pressure.hardware_configuration_sha256 != self.hardware_configuration_sha256
            || self.pressure.test_context_sha256 != self.test_context_sha256
            || self.pressure.protocol_sha256 != self.protocol_sha256
        {
            return Err(MethodPressureError::BoostPressureContextMismatch);
        }
        if !self.boost_reset_observed_after_close {
            return Err(MethodPressureError::BoostDidNotResetAfterFlowClose);
        }

        for (field, flow, uncertainty) in [
            ("regular_flow_lpm", self.regular_flow_lpm, self.regular_flow_uncertainty_lpm),
            ("boost_flow_lpm", self.boost_flow_lpm, self.boost_flow_uncertainty_lpm),
            ("post_close_flow_lpm", self.post_close_flow_lpm, self.post_close_flow_uncertainty_lpm),
        ] {
            if !flow.is_finite() || flow <= 0.0 {
                return Err(MethodPressureError::InvalidFlow);
            }
            if !uncertainty.is_finite() || uncertainty < 0.0 {
                return Err(MethodPressureError::InvalidFlowUncertainty);
            }
            if flow - uncertainty <= 0.0 {
                return Err(MethodPressureError::InvalidFlowUncertainty);
            }
            if self.flow_instrument_id.trim().is_empty() {
                return Err(MethodPressureError::MissingField {
                    field: "flow_instrument_id",
                });
            }
            let _ = field;
        }

        if self.boost_flow_lpm - self.boost_flow_uncertainty_lpm
            <= self.regular_flow_lpm + self.regular_flow_uncertainty_lpm
        {
            return Err(MethodPressureError::BoostIncreaseNotResolved);
        }

        let conservative_boost_upper = self.boost_flow_lpm + self.boost_flow_uncertainty_lpm;
        let conservative_regular_lower = self.regular_flow_lpm - self.regular_flow_uncertainty_lpm;
        if conservative_boost_upper >= 2.0 * conservative_regular_lower {
            return Err(MethodPressureError::BoostNotBelowTwoTimesRegular);
        }

        let post_close_difference_lpm = (self.post_close_flow_lpm - self.regular_flow_lpm).abs();
        let combined_uncertainty =
            self.post_close_flow_uncertainty_lpm + self.regular_flow_uncertainty_lpm;
        if post_close_difference_lpm > combined_uncertainty {
            return Err(MethodPressureError::BoostDidNotReturnToRegularFlow);
        }

        let boost_to_regular_ratio = self.boost_flow_lpm / self.regular_flow_lpm;
        if !boost_to_regular_ratio.is_finite() || !post_close_difference_lpm.is_finite() {
            return Err(MethodPressureError::InvalidFlow);
        }

        Ok(UwlBoostFunctionResult {
            regular_flow_lpm: self.regular_flow_lpm,
            boost_flow_lpm: self.boost_flow_lpm,
            post_close_flow_lpm: self.post_close_flow_lpm,
            boost_to_regular_ratio,
            post_close_difference_lpm,
            post_close_equivalent_within_measurement_uncertainty: true,
            boost_below_two_times_regular_on_conservative_bounds: true,
            method_conditions_supported: true,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MethodPressureError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    InvalidPressure,
    InvalidPressureUncertainty,
    PressureIntervalOutsideMethodTolerance,
    InletChannelIdentityMismatch { field: &'static str },
    InvalidFlow,
    FlowNotStabilized,
    InvalidFlowUncertainty,
    BoostRequiresType1P2,
    BoostPressureContextMismatch,
    BoostDidNotResetAfterFlowClose,
    BoostIncreaseNotResolved,
    BoostNotBelowTwoTimesRegular,
    BoostDidNotReturnToRegularFlow,
}

impl Display for MethodPressureError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing method evidence field: {field}"),
            Self::InvalidDigest { field } => write!(f, "invalid method evidence digest: {field}"),
            Self::InvalidPressure => write!(f, "method pressure must be finite and positive"),
            Self::InvalidPressureUncertainty => {
                write!(f, "method pressure uncertainty must be finite and non-negative")
            }
            Self::PressureIntervalOutsideMethodTolerance => {
                write!(f, "pressure plus uncertainty is outside the method tolerance")
            }
            Self::InletChannelIdentityMismatch { field } => {
                write!(f, "UWL inlet channel identity mismatch: {field}")
            }
            Self::InvalidFlow => write!(f, "stabilized flow must be finite and positive"),
            Self::FlowNotStabilized => {
                write!(f, "ETL flow result cannot be admitted before stabilization")
            }
            Self::InvalidFlowUncertainty => {
                write!(f, "flow uncertainty must be finite and non-negative")
            }
            Self::BoostRequiresType1P2 => {
                write!(f, "UWL boost-function test requires Type 1 P2")
            }
            Self::BoostPressureContextMismatch => {
                write!(f, "UWL boost pressure run does not match the boost observation context")
            }
            Self::BoostDidNotResetAfterFlowClose => {
                write!(f, "boost mode did not reset after the flow control was closed")
            }
            Self::BoostIncreaseNotResolved => {
                write!(f, "boost flow increase is not resolved above regular flow uncertainty")
            }
            Self::BoostNotBelowTwoTimesRegular => {
                write!(f, "boost flow upper bound is not below twice regular flow lower bound")
            }
            Self::BoostDidNotReturnToRegularFlow => {
                write!(f, "post-close flow is not equal to regular flow within combined measurement uncertainty")
            }
        }
    }
}

impl Error for MethodPressureError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn channel(inlet: UwlInlet, pressure: f64) -> UwlInletPressureChannel {
        UwlInletPressureChannel {
            inlet,
            measured_dynamic_pressure_kpa: pressure,
            uncertainty_kpa: 0.5,
            instrument_id: match inlet {
                UwlInlet::Hot => "hot-pressure-01",
                UwlInlet::Cold => "cold-pressure-01",
            }
            .to_owned(),
            calibration_receipt_sha256: digest('a'),
        }
    }

    fn uwl_run(point: UwlPressurePoint, hot: f64, cold: f64) -> UwlPressureRun {
        UwlPressureRun {
            run_id: "run-01".to_owned(),
            specimen_id: "specimen-01".to_owned(),
            market_variant_id: "variant-01".to_owned(),
            mode_id: "RAIN".to_owned(),
            hardware_configuration_sha256: digest('b'),
            test_context_sha256: digest('c'),
            protocol_sha256: digest('d'),
            point,
            hot_inlet: channel(UwlInlet::Hot, hot),
            cold_inlet: channel(UwlInlet::Cold, cold),
        }
    }

    #[test]
    fn accepts_uwl_type1_p1_hot_and_cold_inlets_independently() {
        let run = uwl_run(UwlPressurePoint::Type1P1, 154.0, 158.0);
        assert_eq!(run.validate(), Ok(()));
    }

    #[test]
    fn accepts_uwl_type1_p2_hot_and_cold_inlets_independently() {
        let run = uwl_run(UwlPressurePoint::Type1P2, 304.0, 318.0);
        assert_eq!(run.validate(), Ok(()));
    }

    #[test]
    fn accepts_uwl_type2_p3_low_pressure_point() {
        let run = uwl_run(UwlPressurePoint::Type2P3, 10.5, 11.5);
        assert_eq!(run.validate(), Ok(()));
    }

    #[test]
    fn rejects_swapped_hot_and_cold_channel_identities() {
        let mut run = uwl_run(UwlPressurePoint::Type1P1, 154.0, 158.0);
        run.hot_inlet.inlet = UwlInlet::Cold;
        assert_eq!(
            run.validate(),
            Err(MethodPressureError::InletChannelIdentityMismatch {
                field: "hot_inlet"
            })
        );
    }

    #[test]
    fn rejects_pressure_below_asymmetric_uwl_lower_bound() {
        let run = uwl_run(UwlPressurePoint::Type1P1, 149.0, 154.0);
        assert_eq!(
            run.validate(),
            Err(MethodPressureError::PressureIntervalOutsideMethodTolerance)
        );
    }

    #[test]
    fn rejects_pressure_above_asymmetric_uwl_upper_bound() {
        let run = uwl_run(UwlPressurePoint::Type1P2, 320.0, 308.0);
        assert_eq!(
            run.validate(),
            Err(MethodPressureError::PressureIntervalOutsideMethodTolerance)
        );
    }

    #[test]
    fn rejects_pressure_uncertainty_that_crosses_method_bound() {
        let mut run = uwl_run(UwlPressurePoint::Type2P3, 11.5, 11.0);
        run.hot_inlet.uncertainty_kpa = 1.0;
        assert_eq!(
            run.validate(),
            Err(MethodPressureError::PressureIntervalOutsideMethodTolerance)
        );
    }

    #[test]
    fn requires_separate_calibration_receipts_for_both_inlet_channels() {
        let mut run = uwl_run(UwlPressurePoint::Type1P1, 154.0, 158.0);
        run.cold_inlet.calibration_receipt_sha256 = "unknown".to_owned();
        assert_eq!(
            run.validate(),
            Err(MethodPressureError::InvalidDigest {
                field: "cold_inlet.calibration_receipt_sha256"
            })
        );
    }

    fn etl_faceplate() -> EtlFaceplateFlowObservation {
        EtlFaceplateFlowObservation {
            run_id: "etl-run-01".to_owned(),
            specimen_id: "etl-specimen-01".to_owned(),
            market_variant_id: "etl-variant-01".to_owned(),
            mode_id: "RAIN".to_owned(),
            geometry_sha256: digest('a'),
            test_context_sha256: digest('b'),
            protocol_sha256: digest('c'),
            faceplate_center_pressure_kpa: 310.0,
            faceplate_pressure_uncertainty_kpa: 2.0,
            pressure_instrument_id: "faceplate-pressure-01".to_owned(),
            pressure_calibration_receipt_sha256: digest('d'),
            stabilized_flow_lpm: 7.1,
            flow_stabilized: true,
            stabilization_receipt_sha256: digest('f'),
            flow_uncertainty_lpm: 0.1,
            flow_instrument_id: "flow-01".to_owned(),
            flow_calibration_receipt_sha256: digest('e'),
        }
    }

    #[test]
    fn accepts_etl_p2_pressure_at_faceplate_centre_after_flow_stabilization() {
        assert_eq!(etl_faceplate().validate(), Ok(()));
    }

    #[test]
    fn rejects_etl_faceplate_pressure_outside_p2_tolerance() {
        let mut observation = etl_faceplate();
        observation.faceplate_center_pressure_kpa = 299.0;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::PressureIntervalOutsideMethodTolerance)
        );
    }

    #[test]
    fn rejects_etl_faceplate_flow_without_valid_calibration_evidence() {
        let mut observation = etl_faceplate();
        observation.flow_calibration_receipt_sha256 = "unknown".to_owned();
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::InvalidDigest {
                field: "flow_calibration_receipt_sha256"
            })
        );
    }

    #[test]
    fn rejects_etl_faceplate_flow_before_stabilization() {
        let mut observation = etl_faceplate();
        observation.flow_stabilized = false;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::FlowNotStabilized)
        );
    }


    fn boost_observation() -> UwlBoostFunctionObservation {
        let mut pressure = uwl_run(UwlPressurePoint::Type1P2, 304.0, 310.0);
        pressure.run_id = "boost-run-01".to_owned();
        pressure.specimen_id = "boost-specimen-01".to_owned();
        pressure.market_variant_id = "boost-variant-01".to_owned();
        pressure.mode_id = "RAIN-BOOST".to_owned();
        pressure.hardware_configuration_sha256 = digest('a');
        pressure.test_context_sha256 = digest('b');
        pressure.protocol_sha256 = digest('c');

        UwlBoostFunctionObservation {
            run_id: "boost-run-01".to_owned(),
            specimen_id: "boost-specimen-01".to_owned(),
            market_variant_id: "boost-variant-01".to_owned(),
            mode_id: "RAIN-BOOST".to_owned(),
            hardware_configuration_sha256: digest('a'),
            test_context_sha256: digest('b'),
            protocol_sha256: digest('c'),
            pressure,
            regular_flow_lpm: 3.0,
            regular_flow_uncertainty_lpm: 0.05,
            boost_flow_lpm: 4.5,
            boost_flow_uncertainty_lpm: 0.05,
            post_close_flow_lpm: 3.02,
            post_close_flow_uncertainty_lpm: 0.05,
            boost_activated_receipt_sha256: digest('d'),
            flow_close_reopen_receipt_sha256: digest('e'),
            boost_reset_observed_after_close: true,
            flow_instrument_id: "flow-boost-01".to_owned(),
            flow_calibration_receipt_sha256: digest('f'),
        }
    }

    #[test]
    fn accepts_uwl_boost_with_reset_and_conservative_flow_bounds() {
        let result = boost_observation().validate().expect("valid boost test");
        assert!((result.boost_to_regular_ratio - 1.5).abs() < 1e-12);
        assert!(result.post_close_equivalent_within_measurement_uncertainty);
        assert!(result.boost_below_two_times_regular_on_conservative_bounds);
        assert!(result.method_conditions_supported);
    }

    #[test]
    fn rejects_boost_test_without_type1_p2_pressure() {
        let mut observation = boost_observation();
        observation.pressure.point = UwlPressurePoint::Type1P1;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::BoostRequiresType1P2)
        );
    }

    #[test]
    fn rejects_boost_test_with_pressure_context_mismatch() {
        let mut observation = boost_observation();
        observation.pressure.specimen_id = "different-specimen".to_owned();
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::BoostPressureContextMismatch)
        );
    }

    #[test]
    fn rejects_boost_that_does_not_reset_after_flow_close() {
        let mut observation = boost_observation();
        observation.boost_reset_observed_after_close = false;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::BoostDidNotResetAfterFlowClose)
        );
    }

    #[test]
    fn rejects_boost_not_demonstrably_above_regular_flow() {
        let mut observation = boost_observation();
        observation.boost_flow_lpm = 3.02;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::BoostIncreaseNotResolved)
        );
    }

    #[test]
    fn rejects_boost_at_or_above_two_times_regular_flow_conservative_bound() {
        let mut observation = boost_observation();
        observation.boost_flow_lpm = 6.2;
        observation.boost_flow_uncertainty_lpm = 0.1;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::BoostNotBelowTwoTimesRegular)
        );
    }

    #[test]
    fn rejects_post_close_flow_not_restored_to_regular() {
        let mut observation = boost_observation();
        observation.post_close_flow_lpm = 3.3;
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::BoostDidNotReturnToRegularFlow)
        );
    }

    #[test]
    fn rejects_invalid_boost_activation_receipt() {
        let mut observation = boost_observation();
        observation.boost_activated_receipt_sha256 = "unknown".to_owned();
        assert_eq!(
            observation.validate(),
            Err(MethodPressureError::InvalidDigest {
                field: "boost_activated_receipt_sha256"
            })
        );
    }

}
