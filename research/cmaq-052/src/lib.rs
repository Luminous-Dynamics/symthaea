#![forbid(unsafe_code)]

//! Uncertainty-aware null-reference target for bounded iid stationary streams.
//!
//! Independent reference blocks r_i in R^d obey ||r_i||_2 <= R. Coordinate
//! values therefore lie in [-R,R]. For m independent reference blocks, the
//! simultaneous Hoeffding radius
//!
//!   epsilon = R * sqrt(2 * ln(2d/delta) / m)
//!
//! gives P(mu not in S_hat) <= delta for the true iid mean mu, where S_hat is
//! the coordinatewise box around the reference sample mean. An independent
//! iid test stream then uses the CMAQ-046 e-process at alpha_test conditional
//! on mu being in S_hat. The unconditional false-alarm bound is
//!
//!   P(false alarm ever) <= alpha_test + delta <= alpha_total.
//!
//! This is an iid stationary mean guarantee. It is not a conditional-mean
//! guarantee for arbitrary serially dependent test streams.

use std::collections::HashSet;

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};

const MAX_DIMENSION: usize = 1_024;
const MAX_REFERENCE_BLOCKS: usize = 1_000_000;
const MAX_ID_BYTES: usize = 256;
const DIGEST_HEX_LEN: usize = 64;
const NORM_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    EmptyReferenceInventory,
    TooManyReferenceBlocks,
    EmptyBlockId,
    BlockIdTooLong,
    DuplicateBlockId,
    DuplicateBlockDigest,
    InvalidDigest(&'static str),
    DimensionMismatch,
    DimensionTooLarge,
    InvalidObservationNormBound,
    InvalidRiskFraction(&'static str),
    RiskBudgetExceeded,
    NonFiniteReferenceValue,
    ReferenceNormExceeded,
    NonFiniteTestValue,
    TestNormExceeded,
    DuplicateTestBlockId,
    TestBlockInventoryFull,
    CounterOverflow,
    ArithmeticOverflow,
    Monitor(ApproachabilityError),
}

impl From<ApproachabilityError> for CalibrationError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

/// Exact positive rational in (0,1). Values are not implicitly reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiskFraction {
    numerator: u32,
    denominator: u32,
}

impl RiskFraction {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self, CalibrationError> {
        if numerator == 0 || denominator == 0 || numerator >= denominator {
            return Err(CalibrationError::InvalidRiskFraction("fraction"));
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub fn numerator(&self) -> u32 {
        self.numerator
    }

    pub fn denominator(&self) -> u32 {
        self.denominator
    }

    pub fn as_f64(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

/// One independent calibration block, represented by one bounded vector.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceBlock {
    pub block_id: String,
    pub trace_sha256: String,
    pub value: Vec<f64>,
}

/// Receipt for the fitted reference confidence envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceEnvelopeReceipt {
    pub dimension: usize,
    pub reference_block_count: usize,
    pub observation_norm_bound: f64,
    pub hoeffding_radius: f64,
    pub alpha_test: RiskFraction,
    pub delta_reference_failure: RiskFraction,
    pub alpha_total: RiskFraction,
    pub reference_manifest_sha256: String,
    pub target_sha256: String,
    /// This is the analytical upper bound alpha_test + delta, not an estimate.
    pub unconditional_false_alarm_bound: f64,
}

/// Independent-reference calibrated CMAQ-046 monitor.
///
/// Required assumptions:
/// - reference blocks are mutually independent draws from a single iid null
///   distribution;
/// - the test stream is independent of the reference and iid from that same
///   distribution under the null;
/// - every reference and test vector has L2 norm at most R.
///
/// The hash fields are shape-checked only. They do not authenticate source
/// identity or statistical independence.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedReferenceApproachabilityMonitor {
    target: BoxTarget,
    monitor: ConvexTargetApproachabilityMonitor,
    receipt: ReferenceEnvelopeReceipt,
    seen_test_block_ids: HashSet<String>,
}

impl CalibratedReferenceApproachabilityMonitor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reference_blocks: Vec<ReferenceBlock>,
        observation_norm_bound: f64,
        alpha_test: RiskFraction,
        delta_reference_failure: RiskFraction,
        alpha_total: RiskFraction,
        reference_manifest_sha256: String,
        target_sha256: String,
    ) -> Result<Self, CalibrationError> {
        if reference_blocks.is_empty() {
            return Err(CalibrationError::EmptyReferenceInventory);
        }
        if reference_blocks.len() > MAX_REFERENCE_BLOCKS {
            return Err(CalibrationError::TooManyReferenceBlocks);
        }
        if !observation_norm_bound.is_finite() || observation_norm_bound <= 0.0 {
            return Err(CalibrationError::InvalidObservationNormBound);
        }
        validate_digest(&reference_manifest_sha256, "reference_manifest_sha256")?;
        validate_digest(&target_sha256, "target_sha256")?;
        validate_risk_budget(alpha_test, delta_reference_failure, alpha_total)?;

        let dimension = reference_blocks[0].value.len();
        if dimension == 0 {
            return Err(CalibrationError::DimensionMismatch);
        }
        if dimension > MAX_DIMENSION {
            return Err(CalibrationError::DimensionTooLarge);
        }

        let mut block_ids = HashSet::with_capacity(reference_blocks.len());
        let mut block_digests = HashSet::with_capacity(reference_blocks.len());
        let mut sums = vec![0.0f64; dimension];

        for block in &reference_blocks {
            if block.block_id.trim().is_empty() {
                return Err(CalibrationError::EmptyBlockId);
            }
            if block.block_id.len() > MAX_ID_BYTES {
                return Err(CalibrationError::BlockIdTooLong);
            }
            if !block_ids.insert(block.block_id.clone()) {
                return Err(CalibrationError::DuplicateBlockId);
            }
            validate_digest(&block.trace_sha256, "trace_sha256")?;
            if !block_digests.insert(block.trace_sha256.clone()) {
                return Err(CalibrationError::DuplicateBlockDigest);
            }
            if block.value.len() != dimension {
                return Err(CalibrationError::DimensionMismatch);
            }
            if block.value.iter().any(|value| !value.is_finite()) {
                return Err(CalibrationError::NonFiniteReferenceValue);
            }
            let norm = l2_norm(&block.value);
            if !norm.is_finite() {
                return Err(CalibrationError::ArithmeticOverflow);
            }
            if norm > observation_norm_bound + NORM_TOLERANCE {
                return Err(CalibrationError::ReferenceNormExceeded);
            }
            for (sum, value) in sums.iter_mut().zip(&block.value) {
                *sum += value;
                if !sum.is_finite() {
                    return Err(CalibrationError::ArithmeticOverflow);
                }
            }
        }

        let count = reference_blocks.len() as f64;
        let mean = sums
            .iter()
            .map(|sum| *sum / count)
            .collect::<Vec<_>>();
        if mean.iter().any(|value| !value.is_finite()) {
            return Err(CalibrationError::ArithmeticOverflow);
        }

        let delta = delta_reference_failure.as_f64();
        let confidence_log_argument = 2.0 * dimension as f64 / delta;
        if !confidence_log_argument.is_finite() || confidence_log_argument <= 1.0 {
            return Err(CalibrationError::ArithmeticOverflow);
        }
        let radius = observation_norm_bound
            * (2.0 * confidence_log_argument.ln() / count).sqrt();
        if !radius.is_finite() || radius <= 0.0 {
            return Err(CalibrationError::ArithmeticOverflow);
        }

        let lower = mean.iter().map(|value| value - radius).collect::<Vec<_>>();
        let upper = mean.iter().map(|value| value + radius).collect::<Vec<_>>();
        if lower.iter().chain(&upper).any(|value| !value.is_finite()) {
            return Err(CalibrationError::ArithmeticOverflow);
        }
        let target = BoxTarget::new(lower, upper)?;
        let monitor = ConvexTargetApproachabilityMonitor::new(
            target.clone(),
            observation_norm_bound,
            alpha_test.numerator,
            alpha_test.denominator,
        )?;

        let unconditional_false_alarm_bound =
            alpha_test.as_f64() + delta_reference_failure.as_f64();
        let receipt = ReferenceEnvelopeReceipt {
            dimension,
            reference_block_count: reference_blocks.len(),
            observation_norm_bound,
            hoeffding_radius: radius,
            alpha_test,
            delta_reference_failure,
            alpha_total,
            reference_manifest_sha256,
            target_sha256,
            unconditional_false_alarm_bound,
        };

        Ok(Self {
            target,
            monitor,
            receipt,
            seen_test_block_ids: HashSet::new(),
        })
    }

    /// Adds one independent test observation. Test block IDs are consumed only
    /// after the CMAQ-046 update succeeds.
    pub fn observe(
        &mut self,
        test_block_id: &str,
        value: &[f64],
    ) -> Result<ApproachabilityUpdate, CalibrationError> {
        if test_block_id.trim().is_empty() {
            return Err(CalibrationError::EmptyBlockId);
        }
        if test_block_id.len() > MAX_ID_BYTES {
            return Err(CalibrationError::BlockIdTooLong);
        }
        if self.seen_test_block_ids.contains(test_block_id) {
            return Err(CalibrationError::DuplicateTestBlockId);
        }
        if self.seen_test_block_ids.len() >= MAX_REFERENCE_BLOCKS {
            return Err(CalibrationError::TestBlockInventoryFull);
        }
        if value.len() != self.receipt.dimension {
            return Err(CalibrationError::DimensionMismatch);
        }
        if value.iter().any(|item| !item.is_finite()) {
            return Err(CalibrationError::NonFiniteTestValue);
        }
        let norm = l2_norm(value);
        if !norm.is_finite() {
            return Err(CalibrationError::ArithmeticOverflow);
        }
        if norm > self.receipt.observation_norm_bound + NORM_TOLERANCE {
            return Err(CalibrationError::TestNormExceeded);
        }

        let update = self.monitor.observe(value)?;
        self.seen_test_block_ids.insert(test_block_id.to_string());
        Ok(update)
    }

    pub fn target(&self) -> &BoxTarget {
        &self.target
    }

    pub fn receipt(&self) -> &ReferenceEnvelopeReceipt {
        &self.receipt
    }

    pub fn observations_seen(&self) -> u64 {
        self.monitor.observations_seen()
    }

    pub fn log_e_value(&self) -> f64 {
        self.monitor.log_e_value()
    }

    pub fn is_quarantined(&self) -> bool {
        self.monitor.is_quarantined()
    }
}

fn validate_risk_budget(
    alpha_test: RiskFraction,
    delta: RiskFraction,
    alpha_total: RiskFraction,
) -> Result<(), CalibrationError> {
    // Exact comparison:
    // alpha_test + delta <= alpha_total.
    let alpha_term = u128::from(alpha_test.numerator)
        .checked_mul(u128::from(delta.denominator))
        .and_then(|value| value.checked_mul(u128::from(alpha_total.denominator)))
        .ok_or(CalibrationError::ArithmeticOverflow)?;
    let delta_term = u128::from(delta.numerator)
        .checked_mul(u128::from(alpha_test.denominator))
        .and_then(|value| value.checked_mul(u128::from(alpha_total.denominator)))
        .ok_or(CalibrationError::ArithmeticOverflow)?;
    let left = alpha_term
        .checked_add(delta_term)
        .ok_or(CalibrationError::ArithmeticOverflow)?;
    let right = u128::from(alpha_total.numerator)
        .checked_mul(u128::from(alpha_test.denominator))
        .and_then(|value| value.checked_mul(u128::from(delta.denominator)))
        .ok_or(CalibrationError::ArithmeticOverflow)?;
    if left <= right {
        Ok(())
    } else {
        Err(CalibrationError::RiskBudgetExceeded)
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), CalibrationError> {
    if value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(CalibrationError::InvalidDigest(label))
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const REF_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TARGET_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn alpha_test() -> RiskFraction {
        RiskFraction::new(1, 40).unwrap()
    }

    fn delta() -> RiskFraction {
        RiskFraction::new(1, 40).unwrap()
    }

    fn alpha_total() -> RiskFraction {
        RiskFraction::new(1, 20).unwrap()
    }

    fn references(count: usize, shift: f64) -> Vec<ReferenceBlock> {
        (0..count)
            .map(|index| ReferenceBlock {
                block_id: format!("reference-block-{index}"),
                trace_sha256: format!("{:064x}", index + 1),
                value: vec![shift],
            })
            .collect()
    }

    fn monitor() -> CalibratedReferenceApproachabilityMonitor {
        CalibratedReferenceApproachabilityMonitor::new(
            references(256, 0.0),
            1.0,
            alpha_test(),
            delta(),
            alpha_total(),
            REF_SHA.into(),
            TARGET_SHA.into(),
        )
        .unwrap()
    }

    #[test]
    fn exact_alpha_budget_boundary_is_accepted() {
        assert!(validate_risk_budget(
            RiskFraction::new(1, 40).unwrap(),
            RiskFraction::new(1, 40).unwrap(),
            RiskFraction::new(1, 20).unwrap(),
        )
        .is_ok());
        assert_eq!(
            validate_risk_budget(
                RiskFraction::new(1, 30).unwrap(),
                RiskFraction::new(1, 30).unwrap(),
                RiskFraction::new(1, 20).unwrap(),
            ),
            Err(CalibrationError::RiskBudgetExceeded)
        );
    }

    #[test]
    fn invalid_risk_fractions_are_rejected() {
        assert_eq!(
            RiskFraction::new(0, 20),
            Err(CalibrationError::InvalidRiskFraction("fraction"))
        );
        assert_eq!(
            RiskFraction::new(1, 1),
            Err(CalibrationError::InvalidRiskFraction("fraction"))
        );
    }

    #[test]
    fn hoeffding_radius_matches_closed_form() {
        let fitted = monitor();
        let dimension = 1.0f64;
        let count = 256.0f64;
        let delta = 1.0 / 40.0;
        let expected = (2.0 * (2.0 * dimension / delta).ln() / count).sqrt();
        assert!((fitted.receipt().hoeffding_radius - expected).abs() < 1e-12);
        assert_eq!(fitted.receipt().reference_block_count, 256);
        assert_eq!(fitted.receipt().unconditional_false_alarm_bound, 0.05);
    }

    #[test]
    fn reference_target_contains_empirical_mean_and_known_symmetric_mean() {
        let fitted = monitor();
        let target = fitted.target();
        assert!(target.lower()[0] <= 0.0);
        assert!(target.upper()[0] >= 0.0);
        assert!(fitted.receipt().hoeffding_radius > 0.0);
    }

    #[test]
    fn duplicate_reference_ids_and_trace_digests_are_rejected() {
        let mut duplicate_id = references(4, 0.0);
        duplicate_id[3].block_id = duplicate_id[0].block_id.clone();
        assert_eq!(
            CalibratedReferenceApproachabilityMonitor::new(
                duplicate_id,
                1.0,
                alpha_test(),
                delta(),
                alpha_total(),
                REF_SHA.into(),
                TARGET_SHA.into(),
            )
            .err(),
            Some(CalibrationError::DuplicateBlockId)
        );

        let mut duplicate_digest = references(4, 0.0);
        duplicate_digest[3].trace_sha256 = duplicate_digest[0].trace_sha256.clone();
        assert_eq!(
            CalibratedReferenceApproachabilityMonitor::new(
                duplicate_digest,
                1.0,
                alpha_test(),
                delta(),
                alpha_total(),
                REF_SHA.into(),
                TARGET_SHA.into(),
            )
            .err(),
            Some(CalibrationError::DuplicateBlockDigest)
        );
    }

    #[test]
    fn malformed_reference_values_and_digests_fail_closed() {
        let mut malformed = references(4, 0.0);
        malformed[0].value[0] = f64::NAN;
        assert_eq!(
            CalibratedReferenceApproachabilityMonitor::new(
                malformed,
                1.0,
                alpha_test(),
                delta(),
                alpha_total(),
                REF_SHA.into(),
                TARGET_SHA.into(),
            )
            .err(),
            Some(CalibrationError::NonFiniteReferenceValue)
        );
        assert_eq!(
            CalibratedReferenceApproachabilityMonitor::new(
                references(4, 0.0),
                1.0,
                alpha_test(),
                delta(),
                alpha_total(),
                "bad".into(),
                TARGET_SHA.into(),
            )
            .err(),
            Some(CalibrationError::InvalidDigest("reference_manifest_sha256"))
        );
    }

    #[test]
    fn wrong_dimensions_and_norm_bounds_fail_closed() {
        let mut wrong_dimension = references(4, 0.0);
        wrong_dimension[2].value.push(0.1);
        assert_eq!(
            CalibratedReferenceApproachabilityMonitor::new(
                wrong_dimension,
                1.0,
                alpha_test(),
                delta(),
                alpha_total(),
                REF_SHA.into(),
                TARGET_SHA.into(),
            )
            .err(),
            Some(CalibrationError::DimensionMismatch)
        );

        assert_eq!(
            CalibratedReferenceApproachabilityMonitor::new(
                references(4, 1.1),
                1.0,
                alpha_test(),
                delta(),
                alpha_total(),
                REF_SHA.into(),
                TARGET_SHA.into(),
            )
            .err(),
            Some(CalibrationError::ReferenceNormExceeded)
        );
    }

    #[test]
    fn invalid_test_observation_does_not_mutate_evidence() {
        let mut fitted = monitor();
        let before = fitted.clone();
        assert_eq!(
            fitted.observe("test-1", &[f64::NAN]),
            Err(CalibrationError::NonFiniteTestValue)
        );
        assert_eq!(
            fitted.observe("test-1", &[1.1]),
            Err(CalibrationError::TestNormExceeded)
        );
        assert_eq!(fitted, before);
    }

    #[test]
    fn duplicate_test_block_cannot_consume_evidence_twice() {
        let mut fitted = monitor();
        fitted.observe("test-1", &[0.0]).unwrap();
        assert_eq!(
            fitted.observe("test-1", &[0.0]),
            Err(CalibrationError::DuplicateTestBlockId)
        );
        assert_eq!(fitted.observations_seen(), 1);
    }

    #[test]
    fn persistent_mean_shift_beyond_calibrated_envelope_accumulates_evidence() {
        let mut fitted = monitor();
        let mut quarantined = false;
        for index in 0..500 {
            let value = if index % 2 == 0 { 0.8 } else { 0.7 };
            let update = fitted.observe(&format!("test-{index}"), &[value]).unwrap();
            if update.quarantined {
                quarantined = true;
                break;
            }
        }
        assert!(quarantined);
        assert!(fitted.is_quarantined());
        assert!(fitted.log_e_value() >= (40.0f64).ln());
    }
}
