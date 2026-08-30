// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Audit receipt for the conservative aggregate surveillance screen.
//!
//! The underlying screening function intentionally stays small. This wrapper
//! binds its output to the exact algorithm identifier, caller-supplied
//! configuration, time scope, and a content commitment over the complete ordered
//! input series so a later evidence plane can preserve what was actually
//! evaluated instead of retaining only a disposition label.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::surveillance::{
    AbstentionReason, ChangeDirection, IntervalEstimate, RobustBaseline, ScreeningDisposition,
    SurveillanceAssessment, SurveillancePoint, SurveillanceScreenConfig, SurveillanceScreenError,
    assess_latest_change,
};

pub const SURVEILLANCE_SCREEN_ALGORITHM_V1: &str = "robust-median-mad-interval-guard-v1";
pub const SURVEILLANCE_SCREEN_INPUT_ID_DOMAIN_V1: &[u8] =
    b"symthaea-epidemiology-surveillance-screen-input-v1\0";
pub const SURVEILLANCE_SCREEN_RECEIPT_ID_DOMAIN_V1: &[u8] =
    b"symthaea-epidemiology-surveillance-screen-receipt-v1\0";

/// Typed semantic identity of a surveillance-screen algorithm.
///
/// The enum is intentionally small and versioned rather than storing an
/// arbitrary `&'static str` in evidence receipts. Its semantic string remains
/// the canonical v1 hash input so introducing this type does not redefine the
/// already-frozen receipt identity vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurveillanceScreenAlgorithm {
    RobustMedianMadIntervalGuardV1,
}

impl SurveillanceScreenAlgorithm {
    pub const fn semantic_id(self) -> &'static str {
        match self {
            Self::RobustMedianMadIntervalGuardV1 => SURVEILLANCE_SCREEN_ALGORITHM_V1,
        }
    }
}

impl fmt::Display for SurveillanceScreenAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.semantic_id())
    }
}

/// Content identity for the exact ordered aggregate series supplied to one
/// surveillance screen.
///
/// This is evidence identity, not source authentication. The hash commits to
/// every baseline timestamp, explicit missing marker, observed estimate and
/// uncertainty interval, plus the complete latest point. Floating-point negative
/// zero is canonicalized to positive zero so numerically identical zero values do
/// not acquire different semantic identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurveillanceScreenInputId([u8; 32]);

impl SurveillanceScreenInputId {
    pub fn from_series(baseline_points: &[SurveillancePoint], latest: SurveillancePoint) -> Self {
        let mut h = Sha256::new();
        h.update(SURVEILLANCE_SCREEN_INPUT_ID_DOMAIN_V1);
        put_usize(&mut h, baseline_points.len());
        for point in baseline_points {
            put_point(&mut h, *point);
        }
        // Explicitly separate the historical sequence from the latest point.
        h.update([0xff]);
        put_point(&mut h, latest);
        Self(h.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(self.0)
    }
}

impl fmt::Display for SurveillanceScreenInputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        put_hex(f, self.0)
    }
}

/// Content identity for one complete evidence-bearing screen receipt.
///
/// The identity commits to the exact input identity, semantic algorithm ID,
/// every screen-configuration field, explicit time scope, and every field in the
/// returned assessment. It is a content commitment only: it does not authenticate
/// a source, prove that the measurements are true, or grant operational authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurveillanceScreenReceiptId([u8; 32]);

impl SurveillanceScreenReceiptId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(self.0)
    }
}

impl fmt::Display for SurveillanceScreenReceiptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        put_hex(f, self.0)
    }
}

fn hex(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn put_hex(f: &mut fmt::Formatter<'_>, bytes: [u8; 32]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

fn put_usize(h: &mut Sha256, value: usize) {
    let value = u64::try_from(value).expect("supported Rust targets fit usize into u64");
    h.update(value.to_be_bytes());
}

fn put_point(h: &mut Sha256, point: SurveillancePoint) {
    h.update(point.observed_at_unix_s().to_be_bytes());
    match point.measurement() {
        Some(measurement) => {
            h.update([1]);
            put_interval(h, measurement);
        }
        None => h.update([0]),
    }
}

fn put_interval(h: &mut Sha256, measurement: IntervalEstimate) {
    put_f64(h, measurement.estimate());
    put_f64(h, measurement.lower());
    put_f64(h, measurement.upper());
}

fn put_f64(h: &mut Sha256, value: f64) {
    let bits = if value == 0.0 {
        0.0f64.to_bits()
    } else if value.is_nan() {
        0x7ff8_0000_0000_0000u64
    } else {
        value.to_bits()
    };
    h.update(bits.to_be_bytes());
}

fn put_string(h: &mut Sha256, value: &str) {
    put_usize(h, value.len());
    h.update(value.as_bytes());
}

fn put_config(h: &mut Sha256, config: SurveillanceScreenConfig) {
    put_usize(h, config.min_baseline_observations());
    put_f64(h, config.robust_z_threshold());
    put_f64(h, config.scale_epsilon());
}

fn put_baseline_window(h: &mut Sha256, window: Option<BaselineTimeWindow>) {
    match window {
        Some(window) => {
            h.update([1]);
            h.update(window.start_unix_s.to_be_bytes());
            h.update(window.end_unix_s.to_be_bytes());
        }
        None => h.update([0]),
    }
}

fn put_disposition(h: &mut Sha256, disposition: ScreeningDisposition) {
    match disposition {
        ScreeningDisposition::InsufficientBaseline => h.update([0]),
        ScreeningDisposition::WithinBaseline => h.update([1]),
        ScreeningDisposition::ChangeCandidate(direction) => {
            h.update([2]);
            h.update([match direction {
                ChangeDirection::Upward => 0,
                ChangeDirection::Downward => 1,
            }]);
        }
        ScreeningDisposition::Abstain(reason) => {
            h.update([3]);
            h.update([match reason {
                AbstentionReason::LatestMeasurementMissing => 0,
                AbstentionReason::DegenerateBaselineSpread => 1,
                AbstentionReason::UncertaintyOverlapsThresholdEnvelope => 2,
            }]);
        }
    }
}

fn put_robust_baseline(h: &mut Sha256, baseline: Option<RobustBaseline>) {
    match baseline {
        Some(baseline) => {
            h.update([1]);
            put_f64(h, baseline.center);
            put_f64(h, baseline.mad);
            put_f64(h, baseline.robust_scale);
            put_f64(h, baseline.lower_threshold_envelope);
            put_f64(h, baseline.upper_threshold_envelope);
        }
        None => h.update([0]),
    }
}

fn put_optional_f64(h: &mut Sha256, value: Option<f64>) {
    match value {
        Some(value) => {
            h.update([1]);
            put_f64(h, value);
        }
        None => h.update([0]),
    }
}

fn put_optional_interval(h: &mut Sha256, value: Option<IntervalEstimate>) {
    match value {
        Some(value) => {
            h.update([1]);
            put_interval(h, value);
        }
        None => h.update([0]),
    }
}

fn put_assessment(h: &mut Sha256, assessment: SurveillanceAssessment) {
    put_disposition(h, assessment.disposition);
    put_robust_baseline(h, assessment.baseline);
    put_usize(h, assessment.baseline_observed);
    put_usize(h, assessment.baseline_missing);
    put_optional_f64(h, assessment.robust_z);
    put_optional_interval(h, assessment.latest);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaselineTimeWindow {
    pub start_unix_s: i64,
    pub end_unix_s: i64,
}

/// Evidence-bearing result of one validated invocation of the v1 screen.
///
/// Fields are private so external callers cannot manufacture a receipt with an
/// arbitrary assessment/configuration pair. Receipts are created by
/// [`assess_latest_change_with_receipt`] and exposed through read-only accessors.
/// This is an implementation-integrity boundary only; it does not authenticate
/// the upstream measurements or their institutional provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurveillanceScreenReceipt {
    algorithm: SurveillanceScreenAlgorithm,
    input_id: SurveillanceScreenInputId,
    config: SurveillanceScreenConfig,
    baseline_window: Option<BaselineTimeWindow>,
    latest_observed_at_unix_s: i64,
    assessment: SurveillanceAssessment,
}

impl SurveillanceScreenReceipt {
    pub const fn algorithm(self) -> SurveillanceScreenAlgorithm {
        self.algorithm
    }

    pub const fn algorithm_id(self) -> &'static str {
        self.algorithm.semantic_id()
    }

    pub const fn input_id(self) -> SurveillanceScreenInputId {
        self.input_id
    }

    pub const fn config(self) -> SurveillanceScreenConfig {
        self.config
    }

    pub const fn baseline_window(self) -> Option<BaselineTimeWindow> {
        self.baseline_window
    }

    pub const fn latest_observed_at_unix_s(self) -> i64 {
        self.latest_observed_at_unix_s
    }

    pub const fn assessment(self) -> SurveillanceAssessment {
        self.assessment
    }

    /// Return the semantic content identity of this complete receipt.
    ///
    /// The ID is intentionally independent of Rust Debug output, serde formats,
    /// pointer layout, and machine endianness. Fixed tags, big-endian integers,
    /// and canonical IEEE-754 bit encodings define the v1 identity contract.
    pub fn id(&self) -> SurveillanceScreenReceiptId {
        let mut h = Sha256::new();
        h.update(SURVEILLANCE_SCREEN_RECEIPT_ID_DOMAIN_V1);
        put_string(&mut h, self.algorithm.semantic_id());
        h.update(self.input_id.as_bytes());
        put_config(&mut h, self.config);
        put_baseline_window(&mut h, self.baseline_window);
        h.update(self.latest_observed_at_unix_s.to_be_bytes());
        put_assessment(&mut h, self.assessment);
        SurveillanceScreenReceiptId(h.finalize().into())
    }
}

/// Preferred evidence-bearing entry point for the v1 screen.
///
/// This function does not add authority or epistemic strength. It simply makes
/// the exact input content, algorithm, configuration, time context, and returned
/// assessment travel together for downstream audit and reproducibility.
pub fn assess_latest_change_with_receipt(
    baseline_points: &[SurveillancePoint],
    latest: SurveillancePoint,
    config: SurveillanceScreenConfig,
) -> Result<SurveillanceScreenReceipt, SurveillanceScreenError> {
    let input_id = SurveillanceScreenInputId::from_series(baseline_points, latest);
    let baseline_window =
        baseline_points
            .first()
            .zip(baseline_points.last())
            .map(|(first, last)| BaselineTimeWindow {
                start_unix_s: first.observed_at_unix_s(),
                end_unix_s: last.observed_at_unix_s(),
            });
    let latest_observed_at_unix_s = latest.observed_at_unix_s();
    let assessment = assess_latest_change(baseline_points, latest, config)?;

    Ok(SurveillanceScreenReceipt {
        algorithm: SurveillanceScreenAlgorithm::RobustMedianMadIntervalGuardV1,
        input_id,
        config,
        baseline_window,
        latest_observed_at_unix_s,
        assessment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> [SurveillancePoint; 5] {
        [
            SurveillancePoint::observed(10, 8.0, 7.9, 8.1).unwrap(),
            SurveillancePoint::observed(20, 9.0, 8.9, 9.1).unwrap(),
            SurveillancePoint::observed(30, 10.0, 9.9, 10.1).unwrap(),
            SurveillancePoint::observed(40, 11.0, 10.9, 11.1).unwrap(),
            SurveillancePoint::observed(50, 12.0, 11.9, 12.1).unwrap(),
        ]
    }

    #[test]
    fn receipt_binds_algorithm_config_time_scope_and_exact_input() {
        let history = history();
        let latest = SurveillancePoint::observed(60, 20.0, 19.0, 21.0).unwrap();
        let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();

        let receipt = assess_latest_change_with_receipt(&history, latest, config).unwrap();
        assert_eq!(
            receipt.algorithm,
            SurveillanceScreenAlgorithm::RobustMedianMadIntervalGuardV1
        );
        assert_eq!(receipt.algorithm_id(), SURVEILLANCE_SCREEN_ALGORITHM_V1);
        assert_eq!(
            receipt.input_id,
            SurveillanceScreenInputId::from_series(&history, latest)
        );
        assert_eq!(
            receipt.input_id.to_hex(),
            "31281d529d4fb72626e0f2af0b3874a380f055f7baa57be9c2858c3a3e90d7c6"
        );
        assert_eq!(
            receipt.id().to_hex(),
            "9ab80879d2903714349612bd8faf7f154bf521f7d059290f4a455cb50ee69aaf"
        );
        assert_eq!(receipt.config, config);
        assert_eq!(
            receipt.baseline_window,
            Some(BaselineTimeWindow {
                start_unix_s: 10,
                end_unix_s: 50,
            })
        );
        assert_eq!(receipt.latest_observed_at_unix_s, 60);
        assert_eq!(
            receipt.assessment.disposition,
            ScreeningDisposition::ChangeCandidate(ChangeDirection::Upward)
        );
    }

    #[test]
    fn identical_receipts_have_identical_receipt_identity() {
        let history = history();
        let latest = SurveillancePoint::observed(60, 20.0, 19.0, 21.0).unwrap();
        let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();
        let a = assess_latest_change_with_receipt(&history, latest, config).unwrap();
        let b = assess_latest_change_with_receipt(&history, latest, config).unwrap();

        assert_eq!(a, b);
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn scale_epsilon_is_receipt_identity_significant() {
        let history = history();
        let latest = SurveillancePoint::observed(60, 11.0, 10.9, 11.1).unwrap();
        let a_config = SurveillanceScreenConfig::with_scale_epsilon(5, 3.0, 1e-12).unwrap();
        let b_config = SurveillanceScreenConfig::with_scale_epsilon(5, 3.0, 1e-6).unwrap();
        let a = assess_latest_change_with_receipt(&history, latest, a_config).unwrap();
        let b = assess_latest_change_with_receipt(&history, latest, b_config).unwrap();

        assert_eq!(a.input_id, b.input_id);
        assert_eq!(a.assessment, b.assessment);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn returned_assessment_is_receipt_identity_significant() {
        let history = history();
        let latest = SurveillancePoint::observed(60, 20.0, 19.0, 21.0).unwrap();
        let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();
        let receipt = assess_latest_change_with_receipt(&history, latest, config).unwrap();
        let mut altered = receipt;
        altered.assessment.disposition = ScreeningDisposition::WithinBaseline;

        assert_eq!(receipt.input_id, altered.input_id);
        assert_eq!(receipt.config, altered.config);
        assert_ne!(receipt.id(), altered.id());
    }

    #[test]
    fn typed_algorithm_identity_is_hash_stable() {
        assert_eq!(
            SurveillanceScreenAlgorithm::RobustMedianMadIntervalGuardV1.semantic_id(),
            SURVEILLANCE_SCREEN_ALGORITHM_V1
        );
    }

    #[test]
    fn different_measurements_in_same_time_scope_have_different_input_identity() {
        let history_a = [
            SurveillancePoint::observed(10, 8.0, 7.9, 8.1).unwrap(),
            SurveillancePoint::observed(20, 9.0, 8.9, 9.1).unwrap(),
            SurveillancePoint::observed(30, 10.0, 9.9, 10.1).unwrap(),
        ];
        let history_b = [
            SurveillancePoint::observed(10, 8.0, 7.9, 8.1).unwrap(),
            SurveillancePoint::observed(20, 9.5, 9.4, 9.6).unwrap(),
            SurveillancePoint::observed(30, 10.0, 9.9, 10.1).unwrap(),
        ];
        let latest = SurveillancePoint::observed(40, 11.0, 10.9, 11.1).unwrap();

        assert_ne!(
            SurveillanceScreenInputId::from_series(&history_a, latest),
            SurveillanceScreenInputId::from_series(&history_b, latest)
        );
    }

    #[test]
    fn explicit_missingness_changes_input_identity() {
        let observed_history = [
            SurveillancePoint::observed(10, 8.0, 7.9, 8.1).unwrap(),
            SurveillancePoint::observed(20, 9.0, 8.9, 9.1).unwrap(),
        ];
        let missing_history = [
            SurveillancePoint::observed(10, 8.0, 7.9, 8.1).unwrap(),
            SurveillancePoint::missing(20),
        ];
        let latest = SurveillancePoint::observed(30, 10.0, 9.9, 10.1).unwrap();

        assert_ne!(
            SurveillanceScreenInputId::from_series(&observed_history, latest),
            SurveillanceScreenInputId::from_series(&missing_history, latest)
        );
    }

    #[test]
    fn negative_zero_is_canonicalized_in_input_identity() {
        let negative = [SurveillancePoint::observed(10, -0.0, -0.0, -0.0).unwrap()];
        let positive = [SurveillancePoint::observed(10, 0.0, 0.0, 0.0).unwrap()];
        let latest_negative = SurveillancePoint::observed(20, -0.0, -0.0, -0.0).unwrap();
        let latest_positive = SurveillancePoint::observed(20, 0.0, 0.0, 0.0).unwrap();

        assert_eq!(
            SurveillanceScreenInputId::from_series(&negative, latest_negative),
            SurveillanceScreenInputId::from_series(&positive, latest_positive)
        );
    }

    #[test]
    fn empty_baseline_still_records_latest_time_and_exact_input() {
        let latest = SurveillancePoint::observed(60, 20.0, 19.0, 21.0).unwrap();
        let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();
        let receipt = assess_latest_change_with_receipt(&[], latest, config).unwrap();

        assert_eq!(
            receipt.input_id,
            SurveillanceScreenInputId::from_series(&[], latest)
        );
        assert_eq!(receipt.baseline_window, None);
        assert_eq!(receipt.latest_observed_at_unix_s, 60);
        assert_eq!(receipt.config, config);
        assert_eq!(
            receipt.assessment.disposition,
            ScreeningDisposition::InsufficientBaseline
        );
    }
}
