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
    SurveillanceAssessment, SurveillancePoint, SurveillanceScreenConfig, SurveillanceScreenError,
    assess_latest_change,
};

pub const SURVEILLANCE_SCREEN_ALGORITHM_V1: &str = "robust-median-mad-interval-guard-v1";
pub const SURVEILLANCE_SCREEN_INPUT_ID_DOMAIN_V1: &[u8] =
    b"symthaea-epidemiology-surveillance-screen-input-v1\0";

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
        h.update((baseline_points.len() as u64).to_be_bytes());
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
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

impl fmt::Display for SurveillanceScreenInputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn put_point(h: &mut Sha256, point: SurveillancePoint) {
    h.update(point.observed_at_unix_s().to_be_bytes());
    match point.measurement() {
        Some(measurement) => {
            h.update([1]);
            put_f64(h, measurement.estimate());
            put_f64(h, measurement.lower());
            put_f64(h, measurement.upper());
        }
        None => h.update([0]),
    }
}

fn put_f64(h: &mut Sha256, value: f64) {
    let bits = if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    };
    h.update(bits.to_be_bytes());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaselineTimeWindow {
    pub start_unix_s: i64,
    pub end_unix_s: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurveillanceScreenReceipt {
    /// Stable semantic identifier for the screening algorithm, not a software
    /// build/version claim.
    pub algorithm_id: &'static str,
    /// Content commitment over the complete ordered baseline + latest point.
    /// This prevents two different series with the same time extent/result from
    /// becoming indistinguishable in downstream audit records.
    pub input_id: SurveillanceScreenInputId,
    /// Exact caller-supplied screening configuration used for this result.
    pub config: SurveillanceScreenConfig,
    /// Time extent of the supplied historical series, including explicit
    /// missing observations. `None` only when the baseline slice is empty.
    pub baseline_window: Option<BaselineTimeWindow>,
    pub latest_observed_at_unix_s: i64,
    pub assessment: SurveillanceAssessment,
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
        algorithm_id: SURVEILLANCE_SCREEN_ALGORITHM_V1,
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
    use crate::surveillance::{ChangeDirection, ScreeningDisposition};

    #[test]
    fn receipt_binds_algorithm_config_time_scope_and_exact_input() {
        let history = [
            SurveillancePoint::observed(10, 8.0, 7.9, 8.1).unwrap(),
            SurveillancePoint::observed(20, 9.0, 8.9, 9.1).unwrap(),
            SurveillancePoint::observed(30, 10.0, 9.9, 10.1).unwrap(),
            SurveillancePoint::observed(40, 11.0, 10.9, 11.1).unwrap(),
            SurveillancePoint::observed(50, 12.0, 11.9, 12.1).unwrap(),
        ];
        let latest = SurveillancePoint::observed(60, 20.0, 19.0, 21.0).unwrap();
        let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();

        let receipt = assess_latest_change_with_receipt(&history, latest, config).unwrap();
        assert_eq!(receipt.algorithm_id, SURVEILLANCE_SCREEN_ALGORITHM_V1);
        assert_eq!(
            receipt.input_id,
            SurveillanceScreenInputId::from_series(&history, latest)
        );
        assert_eq!(receipt.input_id.to_hex().len(), 64);
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
