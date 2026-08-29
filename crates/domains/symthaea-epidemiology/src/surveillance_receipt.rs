// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Audit receipt for the conservative aggregate surveillance screen.
//!
//! The underlying screening function intentionally stays small. This wrapper
//! binds its output to the exact algorithm identifier, caller-supplied
//! configuration, and time scope so a later evidence plane can preserve what was
//! actually evaluated instead of retaining only a disposition label.

use crate::surveillance::{
    SurveillanceAssessment, SurveillancePoint, SurveillanceScreenConfig, SurveillanceScreenError,
    assess_latest_change,
};

pub const SURVEILLANCE_SCREEN_ALGORITHM_V1: &str = "robust-median-mad-interval-guard-v1";

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
/// the algorithm/configuration/time context inseparable from the returned
/// assessment for downstream audit and reproducibility.
pub fn assess_latest_change_with_receipt(
    baseline_points: &[SurveillancePoint],
    latest: SurveillancePoint,
    config: SurveillanceScreenConfig,
) -> Result<SurveillanceScreenReceipt, SurveillanceScreenError> {
    let baseline_window = baseline_points
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
    fn receipt_binds_algorithm_config_and_time_scope() {
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
    fn empty_baseline_still_records_latest_time_and_exact_config() {
        let latest = SurveillancePoint::observed(60, 20.0, 19.0, 21.0).unwrap();
        let config = SurveillanceScreenConfig::new(5, 3.0).unwrap();
        let receipt = assess_latest_change_with_receipt(&[], latest, config).unwrap();

        assert_eq!(receipt.baseline_window, None);
        assert_eq!(receipt.latest_observed_at_unix_s, 60);
        assert_eq!(receipt.config, config);
        assert_eq!(
            receipt.assessment.disposition,
            ScreeningDisposition::InsufficientBaseline
        );
    }
}
