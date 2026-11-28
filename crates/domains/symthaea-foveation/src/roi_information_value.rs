// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Expected value of information (EVI) ranking for already-authorized ROIs.
//!
//! This module ranks *processing candidates*. It is not a permission system:
//! camera/sensor capture, privacy, retention and model-inference authorization
//! must already have been established by upstream policy. A high information
//! score can never override those hard gates.
//!
//! The estimator consumes calibrated expected decision-loss estimates rather
//! than raw surprise or uncalibrated confidence. It reports value per 1,000
//! source pixels so candidate selection can be compared under matched compute.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct RoiCalibrationEvidence {
    pub calibration_id: String,
    pub model_digest: String,
    pub task_family: String,
    pub sample_count: u64,
    /// Expected calibration error on a held-out calibration report, [0, 1].
    pub expected_calibration_error: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiValueEstimate {
    pub candidate_id: String,
    pub task_id: String,
    pub source_clock_domain_id: String,
    pub source_frame_id: u64,
    pub source_timestamp_us: u64,
    /// Expected normalized task decision loss without this ROI, [0, 1].
    pub expected_loss_without_roi: f32,
    /// Expected normalized task decision loss after using this ROI, [0, 1].
    pub expected_loss_with_roi: f32,
    /// Expected normalized loss from waiting for this ROI result, [0, 1].
    pub latency_penalty: f32,
    /// Expected normalized loss from evidence becoming stale, [0, 1].
    pub staleness_penalty: f32,
    /// Source crop pixels before resizing/downscaling.
    pub source_pixel_cost: u64,
    pub calibration: RoiCalibrationEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoiInformationValueConfig {
    pub maximum_source_age_us: u64,
    pub minimum_calibration_samples: u64,
    pub maximum_expected_calibration_error: f32,
    pub minimum_net_benefit: f32,
    pub minimum_benefit_per_1000_pixels: f32,
}

impl Default for RoiInformationValueConfig {
    fn default() -> Self {
        Self {
            maximum_source_age_us: 500_000,
            minimum_calibration_samples: 100,
            maximum_expected_calibration_error: 0.10,
            minimum_net_benefit: 0.01,
            minimum_benefit_per_1000_pixels: 0.001,
        }
    }
}

impl RoiInformationValueConfig {
    pub fn validate(&self) -> Result<(), RoiInformationValueError> {
        if self.maximum_source_age_us == 0
            || self.minimum_calibration_samples == 0
            || !self.maximum_expected_calibration_error.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_expected_calibration_error)
            || !self.minimum_net_benefit.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_net_benefit)
            || !self.minimum_benefit_per_1000_pixels.is_finite()
            || self.minimum_benefit_per_1000_pixels < 0.0
        {
            return Err(RoiInformationValueError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiInformationValueScore {
    pub candidate_id: String,
    pub task_id: String,
    pub source_frame_id: u64,
    pub calibration_id: String,
    pub expected_loss_reduction: f32,
    pub net_benefit: f32,
    pub benefit_per_1000_pixels: f32,
    pub source_pixel_cost: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoiInformationValueError {
    InvalidConfig,
    InvalidIdentifier,
    InvalidDigest,
    InvalidEstimate,
    ZeroPixelCost,
    FutureSourceTimestamp,
    StaleSourceFrame,
    ClockDomainMismatch,
    InsufficientCalibrationSamples,
    CalibrationErrorTooHigh,
    NoExpectedLossReduction,
    BelowMinimumNetBenefit,
    BelowMinimumValuePerPixel,
    DuplicateCandidateId,
}

/// Evaluate one ROI candidate against a calibration-backed decision-loss
/// estimate. The input timestamps use the same monotonic clock domain as
/// `now_us`; no cross-clock conversion is inferred here.
pub fn evaluate_roi_information_value(
    estimate: &RoiValueEstimate,
    now_us: u64,
    now_clock_domain_id: &str,
    config: RoiInformationValueConfig,
) -> Result<RoiInformationValueScore, RoiInformationValueError> {
    config.validate()?;
    validate_identifier(now_clock_domain_id)?;
    if estimate.source_clock_domain_id != now_clock_domain_id {
        return Err(RoiInformationValueError::ClockDomainMismatch);
    }
    validate_identifier(&estimate.candidate_id)?;
    validate_identifier(&estimate.task_id)?;
    validate_identifier(&estimate.source_clock_domain_id)?;
    validate_identifier(&estimate.calibration.calibration_id)?;
    validate_identifier(&estimate.calibration.task_family)?;
    validate_digest(&estimate.calibration.model_digest)?;

    if estimate.source_timestamp_us > now_us {
        return Err(RoiInformationValueError::FutureSourceTimestamp);
    }
    let age_us = now_us - estimate.source_timestamp_us;
    if age_us > config.maximum_source_age_us {
        return Err(RoiInformationValueError::StaleSourceFrame);
    }
    if estimate.source_pixel_cost == 0 {
        return Err(RoiInformationValueError::ZeroPixelCost);
    }
    if estimate.calibration.sample_count < config.minimum_calibration_samples {
        return Err(RoiInformationValueError::InsufficientCalibrationSamples);
    }
    if !estimate.calibration.expected_calibration_error.is_finite()
        || !(0.0..=config.maximum_expected_calibration_error)
            .contains(&estimate.calibration.expected_calibration_error)
    {
        return Err(RoiInformationValueError::CalibrationErrorTooHigh);
    }

    for value in [
        estimate.expected_loss_without_roi,
        estimate.expected_loss_with_roi,
        estimate.latency_penalty,
        estimate.staleness_penalty,
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(RoiInformationValueError::InvalidEstimate);
        }
    }

    let expected_loss_reduction =
        estimate.expected_loss_without_roi - estimate.expected_loss_with_roi;
    if expected_loss_reduction <= 0.0 {
        return Err(RoiInformationValueError::NoExpectedLossReduction);
    }
    let net_benefit =
        expected_loss_reduction - estimate.latency_penalty - estimate.staleness_penalty;
    if net_benefit < config.minimum_net_benefit {
        return Err(RoiInformationValueError::BelowMinimumNetBenefit);
    }
    let benefit_per_1000_pixels = net_benefit * 1_000.0 / estimate.source_pixel_cost as f32;
    if benefit_per_1000_pixels < config.minimum_benefit_per_1000_pixels {
        return Err(RoiInformationValueError::BelowMinimumValuePerPixel);
    }

    Ok(RoiInformationValueScore {
        candidate_id: estimate.candidate_id.clone(),
        task_id: estimate.task_id.clone(),
        source_frame_id: estimate.source_frame_id,
        calibration_id: estimate.calibration.calibration_id.clone(),
        expected_loss_reduction,
        net_benefit,
        benefit_per_1000_pixels,
        source_pixel_cost: estimate.source_pixel_cost,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiCandidateRejection {
    pub candidate_id: String,
    pub reason: RoiInformationValueError,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiInformationRanking {
    /// Highest benefit per 1,000 source pixels first. Candidate ID breaks ties.
    pub ranked: Vec<RoiInformationValueScore>,
    /// Rejections are retained rather than silently omitted.
    pub rejected: Vec<RoiCandidateRejection>,
}

pub fn rank_roi_information_value(
    estimates: &[RoiValueEstimate],
    now_us: u64,
    now_clock_domain_id: &str,
    config: RoiInformationValueConfig,
) -> Result<RoiInformationRanking, RoiInformationValueError> {
    config.validate()?;
    validate_identifier(now_clock_domain_id)?;
    let mut counts = std::collections::HashMap::with_capacity(estimates.len());
    for estimate in estimates {
        *counts.entry(estimate.candidate_id.as_str()).or_insert(0usize) += 1;
    }
    let mut seen = HashSet::with_capacity(estimates.len());
    let mut ranked = Vec::with_capacity(estimates.len());
    let mut rejected = Vec::new();

    for estimate in estimates {
        if !seen.insert(estimate.candidate_id.as_str()) {
            continue;
        }
        if counts.get(estimate.candidate_id.as_str()).copied().unwrap_or(0) > 1 {
            for _ in 0..counts[estimate.candidate_id.as_str()] {
                rejected.push(RoiCandidateRejection {
                    candidate_id: estimate.candidate_id.clone(),
                    reason: RoiInformationValueError::DuplicateCandidateId,
                });
            }
            continue;
        }
        match evaluate_roi_information_value(estimate, now_us, now_clock_domain_id, config) {
            Ok(score) => ranked.push(score),
            Err(reason) => rejected.push(RoiCandidateRejection {
                candidate_id: estimate.candidate_id.clone(),
                reason,
            }),
        }
    }

    ranked.sort_by(|left, right| {
        right
            .benefit_per_1000_pixels
            .total_cmp(&left.benefit_per_1000_pixels)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    rejected.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    Ok(RoiInformationRanking { ranked, rejected })
}

fn validate_identifier(value: &str) -> Result<(), RoiInformationValueError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(RoiInformationValueError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), RoiInformationValueError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RoiInformationValueError::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(id: &str, loss_without: f32, loss_with: f32, pixels: u64) -> RoiValueEstimate {
        RoiValueEstimate {
            candidate_id: id.to_owned(),
            task_id: "turn-boundary".to_owned(),
            source_clock_domain_id: "clock:camera-1".to_owned(),
            source_frame_id: 7,
            source_timestamp_us: 1_000,
            expected_loss_without_roi: loss_without,
            expected_loss_with_roi: loss_with,
            latency_penalty: 0.02,
            staleness_penalty: 0.01,
            source_pixel_cost: pixels,
            calibration: RoiCalibrationEvidence {
                calibration_id: "turn-cal-v1".to_owned(),
                model_digest: "a".repeat(64),
                task_family: "turn_floor_state".to_owned(),
                sample_count: 500,
                expected_calibration_error: 0.04,
            },
        }
    }

    #[test]
    fn expected_loss_reduction_minus_costs_produces_net_value() {
        let score = evaluate_roi_information_value(
            &estimate("roi-1", 0.8, 0.5, 10_000),
            1_100,
            "clock:camera-1",
            RoiInformationValueConfig::default(),
        ).unwrap();
        assert!((score.expected_loss_reduction - 0.3).abs() < 1e-6);
        assert!((score.net_benefit - 0.27).abs() < 1e-6);
        assert!((score.benefit_per_1000_pixels - 0.027).abs() < 1e-6);
    }

    #[test]
    fn candidates_rank_by_expected_net_benefit_per_pixel() {
        let candidates = vec![
            estimate("roi-low", 0.6, 0.4, 50_000),
            estimate("roi-high", 0.9, 0.5, 10_000),
        ];
        let ranking = rank_roi_information_value(
            &candidates,
            1_100,
            "clock:camera-1",
            RoiInformationValueConfig::default(),
        ).unwrap();
        assert_eq!(ranking.ranked[0].candidate_id, "roi-high");
        assert_eq!(ranking.ranked[1].candidate_id, "roi-low");
        assert!(ranking.rejected.is_empty());
    }

    #[test]
    fn low_calibration_sample_count_is_rejected() {
        let mut value = estimate("roi-1", 0.8, 0.5, 10_000);
        value.calibration.sample_count = 20;
        assert_eq!(
            evaluate_roi_information_value(&value, 1_100, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::InsufficientCalibrationSamples)
        );
    }

    #[test]
    fn high_calibration_error_is_rejected() {
        let mut value = estimate("roi-1", 0.8, 0.5, 10_000);
        value.calibration.expected_calibration_error = 0.2;
        assert_eq!(
            evaluate_roi_information_value(&value, 1_100, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::CalibrationErrorTooHigh)
        );
    }

    #[test]
    fn zero_or_negative_loss_reduction_is_rejected() {
        let value = estimate("roi-1", 0.4, 0.5, 10_000);
        assert_eq!(
            evaluate_roi_information_value(&value, 1_100, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::NoExpectedLossReduction)
        );
    }

    #[test]
    fn latency_and_staleness_costs_can_make_an_roi_not_worth_sampling() {
        let mut value = estimate("roi-1", 0.5, 0.4, 10_000);
        value.latency_penalty = 0.08;
        value.staleness_penalty = 0.04;
        assert_eq!(
            evaluate_roi_information_value(&value, 1_100, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::BelowMinimumNetBenefit)
        );
    }

    #[test]
    fn source_frame_future_and_staleness_are_rejected() {
        let value = estimate("roi-1", 0.8, 0.5, 10_000);
        assert_eq!(
            evaluate_roi_information_value(&value, 999, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::FutureSourceTimestamp)
        );
        assert_eq!(
            evaluate_roi_information_value(&value, 600_001, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::StaleSourceFrame)
        );
    }

    #[test]
    fn ranking_retains_rejections_instead_of_silently_dropping_them() {
        let good = estimate("roi-good", 0.8, 0.5, 10_000);
        let bad = estimate("roi-bad", 0.4, 0.5, 10_000);
        let ranking = rank_roi_information_value(
            &[good, bad],
            1_100,
            "clock:camera-1",
            RoiInformationValueConfig::default(),
        ).unwrap();
        assert_eq!(ranking.ranked.len(), 1);
        assert_eq!(ranking.rejected.len(), 1);
        assert_eq!(ranking.rejected[0].reason, RoiInformationValueError::NoExpectedLossReduction);
    }

    #[test]
    fn duplicate_candidate_ids_are_explicitly_rejected() {
        let ranking = rank_roi_information_value(
            &[estimate("roi-1", 0.8, 0.5, 10_000), estimate("roi-1", 0.7, 0.5, 10_000)],
            1_100,
            "clock:camera-1",
            RoiInformationValueConfig::default(),
        ).unwrap();
        assert!(ranking.ranked.is_empty());
        assert_eq!(ranking.rejected.len(), 2);
        assert!(ranking.rejected.iter().all(|item| item.reason == RoiInformationValueError::DuplicateCandidateId));
    }

    #[test]
    fn equal_scores_use_stable_candidate_id_tie_break() {
        let ranking = rank_roi_information_value(
            &[estimate("roi-b", 0.8, 0.5, 10_000), estimate("roi-a", 0.8, 0.5, 10_000)],
            1_100,
            "clock:camera-1",
            RoiInformationValueConfig::default(),
        ).unwrap();
        assert_eq!(ranking.ranked[0].candidate_id, "roi-a");
        assert_eq!(ranking.ranked[1].candidate_id, "roi-b");
    }

    #[test]
    fn cross_clock_value_estimates_are_rejected() {
        let value = estimate("roi-1", 0.8, 0.5, 10_000);
        assert_eq!(
            evaluate_roi_information_value(
                &value,
                1_100,
                "clock:other-camera",
                RoiInformationValueConfig::default()
            ),
            Err(RoiInformationValueError::ClockDomainMismatch)
        );
    }

    #[test]
    fn non_finite_estimates_fail_closed() {
        let mut value = estimate("roi-1", 0.8, 0.5, 10_000);
        value.latency_penalty = f32::NAN;
        assert_eq!(
            evaluate_roi_information_value(&value, 1_100, "clock:camera-1", RoiInformationValueConfig::default()),
            Err(RoiInformationValueError::InvalidEstimate)
        );
    }
}
