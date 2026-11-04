// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replayable mapping between a runner monotonic clock and one CPAL stream clock.
//!
//! This module is dependency-light and contains no CPAL types. The live callback
//! adapter records a bounded pair around the CPAL callback and predicted-playback timestamps read:
//! runner time immediately before and after observing the CPAL stream timestamp.
//! The pair is a software observation bracket, not a guarantee about backend
//! callback-start accuracy.
//!
//! A low regression residual establishes consistency of the supplied pairs.
//! It does not establish absolute clock accuracy. V1 therefore reports only a
//! point estimate and must not be promoted to a physical output-latency claim.
//!
//! The resulting mapping can be composed with the CPAL playback-time -> output
//! frame fit from MEL-CLOCK-001 only when session, runner origin, stream
//! identity, descriptor digest, validity interval and uncertainty status all
//! match. Since uncertainty propagation is not yet qualified, such a
//! composition remains non-qualified.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub const RUNNER_CLOCK_BRIDGE_DESCRIPTOR_SCHEMA_V1: &str =
    "mel-runner-cpal-clock-descriptor-v1";
pub const RUNNER_CPAL_CLOCK_PAIR_SCHEMA_V1: &str = "mel-runner-cpal-clock-pair-v1";
pub const RUNNER_CPAL_CLOCK_FIT_SCHEMA_V1: &str = "mel-runner-cpal-clock-fit-v1";
pub const RUNNER_CPAL_CLOCK_BUNDLE_SCHEMA_V1: &str =
    "mel-runner-cpal-clock-fit-bundle-v1";

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Copyable process-local origin for runner monotonic timestamps.
///
/// Construct this once for a session and share the same value with the
/// turn-taking/effect logger and the CPAL callback. It is intentionally not
/// serializable: only relative integer nanoseconds are persisted. The
/// descriptor's `runner_origin_id` identifies this exact origin in receipts.
#[derive(Debug, Clone, Copy)]
pub struct RunnerMonotonicClockOriginV1 {
    origin: Instant,
}

impl RunnerMonotonicClockOriginV1 {
    pub fn new() -> Self {
        Self { origin: Instant::now() }
    }

    pub fn now_ns(&self) -> Option<u64> {
        self.elapsed_ns_at(Instant::now())
    }

    pub fn elapsed_ns_at(&self, instant: Instant) -> Option<u64> {
        instant
            .checked_duration_since(self.origin)
            .and_then(|elapsed| u64::try_from(elapsed.as_nanos()).ok())
    }
}

impl Default for RunnerMonotonicClockOriginV1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Identity of the two clock domains being bridged for one live session.
///
/// `runner_origin_id` must identify the exact monotonic origin used by the
/// runner timestamps in the turn-taking/effect receipts. A clock-domain name
/// alone is insufficient if a process or session can restart and reset its
/// relative time origin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerCpalClockDescriptorV1 {
    pub schema_version: String,
    pub session_id: String,
    pub runner_clock_domain_id: String,
    pub runner_origin_id: String,
    pub stream_instance_id: u64,
    /// SHA-256 digest of the exact CPAL stream descriptor used by the
    /// output-clock fit. It is a join key, not a device-anonymization claim.
    pub cpal_stream_descriptor_sha256: String,
}

impl RunnerCpalClockDescriptorV1 {
    /// Stable digest carried by each fixed-size callback record so an SPSC
    /// stream cannot be accidentally replayed under another session/origin.
    pub fn identity_sha256(&self) -> Option<[u8; 32]> {
        if !self.is_valid() {
            return None;
        }
        let canonical = format!(
            "{RUNNER_CLOCK_BRIDGE_DESCRIPTOR_SCHEMA_V1}\n{}\n{}\n{}\n{}\n{}",
            self.session_id,
            self.runner_clock_domain_id,
            self.runner_origin_id,
            self.stream_instance_id,
            self.cpal_stream_descriptor_sha256
        );
        Some(Sha256::digest(canonical.as_bytes()).into())
    }

    pub fn is_valid(&self) -> bool {
        self.schema_version == RUNNER_CLOCK_BRIDGE_DESCRIPTOR_SCHEMA_V1
            && !self.session_id.trim().is_empty()
            && !self.runner_clock_domain_id.trim().is_empty()
            && !self.runner_origin_id.trim().is_empty()
            && self.stream_instance_id > 0
            && is_lower_hex_digest(&self.cpal_stream_descriptor_sha256)
    }
}

/// Paired observation acquired during one CPAL output callback.
///
/// `runner_time_before_ns` and `runner_time_after_ns` bracket the read of
/// CPAL's callback timestamp using the exact monotonic origin named by the
/// descriptor. The midpoint is used as the OLS observation; the full bracket
/// is retained and checked. It is not a certified uncertainty interval for
/// the backend timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunnerCpalClockPairSchemaV1 {
    #[serde(rename = "mel-runner-cpal-clock-pair-v1")]
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerCpalClockPairV1 {
    pub schema_version: RunnerCpalClockPairSchemaV1,
    pub stream_instance_id: u64,
    /// Fixed-size identity digest computed off the real-time thread from the
    /// bridge descriptor. This keeps the pair Copy and SPSC-safe.
    pub descriptor_identity_sha256: [u8; 32],
    pub pair_index: u64,
    /// CPAL OutputStreamTimestamp.callback normalized to this stream's
    /// callback-time origin, in nanoseconds.
    pub cpal_callback_stream_time_ns: u64,
    /// CPAL OutputStreamTimestamp.playback normalized to this stream's
    /// callback-time origin. This is the clock domain consumed by the
    /// MEL-CLOCK-001 output-frame fit.
    pub cpal_playback_stream_time_ns: u64,
    /// Runner monotonic time sampled immediately before reading/normalizing
    /// the CPAL callback and predicted-playback timestamps.
    pub runner_time_before_ns: u64,
    /// Runner monotonic time sampled immediately after reading/normalizing
    /// the CPAL callback and predicted-playback timestamps.
    pub runner_time_after_ns: u64,
}

impl RunnerCpalClockPairV1 {
    /// Construct the fixed-size callback record after the caller has sampled
    /// the runner clock before/after reading CPAL's callback timestamp. The
    /// descriptor digest is computed off the real-time thread and copied in.
    pub fn from_observation(
        stream_instance_id: u64,
        descriptor_identity_sha256: [u8; 32],
        pair_index: u64,
        cpal_callback_stream_time_ns: u64,
        cpal_playback_stream_time_ns: u64,
        runner_time_before_ns: u64,
        runner_time_after_ns: u64,
    ) -> Option<Self> {
        if stream_instance_id == 0 || runner_time_after_ns < runner_time_before_ns {
            return None;
        }
        Some(Self {
            schema_version: RunnerCpalClockPairSchemaV1::V1,
            stream_instance_id,
            descriptor_identity_sha256,
            pair_index,
            cpal_callback_stream_time_ns,
            cpal_playback_stream_time_ns,
            runner_time_before_ns,
            runner_time_after_ns,
        })
    }

    fn midpoint_runner_time_ns(&self) -> u64 {
        self.runner_time_before_ns
            + (self.runner_time_after_ns - self.runner_time_before_ns) / 2
    }

    fn bracket_width_ns(&self) -> u64 {
        self.runner_time_after_ns - self.runner_time_before_ns
    }
}

/// Frozen acceptance policy. The fit is valid only within its observed CPAL
/// interval plus the explicit extrapolation horizon.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerCpalClockFitPolicyV1 {
    pub minimum_pair_count: usize,
    pub maximum_abs_drift_ppm: f64,
    pub maximum_abs_residual_ns: f64,
    pub maximum_capture_bracket_ns: u64,
    pub maximum_extrapolation_ns: u64,
}

impl Default for RunnerCpalClockFitPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_pair_count: 4,
            maximum_abs_drift_ppm: 500.0,
            maximum_abs_residual_ns: 250_000.0,
            maximum_capture_bracket_ns: 100_000,
            maximum_extrapolation_ns: 100_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerCpalClockAccuracyStatusV1 {
    PointEstimateOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerCpalClockFitV1 {
    pub schema_version: String,
    pub session_id: String,
    pub runner_clock_domain_id: String,
    pub runner_origin_id: String,
    pub stream_instance_id: u64,
    pub cpal_stream_descriptor_sha256: String,
    /// Runner nanoseconds per CPAL callback-stream nanosecond.
    pub runner_ns_per_cpal_ns: f64,
    /// Runner time at the reference CPAL timestamp. Retained as f64 because
    /// the OLS centroid need not fall on an integer nanosecond.
    pub reference_runner_time_ns: f64,
    pub reference_cpal_playback_stream_time_ns: u64,
    pub valid_from_cpal_playback_stream_time_ns: u64,
    pub valid_until_cpal_playback_stream_time_ns: u64,
    pub valid_from_runner_time_ns: f64,
    pub valid_until_runner_time_ns: f64,
    pub maximum_extrapolation_ns: u64,
    pub minimum_required_pair_count: usize,
    pub pair_set_sha256: String,
    pub fit_policy_sha256: String,
    pub pair_count: usize,
    pub missing_pair_records_before_or_between_observations: u64,
    pub maximum_abs_drift_ppm: f64,
    pub maximum_allowed_abs_residual_ns: f64,
    pub estimated_drift_ppm: f64,
    pub median_abs_residual_ns: f64,
    pub p95_abs_residual_ns: f64,
    pub maximum_abs_residual_ns: f64,
    pub maximum_capture_bracket_ns: u64,
    pub p95_capture_bracket_ns: u64,
    pub accuracy_status: RunnerCpalClockAccuracyStatusV1,
    /// Always absent in v1. The bracket is retained as an observation
    /// diagnostic, but clock-source and regression-parameter uncertainty
    /// have not been propagated into a certified mapping interval.
    pub reported_uncertainty_ns: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpalClockEstimateV1 {
    pub cpal_playback_stream_time_ns: f64,
    /// Fit residual only; not a timestamp-accuracy guarantee.
    pub fit_residual_bound_ns: f64,
    pub accuracy_status: RunnerCpalClockAccuracyStatusV1,
    pub reported_uncertainty_ns: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunnerCpalClockFitErrorV1 {
    InvalidDescriptor,
    TooFewPairs { actual: usize, required: usize },
    InvalidPairSchema { index: usize },
    DescriptorMismatch { index: usize },
    InvalidCaptureBracket { index: usize },
    CaptureBracketOutsidePolicy { index: usize, actual_ns: u64, maximum_ns: u64 },
    NonMonotonicPairIndex { index: usize },
    NonMonotonicCpalTime { index: usize },
    NonMonotonicPlaybackTime { index: usize },
    NonMonotonicRunnerBefore { index: usize },
    NonMonotonicRunnerAfter { index: usize },
    ZeroTimeVariance,
    NonFiniteFit,
    DriftOutsidePolicy { drift_ppm: f64, maximum_abs_ppm: f64 },
    ResidualOutsidePolicy { residual_ns: f64, maximum_abs_ns: f64 },
    QueryOutsideValidity { runner_time_ns: u64 },
    SerializationFailure { reason: String },
}

impl std::fmt::Display for RunnerCpalClockFitErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RunnerCpalClockFitErrorV1 {}

impl RunnerCpalClockFitV1 {
    /// Invert the fitted mapping and estimate the CPAL stream-clock instant
    /// corresponding to a runner monotonic timestamp. A caller may then query
    /// the CPAL playback-time -> frame fit with this same stream-clock value.
    pub fn playback_time_at_runner_time(
        &self,
        runner_time_ns: u64,
    ) -> Result<CpalClockEstimateV1, RunnerCpalClockFitErrorV1> {
        let lower = self.valid_from_runner_time_ns - self.maximum_extrapolation_ns as f64;
        let upper = self.valid_until_runner_time_ns + self.maximum_extrapolation_ns as f64;
        if (runner_time_ns as f64) < lower || (runner_time_ns as f64) > upper {
            return Err(RunnerCpalClockFitErrorV1::QueryOutsideValidity {
                runner_time_ns,
            });
        }

        let cpal_time = self.reference_cpal_playback_stream_time_ns as f64
            + (runner_time_ns as f64 - self.reference_runner_time_ns)
                / self.runner_ns_per_cpal_ns;
        if !cpal_time.is_finite() || cpal_time < 0.0 {
            return Err(RunnerCpalClockFitErrorV1::NonFiniteFit);
        }

        Ok(CpalClockEstimateV1 {
            cpal_playback_stream_time_ns: cpal_time,
            fit_residual_bound_ns: self.maximum_abs_residual_ns,
            accuracy_status: self.accuracy_status,
            reported_uncertainty_ns: self.reported_uncertainty_ns,
        })
    }
}

/// Fit a deterministic affine mapping for one continuous runner/CPAL clock
/// segment. No outlier is silently dropped; suspicious observations are
/// rejected under the frozen policy.
pub fn fit_runner_cpal_clock_v1(
    descriptor: &RunnerCpalClockDescriptorV1,
    pairs: &[RunnerCpalClockPairV1],
    policy: RunnerCpalClockFitPolicyV1,
) -> Result<RunnerCpalClockFitV1, RunnerCpalClockFitErrorV1> {
    if !descriptor.is_valid() {
        return Err(RunnerCpalClockFitErrorV1::InvalidDescriptor);
    }

    let required_pairs = policy.minimum_pair_count.max(2);
    let effective_policy = RunnerCpalClockFitPolicyV1 {
        minimum_pair_count: required_pairs,
        ..policy
    };
    if pairs.len() < required_pairs {
        return Err(RunnerCpalClockFitErrorV1::TooFewPairs {
            actual: pairs.len(),
            required: required_pairs,
        });
    }
    if !policy.maximum_abs_drift_ppm.is_finite()
        || policy.maximum_abs_drift_ppm < 0.0
        || !policy.maximum_abs_residual_ns.is_finite()
        || policy.maximum_abs_residual_ns < 0.0
    {
        return Err(RunnerCpalClockFitErrorV1::NonFiniteFit);
    }

    for (index, pair) in pairs.iter().enumerate() {
        if pair.schema_version != RunnerCpalClockPairSchemaV1::V1 {
            return Err(RunnerCpalClockFitErrorV1::InvalidPairSchema { index });
        }
        if pair.stream_instance_id != descriptor.stream_instance_id
            || Some(pair.descriptor_identity_sha256) != descriptor.identity_sha256()
        {
            return Err(RunnerCpalClockFitErrorV1::DescriptorMismatch { index });
        }
        if pair.runner_time_after_ns < pair.runner_time_before_ns {
            return Err(RunnerCpalClockFitErrorV1::InvalidCaptureBracket { index });
        }
        let bracket_width = pair.bracket_width_ns();
        if bracket_width > policy.maximum_capture_bracket_ns {
            return Err(RunnerCpalClockFitErrorV1::CaptureBracketOutsidePolicy {
                index,
                actual_ns: bracket_width,
                maximum_ns: policy.maximum_capture_bracket_ns,
            });
        }
        if index > 0 {
            let previous = &pairs[index - 1];
            if pair.pair_index <= previous.pair_index {
                return Err(RunnerCpalClockFitErrorV1::NonMonotonicPairIndex { index });
            }
            if pair.cpal_callback_stream_time_ns
                <= previous.cpal_callback_stream_time_ns
            {
                return Err(RunnerCpalClockFitErrorV1::NonMonotonicCpalTime { index });
            }
            if pair.cpal_playback_stream_time_ns
                <= previous.cpal_playback_stream_time_ns
            {
                return Err(RunnerCpalClockFitErrorV1::NonMonotonicPlaybackTime { index });
            }
            if pair.runner_time_before_ns <= previous.runner_time_before_ns {
                return Err(RunnerCpalClockFitErrorV1::NonMonotonicRunnerBefore { index });
            }
            if pair.runner_time_after_ns <= previous.runner_time_after_ns {
                return Err(RunnerCpalClockFitErrorV1::NonMonotonicRunnerAfter { index });
            }
        }
    }

    let x0 = pairs[0].cpal_playback_stream_time_ns;
    let y0 = pairs[0].midpoint_runner_time_ns();
    let xs = pairs
        .iter()
        .map(|pair| (pair.cpal_playback_stream_time_ns - x0) as f64)
        .collect::<Vec<_>>();
    let ys = pairs
        .iter()
        .map(|pair| (pair.midpoint_runner_time_ns() as i128 - y0 as i128) as f64)
        .collect::<Vec<_>>();

    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let covariance = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let variance = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    if variance <= 0.0 || !variance.is_finite() {
        return Err(RunnerCpalClockFitErrorV1::ZeroTimeVariance);
    }

    let slope = covariance / variance;
    let reference_cpal_delta_ns = mean_x;
    let reference_cpal_playback_stream_time_ns =
        x0.saturating_add(reference_cpal_delta_ns.max(0.0).round() as u64);
    let rounded_reference_delta_ns =
        reference_cpal_playback_stream_time_ns.saturating_sub(x0) as f64;
    let reference_runner_time_ns =
        y0 as f64 + mean_y + slope * (rounded_reference_delta_ns - reference_cpal_delta_ns);

    if !slope.is_finite() || slope <= 0.0 || !reference_runner_time_ns.is_finite() {
        return Err(RunnerCpalClockFitErrorV1::NonFiniteFit);
    }

    let estimated_drift_ppm = (slope - 1.0) * 1_000_000.0;
    if !estimated_drift_ppm.is_finite() {
        return Err(RunnerCpalClockFitErrorV1::NonFiniteFit);
    }
    if estimated_drift_ppm.abs() > policy.maximum_abs_drift_ppm {
        return Err(RunnerCpalClockFitErrorV1::DriftOutsidePolicy {
            drift_ppm: estimated_drift_ppm,
            maximum_abs_ppm: policy.maximum_abs_drift_ppm,
        });
    }

    let mut residuals = pairs
        .iter()
        .zip(&xs)
        .map(|(pair, x)| {
            let predicted = y0 as f64 + mean_y + slope * (x - mean_x);
            (predicted - pair.midpoint_runner_time_ns() as f64).abs()
        })
        .collect::<Vec<_>>();
    if residuals.iter().any(|residual| !residual.is_finite()) {
        return Err(RunnerCpalClockFitErrorV1::NonFiniteFit);
    }
    residuals.sort_by(f64::total_cmp);
    let maximum_abs_residual_ns = *residuals.last().unwrap_or(&0.0);
    if maximum_abs_residual_ns > policy.maximum_abs_residual_ns {
        return Err(RunnerCpalClockFitErrorV1::ResidualOutsidePolicy {
            residual_ns: maximum_abs_residual_ns,
            maximum_abs_ns: policy.maximum_abs_residual_ns,
        });
    }
    let middle = residuals.len() / 2;
    let median_abs_residual_ns = if residuals.len() % 2 == 0 {
        (residuals[middle - 1] + residuals[middle]) * 0.5
    } else {
        residuals[middle]
    };
    let p95_index = ((residuals.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(residuals.len() - 1);
    let p95_abs_residual_ns = residuals[p95_index];

    let mut brackets = pairs
        .iter()
        .map(RunnerCpalClockPairV1::bracket_width_ns)
        .collect::<Vec<_>>();
    brackets.sort_unstable();
    let maximum_capture_bracket_ns = *brackets.last().unwrap_or(&0);
    let p95_bracket_index = ((brackets.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(brackets.len() - 1);
    let p95_capture_bracket_ns = brackets[p95_bracket_index];

    let pair_set_sha256 = serde_json::to_vec(pairs)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| RunnerCpalClockFitErrorV1::SerializationFailure {
            reason: error.to_string(),
        })?;
    let fit_policy_sha256 = serde_json::to_vec(&effective_policy)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| RunnerCpalClockFitErrorV1::SerializationFailure {
            reason: error.to_string(),
        })?;

    let mut missing_pair_records_before_or_between_observations = pairs[0].pair_index;
    for pair in pairs.windows(2) {
        missing_pair_records_before_or_between_observations +=
            pair[1].pair_index - pair[0].pair_index - 1;
    }

    let valid_from_cpal = x0;
    let valid_until_cpal = pairs[pairs.len() - 1].cpal_playback_stream_time_ns;
    let valid_from_runner = reference_runner_time_ns
        + slope * (valid_from_cpal as f64
            - reference_cpal_playback_stream_time_ns as f64);
    let valid_until_runner = reference_runner_time_ns
        + slope * (valid_until_cpal as f64
            - reference_cpal_playback_stream_time_ns as f64);
    if !valid_from_runner.is_finite()
        || !valid_until_runner.is_finite()
        || valid_until_runner < valid_from_runner
    {
        return Err(RunnerCpalClockFitErrorV1::NonFiniteFit);
    }

    Ok(RunnerCpalClockFitV1 {
        schema_version: RUNNER_CPAL_CLOCK_FIT_SCHEMA_V1.to_owned(),
        session_id: descriptor.session_id.clone(),
        runner_clock_domain_id: descriptor.runner_clock_domain_id.clone(),
        runner_origin_id: descriptor.runner_origin_id.clone(),
        stream_instance_id: descriptor.stream_instance_id,
        cpal_stream_descriptor_sha256: descriptor.cpal_stream_descriptor_sha256.clone(),
        runner_ns_per_cpal_ns: slope,
        reference_runner_time_ns,
        reference_cpal_playback_stream_time_ns,
        valid_from_cpal_playback_stream_time_ns: valid_from_cpal,
        valid_until_cpal_playback_stream_time_ns: valid_until_cpal,
        valid_from_runner_time_ns: valid_from_runner,
        valid_until_runner_time_ns: valid_until_runner,
        maximum_extrapolation_ns: effective_policy.maximum_extrapolation_ns,
        minimum_required_pair_count: effective_policy.minimum_pair_count,
        pair_set_sha256,
        fit_policy_sha256,
        pair_count: pairs.len(),
        missing_pair_records_before_or_between_observations,
        maximum_abs_drift_ppm: effective_policy.maximum_abs_drift_ppm,
        maximum_allowed_abs_residual_ns: effective_policy.maximum_abs_residual_ns,
        estimated_drift_ppm,
        median_abs_residual_ns,
        p95_abs_residual_ns,
        maximum_abs_residual_ns,
        maximum_capture_bracket_ns,
        p95_capture_bracket_ns,
        accuracy_status: RunnerCpalClockAccuracyStatusV1::PointEstimateOnly,
        reported_uncertainty_ns: None,
    })
}

/// Replayable fit receipt binding descriptor, exact pair bytes and fit policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerCpalClockFitBundleV1 {
    pub schema_version: String,
    pub descriptor: RunnerCpalClockDescriptorV1,
    pub fit: RunnerCpalClockFitV1,
}

pub fn fit_runner_cpal_clock_bundle_v1(
    descriptor: &RunnerCpalClockDescriptorV1,
    pairs: &[RunnerCpalClockPairV1],
    policy: RunnerCpalClockFitPolicyV1,
) -> Result<RunnerCpalClockFitBundleV1, RunnerCpalClockFitErrorV1> {
    if !descriptor.is_valid() {
        return Err(RunnerCpalClockFitErrorV1::InvalidDescriptor);
    }
    let fit = fit_runner_cpal_clock_v1(descriptor, pairs, policy)?;
    Ok(RunnerCpalClockFitBundleV1 {
        schema_version: RUNNER_CPAL_CLOCK_BUNDLE_SCHEMA_V1.to_owned(),
        descriptor: descriptor.clone(),
        fit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> RunnerCpalClockDescriptorV1 {
        RunnerCpalClockDescriptorV1 {
            schema_version: RUNNER_CLOCK_BRIDGE_DESCRIPTOR_SCHEMA_V1.to_owned(),
            session_id: "session-001".to_owned(),
            runner_clock_domain_id: "runner-monotonic-v1".to_owned(),
            runner_origin_id: "origin-001".to_owned(),
            stream_instance_id: 7,
            cpal_stream_descriptor_sha256: "a".repeat(64),
        }
    }

    fn pair(index: u64, cpal_ns: u64, runner_midpoint_ns: u64) -> RunnerCpalClockPairV1 {
        RunnerCpalClockPairV1 {
            schema_version: RunnerCpalClockPairSchemaV1::V1,
            stream_instance_id: 7,
            descriptor_identity_sha256: descriptor().identity_sha256().unwrap(),
            pair_index: index,
            cpal_callback_stream_time_ns: cpal_ns,
            cpal_playback_stream_time_ns: cpal_ns + 5_000_000,
            runner_time_before_ns: runner_midpoint_ns - 50,
            runner_time_after_ns: runner_midpoint_ns + 50,
        }
    }

    fn policy() -> RunnerCpalClockFitPolicyV1 {
        RunnerCpalClockFitPolicyV1 {
            minimum_pair_count: 4,
            maximum_abs_drift_ppm: 100.0,
            maximum_abs_residual_ns: 1.0,
            maximum_capture_bracket_ns: 200,
            maximum_extrapolation_ns: 20_000_000,
        }
    }

    fn exact_pairs() -> Vec<RunnerCpalClockPairV1> {
        vec![
            pair(0, 5_000_000, 105_000_000),
            pair(1, 15_000_000, 115_000_000),
            pair(2, 25_000_000, 125_000_000),
            pair(3, 35_000_000, 135_000_000),
        ]
    }

    #[test]
    fn compact_pair_record_is_copyable_and_rejects_invalid_brackets() {
        let d = descriptor();
        let identity = d.identity_sha256().unwrap();
        let record = RunnerCpalClockPairV1::from_observation(
            d.stream_instance_id,
            identity,
            0,
            1_000,
            6_000,
            2_000,
            2_100,
        )
        .unwrap();
        let copied = record;
        assert_eq!(record, copied);
        assert!(RunnerCpalClockPairV1::from_observation(
            d.stream_instance_id,
            identity,
            1,
            2_000,
            7_000,
            3_000,
            2_999,
        ).is_none());
    }

    #[test]
    fn exact_pair_fit_maps_runner_to_cpal_playback_stream_time() {
        let d = descriptor();
        let fit = fit_runner_cpal_clock_v1(&d, &exact_pairs(), policy()).unwrap();
        let estimate = fit.playback_time_at_runner_time(120_000_000).unwrap();

        assert!((fit.estimated_drift_ppm).abs() < 1e-6);
        assert!((estimate.cpal_playback_stream_time_ns - 25_000_000.0).abs() < 1e-6);
        assert_eq!(
            estimate.accuracy_status,
            RunnerCpalClockAccuracyStatusV1::PointEstimateOnly
        );
        assert!(estimate.reported_uncertainty_ns.is_none());
    }

    #[test]
    fn positive_and_negative_clock_drift_are_estimated() {
        let positive = vec![
            pair(0, 0, 100_000_000),
            pair(1, 10_000_000, 110_000_250),
            pair(2, 20_000_000, 120_000_500),
            pair(3, 30_000_000, 130_000_750),
        ];
        let negative = vec![
            pair(0, 0, 100_000_000),
            pair(1, 10_000_000, 109_999_750),
            pair(2, 20_000_000, 119_999_500),
            pair(3, 30_000_000, 129_999_250),
        ];
        let d = descriptor();
        assert!((fit_runner_cpal_clock_v1(&d, &positive, policy()).unwrap().estimated_drift_ppm - 25.0).abs() < 0.1);
        assert!((fit_runner_cpal_clock_v1(&d, &negative, policy()).unwrap().estimated_drift_ppm + 25.0).abs() < 0.1);
    }

    #[test]
    fn wide_capture_bracket_is_rejected() {
        let mut pairs = exact_pairs();
        pairs[2].runner_time_after_ns += 1_000;
        assert!(matches!(
            fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()),
            Err(RunnerCpalClockFitErrorV1::CaptureBracketOutsidePolicy { index: 2, .. })
        ));
    }

    #[test]
    fn runner_clock_regression_is_rejected_even_when_cpal_time_increases() {
        let mut pairs = exact_pairs();
        pairs[2].runner_time_before_ns = pairs[1].runner_time_before_ns - 1;
        pairs[2].runner_time_after_ns = pairs[1].runner_time_after_ns + 1;
        assert!(matches!(
            fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()),
            Err(RunnerCpalClockFitErrorV1::NonMonotonicRunnerBefore { index: 2 })
        ));
    }

    #[test]
    fn out_of_policy_drift_is_rejected() {
        let pairs = vec![
            pair(0, 0, 100_000_000),
            pair(1, 10_000_000, 110_020_000),
            pair(2, 20_000_000, 120_040_000),
            pair(3, 30_000_000, 130_060_000),
        ];
        assert!(matches!(
            fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()),
            Err(RunnerCpalClockFitErrorV1::DriftOutsidePolicy { .. })
        ));
    }

    #[test]
    fn residual_outlier_is_rejected_not_silently_removed() {
        let pairs = vec![
            pair(0, 0, 100_000_000),
            pair(1, 10_000_000, 110_000_000),
            pair(2, 20_000_000, 120_001_000),
            pair(3, 30_000_000, 130_000_000),
        ];
        assert!(matches!(
            fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()),
            Err(RunnerCpalClockFitErrorV1::ResidualOutsidePolicy { .. })
        ));
    }

    #[test]
    fn descriptor_identity_binds_runner_session_clock_domain_and_origin() {
        let pairs = exact_pairs();
        for mutate in 0..3 {
            let mut d = descriptor();
            match mutate {
                0 => d.session_id = "another-session".to_owned(),
                1 => d.runner_clock_domain_id = "another-domain".to_owned(),
                _ => d.runner_origin_id = "another-origin".to_owned(),
            }
            assert!(d.is_valid());
            assert!(matches!(
                fit_runner_cpal_clock_v1(&d, &pairs, policy()),
                Err(RunnerCpalClockFitErrorV1::DescriptorMismatch { index: 0 })
            ));
        }
    }

    #[test]
    fn non_monotonic_clock_observation_is_rejected() {
        let mut pairs = exact_pairs();
        pairs[2].cpal_callback_stream_time_ns = pairs[1].cpal_callback_stream_time_ns;
        assert!(matches!(
            fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()),
            Err(RunnerCpalClockFitErrorV1::NonMonotonicCpalTime { index: 2 })
        ));
    }

    #[test]
    fn missing_pair_records_are_counted_not_filled_with_zero() {
        let mut pairs = exact_pairs();
        pairs[2] = pair(3, 35_000_000, 135_000_000);
        pairs[3] = pair(4, 45_000_000, 145_000_000);
        let fit = fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()).unwrap();
        assert_eq!(fit.missing_pair_records_before_or_between_observations, 1);
    }

    #[test]
    fn calibration_expiry_rejects_queries_beyond_horizon() {
        let fit = fit_runner_cpal_clock_v1(&descriptor(), &exact_pairs(), policy()).unwrap();
        assert!(matches!(
            fit.playback_time_at_runner_time(200_000_000),
            Err(RunnerCpalClockFitErrorV1::QueryOutsideValidity { .. })
        ));
    }

    #[test]
    fn runner_and_cpal_fit_remains_point_estimate_without_error_propagation() {
        let mut pairs = exact_pairs();
        pairs[0].runner_time_before_ns -= 10;
        pairs[0].runner_time_after_ns += 10;
        let fit = fit_runner_cpal_clock_v1(&descriptor(), &pairs, policy()).unwrap();
        assert_eq!(fit.accuracy_status, RunnerCpalClockAccuracyStatusV1::PointEstimateOnly);
        assert!(fit.reported_uncertainty_ns.is_none());
    }

    #[test]
    fn bundle_binds_exact_pairs_and_fit_policy_digests() {
        let d = descriptor();
        let pairs = exact_pairs();
        let bundle = fit_runner_cpal_clock_bundle_v1(&d, &pairs, policy()).unwrap();
        let serialized = serde_json::to_vec(&pairs).unwrap();
        assert_eq!(
            bundle.fit.pair_set_sha256,
            format!("{:x}", Sha256::digest(&serialized))
        );
        assert_eq!(bundle.fit.pair_set_sha256.len(), 64);
        assert_eq!(bundle.fit.fit_policy_sha256.len(), 64);
        assert_eq!(bundle.fit.minimum_required_pair_count, 4);
    }

    #[derive(Deserialize)]
    struct RunnerCpalClockFixtureV1 {
        schema_version: String,
        fixture_id: String,
        expected_disposition: String,
        descriptor: RunnerCpalClockDescriptorV1,
        policy: RunnerCpalClockFitPolicyV1,
        pairs: Vec<RunnerCpalClockPairV1>,
    }

    #[test]
    fn checked_in_runner_cpal_clock_fixtures_match_the_evaluator() {
        let fixtures = [
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-exact-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-drift-pos-25ppm-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-drift-neg-25ppm-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-missing-pair-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-wide-bracket-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-descriptor-mismatch-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-drift-outside-policy-001.json"),
            include_str!("../../../../benchmarks/mel/clock/mel-runner-cpal-residual-outlier-001.json"),
        ];

        for fixture_json in fixtures {
            let fixture: RunnerCpalClockFixtureV1 =
                serde_json::from_str(fixture_json).expect("clock fixture must parse");
            assert_eq!(
                fixture.schema_version,
                "mel-runner-cpal-clock-pair-set-v1",
                "{}",
                fixture.fixture_id
            );
            let result = fit_runner_cpal_clock_bundle_v1(
                &fixture.descriptor,
                &fixture.pairs,
                fixture.policy,
            );
            match fixture.expected_disposition.as_str() {
                "fit" => {
                    let bundle = result.unwrap_or_else(|error| {
                        panic!("{} unexpectedly rejected: {error}", fixture.fixture_id)
                    });
                    assert_eq!(
                        bundle.schema_version,
                        RUNNER_CPAL_CLOCK_BUNDLE_SCHEMA_V1,
                        "{}",
                        fixture.fixture_id
                    );
                    if fixture.fixture_id == "MEL-RUNNER-CPAL-DRIFT-POS-25PPM-001" {
                        assert!((bundle.fit.estimated_drift_ppm - 25.0).abs() < 0.1);
                    }
                    if fixture.fixture_id == "MEL-RUNNER-CPAL-DRIFT-NEG-25PPM-001" {
                        assert!((bundle.fit.estimated_drift_ppm + 25.0).abs() < 0.1);
                    }
                    if fixture.fixture_id == "MEL-RUNNER-CPAL-MISSING-PAIR-001" {
                        assert_eq!(
                            bundle.fit.missing_pair_records_before_or_between_observations,
                            1
                        );
                    }
                }
                "reject_capture_bracket" => assert!(
                    matches!(result, Err(RunnerCpalClockFitErrorV1::CaptureBracketOutsidePolicy { .. })),
                    "{} must fail the capture-bracket gate",
                    fixture.fixture_id
                ),
                "reject_descriptor_mismatch" => assert!(
                    matches!(result, Err(RunnerCpalClockFitErrorV1::DescriptorMismatch { .. })),
                    "{} must fail the descriptor-identity gate",
                    fixture.fixture_id
                ),
                "reject_drift" => assert!(
                    matches!(result, Err(RunnerCpalClockFitErrorV1::DriftOutsidePolicy { .. })),
                    "{} must fail the drift gate",
                    fixture.fixture_id
                ),
                "reject_residual" => assert!(
                    matches!(result, Err(RunnerCpalClockFitErrorV1::ResidualOutsidePolicy { .. })),
                    "{} must fail the residual gate",
                    fixture.fixture_id
                ),
                other => panic!(
                    "{} has unsupported expected disposition {other}",
                    fixture.fixture_id
                ),
            }
        }
    }

    #[test]
    fn bundle_rejects_invalid_descriptor_digest() {
        let mut d = descriptor();
        d.cpal_stream_descriptor_sha256 = "not-a-digest".to_owned();
        assert!(matches!(
            fit_runner_cpal_clock_bundle_v1(&d, &exact_pairs(), policy()),
            Err(RunnerCpalClockFitErrorV1::InvalidDescriptor)
        ));
    }
}
