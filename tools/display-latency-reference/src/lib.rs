#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reference photodiode transition analysis for frameprobe-compatible captures.
//!
//! This is a small, independent implementation of a normalized crossing method.
//! It is intentionally not a copy of frameprobe's analyzer. Temporal-light-
//! modulation compensation is out of scope; its use on flickering traces must
//! remain unqualified until that behavior has a separately validated method.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingReference {
    /// Time relative to the host input call.
    Press,
    /// Time relative to the host's USB receipt of the input report.
    HostDelivery,
}

impl TimingReference {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Press => "PRESS",
            Self::HostDelivery => "HOST_DELIVERY",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyStatus {
    Measured,
    NoDelivery,
    MalformedRow,
    SampleCountMismatch,
    InvalidTimebase,
    InsufficientSamples,
    AdcOutOfRange,
    AdcSaturated,
    NoTransition,
    Incomplete,
}

impl LatencyStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "MEASURED_UNQUALIFIED",
            Self::NoDelivery => "NO_DELIVERY",
            Self::MalformedRow => "MALFORMED_ROW",
            Self::SampleCountMismatch => "SAMPLE_COUNT_MISMATCH",
            Self::InvalidTimebase => "INVALID_TIMEBASE",
            Self::InsufficientSamples => "INSUFFICIENT_SAMPLES",
            Self::AdcOutOfRange => "ADC_OUT_OF_RANGE",
            Self::AdcSaturated => "ADC_SATURATED",
            Self::NoTransition => "NO_TRANSITION",
            Self::Incomplete => "INCOMPLETE",
        }
    }
}

/// One firmware capture row after parsing the numeric CSV fields.
///
/// Frameprobe names the first timing field clickTime. In current firmware it is
/// the duration of the HID input call, not an absolute timestamp. timeTaken is
/// the total capture duration; deliveryTime, when present, is the elapsed time
/// from the input call to host USB receipt. All three are in microseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureRow {
    pub click_duration_us: u64,
    pub total_duration_us: u64,
    pub declared_sample_count: usize,
    pub pre_click_samples: usize,
    pub samples: Vec<u16>,
    /// None means an older row did not record host delivery; Some(0) means the
    /// firmware recorded its explicit "not delivered" sentinel.
    pub delivery_time_us: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencyConfig {
    pub baseline_window_samples: usize,
    pub noise_window_samples: usize,
    pub settled_tail_samples: usize,
    pub minimum_baseline_samples: usize,
    pub minimum_noise_samples: usize,
    pub minimum_settled_samples: usize,
    pub minimum_swing_adc: f64,
    pub noise_swing_multiple: f64,
    pub sustained_crossing_samples: usize,
    pub onset_fraction: f64,
    pub midpoint_fraction: f64,
    pub settled_fraction: f64,
    pub adc_max: u16,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            baseline_window_samples: 200,
            noise_window_samples: 1_000,
            settled_tail_samples: 500,
            minimum_baseline_samples: 20,
            minimum_noise_samples: 20,
            minimum_settled_samples: 20,
            minimum_swing_adc: 50.0,
            noise_swing_multiple: 2.0,
            sustained_crossing_samples: 5,
            onset_fraction: 0.10,
            midpoint_fraction: 0.50,
            settled_fraction: 0.90,
            adc_max: 16_383,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub &'static str);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ConfigError {}

impl LatencyConfig {
    /// Reject an invalid analysis profile instead of silently changing it.
    pub fn validate(self) -> Result<(), ConfigError> {
        if self.baseline_window_samples == 0
            || self.noise_window_samples == 0
            || self.settled_tail_samples == 0
        {
            return Err(ConfigError("analysis windows must be nonzero"));
        }
        if self.minimum_baseline_samples == 0
            || self.minimum_noise_samples == 0
            || self.minimum_settled_samples == 0
        {
            return Err(ConfigError("minimum sample counts must be nonzero"));
        }
        if self.minimum_baseline_samples > self.baseline_window_samples
            || self.minimum_noise_samples > self.noise_window_samples
            || self.minimum_settled_samples > self.settled_tail_samples
        {
            return Err(ConfigError(
                "minimum sample counts cannot exceed their configured windows",
            ));
        }
        if !self.minimum_swing_adc.is_finite() || self.minimum_swing_adc <= 0.0 {
            return Err(ConfigError("minimum swing must be finite and positive"));
        }
        if !self.noise_swing_multiple.is_finite() || self.noise_swing_multiple < 0.0 {
            return Err(ConfigError(
                "noise-to-swing multiplier must be finite and nonnegative",
            ));
        }
        if self.sustained_crossing_samples == 0 {
            return Err(ConfigError("sustained crossing sample count must be nonzero"));
        }
        if !(0.0 < self.onset_fraction
            && self.onset_fraction < self.midpoint_fraction
            && self.midpoint_fraction < self.settled_fraction
            && self.settled_fraction < 1.0)
        {
            return Err(ConfigError(
                "crossing fractions must satisfy 0 < onset < midpoint < settled < 1",
            ));
        }
        if self.adc_max < 2 {
            return Err(ConfigError("ADC maximum must be at least two counts"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LatencyResult {
    pub status: LatencyStatus,
    pub reference: TimingReference,
    pub reason: Option<&'static str>,
    pub baseline_adc: Option<f64>,
    pub settled_adc: Option<f64>,
    pub swing_adc: Option<f64>,
    pub noise_peak_to_peak_adc: Option<f64>,
    pub signal_noise_separation: Option<f64>,
    /// Estimated average sample spacing. It is derived from aggregate capture
    /// duration, not from individually timestamped ADC samples.
    pub estimated_sample_interval_us: Option<f64>,
    pub t10_us: Option<f64>,
    pub t50_us: Option<f64>,
    pub t90_us: Option<f64>,
    /// t90 - t10. This is sensor-plus-display response, not panel-only response.
    pub response_t10_t90_us: Option<f64>,
}

impl LatencyResult {
    fn new(
        status: LatencyStatus,
        reference: TimingReference,
        reason: Option<&'static str>,
    ) -> Self {
        Self {
            status,
            reference,
            reason,
            baseline_adc: None,
            settled_adc: None,
            swing_adc: None,
            noise_peak_to_peak_adc: None,
            signal_noise_separation: None,
            estimated_sample_interval_us: None,
            t10_us: None,
            t50_us: None,
            t90_us: None,
            response_t10_t90_us: None,
        }
    }

    fn skipped(
        reference: TimingReference,
        status: LatencyStatus,
        reason: &'static str,
    ) -> Self {
        Self::new(status, reference, Some(reason))
    }
}

/// Analyze a single capture row.
///
/// The returned status is never a physical qualification. A successful row has
/// a defensible t50 crossing under this algorithm only; calibration, detector
/// bandwidth, actual sample timing, display mode and flicker remain separate.
pub fn analyze_capture(
    row: &CaptureRow,
    reference: TimingReference,
    config: LatencyConfig,
) -> Result<LatencyResult, ConfigError> {
    config.validate()?;

    if row.declared_sample_count != row.samples.len() {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::SampleCountMismatch,
            "declared sample count differs from parsed samples",
        ));
    }
    if row.samples.is_empty() || row.pre_click_samples == 0 || row.pre_click_samples >= row.samples.len()
    {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::MalformedRow,
            "sample vector or pre-click index is invalid",
        ));
    }
    if row.samples.iter().any(|&sample| sample > config.adc_max) {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::AdcOutOfRange,
            "sample exceeds declared ADC range",
        ));
    }
    if row.samples.iter().any(|&sample| sample == 0 || sample == config.adc_max) {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::AdcSaturated,
            "sample touches an ADC rail; transition metrics are not trusted",
        ));
    }

    let baseline_start = row
        .pre_click_samples
        .saturating_sub(config.baseline_window_samples);
    let noise_start = row
        .pre_click_samples
        .saturating_sub(config.noise_window_samples);
    let settled_start = row
        .pre_click_samples
        .max(row.samples.len().saturating_sub(config.settled_tail_samples));

    let baseline_count = row.pre_click_samples - baseline_start;
    let noise_count = row.pre_click_samples - noise_start;
    let settled_count = row.samples.len() - settled_start;
    if baseline_count < config.minimum_baseline_samples
        || noise_count < config.minimum_noise_samples
        || settled_count < config.minimum_settled_samples
    {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::InsufficientSamples,
            "baseline, noise, or settled-tail window is too short",
        ));
    }

    let (pause_us, offset_us) = match (reference, row.delivery_time_us) {
        (_, Some(0)) => {
            return Ok(LatencyResult::skipped(
                reference,
                LatencyStatus::NoDelivery,
                "firmware delivery sentinel is zero",
            ));
        }
        (TimingReference::HostDelivery, None) => {
            return Ok(LatencyResult::skipped(
                reference,
                LatencyStatus::NoDelivery,
                "capture has no host-delivery timestamp",
            ));
        }
        (TimingReference::Press, Some(delivery)) => (delivery, delivery as f64),
        (TimingReference::Press, None) => (row.click_duration_us, row.click_duration_us as f64),
        (TimingReference::HostDelivery, Some(delivery)) => (delivery, 0.0),
    };
    let Some(sampled_duration_us) = row.total_duration_us.checked_sub(pause_us) else {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::InvalidTimebase,
            "pause/delivery duration exceeds total capture duration",
        ));
    };
    let sample_interval_us = sampled_duration_us as f64 / row.samples.len() as f64;
    if !sample_interval_us.is_finite() || sample_interval_us <= 0.0 {
        return Ok(LatencyResult::skipped(
            reference,
            LatencyStatus::InvalidTimebase,
            "estimated average sample interval is not positive and finite",
        ));
    }

    let baseline = upper_median(&row.samples[baseline_start..row.pre_click_samples]);
    let settled = upper_median(&row.samples[settled_start..]);
    let swing = settled - baseline;
    let noise_window = &row.samples[noise_start..row.pre_click_samples];
    let noise_min = noise_window.iter().copied().min().unwrap_or(0);
    let noise_max = noise_window.iter().copied().max().unwrap_or(0);
    let noise_peak_to_peak = f64::from(noise_max - noise_min);

    let mut result = LatencyResult::new(LatencyStatus::Measured, reference, None);
    result.baseline_adc = Some(baseline);
    result.settled_adc = Some(settled);
    result.swing_adc = Some(swing);
    result.noise_peak_to_peak_adc = Some(noise_peak_to_peak);
    result.signal_noise_separation = Some(swing.abs() / noise_peak_to_peak.max(1.0));
    result.estimated_sample_interval_us = Some(sample_interval_us);

    if swing.abs() <= config.minimum_swing_adc.max(noise_peak_to_peak * config.noise_swing_multiple)
    {
        result.status = LatencyStatus::NoTransition;
        result.reason = Some("settled swing does not exceed minimum swing/noise gate");
        return Ok(result);
    }

    let timestamp = |index: usize| {
        offset_us + index.saturating_sub(row.pre_click_samples) as f64 * sample_interval_us
    };
    let t50_index = find_crossing(
        &row.samples,
        baseline,
        swing,
        config.midpoint_fraction,
        config.sustained_crossing_samples,
        row.pre_click_samples,
    );
    let Some(t50_index) = t50_index else {
        result.status = LatencyStatus::Incomplete;
        result.reason = Some("midpoint crossing did not persist before capture end");
        return Ok(result);
    };
    result.t50_us = Some(timestamp(t50_index));

    if swing.abs() * config.onset_fraction > noise_peak_to_peak
        && let Some(t10_index) = find_crossing(
            &row.samples,
            baseline,
            swing,
            config.onset_fraction,
            config.sustained_crossing_samples,
            row.pre_click_samples,
        )
    {
        result.t10_us = Some(timestamp(t10_index));
    }

    if let Some(t90_index) = find_crossing(
        &row.samples,
        baseline,
        swing,
        config.settled_fraction,
        1,
        t50_index,
    ) {
        result.t90_us = Some(timestamp(t90_index));
    }
    if let (Some(t10), Some(t90)) = (result.t10_us, result.t90_us) {
        if t90 >= t10 {
            result.response_t10_t90_us = Some(t90 - t10);
        } else {
            result.status = LatencyStatus::Incomplete;
            result.reason = Some("near-settled crossing precedes onset crossing");
            result.t90_us = None;
        }
    }
    Ok(result)
}

fn upper_median(samples: &[u16]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    f64::from(sorted[sorted.len() / 2])
}

fn find_crossing(
    samples: &[u16],
    baseline: f64,
    swing: f64,
    fraction: f64,
    sustain: usize,
    start: usize,
) -> Option<usize> {
    let level = baseline + swing * fraction;
    let mut consecutive = 0_usize;
    for (index, &sample) in samples.iter().enumerate().skip(start) {
        let beyond = if swing > 0.0 {
            f64::from(sample) >= level
        } else {
            f64::from(sample) <= level
        };
        if beyond {
            consecutive += 1;
            if consecutive >= sustain {
                return Some(index + 1 - consecutive);
            }
        } else {
            consecutive = 0;
        }
    }
    None
}

/// Header mapping for frameprobe's numeric CSV format, including the optional
/// deliveryTime column. Five-column legacy headers are supported; if a newer
/// firmware row appends a sixth field under an older header, it is interpreted
/// as deliveryTime, matching the documented compatibility behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameprobeHeader {
    names: Vec<String>,
    click_time: usize,
    total_time: usize,
    sample_count: usize,
    pre_click_samples: usize,
    samples: usize,
    delivery_time: Option<usize>,
}

impl FrameprobeHeader {
    pub fn parse(line: &str) -> Result<Self, String> {
        let names: Vec<String> = line
            .trim_end_matches('\r')
            .split(',')
            .map(|field| field.trim().to_owned())
            .collect();
        let required = [
            "clickTime",
            "timeTaken",
            "sampleCount",
            "preClickSamples",
            "samples",
        ];
        for name in required {
            if names.iter().filter(|field| field.as_str() == name).count() != 1 {
                return Err(format!("header must contain exactly one {name} column"));
            }
        }
        if names.iter().filter(|field| field.as_str() == "deliveryTime").count() > 1 {
            return Err("header contains duplicate deliveryTime columns".to_owned());
        }
        Ok(Self {
            click_time: find_column(&names, "clickTime")?,
            total_time: find_column(&names, "timeTaken")?,
            sample_count: find_column(&names, "sampleCount")?,
            pre_click_samples: find_column(&names, "preClickSamples")?,
            samples: find_column(&names, "samples")?,
            delivery_time: names.iter().position(|field| field == "deliveryTime"),
            names,
        })
    }

    pub fn parse_row(&self, line: &str) -> Result<CaptureRow, String> {
        let fields: Vec<&str> = line.trim_end_matches('\r').split(',').collect();
        let legacy_appended_delivery = self.delivery_time.is_none() && fields.len() == self.names.len() + 1;
        let missing_optional_delivery = self.delivery_time.is_some() && fields.len() + 1 == self.names.len();
        if fields.len() != self.names.len() && !legacy_appended_delivery && !missing_optional_delivery {
            return Err("row field count does not match the header".to_owned());
        }
        let field = |index: usize| {
            fields
                .get(index)
                .copied()
                .ok_or_else(|| "row is missing a required field".to_owned())
        };
        let click_duration_us = parse_u64(field(self.click_time)?, "clickTime")?;
        let total_duration_us = parse_u64(field(self.total_time)?, "timeTaken")?;
        let declared_sample_count = parse_usize(field(self.sample_count)?, "sampleCount")?;
        let pre_click_samples = parse_usize(field(self.pre_click_samples)?, "preClickSamples")?;
        let sample_text = field(self.samples)?;
        let mut tokens: Vec<&str> = sample_text.split(';').collect();
        while tokens.last().is_some_and(|token| token.trim().is_empty()) {
            tokens.pop();
        }
        if tokens.is_empty() || tokens.iter().any(|token| token.trim().is_empty()) {
            return Err("sample vector is empty or contains an empty interior field".to_owned());
        }
        let mut samples = Vec::with_capacity(tokens.len());
        for token in tokens {
            samples.push(
                token
                    .trim()
                    .parse::<u16>()
                    .map_err(|_| "sample vector contains a non-integer ADC value".to_owned())?,
            );
        }

        let delivery_index = self.delivery_time.or_else(|| {
            if legacy_appended_delivery {
                Some(self.names.len())
            } else {
                None
            }
        });
        let delivery_time_us = delivery_index
            .and_then(|index| fields.get(index))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| parse_u64(value, "deliveryTime"))
            .transpose()?;

        Ok(CaptureRow {
            click_duration_us,
            total_duration_us,
            declared_sample_count,
            pre_click_samples,
            samples,
            delivery_time_us,
        })
    }
}

fn find_column(names: &[String], name: &str) -> Result<usize, String> {
    names
        .iter()
        .position(|field| field == name)
        .ok_or_else(|| format!("header is missing {name}"))
}

fn parse_u64(value: &str, field: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("{field} must be a nonnegative integer"))
}

fn parse_usize(value: &str, field: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("{field} must be a nonnegative integer"))
}

#[cfg(test)]
mod tests {
    use super::{
        CaptureRow, FrameprobeHeader, LatencyConfig, LatencyStatus, TimingReference,
        analyze_capture,
    };

    fn step_capture() -> CaptureRow {
        let pre_click_samples = 1_200;
        let sample_count = 1_800;
        let mut samples = Vec::with_capacity(sample_count);
        for index in 0..sample_count {
            let value = if index < pre_click_samples {
                1_000
            } else if index < pre_click_samples + 100 {
                1_000 + (index - pre_click_samples) as u16
            } else {
                1_100
            };
            samples.push(value);
        }
        CaptureRow {
            click_duration_us: 80,
            total_duration_us: 44_200,
            declared_sample_count: sample_count,
            pre_click_samples,
            samples,
            delivery_time_us: Some(200),
        }
    }

    #[test]
    fn known_answer_crossings_preserve_press_and_delivery_references() {
        let row = step_capture();
        let pressed = analyze_capture(&row, TimingReference::Press, LatencyConfig::default())
            .expect("default config valid");
        assert_eq!(pressed.status, LatencyStatus::Measured);
        assert_eq!(pressed.t10_us, Some(440.0));
        assert_eq!(pressed.t50_us, Some(1_400.0));
        assert_eq!(pressed.t90_us, Some(2_360.0));
        assert_eq!(pressed.response_t10_t90_us, Some(1_920.0));
        assert_eq!(pressed.estimated_sample_interval_us, Some(24.0));
        assert_eq!(pressed.signal_noise_separation, Some(100.0));

        let delivered = analyze_capture(
            &row,
            TimingReference::HostDelivery,
            LatencyConfig::default(),
        )
        .expect("default config valid");
        assert_eq!(delivered.status, LatencyStatus::Measured);
        assert_eq!(delivered.t50_us, Some(1_200.0));
    }

    #[test]
    fn no_delivery_sentinel_fails_closed_for_both_time_references() {
        let mut row = step_capture();
        row.delivery_time_us = Some(0);
        assert_eq!(
            analyze_capture(&row, TimingReference::Press, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::NoDelivery
        );
        assert_eq!(
            analyze_capture(&row, TimingReference::HostDelivery, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::NoDelivery
        );
    }

    #[test]
    fn delivery_reference_requires_delivery_timestamp() {
        let mut row = step_capture();
        row.delivery_time_us = None;
        let result = analyze_capture(&row, TimingReference::HostDelivery, LatencyConfig::default())
            .expect("default config valid");
        assert_eq!(result.status, LatencyStatus::NoDelivery);
        assert_eq!(result.t50_us, None);
    }

    #[test]
    fn no_transition_is_not_reported_as_zero_latency() {
        let mut row = step_capture();
        row.samples.fill(1_000);
        let result = analyze_capture(&row, TimingReference::Press, LatencyConfig::default())
            .expect("default config valid");
        assert_eq!(result.status, LatencyStatus::NoTransition);
        assert_eq!(result.t50_us, None);
        assert_eq!(result.swing_adc, Some(0.0));
    }

    #[test]
    fn unsustained_midpoint_spikes_are_incomplete_not_fast_transitions() {
        let mut row = step_capture();
        for index in row.pre_click_samples..row.samples.len() {
            row.samples[index] = if index % 2 == 0 { 1_100 } else { 1_000 };
        }
        let result = analyze_capture(&row, TimingReference::Press, LatencyConfig::default())
            .expect("default config valid");
        assert_eq!(result.status, LatencyStatus::Incomplete);
        assert_eq!(result.t50_us, None);
    }

    #[test]
    fn noisy_baseline_can_suppress_onset_without_erasing_t50() {
        let mut row = step_capture();
        for index in 200..row.pre_click_samples {
            row.samples[index] = if index % 2 == 0 { 990 } else { 1_010 };
        }
        for index in row.pre_click_samples..row.samples.len() {
            row.samples[index] = if index < row.pre_click_samples + 100 {
                1_010 + (index - row.pre_click_samples) as u16
            } else {
                1_110
            };
        }
        let result = analyze_capture(&row, TimingReference::Press, LatencyConfig::default())
            .expect("default config valid");
        assert_eq!(result.status, LatencyStatus::Measured);
        assert_eq!(result.t10_us, None);
        assert!(result.t50_us.is_some());
    }

    #[test]
    fn sample_count_mismatch_and_adc_rail_values_are_rejected() {
        let mut mismatch = step_capture();
        mismatch.declared_sample_count += 1;
        assert_eq!(
            analyze_capture(&mismatch, TimingReference::Press, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::SampleCountMismatch
        );

        let mut saturated = step_capture();
        saturated.samples[500] = 0;
        assert_eq!(
            analyze_capture(&saturated, TimingReference::Press, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::AdcSaturated
        );

        let mut out_of_range = step_capture();
        out_of_range.samples[500] = 16_384;
        assert_eq!(
            analyze_capture(&out_of_range, TimingReference::Press, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::AdcOutOfRange
        );
    }

    #[test]
    fn invalid_timebase_and_short_windows_are_rejected() {
        let mut invalid = step_capture();
        invalid.total_duration_us = 100;
        assert_eq!(
            analyze_capture(&invalid, TimingReference::Press, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::InvalidTimebase
        );

        let mut too_short = step_capture();
        too_short.samples.truncate(1_100);
        too_short.declared_sample_count = too_short.samples.len();
        assert_eq!(
            analyze_capture(&too_short, TimingReference::Press, LatencyConfig::default())
                .expect("default config valid")
                .status,
            LatencyStatus::InsufficientSamples
        );
    }

    #[test]
    fn frameprobe_headers_and_delivery_column_compatibility_are_explicit() {
        let header = FrameprobeHeader::parse(
            "clickTime,timeTaken,sampleCount,preClickSamples,samples,deliveryTime",
        )
        .expect("current header valid");
        let row = header
            .parse_row("80,44200,6,2,1000;1000;1000;1000;1050;1100;,200")
            .expect("current row valid");
        assert_eq!(row.declared_sample_count, 6);
        assert_eq!(row.pre_click_samples, 2);
        assert_eq!(row.delivery_time_us, Some(200));

        let legacy = FrameprobeHeader::parse(
            "clickTime,timeTaken,sampleCount,preClickSamples,samples",
        )
        .expect("legacy header valid");
        let old = legacy
            .parse_row("80,44200,6,2,1000;1000;1000;1000;1050;1100;")
            .expect("legacy row valid");
        assert_eq!(old.delivery_time_us, None);

        let newer_firmware = legacy
            .parse_row("80,44200,6,2,1000;1000;1000;1000;1050;1100;,200")
            .expect("trailing compatibility delivery field valid");
        assert_eq!(newer_firmware.delivery_time_us, Some(200));
    }

    #[test]
    fn malformed_frameprobe_rows_reject_empty_samples_and_bad_integers() {
        let header = FrameprobeHeader::parse(
            "clickTime,timeTaken,sampleCount,preClickSamples,samples",
        )
        .expect("legacy header valid");
        assert!(header.parse_row("80,44200,6,2,").is_err());
        assert!(header.parse_row("80,44200,6,2,1000;bad;1100").is_err());
        assert!(header.parse_row("80,44200,6,2,1000;;1100").is_err());
        assert!(FrameprobeHeader::parse("clickTime,timeTaken,clickTime,samples").is_err());
    }

    #[test]
    fn invalid_configuration_is_not_silently_normalized() {
        let config = LatencyConfig {
            onset_fraction: 0.6,
            midpoint_fraction: 0.5,
            ..LatencyConfig::default()
        };
        assert!(analyze_capture(&step_capture(), TimingReference::Press, config).is_err());
    }
}
