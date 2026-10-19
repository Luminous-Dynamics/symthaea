//! Evidence-bounded integration of synchronized marine electrical input-power traces.
//!
//! V0 uses piecewise-linear interpolation between adjacent samples and conservative
//! absolute bounds on each power sample. It does not validate the physical sensor,
//! correct environmental conditions, or implement a sea-trial standard.

pub mod paired_trial;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerChannel {
    Propulsion,
    Auxiliary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerReading {
    /// Nominal electrical input power, watts.
    pub nominal_w: f64,
    /// Declared non-negative absolute bound, watts (not a standard deviation).
    pub absolute_bound_w: f64,
}

impl PowerReading {
    fn lower_w(self) -> f64 {
        (self.nominal_w - self.absolute_bound_w).max(0.0)
    }

    fn upper_w(self) -> f64 {
        self.nominal_w + self.absolute_bound_w
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowerSample {
    /// Monotonic elapsed time in integer milliseconds from a single trace origin.
    pub elapsed_ms: u64,
    pub propulsion: Option<PowerReading>,
    pub auxiliary: Option<PowerReading>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceSpec {
    /// Stable caller-assigned trace identifier.
    pub trace_id: String,
    /// SHA-256 hex digest of the raw trace; format is checked, provenance is not authenticated.
    pub raw_trace_sha256: String,
    pub mission_id: String,
    pub condition_digest: String,
    /// SHA-256 digest of the required delivered-service contract (route, distance, payload, speed policy).
    pub service_digest: String,
    /// Identity of the meter chain / calibration epoch used for all channels.
    pub calibration_epoch_id: String,
    /// All channels must refer to the same electrical input boundary convention.
    pub energy_boundary_id: String,
    /// Identifies the declared absolute-bound method; it is not interpreted by this crate.
    pub uncertainty_method_id: String,
    pub expected_start_ms: u64,
    pub expected_end_ms: u64,
    /// No adjacent samples may be farther apart than this.
    pub max_gap_ms: u64,
    pub samples: Vec<PowerSample>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceDisposition {
    Complete,
    Indeterminate,
    Invalid,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraceIssue {
    EmptyIdentity(&'static str),
    InvalidRawTraceDigest,
    InvalidServiceDigest,
    InvalidTimeRange,
    ZeroMaximumGap,
    TooFewSamples,
    TimestampNotIncreasing {
        index: usize,
        previous_ms: u64,
        current_ms: u64,
    },
    MissingStartEndpoint {
        expected_ms: u64,
        observed_ms: u64,
    },
    MissingEndEndpoint {
        expected_ms: u64,
        observed_ms: u64,
    },
    MissingPower {
        index: usize,
        channel: PowerChannel,
    },
    InvalidPower {
        index: usize,
        channel: PowerChannel,
    },
    InvalidUncertainty {
        index: usize,
        channel: PowerChannel,
    },
    GapExceedsLimit {
        start_ms: u64,
        end_ms: u64,
        gap_ms: u64,
        max_gap_ms: u64,
    },
    NonFiniteResult,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyBand {
    pub lower_wh: f64,
    pub nominal_wh: f64,
    pub upper_wh: f64,
}

impl EnergyBand {
    fn plus(&self, other: &Self) -> Self {
        Self {
            lower_wh: self.lower_wh + other.lower_wh,
            nominal_wh: self.nominal_wh + other.nominal_wh,
            upper_wh: self.upper_wh + other.upper_wh,
        }
    }

    fn is_finite_and_ordered(&self) -> bool {
        self.lower_wh.is_finite()
            && self.nominal_wh.is_finite()
            && self.upper_wh.is_finite()
            && self.lower_wh >= 0.0
            && self.lower_wh <= self.nominal_wh
            && self.nominal_wh <= self.upper_wh
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceEnergy {
    pub trace_id: String,
    pub raw_trace_sha256: String,
    pub mission_id: String,
    pub condition_digest: String,
    pub service_digest: String,
    pub calibration_epoch_id: String,
    pub energy_boundary_id: String,
    pub uncertainty_method_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub duration_ms: u64,
    pub sample_count: usize,
    pub max_observed_gap_ms: u64,
    pub propulsion_wh: EnergyBand,
    pub auxiliary_wh: EnergyBand,
    pub total_wh: EnergyBand,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraceOutcome {
    Complete(TraceEnergy),
    Indeterminate(Vec<TraceIssue>),
    Invalid(Vec<TraceIssue>),
}

impl TraceOutcome {
    pub fn disposition(&self) -> TraceDisposition {
        match self {
            Self::Complete(_) => TraceDisposition::Complete,
            Self::Indeterminate(_) => TraceDisposition::Indeterminate,
            Self::Invalid(_) => TraceDisposition::Invalid,
        }
    }
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn reading_for(sample: &PowerSample, channel: PowerChannel) -> Option<&PowerReading> {
    match channel {
        PowerChannel::Propulsion => sample.propulsion.as_ref(),
        PowerChannel::Auxiliary => sample.auxiliary.as_ref(),
    }
}

fn validate_reading(
    reading: &PowerReading,
    index: usize,
    channel: PowerChannel,
    invalid: &mut Vec<TraceIssue>,
) {
    if !reading.nominal_w.is_finite() || reading.nominal_w < 0.0 {
        invalid.push(TraceIssue::InvalidPower { index, channel });
    }
    if !reading.absolute_bound_w.is_finite()
        || reading.absolute_bound_w < 0.0
        || !(reading.nominal_w + reading.absolute_bound_w).is_finite()
    {
        invalid.push(TraceIssue::InvalidUncertainty { index, channel });
    }
}

fn integrate_channel(samples: &[PowerSample], channel: PowerChannel) -> Option<EnergyBand> {
    let mut lower_wh = 0.0;
    let mut nominal_wh = 0.0;
    let mut upper_wh = 0.0;

    for pair in samples.windows(2) {
        let start = reading_for(&pair[0], channel)?;
        let end = reading_for(&pair[1], channel)?;
        let dt_hours = (pair[1].elapsed_ms - pair[0].elapsed_ms) as f64 / 3_600_000.0;

        lower_wh += (start.lower_w() + end.lower_w()) * 0.5 * dt_hours;
        nominal_wh += (start.nominal_w + end.nominal_w) * 0.5 * dt_hours;
        upper_wh += (start.upper_w() + end.upper_w()) * 0.5 * dt_hours;
    }

    let band = EnergyBand {
        lower_wh,
        nominal_wh,
        upper_wh,
    };
    band.is_finite_and_ordered().then_some(band)
}

/// Validate and integrate one complete trace.
///
/// Missing channel readings and over-limit time gaps are Indeterminate, not zero.
/// Malformed metadata, time ordering, or numeric samples are Invalid.
/// Adjacent readings are integrated with trapezoidal interpolation; the crate does not
/// make claims about unsampled fluctuations between them.
pub fn assess_trace(spec: &TraceSpec) -> TraceOutcome {
    let mut invalid = Vec::new();
    let mut indeterminate = Vec::new();

    for (value, field) in [
        (spec.trace_id.as_str(), "trace_id"),
        (spec.mission_id.as_str(), "mission_id"),
        (spec.condition_digest.as_str(), "condition_digest"),
        (spec.calibration_epoch_id.as_str(), "calibration_epoch_id"),
        (spec.energy_boundary_id.as_str(), "energy_boundary_id"),
        (spec.uncertainty_method_id.as_str(), "uncertainty_method_id"),
    ] {
        if !is_nonempty(value) {
            invalid.push(TraceIssue::EmptyIdentity(field));
        }
    }

    if !valid_sha256_hex(&spec.raw_trace_sha256) {
        invalid.push(TraceIssue::InvalidRawTraceDigest);
    }
    if !valid_sha256_hex(&spec.service_digest) {
        invalid.push(TraceIssue::InvalidServiceDigest);
    }
    if spec.expected_end_ms <= spec.expected_start_ms {
        invalid.push(TraceIssue::InvalidTimeRange);
    }
    if spec.max_gap_ms == 0 {
        invalid.push(TraceIssue::ZeroMaximumGap);
    }
    if spec.samples.len() < 2 {
        invalid.push(TraceIssue::TooFewSamples);
    }

    if let Some(first) = spec.samples.first() {
        if first.elapsed_ms != spec.expected_start_ms {
            indeterminate.push(TraceIssue::MissingStartEndpoint {
                expected_ms: spec.expected_start_ms,
                observed_ms: first.elapsed_ms,
            });
        }
    }
    if let Some(last) = spec.samples.last() {
        if last.elapsed_ms != spec.expected_end_ms {
            indeterminate.push(TraceIssue::MissingEndEndpoint {
                expected_ms: spec.expected_end_ms,
                observed_ms: last.elapsed_ms,
            });
        }
    }

    for (index, sample) in spec.samples.iter().enumerate() {
        for channel in [PowerChannel::Propulsion, PowerChannel::Auxiliary] {
            match reading_for(sample, channel) {
                Some(reading) => validate_reading(reading, index, channel, &mut invalid),
                None => indeterminate.push(TraceIssue::MissingPower { index, channel }),
            }
        }
    }

    for (index, pair) in spec.samples.windows(2).enumerate() {
        if pair[1].elapsed_ms <= pair[0].elapsed_ms {
            invalid.push(TraceIssue::TimestampNotIncreasing {
                index: index + 1,
                previous_ms: pair[0].elapsed_ms,
                current_ms: pair[1].elapsed_ms,
            });
        } else {
            let gap_ms = pair[1].elapsed_ms - pair[0].elapsed_ms;
            if gap_ms > spec.max_gap_ms {
                indeterminate.push(TraceIssue::GapExceedsLimit {
                    start_ms: pair[0].elapsed_ms,
                    end_ms: pair[1].elapsed_ms,
                    gap_ms,
                    max_gap_ms: spec.max_gap_ms,
                });
            }
        }
    }

    if !invalid.is_empty() {
        return TraceOutcome::Invalid(invalid);
    }
    if !indeterminate.is_empty() {
        return TraceOutcome::Indeterminate(indeterminate);
    }

    let Some(propulsion_wh) = integrate_channel(&spec.samples, PowerChannel::Propulsion) else {
        return TraceOutcome::Invalid(vec![TraceIssue::NonFiniteResult]);
    };
    let Some(auxiliary_wh) = integrate_channel(&spec.samples, PowerChannel::Auxiliary) else {
        return TraceOutcome::Invalid(vec![TraceIssue::NonFiniteResult]);
    };
    let total_wh = propulsion_wh.plus(&auxiliary_wh);

    if !propulsion_wh.is_finite_and_ordered()
        || !auxiliary_wh.is_finite_and_ordered()
        || !total_wh.is_finite_and_ordered()
    {
        return TraceOutcome::Invalid(vec![TraceIssue::NonFiniteResult]);
    }

    let max_observed_gap_ms = spec
        .samples
        .windows(2)
        .map(|pair| pair[1].elapsed_ms - pair[0].elapsed_ms)
        .max()
        .unwrap_or(0);

    TraceOutcome::Complete(TraceEnergy {
        trace_id: spec.trace_id.clone(),
        raw_trace_sha256: spec.raw_trace_sha256.clone(),
        mission_id: spec.mission_id.clone(),
        condition_digest: spec.condition_digest.clone(),
        service_digest: spec.service_digest.clone(),
        calibration_epoch_id: spec.calibration_epoch_id.clone(),
        energy_boundary_id: spec.energy_boundary_id.clone(),
        uncertainty_method_id: spec.uncertainty_method_id.clone(),
        start_ms: spec.expected_start_ms,
        end_ms: spec.expected_end_ms,
        duration_ms: spec.expected_end_ms - spec.expected_start_ms,
        sample_count: spec.samples.len(),
        max_observed_gap_ms,
        propulsion_wh,
        auxiliary_wh,
        total_wh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(nominal_w: f64, absolute_bound_w: f64) -> PowerReading {
        PowerReading {
            nominal_w,
            absolute_bound_w,
        }
    }

    fn sample(elapsed_ms: u64, propulsion_w: f64, auxiliary_w: f64) -> PowerSample {
        PowerSample {
            elapsed_ms,
            propulsion: Some(reading(propulsion_w, 0.0)),
            auxiliary: Some(reading(auxiliary_w, 0.0)),
        }
    }

    fn spec(samples: Vec<PowerSample>, end_ms: u64, max_gap_ms: u64) -> TraceSpec {
        TraceSpec {
            trace_id: "trace-001".to_owned(),
            raw_trace_sha256: "a".repeat(64),
            mission_id: "mission-A".to_owned(),
            condition_digest: "condition-sha256:example".to_owned(),
            service_digest: "b".repeat(64),
            calibration_epoch_id: "meter-chain-epoch-001".to_owned(),
            energy_boundary_id: "dc-input-boundary-A".to_owned(),
            uncertainty_method_id: "absolute-bound-v1".to_owned(),
            expected_start_ms: 0,
            expected_end_ms: end_ms,
            max_gap_ms,
            samples,
        }
    }

    fn complete(outcome: TraceOutcome) -> TraceEnergy {
        match outcome {
            TraceOutcome::Complete(energy) => energy,
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn constant_power_integrates_exactly() {
        let trace = spec(vec![sample(0, 800.0, 200.0), sample(3_600_000, 800.0, 200.0)], 3_600_000, 3_600_000);
        let energy = complete(assess_trace(&trace));
        approx(energy.propulsion_wh.nominal_wh, 800.0);
        approx(energy.auxiliary_wh.nominal_wh, 200.0);
        approx(energy.total_wh.nominal_wh, 1_000.0);
        assert_eq!(energy.duration_ms, 3_600_000);
    }

    #[test]
    fn linear_ramp_uses_trapezoidal_rule() {
        let trace = spec(vec![sample(0, 0.0, 0.0), sample(2_000, 10.0, 0.0)], 2_000, 2_000);
        let energy = complete(assess_trace(&trace));
        approx(energy.propulsion_wh.nominal_wh, 10.0 / 3_600.0);
    }

    #[test]
    fn auxiliary_channel_is_preserved_and_added() {
        let trace = spec(vec![sample(0, 100.0, 25.0), sample(3_600_000, 100.0, 25.0)], 3_600_000, 3_600_000);
        let energy = complete(assess_trace(&trace));
        approx(energy.propulsion_wh.nominal_wh, 100.0);
        approx(energy.auxiliary_wh.nominal_wh, 25.0);
        approx(energy.total_wh.nominal_wh, 125.0);
    }

    #[test]
    fn power_bounds_propagate_conservatively() {
        let uncertain = |time| PowerSample {
            elapsed_ms: time,
            propulsion: Some(reading(100.0, 10.0)),
            auxiliary: Some(reading(20.0, 2.0)),
        };
        let trace = spec(vec![uncertain(0), uncertain(3_600_000)], 3_600_000, 3_600_000);
        let energy = complete(assess_trace(&trace));
        approx(energy.total_wh.lower_wh, 108.0);
        approx(energy.total_wh.nominal_wh, 120.0);
        approx(energy.total_wh.upper_wh, 132.0);
    }

    #[test]
    fn missing_propulsion_power_is_indeterminate_not_zero() {
        let mut trace = spec(vec![sample(0, 10.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 1000);
        trace.samples[0].propulsion = None;
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Indeterminate);
    }

    #[test]
    fn missing_auxiliary_power_is_indeterminate_not_zero() {
        let mut trace = spec(vec![sample(0, 10.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 1000);
        trace.samples[1].auxiliary = None;
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Indeterminate);
    }

    #[test]
    fn missing_start_endpoint_is_indeterminate() {
        let trace = spec(vec![sample(5, 10.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 1000);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Indeterminate);
    }

    #[test]
    fn missing_end_endpoint_is_indeterminate() {
        let trace = spec(vec![sample(0, 10.0, 2.0), sample(995, 10.0, 2.0)], 1000, 1000);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Indeterminate);
    }

    #[test]
    fn over_limit_gap_is_indeterminate() {
        let trace = spec(vec![sample(0, 10.0, 2.0), sample(2000, 10.0, 2.0)], 2000, 1000);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Indeterminate);
    }

    #[test]
    fn duplicate_timestamps_are_invalid() {
        let trace = spec(vec![sample(0, 10.0, 2.0), sample(0, 10.0, 2.0)], 1000, 1000);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Invalid);
    }

    #[test]
    fn negative_power_is_invalid() {
        let trace = spec(vec![sample(0, -1.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 1000);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Invalid);
    }

    #[test]
    fn non_finite_uncertainty_is_invalid() {
        let mut trace = spec(vec![sample(0, 10.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 1000);
        trace.samples[0].auxiliary = Some(reading(2.0, f64::INFINITY));
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Invalid);
    }

    #[test]
    fn malformed_digest_is_invalid() {
        let mut trace = spec(vec![sample(0, 10.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 1000);
        trace.raw_trace_sha256 = "not-a-hash".to_owned();
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Invalid);
    }

    #[test]
    fn zero_gap_limit_is_invalid() {
        let trace = spec(vec![sample(0, 10.0, 2.0), sample(1000, 10.0, 2.0)], 1000, 0);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Invalid);
    }

    #[test]
    fn higher_uncertainty_cannot_narrow_energy_band() {
        let make = |bound| {
            let s = |time| PowerSample {
                elapsed_ms: time,
                propulsion: Some(reading(100.0, bound)),
                auxiliary: Some(reading(10.0, bound)),
            };
            spec(vec![s(0), s(3_600_000)], 3_600_000, 3_600_000)
        };
        let narrow = complete(assess_trace(&make(2.0))).total_wh;
        let wide = complete(assess_trace(&make(5.0))).total_wh;
        assert!(wide.lower_wh <= narrow.lower_wh);
        assert!(wide.upper_wh >= narrow.upper_wh);
    }

    #[test]
    fn deterministic_replay_returns_identical_record() {
        let trace = spec(vec![sample(0, 80.0, 20.0), sample(1000, 100.0, 30.0)], 1000, 1000);
        assert_eq!(assess_trace(&trace), assess_trace(&trace));
    }

    #[test]
    fn decreasing_timestamps_are_invalid() {
        let trace = spec(vec![sample(0, 10.0, 2.0), sample(1000, 10.0, 2.0), sample(500, 10.0, 2.0)], 500, 1000);
        assert_eq!(assess_trace(&trace).disposition(), TraceDisposition::Invalid);
    }
}
