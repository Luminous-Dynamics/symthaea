// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent reference for synchronized electrical power-series integration.
//!
//! Every input channel must share an exact timestamp grid and declared clock
//! domain. The oracle uses trapezoidal integration and never resamples, fills
//! gaps, corrects clocks or silently drops observations.

use std::collections::BTreeSet;
use std::fmt;

pub const MODEL_VERSION: &str = "electric-power-series-oracle-v0.1.0";

/// Electrical load categories at non-overlapping meter boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PowerSeriesLoadKind {
    PropulsionDcInput,
    ThermalManagement,
    OtherAuxiliary,
    UpstreamDistributionLoss,
    BatteryTerminalTotal,
}

/// One power observation and its supplied absolute uncertainty bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSample {
    pub timestamp_ns: u64,
    pub power_w: f64,
    pub absolute_uncertainty_bound_w: f64,
}

/// One synchronized, evidence-bound electrical load channel.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSeriesChannel {
    pub load_id: String,
    pub meter_boundary_id: String,
    pub kind: PowerSeriesLoadKind,
    pub clock_domain_id: String,
    pub clock_uncertainty_ns: u64,
    pub evidence_manifest_digest: String,
    pub samples: Vec<PowerSample>,
}

/// Integrity gates for a multi-channel integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerSeriesConfig {
    pub expected_clock_domain_id: String,
    pub maximum_sample_gap_ns: u64,
    pub maximum_clock_uncertainty_ns: u64,
}

/// Integrated result for one load channel.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSeriesChannelResult {
    pub load_id: String,
    pub meter_boundary_id: String,
    pub kind: PowerSeriesLoadKind,
    pub peak_power_w: f64,
    pub average_power_w: f64,
    pub energy_j: f64,
    pub absolute_energy_uncertainty_bound_j: f64,
    pub energy_lower_bound_j: f64,
    pub energy_upper_bound_j: f64,
    pub clock_domain_id: String,
    pub clock_uncertainty_ns: u64,
    pub evidence_manifest_digest: String,
}

/// Integrated energy and peak-demand summary for aligned load channels.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSeriesReport {
    pub model_version: &'static str,
    pub clock_domain_id: String,
    pub start_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub duration_s: f64,
    pub battery_terminal_peak_power_w: f64,
    pub battery_terminal_average_power_w: f64,
    pub battery_terminal_energy_j: f64,
    pub absolute_energy_uncertainty_bound_j: f64,
    pub energy_lower_bound_j: f64,
    pub energy_upper_bound_j: f64,
    pub propulsion_energy_j: Option<f64>,
    pub thermal_management_energy_j: Option<f64>,
    pub auxiliary_energy_j: Option<f64>,
    pub distribution_loss_energy_j: Option<f64>,
    pub battery_terminal_meter_energy_j: Option<f64>,
    pub channels: Vec<PowerSeriesChannelResult>,
}

/// Typed non-PASS outcomes for malformed, unsynchronized or unsupported data.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerSeriesError {
    InvalidInput(&'static str),
    DuplicateIdentity(&'static str),
    OverlappingMeterBoundary,
    ClockDomainMismatch,
    TimestampGridMismatch,
    SampleGapExceeded {
        gap_ns: u64,
        maximum_gap_ns: u64,
    },
    ClockUncertaintyExceeded {
        measured_uncertainty_ns: u64,
        maximum_uncertainty_ns: u64,
    },
    NumericalOverflow,
}

impl fmt::Display for PowerSeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PowerSeriesError {}

fn add_finite(total: &mut f64, value: f64) -> Result<(), PowerSeriesError> {
    let next = *total + value;
    if !next.is_finite() {
        return Err(PowerSeriesError::NumericalOverflow);
    }
    *total = next;
    Ok(())
}

fn integrate_channel(
    channel: &PowerSeriesChannel,
    maximum_gap_ns: u64,
) -> Result<(f64, f64, f64, f64), PowerSeriesError> {
    let mut energy_j = 0.0;
    let mut uncertainty_j = 0.0;
    let mut peak_power_w: f64 = 0.0;

    for sample in &channel.samples {
        if !sample.power_w.is_finite()
            || sample.power_w < 0.0
            || !sample.absolute_uncertainty_bound_w.is_finite()
            || sample.absolute_uncertainty_bound_w < 0.0
        {
            return Err(PowerSeriesError::InvalidInput(
                "power and uncertainty must be finite and non-negative",
            ));
        }
        peak_power_w = peak_power_w.max(sample.power_w);
    }

    for pair in channel.samples.windows(2) {
        let gap_ns = pair[1]
            .timestamp_ns
            .checked_sub(pair[0].timestamp_ns)
            .filter(|gap| *gap > 0)
            .ok_or(PowerSeriesError::InvalidInput(
                "timestamps must be strictly increasing",
            ))?;
        if gap_ns > maximum_gap_ns {
            return Err(PowerSeriesError::SampleGapExceeded {
                gap_ns,
                maximum_gap_ns,
            });
        }

        let delta_s = gap_ns as f64 / 1_000_000_000.0;
        let mean_power_w = pair[0].power_w * 0.5 + pair[1].power_w * 0.5;
        let mean_uncertainty_w = pair[0].absolute_uncertainty_bound_w * 0.5
            + pair[1].absolute_uncertainty_bound_w * 0.5;
        add_finite(&mut energy_j, mean_power_w * delta_s)?;
        add_finite(&mut uncertainty_j, mean_uncertainty_w * delta_s)?;
    }

    let duration_ns = channel.samples[channel.samples.len() - 1]
        .timestamp_ns
        .checked_sub(channel.samples[0].timestamp_ns)
        .ok_or(PowerSeriesError::InvalidInput(
            "time-series end must not precede start",
        ))?;
    if duration_ns == 0 {
        return Err(PowerSeriesError::InvalidInput(
            "time-series duration must be positive",
        ));
    }
    let duration_s = duration_ns as f64 / 1_000_000_000.0;
    let average_power_w = energy_j / duration_s;
    if !average_power_w.is_finite() {
        return Err(PowerSeriesError::NumericalOverflow);
    }

    Ok((
        energy_j,
        uncertainty_j,
        peak_power_w,
        average_power_w,
    ))
}

/// Integrate synchronized electrical power channels using the trapezoidal rule.
///
/// All channels must have identical timestamps and one declared clock domain.
/// A battery-terminal-total channel is exclusive and cannot be combined with
/// branch meters. No interpolation or clock correction is performed.
pub fn integrate_power_series(
    channels: &[PowerSeriesChannel],
    config: &PowerSeriesConfig,
) -> Result<PowerSeriesReport, PowerSeriesError> {
    if config.expected_clock_domain_id.trim().is_empty()
        || config.maximum_sample_gap_ns == 0
    {
        return Err(PowerSeriesError::InvalidInput(
            "clock identity and positive maximum sample gap are required",
        ));
    }
    if channels.is_empty() {
        return Err(PowerSeriesError::InvalidInput(
            "at least one channel is required",
        ));
    }

    let mut ordered = channels.to_vec();
    ordered.sort_by(|left, right| left.load_id.cmp(&right.load_id));
    let mut load_ids = BTreeSet::new();
    let mut boundary_ids = BTreeSet::new();
    let mut reference_timestamps: Option<Vec<u64>> = None;
    let mut results = Vec::with_capacity(ordered.len());

    let mut total_energy = 0.0;
    let mut total_uncertainty = 0.0;
    let mut propulsion: Option<f64> = None;
    let mut tms: Option<f64> = None;
    let mut auxiliary: Option<f64> = None;
    let mut distribution: Option<f64> = None;
    let mut battery_total_energy: Option<f64> = None;
    let mut battery_terminal_peak_power: f64 = 0.0;
    let mut battery_total_count = 0usize;

    for channel in &ordered {
        if channel.load_id.trim().is_empty()
            || channel.meter_boundary_id.trim().is_empty()
            || channel.clock_domain_id.trim().is_empty()
            || channel.evidence_manifest_digest.trim().is_empty()
        {
            return Err(PowerSeriesError::InvalidInput(
                "load, meter, clock and evidence identities must not be empty",
            ));
        }
        if !load_ids.insert(channel.load_id.clone()) {
            return Err(PowerSeriesError::DuplicateIdentity("load ID"));
        }
        if !boundary_ids.insert(channel.meter_boundary_id.clone()) {
            return Err(PowerSeriesError::DuplicateIdentity("meter boundary ID"));
        }
        if channel.clock_domain_id != config.expected_clock_domain_id {
            return Err(PowerSeriesError::ClockDomainMismatch);
        }
        if channel.clock_uncertainty_ns > config.maximum_clock_uncertainty_ns {
            return Err(PowerSeriesError::ClockUncertaintyExceeded {
                measured_uncertainty_ns: channel.clock_uncertainty_ns,
                maximum_uncertainty_ns: config.maximum_clock_uncertainty_ns,
            });
        }
        if channel.samples.len() < 2 {
            return Err(PowerSeriesError::InvalidInput(
                "each channel requires at least two samples",
            ));
        }

        let timestamps = channel
            .samples
            .iter()
            .map(|sample| sample.timestamp_ns)
            .collect::<Vec<_>>();
        if let Some(reference) = &reference_timestamps {
            if reference != &timestamps {
                return Err(PowerSeriesError::TimestampGridMismatch);
            }
        } else {
            reference_timestamps = Some(timestamps);
        }

        let (energy, uncertainty, peak, average) =
            integrate_channel(channel, config.maximum_sample_gap_ns)?;
        add_finite(&mut total_energy, energy)?;
        add_finite(&mut total_uncertainty, uncertainty)?;

        match channel.kind {
            PowerSeriesLoadKind::PropulsionDcInput => {
                add_finite(propulsion.get_or_insert(0.0), energy)?;
            }
            PowerSeriesLoadKind::ThermalManagement => {
                add_finite(tms.get_or_insert(0.0), energy)?;
            }
            PowerSeriesLoadKind::OtherAuxiliary => {
                add_finite(auxiliary.get_or_insert(0.0), energy)?;
            }
            PowerSeriesLoadKind::UpstreamDistributionLoss => {
                add_finite(distribution.get_or_insert(0.0), energy)?;
            }
            PowerSeriesLoadKind::BatteryTerminalTotal => {
                battery_total_count += 1;
                battery_total_energy = Some(energy);
            }
        }

        results.push(PowerSeriesChannelResult {
            load_id: channel.load_id.clone(),
            meter_boundary_id: channel.meter_boundary_id.clone(),
            kind: channel.kind,
            peak_power_w: peak,
            average_power_w: average,
            energy_j: energy,
            absolute_energy_uncertainty_bound_j: uncertainty,
            energy_lower_bound_j: (energy - uncertainty).max(0.0),
            energy_upper_bound_j: energy + uncertainty,
            clock_domain_id: channel.clock_domain_id.clone(),
            clock_uncertainty_ns: channel.clock_uncertainty_ns,
            evidence_manifest_digest: channel.evidence_manifest_digest.clone(),
        });
    }

    if battery_total_count > 0 && ordered.len() != 1 {
        return Err(PowerSeriesError::OverlappingMeterBoundary);
    }

    let timestamps = reference_timestamps.ok_or(PowerSeriesError::InvalidInput(
        "no timestamp grid was supplied",
    ))?;
    let start_timestamp_ns = timestamps[0];
    let end_timestamp_ns = timestamps[timestamps.len() - 1];
    let duration_ns = end_timestamp_ns
        .checked_sub(start_timestamp_ns)
        .ok_or(PowerSeriesError::InvalidInput(
            "series end timestamp precedes start",
        ))?;
    if duration_ns == 0 {
        return Err(PowerSeriesError::InvalidInput(
            "integrated time interval must be positive",
        ));
    }
    let duration_s = duration_ns as f64 / 1_000_000_000.0;
    let mut sample_index = 0usize;
    while sample_index < timestamps.len() {
        let mut instantaneous_total = 0.0;
        for channel in &ordered {
            add_finite(
                &mut instantaneous_total,
                channel.samples[sample_index].power_w,
            )?;
        }
        battery_terminal_peak_power =
            battery_terminal_peak_power.max(instantaneous_total);
        sample_index += 1;
    }

    if battery_total_count == 1 {
        propulsion = None;
        tms = None;
        auxiliary = None;
        distribution = None;
    }

    let average_power = total_energy / duration_s;
    let energy_lower = (total_energy - total_uncertainty).max(0.0);
    let energy_upper = total_energy + total_uncertainty;
    if !average_power.is_finite() || !energy_upper.is_finite() {
        return Err(PowerSeriesError::NumericalOverflow);
    }

    Ok(PowerSeriesReport {
        model_version: MODEL_VERSION,
        clock_domain_id: config.expected_clock_domain_id.clone(),
        start_timestamp_ns,
        end_timestamp_ns,
        duration_s,
        battery_terminal_peak_power_w: battery_terminal_peak_power,
        battery_terminal_average_power_w: average_power,
        battery_terminal_energy_j: total_energy,
        absolute_energy_uncertainty_bound_j: total_uncertainty,
        energy_lower_bound_j: energy_lower,
        energy_upper_bound_j: energy_upper,
        propulsion_energy_j: propulsion,
        thermal_management_energy_j: tms,
        auxiliary_energy_j: auxiliary,
        distribution_loss_energy_j: distribution,
        battery_terminal_meter_energy_j: battery_total_energy,
        channels: results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PowerSeriesConfig {
        PowerSeriesConfig {
            expected_clock_domain_id: "ptp-grandmaster-01".to_owned(),
            maximum_sample_gap_ns: 2_000_000_000,
            maximum_clock_uncertainty_ns: 100_000,
        }
    }

    fn channel(
        load_id: &str,
        boundary: &str,
        kind: PowerSeriesLoadKind,
        powers: &[f64],
    ) -> PowerSeriesChannel {
        let samples = powers
            .iter()
            .enumerate()
            .map(|(index, power)| PowerSample {
                timestamp_ns: index as u64 * 1_000_000_000,
                power_w: *power,
                absolute_uncertainty_bound_w: 0.0,
            })
            .collect();
        PowerSeriesChannel {
            load_id: load_id.to_owned(),
            meter_boundary_id: boundary.to_owned(),
            kind,
            clock_domain_id: "ptp-grandmaster-01".to_owned(),
            clock_uncertainty_ns: 10_000,
            evidence_manifest_digest: format!("evidence-{load_id}"),
            samples,
        }
    }

    #[test]
    fn constant_power_integrates_to_exact_energy() {
        let input = channel(
            "propulsion",
            "inverter-dc-terminal",
            PowerSeriesLoadKind::PropulsionDcInput,
            &[1_000.0, 1_000.0, 1_000.0],
        );
        let report = integrate_power_series(&[input], &config()).unwrap();
        assert_eq!(report.duration_s, 2.0);
        assert_eq!(report.battery_terminal_energy_j, 2_000.0);
        assert_eq!(report.battery_terminal_average_power_w, 1_000.0);
        assert_eq!(report.battery_terminal_peak_power_w, 1_000.0);
    }

    #[test]
    fn linear_power_ramp_uses_exact_trapezoidal_energy() {
        let input = channel(
            "propulsion",
            "inverter-dc-terminal",
            PowerSeriesLoadKind::PropulsionDcInput,
            &[0.0, 100.0],
        );
        let report = integrate_power_series(&[input], &config()).unwrap();
        assert_eq!(report.battery_terminal_energy_j, 50.0);
        assert_eq!(report.battery_terminal_average_power_w, 50.0);
    }

    #[test]
    fn aligned_branch_channels_sum_energy_and_preserve_categories() {
        let propulsion = channel(
            "propulsion",
            "motor-drive-dc",
            PowerSeriesLoadKind::PropulsionDcInput,
            &[1_000.0, 1_000.0],
        );
        let tms = channel(
            "tms",
            "pump-drive-dc",
            PowerSeriesLoadKind::ThermalManagement,
            &[100.0, 200.0],
        );
        let report = integrate_power_series(&[tms, propulsion], &config()).unwrap();
        assert_eq!(report.battery_terminal_energy_j, 1_150.0);
        assert_eq!(report.propulsion_energy_j, Some(1_000.0));
        assert_eq!(report.thermal_management_energy_j, Some(150.0));
        assert_eq!(report.channels[0].load_id, "propulsion");
    }

    #[test]
    fn sample_uncertainty_integrates_to_absolute_energy_envelope() {
        let mut input = channel(
            "load",
            "meter",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[100.0, 100.0, 100.0],
        );
        input.samples[0].absolute_uncertainty_bound_w = 2.0;
        input.samples[1].absolute_uncertainty_bound_w = 4.0;
        input.samples[2].absolute_uncertainty_bound_w = 2.0;
        let report = integrate_power_series(&[input], &config()).unwrap();
        assert_eq!(report.battery_terminal_energy_j, 200.0);
        assert_eq!(report.absolute_energy_uncertainty_bound_j, 6.0);
        assert_eq!(report.energy_lower_bound_j, 194.0);
        assert_eq!(report.energy_upper_bound_j, 206.0);
    }

    #[test]
    fn battery_terminal_total_is_exclusive_and_branch_subtotals_unknown() {
        let input = channel(
            "battery",
            "battery-terminal",
            PowerSeriesLoadKind::BatteryTerminalTotal,
            &[1_200.0, 1_200.0],
        );
        let report = integrate_power_series(&[input], &config()).unwrap();
        assert_eq!(report.battery_terminal_meter_energy_j, Some(1_200.0));
        assert_eq!(report.propulsion_energy_j, None);
        assert_eq!(report.thermal_management_energy_j, None);
    }

    #[test]
    fn battery_terminal_total_cannot_be_added_to_branch_channels() {
        let total = channel(
            "battery",
            "battery-terminal",
            PowerSeriesLoadKind::BatteryTerminalTotal,
            &[1_200.0, 1_200.0],
        );
        let tms = channel(
            "tms",
            "tms-terminal",
            PowerSeriesLoadKind::ThermalManagement,
            &[100.0, 100.0],
        );
        assert_eq!(
            integrate_power_series(&[total, tms], &config()),
            Err(PowerSeriesError::OverlappingMeterBoundary)
        );
    }

    #[test]
    fn different_clock_domains_are_rejected() {
        let mut input = channel(
            "load",
            "meter",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[1.0, 1.0],
        );
        input.clock_domain_id = "other-clock".to_owned();
        assert_eq!(
            integrate_power_series(&[input], &config()),
            Err(PowerSeriesError::ClockDomainMismatch)
        );
    }

    #[test]
    fn timestamp_grid_mismatch_is_rejected() {
        let first = channel(
            "first",
            "meter-first",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[10.0, 10.0],
        );
        let mut second = channel(
            "second",
            "meter-second",
            PowerSeriesLoadKind::ThermalManagement,
            &[20.0, 20.0],
        );
        second.samples[1].timestamp_ns += 1;
        assert_eq!(
            integrate_power_series(&[first, second], &config()),
            Err(PowerSeriesError::TimestampGridMismatch)
        );
    }

    #[test]
    fn duplicate_timestamp_and_overlong_sample_gap_fail() {
        let mut duplicate = channel(
            "duplicate",
            "meter-duplicate",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[10.0, 10.0],
        );
        duplicate.samples[1].timestamp_ns = duplicate.samples[0].timestamp_ns;
        assert!(matches!(
            integrate_power_series(&[duplicate], &config()),
            Err(PowerSeriesError::InvalidInput(_))
        ));

        let mut gap = channel(
            "gap",
            "meter-gap",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[10.0, 10.0],
        );
        gap.samples[1].timestamp_ns = 3_000_000_000;
        assert!(matches!(
            integrate_power_series(&[gap], &config()),
            Err(PowerSeriesError::SampleGapExceeded { .. })
        ));
    }

    #[test]
    fn excessive_clock_uncertainty_and_missing_evidence_fail() {
        let mut input = channel(
            "load",
            "meter",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[1.0, 1.0],
        );
        input.clock_uncertainty_ns = 200_000;
        assert!(matches!(
            integrate_power_series(&[input.clone()], &config()),
            Err(PowerSeriesError::ClockUncertaintyExceeded { .. })
        ));
        input.clock_uncertainty_ns = 10_000;
        input.evidence_manifest_digest.clear();
        assert!(matches!(
            integrate_power_series(&[input], &config()),
            Err(PowerSeriesError::InvalidInput(_))
        ));
    }

    #[test]
    fn transient_peak_is_not_hidden_by_average_power() {
        let input = channel(
            "propulsion",
            "inverter-dc-terminal",
            PowerSeriesLoadKind::PropulsionDcInput,
            &[0.0, 200.0, 0.0],
        );
        let report = integrate_power_series(&[input], &config()).unwrap();
        assert_eq!(report.battery_terminal_energy_j, 200.0);
        assert_eq!(report.battery_terminal_average_power_w, 100.0);
        assert_eq!(report.battery_terminal_peak_power_w, 200.0);
    }

    #[test]
    fn duplicate_ids_and_boundaries_are_rejected() {
        let duplicate_load = vec![
            channel("same", "meter-a", PowerSeriesLoadKind::OtherAuxiliary, &[1.0, 1.0]),
            channel("same", "meter-b", PowerSeriesLoadKind::OtherAuxiliary, &[2.0, 2.0]),
        ];
        assert_eq!(
            integrate_power_series(&duplicate_load, &config()),
            Err(PowerSeriesError::DuplicateIdentity("load ID"))
        );

        let duplicate_boundary = vec![
            channel("load-a", "same-meter", PowerSeriesLoadKind::OtherAuxiliary, &[1.0, 1.0]),
            channel("load-b", "same-meter", PowerSeriesLoadKind::OtherAuxiliary, &[2.0, 2.0]),
        ];
        assert_eq!(
            integrate_power_series(&duplicate_boundary, &config()),
            Err(PowerSeriesError::DuplicateIdentity("meter boundary ID"))
        );
    }

    #[test]
    fn input_channel_order_does_not_change_integrated_report() {
        let a = channel("a", "meter-a", PowerSeriesLoadKind::PropulsionDcInput, &[100.0, 200.0]);
        let b = channel("b", "meter-b", PowerSeriesLoadKind::ThermalManagement, &[10.0, 20.0]);
        let first = integrate_power_series(&[a.clone(), b.clone()], &config()).unwrap();
        let second = integrate_power_series(&[b, a], &config()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn energy_overflow_fails_closed() {
        let input = channel(
            "large",
            "large-meter",
            PowerSeriesLoadKind::OtherAuxiliary,
            &[f64::MAX, f64::MAX, f64::MAX],
        );
        let mut cfg = config();
        cfg.maximum_sample_gap_ns = 2_000_000_000;
        assert_eq!(
            integrate_power_series(&[input], &cfg),
            Err(PowerSeriesError::NumericalOverflow)
        );
    }
}
