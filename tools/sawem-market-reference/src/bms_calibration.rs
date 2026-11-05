//! Deterministic calibration of conservative BMS capability-cell limits.
//!
//! Only BMS-reported available-power telemetry and explicit capability-limit
//! tests can contribute to a cell fit. Ordinary delivered dispatch power is
//! censored by the operator's request and is not a valid maximum-capability
//! observation. The fitter uses a lower empirical order statistic and requires
//! a minimum number of source-bound observations.
//!
//! This module computes one cell's numerical limit. The caller must freeze a
//! canonical cell evidence manifest and supply its source reference/digest when
//! constructing a BatteryCapabilitySurface cell.

use std::collections::BTreeSet;

use crate::bms_capability::PowerDirection;

const BASIS_POINTS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityMeasurementKind {
    BmsReportedAvailablePower,
    CapabilityLimitTest,
    OrdinaryDeliveredPower,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityMeasurement<'a> {
    pub record_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub campaign_id: &'a str,
    pub observed_at_utc: i64,
    pub available_at_utc: i64,
    pub soc_bps: u16,
    pub temperature_deci_c: i16,
    pub soh_bps: u16,
    pub direction: PowerDirection,
    pub available_power_w: u64,
    pub measurement_kind: CapabilityMeasurementKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityCellBounds {
    pub soc_lower_bps: u16,
    pub soc_upper_bps: u16,
    pub temperature_lower_deci_c: i16,
    pub temperature_upper_deci_c: i16,
    pub soh_lower_bps: u16,
    pub soh_upper_bps: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityCellFitInputs<'a> {
    pub battery_unit_id: &'a str,
    pub cell_source_ref: &'a str,
    pub cell_source_sha256: &'a str,
    pub calibration_campaign_id: &'a str,
    pub calibration_as_of_utc: i64,
    pub rated_power_w: u64,
    pub bounds: CapabilityCellBounds,
    pub direction: PowerDirection,
    /// Lower empirical quantile in basis points. 500 = lower 5th percentile.
    pub lower_quantile_bps: u16,
    pub minimum_sample_count: u32,
    pub observations: &'a [CapabilityMeasurement<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityCellFitError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidRatedPower,
    InvalidCellBounds,
    InvalidQuantile,
    MissingObservations,
    InsufficientSamples,
    DuplicateRecordId,
    DuplicateSourceReference,
    WrongCampaign,
    WrongDirection,
    ObservationOutsideCell,
    FutureKnownObservation,
    InvalidObservationChronology,
    InvalidMeasurementKind,
    ObservationPowerAboveRated,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCellFit<'a> {
    pub battery_unit_id: &'a str,
    pub cell_source_ref: &'a str,
    pub cell_source_sha256: &'a str,
    pub calibration_campaign_id: &'a str,
    pub calibration_as_of_utc: i64,
    pub bounds: CapabilityCellBounds,
    pub direction: PowerDirection,
    pub lower_quantile_bps: u16,
    pub sample_count: u32,
    /// Zero-based rank = floor(q * (n - 1) / 10,000).
    pub selected_order_statistic_rank: u32,
    pub conservative_power_limit_w: u64,
    /// Source record selected by the lower order statistic.
    pub selected_measurement_record_id: &'a str,
    pub source_manifest_required: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn in_cell(
    bounds: CapabilityCellBounds,
    observation: CapabilityMeasurement<'_>,
) -> bool {
    observation.soc_bps >= bounds.soc_lower_bps
        && observation.soc_bps < bounds.soc_upper_bps
        && observation.temperature_deci_c >= bounds.temperature_lower_deci_c
        && observation.temperature_deci_c < bounds.temperature_upper_deci_c
        && observation.soh_bps >= bounds.soh_lower_bps
        && observation.soh_bps < bounds.soh_upper_bps
}

fn validate_bounds(bounds: CapabilityCellBounds) -> Result<(), CapabilityCellFitError> {
    if bounds.soc_lower_bps >= bounds.soc_upper_bps
        || bounds.soc_upper_bps > 10_000
        || bounds.soh_lower_bps >= bounds.soh_upper_bps
        || bounds.soh_upper_bps > 10_000
        || bounds.temperature_lower_deci_c >= bounds.temperature_upper_deci_c
    {
        return Err(CapabilityCellFitError::InvalidCellBounds);
    }
    Ok(())
}

/// Fit one direction-specific cell from source-bound capability observations.
/// The supplied cell-source digest must identify a frozen manifest containing
/// the full contributing sample set and its declared selection method.
pub fn fit_capability_cell<'a>(
    inputs: CapabilityCellFitInputs<'a>,
) -> Result<CapabilityCellFit<'a>, CapabilityCellFitError> {
    if !nonempty(inputs.battery_unit_id)
        || !nonempty(inputs.cell_source_ref)
        || !nonempty(inputs.calibration_campaign_id)
    {
        return Err(CapabilityCellFitError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.cell_source_sha256) {
        return Err(CapabilityCellFitError::InvalidSourceDigest);
    }
    if inputs.rated_power_w == 0 {
        return Err(CapabilityCellFitError::InvalidRatedPower);
    }
    validate_bounds(inputs.bounds)?;
    if inputs.lower_quantile_bps == 0 || inputs.lower_quantile_bps > 5_000 {
        return Err(CapabilityCellFitError::InvalidQuantile);
    }
    if inputs.minimum_sample_count == 0 {
        return Err(CapabilityCellFitError::InsufficientSamples);
    }
    if inputs.observations.is_empty() {
        return Err(CapabilityCellFitError::MissingObservations);
    }

    let mut record_ids = BTreeSet::new();
    let mut source_refs = BTreeSet::new();
    let mut powers = Vec::new();
    let mut power_record_ids = Vec::new();

    for observation in inputs.observations.iter().copied() {
        if !nonempty(observation.record_id)
            || !nonempty(observation.source_ref)
            || !nonempty(observation.campaign_id)
        {
            return Err(CapabilityCellFitError::InvalidEvidenceReference);
        }
        if !valid_sha256(observation.source_sha256) {
            return Err(CapabilityCellFitError::InvalidSourceDigest);
        }
        if !record_ids.insert(observation.record_id) {
            return Err(CapabilityCellFitError::DuplicateRecordId);
        }
        if !source_refs.insert(observation.source_ref) {
            return Err(CapabilityCellFitError::DuplicateSourceReference);
        }
        if observation.campaign_id != inputs.calibration_campaign_id {
            return Err(CapabilityCellFitError::WrongCampaign);
        }
        if observation.direction != inputs.direction {
            return Err(CapabilityCellFitError::WrongDirection);
        }
        if !in_cell(inputs.bounds, observation) {
            return Err(CapabilityCellFitError::ObservationOutsideCell);
        }
        if observation.observed_at_utc > observation.available_at_utc {
            return Err(CapabilityCellFitError::InvalidObservationChronology);
        }
        if observation.available_at_utc > inputs.calibration_as_of_utc {
            return Err(CapabilityCellFitError::FutureKnownObservation);
        }
        if observation.available_power_w > inputs.rated_power_w {
            return Err(CapabilityCellFitError::ObservationPowerAboveRated);
        }
        match observation.measurement_kind {
            CapabilityMeasurementKind::BmsReportedAvailablePower
            | CapabilityMeasurementKind::CapabilityLimitTest => {}
            CapabilityMeasurementKind::OrdinaryDeliveredPower => {
                return Err(CapabilityCellFitError::InvalidMeasurementKind);
            }
        }
        powers.push(observation.available_power_w);
        power_record_ids.push(observation.record_id);
    }

    let sample_count = u32::try_from(powers.len())
        .map_err(|_| CapabilityCellFitError::ArithmeticOverflow)?;
    if sample_count < inputs.minimum_sample_count {
        return Err(CapabilityCellFitError::InsufficientSamples);
    }

    let mut order = (0..powers.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| powers[*index]);
    let rank_numerator = u128::from(inputs.lower_quantile_bps)
        .checked_mul(u128::from(sample_count.saturating_sub(1)))
        .ok_or(CapabilityCellFitError::ArithmeticOverflow)?;
    let rank = rank_numerator / BASIS_POINTS;
    let rank_usize = usize::try_from(rank).map_err(|_| CapabilityCellFitError::ArithmeticOverflow)?;
    let selected_index = *order
        .get(rank_usize)
        .ok_or(CapabilityCellFitError::ArithmeticOverflow)?;

    Ok(CapabilityCellFit {
        battery_unit_id: inputs.battery_unit_id,
        cell_source_ref: inputs.cell_source_ref,
        cell_source_sha256: inputs.cell_source_sha256,
        calibration_campaign_id: inputs.calibration_campaign_id,
        calibration_as_of_utc: inputs.calibration_as_of_utc,
        bounds: inputs.bounds,
        direction: inputs.direction,
        lower_quantile_bps: inputs.lower_quantile_bps,
        sample_count,
        selected_order_statistic_rank: u32::try_from(rank)
            .map_err(|_| CapabilityCellFitError::ArithmeticOverflow)?,
        conservative_power_limit_w: powers[selected_index],
        selected_measurement_record_id: power_record_ids[selected_index],
        source_manifest_required: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    const BOUNDS: CapabilityCellBounds = CapabilityCellBounds {
        soc_lower_bps: 1_000,
        soc_upper_bps: 5_000,
        temperature_lower_deci_c: 0,
        temperature_upper_deci_c: 300,
        soh_lower_bps: 8_000,
        soh_upper_bps: 10_000,
    };

    fn observation(
        id: &'static str,
        power_w: u64,
        kind: CapabilityMeasurementKind,
    ) -> CapabilityMeasurement<'static> {
        CapabilityMeasurement {
            record_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            campaign_id: "campaign/test-1",
            observed_at_utc: 90,
            available_at_utc: 95,
            soc_bps: 3_000,
            temperature_deci_c: 250,
            soh_bps: 9_000,
            direction: PowerDirection::Discharge,
            available_power_w: power_w,
            measurement_kind: kind,
        }
    }

    fn inputs<'a>(
        observations: &'a [CapabilityMeasurement<'a>],
    ) -> CapabilityCellFitInputs<'a> {
        CapabilityCellFitInputs {
            battery_unit_id: "LMO-01",
            cell_source_ref: "manifest/campaign-test-1/cell-000",
            cell_source_sha256: SOURCE_SHA,
            calibration_campaign_id: "campaign/test-1",
            calibration_as_of_utc: 100,
            rated_power_w: 5_000,
            bounds: BOUNDS,
            direction: PowerDirection::Discharge,
            lower_quantile_bps: 2_000,
            minimum_sample_count: 5,
            observations,
        }
    }

    #[test]
    fn lower_empirical_quantile_uses_exact_floor_rank() {
        let observations = [
            observation("sample-1", 1_000, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-2", 1_200, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-3", 1_400, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-4", 1_600, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-5", 1_800, CapabilityMeasurementKind::BmsReportedAvailablePower),
        ];
        let result = fit_capability_cell(inputs(&observations)).unwrap();
        // floor(0.20 * (5 - 1)) = rank 0, so this conservative limit is 1,000 W.
        assert_eq!(result.selected_order_statistic_rank, 0);
        assert_eq!(result.conservative_power_limit_w, 1_000);
        assert_eq!(result.selected_measurement_record_id, "sample-1");
        assert_eq!(result.sample_count, 5);
        assert!(result.source_manifest_required);
    }

    #[test]
    fn higher_quantile_uses_deterministic_order_statistic() {
        let observations = [
            observation("sample-1", 1_000, CapabilityMeasurementKind::CapabilityLimitTest),
            observation("sample-2", 1_200, CapabilityMeasurementKind::CapabilityLimitTest),
            observation("sample-3", 1_400, CapabilityMeasurementKind::CapabilityLimitTest),
            observation("sample-4", 1_600, CapabilityMeasurementKind::CapabilityLimitTest),
            observation("sample-5", 1_800, CapabilityMeasurementKind::CapabilityLimitTest),
        ];
        let inputs = CapabilityCellFitInputs {
            lower_quantile_bps: 5_000,
            ..inputs(&observations)
        };
        let result = fit_capability_cell(inputs).unwrap();
        // floor(0.50 * (5 - 1)) = rank 2.
        assert_eq!(result.selected_order_statistic_rank, 2);
        assert_eq!(result.conservative_power_limit_w, 1_400);
    }

    #[test]
    fn ordinary_delivered_power_is_not_a_capability_observation() {
        let observations = [
            observation("sample-1", 1_000, CapabilityMeasurementKind::OrdinaryDeliveredPower),
            observation("sample-2", 1_200, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-3", 1_400, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-4", 1_600, CapabilityMeasurementKind::BmsReportedAvailablePower),
            observation("sample-5", 1_800, CapabilityMeasurementKind::BmsReportedAvailablePower),
        ];
        assert_eq!(
            fit_capability_cell(inputs(&observations)),
            Err(CapabilityCellFitError::InvalidMeasurementKind)
        );
    }

    #[test]
    fn sample_count_campaign_direction_and_cell_bounds_are_enforced() {
        let observations = [
            observation("sample-1", 1_000, CapabilityMeasurementKind::CapabilityLimitTest),
            observation("sample-2", 1_200, CapabilityMeasurementKind::CapabilityLimitTest),
        ];
        assert_eq!(
            fit_capability_cell(inputs(&observations)),
            Err(CapabilityCellFitError::InsufficientSamples)
        );

        let wrong_campaign = [CapabilityMeasurement {
            campaign_id: "campaign/test-2",
            ..observation("wrong-campaign", 1_000, CapabilityMeasurementKind::CapabilityLimitTest)
        }];
        assert_eq!(
            fit_capability_cell(inputs(&wrong_campaign)),
            Err(CapabilityCellFitError::WrongCampaign)
        );

        let wrong_direction = [CapabilityMeasurement {
            direction: PowerDirection::Charge,
            ..observation("wrong-direction", 1_000, CapabilityMeasurementKind::CapabilityLimitTest)
        }];
        assert_eq!(
            fit_capability_cell(inputs(&wrong_direction)),
            Err(CapabilityCellFitError::WrongDirection)
        );

        let outside = [CapabilityMeasurement {
            soc_bps: 5_000,
            ..observation("outside-cell", 1_000, CapabilityMeasurementKind::CapabilityLimitTest)
        }];
        assert_eq!(
            fit_capability_cell(inputs(&outside)),
            Err(CapabilityCellFitError::ObservationOutsideCell)
        );
    }

    #[test]
    fn future_known_duplicate_and_bad_digest_samples_fail_closed() {
        let future = [CapabilityMeasurement {
            available_at_utc: 101,
            ..observation("future", 1_000, CapabilityMeasurementKind::CapabilityLimitTest)
        }];
        assert_eq!(
            fit_capability_cell(inputs(&future)),
            Err(CapabilityCellFitError::FutureKnownObservation)
        );

        let duplicate = [
            observation("same", 1_000, CapabilityMeasurementKind::CapabilityLimitTest),
            CapabilityMeasurement {
                source_ref: "duplicate-source",
                ..observation("same", 1_200, CapabilityMeasurementKind::CapabilityLimitTest)
            },
        ];
        assert_eq!(
            fit_capability_cell(inputs(&duplicate)),
            Err(CapabilityCellFitError::DuplicateRecordId)
        );

        let bad_digest = [CapabilityMeasurement {
            source_sha256: "invalid",
            ..observation("bad-digest", 1_000, CapabilityMeasurementKind::CapabilityLimitTest)
        }];
        assert_eq!(
            fit_capability_cell(inputs(&bad_digest)),
            Err(CapabilityCellFitError::InvalidSourceDigest)
        );
    }
}
