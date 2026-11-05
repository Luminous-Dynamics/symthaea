//! Multi-dimensional battery capability surface indexed by SOC, temperature and SOH.
//!
//! Unlike a smooth extrapolating model, this evaluator uses a complete grid of
//! conservative operating cells. Every requested state must fall inside one
//! explicitly evidenced cell. Cell boundaries are half-open except for the
//! final axis endpoint. Missing cells, duplicate cell coordinates, insufficient
//! sample counts and out-of-range state are rejected.
//!
//! Field-measured surfaces should be built from a frozen calibration campaign,
//! with a declared lower-quantile rule and a minimum sample count per cell.
//! This module validates and evaluates those cell limits; it does not fit the
//! envelope from raw telemetry or establish electrochemical safety by itself.

use std::collections::BTreeSet;

use crate::bms_capability::CapabilityEvidenceBasis;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySurfaceCell<'a> {
    pub soc_bin_index: u16,
    pub temperature_bin_index: u16,
    pub soh_bin_index: u16,
    /// Conservative AC-side charge power limit for the whole cell.
    pub maximum_charge_power_w: u64,
    /// Conservative AC-side discharge power limit for the whole cell.
    pub maximum_discharge_power_w: u64,
    /// Number of field observations summarized into this cell. For
    /// manufacturer guarantees and synthetic cells this is zero.
    pub sample_count: u32,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryCapabilitySurface<'a> {
    pub battery_unit_id: &'a str,
    pub surface_source_ref: &'a str,
    pub surface_source_sha256: &'a str,
    pub rated_power_w: u64,
    /// SOC breakpoints in basis points, strictly increasing.
    pub soc_breakpoints_bps: &'a [u16],
    /// Temperature breakpoints in deci-degrees Celsius, strictly increasing.
    pub temperature_breakpoints_deci_c: &'a [i16],
    /// State-of-health breakpoints in basis points, strictly increasing.
    pub soh_breakpoints_bps: &'a [u16],
    pub evidence_basis: CapabilityEvidenceBasis,
    /// Required for FieldMeasuredConservative, e.g. 500 = lower 5th percentile.
    /// Not applicable to manufacturer-guaranteed or synthetic cells.
    pub lower_quantile_bps: Option<u16>,
    /// Minimum per-cell field sample count. Ignored for non-field evidence.
    pub minimum_cell_sample_count: u32,
    pub cells: &'a [CapabilitySurfaceCell<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySurfaceError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidRatedPower,
    InvalidAxis,
    AxisNotStrictlyIncreasing,
    InvalidQuantile,
    MissingFieldSampleCount,
    InsufficientCellSamples,
    InvalidCellCoordinate,
    DuplicateCellCoordinate,
    DuplicateCellSourceReference,
    IncompleteCellGrid,
    CapabilityAboveRatedPower,
    StateOutsideCalibratedSurface,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySurfaceAssessment<'a> {
    pub battery_unit_id: &'a str,
    pub soc_bps: u16,
    pub temperature_deci_c: i16,
    pub soh_bps: u16,
    pub direction: crate::bms_capability::PowerDirection,
    pub requested_power_w: u64,
    pub maximum_supported_power_w: u64,
    pub disposition: crate::bms_capability::CapabilityDisposition,
    /// This only distinguishes synthetic from non-synthetic evidence labels.
    /// It is not an authenticity proof, BMS approval or operational permit.
    pub evidence_basis_is_non_synthetic: bool,
    pub evidence_basis: CapabilityEvidenceBasis,
    pub lower_quantile_bps: Option<u16>,
    pub selected_cell: (u16, u16, u16),
    pub cell_source_ref: &'a str,
    pub cell_source_sha256: &'a str,
    pub surface_source_ref: &'a str,
    pub surface_source_sha256: &'a str,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_axis_u16(axis: &[u16]) -> Result<(), CapabilitySurfaceError> {
    if axis.len() < 2 || axis.iter().any(|value| *value > 10_000) {
        return Err(CapabilitySurfaceError::InvalidAxis);
    }
    if axis.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CapabilitySurfaceError::AxisNotStrictlyIncreasing);
    }
    Ok(())
}

fn validate_temperature_axis(axis: &[i16]) -> Result<(), CapabilitySurfaceError> {
    if axis.len() < 2 {
        return Err(CapabilitySurfaceError::InvalidAxis);
    }
    if axis.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CapabilitySurfaceError::AxisNotStrictlyIncreasing);
    }
    Ok(())
}

fn containing_bin_u16(axis: &[u16], value: u16) -> Option<u16> {
    if value < *axis.first()? || value > *axis.last()? {
        return None;
    }
    if value == *axis.last()? {
        return u16::try_from(axis.len().checked_sub(2)?).ok();
    }
    axis.windows(2)
        .position(|pair| pair[0] <= value && value < pair[1])
        .and_then(|index| u16::try_from(index).ok())
}

fn containing_bin_temperature(axis: &[i16], value: i16) -> Option<u16> {
    if value < *axis.first()? || value > *axis.last()? {
        return None;
    }
    if value == *axis.last()? {
        return u16::try_from(axis.len().checked_sub(2)?).ok();
    }
    axis.windows(2)
        .position(|pair| pair[0] <= value && value < pair[1])
        .and_then(|index| u16::try_from(index).ok())
}

fn expected_cell_count(
    soc_bins: usize,
    temperature_bins: usize,
    soh_bins: usize,
) -> Result<usize, CapabilitySurfaceError> {
    soc_bins
        .checked_mul(temperature_bins)
        .and_then(|value| value.checked_mul(soh_bins))
        .ok_or(CapabilitySurfaceError::ArithmeticOverflow)
}

fn validate_surface(
    surface: BatteryCapabilitySurface<'_>,
) -> Result<(), CapabilitySurfaceError> {
    if !nonempty(surface.battery_unit_id) || !nonempty(surface.surface_source_ref) {
        return Err(CapabilitySurfaceError::InvalidEvidenceReference);
    }
    if !valid_sha256(surface.surface_source_sha256) {
        return Err(CapabilitySurfaceError::InvalidSourceDigest);
    }
    if surface.rated_power_w == 0 {
        return Err(CapabilitySurfaceError::InvalidRatedPower);
    }

    validate_axis_u16(surface.soc_breakpoints_bps)?;
    validate_temperature_axis(surface.temperature_breakpoints_deci_c)?;
    validate_axis_u16(surface.soh_breakpoints_bps)?;

    match surface.evidence_basis {
        CapabilityEvidenceBasis::FieldMeasuredConservative => {
            let quantile = surface
                .lower_quantile_bps
                .ok_or(CapabilitySurfaceError::InvalidQuantile)?;
            if quantile == 0 || quantile > 5_000 {
                return Err(CapabilitySurfaceError::InvalidQuantile);
            }
            if surface.minimum_cell_sample_count == 0 {
                return Err(CapabilitySurfaceError::MissingFieldSampleCount);
            }
        }
        CapabilityEvidenceBasis::ManufacturerGuaranteed
        | CapabilityEvidenceBasis::SyntheticAssumption => {
            if surface.lower_quantile_bps.is_some() {
                return Err(CapabilitySurfaceError::InvalidQuantile);
            }
        }
    }

    let soc_bins = surface.soc_breakpoints_bps.len() - 1;
    let temperature_bins = surface.temperature_breakpoints_deci_c.len() - 1;
    let soh_bins = surface.soh_breakpoints_bps.len() - 1;
    let expected = expected_cell_count(soc_bins, temperature_bins, soh_bins)?;
    if surface.cells.len() != expected {
        return Err(CapabilitySurfaceError::IncompleteCellGrid);
    }

    let mut coordinates = BTreeSet::new();
    let mut source_refs = BTreeSet::new();
    for cell in surface.cells {
        if usize::from(cell.soc_bin_index) >= soc_bins
            || usize::from(cell.temperature_bin_index) >= temperature_bins
            || usize::from(cell.soh_bin_index) >= soh_bins
        {
            return Err(CapabilitySurfaceError::InvalidCellCoordinate);
        }
        if !coordinates.insert((
            cell.soc_bin_index,
            cell.temperature_bin_index,
            cell.soh_bin_index,
        )) {
            return Err(CapabilitySurfaceError::DuplicateCellCoordinate);
        }
        if !nonempty(cell.source_ref) {
            return Err(CapabilitySurfaceError::InvalidEvidenceReference);
        }
        if !valid_sha256(cell.source_sha256) {
            return Err(CapabilitySurfaceError::InvalidSourceDigest);
        }
        if !source_refs.insert(cell.source_ref) {
            return Err(CapabilitySurfaceError::DuplicateCellSourceReference);
        }
        if cell.maximum_charge_power_w > surface.rated_power_w
            || cell.maximum_discharge_power_w > surface.rated_power_w
        {
            return Err(CapabilitySurfaceError::CapabilityAboveRatedPower);
        }
        if surface.evidence_basis == CapabilityEvidenceBasis::FieldMeasuredConservative
            && cell.sample_count < surface.minimum_cell_sample_count
        {
            return Err(CapabilitySurfaceError::InsufficientCellSamples);
        }
    }
    if coordinates.len() != expected {
        return Err(CapabilitySurfaceError::IncompleteCellGrid);
    }
    Ok(())
}

/// Evaluate a state inside a fully evidenced SOC × temperature × SOH cell.
/// No interpolation or extrapolation occurs: the returned limits are the
/// precomputed conservative cell limits provided by the evidence producer.
pub fn assess_capability_surface<'a>(
    surface: BatteryCapabilitySurface<'a>,
    soc_bps: u16,
    temperature_deci_c: i16,
    soh_bps: u16,
    direction: crate::bms_capability::PowerDirection,
    requested_power_w: u64,
) -> Result<CapabilitySurfaceAssessment<'a>, CapabilitySurfaceError> {
    validate_surface(surface)?;

    let soc_bin = containing_bin_u16(surface.soc_breakpoints_bps, soc_bps)
        .ok_or(CapabilitySurfaceError::StateOutsideCalibratedSurface)?;
    let temperature_bin =
        containing_bin_temperature(surface.temperature_breakpoints_deci_c, temperature_deci_c)
            .ok_or(CapabilitySurfaceError::StateOutsideCalibratedSurface)?;
    let soh_bin = containing_bin_u16(surface.soh_breakpoints_bps, soh_bps)
        .ok_or(CapabilitySurfaceError::StateOutsideCalibratedSurface)?;

    let cell = surface
        .cells
        .iter()
        .find(|cell| {
            cell.soc_bin_index == soc_bin
                && cell.temperature_bin_index == temperature_bin
                && cell.soh_bin_index == soh_bin
        })
        .ok_or(CapabilitySurfaceError::IncompleteCellGrid)?;

    let maximum_supported_power_w = match direction {
        crate::bms_capability::PowerDirection::Charge => cell.maximum_charge_power_w,
        crate::bms_capability::PowerDirection::Discharge => cell.maximum_discharge_power_w,
    };
    let disposition = if requested_power_w <= maximum_supported_power_w {
        crate::bms_capability::CapabilityDisposition::WithinEnvelope
    } else {
        crate::bms_capability::CapabilityDisposition::DerateRequired
    };

    Ok(CapabilitySurfaceAssessment {
        battery_unit_id: surface.battery_unit_id,
        soc_bps,
        temperature_deci_c,
        soh_bps,
        direction,
        requested_power_w,
        maximum_supported_power_w,
        disposition,
        evidence_basis_is_non_synthetic: surface.evidence_basis
            != CapabilityEvidenceBasis::SyntheticAssumption,
        evidence_basis: surface.evidence_basis,
        lower_quantile_bps: surface.lower_quantile_bps,
        selected_cell: (soc_bin, temperature_bin, soh_bin),
        cell_source_ref: cell.source_ref,
        cell_source_sha256: cell.source_sha256,
        surface_source_ref: surface.surface_source_ref,
        surface_source_sha256: surface.surface_source_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SOC: [u16; 3] = [1_000, 5_000, 9_000];
    const TEMPERATURE: [i16; 3] = [-100, 250, 450];
    const SOH: [u16; 3] = [7_000, 8_500, 10_000];

    fn cell(
        id: &'static str,
        soc: u16,
        temperature: u16,
        soh: u16,
        charge: u64,
        discharge: u64,
        samples: u32,
    ) -> CapabilitySurfaceCell<'static> {
        CapabilitySurfaceCell {
            soc_bin_index: soc,
            temperature_bin_index: temperature,
            soh_bin_index: soh,
            maximum_charge_power_w: charge,
            maximum_discharge_power_w: discharge,
            sample_count: samples,
            source_ref: id,
            source_sha256: SOURCE_SHA,
        }
    }

    const CELLS: [CapabilitySurfaceCell<'static>; 8] = [
        cell("cell-000", 0, 0, 0, 3_000, 3_500, 40),
        cell("cell-001", 0, 0, 1, 2_800, 3_300, 40),
        cell("cell-010", 0, 1, 0, 2_500, 3_100, 40),
        cell("cell-011", 0, 1, 1, 2_300, 2_900, 40),
        cell("cell-100", 1, 0, 0, 3_500, 4_000, 40),
        cell("cell-101", 1, 0, 1, 3_200, 3_700, 40),
        cell("cell-110", 1, 1, 0, 2_000, 2_500, 40),
        cell("cell-111", 1, 1, 1, 1_500, 2_000, 40),
    ];

    fn surface<'a>(
        basis: CapabilityEvidenceBasis,
        cells: &'a [CapabilitySurfaceCell<'a>],
    ) -> BatteryCapabilitySurface<'a> {
        BatteryCapabilitySurface {
            battery_unit_id: "LMO-01",
            surface_source_ref: "bms/surface/campaign-01",
            surface_source_sha256: SOURCE_SHA,
            rated_power_w: 5_000,
            soc_breakpoints_bps: &SOC,
            temperature_breakpoints_deci_c: &TEMPERATURE,
            soh_breakpoints_bps: &SOH,
            evidence_basis: basis,
            lower_quantile_bps: if basis == CapabilityEvidenceBasis::FieldMeasuredConservative {
                Some(500)
            } else {
                None
            },
            minimum_cell_sample_count: if basis == CapabilityEvidenceBasis::FieldMeasuredConservative {
                20
            } else {
                0
            },
            cells,
        }
    }

    #[test]
    fn selects_directional_power_limit_from_soc_temperature_soh_cell() {
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &CELLS);
        let result = assess_capability_surface(
            profile,
            6_000,
            300,
            9_000,
            crate::bms_capability::PowerDirection::Charge,
            1_400,
        )
        .unwrap();
        assert_eq!(result.selected_cell, (1, 1, 1));
        assert_eq!(result.maximum_supported_power_w, 1_500);
        assert_eq!(result.disposition, crate::bms_capability::CapabilityDisposition::WithinEnvelope);
        assert_eq!(result.lower_quantile_bps, Some(500));
        assert!(result.evidence_basis_is_non_synthetic);
    }

    #[test]
    fn final_axis_endpoints_map_to_last_cell_without_extrapolation() {
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &CELLS);
        let result = assess_capability_surface(
            profile,
            9_000,
            450,
            10_000,
            crate::bms_capability::PowerDirection::Discharge,
            2_100,
        )
        .unwrap();
        assert_eq!(result.selected_cell, (1, 1, 1));
        assert_eq!(result.maximum_supported_power_w, 2_000);
        assert_eq!(result.disposition, crate::bms_capability::CapabilityDisposition::DerateRequired);
    }

    #[test]
    fn state_outside_any_calibrated_axis_is_rejected() {
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &CELLS);
        assert_eq!(
            assess_capability_surface(
                profile,
                9_001,
                300,
                9_000,
                crate::bms_capability::PowerDirection::Discharge,
                500,
            ),
            Err(CapabilitySurfaceError::StateOutsideCalibratedSurface)
        );
        assert_eq!(
            assess_capability_surface(
                profile,
                6_000,
                451,
                9_000,
                crate::bms_capability::PowerDirection::Discharge,
                500,
            ),
            Err(CapabilitySurfaceError::StateOutsideCalibratedSurface)
        );
        assert_eq!(
            assess_capability_surface(
                profile,
                6_000,
                300,
                6_999,
                crate::bms_capability::PowerDirection::Discharge,
                500,
            ),
            Err(CapabilitySurfaceError::StateOutsideCalibratedSurface)
        );
    }

    #[test]
    fn incomplete_or_duplicated_cells_fail_closed() {
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &CELLS[..7]);
        assert_eq!(
            assess_capability_surface(
                profile,
                6_000,
                300,
                9_000,
                crate::bms_capability::PowerDirection::Charge,
                500,
            ),
            Err(CapabilitySurfaceError::IncompleteCellGrid)
        );

        let mut duplicate = CELLS;
        duplicate[7] = cell("cell-duplicate", 0, 0, 0, 1_000, 1_000, 40);
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &duplicate);
        assert_eq!(
            assess_capability_surface(
                profile,
                6_000,
                300,
                9_000,
                crate::bms_capability::PowerDirection::Charge,
                500,
            ),
            Err(CapabilitySurfaceError::DuplicateCellCoordinate)
        );
    }

    #[test]
    fn field_cells_need_minimum_samples_and_valid_cell_digests() {
        let mut sparse = CELLS;
        sparse[7] = cell("sparse-cell", 1, 1, 1, 1_500, 2_000, 19);
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &sparse);
        assert_eq!(
            assess_capability_surface(
                profile,
                6_000,
                300,
                9_000,
                crate::bms_capability::PowerDirection::Charge,
                500,
            ),
            Err(CapabilitySurfaceError::InsufficientCellSamples)
        );

        let mut bad_digest = CELLS;
        bad_digest[7] = CapabilitySurfaceCell {
            source_sha256: "invalid",
            ..cell("bad-digest-cell", 1, 1, 1, 1_500, 2_000, 40)
        };
        let profile = surface(CapabilityEvidenceBasis::FieldMeasuredConservative, &bad_digest);
        assert_eq!(
            assess_capability_surface(
                profile,
                6_000,
                300,
                9_000,
                crate::bms_capability::PowerDirection::Charge,
                500,
            ),
            Err(CapabilitySurfaceError::InvalidSourceDigest)
        );
    }

    #[test]
    fn synthetic_surface_is_not_marked_as_non_synthetic() {
        const SYNTHETIC: [CapabilitySurfaceCell<'static>; 8] = [
            cell("synthetic-000", 0, 0, 0, 3_000, 3_500, 0),
            cell("synthetic-001", 0, 0, 1, 2_800, 3_300, 0),
            cell("synthetic-010", 0, 1, 0, 2_500, 3_100, 0),
            cell("synthetic-011", 0, 1, 1, 2_300, 2_900, 0),
            cell("synthetic-100", 1, 0, 0, 3_500, 4_000, 0),
            cell("synthetic-101", 1, 0, 1, 3_200, 3_700, 0),
            cell("synthetic-110", 1, 1, 0, 2_000, 2_500, 0),
            cell("synthetic-111", 1, 1, 1, 1_500, 2_000, 0),
        ];
        let profile = surface(CapabilityEvidenceBasis::SyntheticAssumption, &SYNTHETIC);
        let result = assess_capability_surface(
            profile,
            6_000,
            300,
            9_000,
            crate::bms_capability::PowerDirection::Discharge,
            1_000,
        )
        .unwrap();
        assert!(!result.evidence_basis_is_non_synthetic);
    }
}
