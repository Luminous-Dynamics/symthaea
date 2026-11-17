// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded 24-hour and multi-day energy-balance screen for high-altitude platforms.
//!
//! This module deliberately consumes a caller-supplied time-resolved irradiance and
//! load profile. It does not calculate sun position, atmospheric transmission, solar
//! geometry, aircraft aerodynamics, station-keeping trajectories, battery chemistry,
//! or wireless-transfer physics. Those remain with their domain owners.
//!
//! Export is surplus-only: the platform serves its own loads, then charges its battery,
//! and only then offers remaining PV energy to a declared receiver demand. The model
//! never discharges the HAPS battery to satisfy an external receiver request.
//!
//! A PASS from this arithmetic screen is not evidence of flight endurance, HAPS
//! station-keeping, a qualified battery, physical energy transfer, airworthiness,
//! airspace permission, or multi-month / multi-year operational availability.

/// Seconds in the fixed 24-hour accounting horizon.
pub const DAY_SECONDS: f64 = 86_400.0;

/// Absolute timing tolerance for summing a 24-hour profile from interval durations.
const DURATION_TOLERANCE_S: f64 = 1.0e-3;
/// Absolute watt-hour tolerance for energy ledgers and reserve comparisons.
const ENERGY_TOLERANCE_WH: f64 = 1.0e-8;

/// Installed-platform parameters for the bounded day-energy screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarHapsConfig {
    /// Active photovoltaic collection area in square metres.
    pub pv_area_m2: f64,
    /// Cell/array conversion efficiency at the declared reference condition [0, 1].
    pub pv_efficiency: f64,
    /// Maximum power-point tracking efficiency [0, 1].
    pub mppt_efficiency: f64,
    /// DC wiring and collection-path efficiency [0, 1].
    pub wiring_efficiency: f64,
    /// Installed battery energy capacity in watt-hours.
    pub battery_capacity_wh: f64,
    /// Battery state of charge at the beginning of the evaluation horizon [Wh].
    pub initial_soc_wh: f64,
    /// Hard minimum battery reserve which cannot be used for export [Wh].
    pub minimum_reserve_wh: f64,
    /// Battery-terminal energy stored per Wh supplied at the charging bus [0, 1].
    pub charge_efficiency: f64,
    /// Bus energy delivered per Wh withdrawn from stored battery energy [0, 1].
    pub discharge_efficiency: f64,
    /// Maximum battery charging power at the DC bus [W].
    pub max_charge_power_w: f64,
    /// Maximum battery discharge power delivered to the DC bus [W].
    pub max_discharge_power_w: f64,
    /// Declared end-to-end HAPS-bus to receiver-useful-load transfer efficiency [0, 1].
    ///
    /// This is an upstream assumption/measurement, not a link model. The value must
    /// include every conversion and propagation stage in the declared meter boundary.
    pub export_link_efficiency: f64,
}

/// One piecewise-constant interval in a frozen HAPS energy profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarHapsInterval {
    /// Interval duration in seconds.
    pub duration_s: f64,
    /// Incident solar irradiance at the array plane before incidence/array derating [W/m²].
    pub incident_irradiance_w_m2: f64,
    /// Solar incidence / usable-area factor [0, 1].
    pub incidence_factor: f64,
    /// PV temperature and operating-condition derate [0, 1].
    pub pv_temperature_derate: f64,
    /// Propulsion power required for station-keeping and flight [W].
    pub station_keeping_power_w: f64,
    /// HAPS mission payload power [W].
    pub payload_power_w: f64,
    /// Avionics, communications and control power [W].
    pub avionics_power_w: f64,
    /// Thermal-management and other auxiliary power [W].
    pub thermal_management_power_w: f64,
    /// Requested useful receiver power at the destination aircraft/load [W].
    ///
    /// It is served only from PV surplus after HAPS own-load service and battery
    /// charging. It is never silently supplied by discharging the HAPS battery.
    pub requested_receiver_power_w: f64,
}

/// Validation and accounting errors for bounded HAPS energy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarHapsError {
    NonFiniteConfig,
    InvalidPvArea,
    InvalidEfficiency,
    InvalidBatteryCapacity,
    InvalidInitialSoc,
    InvalidReserve,
    InvalidPowerLimit,
    EmptyProfile,
    NonFiniteInterval,
    InvalidIntervalDuration,
    InvalidIntervalValue,
    ProfileDurationNot24Hours,
    ProfileDurationNotWholeDays,
    AllocationFailed,
    NonFiniteResult,
}

impl std::fmt::Display for SolarHapsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::NonFiniteConfig => "configuration contains a non-finite value",
            Self::InvalidPvArea => "PV area must be positive",
            Self::InvalidEfficiency => "efficiencies must be finite and in (0, 1]",
            Self::InvalidBatteryCapacity => "battery capacity must be positive",
            Self::InvalidInitialSoc => "initial state of charge must lie within battery limits",
            Self::InvalidReserve => "reserve must lie between zero and battery capacity",
            Self::InvalidPowerLimit => "charge and discharge power limits must be positive",
            Self::EmptyProfile => "the energy profile must contain at least one interval",
            Self::NonFiniteInterval => "an interval contains a non-finite value",
            Self::InvalidIntervalDuration => "interval duration must be positive",
            Self::InvalidIntervalValue => "irradiance, loads and receiver requests must be non-negative and derates within [0, 1]",
            Self::ProfileDurationNot24Hours => "the sum of interval durations must equal 24 hours",
            Self::ProfileDurationNotWholeDays => "the horizon duration must be a positive whole number of 24-hour days",
            Self::AllocationFailed => "could not reserve memory for the requested energy-evaluation result",
            Self::NonFiniteResult => "energy accounting produced a non-finite result",
        };
        f.write_str(message)
    }
}

impl std::error::Error for SolarHapsError {}

/// Auditable per-interval accounting result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarHapsIntervalResult {
    pub interval_index: usize,
    pub duration_s: f64,
    pub solar_generation_wh: f64,
    pub own_load_demand_wh: f64,
    pub own_load_served_wh: f64,
    pub own_load_unserved_wh: f64,
    /// Bus energy drawn from the battery and delivered to HAPS loads [Wh].
    pub battery_discharge_output_wh: f64,
    /// Stored battery energy withdrawn to produce the discharge output [Wh].
    pub battery_discharge_withdrawn_wh: f64,
    /// Bus energy supplied to the battery charger [Wh].
    pub battery_charge_input_wh: f64,
    /// Energy actually stored after charge losses [Wh].
    pub battery_energy_stored_wh: f64,
    /// Solar-bus energy consumed by the HAPS-side transfer system [Wh].
    pub export_source_energy_wh: f64,
    /// Useful energy delivered at the receiver meter plane [Wh].
    pub export_receiver_energy_wh: f64,
    /// Requested but undelivered receiver energy [Wh].
    pub export_unserved_receiver_energy_wh: f64,
    /// PV energy curtailed after own loads, charging and export [Wh].
    pub curtailed_solar_energy_wh: f64,
    /// Battery state of charge at interval end [Wh].
    pub ending_soc_wh: f64,
    /// PV-bus ledger residual: generation less direct load, charge input,
    /// export source and curtailment [Wh]. Should be near zero.
    pub pv_bus_ledger_residual_wh: f64,
}

/// Aggregated result for one supplied 24-hour profile.
#[derive(Debug, Clone, PartialEq)]
pub struct SolarHapsDayResult {
    pub intervals: Vec<SolarHapsIntervalResult>,
    pub initial_soc_wh: f64,
    pub ending_soc_wh: f64,
    pub minimum_soc_wh: f64,
    pub total_solar_generation_wh: f64,
    pub total_own_load_demand_wh: f64,
    pub total_own_load_served_wh: f64,
    pub total_own_load_unserved_wh: f64,
    pub total_battery_charge_input_wh: f64,
    pub total_battery_energy_stored_wh: f64,
    pub total_battery_discharge_output_wh: f64,
    pub total_battery_discharge_withdrawn_wh: f64,
    pub total_export_source_energy_wh: f64,
    pub total_export_receiver_energy_wh: f64,
    pub total_export_unserved_receiver_energy_wh: f64,
    pub total_curtailed_solar_energy_wh: f64,
    pub total_pv_bus_ledger_residual_wh: f64,
    pub reserve_respected: bool,
    /// True only when HAPS own loads were served and the hard reserve was respected.
    pub own_service_continuous: bool,
    /// Ending SoC is at least the starting SoC. This is necessary for repeated
    /// identical daily cycles, but is not a months/years reliability demonstration.
    pub daily_energy_cycle_closed: bool,
    /// A HAPS-to-receiver service claim is admissible only when own loads, reserve,
    /// and the repeated daily energy cycle all close. Calculated export energy is
    /// still reported when this flag is false, but must not be promoted as a
    /// sustainable service capability.
    pub export_service_admissible: bool,
}


/// Aggregated result for a multi-day, caller-supplied HAPS energy horizon.
///
/// The day results preserve interval-level auditability. Horizon-level export
/// admission uses the full-window state-of-charge closure, not the per-day cycle
/// flag: a low-solar day may be recovered by a later surplus day, provided every
/// interval preserves reserve and serves the HAPS's own loads.
#[derive(Debug, Clone, PartialEq)]
pub struct SolarHapsHorizonResult {
    pub days: Vec<SolarHapsDayResult>,
    pub horizon_days: usize,
    pub initial_soc_wh: f64,
    pub ending_soc_wh: f64,
    pub minimum_soc_wh: f64,
    pub total_solar_generation_wh: f64,
    pub total_own_load_demand_wh: f64,
    pub total_own_load_served_wh: f64,
    pub total_own_load_unserved_wh: f64,
    pub total_battery_charge_input_wh: f64,
    pub total_battery_energy_stored_wh: f64,
    pub total_battery_discharge_output_wh: f64,
    pub total_battery_discharge_withdrawn_wh: f64,
    pub total_export_source_energy_wh: f64,
    pub total_export_receiver_energy_wh: f64,
    pub total_export_unserved_receiver_energy_wh: f64,
    pub total_curtailed_solar_energy_wh: f64,
    pub total_pv_bus_ledger_residual_wh: f64,
    pub reserve_respected: bool,
    pub own_service_continuous: bool,
    /// Ending SoC is at least the horizon's initial SoC. This is necessary for
    /// repeating this exact horizon, but does not establish seasonal reliability.
    pub horizon_energy_cycle_closed: bool,
    /// True only when every day preserves reserve and serves own loads and the
    /// complete horizon ends with at least its initial SoC.
    pub export_service_admissible: bool,
}

/// Validate the caller-supplied platform parameters before any accounting.
pub fn validate_config(config: &SolarHapsConfig) -> Result<(), SolarHapsError> {
    let values = [
        config.pv_area_m2,
        config.pv_efficiency,
        config.mppt_efficiency,
        config.wiring_efficiency,
        config.battery_capacity_wh,
        config.initial_soc_wh,
        config.minimum_reserve_wh,
        config.charge_efficiency,
        config.discharge_efficiency,
        config.max_charge_power_w,
        config.max_discharge_power_w,
        config.export_link_efficiency,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SolarHapsError::NonFiniteConfig);
    }
    if config.pv_area_m2 <= 0.0 {
        return Err(SolarHapsError::InvalidPvArea);
    }
    let efficiencies = [
        config.pv_efficiency,
        config.mppt_efficiency,
        config.wiring_efficiency,
        config.charge_efficiency,
        config.discharge_efficiency,
        config.export_link_efficiency,
    ];
    if efficiencies
        .iter()
        .any(|efficiency| *efficiency <= 0.0 || *efficiency > 1.0)
    {
        return Err(SolarHapsError::InvalidEfficiency);
    }
    if config.battery_capacity_wh <= 0.0 {
        return Err(SolarHapsError::InvalidBatteryCapacity);
    }
    if config.minimum_reserve_wh < 0.0
        || config.minimum_reserve_wh > config.battery_capacity_wh
    {
        return Err(SolarHapsError::InvalidReserve);
    }
    if config.initial_soc_wh < config.minimum_reserve_wh
        || config.initial_soc_wh > config.battery_capacity_wh
    {
        return Err(SolarHapsError::InvalidInitialSoc);
    }
    if config.max_charge_power_w <= 0.0 || config.max_discharge_power_w <= 0.0 {
        return Err(SolarHapsError::InvalidPowerLimit);
    }
    Ok(())
}

/// Validate one interval. The irradiance model and all aircraft loads are upstream
/// inputs with their own source, model and applicability evidence.
fn validate_interval(interval: &SolarHapsInterval) -> Result<(), SolarHapsError> {
    let values = [
        interval.duration_s,
        interval.incident_irradiance_w_m2,
        interval.incidence_factor,
        interval.pv_temperature_derate,
        interval.station_keeping_power_w,
        interval.payload_power_w,
        interval.avionics_power_w,
        interval.thermal_management_power_w,
        interval.requested_receiver_power_w,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SolarHapsError::NonFiniteInterval);
    }
    if interval.duration_s <= 0.0 {
        return Err(SolarHapsError::InvalidIntervalDuration);
    }
    if interval.incident_irradiance_w_m2 < 0.0
        || !(0.0..=1.0).contains(&interval.incidence_factor)
        || !(0.0..=1.0).contains(&interval.pv_temperature_derate)
        || interval.station_keeping_power_w < 0.0
        || interval.payload_power_w < 0.0
        || interval.avionics_power_w < 0.0
        || interval.thermal_management_power_w < 0.0
        || interval.requested_receiver_power_w < 0.0
    {
        return Err(SolarHapsError::InvalidIntervalValue);
    }
    Ok(())
}

/// Run the frozen 24-hour profile.
///
/// Accounting priority:
/// 1. PV serves the HAPS's own station-keeping, payload, avionics and thermal loads.
/// 2. Battery discharge covers any remaining own-load deficit, but never crosses reserve.
/// 3. PV surplus charges the HAPS battery, subject to charge-power and capacity limits.
/// 4. Only remaining PV surplus is offered to the external receiver.
///
/// The output reports source-side export energy and useful receiver energy separately.
/// This is a deterministic accounting screen, not an optical/RF/microwave link simulator.
pub fn evaluate_day(
    config: &SolarHapsConfig,
    profile: &[SolarHapsInterval],
) -> Result<SolarHapsDayResult, SolarHapsError> {
    validate_config(config)?;
    if profile.is_empty() {
        return Err(SolarHapsError::EmptyProfile);
    }
    let mut total_duration_s = 0.0;
    for interval in profile {
        validate_interval(interval)?;
        total_duration_s += interval.duration_s;
    }
    if !total_duration_s.is_finite()
        || (total_duration_s - DAY_SECONDS).abs() > DURATION_TOLERANCE_S
    {
        return Err(SolarHapsError::ProfileDurationNot24Hours);
    }

    let mut soc_wh = config.initial_soc_wh;
    let mut minimum_soc_wh = soc_wh;
    let mut results = Vec::new();
    results
        .try_reserve_exact(profile.len())
        .map_err(|_| SolarHapsError::AllocationFailed)?;

    let mut total_solar_generation_wh = 0.0;
    let mut total_own_load_demand_wh = 0.0;
    let mut total_own_load_served_wh = 0.0;
    let mut total_own_load_unserved_wh = 0.0;
    let mut total_battery_charge_input_wh = 0.0;
    let mut total_battery_energy_stored_wh = 0.0;
    let mut total_battery_discharge_output_wh = 0.0;
    let mut total_battery_discharge_withdrawn_wh = 0.0;
    let mut total_export_source_energy_wh = 0.0;
    let mut total_export_receiver_energy_wh = 0.0;
    let mut total_export_unserved_receiver_energy_wh = 0.0;
    let mut total_curtailed_solar_energy_wh = 0.0;
    let mut total_pv_bus_ledger_residual_wh = 0.0;

    for (index, interval) in profile.iter().enumerate() {
        let duration_h = interval.duration_s / 3600.0;
        let solar_power_w = interval.incident_irradiance_w_m2
            * config.pv_area_m2
            * config.pv_efficiency
            * interval.incidence_factor
            * interval.pv_temperature_derate
            * config.mppt_efficiency
            * config.wiring_efficiency;
        let solar_generation_wh = solar_power_w * duration_h;

        let own_load_power_w = interval.station_keeping_power_w
            + interval.payload_power_w
            + interval.avionics_power_w
            + interval.thermal_management_power_w;
        let own_load_demand_wh = own_load_power_w * duration_h;

        // PV is first allocated to the HAPS's own loads.
        let direct_solar_to_own_load_wh = solar_generation_wh.min(own_load_demand_wh);
        let own_load_deficit_wh = own_load_demand_wh - direct_solar_to_own_load_wh;
        let solar_surplus_after_own_load_wh =
            solar_generation_wh - direct_solar_to_own_load_wh;

        // Any own-load deficit may draw the battery only above the declared reserve.
        let max_battery_output_wh = config.max_discharge_power_w * duration_h;
        let battery_available_above_reserve_wh =
            (soc_wh - config.minimum_reserve_wh).max(0.0);
        let battery_available_at_bus_wh =
            (battery_available_above_reserve_wh * config.discharge_efficiency)
                .min(max_battery_output_wh);
        let battery_discharge_output_wh =
            own_load_deficit_wh.min(battery_available_at_bus_wh);
        let battery_discharge_withdrawn_wh =
            battery_discharge_output_wh / config.discharge_efficiency;
        soc_wh -= battery_discharge_withdrawn_wh;

        let own_load_served_wh =
            direct_solar_to_own_load_wh + battery_discharge_output_wh;
        let own_load_unserved_wh = own_load_demand_wh - own_load_served_wh;

        // If PV is in surplus, charge the HAPS battery before exporting any energy.
        let max_battery_charge_input_wh = config.max_charge_power_w * duration_h;
        let battery_headroom_input_wh =
            ((config.battery_capacity_wh - soc_wh).max(0.0)) / config.charge_efficiency;
        let battery_charge_input_wh = solar_surplus_after_own_load_wh
            .min(max_battery_charge_input_wh)
            .min(battery_headroom_input_wh);
        let battery_energy_stored_wh =
            battery_charge_input_wh * config.charge_efficiency;
        soc_wh += battery_energy_stored_wh;

        let solar_surplus_after_charge_wh =
            solar_surplus_after_own_load_wh - battery_charge_input_wh;

        // External export is supplied only from remaining solar surplus. The HAPS
        // battery is not discharged to satisfy a receiver's request.
        let requested_receiver_energy_wh = interval.requested_receiver_power_w * duration_h;
        let requested_source_energy_wh =
            requested_receiver_energy_wh / config.export_link_efficiency;
        let export_source_energy_wh =
            solar_surplus_after_charge_wh.min(requested_source_energy_wh);
        let export_receiver_energy_wh =
            export_source_energy_wh * config.export_link_efficiency;
        let export_unserved_receiver_energy_wh =
            requested_receiver_energy_wh - export_receiver_energy_wh;
        let curtailed_solar_energy_wh =
            solar_surplus_after_charge_wh - export_source_energy_wh;

        let pv_bus_ledger_residual_wh = solar_generation_wh
            - direct_solar_to_own_load_wh
            - battery_charge_input_wh
            - export_source_energy_wh
            - curtailed_solar_energy_wh;

        minimum_soc_wh = minimum_soc_wh.min(soc_wh);

        let result = SolarHapsIntervalResult {
            interval_index: index,
            duration_s: interval.duration_s,
            solar_generation_wh,
            own_load_demand_wh,
            own_load_served_wh,
            own_load_unserved_wh,
            battery_discharge_output_wh,
            battery_discharge_withdrawn_wh,
            battery_charge_input_wh,
            battery_energy_stored_wh,
            export_source_energy_wh,
            export_receiver_energy_wh,
            export_unserved_receiver_energy_wh,
            curtailed_solar_energy_wh,
            ending_soc_wh: soc_wh,
            pv_bus_ledger_residual_wh,
        };

        let interval_values = [
            result.solar_generation_wh,
            result.own_load_demand_wh,
            result.own_load_served_wh,
            result.own_load_unserved_wh,
            result.battery_discharge_output_wh,
            result.battery_discharge_withdrawn_wh,
            result.battery_charge_input_wh,
            result.battery_energy_stored_wh,
            result.export_source_energy_wh,
            result.export_receiver_energy_wh,
            result.export_unserved_receiver_energy_wh,
            result.curtailed_solar_energy_wh,
            result.ending_soc_wh,
            result.pv_bus_ledger_residual_wh,
        ];
        if interval_values.iter().any(|value| !value.is_finite()) {
            return Err(SolarHapsError::NonFiniteResult);
        }

        total_solar_generation_wh += solar_generation_wh;
        total_own_load_demand_wh += own_load_demand_wh;
        total_own_load_served_wh += own_load_served_wh;
        total_own_load_unserved_wh += own_load_unserved_wh;
        total_battery_charge_input_wh += battery_charge_input_wh;
        total_battery_energy_stored_wh += battery_energy_stored_wh;
        total_battery_discharge_output_wh += battery_discharge_output_wh;
        total_battery_discharge_withdrawn_wh += battery_discharge_withdrawn_wh;
        total_export_source_energy_wh += export_source_energy_wh;
        total_export_receiver_energy_wh += export_receiver_energy_wh;
        total_export_unserved_receiver_energy_wh += export_unserved_receiver_energy_wh;
        total_curtailed_solar_energy_wh += curtailed_solar_energy_wh;
        total_pv_bus_ledger_residual_wh += pv_bus_ledger_residual_wh;
        results.push(result);
    }

    let totals = [
        total_solar_generation_wh,
        total_own_load_demand_wh,
        total_own_load_served_wh,
        total_own_load_unserved_wh,
        total_battery_charge_input_wh,
        total_battery_energy_stored_wh,
        total_battery_discharge_output_wh,
        total_battery_discharge_withdrawn_wh,
        total_export_source_energy_wh,
        total_export_receiver_energy_wh,
        total_export_unserved_receiver_energy_wh,
        total_curtailed_solar_energy_wh,
        total_pv_bus_ledger_residual_wh,
        soc_wh,
        minimum_soc_wh,
    ];
    if totals.iter().any(|value| !value.is_finite()) {
        return Err(SolarHapsError::NonFiniteResult);
    }

    let reserve_respected = minimum_soc_wh + ENERGY_TOLERANCE_WH >= config.minimum_reserve_wh;
    let own_service_continuous =
        reserve_respected && total_own_load_unserved_wh <= ENERGY_TOLERANCE_WH;
    let daily_energy_cycle_closed =
        soc_wh + ENERGY_TOLERANCE_WH >= config.initial_soc_wh;
    let export_service_admissible =
        own_service_continuous && daily_energy_cycle_closed;

    Ok(SolarHapsDayResult {
        intervals: results,
        initial_soc_wh: config.initial_soc_wh,
        ending_soc_wh: soc_wh,
        minimum_soc_wh,
        total_solar_generation_wh,
        total_own_load_demand_wh,
        total_own_load_served_wh,
        total_own_load_unserved_wh,
        total_battery_charge_input_wh,
        total_battery_energy_stored_wh,
        total_battery_discharge_output_wh,
        total_battery_discharge_withdrawn_wh,
        total_export_source_energy_wh,
        total_export_receiver_energy_wh,
        total_export_unserved_receiver_energy_wh,
        total_curtailed_solar_energy_wh,
        total_pv_bus_ledger_residual_wh,
        reserve_respected,
        own_service_continuous,
        daily_energy_cycle_closed,
        export_service_admissible,
    })
}


/// Evaluate a contiguous horizon containing one or more complete 24-hour days.
///
/// Input intervals are treated as piecewise-constant. Intervals that cross a
/// day boundary are split without changing their physical input values. The
/// initial state of charge is carried forward from each day to the next.
///
/// Horizon-level export admission requires:
/// - HAPS own loads served throughout the complete horizon;
/// - the battery reserve respected throughout the complete horizon; and
/// - ending SoC at least equal to the horizon's starting SoC.
///
/// This is still an arithmetic model over caller-supplied inputs. It does not
/// supply seasonal weather, battery ageing, thermal dynamics, station-keeping
/// trajectories, buoyancy, or aircraft operational reliability.
pub fn evaluate_horizon(
    config: &SolarHapsConfig,
    profile: &[SolarHapsInterval],
) -> Result<SolarHapsHorizonResult, SolarHapsError> {
    validate_config(config)?;
    if profile.is_empty() {
        return Err(SolarHapsError::EmptyProfile);
    }

    let mut total_duration_s = 0.0;
    for interval in profile {
        validate_interval(interval)?;
        total_duration_s += interval.duration_s;
    }
    if !total_duration_s.is_finite() {
        return Err(SolarHapsError::NonFiniteResult);
    }

    let horizon_days_f64 = (total_duration_s / DAY_SECONDS).round();
    if horizon_days_f64 < 1.0
        || horizon_days_f64 > usize::MAX as f64
        || (total_duration_s - horizon_days_f64 * DAY_SECONDS).abs() > DURATION_TOLERANCE_S
    {
        return Err(SolarHapsError::ProfileDurationNotWholeDays);
    }
    let horizon_days = horizon_days_f64 as usize;
    if horizon_days == 0 {
        return Err(SolarHapsError::ProfileDurationNotWholeDays);
    }

    // Normalize only sub-millisecond summation drift at the final interval.
    // This prevents floating-point accumulation from creating a phantom partial day.
    let target_duration_s = horizon_days_f64 * DAY_SECONDS;
    let mut normalized_profile = Vec::new();
    normalized_profile
        .try_reserve_exact(profile.len())
        .map_err(|_| SolarHapsError::AllocationFailed)?;
    normalized_profile.extend_from_slice(profile);
    let final_interval = normalized_profile
        .last_mut()
        .ok_or(SolarHapsError::EmptyProfile)?;
    final_interval.duration_s += target_duration_s - total_duration_s;
    validate_interval(final_interval)?;

    let mut days = Vec::new();
    days.try_reserve_exact(horizon_days)
        .map_err(|_| SolarHapsError::AllocationFailed)?;
    let mut day_profile = Vec::new();
    let mut day_elapsed_s = 0.0;
    let mut carried_soc_wh = config.initial_soc_wh;
    let mut day_config = *config;

    for interval in normalized_profile {
        let mut remaining_s = interval.duration_s;
        while remaining_s > 0.0 {
            let day_remaining_s = DAY_SECONDS - day_elapsed_s;
            if day_remaining_s <= DURATION_TOLERANCE_S {
                let correction_s = DAY_SECONDS - day_elapsed_s;
                let last = day_profile
                    .last_mut()
                    .ok_or(SolarHapsError::ProfileDurationNotWholeDays)?;
                last.duration_s += correction_s;

                day_config.initial_soc_wh = carried_soc_wh;
                let day_result = evaluate_day(&day_config, &day_profile)?;
                carried_soc_wh = day_result.ending_soc_wh;
                days.push(day_result);
                day_profile.clear();
                day_elapsed_s = 0.0;
                continue;
            }

            let fragment_duration_s = remaining_s.min(day_remaining_s);
            if !fragment_duration_s.is_finite() || fragment_duration_s <= 0.0 {
                return Err(SolarHapsError::ProfileDurationNotWholeDays);
            }

            let mut fragment = interval;
            fragment.duration_s = fragment_duration_s;
            day_profile
                .try_reserve(1)
                .map_err(|_| SolarHapsError::AllocationFailed)?;
            day_profile.push(fragment);
            remaining_s -= fragment_duration_s;
            day_elapsed_s += fragment_duration_s;

            if (DAY_SECONDS - day_elapsed_s).abs() <= DURATION_TOLERANCE_S {
                let correction_s = DAY_SECONDS - day_elapsed_s;
                let last = day_profile
                    .last_mut()
                    .ok_or(SolarHapsError::ProfileDurationNotWholeDays)?;
                last.duration_s += correction_s;

                day_config.initial_soc_wh = carried_soc_wh;
                let day_result = evaluate_day(&day_config, &day_profile)?;
                carried_soc_wh = day_result.ending_soc_wh;
                days.push(day_result);
                day_profile.clear();
                day_elapsed_s = 0.0;
            }
        }
    }

    if !day_profile.is_empty() || days.len() != horizon_days {
        return Err(SolarHapsError::ProfileDurationNotWholeDays);
    }

    let initial_soc_wh = config.initial_soc_wh;
    let ending_soc_wh = carried_soc_wh;
    let minimum_soc_wh = days
        .iter()
        .map(|day| day.minimum_soc_wh)
        .fold(initial_soc_wh, f64::min);

    let total_solar_generation_wh = days.iter().map(|day| day.total_solar_generation_wh).sum();
    let total_own_load_demand_wh = days.iter().map(|day| day.total_own_load_demand_wh).sum();
    let total_own_load_served_wh = days.iter().map(|day| day.total_own_load_served_wh).sum();
    let total_own_load_unserved_wh = days.iter().map(|day| day.total_own_load_unserved_wh).sum();
    let total_battery_charge_input_wh =
        days.iter().map(|day| day.total_battery_charge_input_wh).sum();
    let total_battery_energy_stored_wh =
        days.iter().map(|day| day.total_battery_energy_stored_wh).sum();
    let total_battery_discharge_output_wh =
        days.iter().map(|day| day.total_battery_discharge_output_wh).sum();
    let total_battery_discharge_withdrawn_wh =
        days.iter().map(|day| day.total_battery_discharge_withdrawn_wh).sum();
    let total_export_source_energy_wh =
        days.iter().map(|day| day.total_export_source_energy_wh).sum();
    let total_export_receiver_energy_wh =
        days.iter().map(|day| day.total_export_receiver_energy_wh).sum();
    let total_export_unserved_receiver_energy_wh =
        days.iter().map(|day| day.total_export_unserved_receiver_energy_wh).sum();
    let total_curtailed_solar_energy_wh =
        days.iter().map(|day| day.total_curtailed_solar_energy_wh).sum();
    let total_pv_bus_ledger_residual_wh =
        days.iter().map(|day| day.total_pv_bus_ledger_residual_wh).sum();

    let totals = [
        total_solar_generation_wh,
        total_own_load_demand_wh,
        total_own_load_served_wh,
        total_own_load_unserved_wh,
        total_battery_charge_input_wh,
        total_battery_energy_stored_wh,
        total_battery_discharge_output_wh,
        total_battery_discharge_withdrawn_wh,
        total_export_source_energy_wh,
        total_export_receiver_energy_wh,
        total_export_unserved_receiver_energy_wh,
        total_curtailed_solar_energy_wh,
        total_pv_bus_ledger_residual_wh,
        ending_soc_wh,
        minimum_soc_wh,
    ];
    if totals.iter().any(|value| !value.is_finite()) {
        return Err(SolarHapsError::NonFiniteResult);
    }

    let reserve_respected = days.iter().all(|day| day.reserve_respected);
    let own_service_continuous = days.iter().all(|day| day.own_service_continuous);
    let horizon_energy_cycle_closed =
        ending_soc_wh + ENERGY_TOLERANCE_WH >= initial_soc_wh;
    let export_service_admissible =
        reserve_respected && own_service_continuous && horizon_energy_cycle_closed;

    Ok(SolarHapsHorizonResult {
        days,
        horizon_days,
        initial_soc_wh,
        ending_soc_wh,
        minimum_soc_wh,
        total_solar_generation_wh,
        total_own_load_demand_wh,
        total_own_load_served_wh,
        total_own_load_unserved_wh,
        total_battery_charge_input_wh,
        total_battery_energy_stored_wh,
        total_battery_discharge_output_wh,
        total_battery_discharge_withdrawn_wh,
        total_export_source_energy_wh,
        total_export_receiver_energy_wh,
        total_export_unserved_receiver_energy_wh,
        total_curtailed_solar_energy_wh,
        total_pv_bus_ledger_residual_wh,
        reserve_respected,
        own_service_continuous,
        horizon_energy_cycle_closed,
        export_service_admissible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> SolarHapsConfig {
        SolarHapsConfig {
            pv_area_m2: 1.25,
            pv_efficiency: 0.2,
            mppt_efficiency: 1.0,
            wiring_efficiency: 1.0,
            battery_capacity_wh: 1800.0,
            initial_soc_wh: 600.0,
            minimum_reserve_wh: 600.0,
            charge_efficiency: 1.0,
            discharge_efficiency: 1.0,
            max_charge_power_w: 200.0,
            max_discharge_power_w: 200.0,
            export_link_efficiency: 0.8,
        }
    }

    fn day_night_profile(requested_receiver_power_w: f64) -> Vec<SolarHapsInterval> {
        let mut profile = Vec::with_capacity(24);
        for hour in 0..24 {
            let daylight = hour < 12;
            profile.push(SolarHapsInterval {
                duration_s: 3600.0,
                incident_irradiance_w_m2: if daylight { 1000.0 } else { 0.0 },
                incidence_factor: 1.0,
                pv_temperature_derate: 1.0,
                station_keeping_power_w: 100.0,
                payload_power_w: 0.0,
                avionics_power_w: 0.0,
                thermal_management_power_w: 0.0,
                requested_receiver_power_w: if daylight {
                    requested_receiver_power_w
                } else {
                    0.0
                },
            });
        }
        profile
    }


    fn two_day_recovery_profile() -> Vec<SolarHapsInterval> {
        let mut profile = Vec::with_capacity(48);
        for hour in 0..48 {
            let day_index = hour / 24;
            let hour_of_day = hour % 24;
            let daylight = hour_of_day < 12;
            profile.push(SolarHapsInterval {
                duration_s: 3600.0,
                incident_irradiance_w_m2: if daylight {
                    if day_index == 0 { 800.0 } else { 1000.0 }
                } else {
                    0.0
                },
                incidence_factor: 1.0,
                pv_temperature_derate: 1.0,
                station_keeping_power_w: if day_index == 1 && !daylight {
                    50.0
                } else {
                    100.0
                },
                payload_power_w: 0.0,
                avionics_power_w: 0.0,
                thermal_management_power_w: 0.0,
                requested_receiver_power_w: if daylight { 200.0 } else { 0.0 },
            });
        }
        profile
    }


    fn unserved_load_recovery_profile() -> Vec<SolarHapsInterval> {
        let mut profile = Vec::with_capacity(48);
        for hour in 0..48 {
            let day_index = hour / 24;
            let hour_of_day = hour % 24;
            let daylight = hour_of_day < 12;
            let station_keeping_power_w = match (day_index, daylight) {
                (0, true) => 100.0,
                (0, false) => 200.0,
                (1, _) => 0.0,
                _ => unreachable!(),
            };
            profile.push(SolarHapsInterval {
                duration_s: 3600.0,
                incident_irradiance_w_m2: if daylight { 1000.0 } else { 0.0 },
                incidence_factor: 1.0,
                pv_temperature_derate: 1.0,
                station_keeping_power_w,
                payload_power_w: 0.0,
                avionics_power_w: 0.0,
                thermal_management_power_w: 0.0,
                requested_receiver_power_w: if daylight { 100.0 } else { 0.0 },
            });
        }
        profile
    }

    #[test]
    fn surplus_only_export_preserves_the_haps_daily_energy_cycle() {
        let result = evaluate_day(&base_config(), &day_night_profile(40.0)).unwrap();
        assert!(result.reserve_respected);
        assert!(result.own_service_continuous);
        assert!(result.daily_energy_cycle_closed);
        assert!(result.export_service_admissible);
        assert!((result.ending_soc_wh - 600.0).abs() < 1.0e-8);
        assert!((result.total_export_receiver_energy_wh - 480.0).abs() < 1.0e-8);
        assert!((result.total_export_source_energy_wh - 600.0).abs() < 1.0e-8);
        assert!(result.total_pv_bus_ledger_residual_wh.abs() < 1.0e-8);
        assert!(result.total_export_unserved_receiver_energy_wh.abs() < 1.0e-8);
    }

    #[test]
    fn export_is_not_funded_by_discharging_the_haps_battery() {
        let mut profile = day_night_profile(0.0);
        for interval in &mut profile {
            interval.incident_irradiance_w_m2 = 0.0;
            interval.station_keeping_power_w = 0.0;
            interval.requested_receiver_power_w = 100.0;
        }
        let result = evaluate_day(&base_config(), &profile).unwrap();
        assert_eq!(result.total_export_source_energy_wh, 0.0);
        assert_eq!(result.total_export_receiver_energy_wh, 0.0);
        assert_eq!(result.total_battery_discharge_output_wh, 0.0);
        assert!(result.total_export_unserved_receiver_energy_wh > 0.0);
        assert_eq!(result.ending_soc_wh, 600.0);
    }

    #[test]
    fn insufficient_night_energy_is_not_misreported_as_continuous_service() {
        let mut config = base_config();
        config.battery_capacity_wh = 1000.0;
        config.initial_soc_wh = 700.0;
        config.minimum_reserve_wh = 500.0;
        let mut profile = day_night_profile(0.0);
        for interval in &mut profile {
            interval.station_keeping_power_w = 100.0;
        }
        // Daylight PV produces only 30 W against a 100 W station-keeping load.
        // The battery can serve the deficit only until its 500 Wh hard reserve.
        for interval in profile.iter_mut().take(12) {
            interval.incident_irradiance_w_m2 = 120.0;
        }
        let result = evaluate_day(&config, &profile).unwrap();
        assert!(result.reserve_respected);
        assert!(!result.own_service_continuous);
        assert!(result.total_own_load_unserved_wh > 0.0);
        assert!(!result.daily_energy_cycle_closed);
        assert!(!result.export_service_admissible);
    }

    #[test]
    fn battery_charge_and_discharge_losses_are_accounted_separately() {
        let mut config = base_config();
        config.battery_capacity_wh = 4000.0;
        config.initial_soc_wh = 1500.0;
        config.minimum_reserve_wh = 500.0;
        config.charge_efficiency = 0.8;
        config.discharge_efficiency = 0.9;
        config.max_charge_power_w = 100.0;

        let mut profile = day_night_profile(0.0);
        for interval in profile.iter_mut().skip(12) {
            interval.incident_irradiance_w_m2 = 0.0;
            interval.station_keeping_power_w = 100.0;
        }

        let result = evaluate_day(&config, &profile).unwrap();
        assert!((result.total_battery_charge_input_wh - 1200.0).abs() < 1.0e-8);
        assert!((result.total_battery_energy_stored_wh - 960.0).abs() < 1.0e-8);
        assert!((result.total_battery_discharge_output_wh - 1200.0).abs() < 1.0e-8);
        assert!(
            (result.total_battery_discharge_withdrawn_wh - (1200.0 / 0.9)).abs() < 1.0e-8
        );
        assert!(result.reserve_respected);
        assert!(result.own_service_continuous);
        assert!(!result.daily_energy_cycle_closed);
        assert!(!result.export_service_admissible);
        assert!(result.total_pv_bus_ledger_residual_wh.abs() < 1.0e-8);
    }

    #[test]
    fn profile_must_cover_exactly_twenty_four_hours() {
        let mut profile = day_night_profile(0.0);
        profile.pop();
        assert_eq!(
            evaluate_day(&base_config(), &profile),
            Err(SolarHapsError::ProfileDurationNot24Hours)
        );
    }


    #[test]
    fn multi_day_horizon_carries_soc_and_uses_full_window_closure() {
        let mut config = base_config();
        config.initial_soc_wh = 1000.0;
        let result = evaluate_horizon(&config, &two_day_recovery_profile()).unwrap();

        assert_eq!(result.horizon_days, 2);
        assert_eq!(result.days.len(), 2);
        assert!(result.days[0].reserve_respected);
        assert!(result.days[0].own_service_continuous);
        assert!(!result.days[0].daily_energy_cycle_closed);
        assert!(result.days[1].daily_energy_cycle_closed);
        assert!(result.reserve_respected);
        assert!(result.own_service_continuous);
        assert!(result.horizon_energy_cycle_closed);
        assert!(result.export_service_admissible);
        assert!((result.ending_soc_wh - 1200.0).abs() < 1.0e-8);
        assert!((result.minimum_soc_wh - 600.0).abs() < 1.0e-8);
        assert!((result.total_export_source_energy_wh - 1000.0).abs() < 1.0e-8);
        assert!((result.total_export_receiver_energy_wh - 800.0).abs() < 1.0e-8);
        assert!(result.total_pv_bus_ledger_residual_wh.abs() < 1.0e-8);
    }

    #[test]
    fn multi_day_horizon_splits_intervals_at_day_boundaries() {
        let config = base_config();
        let profile = [SolarHapsInterval {
            duration_s: 2.0 * DAY_SECONDS,
            incident_irradiance_w_m2: 1000.0,
            incidence_factor: 1.0,
            pv_temperature_derate: 1.0,
            station_keeping_power_w: 0.0,
            payload_power_w: 0.0,
            avionics_power_w: 0.0,
            thermal_management_power_w: 0.0,
            requested_receiver_power_w: 0.0,
        }];

        let result = evaluate_horizon(&config, &profile).unwrap();
        assert_eq!(result.horizon_days, 2);
        assert_eq!(result.days.len(), 2);
        assert!(result.reserve_respected);
        assert!(result.own_service_continuous);
        assert!(result.horizon_energy_cycle_closed);
        assert!((result.days[0].intervals[0].duration_s - DAY_SECONDS).abs() < 1.0e-8);
        assert!((result.days[1].intervals[0].duration_s - DAY_SECONDS).abs() < 1.0e-8);
    }

    #[test]
    fn later_solar_surplus_does_not_erase_prior_own_load_failure() {
        let result = evaluate_horizon(
            &base_config(),
            &unserved_load_recovery_profile(),
        )
        .unwrap();

        assert_eq!(result.horizon_days, 2);
        assert!(result.days[0].reserve_respected);
        assert!(!result.days[0].own_service_continuous);
        assert!(result.days[1].own_service_continuous);
        assert!(result.horizon_energy_cycle_closed);
        assert!(result.ending_soc_wh >= result.initial_soc_wh);
        assert!(!result.own_service_continuous);
        assert!(!result.export_service_admissible);
        assert!((result.total_own_load_unserved_wh - 1200.0).abs() < 1.0e-8);
        assert!((result.ending_soc_wh - 1800.0).abs() < 1.0e-8);
        assert!((result.total_export_source_energy_wh - 2100.0).abs() < 1.0e-8);
        assert!((result.total_export_receiver_energy_wh - 1680.0).abs() < 1.0e-8);
    }

    #[test]
    fn multi_day_horizon_rejects_partial_day_duration() {
        let mut profile = day_night_profile(0.0);
        profile.push(SolarHapsInterval {
            duration_s: 3600.0,
            incident_irradiance_w_m2: 0.0,
            incidence_factor: 1.0,
            pv_temperature_derate: 1.0,
            station_keeping_power_w: 0.0,
            payload_power_w: 0.0,
            avionics_power_w: 0.0,
            thermal_management_power_w: 0.0,
            requested_receiver_power_w: 0.0,
        });

        assert_eq!(
            evaluate_horizon(&base_config(), &profile),
            Err(SolarHapsError::ProfileDurationNotWholeDays)
        );
    }
    
    #[test]
    fn profile_rejects_non_finite_or_negative_physical_inputs() {
        let mut profile = day_night_profile(0.0);
        profile[0].incident_irradiance_w_m2 = f64::NAN;
        assert_eq!(
            evaluate_day(&base_config(), &profile),
            Err(SolarHapsError::NonFiniteInterval)
        );

        profile[0].incident_irradiance_w_m2 = -1.0;
        assert_eq!(
            evaluate_day(&base_config(), &profile),
            Err(SolarHapsError::InvalidIntervalValue)
        );
    }
}
