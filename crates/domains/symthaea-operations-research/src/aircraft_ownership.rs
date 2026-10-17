// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bounded aircraft ownership economics.
//!
//! This module compares annual cost and completed passenger-kilometres under
//! one declared mission family. It does not estimate costs from market prices,
//! predict reliability, calculate aircraft performance, or provide a quote.
//! Monetary inputs must all use the declared currency and a common cost
//! boundary. Annual capital charge and all reserve rates are explicit analyst
//! inputs; this crate does not infer financing, tax, depreciation, insurance,
//! or overhaul prices.

use std::collections::HashSet;

/// Annual fixed ownership costs in the scenario's declared currency units.
///
/// annual_capital_charge must be calculated and documented by the caller.
/// Do not also include the same acquisition/depreciation/finance charge in
/// another field.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AnnualFixedCosts {
    pub annual_capital_charge: f64,
    pub annual_insurance: f64,
    pub annual_storage_and_hangar: f64,
    pub annual_calendar_inspection: f64,
    pub annual_taxes_registration_and_fees: f64,
    pub annual_other_fixed: f64,
}

impl AnnualFixedCosts {
    fn total(&self) -> f64 {
        self.annual_capital_charge
            + self.annual_insurance
            + self.annual_storage_and_hangar
            + self.annual_calendar_inspection
            + self.annual_taxes_registration_and_fees
            + self.annual_other_fixed
    }

    fn fields(&self) -> [(&'static str, f64); 6] {
        [
            ("annual_capital_charge", self.annual_capital_charge),
            ("annual_insurance", self.annual_insurance),
            ("annual_storage_and_hangar", self.annual_storage_and_hangar),
            ("annual_calendar_inspection", self.annual_calendar_inspection),
            ("annual_taxes_registration_and_fees", self.annual_taxes_registration_and_fees),
            ("annual_other_fixed", self.annual_other_fixed),
        ]
    }
}

/// Costs that scale with actual flown hours.
///
/// These can include fuel/energy, oil/consumables, time-based maintenance,
/// maintenance labor/parts, and explicit per-hour engine/propeller/battery
/// reserves. Do not put calendar inspection costs here if they are already in
/// AnnualFixedCosts.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HourlyOperatingCosts {
    pub fuel_or_energy: f64,
    pub oil_and_consumables: f64,
    pub maintenance_labor: f64,
    pub maintenance_parts: f64,
    pub engine_overhaul_reserve: f64,
    pub propeller_reserve: f64,
    pub battery_replacement_reserve: f64,
    pub other_hourly: f64,
}

impl HourlyOperatingCosts {
    fn total(&self) -> f64 {
        self.fuel_or_energy
            + self.oil_and_consumables
            + self.maintenance_labor
            + self.maintenance_parts
            + self.engine_overhaul_reserve
            + self.propeller_reserve
            + self.battery_replacement_reserve
            + self.other_hourly
    }

    fn fields(&self) -> [(&'static str, f64); 8] {
        [
            ("fuel_or_energy", self.fuel_or_energy),
            ("oil_and_consumables", self.oil_and_consumables),
            ("maintenance_labor", self.maintenance_labor),
            ("maintenance_parts", self.maintenance_parts),
            ("engine_overhaul_reserve", self.engine_overhaul_reserve),
            ("propeller_reserve", self.propeller_reserve),
            ("battery_replacement_reserve", self.battery_replacement_reserve),
            ("other_hourly", self.other_hourly),
        ]
    }
}

/// Costs incurred per scheduled dispatch attempt, including attempts that are
/// cancelled before a completed mission. A completed-mission cost can represent
/// landing/handling fees or other cost incurred only for completed sorties.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PerMissionCosts {
    pub preparation_per_attempt: f64,
    pub failed_dispatch_per_attempt: f64,
    pub completed_mission_per_success: f64,
}

/// One aircraft evaluated against one frozen intended-use mission family.
///
/// annual_planned_attempts counts planned dispatch attempts, not completed
/// sorties. dispatch_success_fraction is an observed or explicitly assumed
/// fraction in (0, 1]; its source and confidence should be recorded by the caller.
/// annual_scheduling_budget_hours must accommodate every planned attempt at the
/// candidate's declared mission block time, preventing physically impossible
/// annual schedules from being compared accidentally.
#[derive(Debug, Clone, PartialEq)]
pub struct AircraftOwnershipScenario {
    pub aircraft_id: String,
    pub mission_family_id: String,
    /// SHA-256 digest of the canonical mission definition and common operating
    /// assumptions; syntax/equality only are checked by this module.
    pub mission_family_sha256: String,
    /// ISO 4217-style three-letter code; this crate performs no currency conversion.
    pub currency_code: String,
    pub annual_planned_attempts: u32,
    pub annual_scheduling_budget_hours: f64,
    pub mission_block_hours: f64,
    pub route_distance_km: f64,
    /// Number of passengers transported, excluding flight crew.
    pub passengers_per_completed_mission: u32,
    /// Fraction of planned dispatch attempts expected to complete the mission.
    pub dispatch_success_fraction: f64,
    pub annual_fixed_costs: AnnualFixedCosts,
    pub hourly_costs: HourlyOperatingCosts,
    pub per_mission_costs: PerMissionCosts,
}

/// Explicit validation or comparison failure. A failure is not silently mapped
/// to zero cost or success.
#[derive(Debug, Clone, PartialEq)]
pub enum OwnershipCostError {
    EmptyAircraftId,
    DuplicateAircraftId { aircraft_id: String },
    EmptyMissionFamilyId,
    InvalidMissionFamilyDigest,
    InvalidCurrencyCode,
    ZeroPlannedAttempts,
    ZeroPassengerCount,
    InvalidValue { field: &'static str },
    ScheduleBudgetExceeded {
        scheduled_hours: f64,
        available_hours: f64,
    },
    ArithmeticOverflow { field: &'static str },
    ZeroDenominator { metric: &'static str },
    MissionFamilyMismatch { aircraft_id: String },
    MissionDigestMismatch { aircraft_id: String },
    CurrencyMismatch { aircraft_id: String },
    PlannedAttemptsMismatch { aircraft_id: String },
    PassengerCountMismatch { aircraft_id: String },
    RouteDistanceMismatch { aircraft_id: String },
    ScheduleBudgetMismatch { aircraft_id: String },
}

/// Cost decomposition in the scenario's declared currency units.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AnnualCostBreakdown {
    pub annual_capital_charge: f64,
    pub annual_insurance: f64,
    pub annual_storage_and_hangar: f64,
    pub annual_calendar_inspection: f64,
    pub annual_taxes_registration_and_fees: f64,
    pub annual_other_fixed: f64,
    pub annual_fixed_costs: f64,
    pub preparation_costs_all_attempts: f64,
    pub failed_dispatch_costs: f64,
    pub completed_mission_costs: f64,
    pub fuel_or_energy_costs: f64,
    pub oil_and_consumables_costs: f64,
    pub maintenance_labor_costs: f64,
    pub maintenance_parts_costs: f64,
    pub engine_overhaul_reserve_costs: f64,
    pub propeller_reserve_costs: f64,
    pub battery_replacement_reserve_costs: f64,
    pub other_hourly_costs: f64,
    pub variable_flight_hour_costs: f64,
    pub annual_total_cost: f64,
}

/// Annual expected ownership metrics. Attempt/completion counts are expected
/// values and can be fractional when the input dispatch rate is a probability.
/// The underlying planned-attempt denominator remains explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct AircraftOwnershipReport {
    pub aircraft_id: String,
    pub mission_family_id: String,
    pub currency_code: String,
    pub annual_planned_attempts: u32,
    pub dispatch_success_fraction: f64,
    pub expected_completed_missions: f64,
    pub expected_failed_dispatches: f64,
    pub expected_flown_hours: f64,
    pub completed_passenger_km: f64,
    pub breakdown: AnnualCostBreakdown,
    /// Fully loaded annual ownership cost divided by expected completed hours.
    pub fully_loaded_cost_per_flown_hour: f64,
    /// Fully loaded annual ownership cost divided by expected completed missions.
    pub cost_per_completed_mission: f64,
    /// Fully loaded annual ownership cost divided by completed passenger-kilometres.
    pub cost_per_completed_passenger_km: f64,
    /// Cost that scales with one completed flight hour, before annual fixed costs.
    pub variable_cost_per_flight_hour: f64,
}

/// Evaluate one scenario without inserting default prices, reserves, or
/// reliability values.
pub fn evaluate_ownership_scenario(
    scenario: &AircraftOwnershipScenario,
) -> Result<AircraftOwnershipReport, OwnershipCostError> {
    validate_scenario(scenario)?;

    let attempts = f64::from(scenario.annual_planned_attempts);
    let success = scenario.dispatch_success_fraction;
    let completed = attempts * success;
    let failed = attempts * (1.0 - success);
    let flown_hours = completed * scenario.mission_block_hours;
    let completed_passenger_km = completed
        * f64::from(scenario.passengers_per_completed_mission)
        * scenario.route_distance_km;

    if !completed.is_finite() || !failed.is_finite() || !flown_hours.is_finite() {
        return Err(OwnershipCostError::ArithmeticOverflow {
            field: "annual expected mission counts/hours",
        });
    }
    if !completed_passenger_km.is_finite() {
        return Err(OwnershipCostError::ArithmeticOverflow {
            field: "completed_passenger_km",
        });
    }
    if completed <= 0.0 {
        return Err(OwnershipCostError::ZeroDenominator {
            metric: "cost per completed mission",
        });
    }
    if flown_hours <= 0.0 {
        return Err(OwnershipCostError::ZeroDenominator {
            metric: "fully loaded cost per flown hour",
        });
    }
    if completed_passenger_km <= 0.0 {
        return Err(OwnershipCostError::ZeroDenominator {
            metric: "cost per completed passenger-km",
        });
    }

    let fixed = scenario.annual_fixed_costs.total();
    let prep = attempts * scenario.per_mission_costs.preparation_per_attempt;
    let failed_cost = failed * scenario.per_mission_costs.failed_dispatch_per_attempt;
    let completed_cost = completed * scenario.per_mission_costs.completed_mission_per_success;
    let fixed_inputs = scenario.annual_fixed_costs;
    let hourly_inputs = scenario.hourly_costs;
    let fuel_or_energy_costs = flown_hours * hourly_inputs.fuel_or_energy;
    let oil_and_consumables_costs = flown_hours * hourly_inputs.oil_and_consumables;
    let maintenance_labor_costs = flown_hours * hourly_inputs.maintenance_labor;
    let maintenance_parts_costs = flown_hours * hourly_inputs.maintenance_parts;
    let engine_overhaul_reserve_costs = flown_hours * hourly_inputs.engine_overhaul_reserve;
    let propeller_reserve_costs = flown_hours * hourly_inputs.propeller_reserve;
    let battery_replacement_reserve_costs = flown_hours * hourly_inputs.battery_replacement_reserve;
    let other_hourly_costs = flown_hours * hourly_inputs.other_hourly;
    let hourly = fuel_or_energy_costs
        + oil_and_consumables_costs
        + maintenance_labor_costs
        + maintenance_parts_costs
        + engine_overhaul_reserve_costs
        + propeller_reserve_costs
        + battery_replacement_reserve_costs
        + other_hourly_costs;
    let annual_total = fixed + prep + failed_cost + completed_cost + hourly;

    for (field, value) in [
        ("annual_capital_charge", fixed_inputs.annual_capital_charge),
        ("annual_insurance", fixed_inputs.annual_insurance),
        ("annual_storage_and_hangar", fixed_inputs.annual_storage_and_hangar),
        ("annual_calendar_inspection", fixed_inputs.annual_calendar_inspection),
        ("annual_taxes_registration_and_fees", fixed_inputs.annual_taxes_registration_and_fees),
        ("annual_other_fixed", fixed_inputs.annual_other_fixed),
        ("annual_fixed_costs", fixed),
        ("preparation_costs_all_attempts", prep),
        ("failed_dispatch_costs", failed_cost),
        ("completed_mission_costs", completed_cost),
        ("fuel_or_energy_costs", fuel_or_energy_costs),
        ("oil_and_consumables_costs", oil_and_consumables_costs),
        ("maintenance_labor_costs", maintenance_labor_costs),
        ("maintenance_parts_costs", maintenance_parts_costs),
        ("engine_overhaul_reserve_costs", engine_overhaul_reserve_costs),
        ("propeller_reserve_costs", propeller_reserve_costs),
        ("battery_replacement_reserve_costs", battery_replacement_reserve_costs),
        ("other_hourly_costs", other_hourly_costs),
        ("variable_flight_hour_costs", hourly),
        ("annual_total_cost", annual_total),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(OwnershipCostError::ArithmeticOverflow { field });
        }
    }

    let fully_loaded_cost_per_flown_hour = annual_total / flown_hours;
    let cost_per_completed_mission = annual_total / completed;
    let cost_per_completed_passenger_km = annual_total / completed_passenger_km;
    let variable_cost_per_flight_hour = scenario.hourly_costs.total();
    for (field, value) in [
        ("fully_loaded_cost_per_flown_hour", fully_loaded_cost_per_flown_hour),
        ("cost_per_completed_mission", cost_per_completed_mission),
        ("cost_per_completed_passenger_km", cost_per_completed_passenger_km),
        ("variable_cost_per_flight_hour", variable_cost_per_flight_hour),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(OwnershipCostError::ArithmeticOverflow { field });
        }
    }

    Ok(AircraftOwnershipReport {
        aircraft_id: scenario.aircraft_id.clone(),
        mission_family_id: scenario.mission_family_id.clone(),
        currency_code: scenario.currency_code.clone(),
        annual_planned_attempts: scenario.annual_planned_attempts,
        dispatch_success_fraction: success,
        expected_completed_missions: completed,
        expected_failed_dispatches: failed,
        expected_flown_hours: flown_hours,
        completed_passenger_km,
        breakdown: AnnualCostBreakdown {
            annual_capital_charge: fixed_inputs.annual_capital_charge,
            annual_insurance: fixed_inputs.annual_insurance,
            annual_storage_and_hangar: fixed_inputs.annual_storage_and_hangar,
            annual_calendar_inspection: fixed_inputs.annual_calendar_inspection,
            annual_taxes_registration_and_fees: fixed_inputs.annual_taxes_registration_and_fees,
            annual_other_fixed: fixed_inputs.annual_other_fixed,
            annual_fixed_costs: fixed,
            preparation_costs_all_attempts: prep,
            failed_dispatch_costs: failed_cost,
            completed_mission_costs: completed_cost,
            fuel_or_energy_costs,
            oil_and_consumables_costs,
            maintenance_labor_costs,
            maintenance_parts_costs,
            engine_overhaul_reserve_costs,
            propeller_reserve_costs,
            battery_replacement_reserve_costs,
            other_hourly_costs,
            variable_flight_hour_costs: hourly,
            annual_total_cost: annual_total,
        },
        fully_loaded_cost_per_flown_hour,
        cost_per_completed_mission,
        cost_per_completed_passenger_km,
        variable_cost_per_flight_hour,
    })
}

/// Evaluate multiple aircraft only when their mission family, currency, annual
/// attempt count, route distance, passenger count and scheduled time budget are
/// comparable. Candidate mission duration, cost and dispatch reliability may
/// differ. Results are sorted by fully loaded cost per completed passenger-km,
/// then aircraft ID for deterministic ties. This is an economics ranking only.
pub fn compare_ownership_scenarios(
    scenarios: &[AircraftOwnershipScenario],
) -> Result<Vec<AircraftOwnershipReport>, OwnershipCostError> {
    let Some(reference) = scenarios.first() else {
        return Err(OwnershipCostError::EmptyScenarioSet);
    };
    validate_scenario(reference)?;
    let mut ids = HashSet::with_capacity(scenarios.len());
    ids.insert(reference.aircraft_id.as_str());

    for scenario in scenarios.iter().skip(1) {
        validate_scenario(scenario)?;
        if !ids.insert(scenario.aircraft_id.as_str()) {
            return Err(OwnershipCostError::DuplicateAircraftId {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if scenario.mission_family_id != reference.mission_family_id {
            return Err(OwnershipCostError::MissionFamilyMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if !scenario.mission_family_sha256.eq_ignore_ascii_case(
            &reference.mission_family_sha256,
        ) {
            return Err(OwnershipCostError::MissionDigestMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if scenario.currency_code != reference.currency_code {
            return Err(OwnershipCostError::CurrencyMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if scenario.annual_planned_attempts != reference.annual_planned_attempts {
            return Err(OwnershipCostError::PlannedAttemptsMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if scenario.passengers_per_completed_mission != reference.passengers_per_completed_mission {
            return Err(OwnershipCostError::PassengerCountMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if !approximately_equal(scenario.route_distance_km, reference.route_distance_km) {
            return Err(OwnershipCostError::RouteDistanceMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
        if !approximately_equal(
            scenario.annual_scheduling_budget_hours,
            reference.annual_scheduling_budget_hours,
        ) {
            return Err(OwnershipCostError::ScheduleBudgetMismatch {
                aircraft_id: scenario.aircraft_id.clone(),
            });
        }
    }

    let mut reports: Vec<_> = scenarios
        .iter()
        .map(evaluate_ownership_scenario)
        .collect::<Result<_, _>>()?;
    reports.sort_by(|a, b| {
        a.cost_per_completed_passenger_km
            .total_cmp(&b.cost_per_completed_passenger_km)
            .then_with(|| a.aircraft_id.cmp(&b.aircraft_id))
    });
    Ok(reports)
}

fn validate_scenario(scenario: &AircraftOwnershipScenario) -> Result<(), OwnershipCostError> {
    if scenario.aircraft_id.trim().is_empty()
        || scenario.aircraft_id.trim() != scenario.aircraft_id.as_str()
    {
        return Err(OwnershipCostError::EmptyAircraftId);
    }
    if scenario.mission_family_id.trim().is_empty()
        || scenario.mission_family_id.trim() != scenario.mission_family_id.as_str()
    {
        return Err(OwnershipCostError::EmptyMissionFamilyId);
    }
    if !is_sha256_digest(&scenario.mission_family_sha256) {
        return Err(OwnershipCostError::InvalidMissionFamilyDigest);
    }
    if scenario.currency_code.len() != 3
        || !scenario.currency_code.bytes().all(|b| b.is_ascii_uppercase())
    {
        return Err(OwnershipCostError::InvalidCurrencyCode);
    }
    if scenario.annual_planned_attempts == 0 {
        return Err(OwnershipCostError::ZeroPlannedAttempts);
    }
    if scenario.passengers_per_completed_mission == 0 {
        return Err(OwnershipCostError::ZeroPassengerCount);
    }
    for (field, value) in [
        ("annual_scheduling_budget_hours", scenario.annual_scheduling_budget_hours),
        ("mission_block_hours", scenario.mission_block_hours),
        ("route_distance_km", scenario.route_distance_km),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(OwnershipCostError::InvalidValue { field });
        }
    }
    if !scenario.dispatch_success_fraction.is_finite()
        || scenario.dispatch_success_fraction <= 0.0
        || scenario.dispatch_success_fraction > 1.0
    {
        return Err(OwnershipCostError::InvalidValue {
            field: "dispatch_success_fraction",
        });
    }

    let scheduled_hours =
        f64::from(scenario.annual_planned_attempts) * scenario.mission_block_hours;
    if !scheduled_hours.is_finite() {
        return Err(OwnershipCostError::ArithmeticOverflow {
            field: "scheduled_hours",
        });
    }
    if scheduled_hours > scenario.annual_scheduling_budget_hours {
        return Err(OwnershipCostError::ScheduleBudgetExceeded {
            scheduled_hours,
            available_hours: scenario.annual_scheduling_budget_hours,
        });
    }

    for (field, value) in scenario.annual_fixed_costs.fields() {
        validate_cost(field, value)?;
    }
    for (field, value) in scenario.hourly_costs.fields() {
        validate_cost(field, value)?;
    }
    for (field, value) in [
        ("preparation_per_attempt", scenario.per_mission_costs.preparation_per_attempt),
        ("failed_dispatch_per_attempt", scenario.per_mission_costs.failed_dispatch_per_attempt),
        ("completed_mission_per_success", scenario.per_mission_costs.completed_mission_per_success),
    ] {
        validate_cost(field, value)?;
    }

    let fixed = scenario.annual_fixed_costs.total();
    let hourly = scenario.hourly_costs.total();
    if !fixed.is_finite() || !hourly.is_finite() {
        return Err(OwnershipCostError::ArithmeticOverflow {
            field: "input cost component total",
        });
    }
    Ok(())
}

fn validate_cost(field: &'static str, value: f64) -> Result<(), OwnershipCostError> {
    if !value.is_finite() || value < 0.0 {
        Err(OwnershipCostError::InvalidValue { field })
    } else {
        Ok(())
    }
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(aircraft_id: &str, attempts: u32) -> AircraftOwnershipScenario {
        AircraftOwnershipScenario {
            aircraft_id: aircraft_id.into(),
            mission_family_id: "s4-personal-150nm-v1".into(),
            mission_family_sha256: "a".repeat(64),
            currency_code: "USD".into(),
            annual_planned_attempts: attempts,
            annual_scheduling_budget_hours: 500.0,
            mission_block_hours: 1.5,
            route_distance_km: 200.0,
            passengers_per_completed_mission: 2,
            dispatch_success_fraction: 0.9,
            annual_fixed_costs: AnnualFixedCosts {
                annual_capital_charge: 12_000.0,
                annual_insurance: 2_000.0,
                annual_storage_and_hangar: 1_000.0,
                annual_calendar_inspection: 500.0,
                annual_taxes_registration_and_fees: 0.0,
                annual_other_fixed: 500.0,
            },
            hourly_costs: HourlyOperatingCosts {
                fuel_or_energy: 60.0,
                oil_and_consumables: 5.0,
                maintenance_labor: 25.0,
                maintenance_parts: 20.0,
                engine_overhaul_reserve: 30.0,
                propeller_reserve: 5.0,
                battery_replacement_reserve: 0.0,
                other_hourly: 10.0,
            },
            per_mission_costs: PerMissionCosts {
                preparation_per_attempt: 20.0,
                failed_dispatch_per_attempt: 100.0,
                completed_mission_per_success: 50.0,
            },
        }
    }

    #[test]
    fn known_answer_includes_fixed_costs_failed_dispatches_and_reserves() {
        let report = evaluate_ownership_scenario(&scenario("s4", 100)).unwrap();
        assert!((report.expected_completed_missions - 90.0).abs() < 1e-12);
        assert!((report.expected_failed_dispatches - 10.0).abs() < 1e-12);
        assert!((report.expected_flown_hours - 135.0).abs() < 1e-12);
        assert!((report.completed_passenger_km - 36_000.0).abs() < 1e-12);
        assert!((report.breakdown.annual_fixed_costs - 16_000.0).abs() < 1e-12);
        assert!((report.breakdown.preparation_costs_all_attempts - 2_000.0).abs() < 1e-12);
        assert!((report.breakdown.failed_dispatch_costs - 1_000.0).abs() < 1e-12);
        assert!((report.breakdown.completed_mission_costs - 4_500.0).abs() < 1e-12);
        assert!((report.breakdown.variable_flight_hour_costs - 20_925.0).abs() < 1e-12);
        assert!((report.breakdown.annual_total_cost - 44_425.0).abs() < 1e-12);
        assert!((report.fully_loaded_cost_per_flown_hour - 44_425.0 / 135.0).abs() < 1e-9);
        assert!((report.cost_per_completed_mission - 44_425.0 / 90.0).abs() < 1e-9);
        assert!((report.cost_per_completed_passenger_km - 44_425.0 / 36_000.0).abs() < 1e-12);
    }

    #[test]
    fn more_failed_dispatches_raise_cost_per_completed_passenger_km() {
        let baseline = evaluate_ownership_scenario(&scenario("base", 100)).unwrap();
        let mut worse = scenario("worse", 100);
        worse.dispatch_success_fraction = 0.7;
        let worse = evaluate_ownership_scenario(&worse).unwrap();
        assert!(worse.expected_failed_dispatches > baseline.expected_failed_dispatches);
        assert!(worse.cost_per_completed_passenger_km > baseline.cost_per_completed_passenger_km);
    }

    #[test]
    fn zero_fixed_or_hourly_component_is_allowed_when_explicit() {
        let mut input = scenario("s4", 1);
        input.annual_fixed_costs.annual_taxes_registration_and_fees = 0.0;
        input.hourly_costs.battery_replacement_reserve = 0.0;
        assert!(evaluate_ownership_scenario(&input).is_ok());
    }

    #[test]
    fn negative_or_non_finite_costs_fail_closed() {
        let mut negative = scenario("s4", 1);
        negative.hourly_costs.engine_overhaul_reserve = -1.0;
        assert_eq!(
            evaluate_ownership_scenario(&negative),
            Err(OwnershipCostError::InvalidValue {
                field: "engine_overhaul_reserve"
            })
        );

        let mut nan = scenario("s4", 1);
        nan.annual_fixed_costs.annual_insurance = f64::NAN;
        assert_eq!(
            evaluate_ownership_scenario(&nan),
            Err(OwnershipCostError::InvalidValue {
                field: "annual_insurance"
            })
        );
    }

    #[test]
    fn mismatched_currencies_and_mission_families_cannot_be_ranked() {
        let usd = scenario("alpha", 20);
        let mut zar = scenario("beta", 20);
        zar.currency_code = "ZAR".into();
        assert_eq!(
            compare_ownership_scenarios(&[usd.clone(), zar]),
            Err(OwnershipCostError::CurrencyMismatch {
                aircraft_id: "beta".into()
            })
        );

        let mut different_mission = scenario("gamma", 20);
        different_mission.mission_family_id = "different-route".into();
        assert_eq!(
            compare_ownership_scenarios(&[usd, different_mission]),
            Err(OwnershipCostError::MissionFamilyMismatch {
                aircraft_id: "gamma".into()
            })
        );
    }

    #[test]
    fn mission_family_digest_must_match() {
        let first = scenario("alpha", 20);
        let mut second = scenario("beta", 20);
        second.mission_family_sha256 = "b".repeat(64);
        assert_eq!(
            compare_ownership_scenarios(&[first, second]),
            Err(OwnershipCostError::MissionDigestMismatch {
                aircraft_id: "beta".into()
            })
        );
    }

    #[test]
    fn schedule_budget_and_zero_denominators_fail_closed() {
        let mut input = scenario("s4", 100);
        input.annual_scheduling_budget_hours = 100.0;
        assert!(matches!(
            evaluate_ownership_scenario(&input),
            Err(OwnershipCostError::ScheduleBudgetExceeded { .. })
        ));

        let mut impossible = scenario("zero", 1);
        impossible.dispatch_success_fraction = 0.0;
        assert_eq!(
            evaluate_ownership_scenario(&impossible),
            Err(OwnershipCostError::InvalidValue {
                field: "dispatch_success_fraction"
            })
        );
    }

    #[test]
    fn comparisons_are_deterministic_and_sorted_by_cost() {
        let mut expensive = scenario("expensive", 100);
        expensive.hourly_costs.fuel_or_energy = 120.0;
        let cheap = scenario("cheap", 100);
        let first = compare_ownership_scenarios(&[expensive.clone(), cheap.clone()]).unwrap();
        let second = compare_ownership_scenarios(&[cheap, expensive]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first[0].aircraft_id, "cheap");
        assert_eq!(first[1].aircraft_id, "expensive");
    }

    #[test]
    fn route_distance_and_passenger_profile_must_match_for_comparison() {
        let first = scenario("alpha", 20);
        let mut second = scenario("beta", 20);
        second.route_distance_km = 250.0;
        assert_eq!(
            compare_ownership_scenarios(&[first.clone(), second]),
            Err(OwnershipCostError::RouteDistanceMismatch {
                aircraft_id: "beta".into()
            })
        );

        let mut second = scenario("beta", 20);
        second.passengers_per_completed_mission = 1;
        assert_eq!(
            compare_ownership_scenarios(&[first, second]),
            Err(OwnershipCostError::PassengerCountMismatch {
                aircraft_id: "beta".into()
            })
        );
    }

    #[test]
    fn duplicate_aircraft_ids_are_rejected() {
        assert_eq!(
            compare_ownership_scenarios(&[scenario("same", 10), scenario("same", 10)]),
            Err(OwnershipCostError::DuplicateAircraftId {
                aircraft_id: "same".into()
            })
        );
    }
}
