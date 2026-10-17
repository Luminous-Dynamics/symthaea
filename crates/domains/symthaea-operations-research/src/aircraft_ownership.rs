// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Transparent, assumption-explicit ownership economics for general-aviation
//! aircraft families. This module calculates planning scenarios; it does not
//! supply market prices, guarantee costs, or replace aircraft-specific quotes.
//!
//! Accounting boundaries are intentional:
//! - annual fixed operating cash costs are paid regardless of utilization;
//! - variable cost rates are multiplied by all actual flown hours, including
//!   hours spent on aborted flights;
//! - acquisition economics are converted to an equivalent annual capital cost
//!   using a declared horizon, discount rate and end-of-horizon residual value;
//! - owner time is valued separately instead of silently treated as free;
//! - the additional cost of failed missions and maintenance downtime is explicit
//!   and must not duplicate costs already included elsewhere.
//!
//! For comparisons, candidates must share a mission-contract digest, currency,
//! capital horizon and discount-rate assumption. No FX conversion, hidden default
//! price, aircraft ranking, or composite "winner" score is generated.

use std::collections::BTreeSet;

/// One fixed cost paid annually whether or not the aircraft flies.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedAnnualCashCosts {
    pub insurance: f64,
    pub hangar_or_tiedown: f64,
    pub annual_inspections_and_compliance: f64,
    pub recurrent_training: f64,
    pub subscriptions_and_data: f64,
    pub registration_taxes_and_fees: f64,
    pub other_fixed_costs: f64,
}

impl FixedAnnualCashCosts {
    pub fn total(&self) -> f64 {
        self.insurance
            + self.hangar_or_tiedown
            + self.annual_inspections_and_compliance
            + self.recurrent_training
            + self.subscriptions_and_data
            + self.registration_taxes_and_fees
            + self.other_fixed_costs
    }
}

/// Per-flown-hour cash contributions. Reserve items are budget allocations for
/// future replacement/overhaul, not a claim that a component fails at a fixed
/// interval. Include fuel or electricity at the declared energy boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableCostRates {
    pub fuel_or_energy: f64,
    pub oil_and_fluids: f64,
    pub routine_scheduled_maintenance: f64,
    pub unscheduled_maintenance_allowance: f64,
    pub engine_and_propeller_overhaul_reserve: f64,
    pub avionics_replacement_reserve: f64,
    pub battery_replacement_reserve: f64,
    pub wear_parts_and_consumables: f64,
    pub landing_handling_and_usage_fees: f64,
}

impl VariableCostRates {
    pub fn total_per_flown_hour(&self) -> f64 {
        self.fuel_or_energy
            + self.oil_and_fluids
            + self.routine_scheduled_maintenance
            + self.unscheduled_maintenance_allowance
            + self.engine_and_propeller_overhaul_reserve
            + self.avionics_replacement_reserve
            + self.battery_replacement_reserve
            + self.wear_parts_and_consumables
            + self.landing_handling_and_usage_fees
    }
}

/// Acquisition economics are kept separate from actual monthly loan payments
/// to prevent double counting. This model reports an equivalent annual capital
/// cost, not cash-flow timing, tax treatment or a loan quote.
#[derive(Debug, Clone, PartialEq)]
pub struct CapitalEconomics {
    pub acquisition_price: f64,
    pub residual_value_at_horizon: f64,
    pub horizon_years: u32,
    /// Annual effective discount/capital rate as a fraction; e.g. 0.05 = 5%.
    pub discount_rate_annual: f64,
}

/// Owner labour is shown separately and included in fully loaded economic cost.
/// Do not also include the same owner hours in variable maintenance charges.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnerLaborAssumptions {
    pub hourly_value: f64,
    pub annual_administration_and_logbook_hours: f64,
    pub maintenance_hours_per_flown_hour: f64,
}

/// Annual costs associated with downtime/failed attempts that are *not* already
/// present in hourly operating costs. For example, a replacement ground trip or
/// transport to a repair station may be recorded here. Never add the same fuel,
/// maintenance, or labor expense twice.
#[derive(Debug, Clone, PartialEq)]
pub struct DowntimeCostAssumptions {
    pub annual_downtime_support_cost: f64,
    pub cost_per_aborted_mission_excluding_hourly_costs: f64,
}

/// Measured or explicitly assumed annual utilization. All hours count toward
/// operating costs; successful hours/missions/passenger-km are the delivered
/// service denominator. An aborted attempt may have flown and incurred costs.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnualUtilization {
    pub flown_hours_including_aborts: f64,
    pub successful_flight_hours: f64,
    pub attempted_missions: u64,
    pub completed_missions: u64,
    pub completed_passenger_kilometres: f64,
}

/// Full set of assumptions for one aircraft in one usage scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct AircraftOwnershipScenario {
    pub scenario_id: String,
    pub mission_family_id: String,
    /// Digest of the frozen use-case/mission contract shared by compared cases.
    pub mission_contract_sha256: String,
    /// Three uppercase ASCII letters, such as USD or EUR. This is validated for
    /// syntax only; it is not an ISO registry lookup or FX conversion.
    pub currency_code: String,
    pub capital: CapitalEconomics,
    pub fixed_annual_cash_costs: FixedAnnualCashCosts,
    pub variable_cost_rates: VariableCostRates,
    pub owner_labor: OwnerLaborAssumptions,
    pub downtime: DowntimeCostAssumptions,
    pub utilization: AnnualUtilization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnershipCostError {
    EmptyScenarioSet,
    EmptyField { field: String },
    InvalidCurrencyCode { value: String },
    InvalidDigest { field: String },
    InvalidValue { field: String, reason: String },
    ArithmeticOverflow { field: String },
    DuplicateScenarioId { scenario_id: String },
    MissionFamilyMismatch { scenario_id: String },
    MissionContractMismatch { scenario_id: String },
    CurrencyMismatch { scenario_id: String },
    CapitalHorizonMismatch { scenario_id: String },
    DiscountRateMismatch { scenario_id: String },
}

/// A fully explicit cost breakdown for one annual usage scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct AircraftOwnershipCostResult {
    pub scenario_id: String,
    pub mission_family_id: String,
    pub mission_contract_sha256: String,
    pub currency_code: String,
    pub annual_fixed_operating_cash_cost: f64,
    pub variable_cost_rate_per_flown_hour: f64,
    pub annual_variable_operating_cash_cost: f64,
    pub annual_downtime_and_aborted_mission_cost: f64,
    pub annual_operating_cash_cost: f64,
    pub equivalent_annual_capital_cost: f64,
    pub annual_owner_labor_hours: f64,
    pub annual_owner_labor_economic_cost: f64,
    pub annual_fully_loaded_economic_cost: f64,
    pub annual_flown_hours_including_aborts: f64,
    pub annual_successful_flight_hours: f64,
    pub attempted_missions: u64,
    pub completed_missions: u64,
    pub mission_completion_fraction: Option<f64>,
    pub completed_passenger_kilometres: f64,
    pub cost_per_flown_hour: Option<f64>,
    pub cost_per_successful_flight_hour: Option<f64>,
    pub cost_per_completed_mission: Option<f64>,
    pub cost_per_completed_passenger_kilometre: Option<f64>,
}

/// Compare scenarios by displaying their separate outputs. This intentionally
/// does not choose a single winner or conceal differences behind a weighted
/// score. Inputs must share a mission-contract digest, currency and capital
/// assumptions, but may vary in annual utilization and cost structure.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipScenarioComparison {
    pub mission_family_id: String,
    pub mission_contract_sha256: String,
    pub currency_code: String,
    pub horizon_years: u32,
    pub discount_rate_annual: f64,
    pub results: Vec<AircraftOwnershipCostResult>,
}

pub fn calculate_ownership_cost(
    scenario: &AircraftOwnershipScenario,
) -> Result<AircraftOwnershipCostResult, Vec<OwnershipCostError>> {
    let errors = validate_scenario(scenario);
    if !errors.is_empty() {
        return Err(errors);
    }

    let fixed_annual = scenario.fixed_annual_cash_costs.total();
    let variable_rate = scenario.variable_cost_rates.total_per_flown_hour();
    let variable_annual = variable_rate * scenario.utilization.flown_hours_including_aborts;
    let aborted_missions = scenario.utilization.attempted_missions
        - scenario.utilization.completed_missions;
    let downtime_annual = scenario.downtime.annual_downtime_support_cost
        + scenario.downtime.cost_per_aborted_mission_excluding_hourly_costs
            * aborted_missions as f64;
    let operating_cash_annual = fixed_annual + variable_annual + downtime_annual;
    let capital_annual = equivalent_annual_capital_cost(&scenario.capital);
    let owner_labor_hours = scenario.owner_labor.annual_administration_and_logbook_hours
        + scenario.owner_labor.maintenance_hours_per_flown_hour
            * scenario.utilization.flown_hours_including_aborts;
    let owner_labor_annual = owner_labor_hours * scenario.owner_labor.hourly_value;
    let full_economic_annual = operating_cash_annual + capital_annual + owner_labor_annual;

    let numeric_results = [
        ("fixed_annual", fixed_annual),
        ("variable_rate", variable_rate),
        ("variable_annual", variable_annual),
        ("downtime_annual", downtime_annual),
        ("operating_cash_annual", operating_cash_annual),
        ("capital_annual", capital_annual),
        ("owner_labor_hours", owner_labor_hours),
        ("owner_labor_annual", owner_labor_annual),
        ("full_economic_annual", full_economic_annual),
    ];
    let overflows: Vec<_> = numeric_results
        .iter()
        .filter(|(_, value)| !value.is_finite() || *value < 0.0)
        .map(|(field, _)| OwnershipCostError::ArithmeticOverflow {
            field: (*field).into(),
        })
        .collect();
    if !overflows.is_empty() {
        return Err(overflows);
    }

    let flown_hours = scenario.utilization.flown_hours_including_aborts;
    let successful_hours = scenario.utilization.successful_flight_hours;
    let successful_missions = scenario.utilization.completed_missions as f64;
    let attempted_missions = scenario.utilization.attempted_missions as f64;
    let passenger_km = scenario.utilization.completed_passenger_kilometres;

    Ok(AircraftOwnershipCostResult {
        scenario_id: scenario.scenario_id.clone(),
        mission_family_id: scenario.mission_family_id.clone(),
        mission_contract_sha256: scenario.mission_contract_sha256.clone(),
        currency_code: scenario.currency_code.clone(),
        annual_fixed_operating_cash_cost: fixed_annual,
        variable_cost_rate_per_flown_hour: variable_rate,
        annual_variable_operating_cash_cost: variable_annual,
        annual_downtime_and_aborted_mission_cost: downtime_annual,
        annual_operating_cash_cost: operating_cash_annual,
        equivalent_annual_capital_cost: capital_annual,
        annual_owner_labor_hours: owner_labor_hours,
        annual_owner_labor_economic_cost: owner_labor_annual,
        annual_fully_loaded_economic_cost: full_economic_annual,
        annual_flown_hours_including_aborts: flown_hours,
        annual_successful_flight_hours: successful_hours,
        attempted_missions: scenario.utilization.attempted_missions,
        completed_missions: scenario.utilization.completed_missions,
        mission_completion_fraction: (attempted_missions > 0.0)
            .then_some(successful_missions / attempted_missions),
        completed_passenger_kilometres: passenger_km,
        cost_per_flown_hour: positive_denominator_rate(full_economic_annual, flown_hours),
        cost_per_successful_flight_hour: positive_denominator_rate(
            full_economic_annual,
            successful_hours,
        ),
        cost_per_completed_mission: positive_denominator_rate(
            full_economic_annual,
            successful_missions,
        ),
        cost_per_completed_passenger_kilometre: positive_denominator_rate(
            full_economic_annual,
            passenger_km,
        ),
    })
}

/// Evaluate several alternatives with common comparison assumptions. Results
/// are deterministic and sorted by scenario ID. Different utilization levels
/// are valid comparisons; different mission contracts, currency or capital
/// assumptions are not silently normalized.
pub fn compare_ownership_scenarios(
    scenarios: &[AircraftOwnershipScenario],
) -> Result<OwnershipScenarioComparison, Vec<OwnershipCostError>> {
    if scenarios.is_empty() {
        return Err(vec![OwnershipCostError::EmptyScenarioSet]);
    }

    let mut errors = Vec::new();
    let first = &scenarios[0];
    let mut identifiers = BTreeSet::new();
    for scenario in scenarios {
        errors.extend(validate_scenario(scenario));
        if !identifiers.insert(scenario.scenario_id.as_str()) {
            errors.push(OwnershipCostError::DuplicateScenarioId {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if scenario.mission_family_id != first.mission_family_id {
            errors.push(OwnershipCostError::MissionFamilyMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if !scenario
            .mission_contract_sha256
            .eq_ignore_ascii_case(&first.mission_contract_sha256)
        {
            errors.push(OwnershipCostError::MissionContractMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if scenario.currency_code != first.currency_code {
            errors.push(OwnershipCostError::CurrencyMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if scenario.capital.horizon_years != first.capital.horizon_years {
            errors.push(OwnershipCostError::CapitalHorizonMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        if scenario.capital.discount_rate_annual.to_bits()
            != first.capital.discount_rate_annual.to_bits()
        {
            errors.push(OwnershipCostError::DiscountRateMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut results = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        results.push(
            calculate_ownership_cost(scenario)
                .map_err(|errors| errors)?,
        );
    }
    results.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
    Ok(OwnershipScenarioComparison {
        mission_family_id: first.mission_family_id.clone(),
        mission_contract_sha256: first.mission_contract_sha256.clone(),
        currency_code: first.currency_code.clone(),
        horizon_years: first.capital.horizon_years,
        discount_rate_annual: first.capital.discount_rate_annual,
        results,
    })
}

fn equivalent_annual_capital_cost(capital: &CapitalEconomics) -> f64 {
    let years = capital.horizon_years as f64;
    let rate = capital.discount_rate_annual;
    if rate == 0.0 {
        return (capital.acquisition_price - capital.residual_value_at_horizon) / years;
    }

    let growth = (1.0 + rate).powf(years);
    let capital_recovery_factor = rate * growth / (growth - 1.0);
    let residual_present_value = capital.residual_value_at_horizon / growth;
    (capital.acquisition_price - residual_present_value) * capital_recovery_factor
}

fn positive_denominator_rate(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(numerator / denominator)
}

fn validate_scenario(scenario: &AircraftOwnershipScenario) -> Vec<OwnershipCostError> {
    let mut errors = Vec::new();
    nonempty("scenario_id", &scenario.scenario_id, &mut errors);
    nonempty("mission_family_id", &scenario.mission_family_id, &mut errors);
    if !is_sha256_digest(&scenario.mission_contract_sha256) {
        errors.push(OwnershipCostError::InvalidDigest {
            field: "mission_contract_sha256".into(),
        });
    }
    if scenario.currency_code.len() != 3
        || !scenario.currency_code.bytes().all(|byte| byte.is_ascii_uppercase())
    {
        errors.push(OwnershipCostError::InvalidCurrencyCode {
            value: scenario.currency_code.clone(),
        });
    }

    if scenario.capital.horizon_years == 0 {
        errors.push(invalid("capital.horizon_years", "must be greater than zero"));
    }
    for (field, value) in [
        ("capital.acquisition_price", scenario.capital.acquisition_price),
        ("capital.residual_value_at_horizon", scenario.capital.residual_value_at_horizon),
        ("capital.discount_rate_annual", scenario.capital.discount_rate_annual),
        ("owner_labor.hourly_value", scenario.owner_labor.hourly_value),
        (
            "owner_labor.annual_administration_and_logbook_hours",
            scenario.owner_labor.annual_administration_and_logbook_hours,
        ),
        (
            "owner_labor.maintenance_hours_per_flown_hour",
            scenario.owner_labor.maintenance_hours_per_flown_hour,
        ),
        ("downtime.annual_downtime_support_cost", scenario.downtime.annual_downtime_support_cost),
        (
            "downtime.cost_per_aborted_mission_excluding_hourly_costs",
            scenario.downtime.cost_per_aborted_mission_excluding_hourly_costs,
        ),
        (
            "utilization.flown_hours_including_aborts",
            scenario.utilization.flown_hours_including_aborts,
        ),
        (
            "utilization.successful_flight_hours",
            scenario.utilization.successful_flight_hours,
        ),
        (
            "utilization.completed_passenger_kilometres",
            scenario.utilization.completed_passenger_kilometres,
        ),
    ] {
        finite_nonnegative(field, value, &mut errors);
    }

    if scenario.capital.discount_rate_annual >= 1.0 {
        errors.push(invalid("capital.discount_rate_annual", "must be less than 1.0"));
    }
    if scenario.capital.residual_value_at_horizon > scenario.capital.acquisition_price {
        errors.push(invalid(
            "capital.residual_value_at_horizon",
            "must not exceed the acquisition price under this model",
        ));
    }
    if scenario.utilization.successful_flight_hours
        > scenario.utilization.flown_hours_including_aborts
    {
        errors.push(invalid(
            "utilization.successful_flight_hours",
            "must not exceed all flown hours including aborted attempts",
        ));
    }
    if scenario.utilization.completed_missions > scenario.utilization.attempted_missions {
        errors.push(invalid(
            "utilization.completed_missions",
            "must not exceed attempted missions",
        ));
    }

    let fixed = &scenario.fixed_annual_cash_costs;
    for (field, value) in [
        ("fixed.insurance", fixed.insurance),
        ("fixed.hangar_or_tiedown", fixed.hangar_or_tiedown),
        ("fixed.annual_inspections_and_compliance", fixed.annual_inspections_and_compliance),
        ("fixed.recurrent_training", fixed.recurrent_training),
        ("fixed.subscriptions_and_data", fixed.subscriptions_and_data),
        ("fixed.registration_taxes_and_fees", fixed.registration_taxes_and_fees),
        ("fixed.other_fixed_costs", fixed.other_fixed_costs),
    ] {
        finite_nonnegative(field, value, &mut errors);
    }

    let variable = &scenario.variable_cost_rates;
    for (field, value) in [
        ("variable.fuel_or_energy", variable.fuel_or_energy),
        ("variable.oil_and_fluids", variable.oil_and_fluids),
        ("variable.routine_scheduled_maintenance", variable.routine_scheduled_maintenance),
        ("variable.unscheduled_maintenance_allowance", variable.unscheduled_maintenance_allowance),
        (
            "variable.engine_and_propeller_overhaul_reserve",
            variable.engine_and_propeller_overhaul_reserve,
        ),
        ("variable.avionics_replacement_reserve", variable.avionics_replacement_reserve),
        ("variable.battery_replacement_reserve", variable.battery_replacement_reserve),
        ("variable.wear_parts_and_consumables", variable.wear_parts_and_consumables),
        ("variable.landing_handling_and_usage_fees", variable.landing_handling_and_usage_fees),
    ] {
        finite_nonnegative(field, value, &mut errors);
    }

    errors
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn nonempty(field: &str, value: &str, errors: &mut Vec<OwnershipCostError>) {
    if value.trim().is_empty() || value.trim() != value {
        errors.push(OwnershipCostError::EmptyField {
            field: field.into(),
        });
    }
}

fn finite_nonnegative(field: &str, value: f64, errors: &mut Vec<OwnershipCostError>) {
    if !value.is_finite() || value < 0.0 {
        errors.push(invalid(field, "must be finite and non-negative"));
    }
}

fn invalid(field: &str, reason: &str) -> OwnershipCostError {
    OwnershipCostError::InvalidValue {
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_costs() -> FixedAnnualCashCosts {
        FixedAnnualCashCosts {
            insurance: 3_000.0,
            hangar_or_tiedown: 4_800.0,
            annual_inspections_and_compliance: 1_500.0,
            recurrent_training: 500.0,
            subscriptions_and_data: 300.0,
            registration_taxes_and_fees: 200.0,
            other_fixed_costs: 200.0,
        }
    }

    fn variable_rates() -> VariableCostRates {
        VariableCostRates {
            fuel_or_energy: 80.0,
            oil_and_fluids: 5.0,
            routine_scheduled_maintenance: 20.0,
            unscheduled_maintenance_allowance: 15.0,
            engine_and_propeller_overhaul_reserve: 25.0,
            avionics_replacement_reserve: 6.0,
            battery_replacement_reserve: 0.0,
            wear_parts_and_consumables: 8.0,
            landing_handling_and_usage_fees: 6.0,
        }
    }

    fn scenario(id: &str, flown_hours: f64) -> AircraftOwnershipScenario {
        AircraftOwnershipScenario {
            scenario_id: id.into(),
            mission_family_id: "two-seat-local-training-v1".into(),
            mission_contract_sha256: "a".repeat(64),
            currency_code: "USD".into(),
            capital: CapitalEconomics {
                acquisition_price: 150_000.0,
                residual_value_at_horizon: 90_000.0,
                horizon_years: 10,
                discount_rate_annual: 0.0,
            },
            fixed_annual_cash_costs: fixed_costs(),
            variable_cost_rates: variable_rates(),
            owner_labor: OwnerLaborAssumptions {
                hourly_value: 30.0,
                annual_administration_and_logbook_hours: 25.0,
                maintenance_hours_per_flown_hour: 0.2,
            },
            downtime: DowntimeCostAssumptions {
                annual_downtime_support_cost: 400.0,
                cost_per_aborted_mission_excluding_hourly_costs: 75.0,
            },
            utilization: AnnualUtilization {
                flown_hours_including_aborts: flown_hours,
                successful_flight_hours: flown_hours * 0.95,
                attempted_missions: (flown_hours / 1.5).round() as u64,
                completed_missions: ((flown_hours / 1.5).round() as u64).saturating_sub(1),
                completed_passenger_kilometres: flown_hours * 90.0,
            },
        }
    }

    #[test]
    fn acquisition_capital_cost_respects_horizon_residual_and_zero_discount() {
        let result = calculate_ownership_cost(&scenario("base", 100.0)).unwrap();
        assert_eq!(result.equivalent_annual_capital_cost, 6_000.0);
    }

    #[test]
    fn low_utilization_has_higher_fully_loaded_hourly_cost() {
        let low = calculate_ownership_cost(&scenario("low", 50.0)).unwrap();
        let high = calculate_ownership_cost(&scenario("high", 200.0)).unwrap();
        assert!(low.cost_per_flown_hour.unwrap() > high.cost_per_flown_hour.unwrap());
        assert!(low.annual_fully_loaded_economic_cost < high.annual_fully_loaded_economic_cost);
    }

    #[test]
    fn capital_rate_and_horizon_change_equivalent_annual_capital_cost() {
        let mut base = scenario("base", 100.0);
        let zero = calculate_ownership_cost(&base).unwrap().equivalent_annual_capital_cost;
        base.capital.discount_rate_annual = 0.05;
        let discounted = calculate_ownership_cost(&base).unwrap().equivalent_annual_capital_cost;
        assert!(discounted > zero);
        base.capital.horizon_years = 20;
        let longer = calculate_ownership_cost(&base).unwrap().equivalent_annual_capital_cost;
        assert!(longer < discounted);
    }

    #[test]
    fn owner_labor_is_explicit_and_included_in_economic_cost() {
        let base = calculate_ownership_cost(&scenario("base", 100.0)).unwrap();
        let mut no_owner_labor = scenario("no-owner-labor", 100.0);
        no_owner_labor.owner_labor.hourly_value = 0.0;
        let zero_labor = calculate_ownership_cost(&no_owner_labor).unwrap();
        assert!(base.annual_fully_loaded_economic_cost > zero_labor.annual_fully_loaded_economic_cost);
        assert!(base.annual_operating_cash_cost == zero_labor.annual_operating_cash_cost);
    }

    #[test]
    fn no_utilization_returns_unavailable_hourly_metrics_not_nan() {
        let result = calculate_ownership_cost(&scenario("zero-hours", 0.0)).unwrap();
        assert_eq!(result.cost_per_flown_hour, None);
        assert_eq!(result.cost_per_successful_flight_hour, None);
        assert_eq!(result.cost_per_completed_passenger_kilometre, None);
    }

    #[test]
    fn comparison_rejects_different_mission_contracts_and_currencies() {
        let mut second = scenario("second", 100.0);
        second.mission_contract_sha256 = "b".repeat(64);
        assert!(compare_ownership_scenarios(&[scenario("first", 100.0), second.clone()]).is_err());
        second.mission_contract_sha256 = "a".repeat(64);
        second.currency_code = "EUR".into();
        assert!(compare_ownership_scenarios(&[scenario("first", 100.0), second]).is_err());
    }

    #[test]
    fn comparison_is_deterministic_and_sorted_by_scenario_id() {
        let report = compare_ownership_scenarios(&[
            scenario("normal", 100.0),
            scenario("low", 50.0),
            scenario("high", 200.0),
        ]).unwrap();
        assert_eq!(
            report.results.iter().map(|result| result.scenario_id.as_str()).collect::<Vec<_>>(),
            vec!["high", "low", "normal"]
        );
    }

    #[test]
    fn missing_successful_missions_yields_none_per_mission() {
        let mut input = scenario("no-completions", 100.0);
        input.utilization.completed_missions = 0;
        input.utilization.attempted_missions = 0;
        let result = calculate_ownership_cost(&input).unwrap();
        assert_eq!(result.cost_per_completed_mission, None);
        assert_eq!(result.mission_completion_fraction, None);
    }

    #[test]
    fn invalid_costs_counts_and_currency_fail_closed() {
        let mut input = scenario("invalid", 100.0);
        input.fixed_annual_cash_costs.insurance = f64::NAN;
        input.utilization.completed_missions = input.utilization.attempted_missions + 1;
        input.currency_code = "usd".into();
        assert!(calculate_ownership_cost(&input).is_err());
    }

    #[test]
    fn duplicate_ids_are_rejected_in_comparison() {
        assert!(matches!(
            compare_ownership_scenarios(&[scenario("same", 50.0), scenario("same", 100.0)]),
            Err(errors) if errors.iter().any(|error| matches!(error,
                OwnershipCostError::DuplicateScenarioId { scenario_id } if scenario_id == "same"
            ))
        ));
    }

    #[test]
    fn completed_passenger_kilometres_are_the_delivered_service_denominator() {
        let input = scenario("delivered-service", 100.0);
        let result = calculate_ownership_cost(&input).unwrap();
        assert!(result.cost_per_completed_passenger_kilometre.unwrap() > 0.0);
        assert_eq!(
            result.cost_per_completed_passenger_kilometre,
            Some(result.annual_fully_loaded_economic_cost / (100.0 * 90.0))
        );
    }
}
