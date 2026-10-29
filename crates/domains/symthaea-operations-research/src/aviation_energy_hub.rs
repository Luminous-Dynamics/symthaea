// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-first, time-indexed airport electricity/storage benchmark.
//!
//! V1 is a small discrete model, not a power-flow solver or airline dispatch system.
//! Renewable generation is a fixed scenario input and its cost/lifecycle emissions are
//! counted even when energy is curtailed. Grid imports have interval-specific limits,
//! prices and lifecycle factors. Storage starts empty, charges only from renewable
//! surplus, and obeys integer-quantum charge/discharge efficiencies, power limits and
//! a terminal reserve. Grid-to-storage charging, A2G, network flows, electrical losses
//! upstream of the declared bus, physical aircraft constraints and non-CO2 effects
//! are out of scope.
//!
//! A plan is re-evaluated independently from its claimed actions. Exact search returns
//! a Pareto frontier across cost, accounted lifecycle CO2e, and terminal stored energy;
//! budget exhaustion never returns a partial frontier labelled exact.

use std::collections::BTreeSet;
use std::fmt;

use sha2::{Digest, Sha256};

use crate::aviation_energy_accounting::{
    account_energy_ledger, AccountingError, AccountingSummary, EnergyCarrier, EnergyRecord,
    EvidenceClass, NonCo2Evidence, TransportService,
};

const MAX_INTERVALS: usize = 8;
const MAX_QUANTA_PER_INTERVAL: u32 = 32;
const MAX_TOTAL_DEMAND_QUANTA: u32 = 32;
const MAX_SEARCH_NODES: u64 = 250_000;
const PPM: u64 = 1_000_000;

/// A single ordered planning interval. Generation is fixed and unavoidable in V1;
/// unused generation is curtailed but its declared production cost/emissions remain.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyHubInterval {
    pub id: String,
    pub demand_quanta: u32,
    pub charger_capacity_quanta: u32,
    pub renewable_generation_quanta: u32,
    pub grid_import_capacity_quanta: u32,
    pub renewable_cost_per_quantum: f64,
    pub renewable_lifecycle_co2e_kg_per_quantum: f64,
    pub renewable_evidence_ref: String,
    pub renewable_evidence_class: EvidenceClass,
    pub grid_cost_per_quantum: f64,
    pub grid_lifecycle_co2e_kg_per_quantum: f64,
    pub grid_evidence_ref: String,
    pub grid_evidence_class: EvidenceClass,
}

/// Storage is initially empty in v1 to avoid carrying unaccounted pre-horizon energy.
/// Efficiencies are parts per million in (0, 1_000_000]. Integer rounding is
/// conservative: charge output rounds down; SOC needed for discharge rounds up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyStorageSpec {
    pub capacity_quanta: u32,
    /// Physically carried state of charge at the start of this planning horizon.
    pub initial_soc_quanta: u32,
    /// Identity of the external/prefix ledger that already attributes this inventory's
    /// upstream cost and emissions. Use an explicit no-opening-inventory sentinel at SOC 0.
    pub initial_soc_ledger_ref: String,
    pub max_charge_input_quanta_per_interval: u32,
    pub max_discharge_output_quanta_per_interval: u32,
    pub charge_efficiency_ppm: u32,
    pub discharge_efficiency_ppm: u32,
    pub minimum_terminal_soc_quanta: u32,
}

/// A chronological synthetic airport energy-hub scenario. The input digest is
/// SHA-256 over canonical content and is verified by every solver and plan verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeIndexedEnergyHubScenario {
    pub scenario_id: String,
    pub input_digest: String,
    pub currency_code: String,
    pub quantum_kwh: f64,
    pub service: TransportService,
    pub storage: EnergyStorageSpec,
    pub intervals: Vec<EnergyHubInterval>,
}

/// One interval's energy balance and exact storage-state witness.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EnergyHubStepAction {
    pub interval_id: String,
    pub renewable_direct_quanta: u32,
    pub grid_import_quanta: u32,
    pub battery_discharge_output_quanta: u32,
    pub renewable_charge_input_quanta: u32,
    pub renewable_curtailed_quanta: u32,
    pub soc_before_quanta: u32,
    pub soc_after_quanta: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyHubPlan {
    pub scenario_id: String,
    pub input_digest: String,
    pub actions: Vec<EnergyHubStepAction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyHubMetrics {
    /// Accounting ledger. The source-energy aggregate includes secondary storage-output
    /// flow records; use the separate generation/import counters for primary inputs.
    pub source_ledger: AccountingSummary,
    pub renewable_generation_quanta: u32,
    pub renewable_direct_quanta: u32,
    pub renewable_charge_input_quanta: u32,
    pub renewable_curtailed_quanta: u32,
    pub grid_import_quanta: u32,
    pub battery_discharge_output_quanta: u32,
    pub battery_charge_loss_quanta: u32,
    pub battery_discharge_loss_quanta: u32,
    pub final_soc_quanta: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedEnergyHubPlan {
    pub plan: EnergyHubPlan,
    pub metrics: EnergyHubMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactEnergyHubResult {
    pub frontier: Vec<EvaluatedEnergyHubPlan>,
    pub nodes_visited: u64,
    pub feasible_plans_seen: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyHubError {
    MissingIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    TooManyIntervals,
    QuantumLimit,
    InvalidQuantum,
    InvalidCurrencyCode,
    InvalidService,
    InvalidStorage,
    InvalidEfficiency,
    InvalidCapacity(&'static str),
    InvalidNumericValue(&'static str),
    NegativeValue(&'static str),
    InvalidSearchBudget,
    NoFeasiblePlan,
    SearchBudgetExceeded { nodes_visited: u64 },
    ScenarioDigestMismatch,
    PlanIdentityMismatch,
    ActionCountMismatch,
    IntervalOrderMismatch,
    DemandBalanceViolation,
    ResourceBalanceViolation,
    SimultaneousChargeAndDischarge,
    StorageStateMismatch,
    TerminalReserveNotMet,
    ArithmeticOverflow,
    Accounting(AccountingError),
}

impl fmt::Display for EnergyHubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingIdentifier(field) => write!(f, "missing identifier: {field}"),
            Self::DuplicateIdentifier(field) => write!(f, "duplicate identifier: {field}"),
            Self::TooManyIntervals => write!(f, "energy hub exceeds the interval limit"),
            Self::QuantumLimit => write!(f, "scenario exceeds the discrete energy quantum limit"),
            Self::InvalidQuantum => write!(f, "energy quantum must be finite and positive"),
            Self::InvalidCurrencyCode => write!(f, "currency code must be three uppercase ASCII letters"),
            Self::InvalidService => write!(f, "transport service quantities must be finite and positive"),
            Self::InvalidStorage => write!(f, "storage capacities or terminal reserve are inconsistent"),
            Self::InvalidEfficiency => write!(f, "storage efficiency must be between 1 and 1,000,000 ppm"),
            Self::InvalidCapacity(field) => write!(f, "invalid capacity: {field}"),
            Self::InvalidNumericValue(field) => write!(f, "non-finite numeric value: {field}"),
            Self::NegativeValue(field) => write!(f, "negative value: {field}"),
            Self::InvalidSearchBudget => write!(f, "search budget must be between 1 and the hard upper bound"),
            Self::NoFeasiblePlan => write!(f, "no energy plan meets demand, capacity and terminal reserve"),
            Self::SearchBudgetExceeded { nodes_visited } => {
                write!(f, "exact energy-hub search budget exceeded after {nodes_visited} nodes")
            }
            Self::ScenarioDigestMismatch => write!(f, "energy-hub scenario digest does not match canonical content"),
            Self::PlanIdentityMismatch => write!(f, "energy-hub plan scenario ID or digest does not match"),
            Self::ActionCountMismatch => write!(f, "plan action count does not match scenario intervals"),
            Self::IntervalOrderMismatch => write!(f, "plan interval identity/order does not match scenario"),
            Self::DemandBalanceViolation => write!(f, "energy supplied in an interval does not equal demand"),
            Self::ResourceBalanceViolation => write!(f, "source allocation, capacity, curtailment or state balance is invalid"),
            Self::SimultaneousChargeAndDischarge => write!(f, "simultaneous storage charge and discharge is disallowed"),
            Self::StorageStateMismatch => write!(f, "declared storage state does not match independently calculated state"),
            Self::TerminalReserveNotMet => write!(f, "minimum terminal storage reserve is not met"),
            Self::ArithmeticOverflow => write!(f, "energy-hub aggregate arithmetic overflowed"),
            Self::Accounting(error) => write!(f, "source ledger rejected the plan: {error}"),
        }
    }
}

impl std::error::Error for EnergyHubError {}

/// Create a canonical, domain-separated SHA-256 scenario digest. Interval order is
/// meaningful and retained; the digest field itself is omitted to avoid self-reference.
pub fn canonical_energy_hub_digest(
    scenario: &TimeIndexedEnergyHubScenario,
) -> Result<String, EnergyHubError> {
    let mut validation_copy = scenario.clone();
    validation_copy.input_digest = "canonicalization-validation-placeholder".to_owned();
    validate_scenario(&validation_copy)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.time-indexed-energy-hub.v1\0");
    put_string(&mut bytes, &scenario.scenario_id);
    put_string(&mut bytes, &scenario.currency_code);
    put_f64(&mut bytes, scenario.quantum_kwh);
    put_string(&mut bytes, &scenario.service.period_id);
    put_string(&mut bytes, &scenario.service.lifecycle_boundary_id);
    put_f64(&mut bytes, scenario.service.completed_passenger_km);
    put_f64(&mut bytes, scenario.service.available_seat_km);

    put_u32(&mut bytes, scenario.storage.capacity_quanta);
    put_u32(&mut bytes, scenario.storage.initial_soc_quanta);
    put_string(&mut bytes, &scenario.storage.initial_soc_ledger_ref);
    put_u32(&mut bytes, scenario.storage.max_charge_input_quanta_per_interval);
    put_u32(&mut bytes, scenario.storage.max_discharge_output_quanta_per_interval);
    put_u32(&mut bytes, scenario.storage.charge_efficiency_ppm);
    put_u32(&mut bytes, scenario.storage.discharge_efficiency_ppm);
    put_u32(&mut bytes, scenario.storage.minimum_terminal_soc_quanta);

    put_u32(&mut bytes, scenario.intervals.len() as u32);
    for interval in &scenario.intervals {
        put_string(&mut bytes, &interval.id);
        put_u32(&mut bytes, interval.demand_quanta);
        put_u32(&mut bytes, interval.charger_capacity_quanta);
        put_u32(&mut bytes, interval.renewable_generation_quanta);
        put_u32(&mut bytes, interval.grid_import_capacity_quanta);
        put_f64(&mut bytes, interval.renewable_cost_per_quantum);
        put_f64(&mut bytes, interval.renewable_lifecycle_co2e_kg_per_quantum);
        put_string(&mut bytes, &interval.renewable_evidence_ref);
        bytes.push(encode_evidence_class(interval.renewable_evidence_class));
        put_f64(&mut bytes, interval.grid_cost_per_quantum);
        put_f64(&mut bytes, interval.grid_lifecycle_co2e_kg_per_quantum);
        put_string(&mut bytes, &interval.grid_evidence_ref);
        bytes.push(encode_evidence_class(interval.grid_evidence_class));
    }

    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(7 + digest.len() * 2);
    hex.push_str("sha256:");
    for byte in digest.iter() {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(hex)
}

pub fn seal_energy_hub_scenario(
    scenario: &mut TimeIndexedEnergyHubScenario,
) -> Result<(), EnergyHubError> {
    scenario.input_digest = canonical_energy_hub_digest(scenario)?;
    Ok(())
}

pub fn verify_energy_hub_digest(
    scenario: &TimeIndexedEnergyHubScenario,
) -> Result<(), EnergyHubError> {
    if scenario.input_digest != canonical_energy_hub_digest(scenario)? {
        return Err(EnergyHubError::ScenarioDigestMismatch);
    }
    Ok(())
}

/// Independently validate an action sequence, compute storage transitions and
/// aggregate cost/emissions by rerunning the source-ledger accounting routine.
pub fn verify_energy_hub_plan(
    scenario: &TimeIndexedEnergyHubScenario,
    plan: &EnergyHubPlan,
) -> Result<EvaluatedEnergyHubPlan, EnergyHubError> {
    validate_scenario(scenario)?;
    verify_energy_hub_digest(scenario)?;
    if plan.scenario_id != scenario.scenario_id || plan.input_digest != scenario.input_digest {
        return Err(EnergyHubError::PlanIdentityMismatch);
    }
    if plan.actions.len() != scenario.intervals.len() {
        return Err(EnergyHubError::ActionCountMismatch);
    }

    let mut soc = scenario.storage.initial_soc_quanta;
    let mut renewable_generation_quanta = 0_u32;
    let mut renewable_direct_quanta = 0_u32;
    let mut renewable_charge_input_quanta = 0_u32;
    let mut renewable_curtailed_quanta = 0_u32;
    let mut grid_import_quanta = 0_u32;
    let mut battery_discharge_output_quanta = 0_u32;
    let mut battery_charge_loss_quanta = 0_u32;
    let mut battery_discharge_loss_quanta = 0_u32;
    let mut records = Vec::new();

    for (interval, action) in scenario.intervals.iter().zip(plan.actions.iter()) {
        if action.interval_id != interval.id {
            return Err(EnergyHubError::IntervalOrderMismatch);
        }
        if interval.demand_quanta > interval.charger_capacity_quanta {
            return Err(EnergyHubError::NoFeasiblePlan);
        }
        if action.soc_before_quanta != soc {
            return Err(EnergyHubError::StorageStateMismatch);
        }
        if action.renewable_direct_quanta > interval.renewable_generation_quanta
            || action.renewable_direct_quanta > interval.demand_quanta
            || action.grid_import_quanta > interval.grid_import_capacity_quanta
            || action.battery_discharge_output_quanta
                > scenario.storage.max_discharge_output_quanta_per_interval
            || action.renewable_charge_input_quanta
                > scenario.storage.max_charge_input_quanta_per_interval
        {
            return Err(EnergyHubError::ResourceBalanceViolation);
        }

        let supplied = action
            .renewable_direct_quanta
            .checked_add(action.grid_import_quanta)
            .and_then(|sum| sum.checked_add(action.battery_discharge_output_quanta))
            .ok_or(EnergyHubError::ArithmeticOverflow)?;
        if supplied != interval.demand_quanta {
            return Err(EnergyHubError::DemandBalanceViolation);
        }
        if action.battery_discharge_output_quanta > 0
            && action.renewable_charge_input_quanta > 0
        {
            return Err(EnergyHubError::SimultaneousChargeAndDischarge);
        }

        let renewable_used = action
            .renewable_direct_quanta
            .checked_add(action.renewable_charge_input_quanta)
            .ok_or(EnergyHubError::ArithmeticOverflow)?;
        if renewable_used > interval.renewable_generation_quanta {
            return Err(EnergyHubError::ResourceBalanceViolation);
        }
        let expected_curtailed = interval.renewable_generation_quanta - renewable_used;
        if action.renewable_curtailed_quanta != expected_curtailed {
            return Err(EnergyHubError::ResourceBalanceViolation);
        }

        let mut charge_loss = 0_u32;
        let mut discharge_loss = 0_u32;
        let expected_soc = if action.battery_discharge_output_quanta > 0 {
            let consumed = soc_required_for_discharge(
                action.battery_discharge_output_quanta,
                scenario.storage.discharge_efficiency_ppm,
            )?;
            if consumed > soc {
                return Err(EnergyHubError::ResourceBalanceViolation);
            }
            discharge_loss = consumed - action.battery_discharge_output_quanta;
            soc - consumed
        } else if action.renewable_charge_input_quanta > 0 {
            let stored = storage_from_charge(
                action.renewable_charge_input_quanta,
                scenario.storage.charge_efficiency_ppm,
            )?;
            if stored == 0 || soc.checked_add(stored).is_none() {
                return Err(EnergyHubError::ResourceBalanceViolation);
            }
            let next = soc + stored;
            if next > scenario.storage.capacity_quanta {
                return Err(EnergyHubError::ResourceBalanceViolation);
            }
            charge_loss = action.renewable_charge_input_quanta - stored;
            next
        } else {
            soc
        };
        if action.soc_after_quanta != expected_soc {
            return Err(EnergyHubError::StorageStateMismatch);
        }
        soc = expected_soc;

        add_u32(&mut renewable_generation_quanta, interval.renewable_generation_quanta)?;
        add_u32(&mut renewable_direct_quanta, action.renewable_direct_quanta)?;
        add_u32(&mut renewable_charge_input_quanta, action.renewable_charge_input_quanta)?;
        add_u32(&mut renewable_curtailed_quanta, action.renewable_curtailed_quanta)?;
        add_u32(&mut grid_import_quanta, action.grid_import_quanta)?;
        add_u32(
            &mut battery_discharge_output_quanta,
            action.battery_discharge_output_quanta,
        )?;
        add_u32(&mut battery_charge_loss_quanta, charge_loss)?;
        add_u32(&mut battery_discharge_loss_quanta, discharge_loss)?;

        if interval.renewable_generation_quanta > 0 {
            let energy = f64::from(interval.renewable_generation_quanta) * scenario.quantum_kwh;
            records.push(EnergyRecord {
                id: format!("renewable-generation:{}", interval.id),
                source_id: "airport-renewable-generation".to_owned(),
                boundary_id: format!("energy-hub-renewable-bus:{}", interval.id),
                lifecycle_boundary_id: scenario.service.lifecycle_boundary_id.clone(),
                period_id: scenario.service.period_id.clone(),
                evidence_ref: interval.renewable_evidence_ref.clone(),
                evidence_class: interval.renewable_evidence_class,
                carrier: EnergyCarrier::Electricity,
                source_energy_kwh: energy,
                // Delivered energy is energy served to load; charging input remains
                // represented in generation/source cost and emissions, but is not load delivery.
                delivered_energy_kwh: f64::from(action.renewable_direct_quanta) * scenario.quantum_kwh,
                upstream_co2e_kg: f64::from(interval.renewable_generation_quanta)
                    * interval.renewable_lifecycle_co2e_kg_per_quantum,
                direct_operational_co2e_kg: 0.0,
                cost_amount: f64::from(interval.renewable_generation_quanta)
                    * interval.renewable_cost_per_quantum,
                currency_code: scenario.currency_code.clone(),
                non_co2: NonCo2Evidence::NotAssessed,
            });
        }
        if action.battery_discharge_output_quanta > 0 {
            // The primary energy cost and lifecycle emissions are attributed when the
            // storage is charged. This secondary-bus record makes the discharged energy
            // visible as load delivery without re-booking those upstream impacts. Opening
            // SOC must therefore have its prior source ledger reconciled outside this horizon.
            let energy =
                f64::from(action.battery_discharge_output_quanta) * scenario.quantum_kwh;
            records.push(EnergyRecord {
                id: format!("storage-output:{}", interval.id),
                source_id: "pre-accounted-storage-inventory".to_owned(),
                boundary_id: format!("energy-hub-storage-output:{}", interval.id),
                lifecycle_boundary_id: scenario.service.lifecycle_boundary_id.clone(),
                period_id: scenario.service.period_id.clone(),
                evidence_ref: "energy-hub:storage-input-impacts-attributed-at-charge".to_owned(),
                evidence_class: EvidenceClass::UncalibratedModel,
                carrier: EnergyCarrier::Electricity,
                source_energy_kwh: energy,
                delivered_energy_kwh: energy,
                upstream_co2e_kg: 0.0,
                direct_operational_co2e_kg: 0.0,
                cost_amount: 0.0,
                currency_code: scenario.currency_code.clone(),
                non_co2: NonCo2Evidence::NotAssessed,
            });
        }

        if action.grid_import_quanta > 0 {
            let energy = f64::from(action.grid_import_quanta) * scenario.quantum_kwh;
            records.push(EnergyRecord {
                id: format!("grid-import:{}", interval.id),
                source_id: "airport-grid-import".to_owned(),
                boundary_id: format!("energy-hub-grid-bus:{}", interval.id),
                lifecycle_boundary_id: scenario.service.lifecycle_boundary_id.clone(),
                period_id: scenario.service.period_id.clone(),
                evidence_ref: interval.grid_evidence_ref.clone(),
                evidence_class: interval.grid_evidence_class,
                carrier: EnergyCarrier::Electricity,
                source_energy_kwh: energy,
                delivered_energy_kwh: energy,
                upstream_co2e_kg: f64::from(action.grid_import_quanta)
                    * interval.grid_lifecycle_co2e_kg_per_quantum,
                direct_operational_co2e_kg: 0.0,
                cost_amount: f64::from(action.grid_import_quanta) * interval.grid_cost_per_quantum,
                currency_code: scenario.currency_code.clone(),
                non_co2: NonCo2Evidence::NotAssessed,
            });
        }
    }

    if soc < scenario.storage.minimum_terminal_soc_quanta {
        return Err(EnergyHubError::TerminalReserveNotMet);
    }
    let source_ledger = account_energy_ledger(&records, &scenario.service)
        .map_err(EnergyHubError::Accounting)?;

    Ok(EvaluatedEnergyHubPlan {
        plan: EnergyHubPlan {
            scenario_id: plan.scenario_id.clone(),
            input_digest: plan.input_digest.clone(),
            actions: plan.actions.clone(),
        },
        metrics: EnergyHubMetrics {
            source_ledger,
            renewable_generation_quanta,
            renewable_direct_quanta,
            renewable_charge_input_quanta,
            renewable_curtailed_quanta,
            grid_import_quanta,
            battery_discharge_output_quanta,
            battery_charge_loss_quanta,
            battery_discharge_loss_quanta,
            final_soc_quanta: soc,
        },
    })
}

/// Exact bounded search over discrete per-interval allocations. This returns a
/// three-axis Pareto set: minimize cost, minimize lifecycle CO2e, maximize final SOC.
pub fn solve_exact_energy_hub_pareto(
    scenario: &TimeIndexedEnergyHubScenario,
    max_nodes: u64,
) -> Result<ExactEnergyHubResult, EnergyHubError> {
    validate_scenario(scenario)?;
    verify_energy_hub_digest(scenario)?;
    if max_nodes == 0 || max_nodes > MAX_SEARCH_NODES {
        return Err(EnergyHubError::InvalidSearchBudget);
    }

    let mut search = HubSearch {
        scenario,
        max_nodes,
        nodes_visited: 0,
        feasible_plans_seen: 0,
        exhausted: false,
        frontier: Vec::new(),
        actions: Vec::with_capacity(scenario.intervals.len()),
    };
    search.visit(0, scenario.storage.initial_soc_quanta)?;
    if search.exhausted {
        return Err(EnergyHubError::SearchBudgetExceeded {
            nodes_visited: search.nodes_visited,
        });
    }
    if search.frontier.is_empty() {
        return Err(EnergyHubError::NoFeasiblePlan);
    }
    search.frontier.sort_by(compare_hub_plans);
    Ok(ExactEnergyHubResult {
        frontier: search.frontier,
        nodes_visited: search.nodes_visited,
        feasible_plans_seen: search.feasible_plans_seen,
    })
}

/// Myopic benchmark baseline: serve each interval from current renewable generation,
/// then grid import, then battery; charge from unused renewable surplus only.
/// It deliberately ignores future prices, emissions and terminal-state value.
pub fn solve_renewable_first_greedy(
    scenario: &TimeIndexedEnergyHubScenario,
) -> Result<EvaluatedEnergyHubPlan, EnergyHubError> {
    validate_scenario(scenario)?;
    verify_energy_hub_digest(scenario)?;
    let mut soc = scenario.storage.initial_soc_quanta;
    let mut actions = Vec::with_capacity(scenario.intervals.len());

    for interval in &scenario.intervals {
        if interval.demand_quanta > interval.charger_capacity_quanta {
            return Err(EnergyHubError::NoFeasiblePlan);
        }
        let renewable_direct = interval.demand_quanta.min(interval.renewable_generation_quanta);
        let remaining = interval.demand_quanta - renewable_direct;
        let grid_import = remaining.min(interval.grid_import_capacity_quanta);
        let battery_output = remaining - grid_import;

        if battery_output > scenario.storage.max_discharge_output_quanta_per_interval {
            return Err(EnergyHubError::NoFeasiblePlan);
        }
        let charge_input = if battery_output == 0 {
            let available = interval
                .renewable_generation_quanta
                .saturating_sub(renewable_direct)
                .min(scenario.storage.max_charge_input_quanta_per_interval);
            let mut selected = 0;
            for candidate in (1..=available).rev() {
                let stored = storage_from_charge(candidate, scenario.storage.charge_efficiency_ppm)?;
                if stored > 0
                    && soc
                        .checked_add(stored)
                        .is_some_and(|next| next <= scenario.storage.capacity_quanta)
                {
                    selected = candidate;
                    break;
                }
            }
            selected
        } else {
            0
        };

        let charge_gain = storage_from_charge(charge_input, scenario.storage.charge_efficiency_ppm)?;
        let expected_soc = if battery_output > 0 {
            let consumed =
                soc_required_for_discharge(battery_output, scenario.storage.discharge_efficiency_ppm)?;
            if consumed > soc {
                return Err(EnergyHubError::NoFeasiblePlan);
            }
            soc - consumed
        } else {
            soc + charge_gain
        };
        let curtailed =
            interval.renewable_generation_quanta - renewable_direct - charge_input;
        actions.push(EnergyHubStepAction {
            interval_id: interval.id.clone(),
            renewable_direct_quanta: renewable_direct,
            grid_import_quanta: grid_import,
            battery_discharge_output_quanta: battery_output,
            renewable_charge_input_quanta: charge_input,
            renewable_curtailed_quanta: curtailed,
            soc_before_quanta: soc,
            soc_after_quanta: expected_soc,
        });
        soc = expected_soc;
    }

    let plan = EnergyHubPlan {
        scenario_id: scenario.scenario_id.clone(),
        input_digest: scenario.input_digest.clone(),
        actions,
    };
    verify_energy_hub_plan(scenario, &plan)
}

struct HubSearch<'a> {
    scenario: &'a TimeIndexedEnergyHubScenario,
    max_nodes: u64,
    nodes_visited: u64,
    feasible_plans_seen: u64,
    exhausted: bool,
    frontier: Vec<EvaluatedEnergyHubPlan>,
    actions: Vec<EnergyHubStepAction>,
}

impl HubSearch<'_> {
    fn visit(&mut self, index: usize, soc: u32) -> Result<(), EnergyHubError> {
        if self.nodes_visited >= self.max_nodes {
            self.exhausted = true;
            return Ok(());
        }
        self.nodes_visited += 1;

        if index == self.scenario.intervals.len() {
            if soc < self.scenario.storage.minimum_terminal_soc_quanta {
                return Ok(());
            }
            let plan = EnergyHubPlan {
                scenario_id: self.scenario.scenario_id.clone(),
                input_digest: self.scenario.input_digest.clone(),
                actions: self.actions.clone(),
            };
            let evaluated = verify_energy_hub_plan(self.scenario, &plan)?;
            self.feasible_plans_seen = self
                .feasible_plans_seen
                .checked_add(1)
                .ok_or(EnergyHubError::ArithmeticOverflow)?;
            insert_hub_pareto(&mut self.frontier, evaluated);
            return Ok(());
        }

        let interval = &self.scenario.intervals[index];
        let candidates = enumerate_step_actions(self.scenario, interval, soc)?;
        for action in candidates {
            let next_soc = action.soc_after_quanta;
            self.actions.push(action);
            self.visit(index + 1, next_soc)?;
            self.actions.pop();
            if self.exhausted {
                break;
            }
        }
        Ok(())
    }
}

fn enumerate_step_actions(
    scenario: &TimeIndexedEnergyHubScenario,
    interval: &EnergyHubInterval,
    soc: u32,
) -> Result<Vec<EnergyHubStepAction>, EnergyHubError> {
    if interval.demand_quanta > interval.charger_capacity_quanta {
        return Ok(Vec::new());
    }
    let mut candidates = Vec::new();
    let max_direct = interval.demand_quanta.min(interval.renewable_generation_quanta);
    for renewable_direct in 0..=max_direct {
        let residual = interval.demand_quanta - renewable_direct;
        let max_grid = residual.min(interval.grid_import_capacity_quanta);
        for grid_import in 0..=max_grid {
            let battery_output = residual - grid_import;
            if battery_output > scenario.storage.max_discharge_output_quanta_per_interval {
                continue;
            }
            if battery_output > 0 {
                let consumed = soc_required_for_discharge(
                    battery_output,
                    scenario.storage.discharge_efficiency_ppm,
                )?;
                if consumed <= soc {
                    candidates.push(EnergyHubStepAction {
                        interval_id: interval.id.clone(),
                        renewable_direct_quanta: renewable_direct,
                        grid_import_quanta: grid_import,
                        battery_discharge_output_quanta: battery_output,
                        renewable_charge_input_quanta: 0,
                        renewable_curtailed_quanta: interval.renewable_generation_quanta
                            - renewable_direct,
                        soc_before_quanta: soc,
                        soc_after_quanta: soc - consumed,
                    });
                }
                continue;
            }

            let surplus = interval.renewable_generation_quanta - renewable_direct;
            let max_charge = surplus.min(scenario.storage.max_charge_input_quanta_per_interval);
            for charge_input in 0..=max_charge {
                let stored = storage_from_charge(
                    charge_input,
                    scenario.storage.charge_efficiency_ppm,
                )?;
                if charge_input > 0 && stored == 0 {
                    continue;
                }
                let next_soc = soc
                    .checked_add(stored)
                    .ok_or(EnergyHubError::ArithmeticOverflow)?;
                if next_soc > scenario.storage.capacity_quanta {
                    continue;
                }
                candidates.push(EnergyHubStepAction {
                    interval_id: interval.id.clone(),
                    renewable_direct_quanta: renewable_direct,
                    grid_import_quanta: grid_import,
                    battery_discharge_output_quanta: 0,
                    renewable_charge_input_quanta: charge_input,
                    renewable_curtailed_quanta: surplus - charge_input,
                    soc_before_quanta: soc,
                    soc_after_quanta: next_soc,
                });
            }
        }
    }
    candidates.sort();
    Ok(candidates)
}

fn validate_scenario(
    scenario: &TimeIndexedEnergyHubScenario,
) -> Result<(), EnergyHubError> {
    require_identifier(&scenario.scenario_id, "scenario ID")?;
    require_identifier(&scenario.input_digest, "input digest")?;
    require_identifier(&scenario.currency_code, "currency code")?;
    require_identifier(&scenario.service.period_id, "service period ID")?;
    require_identifier(
        &scenario.service.lifecycle_boundary_id,
        "lifecycle boundary ID",
    )?;
    if !scenario.quantum_kwh.is_finite() || scenario.quantum_kwh <= 0.0 {
        return Err(EnergyHubError::InvalidQuantum);
    }
    if scenario.currency_code.len() != 3
        || !scenario.currency_code.bytes().all(|byte| byte.is_ascii_uppercase())
    {
        return Err(EnergyHubError::InvalidCurrencyCode);
    }
    if !scenario.service.completed_passenger_km.is_finite()
        || scenario.service.completed_passenger_km <= 0.0
        || !scenario.service.available_seat_km.is_finite()
        || scenario.service.available_seat_km <= 0.0
    {
        return Err(EnergyHubError::InvalidService);
    }
    if scenario.intervals.is_empty() || scenario.intervals.len() > MAX_INTERVALS {
        return Err(EnergyHubError::TooManyIntervals);
    }
    let storage = &scenario.storage;
    require_identifier(&storage.initial_soc_ledger_ref, "opening SOC inventory ledger reference")?;
    if storage.capacity_quanta > MAX_QUANTA_PER_INTERVAL
        || storage.initial_soc_quanta > storage.capacity_quanta
        || storage.max_charge_input_quanta_per_interval > MAX_QUANTA_PER_INTERVAL
        || storage.max_discharge_output_quanta_per_interval > MAX_QUANTA_PER_INTERVAL
        || storage.minimum_terminal_soc_quanta > storage.capacity_quanta
    {
        return Err(EnergyHubError::InvalidStorage);
    }
    if storage.charge_efficiency_ppm == 0
        || u64::from(storage.charge_efficiency_ppm) > PPM
        || storage.discharge_efficiency_ppm == 0
        || u64::from(storage.discharge_efficiency_ppm) > PPM
    {
        return Err(EnergyHubError::InvalidEfficiency);
    }

    let mut ids = BTreeSet::new();
    let mut total_demand = 0_u32;
    for interval in &scenario.intervals {
        require_identifier(&interval.id, "interval ID")?;
        require_identifier(&interval.renewable_evidence_ref, "renewable evidence ref")?;
        require_identifier(&interval.grid_evidence_ref, "grid evidence ref")?;
        if !ids.insert(interval.id.as_str()) {
            return Err(EnergyHubError::DuplicateIdentifier("interval ID"));
        }
        for (value, field) in [
            (interval.demand_quanta, "demand"),
            (interval.charger_capacity_quanta, "charger capacity"),
            (interval.renewable_generation_quanta, "renewable generation"),
            (interval.grid_import_capacity_quanta, "grid import capacity"),
        ] {
            if value > MAX_QUANTA_PER_INTERVAL {
                return Err(EnergyHubError::InvalidCapacity(field));
            }
        }
        total_demand = total_demand
            .checked_add(interval.demand_quanta)
            .ok_or(EnergyHubError::QuantumLimit)?;
        for (value, field) in [
            (interval.renewable_cost_per_quantum, "renewable cost factor"),
            (
                interval.renewable_lifecycle_co2e_kg_per_quantum,
                "renewable lifecycle factor",
            ),
            (interval.grid_cost_per_quantum, "grid cost factor"),
            (interval.grid_lifecycle_co2e_kg_per_quantum, "grid lifecycle factor"),
        ] {
            if !value.is_finite() {
                return Err(EnergyHubError::InvalidNumericValue(field));
            }
            if value < 0.0 {
                return Err(EnergyHubError::NegativeValue(field));
            }
        }
    }
    if total_demand == 0 || total_demand > MAX_TOTAL_DEMAND_QUANTA {
        return Err(EnergyHubError::QuantumLimit);
    }
    Ok(())
}

fn insert_hub_pareto(
    frontier: &mut Vec<EvaluatedEnergyHubPlan>,
    candidate: EvaluatedEnergyHubPlan,
) {
    if frontier
        .iter()
        .any(|current| hub_dominates(&current.metrics, &candidate.metrics))
    {
        return;
    }
    if frontier.iter().any(|current| {
        same_hub_objectives(&current.metrics, &candidate.metrics)
            && current.plan.actions <= candidate.plan.actions
    }) {
        return;
    }
    frontier.retain(|current| {
        !hub_dominates(&candidate.metrics, &current.metrics)
            && !(same_hub_objectives(&current.metrics, &candidate.metrics)
                && candidate.plan.actions < current.plan.actions)
    });
    frontier.push(candidate);
}

fn hub_dominates(left: &EnergyHubMetrics, right: &EnergyHubMetrics) -> bool {
    left.source_ledger.total_cost <= right.source_ledger.total_cost
        && left.source_ledger.accounted_lifecycle_co2e_kg
            <= right.source_ledger.accounted_lifecycle_co2e_kg
        && left.final_soc_quanta >= right.final_soc_quanta
        && (left.source_ledger.total_cost < right.source_ledger.total_cost
            || left.source_ledger.accounted_lifecycle_co2e_kg
                < right.source_ledger.accounted_lifecycle_co2e_kg
            || left.final_soc_quanta > right.final_soc_quanta)
}

fn same_hub_objectives(left: &EnergyHubMetrics, right: &EnergyHubMetrics) -> bool {
    left.source_ledger.total_cost == right.source_ledger.total_cost
        && left.source_ledger.accounted_lifecycle_co2e_kg
            == right.source_ledger.accounted_lifecycle_co2e_kg
        && left.final_soc_quanta == right.final_soc_quanta
}

fn compare_hub_plans(
    left: &EvaluatedEnergyHubPlan,
    right: &EvaluatedEnergyHubPlan,
) -> std::cmp::Ordering {
    left.metrics
        .source_ledger
        .total_cost
        .total_cmp(&right.metrics.source_ledger.total_cost)
        .then_with(|| {
            left.metrics
                .source_ledger
                .accounted_lifecycle_co2e_kg
                .total_cmp(&right.metrics.source_ledger.accounted_lifecycle_co2e_kg)
        })
        .then_with(|| right.metrics.final_soc_quanta.cmp(&left.metrics.final_soc_quanta))
        .then_with(|| left.plan.actions.cmp(&right.plan.actions))
}

fn storage_from_charge(input: u32, efficiency_ppm: u32) -> Result<u32, EnergyHubError> {
    let value = u64::from(input)
        .checked_mul(u64::from(efficiency_ppm))
        .ok_or(EnergyHubError::ArithmeticOverflow)?
        / PPM;
    u32::try_from(value).map_err(|_| EnergyHubError::ArithmeticOverflow)
}

fn soc_required_for_discharge(output: u32, efficiency_ppm: u32) -> Result<u32, EnergyHubError> {
    let numerator = u64::from(output)
        .checked_mul(PPM)
        .and_then(|value| value.checked_add(u64::from(efficiency_ppm) - 1))
        .ok_or(EnergyHubError::ArithmeticOverflow)?;
    let value = numerator / u64::from(efficiency_ppm);
    u32::try_from(value).map_err(|_| EnergyHubError::ArithmeticOverflow)
}

fn add_u32(total: &mut u32, value: u32) -> Result<(), EnergyHubError> {
    *total = total.checked_add(value).ok_or(EnergyHubError::ArithmeticOverflow)?;
    Ok(())
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), EnergyHubError> {
    if value.trim().is_empty() {
        return Err(EnergyHubError::MissingIdentifier(field));
    }
    Ok(())
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn put_f64(bytes: &mut Vec<u8>, value: f64) {
    put_u64(bytes, (if value == 0.0 { 0.0 } else { value }).to_bits());
}

fn encode_evidence_class(class: EvidenceClass) -> u8 {
    match class {
        EvidenceClass::Measured => 0,
        EvidenceClass::SupplierReported => 1,
        EvidenceClass::LiteratureDerived => 2,
        EvidenceClass::CalibratedModel => 3,
        EvidenceClass::UncalibratedModel => 4,
        EvidenceClass::Synthetic => 5,
    }
}
