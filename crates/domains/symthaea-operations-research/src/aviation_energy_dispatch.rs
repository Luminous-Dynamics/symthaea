// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded, discrete energy-sourcing oracle for CLEAN-AVIATION-001.
//!
//! This first P1 model covers one planning interval and caller-declared delivered
//! energy quanta. It enumerates source allocations by carrier and returns a Pareto
//! frontier over declared cost and accounted lifecycle CO2e. It does not model
//! aircraft feasibility, flight timing, charging queues, storage dynamics, network
//! flows, continuous dispatch, physical fuel production, or a real airport.
//!
//! The scenario identity uses a versioned, domain-separated SHA-256 digest of a
//! canonical encoding. Input vector order is normalized by stable IDs, and plan
//! verification recomputes the digest. This is integrity/content identity, not a
//! signature, evidence of physical validity, or authority to operate an airport.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sha2::{Digest, Sha256};

use crate::aviation_energy_accounting::{
    account_energy_ledger, AccountingError, AccountingSummary, EnergyCarrier, EnergyRecord,
    EvidenceClass, NonCo2Evidence, TransportService,
};

const MAX_DEMANDS: usize = 8;
const MAX_OFFERS: usize = 8;
const MAX_TOTAL_DEMAND_QUANTA: u32 = 32;
const MAX_QUANTA_PER_OFFER: u32 = 32;
const MAX_SEARCH_NODES: u64 = 1_000_000;

/// Required delivered energy in integer units of scenario.quantum_kwh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyDemand {
    pub id: String,
    pub carrier: EnergyCarrier,
    pub required_quanta: u32,
}

/// A single source offer. Capacity and all unit cost/emission factors are expressed
/// per delivered-energy quantum at the declared scenario boundary. Conversion,
/// transmission, storage and production losses must already be reflected in the
/// declared offer capacity, unit cost and upstream lifecycle factor; this small
/// model does not simulate those physical processes.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergySourceOffer {
    pub id: String,
    pub source_id: String,
    pub carrier: EnergyCarrier,
    pub capacity_quanta: u32,
    pub cost_per_quantum: f64,
    pub upstream_co2e_kg_per_quantum: f64,
    pub direct_operational_co2e_kg_per_quantum: f64,
    pub evidence_ref: String,
    pub evidence_class: EvidenceClass,
}

/// A single-period, discrete synthetic airport-energy procurement scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct CleanAirlineEnergyScenario {
    pub scenario_id: String,
    /// Canonical SHA-256 digest, set by `seal_scenario` and checked by every solver/verifier.
    pub input_digest: String,
    pub currency_code: String,
    pub quantum_kwh: f64,
    pub service: TransportService,
    pub demands: Vec<EnergyDemand>,
    pub offers: Vec<EnergySourceOffer>,
}

/// Positive source allocations only. The canonical form is sorted by offer_id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EnergyAllocation {
    pub offer_id: String,
    pub quanta: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyPlan {
    pub scenario_id: String,
    pub input_digest: String,
    pub allocations: Vec<EnergyAllocation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedEnergyPlan {
    pub plan: EnergyPlan,
    pub ledger: AccountingSummary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactParetoResult {
    pub frontier: Vec<EvaluatedEnergyPlan>,
    pub nodes_visited: u64,
    /// Number of feasible allocation witnesses encountered before objective deduplication.
    pub feasible_plans_seen: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyDispatchError {
    MissingIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    TooManyDemands,
    TooManyOffers,
    DemandQuantumLimit,
    InvalidQuantum,
    InvalidCurrencyCode,
    InvalidNumericValue(&'static str),
    NegativeValue(&'static str),
    InvalidCapacity,
    InvalidSearchBudget,
    NoFeasiblePlan,
    SearchBudgetExceeded { nodes_visited: u64 },
    PlanIdentityMismatch,
    ScenarioDigestMismatch,
    UnknownOffer,
    DuplicateAllocation,
    ZeroAllocation,
    AllocationExceedsCapacity,
    DemandExceeded,
    DemandNotMet,
    Accounting(AccountingError),
}

impl fmt::Display for EnergyDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingIdentifier(field) => write!(f, "missing identifier: {field}"),
            Self::DuplicateIdentifier(field) => write!(f, "duplicate identifier: {field}"),
            Self::TooManyDemands => write!(f, "scenario exceeds the discrete demand-count limit"),
            Self::TooManyOffers => write!(f, "scenario exceeds the discrete source-offer limit"),
            Self::DemandQuantumLimit => write!(f, "scenario exceeds the total demand-quantum limit"),
            Self::InvalidQuantum => write!(f, "energy quantum must be finite and positive"),
            Self::InvalidCurrencyCode => write!(f, "currency code must be three uppercase ASCII letters"),
            Self::InvalidNumericValue(field) => write!(f, "invalid numeric value: {field}"),
            Self::NegativeValue(field) => write!(f, "negative value: {field}"),
            Self::InvalidCapacity => write!(f, "capacity must be between 1 and the declared per-offer limit"),
            Self::InvalidSearchBudget => write!(f, "search budget must be between 1 and the hard upper bound"),
            Self::NoFeasiblePlan => write!(f, "no feasible energy allocation satisfies all declared demand"),
            Self::SearchBudgetExceeded { nodes_visited } => {
                write!(f, "exact search budget exceeded after {nodes_visited} nodes")
            }
            Self::PlanIdentityMismatch => write!(f, "plan scenario ID or input digest does not match"),
            Self::ScenarioDigestMismatch => write!(f, "scenario input digest does not match canonical scenario content"),
            Self::UnknownOffer => write!(f, "plan references an unknown source offer"),
            Self::DuplicateAllocation => write!(f, "plan allocates the same source offer more than once"),
            Self::ZeroAllocation => write!(f, "canonical plans may contain only positive allocations"),
            Self::AllocationExceedsCapacity => write!(f, "allocation exceeds source-offer capacity"),
            Self::DemandExceeded => write!(f, "allocation supplies more energy than the declared carrier demand"),
            Self::DemandNotMet => write!(f, "plan does not satisfy all declared carrier demand"),
            Self::Accounting(error) => write!(f, "energy accounting rejected the plan: {error}"),
        }
    }
}

impl std::error::Error for EnergyDispatchError {}

/// Calculate a domain-separated SHA-256 digest over the canonical scenario fields.
///
/// Demands and offers are sorted by their stable IDs; the input_digest field itself is
/// excluded to avoid self-reference. Floating-point values are encoded by normalized
/// IEEE-754 bits (negative zero is canonicalized to positive zero). Any schema change
/// must use a new domain/version string.
///
/// This identifies exact declared inputs; it does not attest to their truth or validity.
pub fn canonical_scenario_digest(
    scenario: &CleanAirlineEnergyScenario,
) -> Result<String, EnergyDispatchError> {
    let mut validation_copy = scenario.clone();
    validation_copy.input_digest = "canonicalization-validation-placeholder".to_owned();
    validate_scenario(&validation_copy)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.energy-scenario.v1\0");
    put_string(&mut bytes, &scenario.scenario_id);
    put_string(&mut bytes, &scenario.currency_code);
    put_f64(&mut bytes, scenario.quantum_kwh);
    put_string(&mut bytes, &scenario.service.period_id);
    put_string(&mut bytes, &scenario.service.lifecycle_boundary_id);
    put_f64(&mut bytes, scenario.service.completed_passenger_km);
    put_f64(&mut bytes, scenario.service.available_seat_km);

    let mut demands = scenario.demands.iter().collect::<Vec<_>>();
    demands.sort_by(|left, right| left.id.cmp(&right.id));
    put_u32(&mut bytes, demands.len() as u32);
    for demand in demands {
        put_string(&mut bytes, &demand.id);
        encode_carrier(&mut bytes, &demand.carrier);
        put_u32(&mut bytes, demand.required_quanta);
    }

    let mut offers = scenario.offers.iter().collect::<Vec<_>>();
    offers.sort_by(|left, right| left.id.cmp(&right.id));
    put_u32(&mut bytes, offers.len() as u32);
    for offer in offers {
        put_string(&mut bytes, &offer.id);
        put_string(&mut bytes, &offer.source_id);
        encode_carrier(&mut bytes, &offer.carrier);
        put_u32(&mut bytes, offer.capacity_quanta);
        put_f64(&mut bytes, offer.cost_per_quantum);
        put_f64(&mut bytes, offer.upstream_co2e_kg_per_quantum);
        put_f64(&mut bytes, offer.direct_operational_co2e_kg_per_quantum);
        put_string(&mut bytes, &offer.evidence_ref);
        bytes.push(encode_evidence_class(offer.evidence_class));
    }

    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(7 + digest.len() * 2);
    hex.push_str("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(hex)
}

/// Seal a mutable scenario with its canonical content digest.
pub fn seal_scenario(
    scenario: &mut CleanAirlineEnergyScenario,
) -> Result<(), EnergyDispatchError> {
    scenario.input_digest = canonical_scenario_digest(scenario)?;
    Ok(())
}

/// Verify that the stored digest exactly identifies the current canonical scenario.
pub fn verify_scenario_digest(
    scenario: &CleanAirlineEnergyScenario,
) -> Result<(), EnergyDispatchError> {
    let expected = canonical_scenario_digest(scenario)?;
    if scenario.input_digest != expected {
        return Err(EnergyDispatchError::ScenarioDigestMismatch);
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
    let normalized = if value == 0.0 { 0.0 } else { value };
    put_u64(bytes, normalized.to_bits());
}

fn encode_carrier(bytes: &mut Vec<u8>, carrier: &EnergyCarrier) {
    match carrier {
        EnergyCarrier::Electricity => bytes.push(0),
        EnergyCarrier::Hydrogen => bytes.push(1),
        EnergyCarrier::SustainableAviationFuel => bytes.push(2),
        EnergyCarrier::SyntheticAviationFuel => bytes.push(3),
        EnergyCarrier::FossilJetFuel => bytes.push(4),
        EnergyCarrier::Other(name) => {
            bytes.push(5);
            put_string(bytes, name);
        }
    }
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

/// Validate a plan independently of the search algorithm and recompute all reported
/// ledger metrics from the frozen scenario and selected source offers.
pub fn verify_energy_plan(
    scenario: &CleanAirlineEnergyScenario,
    plan: &EnergyPlan,
) -> Result<EvaluatedEnergyPlan, EnergyDispatchError> {
    validate_scenario(scenario)?;
    verify_scenario_digest(scenario)?;
    if plan.scenario_id != scenario.scenario_id || plan.input_digest != scenario.input_digest {
        return Err(EnergyDispatchError::PlanIdentityMismatch);
    }

    let offers_by_id: BTreeMap<&str, &EnergySourceOffer> = scenario
        .offers
        .iter()
        .map(|offer| (offer.id.as_str(), offer))
        .collect();
    let mut allocated_offer_ids = BTreeSet::new();
    let mut remaining_demand = demand_totals(scenario);
    let mut allocations = plan.allocations.clone();
    allocations.sort();

    if allocations.is_empty() {
        return Err(EnergyDispatchError::NoFeasiblePlan);
    }

    for allocation in &allocations {
        if allocation.quanta == 0 {
            return Err(EnergyDispatchError::ZeroAllocation);
        }
        if !allocated_offer_ids.insert(allocation.offer_id.as_str()) {
            return Err(EnergyDispatchError::DuplicateAllocation);
        }
        let offer = offers_by_id
            .get(allocation.offer_id.as_str())
            .copied()
            .ok_or(EnergyDispatchError::UnknownOffer)?;
        if allocation.quanta > offer.capacity_quanta {
            return Err(EnergyDispatchError::AllocationExceedsCapacity);
        }
        let remaining = remaining_demand
            .get_mut(&offer.carrier)
            .ok_or(EnergyDispatchError::DemandExceeded)?;
        *remaining = remaining
            .checked_sub(allocation.quanta)
            .ok_or(EnergyDispatchError::DemandExceeded)?;
    }

    if remaining_demand.values().any(|remaining| *remaining != 0) {
        return Err(EnergyDispatchError::DemandNotMet);
    }

    let mut records = Vec::with_capacity(allocations.len());
    for allocation in &allocations {
        let offer = offers_by_id
            .get(allocation.offer_id.as_str())
            .copied()
            .ok_or(EnergyDispatchError::UnknownOffer)?;
        let quanta = f64::from(allocation.quanta);
        let energy_kwh = quanta * scenario.quantum_kwh;
        let cost_amount = quanta * offer.cost_per_quantum;
        let upstream_co2e_kg = quanta * offer.upstream_co2e_kg_per_quantum;
        let direct_operational_co2e_kg =
            quanta * offer.direct_operational_co2e_kg_per_quantum;

        records.push(EnergyRecord {
            id: format!("allocation:{}", offer.id),
            source_id: offer.source_id.clone(),
            boundary_id: format!("energy-use:{}", offer.id),
            lifecycle_boundary_id: scenario.service.lifecycle_boundary_id.clone(),
            period_id: scenario.service.period_id.clone(),
            evidence_ref: offer.evidence_ref.clone(),
            evidence_class: offer.evidence_class,
            carrier: offer.carrier.clone(),
            source_energy_kwh: energy_kwh,
            delivered_energy_kwh: energy_kwh,
            upstream_co2e_kg,
            direct_operational_co2e_kg,
            cost_amount,
            currency_code: scenario.currency_code.clone(),
            non_co2: NonCo2Evidence::NotAssessed,
        });
    }

    let ledger = account_energy_ledger(&records, &scenario.service)
        .map_err(EnergyDispatchError::Accounting)?;
    Ok(EvaluatedEnergyPlan {
        plan: EnergyPlan {
            scenario_id: plan.scenario_id.clone(),
            input_digest: plan.input_digest.clone(),
            allocations,
        },
        ledger,
    })
}

/// Exhaustively enumerate the discrete source allocations and return the Pareto
/// frontier for (total cost, accounted lifecycle CO2e). There is no weighted score.
/// A budget-exhausted run returns an error, never a partial frontier labelled exact.
pub fn solve_exact_pareto(
    scenario: &CleanAirlineEnergyScenario,
    max_nodes: u64,
) -> Result<ExactParetoResult, EnergyDispatchError> {
    validate_scenario(scenario)?;
    verify_scenario_digest(scenario)?;
    if max_nodes == 0 || max_nodes > MAX_SEARCH_NODES {
        return Err(EnergyDispatchError::InvalidSearchBudget);
    }

    let mut remaining = demand_totals(scenario);
    let suffix_capacity = build_suffix_capacity(scenario);
    let mut choices = vec![0_u32; scenario.offers.len()];
    let mut search = ExactSearch {
        scenario,
        suffix_capacity: &suffix_capacity,
        max_nodes,
        nodes_visited: 0,
        feasible_plans_seen: 0,
        exhausted: false,
        frontier: Vec::new(),
    };
    search.visit(0, &mut remaining, &mut choices)?;

    if search.exhausted {
        return Err(EnergyDispatchError::SearchBudgetExceeded {
            nodes_visited: search.nodes_visited,
        });
    }
    if search.frontier.is_empty() {
        return Err(EnergyDispatchError::NoFeasiblePlan);
    }
    search.frontier.sort_by(compare_evaluated_plans);
    Ok(ExactParetoResult {
        frontier: search.frontier,
        nodes_visited: search.nodes_visited,
        feasible_plans_seen: search.feasible_plans_seen,
    })
}

/// Deterministic baseline: satisfy each carrier demand using lowest-cost source
/// offers first, breaking price ties by stable offer ID. It is cost-greedy, not
/// carbon-aware, and makes no optimality claim.
pub fn solve_cost_greedy(
    scenario: &CleanAirlineEnergyScenario,
) -> Result<EvaluatedEnergyPlan, EnergyDispatchError> {
    validate_scenario(scenario)?;
    verify_scenario_digest(scenario)?;
    let mut remaining = demand_totals(scenario);
    let mut offers: Vec<&EnergySourceOffer> = scenario.offers.iter().collect();
    offers.sort_by(|left, right| {
        left.cost_per_quantum
            .total_cmp(&right.cost_per_quantum)
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut allocations = Vec::new();
    for (carrier, required) in remaining.iter_mut() {
        let mut unmet = *required;
        for offer in offers.iter().filter(|offer| &offer.carrier == carrier) {
            if unmet == 0 {
                break;
            }
            let allocated = unmet.min(offer.capacity_quanta);
            if allocated > 0 {
                allocations.push(EnergyAllocation {
                    offer_id: offer.id.clone(),
                    quanta: allocated,
                });
                unmet -= allocated;
            }
        }
        if unmet != 0 {
            return Err(EnergyDispatchError::NoFeasiblePlan);
        }
        *required = 0;
    }
    allocations.sort();

    let plan = EnergyPlan {
        scenario_id: scenario.scenario_id.clone(),
        input_digest: scenario.input_digest.clone(),
        allocations,
    };
    verify_energy_plan(scenario, &plan)
}

fn validate_scenario(
    scenario: &CleanAirlineEnergyScenario,
) -> Result<(), EnergyDispatchError> {
    require_identifier(&scenario.scenario_id, "scenario ID")?;
    require_identifier(&scenario.input_digest, "input digest")?;
    require_identifier(&scenario.currency_code, "currency code")?;
    require_identifier(&scenario.service.period_id, "service period ID")?;
    require_identifier(
        &scenario.service.lifecycle_boundary_id,
        "lifecycle boundary ID",
    )?;

    if !scenario.quantum_kwh.is_finite() || scenario.quantum_kwh <= 0.0 {
        return Err(EnergyDispatchError::InvalidQuantum);
    }
    if scenario.currency_code.len() != 3
        || !scenario
            .currency_code
            .bytes()
            .all(|byte| byte.is_ascii_uppercase())
    {
        return Err(EnergyDispatchError::InvalidCurrencyCode);
    }
    if !scenario.service.completed_passenger_km.is_finite()
        || scenario.service.completed_passenger_km <= 0.0
        || !scenario.service.available_seat_km.is_finite()
        || scenario.service.available_seat_km <= 0.0
    {
        return Err(EnergyDispatchError::InvalidNumericValue("transport service"));
    }
    if scenario.demands.is_empty() {
        return Err(EnergyDispatchError::MissingIdentifier("energy demands"));
    }
    if scenario.demands.len() > MAX_DEMANDS {
        return Err(EnergyDispatchError::TooManyDemands);
    }
    if scenario.offers.is_empty() {
        return Err(EnergyDispatchError::MissingIdentifier("energy source offers"));
    }
    if scenario.offers.len() > MAX_OFFERS {
        return Err(EnergyDispatchError::TooManyOffers);
    }

    let mut demand_ids = BTreeSet::new();
    let mut total_demand = 0_u32;
    for demand in &scenario.demands {
        require_identifier(&demand.id, "demand ID")?;
        if !demand_ids.insert(demand.id.as_str()) {
            return Err(EnergyDispatchError::DuplicateIdentifier("demand ID"));
        }
        if demand.required_quanta == 0 {
            return Err(EnergyDispatchError::InvalidCapacity);
        }
        total_demand = total_demand
            .checked_add(demand.required_quanta)
            .ok_or(EnergyDispatchError::DemandQuantumLimit)?;
    }
    if total_demand > MAX_TOTAL_DEMAND_QUANTA {
        return Err(EnergyDispatchError::DemandQuantumLimit);
    }

    let mut offer_ids = BTreeSet::new();
    let mut source_ids = BTreeSet::new();
    for offer in &scenario.offers {
        require_identifier(&offer.id, "offer ID")?;
        require_identifier(&offer.source_id, "source ID")?;
        require_identifier(&offer.evidence_ref, "offer evidence reference")?;
        if !offer_ids.insert(offer.id.as_str()) {
            return Err(EnergyDispatchError::DuplicateIdentifier("offer ID"));
        }
        // V1 permits one offer per source ID so declared capacity cannot be duplicated.
        if !source_ids.insert(offer.source_id.as_str()) {
            return Err(EnergyDispatchError::DuplicateIdentifier("source ID"));
        }
        if offer.capacity_quanta == 0 || offer.capacity_quanta > MAX_QUANTA_PER_OFFER {
            return Err(EnergyDispatchError::InvalidCapacity);
        }
        if !offer.cost_per_quantum.is_finite() {
            return Err(EnergyDispatchError::InvalidNumericValue("unit cost"));
        }
        if offer.cost_per_quantum < 0.0 {
            return Err(EnergyDispatchError::NegativeValue("unit cost"));
        }
        if !offer.upstream_co2e_kg_per_quantum.is_finite() {
            return Err(EnergyDispatchError::InvalidNumericValue("upstream CO2e factor"));
        }
        // Negative lifecycle factors can encode removal/credit accounting, which
        // must not become a free optimization shortcut without a separately typed
        // and independently qualified removal contract.
        if offer.upstream_co2e_kg_per_quantum < 0.0 {
            return Err(EnergyDispatchError::NegativeValue("upstream CO2e factor"));
        }
        if !offer.direct_operational_co2e_kg_per_quantum.is_finite() {
            return Err(EnergyDispatchError::InvalidNumericValue("direct CO2e factor"));
        }
        if offer.direct_operational_co2e_kg_per_quantum < 0.0 {
            return Err(EnergyDispatchError::NegativeValue("direct CO2e factor"));
        }
    }
    Ok(())
}

fn demand_totals(scenario: &CleanAirlineEnergyScenario) -> BTreeMap<EnergyCarrier, u32> {
    let mut totals = BTreeMap::new();
    for demand in &scenario.demands {
        *totals.entry(demand.carrier.clone()).or_insert(0_u32) += demand.required_quanta;
    }
    totals
}

fn build_suffix_capacity(
    scenario: &CleanAirlineEnergyScenario,
) -> Vec<BTreeMap<EnergyCarrier, u32>> {
    let mut suffix = vec![BTreeMap::new(); scenario.offers.len() + 1];
    for index in (0..scenario.offers.len()).rev() {
        suffix[index] = suffix[index + 1].clone();
        let offer = &scenario.offers[index];
        *suffix[index].entry(offer.carrier.clone()).or_insert(0_u32) +=
            offer.capacity_quanta;
    }
    suffix
}

struct ExactSearch<'a> {
    scenario: &'a CleanAirlineEnergyScenario,
    suffix_capacity: &'a [BTreeMap<EnergyCarrier, u32>],
    max_nodes: u64,
    nodes_visited: u64,
    feasible_plans_seen: u64,
    exhausted: bool,
    frontier: Vec<EvaluatedEnergyPlan>,
}

impl ExactSearch<'_> {
    fn visit(
        &mut self,
        index: usize,
        remaining: &mut BTreeMap<EnergyCarrier, u32>,
        choices: &mut [u32],
    ) -> Result<(), EnergyDispatchError> {
        if self.nodes_visited >= self.max_nodes {
            self.exhausted = true;
            return Ok(());
        }
        self.nodes_visited += 1;

        if remaining.values().all(|needed| *needed == 0) {
            // Offers after this point must have zero allocations, even if a previous
            // branch left stale choices in a deeper stack frame.
            for choice in choices.iter_mut().skip(index) {
                *choice = 0;
            }
            let allocations = choices
                .iter()
                .enumerate()
                .filter_map(|(offer_index, quanta)| {
                    if *quanta == 0 {
                        None
                    } else {
                        Some(EnergyAllocation {
                            offer_id: self.scenario.offers[offer_index].id.clone(),
                            quanta: *quanta,
                        })
                    }
                })
                .collect();
            let plan = EnergyPlan {
                scenario_id: self.scenario.scenario_id.clone(),
                input_digest: self.scenario.input_digest.clone(),
                allocations,
            };
            let evaluation = verify_energy_plan(self.scenario, &plan)?;
            self.feasible_plans_seen = self
                .feasible_plans_seen
                .checked_add(1)
                .ok_or(EnergyDispatchError::DemandQuantumLimit)?;
            insert_pareto(&mut self.frontier, evaluation);
            return Ok(());
        }

        if index >= self.scenario.offers.len() {
            return Ok(());
        }

        for (carrier, needed) in remaining.iter() {
            let available = self.suffix_capacity[index].get(carrier).copied().unwrap_or(0);
            if available < *needed {
                return Ok(());
            }
        }

        let offer = &self.scenario.offers[index];
        let needed = remaining.get(&offer.carrier).copied().unwrap_or(0);
        let max_allocate = needed.min(offer.capacity_quanta);
        for allocation in 0..=max_allocate {
            choices[index] = allocation;
            remaining.insert(offer.carrier.clone(), needed - allocation);
            self.visit(index + 1, remaining, choices)?;
            remaining.insert(offer.carrier.clone(), needed);
            choices[index] = 0;
            if self.exhausted {
                break;
            }
        }
        Ok(())
    }
}

fn insert_pareto(
    frontier: &mut Vec<EvaluatedEnergyPlan>,
    candidate: EvaluatedEnergyPlan,
) {
    if frontier.iter().any(|current| dominates(&current.ledger, &candidate.ledger)) {
        return;
    }

    // For equal objective pairs, keep one stable allocation witness only.
    if frontier.iter().any(|current| {
        same_objectives(&current.ledger, &candidate.ledger)
            && current.plan.allocations <= candidate.plan.allocations
    }) {
        return;
    }

    frontier.retain(|current| {
        !dominates(&candidate.ledger, &current.ledger)
            && !(same_objectives(&current.ledger, &candidate.ledger)
                && candidate.plan.allocations < current.plan.allocations)
    });
    frontier.push(candidate);
}

fn dominates(left: &AccountingSummary, right: &AccountingSummary) -> bool {
    left.total_cost <= right.total_cost
        && left.accounted_lifecycle_co2e_kg <= right.accounted_lifecycle_co2e_kg
        && (left.total_cost < right.total_cost
            || left.accounted_lifecycle_co2e_kg < right.accounted_lifecycle_co2e_kg)
}

fn same_objectives(left: &AccountingSummary, right: &AccountingSummary) -> bool {
    left.total_cost == right.total_cost
        && left.accounted_lifecycle_co2e_kg == right.accounted_lifecycle_co2e_kg
}

fn compare_evaluated_plans(
    left: &EvaluatedEnergyPlan,
    right: &EvaluatedEnergyPlan,
) -> std::cmp::Ordering {
    left.ledger
        .total_cost
        .total_cmp(&right.ledger.total_cost)
        .then_with(|| {
            left.ledger
                .accounted_lifecycle_co2e_kg
                .total_cmp(&right.ledger.accounted_lifecycle_co2e_kg)
        })
        .then_with(|| left.plan.allocations.cmp(&right.plan.allocations))
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), EnergyDispatchError> {
    if value.trim().is_empty() {
        return Err(EnergyDispatchError::MissingIdentifier(field));
    }
    Ok(())
}
