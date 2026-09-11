// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neutral demand and trade-study vocabulary for cislunar infrastructure.
//!
//! This module deliberately contains no optimizer. It defines the evidence-
//! bearing inputs and outputs required to compare transport architectures.

use serde::{Deserialize, Serialize};
use symthaea_infrastructure::{
    transport::{BoundedMetric, CargoKind, TransportMode},
    AssetId, EvidenceStatus,
};

use crate::claims::ClaimId;

/// Named operating regimes only. These labels do not imply hard-coded traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReferenceDemandTier {
    /// Pathfinder science, demonstrations, and small logistics demand.
    D0Research,
    /// Early outpost / intermittent crew and robotic operations.
    D1Outpost,
    /// Sustained surface presence and growing infrastructure demand.
    D2SustainedPresence,
    /// Industrial-scale resource and construction flows.
    D3Industrial,
    /// Mature high-throughput cislunar economy.
    D4MatureNetwork,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CargoDemand {
    pub demand_id: String,
    pub origin: AssetId,
    pub destination: AssetId,
    pub cargo_kind: CargoKind,
    pub annual_mass_kg: BoundedMetric,
    pub peak_shipment_mass_kg: BoundedMetric,
    pub maximum_transit_time_s: Option<BoundedMetric>,
    pub evidence_status: EvidenceStatus,
    pub evidence_refs: Vec<String>,
    pub claim_refs: Vec<ClaimId>,
}

impl CargoDemand {
    pub fn is_well_formed(&self) -> bool {
        !self.demand_id.trim().is_empty()
            && self.origin.is_well_formed()
            && self.destination.is_well_formed()
            && self.origin != self.destination
            && non_negative(&self.annual_mass_kg)
            && non_negative(&self.peak_shipment_mass_kg)
            && self
                .maximum_transit_time_s
                .as_ref()
                .is_none_or(|metric| non_negative(metric) && metric.lower > 0.0)
            && self.claim_refs.iter().all(ClaimId::is_well_formed)
            && evidence_refs_valid(self.evidence_status, &self.evidence_refs)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DemandScenario {
    pub scenario_id: String,
    pub tier: ReferenceDemandTier,
    pub label: String,
    pub demands: Vec<CargoDemand>,
    pub claim_refs: Vec<ClaimId>,
    pub evidence_status: EvidenceStatus,
    pub evidence_refs: Vec<String>,
}

impl DemandScenario {
    pub fn is_well_formed(&self) -> bool {
        !self.scenario_id.trim().is_empty()
            && !self.label.trim().is_empty()
            && !self.demands.is_empty()
            && self.demands.iter().all(CargoDemand::is_well_formed)
            && self.claim_refs.iter().all(ClaimId::is_well_formed)
            && evidence_refs_valid(self.evidence_status, &self.evidence_refs)
            && unique_strings(self.demands.iter().map(|demand| demand.demand_id.as_str()))
    }
}

/// One candidate network architecture. Modes are descriptive; detailed edges
/// and physics remain in the transport graph and domain adapters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportArchitectureOption {
    pub option_id: String,
    pub label: String,
    pub modes: Vec<TransportMode>,
    pub required_assets: Vec<AssetId>,
    pub claim_refs: Vec<ClaimId>,
    pub evidence_refs: Vec<String>,
}

impl TransportArchitectureOption {
    pub fn is_well_formed(&self) -> bool {
        !self.option_id.trim().is_empty()
            && !self.label.trim().is_empty()
            && !self.modes.is_empty()
            && self.required_assets.iter().all(AssetId::is_well_formed)
            && self.claim_refs.iter().all(ClaimId::is_well_formed)
            && self.evidence_refs.iter().all(|reference| !reference.trim().is_empty())
            && unique_modes(&self.modes)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifecycleCostModel {
    pub currency: String,
    pub price_year: u16,
    pub development_cost: BoundedMetric,
    pub deployment_capex: BoundedMetric,
    pub annual_operations_cost: BoundedMetric,
    pub annual_maintenance_cost: BoundedMetric,
    pub annual_replacement_reserve: BoundedMetric,
}

impl LifecycleCostModel {
    pub fn is_well_formed(&self) -> bool {
        !self.currency.trim().is_empty()
            && self.price_year >= 1900
            && non_negative(&self.development_cost)
            && non_negative(&self.deployment_capex)
            && non_negative(&self.annual_operations_cost)
            && non_negative(&self.annual_maintenance_cost)
            && non_negative(&self.annual_replacement_reserve)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilitySummary {
    pub delivery_success_probability: BoundedMetric,
    pub availability_fraction: BoundedMetric,
    pub catastrophic_loss_probability_per_shipment: BoundedMetric,
    pub single_provider_dependency_fraction: BoundedMetric,
    pub mean_time_to_repair_h: BoundedMetric,
}

impl ReliabilitySummary {
    pub fn is_well_formed(&self) -> bool {
        probability(&self.delivery_success_probability)
            && probability(&self.availability_fraction)
            && probability(&self.catastrophic_loss_probability_per_shipment)
            && probability(&self.single_provider_dependency_fraction)
            && non_negative(&self.mean_time_to_repair_h)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveMetric {
    pub key: String,
    pub direction: ObjectiveDirection,
    pub value: BoundedMetric,
}

impl ObjectiveMetric {
    pub fn is_well_formed(&self) -> bool {
        !self.key.trim().is_empty() && self.value.is_well_formed()
    }
}

/// Common architecture result record. Domain-specific metrics may be appended
/// through `additional_objectives`; no scalar "best architecture" score exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeStudyResult {
    pub result_id: String,
    pub scenario_id: String,
    pub option_id: String,
    pub model_version: String,
    pub lifecycle_cost: LifecycleCostModel,
    pub reliability: ReliabilitySummary,
    pub cost_per_delivered_kg: BoundedMetric,
    pub earth_imported_mass_kg_per_delivered_kg: BoundedMetric,
    pub electrical_energy_kwh_per_delivered_kg: BoundedMetric,
    pub propellant_kg_per_delivered_kg: BoundedMetric,
    pub delivered_throughput_kg_per_year: BoundedMetric,
    pub end_to_end_latency_s: BoundedMetric,
    pub spare_inventory_kg: BoundedMetric,
    pub construction_mass_kg: BoundedMetric,
    pub bootstrap_multiplication_factor: BoundedMetric,
    pub additional_objectives: Vec<ObjectiveMetric>,
    pub claim_refs: Vec<ClaimId>,
    pub evidence_refs: Vec<String>,
}

impl TradeStudyResult {
    pub fn is_well_formed(&self) -> bool {
        !self.result_id.trim().is_empty()
            && !self.scenario_id.trim().is_empty()
            && !self.option_id.trim().is_empty()
            && !self.model_version.trim().is_empty()
            && self.lifecycle_cost.is_well_formed()
            && self.reliability.is_well_formed()
            && non_negative(&self.cost_per_delivered_kg)
            && non_negative(&self.earth_imported_mass_kg_per_delivered_kg)
            && non_negative(&self.electrical_energy_kwh_per_delivered_kg)
            && non_negative(&self.propellant_kg_per_delivered_kg)
            && non_negative(&self.delivered_throughput_kg_per_year)
            && non_negative(&self.end_to_end_latency_s)
            && non_negative(&self.spare_inventory_kg)
            && non_negative(&self.construction_mass_kg)
            && non_negative(&self.bootstrap_multiplication_factor)
            && self.additional_objectives.iter().all(ObjectiveMetric::is_well_formed)
            && unique_strings(
                self.additional_objectives
                    .iter()
                    .map(|objective| objective.key.as_str()),
            )
            && self.claim_refs.iter().all(ClaimId::is_well_formed)
            && self.evidence_refs.iter().all(|reference| !reference.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoStudySpec {
    pub study_id: String,
    pub scenario_ids: Vec<String>,
    pub option_ids: Vec<String>,
    /// Metric keys and directions only. This type does not implement ranking.
    pub objectives: Vec<(String, ObjectiveDirection)>,
    pub claim_refs: Vec<ClaimId>,
}

impl ParetoStudySpec {
    pub fn is_well_formed(&self) -> bool {
        !self.study_id.trim().is_empty()
            && !self.scenario_ids.is_empty()
            && !self.option_ids.is_empty()
            && !self.objectives.is_empty()
            && self.scenario_ids.iter().all(|id| !id.trim().is_empty())
            && self.option_ids.iter().all(|id| !id.trim().is_empty())
            && self
                .objectives
                .iter()
                .all(|(key, _)| !key.trim().is_empty())
            && unique_strings(self.scenario_ids.iter().map(String::as_str))
            && unique_strings(self.option_ids.iter().map(String::as_str))
            && unique_strings(self.objectives.iter().map(|(key, _)| key.as_str()))
            && self.claim_refs.iter().all(ClaimId::is_well_formed)
    }
}

fn non_negative(metric: &BoundedMetric) -> bool {
    metric.is_well_formed() && metric.lower >= 0.0
}

fn probability(metric: &BoundedMetric) -> bool {
    metric.is_well_formed() && metric.lower >= 0.0 && metric.upper <= 1.0
}

fn evidence_refs_valid(status: EvidenceStatus, refs: &[String]) -> bool {
    let non_empty = refs.iter().all(|reference| !reference.trim().is_empty());
    match status {
        EvidenceStatus::Tested | EvidenceStatus::Qualified => non_empty && !refs.is_empty(),
        EvidenceStatus::Declared | EvidenceStatus::Modeled => non_empty,
    }
}

fn unique_strings<'a>(items: impl Iterator<Item = &'a str>) -> bool {
    let mut seen: Vec<&str> = Vec::new();
    for item in items {
        if seen.contains(&item) {
            return false;
        }
        seen.push(item);
    }
    true
}

fn unique_modes(items: &[TransportMode]) -> bool {
    for (index, item) in items.iter().enumerate() {
        if items[index + 1..].iter().any(|other| other == item) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(lower: f64, nominal: f64, upper: f64, unit: &str) -> BoundedMetric {
        BoundedMetric {
            lower,
            nominal,
            upper,
            unit: unit.into(),
            evidence_status: EvidenceStatus::Declared,
            evidence_refs: vec![],
        }
    }

    #[test]
    fn demand_tier_does_not_encode_traffic_numbers() {
        let scenario = DemandScenario {
            scenario_id: "d0-reference".into(),
            tier: ReferenceDemandTier::D0Research,
            label: "research/reference".into(),
            demands: vec![CargoDemand {
                demand_id: "science-cargo".into(),
                origin: AssetId::new("earth"),
                destination: AssetId::new("moon"),
                cargo_kind: CargoKind::General,
                annual_mass_kg: metric(1.0, 2.0, 3.0, "kg/year"),
                peak_shipment_mass_kg: metric(1.0, 1.0, 1.0, "kg"),
                maximum_transit_time_s: None,
                evidence_status: EvidenceStatus::Declared,
                evidence_refs: vec![],
                claim_refs: vec![ClaimId::new("demand-assumption")],
            }],
            claim_refs: vec![ClaimId::new("demand-assumption")],
            evidence_status: EvidenceStatus::Declared,
            evidence_refs: vec![],
        };
        assert!(scenario.is_well_formed());
    }

    #[test]
    fn tested_demand_requires_evidence() {
        let mut demand = CargoDemand {
            demand_id: "industrial".into(),
            origin: AssetId::new("south-pole"),
            destination: AssetId::new("anchor"),
            cargo_kind: CargoKind::BulkCommodity,
            annual_mass_kg: metric(1.0, 10.0, 100.0, "kg/year"),
            peak_shipment_mass_kg: metric(1.0, 5.0, 10.0, "kg"),
            maximum_transit_time_s: None,
            evidence_status: EvidenceStatus::Tested,
            evidence_refs: vec![],
            claim_refs: vec![],
        };
        assert!(!demand.is_well_formed());
        demand.evidence_refs.push("measured-flow-study".into());
        assert!(demand.is_well_formed());
    }

    #[test]
    fn option_rejects_duplicate_modes() {
        let option = TransportArchitectureOption {
            option_id: "bad".into(),
            label: "duplicate elevator".into(),
            modes: vec![TransportMode::LunarElevator, TransportMode::LunarElevator],
            required_assets: vec![],
            claim_refs: vec![],
            evidence_refs: vec![],
        };
        assert!(!option.is_well_formed());
    }

    #[test]
    fn reliability_probability_bounds_are_enforced() {
        let reliability = ReliabilitySummary {
            delivery_success_probability: metric(0.9, 0.99, 1.1, "1"),
            availability_fraction: metric(0.5, 0.9, 0.99, "1"),
            catastrophic_loss_probability_per_shipment: metric(0.0, 0.001, 0.01, "1"),
            single_provider_dependency_fraction: metric(0.0, 0.5, 1.0, "1"),
            mean_time_to_repair_h: metric(1.0, 10.0, 100.0, "h"),
        };
        assert!(!reliability.is_well_formed());
    }

    #[test]
    fn pareto_spec_refuses_duplicate_objective_keys() {
        let spec = ParetoStudySpec {
            study_id: "study-1".into(),
            scenario_ids: vec!["d0".into()],
            option_ids: vec!["lander".into(), "elevator".into()],
            objectives: vec![
                ("cost_per_kg".into(), ObjectiveDirection::Minimize),
                ("cost_per_kg".into(), ObjectiveDirection::Minimize),
            ],
            claim_refs: vec![],
        };
        assert!(!spec.is_well_formed());
    }
}
