// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Referential closure for a complete transport trade-study evidence bundle.

use serde::{Deserialize, Serialize};

use crate::{
    claims::{ClaimId, ClaimRegistry},
    trade::{DemandScenario, ParetoStudySpec, TradeStudyResult, TransportArchitectureOption},
};

/// A self-contained study package whose claims, scenarios, options, study spec,
/// and results all refer to records that actually exist in the same bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeStudyBundle {
    pub bundle_id: String,
    pub claims: ClaimRegistry,
    pub scenarios: Vec<DemandScenario>,
    pub options: Vec<TransportArchitectureOption>,
    pub study: ParetoStudySpec,
    pub results: Vec<TradeStudyResult>,
}

impl TradeStudyBundle {
    pub fn is_well_formed(&self) -> bool {
        if self.bundle_id.trim().is_empty()
            || !self.claims.is_well_formed()
            || self.scenarios.iter().any(|scenario| !scenario.is_well_formed())
            || self.options.iter().any(|option| !option.is_well_formed())
            || !self.study.is_well_formed()
            || self.results.iter().any(|result| !result.is_well_formed())
        {
            return false;
        }

        if !unique_strings(self.scenarios.iter().map(|scenario| scenario.scenario_id.as_str()))
            || !unique_strings(self.options.iter().map(|option| option.option_id.as_str()))
            || !unique_strings(self.results.iter().map(|result| result.result_id.as_str()))
        {
            return false;
        }

        if !self
            .study
            .scenario_ids
            .iter()
            .all(|id| self.scenarios.iter().any(|scenario| &scenario.scenario_id == id))
            || !self
                .study
                .option_ids
                .iter()
                .all(|id| self.options.iter().any(|option| &option.option_id == id))
        {
            return false;
        }

        if !self.claim_refs_exist(self.study.claim_refs.iter()) {
            return false;
        }

        for scenario in &self.scenarios {
            if !self.claim_refs_exist(scenario.claim_refs.iter())
                || scenario
                    .demands
                    .iter()
                    .any(|demand| !self.claim_refs_exist(demand.claim_refs.iter()))
            {
                return false;
            }
        }

        for option in &self.options {
            if !self.claim_refs_exist(option.claim_refs.iter()) {
                return false;
            }
        }

        self.results.iter().all(|result| {
            self.scenarios
                .iter()
                .any(|scenario| scenario.scenario_id == result.scenario_id)
                && self
                    .options
                    .iter()
                    .any(|option| option.option_id == result.option_id)
                && self.claim_refs_exist(result.claim_refs.iter())
        })
    }

    fn claim_refs_exist<'a>(&self, refs: impl Iterator<Item = &'a ClaimId>) -> bool {
        refs.all(|id| self.claims.get(id).is_some())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        claims::{ClaimDisposition, ClaimKind, ClaimRecord},
        trade::{
            CargoDemand, LifecycleCostModel, ObjectiveDirection, ReferenceDemandTier,
            ReliabilitySummary,
        },
    };
    use symthaea_infrastructure::{
        transport::{BoundedMetric, CargoKind, TransportMode},
        AssetId, EvidenceStatus,
    };

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

    fn bundle() -> TradeStudyBundle {
        let claim = ClaimRecord {
            claim_id: ClaimId::new("assumption-1"),
            kind: ClaimKind::Assumption,
            subject: "traffic".into(),
            statement: "reference traffic assumption".into(),
            scalar: None,
            evidence_status: EvidenceStatus::Declared,
            disposition: ClaimDisposition::Active,
            source_refs: vec![],
            supersedes: vec![],
        };

        let scenario = DemandScenario {
            scenario_id: "d0".into(),
            tier: ReferenceDemandTier::D0Research,
            label: "research".into(),
            demands: vec![CargoDemand {
                demand_id: "cargo".into(),
                origin: AssetId::new("surface"),
                destination: AssetId::new("eml1"),
                cargo_kind: CargoKind::General,
                annual_mass_kg: metric(1.0, 2.0, 3.0, "kg/year"),
                peak_shipment_mass_kg: metric(1.0, 1.0, 1.0, "kg"),
                maximum_transit_time_s: None,
                evidence_status: EvidenceStatus::Declared,
                evidence_refs: vec![],
                claim_refs: vec![ClaimId::new("assumption-1")],
            }],
            claim_refs: vec![ClaimId::new("assumption-1")],
            evidence_status: EvidenceStatus::Declared,
            evidence_refs: vec![],
        };

        let option = TransportArchitectureOption {
            option_id: "elevator-option".into(),
            label: "elevator reference".into(),
            modes: vec![TransportMode::LunarElevator],
            required_assets: vec![],
            claim_refs: vec![ClaimId::new("assumption-1")],
            evidence_refs: vec![],
        };

        let lifecycle_cost = LifecycleCostModel {
            currency: "USD".into(),
            price_year: 2026,
            development_cost: metric(0.0, 1.0, 2.0, "USD"),
            deployment_capex: metric(0.0, 1.0, 2.0, "USD"),
            annual_operations_cost: metric(0.0, 1.0, 2.0, "USD/year"),
            annual_maintenance_cost: metric(0.0, 1.0, 2.0, "USD/year"),
            annual_replacement_reserve: metric(0.0, 1.0, 2.0, "USD/year"),
        };
        let reliability = ReliabilitySummary {
            delivery_success_probability: metric(0.8, 0.9, 0.99, "1"),
            availability_fraction: metric(0.5, 0.8, 0.99, "1"),
            catastrophic_loss_probability_per_shipment: metric(0.0, 0.01, 0.1, "1"),
            single_provider_dependency_fraction: metric(0.0, 0.5, 1.0, "1"),
            mean_time_to_repair_h: metric(0.0, 10.0, 100.0, "h"),
        };
        let result = TradeStudyResult {
            result_id: "result-1".into(),
            scenario_id: "d0".into(),
            option_id: "elevator-option".into(),
            model_version: "model-v0".into(),
            lifecycle_cost,
            reliability,
            cost_per_delivered_kg: metric(0.0, 1.0, 2.0, "USD/kg"),
            earth_imported_mass_kg_per_delivered_kg: metric(0.0, 1.0, 2.0, "kg/kg"),
            electrical_energy_kwh_per_delivered_kg: metric(0.0, 1.0, 2.0, "kWh/kg"),
            propellant_kg_per_delivered_kg: metric(0.0, 0.1, 1.0, "kg/kg"),
            delivered_throughput_kg_per_year: metric(1.0, 2.0, 3.0, "kg/year"),
            end_to_end_latency_s: metric(1.0, 2.0, 3.0, "s"),
            spare_inventory_kg: metric(0.0, 1.0, 2.0, "kg"),
            construction_mass_kg: metric(0.0, 1.0, 2.0, "kg"),
            bootstrap_multiplication_factor: metric(0.0, 1.0, 2.0, "1"),
            additional_objectives: vec![],
            claim_refs: vec![ClaimId::new("assumption-1")],
            evidence_refs: vec![],
        };

        TradeStudyBundle {
            bundle_id: "bundle-1".into(),
            claims: ClaimRegistry {
                claims: vec![claim],
            },
            scenarios: vec![scenario],
            options: vec![option],
            study: ParetoStudySpec {
                study_id: "study-1".into(),
                scenario_ids: vec!["d0".into()],
                option_ids: vec!["elevator-option".into()],
                objectives: vec![("cost_per_delivered_kg".into(), ObjectiveDirection::Minimize)],
                claim_refs: vec![ClaimId::new("assumption-1")],
            },
            results: vec![result],
        }
    }

    #[test]
    fn complete_bundle_is_referentially_closed() {
        assert!(bundle().is_well_formed());
    }

    #[test]
    fn dangling_claim_reference_fails_closed() {
        let mut bundle = bundle();
        bundle.scenarios[0].claim_refs = vec![ClaimId::new("missing")];
        assert!(!bundle.is_well_formed());
    }

    #[test]
    fn unknown_option_in_result_fails_closed() {
        let mut bundle = bundle();
        bundle.results[0].option_id = "not-present".into();
        assert!(!bundle.is_well_formed());
    }
}
