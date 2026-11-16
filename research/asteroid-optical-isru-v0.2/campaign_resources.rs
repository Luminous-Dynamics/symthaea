// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Multi-resource campaign budget and Pareto decision reference.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/campaign_resources.rs -o /tmp/isru-resource-tests
//   /tmp/isru-resource-tests

#![forbid(unsafe_code)]

//! Preserve campaign resource costs as a typed vector.
//!
//! Hard budget admission is component-wise and happens before ranking. Pareto
//! comparisons preserve incomparable actions. A scalar cost-adjusted score is
//! available only when the caller supplies explicit, versioned shadow prices.
//! The module does not infer physical meter readings or authorize experiments.

use std::error::Error;
use std::fmt;

const RESOURCE_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyMeterBoundary {
    ElectricalInput,
    OpticalSourceOutput,
    ReceiverIncident,
    ReceiverAbsorbed,
    UsefulProcessInput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBasis {
    pub energy_meter_boundary: EnergyMeterBoundary,
    /// ISO currency code or a declared benchmark accounting unit.
    pub assay_cost_unit: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResourceVector {
    pub rig_hours: f64,
    pub source_energy_wh: f64,
    pub feedstock_g: f64,
    pub assay_cost: f64,
    pub maintenance_minutes: f64,
    pub consumables_g: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CampaignAction {
    pub id: String,
    /// Expected reduction in declared decision loss; same units for all actions.
    pub decision_value: f64,
    pub resources: ResourceVector,
    pub basis: ResourceBasis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CampaignBudget {
    pub resources: ResourceVector,
    pub basis: ResourceBasis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BudgetAssessment {
    pub action_id: String,
    pub affordable: bool,
    pub over_budget_resources: Vec<&'static str>,
    pub remaining_if_affordable: Option<ResourceVector>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceShadowPrices {
    pub version: String,
    pub config_digest: String,
    pub rig_hour: f64,
    pub source_energy_wh: f64,
    pub feedstock_g: f64,
    pub assay_cost: f64,
    pub maintenance_minute: f64,
    pub consumable_g: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceError {
    EmptyIdentity(&'static str),
    EmptyCostUnit,
    BasisMismatch,
    InvalidResource(&'static str),
    InvalidDecisionValue,
    InvalidShadowPrice(&'static str),
    AllShadowPricesZero,
    ZeroWeightedCost(String),
    NoAffordableActions,
}

impl fmt::Display for ResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ResourceError {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl ResourceVector {
    pub fn validate(&self) -> Result<(), ResourceError> {
        for (name, value) in [
            ("rig_hours", self.rig_hours),
            ("source_energy_wh", self.source_energy_wh),
            ("feedstock_g", self.feedstock_g),
            ("assay_cost", self.assay_cost),
            ("maintenance_minutes", self.maintenance_minutes),
            ("consumables_g", self.consumables_g),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ResourceError::InvalidResource(name));
            }
        }
        Ok(())
    }

    fn fits_within(&self, budget: &Self) -> bool {
        self.rig_hours <= budget.rig_hours + RESOURCE_TOLERANCE
            && self.source_energy_wh <= budget.source_energy_wh + RESOURCE_TOLERANCE
            && self.feedstock_g <= budget.feedstock_g + RESOURCE_TOLERANCE
            && self.assay_cost <= budget.assay_cost + RESOURCE_TOLERANCE
            && self.maintenance_minutes <= budget.maintenance_minutes + RESOURCE_TOLERANCE
            && self.consumables_g <= budget.consumables_g + RESOURCE_TOLERANCE
    }

    fn over_budget(&self, budget: &Self) -> Vec<&'static str> {
        let mut over = Vec::new();
        if self.rig_hours > budget.rig_hours + RESOURCE_TOLERANCE {
            over.push("rig_hours");
        }
        if self.source_energy_wh > budget.source_energy_wh + RESOURCE_TOLERANCE {
            over.push("source_energy_wh");
        }
        if self.feedstock_g > budget.feedstock_g + RESOURCE_TOLERANCE {
            over.push("feedstock_g");
        }
        if self.assay_cost > budget.assay_cost + RESOURCE_TOLERANCE {
            over.push("assay_cost");
        }
        if self.maintenance_minutes > budget.maintenance_minutes + RESOURCE_TOLERANCE {
            over.push("maintenance_minutes");
        }
        if self.consumables_g > budget.consumables_g + RESOURCE_TOLERANCE {
            over.push("consumables_g");
        }
        over
    }

    fn subtract(&self, cost: &Self) -> Self {
        Self {
            rig_hours: self.rig_hours - cost.rig_hours,
            source_energy_wh: self.source_energy_wh - cost.source_energy_wh,
            feedstock_g: self.feedstock_g - cost.feedstock_g,
            assay_cost: self.assay_cost - cost.assay_cost,
            maintenance_minutes: self.maintenance_minutes - cost.maintenance_minutes,
            consumables_g: self.consumables_g - cost.consumables_g,
        }
    }

    fn componentwise_le(&self, other: &Self) -> bool {
        self.rig_hours <= other.rig_hours + RESOURCE_TOLERANCE
            && self.source_energy_wh <= other.source_energy_wh + RESOURCE_TOLERANCE
            && self.feedstock_g <= other.feedstock_g + RESOURCE_TOLERANCE
            && self.assay_cost <= other.assay_cost + RESOURCE_TOLERANCE
            && self.maintenance_minutes <= other.maintenance_minutes + RESOURCE_TOLERANCE
            && self.consumables_g <= other.consumables_g + RESOURCE_TOLERANCE
    }

    fn componentwise_strict(&self, other: &Self) -> bool {
        self.rig_hours < other.rig_hours - RESOURCE_TOLERANCE
            || self.source_energy_wh < other.source_energy_wh - RESOURCE_TOLERANCE
            || self.feedstock_g < other.feedstock_g - RESOURCE_TOLERANCE
            || self.assay_cost < other.assay_cost - RESOURCE_TOLERANCE
            || self.maintenance_minutes < other.maintenance_minutes - RESOURCE_TOLERANCE
            || self.consumables_g < other.consumables_g - RESOURCE_TOLERANCE
    }
}

impl ResourceBasis {
    fn validate(&self) -> Result<(), ResourceError> {
        if !nonempty(&self.assay_cost_unit) {
            return Err(ResourceError::EmptyCostUnit);
        }
        Ok(())
    }
}

impl CampaignBudget {
    pub fn validate(&self) -> Result<(), ResourceError> {
        self.resources.validate()?;
        self.basis.validate()
    }
}

impl CampaignAction {
    pub fn validate_against(&self, budget: &CampaignBudget) -> Result<(), ResourceError> {
        if !nonempty(&self.id) {
            return Err(ResourceError::EmptyIdentity("action_id"));
        }
        if !self.decision_value.is_finite() || self.decision_value < 0.0 {
            return Err(ResourceError::InvalidDecisionValue);
        }
        self.resources.validate()?;
        if self.basis != budget.basis {
            return Err(ResourceError::BasisMismatch);
        }
        Ok(())
    }
}

/// Assess all candidates against the same frozen component-wise budget.
/// Over-budget candidates remain in the returned table with explicit reasons.
pub fn assess_budget(
    actions: &[CampaignAction],
    budget: &CampaignBudget,
) -> Result<Vec<BudgetAssessment>, ResourceError> {
    budget.validate()?;
    let mut ids = std::collections::BTreeSet::new();
    let mut assessments = Vec::with_capacity(actions.len());

    for action in actions {
        action.validate_against(budget)?;
        if !ids.insert(action.id.as_str()) {
            return Err(ResourceError::EmptyIdentity("duplicate_action_id"));
        }
        let over_budget_resources = action.resources.over_budget(&budget.resources);
        let affordable = over_budget_resources.is_empty();
        assessments.push(BudgetAssessment {
            action_id: action.id.clone(),
            affordable,
            over_budget_resources,
            remaining_if_affordable: if affordable {
                Some(budget.resources.subtract(&action.resources))
            } else {
                None
            },
        });
    }

    Ok(assessments)
}

/// Returns true when lhs is no worse on decision value and uses no more of
/// every resource, with at least one strict improvement.
pub fn dominates(lhs: &CampaignAction, rhs: &CampaignAction) -> bool {
    lhs.basis == rhs.basis
        && lhs.decision_value + RESOURCE_TOLERANCE >= rhs.decision_value
        && lhs.resources.componentwise_le(&rhs.resources)
        && (lhs.decision_value > rhs.decision_value + RESOURCE_TOLERANCE
            || lhs.resources.componentwise_strict(&rhs.resources))
}

/// Pareto frontier over affordable candidates only. Original input order is
/// preserved; candidates are not scalarized and incomparable options remain.
pub fn pareto_frontier<'a>(
    actions: &'a [CampaignAction],
    budget: &CampaignBudget,
) -> Result<Vec<&'a CampaignAction>, ResourceError> {
    let assessments = assess_budget(actions, budget)?;
    let affordable_ids: std::collections::BTreeSet<&str> = assessments
        .iter()
        .filter(|assessment| assessment.affordable)
        .map(|assessment| assessment.action_id.as_str())
        .collect();

    let affordable: Vec<&CampaignAction> = actions
        .iter()
        .filter(|action| affordable_ids.contains(action.id.as_str()))
        .collect();

    Ok(affordable
        .iter()
        .copied()
        .filter(|candidate| {
            !affordable
                .iter()
                .any(|other| other.id != candidate.id && dominates(other, candidate))
        })
        .collect())
}

/// Select maximum expected decision value among affordable actions.
/// Ties preserve input order. No implicit resource scalarization is performed.
pub fn select_max_decision_value<'a>(
    actions: &'a [CampaignAction],
    budget: &CampaignBudget,
) -> Result<&'a CampaignAction, ResourceError> {
    let assessments = assess_budget(actions, budget)?;
    let affordable_ids: std::collections::BTreeSet<&str> = assessments
        .iter()
        .filter(|assessment| assessment.affordable)
        .map(|assessment| assessment.action_id.as_str())
        .collect();

    actions
        .iter()
        .filter(|action| affordable_ids.contains(action.id.as_str()))
        .max_by(|left, right| {
            left.decision_value
                .partial_cmp(&right.decision_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or(ResourceError::NoAffordableActions)
}

impl ResourceShadowPrices {
    pub fn validate(&self) -> Result<(), ResourceError> {
        if !nonempty(&self.version) {
            return Err(ResourceError::EmptyIdentity("shadow_price_version"));
        }
        if !nonempty(&self.config_digest) {
            return Err(ResourceError::EmptyIdentity("shadow_price_config_digest"));
        }
        let values = [
            ("rig_hour", self.rig_hour),
            ("source_energy_wh", self.source_energy_wh),
            ("feedstock_g", self.feedstock_g),
            ("assay_cost", self.assay_cost),
            ("maintenance_minute", self.maintenance_minute),
            ("consumable_g", self.consumable_g),
        ];
        let mut any_positive = false;
        for (name, value) in values {
            if !value.is_finite() || value < 0.0 {
                return Err(ResourceError::InvalidShadowPrice(name));
            }
            any_positive |= value > 0.0;
        }
        if !any_positive {
            return Err(ResourceError::AllShadowPricesZero);
        }
        Ok(())
    }

    pub fn weighted_cost(&self, resources: &ResourceVector) -> Result<f64, ResourceError> {
        self.validate()?;
        resources.validate()?;
        let total = resources.rig_hours * self.rig_hour
            + resources.source_energy_wh * self.source_energy_wh
            + resources.feedstock_g * self.feedstock_g
            + resources.assay_cost * self.assay_cost
            + resources.maintenance_minutes * self.maintenance_minute
            + resources.consumables_g * self.consumable_g;
        if !total.is_finite() {
            return Err(ResourceError::InvalidDecisionValue);
        }
        Ok(total)
    }

    pub fn decision_value_density(
        &self,
        action: &CampaignAction,
    ) -> Result<f64, ResourceError> {
        let cost = self.weighted_cost(&action.resources)?;
        if cost <= 0.0 {
            return Err(ResourceError::ZeroWeightedCost(action.id.clone()));
        }
        Ok(action.decision_value / cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis() -> ResourceBasis {
        ResourceBasis {
            energy_meter_boundary: EnergyMeterBoundary::ElectricalInput,
            assay_cost_unit: "USD".to_string(),
        }
    }

    fn vector(rig: f64, energy: f64, feedstock: f64, assay: f64, maintenance: f64, consumables: f64) -> ResourceVector {
        ResourceVector {
            rig_hours: rig,
            source_energy_wh: energy,
            feedstock_g: feedstock,
            assay_cost: assay,
            maintenance_minutes: maintenance,
            consumables_g: consumables,
        }
    }

    fn budget() -> CampaignBudget {
        CampaignBudget {
            resources: vector(10.0, 1_000.0, 100.0, 500.0, 120.0, 20.0),
            basis: basis(),
        }
    }

    fn action(id: &str, value: f64, resources: ResourceVector) -> CampaignAction {
        CampaignAction {
            id: id.to_string(),
            decision_value: value,
            resources,
            basis: basis(),
        }
    }

    fn shadow_prices() -> ResourceShadowPrices {
        ResourceShadowPrices {
            version: "shadow-prices-v1".to_string(),
            config_digest: "sha256:shadow-price-config".to_string(),
            rig_hour: 10.0,
            source_energy_wh: 0.01,
            feedstock_g: 0.1,
            assay_cost: 1.0,
            maintenance_minute: 0.2,
            consumable_g: 0.5,
        }
    }

    #[test]
    fn hard_budget_rejects_single_resource_overrun_before_ranking() {
        let candidates = vec![
            action("feasible", 1.0, vector(1.0, 100.0, 1.0, 10.0, 10.0, 1.0)),
            action("too_much_energy", 100.0, vector(1.0, 1_001.0, 1.0, 10.0, 10.0, 1.0)),
        ];
        let result = assess_budget(&candidates, &budget()).unwrap();
        assert!(result[0].affordable);
        assert!(!result[1].affordable);
        assert_eq!(result[1].over_budget_resources, vec!["source_energy_wh"]);
        assert!(result[1].remaining_if_affordable.is_none());
    }

    #[test]
    fn remaining_budget_is_componentwise_and_never_negative() {
        let candidate = action("small", 1.0, vector(2.0, 100.0, 10.0, 20.0, 30.0, 2.0));
        let result = assess_budget(&[candidate], &budget()).unwrap();
        let remaining = result[0].remaining_if_affordable.unwrap();
        assert_eq!(remaining.rig_hours, 8.0);
        assert_eq!(remaining.source_energy_wh, 900.0);
        assert_eq!(remaining.maintenance_minutes, 90.0);
    }

    #[test]
    fn high_value_but_unaffordable_action_is_never_selected() {
        let candidates = vec![
            action("affordable", 2.0, vector(1.0, 100.0, 1.0, 10.0, 10.0, 1.0)),
            action("unaffordable", 200.0, vector(1.0, 1_001.0, 1.0, 10.0, 10.0, 1.0)),
        ];
        assert_eq!(
            select_max_decision_value(&candidates, &budget()).unwrap().id,
            "affordable"
        );
    }

    #[test]
    fn pareto_frontier_preserves_incomparable_resource_tradeoffs() {
        let candidates = vec![
            action("low_energy_more_time", 5.0, vector(8.0, 100.0, 5.0, 100.0, 20.0, 1.0)),
            action("high_energy_less_time", 5.0, vector(2.0, 800.0, 5.0, 100.0, 20.0, 1.0)),
            action("dominated", 4.0, vector(9.0, 900.0, 6.0, 120.0, 30.0, 2.0)),
        ];
        let frontier = pareto_frontier(&candidates, &budget()).unwrap();
        let ids: Vec<&str> = frontier.iter().map(|candidate| candidate.id.as_str()).collect();
        assert_eq!(ids, vec!["low_energy_more_time", "high_energy_less_time"]);
    }

    #[test]
    fn dominated_action_is_removed_when_another_is_better_and_uses_less() {
        let candidates = vec![
            action("efficient", 6.0, vector(2.0, 200.0, 3.0, 30.0, 5.0, 1.0)),
            action("dominated", 5.0, vector(3.0, 250.0, 4.0, 40.0, 6.0, 2.0)),
        ];
        let frontier = pareto_frontier(&candidates, &budget()).unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].id, "efficient");
    }

    #[test]
    fn scalar_cost_ranking_requires_explicit_shadow_prices() {
        let candidates = vec![
            action("low-energy", 4.0, vector(4.0, 100.0, 1.0, 10.0, 5.0, 1.0)),
            action("low-time", 4.0, vector(1.0, 700.0, 1.0, 10.0, 5.0, 1.0)),
        ];
        let prices = shadow_prices();
        let first = prices.decision_value_density(&candidates[0]).unwrap();
        let second = prices.decision_value_density(&candidates[1]).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn changed_shadow_prices_can_change_preference_without_changing_raw_costs() {
        let a = action("A", 5.0, vector(1.0, 900.0, 1.0, 1.0, 1.0, 1.0));
        let b = action("B", 5.0, vector(5.0, 100.0, 1.0, 1.0, 1.0, 1.0));
        let energy_valued = ResourceShadowPrices {
            rig_hour: 0.1,
            source_energy_wh: 1.0,
            ..shadow_prices()
        };
        let time_valued = ResourceShadowPrices {
            rig_hour: 10.0,
            source_energy_wh: 0.001,
            ..shadow_prices()
        };
        assert!(energy_valued.decision_value_density(&b).unwrap()
            > energy_valued.decision_value_density(&a).unwrap());
        assert!(time_valued.decision_value_density(&a).unwrap()
            > time_valued.decision_value_density(&b).unwrap());
    }

    #[test]
    fn energy_meter_boundary_mismatch_is_rejected() {
        let mut candidate = action("wrong_boundary", 1.0, vector(1.0, 10.0, 1.0, 1.0, 1.0, 1.0));
        candidate.basis.energy_meter_boundary = EnergyMeterBoundary::OpticalSourceOutput;
        assert_eq!(assess_budget(&[candidate], &budget()), Err(ResourceError::BasisMismatch));
    }

    #[test]
    fn invalid_resource_and_shadow_price_values_are_rejected() {
        let bad = action("negative", 1.0, vector(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
        assert_eq!(assess_budget(&[bad], &budget()), Err(ResourceError::InvalidResource("rig_hours")));

        let mut prices = shadow_prices();
        prices.source_energy_wh = f64::NAN;
        assert_eq!(prices.validate(), Err(ResourceError::InvalidShadowPrice("source_energy_wh")));
    }

    #[test]
    fn zero_weighted_cost_and_all_zero_shadow_prices_are_rejected() {
        let free = action("free", 1.0, vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(shadow_prices().decision_value_density(&free), Err(ResourceError::ZeroWeightedCost("free".to_string())));

        let mut prices = shadow_prices();
        prices.rig_hour = 0.0;
        prices.source_energy_wh = 0.0;
        prices.feedstock_g = 0.0;
        prices.assay_cost = 0.0;
        prices.maintenance_minute = 0.0;
        prices.consumable_g = 0.0;
        assert_eq!(prices.validate(), Err(ResourceError::AllShadowPricesZero));
    }

    #[test]
    fn no_affordable_action_is_an_explicit_result() {
        let candidate = action("unaffordable", 10.0, vector(100.0, 10.0, 1.0, 1.0, 1.0, 1.0));
        assert_eq!(select_max_decision_value(&[candidate], &budget()), Err(ResourceError::NoAffordableActions));
    }
}
