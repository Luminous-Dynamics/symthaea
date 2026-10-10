// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-linked measurement prioritization for regenerative design screening.
//!
//! This module does not run a statistical design-of-experiments algorithm or estimate
//! causal effects. It ranks supplied measurement options only when they target a currently
//! unresolved numeric constraint, using explicit domain weights, estimated interval-width
//! reduction, and same-unit cost. The score is a transparent heuristic, not calibrated
//! expected value of information or a guarantee that the measurement will resolve a decision.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

use symthaea_agribot::soil_process::EvidenceRef;

use crate::regenerative::{
    assess_regenerative_uncertainty, IntervalConstraintStatus, MetricInterval,
    RegenerativeDesignRequirements, RegenerativeMetricIntervals,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementMetric {
    CharYield,
    CarbonRetention,
    SuppliedHeat,
    Cost,
    WaterUse,
    NetClimate,
}

impl MeasurementMetric {
    fn constraint_id(self) -> &'static str {
        match self {
            Self::CharYield => "char_yield_minimum",
            Self::CarbonRetention => "carbon_retention_minimum",
            Self::SuppliedHeat => "supplied_heat_maximum",
            Self::Cost => "cost_maximum",
            Self::WaterUse => "water_use_maximum",
            Self::NetClimate => "climate_objective_interval_available",
        }
    }

    fn interval<'a>(self, intervals: &'a RegenerativeMetricIntervals) -> Option<&'a MetricInterval> {
        match self {
            Self::CharYield => Some(&intervals.char_yield_fraction_dry_basis),
            Self::CarbonRetention => Some(&intervals.carbon_retained_fraction),
            Self::SuppliedHeat => Some(&intervals.supplied_heat_mj_per_kg_dry_feedstock),
            Self::Cost => Some(&intervals.cost_per_kg_dry_feedstock),
            Self::WaterUse => Some(&intervals.water_l_per_kg_dry_feedstock),
            Self::NetClimate => intervals.net_climate_kg_co2e_per_kg_dry_feedstock.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementBenefitBasis {
    /// The metric already has a range, and the activity is expected to narrow it.
    IntervalWidthReduction,
    /// Required metric bounds are absent, and the activity primarily acquires baseline evidence.
    MissingRequiredDataAcquisition,
}

/// A possible measurement or calibration activity. All option costs must use the same
/// cost_unit within one request. Benefit fraction and priority weight are explicit
/// estimates/judgments with provenance; they are not inferred by this planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementOption {
    pub option_id: String,
    pub metric: MeasurementMetric,
    /// Lab method, measurement procedure, instrument calibration, or named experiment.
    pub method_or_experiment_id: String,
    pub estimated_cost: f64,
    pub cost_unit: String,
    /// Declared expected benefit fraction in (0, 1]; meaning depends on benefit_basis.
    pub benefit_basis: MeasurementBenefitBasis,
    pub expected_benefit_fraction: f64,
    /// Caller-supplied decision relevance weight in [0, 1], chosen by the study owner.
    pub decision_relevance_weight: f64,
    pub cost_evidence: EvidenceRef,
    pub expected_reduction_evidence: EvidenceRef,
    pub relevance_weight_evidence: EvidenceRef,
}

impl MeasurementOption {
    fn validate(&self, shared_cost_unit: &str) -> Result<(), MeasurementPlannerError> {
        if self.option_id.trim().is_empty() {
            return Err(MeasurementPlannerError::new("option_id", "cannot be empty"));
        }
        if self.method_or_experiment_id.trim().is_empty() {
            return Err(MeasurementPlannerError::new(
                "method_or_experiment_id",
                "must identify the proposed measurement or experiment",
            ));
        }
        if self.cost_unit.trim().is_empty() || self.cost_unit != shared_cost_unit {
            return Err(MeasurementPlannerError::new(
                "cost_unit",
                "all measurement options must use the identical declared cost unit",
            ));
        }
        if !self.estimated_cost.is_finite() || self.estimated_cost <= 0.0 {
            return Err(MeasurementPlannerError::new(
                "estimated_cost",
                "must be finite and greater than zero",
            ));
        }
        if !self.expected_benefit_fraction.is_finite()
            || self.expected_benefit_fraction <= 0.0
            || self.expected_benefit_fraction > 1.0
        {
            return Err(MeasurementPlannerError::new(
                "expected_benefit_fraction",
                "must be finite and in the interval (0, 1]",
            ));
        }
        if !self.decision_relevance_weight.is_finite()
            || !(0.0..=1.0).contains(&self.decision_relevance_weight)
        {
            return Err(MeasurementPlannerError::new(
                "decision_relevance_weight",
                "must be finite and in [0, 1]",
            ));
        }
        for (field, evidence) in [
            ("cost_evidence", &self.cost_evidence),
            ("expected_reduction_evidence", &self.expected_reduction_evidence),
            ("relevance_weight_evidence", &self.relevance_weight_evidence),
        ] {
            if evidence.evidence_id.trim().is_empty() {
                return Err(MeasurementPlannerError::new(
                    field,
                    "requires a non-empty evidence ID",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementPlannerError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl MeasurementPlannerError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl std::fmt::Display for MeasurementPlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl std::error::Error for MeasurementPlannerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementPlanStatus {
    /// At least one option targets a currently unresolved requirement.
    PrioritiesAvailable,
    /// The supplied intervals resolve every hard numeric threshold.
    NoUnresolvedNumericConstraints,
    /// Constraints remain unresolved, but none of the supplied options targets them.
    NoApplicableMeasurementOptions,
    /// At least one option targets an unresolved constraint, but every such option has zero weight.
    ApplicableOptionsHaveZeroPriorityWeight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementPriority {
    pub rank: Option<u32>,
    pub option_id: String,
    pub metric: MeasurementMetric,
    pub method_or_experiment_id: String,
    pub target_constraint_id: String,
    pub current_constraint_status: Option<IntervalConstraintStatus>,
    pub currently_decision_relevant: bool,
    /// Relevance × declared expected benefit fraction / estimated cost.
    /// Relative heuristic only; not a probability, confidence value, or calibrated VOI.
    pub heuristic_score: Option<f64>,
    pub estimated_cost: f64,
    pub cost_unit: String,
    pub benefit_basis: MeasurementBenefitBasis,
    pub expected_benefit_fraction: f64,
    pub decision_relevance_weight: f64,
    pub cost_evidence: EvidenceRef,
    pub expected_reduction_evidence: EvidenceRef,
    pub relevance_weight_evidence: EvidenceRef,
    pub disposition_note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementPriorityPlan {
    pub candidate_id: String,
    pub specification_id: String,
    pub status: MeasurementPlanStatus,
    pub cost_unit: String,
    pub unresolved_constraint_ids: Vec<String>,
    /// Deterministically ordered by descending heuristic score then option ID.
    /// Options which do not target unresolved constraints are retained without a rank.
    pub options: Vec<MeasurementPriority>,
    pub scope_note: String,
}

/// Produce a deterministic list of potentially useful measurements for an indeterminate
/// regenerative-design screen. Only options mapped to a constraint that is currently
/// unresolved are ranked. The ordering score is:
///
/// decision_relevance_weight * expected_benefit_fraction / estimated_cost.
///
/// All costs must use the same explicit unit and price basis. The result is a triage aid,
/// not an optimal experiment plan: it does not model dependencies/correlation, measurement
/// variance, statistical power, treatment randomization, causal identification, or actual
/// probability of changing a decision. Safety/product-quality questions remain separate gates.
pub fn prioritize_measurements(
    intervals: &RegenerativeMetricIntervals,
    requirements: &RegenerativeDesignRequirements,
    cost_unit: &str,
    options: &[MeasurementOption],
) -> Result<MeasurementPriorityPlan, MeasurementPlannerError> {
    if cost_unit.trim().is_empty() {
        return Err(MeasurementPlannerError::new(
            "cost_unit",
            "the shared measurement-cost unit must be explicit",
        ));
    }

    let assessment = assess_regenerative_uncertainty(intervals, requirements)
        .map_err(|error| MeasurementPlannerError::new(error.field, error.reason))?;

    let unresolved_constraint_ids: Vec<String> = assessment
        .constraints
        .iter()
        .filter(|constraint| constraint.status == IntervalConstraintStatus::Unresolved)
        .map(|constraint| constraint.constraint_id.clone())
        .collect();

    let mut ids = HashSet::new();
    let mut priorities = Vec::with_capacity(options.len());
    for option in options {
        option.validate(cost_unit)?;
        if !ids.insert(option.option_id.as_str()) {
            return Err(MeasurementPlannerError::new(
                "option_id",
                "measurement option IDs must be unique",
            ));
        }

        let target_constraint_id = if option.metric == MeasurementMetric::NetClimate
            && assessment
                .constraints
                .iter()
                .any(|constraint| constraint.constraint_id == "net_climate_maximum")
        {
            "net_climate_maximum"
        } else {
            option.metric.constraint_id()
        };
        let target_constraint = assessment
            .constraints
            .iter()
            .find(|constraint| constraint.constraint_id == target_constraint_id);
        let relevant_status = target_constraint.map(|constraint| constraint.status);
        let metric_interval_available = option.metric.interval(intervals).is_some();
        let is_missing_required_climate = option.metric == MeasurementMetric::NetClimate
            && relevant_status == Some(IntervalConstraintStatus::Unresolved)
            && target_constraint_id == "climate_objective_interval_available";
        let relevant = relevant_status == Some(IntervalConstraintStatus::Unresolved)
            && (metric_interval_available || is_missing_required_climate);

        if relevant {
            let expected_basis = if is_missing_required_climate {
                MeasurementBenefitBasis::MissingRequiredDataAcquisition
            } else {
                MeasurementBenefitBasis::IntervalWidthReduction
            };
            if option.benefit_basis != expected_basis {
                return Err(MeasurementPlannerError::new(
                    "benefit_basis",
                    "benefit basis must match whether the option narrows an existing interval or acquires missing required data",
                ));
            }
        }

        let score = if relevant {
            let value = option.decision_relevance_weight
                * option.expected_benefit_fraction
                / option.estimated_cost;
            if !value.is_finite() {
                return Err(MeasurementPlannerError::new(
                    "heuristic_score",
                    "score overflow; inspect weights and cost units",
                ));
            }
            (value > 0.0).then_some(value)
        } else {
            None
        };

        let disposition_note = if relevant && score.is_some() && is_missing_required_climate {
            "Targets missing required climate data; ranking uses a declared acquisition-benefit fraction and is only a triage heuristic.".to_string()
        } else if relevant && score.is_some() {
            "Targets an unresolved numeric constraint; ranking uses declared heuristic inputs only.".to_string()
        } else if relevant {
            "Constraint is unresolved, but the declared decision relevance weight is zero.".to_string()
        } else if relevant_status.is_none() {
            "No matching constraint is active in this specification; option is not ranked.".to_string()
        } else if relevant_status == Some(IntervalConstraintStatus::Unresolved)
            && option.metric == MeasurementMetric::NetClimate
            && option.metric.interval(intervals).is_none()
        {
            "Required climate interval is missing; the proposed assay can supply evidence, but no numeric interval is available yet.".to_string()
        } else {
            "Does not target a currently unresolved constraint; retained for audit but not ranked.".to_string()
        };

        priorities.push(MeasurementPriority {
            rank: None,
            option_id: option.option_id.clone(),
            metric: option.metric,
            method_or_experiment_id: option.method_or_experiment_id.clone(),
            target_constraint_id: target_constraint_id.to_string(),
            current_constraint_status: relevant_status,
            currently_decision_relevant: relevant,
            heuristic_score: score,
            estimated_cost: option.estimated_cost,
            cost_unit: option.cost_unit.clone(),
            benefit_basis: option.benefit_basis,
            expected_benefit_fraction: option.expected_benefit_fraction,
            decision_relevance_weight: option.decision_relevance_weight,
            cost_evidence: option.cost_evidence.clone(),
            expected_reduction_evidence: option.expected_reduction_evidence.clone(),
            relevance_weight_evidence: option.relevance_weight_evidence.clone(),
            disposition_note,
        });
    }

    priorities.sort_by(|a, b| match (a.heuristic_score, b.heuristic_score) {
        (Some(a_score), Some(b_score)) => b_score
            .partial_cmp(&a_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.option_id.cmp(&b.option_id)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => a.option_id.cmp(&b.option_id),
    });

    let mut rank = 1_u32;
    for item in &mut priorities {
        if item.heuristic_score.is_some() {
            item.rank = Some(rank);
            rank = rank.saturating_add(1);
        }
    }

    let ranked_count = priorities.iter().filter(|item| item.rank.is_some()).count();
    let has_decision_relevant_option = priorities
        .iter()
        .any(|item| item.currently_decision_relevant);
    let status = if ranked_count > 0 {
        MeasurementPlanStatus::PrioritiesAvailable
    } else if unresolved_constraint_ids.is_empty() {
        MeasurementPlanStatus::NoUnresolvedNumericConstraints
    } else if has_decision_relevant_option {
        MeasurementPlanStatus::ApplicableOptionsHaveZeroPriorityWeight
    } else {
        MeasurementPlanStatus::NoApplicableMeasurementOptions
    };

    Ok(MeasurementPriorityPlan {
        candidate_id: intervals.candidate_id.clone(),
        specification_id: requirements.specification_id.clone(),
        status,
        cost_unit: cost_unit.to_string(),
        unresolved_constraint_ids,
        options: priorities,
        scope_note: "measurement triage heuristic only; not a calibrated value-of-information estimate, statistical design-of-experiments plan, safety approval, or agronomic recommendation".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regenerative::{MetricInterval, RegenerativeMetricIntervals};
    use symthaea_agribot::soil_process::EvidenceKind;

    fn evidence(id: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind: EvidenceKind::Scenario,
        }
    }

    fn metric_interval(lower: f64, upper: f64, id: &str) -> MetricInterval {
        MetricInterval {
            lower,
            upper,
            evidence: evidence(id),
        }
    }

    fn requirements() -> RegenerativeDesignRequirements {
        RegenerativeDesignRequirements {
            specification_id: "design-spec-v1".into(),
            cost_unit: "USD_2026_per_kg_dry".into(),
            min_char_yield_fraction: 0.20,
            min_carbon_retained_fraction: 0.40,
            max_supplied_heat_mj_per_kg_dry_feedstock: 8.0,
            max_cost_per_kg_dry_feedstock: 10.0,
            max_water_l_per_kg_dry_feedstock: 8.0,
            include_climate_objective: false,
            max_net_climate_kg_co2e_per_kg_dry_feedstock: None,
        }
    }

    fn intervals() -> RegenerativeMetricIntervals {
        RegenerativeMetricIntervals {
            candidate_id: "candidate-v1".into(),
            char_yield_fraction_dry_basis: metric_interval(0.15, 0.27, "yield-range"),
            carbon_retained_fraction: metric_interval(0.45, 0.60, "carbon-range"),
            supplied_heat_mj_per_kg_dry_feedstock: metric_interval(3.0, 5.0, "heat-range"),
            cost_per_kg_dry_feedstock: metric_interval(2.0, 4.0, "cost-range"),
            water_l_per_kg_dry_feedstock: metric_interval(1.0, 2.0, "water-range"),
            net_climate_kg_co2e_per_kg_dry_feedstock: None,
        }
    }

    fn option(
        id: &str,
        metric: MeasurementMetric,
        cost: f64,
        reduction: f64,
        weight: f64,
    ) -> MeasurementOption {
        MeasurementOption {
            option_id: id.into(),
            metric,
            method_or_experiment_id: format!("method-{id}-v1"),
            estimated_cost: cost,
            cost_unit: "USD_2026_per_measurement".into(),
            benefit_basis: MeasurementBenefitBasis::IntervalWidthReduction,
            expected_benefit_fraction: reduction,
            decision_relevance_weight: weight,
            cost_evidence: evidence(&format!("{id}-cost")),
            expected_reduction_evidence: evidence(&format!("{id}-reduction")),
            relevance_weight_evidence: evidence(&format!("{id}-weight")),
        }
    }

    #[test]
    fn only_measurements_targeting_unresolved_constraints_are_ranked() {
        let options = vec![
            option("yield-lab", MeasurementMetric::CharYield, 2.0, 0.5, 1.0),
            option("heat-meter", MeasurementMetric::SuppliedHeat, 1.0, 0.8, 0.8),
            option("carbon-lab", MeasurementMetric::CarbonRetention, 1.0, 0.9, 1.0),
        ];
        let plan = prioritize_measurements(
            &intervals(),
            &requirements(),
            "USD_2026_per_measurement",
            &options,
        ).unwrap();

        assert_eq!(plan.status, MeasurementPlanStatus::PrioritiesAvailable);
        assert_eq!(plan.unresolved_constraint_ids, vec!["char_yield_minimum"]);
        assert_eq!(plan.options[0].option_id, "yield-lab");
        assert_eq!(plan.options[0].rank, Some(1));
        assert!(plan.options[0].currently_decision_relevant);
        assert_eq!(plan.options[1].option_id, "carbon-lab");
        assert_eq!(plan.options[1].rank, None);
        assert_eq!(plan.options[2].option_id, "heat-meter");
        assert_eq!(plan.options[2].rank, None);
    }

    #[test]
    fn score_uses_explicit_weight_reduction_and_same_unit_cost() {
        let options = vec![
            option("expensive", MeasurementMetric::CharYield, 10.0, 0.8, 1.0),
            option("cheap", MeasurementMetric::CharYield, 2.0, 0.5, 1.0),
        ];
        let plan = prioritize_measurements(
            &intervals(), &requirements(), "USD_2026_per_measurement", &options,
        ).unwrap();
        assert_eq!(plan.options[0].option_id, "cheap");
        assert!((plan.options[0].heuristic_score.unwrap() - 0.25).abs() < 1e-12);
        assert!((plan.options[1].heuristic_score.unwrap() - 0.08).abs() < 1e-12);
    }

    #[test]
    fn missing_required_climate_can_be_targeted_for_data_acquisition() {
        let mut req = requirements();
        req.include_climate_objective = true;
        let mut climate_option = option("climate-inventory", MeasurementMetric::NetClimate, 8.0, 0.6, 1.0);
        climate_option.benefit_basis = MeasurementBenefitBasis::MissingRequiredDataAcquisition;
        let options = vec![climate_option];
        let plan = prioritize_measurements(
            &intervals(), &req, "USD_2026_per_measurement", &options,
        ).unwrap();
        assert_eq!(plan.status, MeasurementPlanStatus::PrioritiesAvailable);
        assert_eq!(plan.options[0].target_constraint_id, "climate_objective_interval_available");
        assert_eq!(plan.options[0].rank, Some(1));
        assert!(plan.options[0].currently_decision_relevant);
    }

    #[test]
    fn climate_interval_reduction_and_missing_data_acquisition_are_distinct() {
        let mut req = requirements();
        req.include_climate_objective = true;
        let mut input = intervals();
        let options = vec![option("climate-inventory", MeasurementMetric::NetClimate, 8.0, 0.6, 1.0)];
        assert!(prioritize_measurements(
            &input, &req, "USD_2026_per_measurement", &options,
        ).is_err());

        input.net_climate_kg_co2e_per_kg_dry_feedstock =
            Some(metric_interval(-1.0, 1.0, "climate-range"));
        assert!(prioritize_measurements(
            &input, &req, "USD_2026_per_measurement", &options,
        ).is_err());
    }

    #[test]
    fn climate_measurement_targets_an_active_climate_threshold() {
        let mut req = requirements();
        req.include_climate_objective = true;
        req.max_net_climate_kg_co2e_per_kg_dry_feedstock = Some(0.0);
        let mut input = intervals();
        input.net_climate_kg_co2e_per_kg_dry_feedstock =
            Some(metric_interval(-1.0, 1.0, "climate-range"));
        let options = vec![option("climate-assay", MeasurementMetric::NetClimate, 3.0, 0.5, 1.0)];
        let plan = prioritize_measurements(
            &input, &req, "USD_2026_per_measurement", &options,
        ).unwrap();
        assert_eq!(plan.options[0].target_constraint_id, "net_climate_maximum");
        assert_eq!(plan.options[0].current_constraint_status, Some(IntervalConstraintStatus::Unresolved));
        assert_eq!(plan.options[0].rank, Some(1));
    }

    #[test]
    fn no_unresolved_constraints_does_not_manufacture_priorities() {
        let mut input = intervals();
        input.char_yield_fraction_dry_basis = metric_interval(0.25, 0.30, "yield-passing");
        let options = vec![option("yield-lab", MeasurementMetric::CharYield, 2.0, 0.5, 1.0)];
        let plan = prioritize_measurements(
            &input, &requirements(), "USD_2026_per_measurement", &options,
        ).unwrap();
        assert_eq!(plan.status, MeasurementPlanStatus::NoUnresolvedNumericConstraints);
        assert_eq!(plan.options[0].rank, None);
    }

    #[test]
    fn invalid_cost_evidence_options_and_duplicate_ids_fail_closed() {
        let mut options = vec![option("yield-lab", MeasurementMetric::CharYield, 2.0, 0.5, 1.0)];
        options[0].cost_unit = "USD_per_hour".into();
        assert!(prioritize_measurements(
            &intervals(), &requirements(), "USD_2026_per_measurement", &options,
        ).is_err());

        options[0] = option("same", MeasurementMetric::CharYield, 2.0, 0.5, 1.0);
        options.push(option("same", MeasurementMetric::CharYield, 3.0, 0.7, 1.0));
        assert!(prioritize_measurements(
            &intervals(), &requirements(), "USD_2026_per_measurement", &options,
        ).is_err());

        options = vec![option("zero-cost", MeasurementMetric::CharYield, 0.0, 0.5, 1.0)];
        assert!(prioritize_measurements(
            &intervals(), &requirements(), "USD_2026_per_measurement", &options,
        ).is_err());
    }

    #[test]
    fn zero_weight_is_not_reported_as_missing_measurement_coverage() {
        let options = vec![option("yield-lab", MeasurementMetric::CharYield, 2.0, 0.5, 0.0)];
        let plan = prioritize_measurements(
            &intervals(), &requirements(), "USD_2026_per_measurement", &options,
        ).unwrap();
        assert_eq!(plan.status, MeasurementPlanStatus::ApplicableOptionsHaveZeroPriorityWeight);
        assert!(plan.options[0].currently_decision_relevant);
        assert_eq!(plan.options[0].rank, None);
    }
}
