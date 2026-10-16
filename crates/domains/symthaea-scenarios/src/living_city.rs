//! Constraint-first assessment of urban-design scenarios.
//!
//! This module is an evaluation boundary, not an urban physics model. It keeps
//! locally ratified service/equity floors separate from Pareto comparison so
//! excellent emissions or cost figures cannot compensate for failing an
//! essential-needs floor. Inputs must come from a versioned evidence package;
//! do not replace missing observations with zeroes.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Per-resident outcomes for a fixed scenario, boundary, time horizon, and
/// evidence snapshot. Ratios are in [0, 1]; non-ratio quantities use the units
/// embedded in their field names. This struct deliberately carries no weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CityMetrics {
    /// Share of residents with accessible routes to locally designated essentials.
    pub essential_access_share: f64,
    /// 90th-percentile housing cost as a share of household income; may exceed 1.
    pub housing_cost_burden_p90: f64,
    /// Annual uptime of critical services, as a ratio.
    pub critical_service_uptime: f64,
    /// Share of modeled demand reliably served by safe water.
    pub water_service_reliability: f64,
    /// Share of residents with safe, accessible cooling/refuge options in the
    /// selected heat scenario.
    pub heat_safe_access_share: f64,
    /// Lifecycle emissions, kg CO2e per resident per year.
    pub lifecycle_emissions_kg_co2e_per_person_year: f64,
    /// Water withdrawal, litres per resident per day. Context-specific scarcity
    /// limits belong in the locally configured constraint set.
    pub water_withdrawal_l_per_person_day: f64,
    /// Habitat-connectivity proxy in [0, 1], with methodology declared externally.
    pub habitat_connectivity_index: f64,
    /// Modeled risk of involuntary displacement attributable to the scenario.
    pub involuntary_displacement_risk: f64,
    /// Lifecycle cost, currency units per resident per year. Compare only when
    /// currency, price year, discount rate, boundary, and lifetime are aligned.
    pub lifecycle_cost_per_person_year: f64,
}

impl CityMetrics {
    pub fn validate(&self) -> Result<(), MetricValidationError> {
        validate_ratio("essential_access_share", self.essential_access_share)?;
        validate_non_negative("housing_cost_burden_p90", self.housing_cost_burden_p90)?;
        validate_ratio("critical_service_uptime", self.critical_service_uptime)?;
        validate_ratio("water_service_reliability", self.water_service_reliability)?;
        validate_ratio("heat_safe_access_share", self.heat_safe_access_share)?;
        validate_non_negative(
            "lifecycle_emissions_kg_co2e_per_person_year",
            self.lifecycle_emissions_kg_co2e_per_person_year,
        )?;
        validate_non_negative(
            "water_withdrawal_l_per_person_day",
            self.water_withdrawal_l_per_person_day,
        )?;
        validate_ratio("habitat_connectivity_index", self.habitat_connectivity_index)?;
        validate_ratio(
            "involuntary_displacement_risk",
            self.involuntary_displacement_risk,
        )?;
        validate_non_negative(
            "lifecycle_cost_per_person_year",
            self.lifecycle_cost_per_person_year,
        )?;
        Ok(())
    }
}

/// Non-negotiable floors for one place and decision context. These values must
/// be set and justified by the responsible community and domain experts; this
/// library intentionally defines no universal default thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CityFloors {
    pub minimum_essential_access_share: f64,
    pub maximum_housing_cost_burden_p90: f64,
    pub minimum_critical_service_uptime: f64,
    pub minimum_water_service_reliability: f64,
    pub minimum_heat_safe_access_share: f64,
    pub maximum_involuntary_displacement_risk: f64,
}

impl CityFloors {
    pub fn validate(&self) -> Result<(), MetricValidationError> {
        validate_ratio("minimum_essential_access_share", self.minimum_essential_access_share)?;
        validate_non_negative(
            "maximum_housing_cost_burden_p90",
            self.maximum_housing_cost_burden_p90,
        )?;
        validate_ratio(
            "minimum_critical_service_uptime",
            self.minimum_critical_service_uptime,
        )?;
        validate_ratio(
            "minimum_water_service_reliability",
            self.minimum_water_service_reliability,
        )?;
        validate_ratio(
            "minimum_heat_safe_access_share",
            self.minimum_heat_safe_access_share,
        )?;
        validate_ratio(
            "maximum_involuntary_displacement_risk",
            self.maximum_involuntary_displacement_risk,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintBound {
    AtLeast,
    AtMost,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FloorViolation {
    pub metric: &'static str,
    pub actual: f64,
    pub bound: ConstraintBound,
    pub threshold: f64,
}

/// Constructed only by assess_scenario, so the feasibility result and its
/// violation list cannot drift apart through public-field mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioAssessment {
    scenario_id: String,
    metrics: CityMetrics,
    violations: Vec<FloorViolation>,
}

impl ScenarioAssessment {
    pub fn scenario_id(&self) -> &str {
        &self.scenario_id
    }

    pub fn metrics(&self) -> CityMetrics {
        self.metrics
    }

    pub fn violations(&self) -> &[FloorViolation] {
        &self.violations
    }

    pub fn is_feasible(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Validation error for a non-finite, negative, or out-of-range input.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricValidationError {
    pub field: &'static str,
    pub value: f64,
    pub reason: &'static str,
}

impl fmt::Display for MetricValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid city metric/floor {}={} ({}); assessment refused",
            self.field, self.value, self.reason
        )
    }
}

impl std::error::Error for MetricValidationError {}

fn validate_ratio(field: &'static str, value: f64) -> Result<(), MetricValidationError> {
    if !value.is_finite() {
        return Err(MetricValidationError {
            field,
            value,
            reason: "must be finite",
        });
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(MetricValidationError {
            field,
            value,
            reason: "must be in [0, 1]",
        });
    }
    Ok(())
}

fn validate_non_negative(
    field: &'static str,
    value: f64,
) -> Result<(), MetricValidationError> {
    if !value.is_finite() {
        return Err(MetricValidationError {
            field,
            value,
            reason: "must be finite",
        });
    }
    if value < 0.0 {
        return Err(MetricValidationError {
            field,
            value,
            reason: "must be non-negative",
        });
    }
    Ok(())
}

/// Assess one scenario against explicit, place-specific minimum floors.
/// Passing this function means only that the supplied values satisfy the
/// configured constraints; it is not a prediction or certification of a city.
pub fn assess_scenario(
    scenario_id: impl Into<String>,
    metrics: CityMetrics,
    floors: CityFloors,
) -> Result<ScenarioAssessment, MetricValidationError> {
    metrics.validate()?;
    floors.validate()?;

    let mut violations = Vec::new();
    push_at_least(
        &mut violations,
        "essential_access_share",
        metrics.essential_access_share,
        floors.minimum_essential_access_share,
    );
    push_at_most(
        &mut violations,
        "housing_cost_burden_p90",
        metrics.housing_cost_burden_p90,
        floors.maximum_housing_cost_burden_p90,
    );
    push_at_least(
        &mut violations,
        "critical_service_uptime",
        metrics.critical_service_uptime,
        floors.minimum_critical_service_uptime,
    );
    push_at_least(
        &mut violations,
        "water_service_reliability",
        metrics.water_service_reliability,
        floors.minimum_water_service_reliability,
    );
    push_at_least(
        &mut violations,
        "heat_safe_access_share",
        metrics.heat_safe_access_share,
        floors.minimum_heat_safe_access_share,
    );
    push_at_most(
        &mut violations,
        "involuntary_displacement_risk",
        metrics.involuntary_displacement_risk,
        floors.maximum_involuntary_displacement_risk,
    );

    Ok(ScenarioAssessment {
        scenario_id: scenario_id.into(),
        metrics,
        violations,
    })
}

fn push_at_least(
    violations: &mut Vec<FloorViolation>,
    metric: &'static str,
    actual: f64,
    threshold: f64,
) {
    if actual < threshold {
        violations.push(FloorViolation {
            metric,
            actual,
            bound: ConstraintBound::AtLeast,
            threshold,
        });
    }
}

fn push_at_most(
    violations: &mut Vec<FloorViolation>,
    metric: &'static str,
    actual: f64,
    threshold: f64,
) {
    if actual > threshold {
        violations.push(FloorViolation {
            metric,
            actual,
            bound: ConstraintBound::AtMost,
            threshold,
        });
    }
}

/// Comparison is constraint-first, then Pareto. Infeasible scenarios are not
/// ranked against each other by a weighted score; both stay explicitly marked
/// infeasible until their floor violations are repaired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioRelation {
    AFeasibleBInfeasible,
    AInfeasibleBFeasible,
    BothInfeasible,
    ADominates,
    BDominates,
    TradeOff,
    Equivalent,
}

pub fn compare_scenarios(
    a: &ScenarioAssessment,
    b: &ScenarioAssessment,
) -> ScenarioRelation {
    match (a.is_feasible(), b.is_feasible()) {
        (true, false) => return ScenarioRelation::AFeasibleBInfeasible,
        (false, true) => return ScenarioRelation::AInfeasibleBFeasible,
        (false, false) => return ScenarioRelation::BothInfeasible,
        (true, true) => {}
    }

    let am = a.metrics;
    let bm = b.metrics;
    let objectives = [
        (am.essential_access_share, bm.essential_access_share, true),
        (am.housing_cost_burden_p90, bm.housing_cost_burden_p90, false),
        (am.critical_service_uptime, bm.critical_service_uptime, true),
        (am.water_service_reliability, bm.water_service_reliability, true),
        (am.heat_safe_access_share, bm.heat_safe_access_share, true),
        (
            am.lifecycle_emissions_kg_co2e_per_person_year,
            bm.lifecycle_emissions_kg_co2e_per_person_year,
            false,
        ),
        (
            am.water_withdrawal_l_per_person_day,
            bm.water_withdrawal_l_per_person_day,
            false,
        ),
        (am.habitat_connectivity_index, bm.habitat_connectivity_index, true),
        (
            am.involuntary_displacement_risk,
            bm.involuntary_displacement_risk,
            false,
        ),
        (
            am.lifecycle_cost_per_person_year,
            bm.lifecycle_cost_per_person_year,
            false,
        ),
    ];

    let mut a_better = false;
    let mut b_better = false;
    for (av, bv, higher_is_better) in objectives {
        if av == bv {
            continue;
        }
        let a_is_better = if higher_is_better { av > bv } else { av < bv };
        if a_is_better {
            a_better = true;
        } else {
            b_better = true;
        }
    }

    match (a_better, b_better) {
        (true, false) => ScenarioRelation::ADominates,
        (false, true) => ScenarioRelation::BDominates,
        (true, true) => ScenarioRelation::TradeOff,
        (false, false) => ScenarioRelation::Equivalent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These are synthetic fixture values only, not measured data or universal
    // city standards. A real pilot must publish its evidence manifest and floor
    // ratification record alongside each assessment.
    fn metrics() -> CityMetrics {
        CityMetrics {
            essential_access_share: 0.94,
            housing_cost_burden_p90: 0.29,
            critical_service_uptime: 0.9995,
            water_service_reliability: 0.997,
            heat_safe_access_share: 0.96,
            lifecycle_emissions_kg_co2e_per_person_year: 850.0,
            water_withdrawal_l_per_person_day: 145.0,
            habitat_connectivity_index: 0.64,
            involuntary_displacement_risk: 0.02,
            lifecycle_cost_per_person_year: 1800.0,
        }
    }

    fn floors() -> CityFloors {
        CityFloors {
            minimum_essential_access_share: 0.90,
            maximum_housing_cost_burden_p90: 0.30,
            minimum_critical_service_uptime: 0.999,
            minimum_water_service_reliability: 0.995,
            minimum_heat_safe_access_share: 0.95,
            maximum_involuntary_displacement_risk: 0.05,
        }
    }

    #[test]
    fn synthetic_scenario_can_meet_locally_configured_floors() {
        let assessed = assess_scenario("fixture-good", metrics(), floors()).unwrap();
        assert!(assessed.is_feasible());
        assert!(assessed.violations().is_empty());
    }

    #[test]
    fn carbon_savings_cannot_compensate_for_failed_access_floor() {
        let good = assess_scenario("good", metrics(), floors()).unwrap();
        let bad_metrics = CityMetrics {
            essential_access_share: 0.40,
            lifecycle_emissions_kg_co2e_per_person_year: 10.0,
            ..metrics()
        };
        let bad = assess_scenario("low-carbon-but-excluding", bad_metrics, floors()).unwrap();

        assert!(!bad.is_feasible());
        assert!(bad.violations().iter().any(|v| v.metric == "essential_access_share"));
        assert_eq!(
            compare_scenarios(&good, &bad),
            ScenarioRelation::AFeasibleBInfeasible
        );
    }

    #[test]
    fn feasible_tradeoffs_are_not_hidden_in_a_weighted_score() {
        let a = assess_scenario("low-carbon", metrics(), floors()).unwrap();
        let alternative = CityMetrics {
            lifecycle_emissions_kg_co2e_per_person_year: 600.0,
            housing_cost_burden_p90: 0.30,
            habitat_connectivity_index: 0.55,
            ..metrics()
        };
        let b = assess_scenario("lower-emissions-higher-housing-burden", alternative, floors())
            .unwrap();
        assert!(b.is_feasible());
        assert_eq!(compare_scenarios(&a, &b), ScenarioRelation::TradeOff);
    }

    #[test]
    fn two_infeasible_scenarios_are_not_implicitly_ranked() {
        let a = assess_scenario(
            "a",
            CityMetrics { essential_access_share: 0.50, ..metrics() },
            floors(),
        )
        .unwrap();
        let b = assess_scenario(
            "b",
            CityMetrics { housing_cost_burden_p90: 0.70, ..metrics() },
            floors(),
        )
        .unwrap();
        assert_eq!(compare_scenarios(&a, &b), ScenarioRelation::BothInfeasible);
    }

    #[test]
    fn invalid_ratio_fails_closed() {
        let invalid = CityMetrics { water_service_reliability: 1.2, ..metrics() };
        let error = assess_scenario("invalid", invalid, floors()).unwrap_err();
        assert_eq!(error.field, "water_service_reliability");
    }

    #[test]
    fn non_finite_values_fail_closed() {
        let invalid = CityMetrics {
            lifecycle_emissions_kg_co2e_per_person_year: f64::NAN,
            ..metrics()
        };
        assert!(assess_scenario("nan", invalid, floors()).is_err());
    }
}
