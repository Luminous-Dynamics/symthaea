//! Exact, synthetic-only cost accounting for the HABITAT-VERTICAL-001 fixture.
//!
//! This is a benchmark oracle, not an ERP or lifecycle-cost engine. It sums
//! declared cost categories using integer minor units, rejects incomplete
//! category sets and invalid denominators, and compares scenarios only when
//! currency and accepted-unit semantics match. Fixture values are arbitrary
//! arithmetic controls and are not commercial prices or measured costs.

use serde_json::Value;

/// Mandatory cost categories in contract version 0.1.
///
/// A new cost basis must version the schema rather than silently dropping a
/// category that makes a candidate look artificially cheap.
pub const REQUIRED_COST_CATEGORIES: [&str; 12] = [
    "materials",
    "yield_loss_scrap_rework",
    "inbound_freight_duties",
    "labor_training",
    "consumables_tooling",
    "equipment_capacity_depreciation",
    "qa_metrology_calibration",
    "facility_utilities",
    "packaging_warehousing",
    "outbound_freight_returns_warranty",
    "maintenance_spares",
    "reverse_logistics_end_of_life",
];

/// Aggregated cost scenario result in explicitly synthetic or source-declared
/// integer minor units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostScenarioReport {
    pub scenario_id: String,
    pub role: String,
    pub currency: String,
    pub accepted_units: u64,
    pub total_cost_minor_units: u64,
    pub expected_total_minor_units: u64,
}

/// Exact-ratio comparison of candidate and reference cost per accepted unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostComparisonReport {
    pub candidate_scenario_id: String,
    pub reference_scenario_id: String,
    pub currency: String,
    /// Floor of the unit-cost ratio multiplied by 10,000. E.g. 8,400 is 84%.
    pub ratio_basis_points_floor: u64,
    pub target_ratio_basis_points: u32,
    /// Calculated by exact cross multiplication, not by the rounded display.
    pub target_met: bool,
}

/// Aggregate one scenario, failing closed if identity, currency, accepted
/// quantity, category coverage, integer amounts or expected total are invalid.
pub fn calculate_cost_scenario(
    root: &Value,
    scenario_id: &str,
) -> Result<CostScenarioReport, String> {
    if scenario_id.trim().is_empty() {
        return Err("scenario_id must be a non-empty string".to_owned());
    }
    let scenarios = root
        .get("synthetic_cost_scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| "synthetic_cost_scenarios must be an array".to_owned())?;
    let matches: Vec<_> = scenarios
        .iter()
        .filter(|scenario| scenario.get("scenario_id").and_then(Value::as_str) == Some(scenario_id))
        .collect();
    let scenario = match matches.as_slice() {
        [one] => *one,
        [] => return Err(format!("cost scenario {scenario_id:?} not found")),
        _ => return Err(format!("cost scenario {scenario_id:?} is duplicated")),
    };

    if scenario.get("status").and_then(Value::as_str) != Some("synthetic_known_answer_only") {
        return Err(format!(
            "cost scenario {scenario_id:?} is not explicitly marked synthetic_known_answer_only"
        ));
    }

    let role = scenario
        .get("role")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("cost scenario {scenario_id:?} has no role"))?
        .to_owned();
    let currency = scenario
        .get("currency")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("cost scenario {scenario_id:?} has no currency basis"))?
        .to_owned();
    let accepted_units = scenario
        .get("accepted_units")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("cost scenario {scenario_id:?} must have positive accepted_units"))?;

    let components = scenario
        .get("cost_components")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("cost scenario {scenario_id:?} has no cost_components array"))?;
    let mut seen = std::collections::HashSet::new();
    let mut total: u64 = 0;

    for (index, component) in components.iter().enumerate() {
        let category = component
            .get("category")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                format!("cost scenario {scenario_id:?} component {index} has no category")
            })?;
        if !REQUIRED_COST_CATEGORIES.contains(&category) {
            return Err(format!(
                "cost scenario {scenario_id:?} has unsupported category {category:?}"
            ));
        }
        if !seen.insert(category) {
            return Err(format!(
                "cost scenario {scenario_id:?} duplicates category {category:?}"
            ));
        }
        let amount = component
            .get("amount_minor_units")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                format!(
                    "cost scenario {scenario_id:?} category {category:?} requires a non-negative integer amount_minor_units"
                )
            })?;
        total = total.checked_add(amount).ok_or_else(|| {
            format!("cost scenario {scenario_id:?} total overflows u64")
        })?;
    }

    let missing: Vec<_> = REQUIRED_COST_CATEGORIES
        .iter()
        .filter(|category| !seen.contains(**category))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "cost scenario {scenario_id:?} is missing required categories: {}",
            missing.join(", ")
        ));
    }
    if total == 0 {
        return Err(format!(
            "cost scenario {scenario_id:?} must have a positive total cost"
        ));
    }

    let expected_total = scenario
        .get("expected_total_minor_units")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            format!("cost scenario {scenario_id:?} has no expected_total_minor_units")
        })?;
    if total != expected_total {
        return Err(format!(
            "cost scenario {scenario_id:?} total mismatch: expected {expected_total}, computed {total}"
        ));
    }

    Ok(CostScenarioReport {
        scenario_id: scenario_id.to_owned(),
        role,
        currency,
        accepted_units,
        total_cost_minor_units: total,
        expected_total_minor_units: expected_total,
    })
}

/// Exact result for one synthetic material stream and accounting boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialBalanceReport {
    pub case_id: String,
    pub stream_id: String,
    pub mass_unit: String,
    pub input_mass_mg: u64,
    pub accounted_mass_mg: u64,
    pub unaccounted_mass_mg: u64,
    pub tolerance_mg: u64,
    /// Floor of accounted / input × 10,000. 10,000 means 100% closure.
    pub closure_basis_points_floor: u64,
    /// Accepted product mass / input mass, in basis points.
    pub accepted_yield_basis_points_floor: u64,
}

/// Reconcile one material stream exactly; refuse missing mass, double-counting,
/// overflow, inconsistent reported residuals, or non-synthetic inputs.
///
/// Mass unit is fixed to milligrams in schema v0.1. A future unit/boundary
/// change requires a schema revision; unrelated materials must not be combined
/// across incompatible process boundaries merely because their numbers add.
pub fn calculate_material_balance(
    root: &Value,
    case_id: &str,
) -> Result<MaterialBalanceReport, String> {
    if case_id.trim().is_empty() {
        return Err("case_id must be a non-empty string".to_owned());
    }
    let cases = root
        .get("synthetic_material_balance_cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "synthetic_material_balance_cases must be an array".to_owned())?;
    let matches: Vec<_> = cases
        .iter()
        .filter(|case| case.get("case_id").and_then(Value::as_str) == Some(case_id))
        .collect();
    let case = match matches.as_slice() {
        [one] => *one,
        [] => return Err(format!("material-balance case {case_id:?} not found")),
        _ => return Err(format!("material-balance case {case_id:?} is duplicated")),
    };
    if case.get("status").and_then(Value::as_str) != Some("synthetic_known_answer_only") {
        return Err(format!(
            "material-balance case {case_id:?} must be marked synthetic_known_answer_only"
        ));
    }

    let stream_id = case
        .get("stream_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("material-balance case {case_id:?} has no stream_id"))?
        .to_owned();
    let mass_unit = case
        .get("mass_unit")
        .and_then(Value::as_str)
        .filter(|value| *value == "mg")
        .ok_or_else(|| {
            format!("material-balance case {case_id:?} must use explicit schema-v0.1 unit \"mg\"")
        })?
        .to_owned();
    if case
        .get("accounting_boundary")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(format!(
            "material-balance case {case_id:?} has no accounting_boundary"
        ));
    }

    let read_mass = |field: &str| -> Result<u64, String> {
        case.get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("material-balance case {case_id:?} requires non-negative integer {field}"))
    };
    let input_mass_mg = read_mass("input_mass_mg")?;
    if input_mass_mg == 0 {
        return Err(format!(
            "material-balance case {case_id:?} input mass must be positive"
        ));
    }
    let accepted_output_mass_mg = read_mass("accepted_output_mass_mg")?;
    let scrap_mass_mg = read_mass("scrap_mass_mg")?;
    let rework_wip_mass_mg = read_mass("rework_wip_mass_mg")?;
    let process_residue_mass_mg = read_mass("process_residue_mass_mg")?;
    let reported_unaccounted_mass_mg = read_mass("reported_unaccounted_mass_mg")?;
    let tolerance_mg = read_mass("tolerance_mg")?;

    let accounted_mass_mg = [
        accepted_output_mass_mg,
        scrap_mass_mg,
        rework_wip_mass_mg,
        process_residue_mass_mg,
    ]
    .into_iter()
    .try_fold(0_u64, |sum, part| sum.checked_add(part))
    .ok_or_else(|| format!("material-balance case {case_id:?} accounted mass overflows u64"))?;

    if accounted_mass_mg > input_mass_mg {
        return Err(format!(
            "material-balance case {case_id:?} accounts for {accounted_mass_mg} mg from only {input_mass_mg} mg input"
        ));
    }
    let computed_unaccounted = input_mass_mg - accounted_mass_mg;
    if reported_unaccounted_mass_mg != computed_unaccounted {
        return Err(format!(
            "material-balance case {case_id:?} residual mismatch: reported {reported_unaccounted_mass_mg} mg, computed {computed_unaccounted} mg"
        ));
    }
    if computed_unaccounted > tolerance_mg {
        return Err(format!(
            "material-balance case {case_id:?} unaccounted mass {computed_unaccounted} mg exceeds tolerance {tolerance_mg} mg"
        ));
    }

    let expected_closure = case
        .get("expected_closure_basis_points_floor")
        .and_then(Value::as_u64)
        .filter(|value| *value <= 10_000)
        .ok_or_else(|| {
            format!("material-balance case {case_id:?} expected_closure_basis_points_floor must be 0..=10000")
        })?;
    let accounted_scaled = (accounted_mass_mg as u128)
        .checked_mul(10_000)
        .ok_or_else(|| format!("material-balance case {case_id:?} closure ratio overflow"))?;
    let closure_basis_points_floor = (accounted_scaled / input_mass_mg as u128) as u64;
    if closure_basis_points_floor != expected_closure {
        return Err(format!(
            "material-balance case {case_id:?} closure known-answer mismatch: expected {expected_closure}, computed {closure_basis_points_floor}"
        ));
    }

    let accepted_scaled = (accepted_output_mass_mg as u128)
        .checked_mul(10_000)
        .ok_or_else(|| format!("material-balance case {case_id:?} accepted-yield ratio overflow"))?;
    let accepted_yield_basis_points_floor = (accepted_scaled / input_mass_mg as u128) as u64;

    Ok(MaterialBalanceReport {
        case_id: case_id.to_owned(),
        stream_id,
        mass_unit,
        input_mass_mg,
        accounted_mass_mg,
        unaccounted_mass_mg: computed_unaccounted,
        tolerance_mg,
        closure_basis_points_floor,
        accepted_yield_basis_points_floor,
    })
}

/// Compare candidate delivered cost per accepted unit against a reference.
///
/// Uses integer cross-multiplication so the pass/fail decision does not depend
/// on floating-point rounding. A 10,000-basis-point ratio is 100%; 8,000 is 80%.
pub fn calculate_cost_comparison(root: &Value) -> Result<CostComparisonReport, String> {
    let comparison = root
        .get("synthetic_cost_comparison")
        .ok_or_else(|| "synthetic_cost_comparison object is missing".to_owned())?;
    if comparison.get("status").and_then(Value::as_str) != Some("synthetic_known_answer_only") {
        return Err("cost comparison must be marked synthetic_known_answer_only".to_owned());
    }
    let candidate_id = comparison
        .get("candidate_scenario_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "candidate_scenario_id is missing".to_owned())?;
    let reference_id = comparison
        .get("reference_scenario_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "reference_scenario_id is missing".to_owned())?;
    let target = comparison
        .get("target_ratio_basis_points")
        .and_then(Value::as_u64)
        .filter(|value| *value <= 1_000_000)
        .ok_or_else(|| "target_ratio_basis_points must be an integer in 0..=1,000,000".to_owned())?
        as u32;

    let candidate = calculate_cost_scenario(root, candidate_id)?;
    let reference = calculate_cost_scenario(root, reference_id)?;
    if candidate.currency != reference.currency {
        return Err(format!(
            "cannot compare currencies {:?} and {:?} without an explicit conversion basis",
            candidate.currency, reference.currency
        ));
    }
    if candidate.role != "candidate" || reference.role != "reference" {
        return Err("comparison roles must be candidate and reference respectively".to_owned());
    }

    // Ratio of (candidate total / candidate accepted units) to
    // (reference total / reference accepted units).
    let numerator = (candidate.total_cost_minor_units as u128)
        .checked_mul(reference.accepted_units as u128)
        .and_then(|value| value.checked_mul(10_000))
        .ok_or_else(|| "candidate cost-ratio numerator overflow".to_owned())?;
    let denominator = (reference.total_cost_minor_units as u128)
        .checked_mul(candidate.accepted_units as u128)
        .ok_or_else(|| "cost-ratio denominator overflow".to_owned())?;
    if denominator == 0 {
        return Err("reference unit cost denominator must be positive".to_owned());
    }
    let ratio_basis_points_floor = numerator / denominator;
    let target_rhs = denominator
        .checked_mul(target as u128)
        .ok_or_else(|| "cost-ratio target cross-product overflow".to_owned())?;

    let expected_ratio = comparison
        .get("expected_ratio_basis_points_floor")
        .and_then(Value::as_u64)
        .ok_or_else(|| "expected_ratio_basis_points_floor is missing".to_owned())?;
    let expected_target_met = comparison
        .get("expected_target_met")
        .and_then(Value::as_bool)
        .ok_or_else(|| "expected_target_met is missing".to_owned())?;
    let target_met = numerator <= target_rhs;

    if ratio_basis_points_floor != expected_ratio {
        return Err(format!(
            "cost-ratio known-answer mismatch: expected {expected_ratio} basis points, computed {ratio_basis_points_floor}"
        ));
    }
    if target_met != expected_target_met {
        return Err(format!(
            "cost target known-answer mismatch: expected {expected_target_met}, computed {target_met}"
        ));
    }

    Ok(CostComparisonReport {
        candidate_scenario_id: candidate.scenario_id,
        reference_scenario_id: reference.scenario_id,
        currency: candidate.currency,
        ratio_basis_points_floor,
        target_ratio_basis_points: target,
        target_met,
    })
}
