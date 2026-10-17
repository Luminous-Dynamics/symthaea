use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

const FIXTURE_PATH: &str = "docs/research/fixtures/env-mobility-make-buy-v1.json";
const FIXED_COST_FIELDS: &[&str] = &[
    "maintenance",
    "site_support",
    "calibration_qualification",
    "training",
    "supplier_setup_or_tooling",
];
const VARIABLE_COST_FIELDS: &[&str] = &[
    "unit_material_or_quote",
    "direct_labor",
    "energy_consumables",
    "logistics",
    "variable_inspection",
    "expected_rework_scrap",
];
const DECISIONS: &[&str] = &[
    "prefer_buy_lower_annual_cost",
    "prefer_make_lower_annual_cost",
    "prefer_buy_qualified_even_if_make_cheaper",
    "prefer_make_qualified",
    "abstain_missing_cost_evidence",
    "abstain_qualification_evidence",
    "no_qualified_alternative",
    "cost_indifferent_within_tolerance",
];
const BREAK_EVEN_CLASSES: &[&str] = &[
    "positive",
    "none",
    "not_computable",
    "all_volumes_equal",
];

#[derive(Debug, Clone)]
struct Alternative {
    qualification_status: String,
    evidence_status: String,
    annual_fixed_cost: f64,
    variable_cost_per_attempt: Option<f64>,
    yield_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EconomicsCaseResult {
    case_id: String,
    demand_good_units_per_year: f64,
    expected_decision: String,
    derived_decision: String,
    decision_matches: bool,
    expected_break_even_class: String,
    derived_break_even_class: String,
    break_even_class_matches: bool,
    matches: bool,
    make_qualified: bool,
    buy_qualified: bool,
    make_annual_fixed_cost: Option<f64>,
    buy_annual_fixed_cost: Option<f64>,
    make_variable_cost_per_good_unit: Option<f64>,
    buy_variable_cost_per_good_unit: Option<f64>,
    make_annual_cost_per_good_unit: Option<f64>,
    buy_annual_cost_per_good_unit: Option<f64>,
    break_even_demand_good_units_per_year: Option<f64>,
    decision_reason: String,
}

#[derive(Debug, Serialize)]
pub struct EconomicsReport {
    pub report_schema: &'static str,
    pub verifier_version: &'static str,
    pub verifier_source_revision: Option<&'static str>,
    pub corpus_id: String,
    pub schema_version: String,
    pub currency: String,
    pub fixture_path: String,
    pub raw_sha256: String,
    pub case_count: usize,
    pub decision_mismatch_count: usize,
    pub break_even_mismatch_count: usize,
    pub total_mismatch_count: usize,
    pub expected_decision_counts: BTreeMap<String, usize>,
    pub derived_decision_counts: BTreeMap<String, usize>,
    pub claim_ceiling: &'static str,
    pub cases: Vec<EconomicsCaseResult>,
}

fn exact_keys(value: &Value, required: &[&str], context: &str) -> Result<()> {
    let object = value.as_object().with_context(|| format!("{context}: expected object"))?;
    let actual: HashSet<&str> = object.keys().map(String::as_str).collect();
    let expected: HashSet<&str> = required.iter().copied().collect();
    ensure!(actual == expected, "{context}: field set differs from the frozen schema");
    Ok(())
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value.as_object().with_context(|| format!("{context}: expected object"))
}

fn string_field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value.get(key).and_then(Value::as_str)
        .with_context(|| format!("{context}: required string field {key:?} is missing or invalid"))
}

fn number_field(value: &Value, key: &str, context: &str) -> Result<f64> {
    let number = value.get(key).and_then(Value::as_f64)
        .with_context(|| format!("{context}: required numeric field {key:?} is missing or invalid"))?;
    ensure!(number.is_finite(), "{context}: {key} must be finite");
    Ok(number)
}

fn nonnegative_field(value: &Value, key: &str, context: &str) -> Result<f64> {
    let number = number_field(value, key, context)?;
    ensure!(number >= 0.0, "{context}: {key} cannot be negative");
    Ok(number)
}

fn optional_nonnegative_field(value: &Value, key: &str, context: &str) -> Result<Option<f64>> {
    let raw = value.get(key).with_context(|| format!("{context}: component {key:?} is missing"))?;
    if raw.is_null() {
        return Ok(None);
    }
    let number = raw.as_f64().with_context(|| format!("{context}: {key} must be a number or null"))?;
    ensure!(number.is_finite() && number >= 0.0, "{context}: {key} must be finite and nonnegative");
    Ok(Some(number))
}

fn sum_components(value: &Value, fields: &[&str], context: &str, allow_missing: bool) -> Result<Option<f64>> {
    exact_keys(value, fields, context)?;
    let mut total = 0.0;
    let mut complete = true;
    for field in fields {
        match optional_nonnegative_field(value, field, context)? {
            Some(number) => total += number,
            None => {
                ensure!(allow_missing, "{context}: fixed component {field:?} cannot be null");
                complete = false;
            }
        }
    }
    ensure!(total.is_finite(), "{context}: component sum overflowed");
    Ok(complete.then_some(total))
}

fn capital_recovery_factor(rate: f64, life_years: f64, context: &str) -> Result<f64> {
    ensure!(life_years > 0.0 && life_years.is_finite(), "{context}: capital life must be positive and finite");
    ensure!(rate.is_finite() && (0.0..=1.0).contains(&rate), "{context}: discount rate must be in [0, 1]");
    if rate == 0.0 {
        return Ok(1.0 / life_years);
    }
    let growth = (1.0 + rate).powf(life_years);
    let denominator = growth - 1.0;
    ensure!(growth.is_finite() && denominator > 0.0, "{context}: invalid capital-recovery calculation");
    let factor = rate * growth / denominator;
    ensure!(factor.is_finite() && factor >= 0.0, "{context}: capital-recovery factor invalid");
    Ok(factor)
}

fn parse_alternative(value: &Value, context: &str) -> Result<Alternative> {
    exact_keys(value, &[
        "qualification_status",
        "capex_upfront",
        "capital_life_years",
        "annual_discount_rate",
        "annual_fixed_costs",
        "variable_cost_per_attempt_components",
        "yield_fraction",
        "evidence_status",
    ], context)?;
    let qualification_status = string_field(value, "qualification_status", context)?.to_owned();
    ensure!(
        ["qualified_for_profile", "not_qualified", "unknown"].contains(&qualification_status.as_str()),
        "{context}: unsupported qualification status"
    );
    let evidence_status = string_field(value, "evidence_status", context)?.to_owned();
    ensure!(
        ["synthetic_complete", "missing_supplier_quote", "missing_process_measurement", "unknown"].contains(&evidence_status.as_str()),
        "{context}: unsupported evidence status"
    );
    let capex = nonnegative_field(value, "capex_upfront", context)?;
    let life = number_field(value, "capital_life_years", context)?;
    let rate = number_field(value, "annual_discount_rate", context)?;
    let fixed_components = value.get("annual_fixed_costs").context("annual_fixed_costs missing")?;
    let fixed_sum = sum_components(fixed_components, FIXED_COST_FIELDS, context, false)?
        .context("fixed cost components unexpectedly incomplete")?;
    let annual_fixed_cost = capex * capital_recovery_factor(rate, life, context)? + fixed_sum;
    ensure!(annual_fixed_cost.is_finite(), "{context}: annual fixed cost overflowed");

    let variable_components = value.get("variable_cost_per_attempt_components")
        .context("variable_cost_per_attempt_components missing")?;
    let variable_cost_per_attempt = sum_components(variable_components, VARIABLE_COST_FIELDS, context, true)?;
    if variable_cost_per_attempt.is_none() {
        ensure!(evidence_status != "synthetic_complete",
            "{context}: missing variable cost component cannot be labelled complete");
    }
    let yield_fraction = number_field(value, "yield_fraction", context)?;
    ensure!(
        yield_fraction.is_finite() && yield_fraction > 0.0 && yield_fraction <= 1.0,
        "{context}: yield_fraction must be in (0, 1]"
    );

    Ok(Alternative {
        qualification_status,
        evidence_status,
        annual_fixed_cost,
        variable_cost_per_attempt,
        yield_fraction,
    })
}

fn derive_break_even(make: &Alternative, buy: &Alternative) -> (String, Option<f64>, Option<f64>, Option<f64>) {
    let (Some(make_variable), Some(buy_variable)) =
        (make.variable_cost_per_attempt, buy.variable_cost_per_attempt)
    else {
        return ("not_computable".to_owned(), None, None, None);
    };
    let make_variable_per_good = make_variable / make.yield_fraction;
    let buy_variable_per_good = buy_variable / buy.yield_fraction;
    let numerator = buy.annual_fixed_cost - make.annual_fixed_cost;
    let denominator = make_variable_per_good - buy_variable_per_good;
    let epsilon = 1e-9;
    if denominator.abs() <= epsilon {
        if numerator.abs() <= epsilon {
            return ("all_volumes_equal".to_owned(), None, Some(make_variable_per_good), Some(buy_variable_per_good));
        }
        return ("none".to_owned(), None, Some(make_variable_per_good), Some(buy_variable_per_good));
    }
    let crossing = numerator / denominator;
    if crossing.is_finite() && crossing > 0.0 {
        ("positive".to_owned(), Some(crossing), Some(make_variable_per_good), Some(buy_variable_per_good))
    } else {
        ("none".to_owned(), None, Some(make_variable_per_good), Some(buy_variable_per_good))
    }
}

fn derive_decision(
    make: &Alternative,
    buy: &Alternative,
    demand: f64,
) -> (String, String, Option<f64>, Option<f64>) {
    let make_variable_per_good = make.variable_cost_per_attempt.map(|v| v / make.yield_fraction);
    let buy_variable_per_good = buy.variable_cost_per_attempt.map(|v| v / buy.yield_fraction);
    let make_annual = make_variable_per_good.map(|v| make.annual_fixed_cost / demand + v);
    let buy_annual = buy_variable_per_good.map(|v| buy.annual_fixed_cost / demand + v);

    if make.qualification_status == "unknown" || buy.qualification_status == "unknown" {
        return ("abstain_qualification_evidence".into(), "one or both qualification states are unknown".into(), make_annual, buy_annual);
    }
    if make.qualification_status != "qualified_for_profile" && buy.qualification_status != "qualified_for_profile" {
        return ("no_qualified_alternative".into(), "neither alternative meets the declared quality/qualification profile".into(), make_annual, buy_annual);
    }
    if make.qualification_status != "qualified_for_profile" {
        return ("prefer_buy_qualified_even_if_make_cheaper".into(), "in-house process is not qualified for the comparable output profile".into(), make_annual, buy_annual);
    }
    if buy.qualification_status != "qualified_for_profile" {
        return ("prefer_make_qualified".into(), "supplier alternative is not qualified for the comparable output profile".into(), make_annual, buy_annual);
    }
    if make.variable_cost_per_attempt.is_none() || buy.variable_cost_per_attempt.is_none() {
        return ("abstain_missing_cost_evidence".into(), "a required variable-cost component is missing; missing is not zero".into(), make_annual, buy_annual);
    }
    let make_cost = make_annual.expect("make variable cost exists after completeness gate");
    let buy_cost = buy_annual.expect("buy variable cost exists after completeness gate");
    let tolerance = 0.01_f64.max(make_cost.abs().min(buy_cost.abs()) * 0.001);
    if (make_cost - buy_cost).abs() <= tolerance {
        return ("cost_indifferent_within_tolerance".into(), "annual unit-cost difference is within declared decision tolerance".into(), Some(make_cost), Some(buy_cost));
    }
    if make_cost < buy_cost {
        ("prefer_make_lower_annual_cost".into(), "qualified in-house alternative has lower expected annual cost per conforming unit".into(), Some(make_cost), Some(buy_cost))
    } else {
        ("prefer_buy_lower_annual_cost".into(), "qualified supplier alternative has lower expected annual cost per conforming unit".into(), Some(make_cost), Some(buy_cost))
    }
}

pub fn run() -> Result<EconomicsReport> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().context("xtask must be one level below workspace root")?;
    let path = root.join(FIXTURE_PATH);
    let bytes = fs::read(&path).with_context(|| format!("read economics fixture {}", path.display()))?;
    let raw_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let parsed: Value = serde_json::from_slice(&bytes).with_context(|| format!("parse economics fixture {}", path.display()))?;
    let context = FIXTURE_PATH;

    exact_keys(&parsed, &[
        "model_id", "schema_version", "status", "currency", "source_class",
        "generated_on", "claim_ceiling", "formula_contract", "cases",
    ], context)?;
    ensure!(string_field(&parsed, "schema_version", context)? == "env-mobility-make-buy-case-v1",
        "{context}: unsupported economics schema version");
    ensure!(string_field(&parsed, "status", context)? == "synthetic_frozen_candidate",
        "{context}: economics fixture must remain explicitly synthetic");
    for key in ["model_id", "currency", "source_class", "generated_on", "claim_ceiling"] {
        ensure!(!string_field(&parsed, key, context)?.trim().is_empty(),
            "{context}: metadata {key:?} is empty");
    }
    let formula = object(parsed.get("formula_contract").context("formula_contract missing")?, "formula_contract")?;
    for key in [
        "annual_fixed_cost", "crf", "variable_cost_per_attempt", "variable_cost_per_accepted_unit",
        "total_annual_cost_per_accepted_unit", "break_even_demand", "eligibility_gate",
    ] {
        let text = formula.get(key).and_then(Value::as_str)
            .with_context(|| format!("formula_contract.{key} must be a string"))?;
        ensure!(!text.trim().is_empty(), "formula_contract.{key} must not be empty");
    }
    ensure!(formula.get("expected_decisions_are_independent_test_labels").and_then(Value::as_bool) == Some(true),
        "{context}: expected decisions must be independent test labels");

    let cases = parsed.get("cases").and_then(Value::as_array).context("cases must be an array")?;
    ensure!(cases.len() == 7, "{context}: expected seven synthetic make/buy cases");
    let mut ids = HashSet::new();
    let mut results = Vec::with_capacity(cases.len());
    let mut expected_counts = BTreeMap::new();
    let mut derived_counts = BTreeMap::new();
    let mut decision_mismatch_count = 0usize;
    let mut break_even_mismatch_count = 0usize;

    for (index, case) in cases.iter().enumerate() {
        let at = format!("{context} case[{index}]");
        exact_keys(case, &[
            "case_id", "description", "demand_good_units_per_year", "make", "buy",
            "expected_decision", "expected_break_even_class",
        ], &[], &at)?;
        let id = string_field(case, "case_id", &at)?.to_owned();
        ensure!(!id.trim().is_empty() && ids.insert(id.clone()), "{at}: empty or duplicate case ID");
        ensure!(!string_field(case, "description", &at)?.trim().is_empty(), "{at}: description empty");
        let demand = number_field(case, "demand_good_units_per_year", &at)?;
        ensure!(demand > 0.0, "{at}: demand must be positive");
        let make = parse_alternative(case.get("make").context("make alternative missing")?, &format!("{at} make"))?;
        let buy = parse_alternative(case.get("buy").context("buy alternative missing")?, &format!("{at} buy"))?;

        let expected_decision = string_field(case, "expected_decision", &at)?.to_owned();
        ensure!(DECISIONS.contains(&expected_decision.as_str()), "{at}: unsupported expected decision");
        let expected_break_even_class = string_field(case, "expected_break_even_class", &at)?.to_owned();
        ensure!(BREAK_EVEN_CLASSES.contains(&expected_break_even_class.as_str()),
            "{at}: unsupported expected break-even class");

        let (derived_break_even_class, break_even_demand, make_variable_per_good, buy_variable_per_good) =
            derive_break_even(&make, &buy);
        let (derived_decision, decision_reason, make_annual, buy_annual) =
            derive_decision(&make, &buy, demand);
        let decision_matches = derived_decision == expected_decision;
        let break_even_class_matches = derived_break_even_class == expected_break_even_class;
        if !decision_matches { decision_mismatch_count += 1; }
        if !break_even_class_matches { break_even_mismatch_count += 1; }
        *expected_counts.entry(expected_decision.clone()).or_insert(0usize) += 1;
        *derived_counts.entry(derived_decision.clone()).or_insert(0usize) += 1;

        results.push(EconomicsCaseResult {
            case_id: id,
            demand_good_units_per_year: demand,
            expected_decision,
            derived_decision,
            decision_matches,
            expected_break_even_class,
            derived_break_even_class,
            break_even_class_matches,
            matches: decision_matches && break_even_class_matches,
            make_qualified: make.qualification_status == "qualified_for_profile",
            buy_qualified: buy.qualification_status == "qualified_for_profile",
            make_annual_fixed_cost: Some(make.annual_fixed_cost),
            buy_annual_fixed_cost: Some(buy.annual_fixed_cost),
            make_variable_cost_per_good_unit: make_variable_per_good,
            buy_variable_cost_per_good_unit: buy_variable_per_good,
            make_annual_cost_per_good_unit: make_annual,
            buy_annual_cost_per_good_unit: buy_annual,
            break_even_demand_good_units_per_year: break_even_demand,
            decision_reason,
        });
    }

    let total_mismatch_count = results.iter().filter(|case| !case.matches).count();
    Ok(EconomicsReport {
        report_schema: "env-mobility-make-buy-report-v1",
        verifier_version: env!("CARGO_PKG_VERSION"),
        verifier_source_revision: option_env!("GITHUB_SHA"),
        corpus_id: string_field(&parsed, "model_id", context)?.to_owned(),
        schema_version: string_field(&parsed, "schema_version", context)?.to_owned(),
        currency: string_field(&parsed, "currency", context)?.to_owned(),
        fixture_path: FIXTURE_PATH.to_owned(),
        raw_sha256,
        case_count: cases.len(),
        decision_mismatch_count,
        break_even_mismatch_count,
        total_mismatch_count,
        expected_decision_counts: expected_counts,
        derived_decision_counts: derived_counts,
        claim_ceiling: "Synthetic make/buy arithmetic only; no real quotes, product qualification, manufacturing capability, cost saving, or procurement approval is established.",
        cases: results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_make_buy_fixture_matches_independent_labels() {
        let report = run().expect("run synthetic make/buy corpus");
        assert_eq!(report.case_count, 7);
        assert_eq!(report.decision_mismatch_count, 0);
        assert_eq!(report.break_even_mismatch_count, 0);
        assert_eq!(report.total_mismatch_count, 0);
        let low = report.cases.iter().find(|c| c.case_id == "MAKEBUY-LOW-05").unwrap();
        let base = report.cases.iter().find(|c| c.case_id == "MAKEBUY-BASE-25").unwrap();
        assert!(low.buy_annual_cost_per_good_unit.unwrap() < low.make_annual_cost_per_good_unit.unwrap());
        assert!(base.make_annual_cost_per_good_unit.unwrap() < base.buy_annual_cost_per_good_unit.unwrap());
        assert!((base.break_even_demand_good_units_per_year.unwrap() - 13.502).abs() < 0.01);
    }

    #[test]
    fn make_buy_missing_quote_is_not_zero_cost() {
        let report = run().expect("run synthetic make/buy corpus");
        let missing = report.cases.iter().find(|c| c.case_id == "MAKEBUY-MISSING-QUOTE").unwrap();
        assert_eq!(missing.derived_decision, "abstain_missing_cost_evidence");
        assert_eq!(missing.buy_variable_cost_per_good_unit, None);
        assert!(missing.buy_annual_cost_per_good_unit.is_none());
    }

    #[test]
    fn ineligible_process_never_wins_on_price_alone() {
        let report = run().expect("run synthetic make/buy corpus");
        let unqualified = report.cases.iter().find(|c| c.case_id == "MAKEBUY-UNQUALIFIED").unwrap();
        assert!(!unqualified.make_qualified);
        assert_eq!(unqualified.derived_decision, "prefer_buy_qualified_even_if_make_cheaper");
    }

    #[test]
    fn zero_discount_rate_uses_linear_amortization() {
        let factor = capital_recovery_factor(0.0, 4.0, "unit test").unwrap();
        assert!((factor - 0.25).abs() < 1e-12);
    }
}
