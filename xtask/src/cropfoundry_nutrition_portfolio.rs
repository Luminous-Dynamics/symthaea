use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use crate::cropfoundry_nutrition_accounting::audit_json_value;

const PORTFOLIO_SCHEMA_V0: &str = "cropfoundry-nutrition-portfolio-input-v0";
const RESOURCE_FIELDS: [(&str, &str); 4] = [
    ("water_liters", "water_boundary"),
    ("land_square_meters", "land_boundary"),
    ("energy_kwh", "energy_boundary"),
    ("cost_amount", "cost_boundary"),
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortfolioInput {
    schema_version: String,
    portfolio_id: String,
    comparison_basis: String,
    comparison_basis_evidence_uri: String,
    expected_nutrient_ids: Vec<String>,
    options: Vec<PortfolioOptionInput>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortfolioOptionInput {
    option_id: String,
    records: Vec<Value>,
}

#[derive(Debug, Clone)]
struct AuditedRow {
    batch_id: String,
    nutrient_id: String,
    input: Value,
    receipt: Value,
}

#[derive(Debug, Serialize)]
struct NutrientPortfolioTotal {
    nutrient_id: String,
    batch_count: usize,
    direct_assay_concentration_batch_count: usize,
    estimated_delivery_batch_count: usize,
    total_delivered_mg: Option<f64>,
    total_delivered_interval_lower_mg: Option<f64>,
    total_delivered_interval_upper_mg: Option<f64>,
    interval_coverage_batch_count: usize,
    estimated_bioavailable_total_mg: Option<f64>,
    bioavailability_coverage_batch_count: usize,
    estimated_bioavailable_interval_lower_mg: Option<f64>,
    estimated_bioavailable_interval_upper_mg: Option<f64>,
    mg_per_liter_water: Option<f64>,
    mg_per_square_meter_land: Option<f64>,
    mg_per_kwh_energy: Option<f64>,
    mg_per_currency_unit: Option<f64>,
    currency_code: Option<String>,
    boundary_labels: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Serialize)]
struct OptionPortfolioReceipt {
    option_id: String,
    batch_count: usize,
    expected_nutrient_ids: Vec<String>,
    nutrients: Vec<NutrientPortfolioTotal>,
    total_water_liters: Option<f64>,
    water_boundary: Option<String>,
    total_land_square_meters: Option<f64>,
    land_boundary: Option<String>,
    total_energy_kwh: Option<f64>,
    energy_boundary: Option<String>,
    total_cost_amount: Option<f64>,
    cost_boundary: Option<String>,
    currency_code: Option<String>,
    nutrient_composition_complete: bool,
    resource_ledger_complete: bool,
    accounting_pass: bool,
    issues: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct NutritionPortfolioReceipt {
    receipt_schema: String,
    input_sha256: String,
    portfolio_id: String,
    comparison_basis: String,
    comparison_basis_evidence_uri: String,
    expected_nutrient_ids: Vec<String>,
    option_count: usize,
    common_resource_count: usize,
    comparable_resource_boundaries: BTreeMap<String, Option<String>>,
    nutrient_interval_comparison_available: BTreeMap<String, bool>,
    bioavailability_comparison_available: BTreeMap<String, bool>,
    accounting_pass: bool,
    dietary_adequacy_evaluated: bool,
    food_safety_validated: bool,
    recommendation_emitted: bool,
    training_authorized: bool,
    options: Vec<OptionPortfolioReceipt>,
    issues: Vec<String>,
    caveats: Vec<String>,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key)
}

fn text_field(value: &Value, key: &str) -> Option<String> {
    field(value, key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

fn number_field(value: &Value, key: &str) -> Option<f64> {
    field(value, key).and_then(Value::as_f64).filter(|number| number.is_finite())
}

fn numeric_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0),
        _ => false,
    }
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    numerator.zip(denominator).and_then(|(n, d)| {
        if n.is_finite() && d.is_finite() && n >= 0.0 && d > 0.0 {
            let value = n / d;
            value.is_finite().then_some(value)
        } else {
            None
        }
    })
}

fn sum_finite(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut total = 0.0;
    for value in values {
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        total += value;
        if !total.is_finite() {
            return None;
        }
    }
    Some(total)
}

fn delivered_nutrient_mass(receipt: &Value) -> Option<f64> {
    number_field(receipt, "measured_nutrient_mass_delivered_mg")
        .or_else(|| number_field(receipt, "modeled_nutrient_mass_delivered_mg"))
}

fn same_batch_resources(rows: &[&AuditedRow], issues: &mut Vec<String>, batch_id: &str) -> bool {
    let Some(first) = rows.first() else {
        return false;
    };
    let first_input = &first.input;
    let mut valid = true;
    for row in rows.iter().skip(1) {
        for (value_key, boundary_key) in RESOURCE_FIELDS {
            if !numeric_equal(
                number_field(first_input, value_key),
                number_field(&row.input, value_key),
            ) || text_field(first_input, boundary_key) != text_field(&row.input, boundary_key)
            {
                issues.push(format!(
                    "batch {batch_id} has inconsistent {value_key}/{boundary_key} across nutrient records"
                ));
                valid = false;
            }
        }
        if text_field(first_input, "currency_code") != text_field(&row.input, "currency_code") {
            issues.push(format!(
                "batch {batch_id} has inconsistent currency_code across nutrient records"
            ));
            valid = false;
        }
    }
    valid
}

fn aggregate_option(
    option: &PortfolioOptionInput,
    expected_nutrients: &[String],
) -> OptionPortfolioReceipt {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let expected: BTreeSet<String> = expected_nutrients.iter().cloned().collect();
    let mut rows = Vec::new();
    let mut seen_pairs = BTreeSet::new();

    if option.option_id.trim().is_empty() {
        issues.push("option_id must be non-empty".to_owned());
    }
    if option.records.is_empty() {
        issues.push(format!("option {} has no accounting records", option.option_id));
    }

    for (index, input) in option.records.iter().enumerate() {
        match audit_json_value(input) {
            Ok(receipt) => {
                let batch_id = text_field(input, "batch_id").unwrap_or_default();
                let nutrient_id = text_field(input, "nutrient_id").unwrap_or_default();
                if batch_id.is_empty() {
                    issues.push(format!("record {index} has no non-empty batch_id"));
                }
                if nutrient_id.is_empty() {
                    issues.push(format!("record {index} has no non-empty nutrient_id"));
                }
                if !seen_pairs.insert((batch_id.clone(), nutrient_id.clone())) {
                    issues.push(format!(
                        "option {} duplicates batch {batch_id} / nutrient {nutrient_id}",
                        option.option_id
                    ));
                }
                if field(&receipt, "accounting_pass").and_then(Value::as_bool) != Some(true) {
                    issues.push(format!(
                        "option {} batch {batch_id} nutrient {nutrient_id} failed row-level accounting",
                        option.option_id
                    ));
                }
                rows.push(AuditedRow {
                    batch_id,
                    nutrient_id,
                    input: input.clone(),
                    receipt,
                });
            }
            Err(error) => issues.push(format!(
                "option {} record {index} is not a valid nutrition-accounting input: {error}",
                option.option_id
            )),
        }
    }

    let mut grouped_batches = BTreeMap::<String, Vec<&AuditedRow>>::new();
    for row in &rows {
        grouped_batches
            .entry(row.batch_id.clone())
            .or_default()
            .push(row);
    }
    let mut nutrient_composition_complete = !grouped_batches.is_empty();
    for (batch_id, batch_rows) in &grouped_batches {
        let actual: BTreeSet<String> = batch_rows.iter().map(|row| row.nutrient_id.clone()).collect();
        if actual != expected || actual.len() != batch_rows.len() {
            nutrient_composition_complete = false;
            let missing: Vec<String> = expected.difference(&actual).cloned().collect();
            let unexpected: Vec<String> = actual.difference(&expected).cloned().collect();
            issues.push(format!(
                "batch {batch_id} nutrient coverage differs from the declared set; missing={missing:?}, unexpected={unexpected:?}"
            ));
        }
        same_batch_resources(batch_rows, &mut issues, batch_id);
    }

    let cost_recorded = grouped_batches
        .values()
        .any(|batch_rows| number_field(&batch_rows[0].input, "cost_amount").is_some());
    let currencies_for_option: BTreeSet<String> = grouped_batches
        .values()
        .filter_map(|batch_rows| text_field(&batch_rows[0].input, "currency_code"))
        .map(|value| value.to_ascii_uppercase())
        .collect();
    if cost_recorded && currencies_for_option.len() > 1 {
        issues.push(format!(
            "option {} mixes currencies across batch resource ledgers",
            option.option_id
        ));
    }

    let batch_count = grouped_batches.len();
    let row_accounting_complete = rows.len() == option.records.len()
        && rows.iter().all(|row| {
            field(&row.receipt, "accounting_pass").and_then(Value::as_bool) == Some(true)
        });
    let mut resource_totals = BTreeMap::<String, Option<f64>>::new();
    let mut resource_boundaries = BTreeMap::<String, Option<String>>::new();
    let mut resource_measured_counts = BTreeMap::<String, usize>::new();
    let mut resource_boundary_sets = BTreeMap::<String, BTreeSet<String>>::new();

    for (value_key, boundary_key) in RESOURCE_FIELDS {
        let mut total = 0.0;
        let mut measured_batches = 0usize;
        let mut values_complete = batch_count > 0;
        for batch_rows in grouped_batches.values() {
            let input = &batch_rows[0].input;
            match number_field(input, value_key) {
                Some(value) => {
                    total += value;
                    measured_batches += 1;
                    if let Some(boundary) = text_field(input, boundary_key) {
                        resource_boundary_sets
                            .entry(boundary_key.to_owned())
                            .or_default()
                            .insert(boundary);
                    } else {
                        values_complete = false;
                        issues.push(format!(
                            "batch {} has {value_key} without {boundary_key}",
                            batch_rows[0].batch_id
                        ));
                    }
                }
                None => values_complete = false,
            }
        }
        resource_measured_counts.insert(value_key.to_owned(), measured_batches);
        if measured_batches == batch_count && values_complete && !total.is_finite() {
            issues.push(format!(
                "option {} resource total {value_key} overflowed the finite numeric range",
                option.option_id
            ));
        }
        let total_option = if row_accounting_complete
            && values_complete
            && measured_batches == batch_count
            && total.is_finite()
        {
            Some(total)
        } else {
            None
        };
        if measured_batches > 0 && measured_batches < batch_count {
            warnings.push(format!(
                "{value_key} covers {measured_batches}/{batch_count} batches; portfolio efficiency for this resource is unavailable"
            ));
        }
        resource_totals.insert(value_key.to_owned(), total_option);
        let boundaries = resource_boundary_sets.get(boundary_key);
        let boundary = boundaries.and_then(|set| (set.len() == 1).then(|| set.iter().next().unwrap().clone()));
        if boundaries.is_some_and(|set| set.len() > 1) {
            issues.push(format!(
                "option {} mixes incompatible {boundary_key} values",
                option.option_id
            ));
        }
        resource_boundaries.insert(boundary_key.to_owned(), boundary);
    }

    let mut nutrients = Vec::new();
    for nutrient_id in expected_nutrients {
        let nutrient_rows: Vec<&AuditedRow> = rows
            .iter()
            .filter(|row| row.nutrient_id == *nutrient_id)
            .collect();
        let total_delivered = if row_accounting_complete
            && nutrient_rows.len() == batch_count
            && nutrient_composition_complete
        {
            let values: Vec<Option<f64>> = nutrient_rows
                .iter()
                .map(|row| delivered_nutrient_mass(&row.receipt))
                .collect();
            if values.iter().all(Option::is_some) {
                let total = sum_finite(values.iter().flatten().copied());
                if total.is_none() {
                    issues.push(format!(
                        "option {} nutrient {nutrient_id} total overflowed the finite numeric range",
                        option.option_id
                    ));
                }
                total
            } else {
                None
            }
        } else {
            None
        };
        if total_delivered.is_none() {
            issues.push(format!(
                "option {} nutrient {nutrient_id} has incomplete delivered-mass coverage",
                option.option_id
            ));
        }

        let lower: Vec<Option<f64>> = nutrient_rows
            .iter()
            .map(|row| number_field(&row.receipt, "nutrient_mass_delivered_interval_lower_mg"))
            .collect();
        let upper: Vec<Option<f64>> = nutrient_rows
            .iter()
            .map(|row| number_field(&row.receipt, "nutrient_mass_delivered_interval_upper_mg"))
            .collect();
        let any_interval = lower.iter().any(Option::is_some) || upper.iter().any(Option::is_some);
        let intervals_complete = !nutrient_rows.is_empty()
            && lower.iter().all(Option::is_some)
            && upper.iter().all(Option::is_some);
        let interval_coverage_batch_count = lower
            .iter()
            .zip(&upper)
            .filter(|(low, high)| low.is_some() && high.is_some())
            .count();
        if any_interval && !intervals_complete {
            issues.push(format!(
                "option {} nutrient {nutrient_id} has partial interval coverage across batches",
                option.option_id
            ));
        }
        let total_lower = if row_accounting_complete
            && intervals_complete
            && nutrient_rows.len() == batch_count
        {
            sum_finite(lower.iter().flatten().copied())
        } else {
            None
        };
        let total_upper = if row_accounting_complete
            && intervals_complete
            && nutrient_rows.len() == batch_count
        {
            sum_finite(upper.iter().flatten().copied())
        } else {
            None
        };
        if intervals_complete && (total_lower.is_none() || total_upper.is_none()) {
            issues.push(format!(
                "option {} nutrient {nutrient_id} interval sum is non-finite or invalid",
                option.option_id
            ));
        }

        let bioavailability_values: Vec<Option<f64>> = nutrient_rows
            .iter()
            .map(|row| number_field(&row.receipt, "estimated_bioavailable_nutrient_mass_delivered_mg"))
            .collect();
        let bioavailability_coverage = bioavailability_values.iter().filter(|value| value.is_some()).count();
        let bioavailable_total = if row_accounting_complete
            && bioavailability_coverage == batch_count
            && batch_count > 0
        {
            let total = sum_finite(bioavailability_values.iter().flatten().copied());
            if total.is_none() {
                issues.push(format!(
                    "option {} nutrient {nutrient_id} bioavailable total overflowed",
                    option.option_id
                ));
            }
            total
        } else {
            None
        };
        if bioavailability_coverage > 0 && bioavailability_coverage < batch_count {
            warnings.push(format!(
                "nutrient {nutrient_id} has bioavailability estimates for {bioavailability_coverage}/{batch_count} batches; total intentionally left unknown"
            ));
        }
        let bio_low: Vec<Option<f64>> = nutrient_rows
            .iter()
            .map(|row| number_field(&row.receipt, "bioavailable_nutrient_mass_delivered_interval_lower_mg"))
            .collect();
        let bio_high: Vec<Option<f64>> = nutrient_rows
            .iter()
            .map(|row| number_field(&row.receipt, "bioavailable_nutrient_mass_delivered_interval_upper_mg"))
            .collect();
        let bio_interval_complete = bio_low.len() == batch_count
            && bio_high.len() == batch_count
            && bio_low.iter().all(Option::is_some)
            && bio_high.iter().all(Option::is_some);
        let bioavailable_lower = if row_accounting_complete
            && bio_interval_complete
            && batch_count > 0
        {
            sum_finite(bio_low.iter().flatten().copied())
        } else {
            None
        };
        let bioavailable_upper = if row_accounting_complete
            && bio_interval_complete
            && batch_count > 0
        {
            sum_finite(bio_high.iter().flatten().copied())
        } else {
            None
        };
        if bio_interval_complete
            && (bioavailable_lower.is_none() || bioavailable_upper.is_none())
        {
            issues.push(format!(
                "option {} nutrient {nutrient_id} bioavailable interval sum is non-finite",
                option.option_id
            ));
        }

        let direct_assay_count = nutrient_rows
            .iter()
            .filter(|row| field(&row.receipt, "concentration_is_direct_assay").and_then(Value::as_bool) == Some(true))
            .count();
        let estimated_delivery_count = nutrient_rows
            .iter()
            .filter(|row| {
                field(&row.receipt, "delivery_mass_is_estimated")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
            })
            .count();
        let water_total = resource_totals.get("water_liters").copied().flatten();
        let land_total = resource_totals.get("land_square_meters").copied().flatten();
        let energy_total = resource_totals.get("energy_kwh").copied().flatten();
        let cost_total = resource_totals.get("cost_amount").copied().flatten();
        let input0 = nutrient_rows.first().map(|row| &row.input);
        let currency = input0.and_then(|input| text_field(input, "currency_code")).map(|value| value.to_ascii_uppercase());
        let boundaries = BTreeMap::from([
            ("water".to_owned(), resource_boundaries.get("water_boundary").cloned().flatten()),
            ("land".to_owned(), resource_boundaries.get("land_boundary").cloned().flatten()),
            ("energy".to_owned(), resource_boundaries.get("energy_boundary").cloned().flatten()),
            ("cost".to_owned(), resource_boundaries.get("cost_boundary").cloned().flatten()),
        ]);
        nutrients.push(NutrientPortfolioTotal {
            nutrient_id: nutrient_id.clone(),
            batch_count: nutrient_rows.len(),
            direct_assay_concentration_batch_count: direct_assay_count,
            estimated_delivery_batch_count: estimated_delivery_count,
            total_delivered_mg: total_delivered,
            total_delivered_interval_lower_mg: total_lower,
            total_delivered_interval_upper_mg: total_upper,
            interval_coverage_batch_count,
            estimated_bioavailable_total_mg: bioavailable_total,
            bioavailability_coverage_batch_count: bioavailability_coverage,
            estimated_bioavailable_interval_lower_mg: bioavailable_lower,
            estimated_bioavailable_interval_upper_mg: bioavailable_upper,
            mg_per_liter_water: ratio(total_delivered, water_total),
            mg_per_square_meter_land: ratio(total_delivered, land_total),
            mg_per_kwh_energy: ratio(total_delivered, energy_total),
            mg_per_currency_unit: ratio(total_delivered, cost_total),
            currency_code: currency,
            boundary_labels: boundaries,
        });
    }

    if grouped_batches.is_empty() {
        issues.push(format!("option {} contains no valid batches", option.option_id));
    }
    let resource_ledger_complete = row_accounting_complete
        && !grouped_batches.is_empty()
        && RESOURCE_FIELDS.iter().all(|(key, _)| {
            let measured = resource_measured_counts.get(*key).copied().unwrap_or(0);
            measured == 0
                || (measured == batch_count
                    && resource_totals.get(*key).map(|value| value.is_some()).unwrap_or(false))
        });
    OptionPortfolioReceipt {
        option_id: option.option_id.clone(),
        batch_count,
        expected_nutrient_ids: expected_nutrients.to_vec(),
        nutrients,
        total_water_liters: resource_totals.get("water_liters").copied().flatten(),
        water_boundary: resource_boundaries.get("water_boundary").cloned().flatten(),
        total_land_square_meters: resource_totals.get("land_square_meters").copied().flatten(),
        land_boundary: resource_boundaries.get("land_boundary").cloned().flatten(),
        total_energy_kwh: resource_totals.get("energy_kwh").copied().flatten(),
        energy_boundary: resource_boundaries.get("energy_boundary").cloned().flatten(),
        total_cost_amount: resource_totals.get("cost_amount").copied().flatten(),
        cost_boundary: resource_boundaries.get("cost_boundary").cloned().flatten(),
        currency_code: rows
            .first()
            .and_then(|row| text_field(&row.input, "currency_code"))
            .map(|value| value.to_ascii_uppercase()),
        nutrient_composition_complete,
        resource_ledger_complete,
        accounting_pass: issues.is_empty(),
        issues,
        warnings,
    }
}

fn reconcile_portfolio(input: PortfolioInput, input_sha256: String) -> NutritionPortfolioReceipt {
    let mut issues = Vec::new();
    let expected_set: BTreeSet<String> = input.expected_nutrient_ids.iter().cloned().collect();
    if input.schema_version != PORTFOLIO_SCHEMA_V0 {
        issues.push("unsupported portfolio schema_version".to_owned());
    }
    if input.portfolio_id.trim().is_empty() {
        issues.push("portfolio_id must be non-empty".to_owned());
    }
    if input.comparison_basis.trim().is_empty() {
        issues.push("comparison_basis must be explicit".to_owned());
    }
    if input.comparison_basis_evidence_uri.trim().is_empty() {
        issues.push("comparison_basis_evidence_uri must identify the study protocol or comparison evidence".to_owned());
    }
    if input.expected_nutrient_ids.is_empty()
        || input.expected_nutrient_ids.iter().any(|id| id.trim().is_empty())
        || expected_set.len() != input.expected_nutrient_ids.len()
    {
        issues.push("expected_nutrient_ids must be non-empty, unique, and non-blank".to_owned());
    }
    if input.options.len() < 2 {
        issues.push("at least two options are required for a portfolio comparison".to_owned());
    }
    let mut seen_options = BTreeSet::new();
    for option in &input.options {
        if option.option_id.trim().is_empty() || !seen_options.insert(option.option_id.clone()) {
            issues.push("option IDs must be non-empty and unique".to_owned());
        }
    }

    let mut options = input
        .options
        .iter()
        .map(|option| aggregate_option(option, &input.expected_nutrient_ids))
        .collect::<Vec<_>>();

    let boundary_key_for_resource = [
        ("water_liters", "water_boundary"),
        ("land_square_meters", "land_boundary"),
        ("energy_kwh", "energy_boundary"),
        ("cost_amount", "cost_boundary"),
    ];
    let mut comparable_resource_boundaries = BTreeMap::new();
    let mut common_resource_count = 0usize;
    for (value_key, boundary_key) in boundary_key_for_resource {
        let completeness: Vec<bool> = options.iter().map(|option| match value_key {
            "water_liters" => option.total_water_liters.is_some(),
            "land_square_meters" => option.total_land_square_meters.is_some(),
            "energy_kwh" => option.total_energy_kwh.is_some(),
            "cost_amount" => option.total_cost_amount.is_some(),
            _ => false,
        }).collect();
        let measured_any = completeness.iter().any(|value| *value);
        let measured_all = !completeness.is_empty() && completeness.iter().all(|value| *value);
        let boundaries: BTreeSet<String> = options
            .iter()
            .filter_map(|option| match boundary_key {
                "water_boundary" => option.water_boundary.clone(),
                "land_boundary" => option.land_boundary.clone(),
                "energy_boundary" => option.energy_boundary.clone(),
                "cost_boundary" => option.cost_boundary.clone(),
                _ => None,
            })
            .collect();
        let common_boundary = (measured_all && boundaries.len() == 1)
            .then(|| boundaries.iter().next().unwrap().clone());
        if measured_any && !measured_all {
            issues.push(format!(
                "resource {value_key} is incomplete across comparison options"
            ));
        }
        if measured_all && boundaries.len() != 1 {
            issues.push(format!(
                "resource {value_key} uses incompatible {boundary_key} values across options"
            ));
        }
        if common_boundary.is_some() {
            common_resource_count += 1;
        }
        comparable_resource_boundaries.insert(value_key.to_owned(), common_boundary);
    }
    let currencies: BTreeSet<String> = options
        .iter()
        .filter_map(|option| option.currency_code.clone())
        .collect();
    let any_cost = options.iter().any(|option| option.total_cost_amount.is_some());
    if any_cost && currencies.len() != 1 {
        issues.push("cost comparison requires one common currency across all options".to_owned());
    }
    if common_resource_count == 0 {
        issues.push("portfolio comparison has no resource measured with a common boundary across options".to_owned());
    }
    for option in &mut options {
        option.accounting_pass = option.accounting_pass && option.issues.is_empty() && common_resource_count > 0;
    }
    let mut nutrient_interval_comparison_available = BTreeMap::new();
    let mut bioavailability_comparison_available = BTreeMap::new();
    for nutrient_id in &input.expected_nutrient_ids {
        let interval_available = !options.is_empty()
            && options.iter().all(|option| {
                option.accounting_pass
                    && option.nutrient_composition_complete
                    && option
                    .nutrients
                    .iter()
                    .find(|nutrient| nutrient.nutrient_id == *nutrient_id)
                    .is_some_and(|nutrient| {
                        nutrient.batch_count == option.batch_count
                            && nutrient.interval_coverage_batch_count == option.batch_count
                            && nutrient.total_delivered_interval_lower_mg.is_some()
                            && nutrient.total_delivered_interval_upper_mg.is_some()
                    })
            });
        let bioavailability_available = !options.is_empty()
            && options.iter().all(|option| {
                option.accounting_pass
                    && option.nutrient_composition_complete
                    && option
                    .nutrients
                    .iter()
                    .find(|nutrient| nutrient.nutrient_id == *nutrient_id)
                    .is_some_and(|nutrient| {
                        nutrient.batch_count == option.batch_count
                            && nutrient.bioavailability_coverage_batch_count == option.batch_count
                            && nutrient.estimated_bioavailable_total_mg.is_some()
                    })
            });
        nutrient_interval_comparison_available
            .insert(nutrient_id.clone(), interval_available);
        bioavailability_comparison_available
            .insert(nutrient_id.clone(), bioavailability_available);
    }
    let accounting_pass = issues.is_empty()
        && options.iter().all(|option| option.accounting_pass);
    NutritionPortfolioReceipt {
        receipt_schema: "cropfoundry-nutrition-portfolio-receipt-v0".to_owned(),
        input_sha256,
        portfolio_id: input.portfolio_id,
        comparison_basis: input.comparison_basis,
        comparison_basis_evidence_uri: input.comparison_basis_evidence_uri,
        expected_nutrient_ids: input.expected_nutrient_ids,
        option_count: options.len(),
        common_resource_count,
        comparable_resource_boundaries,
        nutrient_interval_comparison_available,
        bioavailability_comparison_available,
        accounting_pass,
        dietary_adequacy_evaluated: false,
        food_safety_validated: false,
        recommendation_emitted: false,
        training_authorized: false,
        options,
        issues,
        caveats: vec![
            "Nutrient composition multiplied by edible mass is an accounting calculation, not a health outcome or diet recommendation.".to_owned(),
            "Resource amounts are summed once per batch, not once per nutrient record; the caller must ensure batch resource ledgers are non-overlapping.".to_owned(),
            "All compared options must use the same declared comparison basis and resource measurement boundaries.".to_owned(),
            "Missing nutrient records fail composition completeness rather than being treated as zero nutrient output.".to_owned(),
            "Bioavailability totals remain unknown when evidence/values are missing for any batch; they are not assumed equal to total nutrient mass.".to_owned(),
            "Uncertainty bounds are summed only when each batch provides consistent lower/upper delivered-mass bounds; no statistical interval coverage is inferred.".to_owned(),
            "The receipt does not evaluate dietary adequacy, affordability for a household, safety, clinical outcomes, or causal treatment effects.".to_owned(),
            "No option ranking, recommendation, or training authorization is emitted.".to_owned(),
        ],
    }
}

/// Reconcile multiple crop-option accounting inputs without double-counting shared batch resources.
pub fn run(input_path: PathBuf, out: Option<PathBuf>) -> Result<()> {
    let raw = fs::read(&input_path)
        .with_context(|| format!("cannot read nutrition portfolio input {}", input_path.display()))?;
    let input: PortfolioInput = serde_json::from_slice(&raw)
        .context("input must be a cropfoundry-nutrition-portfolio-input-v0 JSON object")?;
    let receipt = reconcile_portfolio(input, format!("{:x}", Sha256::digest(&raw)));
    let rendered = serde_json::to_string_pretty(&receipt)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("cannot create output directory {}", parent.display()))?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .with_context(|| format!("cannot write nutrition portfolio receipt {}", path.display()))?;
    }
    if receipt.accounting_pass {
        Ok(())
    } else {
        Err(anyhow!(
            "nutrition portfolio reconciliation failed; receipt emitted/saved and missing nutrient/resource values were not imputed"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(batch: &str, nutrient: &str, concentration: f64, source_sha: &str) -> Value {
        serde_json::json!({
            "schema_version": "cropfoundry-nutrition-accounting-input-v1",
            "batch_id": batch,
            "nutrient_id": nutrient,
            "concentration_mg_per_kg": concentration,
            "concentration_mass_basis": "fresh_edible_mass",
            "concentration_stage": "measured_at_serving",
            "concentration_provenance": "direct_assay",
            "concentration_evidence_uri": format!("urn:synthetic-assay:{batch}:{nutrient}"),
            "assay_method": "synthetic-fixture-only",
            "edible_mass_kg_at_concentration_stage": 0.1,
            "edible_mass_basis_at_concentration_stage": "fresh_edible_mass",
            "water_liters": 10.0,
            "water_boundary": "irrigation_applied",
            "land_square_meters": 1.0,
            "land_boundary": "allocated_cultivated_area",
            "source_sha256": source_sha,
        })
    }

    fn option(option_id: &str, iron_concentration: f64, zinc_concentration: f64) -> PortfolioOptionInput {
        PortfolioOptionInput {
            option_id: option_id.to_owned(),
            records: vec![
                record(
                    &format!("{option_id}-batch-1"),
                    "iron",
                    iron_concentration,
                    &"a".repeat(64),
                ),
                record(
                    &format!("{option_id}-batch-1"),
                    "zinc",
                    zinc_concentration,
                    &"b".repeat(64),
                ),
                record(
                    &format!("{option_id}-batch-2"),
                    "iron",
                    iron_concentration,
                    &"c".repeat(64),
                ),
                record(
                    &format!("{option_id}-batch-2"),
                    "zinc",
                    zinc_concentration,
                    &"d".repeat(64),
                ),
            ],
        }
    }

    fn portfolio(options: Vec<PortfolioOptionInput>) -> NutritionPortfolioReceipt {
        reconcile_portfolio(
            PortfolioInput {
                schema_version: PORTFOLIO_SCHEMA_V0.to_owned(),
                portfolio_id: "synthetic-portfolio".to_owned(),
                comparison_basis: "same synthetic site-season unit".to_owned(),
                comparison_basis_evidence_uri: "urn:synthetic-comparison-protocol".to_owned(),
                expected_nutrient_ids: vec!["iron".to_owned(), "zinc".to_owned()],
                options,
            },
            "e".repeat(64),
        )
    }

    fn add_sensitivity_bounds(option: &mut PortfolioOptionInput) {
        for record in &mut option.records {
            let concentration = record["concentration_mg_per_kg"].as_f64().unwrap();
            record["concentration_lower_mg_per_kg"] =
                serde_json::json!(concentration * 0.9);
            record["concentration_upper_mg_per_kg"] =
                serde_json::json!(concentration * 1.1);
            record["edible_mass_lower_kg_at_concentration_stage"] = serde_json::json!(0.09);
            record["edible_mass_upper_kg_at_concentration_stage"] = serde_json::json!(0.11);
            record["uncertainty_interval_kind"] = serde_json::json!("sensitivity_bounds");
            record["uncertainty_evidence_uri"] =
                serde_json::json!("urn:synthetic-interval-method");
        }
    }

    #[test]
    fn cropfoundry_portfolio_checked_in_fixture_has_reproducible_totals() {
        let raw = include_str!("../../docs/research/cropfoundry/nutrition-portfolio-example-v0.json");
        let input: PortfolioInput = serde_json::from_str(raw).unwrap();
        let receipt = reconcile_portfolio(input, format!("{:x}", Sha256::digest(raw.as_bytes())));
        assert!(receipt.accounting_pass, "{:?}", receipt.issues);
        assert_eq!(receipt.common_resource_count, 3);
        let option_a = receipt
            .options
            .iter()
            .find(|option| option.option_id == "leafy-option-a")
            .unwrap();
        assert_eq!(option_a.total_water_liters, Some(22.0));
        let iron = option_a
            .nutrients
            .iter()
            .find(|nutrient| nutrient.nutrient_id == "iron")
            .unwrap();
        assert_eq!(iron.total_delivered_mg, Some(19.0));
        assert_eq!(receipt.nutrient_interval_comparison_available["iron"], false);
        assert_eq!(receipt.bioavailability_comparison_available["iron"], false);
        assert!((iron.mg_per_liter_water.unwrap() - 19.0 / 22.0).abs() < 1e-9);
        assert!(iron.estimated_bioavailable_total_mg.is_none());
        assert_eq!(iron.bioavailability_coverage_batch_count, 0);
    }

    #[test]
    fn cropfoundry_portfolio_schema_locks_unknown_top_level_fields() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../docs/research/cropfoundry/nutrition-portfolio-input-v0.schema.json"
        ))
        .unwrap();
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["options"]["items"]["properties"]["records"]["items"]["$ref"],
            "nutrition-accounting-input-v1.schema.json"
        );

        let mut input: Value = serde_json::from_str(include_str!(
            "../../docs/research/cropfoundry/nutrition-portfolio-example-v0.json"
        ))
        .unwrap();
        input["unreviewed_field"] = serde_json::json!(true);
        assert!(serde_json::from_value::<PortfolioInput>(input).is_err());
    }

    #[test]
    fn cropfoundry_portfolio_keeps_partial_resource_ledgers_incomparable() {
        let mut crop_b = option("crop-b", 80.0, 30.0);
        for record in &mut crop_b.records {
            record.as_object_mut().unwrap().remove("water_liters");
            record.as_object_mut().unwrap().remove("water_boundary");
        }
        let receipt = portfolio(vec![option("crop-a", 100.0, 20.0), crop_b]);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("water_liters is incomplete"))
        );
    }

    #[test]
    fn cropfoundry_portfolio_sums_only_complete_nutrient_intervals() {
        let mut crop_a = option("crop-a", 100.0, 20.0);
        let mut crop_b = option("crop-b", 80.0, 30.0);
        add_sensitivity_bounds(&mut crop_a);
        add_sensitivity_bounds(&mut crop_b);
        let receipt = portfolio(vec![crop_a, crop_b]);
        assert!(receipt.accounting_pass, "{:?}", receipt.issues);
        let iron = receipt.options[0]
            .nutrients
            .iter()
            .find(|nutrient| nutrient.nutrient_id == "iron")
            .unwrap();
        assert_eq!(iron.total_delivered_interval_lower_mg, Some(16.2));
        assert_eq!(iron.total_delivered_interval_upper_mg, Some(24.2));
        assert_eq!(receipt.nutrient_interval_comparison_available["iron"], true);
    }

    #[test]
    fn cropfoundry_portfolio_does_not_claim_interval_comparability_for_point_only_option() {
        let mut crop_a = option("crop-a", 100.0, 20.0);
        add_sensitivity_bounds(&mut crop_a);
        let receipt = portfolio(vec![crop_a, option("crop-b", 80.0, 30.0)]);
        assert!(receipt.accounting_pass, "{:?}", receipt.issues);
        assert_eq!(receipt.nutrient_interval_comparison_available["iron"], false);
        assert_eq!(receipt.nutrient_interval_comparison_available["zinc"], false);
    }

    #[test]
    fn cropfoundry_portfolio_sums_resources_once_per_batch_not_once_per_nutrient() {
        let receipt = portfolio(vec![option("crop-a", 100.0, 20.0), option("crop-b", 80.0, 30.0)]);
        assert!(receipt.accounting_pass, "{:?}", receipt.issues);
        assert_eq!(receipt.common_resource_count, 2);
        let crop_a = receipt.options.iter().find(|item| item.option_id == "crop-a").unwrap();
        assert_eq!(crop_a.batch_count, 2);
        assert_eq!(crop_a.total_water_liters, Some(20.0));
        let iron = crop_a.nutrients.iter().find(|item| item.nutrient_id == "iron").unwrap();
        assert_eq!(iron.total_delivered_mg, Some(20.0));
        assert_eq!(iron.mg_per_liter_water, Some(1.0));
    }

    #[test]
    fn cropfoundry_portfolio_rejects_missing_nutrients_instead_of_imputing_zero() {
        let mut incomplete = option("crop-a", 100.0, 20.0);
        incomplete.records.retain(|record| record["nutrient_id"] != "zinc" || record["batch_id"] != "crop-a-batch-2");
        let receipt = portfolio(vec![incomplete, option("crop-b", 80.0, 30.0)]);
        assert!(!receipt.accounting_pass);
        assert!(receipt.options[0].issues.iter().any(|issue| issue.contains("nutrient coverage")));
    }

    #[test]
    fn cropfoundry_portfolio_withholds_totals_when_one_row_fails() {
        let mut crop_a = option("crop-a", 100.0, 20.0);
        crop_a.records[1]["concentration_mg_per_kg"] = serde_json::json!(-2.0);
        let receipt = portfolio(vec![crop_a, option("crop-b", 80.0, 30.0)]);
        assert!(!receipt.accounting_pass);
        let iron = receipt.options[0]
            .nutrients
            .iter()
            .find(|nutrient| nutrient.nutrient_id == "iron")
            .unwrap();
        let zinc = receipt.options[0]
            .nutrients
            .iter()
            .find(|nutrient| nutrient.nutrient_id == "zinc")
            .unwrap();
        assert_eq!(iron.total_delivered_mg, None);
        assert_eq!(zinc.total_delivered_mg, None);
        assert!(!receipt.options[0].resource_ledger_complete);
        assert_eq!(receipt.nutrient_interval_comparison_available["iron"], false);
    }

    #[test]
    fn cropfoundry_portfolio_rejects_resource_duplication_mismatch_within_batch() {
        let mut crop = option("crop-a", 100.0, 20.0);
        crop.records[1]["water_liters"] = serde_json::json!(99.0);
        let receipt = portfolio(vec![crop, option("crop-b", 80.0, 30.0)]);
        assert!(!receipt.accounting_pass);
        assert!(receipt.options[0].issues.iter().any(|issue| issue.contains("inconsistent water_liters")));
    }

    #[test]
    fn cropfoundry_portfolio_requires_resource_boundaries_to_match_across_options() {
        let mut crop_b = option("crop-b", 80.0, 30.0);
        crop_b.records.iter_mut().for_each(|record| {
            record["water_boundary"] = serde_json::json!("total_applied_water");
        });
        let receipt = portfolio(vec![option("crop-a", 100.0, 20.0), crop_b]);
        assert!(!receipt.accounting_pass);
        assert!(receipt.issues.iter().any(|issue| issue.contains("incompatible water_boundary")));
    }

    #[test]
    fn cropfoundry_portfolio_keeps_missing_bioavailability_unknown() {
        let receipt = portfolio(vec![option("crop-a", 100.0, 20.0), option("crop-b", 80.0, 30.0)]);
        assert!(receipt.accounting_pass);
        assert!(
            receipt.options.iter().flat_map(|option| &option.nutrients)
                .all(|nutrient| nutrient.estimated_bioavailable_total_mg.is_none())
        );
    }

    #[test]
    fn cropfoundry_portfolio_counts_preparation_stage_delivery_as_estimated() {
        let mut crop = option("crop-a", 100.0, 20.0);
        crop.records[0]["concentration_stage"] = serde_json::json!("measured_before_preparation");
        crop.records[0]["retention_fraction"] = serde_json::json!(0.8);
        crop.records[0]["retention_evidence_uri"] =
            serde_json::json!("urn:synthetic-retention-factor");
        let receipt = portfolio(vec![crop, option("crop-b", 80.0, 30.0)]);
        assert!(receipt.accounting_pass, "{:?}", receipt.issues);
        let iron = receipt.options[0]
            .nutrients
            .iter()
            .find(|nutrient| nutrient.nutrient_id == "iron")
            .unwrap();
        assert_eq!(iron.direct_assay_concentration_batch_count, 2);
        assert_eq!(iron.estimated_delivery_batch_count, 1);
    }

    #[test]
    fn cropfoundry_portfolio_sum_overflow_fails_closed() {
        let mut crop_a = option("crop-a", 100.0, 20.0);
        for record in &mut crop_a.records {
            if record["nutrient_id"] == "iron" {
                record["concentration_mg_per_kg"] = serde_json::json!(1.7e308);
                record["edible_mass_kg_at_concentration_stage"] = serde_json::json!(1.0);
            }
        }
        let receipt = portfolio(vec![crop_a, option("crop-b", 80.0, 30.0)]);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt.options[0]
                .issues
                .iter()
                .any(|issue| issue.contains("total overflowed"))
        );
    }

    #[test]
    fn cropfoundry_portfolio_resource_sum_overflow_fails_closed() {
        let mut crop_a = option("crop-a", 100.0, 20.0);
        for record in &mut crop_a.records {
            record["water_liters"] = serde_json::json!(1.0e308);
        }
        let receipt = portfolio(vec![crop_a, option("crop-b", 80.0, 30.0)]);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt.options[0]
                .issues
                .iter()
                .any(|issue| issue.contains("resource total water_liters overflowed"))
        );
        assert_eq!(receipt.options[0].total_water_liters, None);
    }

    #[test]
    fn cropfoundry_portfolio_checks_resource_currency_comparability() {
        let mut crop_b = option("crop-b", 80.0, 30.0);
        crop_b.records.iter_mut().for_each(|record| {
            record["cost_amount"] = serde_json::json!(2.0);
            record["cost_boundary"] = serde_json::json!("farm_gate");
            record["currency_code"] = serde_json::json!("USD");
        });
        let mut crop_a = option("crop-a", 100.0, 20.0);
        crop_a.records.iter_mut().for_each(|record| {
            record["cost_amount"] = serde_json::json!(2.0);
            record["cost_boundary"] = serde_json::json!("farm_gate");
            record["currency_code"] = serde_json::json!("ZAR");
        });
        let receipt = portfolio(vec![crop_a, crop_b]);
        assert!(!receipt.accounting_pass);
        assert!(receipt.issues.iter().any(|issue| issue.contains("common currency")));
    }
}
