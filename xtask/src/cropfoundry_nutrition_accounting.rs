use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MassBasis {
    FreshEdibleMass,
    DryEdibleMass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ConcentrationStage {
    MeasuredAtServing,
    MeasuredBeforePreparation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ConcentrationProvenance {
    DirectAssay,
    LiteratureValue,
    ModelEstimate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NutritionAccountingInput {
    schema_version: String,
    batch_id: String,
    nutrient_id: String,
    concentration_mg_per_kg: Option<f64>,
    concentration_mass_basis: Option<MassBasis>,
    concentration_stage: Option<ConcentrationStage>,
    concentration_provenance: ConcentrationProvenance,
    concentration_evidence_uri: Option<String>,
    assay_method: Option<String>,
    served_edible_mass_kg: Option<f64>,
    served_mass_basis: Option<MassBasis>,
    retention_fraction: Option<f64>,
    retention_evidence_uri: Option<String>,
    bioavailability_fraction: Option<f64>,
    bioavailability_evidence_uri: Option<String>,
    water_liters: Option<f64>,
    land_square_meters: Option<f64>,
    energy_kwh: Option<f64>,
    cost_amount: Option<f64>,
    currency_code: Option<String>,
    source_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct NutritionAccountingReceipt {
    receipt_schema: String,
    batch_id: String,
    nutrient_id: String,
    accounting_pass: bool,
    direct_assay: bool,
    nutrient_mass_at_serving_mg: Option<f64>,
    modeled_nutrient_mass_at_serving_mg: Option<f64>,
    bioavailable_nutrient_mass_mg: Option<f64>,
    nutrient_mg_per_liter_water: Option<f64>,
    nutrient_mg_per_square_meter: Option<f64>,
    nutrient_mg_per_kwh: Option<f64>,
    nutrient_mg_per_currency_unit: Option<f64>,
    currency_code: Option<String>,
    issues: Vec<String>,
    caveats: Vec<String>,
}

fn valid_fraction(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    numerator.zip(denominator).and_then(|(n, d)| {
        (n.is_finite() && d.is_finite() && n >= 0.0 && d > 0.0).then_some(n / d)
    })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn audit_input(input: NutritionAccountingInput) -> NutritionAccountingReceipt {
    let mut issues = Vec::new();
    let concentration = input.concentration_mg_per_kg;
    let serving_mass = input.served_edible_mass_kg;

    if input.schema_version != "cropfoundry-nutrition-accounting-input-v0" {
        issues.push("unsupported input schema_version".to_owned());
    }
    if input.batch_id.trim().is_empty() {
        issues.push("batch_id must be non-empty".to_owned());
    }
    if input.nutrient_id.trim().is_empty() {
        issues.push("nutrient_id must be non-empty".to_owned());
    }
    if concentration.is_none_or(|value| !valid_nonnegative(value)) {
        issues.push("concentration_mg_per_kg must be present, finite and non-negative".to_owned());
    }
    if serving_mass.is_none_or(|value| !value.is_finite() || value <= 0.0) {
        issues.push("served_edible_mass_kg must be present, finite and greater than zero".to_owned());
    }
    if input.concentration_mass_basis.is_none() || input.served_mass_basis.is_none() {
        issues.push("concentration and serving mass bases must be explicit".to_owned());
    } else if input.concentration_mass_basis != input.served_mass_basis {
        issues.push("concentration mass basis does not match served mass basis".to_owned());
    }
    if input
        .retention_fraction
        .is_some_and(|value| !valid_fraction(value))
    {
        issues.push("retention_fraction must be finite and in [0,1]".to_owned());
    }
    if input
        .bioavailability_fraction
        .is_some_and(|value| !valid_fraction(value))
    {
        issues.push("bioavailability_fraction must be finite and in [0,1]".to_owned());
    }
    if input.concentration_evidence_uri.as_deref().is_none_or(str::is_empty) {
        issues.push("concentration_evidence_uri must be explicit".to_owned());
    }
    if input.concentration_provenance == ConcentrationProvenance::DirectAssay
        && input.assay_method.as_deref().is_none_or(str::is_empty)
    {
        issues.push("direct assay provenance requires assay_method".to_owned());
    }
    if input.concentration_provenance == ConcentrationProvenance::DirectAssay
        && input.source_sha256.as_deref().is_none_or(|digest| !valid_sha256(digest))
    {
        issues.push("direct assay provenance requires a 64-character source_sha256".to_owned());
    }
    if input.concentration_stage == Some(ConcentrationStage::MeasuredAtServing)
        && input.retention_fraction.is_some()
    {
        issues.push("retention_fraction must be absent when concentration was measured at serving".to_owned());
    }
    if input.concentration_stage == Some(ConcentrationStage::MeasuredBeforePreparation)
        && input.retention_fraction.is_none()
    {
        issues.push("pre-preparation concentration requires an explicit retention_fraction".to_owned());
    }
    if input.retention_fraction.is_some()
        && input.retention_evidence_uri.as_deref().is_none_or(str::is_empty)
    {
        issues.push("retention_fraction requires its own evidence URI".to_owned());
    }
    if input.bioavailability_fraction.is_some()
        && input.bioavailability_evidence_uri.as_deref().is_none_or(str::is_empty)
    {
        issues.push("bioavailability_fraction requires its own evidence URI".to_owned());
    }

    for (label, value) in [
        ("water_liters", input.water_liters),
        ("land_square_meters", input.land_square_meters),
        ("energy_kwh", input.energy_kwh),
        ("cost_amount", input.cost_amount),
    ] {
        if value.is_some_and(|number| !valid_nonnegative(number)) {
            issues.push(format!("{label} must be finite and non-negative"));
        }
    }
    if input.cost_amount.is_some()
        && input.currency_code.as_deref().is_none_or(|value| value.trim().is_empty())
    {
        issues.push("cost_amount requires an explicit currency_code".to_owned());
    }
    if input
        .currency_code
        .as_deref()
        .is_some_and(|value| value.trim().len() != 3 || !value.trim().bytes().all(|b| b.is_ascii_alphabetic()))
    {
        issues.push("currency_code must be a three-letter alphabetic code".to_owned());
    }

    let accounting_pass = issues.is_empty();
    let direct_assay = input.concentration_provenance == ConcentrationProvenance::DirectAssay;
    let pre_preparation_mass = if accounting_pass {
        concentration.zip(serving_mass).map(|(c, m)| c * m)
    } else {
        None
    };
    let measured_or_modeled_mass = match input.concentration_stage {
        Some(ConcentrationStage::MeasuredAtServing) if accounting_pass => pre_preparation_mass,
        Some(ConcentrationStage::MeasuredBeforePreparation) if accounting_pass => {
            pre_preparation_mass.zip(input.retention_fraction).map(|(mass, retention)| mass * retention)
        }
        _ => None,
    };
    let bioavailable_mass = measured_or_modeled_mass
        .zip(input.bioavailability_fraction)
        .map(|(mass, fraction)| mass * fraction);

    NutritionAccountingReceipt {
        receipt_schema: "cropfoundry-nutrition-accounting-receipt-v0".to_owned(),
        batch_id: input.batch_id,
        nutrient_id: input.nutrient_id,
        accounting_pass,
        direct_assay,
        nutrient_mass_at_serving_mg: if direct_assay
            && input.concentration_stage == Some(ConcentrationStage::MeasuredAtServing)
        {
            measured_or_modeled_mass
        } else {
            None
        },
        modeled_nutrient_mass_at_serving_mg: if !direct_assay || input.concentration_stage == Some(ConcentrationStage::MeasuredBeforePreparation) {
            measured_or_modeled_mass
        } else {
            None
        },
        bioavailable_nutrient_mass_mg: bioavailable_mass,
        nutrient_mg_per_liter_water: ratio(measured_or_modeled_mass, input.water_liters),
        nutrient_mg_per_square_meter: ratio(measured_or_modeled_mass, input.land_square_meters),
        nutrient_mg_per_kwh: ratio(measured_or_modeled_mass, input.energy_kwh),
        nutrient_mg_per_currency_unit: ratio(measured_or_modeled_mass, input.cost_amount),
        currency_code: input.currency_code.map(|value| value.trim().to_ascii_uppercase()),
        issues,
        caveats: vec![
            "Concentration x edible mass is an accounting calculation, not a dietary adequacy or health outcome.".to_owned(),
            "Literature/modelled composition and retention values remain estimates, even when the arithmetic passes.".to_owned(),
            "Missing bioavailability evidence yields no bioavailable-mass estimate; it is not assumed to equal nutrient mass.".to_owned(),
            "Resource ratios are only comparable with matching system boundaries, units, crop stage and allocation rules.".to_owned(),
            "Food safety, dietary requirements, upper limits and population suitability are not evaluated.".to_owned(),
            "This receipt does not authorize model training or farm recommendations.".to_owned(),
        ],
    }
}

/// Validate one explicit JSON accounting record; no dataset is downloaded or trained on.
pub fn run(input_path: PathBuf, out: Option<PathBuf>) -> Result<()> {
    let raw = fs::read(&input_path)
        .with_context(|| format!("cannot read nutrition-accounting input {}", input_path.display()))?;
    let input: NutritionAccountingInput = serde_json::from_slice(&raw)
        .context("input must be a cropfoundry-nutrition-accounting-input-v0 JSON object")?;
    let receipt = audit_input(input);
    let rendered = serde_json::to_string_pretty(&receipt)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("cannot create output directory {}", parent.display()))?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .with_context(|| format!("cannot write nutrition receipt {}", path.display()))?;
    }
    if receipt.accounting_pass {
        Ok(())
    } else {
        Err(anyhow!("nutrition accounting contract failed; receipt emitted/saved and missing values were not imputed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> NutritionAccountingInput {
        NutritionAccountingInput {
            schema_version: "cropfoundry-nutrition-accounting-input-v0".to_owned(),
            batch_id: "batch-1".to_owned(),
            nutrient_id: "iron".to_owned(),
            concentration_mg_per_kg: Some(80.0),
            concentration_mass_basis: Some(MassBasis::FreshEdibleMass),
            concentration_stage: Some(ConcentrationStage::MeasuredAtServing),
            concentration_provenance: ConcentrationProvenance::DirectAssay,
            concentration_evidence_uri: Some("urn:assay:iron:batch-1".to_owned()),
            assay_method: Some("validated-lab-method-v1".to_owned()),
            served_edible_mass_kg: Some(0.15),
            served_mass_basis: Some(MassBasis::FreshEdibleMass),
            retention_fraction: None,
            retention_evidence_uri: None,
            bioavailability_fraction: Some(0.25),
            bioavailability_evidence_uri: Some("urn:literature:iron-bioavailability".to_owned()),
            water_liters: Some(30.0),
            land_square_meters: Some(2.0),
            energy_kwh: Some(0.5),
            cost_amount: Some(1.25),
            currency_code: Some("ZAR".to_owned()),
            source_sha256: Some("a".repeat(64)),
        }
    }

    #[test]
    fn direct_at_serving_assay_computes_nutrient_mass_and_resource_ratios() {
        let receipt = audit_input(fixture());
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mass_at_serving_mg, Some(12.0));
        assert_eq!(receipt.bioavailable_nutrient_mass_mg, Some(3.0));
        assert_eq!(receipt.nutrient_mg_per_liter_water, Some(0.4));
        assert_eq!(receipt.nutrient_mg_per_square_meter, Some(6.0));
        assert_eq!(receipt.nutrient_mg_per_kwh, Some(24.0));
        assert_eq!(receipt.nutrient_mg_per_currency_unit, Some(9.6));
    }

    #[test]
    fn missing_bioavailability_is_not_assumed_to_be_one() {
        let mut input = fixture();
        input.bioavailability_fraction = None;
        input.bioavailability_evidence_uri = None;
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mass_at_serving_mg, Some(12.0));
        assert_eq!(receipt.bioavailable_nutrient_mass_mg, None);
    }

    #[test]
    fn pre_preparation_concentration_requires_a_retention_factor_and_evidence() {
        let mut input = fixture();
        input.concentration_stage = Some(ConcentrationStage::MeasuredBeforePreparation);
        input.retention_fraction = Some(0.75);
        input.retention_evidence_uri = Some("urn:retention:study-v1".to_owned());
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mass_at_serving_mg, None);
        assert_eq!(receipt.modeled_nutrient_mass_at_serving_mg, Some(9.0));
    }

    #[test]
    fn mismatched_fresh_and_dry_mass_basis_fails_closed() {
        let mut input = fixture();
        input.concentration_mass_basis = Some(MassBasis::DryEdibleMass);
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(receipt.issues.iter().any(|issue| issue.contains("mass basis")));
        assert_eq!(receipt.nutrient_mass_at_serving_mg, None);
    }

    #[test]
    fn direct_assay_requires_method_source_and_digest() {
        let mut input = fixture();
        input.assay_method = None;
        input.concentration_evidence_uri = None;
        input.source_sha256 = None;
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(receipt.issues.len() >= 3);
    }

    #[test]
    fn impossible_retention_or_bioavailability_is_rejected() {
        let mut input = fixture();
        input.concentration_stage = Some(ConcentrationStage::MeasuredBeforePreparation);
        input.retention_fraction = Some(1.2);
        input.retention_evidence_uri = Some("urn:retention:bad".to_owned());
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(receipt.issues.iter().any(|issue| issue.contains("retention_fraction")));
    }

    #[test]
    fn unknown_resource_measurements_stay_unknown() {
        let mut input = fixture();
        input.water_liters = None;
        input.land_square_meters = None;
        input.energy_kwh = None;
        input.cost_amount = None;
        input.currency_code = None;
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mg_per_liter_water, None);
        assert_eq!(receipt.nutrient_mg_per_square_meter, None);
        assert_eq!(receipt.nutrient_mg_per_kwh, None);
        assert_eq!(receipt.nutrient_mg_per_currency_unit, None);
    }

    #[test]
    fn literature_value_stays_visibly_modelled_and_cannot_masquerade_as_assay() {
        let mut input = fixture();
        input.concentration_provenance = ConcentrationProvenance::LiteratureValue;
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert!(!receipt.direct_assay);
        assert_eq!(receipt.nutrient_mass_at_serving_mg, None);
        assert_eq!(receipt.modeled_nutrient_mass_at_serving_mg, Some(12.0));
    }
}
