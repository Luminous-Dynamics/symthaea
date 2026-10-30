use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

const INPUT_SCHEMA_V0: &str = "cropfoundry-nutrition-accounting-input-v0";
const INPUT_SCHEMA_V1: &str = "cropfoundry-nutrition-accounting-input-v1";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UncertaintyIntervalKind {
    ConfidenceInterval95,
    CredibleInterval95,
    MeasurementTolerance,
    SensitivityBounds,
    ObservedRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WaterBoundary {
    IrrigationApplied,
    TotalAppliedWater,
    CropConsumptiveUse,
    BlueWaterFootprint,
    GreenWaterFootprint,
    FullSupplyChainWaterFootprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LandBoundary {
    AllocatedCultivatedArea,
    HarvestedArea,
    FullSupplyChainLandFootprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EnergyBoundary {
    OnFarmOperations,
    IrrigationPumping,
    PostharvestProcessing,
    FullSupplyChainEnergy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CostBoundary {
    FarmGate,
    ProductionOnly,
    PostharvestProcessing,
    DeliveredToMarket,
    DeliveredToServing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    concentration_lower_mg_per_kg: Option<f64>,
    concentration_upper_mg_per_kg: Option<f64>,
    edible_mass_kg_at_concentration_stage: Option<f64>,
    edible_mass_basis_at_concentration_stage: Option<MassBasis>,
    edible_mass_lower_kg_at_concentration_stage: Option<f64>,
    edible_mass_upper_kg_at_concentration_stage: Option<f64>,
    servings_in_batch: Option<u64>,
    retention_fraction: Option<f64>,
    retention_evidence_uri: Option<String>,
    retention_lower_fraction: Option<f64>,
    retention_upper_fraction: Option<f64>,
    bioavailability_fraction: Option<f64>,
    bioavailability_evidence_uri: Option<String>,
    bioavailability_lower_fraction: Option<f64>,
    bioavailability_upper_fraction: Option<f64>,
    uncertainty_interval_kind: Option<UncertaintyIntervalKind>,
    uncertainty_evidence_uri: Option<String>,
    water_liters: Option<f64>,
    water_boundary: Option<WaterBoundary>,
    land_square_meters: Option<f64>,
    land_boundary: Option<LandBoundary>,
    energy_kwh: Option<f64>,
    energy_boundary: Option<EnergyBoundary>,
    cost_amount: Option<f64>,
    cost_boundary: Option<CostBoundary>,
    currency_code: Option<String>,
    source_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct NutritionAccountingReceipt {
    receipt_schema: String,
    input_sha256: Option<String>,
    input_snapshot: NutritionAccountingInput,
    batch_id: String,
    nutrient_id: String,
    accounting_pass: bool,
    concentration_is_direct_assay: bool,
    delivery_mass_is_estimated: bool,
    measured_nutrient_mass_delivered_mg: Option<f64>,
    modeled_nutrient_mass_delivered_mg: Option<f64>,
    estimated_bioavailable_nutrient_mass_delivered_mg: Option<f64>,
    nutrient_mass_delivered_interval_lower_mg: Option<f64>,
    nutrient_mass_delivered_interval_upper_mg: Option<f64>,
    bioavailable_nutrient_mass_delivered_interval_lower_mg: Option<f64>,
    bioavailable_nutrient_mass_delivered_interval_upper_mg: Option<f64>,
    uncertainty_interval_kind: Option<UncertaintyIntervalKind>,
    nutrient_mg_per_serving: Option<f64>,
    estimated_bioavailable_nutrient_mg_per_serving: Option<f64>,
    nutrient_mg_per_liter_water: Option<f64>,
    water_boundary: Option<WaterBoundary>,
    nutrient_mg_per_square_meter: Option<f64>,
    land_boundary: Option<LandBoundary>,
    nutrient_mg_per_kwh: Option<f64>,
    energy_boundary: Option<EnergyBoundary>,
    nutrient_mg_per_currency_unit: Option<f64>,
    cost_boundary: Option<CostBoundary>,
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
        if n.is_finite() && d.is_finite() && n >= 0.0 && d > 0.0 {
            let value = n / d;
            value.is_finite().then_some(value)
        } else {
            None
        }
    })
}

fn finite_product(values: &[f64], label: &str, issues: &mut Vec<String>) -> Option<f64> {
    let mut product = 1.0;
    for value in values {
        if !valid_nonnegative(*value) {
            issues.push(format!("{label} contains an invalid factor"));
            return None;
        }
        product *= value;
        if !product.is_finite() {
            issues.push(format!("{label} overflowed the finite numeric range"));
            return None;
        }
    }
    Some(product)
}

fn validate_bound_pair(
    label: &str,
    lower: Option<f64>,
    upper: Option<f64>,
    point: Option<f64>,
    fraction: bool,
    issues: &mut Vec<String>,
) -> bool {
    match (lower, upper) {
        (None, None) => false,
        (Some(low), Some(high)) => {
            let valid = if fraction {
                valid_fraction(low) && valid_fraction(high)
            } else {
                valid_nonnegative(low) && valid_nonnegative(high)
            };
            if !valid {
                issues.push(format!(
                    "{label} bounds must be finite and in the permitted range"
                ));
            }
            if low > high {
                issues.push(format!("{label} lower bound exceeds upper bound"));
            }
            if let Some(value) = point {
                if value.is_finite() && (value < low || value > high) {
                    issues.push(format!(
                        "{label} point estimate is outside the declared bounds"
                    ));
                }
            } else {
                issues.push(format!("{label} bounds require a point estimate"));
            }
            true
        }
        _ => {
            issues.push(format!(
                "{label} lower and upper bounds must be provided together"
            ));
            true
        }
    }
}

fn has_nonblank(value: Option<&str>) -> bool {
    value.is_some_and(|text| !text.trim().is_empty())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn audit_input(input: NutritionAccountingInput) -> NutritionAccountingReceipt {
    let input_snapshot = input.clone();
    let mut issues = Vec::new();
    let concentration = input.concentration_mg_per_kg;
    let stage_mass = input.edible_mass_kg_at_concentration_stage;

    if input.schema_version != INPUT_SCHEMA_V0 && input.schema_version != INPUT_SCHEMA_V1 {
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
    if stage_mass.is_none_or(|value| !value.is_finite() || value <= 0.0) {
        issues.push(
            "edible_mass_kg_at_concentration_stage must be present, finite and greater than zero"
                .to_owned(),
        );
    }
    if input.concentration_stage.is_none() {
        issues.push("concentration_stage must be explicit".to_owned());
    }
    if input.concentration_mass_basis.is_none()
        || input.edible_mass_basis_at_concentration_stage.is_none()
    {
        issues.push("concentration and stage-mass bases must be explicit".to_owned());
    } else if input.concentration_mass_basis != input.edible_mass_basis_at_concentration_stage {
        issues.push(
            "concentration mass basis does not match edible mass basis at the concentration stage"
                .to_owned(),
        );
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
    if !has_nonblank(input.concentration_evidence_uri.as_deref()) {
        issues.push("concentration_evidence_uri must be explicit".to_owned());
    }
    if input.concentration_provenance == ConcentrationProvenance::DirectAssay
        && !has_nonblank(input.assay_method.as_deref())
    {
        issues.push("direct assay provenance requires assay_method".to_owned());
    }
    if input.concentration_provenance == ConcentrationProvenance::DirectAssay
        && input
            .source_sha256
            .as_deref()
            .is_none_or(|digest| !valid_sha256(digest))
    {
        issues.push("direct assay provenance requires a 64-character source_sha256".to_owned());
    }
    if input.concentration_stage == Some(ConcentrationStage::MeasuredAtServing)
        && input.retention_fraction.is_some()
    {
        issues.push(
            "retention_fraction must be absent when concentration was measured at serving"
                .to_owned(),
        );
    }
    if input.concentration_stage == Some(ConcentrationStage::MeasuredBeforePreparation)
        && input.retention_fraction.is_none()
    {
        issues.push(
            "pre-preparation concentration requires an explicit retention_fraction".to_owned(),
        );
    }
    if input.servings_in_batch == Some(0) {
        issues.push("servings_in_batch must be greater than zero when supplied".to_owned());
    }

    let concentration_bounds = validate_bound_pair(
        "concentration",
        input.concentration_lower_mg_per_kg,
        input.concentration_upper_mg_per_kg,
        concentration,
        false,
        &mut issues,
    );
    let edible_mass_bounds = validate_bound_pair(
        "edible mass",
        input.edible_mass_lower_kg_at_concentration_stage,
        input.edible_mass_upper_kg_at_concentration_stage,
        stage_mass,
        false,
        &mut issues,
    );
    let retention_bounds = validate_bound_pair(
        "retention_fraction",
        input.retention_lower_fraction,
        input.retention_upper_fraction,
        input.retention_fraction,
        true,
        &mut issues,
    );
    let bioavailability_bounds = validate_bound_pair(
        "bioavailability_fraction",
        input.bioavailability_lower_fraction,
        input.bioavailability_upper_fraction,
        input.bioavailability_fraction,
        true,
        &mut issues,
    );
    let any_intervals =
        concentration_bounds || edible_mass_bounds || retention_bounds || bioavailability_bounds;
    if input.schema_version == INPUT_SCHEMA_V0
        && (any_intervals
            || input.uncertainty_interval_kind.is_some()
            || has_nonblank(input.uncertainty_evidence_uri.as_deref()))
    {
        issues.push(
            "input schema v0 does not support uncertainty interval fields; use schema v1"
                .to_owned(),
        );
    }
    if any_intervals {
        if input.uncertainty_interval_kind.is_none()
            || !has_nonblank(input.uncertainty_evidence_uri.as_deref())
        {
            issues.push(
                "uncertainty bounds require an explicit interval kind and evidence URI".to_owned(),
            );
        }
        if !concentration_bounds || !edible_mass_bounds {
            issues.push(
                "nutrient-mass intervals require paired concentration and edible-mass bounds"
                    .to_owned(),
            );
        }
        match input.concentration_stage {
            Some(ConcentrationStage::MeasuredAtServing) if retention_bounds => {
                issues.push(
                    "retention bounds must be absent when concentration was measured at serving"
                        .to_owned(),
                );
            }
            Some(ConcentrationStage::MeasuredBeforePreparation) if !retention_bounds => {
                issues.push(
                    "pre-preparation nutrient-mass intervals require retention bounds".to_owned(),
                );
            }
            _ => {}
        }
    } else if input.uncertainty_interval_kind.is_some()
        || has_nonblank(input.uncertainty_evidence_uri.as_deref())
    {
        issues.push(
            "uncertainty interval metadata was provided without any numeric bounds".to_owned(),
        );
    }
    if input.retention_fraction.is_some() && !has_nonblank(input.retention_evidence_uri.as_deref())
    {
        issues.push("retention_fraction requires its own evidence URI".to_owned());
    }
    if input.bioavailability_fraction.is_some()
        && !has_nonblank(input.bioavailability_evidence_uri.as_deref())
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
        && input
            .currency_code
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
    {
        issues.push("cost_amount requires an explicit currency_code".to_owned());
    }

    for (label, measurement_present, boundary_present) in [
        (
            "water_liters",
            input.water_liters.is_some(),
            input.water_boundary.is_some(),
        ),
        (
            "land_square_meters",
            input.land_square_meters.is_some(),
            input.land_boundary.is_some(),
        ),
        (
            "energy_kwh",
            input.energy_kwh.is_some(),
            input.energy_boundary.is_some(),
        ),
        (
            "cost_amount",
            input.cost_amount.is_some(),
            input.cost_boundary.is_some(),
        ),
    ] {
        if measurement_present != boundary_present {
            issues.push(format!(
                "{label} and its resource boundary must be supplied together"
            ));
        }
    }
    if input.currency_code.as_deref().is_some_and(|value| {
        value.trim().len() != 3 || !value.trim().bytes().all(|b| b.is_ascii_alphabetic())
    }) {
        issues.push("currency_code must be a three-letter alphabetic code".to_owned());
    }

    let mut accounting_pass = issues.is_empty();
    let direct_assay = input.concentration_provenance == ConcentrationProvenance::DirectAssay;
    let delivery_mass_is_estimated = !direct_assay
        || input.concentration_stage == Some(ConcentrationStage::MeasuredBeforePreparation);
    let nutrient_mass_at_concentration_stage = if accounting_pass {
        concentration.zip(stage_mass).and_then(|(c, mass)| {
            finite_product(
                &[c, mass],
                "nutrient mass at concentration stage",
                &mut issues,
            )
        })
    } else {
        None
    };
    let delivered_nutrient_mass = match input.concentration_stage {
        Some(ConcentrationStage::MeasuredAtServing) if accounting_pass => {
            nutrient_mass_at_concentration_stage
        }
        Some(ConcentrationStage::MeasuredBeforePreparation) if accounting_pass => {
            nutrient_mass_at_concentration_stage
                .zip(input.retention_fraction)
                .and_then(|(mass, retention)| {
                    finite_product(&[mass, retention], "nutrient mass at serving", &mut issues)
                })
        }
        _ => None,
    };
    let bioavailable_mass = delivered_nutrient_mass
        .zip(input.bioavailability_fraction)
        .and_then(|(mass, fraction)| {
            finite_product(
                &[mass, fraction],
                "estimated bioavailable nutrient mass",
                &mut issues,
            )
        });

    let delivered_interval = if accounting_pass {
        match (
            input.concentration_lower_mg_per_kg,
            input.concentration_upper_mg_per_kg,
            input.edible_mass_lower_kg_at_concentration_stage,
            input.edible_mass_upper_kg_at_concentration_stage,
        ) {
            (
                Some(concentration_low),
                Some(concentration_high),
                Some(mass_low),
                Some(mass_high),
            ) => match input.concentration_stage {
                Some(ConcentrationStage::MeasuredAtServing) => {
                    let lower = finite_product(
                        &[concentration_low, mass_low],
                        "lower delivered nutrient-mass bound",
                        &mut issues,
                    );
                    let upper = finite_product(
                        &[concentration_high, mass_high],
                        "upper delivered nutrient-mass bound",
                        &mut issues,
                    );
                    lower.zip(upper)
                }
                Some(ConcentrationStage::MeasuredBeforePreparation) => {
                    if let (Some(retention_low), Some(retention_high)) = (
                        input.retention_lower_fraction,
                        input.retention_upper_fraction,
                    ) {
                        let lower = finite_product(
                            &[concentration_low, mass_low, retention_low],
                            "lower delivered nutrient-mass bound",
                            &mut issues,
                        );
                        let upper = finite_product(
                            &[concentration_high, mass_high, retention_high],
                            "upper delivered nutrient-mass bound",
                            &mut issues,
                        );
                        lower.zip(upper)
                    } else {
                        None
                    }
                }
                None => None,
            },
            _ => None,
        }
    } else {
        None
    };
    let bioavailable_interval = delivered_interval.and_then(|(low, high)| {
        input
            .bioavailability_lower_fraction
            .zip(input.bioavailability_upper_fraction)
            .and_then(|(factor_low, factor_high)| {
                let lower = finite_product(
                    &[low, factor_low],
                    "lower bioavailable nutrient-mass bound",
                    &mut issues,
                );
                let upper = finite_product(
                    &[high, factor_high],
                    "upper bioavailable nutrient-mass bound",
                    &mut issues,
                );
                lower.zip(upper)
            })
    });
    accounting_pass = accounting_pass && issues.is_empty();
    let nutrient_per_serving = delivered_nutrient_mass
        .zip(input.servings_in_batch)
        .map(|(mass, count)| mass / count as f64);
    let bioavailable_per_serving = bioavailable_mass
        .zip(input.servings_in_batch)
        .map(|(mass, count)| mass / count as f64);

    NutritionAccountingReceipt {
        receipt_schema: "cropfoundry-nutrition-accounting-receipt-v1".to_owned(),
        input_sha256: None,
        input_snapshot,
        batch_id: input.batch_id,
        nutrient_id: input.nutrient_id,
        accounting_pass,
        concentration_is_direct_assay: direct_assay,
        delivery_mass_is_estimated,
        measured_nutrient_mass_delivered_mg: if direct_assay
            && input.concentration_stage == Some(ConcentrationStage::MeasuredAtServing)
        {
            delivered_nutrient_mass
        } else {
            None
        },
        modeled_nutrient_mass_delivered_mg: if delivery_mass_is_estimated {
            delivered_nutrient_mass
        } else {
            None
        },
        estimated_bioavailable_nutrient_mass_delivered_mg: bioavailable_mass,
        nutrient_mass_delivered_interval_lower_mg: delivered_interval.map(|(low, _)| low),
        nutrient_mass_delivered_interval_upper_mg: delivered_interval.map(|(_, high)| high),
        bioavailable_nutrient_mass_delivered_interval_lower_mg: bioavailable_interval.map(|(low, _)| low),
        bioavailable_nutrient_mass_delivered_interval_upper_mg: bioavailable_interval.map(|(_, high)| high),
        uncertainty_interval_kind: input.uncertainty_interval_kind,
        nutrient_mg_per_serving: nutrient_per_serving,
        estimated_bioavailable_nutrient_mg_per_serving: bioavailable_per_serving,
        nutrient_mg_per_liter_water: ratio(delivered_nutrient_mass, input.water_liters),
        water_boundary: input.water_boundary,
        nutrient_mg_per_square_meter: ratio(delivered_nutrient_mass, input.land_square_meters),
        land_boundary: input.land_boundary,
        nutrient_mg_per_kwh: ratio(delivered_nutrient_mass, input.energy_kwh),
        energy_boundary: input.energy_boundary,
        nutrient_mg_per_currency_unit: ratio(delivered_nutrient_mass, input.cost_amount),
        cost_boundary: input.cost_boundary,
        currency_code: input.currency_code.map(|value| value.trim().to_ascii_uppercase()),
        issues,
        caveats: vec![
            "Concentration x edible mass is an accounting calculation, not a dietary adequacy or health outcome.".to_owned(),
            "Literature/modelled composition and retention values remain estimates, even when the arithmetic passes.".to_owned(),
            "source_sha256 is format-checked here; the CLI does not authenticate it against an independent artifact registry.".to_owned(),
            "Bioavailability-adjusted mass is always an estimate in this prototype, not a direct assay result.".to_owned(),
            "Missing bioavailability evidence yields no estimate; it is not assumed to equal nutrient mass.".to_owned(),
            "Resource ratios are only comparable with matching system boundaries, units, crop stage and allocation rules.".to_owned(),
            "Uncertainty intervals are emitted only when lower/upper bounds and their evidence are supplied; point-only inputs are not treated as zero-uncertainty.".to_owned(),
            "Interval arithmetic multiplies declared non-negative bounds conservatively; it does not certify statistical coverage unless the source interval definition supports that claim.".to_owned(),
            "Food safety, dietary requirements, upper limits and population suitability are not evaluated.".to_owned(),
            "This receipt does not authorize model training or farm recommendations.".to_owned(),
        ],
    }
}

/// Evaluate one typed accounting record for reuse by the portfolio reconciler.
pub(crate) fn audit_json_value(value: &serde_json::Value) -> Result<serde_json::Value> {
    let input: NutritionAccountingInput = serde_json::from_value(value.clone())
        .context("portfolio record is not a valid nutrition-accounting input")?;
    serde_json::to_value(audit_input(input))
        .context("cannot serialize the per-record nutrition accounting receipt")
}

/// Validate one explicit JSON accounting record; no dataset is downloaded or trained on.
pub fn run(input_path: PathBuf, out: Option<PathBuf>) -> Result<()> {
    let raw = fs::read(&input_path).with_context(|| {
        format!(
            "cannot read nutrition-accounting input {}",
            input_path.display()
        )
    })?;
    let input: NutritionAccountingInput = serde_json::from_slice(&raw)
        .context("input must be a cropfoundry-nutrition-accounting-input-v0 JSON object")?;
    let mut receipt = audit_input(input);
    receipt.input_sha256 = Some(format!("{:x}", Sha256::digest(&raw)));
    let rendered = serde_json::to_string_pretty(&receipt)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("cannot create output directory {}", parent.display()))?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .with_context(|| format!("cannot write nutrition receipt {}", path.display()))?;
    }
    if receipt.accounting_pass {
        Ok(())
    } else {
        Err(anyhow!(
            "nutrition accounting contract failed; receipt emitted/saved and missing values were not imputed"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> NutritionAccountingInput {
        NutritionAccountingInput {
            schema_version: INPUT_SCHEMA_V1.to_owned(),
            batch_id: "batch-1".to_owned(),
            nutrient_id: "iron".to_owned(),
            concentration_mg_per_kg: Some(80.0),
            concentration_mass_basis: Some(MassBasis::FreshEdibleMass),
            concentration_stage: Some(ConcentrationStage::MeasuredAtServing),
            concentration_provenance: ConcentrationProvenance::DirectAssay,
            concentration_evidence_uri: Some("urn:assay:iron:batch-1".to_owned()),
            assay_method: Some("validated-lab-method-v1".to_owned()),
            concentration_lower_mg_per_kg: None,
            concentration_upper_mg_per_kg: None,
            edible_mass_kg_at_concentration_stage: Some(0.15),
            edible_mass_basis_at_concentration_stage: Some(MassBasis::FreshEdibleMass),
            edible_mass_lower_kg_at_concentration_stage: None,
            edible_mass_upper_kg_at_concentration_stage: None,
            servings_in_batch: Some(1),
            retention_fraction: None,
            retention_evidence_uri: None,
            retention_lower_fraction: None,
            retention_upper_fraction: None,
            bioavailability_fraction: Some(0.25),
            bioavailability_evidence_uri: Some("urn:literature:iron-bioavailability".to_owned()),
            bioavailability_lower_fraction: None,
            bioavailability_upper_fraction: None,
            uncertainty_interval_kind: None,
            uncertainty_evidence_uri: None,
            water_liters: Some(30.0),
            water_boundary: Some(WaterBoundary::IrrigationApplied),
            land_square_meters: Some(2.0),
            land_boundary: Some(LandBoundary::AllocatedCultivatedArea),
            energy_kwh: Some(0.5),
            energy_boundary: Some(EnergyBoundary::OnFarmOperations),
            cost_amount: Some(1.25),
            cost_boundary: Some(CostBoundary::FarmGate),
            currency_code: Some("ZAR".to_owned()),
            source_sha256: Some("a".repeat(64)),
        }
    }

    #[test]
    fn checked_in_synthetic_v1_example_has_reproducible_interval_outputs() {
        let input: NutritionAccountingInput = serde_json::from_str(include_str!(
            "../../docs/research/cropfoundry/nutrition-accounting-example-v1.json"
        ))
        .unwrap();
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, Some(12.0));
        assert_eq!(
            receipt.nutrient_mass_delivered_interval_lower_mg,
            Some(10.5)
        );
        assert_eq!(
            receipt.nutrient_mass_delivered_interval_upper_mg,
            Some(13.6)
        );
        assert_eq!(
            receipt.bioavailable_nutrient_mass_delivered_interval_lower_mg,
            Some(2.1)
        );
        assert_eq!(
            receipt.bioavailable_nutrient_mass_delivered_interval_upper_mg,
            Some(4.08)
        );
    }

    #[test]
    fn v1_json_schema_is_parseable_and_unknown_fields_are_rejected() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../docs/research/cropfoundry/nutrition-accounting-input-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            INPUT_SCHEMA_V1
        );

        let input = serde_json::to_value(fixture()).unwrap();
        let input_fields = input
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let schema_fields = schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(input_fields, schema_fields);
        for required in schema["required"].as_array().unwrap() {
            assert!(schema_fields.contains(required.as_str().unwrap()));
        }

        let mut input = input;
        input["unexpected_extra_field"] = serde_json::json!(true);
        assert!(serde_json::from_value::<NutritionAccountingInput>(input).is_err());
    }

    #[test]
    fn receipt_retains_input_assumptions_and_provenance_fields() {
        let input = fixture();
        let receipt = audit_input(input.clone());
        assert_eq!(receipt.input_snapshot.batch_id, input.batch_id);
        assert_eq!(
            receipt.input_snapshot.concentration_mass_basis,
            Some(MassBasis::FreshEdibleMass)
        );
        assert_eq!(receipt.input_snapshot.bioavailability_fraction, Some(0.25));
        assert_eq!(
            receipt.input_snapshot.bioavailability_evidence_uri,
            Some("urn:literature:iron-bioavailability".to_owned())
        );
    }

    #[test]
    fn delivery_mass_provenance_is_explicit_in_the_receipt() {
        let direct = audit_input(fixture());
        assert!(!direct.delivery_mass_is_estimated);
        let mut estimated = fixture();
        estimated.concentration_provenance = ConcentrationProvenance::LiteratureValue;
        let literature = audit_input(estimated);
        assert!(literature.delivery_mass_is_estimated);
    }

    #[test]
    fn direct_at_serving_assay_computes_nutrient_mass_and_resource_ratios() {
        let receipt = audit_input(fixture());
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, Some(12.0));
        assert_eq!(
            receipt.estimated_bioavailable_nutrient_mass_delivered_mg,
            Some(3.0)
        );
        assert_eq!(receipt.nutrient_mg_per_serving, Some(12.0));
        assert_eq!(
            receipt.estimated_bioavailable_nutrient_mg_per_serving,
            Some(3.0)
        );
        assert_eq!(receipt.nutrient_mg_per_liter_water, Some(0.4));
        assert_eq!(
            receipt.water_boundary,
            Some(WaterBoundary::IrrigationApplied)
        );
        assert_eq!(receipt.nutrient_mg_per_square_meter, Some(6.0));
        assert_eq!(
            receipt.land_boundary,
            Some(LandBoundary::AllocatedCultivatedArea)
        );
        assert_eq!(receipt.nutrient_mg_per_kwh, Some(24.0));
        assert_eq!(
            receipt.energy_boundary,
            Some(EnergyBoundary::OnFarmOperations)
        );
        assert_eq!(receipt.nutrient_mg_per_currency_unit, Some(9.6));
        assert_eq!(receipt.cost_boundary, Some(CostBoundary::FarmGate));
    }

    #[test]
    fn batch_nutrient_mass_is_divided_only_by_declared_serving_count() {
        let mut input = fixture();
        input.servings_in_batch = Some(2);
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, Some(12.0));
        assert_eq!(receipt.nutrient_mg_per_serving, Some(6.0));
    }

    #[test]
    fn missing_bioavailability_is_not_assumed_to_be_one() {
        let mut input = fixture();
        input.bioavailability_fraction = None;
        input.bioavailability_evidence_uri = None;
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, Some(12.0));
        assert_eq!(
            receipt.estimated_bioavailable_nutrient_mass_delivered_mg,
            None
        );
    }

    #[test]
    fn pre_preparation_concentration_requires_a_retention_factor_and_evidence() {
        let mut input = fixture();
        input.concentration_stage = Some(ConcentrationStage::MeasuredBeforePreparation);
        input.retention_fraction = Some(0.75);
        input.retention_evidence_uri = Some("urn:retention:study-v1".to_owned());
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, None);
        assert_eq!(receipt.modeled_nutrient_mass_delivered_mg, Some(9.0));
    }

    #[test]
    fn nutrient_delivery_bounds_propagate_without_zero_uncertainty_assumptions() {
        let mut input = fixture();
        input.concentration_lower_mg_per_kg = Some(70.0);
        input.concentration_upper_mg_per_kg = Some(90.0);
        input.edible_mass_lower_kg_at_concentration_stage = Some(0.14);
        input.edible_mass_upper_kg_at_concentration_stage = Some(0.16);
        input.bioavailability_lower_fraction = Some(0.20);
        input.bioavailability_upper_fraction = Some(0.30);
        input.uncertainty_interval_kind = Some(UncertaintyIntervalKind::SensitivityBounds);
        input.uncertainty_evidence_uri = Some("urn:uncertainty:iron:batch-1".to_owned());

        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mass_delivered_interval_lower_mg, Some(9.8));
        assert_eq!(
            receipt.nutrient_mass_delivered_interval_upper_mg,
            Some(14.4)
        );
        assert_eq!(
            receipt.bioavailable_nutrient_mass_delivered_interval_lower_mg,
            Some(1.96)
        );
        assert_eq!(
            receipt.bioavailable_nutrient_mass_delivered_interval_upper_mg,
            Some(4.32)
        );
        assert_eq!(
            receipt.uncertainty_interval_kind,
            Some(UncertaintyIntervalKind::SensitivityBounds)
        );
    }

    #[test]
    fn interval_bounds_without_evidence_fail_closed() {
        let mut input = fixture();
        input.concentration_lower_mg_per_kg = Some(70.0);
        input.concentration_upper_mg_per_kg = Some(90.0);
        input.edible_mass_lower_kg_at_concentration_stage = Some(0.14);
        input.edible_mass_upper_kg_at_concentration_stage = Some(0.16);
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("interval kind and evidence URI"))
        );
        assert_eq!(receipt.nutrient_mass_delivered_interval_lower_mg, None);
    }

    #[test]
    fn partial_or_inconsistent_intervals_fail_closed() {
        let mut partial = fixture();
        partial.concentration_lower_mg_per_kg = Some(70.0);
        partial.edible_mass_lower_kg_at_concentration_stage = Some(0.14);
        let receipt = audit_input(partial);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("lower and upper bounds"))
        );

        let mut inconsistent = fixture();
        inconsistent.concentration_lower_mg_per_kg = Some(90.0);
        inconsistent.concentration_upper_mg_per_kg = Some(100.0);
        inconsistent.edible_mass_lower_kg_at_concentration_stage = Some(0.14);
        inconsistent.edible_mass_upper_kg_at_concentration_stage = Some(0.16);
        let receipt = audit_input(inconsistent);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("point estimate is outside"))
        );
    }

    #[test]
    fn pre_preparation_delivery_interval_requires_and_uses_retention_bounds() {
        let mut input = fixture();
        input.concentration_stage = Some(ConcentrationStage::MeasuredBeforePreparation);
        input.retention_fraction = Some(0.75);
        input.retention_evidence_uri = Some("urn:retention:study-v1".to_owned());
        input.concentration_lower_mg_per_kg = Some(70.0);
        input.concentration_upper_mg_per_kg = Some(90.0);
        input.edible_mass_lower_kg_at_concentration_stage = Some(0.14);
        input.edible_mass_upper_kg_at_concentration_stage = Some(0.16);
        input.retention_lower_fraction = Some(0.65);
        input.retention_upper_fraction = Some(0.85);
        input.uncertainty_interval_kind = Some(UncertaintyIntervalKind::SensitivityBounds);
        input.uncertainty_evidence_uri = Some("urn:uncertainty:processing:batch-1".to_owned());

        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(
            receipt.nutrient_mass_delivered_interval_lower_mg,
            Some(6.37)
        );
        assert_eq!(
            receipt.nutrient_mass_delivered_interval_upper_mg,
            Some(12.24)
        );
    }

    #[test]
    fn numeric_overflow_fails_closed_instead_of_emitting_infinite_nutrient_mass() {
        let mut input = fixture();
        input.concentration_mg_per_kg = Some(1e308);
        input.edible_mass_kg_at_concentration_stage = Some(1e308);
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, None);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("overflowed the finite numeric range"))
        );
    }

    #[test]
    fn v0_input_remains_supported_without_interval_fields() {
        let mut input = fixture();
        input.schema_version = INPUT_SCHEMA_V0.to_owned();
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mass_delivered_interval_lower_mg, None);
    }

    #[test]
    fn v0_input_cannot_smuggle_v1_interval_fields() {
        let mut input = fixture();
        input.schema_version = INPUT_SCHEMA_V0.to_owned();
        input.concentration_lower_mg_per_kg = Some(70.0);
        input.concentration_upper_mg_per_kg = Some(90.0);
        input.edible_mass_lower_kg_at_concentration_stage = Some(0.14);
        input.edible_mass_upper_kg_at_concentration_stage = Some(0.16);
        input.uncertainty_interval_kind = Some(UncertaintyIntervalKind::SensitivityBounds);
        input.uncertainty_evidence_uri = Some("urn:uncertainty:v1-only".to_owned());
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("use schema v1"))
        );
    }

    #[test]
    fn missing_concentration_stage_is_not_inferred() {
        let mut input = fixture();
        input.concentration_stage = None;
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("concentration_stage"))
        );
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, None);
        assert_eq!(receipt.modeled_nutrient_mass_delivered_mg, None);
    }

    #[test]
    fn mismatched_fresh_and_dry_mass_basis_fails_closed() {
        let mut input = fixture();
        input.concentration_mass_basis = Some(MassBasis::DryEdibleMass);
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("mass basis"))
        );
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, None);
    }

    #[test]
    fn whitespace_only_evidence_fields_fail_closed() {
        let mut input = fixture();
        input.concentration_evidence_uri = Some("   ".to_owned());
        input.assay_method = Some("  ".to_owned());
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("concentration_evidence_uri"))
        );
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("assay_method"))
        );
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
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("retention_fraction"))
        );
    }

    #[test]
    fn resource_efficiency_requires_matching_measurement_boundaries() {
        let mut input = fixture();
        input.water_boundary = None;
        let receipt = audit_input(input);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("water_liters and its resource boundary"))
        );
        assert_eq!(receipt.nutrient_mg_per_liter_water, None);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, None);

        let mut missing_measurement = fixture();
        missing_measurement.water_liters = None;
        let receipt = audit_input(missing_measurement);
        assert!(!receipt.accounting_pass);
        assert!(
            receipt
                .issues
                .iter()
                .any(|issue| issue.contains("water_liters and its resource boundary"))
        );
    }

    #[test]
    fn unknown_resource_measurements_stay_unknown() {
        let mut input = fixture();
        input.water_liters = None;
        input.water_boundary = None;
        input.land_square_meters = None;
        input.land_boundary = None;
        input.energy_kwh = None;
        input.energy_boundary = None;
        input.cost_amount = None;
        input.cost_boundary = None;
        input.currency_code = None;
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert_eq!(receipt.nutrient_mg_per_liter_water, None);
        assert_eq!(receipt.water_boundary, None);
        assert_eq!(receipt.nutrient_mg_per_square_meter, None);
        assert_eq!(receipt.land_boundary, None);
        assert_eq!(receipt.nutrient_mg_per_kwh, None);
        assert_eq!(receipt.energy_boundary, None);
        assert_eq!(receipt.nutrient_mg_per_currency_unit, None);
        assert_eq!(receipt.cost_boundary, None);
    }

    #[test]
    fn literature_value_stays_visibly_modelled_and_cannot_masquerade_as_assay() {
        let mut input = fixture();
        input.concentration_provenance = ConcentrationProvenance::LiteratureValue;
        let receipt = audit_input(input);
        assert!(receipt.accounting_pass);
        assert!(!receipt.concentration_is_direct_assay);
        assert_eq!(receipt.measured_nutrient_mass_delivered_mg, None);
        assert_eq!(receipt.modeled_nutrient_mass_delivered_mg, Some(12.0));
    }
}
