use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-pack-capability-profile/0.1";
const CORPUS_VERSION: &str = "ev-pack-capability-corpus/0.1";
const PROFILE_VERSION: &str = "ev-pack-capability-profile/0.1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    profile: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    profile_version: String,
    profile_id: String,
    asset: Asset,
    population: Population,
    reference_conditions: ReferenceConditions,
    provenance: Provenance,
    units: Vec<ObservedUnit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    vehicle_ref: String,
    vehicle_configuration_digest: String,
    pack_ref: String,
    pack_configuration_digest: String,
    chemistry: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Population {
    comparison_resolution: String,
    declared_unit_count: usize,
    observed_unit_count: usize,
    inventory_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceConditions {
    capacity: CapacityCondition,
    resistance: ResistanceCondition,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityCondition {
    condition_ref: String,
    temperature_c: f64,
    c_rate: f64,
    lower_cutoff_voltage_v: f64,
    upper_cutoff_voltage_v: f64,
    protocol_ref: String,
    protocol_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResistanceCondition {
    condition_ref: String,
    temperature_c: f64,
    soc_fraction: f64,
    pulse_current_a: f64,
    pulse_duration_ms: u64,
    protocol_ref: String,
    protocol_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    source_batch_digest: String,
    analysis_method_ref: String,
    analysis_method_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedUnit {
    unit_ref: String,
    nominal_capacity_ah: f64,
    capacity: CapacityMeasurement,
    resistance: ResistanceMeasurement,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityMeasurement {
    value_ah: f64,
    uncertainty_ah: f64,
    evidence_class: String,
    method: String,
    evidence_digest: String,
    condition_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResistanceMeasurement {
    value_mohm: f64,
    uncertainty_mohm: f64,
    evidence_class: String,
    method: String,
    evidence_digest: String,
    condition_ref: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DistributionSummary {
    pub mean: f64,
    pub median_nearest_rank: f64,
    pub minimum: f64,
    pub p05_nearest_rank: f64,
    pub p95_nearest_rank: f64,
    pub population_standard_deviation: f64,
    pub coefficient_of_variation: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfileSummary {
    pub profile_id: String,
    pub vehicle_ref: String,
    pub pack_ref: String,
    pub chemistry: String,
    pub comparison_resolution: String,
    pub declared_unit_count: usize,
    pub observed_unit_count: usize,
    pub coverage_fraction: f64,
    pub population_scope: String,
    pub soh_fraction: DistributionSummary,
    pub resistance_mohm: DistributionSummary,
    pub weakest_observed_soh_unit_ref: String,
    pub capacity_evidence_class_counts: BTreeMap<String, usize>,
    pub resistance_evidence_class_counts: BTreeMap<String, usize>,
    pub uncertainty_status: String,
    pub pack_energy_power_status: String,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    summary: Option<ProfileSummary>,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    corpus_sha256: String,
    case_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    cases: Vec<CaseAudit>,
}

#[derive(Debug, Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn schema_validator(schema_bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema $id does not match frozen profile identity".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_profile(profile: &Profile) -> Vec<String> {
    let mut errors = Vec::new();
    if profile.profile_version != PROFILE_VERSION {
        errors.push("unsupported profile_version".to_owned());
    }
    if !is_nonempty(&profile.profile_id) {
        errors.push("profile_id must be non-empty".to_owned());
    }
    if !is_nonempty(&profile.asset.vehicle_ref) || !is_nonempty(&profile.asset.pack_ref) {
        errors.push("vehicle_ref and pack_ref must be non-empty".to_owned());
    }
    if !is_nonempty(&profile.asset.chemistry)
        || !is_sha256(&profile.asset.vehicle_configuration_digest)
        || !is_sha256(&profile.asset.pack_configuration_digest)
        || !is_sha256(&profile.population.inventory_digest)
        || !is_sha256(&profile.provenance.source_batch_digest)
        || !is_sha256(&profile.provenance.analysis_method_digest)
        || !is_nonempty(&profile.provenance.analysis_method_ref)
    {
        errors.push("asset, inventory or provenance digest/reference is invalid".to_owned());
    }
    if !matches!(
        profile.population.comparison_resolution.as_str(),
        "cell" | "parallel_group" | "module"
    ) {
        errors.push("comparison_resolution must be cell, parallel_group or module".to_owned());
    }
    if profile.population.declared_unit_count < 2
        || profile.population.observed_unit_count < 2
        || profile.population.observed_unit_count > profile.population.declared_unit_count
        || profile.population.observed_unit_count != profile.units.len()
    {
        errors.push("declared/observed unit counts are inconsistent with the units array".to_owned());
    }
    if profile.units.len() < 2 {
        errors.push("at least two comparable units are required".to_owned());
    }

    let capacity_condition = &profile.reference_conditions.capacity;
    let resistance_condition = &profile.reference_conditions.resistance;
    if !is_nonempty(&capacity_condition.condition_ref)
        || !is_nonempty(&capacity_condition.protocol_ref)
        || !is_sha256(&capacity_condition.protocol_digest)
        || !capacity_condition.temperature_c.is_finite()
        || !capacity_condition.c_rate.is_finite()
        || capacity_condition.c_rate <= 0.0
        || !capacity_condition.lower_cutoff_voltage_v.is_finite()
        || !capacity_condition.upper_cutoff_voltage_v.is_finite()
        || capacity_condition.lower_cutoff_voltage_v <= 0.0
        || capacity_condition.upper_cutoff_voltage_v
            <= capacity_condition.lower_cutoff_voltage_v
    {
        errors.push("capacity reference condition is invalid".to_owned());
    }
    if !is_nonempty(&resistance_condition.condition_ref)
        || !is_nonempty(&resistance_condition.protocol_ref)
        || !is_sha256(&resistance_condition.protocol_digest)
        || !resistance_condition.temperature_c.is_finite()
        || !resistance_condition.soc_fraction.is_finite()
        || !(0.0..=1.0).contains(&resistance_condition.soc_fraction)
        || !resistance_condition.pulse_current_a.is_finite()
        || resistance_condition.pulse_current_a <= 0.0
        || resistance_condition.pulse_duration_ms == 0
    {
        errors.push("resistance reference condition is invalid".to_owned());
    }

    let mut unit_ids = BTreeSet::new();
    for unit in &profile.units {
        if !is_nonempty(&unit.unit_ref) || !unit_ids.insert(unit.unit_ref.clone()) {
            errors.push(format!("unit identity is empty or duplicated: {}", unit.unit_ref));
        }
        if !unit.nominal_capacity_ah.is_finite() || unit.nominal_capacity_ah <= 0.0 {
            errors.push(format!("unit {} has invalid nominal capacity", unit.unit_ref));
        }
        if !unit.capacity.value_ah.is_finite()
            || unit.capacity.value_ah <= 0.0
            || !unit.capacity.uncertainty_ah.is_finite()
            || unit.capacity.uncertainty_ah < 0.0
            || !unit.resistance.value_mohm.is_finite()
            || unit.resistance.value_mohm <= 0.0
            || !unit.resistance.uncertainty_mohm.is_finite()
            || unit.resistance.uncertainty_mohm < 0.0
        {
            errors.push(format!("unit {} has invalid capacity or resistance", unit.unit_ref));
        }
        if unit.capacity.condition_ref != capacity_condition.condition_ref
            || unit.resistance.condition_ref != resistance_condition.condition_ref
        {
            errors.push(format!("unit {} uses a mismatched reference condition", unit.unit_ref));
        }
        if !is_sha256(&unit.capacity.evidence_digest)
            || !is_sha256(&unit.resistance.evidence_digest)
            || !is_nonempty(&unit.capacity.method)
            || !is_nonempty(&unit.resistance.method)
        {
            errors.push(format!("unit {} has invalid measurement provenance", unit.unit_ref));
        }
        for class in [
            unit.capacity.evidence_class.as_str(),
            unit.resistance.evidence_class.as_str(),
        ] {
            if !matches!(
                class,
                "reference_test_measurement"
                    | "derived_estimate"
                    | "model_prediction"
                    | "simulation_output"
                    | "synthetic_fixture"
                    | "vendor_claim"
            ) {
                errors.push(format!("unit {} has unknown evidence class", unit.unit_ref));
            }
        }
    }
    errors
}

fn nearest_rank(values: &[f64], percentile: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let rank = (percentile * ordered.len() as f64).ceil() as usize;
    ordered[rank.max(1).min(ordered.len()) - 1]
}

fn distribution(values: &[f64]) -> DistributionSummary {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let standard_deviation = variance.sqrt();
    DistributionSummary {
        mean,
        median_nearest_rank: nearest_rank(values, 0.50),
        minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
        p05_nearest_rank: nearest_rank(values, 0.05),
        p95_nearest_rank: nearest_rank(values, 0.95),
        population_standard_deviation: standard_deviation,
        coefficient_of_variation: standard_deviation / mean,
    }
}

fn evidence_counts<'a>(classes: impl Iterator<Item = &'a str>) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for class in classes {
        *counts.entry(class.to_owned()).or_insert(0) += 1;
    }
    counts
}

fn summarize(profile: &Profile) -> ProfileSummary {
    let soh_values: Vec<f64> = profile
        .units
        .iter()
        .map(|unit| unit.capacity.value_ah / unit.nominal_capacity_ah)
        .collect();
    let resistance_values: Vec<f64> = profile
        .units
        .iter()
        .map(|unit| unit.resistance.value_mohm)
        .collect();
    let weakest = profile
        .units
        .iter()
        .min_by(|left, right| {
            (left.capacity.value_ah / left.nominal_capacity_ah)
                .total_cmp(&(right.capacity.value_ah / right.nominal_capacity_ah))
        })
        .map(|unit| unit.unit_ref.clone())
        .unwrap_or_default();

    let observed = profile.population.observed_unit_count;
    let declared = profile.population.declared_unit_count;
    ProfileSummary {
        profile_id: profile.profile_id.clone(),
        vehicle_ref: profile.asset.vehicle_ref.clone(),
        pack_ref: profile.asset.pack_ref.clone(),
        chemistry: profile.asset.chemistry.clone(),
        comparison_resolution: profile.population.comparison_resolution.clone(),
        declared_unit_count: declared,
        observed_unit_count: observed,
        coverage_fraction: observed as f64 / declared as f64,
        population_scope: if observed == declared {
            "declared_population_complete".to_owned()
        } else {
            "observed_subset_only".to_owned()
        },
        soh_fraction: distribution(&soh_values),
        resistance_mohm: distribution(&resistance_values),
        weakest_observed_soh_unit_ref: weakest,
        capacity_evidence_class_counts: evidence_counts(
            profile.units.iter().map(|unit| unit.capacity.evidence_class.as_str()),
        ),
        resistance_evidence_class_counts: evidence_counts(
            profile.units.iter().map(|unit| unit.resistance.evidence_class.as_str()),
        ),
        uncertainty_status: "per_unit_uncertainty_preserved_not_propagated_in_v0_1".to_owned(),
        pack_energy_power_status: "NOT_COMPUTABLE_TOPOLOGY_MODEL_NOT_QUALIFIED".to_owned(),
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let corpus: Corpus =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity does not match the frozen contract".to_owned());
    }

    let mut seen_case_ids = BTreeSet::new();
    let mut audits = Vec::new();
    let mut accepted_count = 0usize;
    let mut rejected_count = 0usize;

    for case in corpus.cases {
        if !is_nonempty(&case.case_id) || !seen_case_ids.insert(case.case_id.clone()) {
            return Err(format!("case_id is empty or duplicated: {}", case.case_id));
        }
        if !matches!(case.expectation.as_str(), "accept" | "reject") {
            return Err(format!("case {} has invalid expectation", case.case_id));
        }

        let schema_errors: Vec<String> = validator
            .iter_errors(&case.profile)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Profile>(case.profile.clone()) {
                Ok(profile) => {
                    let semantic_errors = validate_profile(&profile);
                    if semantic_errors.is_empty() {
                        Ok(summarize(&profile))
                    } else {
                        Err(semantic_errors)
                    }
                }
                Err(error) => Err(vec![format!("typed profile parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };

        let accepted = outcome.is_ok();
        let expected_accepted = case.expectation == "accept";
        if accepted != expected_accepted {
            let detail = match &outcome {
                Ok(_) => "unexpected acceptance".to_owned(),
                Err(errors) => errors.join("; "),
            };
            return Err(format!(
                "case {} expected {} but accepted={}: {}",
                case.case_id, case.expectation, accepted, detail
            ));
        }
        let (summary, errors) = match outcome {
            Ok(summary) => {
                accepted_count += 1;
                (Some(summary), Vec::new())
            }
            Err(_) => {
                rejected_count += 1;
                (None, vec!["rejected as expected".to_owned()])
            }
        };
        audits.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted,
            errors,
            summary,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        corpus_sha256: sha256(corpus_bytes),
        case_count: audits.len(),
        accepted_count,
        rejected_count,
        cases: audits,
    };
    let report_bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    let report_sha256 = sha256(&report_bytes);
    Ok(CorpusReport {
        payload,
        report_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_pack_capability_profile_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/pack_capability_profiles_v0_1.json");

    #[test]
    fn frozen_corpus_accepts_three_profiles_and_rejects_six_controls() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        assert_eq!(report.payload.case_count, 9);
        assert_eq!(report.payload.accepted_count, 3);
        assert_eq!(report.payload.rejected_count, 6);
    }

    #[test]
    fn incomplete_coverage_is_disclosed_not_imputed() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        let partial = report
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "positive-partial-coverage-explicit")
            .and_then(|case| case.summary.as_ref())
            .expect("partial profile summary");
        assert_eq!(partial.population_scope, "observed_subset_only");
        assert!((partial.coverage_fraction - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn v0_1_never_claims_topology_derived_pack_energy_or_power() {
        let report = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        let complete = report
            .payload
            .cases
            .iter()
            .find(|case| case.case_id == "positive-complete-cell-population")
            .and_then(|case| case.summary.as_ref())
            .expect("complete profile summary");
        assert_eq!(
            complete.pack_energy_power_status,
            "NOT_COMPUTABLE_TOPOLOGY_MODEL_NOT_QUALIFIED"
        );
    }

    #[test]
    fn distribution_metrics_are_invariant_to_unit_order() {
        let corpus: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        let profile_value = corpus["cases"][0]["profile"].clone();
        let original: Profile = serde_json::from_value(profile_value.clone()).expect("profile");
        let mut reversed_value = profile_value;
        reversed_value["units"]
            .as_array_mut()
            .expect("units array")
            .reverse();
        let reversed: Profile =
            serde_json::from_value(reversed_value).expect("reversed profile");
        assert_eq!(summarize(&original), summarize(&reversed));
    }

    #[test]
    fn quantiles_use_declared_nearest_rank_method() {
        assert_eq!(nearest_rank(&[4.0, 1.0, 3.0, 2.0], 0.50), 2.0);
        assert_eq!(nearest_rank(&[4.0, 1.0, 3.0, 2.0], 0.95), 4.0);
    }
}
