use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    process::ExitCode,
};

const VERSION: &str = "0.1.0";
type Check<T> = Result<T, String>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    suite_id: String,
    suite_version: String,
    evidence_class: String,
    claim_ceiling: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    expected_disposition: Disposition,
    expected_reason_codes: Vec<String>,
    profile: WoodProfile,
    query: PropertyQuery,
    expected_assertions: Vec<String>,
    negative_control: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WoodProfile {
    profile_id: String,
    accepted_scientific_name: String,
    taxon_resolution: TaxonResolution,
    identification_method: IdentificationMethod,
    profile_level: ProfileLevel,
    source_record_id: String,
    source_kind: SourceKind,
    source_license_ref: Option<String>,
    specimen_id: Option<String>,
    lot_id: Option<String>,
    properties: Vec<WoodProperty>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaxonResolution {
    Resolved,
    Ambiguous,
    Unresolved,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum IdentificationMethod {
    SpecimenDiagnosis,
    ScientificNameReference,
    CommonNameOnly,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProfileLevel {
    Specimen,
    Lot,
    SpeciesAggregate,
    GenusAggregate,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SourceKind {
    SyntheticReference,
    PublicReference,
    MeasuredSource,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceLevel {
    SampleMeasured,
    LotMeasured,
    LiteratureReported,
    SpeciesEstimate,
    GenusEstimate,
    QualifiedDesignValue,
    Assumed,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PropertyUnit {
    KilogramsPerCubicMetre,
    Megapascal,
    Gigapascal,
    Percent,
    MillimetresPerMetre,
    Dimensionless,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PropertyBasis {
    Green,
    AirDry,
    OvenDry,
    Conditioned,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GrainDirection {
    Longitudinal,
    Radial,
    Tangential,
    Isotropic,
    NotApplicable,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WoodProperty {
    property_id: String,
    value: u64,
    unit: PropertyUnit,
    basis: PropertyBasis,
    direction: GrainDirection,
    evidence_level: EvidenceLevel,
    property_source_id: String,
    measurement_record_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum QueryPurpose {
    CandidateRanking,
    LotSelection,
    StructuralUse,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PropertyQuery {
    purpose: QueryPurpose,
    required_property_id: String,
    required_unit: PropertyUnit,
    required_basis: PropertyBasis,
    required_direction: GrainDirection,
    minimum_value: Option<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    Pass,
    Reject,
}

#[derive(Debug, Serialize)]
struct Receipt {
    result: &'static str,
    validator_version: &'static str,
    suite_id: String,
    suite_version: String,
    evidence_class: String,
    claim_ceiling: String,
    scenario_count: usize,
    expected_controls_matched: usize,
    corpus_sha256: String,
    source_commit: String,
    scenarios: Vec<ScenarioReceipt>,
}

#[derive(Debug, Serialize)]
struct ScenarioReceipt {
    scenario_id: String,
    profile_id: String,
    profile_level: ProfileLevel,
    query_purpose: QueryPurpose,
    expected_disposition: Disposition,
    actual_disposition: Disposition,
    expected_reason_codes: Vec<String>,
    actual_reason_codes: Vec<String>,
    selected_property_id: Option<String>,
    selected_evidence_level: Option<EvidenceLevel>,
    selected_value: Option<u64>,
    assertions: Vec<String>,
    negative_control: String,
    expected_control_matched: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn evaluate_scenario(scenario: &Scenario) -> ScenarioReceipt {
    let mut reasons = BTreeSet::<String>::new();
    let profile = &scenario.profile;
    let query = &scenario.query;

    if !nonempty(&profile.profile_id) {
        reasons.insert("PROFILE_ID_MISSING".to_owned());
    }
    if !nonempty(&profile.accepted_scientific_name) {
        reasons.insert("SCIENTIFIC_NAME_MISSING".to_owned());
    }
    if !nonempty(&profile.source_record_id) {
        reasons.insert("SOURCE_PROVENANCE_MISSING".to_owned());
    }
    if matches!(profile.source_kind, SourceKind::PublicReference | SourceKind::MeasuredSource)
        && !profile.source_license_ref.as_ref().is_some_and(|value| nonempty(value))
    {
        reasons.insert("DATA_LICENSE_UNRESOLVED".to_owned());
    }
    if !nonempty(&query.required_property_id) {
        reasons.insert("REQUIRED_PROPERTY_ID_MISSING".to_owned());
    }

    if profile.taxon_resolution != TaxonResolution::Resolved
        || profile.identification_method == IdentificationMethod::CommonNameOnly
    {
        reasons.insert("TAXON_IDENTITY_UNRESOLVED".to_owned());
    }

    match profile.profile_level {
        ProfileLevel::Specimen
            if !profile.specimen_id.as_ref().is_some_and(|value| nonempty(value)) =>
        {
            reasons.insert("SPECIMEN_ID_MISSING".to_owned());
        }
        ProfileLevel::Lot if !profile.lot_id.as_ref().is_some_and(|value| nonempty(value)) => {
            reasons.insert("LOT_ID_MISSING".to_owned());
        }
        _ => {}
    }

    if matches!(query.purpose, QueryPurpose::LotSelection | QueryPurpose::StructuralUse) {
        if !matches!(profile.profile_level, ProfileLevel::Lot | ProfileLevel::Specimen) {
            reasons.insert("EVIDENCE_SCOPE_MISMATCH".to_owned());
        }
        if !profile.lot_id.as_ref().is_some_and(|value| nonempty(value)) {
            reasons.insert("LOT_ID_MISSING".to_owned());
        }
    }

    let mut property_ids = BTreeSet::new();
    for property in &profile.properties {
        if !nonempty(&property.property_id) || !property_ids.insert(property.property_id.clone()) {
            reasons.insert("DUPLICATE_OR_EMPTY_PROPERTY_ID".to_owned());
        }
        if !nonempty(&property.property_source_id) {
            reasons.insert("PROPERTY_SOURCE_MISSING".to_owned());
        }
        if matches!(
            property.evidence_level,
            EvidenceLevel::SampleMeasured
                | EvidenceLevel::LotMeasured
                | EvidenceLevel::QualifiedDesignValue
        ) && !property.measurement_record_id.as_ref().is_some_and(|value| nonempty(value))
        {
            reasons.insert("PROPERTY_MEASUREMENT_REFERENCE_MISSING".to_owned());
        }
    }

    let selected = profile.properties.iter()
        .find(|property| property.property_id == query.required_property_id);
    let mut selected_id = None;
    let mut selected_level = None;
    let mut selected_value = None;

    match selected {
        None => {
            reasons.insert("REQUIRED_PROPERTY_MISSING".to_owned());
        }
        Some(property) => {
            selected_id = Some(property.property_id.clone());
            selected_level = Some(property.evidence_level);
            selected_value = Some(property.value);

            if property.unit != query.required_unit {
                reasons.insert("PROPERTY_UNIT_MISMATCH".to_owned());
            }
            if property.basis != query.required_basis {
                reasons.insert("PROPERTY_CONDITION_MISMATCH".to_owned());
            }
            if property.direction != query.required_direction {
                reasons.insert("PROPERTY_DIRECTION_MISMATCH".to_owned());
            }

            if matches!(property.evidence_level, EvidenceLevel::Unknown | EvidenceLevel::Assumed) {
                reasons.insert("PROPERTY_EVIDENCE_UNRESOLVED".to_owned());
            }
            if query.purpose == QueryPurpose::LotSelection
                && !matches!(
                    property.evidence_level,
                    EvidenceLevel::SampleMeasured | EvidenceLevel::LotMeasured
                )
            {
                reasons.insert("EVIDENCE_SCOPE_MISMATCH".to_owned());
            }
            if query.purpose == QueryPurpose::StructuralUse
                && property.evidence_level != EvidenceLevel::QualifiedDesignValue
            {
                reasons.insert("STRUCTURAL_VALUE_NOT_QUALIFIED".to_owned());
            }
            if let Some(minimum) = query.minimum_value {
                if property.value < minimum {
                    reasons.insert("PROPERTY_BELOW_MINIMUM".to_owned());
                }
            }
        }
    }

    let actual_reason_codes: Vec<String> = reasons.into_iter().collect();
    let actual_disposition = if actual_reason_codes.is_empty() {
        Disposition::Pass
    } else {
        Disposition::Reject
    };
    let mut expected_reasons = scenario.expected_reason_codes.clone();
    expected_reasons.sort();
    expected_reasons.dedup();
    let expected_control_matched = actual_disposition == scenario.expected_disposition
        && actual_reason_codes == expected_reasons
        && !scenario.expected_assertions.is_empty()
        && nonempty(&scenario.negative_control);

    ScenarioReceipt {
        scenario_id: scenario.scenario_id.clone(),
        profile_id: profile.profile_id.clone(),
        profile_level: profile.profile_level,
        query_purpose: query.purpose,
        expected_disposition: scenario.expected_disposition,
        actual_disposition,
        expected_reason_codes: expected_reasons,
        actual_reason_codes,
        selected_property_id: selected_id,
        selected_evidence_level: selected_level,
        selected_value,
        assertions: scenario.expected_assertions.clone(),
        negative_control: scenario.negative_control.clone(),
        expected_control_matched,
    }
}

fn validate_corpus(corpus: &Corpus, corpus_sha256: String, source_commit: String) -> Receipt {
    let mut scenarios = Vec::with_capacity(corpus.scenarios.len());
    let mut ids = BTreeSet::new();
    let mut duplicate_id = false;

    for scenario in &corpus.scenarios {
        if !nonempty(&scenario.scenario_id) || !ids.insert(scenario.scenario_id.clone()) {
            duplicate_id = true;
        }
        scenarios.push(evaluate_scenario(scenario));
    }

    let expected_controls_matched = scenarios.iter()
        .filter(|scenario| scenario.expected_control_matched)
        .count();
    let all_controls_matched = !duplicate_id
        && scenarios.len() == 8
        && expected_controls_matched == scenarios.len()
        && corpus.suite_id == "WOOD_MATERIAL_PROFILE_BENCHMARK"
        && corpus.suite_version == "0.1.0"
        && corpus.evidence_class == "synthetic"
        && nonempty(&corpus.claim_ceiling);

    Receipt {
        result: if all_controls_matched { "validated" } else { "contract_mismatch" },
        validator_version: VERSION,
        suite_id: corpus.suite_id.clone(),
        suite_version: corpus.suite_version.clone(),
        evidence_class: corpus.evidence_class.clone(),
        claim_ceiling: corpus.claim_ceiling.clone(),
        scenario_count: scenarios.len(),
        expected_controls_matched,
        corpus_sha256,
        source_commit,
        scenarios,
    }
}

fn run() -> Check<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[1] != "--source-commit" {
        return Err(
            "usage: wood-material-profile-bench <corpus.json> --source-commit <40-hex-commit>"
                .to_owned(),
        );
    }
    let source_commit = args[2].to_ascii_lowercase();
    if source_commit.len() != 40
        || !source_commit.chars().all(|character| character.is_ascii_hexdigit())
    {
        return Err("--source-commit must be exactly 40 hexadecimal characters".to_owned());
    }

    let bytes = fs::read(&args[0]).map_err(|error| format!("cannot read corpus: {error}"))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let corpus: Corpus = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid corpus JSON: {error}"))?;
    let receipt = validate_corpus(&corpus, digest, source_commit);
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("cannot serialize receipt: {error}"))?
    );

    if receipt.result != "validated" {
        return Err("one or more frozen benchmark controls did not match".to_owned());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Corpus {
        serde_json::from_str(include_str!(
            "../../../docs/tech/wood-material-profile-corpus-v0.1.json"
        ))
        .expect("frozen synthetic corpus must parse")
    }

    #[test]
    fn frozen_corpus_matches_all_known_answers() {
        let corpus = corpus();
        let receipt = validate_corpus(&corpus, "0".repeat(64), "0".repeat(40));
        assert_eq!(receipt.result, "validated");
        assert_eq!(receipt.scenario_count, 8);
        assert_eq!(receipt.expected_controls_matched, 8);
        assert_eq!(receipt.scenarios[0].actual_disposition, Disposition::Pass);
    }

    #[test]
    fn unresolved_common_name_cannot_be_treated_as_resolved_for_ranking() {
        let mut corpus = corpus();
        let scenario = &mut corpus.scenarios[0];
        scenario.profile.taxon_resolution = TaxonResolution::Ambiguous;
        let result = evaluate_scenario(scenario);
        assert!(result
            .actual_reason_codes
            .contains(&"TAXON_IDENTITY_UNRESOLVED".to_owned()));
    }

    #[test]
    fn species_average_can_rank_candidates_but_not_qualify_a_lot() {
        let mut corpus = corpus();
        let scenario = &mut corpus.scenarios[0];
        scenario.query.purpose = QueryPurpose::LotSelection;
        let result = evaluate_scenario(scenario);
        assert!(result.actual_reason_codes.contains(&"EVIDENCE_SCOPE_MISMATCH".to_owned()));
    }

    #[test]
    fn grain_direction_is_not_silently_substituted() {
        let mut corpus = corpus();
        let scenario = &mut corpus.scenarios[3];
        scenario.profile.properties[0].direction = GrainDirection::Radial;
        let result = evaluate_scenario(scenario);
        assert!(result.actual_reason_codes.contains(&"PROPERTY_DIRECTION_MISMATCH".to_owned()));
    }

    #[test]
    fn a_literature_value_is_not_a_structural_design_value() {
        let mut corpus = corpus();
        let scenario = &mut corpus.scenarios[1];
        let result = evaluate_scenario(scenario);
        assert!(result.actual_reason_codes.contains(&"STRUCTURAL_VALUE_NOT_QUALIFIED".to_owned()));
        assert!(result.actual_reason_codes.contains(&"TAXON_IDENTITY_UNRESOLVED".to_owned()));
    }

    #[test]
    fn property_moisture_basis_must_match() {
        let mut corpus = corpus();
        let scenario = &mut corpus.scenarios[4];
        let result = evaluate_scenario(scenario);
        assert!(result.actual_reason_codes.contains(&"PROPERTY_CONDITION_MISMATCH".to_owned()));
    }
}
