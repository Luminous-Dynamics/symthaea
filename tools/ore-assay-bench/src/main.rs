use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

const VERSION: &str = "0.1.0";
const MAX_MASS_FRACTION_PPM: u64 = 1_000_000;
type Check<T> = Result<T, String>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    suite_id: String,
    suite_version: String,
    evidence_class: String,
    claim_ceiling: String,
    dry_mass_unit: String,
    assay_unit: String,
    mass_tolerance_ppm: u32,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    expected_disposition: Disposition,
    expected_reason_codes: Vec<String>,
    sample: Sample,
    process_flow: Option<ProcessFlow>,
    grade_comparison: Option<GradeComparison>,
    expected_assertions: Vec<String>,
    negative_control: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    sample_id: String,
    lot_id: String,
    source_record_id: String,
    assay_method_id: String,
    evidence_class: String,
    dry_mass_mg: u64,
    quality_control_passed: bool,
    assays: Vec<Assay>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum QuantityKind {
    ElementalMassFraction,
    OxideEquivalentMassFraction,
    MineralPhaseMassFraction,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum AssayBasis {
    DryMass,
    WetMass,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ResultState {
    Quantified,
    BelowDetection,
    NotTested,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Assay {
    analyte_id: String,
    quantity_kind: QuantityKind,
    expressed_as: String,
    basis: AssayBasis,
    result_state: ResultState,
    value_ppm: Option<u32>,
    detection_limit_ppm: Option<u32>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StreamKind {
    QualifiedProduct,
    Middlings,
    Tailings,
    ExplicitProcessLoss,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessFlow {
    input_sample_id: String,
    input_lot_id: String,
    feed_dry_mass_mg: u64,
    streams: Vec<MaterialStream>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MaterialStream {
    stream_id: String,
    kind: StreamKind,
    dry_mass_mg: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradeComparison {
    required_analyte_id: String,
    required_kind: QuantityKind,
    required_basis: AssayBasis,
    minimum_ppm: u32,
    measured_analyte_id: String,
    measured_kind: QuantityKind,
    measured_basis: AssayBasis,
    measured_value_ppm: Option<u32>,
    conversion_evidence_id: Option<String>,
    normalized_value_ppm: Option<u32>,
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
    sample_id: String,
    lot_id: String,
    expected_disposition: Disposition,
    actual_disposition: Disposition,
    expected_reason_codes: Vec<String>,
    actual_reason_codes: Vec<String>,
    assay_count: usize,
    phase_fraction_totals_ppm: BTreeMap<AssayBasis, u64>,
    flow_input_dry_mass_mg: Option<u64>,
    flow_output_dry_mass_mg: Option<u128>,
    flow_mass_difference_mg: Option<u128>,
    effective_grade_ppm: Option<u32>,
    assertions: Vec<String>,
    negative_control: String,
    expected_control_matched: bool,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn mass_tolerance(source: u128, tolerance_ppm: u32) -> u128 {
    (source / 1_000_000_u128) * tolerance_ppm as u128
        + ((source % 1_000_000_u128) * tolerance_ppm as u128) / 1_000_000_u128
}

fn evaluate_scenario(scenario: &Scenario, tolerance_ppm: u32) -> ScenarioReceipt {
    let mut reasons = BTreeSet::<String>::new();
    let sample = &scenario.sample;

    if !nonempty(&sample.sample_id) || !nonempty(&sample.lot_id) {
        reasons.insert("SAMPLE_ID_OR_LOT_MISSING".to_owned());
    }
    if !nonempty(&sample.source_record_id) {
        reasons.insert("SOURCE_REFERENCE_MISSING".to_owned());
    }
    if !nonempty(&sample.assay_method_id) {
        reasons.insert("ASSAY_METHOD_MISSING".to_owned());
    }
    if sample.evidence_class != "synthetic" {
        reasons.insert("SAMPLE_EVIDENCE_CLASS_UNSUPPORTED".to_owned());
    }
    if sample.dry_mass_mg == 0 {
        reasons.insert("SAMPLE_DRY_MASS_INVALID".to_owned());
    }
    if !sample.quality_control_passed {
        reasons.insert("ASSAY_QUALITY_CONTROL_UNRESOLVED".to_owned());
    }
    if sample.assays.is_empty() {
        reasons.insert("ASSAY_RECORDS_MISSING".to_owned());
    }

    let mut assay_ids = BTreeSet::new();
    let mut phase_totals = BTreeMap::<AssayBasis, u64>::new();

    for assay in &sample.assays {
        if assay.quantity_kind == QuantityKind::MineralPhaseMassFraction
            && assay.basis != AssayBasis::DryMass
        {
            reasons.insert("MINERAL_PHASE_BASIS_UNSUPPORTED".to_owned());
        }
        if !nonempty(&assay.analyte_id) || !nonempty(&assay.expressed_as) {
            reasons.insert("ASSAY_IDENTITY_MISSING".to_owned());
        }

        let key = format!(
            "{}|{:?}|{}|{:?}",
            assay.analyte_id, assay.quantity_kind, assay.expressed_as, assay.basis
        );
        if !assay_ids.insert(key) {
            reasons.insert("DUPLICATE_ANALYTE_MEASUREMENT".to_owned());
        }

        if assay.detection_limit_ppm == Some(0) {
            reasons.insert("DETECTION_LIMIT_INVALID".to_owned());
        }

        match assay.result_state {
            ResultState::Quantified => match assay.value_ppm {
                None => {
                    reasons.insert("QUANTIFIED_VALUE_MISSING".to_owned());
                }
                Some(value) => {
                    if value as u64 > MAX_MASS_FRACTION_PPM {
                        reasons.insert("ASSAY_VALUE_OUT_OF_RANGE".to_owned());
                    }
                    if assay.detection_limit_ppm.is_some_and(|limit| value < limit) {
                        reasons.insert("VALUE_BELOW_DETECTION_LIMIT".to_owned());
                    }
                    if assay.quantity_kind == QuantityKind::MineralPhaseMassFraction {
                        let total = phase_totals.entry(assay.basis).or_insert(0);
                        match total.checked_add(value as u64) {
                            Some(next) => *total = next,
                            None => {
                                reasons.insert("ASSAY_SUM_OVERFLOW".to_owned());
                            }
                        }
                    }
                }
            },
            ResultState::BelowDetection => {
                if assay.value_ppm.is_some() {
                    reasons.insert("BELOW_DETECTION_VALUE_PRESENT".to_owned());
                }
                if assay.detection_limit_ppm.is_none_or(|limit| limit == 0) {
                    reasons.insert("DETECTION_LIMIT_MISSING_OR_INVALID".to_owned());
                }
            }
            ResultState::NotTested => {
                if assay.value_ppm.is_some() || assay.detection_limit_ppm.is_some() {
                    reasons.insert("NOT_TESTED_HAS_NUMERIC_EVIDENCE".to_owned());
                }
            }
        }
    }

    for total in phase_totals.values() {
        if *total > MAX_MASS_FRACTION_PPM {
            reasons.insert("MINERAL_PHASE_TOTAL_EXCEEDS_ONE_MILLION_PPM".to_owned());
        }
    }

    let mut flow_input = None;
    let mut flow_output = None;
    let mut flow_difference = None;

    if let Some(flow) = &scenario.process_flow {
        flow_input = Some(flow.feed_dry_mass_mg);
        if flow.input_sample_id != sample.sample_id || flow.input_lot_id != sample.lot_id {
            reasons.insert("FLOW_SAMPLE_LINEAGE_MISMATCH".to_owned());
        }
        if flow.feed_dry_mass_mg != sample.dry_mass_mg {
            reasons.insert("FLOW_FEED_MASS_DOES_NOT_MATCH_SAMPLE".to_owned());
        }
        if flow.streams.is_empty() {
            reasons.insert("PROCESS_STREAMS_MISSING".to_owned());
        }

        let mut stream_ids = BTreeSet::new();
        let mut has_qualified_product = false;
        let mut output = 0_u128;
        for stream in &flow.streams {
            if !nonempty(&stream.stream_id) || !stream_ids.insert(stream.stream_id.clone()) {
                reasons.insert("DUPLICATE_OR_EMPTY_STREAM_ID".to_owned());
            }
            has_qualified_product |= stream.kind == StreamKind::QualifiedProduct;
            output = match output.checked_add(stream.dry_mass_mg as u128) {
                Some(next) => next,
                None => {
                    reasons.insert("FLOW_SUM_OVERFLOW".to_owned());
                    output
                }
            };
        }
        if !flow.streams.is_empty() && !has_qualified_product {
            reasons.insert("QUALIFIED_PRODUCT_STREAM_MISSING".to_owned());
        }
        flow_output = Some(output);
        let feed = flow.feed_dry_mass_mg as u128;
        let difference = feed.abs_diff(output);
        flow_difference = Some(difference);
        if difference > mass_tolerance(feed, tolerance_ppm) {
            reasons.insert("DRY_MASS_IMBALANCE".to_owned());
        }
    }

    let mut effective_grade = None;
    if let Some(grade) = &scenario.grade_comparison {
        if !nonempty(&grade.required_analyte_id) || !nonempty(&grade.measured_analyte_id) {
            reasons.insert("GRADE_ANALYTE_ID_MISSING".to_owned());
        } else if grade.required_analyte_id != grade.measured_analyte_id {
            reasons.insert("GRADE_ANALYTE_MISMATCH".to_owned());
        } else {
            let basis_matches = grade.required_kind == grade.measured_kind
                && grade.required_basis == grade.measured_basis;
            let mut comparable = true;
            let value = if basis_matches {
                if grade.conversion_evidence_id.as_ref().is_some_and(|id| nonempty(id))
                    || grade.normalized_value_ppm.is_some()
                {
                    reasons.insert("UNEXPECTED_GRADE_CONVERSION".to_owned());
                }
                grade.measured_value_ppm
            } else if grade.conversion_evidence_id.as_ref().is_some_and(|id| nonempty(id)) {
                match grade.normalized_value_ppm {
                    Some(normalized) => Some(normalized),
                    None => {
                        reasons.insert("GRADE_BASIS_MISMATCH".to_owned());
                        comparable = false;
                        None
                    }
                }
            } else {
                reasons.insert("GRADE_BASIS_MISMATCH".to_owned());
                comparable = false;
                None
            };

            match value {
                None => {
                    if comparable {
                        reasons.insert("PRODUCT_GRADE_MISSING".to_owned());
                    }
                }
                Some(value) => {
                    effective_grade = Some(value);
                    if value as u64 > MAX_MASS_FRACTION_PPM {
                        reasons.insert("PRODUCT_GRADE_OUT_OF_RANGE".to_owned());
                    } else if value < grade.minimum_ppm {
                        reasons.insert("PRODUCT_GRADE_BELOW_SPEC".to_owned());
                    }
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
        sample_id: sample.sample_id.clone(),
        lot_id: sample.lot_id.clone(),
        expected_disposition: scenario.expected_disposition,
        actual_disposition,
        expected_reason_codes: expected_reasons,
        actual_reason_codes,
        assay_count: sample.assays.len(),
        phase_fraction_totals_ppm: phase_totals,
        flow_input_dry_mass_mg: flow_input,
        flow_output_dry_mass_mg: flow_output,
        flow_mass_difference_mg: flow_difference,
        effective_grade_ppm: effective_grade,
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
        scenarios.push(evaluate_scenario(scenario, corpus.mass_tolerance_ppm));
    }

    let expected_controls_matched = scenarios
        .iter()
        .filter(|scenario| scenario.expected_control_matched)
        .count();
    let all_controls_matched = !duplicate_id
        && scenarios.len() == 8
        && expected_controls_matched == scenarios.len()
        && corpus.suite_id == "ORE_ASSAY_REFERENCE_BENCHMARK"
        && corpus.suite_version == "0.1.0"
        && corpus.evidence_class == "synthetic"
        && corpus.dry_mass_unit == "mg"
        && corpus.assay_unit == "ppm_mass_fraction"
        && corpus.mass_tolerance_ppm <= 100_000
        && nonempty(&corpus.claim_ceiling);

    Receipt {
        result: if all_controls_matched {
            "validated"
        } else {
            "contract_mismatch"
        },
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
            "usage: ore-assay-bench <corpus.json> --source-commit <40-hex-commit>".to_owned(),
        );
    }
    let source_commit = args[2].to_ascii_lowercase();
    if source_commit.len() != 40
        || !source_commit
            .chars()
            .all(|character| character.is_ascii_hexdigit())
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
            "../../../docs/tech/ore-assay-benchmark-corpus-v0.1.json"
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
    fn below_detection_is_not_encoded_as_zero() {
        let mut corpus = corpus();
        let tolerance = corpus.mass_tolerance_ppm;
        let scenario = &mut corpus.scenarios[1];
        scenario.sample.assays[0].value_ppm = None;
        let result = evaluate_scenario(scenario, tolerance);
        assert_eq!(result.actual_disposition, Disposition::Pass);
        assert!(result.actual_reason_codes.is_empty());
    }

    #[test]
    fn conversion_reference_without_normalized_value_does_not_pass() {
        let mut corpus = corpus();
        let tolerance = corpus.mass_tolerance_ppm;
        let scenario = &mut corpus.scenarios[2];
        scenario.grade_comparison.as_mut().unwrap().conversion_evidence_id =
            Some("conversion-receipt-001".to_owned());
        let result = evaluate_scenario(scenario, tolerance);
        assert!(result
            .actual_reason_codes
            .contains(&"GRADE_BASIS_MISMATCH".to_owned()));
    }

    #[test]
    fn explicit_conversion_and_normalized_grade_can_be_compared() {
        let mut corpus = corpus();
        let tolerance = corpus.mass_tolerance_ppm;
        let scenario = &mut corpus.scenarios[2];
        let grade = scenario.grade_comparison.as_mut().unwrap();
        grade.conversion_evidence_id = Some("conversion-receipt-001".to_owned());
        grade.normalized_value_ppm = Some(650_000);
        let result = evaluate_scenario(scenario, tolerance);
        assert_eq!(result.actual_disposition, Disposition::Pass);
        assert_eq!(result.effective_grade_ppm, Some(650_000));
    }

    #[test]
    fn mineral_phase_fraction_requires_dry_mass_basis() {
        let mut corpus = corpus();
        let tolerance = corpus.mass_tolerance_ppm;
        let scenario = &mut corpus.scenarios[0];
        scenario.sample.assays[1].basis = AssayBasis::WetMass;
        let result = evaluate_scenario(scenario, tolerance);
        assert!(result
            .actual_reason_codes
            .contains(&"MINERAL_PHASE_BASIS_UNSUPPORTED".to_owned()));
    }

    #[test]
    fn below_detection_requires_a_positive_detection_limit() {
        let mut corpus = corpus();
        let tolerance = corpus.mass_tolerance_ppm;
        let scenario = &mut corpus.scenarios[1];
        scenario.sample.assays[0].value_ppm = None;
        scenario.sample.assays[0].detection_limit_ppm = None;
        let result = evaluate_scenario(scenario, tolerance);
        assert!(result
            .actual_reason_codes
            .contains(&"DETECTION_LIMIT_MISSING_OR_INVALID".to_owned()));
    }

    #[test]
    fn source_record_is_required() {
        let mut corpus = corpus();
        let tolerance = corpus.mass_tolerance_ppm;
        let scenario = &mut corpus.scenarios[0];
        scenario.sample.source_record_id.clear();
        let result = evaluate_scenario(scenario, tolerance);
        assert!(result
            .actual_reason_codes
            .contains(&"SOURCE_REFERENCE_MISSING".to_owned()));
    }
}
