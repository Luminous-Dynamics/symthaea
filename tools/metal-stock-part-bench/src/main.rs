use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, fs, process::ExitCode};

const VERSION: &str = "0.1.0";
type Check<T> = Result<T, String>;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    Pass,
    Reject,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum IdentityState {
    Resolved,
    Unresolved,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FeedstockKind {
    Virgin,
    QualifiedSecondary,
    MixedScrap,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ContaminationState {
    Clear,
    Unknown,
    Contaminated,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RouteKind {
    NewPart,
    RepairRemanufacture,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MeasurementSource {
    PhysicalMeasurement,
    SimulationOnly,
    None,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CalibrationState {
    Current,
    Stale,
    Missing,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    suite_id: String,
    suite_version: String,
    evidence_class: String,
    claim_ceiling: String,
    mass_tolerance_ppm: u64,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    scenario_id: String,
    expected_disposition: Disposition,
    expected_reason_codes: BTreeSet<String>,
    expected_assertions: Vec<String>,
    negative_control: String,
    stock: StockRecord,
    route: RoutePlan,
    part: PartRequirement,
    flow: FlowLedger,
    inspection: InspectionRecord,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StockRecord {
    lot_id: String,
    alloy_id: String,
    material_identity: IdentityState,
    material_state: String,
    feedstock_kind: FeedstockKind,
    contamination_state: ContaminationState,
    grade_evidence_ref: Option<String>,
    source_record_ref: Option<String>,
    custody_evidence_ref: Option<String>,
    requalification_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RoutePlan {
    route_id: String,
    route_kind: RouteKind,
    input_lot_id: String,
    input_material_state: String,
    output_material_state: String,
    planned_operations: Vec<String>,
    executed_operations: Vec<String>,
    required_operations: Vec<String>,
    required_capabilities: Vec<String>,
    available_capabilities: Vec<String>,
    process_record_ref: Option<String>,
    repair_scope_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PartRequirement {
    part_id: String,
    requirements_ref: String,
    required_alloy_id: String,
    required_material_state: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FlowLedger {
    input_mass_g: u64,
    part_mass_g: u64,
    clean_offcut_mass_g: u64,
    chips_mass_g: u64,
    mixed_scrap_mass_g: u64,
    residue_mass_g: u64,
    process_loss_mass_g: u64,
    claimed_clean_offcut_recovery_g: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InspectionRecord {
    source: MeasurementSource,
    calibration: CalibrationState,
    calibration_record_ref: Option<String>,
    method_id: Option<String>,
    measurement_record_ref: Option<String>,
    nominal_dimension_um: u64,
    measured_dimension_um: Option<u64>,
    measurement_uncertainty_um: Option<u64>,
    tolerance_um: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ScenarioResult {
    scenario_id: String,
    expected_disposition: Disposition,
    actual_disposition: Disposition,
    expected_reason_codes: Vec<String>,
    actual_reason_codes: Vec<String>,
    expected_control_matched: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Receipt {
    result: String,
    validator_version: String,
    suite_id: String,
    suite_version: String,
    evidence_class: String,
    source_commit: String,
    corpus_sha256: String,
    scenario_count: usize,
    expected_controls_matched: usize,
    scenarios: Vec<ScenarioResult>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn option_nonempty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|item| nonempty(item))
}

fn validate_corpus(corpus: &Corpus) -> Check<()> {
    if corpus.suite_id != "METAL_STOCK_TO_PART_BENCHMARK" {
        return Err("unexpected suite_id".to_owned());
    }
    if !nonempty(&corpus.suite_version)
        || corpus.evidence_class != "synthetic"
        || !nonempty(&corpus.claim_ceiling)
        || corpus.scenarios.is_empty()
        || corpus.mass_tolerance_ppm > 10_000
    {
        return Err("invalid corpus metadata or mass tolerance".to_owned());
    }

    let mut identifiers = BTreeSet::new();
    for scenario in &corpus.scenarios {
        if !nonempty(&scenario.scenario_id)
            || !identifiers.insert(scenario.scenario_id.clone())
            || !nonempty(&scenario.negative_control)
            || scenario.expected_assertions.is_empty()
            || !nonempty(&scenario.stock.lot_id)
            || !nonempty(&scenario.stock.alloy_id)
            || !nonempty(&scenario.route.route_id)
            || !nonempty(&scenario.part.part_id)
            || !nonempty(&scenario.part.requirements_ref)
            || !nonempty(&scenario.part.required_alloy_id)
        {
            return Err("scenario identity or explanatory metadata is invalid".to_owned());
        }
    }
    Ok(())
}

fn evaluate_scenario(scenario: &Scenario, mass_tolerance_ppm: u64) -> ScenarioResult {
    let mut reasons = BTreeSet::new();
    let stock = &scenario.stock;
    let route = &scenario.route;
    let part = &scenario.part;
    let flow = &scenario.flow;
    let inspection = &scenario.inspection;

    if stock.material_identity != IdentityState::Resolved
        || matches!(
            stock.feedstock_kind,
            FeedstockKind::MixedScrap | FeedstockKind::Unknown
        )
    {
        reasons.insert("FEEDSTOCK_IDENTITY_UNRESOLVED".to_owned());
    }
    if stock.alloy_id != part.required_alloy_id {
        reasons.insert("ALLOY_MISMATCH".to_owned());
    }
    if !option_nonempty(&stock.grade_evidence_ref) {
        reasons.insert("GRADE_EVIDENCE_MISSING".to_owned());
    }
    if !option_nonempty(&stock.source_record_ref) {
        reasons.insert("STOCK_PROVENANCE_MISSING".to_owned());
    }

    if stock.feedstock_kind == FeedstockKind::QualifiedSecondary {
        if !option_nonempty(&stock.custody_evidence_ref) {
            reasons.insert("SECONDARY_CUSTODY_MISSING".to_owned());
        }
        match stock.contamination_state {
            ContaminationState::Clear => {}
            ContaminationState::Unknown => {
                reasons.insert("SECONDARY_CONTAMINATION_UNRESOLVED".to_owned());
            }
            ContaminationState::Contaminated => {
                reasons.insert("SECONDARY_CONTAMINATION_REJECTED".to_owned());
            }
        }
        if !option_nonempty(&stock.requalification_ref) {
            reasons.insert("SECONDARY_REQUALIFICATION_MISSING".to_owned());
        }
    }

    if route.input_lot_id != stock.lot_id {
        reasons.insert("LOT_LINEAGE_MISMATCH".to_owned());
    }
    if route.input_material_state != stock.material_state {
        reasons.insert("INPUT_MATERIAL_STATE_MISMATCH".to_owned());
    }
    if !option_nonempty(&route.process_record_ref) {
        reasons.insert("PROCESS_HISTORY_MISSING".to_owned());
    }
    if route
        .required_operations
        .iter()
        .any(|operation| !route.planned_operations.contains(operation))
    {
        reasons.insert("PROCESS_OPERATION_NOT_PLANNED".to_owned());
    }
    if route
        .required_operations
        .iter()
        .any(|operation| !route.executed_operations.contains(operation))
    {
        reasons.insert("PROCESS_OPERATION_MISSING".to_owned());
    }
    if route
        .executed_operations
        .iter()
        .any(|operation| !route.planned_operations.contains(operation))
    {
        reasons.insert("UNPLANNED_OPERATION_EXECUTED".to_owned());
    }
    if route.output_material_state != part.required_material_state {
        reasons.insert("MATERIAL_STATE_MISMATCH".to_owned());
    }
    if route
        .required_capabilities
        .iter()
        .any(|capability| !route.available_capabilities.contains(capability))
    {
        reasons.insert("ROUTE_CAPABILITY_UNAVAILABLE".to_owned());
    }
    if route.route_kind == RouteKind::RepairRemanufacture
        && !option_nonempty(&route.repair_scope_ref)
    {
        reasons.insert("REPAIR_SCOPE_MISSING".to_owned());
    }

    let output_mass = u128::from(flow.part_mass_g)
        + u128::from(flow.clean_offcut_mass_g)
        + u128::from(flow.chips_mass_g)
        + u128::from(flow.mixed_scrap_mass_g)
        + u128::from(flow.residue_mass_g)
        + u128::from(flow.process_loss_mass_g);
    let input_mass = u128::from(flow.input_mass_g);
    let difference = output_mass.abs_diff(input_mass);
    if input_mass == 0 || difference * 1_000_000 > input_mass * u128::from(mass_tolerance_ppm) {
        reasons.insert("MASS_BALANCE_MISMATCH".to_owned());
    }
    if flow.claimed_clean_offcut_recovery_g > flow.clean_offcut_mass_g {
        reasons.insert("SCRAP_CLASS_MISMATCH".to_owned());
    }

    if inspection.source != MeasurementSource::PhysicalMeasurement {
        reasons.insert("PHYSICAL_INSPECTION_MISSING".to_owned());
    }
    match inspection.calibration {
        CalibrationState::Current if !option_nonempty(&inspection.calibration_record_ref) => {
            reasons.insert("METROLOGY_CALIBRATION_REFERENCE_MISSING".to_owned());
        }
        CalibrationState::Current => {}
        CalibrationState::Stale => {
            reasons.insert("METROLOGY_CALIBRATION_STALE".to_owned());
        }
        CalibrationState::Missing => {
            reasons.insert("METROLOGY_CALIBRATION_MISSING".to_owned());
        }
    }
    if !option_nonempty(&inspection.method_id)
        || !option_nonempty(&inspection.measurement_record_ref)
    {
        reasons.insert("INSPECTION_PROVENANCE_MISSING".to_owned());
    }
    match (
        inspection.measured_dimension_um,
        inspection.measurement_uncertainty_um,
    ) {
        (None, _) => {
            reasons.insert("INSPECTION_VALUE_MISSING".to_owned());
        }
        (Some(measured), None) => {
            reasons.insert("INSPECTION_UNCERTAINTY_MISSING".to_owned());
            if measured.abs_diff(inspection.nominal_dimension_um) > inspection.tolerance_um {
                reasons.insert("PART_OUT_OF_TOLERANCE".to_owned());
            }
        }
        (Some(measured), Some(uncertainty)) => {
            let deviation = measured.abs_diff(inspection.nominal_dimension_um);
            if deviation > inspection.tolerance_um {
                reasons.insert("PART_OUT_OF_TOLERANCE".to_owned());
            } else if u128::from(deviation) + u128::from(uncertainty)
                > u128::from(inspection.tolerance_um)
            {
                reasons.insert("INSPECTION_UNCERTAINTY_EXCEEDS_MARGIN".to_owned());
            }
        }
    }

    let actual_disposition = if reasons.is_empty() {
        Disposition::Pass
    } else {
        Disposition::Reject
    };
    let expected_reason_codes: Vec<_> = scenario.expected_reason_codes.iter().cloned().collect();
    let actual_reason_codes: Vec<_> = reasons.into_iter().collect();
    let expected_control_matched = scenario.expected_disposition == actual_disposition
        && expected_reason_codes == actual_reason_codes;

    ScenarioResult {
        scenario_id: scenario.scenario_id.clone(),
        expected_disposition: scenario.expected_disposition,
        actual_disposition,
        expected_reason_codes,
        actual_reason_codes,
        expected_control_matched,
    }
}

fn source_commit_from_args(args: &[String]) -> Check<String> {
    if args.len() != 4 || args[2] != "--source-commit" {
        return Err("usage: metal-stock-part-bench <corpus.json> --source-commit <40-hex-sha>".to_owned());
    }
    let source_commit = args[3].to_ascii_lowercase();
    if source_commit.len() != 40
        || !source_commit
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err("--source-commit must be exactly 40 hexadecimal characters".to_owned());
    }
    Ok(source_commit)
}

fn run() -> Check<ExitCode> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: metal-stock-part-bench <corpus.json> --source-commit <40-hex-sha>"
                .to_owned(),
        );
    }
    let source_commit = source_commit_from_args(&args)?;
    let bytes = fs::read(&args[1]).map_err(|error| format!("cannot read corpus: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&bytes).map_err(|error| format!("invalid corpus JSON: {error}"))?;
    validate_corpus(&corpus)?;
    let corpus_sha256 = format!("{:x}", Sha256::digest(&bytes));

    let scenarios: Vec<_> = corpus
        .scenarios
        .iter()
        .map(|scenario| evaluate_scenario(scenario, corpus.mass_tolerance_ppm))
        .collect();
    let expected_controls_matched = scenarios
        .iter()
        .filter(|scenario| scenario.expected_control_matched)
        .count();
    let all_controls_matched = expected_controls_matched == scenarios.len();

    let receipt = Receipt {
        result: if all_controls_matched {
            "validated".to_owned()
        } else {
            "contract_mismatch".to_owned()
        },
        validator_version: VERSION.to_owned(),
        suite_id: corpus.suite_id,
        suite_version: corpus.suite_version,
        evidence_class: corpus.evidence_class,
        source_commit,
        corpus_sha256,
        scenario_count: scenarios.len(),
        expected_controls_matched,
        scenarios,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("cannot serialize receipt: {error}"))?
    );
    Ok(if all_controls_matched {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Corpus {
        serde_json::from_str(include_str!(
            "../../../docs/tech/metal-stock-part-corpus-v0.1.json"
        ))
        .expect("frozen synthetic corpus must deserialize")
    }

    #[test]
    fn frozen_corpus_matches_all_known_answers() {
        let corpus = fixture();
        validate_corpus(&corpus).expect("corpus metadata must be valid");
        let results: Vec<_> = corpus
            .scenarios
            .iter()
            .map(|scenario| evaluate_scenario(scenario, corpus.mass_tolerance_ppm))
            .collect();
        assert_eq!(24, results.len());
        assert!(results.iter().all(|result| result.expected_control_matched));
    }

    #[test]
    fn mass_creation_is_rejected() {
        let mut corpus = fixture();
        corpus.scenarios[0].flow.chips_mass_g += 50;
        let result = evaluate_scenario(&corpus.scenarios[0], corpus.mass_tolerance_ppm);
        assert!(result
            .actual_reason_codes
            .contains(&"MASS_BALANCE_MISMATCH".to_owned()));
    }

    #[test]
    fn mixed_scrap_cannot_satisfy_an_exact_alloy_requirement() {
        let mut corpus = fixture();
        corpus.scenarios[0].stock.feedstock_kind = FeedstockKind::MixedScrap;
        corpus.scenarios[0].stock.material_identity = IdentityState::Unresolved;
        let result = evaluate_scenario(&corpus.scenarios[0], corpus.mass_tolerance_ppm);
        assert!(result
            .actual_reason_codes
            .contains(&"FEEDSTOCK_IDENTITY_UNRESOLVED".to_owned()));
    }

    #[test]
    fn unknown_json_fields_fail_closed() {
        let raw = include_str!("../../../docs/tech/metal-stock-part-corpus-v0.1.json").replace(
            "\"suite_id\":",
            "\"unexpected_authority\": true,\n  \"suite_id\":",
        );
        assert!(serde_json::from_str::<Corpus>(&raw).is_err());
    }

    #[test]
    fn simulation_is_not_a_physical_inspection() {
        let mut corpus = fixture();
        corpus.scenarios[0].inspection.source = MeasurementSource::SimulationOnly;
        let result = evaluate_scenario(&corpus.scenarios[0], corpus.mass_tolerance_ppm);
        assert!(result
            .actual_reason_codes
            .contains(&"PHYSICAL_INSPECTION_MISSING".to_owned()));
    }
}
