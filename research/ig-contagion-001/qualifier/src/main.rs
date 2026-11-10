use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

const PROTOCOL_ID: &str = "IG-CONTAGION-001";
const PROTOCOL_VERSION: &str = "0.1.0";
const MATRIX_SCHEMA_VERSION: &str = "ig-contagion-scenario-matrix-v0.1";
const EXPECTED_FIXTURES: usize = 40;
const EXPECTED_FAMILIES: usize = 8;
const PER_FAMILY: usize = 5;
const MIN_TOPOLOGIES: usize = 6;

const REQUIRED_STAGES: [&str; 5] = [
    "private_baseline",
    "public_interaction",
    "private_exposure_reelicitation",
    "private_context_reset_control",
    "optional_delayed_private_reelicitation",
];

const REQUIRED_RECEIPT_FIELDS: [&str; 7] = [
    "topology_profile_id_and_digest",
    "rng_algorithm_version_and_seed",
    "ordered_message_log_digest",
    "participant_specific_exposure_memory_digest",
    "baseline_public_private_output_digests",
    "evaluator_identity_and_version",
    "analysis_plan_digest",
];

const REQUIRED_STATISTICAL_GATES: [&str; 8] = [
    "preregister_hypotheses_before_confirmatory_runs",
    "paired_comparisons_for_same_message_different_topology",
    "multiple_seeds_required",
    "separate_exploratory_and_confirmatory_results",
    "report_intervals_and_variance",
    "evaluator_must_be_independent_of_candidate_agent",
    "missing_measurement_is_not_no_change",
    "no_aggregate_contagion_score",
];

#[derive(Debug, PartialEq, Eq)]
struct QualificationReport {
    fixtures: usize,
    families: usize,
    topology_families: usize,
    red_team: usize,
    positive_controls: usize,
    measurement_controls: usize,
    paired_centrality_blocks: usize,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)?
        .as_str()
        .filter(|text| !text.trim().is_empty())
}

fn string_set(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn add(errors: &mut Vec<String>, context: &str, message: impl AsRef<str>) {
    errors.push(format!("{context}: {}", message.as_ref()));
}

fn require_strings(
    actual: &BTreeSet<String>,
    expected: &[&str],
    context: &str,
    errors: &mut Vec<String>,
) {
    for item in expected {
        if !actual.contains(*item) {
            add(
                errors,
                context,
                format!("required item '{item}' is missing"),
            );
        }
    }
}

fn check_schema_header(
    schema: &Value,
    expected_id: &str,
    required_fields: &[&str],
    context: &str,
    errors: &mut Vec<String>,
) {
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        add(errors, context, "unexpected JSON Schema dialect");
    }
    if schema["$id"].as_str() != Some(expected_id) {
        add(
            errors,
            context,
            "schema $id does not match expected identity",
        );
    }
    if schema["type"].as_str() != Some("object") {
        add(errors, context, "top-level schema type must be object");
    }
    if schema["additionalProperties"].as_bool() != Some(false) {
        add(
            errors,
            context,
            "top-level schema must reject additional properties",
        );
    }
    let required = string_set(&schema["required"]);
    require_strings(&required, required_fields, context, errors);
}

fn validate_protocol(protocol: &Value, errors: &mut Vec<String>) {
    if field(protocol, "protocol_id") != Some(PROTOCOL_ID) {
        add(errors, "protocol", "unexpected protocol_id");
    }
    if field(protocol, "version") != Some(PROTOCOL_VERSION) {
        add(errors, "protocol", "unexpected protocol version");
    }
    if field(protocol, "status") != Some("research-proposal") {
        add(errors, "protocol", "protocol must remain research-proposal");
    }

    let stages = protocol["experiment_stages"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let stage_ids: BTreeSet<String> = stages
        .iter()
        .filter_map(|stage| field(stage, "id"))
        .map(ToOwned::to_owned)
        .collect();
    require_strings(
        &stage_ids,
        &REQUIRED_STAGES,
        "protocol.experiment_stages",
        errors,
    );

    let private = &protocol["private_measurement_semantics"];
    for (key, expected, description) in [
        (
            "live_peer_access",
            false,
            "private re-elicitation must remove live peer access",
        ),
        (
            "current_group_messages_visible",
            false,
            "current group messages must be hidden in private re-elicitation",
        ),
        (
            "baseline_answer_visible",
            false,
            "baseline answer must not be shown in post-discussion re-elicitation",
        ),
        (
            "exposure_memory_preserved",
            true,
            "primary persistence measurement must preserve exposure memory",
        ),
        (
            "exposure_memory_identity_required",
            true,
            "exposure memory must have an explicit identity",
        ),
        (
            "context_reset_is_separate_control",
            true,
            "full context reset must remain a separate control",
        ),
    ] {
        if private[key].as_bool() != Some(expected) {
            add(
                errors,
                "protocol.private_measurement_semantics",
                description,
            );
        }
    }

    if protocol["human_study_gate"]["authorized"].as_bool() != Some(false) {
        add(
            errors,
            "protocol.human_study_gate",
            "human-subject recruitment must remain unauthorized",
        );
    }

    let topology_ids: BTreeSet<String> = protocol["topology_profiles"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|profile| field(profile, "id"))
        .map(ToOwned::to_owned)
        .collect();
    if topology_ids.len() < MIN_TOPOLOGIES {
        add(
            errors,
            "protocol.topology_profiles",
            format!("at least {MIN_TOPOLOGIES} unique topologies are required"),
        );
    }

    let receipt_fields = string_set(&protocol["receipt_contract"]["required_commitments"]);
    require_strings(
        &receipt_fields,
        &REQUIRED_RECEIPT_FIELDS,
        "protocol.receipt_contract",
        errors,
    );

    let statistical = &protocol["statistical_protocol"];
    for gate in REQUIRED_STATISTICAL_GATES {
        if statistical[gate].as_bool() != Some(true) {
            add(
                errors,
                "protocol.statistical_protocol",
                format!("required gate '{gate}' must remain true"),
            );
        }
    }

    let invariants = string_set(&protocol["non_compensatory_invariants"]);
    for item in [
        "public_stance_and_private_stance_are_distinct_observations",
        "private_reelicitation_has_no_live_peer_access",
        "exposure_memory_is_explicit_and_content_bound",
        "context_reset_is_not_mislabeled_as_persistence",
        "missing_private_measurement_is_not_no_change",
        "stance_change_is_not_automatically_harmful",
        "evidence_supported_revision_is_a_positive_control",
        "synthetic_agent_results_are_not_human_behavior_evidence",
        "no_result_grants_governance_authority",
    ] {
        if !invariants.contains(item) {
            add(
                errors,
                "protocol.non_compensatory_invariants",
                format!("missing invariant '{item}'"),
            );
        }
    }
}

fn validate_matrix(
    matrix: &Value,
    protocol: &Value,
    errors: &mut Vec<String>,
) -> QualificationReport {
    if field(matrix, "schema_version") != Some(MATRIX_SCHEMA_VERSION) {
        add(errors, "matrix", "unexpected schema_version");
    }
    if field(matrix, "status") != Some("synthetic_scenario_design_unexecuted") {
        add(
            errors,
            "matrix",
            "scenario matrix must remain marked unexecuted",
        );
    }
    if field(matrix, "protocol_id") != Some(PROTOCOL_ID)
        || field(matrix, "protocol_version") != Some(PROTOCOL_VERSION)
    {
        add(errors, "matrix", "protocol identity/version mismatch");
    }

    let families = matrix["families"].as_array().cloned().unwrap_or_default();
    if families.len() != EXPECTED_FAMILIES {
        add(
            errors,
            "matrix.families",
            format!(
                "expected {EXPECTED_FAMILIES} families, found {}",
                families.len()
            ),
        );
    }
    let family_ids: BTreeSet<String> = families
        .iter()
        .filter_map(|family| field(family, "id"))
        .map(ToOwned::to_owned)
        .collect();
    if family_ids.len() != families.len() {
        add(
            errors,
            "matrix.families",
            "family IDs must be non-empty and unique",
        );
    }

    let protocol_topologies: BTreeSet<String> = protocol["topology_profiles"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|profile| field(profile, "id"))
        .map(ToOwned::to_owned)
        .collect();
    let protocol_conditions = string_set(&protocol["experimental_conditions"]);

    let Some(fixtures) = matrix["fixtures"].as_array() else {
        add(errors, "matrix.fixtures", "fixtures must be an array");
        return QualificationReport {
            fixtures: 0,
            families: family_ids.len(),
            topology_families: 0,
            red_team: 0,
            positive_controls: 0,
            measurement_controls: 0,
            paired_centrality_blocks: 0,
        };
    };

    if fixtures.len() != EXPECTED_FIXTURES {
        add(
            errors,
            "matrix.fixtures",
            format!(
                "expected {EXPECTED_FIXTURES} fixtures, found {}",
                fixtures.len()
            ),
        );
    }
    if matrix["fixture_count"].as_u64() != Some(fixtures.len() as u64) {
        add(errors, "matrix", "fixture_count must equal fixtures.length");
    }
    if matrix["family_count"].as_u64() != Some(family_ids.len() as u64) {
        add(
            errors,
            "matrix",
            "family_count must equal unique family count",
        );
    }

    let mut ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut topologies = BTreeSet::new();
    let mut red_team = 0usize;
    let mut positive = 0usize;
    let mut measurement = 0usize;
    let mut paired_blocks = BTreeSet::new();

    for (index, fixture) in fixtures.iter().enumerate() {
        let context = format!("matrix.fixtures[{index}]");
        let id = field(fixture, "fixture_id").unwrap_or_default();
        if id.len() != 7
            || !id.starts_with("IGC-")
            || !id[4..].bytes().all(|byte| byte.is_ascii_digit())
        {
            add(errors, &context, "fixture_id must match IGC-NNN");
        }
        if !ids.insert(id.to_string()) {
            add(errors, &context, format!("duplicate fixture_id '{id}'"));
        }

        let family = field(fixture, "family_id").unwrap_or_default();
        if !family_ids.contains(family) {
            add(errors, &context, format!("undeclared family_id '{family}'"));
        }
        *family_counts.entry(family.to_string()).or_default() += 1;

        let topology = field(fixture, "topology_family").unwrap_or_default();
        if !protocol_topologies.contains(topology) {
            add(
                errors,
                &context,
                format!("topology '{topology}' is not declared by protocol"),
            );
        }
        topologies.insert(topology.to_string());

        if fixture["graph_generator_issue"].as_str()
            != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11248")
        {
            add(
                errors,
                &context,
                "graph generator must compose with Symthaea #11248",
            );
        }
        if field(fixture, "graph_profile_ref").is_none() {
            add(errors, &context, "graph_profile_ref is required");
        }
        if let Some(condition) = field(fixture, "condition") {
            if !protocol_conditions.contains(condition) {
                add(
                    errors,
                    &context,
                    format!("condition '{condition}' is not declared by protocol"),
                );
            }
        } else {
            add(errors, &context, "condition is required");
        }

        match field(fixture, "class") {
            Some("red_team") => red_team += 1,
            Some("positive_control") => positive += 1,
            Some("measurement_control") => measurement += 1,
            Some(other) => add(errors, &context, format!("unknown fixture class '{other}'")),
            None => add(errors, &context, "fixture class is required"),
        }

        match fixture["seed"].as_u64() {
            Some(seed) if seed > 0 => {
                if !seeds.insert(seed) {
                    add(errors, &context, format!("duplicate seed '{seed}'"));
                }
            }
            _ => add(errors, &context, "seed must be a positive integer"),
        }

        let expected = string_set(&fixture["expected_measurement_properties"]);
        for property in [
            "baseline_private_measurement_recorded",
            "public_interaction_trace_recorded",
            "private_post_measurement_recorded_or_explicit_not_run",
            "evidence_exposure_and_provenance_recorded",
            "no_live_peer_access_in_private_reelicitation",
            "exposure_memory_digest_recorded_when_persistence_is_measured",
            "claim_ceiling_preserved",
        ] {
            if !expected.contains(property) {
                add(
                    errors,
                    &context,
                    format!("expected measurement property '{property}' is missing"),
                );
            }
        }

        if family == "centrality_placement" {
            if let Some(pair) = field(fixture, "paired_block_id") {
                paired_blocks.insert(pair.to_string());
            } else {
                add(
                    errors,
                    &context,
                    "centrality placement fixture needs paired_block_id",
                );
            }
        }
    }

    for family in &family_ids {
        let count = family_counts.get(family).copied().unwrap_or_default();
        if count != PER_FAMILY {
            add(
                errors,
                "matrix.families",
                format!("family '{family}' must contain {PER_FAMILY} fixtures; found {count}"),
            );
        }
    }

    if !fixtures.iter().any(|fixture| {
        fixture["family_id"] == "no_discussion_repeat" && fixture["class"] == "measurement_control"
    }) {
        add(
            errors,
            "matrix.fixtures",
            "no-discussion repeat measurement controls are required",
        );
    }
    if !fixtures.iter().any(|fixture| {
        fixture["family_id"] == "evidence_supported_revision"
            && fixture["class"] == "positive_control"
    }) {
        add(
            errors,
            "matrix.fixtures",
            "evidence-supported revision positive controls are required",
        );
    }
    if !fixtures.iter().any(|fixture| {
        fixture["family_id"] == "unsupported_social_influence" && fixture["class"] == "red_team"
    }) {
        add(
            errors,
            "matrix.fixtures",
            "unsupported social influence red-team fixtures are required",
        );
    }
    if paired_blocks.len() < 2 {
        add(
            errors,
            "matrix.fixtures",
            "at least two paired centrality-placement blocks are required",
        );
    }
    if topologies.len() < MIN_TOPOLOGIES {
        add(
            errors,
            "matrix.fixtures",
            format!("at least {MIN_TOPOLOGIES} topology families must be represented"),
        );
    }

    QualificationReport {
        fixtures: fixtures.len(),
        families: family_ids.len(),
        topology_families: topologies.len(),
        red_team,
        positive_controls: positive,
        measurement_controls: measurement,
        paired_centrality_blocks: paired_blocks.len(),
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let content = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&content).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn validate(
    protocol: &Value,
    protocol_schema: &Value,
    matrix: &Value,
    matrix_schema: &Value,
) -> Result<QualificationReport, Vec<String>> {
    let mut errors = Vec::new();
    check_schema_header(
        protocol_schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-contagion-001/IG_CONTAGION_001_PROTOCOL_V0_1.schema.json",
        &[
            "protocol_id",
            "version",
            "status",
            "experiment_stages",
            "private_measurement_semantics",
            "receipt_contract",
        ],
        "protocol_schema",
        &mut errors,
    );
    check_schema_header(
        matrix_schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-contagion-001/IG_CONTAGION_001_SCENARIO_MATRIX_V0_1.schema.json",
        &[
            "schema_version",
            "status",
            "protocol_id",
            "fixture_count",
            "families",
            "fixtures",
        ],
        "matrix_schema",
        &mut errors,
    );
    validate_protocol(protocol, &mut errors);
    let report = validate_matrix(matrix, protocol, &mut errors);
    if errors.is_empty() {
        Ok(report)
    } else {
        Err(errors)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: ig-contagion-qualifier <protocol.json> <protocol.schema.json> <matrix.json> <matrix.schema.json>"
        );
        return ExitCode::from(2);
    }
    let protocol = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let protocol_schema = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let matrix = match read_json(&args[3]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let matrix_schema = match read_json(&args[4]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    match validate(&protocol, &protocol_schema, &matrix, &matrix_schema) {
        Ok(report) => {
            println!(
                "QUALIFIED STRUCTURAL SCENARIO CONTRACT: fixtures={} families={} topologies={} red_team={} positive_controls={} measurement_controls={} paired_centrality_blocks={}",
                report.fixtures,
                report.families,
                report.topology_families,
                report.red_team,
                report.positive_controls,
                report.measurement_controls,
                report.paired_centrality_blocks
            );
            println!(
                "NOT EXECUTED: no agent interaction, stance transition, or human-subject outcome is implied."
            );
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("QUALIFICATION FAILED ({} findings):", errors.len());
            for error in errors {
                eprintln!(" - {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROTOCOL: &str = include_str!("../../IG_CONTAGION_001_PROTOCOL_V0_1.json");
    const PROTOCOL_SCHEMA: &str = include_str!("../../IG_CONTAGION_001_PROTOCOL_V0_1.schema.json");
    const MATRIX: &str = include_str!("../../IG_CONTAGION_001_SCENARIO_MATRIX_V0_1.json");
    const MATRIX_SCHEMA: &str =
        include_str!("../../IG_CONTAGION_001_SCENARIO_MATRIX_V0_1.schema.json");

    fn parsed(text: &str) -> Value {
        serde_json::from_str(text).expect("checked-in JSON must parse")
    }

    #[test]
    fn checked_in_protocol_and_forty_scenarios_validate() {
        let report = validate(
            &parsed(PROTOCOL),
            &parsed(PROTOCOL_SCHEMA),
            &parsed(MATRIX),
            &parsed(MATRIX_SCHEMA),
        )
        .expect("protocol and matrix should validate");
        assert_eq!(report.fixtures, 40);
        assert_eq!(report.families, 8);
        assert!(report.topology_families >= 6);
        assert_eq!(report.measurement_controls, 5);
        assert_eq!(report.positive_controls, 6);
        assert_eq!(report.paired_centrality_blocks, 5);
    }

    #[test]
    fn human_subject_gate_cannot_be_enabled() {
        let mut protocol = parsed(PROTOCOL);
        protocol["human_study_gate"]["authorized"] = Value::Bool(true);
        let errors = validate(
            &protocol,
            &parsed(PROTOCOL_SCHEMA),
            &parsed(MATRIX),
            &parsed(MATRIX_SCHEMA),
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("human-subject recruitment must remain unauthorized"))
        );
    }

    #[test]
    fn private_peer_access_cannot_be_enabled() {
        let mut protocol = parsed(PROTOCOL);
        protocol["private_measurement_semantics"]["live_peer_access"] = Value::Bool(true);
        let errors = validate(
            &protocol,
            &parsed(PROTOCOL_SCHEMA),
            &parsed(MATRIX),
            &parsed(MATRIX_SCHEMA),
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("remove live peer access"))
        );
    }

    #[test]
    fn exposure_memory_cannot_be_dropped() {
        let mut protocol = parsed(PROTOCOL);
        protocol["private_measurement_semantics"]["exposure_memory_preserved"] = Value::Bool(false);
        let errors = validate(
            &protocol,
            &parsed(PROTOCOL_SCHEMA),
            &parsed(MATRIX),
            &parsed(MATRIX_SCHEMA),
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("preserve exposure memory"))
        );
    }

    #[test]
    fn context_reset_must_remain_a_separate_control() {
        let mut protocol = parsed(PROTOCOL);
        protocol["private_measurement_semantics"]["context_reset_is_separate_control"] =
            Value::Bool(false);
        let errors = validate(
            &protocol,
            &parsed(PROTOCOL_SCHEMA),
            &parsed(MATRIX),
            &parsed(MATRIX_SCHEMA),
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("separate control"))
        );
    }

    #[test]
    fn missing_measurement_cannot_be_treated_as_no_change() {
        let mut protocol = parsed(PROTOCOL);
        protocol["statistical_protocol"]["missing_measurement_is_not_no_change"] =
            Value::Bool(false);
        let errors = validate(
            &protocol,
            &parsed(PROTOCOL_SCHEMA),
            &parsed(MATRIX),
            &parsed(MATRIX_SCHEMA),
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("missing_measurement_is_not_no_change"))
        );
    }

    #[test]
    fn thirty_nine_scenarios_are_rejected() {
        let mut matrix = parsed(MATRIX);
        matrix["fixtures"].as_array_mut().unwrap().pop();
        matrix["fixture_count"] = Value::from(39);
        let mut errors = Vec::new();
        let report = validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert_eq!(report.fixtures, 39);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("expected 40 fixtures"))
        );
    }

    #[test]
    fn duplicate_fixture_ids_are_rejected() {
        let mut matrix = parsed(MATRIX);
        let first = matrix["fixtures"][0]["fixture_id"].clone();
        matrix["fixtures"][1]["fixture_id"] = first;
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate fixture_id"))
        );
    }

    #[test]
    fn duplicate_seeds_are_rejected() {
        let mut matrix = parsed(MATRIX);
        let first = matrix["fixtures"][0]["seed"].clone();
        matrix["fixtures"][1]["seed"] = first;
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(errors.iter().any(|error| error.contains("duplicate seed")));
    }

    #[test]
    fn undeclared_topology_is_rejected() {
        let mut matrix = parsed(MATRIX);
        matrix["fixtures"][0]["topology_family"] = Value::String("unregistered-topology".into());
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("not declared by protocol"))
        );
    }

    #[test]
    fn unknown_scenario_family_is_rejected() {
        let mut matrix = parsed(MATRIX);
        matrix["fixtures"][0]["family_id"] = Value::String("invented-family".into());
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("undeclared family_id"))
        );
    }

    #[test]
    fn declared_family_must_have_five_scenarios() {
        let mut matrix = parsed(MATRIX);
        matrix["fixtures"]
            .as_array_mut()
            .unwrap()
            .retain(|fixture| fixture["fixture_id"] != "IGC-001");
        matrix["fixture_count"] = Value::from(39);
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("must contain 5 fixtures"))
        );
    }

    #[test]
    fn centrality_placement_requires_paired_block_identity() {
        let mut matrix = parsed(MATRIX);
        let fixture = matrix["fixtures"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|fixture| fixture["family_id"] == "centrality_placement")
            .unwrap();
        fixture["paired_block_id"] = Value::Null;
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("requires paired_block_id"))
        );
    }

    #[test]
    fn positive_evidence_revision_control_is_required() {
        let mut matrix = parsed(MATRIX);
        for fixture in matrix["fixtures"].as_array_mut().unwrap() {
            if fixture["family_id"] == "evidence_supported_revision" {
                fixture["class"] = Value::String("red_team".into());
            }
        }
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("positive controls are required"))
        );
    }

    #[test]
    fn no_discussion_measurement_control_is_required() {
        let mut matrix = parsed(MATRIX);
        for fixture in matrix["fixtures"].as_array_mut().unwrap() {
            if fixture["family_id"] == "no_discussion_repeat" {
                fixture["class"] = Value::String("red_team".into());
            }
        }
        let mut errors = Vec::new();
        validate_matrix(&matrix, &parsed(PROTOCOL), &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("measurement controls are required"))
        );
    }

    #[test]
    fn context_reset_stage_and_exposure_memory_receipt_are_mandatory() {
        let mut protocol = parsed(PROTOCOL);
        protocol["experiment_stages"]
            .as_array_mut()
            .unwrap()
            .retain(|stage| stage["id"] != "private_context_reset_control");
        protocol["receipt_contract"]["required_commitments"]
            .as_array_mut()
            .unwrap()
            .retain(|field| field != "participant_specific_exposure_memory_digest");
        let mut errors = Vec::new();
        validate_protocol(&protocol, &mut errors);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("private_context_reset_control"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("participant_specific_exposure_memory_digest"))
        );
    }
}
