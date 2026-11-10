use serde_json::Value;
use std::{
    collections::BTreeSet,
    env, fs,
    process::ExitCode,
};

const LEDGER_ID: &str = "IG-REVISION-LEDGER-001";
const MANIFEST_SCHEMA_VERSION: &str = "ig-revision-ledger-fixtures-v0.1";
const EXPECTED_FIXTURES: usize = 17;

#[derive(Debug, PartialEq, Eq)]
struct Report {
    fixtures: usize,
    valid_controls: usize,
    negative_controls: usize,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|value| !value.trim().is_empty())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn add(errors: &mut Vec<String>, context: &str, message: impl AsRef<str>) {
    errors.push(format!("{context}: {}", message.as_ref()));
}

fn string_array(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn string_set(value: &Value) -> BTreeSet<String> {
    string_array(value).into_iter().collect()
}

fn validate_schema_header(
    schema: &Value,
    expected_id: &str,
    required: &[&str],
    context: &str,
    errors: &mut Vec<String>,
) {
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        add(errors, context, "unexpected JSON Schema dialect");
    }
    if schema["$id"].as_str() != Some(expected_id) {
        add(errors, context, "schema $id does not match expected identity");
    }
    if schema["type"].as_str() != Some("object") {
        add(errors, context, "top-level schema type must be object");
    }
    if schema["additionalProperties"].as_bool() != Some(false) {
        add(errors, context, "top-level schema must reject additional properties");
    }
    let actual = string_set(&schema["required"]);
    for item in required {
        if !actual.contains(*item) {
            add(
                errors,
                context,
                format!("required property `{item}` is missing"),
            );
        }
    }
}

fn validate_state(state: &Value, context: &str, errors: &mut Vec<String>) {
    if state.is_null() {
        return;
    }
    if !is_sha256(field(state, "state_sha256").unwrap_or_default()) {
        add(errors, context, "state_sha256 must be a 64-character hexadecimal digest");
    }
    if field(state, "stance").is_none() {
        add(errors, context, "state stance must be non-empty");
    }
    if field(state, "uncertainty_state").is_none() {
        add(errors, context, "state uncertainty_state must be present");
    }
    for key in ["claim_ids", "evidence_ids"] {
        if !state[key].is_array() {
            add(errors, context, format!("state {key} must be an array"));
        }
        let values = string_array(&state[key]);
        if values.len() != string_set(&state[key]).len() {
            add(errors, context, format!("state {key} must not contain duplicates"));
        }
    }
}

fn validate_record(record: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let id = field(record, "revision_event_id").unwrap_or("<missing>");
    let context = format!("record[{id}]");

    for key in ["revision_event_id", "run_id", "scenario_id", "agent_instance_id"] {
        if field(record, key).is_none() {
            add(&mut errors, &context, format!("{key} must be non-empty"));
        }
    }

    let previous = &record["previous_state"];
    let next = &record["next_state"];
    validate_state(previous, &format!("{context}.previous_state"), &mut errors);
    validate_state(next, &format!("{context}.next_state"), &mut errors);

    match field(record, "transition") {
        Some("Changed") => {
            if previous.is_null() || next.is_null() {
                add(&mut errors, &context, "Changed requires previous and next states");
            } else if field(previous, "state_sha256") == field(next, "state_sha256") {
                add(&mut errors, &context, "changed_state_digest_unchanged");
            }
        }
        Some("NoChange") => {
            if previous.is_null() || next.is_null() {
                add(&mut errors, &context, "NoChange requires previous and next states");
            } else if field(previous, "state_sha256") != field(next, "state_sha256") {
                add(&mut errors, &context, "no_change_state_digest_mismatch");
            }
        }
        Some("Abstained") => {
            if next.is_null()
                || (field(next, "stance") != Some("Abstain")
                    && field(next, "uncertainty_state") != Some("Abstained"))
            {
                add(&mut errors, &context, "abstention_state_missing");
            }
        }
        Some("Missing") => {
            if !next.is_null() {
                add(&mut errors, &context, "missing_measurement_post_state_present");
            }
            if !matches!(field(record, "event_status"), Some("NotRun" | "Indeterminate")) {
                add(
                    &mut errors,
                    &context,
                    "missing_measurement_status_must_be_NotRun_or_Indeterminate",
                );
            }
        }
        Some("Invalid") => {
            if field(record, "event_status") != Some("Invalid") {
                add(
                    &mut errors,
                    &context,
                    "Invalid transition requires Invalid event_status",
                );
            }
        }
        Some(_) | None => add(&mut errors, &context, "transition is missing or unsupported"),
    }

    let delta = &record["claim_delta"];
    let mut primary_claims = BTreeSet::new();
    for key in ["added", "removed", "retained"] {
        if !delta[key].is_array() {
            add(&mut errors, &context, format!("claim_delta.{key} must be an array"));
        }
        for claim in string_array(&delta[key]) {
            if !primary_claims.insert(claim) {
                add(&mut errors, &context, "claim_delta_sets_overlap");
            }
        }
    }
    for key in ["contradicted", "retracted"] {
        if !delta[key].is_array() {
            add(&mut errors, &context, format!("claim_delta.{key} must be an array"));
        }
    }

    let evidence_delta = &record["evidence_delta"];
    for key in ["available_before", "newly_available", "newly_unavailable", "invalidated"] {
        if !evidence_delta[key].is_array() {
            add(&mut errors, &context, format!("evidence_delta.{key} must be an array"));
        }
    }

    for message in record["delivered_messages"].as_array().into_iter().flatten() {
        if field(message, "message_id").is_none()
            || field(message, "channel_event_id").is_none()
            || !is_sha256(field(message, "payload_sha256").unwrap_or_default())
        {
            add(
                &mut errors,
                &context,
                "delivered_message_missing_channel_event",
            );
        }
    }

    let report = &record["agent_report"];
    match report["provided"].as_bool() {
        Some(false) => {
            if field(report, "reason_code") != Some("NotProvided")
                || !report["explanation_sha256"].is_null()
            {
                add(
                    &mut errors,
                    &context,
                    "absent agent self-report must not carry a reason or explanation digest",
                );
            }
        }
        Some(true) => {
            if field(report, "reason_code").is_none_or(|reason| reason == "NotProvided")
                || !is_sha256(field(report, "explanation_sha256").unwrap_or_default())
            {
                add(
                    &mut errors,
                    &context,
                    "provided agent self-report requires reason code and explanation digest",
                );
            }
        }
        None => add(&mut errors, &context, "agent_report.provided must be boolean"),
    }
    if field(report, "kind") != Some("self_report") {
        add(&mut errors, &context, "agent report must remain typed as self_report");
    }

    let oracle = &record["oracle"];
    let task_profile = field(oracle, "task_profile");
    let outcome = field(oracle, "outcome");
    match task_profile {
        Some("SyntheticKnownAnswer") => {
            if outcome != Some("NotApplicable") {
                if !is_sha256(field(oracle, "answer_space_sha256").unwrap_or_default()) {
                    add(&mut errors, &context, "synthetic_oracle_missing_answer_space");
                }
                if !is_sha256(field(oracle, "oracle_receipt_sha256").unwrap_or_default()) {
                    add(&mut errors, &context, "synthetic_oracle_missing_receipt");
                }
            }
        }
        Some("OpenWorld" | "NormativePluralism") => {
            if matches!(outcome, Some("Supported" | "Unsupported" | "Contradicted")) {
                add(
                    &mut errors,
                    &context,
                    "open_world_or_pluralism_cannot_have_correctness_outcome",
                );
            }
            if !oracle["answer_space_sha256"].is_null() {
                add(
                    &mut errors,
                    &context,
                    "open_world_or_pluralism_must_not_bind_single_answer_space",
                );
            }
        }
        Some("MeasurementOnly") => {
            if outcome != Some("NotApplicable") || !oracle["answer_space_sha256"].is_null() {
                add(
                    &mut errors,
                    &context,
                    "measurement_only_has_no_decision_correctness_target",
                );
            }
        }
        Some("SocialClaimIntegrity") => {
            if matches!(outcome, Some("Supported" | "Unsupported" | "Contradicted"))
                && !is_sha256(field(oracle, "oracle_receipt_sha256").unwrap_or_default())
            {
                add(
                    &mut errors,
                    &context,
                    "social_claim_integrity_outcome_requires_independent_oracle_receipt",
                );
            }
        }
        _ => add(&mut errors, &context, "unknown oracle task profile"),
    }

    let causal = &record["causal_attribution"];
    match field(causal, "status") {
        Some("MatchedPairEstimate") => {
            let required = field(causal, "matched_pair_id").is_some()
                && field(causal, "treatment_run_id").is_some()
                && field(causal, "control_run_id").is_some()
                && is_sha256(field(causal, "analysis_plan_sha256").unwrap_or_default())
                && is_sha256(field(causal, "effect_estimate_sha256").unwrap_or_default());
            if !required {
                add(
                    &mut errors,
                    &context,
                    "causal_estimate_missing_matched_pair",
                );
            } else if field(causal, "treatment_run_id") == field(causal, "control_run_id") {
                add(
                    &mut errors,
                    &context,
                    "causal_pair_runs_must_be_distinct",
                );
            }
            let matched_inputs = &causal["matched_inputs"];
            let required_dimensions = [
                "task_instance",
                "model_profile",
                "system_prompt",
                "elicitation_prompt",
                "evidence_packet",
                "communication_graph",
                "budget_profile",
                "decoding_profile",
                "rng_stream",
                "evaluator_profile",
            ];
            for dimension in required_dimensions {
                let treatment = field(&matched_inputs[dimension], "treatment_sha256");
                let control = field(&matched_inputs[dimension], "control_sha256");
                if treatment.is_none_or(|value| !is_sha256(value))
                    || control.is_none_or(|value| !is_sha256(value))
                {
                    add(
                        &mut errors,
                        &context,
                        "causal_estimate_missing_matched_pair",
                    );
                    break;
                }
                if treatment != control {
                    add(
                        &mut errors,
                        &context,
                        "causal_pair_shared_input_mismatch",
                    );
                    break;
                }
            }
            let varied_factor = &causal["varied_factor"];
            let treatment_factor = field(varied_factor, "treatment_sha256");
            let control_factor = field(varied_factor, "control_sha256");
            if field(varied_factor, "id").is_none()
                || treatment_factor.is_none_or(|value| !is_sha256(value))
                || control_factor.is_none_or(|value| !is_sha256(value))
            {
                add(
                    &mut errors,
                    &context,
                    "causal_estimate_missing_matched_pair",
                );
            } else if treatment_factor == control_factor {
                add(
                    &mut errors,
                    &context,
                    "causal_pair_has_no_distinct_intervention_factor",
                );
            }
            if field(causal, "pair_validator_id").is_none()
                || !is_sha256(
                    field(causal, "pair_validation_receipt_sha256").unwrap_or_default(),
                )
            {
                add(
                    &mut errors,
                    &context,
                    "causal_estimate_missing_matched_pair",
                );
            }
        }
        Some("NotEstimated" | "InsufficientEvidence") => {
            for key in [
                "matched_pair_id",
                "treatment_run_id",
                "control_run_id",
                "analysis_plan_sha256",
                "effect_estimate_sha256",
                "matched_inputs",
                "varied_factor",
                "pair_validator_id",
                "pair_validation_receipt_sha256",
            ] {
                if !causal[key].is_null() {
                    add(
                        &mut errors,
                        &context,
                        "causal_pair_fields_without_estimate",
                    );
                    break;
                }
            }
        }
        Some(_) | None => add(
            &mut errors,
            &context,
            "causal attribution status is missing or unsupported",
        ),
    }

    let known_evidence = string_set(&evidence_delta["available_before"])
        .into_iter()
        .chain(string_set(&evidence_delta["newly_available"]))
        .collect::<BTreeSet<_>>();
    for link in record["claim_evidence_links"].as_array().into_iter().flatten() {
        if field(link, "claim_id").is_none()
            || field(link, "evidence_id")
                .is_none_or(|evidence| !known_evidence.contains(evidence))
        {
            add(
                &mut errors,
                &context,
                "claim_evidence_link_not_bound_to_available_evidence",
            );
        }
    }

    errors.sort();
    errors.dedup();
    errors
}

fn validate_manifest(
    manifest: &Value,
    manifest_schema: &Value,
    record_schema: &Value,
) -> Result<Report, Vec<String>> {
    let mut errors = Vec::new();

    validate_schema_header(
        manifest_schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-revision-ledger-001/REVISION_LEDGER_FIXTURES_V0_1.schema.json",
        &[
            "schema_version",
            "ledger_id",
            "status",
            "fixture_count",
            "fixtures",
            "claim_ceiling",
        ],
        "manifest_schema",
        &mut errors,
    );
    validate_schema_header(
        record_schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-revision-ledger-001/REVISION_LEDGER_RECORD_V0_1.schema.json",
        &[
            "revision_event_id",
            "run_id",
            "scenario_id",
            "agent_instance_id",
            "stage",
            "transition",
            "causal_attribution",
        ],
        "record_schema",
        &mut errors,
    );

    if field(manifest, "schema_version") != Some(MANIFEST_SCHEMA_VERSION)
        || field(manifest, "ledger_id") != Some(LEDGER_ID)
    {
        add(&mut errors, "manifest", "unexpected ledger or schema identity");
    }
    if field(manifest, "status") != Some("synthetic_fixture_design_unexecuted") {
        add(&mut errors, "manifest", "fixture design must remain marked unexecuted");
    }
    if manifest["owning_issue"].as_str()
        != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11340")
        || manifest["parent_issue"].as_str()
            != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11320")
    {
        add(
            &mut errors,
            "manifest",
            "issue ownership links do not match the declared ledger hierarchy",
        );
    }
    if manifest["communication_graph_owner"].as_str()
        != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11337")
        || manifest["delegation_graph_owner"].as_str()
            != Some("https://github.com/Luminous-Dynamics/symthaea/issues/11248")
    {
        add(
            &mut errors,
            "manifest",
            "communication and delegation graph ownership must remain separate",
        );
    }

    let Some(fixtures) = manifest["fixtures"].as_array() else {
        add(&mut errors, "manifest.fixtures", "fixtures must be an array");
        return Err(errors);
    };
    if fixtures.len() != EXPECTED_FIXTURES
        || manifest["fixture_count"].as_u64() != Some(fixtures.len() as u64)
    {
        add(
            &mut errors,
            "manifest",
            format!("expected exactly {EXPECTED_FIXTURES} fixtures and matching fixture_count"),
        );
    }

    let mut ids = BTreeSet::new();
    let mut valid_controls = 0;
    let mut negative_controls = 0;

    for (index, fixture) in fixtures.iter().enumerate() {
        let context = format!("fixtures[{index}]");
        let id = field(fixture, "fixture_id").unwrap_or_default();
        if id.is_empty() || !ids.insert(id.to_string()) {
            add(&mut errors, &context, "fixture_id must be non-empty and unique");
        }

        let actual = validate_record(&fixture["record"]);
        let expected = string_array(&fixture["expected_violations"]);
        if actual != expected {
            add(
                &mut errors,
                &context,
                format!("oracle mismatch: expected {expected:?}, actual {actual:?}"),
            );
        }
        if expected.is_empty() {
            valid_controls += 1;
        } else {
            negative_controls += 1;
        }
    }

    if valid_controls < 1 || negative_controls < 1 {
        add(
            &mut errors,
            "manifest.fixtures",
            "fixture corpus requires both valid and negative controls",
        );
    }

    if errors.is_empty() {
        Ok(Report {
            fixtures: fixtures.len(),
            valid_controls,
            negative_controls,
        })
    } else {
        Err(errors)
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let content = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&content).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: ig-revision-ledger-qualifier <fixtures.json> <fixtures.schema.json> <record.schema.json>"
        );
        return ExitCode::from(2);
    }

    let manifest = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let manifest_schema = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let record_schema = match read_json(&args[3]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    match validate_manifest(&manifest, &manifest_schema, &record_schema) {
        Ok(report) => {
            println!(
                "QUALIFIED STRUCTURAL REVISION LEDGER: fixtures={} valid_controls={} negative_controls={}",
                report.fixtures, report.valid_controls, report.negative_controls
            );
            println!(
                "NOT CAUSAL EVIDENCE: self-reported reasons remain separate from matched-pair estimates."
            );
            println!(
                "NOT EXECUTED: these fixtures do not establish model behavior or human outcomes."
            );
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!(
                "REVISION LEDGER QUALIFICATION FAILED ({} findings):",
                errors.len()
            );
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

    const MANIFEST: &str =
        include_str!("../../REVISION_LEDGER_FIXTURES_V0_1.json");
    const MANIFEST_SCHEMA: &str =
        include_str!("../../REVISION_LEDGER_FIXTURES_V0_1.schema.json");
    const RECORD_SCHEMA: &str =
        include_str!("../../REVISION_LEDGER_RECORD_V0_1.schema.json");

    fn parsed(text: &str) -> Value {
        serde_json::from_str(text).expect("checked-in JSON must parse")
    }

    #[test]
    fn checked_in_sixteen_fixture_oracles_validate() {
        let report = validate_manifest(
            &parsed(MANIFEST),
            &parsed(MANIFEST_SCHEMA),
            &parsed(RECORD_SCHEMA),
        )
        .expect("manifest and event fixtures must validate");
        assert_eq!(report.fixtures, 17);
        assert_eq!(report.valid_controls, 8);
        assert_eq!(report.negative_controls, 9);
    }

    #[test]
    fn agent_self_report_cannot_supply_a_matched_causal_estimate() {
        let mut record = parsed(MANIFEST)["fixtures"][2]["record"].clone();
        record["agent_report"]["reason_code"] = Value::String("Evidence".into());
        record["causal_attribution"]["status"] =
            Value::String("MatchedPairEstimate".into());
        record["causal_attribution"]["matched_pair_id"] = Value::Null;
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"causal_estimate_missing_matched_pair".to_string()
        ));
    }

    #[test]
    fn matched_pair_estimate_requires_distinct_runs_and_analysis_identity() {
        let mut record = parsed(MANIFEST)["fixtures"][9]["record"].clone();
        record["causal_attribution"]["control_run_id"] =
            record["causal_attribution"]["treatment_run_id"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"causal_pair_runs_must_be_distinct".to_string()
        ));
    }

    #[test]
    fn matched_pair_requires_identical_common_input_commitments() {
        let mut record = parsed(MANIFEST)["fixtures"][9]["record"].clone();
        record["causal_attribution"]["matched_inputs"]["communication_graph"]["control_sha256"] =
            Value::String("f".repeat(64));
        let errors = validate_record(&record);
        assert!(errors.contains(&"causal_pair_shared_input_mismatch".to_string()));
    }

    #[test]
    fn matched_pair_requires_a_distinct_intervention_factor() {
        let mut record = parsed(MANIFEST)["fixtures"][9]["record"].clone();
        record["causal_attribution"]["varied_factor"]["control_sha256"] =
            record["causal_attribution"]["varied_factor"]["treatment_sha256"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(&"causal_pair_has_no_distinct_intervention_factor".to_string()));
    }

    #[test]
    fn non_estimated_event_cannot_carry_a_pair_validator_identity() {
        let mut record = parsed(MANIFEST)["fixtures"][0]["record"].clone();
        record["causal_attribution"]["pair_validator_id"] =
            Value::String("independent-pair-validator/v1".into());
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"causal_pair_fields_without_estimate".to_string()
        ));
    }

    #[test]
    fn a_missing_measurement_is_not_no_change() {
        let mut record = parsed(MANIFEST)["fixtures"][11]["record"].clone();
        record["next_state"] = record["previous_state"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"missing_measurement_post_state_present".to_string()
        ));
    }

    #[test]
    fn normative_pluralism_cannot_be_scored_correct_or_incorrect() {
        let mut record = parsed(MANIFEST)["fixtures"][10]["record"].clone();
        record["oracle"]["outcome"] = Value::String("Supported".into());
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"open_world_or_pluralism_cannot_have_correctness_outcome".to_string()
        ));
    }

    #[test]
    fn open_world_tasks_cannot_be_given_a_single_answer_score() {
        let mut record = parsed(MANIFEST)["fixtures"][3]["record"].clone();
        record["oracle"]["outcome"] = Value::String("Contradicted".into());
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"open_world_or_pluralism_cannot_have_correctness_outcome".to_string()
        ));
    }

    #[test]
    fn self_report_is_not_an_external_oracle() {
        let mut record = parsed(MANIFEST)["fixtures"][2]["record"].clone();
        record["agent_report"]["explanation_sha256"] =
            Value::String("f".repeat(64));
        record["oracle"]["oracle_receipt_sha256"] = Value::Null;
        let errors = validate_record(&record);
        assert!(errors.is_empty());
    }

    #[test]
    fn evidence_link_must_resolve_to_an_available_evidence_reference() {
        let mut record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        record["claim_evidence_links"][0]["evidence_id"] =
            Value::String("MISSING-EVIDENCE".into());
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"claim_evidence_link_not_bound_to_available_evidence".to_string()
        ));
    }

    #[test]
    fn changed_state_requires_distinct_state_digests() {
        let mut record = parsed(MANIFEST)["fixtures"][4]["record"].clone();
        record["next_state"]["state_sha256"] =
            record["previous_state"]["state_sha256"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"changed_state_digest_unchanged".to_string()
        ));
    }

    #[test]
    fn communication_and_delegation_graph_owners_are_distinct() {
        let mut manifest = parsed(MANIFEST);
        manifest["communication_graph_owner"] = Value::String(
            "https://github.com/Luminous-Dynamics/symthaea/issues/11248".into(),
        );
        let errors = validate_manifest(
            &manifest,
            &parsed(MANIFEST_SCHEMA),
            &parsed(RECORD_SCHEMA),
        )
        .unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains(
                "communication and delegation graph ownership"
            ))
        );
    }

    #[test]
    fn claim_added_and_removed_overlap_is_rejected() {
        let mut record = parsed(MANIFEST)["fixtures"][7]["record"].clone();
        record["claim_delta"]["retained"] =
            serde_json::json!(["CLAIM-X"]);
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"claim_delta_sets_overlap".to_string()
        ));
    }

    #[test]
    fn delivered_message_must_bind_a_channel_event() {
        let mut record = parsed(MANIFEST)["fixtures"][8]["record"].clone();
        record["delivered_messages"][0]["channel_event_id"] =
            Value::String("".into());
        let errors = validate_record(&record);
        assert!(errors.contains(
            &"delivered_message_missing_channel_event".to_string()
        ));
    }
}
