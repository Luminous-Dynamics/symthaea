use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, fs, process::ExitCode};

const LEDGER_ID: &str = "IG-REVISION-LEDGER-001";
const MANIFEST_SCHEMA_VERSION: &str = "ig-revision-ledger-fixtures-v0.1";
const EXPECTED_FIXTURES: usize = 22;

#[derive(Debug, PartialEq, Eq)]
struct Report {
    fixtures: usize,
    valid_controls: usize,
    negative_controls: usize,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)?
        .as_str()
        .filter(|value| !value.trim().is_empty())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Hash compact serde_json bytes after removing the state_sha256 field.
/// Default serde_json maps use sorted keys because preserve_order is disabled.
fn state_content_sha256(state: &Value) -> Option<String> {
    let mut content = state.clone();
    content.as_object_mut()?.remove("state_sha256")?;
    let bytes = serde_json::to_vec(&content).ok()?;
    let digest = Sha256::digest(bytes);
    Some(format!("{digest:x}"))
}

fn state_content_without_digest(state: &Value) -> Option<Value> {
    let mut content = state.clone();
    content.as_object_mut()?.remove("state_sha256")?;
    Some(content)
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

const SUPPORTED_SCHEMA_KEYWORDS: [&str; 18] = [
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "minLength",
    "enum",
    "pattern",
    "uniqueItems",
    "items",
    "minimum",
    "maximum",
    "const",
    "minItems",
    "format",
];

fn validate_schema_definition(schema: &Value, context: &str, errors: &mut Vec<String>) {
    let Some(object) = schema.as_object() else {
        add(errors, context, "schema node must be an object");
        return;
    };
    for keyword in object.keys() {
        if !SUPPORTED_SCHEMA_KEYWORDS.contains(&keyword.as_str()) {
            add(
                errors,
                context,
                format!("unsupported schema keyword '{keyword}'"),
            );
        }
    }
    if let Some(kind) = schema.get("type") {
        let supported = [
            "object", "array", "string", "number", "integer", "boolean", "null",
        ];
        let types: Vec<&str> = if let Some(single) = kind.as_str() {
            vec![single]
        } else {
            kind.as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect()
        };
        if types.is_empty() || types.iter().any(|item| !supported.contains(item)) {
            add(errors, context, "schema type contains an unsupported value");
        }
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        if pattern != "^[a-f0-9]{64}$" {
            add(
                errors,
                context,
                format!("unsupported schema pattern '{pattern}'"),
            );
        }
    }
    if let Some(format) = schema.get("format").and_then(Value::as_str) {
        if format != "uri" {
            add(
                errors,
                context,
                format!("unsupported schema format '{format}'"),
            );
        }
    }
    if let Some(properties) = schema.get("properties") {
        if let Some(properties) = properties.as_object() {
            for (name, child) in properties {
                validate_schema_definition(child, &format!("{context}.properties.{name}"), errors);
            }
        } else {
            add(errors, context, "schema properties must be an object");
        }
    }
    if let Some(items) = schema.get("items") {
        validate_schema_definition(items, &format!("{context}.items"), errors);
    }
}

fn json_type_matches(value: &Value, kind: &str) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn validate_json_schema(value: &Value, schema: &Value, context: &str, errors: &mut Vec<String>) {
    if let Some(kind) = schema.get("type") {
        let matches = if let Some(single) = kind.as_str() {
            json_type_matches(value, single)
        } else {
            kind.as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .any(|candidate| json_type_matches(value, candidate))
        };
        if !matches {
            add(
                errors,
                context,
                format!("schema type mismatch: expected {kind}"),
            );
            return;
        }
    }
    if let Some(expected) = schema.get("const") {
        if value != expected {
            add(errors, context, "schema const mismatch");
        }
    }
    if let Some(variants) = schema.get("enum").and_then(Value::as_array) {
        if !variants.contains(value) {
            add(errors, context, "schema enum mismatch");
        }
    }
    if let Some(text) = value.as_str() {
        if let Some(minimum) = schema.get("minLength").and_then(Value::as_u64) {
            if text.chars().count() < minimum as usize {
                add(errors, context, "schema minLength violation");
            }
        }
        if schema.get("pattern").is_some() && !is_sha256(text) {
            add(errors, context, "schema pattern violation");
        }
        if schema.get("format").and_then(Value::as_str) == Some("uri")
            && !(text.starts_with("https://") || text.starts_with("http://"))
        {
            add(errors, context, "schema URI format violation");
        }
    }
    if let Some(number) = value.as_f64() {
        if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64) {
            if number < minimum {
                add(errors, context, "schema minimum violation");
            }
        }
        if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64) {
            if number > maximum {
                add(errors, context, "schema maximum violation");
            }
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(minimum) = schema.get("minItems").and_then(Value::as_u64) {
            if items.len() < minimum as usize {
                add(errors, context, "schema minItems violation");
            }
        }
        if schema.get("uniqueItems").and_then(Value::as_bool) == Some(true) {
            let serialized: BTreeSet<String> = items.iter().map(Value::to_string).collect();
            if serialized.len() != items.len() {
                add(errors, context, "schema uniqueItems violation");
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in items.iter().enumerate() {
                validate_json_schema(item, item_schema, &format!("{context}[{index}]"), errors);
            }
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for property in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(property) {
                    add(
                        errors,
                        context,
                        format!("schema required property '{property}' is missing"),
                    );
                }
            }
        }
        let properties = schema.get("properties").and_then(Value::as_object);
        if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
            for property in object.keys() {
                if !properties.is_some_and(|known| known.contains_key(property)) {
                    add(
                        errors,
                        context,
                        format!("schema additional property '{property}' is not allowed"),
                    );
                }
            }
        }
        if let Some(properties) = properties {
            for (name, child_schema) in properties {
                if let Some(child_value) = object.get(name) {
                    validate_json_schema(
                        child_value,
                        child_schema,
                        &format!("{context}.{name}"),
                        errors,
                    );
                }
            }
        }
    }
}

fn validate_state(state: &Value, context: &str, errors: &mut Vec<String>) {
    if state.is_null() {
        return;
    }
    let declared_digest = field(state, "state_sha256").unwrap_or_default();
    if !is_sha256(declared_digest) {
        add(
            errors,
            context,
            "state_sha256 must be a 64-character hexadecimal digest",
        );
    } else if state_content_sha256(state).as_deref() != Some(declared_digest) {
        add(errors, context, "state_content_digest_mismatch");
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
            add(
                errors,
                context,
                format!("state {key} must not contain duplicates"),
            );
        }
    }

    if !state["confidence"].is_null() {
        let confidence = &state["confidence"];
        let value = confidence["value"].as_f64();
        let scale = field(confidence, "scale");
        let profile = field(confidence, "calibration_profile_sha256");
        let receipt = field(confidence, "calibration_receipt_sha256");
        match field(confidence, "status") {
            Some("NotProvided") => {
                if value.is_some() || scale.is_some() || profile.is_some() || receipt.is_some() {
                    add(errors, context, "confidence_not_provided_must_be_empty");
                }
            }
            Some("SelfReported") => {
                if value.is_none_or(|value| !(0.0..=1.0).contains(&value)) {
                    add(errors, context, "confidence_value_out_of_range");
                }
                if scale.is_none() {
                    add(errors, context, "confidence_scale_missing");
                }
                if profile.is_some() || receipt.is_some() {
                    add(
                        errors,
                        context,
                        "self_report_confidence_cannot_claim_calibration",
                    );
                }
            }
            Some("Calibrated") => {
                if value.is_none_or(|value| !(0.0..=1.0).contains(&value)) {
                    add(errors, context, "confidence_value_out_of_range");
                }
                if scale.is_none()
                    || profile.is_none_or(|value| !is_sha256(value))
                    || receipt.is_none_or(|value| !is_sha256(value))
                {
                    add(errors, context, "confidence_calibration_evidence_missing");
                }
            }
            Some(_) | None => add(errors, context, "confidence_status_invalid"),
        }
    }
}

fn validate_record(record: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let id = field(record, "revision_event_id").unwrap_or("<missing>");
    let context = format!("record[{id}]");

    for key in [
        "revision_event_id",
        "run_id",
        "scenario_id",
        "agent_instance_id",
    ] {
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
                add(
                    &mut errors,
                    &context,
                    "Changed requires previous and next states",
                );
            } else {
                if field(previous, "state_sha256") == field(next, "state_sha256") {
                    add(&mut errors, &context, "changed_state_digest_unchanged");
                }
                if state_content_without_digest(previous) == state_content_without_digest(next) {
                    add(
                        &mut errors,
                        &context,
                        "changed_transition_no_semantic_delta",
                    );
                }
            }
        }
        Some("NoChange") => {
            if previous.is_null() || next.is_null() {
                add(
                    &mut errors,
                    &context,
                    "NoChange requires previous and next states",
                );
            } else {
                if field(previous, "state_sha256") != field(next, "state_sha256") {
                    add(&mut errors, &context, "no_change_state_digest_mismatch");
                }
                if state_content_without_digest(previous) != state_content_without_digest(next) {
                    add(&mut errors, &context, "no_change_state_content_mismatch");
                }
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
                add(
                    &mut errors,
                    &context,
                    "missing_measurement_post_state_present",
                );
            }
            if !matches!(
                field(record, "event_status"),
                Some("NotRun" | "Indeterminate")
            ) {
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
        Some(_) | None => add(
            &mut errors,
            &context,
            "transition is missing or unsupported",
        ),
    }

    let delta = &record["claim_delta"];
    let mut primary_claims = BTreeSet::new();
    for key in ["added", "removed", "retained"] {
        if !delta[key].is_array() {
            add(
                &mut errors,
                &context,
                format!("claim_delta.{key} must be an array"),
            );
        }
        for claim in string_array(&delta[key]) {
            if !primary_claims.insert(claim) {
                add(&mut errors, &context, "claim_delta_sets_overlap");
            }
        }
    }
    for key in ["contradicted", "retracted"] {
        if !delta[key].is_array() {
            add(
                &mut errors,
                &context,
                format!("claim_delta.{key} must be an array"),
            );
        }
    }

    if !previous.is_null() && !next.is_null() {
        let before = string_set(&previous["claim_ids"]);
        let after = string_set(&next["claim_ids"]);
        let added = string_set(&delta["added"]);
        let removed = string_set(&delta["removed"]);
        let retained = string_set(&delta["retained"]);
        let contradicted = string_set(&delta["contradicted"]);
        let retracted = string_set(&delta["retracted"]);
        let actual_added: BTreeSet<String> = after.difference(&before).cloned().collect();
        let actual_removed: BTreeSet<String> = before.difference(&after).cloned().collect();
        let actual_retained: BTreeSet<String> = before.intersection(&after).cloned().collect();
        let reported_removed: BTreeSet<String> = removed.union(&retracted).cloned().collect();

        if added != actual_added
            || reported_removed != actual_removed
            || retained != actual_retained
        {
            add(&mut errors, &context, "claim_delta_state_mismatch");
        }
        if !contradicted.is_subset(&before)
            || !retracted.is_subset(&before)
            || !retracted.is_subset(&reported_removed)
        {
            add(
                &mut errors,
                &context,
                "claim_contradiction_or_retraction_not_bound_to_previous_state",
            );
        }
    }

    let evidence_delta = &record["evidence_delta"];
    for key in [
        "available_before",
        "newly_available",
        "newly_unavailable",
        "invalidated",
    ] {
        if !evidence_delta[key].is_array() {
            add(
                &mut errors,
                &context,
                format!("evidence_delta.{key} must be an array"),
            );
        }
    }

    let available_before = string_set(&evidence_delta["available_before"]);
    let newly_available = string_set(&evidence_delta["newly_available"]);
    let newly_unavailable = string_set(&evidence_delta["newly_unavailable"]);
    let invalidated = string_set(&evidence_delta["invalidated"]);
    if !available_before.is_disjoint(&newly_available) {
        add(
            &mut errors,
            &context,
            "evidence_available_before_overlaps_newly_available",
        );
    }
    if !newly_available.is_disjoint(&newly_unavailable) {
        add(
            &mut errors,
            &context,
            "evidence_newly_available_overlaps_newly_unavailable",
        );
    }
    let available_during: BTreeSet<String> =
        available_before.union(&newly_available).cloned().collect();
    if !invalidated.is_subset(&available_during) {
        add(
            &mut errors,
            &context,
            "invalidated_evidence_not_previously_available",
        );
    }
    if !previous.is_null() && !string_set(&previous["evidence_ids"]).is_subset(&available_before) {
        add(
            &mut errors,
            &context,
            "previous_state_evidence_not_in_available_before",
        );
    }
    if !next.is_null() {
        let mut usable_after = available_during.clone();
        for evidence in newly_unavailable.union(&invalidated) {
            usable_after.remove(evidence);
        }
        if !string_set(&next["evidence_ids"]).is_subset(&usable_after) {
            add(
                &mut errors,
                &context,
                "next_state_evidence_not_available_after_transition",
            );
        }
    }

    let mut seen_message_ids = BTreeSet::new();
    let mut seen_channel_event_ids = BTreeSet::new();
    let mut previous_delivery_sequence = 0_u64;
    for message in record["delivered_messages"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let message_id = field(message, "message_id");
        let channel_event_id = field(message, "channel_event_id");
        let sender = field(message, "sender_agent_instance_id");
        let recipient = field(message, "recipient_agent_instance_id");
        let sequence = message["delivery_sequence"].as_u64();
        if message_id.is_none()
            || channel_event_id.is_none()
            || !is_sha256(field(message, "payload_sha256").unwrap_or_default())
        {
            add(
                &mut errors,
                &context,
                "delivered_message_missing_channel_event",
            );
        }
        if !is_sha256(field(message, "channel_event_sha256").unwrap_or_default())
            || !is_sha256(field(message, "communication_graph_sha256").unwrap_or_default())
        {
            add(
                &mut errors,
                &context,
                "delivered_message_missing_graph_or_event_digest",
            );
        }
        if sender.is_none() || recipient.is_none() || sender == recipient {
            add(&mut errors, &context, "delivered_message_identity_invalid");
        }
        if recipient != field(record, "agent_instance_id") {
            add(
                &mut errors,
                &context,
                "delivered_message_recipient_mismatch",
            );
        }
        if message_id.is_some_and(|id| !seen_message_ids.insert(id.to_string()))
            || channel_event_id.is_some_and(|id| !seen_channel_event_ids.insert(id.to_string()))
        {
            add(
                &mut errors,
                &context,
                "delivered_message_identity_duplicate",
            );
        }
        match sequence {
            Some(value) if value > previous_delivery_sequence => previous_delivery_sequence = value,
            _ => add(&mut errors, &context, "delivered_message_order_invalid"),
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
        None => add(
            &mut errors,
            &context,
            "agent_report.provided must be boolean",
        ),
    }
    if field(report, "kind") != Some("self_report") {
        add(
            &mut errors,
            &context,
            "agent report must remain typed as self_report",
        );
    }

    let oracle = &record["oracle"];
    let task_profile = field(oracle, "task_profile");
    let outcome = field(oracle, "outcome");
    match task_profile {
        Some("SyntheticKnownAnswer") => {
            if outcome == Some("NotApplicable") {
                add(
                    &mut errors,
                    &context,
                    "synthetic_known_answer_outcome_not_applicable",
                );
            }
            if !is_sha256(field(oracle, "answer_space_sha256").unwrap_or_default()) {
                add(
                    &mut errors,
                    &context,
                    "synthetic_oracle_missing_answer_space",
                );
            }
            if !is_sha256(field(oracle, "oracle_receipt_sha256").unwrap_or_default()) {
                add(&mut errors, &context, "synthetic_oracle_missing_receipt");
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
                add(&mut errors, &context, "causal_pair_runs_must_be_distinct");
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
                    add(&mut errors, &context, "causal_pair_shared_input_mismatch");
                    break;
                }
            }
            let varied_factor = &causal["varied_factor"];
            let treatment_factor = field(varied_factor, "treatment_sha256");
            let control_factor = field(varied_factor, "control_sha256");
            let allowed_factors = [
                "persuasion_exposure",
                "communication_placement",
                "confidence_policy",
                "debate_method",
                "evidence_availability",
                "message_order",
                "agent_diversity",
                "abstention_policy",
                "loss_profile",
                "retrieval_policy",
            ];
            if field(varied_factor, "id").is_none_or(|id| !allowed_factors.contains(&id))
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
                || !is_sha256(field(causal, "pair_validation_receipt_sha256").unwrap_or_default())
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
                    add(&mut errors, &context, "causal_pair_fields_without_estimate");
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
    let mut usable_evidence = known_evidence.clone();
    for unavailable in string_set(&evidence_delta["newly_unavailable"])
        .into_iter()
        .chain(string_set(&evidence_delta["invalidated"]))
    {
        usable_evidence.remove(&unavailable);
    }
    let mut known_claims = BTreeSet::new();
    if !previous.is_null() {
        known_claims.extend(string_set(&previous["claim_ids"]));
    }
    if !next.is_null() {
        known_claims.extend(string_set(&next["claim_ids"]));
    }
    let current_claims = if next.is_null() {
        BTreeSet::new()
    } else {
        string_set(&next["claim_ids"])
    };
    for link in record["claim_evidence_links"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let claim_id = field(link, "claim_id");
        let evidence_id = field(link, "evidence_id");
        if claim_id.is_none_or(|claim| !known_claims.contains(claim)) {
            add(
                &mut errors,
                &context,
                "claim_evidence_link_claim_not_bound_to_revision",
            );
        }
        if evidence_id.is_none_or(|evidence| !known_evidence.contains(evidence)) {
            add(
                &mut errors,
                &context,
                "claim_evidence_link_not_bound_to_available_evidence",
            );
        }
        match field(link, "relationship") {
            Some("supports") => {
                if claim_id.is_none_or(|claim| !current_claims.contains(claim))
                    || evidence_id.is_none_or(|evidence| !usable_evidence.contains(evidence))
                {
                    add(
                        &mut errors,
                        &context,
                        "claim_support_link_not_bound_to_current_usable_state",
                    );
                }
            }
            Some("contradicts" | "qualifies" | "contextualizes") => {}
            _ => add(
                &mut errors,
                &context,
                "claim_evidence_link_relationship_invalid",
            ),
        }
    }

    let mut violation_codes: Vec<String> = errors
        .into_iter()
        .map(|error| {
            error
                .split_once(": ")
                .map(|(_, code)| code.to_string())
                .unwrap_or(error)
        })
        .collect();
    violation_codes.sort();
    violation_codes.dedup();
    violation_codes
}

fn validate_manifest(
    manifest: &Value,
    manifest_schema: &Value,
    record_schema: &Value,
) -> Result<Report, Vec<String>> {
    let mut errors = Vec::new();

    validate_schema_definition(manifest_schema, "manifest_schema", &mut errors);
    validate_schema_definition(record_schema, "record_schema", &mut errors);
    validate_json_schema(manifest, manifest_schema, "manifest", &mut errors);

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
        add(
            &mut errors,
            "manifest",
            "unexpected ledger or schema identity",
        );
    }
    if field(manifest, "status") != Some("synthetic_fixture_design_unexecuted") {
        add(
            &mut errors,
            "manifest",
            "fixture design must remain marked unexecuted",
        );
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
        add(
            &mut errors,
            "manifest.fixtures",
            "fixtures must be an array",
        );
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
    let mut revision_event_ids = BTreeSet::new();
    let mut valid_controls = 0;
    let mut negative_controls = 0;

    for (index, fixture) in fixtures.iter().enumerate() {
        let context = format!("fixtures[{index}]");
        let id = field(fixture, "fixture_id").unwrap_or_default();
        if id.is_empty() || !ids.insert(id.to_string()) {
            add(
                &mut errors,
                &context,
                "fixture_id must be non-empty and unique",
            );
        }

        validate_json_schema(
            &fixture["record"],
            record_schema,
            &format!("{context}.record"),
            &mut errors,
        );
        let revision_id = field(&fixture["record"], "revision_event_id").unwrap_or_default();
        if revision_id.is_empty() || !revision_event_ids.insert(revision_id.to_string()) {
            add(
                &mut errors,
                &context,
                "revision_event_id must be non-empty and unique across the corpus",
            );
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

    const MANIFEST: &str = include_str!("../../REVISION_LEDGER_FIXTURES_V0_1.json");
    const MANIFEST_SCHEMA: &str = include_str!("../../REVISION_LEDGER_FIXTURES_V0_1.schema.json");
    const RECORD_SCHEMA: &str = include_str!("../../REVISION_LEDGER_RECORD_V0_1.schema.json");

    fn parsed(text: &str) -> Value {
        serde_json::from_str(text).expect("checked-in JSON must parse")
    }

    #[test]
    fn state_content_digest_is_recomputed_from_canonical_state() {
        let record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        let state = &record["next_state"];
        assert_eq!(
            state_content_sha256(state).as_deref(),
            field(state, "state_sha256")
        );
    }

    #[test]
    fn state_content_tampering_is_rejected_even_with_valid_digest_shape() {
        let mut record = parsed(MANIFEST)["fixtures"][21]["record"].clone();
        record["next_state"]["stance"] = Value::String("tampered-after-commitment".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"state_content_digest_mismatch".to_string()));
    }

    #[test]
    fn changed_transition_requires_semantic_state_change() {
        let mut record = parsed(MANIFEST)["fixtures"][0]["record"].clone();
        record["transition"] = Value::String("Changed".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"changed_transition_no_semantic_delta".to_string()));
    }

    #[test]
    fn delivered_message_must_bind_recipient_graph_and_channel_event() {
        let mut record = parsed(MANIFEST)["fixtures"][2]["record"].clone();
        record["delivered_messages"][0]["recipient_agent_instance_id"] =
            Value::String("AGENT-OTHER".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"delivered_message_recipient_mismatch".to_string()));
    }

    #[test]
    fn duplicate_delivered_message_ids_are_rejected() {
        let mut record = parsed(MANIFEST)["fixtures"][2]["record"].clone();
        let first = record["delivered_messages"][0].clone();
        record["delivered_messages"]
            .as_array_mut()
            .unwrap()
            .push(first);
        let errors = validate_record(&record);
        assert!(errors.contains(&"delivered_message_identity_duplicate".to_string()));
    }

    #[test]
    fn support_links_must_bind_current_claims_and_usable_evidence() {
        let mut record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        record["claim_evidence_links"][0]["claim_id"] = Value::String("UNRELATED-CLAIM".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"claim_evidence_link_claim_not_bound_to_revision".to_string()));
    }

    #[test]
    fn synthetic_known_answer_cannot_be_marked_not_applicable() {
        let mut record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        record["oracle"]["outcome"] = Value::String("NotApplicable".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"synthetic_known_answer_outcome_not_applicable".to_string()));
    }

    #[test]
    fn matched_pair_varied_factor_must_be_registered() {
        let mut record = parsed(MANIFEST)["fixtures"][9]["record"].clone();
        record["causal_attribution"]["varied_factor"]["id"] =
            Value::String("invented-causal-factor".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"causal_estimate_missing_matched_pair".to_string()));
    }

    #[test]
    fn schemas_are_recursively_checked_and_applied_to_every_record() {
        let mut manifest_schema = parsed(MANIFEST_SCHEMA);
        manifest_schema["properties"]["optional_probe"] =
            serde_json::json!({"type":"string","maxLength":12});
        let errors = validate_manifest(&parsed(MANIFEST), &manifest_schema, &parsed(RECORD_SCHEMA))
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unsupported schema keyword 'maxLength'"))
        );
    }

    #[test]
    fn records_reject_undeclared_properties() {
        let mut manifest = parsed(MANIFEST);
        manifest["fixtures"][0]["record"]["unregistered"] = Value::Bool(true);
        let errors = validate_manifest(&manifest, &parsed(MANIFEST_SCHEMA), &parsed(RECORD_SCHEMA))
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("schema additional property 'unregistered'"))
        );
    }

    #[test]
    fn duplicate_revision_event_ids_are_rejected() {
        let mut manifest = parsed(MANIFEST);
        manifest["fixtures"][1]["record"]["revision_event_id"] =
            manifest["fixtures"][0]["record"]["revision_event_id"].clone();
        let errors = validate_manifest(&manifest, &parsed(MANIFEST_SCHEMA), &parsed(RECORD_SCHEMA))
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("revision_event_id must be non-empty and unique"))
        );
    }

    #[test]
    fn claim_deltas_must_match_state_claim_sets() {
        let mut record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        record["claim_delta"]["added"] = serde_json::json!(["CLAIM-FABRICATED"]);
        let errors = validate_record(&record);
        assert!(errors.contains(&"claim_delta_state_mismatch".to_string()));
    }

    #[test]
    fn evidence_deltas_must_not_invent_invalidated_sources() {
        let mut record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        record["evidence_delta"]["invalidated"] = serde_json::json!(["NEVER-AVAILABLE"]);
        let errors = validate_record(&record);
        assert!(errors.contains(&"invalidated_evidence_not_previously_available".to_string()));
    }

    #[test]
    fn checked_in_twenty_two_fixture_oracles_validate() {
        let report = validate_manifest(
            &parsed(MANIFEST),
            &parsed(MANIFEST_SCHEMA),
            &parsed(RECORD_SCHEMA),
        )
        .expect("manifest and event fixtures must validate");
        assert_eq!(report.fixtures, 22);
        assert_eq!(report.valid_controls, 10);
        assert_eq!(report.negative_controls, 12);
    }

    #[test]
    fn agent_self_report_cannot_supply_a_matched_causal_estimate() {
        let mut record = parsed(MANIFEST)["fixtures"][2]["record"].clone();
        record["agent_report"]["reason_code"] = Value::String("Evidence".into());
        record["causal_attribution"]["status"] = Value::String("MatchedPairEstimate".into());
        record["causal_attribution"]["matched_pair_id"] = Value::Null;
        let errors = validate_record(&record);
        assert!(errors.contains(&"causal_estimate_missing_matched_pair".to_string()));
    }

    #[test]
    fn matched_pair_estimate_requires_distinct_runs_and_analysis_identity() {
        let mut record = parsed(MANIFEST)["fixtures"][9]["record"].clone();
        record["causal_attribution"]["control_run_id"] =
            record["causal_attribution"]["treatment_run_id"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(&"causal_pair_runs_must_be_distinct".to_string()));
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
    fn self_reported_confidence_is_not_calibrated_evidence() {
        let record = parsed(MANIFEST)["fixtures"][17]["record"].clone();
        let errors = validate_record(&record);
        assert!(errors.is_empty());
        assert_eq!(
            record["next_state"]["confidence"]["status"].as_str(),
            Some("SelfReported")
        );
        assert!(
            record["evidence_delta"]["newly_available"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn calibrated_confidence_requires_external_profile_and_receipt() {
        let record = parsed(MANIFEST)["fixtures"][19]["record"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(&"confidence_calibration_evidence_missing".to_string()));
    }

    #[test]
    fn confidence_probability_must_remain_in_unit_interval() {
        let record = parsed(MANIFEST)["fixtures"][20]["record"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(&"confidence_value_out_of_range".to_string()));
    }

    #[test]
    fn non_estimated_event_cannot_carry_a_pair_validator_identity() {
        let mut record = parsed(MANIFEST)["fixtures"][0]["record"].clone();
        record["causal_attribution"]["pair_validator_id"] =
            Value::String("independent-pair-validator/v1".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"causal_pair_fields_without_estimate".to_string()));
    }

    #[test]
    fn a_missing_measurement_is_not_no_change() {
        let mut record = parsed(MANIFEST)["fixtures"][11]["record"].clone();
        record["next_state"] = record["previous_state"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(&"missing_measurement_post_state_present".to_string()));
    }

    #[test]
    fn normative_pluralism_cannot_be_scored_correct_or_incorrect() {
        let mut record = parsed(MANIFEST)["fixtures"][10]["record"].clone();
        record["oracle"]["outcome"] = Value::String("Supported".into());
        let errors = validate_record(&record);
        assert!(
            errors.contains(&"open_world_or_pluralism_cannot_have_correctness_outcome".to_string())
        );
    }

    #[test]
    fn open_world_tasks_cannot_be_given_a_single_answer_score() {
        let mut record = parsed(MANIFEST)["fixtures"][3]["record"].clone();
        record["oracle"]["outcome"] = Value::String("Contradicted".into());
        let errors = validate_record(&record);
        assert!(
            errors.contains(&"open_world_or_pluralism_cannot_have_correctness_outcome".to_string())
        );
    }

    #[test]
    fn self_report_is_not_an_external_oracle() {
        let mut record = parsed(MANIFEST)["fixtures"][2]["record"].clone();
        record["agent_report"]["explanation_sha256"] = Value::String("f".repeat(64));
        record["oracle"]["oracle_receipt_sha256"] = Value::Null;
        let errors = validate_record(&record);
        assert!(errors.is_empty());
    }

    #[test]
    fn evidence_link_must_resolve_to_an_available_evidence_reference() {
        let mut record = parsed(MANIFEST)["fixtures"][1]["record"].clone();
        record["claim_evidence_links"][0]["evidence_id"] = Value::String("MISSING-EVIDENCE".into());
        let errors = validate_record(&record);
        assert!(
            errors.contains(&"claim_evidence_link_not_bound_to_available_evidence".to_string())
        );
    }

    #[test]
    fn changed_state_requires_distinct_state_digests() {
        let mut record = parsed(MANIFEST)["fixtures"][4]["record"].clone();
        record["next_state"]["state_sha256"] = record["previous_state"]["state_sha256"].clone();
        let errors = validate_record(&record);
        assert!(errors.contains(&"changed_state_digest_unchanged".to_string()));
    }

    #[test]
    fn communication_and_delegation_graph_owners_are_distinct() {
        let mut manifest = parsed(MANIFEST);
        manifest["communication_graph_owner"] =
            Value::String("https://github.com/Luminous-Dynamics/symthaea/issues/11248".into());
        let errors = validate_manifest(&manifest, &parsed(MANIFEST_SCHEMA), &parsed(RECORD_SCHEMA))
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("communication and delegation graph ownership"))
        );
    }

    #[test]
    fn claim_added_and_removed_overlap_is_rejected() {
        let mut record = parsed(MANIFEST)["fixtures"][7]["record"].clone();
        record["claim_delta"]["retained"] = serde_json::json!(["CLAIM-X"]);
        let errors = validate_record(&record);
        assert!(errors.contains(&"claim_delta_sets_overlap".to_string()));
    }

    #[test]
    fn delivered_message_must_bind_a_channel_event() {
        let mut record = parsed(MANIFEST)["fixtures"][8]["record"].clone();
        record["delivered_messages"][0]["channel_event_id"] = Value::String("".into());
        let errors = validate_record(&record);
        assert!(errors.contains(&"delivered_message_missing_channel_event".to_string()));
    }
}
