//! Test-only reference evaluator for the synthetic First-Aid process contract.
//!
//! This deliberately parses serde_json::Value and maintains its own reference state.
//! It does not import the production trace checker's event/state types or transition
//! function. It is implementation-diversity evidence, not independent clinical truth.

use serde_json::Value;
use std::collections::BTreeSet;

const VIOLATION_CODES: &[&str] = &[
    "invalid_bundle_verification",
    "render_without_active_session",
    "render_after_revocation",
    "render_without_verified_bundle",
    "unknown_normalized",
    "uncertainty_lost_at_handoff",
    "uncertainty_lost_at_session_boundary",
    "stale_resource_claimed_confirmed",
    "offline_core_network_dependency",
    "intermittent_core_network_dependency",
    "sensitive_data_egress_without_consent",
];

const TRACE_FIELDS: &[&str] = &[
    "trace_id",
    "description",
    "criticality",
    "connectivity",
    "events",
    "expected_violation_codes",
];

#[derive(Debug)]
struct ReferenceState {
    session_active: bool,
    loaded_bundle: Option<String>,
    verified_bundle: Option<String>,
    revoked_bundles: BTreeSet<String>,
    uncertainty_pending: bool,
    uncertainty_preserved: bool,
    connectivity: String,
}

impl ReferenceState {
    fn new(connectivity: &str) -> Self {
        Self {
            session_active: false,
            loaded_bundle: None,
            verified_bundle: None,
            revoked_bundles: BTreeSet::new(),
            uncertainty_pending: false,
            uncertainty_preserved: false,
            connectivity: connectivity.to_owned(),
        }
    }
}

fn string_at<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{context}: {key} must be a string"))
}

fn boolean_at(value: &Value, key: &str, context: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{context}: {key} must be a boolean"))
}

fn permitted_fields(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "session_started"
        | "session_interrupted"
        | "session_resumed"
        | "uncertainty_detected"
        | "uncertainty_preserved"
        | "unknown_normalized"
        | "handoff_emitted"
        | "network_required_for_core" => Some(&["sequence", "kind"]),
        "bundle_loaded" => Some(&["sequence", "kind", "bundle_id", "bundle_state"]),
        "bundle_verified" => Some(&[
            "sequence",
            "kind",
            "bundle_id",
            "bundle_state",
            "digest_verified",
        ]),
        "protocol_rendered" => {
            Some(&["sequence", "kind", "bundle_id", "bundle_state"])
        }
        "bundle_revoked" => Some(&["sequence", "kind", "bundle_id"]),
        "resource_presented" => {
            Some(&["sequence", "kind", "resource_state", "presented_as"])
        }
        "connectivity_changed" => Some(&["sequence", "kind", "connectivity"]),
        "sensitive_data_egress" => Some(&[
            "sequence",
            "kind",
            "sensitive_payload",
            "consent_scoped",
        ]),
        _ => None,
    }
}

fn require_nonempty_string(event: &Value, key: &str, context: &str) -> Result<(), String> {
    let value = string_at(event, key, context)?;
    if value.trim().is_empty() {
        return Err(format!("{context}: {key} must not be empty"));
    }
    Ok(())
}

fn one_of(value: &str, allowed: &[&str]) -> bool {
    allowed.contains(&value)
}

fn validate_trace(trace: &Value) -> Result<(), String> {
    let root = trace
        .as_object()
        .ok_or_else(|| "trace: expected object".to_owned())?;

    for key in root.keys() {
        if !TRACE_FIELDS.contains(&key.as_str()) {
            return Err(format!("trace: unexpected field {key}"));
        }
    }
    for field in TRACE_FIELDS {
        if !root.contains_key(*field) {
            return Err(format!("trace: missing field {field}"));
        }
    }

    require_nonempty_string(trace, "trace_id", "trace")?;
    require_nonempty_string(trace, "description", "trace")?;

    let criticality = string_at(trace, "criticality", "trace")?;
    if !one_of(criticality, &["critical", "high", "routine"]) {
        return Err("trace: unsupported criticality".to_owned());
    }

    let initial_connectivity = string_at(trace, "connectivity", "trace")?;
    if !one_of(initial_connectivity, &["online", "offline", "intermittent"]) {
        return Err("trace: unsupported initial connectivity".to_owned());
    }

    let expected = trace["expected_violation_codes"]
        .as_array()
        .ok_or_else(|| "trace: expected_violation_codes must be an array".to_owned())?;
    for code in expected {
        let code = code
            .as_str()
            .ok_or_else(|| "trace: expected violation code must be a string".to_owned())?;
        if !VIOLATION_CODES.contains(&code) {
            return Err(format!("trace: undeclared violation code {code}"));
        }
    }

    let events = trace["events"]
        .as_array()
        .ok_or_else(|| "trace: events must be an array".to_owned())?;
    if events.is_empty() {
        return Err("trace: events must not be empty".to_owned());
    }

    let mut last_sequence: Option<u64> = None;
    for (index, event) in events.iter().enumerate() {
        let label = format!("event[{index}]");
        let object = event
            .as_object()
            .ok_or_else(|| format!("{label}: expected object"))?;
        let kind = string_at(event, "kind", &label)?;
        let fields = permitted_fields(kind)
            .ok_or_else(|| format!("{label}: unsupported kind {kind}"))?;

        for field in object.keys() {
            if !fields.contains(&field.as_str()) {
                return Err(format!("{label}: {kind} does not permit field {field}"));
            }
        }

        for field in fields {
            if !object.contains_key(*field) {
                // bundle_state is optional only on a render event.
                if !(kind == "protocol_rendered" && *field == "bundle_state") {
                    return Err(format!("{label}: {kind} requires field {field}"));
                }
            }
        }

        let sequence = event["sequence"]
            .as_u64()
            .ok_or_else(|| format!("{label}: sequence must be an unsigned integer"))?;
        if last_sequence.is_some_and(|previous| sequence <= previous) {
            return Err(format!("{label}: sequence must increase strictly"));
        }
        last_sequence = Some(sequence);

        match kind {
            "bundle_loaded" | "bundle_verified" => {
                require_nonempty_string(event, "bundle_id", &label)?;
                let state = string_at(event, "bundle_state", &label)?;
                if !one_of(state, &["valid", "expired", "revoked", "digest_mismatch", "unknown"]) {
                    return Err(format!("{label}: unsupported bundle_state"));
                }
                if kind == "bundle_verified" {
                    let _ = boolean_at(event, "digest_verified", &label)?;
                }
            }
            "protocol_rendered" => {
                require_nonempty_string(event, "bundle_id", &label)?;
                if let Some(state) = event.get("bundle_state") {
                    let state = state
                        .as_str()
                        .ok_or_else(|| format!("{label}: bundle_state must be a string"))?;
                    if !one_of(state, &["valid", "expired", "revoked", "digest_mismatch", "unknown"]) {
                        return Err(format!("{label}: unsupported bundle_state"));
                    }
                }
            }
            "bundle_revoked" => require_nonempty_string(event, "bundle_id", &label)?,
            "resource_presented" => {
                let state = string_at(event, "resource_state", &label)?;
                let presentation = string_at(event, "presented_as", &label)?;
                if !one_of(state, &["fresh", "stale", "unknown"]) {
                    return Err(format!("{label}: unsupported resource_state"));
                }
                if !one_of(presentation, &["confirmed", "unconfirmed"]) {
                    return Err(format!("{label}: unsupported presented_as"));
                }
            }
            "connectivity_changed" => {
                let connectivity = string_at(event, "connectivity", &label)?;
                if !one_of(connectivity, &["online", "offline", "intermittent"]) {
                    return Err(format!("{label}: unsupported connectivity"));
                }
            }
            "sensitive_data_egress" => {
                let _ = boolean_at(event, "sensitive_payload", &label)?;
                let _ = boolean_at(event, "consent_scoped", &label)?;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Evaluate one trace according to the reference contract.
///
/// A schema/input failure returns Err and is not equivalent to a successful trace
/// with an empty violation list. Violations remain event-ordered and are not deduplicated.
fn evaluate_trace(trace: &Value) -> Result<Vec<String>, String> {
    validate_trace(trace)?;
    let initial_connectivity = string_at(trace, "connectivity", "trace")?;
    let mut state = ReferenceState::new(initial_connectivity);
    let events = trace["events"]
        .as_array()
        .expect("validate_trace checked the event array");
    let mut violations = Vec::new();

    for (index, event) in events.iter().enumerate() {
        let context = format!("event[{index}]");
        let kind = string_at(event, "kind", &context)?;
        match kind {
            "session_started" => {
                if state.uncertainty_pending && !state.uncertainty_preserved {
                    violations.push("uncertainty_lost_at_session_boundary".to_owned());
                }
                state.session_active = true;
                state.loaded_bundle = None;
                state.verified_bundle = None;
            }
            "bundle_loaded" => {
                state.loaded_bundle = Some(
                    string_at(event, "bundle_id", &context)?.to_owned(),
                );
                state.verified_bundle = None;
            }
            "bundle_verified" => {
                let id = string_at(event, "bundle_id", &context)?;
                let valid = state.session_active
                    && state.loaded_bundle.as_deref() == Some(id)
                    && string_at(event, "bundle_state", &context)? == "valid"
                    && boolean_at(event, "digest_verified", &context)?
                    && !state.revoked_bundles.contains(id);
                if valid {
                    state.verified_bundle = Some(id.to_owned());
                } else {
                    state.verified_bundle = None;
                    violations.push("invalid_bundle_verification".to_owned());
                }
            }
            "protocol_rendered" => {
                let id = string_at(event, "bundle_id", &context)?;
                if !state.session_active {
                    violations.push("render_without_active_session".to_owned());
                }
                let revoked = state.revoked_bundles.contains(id)
                    || event.get("bundle_state").and_then(Value::as_str) == Some("revoked");
                if revoked {
                    violations.push("render_after_revocation".to_owned());
                } else if state.verified_bundle.as_deref() != Some(id) {
                    violations.push("render_without_verified_bundle".to_owned());
                }
            }
            "bundle_revoked" => {
                let id = string_at(event, "bundle_id", &context)?.to_owned();
                state.revoked_bundles.insert(id.clone());
                if state.loaded_bundle.as_deref() == Some(id.as_str()) {
                    state.verified_bundle = None;
                }
            }
            "session_interrupted" => {
                state.session_active = false;
                state.verified_bundle = None;
            }
            "session_resumed" => {
                state.session_active = true;
                state.verified_bundle = None;
            }
            "uncertainty_detected" => {
                state.uncertainty_pending = true;
                state.uncertainty_preserved = false;
            }
            "uncertainty_preserved" => {
                if state.uncertainty_pending {
                    state.uncertainty_preserved = true;
                }
            }
            "unknown_normalized" => {
                violations.push("unknown_normalized".to_owned());
            }
            "handoff_emitted" => {
                if state.uncertainty_pending && !state.uncertainty_preserved {
                    violations.push("uncertainty_lost_at_handoff".to_owned());
                }
                state.uncertainty_pending = false;
                state.uncertainty_preserved = false;
            }
            "resource_presented" => {
                let freshness = string_at(event, "resource_state", &context)?;
                let presentation = string_at(event, "presented_as", &context)?;
                if one_of(freshness, &["stale", "unknown"]) && presentation == "confirmed" {
                    violations.push("stale_resource_claimed_confirmed".to_owned());
                }
            }
            "connectivity_changed" => {
                state.connectivity = string_at(event, "connectivity", &context)?.to_owned();
            }
            "network_required_for_core" => match state.connectivity.as_str() {
                "offline" => violations.push("offline_core_network_dependency".to_owned()),
                "intermittent" => {
                    violations.push("intermittent_core_network_dependency".to_owned())
                }
                "online" => {}
                _ => unreachable!("connectivity was validated"),
            },
            "sensitive_data_egress" => {
                if boolean_at(event, "sensitive_payload", &context)?
                    && !boolean_at(event, "consent_scoped", &context)?
                {
                    violations.push("sensitive_data_egress_without_consent".to_owned());
                }
            }
            _ => unreachable!("event kinds were validated"),
        }
    }

    Ok(violations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn corpus() -> Value {
        serde_json::from_str(include_str!(
            "../../docs/research/first-aid/trajectory-corpus-v0.json"
        ))
        .expect("frozen synthetic corpus parses")
    }

    fn base_trace(connectivity: &str, events: Vec<Value>) -> Value {
        json!({
            "trace_id": "REF-CASE",
            "description": "Synthetic reference-oracle unit case",
            "criticality": "critical",
            "connectivity": connectivity,
            "events": events,
            "expected_violation_codes": []
        })
    }

    fn event(sequence: u64, kind: &str) -> Value {
        json!({"sequence": sequence, "kind": kind})
    }

    fn resequence(trace: &mut Value) {
        for (index, value) in trace["events"]
            .as_array_mut()
            .expect("trace has an event array")
            .iter_mut()
            .enumerate()
        {
            value["sequence"] = json!(u64::try_from(index + 1).expect("test trace is small"));
        }
    }

    #[test]
    fn frozen_corpus_matches_declared_outcomes_and_production_checker() {
        let parsed = corpus();
        let scenarios = parsed["scenarios"].as_array().expect("scenarios array");
        assert_eq!(scenarios.len(), 12);
        for trace in scenarios {
            let actual = evaluate_trace(trace).expect("frozen trace passes input validation");
            let expected: Vec<String> = trace["expected_violation_codes"]
                .as_array()
                .expect("expected codes")
                .iter()
                .map(|value| value.as_str().expect("code string").to_owned())
                .collect();
            assert_eq!(actual, expected, "reference result for {}", trace["trace_id"]);
            let production = crate::first_aid_trajectory::inspect_trace_json(trace)
                .expect("production trace parser accepts frozen trace");
            assert_eq!(actual, production, "differential result for {}", trace["trace_id"]);
        }
        println!("reference_oracle_frozen_traces=12");
    }

    #[test]
    fn render_violation_precedence_is_explicit_and_differential() {
        let trace = base_trace(
            "offline",
            vec![json!({
                "sequence": 1,
                "kind": "protocol_rendered",
                "bundle_id": "revoked-bundle",
                "bundle_state": "revoked"
            })],
        );
        let expected = vec![
            "render_without_active_session".to_owned(),
            "render_after_revocation".to_owned(),
        ];
        assert_eq!(evaluate_trace(&trace).expect("valid trace"), expected);
        assert_eq!(
            evaluate_trace(&trace).expect("valid trace"),
            crate::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("production accepts trace")
        );
    }

    #[test]
    fn revocation_remains_monotonic_across_session_and_reload() {
        let trace = base_trace(
            "online",
            vec![
                event(1, "session_started"),
                json!({"sequence": 2, "kind": "bundle_loaded", "bundle_id": "bundle-a", "bundle_state": "valid"}),
                json!({"sequence": 3, "kind": "bundle_verified", "bundle_id": "bundle-a", "bundle_state": "valid", "digest_verified": true}),
                json!({"sequence": 4, "kind": "bundle_revoked", "bundle_id": "bundle-a"}),
                event(5, "session_started"),
                json!({"sequence": 6, "kind": "bundle_loaded", "bundle_id": "bundle-a", "bundle_state": "valid"}),
                json!({"sequence": 7, "kind": "bundle_verified", "bundle_id": "bundle-a", "bundle_state": "valid", "digest_verified": true}),
                json!({"sequence": 8, "kind": "protocol_rendered", "bundle_id": "bundle-a"})
            ],
        );
        let expected = vec![
            "invalid_bundle_verification".to_owned(),
            "render_after_revocation".to_owned(),
        ];
        let actual = evaluate_trace(&trace).expect("valid trace");
        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            crate::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("production accepts trace")
        );
    }

    #[test]
    fn event_time_connectivity_does_not_retroactively_change_prior_events() {
        let trace = base_trace(
            "offline",
            vec![
                event(1, "session_started"),
                event(2, "network_required_for_core"),
                json!({"sequence": 3, "kind": "connectivity_changed", "connectivity": "online"}),
                event(4, "network_required_for_core"),
                json!({"sequence": 5, "kind": "connectivity_changed", "connectivity": "intermittent"}),
                event(6, "network_required_for_core")
            ],
        );
        let expected = vec![
            "offline_core_network_dependency".to_owned(),
            "intermittent_core_network_dependency".to_owned(),
        ];
        let actual = evaluate_trace(&trace).expect("valid trace");
        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            crate::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("production accepts trace")
        );
    }

    #[test]
    fn a_new_uncertainty_requires_new_preservation_before_handoff() {
        let trace = base_trace(
            "online",
            vec![
                event(1, "session_started"),
                event(2, "uncertainty_detected"),
                event(3, "uncertainty_preserved"),
                event(4, "uncertainty_detected"),
                event(5, "handoff_emitted"),
            ],
        );
        let expected = vec!["uncertainty_lost_at_handoff".to_owned()];
        let actual = evaluate_trace(&trace).expect("valid trace");
        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            crate::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("production accepts trace")
        );
    }

    #[test]
    fn strict_schema_rejects_unknown_root_and_event_fields() {
        let mut trace = base_trace("online", vec![event(1, "session_started")]);
        trace["unexpected_authority"] = json!(true);
        assert!(evaluate_trace(&trace).is_err());
        assert!(crate::first_aid_trajectory::inspect_trace_json(&trace).is_err());

        let mut trace = base_trace("online", vec![event(1, "session_started")]);
        trace["events"][0]["surprise"] = json!("unsafe-default");
        assert!(evaluate_trace(&trace).is_err());
        assert!(crate::first_aid_trajectory::inspect_trace_json(&trace).is_err());
    }

    #[test]
    fn all_sensitive_payload_and_consent_combinations_are_accounted_for() {
        for sensitive in [false, true] {
            for consent in [false, true] {
                let trace = base_trace(
                    "online",
                    vec![
                        event(1, "session_started"),
                        json!({
                            "sequence": 2,
                            "kind": "sensitive_data_egress",
                            "sensitive_payload": sensitive,
                            "consent_scoped": consent
                        })
                    ],
                );
                let expected = if sensitive && !consent {
                    vec!["sensitive_data_egress_without_consent".to_owned()]
                } else {
                    Vec::new()
                };
                let actual = evaluate_trace(&trace).expect("valid trace");
                assert_eq!(actual, expected);
                assert_eq!(
                    actual,
                    crate::first_aid_trajectory::inspect_trace_json(&trace)
                        .expect("production accepts trace")
                );
            }
        }
    }

    #[test]
    fn loading_a_different_bundle_invalidates_the_previous_verification() {
        let trace = base_trace(
            "online",
            vec![
                event(1, "session_started"),
                json!({"sequence": 2, "kind": "bundle_loaded", "bundle_id": "bundle-a", "bundle_state": "valid"}),
                json!({"sequence": 3, "kind": "bundle_verified", "bundle_id": "bundle-a", "bundle_state": "valid", "digest_verified": true}),
                json!({"sequence": 4, "kind": "bundle_loaded", "bundle_id": "bundle-b", "bundle_state": "valid"}),
                json!({"sequence": 5, "kind": "protocol_rendered", "bundle_id": "bundle-a"})
            ],
        );
        let expected = vec!["render_without_verified_bundle".to_owned()];
        let actual = evaluate_trace(&trace).expect("valid trace");
        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            crate::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("production accepts trace")
        );
    }

    #[test]
    fn seeded_event_suffix_differential_catches_ordering_divergence() {
        let parsed = corpus();
        let scenarios = parsed["scenarios"].as_array().expect("scenarios array");
        const CASES: usize = 2048;
        for case in 0..CASES {
            let source = &scenarios[case % scenarios.len()];
            let mut trace = source.clone();
            let original_events = trace["events"]
                .as_array()
                .expect("frozen event array")
                .clone();
            let suffix_len = case % 4 + 1;
            for suffix in 0..suffix_len {
                let index = (case.wrapping_mul(17) + suffix.wrapping_mul(7)) % original_events.len();
                trace["events"]
                    .as_array_mut()
                    .expect("mutable event array")
                    .push(original_events[index].clone());
            }
            resequence(&mut trace);
            let reference = evaluate_trace(&trace).expect("generated suffix remains schema-valid");
            let production = crate::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("production accepts generated suffix");
            assert_eq!(reference, production, "generated case {case}");
        }
        println!("reference_oracle_differential_cases={CASES}");
    }
}
