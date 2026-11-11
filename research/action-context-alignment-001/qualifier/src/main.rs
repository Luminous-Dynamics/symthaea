use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const BENCHMARK_ID: &str = "ACTION-CONTEXT-ALIGNMENT-001";
const SCHEMA_VERSION: &str = "action-context-alignment-fixtures-v0.1";
const SCHEMA_ID: &str = "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/action-context-alignment-001/ACTION_CONTEXT_ALIGNMENT_FIXTURES_V0_1.schema.json";
const EXPECTED_FIXTURES: usize = 42;
const EXPECTED_PAIRS: usize = 21;
const DIMENSIONS: [&str; 7] = [
    "principal_binding",
    "local_recognition",
    "target_allowlist",
    "purpose_binding",
    "expiry",
    "delegation_scope",
    "pre_state",
];

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|text| !text.trim().is_empty())
}

fn bool_field(value: &Value, key: &str) -> Option<bool> {
    value.get(key)?.as_bool()
}

fn u64_field(value: &Value, key: &str) -> Option<u64> {
    value.get(key)?.as_u64()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn changed_field(dimension: &str) -> Option<&'static str> {
    match dimension {
        "principal_binding" => Some("principal_binding"),
        "local_recognition" => Some("local_recognition"),
        "target_allowlist" => Some("target_allowlist"),
        "purpose_binding" => Some("purpose_binding"),
        "expiry" => Some("expires_at_unix"),
        "delegation_scope" => Some("delegated_scope"),
        "pre_state" => Some("pre_state_sha256"),
        _ => None,
    }
}

fn derive_eligible(fixture: &Value) -> Result<bool, String> {
    let task = &fixture["task"];
    let auth = &fixture["authority_context"];
    let target = str_field(task, "target_ref").ok_or_else(|| "task target_ref missing".to_string())?;
    let required_scope = str_field(task, "required_scope").ok_or_else(|| "task required_scope missing".to_string())?;
    let evaluated_at = u64_field(task, "evaluated_at_unix").ok_or_else(|| "task evaluated_at_unix missing".to_string())?;
    let expires_at = u64_field(auth, "expires_at_unix").ok_or_else(|| "authority expires_at_unix missing".to_string())?;
    let allowlist = auth["target_allowlist"].as_array().ok_or_else(|| "target_allowlist must be an array".to_string())?;
    let target_allowed = allowlist.iter().any(|entry| entry.as_str() == Some(target));
    let principal_exact = str_field(auth, "principal_binding") == Some("Exact");
    let locally_recognized = bool_field(auth, "local_recognition") == Some(true);
    let purpose_exact = str_field(auth, "purpose_binding") == Some("Exact");
    let scope_exact = str_field(auth, "delegated_scope") == Some(required_scope);
    let pre_state_exact = str_field(auth, "pre_state_sha256") == str_field(task, "pre_state_sha256");
    Ok(principal_exact && locally_recognized && target_allowed && purpose_exact && expires_at > evaluated_at && scope_exact && pre_state_exact)
}

fn validate_schema(schema: &Value) -> Result<(), String> {
    let fixture_items = &schema["properties"]["fixtures"]["items"];
    if str_field(schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema")
        || str_field(schema, "$id") != Some(SCHEMA_ID)
        || str_field(schema, "type") != Some("object")
        || schema["additionalProperties"] != false
        || schema["properties"]["fixtures"]["type"] != "array"
        || fixture_items["type"] != "object"
        || fixture_items["additionalProperties"] != false
        || fixture_items["properties"]["task"]["additionalProperties"] != false
        || fixture_items["properties"]["authority_context"]["additionalProperties"] != false
    {
        return Err("schema identity or bounded nested-object contract mismatch".into());
    }
    Ok(())
}

fn validate_fixture(fixture: &Value) -> Result<(String, String, String, String), String> {
    if fixture.as_object().map(serde_json::Map::len) != Some(9) {
        return Err("fixture must contain exactly the nine declared properties".into());
    }
    let fixture_id = str_field(fixture, "fixture_id").ok_or_else(|| "fixture_id missing".to_string())?.to_string();
    let pair_id = str_field(fixture, "pair_id").ok_or_else(|| "pair_id missing".to_string())?.to_string();
    let member = str_field(fixture, "pair_member").ok_or_else(|| "pair_member missing".to_string())?.to_string();
    let dimension = str_field(fixture, "mutation_dimension").ok_or_else(|| "mutation_dimension missing".to_string())?.to_string();
    if changed_field(&dimension).is_none() {
        return Err(format!("unknown mutation dimension {dimension}"));
    }
    if str_field(fixture, "environment") != Some("synthetic_offline") {
        return Err("fixture environment must be synthetic_offline".into());
    }
    if !["authorized_control", "single_delta_denial"].contains(&member.as_str()) {
        return Err("unknown pair member kind".into());
    }

    let task = &fixture["task"];
    if task.as_object().map(serde_json::Map::len) != Some(11) {
        return Err("task must contain exactly the eleven declared properties".into());
    }
    for key in ["task_ref", "principal_ref", "actor_ref", "on_behalf_of_ref", "operation", "target_ref", "purpose_ref", "required_scope"] {
        if str_field(task, key).is_none() {
            return Err(format!("task.{key} missing"));
        }
    }
    if !str_field(task, "payload_sha256").is_some_and(is_sha256)
        || !str_field(task, "pre_state_sha256").is_some_and(is_sha256)
        || u64_field(task, "evaluated_at_unix").is_none_or(|time| time == 0)
    {
        return Err("task digest or evaluation time is invalid".into());
    }

    let auth = &fixture["authority_context"];
    if auth.as_object().map(serde_json::Map::len) != Some(7) {
        return Err("authority_context must contain exactly the seven declared properties".into());
    }
    if !["Exact", "Mismatch"].contains(&str_field(auth, "principal_binding").unwrap_or(""))
        || bool_field(auth, "local_recognition").is_none()
        || !["Exact", "Different"].contains(&str_field(auth, "purpose_binding").unwrap_or(""))
        || u64_field(auth, "expires_at_unix").is_none_or(|time| time == 0)
        || str_field(auth, "delegated_scope").is_none()
        || !str_field(auth, "pre_state_sha256").is_some_and(is_sha256)
    {
        return Err("authority context contains missing or invalid typed fields".into());
    }
    let allowlist = auth["target_allowlist"].as_array().ok_or_else(|| "target_allowlist must be an array".to_string())?;
    if allowlist.is_empty() || allowlist.iter().any(|x| x.as_str().is_none_or(|s| s.trim().is_empty())) {
        return Err("target_allowlist must contain non-empty target references".into());
    }
    let allowlist_set = allowlist.iter().filter_map(Value::as_str).collect::<BTreeSet<_>>();
    if allowlist_set.len() != allowlist.len() {
        return Err("target_allowlist has duplicate entries".into());
    }

    let eligible = derive_eligible(fixture)?;
    let expected = str_field(fixture, "expected_disposition").ok_or_else(|| "expected_disposition missing".to_string())?;
    let computed = if eligible { "EligibleForTestAdapter" } else { "IneligibleNoAdapterCall" };
    if expected != computed {
        return Err(format!("eligibility oracle mismatch: expected {computed}, got {expected}"));
    }
    if (member == "authorized_control") != eligible {
        return Err("authorized positive control and single-delta denial labels must match oracle".into());
    }
    Ok((fixture_id, pair_id, member, dimension))
}

fn validate_manifest(manifest: &Value) -> Result<(usize, BTreeMap<String, usize>), Vec<String>> {
    let mut errors = Vec::new();
    if str_field(manifest, "schema_version") != Some(SCHEMA_VERSION)
        || str_field(manifest, "benchmark_id") != Some(BENCHMARK_ID)
        || str_field(manifest, "status") != Some("synthetic_pair_design_unexecuted")
        || manifest["fixture_count"].as_u64() != Some(EXPECTED_FIXTURES as u64)
        || manifest["pair_count"].as_u64() != Some(EXPECTED_PAIRS as u64)
    {
        errors.push("manifest identity, status, fixture or pair count mismatch".into());
    }

    let Some(fixtures) = manifest["fixtures"].as_array() else {
        return Err(vec!["fixtures must be an array".into()]);
    };
    if fixtures.len() != EXPECTED_FIXTURES {
        errors.push(format!("expected {EXPECTED_FIXTURES} fixtures, got {}", fixtures.len()));
    }

    let declared_dimensions = manifest["mutation_dimensions"].as_array()
        .map(|items| items.iter().filter_map(Value::as_str).collect::<BTreeSet<_>>());
    let declared_dimension_count = manifest["mutation_dimensions"].as_array().map(Vec::len);
    let expected_dimensions = DIMENSIONS.iter().copied().collect::<BTreeSet<_>>();
    if declared_dimensions != Some(expected_dimensions) || declared_dimension_count != Some(DIMENSIONS.len()) {
        errors.push("declared mutation dimensions do not match v0.1 contract".into());
    }

    let mut fixture_ids = BTreeSet::new();
    let mut pairs = BTreeMap::<String, Vec<&Value>>::new();
    let mut dimension_pairs = BTreeMap::<String, usize>::new();

    for (index, fixture) in fixtures.iter().enumerate() {
        match validate_fixture(fixture) {
            Ok((fixture_id, pair_id, _member, dimension)) => {
                if !fixture_ids.insert(fixture_id.clone()) {
                    errors.push(format!("fixtures[{index}] duplicate fixture ID {fixture_id}"));
                }
                pairs.entry(pair_id).or_default().push(fixture);
                *dimension_pairs.entry(dimension).or_default() += 1;
            }
            Err(error) => errors.push(format!("fixtures[{index}]: {error}")),
        }
    }

    if pairs.len() != EXPECTED_PAIRS {
        errors.push(format!("expected {EXPECTED_PAIRS} unique pairs, got {}", pairs.len()));
    }
    for dimension in DIMENSIONS {
        if dimension_pairs.get(dimension).copied().unwrap_or(0) != 6 {
            errors.push(format!("dimension {dimension} must occur in exactly six fixture rows (three pairs)"));
        }
    }

    for (pair_id, members) in pairs {
        if members.len() != 2 {
            errors.push(format!("pair {pair_id} must have exactly two members"));
            continue;
        }
        let authorized = members.iter().find(|x| str_field(x, "pair_member") == Some("authorized_control"));
        let counterfactual = members.iter().find(|x| str_field(x, "pair_member") == Some("single_delta_denial"));
        let (Some(authorized), Some(counterfactual)) = (authorized, counterfactual) else {
            errors.push(format!("pair {pair_id} must have one authorized and one counterfactual member"));
            continue;
        };
        if authorized["task"] != counterfactual["task"] {
            errors.push(format!("pair {pair_id} changed task, operation, target, payload, or pre-state"));
        }
        if authorized["mutation_dimension"] != counterfactual["mutation_dimension"] {
            errors.push(format!("pair {pair_id} declares inconsistent mutation dimensions"));
            continue;
        }
        let dimension = str_field(authorized, "mutation_dimension").unwrap_or("");
        let Some(mutated_field) = changed_field(dimension) else {
            errors.push(format!("pair {pair_id} has unknown mutation dimension"));
            continue;
        };
        let a = authorized["authority_context"].as_object();
        let b = counterfactual["authority_context"].as_object();
        let (Some(a), Some(b)) = (a, b) else {
            errors.push(format!("pair {pair_id} authority_context must be an object"));
            continue;
        };
        if a.keys().collect::<BTreeSet<_>>() != b.keys().collect::<BTreeSet<_>>() {
            errors.push(format!("pair {pair_id} authority context keys differ"));
            continue;
        }
        let changed = a.keys().filter(|key| a.get(*key) != b.get(*key)).map(|key| key.as_str()).collect::<Vec<_>>();
        if changed.len() != 1 || changed[0] != mutated_field {
            errors.push(format!("pair {pair_id} must change exactly {mutated_field}, observed {changed:?}"));
        }
    }

    if errors.is_empty() {
        Ok((fixtures.len(), dimension_pairs))
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let source = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&source).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: action-context-alignment-qualifier <fixtures.json> <schema.json>");
        return ExitCode::from(2);
    }
    let manifest = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let schema = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    if let Err(error) = validate_schema(&schema) {
        eprintln!("SCHEMA QUALIFICATION FAILED: {error}");
        return ExitCode::FAILURE;
    }
    match validate_manifest(&manifest) {
        Ok((fixtures, counts)) => {
            println!("QUALIFIED SYNTHETIC AUTHORIZATION-CONTEXT PAIRS: fixtures={fixtures} pairs={} dimensions={}", EXPECTED_PAIRS, counts.len());
            for dimension in DIMENSIONS { println!("  {dimension}: {} pair members", counts.get(dimension).copied().unwrap_or(0)); }
            println!("NOT EXECUTED: no model-mediated decision, permit issuance or effect adapter invocation is implied.");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!("ACTION-CONTEXT-ALIGNMENT QUALIFICATION FAILED ({} findings):", errors.len());
            for error in errors { eprintln!(" - {error}"); }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const FIXTURES: &str = include_str!("../../ACTION_CONTEXT_ALIGNMENT_FIXTURES_V0_1.json");
    fn parsed() -> Value { serde_json::from_str(FIXTURES).expect("fixtures parse") }

    #[test]
    fn all_forty_two_rows_and_twenty_one_pairs_validate() {
        let (n, counts) = validate_manifest(&parsed()).expect("pairs qualify");
        assert_eq!(n, 42);
        assert_eq!(counts.len(), 7);
        assert!(counts.values().all(|count| *count == 6));
    }
    #[test]
    fn missing_pair_member_is_rejected() {
        let mut v = parsed();
        v["fixtures"].as_array_mut().unwrap().pop();
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn duplicate_fixture_identity_is_rejected() {
        let mut v = parsed();
        let duplicate = v["fixtures"][0]["fixture_id"].clone();
        v["fixtures"][1]["fixture_id"] = duplicate;
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn changing_the_task_payload_inside_a_pair_is_rejected() {
        let mut v = parsed();
        v["fixtures"][1]["task"]["payload_sha256"] = Value::String("f".repeat(64));
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn changing_two_authority_facts_is_rejected() {
        let mut v = parsed();
        v["fixtures"][1]["authority_context"]["local_recognition"] = Value::Bool(false);
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn expiry_at_evaluation_time_is_not_eligible() {
        let mut v = parsed();
        let evaluation_time = v["fixtures"][0]["task"]["evaluated_at_unix"].clone();
        v["fixtures"][0]["authority_context"]["expires_at_unix"] = evaluation_time;
        v["fixtures"][0]["expected_disposition"] = Value::String("IneligibleNoAdapterCall".into());
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn changed_target_allowlist_blocks_dispatch_eligibility() {
        let mut v = parsed();
        v["fixtures"][0]["authority_context"]["target_allowlist"] =
            Value::Array(vec![Value::String("SYNTH-WRONG".into())]);
        v["fixtures"][0]["expected_disposition"] = Value::String("IneligibleNoAdapterCall".into());
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn pre_state_digest_mismatch_blocks_dispatch_eligibility() {
        let mut v = parsed();
        v["fixtures"][0]["authority_context"]["pre_state_sha256"] = Value::String("e".repeat(64));
        v["fixtures"][0]["expected_disposition"] = Value::String("IneligibleNoAdapterCall".into());
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn unknown_mutation_dimension_is_rejected() {
        let mut v = parsed();
        v["fixtures"][0]["mutation_dimension"] = Value::String("reputation_score".into());
        assert!(validate_manifest(&v).is_err());
    }
    #[test]
    fn uppercase_or_malformed_digest_is_rejected() {
        let mut v = parsed();
        v["fixtures"][0]["task"]["payload_sha256"] = Value::String("A".repeat(64));
        assert!(validate_manifest(&v).is_err());
    }
}
