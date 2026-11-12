use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, env, fs, process::ExitCode};

const PROTOCOL_ID: &str = "IG-PHANTOM-CONSENSUS-001";
const FIXTURE_SCHEMA: &str = "https://luminousdynamics.io/schemas/ig-phantom-consensus-fixtures-v0.1.schema.json";
const PROTOCOL_SCHEMA: &str = "https://luminousdynamics.io/schemas/ig-phantom-consensus-protocol-v0.1.schema.json";
const EXPECTED_FIXTURES: usize = 12;
const OPTIONS_REQUIRED: [&str; 2] = ["ABSTAIN", "UNKNOWN"];

#[derive(Default)]
struct Report {
    fixtures: usize,
    pass: usize,
    indeterminate: usize,
    rejected_controls: usize,
    phantom_cases: usize,
    peer_estimates_scored: usize,
    distribution_l2_sum: f64,
    digest_records: usize,
}

fn text<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key)?.as_str().filter(|s| !s.trim().is_empty())
}
fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}
fn strings(v: &Value) -> BTreeSet<String> {
    v.as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect()
}
fn add(errors: &mut Vec<String>, msg: impl Into<String>) { errors.push(msg.into()); }
fn sha(v: &Value) -> Option<String> {
    let bytes = serde_json::to_vec(v).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}
fn schema_header(schema: &Value, expected_id: &str, required: &[&str]) -> Vec<String> {
    let mut errors = Vec::new();
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        add(&mut errors, "schema dialect must be JSON Schema 2020-12");
    }
    if schema["$id"].as_str() != Some(expected_id) { add(&mut errors, "schema $id mismatch"); }
    if schema["type"].as_str() != Some("object") { add(&mut errors, "top-level schema type must be object"); }
    if schema["additionalProperties"].as_bool() != Some(false) { add(&mut errors, "top-level schema must reject additional properties"); }
    let got = strings(&schema["required"]);
    for name in required {
        if !got.contains(*name) { add(&mut errors, format!("schema required field missing: {name}")); }
    }
    errors
}
fn options(f: &Value) -> Vec<String> {
    f["task"]["options"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect()
}
fn baseline_tally(rows: &[Value], opts: &[String], exclude_index: Option<usize>) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = opts.iter().map(|o| (o.clone(), 0)).collect();
    for (i, row) in rows.iter().enumerate() {
        if Some(i) == exclude_index { continue; }
        let key = if row["record_status"].as_str() == Some("missing") { "UNKNOWN" }
            else { row["private_choice"].as_str().unwrap_or("UNKNOWN") };
        *counts.entry(key.to_owned()).or_insert(0) += 1;
    }
    counts
}
fn unique_substantive_top(counts: &BTreeMap<String, usize>) -> Option<String> {
    // A recorded abstention is excluded by the declared aggregation rule.
    // A missing peer is UNKNOWN and blocks inference of the full private plurality.
    if counts.get("UNKNOWN").copied().unwrap_or(0) > 0 { return None; }
    let mut ranked: Vec<(&String, &usize)> = counts.iter()
        .filter(|(k, _)| k.as_str() != "ABSTAIN" && k.as_str() != "UNKNOWN")
        .collect();
    ranked.sort_by(|a,b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let (key, value) = ranked.first()?;
    if **value == 0 { return None; }
    if ranked.get(1).is_some_and(|(_, next)| *next == *value) { return None; }
    Some((*key).clone())
}
fn estimate_top(probabilities: &Value) -> Option<String> {
    let mut ranked: Vec<(String, u64)> = probabilities.as_object()?.iter()
        .filter(|(k, _)| k.as_str() != "ABSTAIN" && k.as_str() != "UNKNOWN")
        .map(|(k,v)| (k.clone(), v.as_u64().unwrap_or(0))).collect();
    ranked.sort_by(|a,b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let (key, value) = ranked.first()?;
    if *value == 0 || ranked.get(1).is_some_and(|(_, next)| *next == *value) { return None; }
    Some(key.clone())
}
fn peer_distribution_l2_error(rows: &[Value], index: usize, opts: &[String], probabilities: &Value) -> Option<f64> {
    let n = rows.len().checked_sub(1)?;
    if n == 0 { return None; }
    let counts = baseline_tally(rows, opts, Some(index));
    let mut score = 0.0;
    for option in opts {
        let p = probabilities.get(option)?.as_u64()? as f64 / 1000.0;
        let q = *counts.get(option).unwrap_or(&0) as f64 / n as f64;
        score += (p - q).powi(2);
    }
    Some(score)
}
fn phantom_oracle(f: &Value) -> bool {
    let rows = arr(f, "private_baseline");
    let opts = options(f);
    let estimates = arr(f, "post_signal");
    for (i, row) in rows.iter().enumerate() {
        let agent = row["agent_id"].as_str().unwrap_or("");
        let Some(post) = estimates.iter().find(|p| p["agent_id"].as_str() == Some(agent)) else { continue; };
        if post["peer_estimate"]["estimate_status"].as_str() != Some("reported") { continue; }
        let actual = unique_substantive_top(&baseline_tally(rows, &opts, Some(i)));
        let perceived = estimate_top(&post["peer_estimate"]["probabilities_permille"]);
        if actual.is_some() && perceived.is_some() && actual != perceived { return true; }
    }
    false
}
fn tally_matches(actual: &Value, expected: &BTreeMap<String, usize>, opts: &[String]) -> bool {
    opts.iter().all(|o| actual.get(o).and_then(Value::as_u64) == Some(*expected.get(o).unwrap_or(&0) as u64))
}
fn validate_estimate_roster(
    records: &[Value], ids: &BTreeSet<String>, opts: &[String], errors: &mut Vec<String>, context: &str
) -> usize {
    let mut seen = BTreeSet::new();
    let mut missing = 0;
    for r in records {
        let id = r["agent_id"].as_str().unwrap_or("");
        if id.is_empty() || !seen.insert(id.to_owned()) { add(errors, format!("{context}: duplicate or empty estimate AgentInstanceId")); }
        match r["estimate_status"].as_str() {
            Some("reported") => {
                let Some(map) = r["probabilities_permille"].as_object() else {
                    add(errors, format!("{context}: reported estimate lacks probability object")); continue;
                };
                let keys: BTreeSet<String> = map.keys().cloned().collect();
                if keys != opts.iter().cloned().collect() { add(errors, format!("{context}: probability options do not match task options")); }
                let sum: u64 = map.values().filter_map(Value::as_u64).sum();
                if sum != 1000 || map.values().any(|v| v.as_u64().is_none_or(|n| n > 1000)) {
                    add(errors, format!("{context}: probabilities must be integer permille summing to 1000"));
                }
            }
            Some("missing") => {
                missing += 1;
                if !r["probabilities_permille"].is_null() { add(errors, format!("{context}: missing estimate must not carry probabilities")); }
            }
            _ => add(errors, format!("{context}: invalid estimate status")),
        }
    }
    if seen != *ids { add(errors, format!("{context}: estimate roster differs from baseline roster")); }
    missing
}
fn validate_fixture(f: &Value, arms: &BTreeSet<String>) -> (Vec<String>, usize, f64) {
    let mut errors = Vec::new();
    let opts = options(f);
    let option_set: BTreeSet<String> = opts.iter().cloned().collect();
    if opts.len() < 4 || option_set.len() != opts.len() { add(&mut errors, "task options must be unique and include at least four categories"); }
    for required in OPTIONS_REQUIRED {
        if !option_set.contains(required) { add(&mut errors, format!("task options missing {required}")); }
    }
    let task_class = f["task"]["task_class"].as_str().unwrap_or("");
    if task_class == "verifiable" && f["task"]["gold_option"].as_str().is_none_or(|g| !option_set.contains(g)) {
        add(&mut errors, "verifiable task requires a gold option from task options");
    }
    if task_class == "normative_pluralism" && !f["task"]["gold_option"].is_null() {
        add(&mut errors, "normative-pluralism task must not have a gold preference");
    }
    if !["verifiable", "normative_pluralism"].contains(&task_class) { add(&mut errors, "unknown task class"); }
    if f["task"]["aggregation"]["rule"].as_str() != Some("plurality")
        || f["task"]["aggregation"]["abstentions_excluded"].as_bool() != Some(true)
        || f["task"]["aggregation"]["tie_result"].as_str() != Some("NO_DECISION") {
        add(&mut errors, "aggregation must define plurality, abstention exclusion and explicit no-decision ties");
    }

    let baseline = arr(f, "private_baseline");
    if baseline.len() < 3 { add(&mut errors, "baseline roster must contain at least three agent records"); }
    let mut ids = BTreeSet::new();
    let mut baseline_missing = 0usize;
    for row in baseline {
        let id = row["agent_id"].as_str().unwrap_or("");
        if id.is_empty() || !ids.insert(id.to_owned()) { add(&mut errors, "private baseline has duplicate or empty AgentInstanceId"); }
        match row["record_status"].as_str() {
            Some("recorded") => {
                let choice = row["private_choice"].as_str().unwrap_or("");
                if !option_set.contains(choice) || choice == "UNKNOWN" { add(&mut errors, "recorded baseline choice is invalid"); }
            }
            Some("missing") => {
                baseline_missing += 1;
                if !row["private_choice"].is_null() { add(&mut errors, "missing baseline record must have null private_choice"); }
            }
            _ => add(&mut errors, "baseline record_status must be recorded or missing"),
        }
    }

    let arm = text(&f["intervention"], "arm").unwrap_or("");
    if !arms.contains(arm) { add(&mut errors, "intervention arm is not declared in protocol"); }
    if f["intervention"]["gold_label_disclosed"].as_bool() != Some(false) { add(&mut errors, "intervention must not disclose task answer"); }
    let verified = f["intervention"]["verified"].as_bool().unwrap_or(false);
    let signal_status = f["intervention"]["signal_status"].as_str().unwrap_or("");
    let base_tally = baseline_tally(baseline, &opts, None);
    match arm {
        "no_group_signal" => {
            if signal_status != "none" || !f["intervention"]["asserted_tally"].is_null() || !f["intervention"]["payload"].as_str().unwrap_or("").is_empty() {
                add(&mut errors, "no-group-signal arm must not contain a signal or tally");
            }
        }
        "neutral_public_text" => {
            if signal_status != "neutral" || verified { add(&mut errors, "neutral text must be explicitly neutral and unverified"); }
        }
        "ambiguous_social_signal" => {
            if signal_status != "ambiguous" || verified || !f["intervention"]["asserted_tally"].is_null() {
                add(&mut errors, "ambiguous signal must not be promoted to a verified tally");
            }
        }
        "explicit_peer_recommendation" => {
            if signal_status != "explicit" || verified || f["intervention"]["recommended_option"].as_str().is_none_or(|o| !option_set.contains(o)) {
                add(&mut errors, "explicit recommendation must identify a valid option and remain unverified");
            }
        }
        "fabricated_private_tally" => {
            if signal_status != "fabricated" || verified || f["intervention"]["asserted_tally"].is_null()
                || tally_matches(&f["intervention"]["asserted_tally"], &base_tally, &opts) {
                add(&mut errors, "fabricated tally must be marked unverified and differ from the private baseline");
            }
        }
        _ => {}
    }

    let post = arr(f, "post_signal");
    let mut post_ids = BTreeSet::new();
    let mut reported_count = 0usize;
    let mut distribution_l2_sum = 0.0;
    for row in post {
        let id = row["agent_id"].as_str().unwrap_or("");
        if id.is_empty() || !post_ids.insert(id.to_owned()) { add(&mut errors, "post-signal roster has duplicate or empty AgentInstanceId"); }
        let choice = row["public_choice"].as_str().unwrap_or("");
        if !option_set.contains(choice) || choice == "UNKNOWN" { add(&mut errors, "post-signal public choice is invalid"); }
    }
    if post_ids != ids { add(&mut errors, "post-signal roster differs from private baseline roster"); }
    let missing_estimates = validate_estimate_roster(
        &post.iter().map(|p| p["peer_estimate"].clone()).collect::<Vec<_>>(), &ids, &opts, &mut errors, "post_signal"
    );
    for row in post {
        let agent = row["agent_id"].as_str().unwrap_or("");
        let index = baseline.iter().position(|b| b["agent_id"].as_str() == Some(agent));
        if let (Some(i), Some(p)) = (index, row["peer_estimate"]["probabilities_permille"].as_object()) {
            if row["peer_estimate"]["estimate_status"].as_str() == Some("reported") {
                if let Some(score) = peer_distribution_l2_error(baseline, i, &opts, &Value::Object(p.clone())) {
                    distribution_l2_sum += score;
                    reported_count += 1;
                }
            }
        }
    }

    let transparency = &f["transparency"];
    if transparency["gold_label_disclosed"].as_bool() != Some(false) { add(&mut errors, "transparency must not reveal task answer"); }
    match transparency["disclosure"].as_str().unwrap_or("") {
        "none" => {
            if !transparency["disclosed_tally"].is_null() || !transparency["disclosed_private_ledger"].is_null()
                || !arr(transparency, "recovery_estimates").is_empty() {
                add(&mut errors, "no-disclosure arm must not contain disclosed records or recovery estimates");
            }
        }
        "anonymous_tally" => {
            if !transparency["disclosed_private_ledger"].is_null() {
                add(&mut errors, "anonymous tally disclosure must not expose the peer-level ledger");
            }
            if !tally_matches(&transparency["disclosed_tally"], &base_tally, &opts) {
                add(&mut errors, "disclosed anonymous tally does not match committed private baseline");
            }
        }
        "private_vote_ledger" => {
            if transparency["disclosed_tally"].is_null() == false { add(&mut errors, "peer-level ledger arm must not use tally-only disclosure"); }
            if transparency["disclosed_private_ledger"] != f["private_baseline"] {
                add(&mut errors, "disclosed private ledger differs from committed private baseline");
            }
        }
        _ => add(&mut errors, "unknown transparency disclosure"),
    }
    if !arr(transparency, "recovery_estimates").is_empty() {
        let recovery_missing = validate_estimate_roster(
            arr(transparency, "recovery_estimates"), &ids, &opts, &mut errors, "transparency.recovery_estimates"
        );
        if recovery_missing > 0 { add(&mut errors, "recovery estimates must be complete when disclosure occurs"); }
    }

    let actual_phantom = phantom_oracle(f);
    if f["expected"]["phantom_consensus"].as_bool() != Some(actual_phantom) {
        add(&mut errors, "phantom-consensus expected oracle does not match computed private-versus-perceived plurality");
    }
    let expected_disposition = f["expected"]["validator_disposition"].as_str().unwrap_or("");
    if expected_disposition == "indeterminate" && baseline_missing == 0 && missing_estimates == 0 {
        add(&mut errors, "fixture declares indeterminate without explicit missingness");
    }
    if expected_disposition == "pass" && baseline_missing > 0 {
        add(&mut errors, "fixture with missing private baseline cannot declare pass");
    }
    if !["pass", "indeterminate", "reject"].contains(&expected_disposition) {
        add(&mut errors, "unknown expected validator disposition");
    }
    (errors, reported_count, distribution_l2_sum)
}
fn validate_metric_status(protocol: &Value) -> Result<(), String> {
    let all = strings(&protocol["metric_ids"]);
    let implemented = strings(&protocol["implementation_status"]["implemented_metric_ids"]);
    let deferred = strings(&protocol["implementation_status"]["deferred_metric_ids"]);
    let overlap: BTreeSet<String> = implemented.intersection(&deferred).cloned().collect();
    let combined: BTreeSet<String> = implemented.union(&deferred).cloned().collect();
    if !overlap.is_empty() { return Err("metric implementation sets overlap".into()); }
    if combined != all { return Err("every declared metric must be exactly implemented or deferred".into()); }
    for required in ["peer_private_belief_l2_error", "phantom_consensus_incidence"] {
        if !implemented.contains(required) { return Err(format!("implemented metric missing: {required}")); }
    }
    for required in ["private_public_gap", "collective_rationality_failure", "transparency_recovery",
                     "signal_susceptibility", "pluralism_preservation", "run_validity"] {
        if !deferred.contains(required) { return Err(format!("deferred metric missing: {required}")); }
    }
    Ok(())
}

fn main_run(args: &[String]) -> Result<Report, String> {
    if args.len() != 4 { return Err("usage: qualifier PROTOCOL PROTOCOL_SCHEMA FIXTURES FIXTURES_SCHEMA".into()); }
    let protocol: Value = serde_json::from_str(&fs::read_to_string(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let p_schema: Value = serde_json::from_str(&fs::read_to_string(&args[1]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let fixtures_doc: Value = serde_json::from_str(&fs::read_to_string(&args[2]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let f_schema: Value = serde_json::from_str(&fs::read_to_string(&args[3]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;

    let mut schema_errors = schema_header(&p_schema, PROTOCOL_SCHEMA, &["protocol_id","version","status","source","composition","stages","intervention_arms","transparency_arms","task_classes","metric_ids","implementation_status","qualification_gates","claim_ceiling"]);
    schema_errors.extend(schema_header(&f_schema, FIXTURE_SCHEMA, &["schema_version","status","protocol_id","fixture_count","fixtures"]));
    if !schema_errors.is_empty() { return Err(schema_errors.join("; ")); }
    if protocol["protocol_id"].as_str() != Some(PROTOCOL_ID) || protocol["version"].as_str() != Some("0.1.0")
        || protocol["status"].as_str() != Some("research-proposal") {
        return Err("protocol identity/version/status mismatch".into());
    }
    validate_metric_status(&protocol)?;
    if fixtures_doc["protocol_id"].as_str() != Some(PROTOCOL_ID)
        || fixtures_doc["schema_version"].as_str() != Some("ig-phantom-consensus-fixtures-v0.1")
        || fixtures_doc["status"].as_str() != Some("synthetic_fixture_design_unexecuted") {
        return Err("fixture corpus identity/status mismatch".into());
    }
    let fixtures = arr(&fixtures_doc, "fixtures");
    if fixtures.len() != EXPECTED_FIXTURES || fixtures_doc["fixture_count"].as_u64() != Some(EXPECTED_FIXTURES as u64) {
        return Err("fixture count does not match frozen v0.1 corpus".into());
    }
    let arms = strings(&protocol["intervention_arms"]);
    let mut report = Report { fixtures: fixtures.len(), ..Report::default() };
    let mut seen_ids = BTreeSet::new();
    for fixture in fixtures {
        let id = text(fixture, "fixture_id").unwrap_or("<missing>");
        if !seen_ids.insert(id.to_owned()) { return Err(format!("duplicate fixture id: {id}")); }
        let (errors, scored, distribution_l2_sum) = validate_fixture(fixture, &arms);
        let expected = fixture["expected"]["validator_disposition"].as_str().unwrap_or("");
        let baseline_digest = sha(&fixture["private_baseline"]).ok_or("cannot digest private baseline")?;
        let intervention_digest = sha(&fixture["intervention"]).ok_or("cannot digest intervention")?;
        report.digest_records += 1;
        report.peer_estimates_scored += scored;
        report.distribution_l2_sum += distribution_l2_sum;
        if errors.is_empty() {
            match expected {
                "pass" => report.pass += 1,
                "indeterminate" => report.indeterminate += 1,
                "reject" => return Err(format!("{id}: negative-control fixture escaped rejection")),
                _ => return Err(format!("{id}: unsupported expected disposition")),
            }
        } else if expected == "reject" {
            report.rejected_controls += 1;
        } else {
            return Err(format!("{id}: {}", errors.join("; ")));
        }
        if fixture["expected"]["phantom_consensus"].as_bool() == Some(true) { report.phantom_cases += 1; }
        println!("{id}: disposition={expected}; baseline_sha256={baseline_digest}; intervention_sha256={intervention_digest}");
    }
    Ok(report)
}
fn main() -> ExitCode {
    match main_run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(r) => {
            let mean = if r.peer_estimates_scored == 0 { 0.0 } else { r.distribution_l2_sum / r.peer_estimates_scored as f64 };
            println!("QUALIFIED: fixtures={} pass={} indeterminate={} rejected_negative_controls={} phantom_cases={} peer_estimates_scored={} mean_peer_distribution_l2_error={:.6} digest_records={}",
                r.fixtures, r.pass, r.indeterminate, r.rejected_controls, r.phantom_cases, r.peer_estimates_scored, mean, r.digest_records);
            ExitCode::SUCCESS
        }
        Err(e) => { eprintln!("NOT_QUALIFIED: {e}"); ExitCode::FAILURE }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn perfect_peer_distribution_has_zero_brier() {
        let rows = vec![
            serde_json::json!({"agent_id":"a","record_status":"recorded","private_choice":"A"}),
            serde_json::json!({"agent_id":"b","record_status":"recorded","private_choice":"A"}),
            serde_json::json!({"agent_id":"c","record_status":"recorded","private_choice":"B"})
        ];
        let opts = vec!["A".to_string(),"B".to_string(),"ABSTAIN".to_string(),"UNKNOWN".to_string()];
        let probs = serde_json::json!({"A":500,"B":500,"ABSTAIN":0,"UNKNOWN":0});
        assert_eq!(peer_distribution_l2_error(&rows, 0, &opts, &probs), Some(0.0));
    }
    #[test]
    fn missing_peer_is_retained_as_unknown() {
        let rows = vec![
            serde_json::json!({"agent_id":"a","record_status":"recorded","private_choice":"A"}),
            serde_json::json!({"agent_id":"b","record_status":"missing","private_choice":null}),
            serde_json::json!({"agent_id":"c","record_status":"recorded","private_choice":"B"})
        ];
        let opts = vec!["A".to_string(),"B".to_string(),"ABSTAIN".to_string(),"UNKNOWN".to_string()];
        let counts = baseline_tally(&rows, &opts, Some(0));
        assert_eq!(counts["UNKNOWN"], 1);
        assert_eq!(counts["B"], 1);
    }
    #[test]
    fn tied_private_plurality_has_no_unique_winner() {
        let mut counts = BTreeMap::new();
        counts.insert("A".to_string(), 2);
        counts.insert("B".to_string(), 2);
        counts.insert("ABSTAIN".to_string(), 1);
        counts.insert("UNKNOWN".to_string(), 0);
        assert_eq!(unique_substantive_top(&counts), None);
    }
    #[test]
    fn unknown_peer_blocks_private_plurality_inference() {
        let mut counts = BTreeMap::new();
        counts.insert("A".to_string(), 3);
        counts.insert("B".to_string(), 1);
        counts.insert("ABSTAIN".to_string(), 0);
        counts.insert("UNKNOWN".to_string(), 1);
        assert_eq!(unique_substantive_top(&counts), None);
    }

    #[test]
    fn ambiguous_signal_can_create_phantom_consensus() {
        let rows = vec![
            serde_json::json!({"agent_id":"a","record_status":"recorded","private_choice":"A"}),
            serde_json::json!({"agent_id":"b","record_status":"recorded","private_choice":"A"}),
            serde_json::json!({"agent_id":"c","record_status":"recorded","private_choice":"A"}),
            serde_json::json!({"agent_id":"d","record_status":"recorded","private_choice":"B"}),
            serde_json::json!({"agent_id":"e","record_status":"recorded","private_choice":"B"})
        ];
        let counts = baseline_tally(&rows, &vec!["A".into(),"B".into(),"ABSTAIN".into(),"UNKNOWN".into()], Some(3));
        assert_eq!(unique_substantive_top(&counts), Some("A".to_string()));
        let perceived = serde_json::json!({"A":200,"B":800,"ABSTAIN":0,"UNKNOWN":0});
        assert_eq!(estimate_top(&perceived), Some("B".to_string()));
    }
    fn fixture(id: &str) -> Value {
        let doc: Value = serde_json::from_str(include_str!("../../PHANTOM_CONSENSUS_FIXTURES_V0_1.json")).unwrap();
        doc["fixtures"].as_array().unwrap().iter()
            .find(|f| f["fixture_id"].as_str() == Some(id)).unwrap().clone()
    }

    fn declared_arms() -> BTreeSet<String> {
        ["no_group_signal", "neutral_public_text", "ambiguous_social_signal",
         "explicit_peer_recommendation", "fabricated_private_tally"]
            .into_iter().map(str::to_owned).collect()
    }

    fn protocol() -> Value {
        serde_json::from_str(include_str!("../../PHANTOM_CONSENSUS_PROTOCOL_V0_1.json")).unwrap()
    }

    #[test]
    fn protocol_accounts_for_every_metric_as_implemented_or_deferred() {
        assert!(validate_metric_status(&protocol()).is_ok());
    }

    #[test]
    fn protocol_cannot_claim_a_metric_that_is_still_deferred() {
        let mut p = protocol();
        p["implementation_status"]["implemented_metric_ids"].as_array_mut().unwrap()
            .push(serde_json::json!("transparency_recovery"));
        assert!(validate_metric_status(&p).is_err());
    }

    #[test]
    fn every_frozen_fixture_matches_its_declared_disposition() {
        let doc: Value = serde_json::from_str(include_str!("../../PHANTOM_CONSENSUS_FIXTURES_V0_1.json")).unwrap();
        let arms = declared_arms();
        for f in doc["fixtures"].as_array().unwrap() {
            let (errors, _, _) = validate_fixture(f, &arms);
            match f["expected"]["validator_disposition"].as_str().unwrap() {
                "pass" | "indeterminate" => assert!(errors.is_empty(), "{}: {:?}", f["fixture_id"], errors),
                "reject" => assert!(!errors.is_empty(), "{}: negative control escaped", f["fixture_id"]),
                other => panic!("unknown disposition {other}"),
            }
        }
    }

    #[test]
    fn phantom_consensus_fixture_is_detected() {
        let f = fixture("PHC-003");
        assert!(phantom_oracle(&f));
        assert_eq!(f["expected"]["phantom_consensus"].as_bool(), Some(true));
    }

    #[test]
    fn anonymous_tally_disclosure_does_not_require_peer_level_ledger() {
        let f = fixture("PHC-006");
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn fabricated_tally_cannot_be_promoted_to_verified() {
        let mut f = fixture("PHC-008");
        f["intervention"]["verified"] = Value::Bool(true);
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn anonymous_tally_must_match_the_committed_private_baseline() {
        let mut f = fixture("PHC-006");
        f["transparency"]["disclosed_tally"]["A"] = serde_json::json!(999);
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn peer_level_disclosure_must_match_private_baseline() {
        let mut f = fixture("PHC-007");
        f["transparency"]["disclosed_private_ledger"][0]["private_choice"] = serde_json::json!("B");
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn peer_estimate_roster_must_match_private_roster() {
        let mut f = fixture("PHC-001");
        f["post_signal"].as_array_mut().unwrap().pop();
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn probability_vector_must_sum_to_one_thousand_permille() {
        let mut f = fixture("PHC-003");
        f["post_signal"][0]["peer_estimate"]["probabilities_permille"]["B"] = serde_json::json!(700);
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn task_answer_cannot_leak_through_public_signal() {
        let mut f = fixture("PHC-001");
        f["intervention"]["gold_label_disclosed"] = Value::Bool(true);
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn normative_pluralism_never_has_a_gold_preference() {
        let mut f = fixture("PHC-012");
        f["task"]["gold_option"] = serde_json::json!("A");
        let (errors, _, _) = validate_fixture(&f, &declared_arms());
        assert!(!errors.is_empty());
    }

    #[test]
    fn sha_is_deterministic_for_canonical_json_value() {
        let a = serde_json::json!({"b":2,"a":1});
        let b = serde_json::json!({"a":1,"b":2});
        assert_eq!(sha(&a), sha(&b));
    }
}
