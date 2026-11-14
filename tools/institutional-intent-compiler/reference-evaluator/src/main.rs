use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process,
};

fn object<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_object()?.get(key)
}

fn string<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    object(v, key)?.as_str()
}

fn number(v: &Value, key: &str) -> Option<u64> {
    object(v, key)?.as_u64()
}

fn array<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    object(v, key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

fn present_non_null(v: &Value, key: &str) -> bool {
    object(v, key).is_some_and(|x| !x.is_null())
}

fn add_decision_faults(
    requirement: &Value,
    manifest: &Value,
    reference_key: &str,
    allowed: &[&str],
    out: &mut BTreeSet<String>,
) {
    let Some(reference) = string(requirement, reference_key) else {
        out.insert("missing_decision".to_owned());
        return;
    };
    let decisions = array(manifest, "decisions");
    let Some(decision) = decisions.iter().find(|d| string(d, "id") == Some(reference)) else {
        out.insert("unknown_decision".to_owned());
        return;
    };
    let req_id = string(requirement, "id");
    if string(decision, "subject_type") != Some("requirement")
        || string(decision, "subject_id") != req_id
    {
        out.insert("decision_subject_mismatch".to_owned());
    }
    if !string(decision, "action").is_some_and(|a| allowed.contains(&a)) {
        out.insert("decision_action_mismatch".to_owned());
    }
    if number(decision, "baseline_version") != number(manifest, "baseline_version") {
        out.insert("decision_baseline_mismatch".to_owned());
    }
}

fn evaluate(manifest: &Value) -> BTreeSet<String> {
    let mut faults = BTreeSet::new();

    let mut source_lengths = BTreeMap::<String, u64>::new();
    for source in array(manifest, "sources") {
        if let (Some(id), Some(length)) = (string(source, "id"), number(source, "byte_length")) {
            source_lengths.insert(id.to_owned(), length);
        }
    }

    let requirements = array(manifest, "requirements");
    if requirements.is_empty() {
        faults.insert("empty_requirements".to_owned());
    }

    for requirement in requirements {
        let accepted = string(requirement, "lifecycle") == Some("accepted");
        let requirement_id = string(requirement, "id");
        let trace = object(requirement, "traceability");
        let spans = trace.and_then(|t| object(t, "source_spans")).and_then(Value::as_array);
        let spans = spans.map(Vec::as_slice).unwrap_or(&[]);
        let has_derivation = trace.is_some_and(|t| present_non_null(t, "derivation_ref"));

        if accepted && spans.is_empty() && !has_derivation {
            faults.insert("accepted_without_trace".to_owned());
        }

        for span in spans {
            let source_id = string(span, "source_id");
            let Some(length) = source_id.and_then(|id| source_lengths.get(id).copied()) else {
                faults.insert("unknown_source".to_owned());
                continue;
            };
            let start = number(span, "start_byte").unwrap_or(u64::MAX);
            let end = number(span, "end_byte").unwrap_or(u64::MAX);
            if start >= end || end > length {
                faults.insert("invalid_source_span".to_owned());
            }
        }

        if accepted {
            add_decision_faults(
                requirement,
                manifest,
                "review_decision_ref",
                &["accept", "amend"],
                &mut faults,
            );

            if string(requirement, "declared_modality") != string(requirement, "proposed_modality") {
                add_decision_faults(
                    requirement,
                    manifest,
                    "modality_change_decision_ref",
                    &["approve_modality_change"],
                    &mut faults,
                );
            }
        }

        for link in array(requirement, "links") {
            if string(link, "kind") != Some("depends_on") || !accepted {
                continue;
            }
            let target = string(link, "target_requirement_id");
            let target_requirement = requirements.iter().find(|candidate| string(candidate, "id") == target);
            if target_requirement.is_some_and(|candidate| string(candidate, "lifecycle") != Some("accepted")) {
                faults.insert("accepted_dependency_not_accepted".to_owned());
            }
        }

        // Explicitly keep the reference's decision subject bound to this candidate.
        let _ = requirement_id;
    }

    faults
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let Some(cases_dir) = args.next() else {
        return Err("usage: institutional-intent-reference-evaluator <fixtures-dir> <known-answers.json>".to_owned());
    };
    let Some(catalog_path) = args.next() else {
        return Err("missing known-answers.json path".to_owned());
    };
    if args.next().is_some() {
        return Err("expected exactly two arguments".to_owned());
    }

    let catalog_text = fs::read_to_string(&catalog_path)
        .map_err(|e| format!("cannot read catalog '{catalog_path}': {e}"))?;
    let catalog: Value = serde_json::from_str(&catalog_text)
        .map_err(|e| format!("invalid catalog JSON: {e}"))?;
    let cases = array(&catalog, "cases");
    if cases.is_empty() {
        return Err("catalog contains no cases".to_owned());
    }

    let root = PathBuf::from(cases_dir);
    let mut results = Vec::new();
    let mut failed = 0usize;

    for case in cases {
        let case_id = string(case, "id").unwrap_or("<missing-id>");
        let file = string(case, "file").ok_or_else(|| format!("case {case_id} has no file"))?;
        let expected_status = string(case, "expected_status").unwrap_or("REJECTED");
        let expected_codes: BTreeSet<String> = array(case, "expected_diagnostics")
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();

        let path = root.join(file);
        let input = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read fixture '{}': {e}", path.display()))?;
        let manifest: Value = serde_json::from_str(&input)
            .map_err(|e| format!("invalid fixture '{}': {e}", path.display()))?;
        let actual_codes = evaluate(&manifest);
        let actual_status = if actual_codes.is_empty() { "STRUCTURALLY_VALID" } else { "REJECTED" };
        let passed = actual_status == expected_status && actual_codes == expected_codes;
        if !passed {
            failed += 1;
        }
        results.push(json!({
            "id": case_id,
            "file": file,
            "passed": passed,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "expected_diagnostics": expected_codes,
            "actual_diagnostics": actual_codes
        }));
    }

    let report = json!({
        "evaluator": "independent-structural-reference-v0.1",
        "case_count": results.len(),
        "passed": results.len() - failed,
        "failed": failed,
        "results": results
    });
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| format!("cannot encode report: {e}"))?);
    Ok(if failed == 0 { 0 } else { 1 })
}

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            2
        }
    };
    process::exit(code);
}
