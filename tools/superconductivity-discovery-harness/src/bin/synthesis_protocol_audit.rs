use jsonschema::validator_for;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn nonempty_text<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {field}"))
}

fn has_actual_value(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.values().any(|item| !item.is_null()))
}

fn has_locator(value: &Value) -> bool {
    let Some(locator) = value.as_object() else {
        return false;
    };
    ["page", "section", "table", "figure", "row_or_record_id"]
        .iter()
        .any(|field| {
            locator.get(*field).is_some_and(|item| match item {
                Value::String(text) => !text.trim().is_empty(),
                Value::Number(number) => number.as_u64().is_some_and(|n| n > 0),
                _ => false,
            })
        })
}

fn validate_protocol(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let protocol: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("protocol fixture JSON parse failed: {error}"))?;

    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }

    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    for error in validator.iter_errors(&protocol) {
        return Err(format!(
            "protocol schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }

    if protocol.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("this executable accepts synthetic fixture records only".to_owned());
    }

    let protocol_id = nonempty_text(&protocol, "protocol_id")?;
    let source_values = protocol
        .get("source_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "source_ids must be an array".to_owned())?;
    let mut source_ids = HashSet::with_capacity(source_values.len());
    for value in source_values {
        let source_id = value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "source_ids must contain non-empty strings".to_owned())?;
        if !source_ids.insert(source_id.to_owned()) {
            return Err(format!("duplicate protocol source id: {source_id}"));
        }
    }

    let precursors = protocol
        .get("precursors")
        .and_then(Value::as_array)
        .ok_or_else(|| "precursors must be an array".to_owned())?;
    let mut precursor_ids = HashSet::with_capacity(precursors.len());
    for precursor in precursors {
        let id = nonempty_text(precursor, "precursor_id")?;
        if !precursor_ids.insert(id.to_owned()) {
            return Err(format!("duplicate precursor id: {id}"));
        }
    }

    let planned = protocol
        .get("planned_steps")
        .and_then(Value::as_array)
        .ok_or_else(|| "planned_steps must be an array".to_owned())?;
    if planned.is_empty() {
        return Err("planned_steps must not be empty".to_owned());
    }

    let mut planned_by_id: HashMap<String, &Value> = HashMap::with_capacity(planned.len());
    let mut planned_ids = HashSet::with_capacity(planned.len());
    let mut available_artifacts = precursor_ids.clone();
    let mut previous_sequence: Option<u64> = None;

    for step in planned {
        let step_id = nonempty_text(step, "step_id")?;
        if !planned_ids.insert(step_id.to_owned()) {
            return Err(format!("duplicate planned step id: {step_id}"));
        }
        let sequence = step
            .get("sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{step_id}: sequence must be a non-negative integer"))?;
        if previous_sequence.is_some_and(|previous| sequence <= previous) {
            return Err("planned step sequence must be strictly increasing".to_owned());
        }
        previous_sequence = Some(sequence);

        let inputs = step
            .get("inputs")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{step_id}: inputs must be an array"))?;
        for input in inputs {
            let input = input
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| format!("{step_id}: inputs must contain non-empty strings"))?;
            if !available_artifacts.contains(input) {
                return Err(format!(
                    "{step_id}: input {input} is not a precursor or earlier planned output"
                ));
            }
        }

        let outputs = step
            .get("outputs")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{step_id}: outputs must be an array"))?;
        for output in outputs {
            let output = output
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| format!("{step_id}: outputs must contain non-empty strings"))?;
            if !available_artifacts.insert(output.to_owned()) {
                return Err(format!(
                    "{step_id}: duplicate or reused output artifact {output}"
                ));
            }
        }

        planned_by_id.insert(step_id.to_owned(), step);
    }

    let runs = protocol
        .get("execution_runs")
        .and_then(Value::as_array)
        .ok_or_else(|| "execution_runs must be an array".to_owned())?;
    let mut run_ids = HashSet::with_capacity(runs.len());
    let mut run_step_count = 0usize;
    let mut sensor_verified_count = 0usize;
    let mut failed_step_count = 0usize;

    for run in runs {
        let run_id = nonempty_text(run, "run_id")?;
        if !run_ids.insert(run_id.to_owned()) {
            return Err(format!("duplicate execution run id: {run_id}"));
        }
        let run_status = nonempty_text(run, "status")?;
        let steps = run
            .get("steps")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{run_id}: steps must be an array"))?;
        let mut run_step_ids = HashSet::with_capacity(steps.len());
        let mut run_has_failure = false;

        for step in steps {
            run_step_count += 1;
            let step_id = nonempty_text(step, "step_id")?;
            if !run_step_ids.insert(step_id.to_owned()) {
                return Err(format!("{run_id}: duplicate execution of step {step_id}"));
            }
            if !planned_by_id.contains_key(step_id) {
                return Err(format!(
                    "{run_id}: execution references unknown planned step {step_id}"
                ));
            }

            let execution_state = nonempty_text(step, "execution_state")?;
            let actual = step
                .get("actual")
                .ok_or_else(|| format!("{run_id}/{step_id}: missing actual field"))?;
            let actual_evidence = step
                .get("actual_evidence")
                .ok_or_else(|| format!("{run_id}/{step_id}: missing actual_evidence field"))?;

            match execution_state {
                "commanded" => {
                    if !actual.is_null() || !actual_evidence.is_null() {
                        return Err(format!(
                            "{run_id}/{step_id}: commanded is intent, not proof of actual execution"
                        ));
                    }
                }
                "executed_unverified" => {
                    if !has_actual_value(actual) {
                        return Err(format!(
                            "{run_id}/{step_id}: executed_unverified requires at least one actual value"
                        ));
                    }
                    if actual_evidence.is_object() {
                        let evidence_kind = nonempty_text(actual_evidence, "evidence_kind")?;
                        if matches!(evidence_kind, "sensor_trace" | "controller_log") {
                            return Err(format!(
                                "{run_id}/{step_id}: sensor/controller evidence requires sensor_verified state"
                            ));
                        }
                    }
                }
                "sensor_verified" => {
                    if !has_actual_value(actual) {
                        return Err(format!(
                            "{run_id}/{step_id}: sensor_verified requires at least one actual value"
                        ));
                    }
                    let evidence = actual_evidence.as_object().ok_or_else(|| {
                        format!("{run_id}/{step_id}: sensor_verified requires actual_evidence")
                    })?;
                    let evidence_kind = evidence
                        .get("evidence_kind")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("{run_id}/{step_id}: missing evidence_kind"))?;
                    if !matches!(evidence_kind, "sensor_trace" | "controller_log") {
                        return Err(format!(
                            "{run_id}/{step_id}: sensor_verified requires sensor_trace or controller_log evidence"
                        ));
                    }
                    let source_id = evidence
                        .get("source_id")
                        .and_then(Value::as_str)
                        .filter(|text| !text.trim().is_empty())
                        .ok_or_else(|| {
                            format!("{run_id}/{step_id}: actual evidence requires source_id")
                        })?;
                    if !source_ids.contains(source_id) {
                        return Err(format!(
                            "{run_id}/{step_id}: unresolved actual evidence source {source_id}"
                        ));
                    }
                    if !evidence.get("source_locator").is_some_and(|value| {
                        value.as_str().is_some_and(|text| !text.trim().is_empty())
                    }) {
                        return Err(format!(
                            "{run_id}/{step_id}: sensor evidence requires a non-empty source_locator"
                        ));
                    }
                    sensor_verified_count += 1;
                }
                "failed" | "aborted" => {
                    run_has_failure = true;
                    failed_step_count += 1;
                    let reason = step
                        .get("failure_reason")
                        .and_then(Value::as_str)
                        .filter(|text| !text.trim().is_empty());
                    if reason.is_none() {
                        return Err(format!(
                            "{run_id}/{step_id}: failed/aborted step requires failure_reason"
                        ));
                    }
                }
                "skipped" => {
                    if !actual.is_null() || !actual_evidence.is_null() {
                        return Err(format!(
                            "{run_id}/{step_id}: skipped step cannot claim actual execution"
                        ));
                    }
                }
                other => {
                    return Err(format!(
                        "{run_id}/{step_id}: unsupported execution state {other}"
                    ));
                }
            }
        }

        if run_status == "completed" && run_has_failure {
            return Err(format!(
                "{run_id}: completed run contains failed/aborted steps"
            ));
        }
        if matches!(run_status, "failed" | "aborted") && !run_has_failure {
            return Err(format!(
                "{run_id}: failed/aborted run must preserve a failed/aborted step"
            ));
        }
    }

    let outcomes = protocol
        .get("outcome_records")
        .and_then(Value::as_array)
        .ok_or_else(|| "outcome_records must be an array".to_owned())?;
    let mut outcome_ids = HashSet::with_capacity(outcomes.len());
    let mut phase_count = 0usize;

    for outcome in outcomes {
        let outcome_id = nonempty_text(outcome, "outcome_id")?;
        if !outcome_ids.insert(outcome_id.to_owned()) {
            return Err(format!("duplicate outcome id: {outcome_id}"));
        }
        let run_id = nonempty_text(outcome, "run_id")?;
        if !run_ids.contains(run_id) {
            return Err(format!(
                "{outcome_id}: outcome references unknown run {run_id}"
            ));
        }

        let phases = outcome
            .get("realized_phases")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{outcome_id}: realized_phases must be an array"))?;
        let mut phase_ids = HashSet::with_capacity(phases.len());
        let mut known_phase_fraction = 0.0f64;
        for phase in phases {
            let phase_id = nonempty_text(phase, "phase_id")?;
            if !phase_ids.insert(phase_id.to_owned()) {
                return Err(format!("{outcome_id}: duplicate realized phase {phase_id}"));
            }
            phase_count += 1;
            if let Some(fraction) = phase.get("phase_fraction_percent").and_then(Value::as_f64) {
                known_phase_fraction += fraction;
            }
        }
        if known_phase_fraction > 100.0 + 1e-6 {
            return Err(format!(
                "{outcome_id}: sum of reported phase fractions exceeds 100 percent"
            ));
        }
    }

    let rights_review = nonempty_text(&protocol, "rights_review")?;

    Ok(json!({
        "status": "pass",
        "contract": "SC-SYNTHESIS-PROTOCOL-0.1",
        "protocol_id": protocol_id,
        "source_count": source_ids.len(),
        "precursor_count": precursor_ids.len(),
        "planned_step_count": planned.len(),
        "execution_run_count": runs.len(),
        "execution_step_count": run_step_count,
        "sensor_verified_step_count": sensor_verified_count,
        "failed_or_aborted_step_count": failed_step_count,
        "outcome_count": outcomes.len(),
        "realized_phase_count": phase_count,
        "rights_review": rights_review,
        "checks": {
            "schema_valid": true,
            "source_ids_unique": true,
            "planned_step_ids_unique": true,
            "planned_sequences_strictly_increasing": true,
            "planned_inputs_resolve_to_precursor_or_prior_output": true,
            "execution_runs_unique": true,
            "execution_steps_resolve_to_plan": true,
            "commanded_is_not_execution": true,
            "sensor_verified_steps_have_actual_source_evidence": true,
            "failed_attempts_preserved": true,
            "outcomes_resolve_to_runs": true,
            "phase_fractions_do_not_exceed_100_percent": true
        },
        "claim_ceiling": "Synthetic protocol-contract validation only; this is not an executed synthesis, validated recipe or material discovery."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <protocol-schema.json> --fixture <protocol.json> [--out <receipt.json>]"
    );
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path: Option<PathBuf> = None;
    let mut fixture_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }

    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let schema_bytes = read(&schema_path);
    let fixture_bytes = read(&fixture_path);
    let receipt = validate_protocol(&schema_bytes, &fixture_bytes).unwrap_or_else(|error| {
        eprintln!("SC-SYNTHESIS-PROTOCOL-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing protocol audit receipt should not fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn committed_inputs() -> (Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_synthesis_protocol_v0_1.schema.json"),
        )
        .expect("protocol schema should be readable");
        let fixture: Value = serde_json::from_slice(
            &fs::read(root.join("tools/superconductivity-discovery-harness/fixtures/synthesis_protocol_smoke_v0_1.json"))
                .expect("protocol fixture should be readable")
        ).expect("protocol fixture should parse");
        (schema, fixture)
    }

    fn audit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = committed_inputs();
        let bytes = serde_json::to_vec(&fixture).unwrap();
        validate_protocol(&schema, &bytes)
    }

    #[test]
    fn committed_protocol_fixture_passes() {
        let (schema, fixture) = committed_inputs();
        let bytes = serde_json::to_vec(&fixture).unwrap();
        let receipt = validate_protocol(&schema, &bytes).unwrap();
        assert_eq!(receipt["planned_step_count"], 2);
        assert_eq!(receipt["execution_run_count"], 1);
        assert_eq!(receipt["sensor_verified_step_count"], 2);
        assert_eq!(receipt["outcome_count"], 1);
    }

    #[test]
    fn commanded_step_cannot_claim_actual_execution() {
        let (_, mut fixture) = committed_inputs();
        fixture["execution_runs"][0]["steps"][0]["execution_state"] = json!("commanded");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn sensor_verified_requires_a_resolved_source() {
        let (_, mut fixture) = committed_inputs();
        fixture["execution_runs"][0]["steps"][0]["actual_evidence"]["source_id"] =
            json!("missing-source");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn failed_attempts_require_a_failure_reason_and_are_preserved() {
        let (_, mut fixture) = committed_inputs();
        fixture["execution_runs"][0]["steps"][0]["execution_state"] = json!("failed");
        fixture["execution_runs"][0]["steps"][0]["actual_evidence"] = Value::Null;
        fixture["execution_runs"][0]["steps"][0]["failure_reason"] = json!("synthetic failure");
        fixture["execution_runs"][0]["status"] = json!("failed");
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["failed_or_aborted_step_count"], 1);
    }

    #[test]
    fn outcomes_must_reference_an_existing_run() {
        let (_, mut fixture) = committed_inputs();
        fixture["outcome_records"][0]["run_id"] = json!("missing-run");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn phase_fractions_cannot_exceed_one_hundred_percent() {
        let (_, mut fixture) = committed_inputs();
        fixture["outcome_records"][0]["realized_phases"].push(json!({
            "phase_id": "second-phase",
            "identity_status": "identified",
            "formula": "synthetic second phase",
            "structure": null,
            "phase_fraction_percent": 40.0,
            "microstructure_notes": null
        }));
        fixture["outcome_records"][0]["realized_phases"][0]["phase_fraction_percent"] = json!(70.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn planned_step_sequences_must_be_increasing() {
        let (_, mut fixture) = committed_inputs();
        fixture["planned_steps"][1]["sequence"] = json!(0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn execution_must_reference_a_planned_step() {
        let (_, mut fixture) = committed_inputs();
        fixture["execution_runs"][0]["steps"][0]["step_id"] = json!("missing-step");
        assert!(audit_mutated(fixture).is_err());
    }
}
