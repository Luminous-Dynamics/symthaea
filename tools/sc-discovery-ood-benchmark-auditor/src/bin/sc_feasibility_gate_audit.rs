use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {key}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value.as_f64().ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() { return Err(format!("{context} must be finite")); }
    Ok(number)
}

fn nullable_number(value: &Value, context: &str) -> Result<Option<f64>, String> {
    if value.is_null() { return Ok(None); }
    let number = finite_number(value, context)?;
    Ok(Some(number))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_iso_date(value: &str) -> bool {
    if value.len() != 10 || value.as_bytes()[4] != b'-' || value.as_bytes()[7] != b'-' {
        return false;
    }
    let Ok(year) = value[0..4].parse::<u32>() else { return false; };
    let Ok(month) = value[5..7].parse::<u32>() else { return false; };
    let Ok(day) = value[8..10].parse::<u32>() else { return false; };
    if year == 0 || !(1..=12).contains(&month) { return false; }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&day)
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!("feasibility schema validation failed: {error} at {}", error.instance_path()));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("feasibility-gate auditor accepts synthetic fixtures only".to_owned());
    }
    if required_text(&fixture, "contract")? != "SC-FEASIBILITY-GATE-0.1" {
        return Err("unsupported feasibility-gate contract".to_owned());
    }
    let instance_id = required_text(&fixture, "instance_id")?;
    let records = fixture.get("records").and_then(Value::as_array)
        .ok_or_else(|| "records must be an array".to_owned())?;
    let predictions = fixture.get("predictions").and_then(Value::as_array)
        .ok_or_else(|| "predictions must be an array".to_owned())?;
    let mut record_map: HashMap<&str, &Value> = HashMap::with_capacity(records.len());
    let mut material_sources = HashSet::new();
    for record in records {
        let id = required_text(record, "record_id")?;
        if record_map.insert(id, record).is_some() {
            return Err(format!("duplicate record_id {id}"));
        }
        let publication_date = required_text(record, "publication_date")?;
        if !valid_iso_date(publication_date) {
            return Err(format!("record {id} has invalid ISO publication_date"));
        }
        // The identity includes the complete condition context: one material
        // can legitimately have multiple pressure/temperature/sample observations
        // in a single paper and those must not be collapsed into one label.
        let identity = [
            required_text(record, "material_id")?,
            required_text(record, "source_id")?,
            required_text(record, "publication_date")?,
            record["condition_context"].to_string(),
        ].join("|");
        if !material_sources.insert(identity) {
            return Err(format!("duplicate material/source/date evidence identity at {id}"));
        }
        let label = required_text(record, "label_state")?;
        let observed_tc = nullable_number(&record["observed_tc_k"], &format!("record {id} observed_tc_k"))?;
        let condition = &record["condition_context"];
        let range = condition["measurement_temperature_range_k"].as_array()
            .ok_or_else(|| format!("record {id} temperature range must be an array"))?;
        let lower_t = finite_number(&range[0], "temperature range lower bound")?;
        let upper_t = finite_number(&range[1], "temperature range upper bound")?;
        if lower_t > upper_t {
            return Err(format!("record {id} temperature range is reversed"));
        }
        let pressure = nullable_number(&condition["pressure_gpa"], "pressure_gpa")?;
        if pressure.is_some_and(|value| value < 0.0) {
            return Err(format!("record {id} pressure must be non-negative"));
        }
        let evidence_scope = required_text(condition, "evidence_scope")?;
        let criterion = required_text(condition, "transition_criterion")?;
        match label {
            "observed_superconducting" => {
                if evidence_scope == "unknown_or_not_reported" || criterion == "unknown" || criterion == "none" {
                    return Err(format!("positive label {id} lacks an observed transition criterion/scope"));
                }
            }
            "observed_non_superconducting_under_conditions" => {
                if evidence_scope == "unknown_or_not_reported" || evidence_scope == "database_absence" {
                    return Err(format!("negative label {id} is not supported by condition-qualified evidence"));
                }
                if criterion != "none" {
                    return Err(format!("negative label {id} must state criterion=none under its observation conditions"));
                }
            }
            "unknown_or_not_reported" => {
                if observed_tc.is_some() {
                    return Err(format!("unknown label {id} cannot carry an observed Tc value"));
                }
            }
            other => return Err(format!("unsupported label state {other} at {id}")),
        }
        if label != "observed_superconducting" && observed_tc.is_some() {
            return Err(format!("only observed-superconducting record {id} may carry observed Tc"));
        }
    }

    let mut prediction_map: HashMap<&str, &Value> = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        let id = required_text(prediction, "record_id")?;
        if !record_map.contains_key(id) {
            return Err(format!("prediction references unknown record {id}"));
        }
        if prediction_map.insert(id, prediction).is_some() {
            return Err(format!("duplicate prediction for {id}"));
        }
    }
    if prediction_map.len() != record_map.len() {
        let missing: Vec<&str> = record_map.keys()
            .filter(|id| !prediction_map.contains_key(**id)).copied().collect();
        return Err(format!("prediction coverage incomplete; missing {missing:?}"));
    }

    let mut labeled_count = 0_u64;
    let mut labeled_predicted_count = 0_u64;
    let mut unknown_count = 0_u64;
    let mut unknown_abstained_count = 0_u64;
    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut tc_count = 0_u64;
    let mut tc_absolute_error_sum = 0.0;
    let mut tc_interval_covered = 0_u64;
    let mut positive_with_tc_count = 0_u64;

    for (id, record) in &record_map {
        let prediction = prediction_map[id];
        let label = required_text(record, "label_state")?;
        let abstained = prediction["feasibility_abstained"].as_bool()
            .ok_or_else(|| format!("prediction {id} feasibility_abstained must be boolean"))?;
        let probability = nullable_number(
            &prediction["p_superconducting"],
            &format!("prediction {id} p_superconducting"),
        )?;
        let tc_prediction = nullable_number(
            &prediction["conditional_tc_pred_k"],
            &format!("prediction {id} conditional_tc_pred_k"),
        )?;
        let tc_lower = nullable_number(&prediction["tc_lower_k"], &format!("prediction {id} tc_lower_k"))?;
        let tc_upper = nullable_number(&prediction["tc_upper_k"], &format!("prediction {id} tc_upper_k"))?;

        if abstained && probability.is_some() {
            return Err(format!("abstained feasibility prediction {id} must not emit a probability"));
        }
        if !abstained && probability.is_none() {
            return Err(format!("non-abstained feasibility prediction {id} must emit p_superconducting"));
        }
        if label == "unknown_or_not_reported" {
            unknown_count += 1;
            if !abstained || probability.is_some() || tc_prediction.is_some() || tc_lower.is_some() || tc_upper.is_some() {
                return Err(format!("unknown/unreported record {id} must abstain at both stages"));
            }
            unknown_abstained_count += 1;
            continue;
        }

        labeled_count += 1;
        if !abstained {
            let p = probability.ok_or_else(|| format!("labeled prediction {id} missing probability"))?;
            let truth = if label == "observed_superconducting" { 1.0 } else { 0.0 };
            brier_sum += (p - truth).powi(2);
            let clipped = p.clamp(1.0e-12, 1.0 - 1.0e-12);
            log_loss_sum += if truth > 0.5 { -clipped.ln() } else { -(1.0 - clipped).ln() };
            labeled_predicted_count += 1;
        }

        let observed_tc = nullable_number(&record["observed_tc_k"], &format!("record {id} observed_tc_k"))?;
        if label == "observed_superconducting" && observed_tc.is_some() {
            positive_with_tc_count += 1;
            match (tc_prediction, tc_lower, tc_upper) {
                (Some(predicted), Some(lower), Some(upper)) => {
                    if lower > predicted || predicted > upper || lower > upper {
                        return Err(format!("Tc prediction interval is inconsistent for {id}"));
                    }
                    let observed = observed_tc.unwrap();
                    if lower < 0.0 || predicted <= 0.0 {
                        return Err(format!("Tc prediction must be positive for {id}"));
                    }
                    tc_count += 1;
                    tc_absolute_error_sum += (predicted - observed).abs();
                    if lower <= observed && observed <= upper {
                        tc_interval_covered += 1;
                    }
                }
                (None, None, None) => {}
                _ => return Err(format!("Tc prediction and interval must be all present or all null for {id}")),
            }
        } else if tc_prediction.is_some() || tc_lower.is_some() || tc_upper.is_some() {
            return Err(format!("conditional Tc prediction is forbidden for non-positive/no-Tc record {id}"));
        }
    }

    if labeled_count == 0 || labeled_predicted_count == 0 {
        return Err("no scored labeled feasibility predictions remain after abstention".to_owned());
    }
    let feasibility_coverage = labeled_predicted_count as f64 / labeled_count as f64;
    let brier = brier_sum / labeled_predicted_count as f64;
    let log_loss = log_loss_sum / labeled_predicted_count as f64;
    let tc_mae = if tc_count > 0 { Some(tc_absolute_error_sum / tc_count as f64) } else { None };
    let tc_coverage = if positive_with_tc_count > 0 {
        Some(tc_count as f64 / positive_with_tc_count as f64)
    } else { None };
    let tc_interval_coverage = if tc_count > 0 {
        Some(tc_interval_covered as f64 / tc_count as f64)
    } else { None };

    Ok(json!({
        "status": "synthetic_contract_pass",
        "contract": "SC-FEASIBILITY-GATE-AUDIT-0.1",
        "instance_id": instance_id,
        "schema_sha256": sha256(schema_bytes),
        "fixture_sha256": sha256(fixture_bytes),
        "record_count": records.len(),
        "prediction_count": predictions.len(),
        "label_counts": {
            "observed_superconducting": records.iter().filter(|record| record["label_state"] == "observed_superconducting").count(),
            "observed_non_superconducting_under_conditions": records.iter().filter(|record| record["label_state"] == "observed_non_superconducting_under_conditions").count(),
            "unknown_or_not_reported": unknown_count
        },
        "feasibility_stage": {
            "scored_labeled_count": labeled_predicted_count,
            "labeled_count": labeled_count,
            "coverage_on_labeled_records": feasibility_coverage,
            "selective_brier_score": brier,
            "selective_log_loss": log_loss,
            "unknown_records_required_to_abstain": unknown_count,
            "unknown_records_correctly_abstained": unknown_abstained_count
        },
        "conditional_tc_stage": {
            "eligible_positive_records_with_observed_tc": positive_with_tc_count,
            "predicted_positive_records": tc_count,
            "prediction_coverage": tc_coverage,
            "mae_k": tc_mae,
            "interval_coverage": tc_interval_coverage,
            "scoring_population": "observed_superconducting records with an observed Tc label only"
        },
        "claim_ceiling": "Synthetic feasibility-gate contract only; no external data, superconducting material, Tc, or model performance is established."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
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
            "--help" | "-h" => { usage(&args[0]); return; }
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
    let receipt = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
        eprintln!("SC-FEASIBILITY-GATE-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing feasibility-gate receipt should not fail");
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
    use std::fs;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/sc_feasibility_gate_v0_1.schema.json")).unwrap(),
            fs::read(root.join("data/research/sc_feasibility_gate_fixture_v0_1.json")).unwrap(),
        )
    }

    fn mutated_fixture(mut mutation: impl FnMut(&mut Value)) -> Result<Value, String> {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutation(&mut fixture);
        audit(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn unknown_or_unreported_is_required_to_abstain() {
        assert!(mutated_fixture(|fixture| {
            fixture["predictions"][2]["feasibility_abstained"] = json!(false);
            fixture["predictions"][2]["p_superconducting"] = json!(0.0);
        }).is_err());
    }

    #[test]
    fn unknown_records_cannot_receive_conditional_tc() {
        assert!(mutated_fixture(|fixture| {
            fixture["predictions"][2]["conditional_tc_pred_k"] = json!(10.0);
            fixture["predictions"][2]["tc_lower_k"] = json!(9.0);
            fixture["predictions"][2]["tc_upper_k"] = json!(11.0);
        }).is_err());
    }

    #[test]
    fn non_superconducting_control_cannot_receive_conditional_tc_prediction() {
        assert!(mutated_fixture(|fixture| {
            fixture["predictions"][1]["conditional_tc_pred_k"] = json!(1.0);
            fixture["predictions"][1]["tc_lower_k"] = json!(0.8);
            fixture["predictions"][1]["tc_upper_k"] = json!(1.2);
        }).is_err());
    }

    #[test]
    fn negative_label_requires_condition_qualified_observation_not_database_absence() {
        assert!(mutated_fixture(|fixture| {
            fixture["records"][1]["condition_context"]["evidence_scope"] =
                json!("unknown_or_not_reported");
        }).is_err());
    }

    #[test]
    fn positive_record_without_observed_tc_cannot_receive_conditional_tc_prediction() {
        assert!(mutated_fixture(|fixture| {
            fixture["records"][0]["observed_tc_k"] = Value::Null;
        }).is_err() || mutated_fixture(|fixture| {
            fixture["records"][0]["observed_tc_k"] = Value::Null;
            fixture["predictions"][0]["conditional_tc_pred_k"] = json!(6.0);
        }).is_ok());
        assert!(mutated_fixture(|fixture| {
            fixture["records"][0]["observed_tc_k"] = Value::Null;
            fixture["predictions"][0]["conditional_tc_pred_k"] = json!(6.0);
            fixture["predictions"][0]["tc_lower_k"] = json!(5.0);
            fixture["predictions"][0]["tc_upper_k"] = json!(7.0);
        }).is_err());
    }

    #[test]
    fn publication_date_must_be_a_real_calendar_date() {
        assert!(mutated_fixture(|fixture| {
            fixture["records"][0]["publication_date"] = json!("2026-02-30");
        }).is_err());
    }

    #[test]
    fn unknown_label_cannot_carry_observed_tc() {
        assert!(mutated_fixture(|fixture| {
            fixture["records"][2]["observed_tc_k"] = json!(5.0);
        }).is_err());
    }

    #[test]
    fn conditional_tc_metric_uses_only_positive_observed_tc_records() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        assert_eq!(receipt["conditional_tc_stage"]["eligible_positive_records_with_observed_tc"], 2);
        assert_eq!(receipt["conditional_tc_stage"]["predicted_positive_records"], 2);
        assert_eq!(receipt["feasibility_stage"]["labeled_count"], 4);
        assert_eq!(receipt["feasibility_stage"]["unknown_records_correctly_abstained"], 2);
    }

    #[test]
    fn metrics_are_deterministic() {
        let (schema, fixture) = inputs();
        assert_eq!(audit(&schema, &fixture).unwrap(), audit(&schema, &fixture).unwrap());
    }
}
