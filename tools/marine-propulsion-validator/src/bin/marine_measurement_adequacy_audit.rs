use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Serialize)]
struct Diagnostic {
    path: String,
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct Report {
    auditor: &'static str,
    auditor_version: &'static str,
    study_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    parameter_count: usize,
    observation_count: usize,
    independent_observation_count: usize,
    sensitivity_rank: usize,
    structural_nullity: usize,
    identifiable_parameter_set: bool,
    excluded_observation_ids: Vec<String>,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
            "INCOMPLETE" => self.incomplete = true,
            _ => {}
        }
        self.diagnostics.push(Diagnostic {
            path: path.into(),
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn disposition(&self) -> &'static str {
        if self.invalid { "INVALID" }
        else if self.incomplete { "INCOMPLETE" }
        else { "IDENTIFIABILITY_GATE_PASSED_FOR_DECLARED_SET" }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON in {path}: {e}"))
}

fn validate_schema(schema: &Value, instance: &Value, f: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        f.push("/$schema", "INVALID", "only JSON Schema Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        f.push("/", "INVALID", "schema does not validate against Draft 2020-12 meta-schema");
        return;
    }
    match jsonschema::draft202012::options().should_validate_formats(true).build(schema) {
        Ok(validator) => for error in validator.iter_errors(instance) {
            f.push(error.instance_path().to_string(), "INVALID", format!("schema violation: {error}"));
        },
        Err(error) => f.push("/", "INVALID", format!("cannot compile schema: {error}")),
    }
}

fn matrix_rank(matrix: &[Vec<f64>], tolerance: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let mut values = matrix.to_vec();
    let rows = values.len();
    let columns = values[0].len();
    let scale = values.iter().flat_map(|row| row.iter()).fold(0.0_f64, |m, x| m.max(x.abs())).max(1.0);
    let threshold = tolerance * scale;
    let mut pivot_row = 0usize;

    for column in 0..columns {
        if pivot_row >= rows {
            break;
        }
        let mut best_row = pivot_row;
        let mut best_value = values[pivot_row][column].abs();
        for row in (pivot_row + 1)..rows {
            let candidate = values[row][column].abs();
            if candidate > best_value {
                best_value = candidate;
                best_row = row;
            }
        }
        if best_value <= threshold {
            continue;
        }
        values.swap(pivot_row, best_row);
        let pivot = values[pivot_row][column];
        for col in column..columns {
            values[pivot_row][col] /= pivot;
        }
        for row in (pivot_row + 1)..rows {
            let factor = values[row][column];
            if factor.abs() <= threshold {
                continue;
            }
            for col in column..columns {
                values[row][col] -= factor * values[pivot_row][col];
            }
        }
        pivot_row += 1;
    }
    pivot_row
}

fn check_semantics(instance: &Value, f: &mut Findings) -> (usize, usize, usize, usize, bool, Vec<String>) {
    let parameters = instance.get("parameters").and_then(Value::as_array).cloned().unwrap_or_default();
    let observations = instance.get("observations").and_then(Value::as_array).cloned().unwrap_or_default();
    let matrix = instance.get("sensitivity_matrix").unwrap_or(&Value::Null);
    let parameter_ids: Vec<String> = parameters.iter().filter_map(|p| p.get("parameter_id").and_then(Value::as_str).map(str::to_owned)).collect();
    let mut parameter_set = BTreeSet::new();
    for (i, id) in parameter_ids.iter().enumerate() {
        if !parameter_set.insert(id.clone()) {
            f.push(format!("/parameters/{i}/parameter_id"), "INVALID", format!("duplicate parameter id '{id}'"));
        }
    }

    let mut observation_by_id = BTreeMap::new();
    for (i, obs) in observations.iter().enumerate() {
        let Some(id) = obs.get("observation_id").and_then(Value::as_str) else {
            f.push(format!("/observations/{i}/observation_id"), "INVALID", "observation_id is missing");
            continue;
        };
        if observation_by_id.insert(id.to_owned(), obs.clone()).is_some() {
            f.push(format!("/observations/{i}/observation_id"), "INVALID", format!("duplicate observation id '{id}'"));
        }
        if obs.get("independence_class").and_then(Value::as_str) == Some("derived") {
            let parents = obs.get("derived_from_ids").and_then(Value::as_array);
            if parents.is_none_or(|v| v.is_empty()) {
                f.push(format!("/observations/{i}/derived_from_ids"), "INVALID", "derived observation must name at least one parent observation");
            } else {
                for (j, parent) in parents.unwrap().iter().enumerate() {
                    if let Some(parent_id) = parent.as_str() {
                        if !observations.iter().any(|candidate| candidate.get("observation_id").and_then(Value::as_str)==Some(parent_id)) {
                            f.push(format!("/observations/{i}/derived_from_ids/{j}"), "INVALID", format!("derived-from observation '{parent_id}' does not resolve"));
                        }
                    }
                }
            }
        }
    }

    let matrix_parameters: Vec<String> = matrix.get("parameter_ids").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect();
    if matrix_parameters != parameter_ids {
        f.push("/sensitivity_matrix/parameter_ids", "INVALID", "sensitivity parameter order must exactly match the declared parameter order");
    }
    let row_ids: Vec<String> = matrix.get("row_observation_ids").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect();
    let rows: Vec<Vec<f64>> = matrix.get("values").and_then(Value::as_array).into_iter().flatten().map(|row| row.as_array().into_iter().flatten().filter_map(Value::as_f64).collect()).collect();
    let tolerance = matrix.get("rank_tolerance").and_then(Value::as_f64).unwrap_or(1e-8);
    if rows.len() != row_ids.len() {
        f.push("/sensitivity_matrix/values", "INVALID", "matrix row count must equal row_observation_ids count");
    }
    if rows.iter().any(|r| r.len() != parameter_ids.len()) {
        f.push("/sensitivity_matrix/values", "INVALID", "each sensitivity row must have one value per declared parameter");
    }
    for id in &row_ids {
        if !observation_by_id.contains_key(id) {
            f.push("/sensitivity_matrix/row_observation_ids", "INVALID", format!("sensitivity row observation '{id}' does not resolve"));
        }
    }

    let mut independent_rows = Vec::new();
    let mut excluded = Vec::new();
    for (i, id) in row_ids.iter().enumerate() {
        let Some(obs) = observation_by_id.get(id) else { continue };
        match obs.get("independence_class").and_then(Value::as_str) {
            Some("independent_measured") => {
                independent_rows.push(rows.get(i).cloned().unwrap_or_default());
                if obs.get("calibration_status").and_then(Value::as_str) != Some("in_tolerance") {
                    f.push(format!("/observations/{id}/calibration_status"), "INCOMPLETE", "independent measured observation is not evidenced as in tolerance");
                }
                if obs.get("timebase_status").and_then(Value::as_str) != Some("qualified") {
                    f.push(format!("/observations/{id}/timebase_status"), "INCOMPLETE", "independent measured observation does not have a qualified timebase");
                }
            }
            Some("derived") | Some("inferred") | Some("manual") | Some("unknown") | None => {
                excluded.push(id.clone());
                f.push(format!("/observations/{id}/independence_class"), "INCOMPLETE", "observation is not admitted as an independent measured channel for sensitivity-rank calculation");
            }
            _ => {
                excluded.push(id.clone());
                f.push(format!("/observations/{id}/independence_class"), "INVALID", "unsupported independence class");
            }
        }
    }

    let rank = matrix_rank(&independent_rows, tolerance);
    let count = parameter_ids.len();
    let nullity = count.saturating_sub(rank);
    let identifiable = rank == count && count > 0;
    if rank < count {
        f.push("/sensitivity_matrix", "INCOMPLETE", format!("independent observation sensitivity rank {rank} is below {count} declared parameters; structural nullity is {nullity}"));
    }
    if instance.get("data_class").and_then(Value::as_str) == Some("synthetic") {
        f.push("/data_class", "INCOMPLETE", "synthetic sensitivity matrix is a software fixture, not evidence of physical parameter identifiability");
    }
    (count, observations.len(), independent_rows.len(), rank, identifiable, excluded)
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-measurement-adequacy-audit <schema.json> <case.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let instance = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &instance, &mut findings);
    let (parameters, observations, independent, rank, identifiable, excluded) = check_semantics(&instance, &mut findings);
    let report = Report {
        auditor: "marine-measurement-adequacy-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        study_id: instance.get("study_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Observation dependency and declared sensitivity-rank checks only. A full-rank matrix does not prove the observation model is correct, sensors are authentic, or parameters are physical truth.",
        parameter_count: parameters,
        observation_count: observations,
        independent_observation_count: independent,
        sensitivity_rank: rank,
        structural_nullity: parameters.saturating_sub(rank),
        identifiable_parameter_set: identifiable,
        excluded_observation_ids: excluded,
        diagnostics: findings.diagnostics,
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(if report.disposition == "IDENTIFIABILITY_GATE_PASSED_FOR_DECLARED_SET" {0} else {2})
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => { eprintln!("{error}"); std::process::exit(64); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-measurement-adequacy-v0.1.0.schema.json");
    const FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-measurement-adequacy-v0.1.0.json");

    #[test]
    fn synthetic_five_parameter_case_reports_rank_two_and_nullity_three() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        assert!(jsonschema::draft202012::options().should_validate_formats(true).build(&schema).unwrap().is_valid(&fixture));
        let mut findings = Findings::default();
        let (parameters, observations, independent, rank, identified, excluded) = check_semantics(&fixture, &mut findings);
        assert_eq!(parameters, 5);
        assert_eq!(observations, 4);
        assert_eq!(independent, 2);
        assert_eq!(rank, 2);
        assert!(!identified);
        assert_eq!(excluded.len(), 2);
        assert_eq!(findings.disposition(), "INCOMPLETE");
    }

    #[test]
    fn gaussian_elimination_rank_known_answer() {
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];
        assert_eq!(matrix_rank(&matrix, 1e-8), 2);
    }

    #[test]
    fn derived_channel_does_not_increase_independent_rank() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let mut findings = Findings::default();
        let (_, _, independent, rank, _, _) = check_semantics(&fixture, &mut findings);
        assert_eq!(independent, 2);
        assert_eq!(rank, 2);
    }

    #[test]
    fn missing_derived_parent_is_invalid() {
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["observations"][2]["derived_from_ids"][0] = serde_json::json!("MISSING-PARENT");
        let mut findings = Findings::default();
        check_semantics(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn poor_sensor_calibration_blocks_decision_grade_identifiability() {
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["observations"][0]["calibration_status"] = serde_json::json!("unknown");
        let mut findings = Findings::default();
        check_semantics(&fixture, &mut findings);
        assert!(findings.incomplete);
    }
}
