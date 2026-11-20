use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const CONTRACT: &str = "SC-ONSET-TRAINING-COHORT-0.1";
const CLAIM_CEILING: &str = "Candidate-cohort leakage audit only; no original PG-LIT completeness, model training, onset prediction or laboratory outcome is established.";

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn canonical_pair(a: &str, b: &str) -> Result<String, String> {
    if a.trim().is_empty() || b.trim().is_empty() {
        return Err("precursor formula fields must be non-empty".to_owned());
    }
    if a == b {
        return Err("precursor pair must contain two distinct formula identities".to_owned());
    }
    let (left, right) = if a <= b { (a, b) } else { (b, a) };
    Ok(format!("{left} + {right}"))
}
fn validate_holdout(value: &Value) -> Result<(HashSet<String>, HashSet<String>), String> {
    if text_field(value, "contract")? != "SC-ONSET-PAIR-HOLDOUT-MANIFEST-0.1" {
        return Err("unsupported pair holdout manifest contract".to_owned());
    }
    let policy = value.get("exclusion_policy").ok_or("holdout exclusion_policy is missing")?;
    let expected_pair_count = policy.get("heldout_pair_count").and_then(Value::as_u64)
        .ok_or("holdout heldout_pair_count is missing")?;
    let expected_sample_count = policy.get("corresponding_precursor_genome_sample_count").and_then(Value::as_u64)
        .ok_or("holdout corresponding_precursor_genome_sample_count is missing")?;
    let pairs = value.get("pairs").and_then(Value::as_array).ok_or("holdout pairs must be an array")?;
    if pairs.len() as u64 != expected_pair_count || pairs.len() as u64 != expected_sample_count {
        return Err("holdout pair/sample counts do not match manifest policy".to_owned());
    }
    let mut pair_keys = HashSet::new();
    let mut sample_ids = HashSet::new();
    let mut reference_ids = HashSet::new();
    for pair in pairs {
        let reference_id = text_field(pair, "reference_id")?.to_owned();
        let key = text_field(pair, "precursor_pair_key")?.to_owned();
        let sample_id = text_field(pair, "precursor_genome_screen_sample_id")?.to_owned();
        if !reference_ids.insert(reference_id.clone()) {
            return Err(format!("duplicate holdout reference_id {reference_id}"));
        }
        if !pair_keys.insert(key.clone()) {
            return Err(format!("duplicate holdout precursor pair {key}"));
        }
        if !sample_ids.insert(sample_id.clone()) {
            return Err(format!("duplicate held-out Precursor Genome sample ID {sample_id}"));
        }
    }
    if sample_ids.len() != 26 || pair_keys.len() != 26 {
        return Err("this v0.1 holdout contract requires 26 unique pair and PG sample identities".to_owned());
    }
    if policy.get("same_pair_literature_rows_excluded").and_then(Value::as_u64) != Some(218) {
        return Err("holdout manifest must preserve the paper's 218 same-pair PG-LIT exclusion count".to_owned());
    }
    Ok((pair_keys, sample_ids))
}
fn audit(schema_bytes: &[u8], holdout_bytes: &[u8], cohort_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes).map_err(|e| format!("schema JSON parse failed: {e}"))?;
    let holdout: Value = serde_json::from_slice(holdout_bytes).map_err(|e| format!("holdout JSON parse failed: {e}"))?;
    let cohort: Value = serde_json::from_slice(cohort_bytes).map_err(|e| format!("cohort JSON parse failed: {e}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("training cohort schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema).map_err(|e| format!("training cohort schema compilation failed: {e}"))?;
    if let Some(error) = validator.iter_errors(&cohort).next() {
        return Err(format!("training cohort schema validation failed: {error} at {}", error.instance_path()));
    }
    if text_field(&cohort, "contract")? != CONTRACT || text_field(&cohort, "claim_ceiling")? != CLAIM_CEILING {
        return Err("unsupported training cohort contract or claim ceiling".to_owned());
    }
    let (holdout_pairs, holdout_samples) = validate_holdout(&holdout)?;
    let rows = cohort.get("rows").and_then(Value::as_array).ok_or("training rows must be an array")?;
    let mut row_ids = HashSet::new();
    let mut passed_rows = Vec::with_capacity(rows.len());
    let mut source_counts = std::collections::BTreeMap::<String, usize>::new();
    for row in rows {
        let row_id = text_field(row, "row_id")?.to_owned();
        if !row_ids.insert(row_id.clone()) {
            return Err(format!("duplicate training row_id {row_id}"));
        }
        let source = text_field(row, "source_kind")?.to_owned();
        let precursor_a = text_field(row, "precursor_a")?;
        let precursor_b = text_field(row, "precursor_b")?;
        let key = canonical_pair(precursor_a, precursor_b)?;
        if holdout_pairs.contains(&key) {
            return Err(format!(
                "held-out precursor-pair leakage: training row {row_id} ({source}) matches {key}"
            ));
        }
        match source.as_str() {
            "precursor_genome" => {
                let sample_id = text_field(row, "sample_id")?.to_owned();
                if holdout_samples.contains(&sample_id) {
                    return Err(format!(
                        "held-out Precursor Genome sample leakage: training row {row_id} references {sample_id}"
                    ));
                }
            }
            "pg_lit_literature" => {
                if !row.get("sample_id").is_some_and(Value::is_null) {
                    return Err(format!("PG-LIT row {row_id} must have null sample_id"));
                }
            }
            _ => return Err(format!("unsupported training source_kind {source}")),
        }
        *source_counts.entry(source.clone()).or_default() += 1;
        passed_rows.push(json!({
            "row_id": row_id,
            "source_kind": source,
            "canonical_precursor_pair_key": key
        }));
    }
    passed_rows.sort_by(|left, right| left["row_id"].as_str().cmp(&right["row_id"].as_str()));
    Ok(json!({
        "status": "onset_holdout_leakage_audit_pass",
        "contract": "SC-ONSET-HOLDOUT-LEAKAGE-AUDIT-0.1",
        "cohort_id": text_field(&cohort, "cohort_id")?,
        "dataset_snapshot_sha256": text_field(&cohort, "dataset_snapshot_sha256")?,
        "training_cohort_sha256": sha256_hex(cohort_bytes),
        "holdout_manifest_sha256": sha256_hex(holdout_bytes),
        "schema_sha256": sha256_hex(schema_bytes),
        "training_row_count": rows.len(),
        "source_counts": source_counts,
        "checked_heldout_pair_count": holdout_pairs.len(),
        "checked_heldout_pg_sample_count": holdout_samples.len(),
        "excluded_same_pair_pg_lit_count_from_paper": 218,
        "literature_row_ids_available": false,
        "rows": passed_rows,
        "claim_ceiling": CLAIM_CEILING
    }))
}
fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --holdout <holdout-manifest.json> --cohort <training-cohort.json> [--out <receipt.json>]");
}
fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut holdout_path = None;
    let mut cohort_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--holdout" | "--cohort" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--holdout" => holdout_path = Some(path),
                    "--cohort" => cohort_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(holdout_path), Some(cohort_path)) =
        (schema_path, holdout_path, cohort_path)
    else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = audit(&read(&schema_path), &read(&holdout_path), &read(&cohort_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-ONSET-HOLDOUT-LEAKAGE-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
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
    fn inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(root.join("data/schemas/sc_onset_training_cohort_v0_1.schema.json")).unwrap();
        let holdout = fs::read(root.join("data/research/sc_onset_pair_holdout_manifest_v0_1.json")).unwrap();
        let cohort = json!({
            "contract": CONTRACT,
            "cohort_id": "synthetic_training",
            "dataset_snapshot_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "rows": [
                {"row_id":"PG-SYN-01","source_kind":"precursor_genome","precursor_a":"Li2CO3","precursor_b":"Al2O3","sample_id":"PG_SYN_01"},
                {"row_id":"LIT-SYN-01","source_kind":"pg_lit_literature","precursor_a":"Na2CO3","precursor_b":"ZrO2","sample_id":null}
            ],
            "claim_ceiling": CLAIM_CEILING
        });
        (schema, holdout, serde_json::to_vec(&cohort).unwrap())
    }
    fn audit_mutated(data: Value) -> Result<Value, String> {
        let (schema, holdout, _) = inputs();
        audit(&schema, &holdout, &serde_json::to_vec(&data).unwrap())
    }
    #[test]
    fn clean_training_cohort_passes() {
        let (schema, holdout, cohort) = inputs();
        let receipt = audit(&schema, &holdout, &cohort).unwrap();
        assert_eq!(receipt["status"], "onset_holdout_leakage_audit_pass");
        assert_eq!(receipt["training_row_count"], 2);
        assert_eq!(receipt["checked_heldout_pair_count"], 26);
    }
    #[test]
    fn direct_pg_sample_id_leak_is_rejected_even_when_formula_key_differs() {
        let (_, _, cohort) = inputs();
        let mut data: Value = serde_json::from_slice(&cohort).unwrap();
        data["rows"][0]["sample_id"] = json!("PG_2238");
        let error = audit_mutated(data).unwrap_err();
        assert!(error.contains("Precursor Genome sample leakage"));
    }
    #[test]
    fn pg_lit_same_pair_leak_is_rejected() {
        let (_, _, cohort) = inputs();
        let mut data: Value = serde_json::from_slice(&cohort).unwrap();
        data["rows"][1]["precursor_a"] = json!("Nb2O5");
        data["rows"][1]["precursor_b"] = json!("Li2CO3");
        let error = audit_mutated(data).unwrap_err();
        assert!(error.contains("held-out precursor-pair leakage"));
    }
    #[test]
    fn reversed_pair_order_is_still_rejected() {
        let (_, _, cohort) = inputs();
        let mut data: Value = serde_json::from_slice(&cohort).unwrap();
        data["rows"][0]["precursor_a"] = json!("Nb2O5");
        data["rows"][0]["precursor_b"] = json!("Li2CO3");
        let error = audit_mutated(data).unwrap_err();
        assert!(error.contains("held-out precursor-pair leakage"));
    }
    #[test]
    fn duplicate_row_ids_are_rejected() {
        let (_, _, cohort) = inputs();
        let mut data: Value = serde_json::from_slice(&cohort).unwrap();
        let mut duplicate = data["rows"][0].clone();
        duplicate["row_id"] = json!("PG-SYN-01");
        data["rows"].as_array_mut().unwrap().push(duplicate);
        let error = audit_mutated(data).unwrap_err();
        assert!(error.contains("duplicate training row_id"));
    }
    #[test]
    fn repeated_precursor_formula_is_rejected() {
        let (_, _, cohort) = inputs();
        let mut data: Value = serde_json::from_slice(&cohort).unwrap();
        data["rows"][0]["precursor_b"] = json!("Li2CO3");
        let error = audit_mutated(data).unwrap_err();
        assert!(error.contains("distinct formula identities"));
    }
}
