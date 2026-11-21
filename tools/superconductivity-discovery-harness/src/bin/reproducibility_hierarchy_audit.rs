use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
const CLAIM_CEILING: &str = "group-split-and-reproducibility-design-semantics-only";
const DEFAULT_SCHEMA: &str = "data/schemas/sc_reproducibility_design_v0_1.schema.json";
const DEFAULT_FIXTURE: &str =
    "tools/superconductivity-discovery-harness/fixtures/reproducibility_hierarchy_smoke_v0_1.json";

fn nonempty_text<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{path}.{key} must be a non-empty string"))
}

fn array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{path} must be an array"))
}

fn canonical_utc(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || !bytes.iter().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit()
        })
    {
        return false;
    }
    let number = |start: usize, end: usize| {
        std::str::from_utf8(&bytes[start..end])
            .ok()
            .and_then(|text| text.parse::<u32>().ok())
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        number(0, 4),
        number(5, 7),
        number(8, 10),
        number(11, 13),
        number(14, 16),
        number(17, 19),
    ) else {
        return false;
    };
    if year == 0
        || !(1..=12).contains(&month)
        || hour > 23
        || minute > 59
        || second > 59
        || day == 0
    {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    day <= days_in_month
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn get_group<'a>(observations: &'a [Value], field: &str) -> Result<BTreeSet<&'a str>, String> {
    observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            nonempty_text(observation, field, &format!("observations[{index}]"))
        })
        .collect()
}

fn require_disjoint<T: Ord + Clone>(
    left: &BTreeSet<T>,
    right: &BTreeSet<T>,
    label: &str,
) -> Result<(), String> {
    if let Some(shared) = left.intersection(right).next() {
        return Err(format!("{label} leakage across train/test: {shared:?}"));
    }
    Ok(())
}

fn require_same_single<T: Ord + Clone + std::fmt::Debug>(
    left: &BTreeSet<T>,
    right: &BTreeSet<T>,
    label: &str,
) -> Result<(), String> {
    if left.len() != 1 || left != right {
        return Err(format!(
            "{label} must be the same single identity in train and test; train={left:?}, test={right:?}"
        ));
    }
    Ok(())
}

fn validate_time_forward(train: &[Value], test: &[Value]) -> Result<(), String> {
    let mut train_times = Vec::with_capacity(train.len());
    let mut test_times = Vec::with_capacity(test.len());
    for (index, observation) in train.iter().enumerate() {
        let value = nonempty_text(observation, "timestamp_utc", &format!("train[{index}]"))?;
        if !canonical_utc(value) {
            return Err(format!(
                "train[{index}].timestamp_utc must use canonical UTC YYYY-MM-DDTHH:MM:SSZ"
            ));
        }
        train_times.push(value);
    }
    for (index, observation) in test.iter().enumerate() {
        let value = nonempty_text(observation, "timestamp_utc", &format!("test[{index}]"))?;
        if !canonical_utc(value) {
            return Err(format!(
                "test[{index}].timestamp_utc must use canonical UTC YYYY-MM-DDTHH:MM:SSZ"
            ));
        }
        test_times.push(value);
    }
    let latest_train = train_times.iter().max().expect("train is non-empty");
    let earliest_test = test_times.iter().min().expect("test is non-empty");
    if latest_train >= earliest_test {
        return Err(format!(
            "time-forward violation: latest train {latest_train} is not earlier than earliest test {earliest_test}"
        ));
    }
    Ok(())
}

fn validate_task(
    task: &Value,
    observations_by_id: &HashMap<String, Value>,
) -> Result<Value, String> {
    let task_id = nonempty_text(task, "task_id", "task")?;
    let task_type = nonempty_text(task, "task_type", "task")?;
    let train_ids = array(
        task.get("train_observation_ids")
            .ok_or_else(|| format!("{task_id}: missing train_observation_ids"))?,
        &format!("{task_id}.train_observation_ids"),
    )?;
    let test_ids = array(
        task.get("test_observation_ids")
            .ok_or_else(|| format!("{task_id}: missing test_observation_ids"))?,
        &format!("{task_id}.test_observation_ids"),
    )?;

    let mut train_seen = BTreeSet::new();
    let mut test_seen = BTreeSet::new();
    let mut train = Vec::with_capacity(train_ids.len());
    let mut test = Vec::with_capacity(test_ids.len());

    for id in train_ids {
        let id = id
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{task_id}: train IDs must be non-empty strings"))?;
        if !train_seen.insert(id.to_owned()) {
            return Err(format!("{task_id}: duplicate training observation ID {id}"));
        }
        train.push(
            observations_by_id
                .get(id)
                .ok_or_else(|| format!("{task_id}: unknown training observation ID {id}"))?
                .clone(),
        );
    }
    for id in test_ids {
        let id = id
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{task_id}: test IDs must be non-empty strings"))?;
        if !test_seen.insert(id.to_owned()) {
            return Err(format!("{task_id}: duplicate test observation ID {id}"));
        }
        if train_seen.contains(id) {
            return Err(format!(
                "{task_id}: observation {id} occurs in both train and test"
            ));
        }
        test.push(
            observations_by_id
                .get(id)
                .ok_or_else(|| format!("{task_id}: unknown test observation ID {id}"))?
                .clone(),
        );
    }
    if train.is_empty() || test.is_empty() {
        return Err(format!(
            "{task_id}: train and test sets must both be non-empty"
        ));
    }

    let train_specimens = get_group(&train, "specimen_id")?;
    let test_specimens = get_group(&test, "specimen_id")?;
    let train_batches = get_group(&train, "batch_id")?;
    let test_batches = get_group(&test, "batch_id")?;
    let train_campaigns = get_group(&train, "campaign_id")?;
    let test_campaigns = get_group(&test, "campaign_id")?;
    let train_labs = get_group(&train, "lab_id")?;
    let test_labs = get_group(&test, "lab_id")?;
    let train_instruments = get_group(&train, "instrument_id")?;
    let test_instruments = get_group(&test, "instrument_id")?;
    let train_protocols = get_group(&train, "protocol_id")?;
    let test_protocols = get_group(&test, "protocol_id")?;
    let train_conditions = get_group(&train, "condition_id")?;
    let test_conditions = get_group(&test, "condition_id")?;
    let train_properties = get_group(&train, "property_id")?;
    let test_properties = get_group(&test, "property_id")?;

    if train_properties != test_properties || train_properties.len() != 1 {
        return Err(format!(
            "{task_id}: property identity must match and be singular"
        ));
    }

    let paired = task
        .get("paired_specimen_transfer")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{task_id}: paired_specimen_transfer must be boolean"))?;

    match task_type {
        "same_specimen_repeatability" => {
            if paired {
                return Err(format!(
                    "{task_id}: repeatability is not a cross-instrument paired transfer"
                ));
            }
            require_same_single(&train_specimens, &test_specimens, "specimen")?;
            require_same_single(&train_batches, &test_batches, "batch")?;
            require_same_single(&train_campaigns, &test_campaigns, "campaign")?;
            require_same_single(&train_labs, &test_labs, "laboratory")?;
            require_same_single(&train_instruments, &test_instruments, "instrument")?;
            require_same_single(&train_protocols, &test_protocols, "protocol")?;
            require_same_single(&train_conditions, &test_conditions, "condition")?;
            let mut replicates = BTreeSet::new();
            for observation in train.iter().chain(test.iter()) {
                let replicate = observation
                    .get("replicate_index")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        format!("{task_id}: replicate_index must be positive integer")
                    })?;
                if !replicates.insert(replicate) {
                    return Err(format!(
                        "{task_id}: replicate_index {replicate} is repeated"
                    ));
                }
            }
        }
        "instrument_transfer" => {
            if !paired {
                return Err(format!(
                    "{task_id}: instrument_transfer must declare paired_specimen_transfer=true"
                ));
            }
            require_same_single(&train_specimens, &test_specimens, "paired specimen")?;
            require_same_single(&train_batches, &test_batches, "batch")?;
            require_same_single(&train_campaigns, &test_campaigns, "campaign")?;
            require_same_single(&train_labs, &test_labs, "laboratory")?;
            require_same_single(&train_protocols, &test_protocols, "protocol")?;
            require_same_single(&train_conditions, &test_conditions, "condition")?;
            require_disjoint(&train_instruments, &test_instruments, "instrument")?;
        }
        "within_batch_specimen_holdout" => {
            if paired {
                return Err(format!(
                    "{task_id}: within-batch specimen holdout must not be paired same-specimen transfer"
                ));
            }
            require_disjoint(&train_specimens, &test_specimens, "specimen")?;
            require_same_single(&train_batches, &test_batches, "batch")?;
            require_same_single(&train_campaigns, &test_campaigns, "campaign")?;
        }
        "between_batch_holdout" => {
            if paired {
                return Err(format!(
                    "{task_id}: between-batch holdout cannot be a paired same-specimen transfer"
                ));
            }
            require_disjoint(&train_specimens, &test_specimens, "specimen")?;
            require_disjoint(&train_batches, &test_batches, "batch")?;
        }
        "between_campaign_holdout" => {
            if paired {
                return Err(format!(
                    "{task_id}: between-campaign holdout cannot be a paired same-specimen transfer"
                ));
            }
            require_disjoint(&train_specimens, &test_specimens, "specimen")?;
            require_disjoint(&train_batches, &test_batches, "batch")?;
            require_disjoint(&train_campaigns, &test_campaigns, "campaign")?;
        }
        "inter_lab_transfer" => {
            require_disjoint(&train_labs, &test_labs, "laboratory")?;
            if paired {
                require_same_single(&train_specimens, &test_specimens, "paired specimen")?;
                require_same_single(&train_batches, &test_batches, "batch")?;
                require_same_single(&train_campaigns, &test_campaigns, "campaign")?;
                require_same_single(&train_conditions, &test_conditions, "condition")?;
                require_same_single(&train_protocols, &test_protocols, "protocol")?;
            } else {
                require_disjoint(&train_specimens, &test_specimens, "specimen")?;
            }
        }
        "protocol_transfer" => {
            if paired {
                return Err(format!(
                    "{task_id}: protocol transfer cannot claim a same-specimen paired transfer"
                ));
            }
            require_disjoint(&train_protocols, &test_protocols, "protocol")?;
        }
        "time_forward" => {
            if paired {
                return Err(format!(
                    "{task_id}: time-forward holdout cannot declare paired specimen transfer"
                ));
            }
            require_disjoint(&train_specimens, &test_specimens, "specimen")?;
            require_disjoint(&train_batches, &test_batches, "batch")?;
            validate_time_forward(&train, &test)?;
        }
        other => return Err(format!("{task_id}: unsupported task_type {other}")),
    }

    Ok(json!({
        "task_id": task_id,
        "task_type": task_type,
        "train_observation_count": train.len(),
        "test_observation_count": test.len(),
        "train_specimen_count": train_specimens.len(),
        "test_specimen_count": test_specimens.len(),
        "train_batch_count": train_batches.len(),
        "test_batch_count": test_batches.len(),
        "train_campaign_count": train_campaigns.len(),
        "test_campaign_count": test_campaigns.len(),
        "train_laboratory_count": train_labs.len(),
        "test_laboratory_count": test_labs.len(),
        "disposition": "PASS_GROUP_SEMANTICS"
    }))
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    for error in validator.iter_errors(&fixture) {
        return Err(format!(
            "fixture schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture.get("evidence_class").and_then(Value::as_str) != Some("synthetic") {
        return Err("auditor accepts synthetic fixtures only".to_owned());
    }
    if fixture.get("claim_ceiling").and_then(Value::as_str) != Some(CLAIM_CEILING) {
        return Err("fixture claim ceiling does not match the reproducibility audit".to_owned());
    }

    let state = fixture
        .get("material_state")
        .ok_or_else(|| "material_state is required".to_owned())?;
    let state_id = nonempty_text(state, "state_id", "material_state")?;
    let property = nonempty_text(state, "property_id", "material_state")?;
    let unit = nonempty_text(state, "property_unit", "material_state")?;
    let observations = array(
        fixture
            .get("observations")
            .ok_or_else(|| "observations are required".to_owned())?,
        "observations",
    )?;
    let mut observations_by_id = HashMap::with_capacity(observations.len());
    let mut specimens = BTreeSet::new();
    let mut batches = BTreeSet::new();
    let mut campaigns = BTreeSet::new();
    let mut labs = BTreeSet::new();
    let mut instruments = BTreeSet::new();
    let mut protocols = BTreeSet::new();

    for (index, observation) in observations.iter().enumerate() {
        let path = format!("observations[{index}]");
        if nonempty_text(observation, "material_state_id", &path)? != state_id
            || nonempty_text(observation, "property_id", &path)? != property
            || nonempty_text(observation, "property_unit", &path)? != unit
        {
            return Err(format!(
                "{path}: material_state_id/property_id/property_unit does not match material_state"
            ));
        }
        if nonempty_text(observation, "evidence_class", &path)? != "synthetic"
            || nonempty_text(observation, "evidence_authority", &path)? != "synthetic_measurement"
        {
            return Err(format!(
                "{path}: synthetic fixture cannot claim non-synthetic evidence authority"
            ));
        }
        let timestamp = nonempty_text(observation, "timestamp_utc", &path)?;
        if !canonical_utc(timestamp) {
            return Err(format!(
                "{path}.timestamp_utc must use canonical UTC YYYY-MM-DDTHH:MM:SSZ"
            ));
        }
        let observation_id = nonempty_text(observation, "observation_id", &path)?.to_owned();
        if observations_by_id
            .insert(observation_id.clone(), observation.clone())
            .is_some()
        {
            return Err(format!("duplicate observation_id {observation_id}"));
        }
        specimens.insert(nonempty_text(observation, "specimen_id", &path)?.to_owned());
        batches.insert(nonempty_text(observation, "batch_id", &path)?.to_owned());
        campaigns.insert(nonempty_text(observation, "campaign_id", &path)?.to_owned());
        labs.insert(nonempty_text(observation, "lab_id", &path)?.to_owned());
        instruments.insert(nonempty_text(observation, "instrument_id", &path)?.to_owned());
        protocols.insert(nonempty_text(observation, "protocol_id", &path)?.to_owned());
    }

    let tasks = array(
        fixture
            .get("tasks")
            .ok_or_else(|| "tasks are required".to_owned())?,
        "tasks",
    )?;
    let mut task_ids = BTreeSet::new();
    let mut task_results = Vec::with_capacity(tasks.len());
    for task in tasks {
        let task_id = nonempty_text(task, "task_id", "task")?.to_owned();
        if !task_ids.insert(task_id.clone()) {
            return Err(format!("duplicate task_id {task_id}"));
        }
        task_results.push(validate_task(task, &observations_by_id)?);
    }

    Ok(json!({
        "schema_version": "sc-reproducibility-audit-receipt.v0.1",
        "schema_sha256": sha256_hex(schema_bytes),
        "fixture_sha256": sha256_hex(fixture_bytes),
        "observation_count": observations.len(),
        "unique_specimen_count": specimens.len(),
        "unique_batch_count": batches.len(),
        "unique_campaign_count": campaigns.len(),
        "unique_laboratory_count": labs.len(),
        "unique_instrument_count": instruments.len(),
        "unique_protocol_count": protocols.len(),
        "task_count": task_results.len(),
        "tasks": task_results,
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn parse_args() -> Result<(PathBuf, PathBuf, Option<PathBuf>), String> {
    let mut schema = PathBuf::from(DEFAULT_SCHEMA);
    let mut fixture = PathBuf::from(DEFAULT_FIXTURE);
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for argument {arg}"))?;
        match arg.as_str() {
            "--schema" => schema = PathBuf::from(value),
            "--fixture" => fixture = PathBuf::from(value),
            "--out" => output = Some(PathBuf::from(value)),
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    Ok((schema, fixture, output))
}

fn run() -> Result<(), String> {
    let (schema_path, fixture_path, output_path) = parse_args()?;
    let schema_bytes = fs::read(&schema_path)
        .map_err(|error| format!("cannot read schema {}: {error}", schema_path.display()))?;
    let fixture_bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read fixture {}: {error}", fixture_path.display()))?;
    let report = audit(&schema_bytes, &fixture_bytes)?;
    let encoded = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize report: {error}"))?;
    if let Some(path) = output_path {
        fs::write(&path, &encoded)
            .map_err(|error| format!("cannot write receipt {}: {error}", path.display()))?;
    } else {
        println!(
            "{}",
            String::from_utf8(encoded).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sc_reproducibility_audit: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_fixture() -> Value {
        let schema =
            include_bytes!("../../../../data/schemas/sc_reproducibility_design_v0_1.schema.json");
        let fixture = include_bytes!("../../fixtures/reproducibility_hierarchy_smoke_v0_1.json");
        audit(schema, fixture).expect("synthetic reproducibility fixture must pass")
    }

    fn fixture_value() -> Value {
        serde_json::from_slice(include_bytes!(
            "../../fixtures/reproducibility_hierarchy_smoke_v0_1.json"
        ))
        .expect("fixture JSON must parse")
    }

    #[test]
    fn synthetic_fixture_audits_all_declared_group_designs() {
        let result = run_fixture();
        assert_eq!(result["observation_count"], 8);
        assert_eq!(result["task_count"], 8);
        assert_eq!(result["unique_specimen_count"], 5);
        assert_eq!(result["unique_batch_count"], 4);
        assert_eq!(result["unique_campaign_count"], 3);
        assert_eq!(result["unique_laboratory_count"], 3);
        assert_eq!(result["unique_instrument_count"], 4);
        assert_eq!(result["unique_protocol_count"], 2);
        assert_eq!(result["claim_ceiling"], CLAIM_CEILING);
    }

    #[test]
    fn repeatability_task_allows_same_specimen_and_same_instrument() {
        let result = run_fixture();
        assert_eq!(result["tasks"][0]["disposition"], "PASS_GROUP_SEMANTICS");
    }

    #[test]
    fn instrument_transfer_requires_same_specimen_and_disjoint_instruments() {
        let result = run_fixture();
        assert_eq!(result["tasks"][1]["disposition"], "PASS_GROUP_SEMANTICS");
    }

    #[test]
    fn within_batch_holdout_requires_disjoint_specimens_but_same_batch() {
        let result = run_fixture();
        assert_eq!(result["tasks"][2]["disposition"], "PASS_GROUP_SEMANTICS");
    }

    #[test]
    fn between_batch_holdout_rejects_specimen_or_batch_overlap() {
        let mut fixture = fixture_value();
        fixture["observations"][4]["batch_id"] = json!("BATCH-001");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("batch leakage")
        );
    }

    #[test]
    fn between_campaign_holdout_rejects_campaign_overlap() {
        let mut fixture = fixture_value();
        fixture["observations"][5]["campaign_id"] = json!("CAMP-001");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("campaign leakage")
        );
    }

    #[test]
    fn inter_lab_transfer_requires_disjoint_labs_even_for_paired_specimen() {
        let mut fixture = fixture_value();
        fixture["observations"][6]["lab_id"] = json!("LAB-001");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("laboratory leakage")
        );
    }

    #[test]
    fn inter_lab_paired_transfer_preserves_same_specimen_identity() {
        let result = run_fixture();
        assert_eq!(result["tasks"][5]["disposition"], "PASS_GROUP_SEMANTICS");
    }

    #[test]
    fn inter_lab_unpaired_transfer_rejects_shared_specimen() {
        let mut fixture = fixture_value();
        fixture["tasks"][5]["paired_specimen_transfer"] = json!(false);
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("specimen leakage")
        );
    }

    #[test]
    fn protocol_transfer_rejects_protocol_overlap() {
        let mut fixture = fixture_value();
        fixture["observations"][7]["protocol_id"] = json!("PROTO-TRANSPORT-A");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("protocol leakage")
        );
    }

    #[test]
    fn time_forward_task_rejects_future_training_observation() {
        let mut fixture = fixture_value();
        fixture["observations"][0]["timestamp_utc"] = json!("2026-11-20T15:00:00Z");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("time-forward violation")
        );
    }

    #[test]
    fn repeatability_rejects_condition_mismatch() {
        let mut fixture = fixture_value();
        fixture["observations"][1]["condition_id"] = json!("COND-OTHER");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("condition must be the same single identity")
        );
    }

    #[test]
    fn instrument_transfer_rejects_same_instrument() {
        let mut fixture = fixture_value();
        fixture["observations"][2]["instrument_id"] = json!("INST-001");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("instrument leakage")
        );
    }

    #[test]
    fn rejects_observation_from_different_material_state() {
        let mut fixture = fixture_value();
        fixture["observations"][0]["material_state_id"] = json!("OTHER-STATE");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("does not match material_state")
        );
    }

    #[test]
    fn rejects_impossible_calendar_date_even_when_timestamp_shape_matches() {
        let mut fixture = fixture_value();
        fixture["observations"][0]["timestamp_utc"] = json!("2026-02-30T09:00:00Z");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("canonical UTC")
        );
    }

    #[test]
    fn rejects_unknown_observation_references() {
        let mut fixture = fixture_value();
        fixture["tasks"][0]["train_observation_ids"][0] = json!("MISSING");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("unknown training observation")
        );
    }

    #[test]
    fn rejects_duplicate_observation_identifiers() {
        let mut fixture = fixture_value();
        fixture["observations"][1]["observation_id"] = json!("OBS-001");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("duplicate observation_id")
        );
    }

    #[test]
    fn rejects_duplicate_task_identifiers() {
        let mut fixture = fixture_value();
        fixture["tasks"][1]["task_id"] = json!("TASK-REPEAT");
        assert!(
            audit_fixture_value(&fixture)
                .unwrap_err()
                .contains("duplicate task_id")
        );
    }

    fn audit_fixture_value(fixture: &Value) -> Result<Value, String> {
        let schema =
            include_bytes!("../../../../data/schemas/sc_reproducibility_design_v0_1.schema.json");
        let bytes = serde_json::to_vec(fixture).expect("fixture serializes");
        audit(schema, &bytes)
    }
}
