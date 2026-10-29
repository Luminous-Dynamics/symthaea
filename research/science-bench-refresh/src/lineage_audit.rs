//! Evaluator-side lineage ledger validation and cross-split ancestry audit.
//!
//! This is separate from policy-input similarity. It detects shared declared
//! world/source ancestry, validates ledger coverage against the exact task set,
//! and blocks refresh records marked as previously exposed. It does not prove
//! the runner kept this ledger hidden or that the ledger's author is honest.

use crate::{
    TaskSetArtifactV1, TASK_SET_SCHEMA, leakage_audit::canonical_policy_input_sha256,
    leakage_audit::validate_split_label, sha256_hex,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const LEDGER_SCHEMA: &str = "symthaea.science_bench_refresh_lineage_ledger.v1";
const REPORT_SCHEMA: &str = "symthaea.science_bench_refresh_lineage_report.v1";
const MAX_IDS_PER_GROUP: usize = 20;
const LINEAGE_KINDS: &[&str] = &[
    "parent_world",
    "source_task",
    "source_item",
    "transform_ancestor",
];
const EXPOSURE_STATES: &[&str] = &[
    "never_exposed",
    "public_development",
    "prior_scored_refresh",
    "unknown",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageLedgerV1 {
    pub schema: String,
    pub generation_id: String,
    pub split_label: String,
    pub task_set_sha256: String,
    pub records: Vec<LineageRecordV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageRecordV1 {
    pub task_id: String,
    pub policy_input_sha256: String,
    pub exposure_state: String,
    pub lineage_commitments: Vec<LineageCommitmentV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageCommitmentV1 {
    pub kind: String,
    pub commitment_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedLineageGroupV1 {
    pub kind: String,
    pub commitment_sha256: String,
    pub left_task_count: u64,
    pub right_task_count: u64,
    pub cross_split_task_pair_count: u64,
    pub left_task_ids_sample: Vec<String>,
    pub right_task_ids_sample: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LineageAuditReportV1 {
    pub schema: String,
    pub valid: bool,
    pub left_split_label: String,
    pub right_split_label: String,
    pub left_generation_id: String,
    pub right_generation_id: String,
    pub left_task_set_sha256: String,
    pub right_task_set_sha256: String,
    pub left_ledger_sha256: String,
    pub right_ledger_sha256: String,
    pub left_task_count: u64,
    pub right_task_count: u64,
    pub shared_lineage_root_count: u64,
    pub shared_lineage_task_pair_count: u64,
    pub shared_lineage: Vec<SharedLineageGroupV1>,
    pub left_exposed_refresh_task_count: u64,
    pub left_exposed_refresh_task_ids_sample: Vec<String>,
    pub right_exposed_refresh_task_count: u64,
    pub right_exposed_refresh_task_ids_sample: Vec<String>,
    pub admission: String,
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_task_set(bytes: &[u8], label: &str) -> Result<TaskSetArtifactV1, String> {
    let task_set: TaskSetArtifactV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("{label} task set does not match strict v1 shape: {error}"))?;
    if task_set.schema != TASK_SET_SCHEMA {
        return Err(format!("{label} task-set schema must be {TASK_SET_SCHEMA}"));
    }
    if task_set.generation_id.trim().is_empty() || task_set.generation_id.len() > 128 {
        return Err(format!("{label} generation_id must not be empty"));
    }
    if task_set.tasks.is_empty() {
        return Err(format!("{label} task set must not be empty"));
    }
    let mut task_ids = BTreeSet::new();
    for task in &task_set.tasks {
        if task.task_id.trim().is_empty() || task.task_id.len() > 256 {
            return Err(format!("{label} task set contains an empty task ID"));
        }
        if task.family_id.trim().is_empty() || task.family_id.len() > 128 {
            return Err(format!("{label} task {} has invalid family_id", task.task_id));
        }
        if !task_ids.insert(task.task_id.as_str()) {
            return Err(format!(
                "{label} task set contains duplicate task ID {}",
                task.task_id
            ));
        }
        if !task.policy_input.is_object() || !task.evaluator_only.is_object() {
            return Err(format!(
                "{label} task {} must have object-shaped policy_input and evaluator_only",
                task.task_id
            ));
        }
    }
    Ok(task_set)
}

fn read_and_validate_ledger(
    task_set_bytes: &[u8],
    ledger_bytes: &[u8],
    label: &str,
) -> Result<(TaskSetArtifactV1, LineageLedgerV1), String> {
    let task_set = read_task_set(task_set_bytes, label)?;
    let ledger: LineageLedgerV1 = serde_json::from_slice(ledger_bytes)
        .map_err(|error| {
            format!("{label} lineage ledger does not match strict v1 shape: {error}")
        })?;

    if ledger.schema != LEDGER_SCHEMA {
        return Err(format!("{label} lineage-ledger schema must be {LEDGER_SCHEMA}"));
    }
    if ledger.generation_id != task_set.generation_id {
        return Err(format!(
            "{label} lineage ledger generation_id does not match the task set"
        ));
    }
    validate_split_label(&ledger.split_label)
        .map_err(|error| format!("{label} lineage ledger: {error}"))?;
    if !is_lower_hex_digest(&ledger.task_set_sha256) {
        return Err(format!(
            "{label} ledger task_set_sha256 must be 64 lowercase hex characters"
        ));
    }
    let actual_task_set_sha256 = sha256_hex(task_set_bytes);
    if ledger.task_set_sha256 != actual_task_set_sha256 {
        return Err(format!("{label} ledger does not bind the exact task-set bytes"));
    }
    if ledger.records.len() != task_set.tasks.len() {
        return Err(format!(
            "{label} lineage ledger record count {} does not match task count {}",
            ledger.records.len(),
            task_set.tasks.len()
        ));
    }

    let mut task_by_id = BTreeMap::new();
    for task in &task_set.tasks {
        task_by_id.insert(task.task_id.as_str(), task);
    }
    let mut observed = BTreeSet::new();
    for record in &ledger.records {
        if record.task_id.trim().is_empty() {
            return Err(format!("{label} lineage ledger contains an empty task ID"));
        }
        if !observed.insert(record.task_id.as_str()) {
            return Err(format!(
                "{label} lineage ledger contains duplicate task ID {}",
                record.task_id
            ));
        }
        let task = task_by_id
            .get(record.task_id.as_str())
            .ok_or_else(|| {
                format!("{label} lineage ledger has orphan task ID {}", record.task_id)
            })?;

        if !is_lower_hex_digest(&record.policy_input_sha256) {
            return Err(format!(
                "{label} task {} policy_input_sha256 must be 64 lowercase hex characters",
                record.task_id
            ));
        }
        let expected_policy_sha = canonical_policy_input_sha256(&task.policy_input)?;
        if record.policy_input_sha256 != expected_policy_sha {
            return Err(format!(
                "{label} task {} policy input digest does not match",
                record.task_id
            ));
        }
        if !EXPOSURE_STATES.contains(&record.exposure_state.as_str()) {
            return Err(format!(
                "{label} task {} has unsupported exposure_state {}",
                record.task_id, record.exposure_state
            ));
        }
        if record.lineage_commitments.is_empty() {
            return Err(format!(
                "{label} task {} must declare at least one lineage commitment",
                record.task_id
            ));
        }

        let mut seen_commitments = BTreeSet::new();
        for commitment in &record.lineage_commitments {
            if !LINEAGE_KINDS.contains(&commitment.kind.as_str()) {
                return Err(format!(
                    "{label} task {} has unsupported lineage kind {}",
                    record.task_id, commitment.kind
                ));
            }
            if !is_lower_hex_digest(&commitment.commitment_sha256) {
                return Err(format!(
                    "{label} task {} lineage commitment must be 64 lowercase hex characters",
                    record.task_id
                ));
            }
            if !seen_commitments.insert((
                commitment.kind.as_str(),
                commitment.commitment_sha256.as_str(),
            )) {
                return Err(format!(
                    "{label} task {} repeats a lineage commitment",
                    record.task_id
                ));
            }
        }
    }

    if observed.len() != task_by_id.len() {
        return Err(format!("{label} lineage ledger does not cover every task ID"));
    }
    Ok((task_set, ledger))
}

type RootIndex = BTreeMap<(String, String), Vec<String>>;

fn lineage_index(ledger: &LineageLedgerV1) -> RootIndex {
    let mut index = RootIndex::new();
    for record in &ledger.records {
        for commitment in &record.lineage_commitments {
            index
                .entry((
                    commitment.kind.clone(),
                    commitment.commitment_sha256.clone(),
                ))
                .or_default()
                .push(record.task_id.clone());
        }
    }
    for task_ids in index.values_mut() {
        task_ids.sort();
        task_ids.dedup();
    }
    index
}

fn exposed_refresh_tasks(ledger: &LineageLedgerV1) -> Vec<String> {
    if !ledger.split_label.starts_with("refresh") {
        return Vec::new();
    }
    let mut tasks: Vec<String> = ledger
        .records
        .iter()
        .filter(|record| record.exposure_state != "never_exposed")
        .map(|record| record.task_id.clone())
        .collect();
    tasks.sort();
    tasks
}

fn bounded_sample(values: &[String]) -> Vec<String> {
    values.iter().take(MAX_IDS_PER_GROUP).cloned().collect()
}

fn checked_pairs(left: usize, right: usize) -> Result<u64, String> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| "lineage overlap pair count overflows usize".to_owned())?;
    u64::try_from(value).map_err(|_| "lineage overlap pair count exceeds u64".to_owned())
}

/// Validate each evaluator-side ledger against its exact task-set artifact,
/// then detect shared world/source ancestry across the declared split boundary.
pub fn audit_lineage_overlap(
    left_task_set_bytes: &[u8],
    left_ledger_bytes: &[u8],
    right_task_set_bytes: &[u8],
    right_ledger_bytes: &[u8],
) -> Result<LineageAuditReportV1, String> {
    let (left_set, left_ledger) =
        read_and_validate_ledger(left_task_set_bytes, left_ledger_bytes, "left")?;
    let (right_set, right_ledger) =
        read_and_validate_ledger(right_task_set_bytes, right_ledger_bytes, "right")?;

    if left_ledger.split_label == right_ledger.split_label {
        return Err("lineage ledgers must identify different split labels".to_owned());
    }

    let left_roots = lineage_index(&left_ledger);
    let right_roots = lineage_index(&right_ledger);
    let mut shared_lineage = Vec::new();
    let mut shared_lineage_task_pair_count = 0_u64;

    for ((kind, digest), left_ids) in &left_roots {
        let Some(right_ids) = right_roots.get(&(kind.clone(), digest.clone())) else {
            continue;
        };
        let pairs = checked_pairs(left_ids.len(), right_ids.len())?;
        shared_lineage_task_pair_count = shared_lineage_task_pair_count
            .checked_add(pairs)
            .ok_or_else(|| "total shared lineage task-pair count overflows u64".to_owned())?;
        shared_lineage.push(SharedLineageGroupV1 {
            kind: kind.clone(),
            commitment_sha256: digest.clone(),
            left_task_count: u64::try_from(left_ids.len())
                .map_err(|_| "left task count overflow")?,
            right_task_count: u64::try_from(right_ids.len())
                .map_err(|_| "right task count overflow")?,
            cross_split_task_pair_count: pairs,
            left_task_ids_sample: bounded_sample(left_ids),
            right_task_ids_sample: bounded_sample(right_ids),
        });
    }

    let left_exposed_refresh_tasks = exposed_refresh_tasks(&left_ledger);
    let right_exposed_refresh_tasks = exposed_refresh_tasks(&right_ledger);
    let shared_lineage_root_count = u64::try_from(shared_lineage.len())
        .map_err(|_| "shared lineage root count overflow")?;
    let left_exposed_refresh_task_count = u64::try_from(left_exposed_refresh_tasks.len())
        .map_err(|_| "left exposed refresh task count overflow")?;
    let right_exposed_refresh_task_count = u64::try_from(right_exposed_refresh_tasks.len())
        .map_err(|_| "right exposed refresh task count overflow")?;

    let has_shared_lineage = shared_lineage_root_count > 0;
    let has_exposed_refresh =
        left_exposed_refresh_task_count > 0 || right_exposed_refresh_task_count > 0;
    let admission = match (has_shared_lineage, has_exposed_refresh) {
        (true, true) => "blocked_shared_lineage_and_exposed_refresh_tasks",
        (true, false) => "blocked_shared_cross_split_lineage",
        (false, true) => "blocked_exposed_refresh_tasks",
        (false, false) => "no_detected_shared_lineage_or_exposed_refresh",
    };

    Ok(LineageAuditReportV1 {
        schema: REPORT_SCHEMA.to_owned(),
        valid: true,
        left_split_label: left_ledger.split_label.clone(),
        right_split_label: right_ledger.split_label.clone(),
        left_generation_id: left_set.generation_id,
        right_generation_id: right_set.generation_id,
        left_task_set_sha256: sha256_hex(left_task_set_bytes),
        right_task_set_sha256: sha256_hex(right_task_set_bytes),
        left_ledger_sha256: sha256_hex(left_ledger_bytes),
        right_ledger_sha256: sha256_hex(right_ledger_bytes),
        left_task_count: u64::try_from(left_set.tasks.len())
            .map_err(|_| "left task count overflow")?,
        right_task_count: u64::try_from(right_set.tasks.len())
            .map_err(|_| "right task count overflow")?,
        shared_lineage_root_count,
        shared_lineage_task_pair_count,
        shared_lineage,
        left_exposed_refresh_task_count,
        left_exposed_refresh_task_ids_sample: bounded_sample(&left_exposed_refresh_tasks),
        right_exposed_refresh_task_count,
        right_exposed_refresh_task_ids_sample: bounded_sample(&right_exposed_refresh_tasks),
        admission: admission.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leakage_audit::canonical_policy_input_sha256;
    use serde_json::{Value, json};

    fn task_set(generation: &str, task_id: &str, x: i64, y: i64) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": TASK_SET_SCHEMA,
            "generation_id": generation,
            "tasks": [{
                "task_id": task_id,
                "family_id": "shared-schema-is-not-lineage",
                "policy_input": {
                    "instruction": "Recover relationship",
                    "observations": [{"x": x, "y": y}]
                },
                "evaluator_only": {"expected_law":"must-not-appear-in-report"}
            }]
        }))
        .expect("task set serializes")
    }

    fn ledger_for(
        task_set_bytes: &[u8],
        split_label: &str,
        lineage_digest: &str,
        exposure_state: &str,
    ) -> Vec<u8> {
        let set: TaskSetArtifactV1 = serde_json::from_slice(task_set_bytes).unwrap();
        let policy_sha = canonical_policy_input_sha256(&set.tasks[0].policy_input).unwrap();
        serde_json::to_vec(&json!({
            "schema": LEDGER_SCHEMA,
            "generation_id": set.generation_id,
            "split_label": split_label,
            "task_set_sha256": sha256_hex(task_set_bytes),
            "records": [{
                "task_id": set.tasks[0].task_id,
                "policy_input_sha256": policy_sha,
                "exposure_state": exposure_state,
                "lineage_commitments": [{
                    "kind": "parent_world",
                    "commitment_sha256": lineage_digest
                }]
            }]
        })).unwrap()
    }

    #[test]
    fn shared_parent_world_blocks_even_when_inputs_and_ids_differ() {
        let left_set = task_set("dev-g1", "dev-1", 1, 2);
        let right_set = task_set("refresh-g2", "refresh-9", 7, 14);
        let root = sha256_hex(b"same-latent-world");
        let left_ledger = ledger_for(&left_set, "development:g1", &root, "public_development");
        let right_ledger = ledger_for(&right_set, "refresh:g2", &root, "never_exposed");
        let report = audit_lineage_overlap(
            &left_set, &left_ledger, &right_set, &right_ledger
        ).unwrap();

        assert_eq!(report.shared_lineage_root_count, 1);
        assert_eq!(report.shared_lineage_task_pair_count, 1);
        assert_eq!(report.admission, "blocked_shared_cross_split_lineage");
    }

    #[test]
    fn shared_schema_and_family_do_not_block_when_lineage_is_disjoint() {
        let left_set = task_set("dev-g1", "dev-1", 1, 2);
        let right_set = task_set("refresh-g2", "refresh-9", 7, 14);
        let left_ledger = ledger_for(
            &left_set, "development:g1", &sha256_hex(b"world-a"), "public_development"
        );
        let right_ledger = ledger_for(
            &right_set, "refresh:g2", &sha256_hex(b"world-b"), "never_exposed"
        );
        let report = audit_lineage_overlap(
            &left_set, &left_ledger, &right_set, &right_ledger
        ).unwrap();

        assert_eq!(report.shared_lineage_root_count, 0);
        assert_eq!(
            report.admission,
            "no_detected_shared_lineage_or_exposed_refresh"
        );
    }

    #[test]
    fn exposed_refresh_task_blocks_admission_even_without_shared_lineage() {
        let left_set = task_set("dev-g1", "dev-1", 1, 2);
        let right_set = task_set("refresh-g2", "refresh-9", 7, 14);
        let left_ledger = ledger_for(
            &left_set, "development:g1", &sha256_hex(b"world-a"), "public_development"
        );
        let right_ledger = ledger_for(
            &right_set, "refresh:g2", &sha256_hex(b"world-b"), "prior_scored_refresh"
        );
        let report = audit_lineage_overlap(
            &left_set, &left_ledger, &right_set, &right_ledger
        ).unwrap();

        assert_eq!(report.right_exposed_refresh_task_count, 1);
        assert_eq!(report.admission, "blocked_exposed_refresh_tasks");
    }

    #[test]
    fn task_set_mutation_or_missing_lineage_record_fails_closed() {
        let set = task_set("refresh-g2", "refresh-9", 7, 14);
        let mut parsed: Value = serde_json::from_slice(&set).unwrap();
        let ledger = ledger_for(
            &set, "refresh:g2", &sha256_hex(b"world-b"), "never_exposed"
        );
        parsed["tasks"][0]["policy_input"]["observations"][0]["y"] = json!(99);
        let changed_set = serde_json::to_vec(&parsed).unwrap();
        assert!(audit_lineage_overlap(
            &changed_set, &ledger, &task_set("dev-g1", "dev-1", 1, 2),
            &ledger_for(&task_set("dev-g1", "dev-1", 1, 2), "development:g1", &sha256_hex(b"world-a"), "public_development")
        ).is_err());

        let mut broken: Value = serde_json::from_slice(&ledger).unwrap();
        broken["records"] = json!([]);
        let broken_ledger = serde_json::to_vec(&broken).unwrap();
        assert!(audit_lineage_overlap(
            &set, &broken_ledger, &task_set("dev-g1", "dev-1", 1, 2),
            &ledger_for(&task_set("dev-g1", "dev-1", 1, 2), "development:g1", &sha256_hex(b"world-a"), "public_development")
        ).is_err());
    }

    #[test]
    fn report_is_deterministic_and_does_not_emit_evaluator_targets() {
        let left_set = task_set("dev-g1", "dev-1", 1, 2);
        let right_set = task_set("refresh-g2", "refresh-9", 7, 14);
        let left_ledger = ledger_for(
            &left_set, "development:g1", &sha256_hex(b"world-a"), "public_development"
        );
        let right_ledger = ledger_for(
            &right_set, "refresh:g2", &sha256_hex(b"world-b"), "never_exposed"
        );
        let first = audit_lineage_overlap(
            &left_set, &left_ledger, &right_set, &right_ledger
        ).unwrap();
        let second = audit_lineage_overlap(
            &left_set, &left_ledger, &right_set, &right_ledger
        ).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert!(
            !serde_json::to_string(&first)
                .unwrap()
                .contains("must-not-appear-in-report")
        );
    }
}
