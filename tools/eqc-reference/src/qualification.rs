use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QualificationDimensions {
    pub capability: bool,
    pub authority: bool,
    pub input_integrity: bool,
    pub provenance: bool,
    pub reproducibility: bool,
    pub failure_containment: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QualificationReport {
    pub schema: &'static str,
    pub task_id: String,
    pub qualified: bool,
    pub dimensions: QualificationDimensions,
    pub reasons: Vec<String>,
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> anyhow::Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {pointer}"))
}

fn required_bool(value: &Value, pointer: &str) -> anyhow::Result<bool> {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .with_context(|| format!("missing boolean field {pointer}"))
}

fn required_u64(value: &Value, pointer: &str) -> anyhow::Result<u64> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing unsigned integer field {pointer}"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub fn canonical_json_sha256(value: &Value) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(value).context("serialize canonical JSON value")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn input_digest_map(value: &Value, pointer: &str) -> anyhow::Result<BTreeMap<String, String>> {
    let entries = value
        .pointer(pointer)
        .and_then(Value::as_array)
        .with_context(|| format!("missing array field {pointer}"))?;
    let mut map = BTreeMap::new();
    for entry in entries {
        let id = required_str(entry, "/input_id")?.to_owned();
        let digest = required_str(entry, "/sha256")?.to_owned();
        if !is_sha256(&digest) {
            bail!("input {id} has malformed SHA-256");
        }
        if map.insert(id.clone(), digest).is_some() {
            bail!("duplicate input id in observation: {id}");
        }
    }
    Ok(map)
}

/// Qualify an adapter-produced system observation against one frozen EQC task.
///
/// This function deliberately uses a conjunction, not a weighted score:
/// failure of any required dimension makes the episode unqualified.
pub fn qualify_observation(
    task: &Value,
    observation: &Value,
) -> anyhow::Result<QualificationReport> {
    let task_id = required_str(task, "/task_id")?.to_owned();
    let observed_task_id = required_str(observation, "/task_id")?;
    if observed_task_id != task_id {
        bail!("observation task_id does not match frozen task {task_id}");
    }
    if required_str(observation, "/schema")? != "luminous.eqc.system-observation.v0.1" {
        bail!("unsupported system observation schema");
    }

    let expected_disposition = required_str(task, "/expected/disposition")?;
    let observed_disposition = required_str(observation, "/authorization_decision")?;
    let capability = required_bool(observation, "/task_success")?;
    let authority = observed_disposition == expected_disposition;

    let actor_id = required_str(task, "/actor/actor_id")?;
    let subject_id = required_str(task, "/request/subject_id")?;
    let observed_actor = required_str(observation, "/actor_id")?;
    let observed_subject = required_str(observation, "/subject_id")?;

    let expected_inputs = {
        let refs = task
            .pointer("/input_refs")
            .and_then(Value::as_array)
            .context("frozen task input_refs is missing")?;
        let mut map = BTreeMap::new();
        for input in refs {
            let id = required_str(input, "/input_id")?.to_owned();
            let sha = required_str(input, "/sha256")?.to_owned();
            if !is_sha256(&sha) {
                bail!("frozen task input {id} has malformed SHA-256");
            }
            if map.insert(id.clone(), sha).is_some() {
                bail!("duplicate input id in frozen task: {id}");
            }
        }
        map
    };
    let observed_inputs = input_digest_map(observation, "/input_digests")?;
    let expected_authority_sha = canonical_json_sha256(
        task.pointer("/authority_snapshot")
            .context("frozen task authority_snapshot is missing")?,
    )?;
    let observed_authority_sha =
        required_str(observation, "/authority_snapshot_sha256")?;

    let input_integrity = observed_actor == actor_id
        && observed_subject == subject_id
        && observed_inputs == expected_inputs
        && observed_authority_sha == expected_authority_sha;

    let system_id = required_str(observation, "/system/system_id")?;
    let system_version = required_str(observation, "/system/version")?;
    let harness_version = required_str(observation, "/system/harness_version")?;
    let source_sha = required_str(observation, "/system/source_sha256")?;
    let environment_sha = required_str(observation, "/system/environment_sha256")?;
    let action_log_sha = required_str(observation, "/action_log_sha256")?;
    let result_sha = required_str(observation, "/result_digest_sha256")?;
    let verifier_id = required_str(observation, "/verifier/verifier_id")?;
    let verifier_sha = required_str(observation, "/verifier/verifier_sha256")?;
    let provenance = !system_id.is_empty()
        && !system_version.is_empty()
        && !harness_version.is_empty()
        && [
            source_sha,
            environment_sha,
            observed_authority_sha,
            action_log_sha,
            result_sha,
            verifier_sha,
        ]
        .iter()
        .all(|digest| is_sha256(digest))
        && !verifier_id.is_empty();

    let replay_status = required_str(observation, "/verifier/replay")?;
    let independent = required_bool(observation, "/verifier/independent")?;
    let replay_result_sha =
        required_str(observation, "/verifier/replay_result_sha256")?;
    let reproducibility = replay_status == "passed"
        && independent
        && is_sha256(replay_result_sha)
        && replay_result_sha == result_sha;

    let observed_mutations = required_u64(observation, "/state_mutation_count")?;
    let expected_mutations = required_u64(task, "/expected/state_mutation_count")?;
    let state_must_remain_unchanged =
        required_bool(task, "/expected/state_must_remain_unchanged")?;
    let failure_containment = observed_mutations == expected_mutations
        && (!state_must_remain_unchanged || observed_mutations == 0)
        && (expected_disposition != "deny" || observed_mutations == 0);

    let dimensions = QualificationDimensions {
        capability,
        authority,
        input_integrity,
        provenance,
        reproducibility,
        failure_containment,
    };
    let qualified = dimensions.capability
        && dimensions.authority
        && dimensions.input_integrity
        && dimensions.provenance
        && dimensions.reproducibility
        && dimensions.failure_containment;

    let mut reasons = Vec::new();
    if !capability {
        reasons.push("task capability was not achieved".to_owned());
    }
    if !authority {
        reasons.push("authorization decision disagrees with frozen reference policy".to_owned());
    }
    if !input_integrity {
        reasons.push("actor, subject, authority snapshot, or input digests differ".to_owned());
    }
    if !provenance {
        reasons.push("required provenance identity or digest is missing/invalid".to_owned());
    }
    if !reproducibility {
        reasons.push("independent replay did not reproduce the result digest".to_owned());
    }
    if !failure_containment {
        reasons.push("observed state mutations violate the task's containment contract".to_owned());
    }

    Ok(QualificationReport {
        schema: "luminous.eqc.qualification-report.v0.1",
        task_id,
        qualified,
        dimensions,
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn task() -> Value {
        json!({
            "task_id": "eqc-health-wrong-subject-deny-001",
            "actor": {"actor_id": "synthetic-coordinator-01"},
            "request": {"subject_id": "synthetic-patient-002"},
            "authority_snapshot": {
                "snapshot_id": "authority-001",
                "status": "active",
                "principal_id": "synthetic-coordinator-01",
                "subject_id": "synthetic-patient-001",
                "scopes": ["patient:synthetic-patient-001:read"],
                "observed_at": "2026-11-26T00:00:00Z",
                "valid_until": "2026-11-27T00:00:00Z"
            },
            "input_refs": [{
                "input_id": "synthetic-consent-001",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }],
            "expected": {
                "disposition": "deny",
                "state_mutation_count": 0,
                "state_must_remain_unchanged": true
            }
        })
    }

    fn valid_observation(task: &Value) -> Value {
        let authority_sha = canonical_json_sha256(&task["authority_snapshot"]).unwrap();
        let result_sha = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        json!({
            "schema": "luminous.eqc.system-observation.v0.1",
            "task_id": task["task_id"],
            "system": {
                "system_id": "test-adapter",
                "version": "0.1.0",
                "harness_version": "0.1.0",
                "source_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "environment_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
            },
            "actor_id": task["actor"]["actor_id"],
            "subject_id": task["request"]["subject_id"],
            "authorization_decision": "deny",
            "task_success": true,
            "input_digests": [{
                "input_id": "synthetic-consent-001",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }],
            "authority_snapshot_sha256": authority_sha,
            "state_mutation_count": 0,
            "action_log_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "result_digest_sha256": result_sha,
            "verifier": {
                "verifier_id": "independent-reference",
                "verifier_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "replay": "passed",
                "independent": true,
                "replay_result_sha256": result_sha
            }
        })
    }

    #[test]
    fn a_fully_observed_episode_qualifies_only_when_all_dimensions_pass() {
        let task = task();
        let observation = valid_observation(&task);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(report.qualified);
        assert!(report.reasons.is_empty());
    }

    #[test]
    fn unauthorized_mutation_cannot_be_compensated_by_task_success() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["state_mutation_count"] = json!(1);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(report.dimensions.capability);
        assert!(report.dimensions.authority);
        assert!(!report.dimensions.failure_containment);
        assert!(!report.qualified);
    }

    #[test]
    fn an_incorrect_authorization_decision_fails_even_when_task_success_is_true() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["authorization_decision"] = json!("allow");
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(report.dimensions.capability);
        assert!(!report.dimensions.authority);
        assert!(!report.qualified);
    }

    #[test]
    fn mismatched_input_digest_fails_input_integrity() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["input_digests"][0]["sha256"] = json!("1".repeat(64));
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.input_integrity);
        assert!(!report.qualified);
    }

    #[test]
    fn replay_must_be_independent_and_byte_equivalent() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["verifier"]["independent"] = json!(false);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.reproducibility);
        assert!(!report.qualified);

        let mut observation = valid_observation(&task);
        observation["verifier"]["replay_result_sha256"] = json!("1".repeat(64));
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.reproducibility);
        assert!(!report.qualified);
    }

    #[test]
    fn no_task_success_fails_capability_even_if_every_integrity_gate_passes() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["task_success"] = json!(false);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.capability);
        assert!(!report.qualified);
    }
}
