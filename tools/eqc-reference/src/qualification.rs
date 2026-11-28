use anyhow::{bail, Context};
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
    pub replay_claim_consistency: bool,
    pub failure_containment: bool,
    pub budget_compliance: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QualificationReport {
    pub schema: &'static str,
    pub task_id: String,
    pub contract_checks_pass: bool,
    pub independent_evidence_verified: bool,
    pub qualification_status: &'static str,
    pub qualified: bool,
    pub dimensions: QualificationDimensions,
    pub resource_usage: Value,
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

pub fn validate_observation_schema(
    observation: &Value,
    repository_root: &Path,
) -> anyhow::Result<()> {
    let schema_path = repository_root.join("docs/research/eqc-system-observation-v0.1.schema.json");
    let schema_bytes = std::fs::read(&schema_path)
        .with_context(|| format!("read system observation schema {}", schema_path.display()))?;
    let schema_value: Value = serde_json::from_slice(&schema_bytes)
        .with_context(|| format!("parse system observation schema {}", schema_path.display()))?;
    let validator = jsonschema::options()
        .offline()
        .should_validate_formats(true)
        .build(&schema_value)
        .with_context(|| {
            format!(
                "compile offline observation schema {}",
                schema_path.display()
            )
        })?;
    let errors = validator
        .iter_errors(observation)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        bail!(
            "system observation schema validation failed: {}",
            errors.join("; ")
        );
    }
    Ok(())
}

/// Qualify an adapter-produced system observation against one frozen EQC task.
///
/// This function deliberately uses a conjunction, not a weighted score:
/// failure of any required dimension makes the episode unqualified.
fn check_budget_compliance(task: &Value, observation: &Value) -> anyhow::Result<bool> {
    let budget = task
        .pointer("/resource_budget")
        .context("frozen task resource_budget is missing")?;
    let usage = observation
        .pointer("/resource_usage")
        .context("system observation resource_usage is missing")?;

    let timeout_seconds = required_u64(budget, "/timeout_seconds")?;
    let max_tool_calls = required_u64(budget, "/max_tool_calls")?;
    let elapsed_ms = required_u64(usage, "/elapsed_ms")?;
    let cpu_time_ms = required_u64(usage, "/cpu_time_ms")?;
    let peak_memory_bytes = required_u64(usage, "/peak_memory_bytes")?;
    let tool_calls = required_u64(usage, "/tool_calls")?;
    let network_bytes = required_u64(usage, "/network_bytes")?;
    let network_endpoints = usage
        .pointer("/network_endpoints")
        .and_then(Value::as_array)
        .context("resource_usage.network_endpoints is missing")?;

    let Some(timeout_ms) = timeout_seconds.checked_mul(1_000) else {
        return Ok(false);
    };
    if elapsed_ms > timeout_ms || tool_calls > max_tool_calls {
        return Ok(false);
    }

    if let Some(cpu_seconds) = budget.get("cpu_seconds").and_then(Value::as_u64) {
        let Some(cpu_limit_ms) = cpu_seconds.checked_mul(1_000) else {
            return Ok(false);
        };
        if cpu_time_ms > cpu_limit_ms {
            return Ok(false);
        }
    }

    if let Some(memory_mib) = budget.get("memory_mib").and_then(Value::as_u64) {
        let Some(memory_limit_bytes) = memory_mib.checked_mul(1024 * 1024) else {
            return Ok(false);
        };
        if peak_memory_bytes > memory_limit_bytes {
            return Ok(false);
        }
    }

    // v0.1 public tasks all use a closed network policy. Other policy modes
    // require an explicit endpoint allow-list and are intentionally not
    // accepted by this first evaluator.
    match required_str(budget, "/network_policy")? {
        "none" => Ok(network_bytes == 0 && network_endpoints.is_empty()),
        "declared_endpoints_only" | "open" => Ok(false),
        _ => Ok(false),
    }
}

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
    let observed_authority_sha = required_str(observation, "/authority_snapshot_sha256")?;

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
    let replay_result_sha = required_str(observation, "/verifier/replay_result_sha256")?;
    // These values are adapter claims. We can check internal consistency,
    // but the adapter's independent=true field is not proof that another
    // process actually replayed the result.
    let replay_claim_consistency = replay_status == "passed"
        && independent
        && is_sha256(replay_result_sha)
        && replay_result_sha == result_sha;

    let observed_mutations = required_u64(observation, "/state_mutation_count")?;
    let expected_mutations = required_u64(task, "/expected/state_mutation_count")?;
    let state_must_remain_unchanged = required_bool(task, "/expected/state_must_remain_unchanged")?;
    let failure_containment = observed_mutations == expected_mutations
        && (!state_must_remain_unchanged || observed_mutations == 0)
        && (expected_disposition != "deny" || observed_mutations == 0);
    let budget_compliance = check_budget_compliance(task, observation)?;
    let resource_usage = observation
        .get("resource_usage")
        .cloned()
        .context("system observation resource_usage is missing")?;

    let dimensions = QualificationDimensions {
        capability,
        authority,
        input_integrity,
        provenance,
        replay_claim_consistency,
        failure_containment,
        budget_compliance,
    };
    // v0.1 has no trusted observer or resource-meter attestation verifier.
    // Therefore no adapter-submitted observation can independently qualify,
    // even when every self-reported contract dimension is internally valid.
    let contract_checks_pass = dimensions.capability
        && dimensions.authority
        && dimensions.input_integrity
        && dimensions.provenance
        && dimensions.replay_claim_consistency
        && dimensions.failure_containment
        && dimensions.budget_compliance;
    let independent_evidence_verified = false;
    let qualification_status = if contract_checks_pass {
        "unqualified_untrusted_adapter_observation"
    } else {
        "unqualified_contract_checks_failed"
    };
    let qualified = contract_checks_pass && independent_evidence_verified;

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
    if !replay_claim_consistency {
        reasons
            .push("adapter-declared replay failed or its result digest did not match".to_owned());
    }
    if !failure_containment {
        reasons.push("observed state mutations violate the task's containment contract".to_owned());
    }
    if !budget_compliance {
        reasons.push("adapter-reported elapsed time, CPU, memory, tool, or network usage exceeds the task budget".to_owned());
    }
    if !independent_evidence_verified {
        reasons.push(
            "no trusted observer attestation verifies resource metering, state observations, or independent replay in EQC v0.1".to_owned(),
        );
    }

    Ok(QualificationReport {
        schema: "luminous.eqc.qualification-report.v0.1",
        task_id,
        contract_checks_pass,
        independent_evidence_verified,
        qualification_status,
        qualified,
        dimensions,
        resource_usage,
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
            "resource_budget": {
                "timeout_seconds": 10,
                "max_tool_calls": 3,
                "network_policy": "none",
                "cpu_seconds": 5,
                "memory_mib": 128
            },
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
            "resource_usage": {
                "elapsed_ms": 250,
                "cpu_time_ms": 200,
                "peak_memory_bytes": 16_777_216,
                "tool_calls": 2,
                "network_bytes": 0,
                "network_endpoints": [],
                "human_interventions": 0,
                "energy_millijoules": null,
                "cost_microusd": null
            },
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
    fn system_observation_schema_meta_validates_and_accepts_the_adapter_contract() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root resolves");
        let schema_path = root.join("docs/research/eqc-system-observation-v0.1.schema.json");
        let schema: Value =
            serde_json::from_slice(&std::fs::read(schema_path).expect("read observation schema"))
                .expect("observation schema parses");
        let validator = jsonschema::options()
            .offline()
            .should_validate_formats(true)
            .build(&schema)
            .expect("Draft 2020-12 observation schema meta-validates");
        let task = task();
        let observation = valid_observation(&task);
        assert!(validator.is_valid(&observation));
        validate_observation_schema(&observation, &root)
            .expect("observation validates against the adapter schema");
    }

    #[test]
    fn observation_schema_rejects_unknown_fields_and_missing_digests() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root resolves");
        let task = task();
        let mut observation = valid_observation(&task);
        observation["unreviewed_override"] = json!(true);
        assert!(validate_observation_schema(&observation, &root).is_err());

        let mut observation = valid_observation(&task);
        observation["action_log_sha256"] = Value::Null;
        assert!(validate_observation_schema(&observation, &root).is_err());
    }

    #[test]
    fn self_reported_observation_passes_contract_checks_but_is_not_independently_qualified() {
        let task = task();
        let observation = valid_observation(&task);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(report.contract_checks_pass);
        assert!(!report.independent_evidence_verified);
        assert_eq!(
            report.qualification_status,
            "unqualified_untrusted_adapter_observation"
        );
        assert!(!report.qualified);
        assert!(report.dimensions.budget_compliance);
        assert_eq!(report.resource_usage["human_interventions"], 0);
        assert!(report.resource_usage["energy_millijoules"].is_null());
        assert_eq!(report.reasons.len(), 1);
        assert!(report.reasons[0].contains("no trusted observer attestation"));
    }

    #[test]
    fn adapter_cannot_self_attest_independence_to_qualify() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["verifier"]["independent"] = json!(true);
        observation["verifier"]["replay"] = json!("passed");
        let report = qualify_observation(&task, &observation).unwrap();

        assert!(report.contract_checks_pass);
        assert!(!report.independent_evidence_verified);
        assert!(!report.qualified);
        assert_eq!(
            report.qualification_status,
            "unqualified_untrusted_adapter_observation"
        );
    }

    #[test]
    fn over_budget_time_or_tool_calls_prevents_qualification() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["resource_usage"]["elapsed_ms"] = json!(10_001);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(report.dimensions.capability);
        assert!(!report.dimensions.budget_compliance);
        assert!(!report.qualified);

        let mut observation = valid_observation(&task);
        observation["resource_usage"]["tool_calls"] = json!(4);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.budget_compliance);
        assert!(!report.qualified);
    }

    #[test]
    fn resource_budget_enforces_cpu_and_memory_limits() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["resource_usage"]["cpu_time_ms"] = json!(5_001);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.budget_compliance);

        let mut observation = valid_observation(&task);
        observation["resource_usage"]["peak_memory_bytes"] = json!(128_u64 * 1024 * 1024 + 1);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.budget_compliance);
    }

    #[test]
    fn closed_network_budget_rejects_network_activity() {
        let task = task();
        let mut observation = valid_observation(&task);
        observation["resource_usage"]["network_bytes"] = json!(1);
        observation["resource_usage"]["network_endpoints"] = json!(["https://example.invalid"]);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.budget_compliance);
        assert!(!report.qualified);
    }

    #[test]
    fn unsupported_network_budget_modes_fail_closed() {
        let mut task = task();
        task["resource_budget"]["network_policy"] = json!("declared_endpoints_only");
        let observation = valid_observation(&task);
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.budget_compliance);
        assert!(!report.qualified);
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
        assert!(!report.dimensions.replay_claim_consistency);
        assert!(!report.qualified);

        let mut observation = valid_observation(&task);
        observation["verifier"]["replay_result_sha256"] = json!("1".repeat(64));
        let report = qualify_observation(&task, &observation).unwrap();
        assert!(!report.dimensions.replay_claim_consistency);
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
