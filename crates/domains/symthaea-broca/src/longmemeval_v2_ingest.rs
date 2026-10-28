// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic LongMemEval-V2 trajectory-to-evidence ingestion.
//!
//! This is an ingestion primitive, not the complete official benchmark worker.
//! It deliberately accepts trajectory records only: question records, gold
//! answers, evaluator instructions and question IDs are rejected. Each emitted
//! record is tied to a canonical per-state source digest and a stable opaque ID.
//! A trusted caller must map (domain, environment) to an authorized scope and
//! must use a separately pinned encoder if it also populates the HDC vector
//! store. This module does not claim that its text payload is a semantic HDC
//! encoding.

use super::memory_evidence::{
    EVIDENCE_SCHEMA_VERSION, EvidenceLifecycle, EvidencePayload, EvidenceRecord, EvidenceSource,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Canonicalization contract. Changing this requires a new version and fixture.
pub const LME_V2_INGEST_SCHEMA_VERSION: &str = "BROCA-LME-V2-TRAJECTORY-INGEST-V1";

const FORBIDDEN_METADATA_KEYS: &[&str] = &[
    "question",
    "questionid",
    "questiontype",
    "benchmarkquestionid",
    "answer",
    "goldanswer",
    "referenceanswer",
    "correctanswer",
    "expectedanswer",
    "answerkey",
    "groundtruth",
    "evalfunction",
    "evaluationfunction",
    "evaluator",
];

/// Official LongMemEval-V2 trajectory JSONL record.
///
/// Unknown fields are rejected so a schema expansion cannot silently affect the
/// memory write path. In particular, question/answer/evaluator fields belong to
/// the question file, not to trajectory ingestion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongMemEvalTrajectory {
    pub id: String,
    pub domain: String,
    pub environment: String,
    pub goal: String,
    pub outcome: String,
    pub start_url: Option<String>,
    pub states: Vec<LongMemEvalState>,
}

/// One ordered state/action observation from a trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongMemEvalState {
    pub state_index: u64,
    pub step: Option<Value>,
    pub url: Option<String>,
    pub action: Option<Value>,
    pub thought: Option<String>,
    pub accessibility_tree: Option<String>,
    pub screenshot: Option<String>,
}

/// Parse, validate and ingest JSONL trajectory records into deterministic text
/// evidence. The resolver must be backed by trusted configuration; source
/// metadata is a lookup key, never authority to grant its own scope.
///
/// The whole batch fails on an invalid line, unknown schema, duplicate evidence
/// identity or 64-bit HDC-ID collision. No partial batch is returned.
pub fn ingest_longmemeval_v2_jsonl<I, S, F>(
    lines: I,
    mut resolve_scope: F,
) -> Result<Vec<EvidenceRecord>, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<[u8]>,
    F: FnMut(&str, &str) -> Option<String>,
{
    let mut records = Vec::new();
    let mut evidence_ids = BTreeSet::new();
    let mut hdc_ids: BTreeMap<u64, String> = BTreeMap::new();

    for (line_idx, line) in lines.into_iter().enumerate() {
        let raw = line.as_ref();
        if raw.iter().all(u8::is_ascii_whitespace) {
            return Err(format!("trajectory JSONL line {} is blank", line_idx + 1));
        }

        let value: Value = serde_json::from_slice(raw)
            .map_err(|error| format!("trajectory JSONL line {} is invalid JSON: {error}", line_idx + 1))?;
        reject_question_metadata(&value)
            .map_err(|error| format!("trajectory JSONL line {} rejected: {error}", line_idx + 1))?;

        let trajectory: LongMemEvalTrajectory = serde_json::from_value(value)
            .map_err(|error| format!("trajectory JSONL line {} violates the pinned schema: {error}", line_idx + 1))?;
        validate_trajectory(&trajectory)
            .map_err(|error| format!("trajectory JSONL line {} rejected: {error}", line_idx + 1))?;

        let scope_id = resolve_scope(&trajectory.domain, &trajectory.environment)
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| {
                format!(
                    "no trusted scope mapping for domain {:?}, environment {:?}",
                    trajectory.domain, trajectory.environment
                )
            })?;

        for state in &trajectory.states {
            let source_bytes = canonical_source_bytes(&trajectory, state)?;
            let source_sha256 = sha256_hex(&source_bytes);
            let identity_bytes = canonical_identity_bytes(&trajectory, state)?;
            let identity_digest = sha256_hex(&identity_bytes);
            let evidence_id = format!("lme-v2-state:{identity_digest}");
            let hdc_id = u64::from_str_radix(&identity_digest[..16], 16)
                .map_err(|error| format!("invalid stable-ID digest: {error}"))?;

            if !evidence_ids.insert(evidence_id.clone()) {
                return Err(format!("duplicate trajectory-state identity {evidence_id}"));
            }
            if let Some(previous) = hdc_ids.insert(hdc_id, evidence_id.clone()) {
                if previous != evidence_id {
                    return Err(format!(
                        "stable HDC-ID collision between {previous} and {evidence_id}; refusing ingestion"
                    ));
                }
            }

            let text = render_state_evidence(&trajectory, state);
            let content_sha256 = sha256_hex(text.as_bytes());
            let record = EvidenceRecord {
                schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
                evidence_id,
                hdc_id,
                scope_id: scope_id.clone(),
                source: EvidenceSource {
                    trajectory_id: trajectory.id.clone(),
                    step_id: Some(format!("state:{}", state.state_index)),
                    step_index: Some(state.state_index),
                    source_sha256,
                },
                payload: EvidencePayload::Text { value: text },
                content_sha256,
                lifecycle: EvidenceLifecycle::Active,
                occurred_at_unix_ms: None,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                superseded_by: None,
            };
            record.validate().map_err(|error| format!("invalid emitted evidence: {error}"))?;
            records.push(record);
        }
    }

    Ok(records)
}

fn validate_trajectory(trajectory: &LongMemEvalTrajectory) -> Result<(), String> {
    if trajectory.id.trim().is_empty() {
        return Err("trajectory id must be non-empty".into());
    }
    if !matches!(trajectory.domain.as_str(), "web" | "enterprise") {
        return Err(format!("unsupported LongMemEval-V2 domain {:?}", trajectory.domain));
    }
    if trajectory.environment.trim().is_empty() {
        return Err("environment must be non-empty".into());
    }
    if trajectory.goal.trim().is_empty() {
        return Err("trajectory goal must be non-empty".into());
    }
    if !matches!(trajectory.outcome.as_str(), "success" | "failure") {
        return Err(format!("unsupported trajectory outcome {:?}", trajectory.outcome));
    }
    if trajectory.states.is_empty() {
        return Err("trajectory must contain at least one state".into());
    }

    let mut previous_index = None;
    for state in &trajectory.states {
        if previous_index.is_some_and(|previous| state.state_index <= previous) {
            return Err("state_index values must be strictly increasing and unique".into());
        }
        previous_index = Some(state.state_index);
    }
    Ok(())
}

/// Stable canonical bytes for a state and the trajectory metadata that gives it
/// meaning. This function is public so an audit/replay tool can independently
/// recompute EvidenceSource.source_sha256.
pub fn canonical_source_bytes(
    trajectory: &LongMemEvalTrajectory,
    state: &LongMemEvalState,
) -> Result<Vec<u8>, String> {
    let envelope = serde_json::json!({
        "schema_version": LME_V2_INGEST_SCHEMA_VERSION,
        "trajectory": trajectory,
        "state": state,
    });
    let canonical = canonicalize_value(envelope);
    serde_json::to_vec(&canonical)
        .map_err(|error| format!("failed to canonicalize trajectory state source: {error}"))
}

fn canonical_identity_bytes(
    trajectory: &LongMemEvalTrajectory,
    state: &LongMemEvalState,
) -> Result<Vec<u8>, String> {
    let identity = serde_json::json!({
        "schema_version": LME_V2_INGEST_SCHEMA_VERSION,
        "domain": trajectory.domain,
        "environment": trajectory.environment,
        "trajectory_id": trajectory.id,
        "state_index": state.state_index,
    });
    serde_json::to_vec(&canonicalize_value(identity))
        .map_err(|error| format!("failed to canonicalize trajectory-state identity: {error}"))
}

fn canonicalize_value(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalize_value).collect()),
        Value::Object(object) => {
            let mut entries: Vec<(String, Value)> = object.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key, canonicalize_value(value));
            }
            Value::Object(sorted)
        }
        scalar => scalar,
    }
}

fn reject_question_metadata(value: &Value) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            for item in items {
                reject_question_metadata(item)?;
            }
        }
        Value::Object(object) => {
            for (key, nested) in object {
                let normalized: String = key
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect();
                if FORBIDDEN_METADATA_KEYS.contains(&normalized.as_str()) {
                    return Err(format!("question/evaluation metadata key {key:?} is forbidden"));
                }
                reject_question_metadata(nested)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn render_state_evidence(
    trajectory: &LongMemEvalTrajectory,
    state: &LongMemEvalState,
) -> String {
    let mut output = String::new();
    output.push_str("Trajectory goal: ");
    output.push_str(&trajectory.goal);
    output.push_str("\nOutcome: ");
    output.push_str(&trajectory.outcome);
    output.push_str("\nDomain: ");
    output.push_str(&trajectory.domain);
    output.push_str("\nEnvironment: ");
    output.push_str(&trajectory.environment);

    if let Some(start_url) = &trajectory.start_url {
        output.push_str("\nStart URL: ");
        output.push_str(start_url);
    }
    output.push_str(&format!("\nState index: {}", state.state_index));

    if let Some(step) = &state.step {
        output.push_str("\nOriginal step: ");
        output.push_str(&canonicalize_value(step.clone()).to_string());
    }
    if let Some(url) = &state.url {
        output.push_str("\nURL: ");
        output.push_str(url);
    }
    if let Some(action) = &state.action {
        if !action.is_null() {
            output.push_str("\nAction: ");
            output.push_str(&canonicalize_value(action.clone()).to_string());
        }
    }
    if let Some(thought) = &state.thought {
        if !thought.trim().is_empty() {
            output.push_str("\nThought: ");
            output.push_str(thought);
        }
    }
    if let Some(tree) = &state.accessibility_tree {
        if !tree.trim().is_empty() {
            output.push_str("\nAccessibility tree:\n");
            output.push_str(tree);
        }
    }

    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}


#[cfg(test)]
mod tests {
    use super::*;

    fn sample_line() -> String {
        serde_json::json!({
            "id": "traj-001",
            "domain": "web",
            "environment": "shop-alpha",
            "goal": "Find the replacement filter and add it to the cart",
            "outcome": "success",
            "start_url": "https://shop.example/",
            "states": [{
                "state_index": 0,
                "step": 1,
                "url": "https://shop.example/search",
                "action": {"type": "click", "target": "search"},
                "thought": "Search for the exact replacement part",
                "accessibility_tree": "textbox Search; button Find",
                "screenshot": "screenshots/traj-001/0.png"
            }, {
                "state_index": 2,
                "step": 3,
                "url": "https://shop.example/item/7",
                "action": "click add to cart",
                "thought": null,
                "accessibility_tree": "button Add to cart",
                "screenshot": null
            }]
        })
        .to_string()
    }

    #[test]
    fn ingestion_emits_one_valid_evidence_record_per_state() {
        let line = sample_line();
        let records = ingest_longmemeval_v2_jsonl([line.as_bytes()], |domain, environment| {
            (domain == "web" && environment == "shop-alpha")
                .then(|| "test:shop-alpha".to_string())
        })
        .expect("valid trajectory");

        assert_eq!(records.len(), 2);
        assert_ne!(records[0].evidence_id, records[1].evidence_id);
        assert_ne!(records[0].hdc_id, records[1].hdc_id);
        assert_eq!(records[0].scope_id, "test:shop-alpha");
        assert_eq!(records[0].source.trajectory_id, "traj-001");
        assert_eq!(records[0].source.step_index, Some(0));
        assert!(records.iter().all(|record| record.validate().is_ok()));
        assert!(matches!(
            &records[0].payload,
            EvidencePayload::Text { value } if value.contains("Accessibility tree:\ntextbox Search")
        ));
    }

    #[test]
    fn source_digest_is_deterministic_across_whitespace_and_serialization() {
        let first = sample_line();
        let second_value: Value = serde_json::from_str(&first).unwrap();
        let second = serde_json::to_string_pretty(&second_value).unwrap();
        let resolver = |_: &str, _: &str| Some("scope".to_string());

        let first_records = ingest_longmemeval_v2_jsonl([first.as_bytes()], resolver).unwrap();
        let second_records = ingest_longmemeval_v2_jsonl([second.as_bytes()], resolver).unwrap();
        assert_eq!(first_records.len(), second_records.len());

        for (left, right) in first_records.iter().zip(second_records.iter()) {
            assert_eq!(left.evidence_id, right.evidence_id);
            assert_eq!(left.hdc_id, right.hdc_id);
            assert_eq!(left.source.source_sha256, right.source.source_sha256);
            let value: Value = serde_json::from_slice(first.as_bytes()).unwrap();
            let trajectory: LongMemEvalTrajectory = serde_json::from_value(value).unwrap();
            let state = trajectory.states.iter().find(|state| {
                state.state_index == left.source.step_index.unwrap()
            }).unwrap();
            let canonical = canonical_source_bytes(&trajectory, state).unwrap();
            assert!(left.verify_source_bytes(&canonical).is_ok());
        }
    }

    #[test]
    fn question_answer_and_evaluator_metadata_fail_closed() {
        let base: Value = serde_json::from_str(&sample_line()).unwrap();
        for (key, value) in [
            ("question_id", Value::String("q-001".into())),
            ("answer", Value::String("gold".into())),
            ("eval_function", Value::String("contains".into())),
        ] {
            let mut injected = base.clone();
            injected.as_object_mut().unwrap().insert(key.into(), value);
            assert!(
                ingest_longmemeval_v2_jsonl([injected.to_string().into_bytes()], |_, _| Some("scope".into()))
                    .is_err(),
                "must reject top-level {key}"
            );
        }

        let mut nested = base;
        nested["states"][0].as_object_mut().unwrap()
            .insert("questionType".into(), Value::String("static".into()));
        assert!(
            ingest_longmemeval_v2_jsonl([nested.to_string().into_bytes()], |_, _| Some("scope".into())).is_err()
        );
    }

    #[test]
    fn unknown_fields_and_invalid_order_are_rejected() {
        let mut unknown: Value = serde_json::from_str(&sample_line()).unwrap();
        unknown.as_object_mut().unwrap().insert("new_field".into(), Value::Bool(true));
        assert!(
            ingest_longmemeval_v2_jsonl([unknown.to_string().into_bytes()], |_, _| Some("scope".into())).is_err()
        );

        let mut unordered: Value = serde_json::from_str(&sample_line()).unwrap();
        unordered["states"][1]["state_index"] = Value::from(0);
        assert!(
            ingest_longmemeval_v2_jsonl([unordered.to_string().into_bytes()], |_, _| Some("scope".into())).is_err()
        );
    }

    #[test]
    fn missing_trusted_scope_mapping_is_an_error() {
        let line = sample_line();
        let result = ingest_longmemeval_v2_jsonl([line.as_bytes()], |_, _| None);
        assert!(result.unwrap_err().contains("no trusted scope mapping"));
    }

    #[test]
    fn repeated_trajectory_state_identity_is_rejected() {
        let line = sample_line();
        let result = ingest_longmemeval_v2_jsonl(
            [line.as_bytes(), line.as_bytes()],
            |_, _| Some("scope".to_string()),
        );
        assert!(result.unwrap_err().contains("duplicate trajectory-state identity"));
    }

    #[test]
    fn blank_lines_and_unsupported_domains_fail_closed() {
        assert!(
            ingest_longmemeval_v2_jsonl([&b"  "[..]], |_, _| Some("scope".into())).is_err()
        );
        let mut invalid: Value = serde_json::from_str(&sample_line()).unwrap();
        invalid["domain"] = Value::String("mobile".into());
        assert!(
            ingest_longmemeval_v2_jsonl([invalid.to_string().into_bytes()], |_, _| Some("scope".into())).is_err()
        );
    }
}
