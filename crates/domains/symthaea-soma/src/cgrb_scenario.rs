// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Information-flow boundary for CGRB scenario manifests.
//!
//! The frozen evaluator manifest can contain gold labels, postconditions,
//! planned interruptions and oracle identity. None of those are model inputs.
//! This module projects an explicit allowlist into a separate serializable view.
//! It does not validate the entire Draft 2020-12 schema; run the independent
//! schema validator first and preserve the original manifest digest.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CGRB_SCENARIO_PROJECTION_V1: &str = "symthaea.cgrb-scenario-projection.v1";

fn is_digest(value: &str) -> bool {
    let Some((algorithm, hex)) = value.split_once(':') else {
        return false;
    };
    matches!(algorithm, "sha256" | "blake3")
        && hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn object_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &'static str,
) -> Result<&'a Value, CgrbScenarioProjectionErrorV1> {
    object
        .get(key)
        .ok_or(CgrbScenarioProjectionErrorV1::MissingField(key))
}

fn string_field(
    object: &serde_json::Map<String, Value>,
    key: &'static str,
) -> Result<String, CgrbScenarioProjectionErrorV1> {
    object_field(object, key)?
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or(CgrbScenarioProjectionErrorV1::InvalidField(key))
}

fn digest_field(
    object: &serde_json::Map<String, Value>,
    key: &'static str,
) -> Result<Option<String>, CgrbScenarioProjectionErrorV1> {
    match object_field(object, key)? {
        Value::Null => Ok(None),
        Value::String(value) if is_digest(value) => Ok(Some(value.clone())),
        _ => Err(CgrbScenarioProjectionErrorV1::InvalidField(key)),
    }
}

fn string_array(
    object: &serde_json::Map<String, Value>,
    key: &'static str,
    require_nonempty: bool,
) -> Result<Vec<String>, CgrbScenarioProjectionErrorV1> {
    let values = object_field(object, key)?
        .as_array()
        .ok_or(CgrbScenarioProjectionErrorV1::InvalidField(key))?;
    if require_nonempty && values.is_empty() {
        return Err(CgrbScenarioProjectionErrorV1::InvalidField(key));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .map(str::to_owned)
                .ok_or(CgrbScenarioProjectionErrorV1::InvalidField(key))
        })
        .collect()
}

fn u64_field(
    object: &serde_json::Map<String, Value>,
    key: &'static str,
) -> Result<u64, CgrbScenarioProjectionErrorV1> {
    object_field(object, key)?
        .as_u64()
        .ok_or(CgrbScenarioProjectionErrorV1::InvalidField(key))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbModelVisibleConversationV1 {
    pub input_mode: String,
    pub audio_digest: Option<String>,
    pub video_digest: Option<String>,
    pub session_token: String,
    pub capture_token: String,
    pub speaker_tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbModelVisibleAuthorityV1 {
    pub permitted_action_types: Vec<String>,
    pub target_scope: Vec<String>,
    pub purpose: String,
    pub destination: String,
    pub effective_from_tick: u64,
    pub effective_until_tick: u64,
    pub policy_version: u64,
}

/// Only this allowlisted projection may be supplied as scenario metadata to
/// the model. Raw media are delivered through a separately permissioned stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbModelVisibleScenarioV1 {
    pub projection_version: String,
    pub benchmark_id: String,
    pub scenario_id: String,
    pub task_family: String,
    pub conversation: CgrbModelVisibleConversationV1,
    pub authority: CgrbModelVisibleAuthorityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CgrbEvaluatorGoldLabelsV1 {
    pub visibility: String,
    pub gold_intent: String,
    pub gold_addressee: String,
    pub allowed_action_ids: Vec<String>,
    pub expected_outcome: String,
    pub expected_final_claim: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CgrbScenarioProjectionErrorV1 {
    InvalidJson,
    InvalidRoot,
    MissingField(&'static str),
    InvalidField(&'static str),
    GoldLabelsNotEvaluatorOnly,
    InvalidAuthorityWindow,
    UnsupportedInputMode,
}

impl std::fmt::Display for CgrbScenarioProjectionErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CgrbScenarioProjectionErrorV1 {}

/// Evaluator-side envelope. Gold labels are intentionally not serializable as
/// part of the model view and are available only through an explicit grader API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CgrbScenarioEnvelopeV1 {
    model_view: CgrbModelVisibleScenarioV1,
    gold_labels: CgrbEvaluatorGoldLabelsV1,
    manifest_digest: String,
}

impl CgrbScenarioEnvelopeV1 {
    /// Project a previously schema-validated manifest into a model-safe view.
    pub fn from_validated_json(
        raw_manifest: &str,
    ) -> Result<Self, CgrbScenarioProjectionErrorV1> {
        let root: Value = serde_json::from_str(raw_manifest)
            .map_err(|_| CgrbScenarioProjectionErrorV1::InvalidJson)?;
        let root = root
            .as_object()
            .ok_or(CgrbScenarioProjectionErrorV1::InvalidRoot)?;

        let benchmark_id = string_field(root, "benchmark_id")?;
        let scenario_id = string_field(root, "scenario_id")?;
        let task_family = string_field(root, "task_family")?;

        let conversation_value = object_field(root, "conversation")?
            .as_object()
            .ok_or(CgrbScenarioProjectionErrorV1::InvalidField("conversation"))?;
        let input_mode = string_field(conversation_value, "input_mode")?;
        if !matches!(
            input_mode.as_str(),
            "audio_only" | "audio_video" | "transcript_oracle"
        ) {
            return Err(CgrbScenarioProjectionErrorV1::UnsupportedInputMode);
        }
        let audio_digest = digest_field(conversation_value, "audio_digest")?;
        let video_digest = digest_field(conversation_value, "video_digest")?;
        let session_token = string_field(conversation_value, "session_token")?;
        let capture_token = string_field(conversation_value, "capture_token")?;
        let speaker_tokens = string_array(conversation_value, "speaker_tokens", true)?;

        let authority_value = object_field(root, "authority")?
            .as_object()
            .ok_or(CgrbScenarioProjectionErrorV1::InvalidField("authority"))?;
        let permitted_action_types =
            string_array(authority_value, "permitted_action_types", false)?;
        let target_scope = string_array(authority_value, "target_scope", true)?;
        let purpose = string_field(authority_value, "purpose")?;
        let destination = string_field(authority_value, "destination")?;
        let effective_from_tick = u64_field(authority_value, "effective_from_tick")?;
        let effective_until_tick = u64_field(authority_value, "effective_until_tick")?;
        let policy_version = u64_field(authority_value, "policy_version")?;
        if effective_until_tick < effective_from_tick {
            return Err(CgrbScenarioProjectionErrorV1::InvalidAuthorityWindow);
        }

        let gold_value = object_field(root, "gold_labels")?;
        let gold_labels: CgrbEvaluatorGoldLabelsV1 = serde_json::from_value(gold_value.clone())
            .map_err(|_| CgrbScenarioProjectionErrorV1::InvalidField("gold_labels"))?;
        if gold_labels.visibility != "evaluator_only" {
            return Err(CgrbScenarioProjectionErrorV1::GoldLabelsNotEvaluatorOnly);
        }

        let model_view = CgrbModelVisibleScenarioV1 {
            projection_version: CGRB_SCENARIO_PROJECTION_V1.to_owned(),
            benchmark_id,
            scenario_id,
            task_family,
            conversation: CgrbModelVisibleConversationV1 {
                input_mode,
                audio_digest,
                video_digest,
                session_token,
                capture_token,
                speaker_tokens,
            },
            authority: CgrbModelVisibleAuthorityV1 {
                permitted_action_types,
                target_scope,
                purpose,
                destination,
                effective_from_tick,
                effective_until_tick,
                policy_version,
            },
        };
        let manifest_digest = format!(
            "blake3:{}",
            blake3::hash(raw_manifest.as_bytes()).to_hex()
        );
        Ok(Self {
            model_view,
            gold_labels,
            manifest_digest,
        })
    }

    pub fn model_view(&self) -> &CgrbModelVisibleScenarioV1 {
        &self.model_view
    }

    pub fn evaluator_gold_labels(&self) -> &CgrbEvaluatorGoldLabelsV1 {
        &self.gold_labels
    }

    pub fn manifest_digest(&self) -> &str {
        &self.manifest_digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_manifest() -> String {
        json!({
            "schema_version": "cgrb-scenario-v0.1",
            "benchmark_id": "CGRB-v0.1",
            "scenario_id": "dev-case-01",
            "task_family": "create_update",
            "split": "dev",
            "environment": {
                "environment_id": "world-01",
                "snapshot_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "seed": 4,
                "state_model_version": "world-v1"
            },
            "conversation": {
                "input_mode": "audio_video",
                "audio_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "video_digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "session_token": "session-opaque",
                "capture_token": "capture-opaque",
                "speaker_tokens": ["speaker-a", "speaker-b"]
            },
            "requester": {
                "speaker_token": "speaker-a",
                "intended_addressee": "agent",
                "intent_type": "explicit_request"
            },
            "authority": {
                "policy_digest": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "policy_version": 3,
                "grant_token": "grant-secret",
                "permitted_action_types": ["record.update"],
                "target_scope": ["record-1"],
                "purpose": "user-requested-update",
                "destination": "sandbox",
                "effective_from_tick": 10,
                "effective_until_tick": 100
            },
            "initial_state": {
                "state_version": "v1",
                "state_digest": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "observed_tick": 10
            },
            "goal": {
                "goal_id": "goal-1",
                "postconditions": [{
                    "predicate_id": "post-1",
                    "subject": "record-1",
                    "path": "status",
                    "operator": "equals",
                    "expected": "confirmed"
                }]
            },
            "actions": [{
                "action_id": "gold-action-1",
                "action_type": "record.update",
                "target": "record-1",
                "preconditions": [],
                "postconditions": [],
                "idempotency_class": "keyed_idempotent",
                "compensation_action_id": null
            }],
            "interruptions": [{
                "interrupt_id": "hidden-interrupt-1",
                "trigger": "tool_timeout",
                "lifecycle_phase": "after_dispatch",
                "expected_response": "reconcile",
                "offset_ms": 300
            }],
            "oracle": {
                "oracle_id": "hidden-oracle",
                "oracle_build_digest": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "state_model_digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
                "independence": "separate_read_only_service",
                "consistency": "versioned_read",
                "max_age_ms": 50
            },
            "recovery": {
                "unknown_outcome_policy": "reconcile_before_retry",
                "retry_policy": "retry_only_after_reconciliation",
                "compensation_policy": "not_available",
                "human_escalation_available": true
            },
            "rights": {
                "license_record_digest": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
                "consent_scope_digest": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
                "capture_permitted": true,
                "evaluation_permitted": true,
                "publication_permitted": false
            },
            "gold_labels": {
                "visibility": "evaluator_only",
                "gold_intent": "update record-1",
                "gold_addressee": "agent",
                "allowed_action_ids": ["gold-action-1"],
                "expected_outcome": "committed",
                "expected_final_claim": "completed"
            }
        })
        .to_string()
    }

    #[test]
    fn model_projection_excludes_gold_oracle_and_interruption_details() {
        let envelope =
            CgrbScenarioEnvelopeV1::from_validated_json(&valid_manifest()).expect("valid envelope");
        let view = serde_json::to_string(envelope.model_view()).expect("serialize model view");
        for forbidden in [
            "gold_labels",
            "gold-action-1",
            "hidden-oracle",
            "hidden-interrupt-1",
            "postconditions",
            "grant-secret",
            "evaluator_only",
        ] {
            assert!(!view.contains(forbidden), "model view leaked {forbidden}");
        }
        assert!(view.contains("record.update"));
        assert!(view.contains("record-1"));
    }

    #[test]
    fn evaluator_keeps_gold_labels_outside_model_projection() {
        let envelope =
            CgrbScenarioEnvelopeV1::from_validated_json(&valid_manifest()).expect("valid envelope");
        assert_eq!(
            envelope.evaluator_gold_labels().expected_final_claim,
            "completed"
        );
        let view = serde_json::to_value(envelope.model_view()).expect("model view");
        assert!(view.get("gold_labels").is_none());
    }

    #[test]
    fn invalid_gold_visibility_is_rejected() {
        let mut manifest: Value = serde_json::from_str(&valid_manifest()).expect("json");
        manifest["gold_labels"]["visibility"] = Value::String("model_visible".into());
        assert_eq!(
            CgrbScenarioEnvelopeV1::from_validated_json(&manifest.to_string()),
            Err(CgrbScenarioProjectionErrorV1::GoldLabelsNotEvaluatorOnly)
        );
    }

    #[test]
    fn invalid_authority_window_is_rejected() {
        let mut manifest: Value = serde_json::from_str(&valid_manifest()).expect("json");
        manifest["authority"]["effective_from_tick"] = Value::from(101);
        assert_eq!(
            CgrbScenarioEnvelopeV1::from_validated_json(&manifest.to_string()),
            Err(CgrbScenarioProjectionErrorV1::InvalidAuthorityWindow)
        );
    }

    #[test]
    fn malformed_media_digest_is_rejected() {
        let mut manifest: Value = serde_json::from_str(&valid_manifest()).expect("json");
        manifest["conversation"]["audio_digest"] = Value::String("sha256:bad".into());
        assert_eq!(
            CgrbScenarioEnvelopeV1::from_validated_json(&manifest.to_string()),
            Err(CgrbScenarioProjectionErrorV1::InvalidField("audio_digest"))
        );
    }

    #[test]
    fn projection_manifest_digest_binds_hidden_and_visible_sections() {
        let original = valid_manifest();
        let envelope_a =
            CgrbScenarioEnvelopeV1::from_validated_json(&original).expect("original");
        let mut changed: Value = serde_json::from_str(&original).expect("json");
        changed["gold_labels"]["expected_final_claim"] = Value::String("unknown".into());
        let envelope_b =
            CgrbScenarioEnvelopeV1::from_validated_json(&changed.to_string()).expect("changed");
        assert_ne!(envelope_a.manifest_digest(), envelope_b.manifest_digest());
        assert_eq!(envelope_a.model_view(), envelope_b.model_view());
    }
}
