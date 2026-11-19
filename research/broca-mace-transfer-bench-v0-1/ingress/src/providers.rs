// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provider-shaped agent trace capture adapters.
//!
//! These adapters preserve provider event identity and content digests.
//! They never turn provider messages, tool results or session status into
//! independently verified execution or world-outcome evidence.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    AnthropicManagedAgents,
    OpenAiAgentsSdk,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AnthropicManagedAgents => "anthropic_managed_agents",
            Self::OpenAiAgentsSdk => "openai_agents_sdk",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceClass {
    UserMessage,
    AssistantMessage,
    SystemContext,
    GoalDeclaration,
    ToolCallIntent,
    ToolOutputClaim,
    PermissionDecision,
    SessionStatus,
    AgentProgress,
    Handoff,
    ContextManagement,
    Unknown,
}

impl TraceClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UserMessage => "user_message",
            Self::AssistantMessage => "assistant_message",
            Self::SystemContext => "system_context",
            Self::GoalDeclaration => "goal_declaration",
            Self::ToolCallIntent => "tool_call_intent",
            Self::ToolOutputClaim => "tool_output_claim",
            Self::PermissionDecision => "permission_decision",
            Self::SessionStatus => "session_status",
            Self::AgentProgress => "agent_progress",
            Self::Handoff => "handoff",
            Self::ContextManagement => "context_management",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderTraceReject {
    EmptySessionId,
    InvalidJsonLine(usize),
    InvalidEnvelope(usize),
    UnsupportedSchemaVersion(usize),
    SessionMismatch(usize),
    InvalidSequence(usize),
    InvalidCaptureTime(usize),
    MissingNativeType(usize),
    MissingNativeEventId(usize),
    StreamingOnlyEvent(usize),
    ConflictingDuplicateIdentity(usize),
    NonMonotonicSequence(usize),
    SequenceOverflow,
}

#[derive(Clone, Debug)]
pub struct ProviderTraceEvent {
    pub provider: Provider,
    pub session_id: String,
    pub thread_id: Option<String>,
    /// Sequence within the provider thread/session event stream.
    pub sequence: u64,
    // Importer capture time, not underlying action or outcome time.
    pub captured_at_ms: u64,
    pub native_event_id: Option<String>,
    pub native_event_type: String,
    pub class: TraceClass,
    pub correlation_id: Option<String>,
    pub provider_processed_at: Option<String>,
    pub event_payload_sha256: String,
    pub capture_line_sha256: String,
}

impl ProviderTraceEvent {
    pub const fn is_candidate_evidence(&self) -> bool {
        false
    }

    pub const fn is_execution_authority(&self) -> bool {
        false
    }

    pub const fn is_outcome_authority(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub struct ProviderTraceBatch {
    pub provider: Provider,
    pub session_id: String,
    pub events: Vec<ProviderTraceEvent>,
    pub duplicate_event_count: usize,
    pub source_capture_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn string_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn correlation_id(event: &Value) -> Option<String> {
    let raw = event.get("raw_item").unwrap_or(event);
    ["call_id", "tool_use_id", "custom_tool_use_id", "tool_call_id"]
        .iter()
        .find_map(|key| string_at(raw, key).map(str::to_owned))
}

fn classify_anthropic(native_type: &str) -> Option<TraceClass> {
    let class = match native_type {
        "event_start" | "event_delta" => return None,
        "user.message" => TraceClass::UserMessage,
        "system.message" => TraceClass::SystemContext,
        "user.define_outcome" => TraceClass::GoalDeclaration,
        "user.tool_confirmation" => TraceClass::PermissionDecision,
        "user.tool_result" | "user.custom_tool_result" => TraceClass::ToolOutputClaim,
        "agent.message" => TraceClass::AssistantMessage,
        "agent.tool_use" | "agent.mcp_tool_use" | "agent.custom_tool_use" => {
            TraceClass::ToolCallIntent
        }
        "agent.tool_result" | "agent.mcp_tool_result" => TraceClass::ToolOutputClaim,
        "agent.thinking" => TraceClass::AgentProgress,
        "agent.thread_message_sent" | "agent.thread_message_received" => TraceClass::Handoff,
        "agent.thread_context_compacted" => TraceClass::ContextManagement,
        value if value.starts_with("session.status_")
            || value == "session.error"
            || value == "session.usage"
            || value == "session.updated"
            || value == "session.deleted"
            || value.starts_with("workflow_run.") => TraceClass::SessionStatus,
        value if value.starts_with("span.") => TraceClass::AgentProgress,
        _ => TraceClass::Unknown,
    };
    Some(class)
}

fn classify_openai(event: &Value) -> (String, TraceClass) {
    let wrapper_type = string_at(event, "type").unwrap_or("");
    let raw = event.get("raw_item").unwrap_or(event);
    let raw_type = string_at(raw, "type").unwrap_or("");
    let native_type = if !wrapper_type.is_empty() {
        wrapper_type
    } else if !raw_type.is_empty() {
        raw_type
    } else if string_at(raw, "role").is_some() {
        "message"
    } else {
        "unknown"
    };

    let class = match wrapper_type {
        "message_output_item" => TraceClass::AssistantMessage,
        "tool_call_item" | "tool_search_call_item" => TraceClass::ToolCallIntent,
        "tool_call_output_item" | "tool_search_output_item" => TraceClass::ToolOutputClaim,
        "handoff_call_item" | "handoff_output_item" => TraceClass::Handoff,
        _ => classify_openai_raw_item(raw),
    };
    (native_type.to_owned(), class)
}

fn classify_openai_raw_item(raw: &Value) -> TraceClass {
    match string_at(raw, "role").unwrap_or("") {
        "user" => return TraceClass::UserMessage,
        "assistant" => return TraceClass::AssistantMessage,
        "system" | "developer" => return TraceClass::SystemContext,
        _ => {}
    }
    match string_at(raw, "type").unwrap_or("") {
        "function_call"
        | "computer_call"
        | "shell_call"
        | "local_shell_call"
        | "mcp_call"
        | "tool_search_call"
        | "apply_patch_call" => TraceClass::ToolCallIntent,
        "function_call_output"
        | "computer_call_output"
        | "shell_call_output"
        | "local_shell_call_output"
        | "mcp_call_output"
        | "tool_search_output"
        | "apply_patch_output" => TraceClass::ToolOutputClaim,
        "reasoning" | "compaction" => TraceClass::ContextManagement,
        _ => TraceClass::Unknown,
    }
}

/// Normalize caller-captured provider records. Each JSONL line is a v1
/// envelope containing session_id, sequence, captured_at_ms and event.
/// This is an adapter contract, not a provider HTTP/API client.
pub fn normalize_capture_jsonl(
    provider: Provider,
    expected_session_id: &str,
    input: &str,
) -> Result<ProviderTraceBatch, ProviderTraceReject> {
    if expected_session_id.trim().is_empty() {
        return Err(ProviderTraceReject::EmptySessionId);
    }

    let mut events = Vec::new();
    let mut seen = BTreeMap::<String, String>::new();
    let mut last_sequence_by_thread = BTreeMap::<Option<String>, u64>::new();
    let mut duplicate_event_count = 0usize;

    for (line_index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_number = line_index + 1;
        let envelope: Value = serde_json::from_str(line)
            .map_err(|_| ProviderTraceReject::InvalidJsonLine(line_number))?;
        if !envelope.is_object()
            || envelope.get("payload").is_some()
            || envelope.get("event").is_none_or(|event| !event.is_object())
        {
            return Err(ProviderTraceReject::InvalidEnvelope(line_number));
        }
        if envelope.get("schema_version").and_then(Value::as_u64) != Some(1) {
            return Err(ProviderTraceReject::UnsupportedSchemaVersion(line_number));
        }
        if string_at(&envelope, "session_id") != Some(expected_session_id) {
            return Err(ProviderTraceReject::SessionMismatch(line_number));
        }
        let sequence = envelope
            .get("sequence")
            .and_then(Value::as_u64)
            .ok_or(ProviderTraceReject::InvalidSequence(line_number))?;
        let thread_id = string_at(&envelope, "thread_id")
            .or_else(|| string_at(&envelope, "stream_id"))
            .map(str::to_owned);
        let captured_at_ms = envelope
            .get("captured_at_ms")
            .and_then(Value::as_u64)
            .ok_or(ProviderTraceReject::InvalidCaptureTime(line_number))?;
        let event = envelope
            .get("event")
            .cloned()
            .ok_or(ProviderTraceReject::InvalidEnvelope(line_number))?;

        let (native_event_type, class) = match provider {
            Provider::AnthropicManagedAgents => {
                let native_type = string_at(&event, "type")
                    .ok_or(ProviderTraceReject::MissingNativeType(line_number))?;
                let class = classify_anthropic(native_type)
                    .ok_or(ProviderTraceReject::StreamingOnlyEvent(line_number))?;
                (native_type.to_owned(), class)
            }
            Provider::OpenAiAgentsSdk => classify_openai(&event),
        };

        let native_event_id = match provider {
            Provider::AnthropicManagedAgents => Some(
                string_at(&event, "id")
                    .filter(|value| !value.is_empty())
                    .ok_or(ProviderTraceReject::MissingNativeEventId(line_number))?
                    .to_owned(),
            ),
            Provider::OpenAiAgentsSdk => string_at(&event, "input_id")
                .or_else(|| string_at(event.get("raw_item").unwrap_or(&event), "input_id"))
                .or_else(|| string_at(&envelope, "input_id"))
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        };
        let stream_identity = thread_id
            .as_ref()
            .map(|id| format!("thread:{id}"))
            .unwrap_or_else(|| "session-stream".to_owned());
        let identity = native_event_id
            .as_ref()
            .map(|id| format!("{stream_identity}:native:{id}"))
            .unwrap_or_else(|| format!("{stream_identity}:sequence:{sequence}"));
        let canonical_event = serde_json::to_vec(&event)
            .map_err(|_| ProviderTraceReject::InvalidEnvelope(line_number))?;
        let event_payload_sha256 = sha256_hex(&canonical_event);

        if let Some(previous_digest) = seen.get(&identity) {
            if previous_digest != &event_payload_sha256 {
                return Err(ProviderTraceReject::ConflictingDuplicateIdentity(line_number));
            }
            duplicate_event_count += 1;
            let last = last_sequence_by_thread.get(&thread_id).copied();
            if last.is_none_or(|previous| sequence > previous) {
                last_sequence_by_thread.insert(thread_id.clone(), sequence);
            }
            continue;
        }

        if last_sequence_by_thread
            .get(&thread_id)
            .is_some_and(|previous| sequence <= *previous)
        {
            return Err(ProviderTraceReject::NonMonotonicSequence(line_number));
        }
        last_sequence_by_thread.insert(thread_id.clone(), sequence);
        seen.insert(identity, event_payload_sha256.clone());

        events.push(ProviderTraceEvent {
            provider,
            session_id: expected_session_id.to_owned(),
            thread_id,
            sequence,
            captured_at_ms,
            native_event_id,
            native_event_type,
            class,
            correlation_id: correlation_id(&event),
            provider_processed_at: string_at(&event, "processed_at")
                .or_else(|| string_at(&envelope, "provider_processed_at"))
                .map(str::to_owned),
            event_payload_sha256,
            capture_line_sha256: sha256_hex(line.as_bytes()),
        });
    }

    if events.is_empty() {
        return Err(ProviderTraceReject::InvalidEnvelope(1));
    }

    Ok(ProviderTraceBatch {
        provider,
        session_id: expected_session_id.to_owned(),
        events,
        duplicate_event_count,
        source_capture_sha256: sha256_hex(input.as_bytes()),
    })
}

impl ProviderTraceBatch {
    /// Export trace metadata only. Transcript and tool-output bodies remain
    /// in the caller-controlled capture artifact, outside MACE's evidence log.
    pub fn to_mace_jsonl(&self) -> Result<String, ProviderTraceReject> {
        let mut lines = Vec::with_capacity(self.events.len());
        for (index, event) in self.events.iter().enumerate() {
            let event_id = u32::try_from(index + 1)
                .map_err(|_| ProviderTraceReject::SequenceOverflow)?;
            let payload = json!({
                "provider": event.provider.as_str(),
                "provider_thread_id": event.thread_id,
                "provider_sequence": event.sequence,
                "provider_event_class": event.class.as_str(),
                "native_event_id": event.native_event_id,
                "native_event_type": event.native_event_type,
                "correlation_id": event.correlation_id,
                "provider_processed_at": event.provider_processed_at,
                "captured_at_ms": event.captured_at_ms,
                "timestamp_semantics": "capture_time_not_action_time",
                "event_payload_sha256": event.event_payload_sha256,
                "capture_line_sha256": event.capture_line_sha256,
                "source_capture_sha256": self.source_capture_sha256,
                "duplicate_event_count": self.duplicate_event_count,
                "candidate_evidence": false,
                "execution_authority": false,
                "outcome_authority": false
            });
            let record = json!({
                "schema_version": 1,
                "event_id": event_id,
                "session_id": self.session_id,
                "sequence": index as u64 + 1,
                "event_type": "provider_trace_capture",
                "occurred_at_ms": event.captured_at_ms,
                "recorded_at_ms": event.captured_at_ms,
                "valid_from_ms": null,
                "valid_to_ms": null,
                "payload": payload
            });
            lines.push(
                serde_json::to_string(&record)
                    .map_err(|_| ProviderTraceReject::InvalidEnvelope(index + 1))?,
            );
        }
        Ok(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(session: &str, sequence: u64, captured_at_ms: u64, event: Value) -> String {
        json!({
            "schema_version": 1,
            "session_id": session,
            "sequence": sequence,
            "captured_at_ms": captured_at_ms,
            "event": event
        })
        .to_string()
    }

    #[test]
    fn anthropic_tool_use_and_result_are_trace_claims_not_authority() {
        let input = [
            envelope("anthropic", 1, 100, json!({
                "id": "evt-use",
                "type": "agent.tool_use",
                "call_id": "call-1",
                "name": "Bash"
            })),
            envelope("anthropic", 2, 110, json!({
                "id": "evt-result",
                "type": "user.tool_result",
                "tool_use_id": "call-1",
                "content": "hello"
            }))
        ].join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &input
        ).unwrap();
        assert_eq!(batch.events[0].class, TraceClass::ToolCallIntent);
        assert_eq!(batch.events[1].class, TraceClass::ToolOutputClaim);
        assert_eq!(batch.events[1].correlation_id.as_deref(), Some("call-1"));
        assert!(batch.events.iter().all(|event| !event.is_candidate_evidence()));
        assert!(batch.events.iter().all(|event| !event.is_execution_authority()));
        assert!(batch.events.iter().all(|event| !event.is_outcome_authority()));
    }

    #[test]
    fn anthropic_stream_previews_are_rejected() {
        let input = envelope(
            "anthropic",
            1,
            100,
            json!({"type": "event_delta", "event_id": "evt-1"})
        );
        assert_eq!(
            normalize_capture_jsonl(
                Provider::AnthropicManagedAgents,
                "anthropic",
                &input
            ).unwrap_err(),
            ProviderTraceReject::StreamingOnlyEvent(1)
        );
    }

    #[test]
    fn anthropic_reconnect_overlap_is_idempotent_by_native_event_id() {
        let event = json!({
            "id": "evt-shared",
            "type": "agent.message",
            "content": [{"type": "text", "text": "done"}]
        });
        let input = [
            envelope("anthropic", 10, 100, event.clone()),
            envelope("anthropic", 10, 101, event),
            envelope("anthropic", 11, 102, json!({
                "id": "evt-next",
                "type": "session.status_idle"
            }))
        ].join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &input
        ).unwrap();
        assert_eq!(batch.events.len(), 2);
        assert_eq!(batch.duplicate_event_count, 1);
    }

    #[test]
    fn changed_payload_under_same_anthropic_id_is_rejected() {
        let input = [
            envelope("anthropic", 1, 100, json!({
                "id": "evt-1", "type": "agent.message", "content": "first"
            })),
            envelope("anthropic", 2, 101, json!({
                "id": "evt-1", "type": "agent.message", "content": "changed"
            }))
        ].join("\n");
        assert_eq!(
            normalize_capture_jsonl(
                Provider::AnthropicManagedAgents,
                "anthropic",
                &input
            ).unwrap_err(),
            ProviderTraceReject::ConflictingDuplicateIdentity(2)
        );
    }

    #[test]
    fn openai_run_items_keep_tool_intent_and_output_separate() {
        let input = [
            envelope("openai", 0, 100, json!({
                "type": "tool_call_item",
                "raw_item": {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "lookup"
                }
            })),
            envelope("openai", 1, 110, json!({
                "type": "tool_call_output_item",
                "raw_item": {
                    "type": "function_call_output",
                    "call_id": "call-2",
                    "output": "ok"
                }
            }))
        ].join("\n");
        let batch = normalize_capture_jsonl(
            Provider::OpenAiAgentsSdk,
            "openai",
            &input
        ).unwrap();
        assert_eq!(batch.events[0].class, TraceClass::ToolCallIntent);
        assert_eq!(batch.events[1].class, TraceClass::ToolOutputClaim);
        assert_eq!(batch.events[1].correlation_id.as_deref(), Some("call-2"));
    }

    #[test]
    fn openai_raw_session_items_keep_roles_separate() {
        let input = [
            envelope("openai", 5, 200, json!({"role": "user", "content": "inspect"})),
            envelope("openai", 6, 210, json!({"role": "assistant", "content": "done"}))
        ].join("\n");
        let batch = normalize_capture_jsonl(
            Provider::OpenAiAgentsSdk,
            "openai",
            &input
        ).unwrap();
        assert_eq!(batch.events[0].class, TraceClass::UserMessage);
        assert_eq!(batch.events[1].class, TraceClass::AssistantMessage);
    }

    #[test]
    fn mixed_sessions_and_nonmonotonic_new_events_fail_closed() {
        let wrong_session = envelope(
            "other",
            1,
            100,
            json!({"id": "evt-1", "type": "agent.message"})
        );
        assert_eq!(
            normalize_capture_jsonl(
                Provider::AnthropicManagedAgents,
                "anthropic",
                &wrong_session
            ).unwrap_err(),
            ProviderTraceReject::SessionMismatch(1)
        );

        let reordered = [
            envelope("openai", 2, 100, json!({"role": "user"})),
            envelope("openai", 1, 101, json!({"role": "assistant"}))
        ].join("\n");
        assert_eq!(
            normalize_capture_jsonl(
                Provider::OpenAiAgentsSdk,
                "openai",
                &reordered
            ).unwrap_err(),
            ProviderTraceReject::NonMonotonicSequence(2)
        );
    }

    #[test]
    fn thread_sequences_are_independent_and_mace_export_uses_import_order() {
        let input = [
            json!({
                "schema_version": 1,
                "session_id": "anthropic",
                "thread_id": "primary",
                "sequence": 1,
                "captured_at_ms": 100,
                "event": {"id": "primary-1", "type": "agent.message"}
            }).to_string(),
            json!({
                "schema_version": 1,
                "session_id": "anthropic",
                "thread_id": "child",
                "sequence": 1,
                "captured_at_ms": 101,
                "event": {"id": "child-1", "type": "agent.message"}
            }).to_string(),
            json!({
                "schema_version": 1,
                "session_id": "anthropic",
                "thread_id": "primary",
                "sequence": 2,
                "captured_at_ms": 102,
                "event": {"id": "primary-2", "type": "agent.message"}
            }).to_string(),
        ].join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &input
        ).unwrap();

        assert_eq!(batch.events[0].sequence, 1);
        assert_eq!(batch.events[1].sequence, 1);
        assert_eq!(batch.events[2].sequence, 2);
        assert_eq!(batch.events[0].thread_id.as_deref(), Some("primary"));
        assert_eq!(batch.events[1].thread_id.as_deref(), Some("child"));

        let normalized = batch.to_mace_jsonl().unwrap();
        let records: Vec<Value> = normalized
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records[0]["sequence"], 1);
        assert_eq!(records[1]["sequence"], 2);
        assert_eq!(records[2]["sequence"], 3);
        assert_eq!(records[1]["payload"]["provider_sequence"], 1);
        assert_eq!(records[1]["payload"]["provider_thread_id"], "child");
    }

    #[test]
    fn openai_raw_item_input_id_is_used_for_reconnect_deduplication() {
        let event = json!({
            "type": "input_item",
            "raw_item": {
                "input_id": "input-123",
                "role": "user",
                "content": "hello"
            }
        });
        let input = [
            envelope("openai", 1, 100, event.clone()),
            envelope("openai", 1, 101, event)
        ].join("\n");
        let batch = normalize_capture_jsonl(
            Provider::OpenAiAgentsSdk,
            "openai",
            &input
        ).unwrap();
        assert_eq!(batch.events.len(), 1);
        assert_eq!(batch.duplicate_event_count, 1);
        assert_eq!(batch.events[0].native_event_id.as_deref(), Some("input-123"));
    }

    #[test]
    fn unknown_anthropic_events_are_preserved_as_unknown() {
        let input = envelope(
            "anthropic",
            1,
            100,
            json!({"id": "evt-new", "type": "future.new_event"})
        );
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &input
        ).unwrap();
        assert_eq!(batch.events[0].class, TraceClass::Unknown);
    }

    #[test]
    fn mace_export_is_digest_only() {
        let input = envelope("anthropic", 1, 100, json!({
            "id": "evt-tool",
            "type": "user.tool_result",
            "tool_use_id": "call-1",
            "content": "secret tool output"
        }));
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &input
        ).unwrap();
        let normalized = batch.to_mace_jsonl().unwrap();
        assert!(!normalized.contains("secret tool output"));
        assert!(normalized.contains("provider_trace_capture"));
        assert!(normalized.contains("\"outcome_authority\":false"));
        assert!(normalized.contains("\"candidate_evidence\":false"));
    }
}
