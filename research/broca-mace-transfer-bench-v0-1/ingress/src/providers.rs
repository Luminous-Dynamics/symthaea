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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderDataRetentionMode {
    ProviderManaged,
    ClientManaged,
    ZeroDataRetention,
    Unknown,
}

impl ProviderDataRetentionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProviderManaged => "provider_managed",
            Self::ClientManaged => "client_managed",
            Self::ZeroDataRetention => "zero_data_retention",
            Self::Unknown => "unknown",
        }
    }
}

/// Caller-declared experiment configuration, bound to exactly one trace capture.
/// It is not proof that the provider actually executed the declared configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderRunManifestV1 {
    pub provider: Provider,
    pub session_id: String,
    pub run_id: String,
    pub model_id: String,
    pub model_release: String,
    pub api_surface: String,
    pub api_version: String,
    pub endpoint: String,
    pub harness_id: String,
    pub harness_version: String,
    pub harness_revision_sha256: String,
    pub instructions_sha256: String,
    pub tool_schema_sha256: String,
    pub sandbox_image_sha256: String,
    pub memory_policy_sha256: String,
    pub memory_state_sha256: String,
    pub runtime_config_sha256: String,
    pub source_capture_sha256: String,
    pub started_at_ms: i64,
    pub ended_at_ms: i64,
    pub retention_mode: ProviderDataRetentionMode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderRunManifestReject {
    EmptyRunOrConfigurationField,
    InvalidDigest,
    ProviderMismatch,
    SessionMismatch,
    CaptureDigestMismatch,
    InvalidTimeRange,
    CaptureOutsideRunWindow,
}

/// The binding establishes internal consistency only. A separate provider/runtime
/// attestation verifier is required before these declarations can be trusted.
#[derive(Clone, Debug)]
pub struct BoundProviderTraceBatchV1 {
    pub manifest: ProviderRunManifestV1,
    pub manifest_sha256: String,
    pub batch: ProviderTraceBatch,
}


/// One provider page in a paginated session export. Cursor values are stored
/// only as digests; raw cursors may contain capability-like tokens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCapturePageV1 {
    pub page_index: u32,
    pub request_cursor_sha256: String,
    pub next_cursor_sha256: Option<String>,
    pub page_sha256: String,
    pub raw_event_count: usize,
}

/// Caller/collector claim that every page in one provider export was captured.
/// This claim is not trusted until a ProviderCaptureClosureVerifierV1 accepts it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCaptureClosureClaimV1 {
    pub provider: Provider,
    pub session_id: String,
    pub run_id: String,
    pub source_capture_sha256: String,
    pub pages: Vec<ProviderCapturePageV1>,
    pub expected_unique_event_count: usize,
    pub expected_duplicate_event_count: usize,
    pub terminal_page_confirmed: bool,
    pub export_receipt_id: String,
    pub export_receipt_sha256: String,
    pub verifier_profile_id: String,
}

/// A production verifier must authenticate the provider/collector receipt and
/// validate the export request, cursor chain, terminal page and page digests.
/// Returning true is a trust decision; test doubles are not production trust.
pub trait ProviderCaptureClosureVerifierV1 {
    fn verify_export_closure(
        &self,
        claim: &ProviderCaptureClosureClaimV1,
        closure_sha256: &str,
    ) -> bool;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderCaptureClosureReject {
    Manifest(ProviderRunManifestReject),
    ProviderMismatch,
    SessionMismatch,
    RunMismatch,
    CaptureDigestMismatch,
    EmptyPageChain,
    InvalidPageIndex { expected: u32, actual: u32 },
    InvalidCursorDigest { page_index: u32 },
    InvalidPageDigest { page_index: u32 },
    RepeatedRequestCursor { page_index: u32 },
    CursorChainMismatch { page_index: u32 },
    MissingTerminalPage,
    TerminalCursorNotClosed,
    EventCountOverflow,
    UniqueEventCountMismatch,
    DuplicateEventCountMismatch,
    TotalEventCountMismatch,
    InvalidExportReceipt,
    UnverifiedExportReceipt,
}

/// A trace binding that has passed both internal manifest consistency and an
/// injected provider/collector export-closure verifier. Its fields are private
/// so callers cannot relabel a merely bound trace as complete.
#[derive(Clone, Debug)]
pub struct VerifiedProviderTraceBatchV1 {
    bound: BoundProviderTraceBatchV1,
    closure_sha256: String,
    unique_event_count: usize,
    duplicate_event_count: usize,
    page_count: usize,
}

impl VerifiedProviderTraceBatchV1 {
    pub fn batch(&self) -> &ProviderTraceBatch {
        &self.bound.batch
    }

    pub fn manifest(&self) -> &ProviderRunManifestV1 {
        &self.bound.manifest
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.bound.manifest_sha256
    }

    pub fn closure_sha256(&self) -> &str {
        &self.closure_sha256
    }

    pub fn unique_event_count(&self) -> usize {
        self.unique_event_count
    }

    pub fn duplicate_event_count(&self) -> usize {
        self.duplicate_event_count
    }

    pub fn page_count(&self) -> usize {
        self.page_count
    }
}

fn provider_capture_closure_digest(
    claim: &ProviderCaptureClosureClaimV1,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mace:provider-capture-closure:v1\0");
    append_manifest_field(&mut hasher, claim.provider.as_str());
    append_manifest_field(&mut hasher, &claim.session_id);
    append_manifest_field(&mut hasher, &claim.run_id);
    append_manifest_field(&mut hasher, &claim.source_capture_sha256);
    append_manifest_field(
        &mut hasher,
        &claim.expected_unique_event_count.to_string(),
    );
    append_manifest_field(
        &mut hasher,
        &claim.expected_duplicate_event_count.to_string(),
    );
    append_manifest_field(
        &mut hasher,
        if claim.terminal_page_confirmed { "true" } else { "false" },
    );
    append_manifest_field(&mut hasher, &claim.export_receipt_id);
    append_manifest_field(&mut hasher, &claim.export_receipt_sha256);
    append_manifest_field(&mut hasher, &claim.verifier_profile_id);
    append_manifest_field(&mut hasher, &claim.pages.len().to_string());
    for page in &claim.pages {
        append_manifest_field(&mut hasher, &page.page_index.to_string());
        append_manifest_field(&mut hasher, &page.request_cursor_sha256);
        append_manifest_field(
            &mut hasher,
            page.next_cursor_sha256.as_deref().unwrap_or("<terminal>"),
        );
        append_manifest_field(&mut hasher, &page.page_sha256);
        append_manifest_field(&mut hasher, &page.raw_event_count.to_string());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Promote an internally bound trace only after its complete export chain has
/// been verified. The existing bind_provider_run_manifest remains a weaker
/// internal-consistency operation and does not imply capture completeness.
pub fn bind_verified_provider_trace_batch(
    batch: &ProviderTraceBatch,
    manifest: ProviderRunManifestV1,
    claim: ProviderCaptureClosureClaimV1,
    verifier: &impl ProviderCaptureClosureVerifierV1,
) -> Result<VerifiedProviderTraceBatchV1, ProviderCaptureClosureReject> {
    let bound = bind_provider_run_manifest(batch, manifest)
        .map_err(ProviderCaptureClosureReject::Manifest)?;

    if claim.provider != batch.provider {
        return Err(ProviderCaptureClosureReject::ProviderMismatch);
    }
    if claim.session_id != batch.session_id {
        return Err(ProviderCaptureClosureReject::SessionMismatch);
    }
    if claim.run_id != bound.manifest.run_id {
        return Err(ProviderCaptureClosureReject::RunMismatch);
    }
    if claim.source_capture_sha256 != batch.source_capture_sha256 {
        return Err(ProviderCaptureClosureReject::CaptureDigestMismatch);
    }
    if claim.pages.is_empty() {
        return Err(ProviderCaptureClosureReject::EmptyPageChain);
    }
    if !claim.terminal_page_confirmed {
        return Err(ProviderCaptureClosureReject::MissingTerminalPage);
    }
    if claim.export_receipt_id.trim().is_empty()
        || claim.verifier_profile_id.trim().is_empty()
        || !is_lower_sha256(&claim.export_receipt_sha256)
    {
        return Err(ProviderCaptureClosureReject::InvalidExportReceipt);
    }

    let mut request_cursors = std::collections::BTreeSet::new();
    let mut raw_event_count = 0usize;
    for (index, page) in claim.pages.iter().enumerate() {
        let expected_index = u32::try_from(index)
            .map_err(|_| ProviderCaptureClosureReject::EventCountOverflow)?;
        if page.page_index != expected_index {
            return Err(ProviderCaptureClosureReject::InvalidPageIndex {
                expected: expected_index,
                actual: page.page_index,
            });
        }
        if !is_lower_sha256(&page.request_cursor_sha256)
            || page
                .next_cursor_sha256
                .as_deref()
                .is_some_and(|cursor| !is_lower_sha256(cursor))
        {
            return Err(ProviderCaptureClosureReject::InvalidCursorDigest {
                page_index: page.page_index,
            });
        }
        if !is_lower_sha256(&page.page_sha256) || page.raw_event_count == 0 {
            return Err(ProviderCaptureClosureReject::InvalidPageDigest {
                page_index: page.page_index,
            });
        }
        if !request_cursors.insert(page.request_cursor_sha256.clone()) {
            return Err(ProviderCaptureClosureReject::RepeatedRequestCursor {
                page_index: page.page_index,
            });
        }
        raw_event_count = raw_event_count
            .checked_add(page.raw_event_count)
            .ok_or(ProviderCaptureClosureReject::EventCountOverflow)?;

        match (page.next_cursor_sha256.as_deref(), claim.pages.get(index + 1)) {
            (Some(next), Some(next_page)) if next == next_page.request_cursor_sha256 => {}
            (Some(_), Some(_)) | (None, Some(_)) => {
                return Err(ProviderCaptureClosureReject::CursorChainMismatch {
                    page_index: page.page_index,
                });
            }
            (Some(_), None) => {
                return Err(ProviderCaptureClosureReject::TerminalCursorNotClosed);
            }
            (None, None) => {}
        }
    }

    let last_page = claim.pages.last().ok_or(ProviderCaptureClosureReject::EmptyPageChain)?;
    if last_page.next_cursor_sha256.is_some() {
        return Err(ProviderCaptureClosureReject::TerminalCursorNotClosed);
    }
    if claim.expected_unique_event_count != batch.events.len() {
        return Err(ProviderCaptureClosureReject::UniqueEventCountMismatch);
    }
    if claim.expected_duplicate_event_count != batch.duplicate_event_count {
        return Err(ProviderCaptureClosureReject::DuplicateEventCountMismatch);
    }
    let expected_total = batch
        .events
        .len()
        .checked_add(batch.duplicate_event_count)
        .ok_or(ProviderCaptureClosureReject::EventCountOverflow)?;
    if raw_event_count != expected_total {
        return Err(ProviderCaptureClosureReject::TotalEventCountMismatch);
    }

    let closure_sha256 = provider_capture_closure_digest(&claim);
    if !verifier.verify_export_closure(&claim, &closure_sha256) {
        return Err(ProviderCaptureClosureReject::UnverifiedExportReceipt);
    }

    Ok(VerifiedProviderTraceBatchV1 {
        bound,
        closure_sha256,
        unique_event_count: batch.events.len(),
        duplicate_event_count: batch.duplicate_event_count,
        page_count: claim.pages.len(),
    })
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

fn append_manifest_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
}

fn provider_run_manifest_digest(manifest: &ProviderRunManifestV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mace:provider-run-manifest:v1\0");
    append_manifest_field(&mut hasher, manifest.provider.as_str());
    append_manifest_field(&mut hasher, &manifest.session_id);
    append_manifest_field(&mut hasher, &manifest.run_id);
    append_manifest_field(&mut hasher, &manifest.model_id);
    append_manifest_field(&mut hasher, &manifest.model_release);
    append_manifest_field(&mut hasher, &manifest.api_surface);
    append_manifest_field(&mut hasher, &manifest.api_version);
    append_manifest_field(&mut hasher, &manifest.endpoint);
    append_manifest_field(&mut hasher, &manifest.harness_id);
    append_manifest_field(&mut hasher, &manifest.harness_version);
    append_manifest_field(&mut hasher, &manifest.harness_revision_sha256);
    append_manifest_field(&mut hasher, &manifest.instructions_sha256);
    append_manifest_field(&mut hasher, &manifest.tool_schema_sha256);
    append_manifest_field(&mut hasher, &manifest.sandbox_image_sha256);
    append_manifest_field(&mut hasher, &manifest.memory_policy_sha256);
    append_manifest_field(&mut hasher, &manifest.memory_state_sha256);
    append_manifest_field(&mut hasher, &manifest.runtime_config_sha256);
    append_manifest_field(&mut hasher, &manifest.source_capture_sha256);
    append_manifest_field(&mut hasher, &manifest.started_at_ms.to_string());
    append_manifest_field(&mut hasher, &manifest.ended_at_ms.to_string());
    append_manifest_field(&mut hasher, manifest.retention_mode.as_str());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Bind the declared model/harness/memory configuration to one exact trace capture.
/// This does not authenticate the declaration or confer execution/outcome authority.
pub fn bind_provider_run_manifest(
    batch: &ProviderTraceBatch,
    manifest: ProviderRunManifestV1,
) -> Result<BoundProviderTraceBatchV1, ProviderRunManifestReject> {
    let required = [
        manifest.session_id.as_str(),
        manifest.run_id.as_str(),
        manifest.model_id.as_str(),
        manifest.model_release.as_str(),
        manifest.api_surface.as_str(),
        manifest.api_version.as_str(),
        manifest.endpoint.as_str(),
        manifest.harness_id.as_str(),
        manifest.harness_version.as_str(),
    ];
    if required.iter().any(|value| value.trim().is_empty()) {
        return Err(ProviderRunManifestReject::EmptyRunOrConfigurationField);
    }

    let digests = [
        manifest.harness_revision_sha256.as_str(),
        manifest.instructions_sha256.as_str(),
        manifest.tool_schema_sha256.as_str(),
        manifest.sandbox_image_sha256.as_str(),
        manifest.memory_policy_sha256.as_str(),
        manifest.memory_state_sha256.as_str(),
        manifest.runtime_config_sha256.as_str(),
        manifest.source_capture_sha256.as_str(),
    ];
    if digests.iter().any(|value| !is_lower_sha256(value)) {
        return Err(ProviderRunManifestReject::InvalidDigest);
    }
    if manifest.provider != batch.provider {
        return Err(ProviderRunManifestReject::ProviderMismatch);
    }
    if manifest.session_id != batch.session_id {
        return Err(ProviderRunManifestReject::SessionMismatch);
    }
    if manifest.source_capture_sha256 != batch.source_capture_sha256 {
        return Err(ProviderRunManifestReject::CaptureDigestMismatch);
    }
    if manifest.started_at_ms < 0 || manifest.ended_at_ms < manifest.started_at_ms {
        return Err(ProviderRunManifestReject::InvalidTimeRange);
    }

    let min_capture = batch
        .events
        .iter()
        .map(|event| event.captured_at_ms)
        .min()
        .ok_or(ProviderRunManifestReject::CaptureOutsideRunWindow)?;
    let max_capture = batch
        .events
        .iter()
        .map(|event| event.captured_at_ms)
        .max()
        .ok_or(ProviderRunManifestReject::CaptureOutsideRunWindow)?;
    if min_capture < manifest.started_at_ms as u64
        || max_capture > manifest.ended_at_ms as u64
    {
        return Err(ProviderRunManifestReject::CaptureOutsideRunWindow);
    }

    let manifest_sha256 = provider_run_manifest_digest(&manifest);
    Ok(BoundProviderTraceBatchV1 {
        manifest,
        manifest_sha256,
        batch: batch.clone(),
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

    fn provider_manifest_fixture(batch: &ProviderTraceBatch) -> ProviderRunManifestV1 {
        let digest = "a".repeat(64);
        ProviderRunManifestV1 {
            provider: batch.provider,
            session_id: batch.session_id.clone(),
            run_id: "run-2026-11-20-001".into(),
            model_id: "gpt-6.2-astra".into(),
            model_release: "2026-11-17".into(),
            api_surface: "responses".into(),
            api_version: "v1".into(),
            endpoint: "/v1/responses".into(),
            harness_id: "broca-harness".into(),
            harness_version: "0.4.0".into(),
            harness_revision_sha256: digest.clone(),
            instructions_sha256: digest.clone(),
            tool_schema_sha256: digest.clone(),
            sandbox_image_sha256: digest.clone(),
            memory_policy_sha256: digest.clone(),
            memory_state_sha256: digest.clone(),
            runtime_config_sha256: digest,
            source_capture_sha256: batch.source_capture_sha256.clone(),
            started_at_ms: batch.events[0].captured_at_ms as i64,
            ended_at_ms: batch.events.last().unwrap().captured_at_ms as i64,
            retention_mode: ProviderDataRetentionMode::ClientManaged,
        }
    }

    #[test]
    fn run_manifest_binds_model_harness_memory_and_exact_trace_capture() {
        let input = [
            envelope("manifest-session", 1, 100, json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "hello"
            })),
            envelope("manifest-session", 2, 120, json!({
                "id": "evt-2",
                "type": "session.status_idle"
            })),
        ]
        .join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "manifest-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let bound = bind_provider_run_manifest(&batch, manifest.clone()).unwrap();
        assert_eq!(bound.batch.source_capture_sha256, batch.source_capture_sha256);
        assert_eq!(bound.manifest, manifest);
        assert_eq!(
            bound.manifest_sha256,
            provider_run_manifest_digest(&bound.manifest)
        );

        let mut changed_harness = manifest;
        changed_harness.harness_revision_sha256 = "b".repeat(64);
        let changed = bind_provider_run_manifest(&batch, changed_harness).unwrap();
        assert_ne!(bound.manifest_sha256, changed.manifest_sha256);
    }

    #[test]
    fn run_manifest_rejects_provider_session_and_capture_rebinding() {
        let input = envelope(
            "manifest-session",
            1,
            100,
            json!({"id": "evt-1", "type": "agent.message"}),
        );
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "manifest-session",
            &input,
        )
        .unwrap();

        let mut wrong_provider = provider_manifest_fixture(&batch);
        wrong_provider.provider = Provider::OpenAiAgentsSdk;
        assert_eq!(
            bind_provider_run_manifest(&batch, wrong_provider).unwrap_err(),
            ProviderRunManifestReject::ProviderMismatch
        );

        let mut wrong_session = provider_manifest_fixture(&batch);
        wrong_session.session_id = "other-session".into();
        assert_eq!(
            bind_provider_run_manifest(&batch, wrong_session).unwrap_err(),
            ProviderRunManifestReject::SessionMismatch
        );

        let mut wrong_capture = provider_manifest_fixture(&batch);
        wrong_capture.source_capture_sha256 = "b".repeat(64);
        assert_eq!(
            bind_provider_run_manifest(&batch, wrong_capture).unwrap_err(),
            ProviderRunManifestReject::CaptureDigestMismatch
        );
    }

    #[test]
    fn run_manifest_rejects_bad_digests_and_invalid_time_windows() {
        let input = envelope(
            "manifest-session",
            1,
            100,
            json!({"id": "evt-1", "type": "agent.message"}),
        );
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "manifest-session",
            &input,
        )
        .unwrap();

        let mut bad_digest = provider_manifest_fixture(&batch);
        bad_digest.tool_schema_sha256 = "NOT-A-DIGEST".into();
        assert_eq!(
            bind_provider_run_manifest(&batch, bad_digest).unwrap_err(),
            ProviderRunManifestReject::InvalidDigest
        );

        let mut before_capture = provider_manifest_fixture(&batch);
        before_capture.started_at_ms = 101;
        before_capture.ended_at_ms = 120;
        assert_eq!(
            bind_provider_run_manifest(&batch, before_capture).unwrap_err(),
            ProviderRunManifestReject::CaptureOutsideRunWindow
        );

        let mut reversed = provider_manifest_fixture(&batch);
        reversed.started_at_ms = 101;
        reversed.ended_at_ms = 100;
        assert_eq!(
            bind_provider_run_manifest(&batch, reversed).unwrap_err(),
            ProviderRunManifestReject::InvalidTimeRange
        );
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

    struct AcceptingExportClosure;

    impl ProviderCaptureClosureVerifierV1 for AcceptingExportClosure {
        fn verify_export_closure(
            &self,
            claim: &ProviderCaptureClosureClaimV1,
            closure_sha256: &str,
        ) -> bool {
            claim.export_receipt_id == "signed-export-receipt"
                && claim.verifier_profile_id == "mycelix-test-profile-v1"
                && is_lower_sha256(closure_sha256)
        }
    }

    struct RejectingExportClosure;

    impl ProviderCaptureClosureVerifierV1 for RejectingExportClosure {
        fn verify_export_closure(
            &self,
            _claim: &ProviderCaptureClosureClaimV1,
            _closure_sha256: &str,
        ) -> bool {
            false
        }
    }

    fn closure_claim_fixture(
        batch: &ProviderTraceBatch,
        manifest: &ProviderRunManifestV1,
    ) -> ProviderCaptureClosureClaimV1 {
        ProviderCaptureClosureClaimV1 {
            provider: batch.provider,
            session_id: batch.session_id.clone(),
            run_id: manifest.run_id.clone(),
            source_capture_sha256: batch.source_capture_sha256.clone(),
            pages: vec![ProviderCapturePageV1 {
                page_index: 0,
                request_cursor_sha256: "1".repeat(64),
                next_cursor_sha256: None,
                page_sha256: "2".repeat(64),
                raw_event_count: batch.events.len() + batch.duplicate_event_count,
            }],
            expected_unique_event_count: batch.events.len(),
            expected_duplicate_event_count: batch.duplicate_event_count,
            terminal_page_confirmed: true,
            export_receipt_id: "signed-export-receipt".into(),
            export_receipt_sha256: "3".repeat(64),
            verifier_profile_id: "mycelix-test-profile-v1".into(),
        }
    }

    #[test]
    fn capture_closure_requires_terminal_page_counts_and_external_verification() {
        let input = [
            envelope(
                "closure-session",
                1,
                100,
                json!({"id": "evt-1", "type": "agent.message"}),
            ),
            envelope(
                "closure-session",
                2,
                110,
                json!({"id": "evt-2", "type": "session.status_idle"}),
            ),
        ]
        .join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "closure-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let claim = closure_claim_fixture(&batch, &manifest);
        let verified = bind_verified_provider_trace_batch(
            &batch,
            manifest,
            claim,
            &AcceptingExportClosure,
        )
        .unwrap();

        assert_eq!(verified.unique_event_count(), 2);
        assert_eq!(verified.duplicate_event_count(), 0);
        assert_eq!(verified.page_count(), 1);
        assert!(is_lower_sha256(verified.closure_sha256()));
        assert_eq!(verified.batch().events.len(), 2);
    }

    #[test]
    fn capture_closure_accepts_verified_multi_page_overlap_and_deduplication() {
        let duplicate_event = json!({
            "id": "evt-1",
            "type": "agent.message",
            "content": "hello"
        });
        let input = [
            envelope("closure-session", 1, 100, duplicate_event.clone()),
            envelope("closure-session", 1, 100, duplicate_event),
            envelope(
                "closure-session",
                2,
                110,
                json!({"id": "evt-2", "type": "session.status_idle"}),
            ),
        ]
        .join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "closure-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let mut claim = closure_claim_fixture(&batch, &manifest);
        claim.pages = vec![
            ProviderCapturePageV1 {
                page_index: 0,
                request_cursor_sha256: "1".repeat(64),
                next_cursor_sha256: Some("2".repeat(64)),
                page_sha256: "4".repeat(64),
                raw_event_count: 2,
            },
            ProviderCapturePageV1 {
                page_index: 1,
                request_cursor_sha256: "2".repeat(64),
                next_cursor_sha256: None,
                page_sha256: "5".repeat(64),
                raw_event_count: 1,
            },
        ];
        let verified = bind_verified_provider_trace_batch(
            &batch,
            manifest,
            claim,
            &AcceptingExportClosure,
        )
        .unwrap();

        assert_eq!(verified.unique_event_count(), 2);
        assert_eq!(verified.duplicate_event_count(), 1);
        assert_eq!(verified.page_count(), 2);
    }

    #[test]
    fn capture_closure_rejects_missing_terminal_page() {
        let input = envelope(
            "closure-session",
            1,
            100,
            json!({"id": "evt-1", "type": "agent.message"}),
        );
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "closure-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let mut claim = closure_claim_fixture(&batch, &manifest);
        claim.terminal_page_confirmed = false;

        assert_eq!(
            bind_verified_provider_trace_batch(
                &batch,
                manifest,
                claim,
                &AcceptingExportClosure,
            )
            .unwrap_err(),
            ProviderCaptureClosureReject::MissingTerminalPage
        );
    }

    #[test]
    fn capture_closure_rejects_truncated_event_count() {
        let input = [
            envelope(
                "closure-session",
                1,
                100,
                json!({"id": "evt-1", "type": "agent.message"}),
            ),
            envelope(
                "closure-session",
                2,
                110,
                json!({"id": "evt-2", "type": "session.status_idle"}),
            ),
        ]
        .join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "closure-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let mut claim = closure_claim_fixture(&batch, &manifest);
        claim.expected_unique_event_count = 1;

        assert_eq!(
            bind_verified_provider_trace_batch(
                &batch,
                manifest,
                claim,
                &AcceptingExportClosure,
            )
            .unwrap_err(),
            ProviderCaptureClosureReject::UniqueEventCountMismatch
        );
    }

    #[test]
    fn capture_closure_rejects_broken_or_repeated_page_cursors() {
        let input = [
            envelope(
                "closure-session",
                1,
                100,
                json!({"id": "evt-1", "type": "agent.message"}),
            ),
            envelope(
                "closure-session",
                2,
                110,
                json!({"id": "evt-2", "type": "session.status_idle"}),
            ),
        ]
        .join("\n");
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "closure-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let mut claim = closure_claim_fixture(&batch, &manifest);
        claim.pages = vec![
            ProviderCapturePageV1 {
                page_index: 0,
                request_cursor_sha256: "1".repeat(64),
                next_cursor_sha256: Some("2".repeat(64)),
                page_sha256: "4".repeat(64),
                raw_event_count: 1,
            },
            ProviderCapturePageV1 {
                page_index: 1,
                request_cursor_sha256: "3".repeat(64),
                next_cursor_sha256: None,
                page_sha256: "5".repeat(64),
                raw_event_count: 1,
            },
        ];
        assert_eq!(
            bind_verified_provider_trace_batch(
                &batch,
                manifest.clone(),
                claim.clone(),
                &AcceptingExportClosure,
            )
            .unwrap_err(),
            ProviderCaptureClosureReject::CursorChainMismatch { page_index: 0 }
        );

        claim.pages[1].request_cursor_sha256 = claim.pages[0]
            .request_cursor_sha256
            .clone();
        claim.pages[0].next_cursor_sha256 =
            Some(claim.pages[1].request_cursor_sha256.clone());
        assert_eq!(
            bind_verified_provider_trace_batch(
                &batch,
                manifest,
                claim,
                &AcceptingExportClosure,
            )
            .unwrap_err(),
            ProviderCaptureClosureReject::RepeatedRequestCursor { page_index: 1 }
        );
    }

    #[test]
    fn capture_closure_requires_export_receipt_verifier_acceptance() {
        let input = envelope(
            "closure-session",
            1,
            100,
            json!({"id": "evt-1", "type": "agent.message"}),
        );
        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "closure-session",
            &input,
        )
        .unwrap();
        let manifest = provider_manifest_fixture(&batch);
        let claim = closure_claim_fixture(&batch, &manifest);

        assert_eq!(
            bind_verified_provider_trace_batch(
                &batch,
                manifest,
                claim,
                &RejectingExportClosure,
            )
            .unwrap_err(),
            ProviderCaptureClosureReject::UnverifiedExportReceipt
        );
    }

}
