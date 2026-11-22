// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact source-span binding for extracted provider-trace claims.
//!
//! A valid span proves only that the quoted bytes occur at the referenced
//! location in the captured provider event. The assertion remains untrusted:
//! this module does not verify truth, entailment, execution, outcome or authority.

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::providers::{
    normalize_capture_jsonl, Provider, ProviderTraceReject, TraceClass,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderClaimKind {
    ActionIntent,
    ToolOutcomeClaim,
    WorldObservationClaim,
    GoalOrConstraint,
    CorrectionOrRetraction,
    Preference,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderClaimSpanV1 {
    /// RFC 6901 JSON Pointer into the decoded provider event object.
    pub json_pointer: String,
    /// UTF-8 byte offsets within the pointed-to decoded string value.
    pub byte_start: usize,
    pub byte_end: usize,
    /// SHA-256 over exactly the decoded UTF-8 bytes in [byte_start, byte_end).
    pub span_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderClaimProposalV1 {
    pub provider: Provider,
    pub session_id: String,
    pub thread_id: Option<String>,
    pub provider_sequence: u64,
    pub native_event_id: Option<String>,
    pub native_event_type: String,
    pub trace_class: TraceClass,
    pub source_capture_sha256: String,
    pub capture_line_sha256: String,
    pub event_payload_sha256: String,
    pub kind: ProviderClaimKind,
    /// Extractor-generated assertion. It is not treated as source truth.
    pub assertion: String,
    pub assertion_sha256: String,
    pub source_span: ProviderClaimSpanV1,
    pub extractor_id: String,
    pub extractor_version: String,
    pub extractor_config_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderClaimReject {
    EmptySessionId,
    EmptyAssertion,
    InvalidAssertionDigest,
    InvalidExtractorMetadata,
    InvalidExtractorDigest,
    InvalidCaptureDigest,
    CaptureDigestMismatch,
    TraceRejected(ProviderTraceReject),
    ReferencedEventNotFound,
    EventIdentityMismatch,
    CaptureLineNotFound,
    InvalidCaptureLine,
    EventPayloadDigestMismatch,
    JsonPointerMissing,
    SpanIsNotString,
    EmptySpan,
    SpanBoundsInvalid,
    SpanSplitsUtf8Character,
    SpanDigestMismatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderClaimSourceRefV1 {
    pub provider: Provider,
    pub session_id: String,
    pub thread_id: Option<String>,
    pub provider_sequence: u64,
    pub native_event_id: Option<String>,
    pub native_event_type: String,
    pub trace_class: TraceClass,
    pub source_capture_sha256: String,
    pub capture_line_sha256: String,
    pub event_payload_sha256: String,
    pub json_pointer: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub span_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UntrustedProviderClaimV1 {
    kind: ProviderClaimKind,
    assertion: String,
    assertion_sha256: String,
    source_text: String,
    source_ref: ProviderClaimSourceRefV1,
    extractor_id: String,
    extractor_version: String,
    extractor_config_sha256: String,
}

impl UntrustedProviderClaimV1 {
    pub fn kind(&self) -> ProviderClaimKind {
        self.kind
    }

    pub fn assertion(&self) -> &str {
        &self.assertion
    }

    pub fn assertion_sha256(&self) -> &str {
        &self.assertion_sha256
    }

    /// Exact captured text selected by the source-span proposal.
    pub fn source_text(&self) -> &str {
        &self.source_text
    }

    pub fn source_ref(&self) -> &ProviderClaimSourceRefV1 {
        &self.source_ref
    }

    pub fn extractor_id(&self) -> &str {
        &self.extractor_id
    }

    pub fn extractor_version(&self) -> &str {
        &self.extractor_version
    }

    pub fn extractor_config_sha256(&self) -> &str {
        &self.extractor_config_sha256
    }

    pub const fn is_candidate_evidence(&self) -> bool {
        false
    }

    pub const fn is_execution_authority(&self) -> bool {
        false
    }

    pub const fn is_outcome_authority(&self) -> bool {
        false
    }

    pub const fn requires_independent_grounding(&self) -> bool {
        true
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate the source reference against the original privacy-controlled
/// capture artifact. This returns an untrusted claim, never an Experience.
pub fn validate_untrusted_provider_claim(
    proposal: &ProviderClaimProposalV1,
    capture_jsonl: &str,
) -> Result<UntrustedProviderClaimV1, ProviderClaimReject> {
    if proposal.session_id.trim().is_empty() {
        return Err(ProviderClaimReject::EmptySessionId);
    }
    if proposal.assertion.trim().is_empty() {
        return Err(ProviderClaimReject::EmptyAssertion);
    }
    if sha256_hex(proposal.assertion.as_bytes()) != proposal.assertion_sha256 {
        return Err(ProviderClaimReject::InvalidAssertionDigest);
    }
    if proposal.extractor_id.trim().is_empty()
        || proposal.extractor_version.trim().is_empty()
    {
        return Err(ProviderClaimReject::InvalidExtractorMetadata);
    }
    if !is_lower_sha256(&proposal.extractor_config_sha256) {
        return Err(ProviderClaimReject::InvalidExtractorDigest);
    }
    if !is_lower_sha256(&proposal.source_capture_sha256)
        || !is_lower_sha256(&proposal.capture_line_sha256)
        || !is_lower_sha256(&proposal.event_payload_sha256)
        || !is_lower_sha256(&proposal.source_span.span_sha256)
    {
        return Err(ProviderClaimReject::InvalidCaptureDigest);
    }
    let actual_capture_digest = sha256_hex(capture_jsonl.as_bytes());
    if actual_capture_digest != proposal.source_capture_sha256 {
        return Err(ProviderClaimReject::CaptureDigestMismatch);
    }

    let batch = normalize_capture_jsonl(
        proposal.provider,
        &proposal.session_id,
        capture_jsonl,
    )
    .map_err(ProviderClaimReject::TraceRejected)?;
    if batch.source_capture_sha256 != proposal.source_capture_sha256 {
        return Err(ProviderClaimReject::CaptureDigestMismatch);
    }

    let event = batch
        .events
        .iter()
        .find(|event| {
            event.thread_id == proposal.thread_id
                && event.sequence == proposal.provider_sequence
                && event.native_event_id == proposal.native_event_id
                && event.native_event_type == proposal.native_event_type
        })
        .ok_or(ProviderClaimReject::ReferencedEventNotFound)?;

    if event.provider != proposal.provider
        || event.session_id != proposal.session_id
        || event.class != proposal.trace_class
    {
        return Err(ProviderClaimReject::EventIdentityMismatch);
    }
    if event.capture_line_sha256 != proposal.capture_line_sha256 {
        return Err(ProviderClaimReject::EventIdentityMismatch);
    }
    if event.event_payload_sha256 != proposal.event_payload_sha256 {
        return Err(ProviderClaimReject::EventPayloadDigestMismatch);
    }

    let raw_line = capture_jsonl
        .lines()
        .find(|line| sha256_hex(line.as_bytes()) == proposal.capture_line_sha256)
        .ok_or(ProviderClaimReject::CaptureLineNotFound)?;
    let envelope: Value = serde_json::from_str(raw_line)
        .map_err(|_| ProviderClaimReject::InvalidCaptureLine)?;
    let raw_event = envelope
        .get("event")
        .filter(|value| value.is_object())
        .ok_or(ProviderClaimReject::InvalidCaptureLine)?;
    let canonical_event = serde_json::to_vec(raw_event)
        .map_err(|_| ProviderClaimReject::InvalidCaptureLine)?;
    if sha256_hex(&canonical_event) != proposal.event_payload_sha256 {
        return Err(ProviderClaimReject::EventPayloadDigestMismatch);
    }

    let source_value = raw_event
        .pointer(&proposal.source_span.json_pointer)
        .ok_or(ProviderClaimReject::JsonPointerMissing)?;
    let source_text = source_value
        .as_str()
        .ok_or(ProviderClaimReject::SpanIsNotString)?;
    if proposal.source_span.byte_start >= proposal.source_span.byte_end
        || proposal.source_span.byte_end > source_text.len()
    {
        return Err(ProviderClaimReject::SpanBoundsInvalid);
    }
    if !source_text.is_char_boundary(proposal.source_span.byte_start)
        || !source_text.is_char_boundary(proposal.source_span.byte_end)
    {
        return Err(ProviderClaimReject::SpanSplitsUtf8Character);
    }
    let exact_span = &source_text[
        proposal.source_span.byte_start..proposal.source_span.byte_end
    ];
    if exact_span.is_empty() {
        return Err(ProviderClaimReject::EmptySpan);
    }
    if sha256_hex(exact_span.as_bytes()) != proposal.source_span.span_sha256 {
        return Err(ProviderClaimReject::SpanDigestMismatch);
    }

    Ok(UntrustedProviderClaimV1 {
        kind: proposal.kind,
        assertion: proposal.assertion.clone(),
        assertion_sha256: proposal.assertion_sha256.clone(),
        source_text: exact_span.to_owned(),
        source_ref: ProviderClaimSourceRefV1 {
            provider: event.provider,
            session_id: event.session_id.clone(),
            thread_id: event.thread_id.clone(),
            provider_sequence: event.sequence,
            native_event_id: event.native_event_id.clone(),
            native_event_type: event.native_event_type.clone(),
            trace_class: event.class,
            source_capture_sha256: batch.source_capture_sha256,
            capture_line_sha256: event.capture_line_sha256.clone(),
            event_payload_sha256: event.event_payload_sha256.clone(),
            json_pointer: proposal.source_span.json_pointer.clone(),
            byte_start: proposal.source_span.byte_start,
            byte_end: proposal.source_span.byte_end,
            span_sha256: proposal.source_span.span_sha256.clone(),
        },
        extractor_id: proposal.extractor_id.clone(),
        extractor_version: proposal.extractor_version.clone(),
        extractor_config_sha256: proposal.extractor_config_sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn envelope(
        session: &str,
        sequence: u64,
        captured_at_ms: u64,
        thread_id: Option<&str>,
        event: Value,
    ) -> String {
        let mut record = json!({
            "schema_version": 1,
            "session_id": session,
            "sequence": sequence,
            "captured_at_ms": captured_at_ms,
            "event": event
        });
        if let Some(thread_id) = thread_id {
            record["thread_id"] = json!(thread_id);
        }
        record.to_string()
    }

    fn proposal(
        provider: Provider,
        session: &str,
        capture: &str,
        pointer: &str,
        start: usize,
        end: usize,
        assertion: &str,
    ) -> ProviderClaimProposalV1 {
        let batch = normalize_capture_jsonl(provider, session, capture).unwrap();
        let event = &batch.events[0];
        let raw_line = capture
            .lines()
            .find(|line| sha256_hex(line.as_bytes()) == event.capture_line_sha256)
            .unwrap();
        let envelope: Value = serde_json::from_str(raw_line).unwrap();
        let raw_event = &envelope["event"];
        let source = raw_event.pointer(pointer).unwrap().as_str().unwrap();
        let span = &source[start..end];
        ProviderClaimProposalV1 {
            provider,
            session_id: session.to_owned(),
            thread_id: event.thread_id.clone(),
            provider_sequence: event.sequence,
            native_event_id: event.native_event_id.clone(),
            native_event_type: event.native_event_type.clone(),
            trace_class: event.class,
            source_capture_sha256: batch.source_capture_sha256,
            capture_line_sha256: event.capture_line_sha256.clone(),
            event_payload_sha256: event.event_payload_sha256.clone(),
            kind: ProviderClaimKind::ToolOutcomeClaim,
            assertion: assertion.to_owned(),
            assertion_sha256: sha256_hex(assertion.as_bytes()),
            source_span: ProviderClaimSpanV1 {
                json_pointer: pointer.to_owned(),
                byte_start: start,
                byte_end: end,
                span_sha256: sha256_hex(span.as_bytes()),
            },
            extractor_id: "fixture-extractor".into(),
            extractor_version: "1".into(),
            extractor_config_sha256: "a".repeat(64),
        }
    }

    #[test]
    fn exact_anthropic_span_is_bound_but_remains_untrusted() {
        let event = json!({
            "id": "evt-result",
            "type": "user.tool_result",
            "tool_use_id": "call-1",
            "content": [{"type": "text", "text": "deployment status: complete"}]
        });
        let capture = envelope(
            "anthropic",
            1,
            100,
            Some("primary"),
            event,
        );
        let span_text = "deployment status: complete";
        let proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content/0/text",
            0,
            span_text.len(),
            "The tool output claims deployment completed.",
        );
        let claim = validate_untrusted_provider_claim(&proposal, &capture).unwrap();

        assert_eq!(claim.source_text(), span_text);
        assert_eq!(claim.assertion(), "The tool output claims deployment completed.");
        assert_eq!(claim.source_ref().native_event_id.as_deref(), Some("evt-result"));
        assert_eq!(claim.source_ref().thread_id.as_deref(), Some("primary"));
        assert!(!claim.is_candidate_evidence());
        assert!(!claim.is_execution_authority());
        assert!(!claim.is_outcome_authority());
        assert!(claim.requires_independent_grounding());
    }

    #[test]
    fn openai_run_item_output_can_be_referenced_without_becoming_outcome_authority() {
        let event = json!({
            "type": "tool_call_output_item",
            "raw_item": {
                "type": "function_call_output",
                "call_id": "call-9",
                "output": "status=done"
            }
        });
        let capture = envelope("openai", 7, 200, None, event);
        let output = "status=done";
        let proposal = proposal(
            Provider::OpenAiAgentsSdk,
            "openai",
            &capture,
            "/raw_item/output",
            0,
            output.len(),
            "The tool reports status done.",
        );
        let claim = validate_untrusted_provider_claim(&proposal, &capture).unwrap();
        assert_eq!(claim.source_text(), "status=done");
        assert!(!claim.is_outcome_authority());
    }

    #[test]
    fn capture_mutation_invalidates_the_claim_proposal() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            None,
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "alpha beta"
            }),
        );
        let proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content",
            0,
            10,
            "The agent said alpha beta.",
        );
        let changed = capture.replace("alpha beta", "alpha zeta");
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &changed).unwrap_err(),
            ProviderClaimReject::CaptureDigestMismatch
        );
    }

    #[test]
    fn changed_span_bytes_fail_even_when_the_outer_event_is_still_valid() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            None,
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "alpha beta"
            }),
        );
        let mut proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content",
            0,
            5,
            "The agent said alpha.",
        );
        proposal.source_span.span_sha256 = "f".repeat(64);
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::SpanDigestMismatch
        );
    }

    #[test]
    fn non_string_json_pointer_is_not_a_text_source() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            None,
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": [{"type": "text", "text": "alpha"}]
            }),
        );
        let mut proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content/0/text",
            0,
            5,
            "The agent said alpha.",
        );
        proposal.source_span.json_pointer = "/content/0".into();
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::SpanIsNotString
        );
    }

    #[test]
    fn utf8_byte_offsets_must_not_split_a_multibyte_character() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            None,
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "café"
            }),
        );
        let mut proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content",
            0,
            5,
            "The agent wrote café.",
        );
        proposal.source_span.byte_end = 4;
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::SpanSplitsUtf8Character
        );
    }

    #[test]
    fn assertion_digest_is_bound_to_the_extractor_output() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            None,
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "alpha"
            }),
        );
        let mut proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content",
            0,
            5,
            "The agent said alpha.",
        );
        proposal.assertion.push_str(" altered");
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::InvalidAssertionDigest
        );
    }

    #[test]
    fn wrong_provider_thread_or_native_event_identity_is_rejected() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            Some("thread-a"),
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "alpha"
            }),
        );
        let mut proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content",
            0,
            5,
            "The agent said alpha.",
        );
        proposal.thread_id = Some("thread-b".into());
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::ReferencedEventNotFound
        );

        proposal.thread_id = Some("thread-a".into());
        proposal.native_event_id = Some("evt-other".into());
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::ReferencedEventNotFound
        );
    }

    #[test]
    fn malformed_span_bounds_and_missing_pointer_fail_closed() {
        let capture = envelope(
            "anthropic",
            1,
            100,
            None,
            json!({
                "id": "evt-1",
                "type": "agent.message",
                "content": "alpha"
            }),
        );
        let mut proposal = proposal(
            Provider::AnthropicManagedAgents,
            "anthropic",
            &capture,
            "/content",
            0,
            5,
            "The agent said alpha.",
        );
        proposal.source_span.byte_end = 99;
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::SpanBoundsInvalid
        );
        proposal.source_span.byte_end = 5;
        proposal.source_span.json_pointer = "/missing".into();
        assert_eq!(
            validate_untrusted_provider_claim(&proposal, &capture).unwrap_err(),
            ProviderClaimReject::JsonPointerMissing
        );
    }
}
