#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const TRACE_SCHEMA_VERSION: &str = "mel-agency-transaction-trace-v1";
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomyModeV1 {
    SuggestOnly,
    PreviewOnly,
    ApplyApproved,
    BoundedExploration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalSubjectV1 {
    pub source_artifact_sha256: String,
    pub candidate_artifact_sha256: String,
    pub scope_sha256: String,
    pub protected_set_sha256: String,
    pub requested_change_sha256: String,
    pub edit_plan_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposedV1 {
    pub proposal_id: String,
    pub subject: ProposalSubjectV1,
    pub subject_commitment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewedV1 {
    pub proposal_id: String,
    pub subject_commitment_sha256: String,
    pub preview_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovedV1 {
    pub proposal_id: String,
    pub subject_commitment_sha256: String,
    pub actor_id: String,
    pub authority_receipt_sha256: String,
    pub expires_at_monotonic_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VetoedV1 {
    pub proposal_id: String,
    pub subject_commitment_sha256: String,
    pub actor_id: String,
    pub veto_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAuthorityGrantedV1 {
    pub proposal_id: String,
    pub subject_commitment_sha256: String,
    pub actor_id: String,
    pub policy_sha256: String,
    pub authority_receipt_sha256: String,
    pub expires_at_monotonic_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommittedV1 {
    pub proposal_id: String,
    pub subject_commitment_sha256: String,
    pub authorization_event_id: String,
    pub result_artifact_sha256: String,
    pub preservation_receipt_sha256: String,
    pub preservation_verifier_id: String,
    pub preservation_verifier_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UndoneV1 {
    pub proposal_id: String,
    pub subject_commitment_sha256: String,
    pub commit_event_id: String,
    pub restored_artifact_sha256: String,
    pub undo_receipt_sha256: String,
    pub undo_verifier_id: String,
    pub undo_verifier_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "payload", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgencyEventKindV1 {
    Proposed(ProposedV1),
    Previewed(PreviewedV1),
    Approved(ApprovedV1),
    Vetoed(VetoedV1),
    BoundedAuthorityGranted(BoundedAuthorityGrantedV1),
    Committed(CommittedV1),
    Undone(UndoneV1),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceEventV1 {
    pub event_id: String,
    pub sequence: u64,
    pub monotonic_time_ns: u64,
    pub previous_event_sha256: String,
    pub event_sha256: String,
    pub kind: AgencyEventKindV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgencyTraceV1 {
    pub schema_version: String,
    pub trace_id: String,
    pub autonomy_mode: AutonomyModeV1,
    pub base_artifact_sha256: String,
    pub clock_domain_id: String,
    pub clock_origin_id: String,
    pub events: Vec<TraceEventV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssueV1 {
    pub code: String,
    pub event_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceReportV1 {
    pub report_version: String,
    pub trace_id: String,
    pub valid: bool,
    pub terminal_disposition: String,
    pub event_count: usize,
    pub previewed: bool,
    pub authorization_recorded: bool,
    pub vetoed: bool,
    pub committed: bool,
    pub undone: bool,
    pub issues: Vec<ValidationIssueV1>,
}

#[derive(Serialize)]
struct SubjectHashInput<'a> {
    source_artifact_sha256: &'a str,
    candidate_artifact_sha256: &'a str,
    scope_sha256: &'a str,
    protected_set_sha256: &'a str,
    requested_change_sha256: &'a str,
    edit_plan_sha256: &'a str,
}

#[derive(Serialize)]
struct EventHashInput<'a> {
    event_id: &'a str,
    sequence: u64,
    monotonic_time_ns: u64,
    previous_event_sha256: &'a str,
    kind: &'a AgencyEventKindV1,
}

#[derive(Debug, Clone)]
enum AuthorizationV1 {
    HumanApproval { event_id: String, expires_at_monotonic_ns: u64 },
    BoundedGrant { event_id: String, expires_at_monotonic_ns: u64 },
}

impl AuthorizationV1 {
    fn event_id(&self) -> &str {
        match self {
            Self::HumanApproval { event_id, .. } | Self::BoundedGrant { event_id, .. } => event_id,
        }
    }

    fn expires_at_monotonic_ns(&self) -> u64 {
        match self {
            Self::HumanApproval { expires_at_monotonic_ns, .. }
            | Self::BoundedGrant { expires_at_monotonic_ns, .. } => *expires_at_monotonic_ns,
        }
    }
}

#[derive(Default)]
struct ReplayStateV1 {
    proposal: Option<ProposedV1>,
    previewed: bool,
    authorization: Option<AuthorizationV1>,
    vetoed: bool,
    commit_event_id: Option<String>,
    committed: bool,
    undone: bool,
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn proposal_subject_sha256(subject: &ProposalSubjectV1) -> Result<String, String> {
    digest_serializable(&SubjectHashInput {
        source_artifact_sha256: &subject.source_artifact_sha256,
        candidate_artifact_sha256: &subject.candidate_artifact_sha256,
        scope_sha256: &subject.scope_sha256,
        protected_set_sha256: &subject.protected_set_sha256,
        requested_change_sha256: &subject.requested_change_sha256,
        edit_plan_sha256: &subject.edit_plan_sha256,
    })
}

fn event_sha256(event: &TraceEventV1) -> Result<String, String> {
    digest_serializable(&EventHashInput {
        event_id: &event.event_id,
        sequence: event.sequence,
        monotonic_time_ns: event.monotonic_time_ns,
        previous_event_sha256: &event.previous_event_sha256,
        kind: &event.kind,
    })
}

/// Recomputes the append-only event hash chain. This seals a trace; it does not
/// authorize an action or authenticate an actor.
pub fn seal_trace(trace: &mut AgencyTraceV1) -> Result<(), String> {
    let mut previous = ZERO_SHA256.to_owned();
    for event in &mut trace.events {
        event.previous_event_sha256 = previous.clone();
        event.event_sha256 = event_sha256(event)?;
        previous = event.event_sha256.clone();
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn issue(issues: &mut Vec<ValidationIssueV1>, code: &str, event_id: Option<&str>, detail: &str) {
    issues.push(ValidationIssueV1 {
        code: code.to_owned(),
        event_id: event_id.map(str::to_owned),
        detail: detail.to_owned(),
    });
}

fn check_digest(
    issues: &mut Vec<ValidationIssueV1>,
    event_id: Option<&str>,
    field: &str,
    value: &str,
) {
    if !is_sha256(value) {
        issue(
            issues,
            "invalid_sha256",
            event_id,
            &format!("{field} must be 64 lowercase hexadecimal characters"),
        );
    }
}

fn proposal_matches(proposal: &ProposedV1, proposal_id: &str, commitment: &str) -> bool {
    proposal.proposal_id == proposal_id && proposal.subject_commitment_sha256 == commitment
}

/// Validates lifecycle ordering, subject binding, autonomy mode, veto precedence,
/// expiry, preservation-receipt references, exact undo and event-chain consistency.
/// Artifact bytes, actor identity and external consent receipts are not opened here.
pub fn validate_trace(trace: &AgencyTraceV1) -> TraceReportV1 {
    let mut issues = Vec::new();
    let mut state = ReplayStateV1::default();
    let mut event_ids = BTreeSet::new();
    let mut expected_previous = ZERO_SHA256.to_owned();
    let mut previous_time: Option<u64> = None;

    if trace.schema_version != TRACE_SCHEMA_VERSION {
        issue(&mut issues, "schema_version_mismatch", None, "unsupported trace schema_version");
    }
    if !non_empty(&trace.trace_id) {
        issue(&mut issues, "empty_trace_id", None, "trace_id must not be empty");
    }
    if !non_empty(&trace.clock_domain_id) || !non_empty(&trace.clock_origin_id) {
        issue(&mut issues, "missing_clock_identity", None, "clock domain and origin identifiers are required");
    }
    check_digest(&mut issues, None, "base_artifact_sha256", &trace.base_artifact_sha256);
    if trace.events.is_empty() {
        issue(&mut issues, "empty_event_trace", None, "a trace must contain at least one event");
    }

    for (index, event) in trace.events.iter().enumerate() {
        if !non_empty(&event.event_id) {
            issue(&mut issues, "empty_event_id", Some(&event.event_id), "event_id must not be empty");
        }
        if !event_ids.insert(event.event_id.clone()) {
            issue(&mut issues, "duplicate_event_id", Some(&event.event_id), "event identifiers must be unique");
        }
        let expected_sequence = (index as u64) + 1;
        if event.sequence != expected_sequence {
            issue(
                &mut issues,
                "non_contiguous_sequence",
                Some(&event.event_id),
                &format!("expected sequence {expected_sequence}, received {}", event.sequence),
            );
        }
        if previous_time.is_some_and(|previous| event.monotonic_time_ns <= previous) {
            issue(
                &mut issues,
                "non_monotonic_event_time",
                Some(&event.event_id),
                "event monotonic time must increase strictly",
            );
        }
        previous_time = Some(event.monotonic_time_ns);

        if event.previous_event_sha256 != expected_previous {
            issue(
                &mut issues,
                "previous_hash_mismatch",
                Some(&event.event_id),
                "previous_event_sha256 does not identify the preceding event",
            );
        }
        check_digest(&mut issues, Some(&event.event_id), "previous_event_sha256", &event.previous_event_sha256);
        check_digest(&mut issues, Some(&event.event_id), "event_sha256", &event.event_sha256);
        match event_sha256(event) {
            Ok(actual) if actual != event.event_sha256 => issue(
                &mut issues,
                "event_hash_mismatch",
                Some(&event.event_id),
                "event_sha256 does not match the canonical event payload",
            ),
            Err(error) => issue(&mut issues, "event_hash_serialization_failed", Some(&event.event_id), &error),
            _ => {}
        }
        expected_previous = event.event_sha256.clone();

        if state.undone {
            issue(&mut issues, "event_after_undo", Some(&event.event_id), "undo is terminal for this single-transaction trace");
        }
        if state.committed && !matches!(&event.kind, AgencyEventKindV1::Undone(_)) {
            issue(&mut issues, "event_after_commit", Some(&event.event_id), "only an exact undo event may follow a commit");
        }
        if state.vetoed && !matches!(&event.kind, AgencyEventKindV1::Vetoed(_)) {
            issue(&mut issues, "event_after_veto", Some(&event.event_id), "a veto is terminal; a new proposal requires a new trace");
        }

        match &event.kind {
            AgencyEventKindV1::Proposed(payload) => {
                if index != 0 || state.proposal.is_some() {
                    issue(&mut issues, "proposal_order_invalid", Some(&event.event_id), "exactly one proposal must begin the trace");
                }
                if !non_empty(&payload.proposal_id) {
                    issue(&mut issues, "empty_proposal_id", Some(&event.event_id), "proposal_id must not be empty");
                }
                let subject = &payload.subject;
                for (field, value) in [
                    ("source_artifact_sha256", subject.source_artifact_sha256.as_str()),
                    ("candidate_artifact_sha256", subject.candidate_artifact_sha256.as_str()),
                    ("scope_sha256", subject.scope_sha256.as_str()),
                    ("protected_set_sha256", subject.protected_set_sha256.as_str()),
                    ("requested_change_sha256", subject.requested_change_sha256.as_str()),
                    ("edit_plan_sha256", subject.edit_plan_sha256.as_str()),
                    ("subject_commitment_sha256", payload.subject_commitment_sha256.as_str()),
                ] {
                    check_digest(&mut issues, Some(&event.event_id), field, value);
                }
                if subject.source_artifact_sha256 != trace.base_artifact_sha256 {
                    issue(&mut issues, "proposal_source_mismatch", Some(&event.event_id), "proposal source must equal the trace base artifact");
                }
                if subject.source_artifact_sha256 == subject.candidate_artifact_sha256 {
                    issue(&mut issues, "no_op_candidate", Some(&event.event_id), "candidate artifact must differ from source");
                }
                match proposal_subject_sha256(subject) {
                    Ok(expected) if expected != payload.subject_commitment_sha256 => issue(
                        &mut issues,
                        "proposal_commitment_mismatch",
                        Some(&event.event_id),
                        "subject commitment does not match source, candidate, scope, protection, change and plan digests",
                    ),
                    Err(error) => issue(&mut issues, "proposal_commitment_failed", Some(&event.event_id), &error),
                    _ => {}
                }
                state.proposal = Some(payload.clone());
            }
            AgencyEventKindV1::Previewed(payload) => {
                let Some(proposal) = state.proposal.as_ref() else {
                    issue(&mut issues, "preview_without_proposal", Some(&event.event_id), "preview requires an earlier proposal");
                    continue;
                };
                if !proposal_matches(proposal, &payload.proposal_id, &payload.subject_commitment_sha256) {
                    issue(&mut issues, "preview_subject_mismatch", Some(&event.event_id), "preview must bind the exact proposal commitment");
                }
                check_digest(&mut issues, Some(&event.event_id), "preview_receipt_sha256", &payload.preview_receipt_sha256);
                if state.previewed {
                    issue(&mut issues, "duplicate_preview", Some(&event.event_id), "v1 permits one preview receipt per proposal");
                }
                state.previewed = true;
            }
            AgencyEventKindV1::Approved(payload) => {
                if trace.autonomy_mode != AutonomyModeV1::ApplyApproved {
                    issue(&mut issues, "approval_mode_mismatch", Some(&event.event_id), "human approval is executable only in apply_approved mode");
                }
                validate_authority_subject(
                    &mut issues, &state, &event.event_id, &payload.proposal_id,
                    &payload.subject_commitment_sha256, event.monotonic_time_ns,
                    payload.expires_at_monotonic_ns, &payload.actor_id, &payload.authority_receipt_sha256,
                );
                if state.authorization.is_some() {
                    issue(&mut issues, "duplicate_authorization", Some(&event.event_id), "v1 permits one authorization event per proposal");
                } else {
                    state.authorization = Some(AuthorizationV1::HumanApproval {
                        event_id: event.event_id.clone(),
                        expires_at_monotonic_ns: payload.expires_at_monotonic_ns,
                    });
                }
            }
            AgencyEventKindV1::Vetoed(payload) => {
                let Some(proposal) = state.proposal.as_ref() else {
                    issue(&mut issues, "veto_without_proposal", Some(&event.event_id), "veto requires an earlier proposal");
                    continue;
                };
                if !state.previewed {
                    issue(&mut issues, "veto_before_preview", Some(&event.event_id), "veto must refer to a proposal shown to the collaborator");
                }
                if !proposal_matches(proposal, &payload.proposal_id, &payload.subject_commitment_sha256) {
                    issue(&mut issues, "veto_subject_mismatch", Some(&event.event_id), "veto must bind the exact proposal commitment");
                }
                if !non_empty(&payload.actor_id) {
                    issue(&mut issues, "missing_veto_actor", Some(&event.event_id), "veto actor identifier is required");
                }
                check_digest(&mut issues, Some(&event.event_id), "veto_receipt_sha256", &payload.veto_receipt_sha256);
                if state.vetoed {
                    issue(&mut issues, "duplicate_veto", Some(&event.event_id), "proposal is already vetoed");
                }
                if state.committed {
                    issue(&mut issues, "veto_after_commit", Some(&event.event_id), "a veto cannot retroactively erase a commit");
                }
                state.vetoed = true;
                state.authorization = None;
            }
            AgencyEventKindV1::BoundedAuthorityGranted(payload) => {
                if trace.autonomy_mode != AutonomyModeV1::BoundedExploration {
                    issue(&mut issues, "bounded_grant_mode_mismatch", Some(&event.event_id), "bounded authority requires bounded_exploration mode");
                }
                validate_authority_subject(
                    &mut issues, &state, &event.event_id, &payload.proposal_id,
                    &payload.subject_commitment_sha256, event.monotonic_time_ns,
                    payload.expires_at_monotonic_ns, &payload.actor_id, &payload.authority_receipt_sha256,
                );
                check_digest(&mut issues, Some(&event.event_id), "policy_sha256", &payload.policy_sha256);
                if state.authorization.is_some() {
                    issue(&mut issues, "duplicate_authorization", Some(&event.event_id), "v1 permits one authorization event per proposal");
                } else {
                    state.authorization = Some(AuthorizationV1::BoundedGrant {
                        event_id: event.event_id.clone(),
                        expires_at_monotonic_ns: payload.expires_at_monotonic_ns,
                    });
                }
            }
            AgencyEventKindV1::Committed(payload) => {
                let Some(proposal) = state.proposal.as_ref() else {
                    issue(&mut issues, "commit_without_proposal", Some(&event.event_id), "commit requires an earlier proposal");
                    continue;
                };
                if !state.previewed {
                    issue(&mut issues, "commit_without_preview", Some(&event.event_id), "commit requires a preview receipt");
                }
                if !matches!(trace.autonomy_mode, AutonomyModeV1::ApplyApproved | AutonomyModeV1::BoundedExploration) {
                    issue(&mut issues, "commit_forbidden_by_autonomy_mode", Some(&event.event_id), "suggest_only and preview_only modes cannot commit");
                }
                if state.vetoed {
                    issue(&mut issues, "vetoed_proposal_commit", Some(&event.event_id), "a vetoed proposal cannot be committed");
                }
                if payload.proposal_id != proposal.proposal_id || payload.subject_commitment_sha256 != proposal.subject_commitment_sha256 {
                    issue(&mut issues, "commit_subject_mismatch", Some(&event.event_id), "commit must bind the exact proposed candidate and constraints");
                }
                match &state.authorization {
                    None => issue(&mut issues, "commit_without_authorization", Some(&event.event_id), "commit requires a valid prior authorization event"),
                    Some(authorization) => {
                        if authorization.event_id() != payload.authorization_event_id {
                            issue(&mut issues, "authorization_reference_mismatch", Some(&event.event_id), "commit must reference the exact authorization event");
                        }
                        let correct_kind = matches!(
                            (trace.autonomy_mode, authorization),
                            (AutonomyModeV1::ApplyApproved, AuthorizationV1::HumanApproval { .. })
                                | (AutonomyModeV1::BoundedExploration, AuthorizationV1::BoundedGrant { .. })
                        );
                        if !correct_kind {
                            issue(&mut issues, "authorization_kind_mismatch", Some(&event.event_id), "authorization type does not match autonomy mode");
                        }
                        if event.monotonic_time_ns > authorization.expires_at_monotonic_ns() {
                            issue(&mut issues, "expired_authorization", Some(&event.event_id), "authorization expired before the commit event");
                        }
                    }
                }
                if payload.result_artifact_sha256 != proposal.subject.candidate_artifact_sha256 {
                    issue(&mut issues, "commit_result_mismatch", Some(&event.event_id), "committed result must be the exact previewed and authorized candidate");
                }
                check_digest(&mut issues, Some(&event.event_id), "result_artifact_sha256", &payload.result_artifact_sha256);
                check_digest(&mut issues, Some(&event.event_id), "preservation_receipt_sha256", &payload.preservation_receipt_sha256);
                if !non_empty(&payload.preservation_verifier_id) || !non_empty(&payload.preservation_verifier_version) {
                    issue(&mut issues, "missing_preservation_verifier_identity", Some(&event.event_id), "preservation verifier identity and version are required");
                }
                if state.committed {
                    issue(&mut issues, "duplicate_commit", Some(&event.event_id), "v1 permits at most one commit per trace");
                }
                state.committed = true;
                state.commit_event_id = Some(event.event_id.clone());
            }
            AgencyEventKindV1::Undone(payload) => {
                let Some(proposal) = state.proposal.as_ref() else {
                    issue(&mut issues, "undo_without_proposal", Some(&event.event_id), "undo requires an earlier proposal");
                    continue;
                };
                if !state.committed {
                    issue(&mut issues, "undo_without_commit", Some(&event.event_id), "undo must reference a prior commit");
                }
                if payload.proposal_id != proposal.proposal_id || payload.subject_commitment_sha256 != proposal.subject_commitment_sha256 {
                    issue(&mut issues, "undo_subject_mismatch", Some(&event.event_id), "undo must bind the same proposal and source transaction");
                }
                if state.commit_event_id.as_deref() != Some(payload.commit_event_id.as_str()) {
                    issue(&mut issues, "undo_commit_reference_mismatch", Some(&event.event_id), "undo must reference the exact preceding commit event");
                }
                if payload.restored_artifact_sha256 != proposal.subject.source_artifact_sha256 {
                    issue(&mut issues, "undo_not_exact", Some(&event.event_id), "undo must restore the exact source artifact digest");
                }
                check_digest(&mut issues, Some(&event.event_id), "restored_artifact_sha256", &payload.restored_artifact_sha256);
                check_digest(&mut issues, Some(&event.event_id), "undo_receipt_sha256", &payload.undo_receipt_sha256);
                if !non_empty(&payload.undo_verifier_id) || !non_empty(&payload.undo_verifier_version) {
                    issue(&mut issues, "missing_undo_verifier_identity", Some(&event.event_id), "undo verifier identity and version are required");
                }
                if state.undone {
                    issue(&mut issues, "duplicate_undo", Some(&event.event_id), "v1 permits one exact undo event");
                }
                state.undone = true;
            }
        }
    }

    let terminal_disposition = if state.undone {
        "undone"
    } else if state.vetoed {
        "vetoed"
    } else if state.committed {
        "committed"
    } else if state.authorization.is_some() {
        "authorized_pending_commit"
    } else if state.previewed {
        "previewed"
    } else if state.proposal.is_some() {
        "proposed"
    } else {
        "empty"
    };

    TraceReportV1 {
        report_version: "mel-agency-trace-report-v1".to_owned(),
        trace_id: trace.trace_id.clone(),
        valid: issues.is_empty(),
        terminal_disposition: terminal_disposition.to_owned(),
        event_count: trace.events.len(),
        previewed: state.previewed,
        authorization_recorded: state.authorization.is_some(),
        vetoed: state.vetoed,
        committed: state.committed,
        undone: state.undone,
        issues,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_authority_subject(
    issues: &mut Vec<ValidationIssueV1>,
    state: &ReplayStateV1,
    event_id: &str,
    proposal_id: &str,
    commitment: &str,
    event_time: u64,
    expires_at: u64,
    actor_id: &str,
    receipt_sha256: &str,
) {
    let Some(proposal) = state.proposal.as_ref() else {
        issue(issues, "authorization_without_proposal", Some(event_id), "authorization requires an earlier proposal");
        return;
    };
    if !state.previewed {
        issue(issues, "authorization_without_preview", Some(event_id), "authorization requires a prior preview receipt");
    }
    if state.vetoed {
        issue(issues, "authorization_after_veto", Some(event_id), "a veto is terminal for this proposal");
    }
    if !proposal_matches(proposal, proposal_id, commitment) {
        issue(issues, "authorization_subject_mismatch", Some(event_id), "authorization must bind the exact proposal commitment");
    }
    if !non_empty(actor_id) {
        issue(issues, "missing_authority_actor", Some(event_id), "authority actor identifier is required");
    }
    check_digest(issues, Some(event_id), "authority_receipt_sha256", receipt_sha256);
    if expires_at <= event_time {
        issue(issues, "invalid_authorization_expiry", Some(event_id), "authorization expiry must follow its own event time");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn subject() -> ProposalSubjectV1 {
        ProposalSubjectV1 {
            source_artifact_sha256: digest('a'),
            candidate_artifact_sha256: digest('b'),
            scope_sha256: digest('c'),
            protected_set_sha256: digest('d'),
            requested_change_sha256: digest('e'),
            edit_plan_sha256: digest('f'),
        }
    }

    fn event(event_id: &str, sequence: u64, time: u64, kind: AgencyEventKindV1) -> TraceEventV1 {
        TraceEventV1 {
            event_id: event_id.to_owned(),
            sequence,
            monotonic_time_ns: time,
            previous_event_sha256: ZERO_SHA256.to_owned(),
            event_sha256: ZERO_SHA256.to_owned(),
            kind,
        }
    }

    fn proposed() -> ProposedV1 {
        let subject = subject();
        ProposedV1 {
            proposal_id: "proposal-001".to_owned(),
            subject_commitment_sha256: proposal_subject_sha256(&subject).unwrap(),
            subject,
        }
    }

    fn apply_approved_trace() -> AgencyTraceV1 {
        let proposal = proposed();
        let commitment = proposal.subject_commitment_sha256.clone();
        let proposal_id = proposal.proposal_id.clone();
        let mut trace = AgencyTraceV1 {
            schema_version: TRACE_SCHEMA_VERSION.to_owned(),
            trace_id: "trace-001".to_owned(),
            autonomy_mode: AutonomyModeV1::ApplyApproved,
            base_artifact_sha256: digest('a'),
            clock_domain_id: "session-monotonic-v1".to_owned(),
            clock_origin_id: "origin-001".to_owned(),
            events: vec![
                event("evt-propose", 1, 10, AgencyEventKindV1::Proposed(proposal)),
                event("evt-preview", 2, 20, AgencyEventKindV1::Previewed(PreviewedV1 {
                    proposal_id: proposal_id.clone(),
                    subject_commitment_sha256: commitment.clone(),
                    preview_receipt_sha256: digest('1'),
                })),
                event("evt-approve", 3, 30, AgencyEventKindV1::Approved(ApprovedV1 {
                    proposal_id: proposal_id.clone(),
                    subject_commitment_sha256: commitment.clone(),
                    actor_id: "musician-01".to_owned(),
                    authority_receipt_sha256: digest('2'),
                    expires_at_monotonic_ns: 100,
                })),
                event("evt-commit", 4, 40, AgencyEventKindV1::Committed(CommittedV1 {
                    proposal_id: proposal_id.clone(),
                    subject_commitment_sha256: commitment.clone(),
                    authorization_event_id: "evt-approve".to_owned(),
                    result_artifact_sha256: digest('b'),
                    preservation_receipt_sha256: digest('3'),
                    preservation_verifier_id: "score-preservation-verifier".to_owned(),
                    preservation_verifier_version: "1.0.0".to_owned(),
                })),
                event("evt-undo", 5, 50, AgencyEventKindV1::Undone(UndoneV1 {
                    proposal_id,
                    subject_commitment_sha256: commitment,
                    commit_event_id: "evt-commit".to_owned(),
                    restored_artifact_sha256: digest('a'),
                    undo_receipt_sha256: digest('4'),
                    undo_verifier_id: "artifact-restore-verifier".to_owned(),
                    undo_verifier_version: "1.0.0".to_owned(),
                })),
            ],
        };
        seal_trace(&mut trace).unwrap();
        trace
    }

    fn proposed_preview_trace(mode: AutonomyModeV1) -> AgencyTraceV1 {
        let proposal = proposed();
        let commitment = proposal.subject_commitment_sha256.clone();
        let proposal_id = proposal.proposal_id.clone();
        let mut trace = AgencyTraceV1 {
            schema_version: TRACE_SCHEMA_VERSION.to_owned(),
            trace_id: "trace-preview".to_owned(),
            autonomy_mode: mode,
            base_artifact_sha256: digest('a'),
            clock_domain_id: "session-monotonic-v1".to_owned(),
            clock_origin_id: "origin-001".to_owned(),
            events: vec![
                event("evt-propose", 1, 10, AgencyEventKindV1::Proposed(proposal)),
                event("evt-preview", 2, 20, AgencyEventKindV1::Previewed(PreviewedV1 {
                    proposal_id,
                    subject_commitment_sha256: commitment,
                    preview_receipt_sha256: digest('1'),
                })),
            ],
        };
        seal_trace(&mut trace).unwrap();
        trace
    }

    #[test]
    fn approved_commit_and_exact_undo_pass() {
        let report = validate_trace(&apply_approved_trace());
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.terminal_disposition, "undone");
        assert!(report.committed && report.undone);
    }

    #[test]
    fn suggest_only_proposal_and_preview_are_non_mutating() {
        let report = validate_trace(&proposed_preview_trace(AutonomyModeV1::SuggestOnly));
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.terminal_disposition, "previewed");
        assert!(!report.committed);
    }

    #[test]
    fn preview_only_is_non_mutating() {
        let report = validate_trace(&proposed_preview_trace(AutonomyModeV1::PreviewOnly));
        assert!(report.valid, "{:?}", report.issues);
        assert!(!report.committed);
    }

    #[test]
    fn suggest_only_cannot_commit_even_with_an_approval_event() {
        let mut trace = apply_approved_trace();
        trace.autonomy_mode = AutonomyModeV1::SuggestOnly;
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "commit_forbidden_by_autonomy_mode"));
        assert!(report.issues.iter().any(|item| item.code == "approval_mode_mismatch"));
    }

    #[test]
    fn stale_approval_is_rejected_at_commit_time() {
        let mut trace = apply_approved_trace();
        if let AgencyEventKindV1::Approved(payload) = &mut trace.events[2].kind {
            payload.expires_at_monotonic_ns = 35;
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "expired_authorization"));
    }

    #[test]
    fn approval_must_bind_the_exact_proposal_commitment() {
        let mut trace = apply_approved_trace();
        if let AgencyEventKindV1::Approved(payload) = &mut trace.events[2].kind {
            payload.subject_commitment_sha256 = digest('9');
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "authorization_subject_mismatch"));
    }

    #[test]
    fn veto_revokes_prior_approval_and_blocks_commit() {
        let mut trace = apply_approved_trace();
        trace.events.truncate(3);
        let commitment = proposed().subject_commitment_sha256;
        trace.events.push(event("evt-veto", 4, 40, AgencyEventKindV1::Vetoed(VetoedV1 {
            proposal_id: "proposal-001".to_owned(),
            subject_commitment_sha256: commitment.clone(),
            actor_id: "musician-01".to_owned(),
            veto_receipt_sha256: digest('5'),
        })));
        trace.events.push(event("evt-commit", 5, 50, AgencyEventKindV1::Committed(CommittedV1 {
            proposal_id: "proposal-001".to_owned(),
            subject_commitment_sha256: commitment,
            authorization_event_id: "evt-approve".to_owned(),
            result_artifact_sha256: digest('b'),
            preservation_receipt_sha256: digest('3'),
            preservation_verifier_id: "score-preservation-verifier".to_owned(),
            preservation_verifier_version: "1.0.0".to_owned(),
        })));
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "vetoed_proposal_commit"));
    }

    #[test]
    fn no_op_candidate_is_rejected() {
        let mut trace = apply_approved_trace();
        let commitment = if let AgencyEventKindV1::Proposed(payload) = &mut trace.events[0].kind {
            payload.subject.candidate_artifact_sha256 = payload.subject.source_artifact_sha256.clone();
            payload.subject_commitment_sha256 = proposal_subject_sha256(&payload.subject).unwrap();
            payload.subject_commitment_sha256.clone()
        } else {
            unreachable!()
        };
        for event in &mut trace.events[1..] {
            match &mut event.kind {
                AgencyEventKindV1::Previewed(payload) => payload.subject_commitment_sha256 = commitment.clone(),
                AgencyEventKindV1::Approved(payload) => payload.subject_commitment_sha256 = commitment.clone(),
                AgencyEventKindV1::Committed(payload) => {
                    payload.subject_commitment_sha256 = commitment.clone();
                    payload.result_artifact_sha256 = digest('a');
                }
                AgencyEventKindV1::Undone(payload) => payload.subject_commitment_sha256 = commitment.clone(),
                _ => {}
            }
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "no_op_candidate"));
    }

    #[test]
    fn commit_requires_a_preservation_receipt_and_verifier() {
        let mut trace = apply_approved_trace();
        if let AgencyEventKindV1::Committed(payload) = &mut trace.events[3].kind {
            payload.preservation_receipt_sha256.clear();
            payload.preservation_verifier_id.clear();
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "invalid_sha256"));
        assert!(report.issues.iter().any(|item| item.code == "missing_preservation_verifier_identity"));
    }

    #[test]
    fn undo_must_restore_the_exact_source_artifact() {
        let mut trace = apply_approved_trace();
        if let AgencyEventKindV1::Undone(payload) = &mut trace.events[4].kind {
            payload.restored_artifact_sha256 = digest('9');
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "undo_not_exact"));
    }

    #[test]
    fn bounded_exploration_requires_a_subject_bound_policy_grant() {
        let proposal = proposed();
        let commitment = proposal.subject_commitment_sha256.clone();
        let proposal_id = proposal.proposal_id.clone();
        let mut trace = AgencyTraceV1 {
            schema_version: TRACE_SCHEMA_VERSION.to_owned(),
            trace_id: "trace-bounded".to_owned(),
            autonomy_mode: AutonomyModeV1::BoundedExploration,
            base_artifact_sha256: digest('a'),
            clock_domain_id: "session-monotonic-v1".to_owned(),
            clock_origin_id: "origin-001".to_owned(),
            events: vec![
                event("evt-propose", 1, 10, AgencyEventKindV1::Proposed(proposal)),
                event("evt-preview", 2, 20, AgencyEventKindV1::Previewed(PreviewedV1 {
                    proposal_id: proposal_id.clone(),
                    subject_commitment_sha256: commitment.clone(),
                    preview_receipt_sha256: digest('1'),
                })),
                event("evt-grant", 3, 30, AgencyEventKindV1::BoundedAuthorityGranted(BoundedAuthorityGrantedV1 {
                    proposal_id: proposal_id.clone(),
                    subject_commitment_sha256: commitment.clone(),
                    actor_id: "musician-01".to_owned(),
                    policy_sha256: digest('6'),
                    authority_receipt_sha256: digest('2'),
                    expires_at_monotonic_ns: 100,
                })),
                event("evt-commit", 4, 40, AgencyEventKindV1::Committed(CommittedV1 {
                    proposal_id,
                    subject_commitment_sha256: commitment,
                    authorization_event_id: "evt-grant".to_owned(),
                    result_artifact_sha256: digest('b'),
                    preservation_receipt_sha256: digest('3'),
                    preservation_verifier_id: "score-preservation-verifier".to_owned(),
                    preservation_verifier_version: "1.0.0".to_owned(),
                })),
            ],
        };
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.terminal_disposition, "committed");
    }

    #[test]
    fn tampered_event_is_detected_by_the_hash_chain() {
        let mut trace = apply_approved_trace();
        if let AgencyEventKindV1::Previewed(payload) = &mut trace.events[1].kind {
            payload.preview_receipt_sha256 = digest('9');
        }
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "event_hash_mismatch"));
    }

    #[test]
    fn commit_must_reference_the_exact_authorization_event() {
        let mut trace = apply_approved_trace();
        if let AgencyEventKindV1::Committed(payload) = &mut trace.events[3].kind {
            payload.authorization_event_id = "some-other-approval".to_owned();
        }
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "authorization_reference_mismatch"));
    }

    #[test]
    fn trace_rejects_non_contiguous_sequences_and_duplicate_event_ids() {
        let mut trace = apply_approved_trace();
        trace.events[1].sequence = 4;
        trace.events[2].event_id = trace.events[1].event_id.clone();
        seal_trace(&mut trace).unwrap();
        let report = validate_trace(&trace);
        assert!(!report.valid);
        assert!(report.issues.iter().any(|item| item.code == "non_contiguous_sequence"));
        assert!(report.issues.iter().any(|item| item.code == "duplicate_event_id"));
    }
}
