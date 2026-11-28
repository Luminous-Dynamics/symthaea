#![forbid(unsafe_code)]

//! Execution-evidence contract for personal-agent interaction preferences.
//!
//! Consumes a selection from the existing #13153 action policy. This crate is
//! not a preference store, selector, consent authority, tool authorizer,
//! semantic judge, or memory backend.
//!
//! Digests are integrity commitments, not signatures. Production must inject
//! verifiers that authenticate selection/source authority and the executor's
//! complete append-only event journal. A response hash proves byte identity,
//! not that prose semantically follows a style preference.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION_V1: &str = "mace_interaction_execution_receipt_v1";

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, InteractionExecutionRejectV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| InteractionExecutionRejectV1::SerializationFailure)?;
    Ok(digest_bytes(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionExecutionContextV1 {
    pub request_sha256: String,
    pub subject_id: String,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub valid_at_ms: u64,
    pub known_at_ms: u64,
    pub policy_sha256: String,
}

impl InteractionExecutionContextV1 {
    pub fn validate(&self) -> Result<(), InteractionExecutionRejectV1> {
        if !valid_sha256(&self.request_sha256)
            || !valid_sha256(&self.policy_sha256)
            || !nonempty(&self.subject_id)
            || !nonempty(&self.scope_id)
            || !nonempty(&self.purpose_id)
            || !nonempty(&self.audience_id)
        {
            return Err(InteractionExecutionRejectV1::InvalidContext);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BehaviorRequirementV1 {
    ControlEquals {
        control_id: String,
        setting_sha256: String,
    },
    ActionMustOccur {
        action_class_sha256: String,
    },
    ActionMustNotOccur {
        action_class_sha256: String,
        window_start_ms: u64,
        window_end_ms: u64,
    },
}

impl BehaviorRequirementV1 {
    fn validate(&self) -> Result<(), InteractionExecutionRejectV1> {
        match self {
            Self::ControlEquals { control_id, setting_sha256 } => {
                if !nonempty(control_id) || !valid_sha256(setting_sha256) {
                    return Err(InteractionExecutionRejectV1::InvalidRequirement);
                }
            }
            Self::ActionMustOccur { action_class_sha256 } => {
                if !valid_sha256(action_class_sha256) {
                    return Err(InteractionExecutionRejectV1::InvalidRequirement);
                }
            }
            Self::ActionMustNotOccur {
                action_class_sha256, window_start_ms, window_end_ms
            } => {
                if !valid_sha256(action_class_sha256) || window_start_ms >= window_end_ms {
                    return Err(InteractionExecutionRejectV1::InvalidRequirement);
                }
            }
        }
        Ok(())
    }

    fn key(&self) -> String {
        match self {
            Self::ControlEquals { control_id, setting_sha256 } =>
                format!("control:{control_id}:{setting_sha256}"),
            Self::ActionMustOccur { action_class_sha256 } =>
                format!("action:must:{action_class_sha256}"),
            Self::ActionMustNotOccur {
                action_class_sha256, window_start_ms, window_end_ms
            } => format!("action:must_not:{action_class_sha256}:{window_start_ms}:{window_end_ms}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedInteractionPreferenceV1 {
    pub preference_id: String,
    pub dimension_id: String,
    pub setting_sha256: String,
    pub source_evidence_receipt_sha256: String,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    pub known_from_ms: u64,
    pub known_until_ms: u64,
    pub requirements: Vec<BehaviorRequirementV1>,
}

impl SelectedInteractionPreferenceV1 {
    fn validate(
        &self,
        context: &InteractionExecutionContextV1,
    ) -> Result<(), InteractionExecutionRejectV1> {
        if !nonempty(&self.preference_id)
            || !nonempty(&self.dimension_id)
            || !valid_sha256(&self.setting_sha256)
            || !valid_sha256(&self.source_evidence_receipt_sha256)
            || self.valid_from_ms >= self.valid_until_ms
            || self.known_from_ms >= self.known_until_ms
        {
            return Err(InteractionExecutionRejectV1::InvalidPreference);
        }
        if self.scope_id != context.scope_id
            || self.purpose_id != context.purpose_id
            || self.audience_id != context.audience_id
        {
            return Err(InteractionExecutionRejectV1::PreferenceContextMismatch);
        }
        if !(self.valid_from_ms <= context.valid_at_ms
            && context.valid_at_ms < self.valid_until_ms)
            || !(self.known_from_ms <= context.known_at_ms
                && context.known_at_ms < self.known_until_ms)
        {
            return Err(InteractionExecutionRejectV1::PreferenceNotCurrent);
        }
        if self.requirements.is_empty() {
            return Err(InteractionExecutionRejectV1::NoBehaviorRequirements);
        }
        let mut requirement_keys = BTreeSet::new();
        for requirement in &self.requirements {
            requirement.validate()?;
            if !requirement_keys.insert(requirement.key()) {
                return Err(InteractionExecutionRejectV1::DuplicateRequirement);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionSelectionV1 {
    pub selection_id: String,
    pub subject_id: String,
    pub context_sha256: String,
    pub decision_policy_sha256: String,
    pub selection_authority_receipt_sha256: String,
    pub preferences: Vec<SelectedInteractionPreferenceV1>,
    pub selection_sha256: String,
}

fn selection_digest(selection: &InteractionSelectionV1) -> Result<String, InteractionExecutionRejectV1> {
    let mut unsigned = selection.clone();
    unsigned.selection_sha256.clear();
    digest_serialized(&unsigned)
}

impl InteractionSelectionV1 {
    pub fn seal_integrity(mut self) -> Result<Self, InteractionExecutionRejectV1> {
        if !nonempty(&self.selection_id)
            || !nonempty(&self.subject_id)
            || !valid_sha256(&self.context_sha256)
            || !valid_sha256(&self.decision_policy_sha256)
            || !valid_sha256(&self.selection_authority_receipt_sha256)
        {
            return Err(InteractionExecutionRejectV1::InvalidSelection);
        }
        self.selection_sha256.clear();
        self.selection_sha256 = selection_digest(&self)?;
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        valid_sha256(&self.selection_sha256)
            && selection_digest(self).is_ok_and(|digest| digest == self.selection_sha256)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InteractionExecutionEventKindV1 {
    ControlApplied { control_id: String, setting_sha256: String },
    ActionEmitted { action_class_sha256: String },
    NoActionWindowClosed {
        action_class_sha256: String,
        window_start_ms: u64,
        window_end_ms: u64,
        observed_through_ms: u64,
        action_events_sha256: String,
    },
    ResponseCommitted { response_sha256: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionExecutionEventV1 {
    pub event_id: String,
    pub selection_id: String,
    pub subject_id: String,
    pub occurred_at_ms: u64,
    pub source_receipt_sha256: String,
    pub kind: InteractionExecutionEventKindV1,
}

impl InteractionExecutionEventV1 {
    fn validate(&self, selection: &InteractionSelectionV1) -> Result<(), InteractionExecutionRejectV1> {
        if !nonempty(&self.event_id)
            || !valid_sha256(&self.source_receipt_sha256)
            || self.selection_id != selection.selection_id
            || self.subject_id != selection.subject_id
        {
            return Err(InteractionExecutionRejectV1::InvalidExecutionEvent);
        }
        match &self.kind {
            InteractionExecutionEventKindV1::ControlApplied { control_id, setting_sha256 } => {
                if !nonempty(control_id) || !valid_sha256(setting_sha256) {
                    return Err(InteractionExecutionRejectV1::InvalidExecutionEvent);
                }
            }
            InteractionExecutionEventKindV1::ActionEmitted { action_class_sha256 } => {
                if !valid_sha256(action_class_sha256) {
                    return Err(InteractionExecutionRejectV1::InvalidExecutionEvent);
                }
            }
            InteractionExecutionEventKindV1::NoActionWindowClosed {
                action_class_sha256, window_start_ms, window_end_ms,
                observed_through_ms, action_events_sha256
            } => {
                if !valid_sha256(action_class_sha256)
                    || !valid_sha256(action_events_sha256)
                    || window_start_ms >= window_end_ms
                    || observed_through_ms < *window_end_ms
                    || self.occurred_at_ms < *window_end_ms
                {
                    return Err(InteractionExecutionRejectV1::InvalidExecutionEvent);
                }
            }
            InteractionExecutionEventKindV1::ResponseCommitted { response_sha256 } => {
                if !valid_sha256(response_sha256) {
                    return Err(InteractionExecutionRejectV1::InvalidExecutionEvent);
                }
            }
        }
        Ok(())
    }
}

/// Production implementations authenticate source authority and the complete
/// executor event journal. No-action coverage must establish that the journal
/// is complete through the requested window; an absent event in an incomplete
/// log is not evidence that the action did not occur.
pub trait InteractionExecutionVerifierV1 {
    fn verify_selection(
        &self,
        selection: &InteractionSelectionV1,
        context: &InteractionExecutionContextV1,
    ) -> Result<(), InteractionExecutionRejectV1>;

    fn verify_execution_event(
        &self,
        event: &InteractionExecutionEventV1,
        selection: &InteractionSelectionV1,
        context: &InteractionExecutionContextV1,
    ) -> Result<(), InteractionExecutionRejectV1>;

    fn verify_no_action_window_coverage(
        &self,
        closure_event: &InteractionExecutionEventV1,
        expected_action_events_sha256: &str,
        selection: &InteractionSelectionV1,
        context: &InteractionExecutionContextV1,
    ) -> Result<(), InteractionExecutionRejectV1>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionExecutionReceiptV1 {
    pub schema_version: String,
    pub selection_id: String,
    pub request_sha256: String,
    pub response_sha256: String,
    pub context_sha256: String,
    pub selection_sha256: String,
    pub event_manifest_sha256: String,
    pub executor_build_sha256: String,
    pub receipt_sha256: String,
}

fn receipt_digest(receipt: &InteractionExecutionReceiptV1) -> Result<String, InteractionExecutionRejectV1> {
    let mut unsigned = receipt.clone();
    unsigned.receipt_sha256.clear();
    digest_serialized(&unsigned)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InteractionExecutionRejectV1 {
    InvalidContext,
    InvalidRequirement,
    InvalidPreference,
    PreferenceContextMismatch,
    PreferenceNotCurrent,
    NoBehaviorRequirements,
    InvalidSelection,
    SelectionIntegrityMismatch,
    SelectionContextMismatch,
    DuplicatePreferenceId,
    DuplicateRequirement,
    InvalidExecutionEvent,
    NonCanonicalEventOrder,
    DuplicateEventId,
    ExecutionVerifierRejected,
    MissingRequiredControl,
    MissingRequiredAction,
    EventAfterResponseCommit,
    ProhibitedActionOccurred,
    MissingNoActionCoverage,
    IncompleteNoActionCoverage,
    MissingResponseCommit,
    DuplicateResponseCommit,
    ResponseDigestMismatch,
    RequestDigestMismatch,
    ContextDigestMismatch,
    EventManifestDigestMismatch,
    ExecutorBuildDigestMismatch,
    ReceiptIntegrityMismatch,
    InvalidReceipt,
    SerializationFailure,
}

impl InteractionExecutionRejectV1 {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidContext => "invalid_context",
            Self::InvalidRequirement => "invalid_requirement",
            Self::InvalidPreference => "invalid_preference",
            Self::PreferenceContextMismatch => "preference_context_mismatch",
            Self::PreferenceNotCurrent => "preference_not_current",
            Self::NoBehaviorRequirements => "no_behavior_requirements",
            Self::InvalidSelection => "invalid_selection",
            Self::SelectionIntegrityMismatch => "selection_integrity_mismatch",
            Self::SelectionContextMismatch => "selection_context_mismatch",
            Self::DuplicatePreferenceId => "duplicate_preference_id",
            Self::DuplicateRequirement => "duplicate_requirement",
            Self::InvalidExecutionEvent => "invalid_execution_event",
            Self::NonCanonicalEventOrder => "non_canonical_event_order",
            Self::DuplicateEventId => "duplicate_event_id",
            Self::ExecutionVerifierRejected => "execution_verifier_rejected",
            Self::MissingRequiredControl => "missing_required_control",
            Self::MissingRequiredAction => "missing_required_action",
            Self::EventAfterResponseCommit => "event_after_response_commit",
            Self::ProhibitedActionOccurred => "prohibited_action_occurred",
            Self::MissingNoActionCoverage => "missing_no_action_coverage",
            Self::IncompleteNoActionCoverage => "incomplete_no_action_coverage",
            Self::MissingResponseCommit => "missing_response_commit",
            Self::DuplicateResponseCommit => "duplicate_response_commit",
            Self::ResponseDigestMismatch => "response_digest_mismatch",
            Self::RequestDigestMismatch => "request_digest_mismatch",
            Self::ContextDigestMismatch => "context_digest_mismatch",
            Self::EventManifestDigestMismatch => "event_manifest_digest_mismatch",
            Self::ExecutorBuildDigestMismatch => "executor_build_digest_mismatch",
            Self::ReceiptIntegrityMismatch => "receipt_integrity_mismatch",
            Self::InvalidReceipt => "invalid_receipt",
            Self::SerializationFailure => "serialization_failure",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedInteractionExecutionV1 {
    receipt_sha256: String,
    response_sha256: String,
    event_manifest_sha256: String,
}

impl VerifiedInteractionExecutionV1 {
    pub fn receipt_sha256(&self) -> &str { &self.receipt_sha256 }
    pub fn response_sha256(&self) -> &str { &self.response_sha256 }
    pub fn event_manifest_sha256(&self) -> &str { &self.event_manifest_sha256 }
}

fn occurs_after_response_commit(
    event: &InteractionExecutionEventV1,
    response_commit: &InteractionExecutionEventV1,
) -> bool {
    (event.occurred_at_ms, event.event_id.as_str())
        > (response_commit.occurred_at_ms, response_commit.event_id.as_str())
}

fn canonical_events_digest(
    events: &[InteractionExecutionEventV1],
) -> Result<String, InteractionExecutionRejectV1> {
    let mut previous: Option<(u64, &str)> = None;
    let mut ids = BTreeSet::new();
    for event in events {
        if !ids.insert(event.event_id.as_str()) {
            return Err(InteractionExecutionRejectV1::DuplicateEventId);
        }
        let current = (event.occurred_at_ms, event.event_id.as_str());
        if previous.is_some_and(|prior| prior > current) {
            return Err(InteractionExecutionRejectV1::NonCanonicalEventOrder);
        }
        previous = Some(current);
    }
    digest_serialized(events)
}

fn action_events_digest_for_window(
    events: &[InteractionExecutionEventV1],
    action_class_sha256: &str,
    window_start_ms: u64,
    window_end_ms: u64,
) -> Result<String, InteractionExecutionRejectV1> {
    let actions: Vec<&InteractionExecutionEventV1> = events.iter().filter(|event| {
        event.occurred_at_ms >= window_start_ms
            && event.occurred_at_ms < window_end_ms
            && matches!(
                &event.kind,
                InteractionExecutionEventKindV1::ActionEmitted { action_class_sha256: actual }
                    if actual == action_class_sha256
            )
    }).collect();
    digest_serialized(&actions)
}

pub fn seal_interaction_execution_receipt_v1(
    selection: &InteractionSelectionV1,
    context: &InteractionExecutionContextV1,
    request_bytes: &[u8],
    response_bytes: &[u8],
    events: &[InteractionExecutionEventV1],
    executor_build_sha256: &str,
) -> Result<InteractionExecutionReceiptV1, InteractionExecutionRejectV1> {
    context.validate()?;
    if !selection.verify_integrity() {
        return Err(InteractionExecutionRejectV1::SelectionIntegrityMismatch);
    }
    if !valid_sha256(executor_build_sha256) {
        return Err(InteractionExecutionRejectV1::ExecutorBuildDigestMismatch);
    }
    let event_manifest_sha256 = canonical_events_digest(events)?;
    let mut receipt = InteractionExecutionReceiptV1 {
        schema_version: SCHEMA_VERSION_V1.to_owned(),
        selection_id: selection.selection_id.clone(),
        request_sha256: digest_bytes(request_bytes),
        response_sha256: digest_bytes(response_bytes),
        context_sha256: digest_serialized(context)?,
        selection_sha256: selection.selection_sha256.clone(),
        event_manifest_sha256,
        executor_build_sha256: executor_build_sha256.to_owned(),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    Ok(receipt)
}

pub fn verify_interaction_execution_v1<V: InteractionExecutionVerifierV1>(
    receipt: &InteractionExecutionReceiptV1,
    selection: &InteractionSelectionV1,
    context: &InteractionExecutionContextV1,
    request_bytes: &[u8],
    response_bytes: &[u8],
    events: &[InteractionExecutionEventV1],
    expected_executor_build_sha256: &str,
    verifier: &V,
) -> Result<VerifiedInteractionExecutionV1, InteractionExecutionRejectV1> {
    context.validate()?;
    if !selection.verify_integrity() {
        return Err(InteractionExecutionRejectV1::SelectionIntegrityMismatch);
    }
    if selection.subject_id != context.subject_id {
        return Err(InteractionExecutionRejectV1::SelectionContextMismatch);
    }
    let context_sha256 = digest_serialized(context)?;
    if selection.context_sha256 != context_sha256 {
        return Err(InteractionExecutionRejectV1::SelectionContextMismatch);
    }
    verifier.verify_selection(selection, context)
        .map_err(|_| InteractionExecutionRejectV1::ExecutionVerifierRejected)?;

    let mut preference_ids = BTreeSet::new();
    for preference in &selection.preferences {
        preference.validate(context)?;
        if !preference_ids.insert(preference.preference_id.as_str()) {
            return Err(InteractionExecutionRejectV1::DuplicatePreferenceId);
        }
    }

    for event in events {
        event.validate(selection)?;
        verifier.verify_execution_event(event, selection, context)
            .map_err(|_| InteractionExecutionRejectV1::ExecutionVerifierRejected)?;
    }
    let event_manifest_sha256 = canonical_events_digest(events)?;
    let expected_response_sha256 = digest_bytes(response_bytes);
    let mut response_commit_count = 0usize;
    for event in events {
        if let InteractionExecutionEventKindV1::ResponseCommitted { response_sha256 } = &event.kind {
            response_commit_count += 1;
            if response_sha256 != &expected_response_sha256 {
                return Err(InteractionExecutionRejectV1::ResponseDigestMismatch);
            }
        }
    }
    if response_commit_count == 0 {
        return Err(InteractionExecutionRejectV1::MissingResponseCommit);
    }
    if response_commit_count > 1 {
        return Err(InteractionExecutionRejectV1::DuplicateResponseCommit);
    }
    let response_commit_event = events
        .iter()
        .find(|event| {
            matches!(
                &event.kind,
                InteractionExecutionEventKindV1::ResponseCommitted { .. }
            )
        })
        .ok_or(InteractionExecutionRejectV1::MissingResponseCommit)?;

    for preference in &selection.preferences {
        for requirement in &preference.requirements {
            match requirement {
                BehaviorRequirementV1::ControlEquals { control_id, setting_sha256 } => {
                    let matching_controls: Vec<&InteractionExecutionEventV1> = events.iter().filter(|event| {
                        matches!(
                            &event.kind,
                            InteractionExecutionEventKindV1::ControlApplied {
                                control_id: actual_control, setting_sha256: actual_setting
                            } if actual_control == control_id && actual_setting == setting_sha256
                        )
                    }).collect();
                    if matching_controls.iter().any(|event| occurs_after_response_commit(event, response_commit_event)) {
                        return Err(InteractionExecutionRejectV1::EventAfterResponseCommit);
                    }
                    if !matching_controls.iter().any(|event| !occurs_after_response_commit(event, response_commit_event)) {
                        return Err(InteractionExecutionRejectV1::MissingRequiredControl);
                    }
                }
                BehaviorRequirementV1::ActionMustOccur { action_class_sha256 } => {
                    let matching_actions: Vec<&InteractionExecutionEventV1> = events.iter().filter(|event| {
                        matches!(
                            &event.kind,
                            InteractionExecutionEventKindV1::ActionEmitted { action_class_sha256: actual }
                                if actual == action_class_sha256
                        )
                    }).collect();
                    if matching_actions.iter().any(|event| occurs_after_response_commit(event, response_commit_event)) {
                        return Err(InteractionExecutionRejectV1::EventAfterResponseCommit);
                    }
                    if !matching_actions.iter().any(|event| !occurs_after_response_commit(event, response_commit_event)) {
                        return Err(InteractionExecutionRejectV1::MissingRequiredAction);
                    }
                }
                BehaviorRequirementV1::ActionMustNotOccur {
                    action_class_sha256, window_start_ms, window_end_ms
                } => {
                    let occurred = events.iter().any(|event| {
                        event.occurred_at_ms >= *window_start_ms
                            && event.occurred_at_ms < *window_end_ms
                            && matches!(
                                &event.kind,
                                InteractionExecutionEventKindV1::ActionEmitted { action_class_sha256: actual }
                                    if actual == action_class_sha256
                            )
                    });
                    if occurred {
                        return Err(InteractionExecutionRejectV1::ProhibitedActionOccurred);
                    }
                    let expected_action_events_sha256 = action_events_digest_for_window(
                        events, action_class_sha256, *window_start_ms, *window_end_ms,
                    )?;
                    let matching_closures: Vec<&InteractionExecutionEventV1> = events.iter().filter(|event| {
                        matches!(
                            &event.kind,
                            InteractionExecutionEventKindV1::NoActionWindowClosed {
                                action_class_sha256: actual_class,
                                window_start_ms: actual_start,
                                window_end_ms: actual_end,
                                observed_through_ms,
                                action_events_sha256
                            } if actual_class == action_class_sha256
                                && actual_start == window_start_ms
                                && actual_end == window_end_ms
                                && observed_through_ms >= window_end_ms
                                && action_events_sha256 == &expected_action_events_sha256
                        )
                    }).collect();
                    if matching_closures.iter().any(|event| occurs_after_response_commit(event, response_commit_event)) {
                        return Err(InteractionExecutionRejectV1::EventAfterResponseCommit);
                    }
                    let closure = matching_closures.iter()
                        .copied()
                        .find(|event| !occurs_after_response_commit(event, response_commit_event))
                        .ok_or(InteractionExecutionRejectV1::MissingNoActionCoverage)?;
                    verifier.verify_no_action_window_coverage(
                        closure, &expected_action_events_sha256, selection, context,
                    ).map_err(|_| InteractionExecutionRejectV1::IncompleteNoActionCoverage)?;
                }
            }
        }
    }

    if receipt.schema_version != SCHEMA_VERSION_V1 || receipt.selection_id != selection.selection_id {
        return Err(InteractionExecutionRejectV1::InvalidReceipt);
    }
    if receipt.request_sha256 != digest_bytes(request_bytes) {
        return Err(InteractionExecutionRejectV1::RequestDigestMismatch);
    }
    if receipt.response_sha256 != expected_response_sha256 {
        return Err(InteractionExecutionRejectV1::ResponseDigestMismatch);
    }
    if receipt.context_sha256 != context_sha256 {
        return Err(InteractionExecutionRejectV1::ContextDigestMismatch);
    }
    if receipt.selection_sha256 != selection.selection_sha256 {
        return Err(InteractionExecutionRejectV1::SelectionIntegrityMismatch);
    }
    if receipt.event_manifest_sha256 != event_manifest_sha256 {
        return Err(InteractionExecutionRejectV1::EventManifestDigestMismatch);
    }
    if receipt.executor_build_sha256 != expected_executor_build_sha256
        || !valid_sha256(&receipt.executor_build_sha256)
    {
        return Err(InteractionExecutionRejectV1::ExecutorBuildDigestMismatch);
    }
    if !valid_sha256(&receipt.receipt_sha256)
        || receipt_digest(receipt)? != receipt.receipt_sha256
    {
        return Err(InteractionExecutionRejectV1::ReceiptIntegrityMismatch);
    }

    Ok(VerifiedInteractionExecutionV1 {
        receipt_sha256: receipt.receipt_sha256.clone(),
        response_sha256: receipt.response_sha256.clone(),
        event_manifest_sha256: receipt.event_manifest_sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Default)]
    struct FixtureVerifier;

    impl InteractionExecutionVerifierV1 for FixtureVerifier {
        fn verify_selection(
            &self,
            selection: &InteractionSelectionV1,
            _context: &InteractionExecutionContextV1,
        ) -> Result<(), InteractionExecutionRejectV1> {
            if selection.selection_authority_receipt_sha256 == sha("selection-authority") {
                Ok(())
            } else {
                Err(InteractionExecutionRejectV1::ExecutionVerifierRejected)
            }
        }

        fn verify_execution_event(
            &self,
            event: &InteractionExecutionEventV1,
            _selection: &InteractionSelectionV1,
            _context: &InteractionExecutionContextV1,
        ) -> Result<(), InteractionExecutionRejectV1> {
            if event.source_receipt_sha256 == sha("executor-event") {
                Ok(())
            } else {
                Err(InteractionExecutionRejectV1::ExecutionVerifierRejected)
            }
        }

        fn verify_no_action_window_coverage(
            &self,
            event: &InteractionExecutionEventV1,
            expected_action_events_sha256: &str,
            _selection: &InteractionSelectionV1,
            _context: &InteractionExecutionContextV1,
        ) -> Result<(), InteractionExecutionRejectV1> {
            match &event.kind {
                InteractionExecutionEventKindV1::NoActionWindowClosed {
                    action_events_sha256, observed_through_ms, window_end_ms, ..
                } if action_events_sha256 == expected_action_events_sha256
                    && observed_through_ms >= window_end_ms => Ok(()),
                _ => Err(InteractionExecutionRejectV1::IncompleteNoActionCoverage),
            }
        }
    }

    fn sha(value: &str) -> String { digest_bytes(value.as_bytes()) }

    fn context() -> InteractionExecutionContextV1 {
        InteractionExecutionContextV1 {
            request_sha256: sha("request"),
            subject_id: "user:alice".into(),
            scope_id: "workspace:alpha".into(),
            purpose_id: "assist:calendar".into(),
            audience_id: "user:alice".into(),
            valid_at_ms: 100,
            known_at_ms: 100,
            policy_sha256: sha("policy"),
        }
    }

    fn pref(requirements: Vec<BehaviorRequirementV1>) -> SelectedInteractionPreferenceV1 {
        SelectedInteractionPreferenceV1 {
            preference_id: "pref:outreach".into(),
            dimension_id: "proactive_outreach".into(),
            setting_sha256: sha("low"),
            source_evidence_receipt_sha256: sha("preference-evidence"),
            scope_id: "workspace:alpha".into(),
            purpose_id: "assist:calendar".into(),
            audience_id: "user:alice".into(),
            valid_from_ms: 0,
            valid_until_ms: 1_000,
            known_from_ms: 0,
            known_until_ms: 1_000,
            requirements,
        }
    }

    fn selection(preferences: Vec<SelectedInteractionPreferenceV1>) -> InteractionSelectionV1 {
        let ctx = context();
        InteractionSelectionV1 {
            selection_id: "selection:1".into(),
            subject_id: ctx.subject_id.clone(),
            context_sha256: digest_serialized(&ctx).unwrap(),
            decision_policy_sha256: sha("decision-policy"),
            selection_authority_receipt_sha256: sha("selection-authority"),
            preferences,
            selection_sha256: String::new(),
        }.seal_integrity().unwrap()
    }

    fn event(
        event_id: &str,
        occurred_at_ms: u64,
        kind: InteractionExecutionEventKindV1,
    ) -> InteractionExecutionEventV1 {
        InteractionExecutionEventV1 {
            event_id: event_id.into(),
            selection_id: "selection:1".into(),
            subject_id: "user:alice".into(),
            occurred_at_ms,
            source_receipt_sha256: sha("executor-event"),
            kind,
        }
    }

    fn response_event(response: &[u8]) -> InteractionExecutionEventV1 {
        event("event:response", 300,
            InteractionExecutionEventKindV1::ResponseCommitted {
                response_sha256: digest_bytes(response),
            })
    }

    fn seal(
        selection: &InteractionSelectionV1,
        events: &[InteractionExecutionEventV1],
        response: &[u8],
    ) -> InteractionExecutionReceiptV1 {
        seal_interaction_execution_receipt_v1(
            selection, &context(), b"request", response, events, &sha("executor-build"),
        ).unwrap()
    }

    fn verify(
        selection: &InteractionSelectionV1,
        events: &[InteractionExecutionEventV1],
        response: &[u8],
        receipt: &InteractionExecutionReceiptV1,
    ) -> Result<VerifiedInteractionExecutionV1, InteractionExecutionRejectV1> {
        verify_interaction_execution_v1(
            receipt, selection, &context(), b"request", response, events,
            &sha("executor-build"), &FixtureVerifier,
        )
    }

    fn sort_events(events: &mut [InteractionExecutionEventV1]) {
        events.sort_by(|a, b| {
            (a.occurred_at_ms, a.event_id.as_str())
                .cmp(&(b.occurred_at_ms, b.event_id.as_str()))
        });
    }

    #[test]
    fn selected_control_is_bound_to_actual_executor_event_and_response() {
        let response = b"I will keep this concise.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }])]);
        let events = vec![
            event("event:control", 200, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        let verified = verify(&selection, &events, response, &receipt).unwrap();
        assert_eq!(verified.response_sha256(), digest_bytes(response));
        assert_eq!(verified.receipt_sha256(), receipt.receipt_sha256);
    }

    #[test]
    fn prohibited_outreach_requires_a_complete_no_action_window() {
        let response = b"I will wait until you ask.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustNotOccur {
            action_class_sha256: sha("outbound_message"), window_start_ms: 100, window_end_ms: 300,
        }])]);
        let empty_action_digest = digest_serialized(&Vec::<&InteractionExecutionEventV1>::new()).unwrap();
        let events = vec![
            event("event:no-outreach-window", 300,
                InteractionExecutionEventKindV1::NoActionWindowClosed {
                    action_class_sha256: sha("outbound_message"),
                    window_start_ms: 100, window_end_ms: 300, observed_through_ms: 300,
                    action_events_sha256: empty_action_digest,
                }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert!(verify(&selection, &events, response, &receipt).is_ok());
    }

    #[test]
    fn actual_prohibited_outreach_fails_even_with_a_window_closure() {
        let response = b"I will wait.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustNotOccur {
            action_class_sha256: sha("outbound_message"), window_start_ms: 100, window_end_ms: 300,
        }])]);
        let actions = vec![event("event:outreach", 200,
            InteractionExecutionEventKindV1::ActionEmitted {
                action_class_sha256: sha("outbound_message"),
            })];
        let action_digest = digest_serialized(&actions.iter().collect::<Vec<_>>()).unwrap();
        let mut events = actions;
        events.push(event("event:no-outreach-window", 300,
            InteractionExecutionEventKindV1::NoActionWindowClosed {
                action_class_sha256: sha("outbound_message"),
                window_start_ms: 100, window_end_ms: 300, observed_through_ms: 300,
                action_events_sha256: action_digest,
            }));
        events.push(response_event(response));
        sort_events(&mut events);
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::ProhibitedActionOccurred));
    }

    #[test]
    fn control_applied_after_response_cannot_retroactively_qualify_it() {
        let response = b"I will keep it concise.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }])]);
        let events = vec![
            response_event(response),
            event("event:late-control", 310, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
        ];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::EventAfterResponseCommit));
    }

    #[test]
    fn required_action_after_response_cannot_retroactively_qualify_it() {
        let response = b"Calendar event created.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustOccur {
            action_class_sha256: sha("calendar_create"),
        }])]);
        let events = vec![
            response_event(response),
            event("event:late-calendar", 310, InteractionExecutionEventKindV1::ActionEmitted {
                action_class_sha256: sha("calendar_create"),
            }),
        ];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::EventAfterResponseCommit));
    }

    #[test]
    fn same_timestamp_control_ordered_after_response_cannot_retroactively_qualify_it() {
        let response = b"I will keep it concise.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }])]);
        let mut events = vec![
            response_event(response),
            event("zz:late-control", 300, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
        ];
        sort_events(&mut events);
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::EventAfterResponseCommit));
    }

    #[test]
    fn same_timestamp_no_action_closure_ordered_after_response_is_not_coverage() {
        let response = b"I will wait until you ask.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustNotOccur {
            action_class_sha256: sha("outbound_message"),
            window_start_ms: 100,
            window_end_ms: 300,
        }])]);
        let empty_digest = digest_serialized(&Vec::<&InteractionExecutionEventV1>::new()).unwrap();
        let mut events = vec![
            response_event(response),
            event("zz:no-action-window", 300,
                InteractionExecutionEventKindV1::NoActionWindowClosed {
                    action_class_sha256: sha("outbound_message"),
                    window_start_ms: 100,
                    window_end_ms: 300,
                    observed_through_ms: 300,
                    action_events_sha256: empty_digest,
                }),
        ];
        sort_events(&mut events);
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::EventAfterResponseCommit));
    }

    #[test]
    fn missing_no_action_coverage_does_not_prove_silence() {
        let response = b"No message sent.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustNotOccur {
            action_class_sha256: sha("outbound_message"), window_start_ms: 100, window_end_ms: 300,
        }])]);
        let events = vec![response_event(response)];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::MissingNoActionCoverage));
    }

    #[test]
    fn incomplete_observation_window_fails_closed() {
        let response = b"I will wait.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustNotOccur {
            action_class_sha256: sha("outbound_message"), window_start_ms: 100, window_end_ms: 300,
        }])]);
        let events = vec![
            event("event:window", 250,
                InteractionExecutionEventKindV1::NoActionWindowClosed {
                    action_class_sha256: sha("outbound_message"),
                    window_start_ms: 100, window_end_ms: 300, observed_through_ms: 250,
                    action_events_sha256: digest_serialized(&Vec::<&InteractionExecutionEventV1>::new()).unwrap(),
                }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::InvalidExecutionEvent));
    }

    #[test]
    fn action_must_occur_requires_a_matching_committed_action() {
        let response = b"Calendar event created.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustOccur {
            action_class_sha256: sha("calendar_create"),
        }])]);
        let events = vec![response_event(response)];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::MissingRequiredAction));
    }

    #[test]
    fn required_tool_action_is_accepted_with_executor_receipt() {
        let response = b"Calendar event created.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ActionMustOccur {
            action_class_sha256: sha("calendar_create"),
        }])]);
        let events = vec![
            event("event:calendar", 200,
                InteractionExecutionEventKindV1::ActionEmitted {
                    action_class_sha256: sha("calendar_create"),
                }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert!(verify(&selection, &events, response, &receipt).is_ok());
    }

    #[test]
    fn wrong_scope_or_audience_cannot_reuse_a_preference() {
        let response = b"Hello.";
        let mut preference = pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }]);
        preference.scope_id = "workspace:other".into();
        let selection = selection(vec![preference]);
        let events = vec![
            event("event:control", 200, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::PreferenceContextMismatch));
    }

    #[test]
    fn expired_or_not_yet_known_preference_cannot_be_executed() {
        let response = b"Hello.";
        let mut preference = pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }]);
        preference.known_from_ms = 150;
        let selection = selection(vec![preference]);
        let events = vec![
            event("event:control", 200, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::PreferenceNotCurrent));
    }

    #[test]
    fn wrong_selected_subject_fails_before_execution_is_accepted() {
        let response = b"Hello.";
        let mut selection = selection(vec![pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }])]);
        selection.subject_id = "user:bob".into();
        selection = selection.seal_integrity().unwrap();
        let events = vec![
            event("event:control", 200, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::SelectionContextMismatch));
    }

    #[test]
    fn response_must_match_the_executor_commit_event_and_receipt() {
        let response = b"Original response.";
        let selection = selection(vec![]);
        let events = vec![response_event(response)];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, b"Changed response.", &receipt),
            Err(InteractionExecutionRejectV1::ResponseDigestMismatch));
    }

    #[test]
    fn tampered_selection_or_receipt_integrity_is_rejected() {
        let response = b"Hello.";
        let selection = selection(vec![]);
        let events = vec![response_event(response)];
        let mut receipt = seal(&selection, &events, response);
        receipt.executor_build_sha256 = sha("different-build");
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::ExecutorBuildDigestMismatch));
        let mut changed = selection;
        changed.decision_policy_sha256 = sha("tampered");
        assert!(!changed.verify_integrity());
    }

    #[test]
    fn duplicate_and_noncanonical_event_streams_are_rejected() {
        let response = b"Hello.";
        let selection = selection(vec![]);
        let first = event("event:control", 200, InteractionExecutionEventKindV1::ControlApplied {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        });
        let mut duplicate = vec![first.clone(), first, response_event(response)];
        sort_events(&mut duplicate);
        assert_eq!(canonical_events_digest(&duplicate),
            Err(InteractionExecutionRejectV1::DuplicateEventId));

        let reversed = vec![response_event(response), event(
            "event:control", 200, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            },
        )];
        assert_eq!(
            seal_interaction_execution_receipt_v1(
                &selection, &context(), b"request", response, &reversed, &sha("executor-build"),
            ),
            Err(InteractionExecutionRejectV1::NonCanonicalEventOrder)
        );
    }

    #[test]
    fn untrusted_executor_event_is_rejected() {
        let response = b"Hello.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }])]);
        let mut untrusted = event("event:control", 200,
            InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            });
        untrusted.source_receipt_sha256 = sha("forged");
        let events = vec![untrusted, response_event(response)];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::ExecutionVerifierRejected));
    }

    #[test]
    fn selected_setting_without_execution_receipt_fails() {
        let response = b"Concise.";
        let selection = selection(vec![pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }])]);
        let events = vec![response_event(response)];
        let receipt = seal(&selection, &events, response);
        assert_eq!(verify(&selection, &events, response, &receipt),
            Err(InteractionExecutionRejectV1::MissingRequiredControl));
    }

    #[test]
    fn multiple_preferences_are_checked_independently() {
        let response = b"I will keep it brief and avoid initiating a message.";
        let mut no_outreach = pref(vec![BehaviorRequirementV1::ActionMustNotOccur {
            action_class_sha256: sha("outbound_message"), window_start_ms: 100, window_end_ms: 300,
        }]);
        no_outreach.preference_id = "pref:no-outreach".into();
        let mut terse = pref(vec![BehaviorRequirementV1::ControlEquals {
            control_id: "verbosity".into(), setting_sha256: sha("terse"),
        }]);
        terse.preference_id = "pref:verbosity".into();
        let selection = selection(vec![no_outreach, terse]);
        let events = vec![
            event("event:verbosity", 250, InteractionExecutionEventKindV1::ControlApplied {
                control_id: "verbosity".into(), setting_sha256: sha("terse"),
            }),
            event("event:no-outreach-window", 300,
                InteractionExecutionEventKindV1::NoActionWindowClosed {
                    action_class_sha256: sha("outbound_message"),
                    window_start_ms: 100, window_end_ms: 300, observed_through_ms: 300,
                    action_events_sha256: digest_serialized(&Vec::<&InteractionExecutionEventV1>::new()).unwrap(),
                }),
            response_event(response),
        ];
        let receipt = seal(&selection, &events, response);
        assert!(verify(&selection, &events, response, &receipt).is_ok());
    }
}
