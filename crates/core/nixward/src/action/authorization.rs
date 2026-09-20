// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed action-intent and authorization primitives for the Nixward authority migration.
//!
//! This module deliberately separates:
//!
//! - immutable action intent;
//! - serializable authorization records used for audit;
//! - non-serializable live one-shot authorization state;
//! - execution receipts.
//!
//! A persisted authorization record is not a live execution capability. Likewise,
//! Phi/confidence and command-risk classification remain advisory inputs and do not
//! mint authorization here.

use super::executor::{ChannelOperation, FlakeOperation, NixOSCommand, SafetyLevel};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const ACTION_INTENT_DOMAIN: &[u8] = b"nixward-action-intent-v1";
const AUTHORIZATION_RECORD_DOMAIN: &[u8] = b"nixward-authorization-record-v1";
const EXECUTION_RECEIPT_DOMAIN: &[u8] = b"nixward-execution-receipt-v1";

/// Maximum mutation scope requested by an action intent.
///
/// Ordering is intentional: a scope may authorize the same or a narrower action
/// class, never a broader one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NixActionScopeV1 {
    ReadOnly,
    UserModify,
    SystemModify,
    SystemCritical,
    Destructive,
}

impl From<SafetyLevel> for NixActionScopeV1 {
    fn from(value: SafetyLevel) -> Self {
        match value {
            SafetyLevel::ReadOnly => Self::ReadOnly,
            SafetyLevel::UserModify => Self::UserModify,
            SafetyLevel::SystemModify => Self::SystemModify,
            SafetyLevel::SystemCritical => Self::SystemCritical,
            SafetyLevel::Destructive => Self::Destructive,
        }
    }
}

/// Typed representation of the currently supported NixOS command vocabulary.
///
/// `NixOSCommand::Custom` is deliberately excluded from v1. Free-form shell
/// commands need a separate policy for shell parsing, secret-bearing arguments,
/// canonical parameter identity, and display-before-approval. The legacy executor
/// remains available during migration, but a custom shell command cannot enter the
/// new governed authority path merely by being wrapped in an enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixActionDescriptorV1 {
    RebuildSwitch {
        flake: Option<String>,
        extra_args: Vec<String>,
    },
    RebuildTest {
        flake: Option<String>,
        extra_args: Vec<String>,
    },
    RebuildBoot {
        flake: Option<String>,
        extra_args: Vec<String>,
    },
    EnvInstall { packages: Vec<String> },
    EnvRemove { packages: Vec<String> },
    EnvRollback,
    Search { query: String, json: bool },
    ChannelUpdate { channel: Option<String> },
    ChannelAdd { url: String, name: String },
    ChannelRemove { name: String },
    ChannelList,
    FlakeUpdate { inputs: Vec<String> },
    FlakeLock { inputs: Vec<String> },
    FlakeShow,
    FlakeCheck,
    HomeManagerSwitch { flake: Option<String> },
    CollectGarbage {
        older_than_days: Option<u32>,
        delete_all: bool,
    },
}

impl TryFrom<&NixOSCommand> for NixActionDescriptorV1 {
    type Error = NixAuthorizationErrorV1;

    fn try_from(value: &NixOSCommand) -> Result<Self, Self::Error> {
        Ok(match value {
            NixOSCommand::RebuildSwitch { flake, extra_args } => Self::RebuildSwitch {
                flake: flake.clone(),
                extra_args: extra_args.clone(),
            },
            NixOSCommand::RebuildTest { flake, extra_args } => Self::RebuildTest {
                flake: flake.clone(),
                extra_args: extra_args.clone(),
            },
            NixOSCommand::RebuildBoot { flake, extra_args } => Self::RebuildBoot {
                flake: flake.clone(),
                extra_args: extra_args.clone(),
            },
            NixOSCommand::EnvInstall { packages } => Self::EnvInstall {
                packages: packages.clone(),
            },
            NixOSCommand::EnvRemove { packages } => Self::EnvRemove {
                packages: packages.clone(),
            },
            NixOSCommand::EnvRollback => Self::EnvRollback,
            NixOSCommand::Search { query, json } => Self::Search {
                query: query.clone(),
                json: *json,
            },
            NixOSCommand::Channel { operation } => match operation {
                ChannelOperation::Update { channel } => Self::ChannelUpdate {
                    channel: channel.clone(),
                },
                ChannelOperation::Add { url, name } => Self::ChannelAdd {
                    url: url.clone(),
                    name: name.clone(),
                },
                ChannelOperation::Remove { name } => Self::ChannelRemove { name: name.clone() },
                ChannelOperation::List => Self::ChannelList,
            },
            NixOSCommand::Flake { operation } => match operation {
                FlakeOperation::Update { inputs } => Self::FlakeUpdate {
                    inputs: inputs.clone(),
                },
                FlakeOperation::Lock { inputs } => Self::FlakeLock {
                    inputs: inputs.clone(),
                },
                FlakeOperation::Show => Self::FlakeShow,
                FlakeOperation::Check => Self::FlakeCheck,
            },
            NixOSCommand::HomeManagerSwitch { flake } => Self::HomeManagerSwitch {
                flake: flake.clone(),
            },
            NixOSCommand::CollectGarbage {
                older_than_days,
                delete_all,
            } => Self::CollectGarbage {
                older_than_days: *older_than_days,
                delete_all: *delete_all,
            },
            NixOSCommand::Custom { .. } => {
                return Err(NixAuthorizationErrorV1::UnsupportedCustomCommand)
            }
        })
    }
}

/// Immutable description of the exact operation for which authorization may be sought.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixActionIntentV1 {
    pub subject_identity: String,
    pub pre_state_identity: Option<String>,
    pub action: NixActionDescriptorV1,
    pub maximum_scope: NixActionScopeV1,
    #[serde(default)]
    pub preconditions: Vec<String>,
    #[serde(default)]
    pub required_postconditions: Vec<String>,
    pub rollback_or_recovery_ref: Option<String>,
}

impl NixActionIntentV1 {
    /// Build a governed intent from a typed NixOS command.
    ///
    /// V1 refuses free-form custom commands rather than pretending shell text has
    /// the same authorization semantics as a typed operation.
    pub fn from_command(
        subject_identity: impl Into<String>,
        pre_state_identity: Option<String>,
        command: &NixOSCommand,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        let action = NixActionDescriptorV1::try_from(command)?;
        let intent = Self {
            subject_identity: subject_identity.into(),
            pre_state_identity,
            action,
            maximum_scope: command.safety_level().into(),
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };
        intent.validate_shape()?;
        Ok(intent)
    }

    pub fn validate_shape(&self) -> Result<(), NixAuthorizationErrorV1> {
        require_nonempty(&self.subject_identity, "subject identity")?;
        validate_optional_nonempty(self.pre_state_identity.as_deref(), "pre-state identity")?;
        validate_optional_nonempty(
            self.rollback_or_recovery_ref.as_deref(),
            "rollback/recovery ref",
        )?;
        validate_list(&self.preconditions, "precondition")?;
        validate_list(&self.required_postconditions, "required postcondition")?;
        validate_action_shape(&self.action)?;

        let minimum_scope = minimum_scope_for_action(&self.action);
        if self.maximum_scope < minimum_scope {
            return Err(NixAuthorizationErrorV1::ScopeTooNarrow {
                requested: self.maximum_scope,
                minimum: minimum_scope,
            });
        }
        Ok(())
    }

    /// Deterministic semantic identity independent of serde/JSON representation.
    pub fn digest(&self) -> Result<String, NixAuthorizationErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(ACTION_INTENT_DOMAIN);
        put_str(&mut h, &self.subject_identity);
        put_opt_str(&mut h, self.pre_state_identity.as_deref());
        put_action(&mut h, &self.action);
        put_u8(&mut h, scope_tag(self.maximum_scope));
        put_str_vec(&mut h, &self.preconditions);
        put_str_vec(&mut h, &self.required_postconditions);
        put_opt_str(&mut h, self.rollback_or_recovery_ref.as_deref());
        Ok(h.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixAuthorizationProfileV1 {
    LocalExplicitConfirmation,
    DelegatedLocalPolicy,
    ExternalBoundPermit,
    EmergencyRecoveryPermit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixAuthorizationDecisionV1 {
    Approved,
    Denied,
}

/// Serializable authorization evidence for audit and provenance.
///
/// Deserializing this record does not create a live authorization capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixExecutionAuthorizationRecordV1 {
    pub action_intent_digest: String,
    pub profile: NixAuthorizationProfileV1,
    pub authority_ref: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: Option<u64>,
    pub decision: NixAuthorizationDecisionV1,
}

impl NixExecutionAuthorizationRecordV1 {
    pub fn validate_shape(&self) -> Result<(), NixAuthorizationErrorV1> {
        require_nonempty(&self.action_intent_digest, "action intent digest")?;
        require_nonempty(&self.authority_ref, "authority ref")?;
        if let Some(expires) = self.expires_at_unix_ms
            && expires < self.issued_at_unix_ms
        {
            return Err(NixAuthorizationErrorV1::InvalidExpiryWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixAuthorizationErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(AUTHORIZATION_RECORD_DOMAIN);
        put_str(&mut h, &self.action_intent_digest);
        put_u8(&mut h, authorization_profile_tag(self.profile));
        put_str(&mut h, &self.authority_ref);
        put_u64(&mut h, self.issued_at_unix_ms);
        put_opt_u64(&mut h, self.expires_at_unix_ms);
        put_u8(&mut h, authorization_decision_tag(self.decision));
        Ok(h.finalize().to_hex().to_string())
    }
}

/// Live local authorization state.
///
/// This type intentionally does not implement `Serialize`, `Deserialize`, or `Clone`.
/// Persisted audit records therefore cannot be deserialized back into live authority.
/// It is crate-private until an independently verified external-permit adapter exists.
pub(crate) struct LiveNixAuthorizationV1 {
    record: NixExecutionAuthorizationRecordV1,
    consumed: bool,
}

impl LiveNixAuthorizationV1 {
    /// Construct the current local explicit-confirmation profile.
    ///
    /// The caller must already have performed the real one-shot human approval
    /// ceremony. This function binds that approval to one exact action intent; it
    /// does not infer approval from Phi/confidence.
    pub(crate) fn local_explicit_confirmation(
        intent: &NixActionIntentV1,
        authority_ref: impl Into<String>,
        issued_at_unix_ms: u64,
        expires_at_unix_ms: Option<u64>,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        let record = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest()?,
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: authority_ref.into(),
            issued_at_unix_ms,
            expires_at_unix_ms,
            decision: NixAuthorizationDecisionV1::Approved,
        };
        record.validate_shape()?;
        Ok(Self {
            record,
            consumed: false,
        })
    }

    pub(crate) fn audit_record(&self) -> &NixExecutionAuthorizationRecordV1 {
        &self.record
    }

    /// Consume the authorization exactly once for the exact bound intent.
    pub(crate) fn consume_for(
        &mut self,
        intent: &NixActionIntentV1,
        now_unix_ms: u64,
    ) -> Result<NixExecutionAuthorizationRecordV1, NixAuthorizationErrorV1> {
        if self.consumed {
            return Err(NixAuthorizationErrorV1::AlreadyConsumed);
        }
        if self.record.decision != NixAuthorizationDecisionV1::Approved {
            return Err(NixAuthorizationErrorV1::NotApproved);
        }
        let digest = intent.digest()?;
        if digest != self.record.action_intent_digest {
            return Err(NixAuthorizationErrorV1::IntentMismatch);
        }
        if now_unix_ms < self.record.issued_at_unix_ms {
            return Err(NixAuthorizationErrorV1::NotYetValid);
        }
        if let Some(expires) = self.record.expires_at_unix_ms
            && now_unix_ms > expires
        {
            // Expiry is terminal for a one-shot capability. Mark it consumed so a
            // later caller cannot revive it by supplying an earlier wall-clock value.
            self.consumed = true;
            return Err(NixAuthorizationErrorV1::Expired);
        }
        self.consumed = true;
        Ok(self.record.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixMechanicalResultV1 {
    Succeeded,
    Failed,
    RolledBack,
    Partial,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixPostconditionStatusV1 {
    NotEvaluated,
    Satisfied,
    Violated,
    Unproven,
}

/// Audit receipt for one exact governed execution attempt.
///
/// Mechanical success and postcondition satisfaction are intentionally separate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixExecutionReceiptV1 {
    pub action_intent_digest: String,
    pub authorization_record_digest: String,
    pub executor_identity: String,
    pub actual_pre_state_ref: Option<String>,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
    pub mechanical_result: NixMechanicalResultV1,
    pub actual_post_state_ref: Option<String>,
    pub postcondition_status: NixPostconditionStatusV1,
    pub rollback_result_ref: Option<String>,
    pub risk_assessment_ref: Option<String>,
}

impl NixExecutionReceiptV1 {
    pub fn validate_shape(&self) -> Result<(), NixAuthorizationErrorV1> {
        require_nonempty(&self.action_intent_digest, "action intent digest")?;
        require_nonempty(
            &self.authorization_record_digest,
            "authorization record digest",
        )?;
        require_nonempty(&self.executor_identity, "executor identity")?;
        validate_optional_nonempty(self.actual_pre_state_ref.as_deref(), "pre-state ref")?;
        validate_optional_nonempty(self.actual_post_state_ref.as_deref(), "post-state ref")?;
        validate_optional_nonempty(self.rollback_result_ref.as_deref(), "rollback result ref")?;
        validate_optional_nonempty(self.risk_assessment_ref.as_deref(), "risk assessment ref")?;
        if self.finished_at_unix_ms < self.started_at_unix_ms {
            return Err(NixAuthorizationErrorV1::InvalidExecutionWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixAuthorizationErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(EXECUTION_RECEIPT_DOMAIN);
        put_str(&mut h, &self.action_intent_digest);
        put_str(&mut h, &self.authorization_record_digest);
        put_str(&mut h, &self.executor_identity);
        put_opt_str(&mut h, self.actual_pre_state_ref.as_deref());
        put_u64(&mut h, self.started_at_unix_ms);
        put_u64(&mut h, self.finished_at_unix_ms);
        put_u8(&mut h, mechanical_result_tag(self.mechanical_result));
        put_opt_str(&mut h, self.actual_post_state_ref.as_deref());
        put_u8(&mut h, postcondition_status_tag(self.postcondition_status));
        put_opt_str(&mut h, self.rollback_result_ref.as_deref());
        put_opt_str(&mut h, self.risk_assessment_ref.as_deref());
        Ok(h.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixAuthorizationErrorV1 {
    #[error("empty required field: {0}")]
    EmptyField(&'static str),
    #[error("empty {field} at index {index}")]
    EmptyListItem { field: &'static str, index: usize },
    #[error("free-form custom commands are not supported by governed action-intent v1")]
    UnsupportedCustomCommand,
    #[error("maximum scope {requested:?} is narrower than action minimum {minimum:?}")]
    ScopeTooNarrow {
        requested: NixActionScopeV1,
        minimum: NixActionScopeV1,
    },
    #[error("authorization decision is not approved")]
    NotApproved,
    #[error("authorization is bound to a different action intent")]
    IntentMismatch,
    #[error("authorization is not yet valid")]
    NotYetValid,
    #[error("authorization has expired")]
    Expired,
    #[error("one-shot authorization was already consumed")]
    AlreadyConsumed,
    #[error("authorization expiry precedes issuance")]
    InvalidExpiryWindow,
    #[error("execution finish time precedes start time")]
    InvalidExecutionWindow,
}

fn validate_action_shape(action: &NixActionDescriptorV1) -> Result<(), NixAuthorizationErrorV1> {
    match action {
        NixActionDescriptorV1::RebuildSwitch { flake, extra_args }
        | NixActionDescriptorV1::RebuildTest { flake, extra_args }
        | NixActionDescriptorV1::RebuildBoot { flake, extra_args } => {
            validate_optional_nonempty(flake.as_deref(), "flake ref")?;
            validate_list(extra_args, "extra arg")?;
        }
        NixActionDescriptorV1::EnvInstall { packages }
        | NixActionDescriptorV1::EnvRemove { packages } => {
            validate_nonempty_list(packages, "package")?;
        }
        NixActionDescriptorV1::Search { query, .. } => require_nonempty(query, "search query")?,
        NixActionDescriptorV1::ChannelUpdate { channel } => {
            validate_optional_nonempty(channel.as_deref(), "channel")?
        }
        NixActionDescriptorV1::ChannelAdd { url, name } => {
            require_nonempty(url, "channel url")?;
            require_nonempty(name, "channel name")?;
        }
        NixActionDescriptorV1::ChannelRemove { name } => require_nonempty(name, "channel name")?,
        NixActionDescriptorV1::FlakeUpdate { inputs }
        | NixActionDescriptorV1::FlakeLock { inputs } => validate_list(inputs, "flake input")?,
        NixActionDescriptorV1::HomeManagerSwitch { flake } => {
            validate_optional_nonempty(flake.as_deref(), "home-manager flake ref")?
        }
        NixActionDescriptorV1::EnvRollback
        | NixActionDescriptorV1::ChannelList
        | NixActionDescriptorV1::FlakeShow
        | NixActionDescriptorV1::FlakeCheck
        | NixActionDescriptorV1::CollectGarbage { .. } => {}
    }
    Ok(())
}

fn minimum_scope_for_action(action: &NixActionDescriptorV1) -> NixActionScopeV1 {
    match action {
        NixActionDescriptorV1::Search { .. }
        | NixActionDescriptorV1::ChannelList
        | NixActionDescriptorV1::FlakeShow
        | NixActionDescriptorV1::FlakeCheck => NixActionScopeV1::ReadOnly,

        NixActionDescriptorV1::EnvInstall { .. }
        | NixActionDescriptorV1::EnvRemove { .. }
        | NixActionDescriptorV1::EnvRollback
        | NixActionDescriptorV1::ChannelUpdate { .. }
        | NixActionDescriptorV1::ChannelAdd { .. }
        | NixActionDescriptorV1::ChannelRemove { .. }
        | NixActionDescriptorV1::FlakeUpdate { .. }
        | NixActionDescriptorV1::FlakeLock { .. }
        | NixActionDescriptorV1::HomeManagerSwitch { .. } => NixActionScopeV1::UserModify,

        NixActionDescriptorV1::RebuildTest { .. } | NixActionDescriptorV1::RebuildBoot { .. } => {
            NixActionScopeV1::SystemModify
        }
        NixActionDescriptorV1::RebuildSwitch { .. } => NixActionScopeV1::SystemCritical,
        NixActionDescriptorV1::CollectGarbage { .. } => NixActionScopeV1::Destructive,
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), NixAuthorizationErrorV1> {
    if value.trim().is_empty() {
        Err(NixAuthorizationErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), NixAuthorizationErrorV1> {
    if let Some(value) = value {
        require_nonempty(value, field)?;
    }
    Ok(())
}

fn validate_list(values: &[String], field: &'static str) -> Result<(), NixAuthorizationErrorV1> {
    for (index, value) in values.iter().enumerate() {
        if value.trim().is_empty() {
            return Err(NixAuthorizationErrorV1::EmptyListItem { field, index });
        }
    }
    Ok(())
}

fn validate_nonempty_list(
    values: &[String],
    field: &'static str,
) -> Result<(), NixAuthorizationErrorV1> {
    if values.is_empty() {
        return Err(NixAuthorizationErrorV1::EmptyField(field));
    }
    validate_list(values, field)
}

fn put_u8(h: &mut Hasher, value: u8) {
    h.update(&[value]);
}

fn put_u32(h: &mut Hasher, value: u32) {
    h.update(&value.to_be_bytes());
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_bool(h: &mut Hasher, value: bool) {
    put_u8(h, u8::from(value));
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

fn put_opt_str(h: &mut Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_str(h, value);
        }
        None => put_u8(h, 0),
    }
}

fn put_opt_u32(h: &mut Hasher, value: Option<u32>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_u32(h, value);
        }
        None => put_u8(h, 0),
    }
}

fn put_opt_u64(h: &mut Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            put_u8(h, 1);
            put_u64(h, value);
        }
        None => put_u8(h, 0),
    }
}

fn put_str_vec(h: &mut Hasher, values: &[String]) {
    put_u64(h, values.len() as u64);
    for value in values {
        put_str(h, value);
    }
}

fn put_action(h: &mut Hasher, action: &NixActionDescriptorV1) {
    match action {
        NixActionDescriptorV1::RebuildSwitch { flake, extra_args } => {
            put_u8(h, 0);
            put_opt_str(h, flake.as_deref());
            put_str_vec(h, extra_args);
        }
        NixActionDescriptorV1::RebuildTest { flake, extra_args } => {
            put_u8(h, 1);
            put_opt_str(h, flake.as_deref());
            put_str_vec(h, extra_args);
        }
        NixActionDescriptorV1::RebuildBoot { flake, extra_args } => {
            put_u8(h, 2);
            put_opt_str(h, flake.as_deref());
            put_str_vec(h, extra_args);
        }
        NixActionDescriptorV1::EnvInstall { packages } => {
            put_u8(h, 3);
            put_str_vec(h, packages);
        }
        NixActionDescriptorV1::EnvRemove { packages } => {
            put_u8(h, 4);
            put_str_vec(h, packages);
        }
        NixActionDescriptorV1::EnvRollback => put_u8(h, 5),
        NixActionDescriptorV1::Search { query, json } => {
            put_u8(h, 6);
            put_str(h, query);
            put_bool(h, *json);
        }
        NixActionDescriptorV1::ChannelUpdate { channel } => {
            put_u8(h, 7);
            put_opt_str(h, channel.as_deref());
        }
        NixActionDescriptorV1::ChannelAdd { url, name } => {
            put_u8(h, 8);
            put_str(h, url);
            put_str(h, name);
        }
        NixActionDescriptorV1::ChannelRemove { name } => {
            put_u8(h, 9);
            put_str(h, name);
        }
        NixActionDescriptorV1::ChannelList => put_u8(h, 10),
        NixActionDescriptorV1::FlakeUpdate { inputs } => {
            put_u8(h, 11);
            put_str_vec(h, inputs);
        }
        NixActionDescriptorV1::FlakeLock { inputs } => {
            put_u8(h, 12);
            put_str_vec(h, inputs);
        }
        NixActionDescriptorV1::FlakeShow => put_u8(h, 13),
        NixActionDescriptorV1::FlakeCheck => put_u8(h, 14),
        NixActionDescriptorV1::HomeManagerSwitch { flake } => {
            put_u8(h, 15);
            put_opt_str(h, flake.as_deref());
        }
        NixActionDescriptorV1::CollectGarbage {
            older_than_days,
            delete_all,
        } => {
            put_u8(h, 16);
            put_opt_u32(h, *older_than_days);
            put_bool(h, *delete_all);
        }
    }
}

fn scope_tag(value: NixActionScopeV1) -> u8 {
    match value {
        NixActionScopeV1::ReadOnly => 0,
        NixActionScopeV1::UserModify => 1,
        NixActionScopeV1::SystemModify => 2,
        NixActionScopeV1::SystemCritical => 3,
        NixActionScopeV1::Destructive => 4,
    }
}

fn authorization_profile_tag(value: NixAuthorizationProfileV1) -> u8 {
    match value {
        NixAuthorizationProfileV1::LocalExplicitConfirmation => 0,
        NixAuthorizationProfileV1::DelegatedLocalPolicy => 1,
        NixAuthorizationProfileV1::ExternalBoundPermit => 2,
        NixAuthorizationProfileV1::EmergencyRecoveryPermit => 3,
    }
}

fn authorization_decision_tag(value: NixAuthorizationDecisionV1) -> u8 {
    match value {
        NixAuthorizationDecisionV1::Approved => 0,
        NixAuthorizationDecisionV1::Denied => 1,
    }
}

fn mechanical_result_tag(value: NixMechanicalResultV1) -> u8 {
    match value {
        NixMechanicalResultV1::Succeeded => 0,
        NixMechanicalResultV1::Failed => 1,
        NixMechanicalResultV1::RolledBack => 2,
        NixMechanicalResultV1::Partial => 3,
        NixMechanicalResultV1::Unknown => 4,
    }
}

fn postcondition_status_tag(value: NixPostconditionStatusV1) -> u8 {
    match value {
        NixPostconditionStatusV1::NotEvaluated => 0,
        NixPostconditionStatusV1::Satisfied => 1,
        NixPostconditionStatusV1::Violated => 2,
        NixPostconditionStatusV1::Unproven => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rebuild() -> NixOSCommand {
        NixOSCommand::RebuildSwitch {
            flake: Some(".#workstation".to_string()),
            extra_args: vec!["--show-trace".to_string()],
        }
    }

    #[test]
    fn action_intent_digest_is_deterministic_and_not_serde_based() {
        let intent = NixActionIntentV1::from_command(
            "host:workstation",
            Some("generation:42".to_string()),
            &rebuild(),
        )
        .unwrap();
        assert_eq!(intent.digest().unwrap(), intent.digest().unwrap());

        let json = serde_json::to_string(&intent).unwrap();
        assert!(!json.is_empty());
        assert_ne!(
            intent.digest().unwrap(),
            blake3::hash(json.as_bytes()).to_hex().to_string(),
            "canonical action identity must not accidentally become serde_json hashing",
        );
    }

    #[test]
    fn action_parameter_or_state_change_changes_identity() {
        let a = NixActionIntentV1::from_command(
            "host:workstation",
            Some("generation:42".to_string()),
            &rebuild(),
        )
        .unwrap();
        let b = NixActionIntentV1::from_command(
            "host:workstation",
            Some("generation:43".to_string()),
            &rebuild(),
        )
        .unwrap();
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());

        let changed = NixOSCommand::RebuildSwitch {
            flake: Some(".#workstation".to_string()),
            extra_args: vec![],
        };
        let c = NixActionIntentV1::from_command(
            "host:workstation",
            Some("generation:42".to_string()),
            &changed,
        )
        .unwrap();
        assert_ne!(a.digest().unwrap(), c.digest().unwrap());
    }

    #[test]
    fn manually_narrowed_scope_is_rejected() {
        let mut intent = NixActionIntentV1::from_command("host:x", None, &rebuild()).unwrap();
        intent.maximum_scope = NixActionScopeV1::ReadOnly;
        assert_eq!(
            intent.validate_shape().unwrap_err(),
            NixAuthorizationErrorV1::ScopeTooNarrow {
                requested: NixActionScopeV1::ReadOnly,
                minimum: NixActionScopeV1::SystemCritical,
            }
        );
    }

    #[test]
    fn custom_shell_commands_are_outside_governed_v1() {
        let command = NixOSCommand::Custom {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), "echo hello".to_string()],
            safety_level: SafetyLevel::SystemCritical,
        };
        assert_eq!(
            NixActionIntentV1::from_command("host:x", None, &command).unwrap_err(),
            NixAuthorizationErrorV1::UnsupportedCustomCommand,
        );
    }

    #[test]
    fn local_authorization_is_exact_and_one_shot() {
        let intent = NixActionIntentV1::from_command(
            "host:workstation",
            Some("generation:42".to_string()),
            &rebuild(),
        )
        .unwrap();
        let mut live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "tui-approval:abc",
            1_000,
            Some(2_000),
        )
        .unwrap();

        let consumed = live.consume_for(&intent, 1_500).unwrap();
        assert_eq!(
            consumed.profile,
            NixAuthorizationProfileV1::LocalExplicitConfirmation
        );
        assert_eq!(
            live.consume_for(&intent, 1_500).unwrap_err(),
            NixAuthorizationErrorV1::AlreadyConsumed
        );
    }

    #[test]
    fn authorization_cannot_be_rebound_to_another_intent() {
        let intent = NixActionIntentV1::from_command(
            "host:workstation",
            Some("generation:42".to_string()),
            &rebuild(),
        )
        .unwrap();
        let other = NixActionIntentV1::from_command(
            "host:other",
            Some("generation:42".to_string()),
            &rebuild(),
        )
        .unwrap();
        let mut live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "tui-approval:abc",
            1_000,
            Some(2_000),
        )
        .unwrap();
        assert_eq!(
            live.consume_for(&other, 1_500).unwrap_err(),
            NixAuthorizationErrorV1::IntentMismatch
        );
        // A failed rebinding attempt does not consume the valid capability.
        assert!(live.consume_for(&intent, 1_500).is_ok());
    }

    #[test]
    fn authorization_cannot_be_used_before_issue_time() {
        let intent = NixActionIntentV1::from_command("host:x", None, &rebuild()).unwrap();
        let mut live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "tui-approval:abc",
            1_000,
            Some(2_000),
        )
        .unwrap();
        assert_eq!(
            live.consume_for(&intent, 999).unwrap_err(),
            NixAuthorizationErrorV1::NotYetValid
        );
        assert!(live.consume_for(&intent, 1_000).is_ok());
    }

    #[test]
    fn expired_authorization_is_terminal() {
        let intent = NixActionIntentV1::from_command("host:x", None, &rebuild()).unwrap();
        let mut live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "tui-approval:abc",
            1_000,
            Some(1_100),
        )
        .unwrap();
        assert_eq!(
            live.consume_for(&intent, 1_101).unwrap_err(),
            NixAuthorizationErrorV1::Expired
        );
        assert_eq!(
            live.consume_for(&intent, 1_050).unwrap_err(),
            NixAuthorizationErrorV1::AlreadyConsumed
        );
    }

    #[test]
    fn persisted_record_is_audit_data_not_live_capability() {
        let intent = NixActionIntentV1::from_command("host:x", None, &rebuild()).unwrap();
        let live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "tui-approval:abc",
            1_000,
            None,
        )
        .unwrap();
        let json = serde_json::to_string(live.audit_record()).unwrap();
        let restored: NixExecutionAuthorizationRecordV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, *live.audit_record());
        assert_eq!(restored.action_intent_digest, intent.digest().unwrap());
        // There is intentionally no public/serde conversion from `restored` back
        // into `LiveNixAuthorizationV1`.
    }

    #[test]
    fn mechanical_success_is_distinct_from_postcondition_status() {
        let intent = NixActionIntentV1::from_command("host:x", None, &rebuild()).unwrap();
        let live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "tui-approval:abc",
            1_000,
            None,
        )
        .unwrap();
        let receipt = NixExecutionReceiptV1 {
            action_intent_digest: intent.digest().unwrap(),
            authorization_record_digest: live.audit_record().digest().unwrap(),
            executor_identity: "nixward:test-executor".to_string(),
            actual_pre_state_ref: Some("generation:42".to_string()),
            started_at_unix_ms: 1_100,
            finished_at_unix_ms: 1_200,
            mechanical_result: NixMechanicalResultV1::Succeeded,
            actual_post_state_ref: Some("generation:43".to_string()),
            postcondition_status: NixPostconditionStatusV1::Violated,
            rollback_result_ref: None,
            risk_assessment_ref: None,
        };
        receipt.validate_shape().unwrap();
        assert_eq!(receipt.mechanical_result, NixMechanicalResultV1::Succeeded);
        assert_eq!(
            receipt.postcondition_status,
            NixPostconditionStatusV1::Violated
        );
    }
}