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
use super::local_approval::LocalApprovalDecisionKindV1;
use super::local_approval_store::ConsumedLocalApprovalDecisionV1;
use super::service_domain::{NixServiceOperationKindV1, validate_canonical_service_operation_v1};
use super::service_effect::{
    NixServiceEffectContextErrorV1, NixServiceEffectContextV1,
    NixSystemdUnitDefinitionContentEvidenceV1,
    NixSystemdUnitDefinitionContentFileV1, NixVerifiedServiceDefinitionContentV1,
};
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
    EnvInstall {
        packages: Vec<String>,
    },
    EnvRemove {
        packages: Vec<String>,
    },
    EnvRollback,
    EnvSwitchGeneration {
        generation: u32,
    },
    EnvDeleteGenerations {
        keep_last: usize,
    },
    EnvDeleteGenerationsOlderThan {
        days: u32,
    },
    Search {
        query: String,
        json: bool,
    },
    ChannelUpdate {
        channel: Option<String>,
    },
    ChannelAdd {
        url: String,
        name: String,
    },
    ChannelRemove {
        name: String,
    },
    ChannelList,
    FlakeUpdate {
        inputs: Vec<String>,
    },
    FlakeLock {
        inputs: Vec<String>,
    },
    FlakeInit {
        template: Option<String>,
    },
    FlakeShow,
    FlakeCheck,
    HomeManagerSwitch {
        flake: Option<String>,
    },
    CollectGarbage {
        older_than_days: Option<u32>,
        delete_all: bool,
    },
    Service {
        operation: NixServiceOperationKindV1,
        unit: String,
    },
    ConfigPatch {
        option_path: String,
        value: String,
        expected_config_digest: String,
    },
}

impl TryFrom<&NixOSCommand> for NixActionDescriptorV1 {
    type Error = NixAuthorizationErrorV1;

    fn try_from(value: &NixOSCommand) -> Result<Self, Self::Error> {
        value
            .validate_shape()
            .map_err(NixAuthorizationErrorV1::InvalidTypedCommand)?;

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
            NixOSCommand::EnvSwitchGeneration { generation } => Self::EnvSwitchGeneration {
                generation: *generation,
            },
            NixOSCommand::EnvDeleteGenerations { keep_last } => Self::EnvDeleteGenerations {
                keep_last: *keep_last,
            },
            NixOSCommand::EnvDeleteGenerationsOlderThan { days } => {
                Self::EnvDeleteGenerationsOlderThan { days: *days }
            }
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
                FlakeOperation::Init { template } => Self::FlakeInit {
                    template: template.clone(),
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
            NixOSCommand::Service { operation, unit } => Self::Service {
                operation: *operation,
                unit: unit.clone(),
            },
            NixOSCommand::ConfigPatch {
                option_path,
                value,
                expected_config_digest,
            } => Self::ConfigPatch {
                option_path: option_path.clone(),
                value: value.clone(),
                expected_config_digest: expected_config_digest.clone(),
            },
            NixOSCommand::Custom { .. } => {
                return Err(NixAuthorizationErrorV1::UnsupportedCustomCommand);
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
    /// Explicit service-effect contract. Required before a Service intent can
    /// become approved authorization; optional only while representing an
    /// unqualified serializable intent.
    #[serde(default)]
    pub service_effect_context: Option<NixServiceEffectContextV1>,
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
            service_effect_context: None,
            maximum_scope: command.safety_level().into(),
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };
        intent.validate_shape()?;
        Ok(intent)
    }

    pub fn with_service_effect_context(
        mut self,
        context: NixServiceEffectContextV1,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        self.service_effect_context = Some(context);
        self.validate_shape()?;
        Ok(self)
    }

    pub fn from_command_with_service_effect_context(
        subject_identity: impl Into<String>,
        pre_state_identity: String,
        command: &NixOSCommand,
        context: NixServiceEffectContextV1,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        let intent = Self::from_command(subject_identity, Some(pre_state_identity), command)?;
        intent.with_service_effect_context(context)
    }

    pub fn service_effect_context(&self) -> Option<&NixServiceEffectContextV1> {
        self.service_effect_context.as_ref()
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

        match (&self.action, &self.service_effect_context) {
            (NixActionDescriptorV1::Service { operation, unit }, Some(context)) => {
                validate_service_effect_context_binding(
                    context,
                    *operation,
                    unit,
                    self.pre_state_identity.as_deref(),
                )?;
            }
            (NixActionDescriptorV1::Service { .. }, None) => {}
            (_, Some(_)) => return Err(NixAuthorizationErrorV1::UnexpectedServiceEffectContext),
            (_, None) => {}
        }

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
        match &self.service_effect_context {
            Some(context) => {
                put_u8(&mut h, 1);
                put_str(&mut h, &context.digest()?);
            }
            None => put_u8(&mut h, 0),
        }
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
    /// Digest of the explicit service-effect context, when the authorized action
    /// is a Service operation.
    #[serde(default)]
    pub service_effect_context_digest: Option<String>,
    pub profile: NixAuthorizationProfileV1,
    pub authority_ref: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: Option<u64>,
    pub decision: NixAuthorizationDecisionV1,
}

impl NixExecutionAuthorizationRecordV1 {
    pub fn validate_shape(&self) -> Result<(), NixAuthorizationErrorV1> {
        require_nonempty(&self.action_intent_digest, "action intent digest")?;
        if let Some(digest) = &self.service_effect_context_digest {
            validate_hex_digest(digest, "service effect context digest")?;
        }
        require_nonempty(&self.authority_ref, "authority ref")?;
        if let Some(expires) = self.expires_at_unix_ms
            && expires < self.issued_at_unix_ms
        {
            return Err(NixAuthorizationErrorV1::InvalidExpiryWindow);
        }
        Ok(())
    }

    pub fn validate_against_intent(
        &self,
        intent: &NixActionIntentV1,
    ) -> Result<(), NixAuthorizationErrorV1> {
        self.validate_shape()?;
        if self.decision != NixAuthorizationDecisionV1::Approved {
            return Err(NixAuthorizationErrorV1::NotApproved);
        }
        let intent_digest = intent.digest()?;
        if self.action_intent_digest != intent_digest {
            return Err(NixAuthorizationErrorV1::IntentMismatch);
        }

        let expected_context_digest = service_effect_context_digest_for_intent(intent)?;
        if self.service_effect_context_digest != expected_context_digest {
            return Err(NixAuthorizationErrorV1::ServiceEffectContextMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixAuthorizationErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(AUTHORIZATION_RECORD_DOMAIN);
        put_str(&mut h, &self.action_intent_digest);
        put_opt_str(&mut h, self.service_effect_context_digest.as_deref());
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
/// Live execution authority minted from one already-consumed local approval.
///
/// This object is intentionally non-serializable and non-cloneable. Possession
/// represents the exact semantic binding between the approved intent and the
/// one-shot local approval evidence consumed by the live daemon runtime.
pub struct NixLocalExecutionAuthorityV1 {
    intent: NixActionIntentV1,
    approval: ConsumedLocalApprovalDecisionV1,
    /// Exact observer-sealed definition commitment used when this Service authority was promoted.
    /// `None` for non-Service authorities.
    service_definition_content_digest: Option<String>,
}

impl NixLocalExecutionAuthorityV1 {
    /// Promote one consumed approval into Nixward execution authority.
    ///
    /// The consumed approval token is moved into this object, preventing later
    /// reconstruction or duplicate consumption by ordinary data copying.
    pub fn from_consumed_local_approval(
        intent: NixActionIntentV1,
        approval: ConsumedLocalApprovalDecisionV1,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        if approval.decision_kind() != LocalApprovalDecisionKindV1::Approved {
            return Err(NixAuthorizationErrorV1::NotApproved);
        }
        let digest = intent.digest()?;
        if approval.decision_evidence().action_intent_digest != digest {
            return Err(NixAuthorizationErrorV1::IntentMismatch);
        }
        if matches!(intent.action, NixActionDescriptorV1::Service { .. }) {
            return Err(NixAuthorizationErrorV1::MissingServiceDefinitionContentCapture);
        }
        service_effect_context_digest_for_intent(&intent)?;
        Ok(Self {
            intent,
            approval,
            service_definition_content_digest: None,
        })
    }

    /// Promote a consumed Service approval only after an observer-sealed definition-content capture.
    ///
    /// The content token is not caller-fabricable; it can only be obtained from the
    /// read-only observer boundary. When an intent already carries a service-effect
    /// context, this constructor additionally requires the sealed capture to match
    /// both its source-identity and content commitments.
    pub fn from_consumed_local_approval_with_definition_capture(
        intent: NixActionIntentV1,
        approval: ConsumedLocalApprovalDecisionV1,
        content: &NixVerifiedServiceDefinitionContentV1,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        if approval.decision_kind() != LocalApprovalDecisionKindV1::Approved {
            return Err(NixAuthorizationErrorV1::NotApproved);
        }
        let digest = intent.digest()?;
        if approval.decision_evidence().action_intent_digest != digest {
            return Err(NixAuthorizationErrorV1::IntentMismatch);
        }

        let content_digest = validate_service_definition_capture_binding(&intent, content)?;
        service_effect_context_digest_for_intent(&intent)?;
        Ok(Self {
            intent,
            approval,
            service_definition_content_digest: Some(content_digest),
        })
    }

    /// Validate that the execution command is exactly the action that was approved.
    pub(crate) fn validate_command(
        &self,
        command: &NixOSCommand,
    ) -> Result<(), NixAuthorizationErrorV1> {
        let descriptor = NixActionDescriptorV1::try_from(command)?;
        if descriptor != self.intent.action {
            return Err(NixAuthorizationErrorV1::IntentMismatch);
        }
        Ok(())
    }

    pub(crate) fn action_intent_digest(&self) -> Result<String, NixAuthorizationErrorV1> {
        self.intent.digest()
    }

    /// Return the pre-state identity bound into the approved intent.
    ///
    /// The executor may use this only for execution-time freshness validation;
    /// it does not grant or expand authority.
    pub(crate) fn pre_state_identity(&self) -> Option<&str> {
        self.intent.pre_state_identity.as_deref()
    }

    pub(crate) fn approval_request_id(&self) -> &str {
        self.approval.request_id()
    }

    pub(crate) fn projection_digest(&self) -> &str {
        self.approval.projection_digest()
    }

    /// Exact content commitment captured before Service authority promotion.
    ///
    /// This is deliberately metadata-only; the raw service definition bytes never
    /// become part of the live authority object.
    pub(crate) fn service_definition_content_digest(&self) -> Option<&str> {
        self.service_definition_content_digest.as_deref()
    }
}

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
        if matches!(intent.action, NixActionDescriptorV1::Service { .. }) {
            return Err(NixAuthorizationErrorV1::MissingServiceDefinitionContentCapture);
        }
        let service_effect_context_digest = service_effect_context_digest_for_intent(intent)?;
        let record = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest()?,
            service_effect_context_digest,
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

    pub(crate) fn local_explicit_confirmation_with_definition_capture(
        intent: &NixActionIntentV1,
        authority_ref: impl Into<String>,
        issued_at_unix_ms: u64,
        expires_at_unix_ms: Option<u64>,
        content: &NixVerifiedServiceDefinitionContentV1,
    ) -> Result<Self, NixAuthorizationErrorV1> {
        validate_service_definition_capture(intent, content)?;
        let service_effect_context_digest = service_effect_context_digest_for_intent(intent)?;
        let record = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest()?,
            service_effect_context_digest,
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: authority_ref.into(),
            issued_at_unix_ms,
            expires_at_unix_ms,
            decision: NixAuthorizationDecisionV1::Approved,
        };
        record.validate_shape()?;
        Ok(Self { record, consumed: false })
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
    #[error("invalid typed command: {0}")]
    InvalidTypedCommand(String),
    #[error("maximum scope {requested:?} is narrower than action minimum {minimum:?}")]
    ScopeTooNarrow {
        requested: NixActionScopeV1,
        minimum: NixActionScopeV1,
    },
    #[error("authorization decision is not approved")]
    NotApproved,
    #[error("authorization is bound to a different action intent")]
    IntentMismatch,
    #[error("service action is missing its authority-bound effect context")]
    MissingServiceEffectContext,
    #[error("service effect context does not match the typed service action")]
    ServiceEffectContextMismatch,
    #[error("service effect context is present for a non-service action")]
    UnexpectedServiceEffectContext,
    #[error("service authorization requires an observer-sealed definition content capture")]
    MissingServiceDefinitionContentCapture,
    #[error("observer-sealed definition content does not match the service intent")]
    DefinitionContentCaptureMismatch,
    #[error("invalid service effect context: {0}")]
    InvalidServiceEffectContext(NixServiceEffectContextErrorV1),
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

fn validate_service_effect_context_binding(
    context: &NixServiceEffectContextV1,
    operation: NixServiceOperationKindV1,
    unit: &str,
    pre_state_identity: Option<&str>,
) -> Result<(), NixAuthorizationErrorV1> {
    context
        .validate_shape()
        .map_err(NixAuthorizationErrorV1::InvalidServiceEffectContext)?;
    if context.operation != operation || context.unit != unit {
        return Err(NixAuthorizationErrorV1::ServiceEffectContextMismatch);
    }

    let Some(pre_state_identity) = pre_state_identity else {
        return Err(NixAuthorizationErrorV1::MissingServiceEffectContext);
    };
    let prefix = "nixward-service-pre-state-v1|generation=";
    let Some(rest) = pre_state_identity.strip_prefix(prefix) else {
        return Err(NixAuthorizationErrorV1::MissingServiceEffectContext);
    };
    let (generation, rest) = rest
        .split_once("|unit=")
        .ok_or(NixAuthorizationErrorV1::InvalidServiceEffectContext(
            NixServiceEffectContextErrorV1::InvalidDigest("pre-state identity"),
        ))?;
    let generation = generation
        .parse::<u64>()
        .map_err(|_| NixAuthorizationErrorV1::InvalidServiceEffectContext(
            NixServiceEffectContextErrorV1::InvalidDigest("pre-state identity"),
        ))?;
    let (identity_unit, state_digest) = rest
        .split_once("|state=")
        .ok_or(NixAuthorizationErrorV1::InvalidServiceEffectContext(
            NixServiceEffectContextErrorV1::InvalidDigest("pre-state identity"),
        ))?;
    if generation != context.authorized_generation
        || identity_unit != context.unit
        || state_digest != context.pre_state_digest
    {
        return Err(NixAuthorizationErrorV1::ServiceEffectContextMismatch);
    }
    Ok(())
}

fn validate_service_definition_capture(
    intent: &NixActionIntentV1,
    content: &NixVerifiedServiceDefinitionContentV1,
) -> Result<(), NixAuthorizationErrorV1> {
    let NixActionDescriptorV1::Service { operation, unit } = &intent.action else {
        return Err(NixAuthorizationErrorV1::UnexpectedServiceEffectContext);
    };
    let context = intent.service_effect_context.as_ref()
        .ok_or(NixAuthorizationErrorV1::MissingServiceEffectContext)?;
    let evidence = content.as_ref();
    if evidence.unit != *unit
        || context.operation != *operation
        || context.unit != *unit
        || context.authorized_definition_digest != evidence.source_identity_digest
    {
        return Err(NixAuthorizationErrorV1::DefinitionContentCaptureMismatch);
    }
    let content_digest = content.digest()
        .map_err(NixAuthorizationErrorV1::InvalidServiceEffectContext)?;
    if context.authorized_definition_content_digest != content_digest {
        return Err(NixAuthorizationErrorV1::DefinitionContentCaptureMismatch);
    }
    Ok(())
}

fn validate_service_definition_capture_binding(
    intent: &NixActionIntentV1,
    content: &NixVerifiedServiceDefinitionContentV1,
) -> Result<String, NixAuthorizationErrorV1> {
    let NixActionDescriptorV1::Service { unit, .. } = &intent.action else {
        return Err(NixAuthorizationErrorV1::UnexpectedServiceEffectContext);
    };

    let evidence = content.as_ref();
    if evidence.unit != *unit {
        return Err(NixAuthorizationErrorV1::DefinitionContentCaptureMismatch);
    }

    let content_digest = content
        .digest()
        .map_err(NixAuthorizationErrorV1::InvalidServiceEffectContext)?;

    if let Some(context) = intent.service_effect_context() {
        if context.unit != *unit
            || context.authorized_definition_digest != evidence.source_identity_digest
            || context.authorized_definition_content_digest != content_digest
        {
            return Err(NixAuthorizationErrorV1::DefinitionContentCaptureMismatch);
        }
    }

    Ok(content_digest)
}

fn service_effect_context_digest_for_intent(
    intent: &NixActionIntentV1,
) -> Result<Option<String>, NixAuthorizationErrorV1> {
    match &intent.action {
        NixActionDescriptorV1::Service { operation, unit } => {
            let context = intent
                .service_effect_context
                .as_ref()
                .ok_or(NixAuthorizationErrorV1::MissingServiceEffectContext)?;
            validate_service_effect_context_binding(
                context,
                *operation,
                unit,
                intent.pre_state_identity.as_deref(),
            )?;
            Ok(Some(
                context
                    .digest()
                    .map_err(NixAuthorizationErrorV1::InvalidServiceEffectContext)?,
            ))
        }
        _ => {
            if intent.service_effect_context.is_some() {
                Err(NixAuthorizationErrorV1::UnexpectedServiceEffectContext)
            } else {
                Ok(None)
            }
        }
    }
}

fn validate_hex_digest(
    value: &str,
    field: &'static str,
) -> Result<(), NixAuthorizationErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NixAuthorizationErrorV1::InvalidServiceEffectContext(
            NixServiceEffectContextErrorV1::InvalidDigest(field),
        ));
    }
    Ok(())
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
        NixActionDescriptorV1::EnvSwitchGeneration { generation } => {
            if *generation == 0 {
                return Err(NixAuthorizationErrorV1::InvalidTypedCommand(
                    "generation number must be greater than zero".to_string(),
                ));
            }
        }
        NixActionDescriptorV1::EnvDeleteGenerations { keep_last } => {
            if *keep_last == 0 {
                return Err(NixAuthorizationErrorV1::InvalidTypedCommand(
                    "keep_last must be greater than zero".to_string(),
                ));
            }
        }
        NixActionDescriptorV1::EnvDeleteGenerationsOlderThan { days } => {
            if *days == 0 {
                return Err(NixAuthorizationErrorV1::InvalidTypedCommand(
                    "age in days must be greater than zero".to_string(),
                ));
            }
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
        NixActionDescriptorV1::FlakeInit { template } => {
            validate_optional_nonempty(template.as_deref(), "flake init template")?
        }
        NixActionDescriptorV1::HomeManagerSwitch { flake } => {
            validate_optional_nonempty(flake.as_deref(), "home-manager flake ref")?
        }
        NixActionDescriptorV1::Service { operation, unit } => {
            validate_canonical_service_operation_v1(unit, *operation)
                .map_err(|error| NixAuthorizationErrorV1::InvalidTypedCommand(error.to_string()))?;
        }
        NixActionDescriptorV1::ConfigPatch {
            option_path,
            value,
            expected_config_digest,
        } => {
            if option_path.trim().is_empty() {
                return Err(NixAuthorizationErrorV1::EmptyField(
                    "config patch option path",
                ));
            }
            if value.trim().is_empty() {
                return Err(NixAuthorizationErrorV1::EmptyField("config patch value"));
            }
            if expected_config_digest.len() != 64
                || !expected_config_digest
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit())
            {
                return Err(NixAuthorizationErrorV1::InvalidTypedCommand(
                    "config patch expected config digest must be 64 hex characters".to_string(),
                ));
            }
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
        | NixActionDescriptorV1::FlakeInit { .. }
        | NixActionDescriptorV1::HomeManagerSwitch { .. } => NixActionScopeV1::UserModify,
        NixActionDescriptorV1::EnvSwitchGeneration { .. } => NixActionScopeV1::SystemCritical,
        NixActionDescriptorV1::EnvDeleteGenerations { .. }
        | NixActionDescriptorV1::EnvDeleteGenerationsOlderThan { .. } => NixActionScopeV1::Destructive,

        NixActionDescriptorV1::RebuildTest { .. }
        | NixActionDescriptorV1::RebuildBoot { .. }
        | NixActionDescriptorV1::Service { .. } => NixActionScopeV1::SystemModify,
        NixActionDescriptorV1::ConfigPatch { .. } | NixActionDescriptorV1::RebuildSwitch { .. } => {
            NixActionScopeV1::SystemCritical
        }
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
        NixActionDescriptorV1::EnvSwitchGeneration { generation } => {
            put_u8(h, 19);
            put_u32(h, *generation);
        }
        NixActionDescriptorV1::EnvDeleteGenerations { keep_last } => {
            put_u8(h, 20);
            put_u64(h, *keep_last as u64);
        }
        NixActionDescriptorV1::EnvDeleteGenerationsOlderThan { days } => {
            put_u8(h, 21);
            put_u32(h, *days);
        }
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
        NixActionDescriptorV1::FlakeInit { template } => {
            put_u8(h, 22);
            put_opt_str(h, template.as_deref());
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
        NixActionDescriptorV1::Service { operation, unit } => {
            put_u8(h, 17);
            put_u8(
                h,
                match operation {
                    NixServiceOperationKindV1::Start => 0,
                    NixServiceOperationKindV1::Stop => 1,
                    NixServiceOperationKindV1::Restart => 2,
                    NixServiceOperationKindV1::Reload => 3,
                    NixServiceOperationKindV1::Enable => 4,
                    NixServiceOperationKindV1::Disable => 5,
                },
            );
            put_str(h, unit);
        }
        NixActionDescriptorV1::ConfigPatch {
            option_path,
            value,
            expected_config_digest,
        } => {
            put_u8(h, 18);
            put_str(h, option_path);
            put_str(h, value);
            put_str(h, expected_config_digest);
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
    fn new_typed_effect_variants_are_exactly_described_and_scoped() {
        let switch_generation = NixOSCommand::EnvSwitchGeneration { generation: 42 };
        let descriptor = NixActionDescriptorV1::try_from(&switch_generation).unwrap();
        assert_eq!(
            descriptor,
            NixActionDescriptorV1::EnvSwitchGeneration { generation: 42 }
        );
        assert_eq!(
            minimum_scope_for_action(&descriptor),
            NixActionScopeV1::SystemCritical
        );

        let delete_count = NixOSCommand::EnvDeleteGenerations { keep_last: 5 };
        let descriptor = NixActionDescriptorV1::try_from(&delete_count).unwrap();
        assert_eq!(
            descriptor,
            NixActionDescriptorV1::EnvDeleteGenerations { keep_last: 5 }
        );
        assert_eq!(
            minimum_scope_for_action(&descriptor),
            NixActionScopeV1::Destructive
        );

        let delete_age = NixOSCommand::EnvDeleteGenerationsOlderThan { days: 30 };
        let descriptor = NixActionDescriptorV1::try_from(&delete_age).unwrap();
        assert_eq!(
            descriptor,
            NixActionDescriptorV1::EnvDeleteGenerationsOlderThan { days: 30 }
        );
        assert_eq!(
            minimum_scope_for_action(&descriptor),
            NixActionScopeV1::Destructive
        );

        let init = NixOSCommand::Flake {
            operation: FlakeOperation::Init {
                template: Some("templates#minimal".to_string()),
            },
        };
        let descriptor = NixActionDescriptorV1::try_from(&init).unwrap();
        assert_eq!(
            descriptor,
            NixActionDescriptorV1::FlakeInit {
                template: Some("templates#minimal".to_string())
            }
        );
        assert_eq!(
            minimum_scope_for_action(&descriptor),
            NixActionScopeV1::UserModify
        );
    }

    #[test]
    fn typed_effect_parameter_changes_produce_distinct_authority_digests() {
        let a = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::EnvSwitchGeneration { generation: 42 },
        )
        .unwrap();
        let b = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::EnvSwitchGeneration { generation: 43 },
        )
        .unwrap();
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());

        let c = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::Flake {
                operation: FlakeOperation::Init {
                    template: Some("templates#a".to_string()),
                },
            },
        )
        .unwrap();
        let d = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::Flake {
                operation: FlakeOperation::Init {
                    template: Some("templates#b".to_string()),
                },
            },
        )
        .unwrap();
        assert_ne!(c.digest().unwrap(), d.digest().unwrap());
    }

    #[test]
    fn invalid_new_typed_effect_parameters_fail_closed() {
        assert!(validate_action_shape(&NixActionDescriptorV1::EnvSwitchGeneration {
            generation: 0,
        })
        .is_err());
        assert!(validate_action_shape(&NixActionDescriptorV1::EnvDeleteGenerations {
            keep_last: 0,
        })
        .is_err());
        assert!(validate_action_shape(
            &NixActionDescriptorV1::EnvDeleteGenerationsOlderThan { days: 0 }
        )
        .is_err());
        assert!(validate_action_shape(&NixActionDescriptorV1::FlakeInit {
            template: Some(String::new()),
        })
        .is_err());
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

    fn service_context() -> NixServiceEffectContextV1 {
        NixServiceEffectContextV1::new(
            NixServiceOperationKindV1::Restart,
            "nginx.service",
            42,
            &"11".repeat(32),
            &"22".repeat(32),
            &"44".repeat(32),
            Some("33".repeat(16)),
            1_000,
        )
        .unwrap()
    }

    fn contextual_service_intent() -> NixActionIntentV1 {
        NixActionIntentV1::from_command_with_service_effect_context(
            "host:x",
            "nixward-service-pre-state-v1|generation=42|unit=nginx.service|state=1111111111111111111111111111111111111111111111111111111111111111".into(),
            &NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit: "nginx.service".to_string(),
            },
            NixServiceEffectContextV1::new(
                NixServiceOperationKindV1::Restart,
                "nginx.service",
                42,
                "1111111111111111111111111111111111111111111111111111111111111111",
                "2222222222222222222222222222222222222222222222222222222222222222",
                "4444444444444444444444444444444444444444444444444444444444444444",
                Some("3333333333333333333333333333333333".into()),
                1_000,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn service_effect_context_is_part_of_intent_identity() {
        let base = contextual_service_intent();
        let mut changed = base.clone();
        changed.service_effect_context.as_mut().unwrap().authorized_definition_digest =
            "4444444444444444444444444444444444444444444444444444444444444444".into();
        changed.validate_shape().unwrap();
        assert_ne!(base.digest().unwrap(), changed.digest().unwrap());
    }

    #[test]
    fn service_definition_capture_binding_matches_unit_and_context_commitments() {
        let evidence = NixSystemdUnitDefinitionContentEvidenceV1 {
            unit: "nginx.service".into(),
            source_identity_digest: "11".repeat(32),
            manager_owner: ":1.42".into(),
            bus_id: "0123456789abcdef0123456789abcdef".into(),
            files: vec![NixSystemdUnitDefinitionContentFileV1 {
                path: "/nix/store/nginx.service".into(),
                resolved_path: None,
                byte_len: 3,
                content_digest: "22".repeat(32),
            }],
            captured_at_monotonic_us: 1,
        };
        let sealed =
            NixVerifiedServiceDefinitionContentV1::from_observer(evidence.clone()).unwrap();

        let bare_intent = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".into()),
            &NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit: "nginx.service".into(),
            },
        )
        .unwrap();
        assert_eq!(
            validate_service_definition_capture_binding(&bare_intent, &sealed).unwrap(),
            sealed.digest().unwrap()
        );

        let context = NixServiceEffectContextV1::from_verified_definition_content(
            NixServiceOperationKindV1::Restart,
            "nginx.service",
            42,
            "aa".repeat(32),
            Some("bb".repeat(16)),
            1_000,
            &sealed,
        )
        .unwrap();
        let contextual = bare_intent.with_service_effect_context(context).unwrap();
        assert_eq!(
            validate_service_definition_capture_binding(&contextual, &sealed).unwrap(),
            sealed.digest().unwrap()
        );

        let mut altered_evidence = evidence;
        altered_evidence.files[0].content_digest = "33".repeat(32);
        let altered =
            NixVerifiedServiceDefinitionContentV1::from_observer(altered_evidence).unwrap();
        assert_eq!(
            validate_service_definition_capture_binding(&contextual, &altered).unwrap_err(),
            NixAuthorizationErrorV1::DefinitionContentCaptureMismatch
        );
    }

    #[test]
    fn service_authorization_requires_effect_context() {
        let intent = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".into()),
            &NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit: "nginx.service".into(),
            },
        )
        .unwrap();
        assert_eq!(
            LiveNixAuthorizationV1::local_explicit_confirmation(
                &intent,
                "approval:test",
                1,
                None,
            )
            .unwrap_err(),
            NixAuthorizationErrorV1::MissingServiceDefinitionContentCapture
        );
    }

    #[test]
    fn service_authorization_with_unsealed_context_still_fails_closed() {
        let intent = contextual_service_intent();
        assert_eq!(
            LiveNixAuthorizationV1::local_explicit_confirmation(
                &intent,
                "approval:test",
                1,
                None,
            )
            .unwrap_err(),
            NixAuthorizationErrorV1::MissingServiceDefinitionContentCapture
        );
    }

    #[test]
    fn service_context_fields_must_match_typed_action() {
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx.service".into(),
        };
        let intent = NixActionIntentV1::from_command("host:x", Some("generation:42".into()), &command)
            .unwrap();
        let mut context = service_context();
        context.operation = NixServiceOperationKindV1::Stop;
        assert_eq!(
            intent.with_service_effect_context(context).unwrap_err(),
            NixAuthorizationErrorV1::ServiceEffectContextMismatch
        );
    }

    #[test]
    fn typed_service_action_is_governed_and_exact() {
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx.service".to_string(),
        };
        let intent =
            NixActionIntentV1::from_command("host:x", Some("generation:42".to_string()), &command)
                .unwrap();

        assert_eq!(intent.maximum_scope, NixActionScopeV1::SystemModify);
        assert!(matches!(
            intent.action,
            NixActionDescriptorV1::Service {
                operation: NixServiceOperationKindV1::Restart,
                ref unit,
            } if unit == "nginx.service"
        ));
    }

    #[test]
    fn config_patch_intent_is_exact_and_system_critical() {
        let a = NixOSCommand::ConfigPatch {
            option_path: "services.nginx.enable".to_string(),
            value: "true".to_string(),
            expected_config_digest: "ab".repeat(32),
        };
        let intent =
            NixActionIntentV1::from_command("host:x", Some("generation:42".to_string()), &a)
                .unwrap();
        assert_eq!(intent.maximum_scope, NixActionScopeV1::SystemCritical);

        let mut changed = a.clone();
        if let NixOSCommand::ConfigPatch { value, .. } = &mut changed {
            *value = "false".to_string();
        }
        let changed_intent =
            NixActionIntentV1::from_command("host:x", Some("generation:42".to_string()), &changed)
                .unwrap();
        assert_ne!(intent.digest().unwrap(), changed_intent.digest().unwrap());
    }

    #[test]
    fn direct_descriptor_conversion_rejects_invalid_typed_service() {
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx*.service".to_string(),
        };
        assert!(matches!(
            NixActionDescriptorV1::try_from(&command),
            Err(NixAuthorizationErrorV1::InvalidTypedCommand(_))
        ));
    }

    #[test]
    fn typed_service_identity_is_operation_and_unit_sensitive() {
        let restart = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit: "nginx.service".to_string(),
            },
        )
        .unwrap();

        let enable = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Enable,
                unit: "nginx.service".to_string(),
            },
        )
        .unwrap();

        let postgres = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit: "postgresql.service".to_string(),
            },
        )
        .unwrap();

        assert_ne!(restart.digest().unwrap(), enable.digest().unwrap());
        assert_ne!(restart.digest().unwrap(), postgres.digest().unwrap());
    }

    #[test]
    fn invalid_typed_service_cannot_enter_governed_v1() {
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx*.service".to_string(),
        };
        assert!(matches!(
            NixActionIntentV1::from_command("host:x", None, &command),
            Err(NixAuthorizationErrorV1::InvalidTypedCommand(_))
        ));
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
    fn live_authority_preserves_bound_pre_state_identity() {
        let intent = NixActionIntentV1::from_command(
            "host:x",
            Some("generation:42".to_string()),
            &rebuild(),
        )
        .unwrap();
        assert_eq!(intent.pre_state_identity.as_deref(), Some("generation:42"));
    }

    #[test]
    fn local_service_authorization_carries_effect_context_digest() {
        let intent = contextual_service_intent();
        let live = LiveNixAuthorizationV1::local_explicit_confirmation(
            &intent,
            "approval:test",
            1_000,
            Some(2_000),
        )
        .unwrap();
        let expected = intent
            .service_effect_context
            .as_ref()
            .unwrap()
            .digest()
            .unwrap();
        assert_eq!(
            live.audit_record()
                .service_effect_context_digest
                .as_deref(),
            Some(expected.as_str())
        );
        live.audit_record().validate_against_intent(&intent).unwrap();
    }

    #[test]
    fn authorization_record_without_service_context_cannot_validate_service_intent() {
        let intent = contextual_service_intent();
        let record = NixExecutionAuthorizationRecordV1 {
            action_intent_digest: intent.digest().unwrap(),
            service_effect_context_digest: None,
            profile: NixAuthorizationProfileV1::LocalExplicitConfirmation,
            authority_ref: "approval:test".into(),
            issued_at_unix_ms: 1,
            expires_at_unix_ms: None,
            decision: NixAuthorizationDecisionV1::Approved,
        };
        assert_eq!(
            record.validate_against_intent(&intent).unwrap_err(),
            NixAuthorizationErrorV1::ServiceEffectContextMismatch
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
