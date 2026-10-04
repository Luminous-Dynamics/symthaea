// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sanitized, exact operator-facing projection of a pending Nixward approval.
//!
//! This is a disclosure contract, not a capability. It intentionally contains
//! only the fields needed for an operator to identify exactly what is awaiting
//! approval. It contains no socket credentials, peer metadata, store internals,
//! or live authority material.
//!
//! The projection digest commits to the complete canonical projection. A later
//! approval bridge may require that digest in addition to the exact request and
//! approver evidence so that "what was displayed" cannot silently diverge from
//! "what was approved".

use super::local_approval::{digest_display, LocalApprovalErrorV1, PendingNixApprovalRequestV1};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const PROJECTION_DOMAIN: &[u8] = b"nixward-pending-nix-approval-projection-v1";
const PROJECTION_SCHEMA_VERSION: u16 = 1;
const MAX_OPERATOR_VISIBLE_ACTION_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingNixApprovalProjectionV1 {
    pub schema_version: u16,
    pub request_id: String,
    pub daemon_incarnation_ref: String,
    pub exact_action_intent_digest: String,
    pub machine_target_ref: String,
    pub operator_visible_action: String,
    pub operator_visible_action_digest: String,
    pub required_approval_profile: String,
    pub created_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub projection_digest: String,
}

impl PendingNixApprovalProjectionV1 {
    /// Build only from the installed request's exact canonical data.
    ///
    /// The operator-visible string is supplied separately because the pending
    /// request stores its digest, not the original disclosure text.
    pub fn from_request(
        request: &PendingNixApprovalRequestV1,
        operator_visible_action: &str,
    ) -> Result<Self, LocalApprovalProjectionErrorV1> {
        request.validate_shape()?;
        if operator_visible_action.trim().is_empty() {
            return Err(LocalApprovalProjectionErrorV1::EmptyOperatorVisibleAction);
        }
        if operator_visible_action.len() > MAX_OPERATOR_VISIBLE_ACTION_BYTES {
            return Err(LocalApprovalProjectionErrorV1::OversizedOperatorVisibleAction);
        }

        let action_digest = digest_display(operator_visible_action);
        if action_digest != request.displayed_action_digest {
            return Err(LocalApprovalProjectionErrorV1::DisplayedActionDigestMismatch);
        }

        let request_id = request.request_id()?;
        let mut projection = Self {
            schema_version: PROJECTION_SCHEMA_VERSION,
            request_id,
            daemon_incarnation_ref: request.daemon_incarnation_id.clone(),
            exact_action_intent_digest: request.action_intent_digest.clone(),
            machine_target_ref: request.machine_target_ref.clone(),
            operator_visible_action: operator_visible_action.to_owned(),
            operator_visible_action_digest: action_digest,
            required_approval_profile: request.authority_profile_ref.clone(),
            created_at_unix_ms: request.created_at_unix_ms,
            expires_at_unix_ms: request.expires_at_unix_ms,
            projection_digest: String::new(),
        };
        projection.projection_digest = projection.compute_digest()?;
        Ok(projection)
    }

    pub fn compute_digest(&self) -> Result<String, LocalApprovalProjectionErrorV1> {
        self.validate_without_self_digest()?;
        let mut h = Hasher::new();
        h.update(PROJECTION_DOMAIN);
        put_u16(&mut h, self.schema_version);
        put_str(&mut h, &self.request_id);
        put_str(&mut h, &self.daemon_incarnation_ref);
        put_str(&mut h, &self.exact_action_intent_digest);
        put_str(&mut h, &self.machine_target_ref);
        put_str(&mut h, &self.operator_visible_action);
        put_str(&mut h, &self.operator_visible_action_digest);
        put_str(&mut h, &self.required_approval_profile);
        put_u64(&mut h, self.created_at_unix_ms);
        put_u64(&mut h, self.expires_at_unix_ms);
        Ok(h.finalize().to_hex().to_string())
    }

    pub fn validate(&self) -> Result<(), LocalApprovalProjectionErrorV1> {
        self.validate_without_self_digest()?;
        if self.projection_digest != self.compute_digest()? {
            return Err(LocalApprovalProjectionErrorV1::ProjectionDigestMismatch);
        }
        Ok(())
    }

    pub fn matches_request(
        &self,
        request: &PendingNixApprovalRequestV1,
        operator_visible_action: &str,
    ) -> Result<bool, LocalApprovalProjectionErrorV1> {
        let expected = Self::from_request(request, operator_visible_action)?;
        Ok(self.validate().is_ok()
            && self.projection_digest == expected.projection_digest
            && self.request_id == expected.request_id)
    }

    fn validate_without_self_digest(&self) -> Result<(), LocalApprovalProjectionErrorV1> {
        if self.schema_version != PROJECTION_SCHEMA_VERSION {
            return Err(LocalApprovalProjectionErrorV1::UnsupportedSchemaVersion);
        }
        validate_digest(&self.request_id, "request id")?;
        validate_identifier(&self.daemon_incarnation_ref, "daemon incarnation ref")?;
        validate_digest(&self.exact_action_intent_digest, "exact action intent digest")?;
        validate_identifier(&self.machine_target_ref, "machine target ref")?;
        validate_identifier(&self.required_approval_profile, "required approval profile")?;
        if self.operator_visible_action.trim().is_empty() {
            return Err(LocalApprovalProjectionErrorV1::EmptyOperatorVisibleAction);
        }
        if self.operator_visible_action.len() > MAX_OPERATOR_VISIBLE_ACTION_BYTES {
            return Err(LocalApprovalProjectionErrorV1::OversizedOperatorVisibleAction);
        }
        if self.operator_visible_action.chars().any(char::is_control) {
            return Err(LocalApprovalProjectionErrorV1::ControlCharacterInOperatorVisibleAction);
        }
        validate_digest(
            &self.operator_visible_action_digest,
            "operator visible action digest",
        )?;
        if digest_display(&self.operator_visible_action) != self.operator_visible_action_digest {
            return Err(LocalApprovalProjectionErrorV1::DisplayedActionDigestMismatch);
        }
        if self.expires_at_unix_ms < self.created_at_unix_ms {
            return Err(LocalApprovalProjectionErrorV1::InvalidRequestWindow);
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LocalApprovalProjectionErrorV1 {
    #[error(transparent)]
    Approval(#[from] LocalApprovalErrorV1),
    #[error("operator-visible action must not be blank")]
    EmptyOperatorVisibleAction,
    #[error("operator-visible action exceeds projection size ceiling")]
    OversizedOperatorVisibleAction,
    #[error("operator-visible action contains a control character")]
    ControlCharacterInOperatorVisibleAction,
    #[error("operator-visible action does not match the request display digest")]
    DisplayedActionDigestMismatch,
    #[error("unsupported approval projection schema version")]
    UnsupportedSchemaVersion,
    #[error("invalid projection field: {0}")]
    InvalidField(&'static str),
    #[error("invalid canonical projection digest field: {0}")]
    InvalidDigest(&'static str),
    #[error("projection digest does not match canonical projection fields")]
    ProjectionDigestMismatch,
    #[error("approval request expires before it is created")]
    InvalidRequestWindow,
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), LocalApprovalProjectionErrorV1> {
    if value.trim().is_empty() || value.len() > 1024 {
        Err(LocalApprovalProjectionErrorV1::InvalidField(field))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), LocalApprovalProjectionErrorV1> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(LocalApprovalProjectionErrorV1::InvalidDigest(field));
    }
    Ok(())
}

fn put_u16(h: &mut Hasher, value: u16) {
    h.update(&value.to_be_bytes());
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::authorization::NixActionIntentV1;
    use crate::action::executor::NixOSCommand;

    fn request() -> PendingNixApprovalRequestV1 {
        let intent = NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &NixOSCommand::RebuildSwitch {
                flake: Some(".#workstation".to_string()),
                extra_args: vec![],
            },
        )
        .unwrap();
        PendingNixApprovalRequestV1::from_intent(
            &intent,
            "daemon-incarnation:test",
            "nixos-rebuild switch --flake .#workstation",
            "same-uid-process-v1",
            crate::action::temporal::UnixMillisV1::new(1_000),
            crate::action::temporal::UnixMillisV1::new(2_000),
            [7; 32],
        )
        .unwrap()
    }

    #[test]
    fn projection_is_exactly_derived_and_digest_stable() {
        let request = request();
        let projection =
            PendingNixApprovalProjectionV1::from_request(&request, "nixos-rebuild switch --flake .#workstation")
                .unwrap();
        assert_eq!(projection.request_id, request.request_id().unwrap());
        assert_eq!(projection.daemon_incarnation_ref, request.daemon_incarnation_id);
        assert_eq!(projection.exact_action_intent_digest, request.action_intent_digest);
        assert_eq!(projection.machine_target_ref, request.machine_target_ref);
        assert_eq!(projection.operator_visible_action_digest, request.displayed_action_digest);
        assert_eq!(projection.required_approval_profile, request.authority_profile_ref);
        assert_eq!(projection.projection_digest, projection.compute_digest().unwrap());
        projection.validate().unwrap();
    }

    #[test]
    fn changed_operator_text_cannot_be_projected_as_the_same_request() {
        let request = request();
        assert_eq!(
            PendingNixApprovalProjectionV1::from_request(&request, "different action").unwrap_err(),
            LocalApprovalProjectionErrorV1::DisplayedActionDigestMismatch
        );
    }

    #[test]
    fn changing_semantic_projection_fields_changes_digest() {
        let request = request();
        let mut projection =
            PendingNixApprovalProjectionV1::from_request(&request, "nixos-rebuild switch --flake .#workstation")
                .unwrap();
        let original = projection.projection_digest.clone();

        projection.machine_target_ref = "machine:other".to_string();
        assert_ne!(projection.compute_digest().unwrap(), original);

        projection.machine_target_ref = request.machine_target_ref.clone();
        projection.required_approval_profile = "different-profile".to_string();
        assert_ne!(projection.compute_digest().unwrap(), original);
    }

    #[test]
    fn tampered_projection_digest_is_rejected() {
        let request = request();
        let mut projection =
            PendingNixApprovalProjectionV1::from_request(&request, "nixos-rebuild switch --flake .#workstation")
                .unwrap();
        projection.projection_digest = "11".repeat(32);
        assert_eq!(
            projection.validate().unwrap_err(),
            LocalApprovalProjectionErrorV1::ProjectionDigestMismatch
        );
    }

    #[test]
    fn serde_roundtrip_preserves_canonical_digest() {
        let request = request();
        let projection =
            PendingNixApprovalProjectionV1::from_request(&request, "nixos-rebuild switch --flake .#workstation")
                .unwrap();
        let restored: PendingNixApprovalProjectionV1 =
            serde_json::from_str(&serde_json::to_string(&projection).unwrap()).unwrap();
        assert_eq!(projection, restored);
        assert_eq!(projection.projection_digest, restored.compute_digest().unwrap());
    }

    #[test]
    fn control_character_in_operator_visible_action_is_rejected() {
        let request = request();
        assert_eq!(
            PendingNixApprovalProjectionV1::from_request(
                &request,
                "nixos-rebuild switch --flake .#workstation\n[APPROVED]",
            )
            .unwrap_err(),
            LocalApprovalProjectionErrorV1::ControlCharacterInOperatorVisibleAction
        );
    }

    #[test]
    fn projection_contains_no_transport_or_execution_authority_fields() {
        let request = request();
        let projection =
            PendingNixApprovalProjectionV1::from_request(&request, "nixos-rebuild switch --flake .#workstation")
                .unwrap();
        let object = serde_json::to_value(&projection).unwrap();
        let object = object.as_object().unwrap();
        for forbidden in [
            "uid",
            "gid",
            "pid",
            "socket_path",
            "socket_credentials",
            "dispatch_permit",
            "live_authorization",
            "xenia_credential",
            "polkit_credential",
            "request_nonce",
        ] {
            assert!(!object.contains_key(forbidden), "forbidden field {forbidden}");
        }
    }
}
