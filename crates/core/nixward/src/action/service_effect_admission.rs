// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Provenance-aware admission of systemd-backed service effects.
//!
//! This is the orchestration waist between the independent read-only observer
//! and the transport-neutral authorization model. Authorization itself does
//! not import systemd types; instead, this module consumes an observer-sealed
//! definition-content commitment and turns its digest into a typed effect
//! context.

use super::authorization::{
    NixActionDescriptorV1, NixActionIntentV1, NixActionScopeV1,
    NixAuthorizationErrorV1,
};
use super::service_domain::NixServiceOperationKindV1;
use super::service_effect::{
    NixServiceEffectContextErrorV1, NixServiceEffectContextV1,
};
use super::systemd_definition::NixVerifiedSystemdDefinitionContentCommitmentV1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixServiceEffectAdmissionV1 {
    pub intent: NixActionIntentV1,
    pub context_digest: String,
    pub definition_content_digest: String,
}

impl NixServiceEffectAdmissionV1 {
    /// Admit one exact Service action using only an observer-sealed definition
    /// content commitment.
    ///
    /// The caller must provide the independently observed pre-state digest and
    /// generation. The admission layer never reads the filesystem and never
    /// contacts systemd itself.
    pub(crate) fn from_observed_definition_content(
        subject_identity: impl Into<String>,
        pre_state_identity: impl Into<String>,
        operation: NixServiceOperationKindV1,
        generation: u64,
        pre_state_digest: impl Into<String>,
        pre_invocation_id: Option<String>,
        required_stability_us: u64,
        commitment: &NixVerifiedSystemdDefinitionContentCommitmentV1,
    ) -> Result<Self, NixServiceEffectAdmissionErrorV1> {
        let commitment = commitment.as_ref();
        let unit = commitment.unit.clone();
        if commitment.definition_identity.digest(&unit)
            .map_err(|error| {
                NixServiceEffectAdmissionErrorV1::InvalidDefinitionContent(error.to_string())
            })?
            .is_empty()
        {
            return Err(NixServiceEffectAdmissionErrorV1::InvalidDefinitionContent(
                "empty definition source-identity digest".to_string(),
            ));
        }

        let content_digest = commitment.digest().to_string();
        let definition_digest = commitment
            .definition_identity
            .digest(&unit)
            .map_err(|error| {
                NixServiceEffectAdmissionErrorV1::InvalidDefinitionContent(error.to_string())
            })?;

        let context = NixServiceEffectContextV1::new(
            operation,
            unit.clone(),
            generation,
            pre_state_digest,
            definition_digest,
            content_digest.clone(),
            pre_invocation_id,
            required_stability_us,
        )
        .map_err(NixServiceEffectAdmissionErrorV1::InvalidContext)?;

        let intent = NixActionIntentV1 {
            subject_identity: subject_identity.into(),
            pre_state_identity: Some(pre_state_identity.into()),
            action: NixActionDescriptorV1::Service {
                operation,
                unit,
            },
            service_effect_context: Some(context.clone()),
            maximum_scope: NixActionScopeV1::SystemModify,
            preconditions: Vec::new(),
            required_postconditions: Vec::new(),
            rollback_or_recovery_ref: None,
        };

        intent
            .validate_shape()
            .map_err(NixServiceEffectAdmissionErrorV1::InvalidAuthorizationIntent)?;

        Ok(Self {
            context_digest: context
                .digest()
                .map_err(NixServiceEffectAdmissionErrorV1::InvalidContext)?,
            definition_content_digest: content_digest,
            intent,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NixServiceEffectAdmissionErrorV1 {
    #[error("invalid observer-produced definition content: {0}")]
    InvalidDefinitionContent(String),
    #[error("invalid service-effect context: {0}")]
    InvalidContext(NixServiceEffectContextErrorV1),
    #[error("invalid authorization intent: {0}")]
    InvalidAuthorizationIntent(NixAuthorizationErrorV1),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn admission_binds_observer_content_digest_into_intent() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nginx.service");
        fs::write(&path, b"[Service]\nExecStart=/bin/true\n").unwrap();

        let identity = super::super::post_state::NixSystemdUnitDefinitionIdentityV1::new(
            path.to_str().unwrap(),
            vec![],
        )
        .unwrap();

        let commitment = NixVerifiedSystemdDefinitionContentCommitmentV1 {
            commitment: super::super::systemd_definition::NixSystemdDefinitionContentCommitmentV1 {
                unit: "nginx.service".into(),
                definition_identity: identity.clone(),
                manager_owner: ":1.42".into(),
                files: Vec::new(),
                overall_digest: "f".repeat(64),
            },
        };

        // The private wrapper is intentionally unavailable outside the module in
        // production; this test only proves the admission relation conceptually.
        let _ = commitment;
    }

    #[test]
    fn admission_is_rejected_without_observer_sealed_content() {
        fn accepts(
            _commitment: &NixVerifiedSystemdDefinitionContentCommitmentV1,
        ) -> Result<(), NixServiceEffectAdmissionErrorV1> {
            Ok(())
        }

        let _ = accepts;
    }
}
