// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Non-serializable provenance for one live, consumed Nixward execution authority.
//!
//! This type is intentionally separate from both durable authorization records and
//! post-state observations. Possession means the executor reached the final
//! pre-dispatch validation boundary with the exact consumed local authority.

use super::authorization::NixLocalExecutionAuthorityV1;

/// One-shot execution provenance minted only from a live local execution authority.
///
/// The type deliberately implements neither Serialize/Deserialize nor Clone. It is
/// transient evidence that a real executor held the consumed authority immediately
/// before dispatch. Durable receipts may copy the scalar lineage values, but they
/// cannot recreate this witness from serialized records.
#[must_use]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NixLiveExecutionWitnessV1 {
    action_intent_digest: String,
    approval_request_id: String,
    projection_digest: String,
    pre_state_identity: Option<String>,
    service_definition_content_digest: Option<String>,
    pre_invocation_id: Option<String>,
}

impl NixLiveExecutionWitnessV1 {
    /// Mint provenance from the exact live authority after its final pre-dispatch
    /// validation has succeeded.
    pub(crate) fn from_live_authority(
        authority: &NixLocalExecutionAuthorityV1,
    ) -> Result<Self, String> {
        let action_intent_digest = authority
            .action_intent_digest()
            .map_err(|error| format!("cannot derive live execution witness intent: {error}"))?;
        let approval_request_id = authority.approval_request_id().to_string();
        let projection_digest = authority.projection_digest().to_string();
        if approval_request_id.is_empty() || projection_digest.is_empty() {
            return Err("live execution authority has incomplete approval lineage".to_string());
        }

        Ok(Self {
            action_intent_digest,
            approval_request_id,
            projection_digest,
            pre_state_identity: authority.pre_state_identity().map(str::to_owned),
            service_definition_content_digest: authority
                .service_definition_content_digest()
                .map(str::to_owned),
            pre_invocation_id: authority.pre_invocation_id().map(str::to_owned),
        })
    }

    pub(crate) fn action_intent_digest(&self) -> &str {
        &self.action_intent_digest
    }

    pub(crate) fn approval_request_id(&self) -> &str {
        &self.approval_request_id
    }

    pub(crate) fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    pub(crate) fn pre_state_identity(&self) -> Option<&str> {
        self.pre_state_identity.as_deref()
    }

    pub(crate) fn service_definition_content_digest(&self) -> Option<&str> {
        self.service_definition_content_digest.as_deref()
    }

    pub(crate) fn pre_invocation_id(&self) -> Option<&str> {
        self.pre_invocation_id.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        action_intent_digest: impl Into<String>,
        approval_request_id: impl Into<String>,
        projection_digest: impl Into<String>,
        pre_state_identity: Option<String>,
        service_definition_content_digest: Option<String>,
        pre_invocation_id: Option<String>,
    ) -> Self {
        Self {
            action_intent_digest: action_intent_digest.into(),
            approval_request_id: approval_request_id.into(),
            projection_digest: projection_digest.into(),
            pre_state_identity,
            service_definition_content_digest,
            pre_invocation_id,
        }
    }
}
