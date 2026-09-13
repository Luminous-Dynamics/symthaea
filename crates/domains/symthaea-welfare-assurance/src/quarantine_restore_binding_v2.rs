// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed cross-ledger binding for V2 episodic restore preparation.
//!
//! This object carries correlation evidence only. It does not authorize a restore and cannot be
//! constructed from a permit or arbitrary digest by external callers. The generic V2 execution
//! adapter first produces `PreparedExecutionContextV2` from a durably persisted generic Prepared
//! record; the domain restore layer may then derive this binding and commit it into its own
//! hash-chained `RestorePreparedV2` event.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use thiserror::Error;

use crate::prepared_execution_context_v2::PreparedExecutionContextV2;

const MAX_EXECUTION_ID_BYTES: usize = 256;
const MAX_PREPARED_PERSISTENCE_REF_BYTES: usize = 2048;

/// Exact generic-Prepared evidence to be embedded by the domain restore write-ahead transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestorePreparedBindingV2 {
    execution_id: String,
    generic_prepared_digest: Sha256Digest,
    generic_prepared_at_unix_s: u64,
    generic_prepared_persistence_ref: String,
}

impl RestorePreparedBindingV2 {
    /// Derive a domain-persistable correlation binding from the opaque V2 execution context.
    ///
    /// `domain_prepared_at_unix_s` is the timestamp the quarantine ledger intends to record. It may
    /// equal or follow the generic Prepared time, but cannot precede it.
    pub fn from_context(
        context: &PreparedExecutionContextV2,
        domain_prepared_at_unix_s: u64,
    ) -> Result<Self, RestorePreparedBindingV2Error> {
        let value = Self {
            execution_id: context.execution_id().to_string(),
            generic_prepared_digest: context.prepared_digest(),
            generic_prepared_at_unix_s: context.prepared_at_unix_s(),
            generic_prepared_persistence_ref: context.prepared_persistence_ref().to_string(),
        };
        value.validate(domain_prepared_at_unix_s)?;
        Ok(value)
    }

    pub fn validate(
        &self,
        domain_prepared_at_unix_s: u64,
    ) -> Result<(), RestorePreparedBindingV2Error> {
        validate_text("execution_id", &self.execution_id, MAX_EXECUTION_ID_BYTES)?;
        validate_text(
            "generic_prepared_persistence_ref",
            &self.generic_prepared_persistence_ref,
            MAX_PREPARED_PERSISTENCE_REF_BYTES,
        )?;
        if self.generic_prepared_digest.0 == [0; 32] {
            return Err(RestorePreparedBindingV2Error::ZeroPreparedDigest);
        }
        if self.generic_prepared_at_unix_s > domain_prepared_at_unix_s {
            return Err(RestorePreparedBindingV2Error::DomainPreparePredatesGenericPrepare {
                generic_prepared_at_unix_s: self.generic_prepared_at_unix_s,
                domain_prepared_at_unix_s,
            });
        }
        Ok(())
    }

    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    pub fn generic_prepared_digest(&self) -> Sha256Digest {
        self.generic_prepared_digest
    }

    pub fn generic_prepared_at_unix_s(&self) -> u64 {
        self.generic_prepared_at_unix_s
    }

    pub fn generic_prepared_persistence_ref(&self) -> &str {
        &self.generic_prepared_persistence_ref
    }
}

fn validate_text(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), RestorePreparedBindingV2Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(RestorePreparedBindingV2Error::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RestorePreparedBindingV2Error {
    #[error("invalid restore-prepared V2 text field `{field}`: {value:?}")]
    InvalidText { field: &'static str, value: String },
    #[error("generic Prepared digest must not be zero")]
    ZeroPreparedDigest,
    #[error(
        "domain RestorePreparedV2 time predates generic Prepared time: generic={generic_prepared_at_unix_s}, domain={domain_prepared_at_unix_s}"
    )]
    DomainPreparePredatesGenericPrepare {
        generic_prepared_at_unix_s: u64,
        domain_prepared_at_unix_s: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;

    use crate::execution_recovery::{
        EXECUTION_JOURNAL_SCHEMA, PreparedInterventionExecution, digest_prepared_execution,
    };

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared() -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: "exec:restore:binding:v2".into(),
            authority_id: "authority:restore:binding:v2".into(),
            target_id: "symthaea:self:episodic-memory:instance:binding-v2".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:binding:v2".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    fn context() -> PreparedExecutionContextV2 {
        let prepared = prepared();
        let prepared_digest = digest_prepared_execution(&prepared).unwrap();
        PreparedExecutionContextV2::verify_exact_prepared_digest(&prepared, prepared_digest)
            .unwrap();
        PreparedExecutionContextV2::from_verified_durable(
            &prepared,
            prepared_digest,
            "execution-journal:prepared:binding:v2".into(),
        )
    }

    #[test]
    fn context_derives_exact_domain_binding() {
        let context = context();
        let binding = RestorePreparedBindingV2::from_context(&context, 110).unwrap();
        assert_eq!(binding.execution_id(), context.execution_id());
        assert_eq!(binding.generic_prepared_digest(), context.prepared_digest());
        assert_eq!(
            binding.generic_prepared_at_unix_s(),
            context.prepared_at_unix_s()
        );
        assert_eq!(
            binding.generic_prepared_persistence_ref(),
            context.prepared_persistence_ref()
        );
    }

    #[test]
    fn domain_prepare_cannot_predate_generic_prepare() {
        let context = context();
        let error = RestorePreparedBindingV2::from_context(&context, 99).unwrap_err();
        assert_eq!(
            error,
            RestorePreparedBindingV2Error::DomainPreparePredatesGenericPrepare {
                generic_prepared_at_unix_s: 100,
                domain_prepared_at_unix_s: 99,
            }
        );
    }

    #[test]
    fn binding_serialization_roundtrips_exactly() {
        let binding = RestorePreparedBindingV2::from_context(&context(), 110).unwrap();
        let bytes = bincode::serialize(&binding).unwrap();
        let recovered: RestorePreparedBindingV2 = bincode::deserialize(&bytes).unwrap();
        assert_eq!(recovered, binding);
        recovered.validate(110).unwrap();
    }
}
