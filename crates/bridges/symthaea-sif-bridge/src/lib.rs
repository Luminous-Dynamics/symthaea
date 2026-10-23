// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea bridge for Sovereign Intelligence Fabric (SIF) v0.1.
//!
//! This crate defines the *public statement binding* for a verifiable,
//! minimum-disclosure query. It does not claim that every configured proof system
//! proves this statement yet; proof-generation backends remain separately gated and
//! must publish their own evidence. The bridge exists so Mycelix receipts can bind to
//! Symthaea proof artifacts without depending on Symthaea's internal representations.

use serde::{Deserialize, Serialize};

/// Stable schema label for the v0.1 verifiable-query binding.
pub const SIF_VERIFIABLE_QUERY_SCHEMA: &str = "symthaea-sif-verifiable-query-v1";

/// Cross-stack proof kind used by a generic SIF proof binding.
pub const SIF_PROOF_KIND: &str = "symthaea/verifiable-query";

/// Cross-stack proof-binding version.
pub const SIF_PROOF_VERSION: u16 = 1;

/// The strongest class of information returned by the query.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum MinimumDisclosureMode {
    /// Boolean/categorical predicate only; source records stay local.
    PredicateOnly,
    /// Aggregate statistic rather than person-level records.
    Aggregate,
    /// Explicit subset of authorized fields.
    SelectiveFields,
    /// Underlying evidence was released under a separately authorized capability.
    RawEvidence,
}

impl MinimumDisclosureMode {
    fn code(self) -> u8 {
        match self {
            Self::PredicateOnly => 1,
            Self::Aggregate => 2,
            Self::SelectiveFields => 3,
            Self::RawEvidence => 4,
        }
    }
}

/// Budget state bound into a query proof/attestation.
///
/// Mycelix owns the policy decision. Symthaea merely commits to the budget state it
/// was given so a verifier can detect if the computation was replayed beside a
/// different query/privacy-budget context.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct QueryBudgetWitness {
    pub budget_class: String,
    pub limit: u64,
    pub consumed_before: u64,
    pub consumed_after: u64,
    pub policy_commitment: [u8; 32],
}

impl QueryBudgetWitness {
    /// Structural sanity only; policy remains external.
    pub fn is_structurally_valid(&self) -> bool {
        !self.budget_class.is_empty()
            && self.policy_commitment != [0; 32]
            && self.consumed_before <= self.consumed_after
    }

    /// Whether the supplied post-query budget state is still within the stated limit.
    pub fn within_budget(&self) -> bool {
        self.is_structurally_valid() && self.consumed_after <= self.limit
    }
}

/// Stable export shape consumed by a generic SIF `ProofBinding` adapter.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SifProofExport {
    pub scheme: String,
    pub version: u16,
    pub digest: [u8; 32],
}

/// Public statement binding for a verifiable person-linked query.
///
/// Every digest is opaque to this bridge. The producer of each commitment owns the
/// canonical encoding of the underlying object. The bridge's job is to bind those
/// commitments together so a valid proof cannot be transplanted onto a different
/// query, policy, subject pseudonym, result, or disclosure shape.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SifVerifiableQueryBinding {
    pub schema: String,
    /// Mycelix SIF `QueryCapability` commitment.
    pub query_commitment: [u8; 32],
    /// Policy/version commitment authorizing the computation.
    pub policy_commitment: [u8; 32],
    /// Pseudonymous subject commitment; raw identity is not required here.
    pub subject_commitment: [u8; 32],
    /// Commitment to the exact predicate/model/statement evaluated.
    pub predicate_commitment: [u8; 32],
    /// Commitment to the returned result.
    pub result_commitment: [u8; 32],
    /// Commitment to the disclosure shape/fields exposed by the result.
    pub disclosure_commitment: [u8; 32],
    pub disclosure_mode: MinimumDisclosureMode,
    /// Backend/circuit family, e.g. `binius-hdc`, `winterfell-range`, `risc0`.
    pub proof_system: String,
    /// Version of the proof statement/circuit contract.
    pub statement_version: u16,
    /// Digest of the actual proof or attestation artifact.
    pub proof_digest: [u8; 32],
    /// Optional budget state committed beside the proof.
    pub budget_witness: Option<QueryBudgetWitness>,
}

impl SifVerifiableQueryBinding {
    /// Create a binding with the stable v0.1 schema label.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_commitment: [u8; 32],
        policy_commitment: [u8; 32],
        subject_commitment: [u8; 32],
        predicate_commitment: [u8; 32],
        result_commitment: [u8; 32],
        disclosure_commitment: [u8; 32],
        disclosure_mode: MinimumDisclosureMode,
        proof_system: impl Into<String>,
        statement_version: u16,
        proof_digest: [u8; 32],
        budget_witness: Option<QueryBudgetWitness>,
    ) -> Self {
        Self {
            schema: SIF_VERIFIABLE_QUERY_SCHEMA.to_string(),
            query_commitment,
            policy_commitment,
            subject_commitment,
            predicate_commitment,
            result_commitment,
            disclosure_commitment,
            disclosure_mode,
            proof_system: proof_system.into(),
            statement_version,
            proof_digest,
            budget_witness,
        }
    }

    /// Structural validation before publishing a proof binding.
    pub fn validate(&self) -> Result<(), SifBindingError> {
        if self.schema != SIF_VERIFIABLE_QUERY_SCHEMA {
            return Err(SifBindingError::UnsupportedSchema(self.schema.clone()));
        }
        if self.query_commitment == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("query"));
        }
        if self.policy_commitment == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("policy"));
        }
        if self.subject_commitment == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("subject"));
        }
        if self.predicate_commitment == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("predicate"));
        }
        if self.result_commitment == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("result"));
        }
        if self.disclosure_commitment == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("disclosure"));
        }
        if self.proof_digest == [0; 32] {
            return Err(SifBindingError::EmptyCommitment("proof"));
        }
        if self.proof_system.is_empty() {
            return Err(SifBindingError::EmptyProofSystem);
        }
        if self.statement_version == 0 {
            return Err(SifBindingError::ZeroStatementVersion);
        }
        if let Some(budget) = &self.budget_witness {
            if !budget.is_structurally_valid() {
                return Err(SifBindingError::InvalidBudgetWitness);
            }
        }
        Ok(())
    }

    /// Canonical public statement bytes committed by the cross-stack digest.
    pub fn statement_message(&self) -> Result<Vec<u8>, SifBindingError> {
        self.validate()?;
        let mut out = Vec::with_capacity(384);
        push_bytes(&mut out, b"symthaea:sif-verifiable-query:v1");
        push_bytes(&mut out, self.schema.as_bytes());
        out.extend_from_slice(&self.query_commitment);
        out.extend_from_slice(&self.policy_commitment);
        out.extend_from_slice(&self.subject_commitment);
        out.extend_from_slice(&self.predicate_commitment);
        out.extend_from_slice(&self.result_commitment);
        out.extend_from_slice(&self.disclosure_commitment);
        out.push(self.disclosure_mode.code());
        push_bytes(&mut out, self.proof_system.as_bytes());
        out.extend_from_slice(&self.statement_version.to_le_bytes());
        out.extend_from_slice(&self.proof_digest);
        match &self.budget_witness {
            Some(budget) => {
                out.push(1);
                push_bytes(&mut out, budget.budget_class.as_bytes());
                out.extend_from_slice(&budget.limit.to_le_bytes());
                out.extend_from_slice(&budget.consumed_before.to_le_bytes());
                out.extend_from_slice(&budget.consumed_after.to_le_bytes());
                out.extend_from_slice(&budget.policy_commitment);
            }
            None => out.push(0),
        }
        Ok(out)
    }

    /// Domain-separated BLAKE3 binding consumed by a SIF access receipt.
    pub fn commitment(&self) -> Result<[u8; 32], SifBindingError> {
        Ok(*blake3::hash(&self.statement_message()?).as_bytes())
    }

    /// Generic export suitable for mapping into Mycelix's proof binding type.
    pub fn export(&self) -> Result<SifProofExport, SifBindingError> {
        Ok(SifProofExport {
            scheme: SIF_PROOF_KIND.to_string(),
            version: SIF_PROOF_VERSION,
            digest: self.commitment()?,
        })
    }
}

/// Structural errors in a SIF verifiable-query binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SifBindingError {
    UnsupportedSchema(String),
    EmptyCommitment(&'static str),
    EmptyProofSystem,
    ZeroStatementVersion,
    InvalidBudgetWitness,
}

impl core::fmt::Display for SifBindingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedSchema(schema) => write!(f, "unsupported SIF binding schema: {schema}"),
            Self::EmptyCommitment(name) => write!(f, "{name} commitment must not be all-zero"),
            Self::EmptyProofSystem => write!(f, "proof_system must not be empty"),
            Self::ZeroStatementVersion => write!(f, "statement_version must be non-zero"),
            Self::InvalidBudgetWitness => write!(f, "budget witness is structurally invalid"),
        }
    }
}

impl std::error::Error for SifBindingError {}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(mode: MinimumDisclosureMode) -> SifVerifiableQueryBinding {
        SifVerifiableQueryBinding::new(
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            [5; 32],
            [6; 32],
            mode,
            "binius-hdc",
            1,
            [7; 32],
            Some(QueryBudgetWitness {
                budget_class: "person-linked-standard".into(),
                limit: 100,
                consumed_before: 4,
                consumed_after: 5,
                policy_commitment: [2; 32],
            }),
        )
    }

    #[test]
    fn predicate_only_export_has_stable_shape() {
        let export = binding(MinimumDisclosureMode::PredicateOnly)
            .export()
            .unwrap();
        assert_eq!(export.scheme, SIF_PROOF_KIND);
        assert_eq!(export.version, SIF_PROOF_VERSION);
        assert_ne!(export.digest, [0; 32]);
    }

    #[test]
    fn binding_changes_with_query_result_and_disclosure_mode() {
        let original = binding(MinimumDisclosureMode::PredicateOnly);

        let mut changed_query = original.clone();
        changed_query.query_commitment = [9; 32];
        assert_ne!(original.commitment().unwrap(), changed_query.commitment().unwrap());

        let mut changed_result = original.clone();
        changed_result.result_commitment = [10; 32];
        assert_ne!(original.commitment().unwrap(), changed_result.commitment().unwrap());

        let raw = binding(MinimumDisclosureMode::RawEvidence);
        assert_ne!(original.commitment().unwrap(), raw.commitment().unwrap());
    }

    #[test]
    fn budget_state_is_bound_into_the_proof_commitment() {
        let original = binding(MinimumDisclosureMode::PredicateOnly);
        let mut changed = original.clone();
        changed.budget_witness.as_mut().unwrap().consumed_after = 6;
        assert_ne!(original.commitment().unwrap(), changed.commitment().unwrap());
    }

    #[test]
    fn budget_helper_does_not_make_policy_decisions() {
        let within = QueryBudgetWitness {
            budget_class: "standard".into(),
            limit: 10,
            consumed_before: 8,
            consumed_after: 9,
            policy_commitment: [1; 32],
        };
        assert!(within.within_budget());

        let exhausted = QueryBudgetWitness {
            consumed_after: 11,
            ..within
        };
        assert!(!exhausted.within_budget());
        assert!(exhausted.is_structurally_valid());
    }

    #[test]
    fn placeholders_are_rejected() {
        let mut value = binding(MinimumDisclosureMode::PredicateOnly);
        value.proof_digest = [0; 32];
        assert_eq!(
            value.commitment().unwrap_err(),
            SifBindingError::EmptyCommitment("proof")
        );
    }
}
