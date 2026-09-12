// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit accepted-requirement -> proof-obligation relation boundary for ETK.
//!
//! Core theorem:
//!
//! ```text
//! requirement + obligation in the same plan != proven requirement/obligation relationship
//! ```
//!
//! This crate makes that relationship explicit and content-addressed.  An exact
//! restatement can be checked mechanically.  A derived safety obligation must
//! instead carry explicit derivation-policy, derivation-record, and acceptance
//! identities.  Those identities are audit bindings only; this crate does not
//! authenticate their producers or prove the derivation itself.

#![deny(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt;
use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionIdV1, AcceptedRequirementRevisionV1, ObligationRevisionIdV1,
    Sha256DigestV1,
};
use symthaea_formal_safety::{EvidenceKind, ProofObligation};
use thiserror::Error;

const BINDING_DOMAIN_V1: &[u8] = b"symthaea.etk-requirement-obligation-binding.v1\0";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RequirementBindingErrorV1 {
    #[error("requirement and obligation expect different evidence kinds")]
    EvidenceKindMismatch,
    #[error("ExactRestatement requires byte-exact requirement statement == obligation claim")]
    ExactRestatementClaimMismatch,
}

/// Role-safe identity of a derivation artifact describing how an accepted
/// requirement was transformed into a downstream proof obligation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DerivationRecordDigestV1(Sha256DigestV1);

impl DerivationRecordDigestV1 {
    pub fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Role-safe content identity of the policy/ruleset under which requirement
/// derivation was performed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DerivationPolicyRevisionDigestV1(Sha256DigestV1);

impl DerivationPolicyRevisionDigestV1 {
    pub fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Role-safe identity of the record that explicitly accepted the derived
/// requirement/obligation relationship into the engineering case.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingAcceptanceRecordDigestV1(Sha256DigestV1);

impl BindingAcceptanceRecordDigestV1 {
    pub fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Non-interchangeable content identity of the relationship itself.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequirementObligationBindingIdV1(Sha256DigestV1);

impl RequirementObligationBindingIdV1 {
    fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for RequirementObligationBindingIdV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why the requirement/obligation pair is claimed to correspond.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementObligationRelationV1 {
    /// The proof obligation is a direct byte-exact restatement of the accepted
    /// requirement proposition. The constructor mechanically enforces equality.
    ExactRestatement,

    /// The obligation is a derived safety/verification proposition.  The three
    /// identities commit to the derivation evidence, governing policy, and the
    /// acceptance of the relationship, but do not authenticate them.
    DerivedSafetyObligation {
        derivation_record_digest: DerivationRecordDigestV1,
        derivation_policy_revision_digest: DerivationPolicyRevisionDigestV1,
        binding_acceptance_record_digest: BindingAcceptanceRecordDigestV1,
    },
}

impl RequirementObligationRelationV1 {
    fn as_value(&self) -> Value {
        match self {
            Self::ExactRestatement => json!({
                "kind": "exact_restatement",
            }),
            Self::DerivedSafetyObligation {
                derivation_record_digest,
                derivation_policy_revision_digest,
                binding_acceptance_record_digest,
            } => json!({
                "binding_acceptance_record_digest": binding_acceptance_record_digest.as_str(),
                "derivation_policy_revision_digest": derivation_policy_revision_digest.as_str(),
                "derivation_record_digest": derivation_record_digest.as_str(),
                "kind": "derived_safety_obligation",
            }),
        }
    }
}

/// Content-addressed statement that one exact accepted requirement revision is
/// related to one exact proof-obligation revision under an explicit relation.
///
/// This is **not** evidence that the obligation is discharged, nor proof that a
/// derived relation is correct.  It only eliminates implicit claim-string,
/// vector-position, or “these happened to be passed together” inference.
#[must_use = "a requirement/obligation binding is a relationship record, not discharge authority"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementObligationBindingV1 {
    binding_id: RequirementObligationBindingIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    relation: RequirementObligationRelationV1,
}

impl RequirementObligationBindingV1 {
    /// Bind a direct restatement. This path is intentionally strict: any textual
    /// difference means the caller must use an explicit derived relation instead.
    pub fn exact_restatement(
        requirement: &AcceptedRequirementRevisionV1,
        obligation: &ProofObligation,
    ) -> Result<Self, RequirementBindingErrorV1> {
        ensure_evidence_kind_matches(requirement, obligation)?;
        if requirement.statement() != obligation.claim {
            return Err(RequirementBindingErrorV1::ExactRestatementClaimMismatch);
        }
        Self::construct(
            requirement,
            obligation,
            RequirementObligationRelationV1::ExactRestatement,
        )
    }

    /// Bind a genuinely derived safety obligation.
    ///
    /// No semantic-correctness claim is inferred from text similarity. The
    /// derivation, policy and relationship-acceptance artifacts are required as
    /// explicit premises and remain separately auditable/authenticatable.
    pub fn derived_safety_obligation(
        requirement: &AcceptedRequirementRevisionV1,
        obligation: &ProofObligation,
        derivation_record_digest: DerivationRecordDigestV1,
        derivation_policy_revision_digest: DerivationPolicyRevisionDigestV1,
        binding_acceptance_record_digest: BindingAcceptanceRecordDigestV1,
    ) -> Result<Self, RequirementBindingErrorV1> {
        ensure_evidence_kind_matches(requirement, obligation)?;
        Self::construct(
            requirement,
            obligation,
            RequirementObligationRelationV1::DerivedSafetyObligation {
                derivation_record_digest,
                derivation_policy_revision_digest,
                binding_acceptance_record_digest,
            },
        )
    }

    fn construct(
        requirement: &AcceptedRequirementRevisionV1,
        obligation: &ProofObligation,
        relation: RequirementObligationRelationV1,
    ) -> Result<Self, RequirementBindingErrorV1> {
        let obligation_revision_id = ObligationRevisionIdV1::for_obligation(obligation)
            .expect("proof-obligation snapshot IDs are generated as canonical SHA-256 identities");
        let obligation_id = obligation.id.to_string();
        let preimage = json!({
            "obligation_id": obligation_id,
            "obligation_revision_id": obligation_revision_id.as_str(),
            "relation": relation.as_value(),
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": "symthaea.etk-requirement-obligation-binding.v1",
        });
        let binding_id = RequirementObligationBindingIdV1::from_digest(domain_hash(
            BINDING_DOMAIN_V1,
            &preimage,
        ));
        Ok(Self {
            binding_id,
            requirement_revision_id: requirement.revision_id().clone(),
            obligation_id,
            obligation_revision_id,
            relation,
        })
    }

    pub fn binding_id(&self) -> &RequirementObligationBindingIdV1 {
        &self.binding_id
    }

    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }

    pub fn obligation_id(&self) -> &str {
        &self.obligation_id
    }

    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }

    pub fn relation(&self) -> &RequirementObligationRelationV1 {
        &self.relation
    }

    /// Data-only representation suitable for audit/storage. It cannot be
    /// deserialized into this capability-bearing type by this crate.
    pub fn audit_record_v1(&self) -> Value {
        json!({
            "authority": "relationship-binding-only",
            "binding_id": self.binding_id.as_str(),
            "obligation_id": self.obligation_id,
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "relation": self.relation.as_value(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
        })
    }
}

fn ensure_evidence_kind_matches(
    requirement: &AcceptedRequirementRevisionV1,
    obligation: &ProofObligation,
) -> Result<(), RequirementBindingErrorV1> {
    if requirement.expected_evidence_kind() != evidence_kind_name(&obligation.expected_evidence) {
        return Err(RequirementBindingErrorV1::EvidenceKindMismatch);
    }
    Ok(())
}

fn evidence_kind_name(kind: &EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::FormalProof => "FormalProof",
        EvidenceKind::Simulation => "Simulation",
        EvidenceKind::Test => "Test",
        EvidenceKind::Telemetry => "Telemetry",
        EvidenceKind::Standard => "Standard",
    }
}

fn domain_hash(domain: &[u8], value: &Value) -> Sha256DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(canonical_json(value).as_bytes());
    Sha256DigestV1::parse(format!("sha256:{}", hex::encode(hasher.finalize())))
        .expect("SHA-256 output is canonical lowercase hex")
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value)
            .expect("serializing an in-memory JSON string cannot fail"),
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let encoded_key = serde_json::to_string(key)
                        .expect("serializing an in-memory JSON key cannot fail");
                    format!("{encoded_key}:{}", canonical_json(&map[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_engineering_evidence_plan::RequirementCriticalityV1;
    use symthaea_sim_bridge::EngineeringDomain;

    fn digest(ch: char) -> Sha256DigestV1 {
        Sha256DigestV1::parse(format!("sha256:{}", ch.to_string().repeat(64))).unwrap()
    }

    fn requirement(statement: &str, evidence: EvidenceKind) -> AcceptedRequirementRevisionV1 {
        AcceptedRequirementRevisionV1::new(
            "REQ-42",
            EngineeringDomain::Civil,
            statement,
            RequirementCriticalityV1::Blocking,
            evidence,
            ["structural invariant"],
            digest('a'),
        )
        .unwrap()
    }

    #[test]
    fn exact_restatement_requires_exact_proposition() {
        let requirement = requirement("stress <= 250 MPa", EvidenceKind::Simulation);
        let obligation = ProofObligation::new("stress <= 250 MPa", EvidenceKind::Simulation);
        let binding =
            RequirementObligationBindingV1::exact_restatement(&requirement, &obligation).unwrap();
        assert_eq!(
            binding.requirement_revision_id(),
            requirement.revision_id()
        );
        assert_eq!(binding.obligation_id(), obligation.id.to_string());
        assert_eq!(binding.audit_record_v1()["authority"], "relationship-binding-only");
    }

    #[test]
    fn exact_restatement_rejects_similar_but_different_claim() {
        let requirement = requirement("stress <= 250 MPa", EvidenceKind::Simulation);
        let obligation = ProofObligation::new(
            "stress remains below allowable under service load",
            EvidenceKind::Simulation,
        );
        assert_eq!(
            RequirementObligationBindingV1::exact_restatement(&requirement, &obligation)
                .unwrap_err(),
            RequirementBindingErrorV1::ExactRestatementClaimMismatch
        );
    }

    #[test]
    fn evidence_kind_mismatch_is_denied_for_all_relation_paths() {
        let requirement = requirement("stress <= 250 MPa", EvidenceKind::Simulation);
        let obligation = ProofObligation::new("stress <= 250 MPa", EvidenceKind::FormalProof);
        assert_eq!(
            RequirementObligationBindingV1::exact_restatement(&requirement, &obligation)
                .unwrap_err(),
            RequirementBindingErrorV1::EvidenceKindMismatch
        );
        assert_eq!(
            RequirementObligationBindingV1::derived_safety_obligation(
                &requirement,
                &obligation,
                DerivationRecordDigestV1::from_digest(digest('b')),
                DerivationPolicyRevisionDigestV1::from_digest(digest('c')),
                BindingAcceptanceRecordDigestV1::from_digest(digest('d')),
            )
            .unwrap_err(),
            RequirementBindingErrorV1::EvidenceKindMismatch
        );
    }

    #[test]
    fn derived_obligation_requires_explicit_derivation_relationship() {
        let requirement = requirement("bridge shall be safe in service", EvidenceKind::Simulation);
        let obligation = ProofObligation::new(
            "maximum principal stress <= material allowable under LC9",
            EvidenceKind::Simulation,
        );
        let binding = RequirementObligationBindingV1::derived_safety_obligation(
            &requirement,
            &obligation,
            DerivationRecordDigestV1::from_digest(digest('b')),
            DerivationPolicyRevisionDigestV1::from_digest(digest('c')),
            BindingAcceptanceRecordDigestV1::from_digest(digest('d')),
        )
        .unwrap();
        assert!(matches!(
            binding.relation(),
            RequirementObligationRelationV1::DerivedSafetyObligation { .. }
        ));
    }

    #[test]
    fn derivation_artifact_changes_binding_identity() {
        let requirement = requirement("bridge shall be safe in service", EvidenceKind::Simulation);
        let obligation = ProofObligation::new(
            "maximum principal stress <= material allowable under LC9",
            EvidenceKind::Simulation,
        );
        let a = RequirementObligationBindingV1::derived_safety_obligation(
            &requirement,
            &obligation,
            DerivationRecordDigestV1::from_digest(digest('b')),
            DerivationPolicyRevisionDigestV1::from_digest(digest('c')),
            BindingAcceptanceRecordDigestV1::from_digest(digest('d')),
        )
        .unwrap();
        let b = RequirementObligationBindingV1::derived_safety_obligation(
            &requirement,
            &obligation,
            DerivationRecordDigestV1::from_digest(digest('e')),
            DerivationPolicyRevisionDigestV1::from_digest(digest('c')),
            BindingAcceptanceRecordDigestV1::from_digest(digest('d')),
        )
        .unwrap();
        assert_ne!(a.binding_id(), b.binding_id());
    }

    #[test]
    fn requirement_semantic_change_changes_binding_identity() {
        let obligation = ProofObligation::new(
            "maximum principal stress <= material allowable under LC9",
            EvidenceKind::Simulation,
        );
        let a = RequirementObligationBindingV1::derived_safety_obligation(
            &requirement("bridge shall be safe in service", EvidenceKind::Simulation),
            &obligation,
            DerivationRecordDigestV1::from_digest(digest('b')),
            DerivationPolicyRevisionDigestV1::from_digest(digest('c')),
            BindingAcceptanceRecordDigestV1::from_digest(digest('d')),
        )
        .unwrap();
        let b = RequirementObligationBindingV1::derived_safety_obligation(
            &requirement(
                "bridge shall be safe in service with fatigue margin",
                EvidenceKind::Simulation,
            ),
            &obligation,
            DerivationRecordDigestV1::from_digest(digest('b')),
            DerivationPolicyRevisionDigestV1::from_digest(digest('c')),
            BindingAcceptanceRecordDigestV1::from_digest(digest('d')),
        )
        .unwrap();
        assert_ne!(a.binding_id(), b.binding_id());
    }

    #[test]
    fn relationship_binding_does_not_discharge_obligation() {
        let requirement = requirement("stress <= 250 MPa", EvidenceKind::Simulation);
        let obligation = ProofObligation::new("stress <= 250 MPa", EvidenceKind::Simulation);
        let before = obligation.status;
        let _binding =
            RequirementObligationBindingV1::exact_restatement(&requirement, &obligation).unwrap();
        assert_eq!(obligation.status, before);
    }
}
