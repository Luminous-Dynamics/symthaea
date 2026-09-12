// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed Engineering Trust Kernel composition above evidence admission.
//!
//! ```text
//! historical discharge receipt
//! != current obligation-discharge fact
//! != complete requirement-verification contract
//! != current requirement satisfaction
//! != qualified design / certification / manufacturing / actuation authority
//! ```
//!
//! This crate deliberately owns no solver-admission theorem. It delegates
//! receipt applicability to `symthaea-engineering-trust`, consumes explicit
//! requirement/obligation relationships, and implements only conservative
//! requirement composition. V1 supports `AllOf` only.

#![deny(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_engineering_evidence_plan::{
    AcceptedRequirementRevisionIdV1, AcceptedRequirementRevisionV1, ObligationRevisionIdV1,
    Sha256DigestV1,
};
use symthaea_engineering_requirement_binding::{
    RequirementObligationBindingIdV1, RequirementObligationBindingV1,
};
use symthaea_engineering_trust::{
    DischargeContextV1, ObligationDischargeReceiptV1, is_obligation_discharged_v1,
};
use symthaea_formal_safety::ProofObligation;
use thiserror::Error;

const CURRENT_DISCHARGE_FACT_SCHEMA_V1: &str =
    "symthaea.etk-current-obligation-discharge-fact.v1";
const CURRENT_DISCHARGE_FACT_DOMAIN_V1: &[u8] =
    b"symthaea.etk-current-obligation-discharge-fact.v1\0";
const VERIFICATION_CONTRACT_SCHEMA_V1: &str =
    "symthaea.etk-requirement-verification-contract.v1";
const VERIFICATION_CONTRACT_DOMAIN_V1: &[u8] =
    b"symthaea.etk-requirement-verification-contract.v1\0";
const SATISFACTION_RECEIPT_SCHEMA_V1: &str =
    "symthaea.etk-requirement-satisfaction-receipt.v1";
const SATISFACTION_RECEIPT_DOMAIN_V1: &[u8] =
    b"symthaea.etk-requirement-satisfaction-receipt.v1\0";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssuranceErrorV1 {
    #[error("{0} cannot be empty or have leading/trailing whitespace")]
    InvalidText(&'static str),
    #[error("{0} is not a canonical SHA-256 identity")]
    InvalidDigest(&'static str),
    #[error("no historical discharge receipt applies to the exact current obligation/context")]
    NoApplicableReceipt,
    #[error("an AllOf requirement-verification contract requires at least one relationship")]
    EmptyAllOf,
    #[error("a requirement/obligation relationship targets a different requirement revision")]
    RelationshipRequirementMismatch,
    #[error("duplicate requirement/obligation relationship identity")]
    DuplicateRelationship,
    #[error("duplicate proof-obligation revision in one AllOf contract")]
    DuplicateObligation,
}

macro_rules! digest_role {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            pub fn from_digest(digest: Sha256DigestV1) -> Self {
                Self(digest)
            }

            pub fn parse(value: impl Into<String>) -> Result<Self, AssuranceErrorV1> {
                Sha256DigestV1::parse(value.into())
                    .map(Self)
                    .map_err(|_| AssuranceErrorV1::InvalidDigest(stringify!($name)))
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

digest_role!(RequirementDecompositionPolicyRevisionDigestV1);
digest_role!(RequirementDecompositionAcceptanceRecordDigestV1);
digest_role!(RequirementCurrentnessAssertionIdV1);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CurrentObligationDischargeFactIdV1(Sha256DigestV1);

impl CurrentObligationDischargeFactIdV1 {
    fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for CurrentObligationDischargeFactIdV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Present-tense, context-bound proof that one exact obligation is currently
/// discharged according to the lower ETK receipt-applicability theorem.
///
/// The fields are private and this type intentionally has no deserializer.
#[must_use = "a current discharge fact is still not requirement satisfaction"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentObligationDischargeFactV1 {
    fact_id: CurrentObligationDischargeFactIdV1,
    obligation_id: String,
    obligation_revision_id: ObligationRevisionIdV1,
    subject_id: String,
    twin_revision: String,
    requirement_revision_id: Sha256DigestV1,
    validity_domain_id: String,
    currentness_proof_id: String,
    witness_receipt_id: String,
}

impl CurrentObligationDischargeFactV1 {
    pub fn fact_id(&self) -> &CurrentObligationDischargeFactIdV1 {
        &self.fact_id
    }

    pub fn obligation_id(&self) -> &str {
        &self.obligation_id
    }

    pub fn obligation_revision_id(&self) -> &ObligationRevisionIdV1 {
        &self.obligation_revision_id
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn twin_revision(&self) -> &str {
        &self.twin_revision
    }

    pub fn requirement_revision_id(&self) -> &str {
        self.requirement_revision_id.as_str()
    }

    pub fn validity_domain_id(&self) -> &str {
        &self.validity_domain_id
    }

    pub fn currentness_proof_id(&self) -> &str {
        &self.currentness_proof_id
    }

    pub fn witness_receipt_id(&self) -> &str {
        &self.witness_receipt_id
    }

    pub fn audit_record_v1(&self) -> Value {
        json!({
            "authority": "current-obligation-discharge-only",
            "current_discharge_fact_id": self.fact_id.as_str(),
            "currentness_proof_id": self.currentness_proof_id,
            "obligation_id": self.obligation_id,
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "subject_id": self.subject_id,
            "twin_revision": self.twin_revision,
            "validity_domain_id": self.validity_domain_id,
            "witness_receipt_id": self.witness_receipt_id,
        })
    }
}

/// Promote the lower ETK boolean applicability theorem into a typed,
/// content-addressed present-tense fact.
///
/// Receipt currentness is *not* reimplemented here. Each candidate receipt is
/// tested by `is_obligation_discharged_v1` in isolation, and the lexical-minimum
/// applicable receipt identity becomes the deterministic witness. Historical or
/// stale receipts therefore cannot be minted into present-discharge authority.
pub fn derive_current_obligation_discharge_fact_v1(
    obligation: &ProofObligation,
    context: &DischargeContextV1,
    receipts: &[ObligationDischargeReceiptV1],
) -> Result<CurrentObligationDischargeFactV1, AssuranceErrorV1> {
    let subject_id = canonical_text(context.subject_id(), "subject id")?;
    let twin_revision = canonical_text(context.twin_revision(), "twin revision")?;
    let validity_domain_id = canonical_text(context.validity_domain_id(), "validity domain id")?;
    let currentness_proof_id = canonical_text(context.currentness_proof_id(), "currentness proof id")?;
    let requirement_revision_id = Sha256DigestV1::parse(context.requirement_revision().to_string())
        .map_err(|_| AssuranceErrorV1::InvalidDigest("requirement revision"))?;
    let obligation_revision_id = ObligationRevisionIdV1::for_obligation(obligation)
        .map_err(|_| AssuranceErrorV1::InvalidDigest("obligation revision"))?;

    let witness_receipt_id = receipts
        .iter()
        .filter(|receipt| {
            is_obligation_discharged_v1(obligation, context, std::slice::from_ref(*receipt))
        })
        .map(ObligationDischargeReceiptV1::receipt_id)
        .min()
        .ok_or(AssuranceErrorV1::NoApplicableReceipt)?
        .to_string();

    let obligation_id = obligation.id.to_string();
    let preimage = json!({
        "currentness_proof_id": currentness_proof_id,
        "obligation_id": obligation_id,
        "obligation_revision": obligation_revision_id.as_str(),
        "requirement_revision": requirement_revision_id.as_str(),
        "schema": CURRENT_DISCHARGE_FACT_SCHEMA_V1,
        "subject_id": subject_id,
        "twin_revision": twin_revision,
        "validity_domain_id": validity_domain_id,
        "witness_receipt_id": witness_receipt_id,
    });
    let fact_id = CurrentObligationDischargeFactIdV1::from_digest(domain_hash(
        CURRENT_DISCHARGE_FACT_DOMAIN_V1,
        &preimage,
    ));

    Ok(CurrentObligationDischargeFactV1 {
        fact_id,
        obligation_id,
        obligation_revision_id,
        subject_id,
        twin_revision,
        requirement_revision_id,
        validity_domain_id,
        currentness_proof_id,
        witness_receipt_id,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequirementVerificationContractIdV1(Sha256DigestV1);

impl RequirementVerificationContractIdV1 {
    fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for RequirementVerificationContractIdV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerificationMemberV1 {
    relationship_id: RequirementObligationBindingIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    obligation_revision_id: ObligationRevisionIdV1,
}

impl VerificationMemberV1 {
    fn as_value(&self) -> Value {
        json!({
            "obligation_revision_id": self.obligation_revision_id.as_str(),
            "relationship_id": self.relationship_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
        })
    }
}

/// Explicit, content-addressed statement that every enumerated relationship is
/// required for one accepted requirement revision.
///
/// V1 is intentionally conservative: `AllOf` is the only composition form.
#[must_use = "a verification contract is a decomposition contract, not satisfaction"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementVerificationContractV1 {
    contract_id: RequirementVerificationContractIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    members: Vec<VerificationMemberV1>,
    decomposition_policy_revision_id: RequirementDecompositionPolicyRevisionDigestV1,
    decomposition_acceptance_record_digest: RequirementDecompositionAcceptanceRecordDigestV1,
}

impl RequirementVerificationContractV1 {
    pub fn all_of(
        requirement: &AcceptedRequirementRevisionV1,
        bindings: &[RequirementObligationBindingV1],
        decomposition_policy_revision_id: RequirementDecompositionPolicyRevisionDigestV1,
        decomposition_acceptance_record_digest: RequirementDecompositionAcceptanceRecordDigestV1,
    ) -> Result<Self, AssuranceErrorV1> {
        if bindings.is_empty() {
            return Err(AssuranceErrorV1::EmptyAllOf);
        }

        let mut relationship_ids = BTreeSet::new();
        let mut obligation_ids = BTreeSet::new();
        let mut members = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if binding.requirement_revision_id() != requirement.revision_id() {
                return Err(AssuranceErrorV1::RelationshipRequirementMismatch);
            }
            if !relationship_ids.insert(binding.binding_id().as_str().to_string()) {
                return Err(AssuranceErrorV1::DuplicateRelationship);
            }
            if !obligation_ids.insert(binding.obligation_revision_id().as_str().to_string()) {
                return Err(AssuranceErrorV1::DuplicateObligation);
            }
            members.push(VerificationMemberV1 {
                relationship_id: binding.binding_id().clone(),
                requirement_revision_id: binding.requirement_revision_id().clone(),
                obligation_revision_id: binding.obligation_revision_id().clone(),
            });
        }
        members.sort_by(|a, b| {
            a.obligation_revision_id
                .as_str()
                .cmp(b.obligation_revision_id.as_str())
                .then_with(|| a.relationship_id.as_str().cmp(b.relationship_id.as_str()))
        });

        let member_values = members
            .iter()
            .map(VerificationMemberV1::as_value)
            .collect::<Vec<_>>();
        let preimage = json!({
            "composition": "AllOf",
            "decomposition_acceptance_record_digest": decomposition_acceptance_record_digest.as_str(),
            "decomposition_policy_revision_id": decomposition_policy_revision_id.as_str(),
            "relationships": member_values,
            "requirement_revision_id": requirement.revision_id().as_str(),
            "schema": VERIFICATION_CONTRACT_SCHEMA_V1,
        });
        let contract_id = RequirementVerificationContractIdV1::from_digest(domain_hash(
            VERIFICATION_CONTRACT_DOMAIN_V1,
            &preimage,
        ));

        Ok(Self {
            contract_id,
            requirement_revision_id: requirement.revision_id().clone(),
            members,
            decomposition_policy_revision_id,
            decomposition_acceptance_record_digest,
        })
    }

    pub fn contract_id(&self) -> &RequirementVerificationContractIdV1 {
        &self.contract_id
    }

    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }

    pub fn required_obligation_revision_ids(&self) -> impl Iterator<Item = &ObligationRevisionIdV1> {
        self.members.iter().map(|member| &member.obligation_revision_id)
    }

    pub fn audit_record_v1(&self) -> Value {
        let relationships = self
            .members
            .iter()
            .map(VerificationMemberV1::as_value)
            .collect::<Vec<_>>();
        json!({
            "authority": "requirement-decomposition-only",
            "composition": "AllOf",
            "decomposition_acceptance_record_digest": self.decomposition_acceptance_record_digest.as_str(),
            "decomposition_policy_revision_id": self.decomposition_policy_revision_id.as_str(),
            "relationships": relationships,
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "verification_contract_id": self.contract_id.as_str(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequirementSatisfactionReceiptIdV1(Sha256DigestV1);

impl RequirementSatisfactionReceiptIdV1 {
    fn from_digest(digest: Sha256DigestV1) -> Self {
        Self(digest)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for RequirementSatisfactionReceiptIdV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UsedDischargeFactV1 {
    obligation_revision_id: ObligationRevisionIdV1,
    current_discharge_fact_id: CurrentObligationDischargeFactIdV1,
}

impl UsedDischargeFactV1 {
    fn as_value(&self) -> Value {
        json!({
            "current_discharge_fact_id": self.current_discharge_fact_id.as_str(),
            "obligation_revision_id": self.obligation_revision_id.as_str(),
        })
    }
}

/// Present-tense satisfaction receipt for one exact requirement-verification
/// contract and one exact subject/twin context.
///
/// This is explicitly *not* design qualification or downstream release authority.
#[must_use = "requirement satisfaction is not design qualification or release authority"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementSatisfactionReceiptV1 {
    receipt_id: RequirementSatisfactionReceiptIdV1,
    verification_contract_id: RequirementVerificationContractIdV1,
    requirement_revision_id: AcceptedRequirementRevisionIdV1,
    current_subject_id: String,
    current_twin_revision: String,
    currentness_assertion_id: RequirementCurrentnessAssertionIdV1,
    used_discharge_facts: Vec<UsedDischargeFactV1>,
}

impl RequirementSatisfactionReceiptV1 {
    pub fn receipt_id(&self) -> &RequirementSatisfactionReceiptIdV1 {
        &self.receipt_id
    }

    pub fn verification_contract_id(&self) -> &RequirementVerificationContractIdV1 {
        &self.verification_contract_id
    }

    pub fn requirement_revision_id(&self) -> &AcceptedRequirementRevisionIdV1 {
        &self.requirement_revision_id
    }

    pub fn current_subject_id(&self) -> &str {
        &self.current_subject_id
    }

    pub fn current_twin_revision(&self) -> &str {
        &self.current_twin_revision
    }

    pub fn currentness_assertion_id(&self) -> &RequirementCurrentnessAssertionIdV1 {
        &self.currentness_assertion_id
    }

    pub fn audit_record_v1(&self) -> Value {
        let facts = self
            .used_discharge_facts
            .iter()
            .map(UsedDischargeFactV1::as_value)
            .collect::<Vec<_>>();
        json!({
            "authority": "current-requirement-satisfaction-only",
            "current_discharge_facts": facts,
            "current_subject_id": self.current_subject_id,
            "current_twin_revision": self.current_twin_revision,
            "currentness_assertion_id": self.currentness_assertion_id.as_str(),
            "requirement_revision_id": self.requirement_revision_id.as_str(),
            "requirement_satisfaction_receipt_id": self.receipt_id.as_str(),
            "verification_contract_id": self.verification_contract_id.as_str(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementUnsatisfiedV1 {
    verification_contract_id: RequirementVerificationContractIdV1,
    missing_obligation_revision_ids: Vec<ObligationRevisionIdV1>,
}

impl RequirementUnsatisfiedV1 {
    pub fn verification_contract_id(&self) -> &RequirementVerificationContractIdV1 {
        &self.verification_contract_id
    }

    pub fn missing_obligation_revision_ids(&self) -> &[ObligationRevisionIdV1] {
        &self.missing_obligation_revision_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequirementSatisfactionDecisionV1 {
    HistoricalVerificationContract {
        verification_contract_id: RequirementVerificationContractIdV1,
        contract_requirement_revision_id: AcceptedRequirementRevisionIdV1,
        current_requirement_revision_id: AcceptedRequirementRevisionIdV1,
    },
    RequirementUnsatisfied(RequirementUnsatisfiedV1),
    CurrentRequirementSatisfied(RequirementSatisfactionReceiptV1),
}

/// Evaluate conservative `AllOf` satisfaction for one exact current design
/// context. Facts from another subject, twin, or requirement revision are
/// ignored and cannot compensate for a missing required obligation.
pub fn evaluate_requirement_satisfaction_v1(
    contract: &RequirementVerificationContractV1,
    current_requirement: &AcceptedRequirementRevisionV1,
    current_subject_id: impl Into<String>,
    current_twin_revision: impl Into<String>,
    facts: &[CurrentObligationDischargeFactV1],
    currentness_assertion_id: RequirementCurrentnessAssertionIdV1,
) -> Result<RequirementSatisfactionDecisionV1, AssuranceErrorV1> {
    let current_subject_id = canonical_text(current_subject_id.into(), "current subject id")?;
    let current_twin_revision =
        canonical_text(current_twin_revision.into(), "current twin revision")?;

    if current_requirement.revision_id() != contract.requirement_revision_id() {
        return Ok(RequirementSatisfactionDecisionV1::HistoricalVerificationContract {
            verification_contract_id: contract.contract_id.clone(),
            contract_requirement_revision_id: contract.requirement_revision_id.clone(),
            current_requirement_revision_id: current_requirement.revision_id().clone(),
        });
    }

    let mut applicable: BTreeMap<String, &CurrentObligationDischargeFactV1> = BTreeMap::new();
    for fact in facts {
        if fact.subject_id != current_subject_id
            || fact.twin_revision != current_twin_revision
            || fact.requirement_revision_id.as_str() != current_requirement.revision_id().as_str()
        {
            continue;
        }
        let key = fact.obligation_revision_id.as_str().to_string();
        match applicable.get(&key) {
            Some(previous) if previous.fact_id.as_str() <= fact.fact_id.as_str() => {}
            _ => {
                applicable.insert(key, fact);
            }
        }
    }

    let mut missing = Vec::new();
    let mut used = Vec::with_capacity(contract.members.len());
    for member in &contract.members {
        let key = member.obligation_revision_id.as_str();
        if let Some(fact) = applicable.get(key) {
            used.push(UsedDischargeFactV1 {
                obligation_revision_id: member.obligation_revision_id.clone(),
                current_discharge_fact_id: fact.fact_id.clone(),
            });
        } else {
            missing.push(member.obligation_revision_id.clone());
        }
    }

    if !missing.is_empty() {
        missing.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        return Ok(RequirementSatisfactionDecisionV1::RequirementUnsatisfied(
            RequirementUnsatisfiedV1 {
                verification_contract_id: contract.contract_id.clone(),
                missing_obligation_revision_ids: missing,
            },
        ));
    }

    used.sort_by(|a, b| a.obligation_revision_id.as_str().cmp(b.obligation_revision_id.as_str()));
    let used_values = used
        .iter()
        .map(UsedDischargeFactV1::as_value)
        .collect::<Vec<_>>();
    let preimage = json!({
        "current_discharge_facts": used_values,
        "current_subject_id": current_subject_id,
        "current_twin_revision": current_twin_revision,
        "currentness_assertion_id": currentness_assertion_id.as_str(),
        "requirement_revision_id": current_requirement.revision_id().as_str(),
        "schema": SATISFACTION_RECEIPT_SCHEMA_V1,
        "verification_contract_id": contract.contract_id.as_str(),
    });
    let receipt_id = RequirementSatisfactionReceiptIdV1::from_digest(domain_hash(
        SATISFACTION_RECEIPT_DOMAIN_V1,
        &preimage,
    ));

    Ok(RequirementSatisfactionDecisionV1::CurrentRequirementSatisfied(
        RequirementSatisfactionReceiptV1 {
            receipt_id,
            verification_contract_id: contract.contract_id.clone(),
            requirement_revision_id: current_requirement.revision_id().clone(),
            current_subject_id,
            current_twin_revision,
            currentness_assertion_id,
            used_discharge_facts: used,
        },
    ))
}

fn canonical_text(value: impl Into<String>, field: &'static str) -> Result<String, AssuranceErrorV1> {
    let value = value.into();
    if value.is_empty() || value.trim() != value {
        Err(AssuranceErrorV1::InvalidText(field))
    } else {
        Ok(value)
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
