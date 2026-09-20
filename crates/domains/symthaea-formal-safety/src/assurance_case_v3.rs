// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict assurance evaluation over admitted verification evidence only.
//!
//! `AssuranceCaseV2` remains available for compatibility, but its raw
//! `EvidenceRecord` inputs are not an admission boundary. V3 deliberately cannot
//! accept raw evidence records or legacy claims: qualification is derived only
//! from admitted evidence and V3 claims that bind each required property to the
//! evidence methods permitted to establish it.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::assurance_case::{
    AssuranceProfile, Assumption, AssumptionStatus, ClaimState, EvidenceMethod, SubjectRef,
};
use crate::assurance_state::PropertyState;
use crate::receipt_admission::{
    AdmittedVerificationEvidence, MAX_ADMISSION_GRANT_PROPERTIES,
    MAX_ADMISSION_GRANT_STRING_BYTES,
};

pub const ASSURANCE_CASE_V3_SCHEMA_VERSION: u16 = 3;

/// One property requirement and the evidence methods allowed to establish it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRequirementV3 {
    pub property: String,
    pub allowed_methods: Vec<EvidenceMethod>,
}

impl EvidenceRequirementV3 {
    pub fn validate(&self) -> bool {
        bounded_nonempty(&self.property)
            && !self.allowed_methods.is_empty()
            && unique(&self.allowed_methods)
    }

    pub fn accepts(&self, method: EvidenceMethod) -> bool {
        self.allowed_methods.contains(&method)
    }
}

/// Strict assurance claim.
///
/// Receipt references are named explicitly so they cannot be confused with the
/// raw V2 `EvidenceRecord.id` namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceClaimV3 {
    pub id: Uuid,
    pub subject: SubjectRef,
    pub proposition: String,
    pub evidence_requirements: Vec<EvidenceRequirementV3>,
    pub assumption_ids: Vec<Uuid>,
    pub dependency_claim_ids: Vec<Uuid>,
    pub admitted_receipt_ids: Vec<Uuid>,
}

impl AssuranceClaimV3 {
    pub fn validate(&self) -> bool {
        self.id != Uuid::nil()
            && self.subject.validate()
            && bounded_nonempty(&self.proposition)
            && !self.evidence_requirements.is_empty()
            && self.evidence_requirements.len() <= MAX_ADMISSION_GRANT_PROPERTIES
            && self
                .evidence_requirements
                .iter()
                .all(EvidenceRequirementV3::validate)
            && unique_by(
                &self.evidence_requirements,
                |requirement| requirement.property.as_str(),
            )
            && unique(&self.assumption_ids)
            && unique(&self.dependency_claim_ids)
            && unique(&self.admitted_receipt_ids)
            && !self.dependency_claim_ids.contains(&self.id)
    }

    fn requires_property(&self, property: &str) -> bool {
        self.evidence_requirements
            .iter()
            .any(|requirement| requirement.property == property)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceIssueV3 {
    InvalidCase,
    InvalidProfile,
    ClaimSubjectMismatch {
        claim_id: Uuid,
    },
    UnknownAssumption {
        claim_id: Uuid,
        assumption_id: Uuid,
    },
    AssumptionProfileMismatch {
        claim_id: Uuid,
        assumption_id: Uuid,
        accepted_profile_id: String,
    },
    OpenAssumption {
        claim_id: Uuid,
        assumption_id: Uuid,
    },
    RefutedAssumption {
        claim_id: Uuid,
        assumption_id: Uuid,
    },
    StaleAssumption {
        claim_id: Uuid,
        assumption_id: Uuid,
    },
    UnknownDependency {
        claim_id: Uuid,
        dependency_claim_id: Uuid,
    },
    DependencyNotQualified {
        claim_id: Uuid,
        dependency_claim_id: Uuid,
        dependency_state: ClaimState,
    },
    DependencyCycle(Uuid),
    UnknownAdmittedReceipt {
        claim_id: Uuid,
        receipt_id: Uuid,
    },
    EvidenceSubjectMismatch {
        claim_id: Uuid,
        receipt_id: Uuid,
    },
    EvidenceProfileMismatch {
        claim_id: Uuid,
        receipt_id: Uuid,
        evidence_profile_id: String,
    },
    EvidenceEnvironmentMismatch {
        claim_id: Uuid,
        receipt_id: Uuid,
    },
    EvidenceExpired {
        claim_id: Uuid,
        receipt_id: Uuid,
    },
    EvidenceNotYetValid {
        claim_id: Uuid,
        receipt_id: Uuid,
    },
    EvidenceMethodNotAllowed {
        claim_id: Uuid,
        receipt_id: Uuid,
        property: String,
        method: EvidenceMethod,
    },
    IndeterminateEvidence {
        claim_id: Uuid,
        receipt_id: Uuid,
        property: String,
    },
    MissingPropertyEvidence {
        claim_id: Uuid,
        property: String,
    },
    CounterexampleEvidence {
        claim_id: Uuid,
        receipt_id: Uuid,
        property: String,
        referenced_by_claim: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceEvaluationV3 {
    pub schema_version: u16,
    pub case_id: Uuid,
    pub profile_id: String,
    pub claim_states: BTreeMap<Uuid, ClaimState>,
    pub issues: Vec<AssuranceIssueV3>,
}

impl AssuranceEvaluationV3 {
    pub fn is_qualified(&self, claim_id: Uuid) -> bool {
        self.claim_states.get(&claim_id) == Some(&ClaimState::QualifiedUnderProfile)
    }
}

/// Strict assurance case: evidence is admitted evidence, never a raw evidence
/// reference supplied directly by a caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceCaseV3 {
    pub schema_version: u16,
    pub case_id: Uuid,
    pub subject: SubjectRef,
    pub claims: Vec<AssuranceClaimV3>,
    pub assumptions: Vec<Assumption>,
    pub admitted_evidence: Vec<AdmittedVerificationEvidence>,
}

impl AssuranceCaseV3 {
    pub fn new(subject: SubjectRef) -> Self {
        Self {
            schema_version: ASSURANCE_CASE_V3_SCHEMA_VERSION,
            case_id: Uuid::new_v4(),
            subject,
            claims: Vec::new(),
            assumptions: Vec::new(),
            admitted_evidence: Vec::new(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == ASSURANCE_CASE_V3_SCHEMA_VERSION
            && self.case_id != Uuid::nil()
            && self.subject.validate()
            && unique_ids(self.claims.iter().map(|value| value.id))
            && unique_ids(self.assumptions.iter().map(|value| value.id))
            && unique_ids(self.admitted_evidence.iter().map(|value| value.receipt_id))
            && self.claims.iter().all(AssuranceClaimV3::validate)
            && self.assumptions.iter().all(Assumption::validate)
            && self
                .admitted_evidence
                .iter()
                .all(|evidence| evidence.subject == self.subject && validate_admitted_evidence(evidence))
    }

    pub fn evaluate(&self, profile: &AssuranceProfile) -> AssuranceEvaluationV3 {
        let mut evaluation = AssuranceEvaluationV3 {
            schema_version: ASSURANCE_CASE_V3_SCHEMA_VERSION,
            case_id: self.case_id,
            profile_id: profile.profile_id.clone(),
            claim_states: BTreeMap::new(),
            issues: Vec::new(),
        };

        if !self.validate() {
            evaluation.issues.push(AssuranceIssueV3::InvalidCase);
            for claim in &self.claims {
                evaluation
                    .claim_states
                    .insert(claim.id, ClaimState::Indeterminate);
            }
            return evaluation;
        }
        if !profile.validate() || profile.subject != self.subject {
            evaluation.issues.push(AssuranceIssueV3::InvalidProfile);
            for claim in &self.claims {
                evaluation
                    .claim_states
                    .insert(claim.id, ClaimState::Indeterminate);
            }
            return evaluation;
        }

        let claims: BTreeMap<_, _> = self.claims.iter().map(|value| (value.id, value)).collect();
        let assumptions: BTreeMap<_, _> = self
            .assumptions
            .iter()
            .map(|value| (value.id, value))
            .collect();
        let evidence: BTreeMap<_, _> = self
            .admitted_evidence
            .iter()
            .map(|value| (value.receipt_id, value))
            .collect();
        let mut visiting = BTreeSet::new();

        for claim in &self.claims {
            evaluate_claim_v3(
                claim.id,
                profile,
                &claims,
                &assumptions,
                &evidence,
                &mut visiting,
                &mut evaluation,
            );
        }

        evaluation
    }

    /// Conservative requalification impact set for one receipt. Because V3 lets
    /// an active refutation dominate even when a claim omitted the refuting
    /// receipt from `admitted_receipt_ids`, every claim sharing a property with
    /// the receipt is included in addition to explicit references.
    pub fn claims_affected_by_receipt(&self, receipt_id: Uuid) -> BTreeSet<Uuid> {
        let Some(evidence) = self
            .admitted_evidence
            .iter()
            .find(|value| value.receipt_id == receipt_id)
        else {
            return BTreeSet::new();
        };
        let evidence_properties: BTreeSet<_> = evidence
            .properties
            .iter()
            .map(|value| value.property.as_str())
            .collect();
        let mut affected: BTreeSet<_> = self
            .claims
            .iter()
            .filter(|claim| {
                claim.admitted_receipt_ids.contains(&receipt_id)
                    || claim
                        .evidence_requirements
                        .iter()
                        .any(|requirement| evidence_properties.contains(requirement.property.as_str()))
            })
            .map(|claim| claim.id)
            .collect();
        close_over_dependents(&self.claims, &mut affected);
        affected
    }

    pub fn claims_affected_by_assumption(&self, assumption_id: Uuid) -> BTreeSet<Uuid> {
        let mut affected: BTreeSet<_> = self
            .claims
            .iter()
            .filter(|claim| claim.assumption_ids.contains(&assumption_id))
            .map(|claim| claim.id)
            .collect();
        close_over_dependents(&self.claims, &mut affected);
        affected
    }

    pub fn claims_affected_by_signer_authorization_grant(
        &self,
        grant_id: Uuid,
    ) -> BTreeSet<Uuid> {
        self.admitted_evidence
            .iter()
            .filter(|value| value.signer_authorization_grant_id == grant_id)
            .flat_map(|value| self.claims_affected_by_receipt(value.receipt_id))
            .collect()
    }

    pub fn claims_affected_by_evidence_admission_grant(
        &self,
        grant_id: Uuid,
    ) -> BTreeSet<Uuid> {
        self.admitted_evidence
            .iter()
            .filter(|value| value.evidence_admission_grant_id == grant_id)
            .flat_map(|value| self.claims_affected_by_receipt(value.receipt_id))
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_claim_v3(
    claim_id: Uuid,
    profile: &AssuranceProfile,
    claims: &BTreeMap<Uuid, &AssuranceClaimV3>,
    assumptions: &BTreeMap<Uuid, &Assumption>,
    evidence: &BTreeMap<Uuid, &AdmittedVerificationEvidence>,
    visiting: &mut BTreeSet<Uuid>,
    evaluation: &mut AssuranceEvaluationV3,
) -> ClaimState {
    if let Some(state) = evaluation.claim_states.get(&claim_id).copied() {
        return state;
    }
    let Some(claim) = claims.get(&claim_id).copied() else {
        return ClaimState::Indeterminate;
    };
    if !visiting.insert(claim_id) {
        evaluation
            .issues
            .push(AssuranceIssueV3::DependencyCycle(claim_id));
        return ClaimState::Indeterminate;
    }

    if claim.subject != profile.subject {
        evaluation
            .issues
            .push(AssuranceIssueV3::ClaimSubjectMismatch { claim_id });
        visiting.remove(&claim_id);
        evaluation
            .claim_states
            .insert(claim_id, ClaimState::Invalidated);
        return ClaimState::Invalidated;
    }

    let mut saw_stale = false;
    let mut saw_evidence = false;
    let mut invalidated = false;
    let mut indeterminate = false;

    for assumption_id in &claim.assumption_ids {
        let Some(assumption) = assumptions.get(assumption_id).copied() else {
            evaluation.issues.push(AssuranceIssueV3::UnknownAssumption {
                claim_id,
                assumption_id: *assumption_id,
            });
            indeterminate = true;
            continue;
        };
        if assumption.subject != profile.subject {
            invalidated = true;
            continue;
        }
        match &assumption.status {
            AssumptionStatus::AcceptedUnderProfile(accepted)
                if accepted == &profile.profile_id => {}
            AssumptionStatus::AcceptedUnderProfile(accepted) => {
                evaluation
                    .issues
                    .push(AssuranceIssueV3::AssumptionProfileMismatch {
                        claim_id,
                        assumption_id: *assumption_id,
                        accepted_profile_id: accepted.clone(),
                    });
                invalidated = true;
            }
            AssumptionStatus::Open => {
                evaluation.issues.push(AssuranceIssueV3::OpenAssumption {
                    claim_id,
                    assumption_id: *assumption_id,
                });
                invalidated = true;
            }
            AssumptionStatus::Refuted => {
                evaluation.issues.push(AssuranceIssueV3::RefutedAssumption {
                    claim_id,
                    assumption_id: *assumption_id,
                });
                invalidated = true;
            }
            AssumptionStatus::Stale => {
                evaluation.issues.push(AssuranceIssueV3::StaleAssumption {
                    claim_id,
                    assumption_id: *assumption_id,
                });
                saw_stale = true;
            }
        }
    }

    for dependency_id in &claim.dependency_claim_ids {
        if !claims.contains_key(dependency_id) {
            evaluation.issues.push(AssuranceIssueV3::UnknownDependency {
                claim_id,
                dependency_claim_id: *dependency_id,
            });
            indeterminate = true;
            continue;
        }
        let state = evaluate_claim_v3(
            *dependency_id,
            profile,
            claims,
            assumptions,
            evidence,
            visiting,
            evaluation,
        );
        if state != ClaimState::QualifiedUnderProfile {
            evaluation
                .issues
                .push(AssuranceIssueV3::DependencyNotQualified {
                    claim_id,
                    dependency_claim_id: *dependency_id,
                    dependency_state: state,
                });
            match state {
                ClaimState::Stale => saw_stale = true,
                ClaimState::CounterexampleFound | ClaimState::Invalidated => invalidated = true,
                ClaimState::Indeterminate => indeterminate = true,
                ClaimState::Unexamined | ClaimState::EvidenceCollected => {}
                ClaimState::QualifiedUnderProfile => unreachable!(),
            }
        }
    }

    let mut supported_properties = BTreeSet::new();
    let mut counterexample = false;

    for receipt_id in &claim.admitted_receipt_ids {
        let Some(record) = evidence.get(receipt_id).copied() else {
            evaluation
                .issues
                .push(AssuranceIssueV3::UnknownAdmittedReceipt {
                    claim_id,
                    receipt_id: *receipt_id,
                });
            indeterminate = true;
            continue;
        };
        saw_evidence = true;
        match evidence_usability(record, profile) {
            EvidenceUsability::Active => {}
            EvidenceUsability::SubjectMismatch => {
                evaluation
                    .issues
                    .push(AssuranceIssueV3::EvidenceSubjectMismatch {
                        claim_id,
                        receipt_id: *receipt_id,
                    });
                invalidated = true;
                continue;
            }
            EvidenceUsability::ProfileMismatch => {
                evaluation
                    .issues
                    .push(AssuranceIssueV3::EvidenceProfileMismatch {
                        claim_id,
                        receipt_id: *receipt_id,
                        evidence_profile_id: record.profile_id.clone(),
                    });
                invalidated = true;
                continue;
            }
            EvidenceUsability::EnvironmentMismatch => {
                evaluation
                    .issues
                    .push(AssuranceIssueV3::EvidenceEnvironmentMismatch {
                        claim_id,
                        receipt_id: *receipt_id,
                    });
                saw_stale = true;
                continue;
            }
            EvidenceUsability::Expired => {
                evaluation.issues.push(AssuranceIssueV3::EvidenceExpired {
                    claim_id,
                    receipt_id: *receipt_id,
                });
                saw_stale = true;
                continue;
            }
            EvidenceUsability::NotYetValid => {
                evaluation
                    .issues
                    .push(AssuranceIssueV3::EvidenceNotYetValid {
                        claim_id,
                        receipt_id: *receipt_id,
                    });
                indeterminate = true;
                continue;
            }
        }

        for requirement in &claim.evidence_requirements {
            let Some(state) = record.property_state(&requirement.property) else {
                continue;
            };
            match state {
                PropertyState::Supported => {
                    if requirement.accepts(record.method) {
                        supported_properties.insert(requirement.property.clone());
                    } else {
                        evaluation
                            .issues
                            .push(AssuranceIssueV3::EvidenceMethodNotAllowed {
                                claim_id,
                                receipt_id: *receipt_id,
                                property: requirement.property.clone(),
                                method: record.method,
                            });
                    }
                }
                PropertyState::Refuted => {
                    counterexample = true;
                    evaluation
                        .issues
                        .push(AssuranceIssueV3::CounterexampleEvidence {
                            claim_id,
                            receipt_id: *receipt_id,
                            property: requirement.property.clone(),
                            referenced_by_claim: true,
                        });
                }
                PropertyState::Indeterminate => {
                    if requirement.accepts(record.method) {
                        indeterminate = true;
                        evaluation
                            .issues
                            .push(AssuranceIssueV3::IndeterminateEvidence {
                                claim_id,
                                receipt_id: *receipt_id,
                                property: requirement.property.clone(),
                            });
                    }
                }
                PropertyState::Unexamined | PropertyState::Checked | PropertyState::Proved => {
                    indeterminate = true;
                }
            }
        }
    }

    // Counterexample completeness rule: a current admitted refutation already
    // present in the case dominates even if this claim omitted that receipt ID.
    // Positive support still requires explicit claim reference; negative evidence
    // cannot be hidden by selective referencing.
    for record in evidence.values().copied() {
        if evidence_usability(record, profile) != EvidenceUsability::Active {
            continue;
        }
        for requirement in &claim.evidence_requirements {
            if record.property_state(&requirement.property) == Some(PropertyState::Refuted) {
                if !claim.admitted_receipt_ids.contains(&record.receipt_id) {
                    evaluation
                        .issues
                        .push(AssuranceIssueV3::CounterexampleEvidence {
                            claim_id,
                            receipt_id: record.receipt_id,
                            property: requirement.property.clone(),
                            referenced_by_claim: false,
                        });
                }
                counterexample = true;
            }
        }
    }

    let mut missing_property = false;
    for requirement in &claim.evidence_requirements {
        if !supported_properties.contains(&requirement.property) {
            evaluation
                .issues
                .push(AssuranceIssueV3::MissingPropertyEvidence {
                    claim_id,
                    property: requirement.property.clone(),
                });
            missing_property = true;
        }
    }

    let dependency_ready = claim.dependency_claim_ids.iter().all(|id| {
        evaluation.claim_states.get(id) == Some(&ClaimState::QualifiedUnderProfile)
    });

    let state = if counterexample {
        ClaimState::CounterexampleFound
    } else if indeterminate {
        ClaimState::Indeterminate
    } else if invalidated {
        ClaimState::Invalidated
    } else if saw_stale {
        ClaimState::Stale
    } else if !missing_property && dependency_ready {
        ClaimState::QualifiedUnderProfile
    } else if saw_evidence {
        ClaimState::EvidenceCollected
    } else {
        ClaimState::Unexamined
    };

    visiting.remove(&claim_id);
    evaluation.claim_states.insert(claim_id, state);
    state
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceUsability {
    Active,
    SubjectMismatch,
    ProfileMismatch,
    EnvironmentMismatch,
    Expired,
    NotYetValid,
}

fn evidence_usability(
    evidence: &AdmittedVerificationEvidence,
    profile: &AssuranceProfile,
) -> EvidenceUsability {
    if evidence.subject != profile.subject {
        return EvidenceUsability::SubjectMismatch;
    }
    if evidence.profile_id != profile.profile_id {
        return EvidenceUsability::ProfileMismatch;
    }
    if evidence.receipt_issued_unix_s > profile.now_unix_s
        || evidence.admitted_at_unix_s > profile.now_unix_s
    {
        return EvidenceUsability::NotYetValid;
    }
    if evidence.is_expired_at(profile.now_unix_s) {
        return EvidenceUsability::Expired;
    }
    if evidence
        .properties
        .first()
        .is_some_and(|property| property.environment_digest != profile.environment_digest)
    {
        return EvidenceUsability::EnvironmentMismatch;
    }
    EvidenceUsability::Active
}

fn validate_admitted_evidence(evidence: &AdmittedVerificationEvidence) -> bool {
    if evidence.receipt_id == Uuid::nil()
        || !evidence.subject.validate()
        || !bounded_nonempty(&evidence.profile_id)
        || evidence.signer_authorization_grant_id == Uuid::nil()
        || evidence.evidence_admission_grant_id == Uuid::nil()
        || !bounded_nonempty(&evidence.authority_scope)
        || evidence.receipt_issued_unix_s > evidence.admitted_at_unix_s
        || evidence
            .valid_until_unix_s
            .is_some_and(|deadline| deadline < evidence.admitted_at_unix_s)
        || evidence.properties.is_empty()
        || evidence.properties.len() > MAX_ADMISSION_GRANT_PROPERTIES
        || evidence.does_not_establish.len() > MAX_ADMISSION_GRANT_PROPERTIES
        || !all_bounded_nonempty_unique(&evidence.does_not_establish)
    {
        return false;
    }

    let first = &evidence.properties[0];
    let mut property_names = BTreeSet::new();
    for property in &evidence.properties {
        if property.receipt_id != evidence.receipt_id
            || property.subject != evidence.subject
            || property.profile_id != evidence.profile_id
            || property.signer_authorization_grant_id != evidence.signer_authorization_grant_id
            || property.evidence_admission_grant_id != evidence.evidence_admission_grant_id
            || property.authority_scope != evidence.authority_scope
            || property.method != evidence.method
            || property.valid_until_unix_s != evidence.valid_until_unix_s
            || !bounded_nonempty(&property.property)
            || !property_names.insert(property.property.as_str())
            || !bounded_nonempty(&property.verifier_id)
            || !bounded_nonempty(&property.verifier_version)
            || !bounded_nonempty(&property.verifier_artifact_digest)
            || !bounded_nonempty(&property.environment_digest)
            || !bounded_nonempty(&property.input_digest)
            || !bounded_nonempty(&property.output_digest)
            || property
                .assumptions_digest
                .as_ref()
                .is_some_and(|value| !bounded_nonempty(value))
            || !matches!(
                property.state,
                PropertyState::Supported | PropertyState::Refuted | PropertyState::Indeterminate
            )
        {
            return false;
        }

        if property.verifier_id != first.verifier_id
            || property.verifier_version != first.verifier_version
            || property.verifier_artifact_digest != first.verifier_artifact_digest
            || property.environment_digest != first.environment_digest
            || property.input_digest != first.input_digest
            || property.output_digest != first.output_digest
            || property.assumptions_digest != first.assumptions_digest
        {
            return false;
        }
    }

    let limitation_names: BTreeSet<_> = evidence
        .does_not_establish
        .iter()
        .map(String::as_str)
        .collect();
    property_names.is_disjoint(&limitation_names)
}

fn bounded_nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= MAX_ADMISSION_GRANT_STRING_BYTES
}

fn all_bounded_nonempty_unique(values: &[String]) -> bool {
    values.iter().all(|value| bounded_nonempty(value))
        && BTreeSet::<_>::from_iter(values.iter().map(String::as_str)).len() == values.len()
}

fn close_over_dependents(claims: &[AssuranceClaimV3], affected: &mut BTreeSet<Uuid>) {
    loop {
        let before = affected.len();
        for claim in claims {
            if claim
                .dependency_claim_ids
                .iter()
                .any(|dependency| affected.contains(dependency))
            {
                affected.insert(claim.id);
            }
        }
        if affected.len() == before {
            break;
        }
    }
}

fn unique<T: Ord>(values: &[T]) -> bool {
    BTreeSet::<_>::from_iter(values.iter()).len() == values.len()
}

fn unique_by<'a, T, K: Ord + 'a>(values: &'a [T], key: impl Fn(&'a T) -> K) -> bool {
    let mut keys = BTreeSet::new();
    values.iter().all(|value| keys.insert(key(value)))
}

fn unique_ids(mut values: impl Iterator<Item = Uuid>) -> bool {
    let mut ids = BTreeSet::new();
    values.all(|id| id != Uuid::nil() && ids.insert(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt_admission::AdmittedPropertyEvidence;

    fn subject() -> SubjectRef {
        SubjectRef {
            namespace: "symthaea-test".into(),
            subject_id: "artifact-a".into(),
            digest: "sha256:subject".into(),
        }
    }

    fn profile(subject: SubjectRef) -> AssuranceProfile {
        AssuranceProfile {
            profile_id: "production".into(),
            subject,
            environment_digest: "sha256:env".into(),
            now_unix_s: 150,
        }
    }

    fn admitted(
        receipt_id: Uuid,
        subject: SubjectRef,
        profile_id: &str,
        method: EvidenceMethod,
        property: &str,
        state: PropertyState,
    ) -> AdmittedVerificationEvidence {
        let signer_grant = Uuid::new_v4();
        let admission_grant = Uuid::new_v4();
        AdmittedVerificationEvidence {
            receipt_id,
            subject: subject.clone(),
            profile_id: profile_id.into(),
            signer_authorization_grant_id: signer_grant,
            evidence_admission_grant_id: admission_grant,
            authority_scope: "attest:verification-receipt".into(),
            method,
            receipt_issued_unix_s: 100,
            admitted_at_unix_s: 120,
            valid_until_unix_s: Some(200),
            properties: vec![AdmittedPropertyEvidence {
                receipt_id,
                subject,
                profile_id: profile_id.into(),
                signer_authorization_grant_id: signer_grant,
                evidence_admission_grant_id: admission_grant,
                authority_scope: "attest:verification-receipt".into(),
                method,
                property: property.into(),
                state,
                verifier_id: "verifier".into(),
                verifier_version: "1.0".into(),
                verifier_artifact_digest: "sha256:verifier".into(),
                environment_digest: "sha256:env".into(),
                input_digest: "sha256:input".into(),
                output_digest: "sha256:output".into(),
                assumptions_digest: Some("sha256:assumptions".into()),
                valid_until_unix_s: Some(200),
            }],
            does_not_establish: vec!["property.unrelated".into()],
        }
    }

    fn claim(
        subject: SubjectRef,
        property: &str,
        allowed_methods: Vec<EvidenceMethod>,
        receipt_ids: Vec<Uuid>,
    ) -> AssuranceClaimV3 {
        AssuranceClaimV3 {
            id: Uuid::new_v4(),
            subject,
            proposition: "requested property holds under the profile".into(),
            evidence_requirements: vec![EvidenceRequirementV3 {
                property: property.into(),
                allowed_methods,
            }],
            assumption_ids: Vec::new(),
            dependency_claim_ids: Vec::new(),
            admitted_receipt_ids: receipt_ids,
        }
    }

    fn case(
        subject: SubjectRef,
        claims: Vec<AssuranceClaimV3>,
        evidence: Vec<AdmittedVerificationEvidence>,
    ) -> AssuranceCaseV3 {
        AssuranceCaseV3 {
            schema_version: ASSURANCE_CASE_V3_SCHEMA_VERSION,
            case_id: Uuid::new_v4(),
            subject,
            claims,
            assumptions: Vec::new(),
            admitted_evidence: evidence,
        }
    }

    #[test]
    fn exact_admitted_support_with_allowed_method_can_qualify() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Supported,
        );
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert!(evaluation.is_qualified(claim_id));
    }

    #[test]
    fn support_from_disallowed_method_cannot_qualify() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::Simulation,
            "auth.request-bound",
            PropertyState::Supported,
        );
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::EvidenceCollected)
        );
        assert!(evaluation.issues.iter().any(|issue| matches!(
            issue,
            AssuranceIssueV3::EvidenceMethodNotAllowed { receipt_id: id, .. }
                if *id == receipt_id
        )));
    }

    #[test]
    fn evidence_admitted_under_another_profile_cannot_qualify() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let evidence = admitted(
            receipt_id,
            subject.clone(),
            "lab",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Supported,
        );
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::Invalidated)
        );
    }

    #[test]
    fn expired_admission_makes_claim_stale() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let mut evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Supported,
        );
        evidence.valid_until_unix_s = Some(149);
        evidence.properties[0].valid_until_unix_s = Some(149);
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert_eq!(evaluation.claim_states.get(&claim_id), Some(&ClaimState::Stale));
    }

    #[test]
    fn future_admission_is_indeterminate() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let mut evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Supported,
        );
        evidence.receipt_issued_unix_s = 151;
        evidence.admitted_at_unix_s = 152;
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::Indeterminate)
        );
    }

    #[test]
    fn unreferenced_active_counterexample_dominates_referenced_support() {
        let subject = subject();
        let support_id = Uuid::new_v4();
        let refute_id = Uuid::new_v4();
        let support = admitted(
            support_id,
            subject.clone(),
            "production",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Supported,
        );
        let refute = admitted(
            refute_id,
            subject.clone(),
            "production",
            EvidenceMethod::RuntimeObservation,
            "auth.request-bound",
            PropertyState::Refuted,
        );
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![support_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![support, refute])
            .evaluate(&profile(subject));
        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::CounterexampleFound)
        );
        assert!(evaluation.issues.iter().any(|issue| matches!(
            issue,
            AssuranceIssueV3::CounterexampleEvidence {
                receipt_id,
                referenced_by_claim: false,
                ..
            } if *receipt_id == refute_id
        )));
    }

    #[test]
    fn admitted_inconclusive_evidence_cannot_become_support() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Indeterminate,
        );
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::Indeterminate)
        );
    }

    #[test]
    fn tampered_child_binding_invalidates_entire_case_structure() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let mut evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::FormalProof,
            "auth.request-bound",
            PropertyState::Supported,
        );
        evidence.properties[0].receipt_id = Uuid::new_v4();
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![receipt_id],
        );
        let claim_id = claim.id;

        let evaluation = case(subject.clone(), vec![claim], vec![evidence])
            .evaluate(&profile(subject));
        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::Indeterminate)
        );
        assert!(evaluation.issues.contains(&AssuranceIssueV3::InvalidCase));
    }

    #[test]
    fn receipt_change_impacts_property_consumers_even_if_not_explicitly_referenced() {
        let subject = subject();
        let receipt_id = Uuid::new_v4();
        let evidence = admitted(
            receipt_id,
            subject.clone(),
            "production",
            EvidenceMethod::RuntimeObservation,
            "auth.request-bound",
            PropertyState::Refuted,
        );
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![EvidenceMethod::FormalProof],
            vec![],
        );
        let claim_id = claim.id;
        let case = case(subject, vec![claim], vec![evidence]);

        assert_eq!(
            case.claims_affected_by_receipt(receipt_id),
            BTreeSet::from([claim_id])
        );
    }
}