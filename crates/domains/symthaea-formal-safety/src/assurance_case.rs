// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound assurance-case primitives.
//!
//! This module is deliberately stricter than the legacy `ProofObligation` API.
//! Evidence is not authority merely because a reference exists: qualification is
//! derived from exact subject/environment binding, explicit claim properties,
//! profile-bound assumptions, dependencies, freshness, and contradiction
//! handling.
//!
//! The types here do not perform cryptographic verification themselves. Digests
//! and verifier receipts are inputs from independently qualified mechanisms such
//! as Xenia, Forge, Nixward, HIL systems, proof assistants, or test harnesses.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const ASSURANCE_CASE_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SubjectRef {
    pub namespace: String,
    pub subject_id: String,
    pub digest: String,
}

impl SubjectRef {
    pub fn validate(&self) -> bool {
        !self.namespace.trim().is_empty()
            && !self.subject_id.trim().is_empty()
            && !self.digest.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceMethod {
    Inspection,
    StaticAnalysis,
    ExampleTest,
    PropertyTest,
    FuzzCampaign,
    AdversarialCampaign,
    Simulation,
    RuntimeObservation,
    ModelCheck,
    FormalProof,
    StandardReference,
    SignedAttestation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceDisposition {
    Supports,
    Contradicts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceValidity {
    Active,
    Withdrawn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub id: Uuid,
    pub subject: SubjectRef,
    pub method: EvidenceMethod,
    pub disposition: EvidenceDisposition,
    pub artifact_digest: String,
    pub environment_digest: String,
    pub establishes: Vec<String>,
    pub does_not_establish: Vec<String>,
    pub verifier_receipt: Option<String>,
    pub valid_until_unix_s: Option<u64>,
    pub validity: EvidenceValidity,
}

impl EvidenceRecord {
    pub fn validate(&self) -> bool {
        if !self.subject.validate()
            || self.artifact_digest.trim().is_empty()
            || self.environment_digest.trim().is_empty()
            || self.establishes.is_empty()
            || self.establishes.iter().any(|value| value.trim().is_empty())
            || self
                .does_not_establish
                .iter()
                .any(|value| value.trim().is_empty())
            || !unique(&self.establishes)
            || !unique(&self.does_not_establish)
        {
            return false;
        }

        let establishes: BTreeSet<_> = self.establishes.iter().collect();
        let limitations: BTreeSet<_> = self.does_not_establish.iter().collect();
        establishes.is_disjoint(&limitations)
    }

    fn is_expired(&self, now_unix_s: u64) -> bool {
        self.valid_until_unix_s
            .is_some_and(|deadline| now_unix_s > deadline)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssumptionStatus {
    Open,
    AcceptedUnderProfile(String),
    Refuted,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assumption {
    pub id: Uuid,
    pub subject: SubjectRef,
    pub statement: String,
    pub status: AssumptionStatus,
}

impl Assumption {
    pub fn validate(&self) -> bool {
        self.subject.validate()
            && !self.statement.trim().is_empty()
            && match &self.status {
                AssumptionStatus::AcceptedUnderProfile(profile_id) => {
                    !profile_id.trim().is_empty()
                }
                _ => true,
            }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceClaim {
    pub id: Uuid,
    pub subject: SubjectRef,
    pub proposition: String,
    pub required_properties: Vec<String>,
    pub assumption_ids: Vec<Uuid>,
    pub dependency_claim_ids: Vec<Uuid>,
    pub evidence_ids: Vec<Uuid>,
}

impl AssuranceClaim {
    pub fn validate(&self) -> bool {
        self.subject.validate()
            && !self.proposition.trim().is_empty()
            && !self.required_properties.is_empty()
            && self
                .required_properties
                .iter()
                .all(|property| !property.trim().is_empty())
            && unique(&self.required_properties)
            && unique(&self.assumption_ids)
            && unique(&self.dependency_claim_ids)
            && unique(&self.evidence_ids)
            && !self.dependency_claim_ids.contains(&self.id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceProfile {
    pub profile_id: String,
    pub subject: SubjectRef,
    pub environment_digest: String,
    pub now_unix_s: u64,
}

impl AssuranceProfile {
    pub fn validate(&self) -> bool {
        !self.profile_id.trim().is_empty()
            && self.subject.validate()
            && !self.environment_digest.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimState {
    Unexamined,
    EvidenceCollected,
    QualifiedUnderProfile,
    CounterexampleFound,
    Invalidated,
    Stale,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssuranceIssue {
    InvalidCase,
    InvalidProfile,
    SubjectMismatch {
        claim_id: Uuid,
        evidence_id: Option<Uuid>,
    },
    EnvironmentMismatch {
        claim_id: Uuid,
        evidence_id: Uuid,
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
    UnknownEvidence {
        claim_id: Uuid,
        evidence_id: Uuid,
    },
    WithdrawnEvidence {
        claim_id: Uuid,
        evidence_id: Uuid,
    },
    ExpiredEvidence {
        claim_id: Uuid,
        evidence_id: Uuid,
    },
    MissingPropertyEvidence {
        claim_id: Uuid,
        property: String,
    },
    ContradictoryEvidence {
        claim_id: Uuid,
        evidence_id: Uuid,
        property: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceEvaluation {
    pub schema_version: u16,
    pub case_id: Uuid,
    pub profile_id: String,
    pub claim_states: BTreeMap<Uuid, ClaimState>,
    pub issues: Vec<AssuranceIssue>,
}

impl AssuranceEvaluation {
    pub fn is_qualified(&self, claim_id: Uuid) -> bool {
        self.claim_states.get(&claim_id) == Some(&ClaimState::QualifiedUnderProfile)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceCaseV2 {
    pub schema_version: u16,
    pub case_id: Uuid,
    pub subject: SubjectRef,
    pub claims: Vec<AssuranceClaim>,
    pub assumptions: Vec<Assumption>,
    pub evidence: Vec<EvidenceRecord>,
}

impl AssuranceCaseV2 {
    pub fn new(subject: SubjectRef) -> Self {
        Self {
            schema_version: ASSURANCE_CASE_SCHEMA_VERSION,
            case_id: Uuid::new_v4(),
            subject,
            claims: Vec::new(),
            assumptions: Vec::new(),
            evidence: Vec::new(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == ASSURANCE_CASE_SCHEMA_VERSION
            && self.subject.validate()
            && unique_ids(self.claims.iter().map(|value| value.id))
            && unique_ids(self.assumptions.iter().map(|value| value.id))
            && unique_ids(self.evidence.iter().map(|value| value.id))
            && self.claims.iter().all(AssuranceClaim::validate)
            && self.assumptions.iter().all(Assumption::validate)
            && self.evidence.iter().all(EvidenceRecord::validate)
    }

    pub fn evaluate(&self, profile: &AssuranceProfile) -> AssuranceEvaluation {
        let mut evaluation = AssuranceEvaluation {
            schema_version: ASSURANCE_CASE_SCHEMA_VERSION,
            case_id: self.case_id,
            profile_id: profile.profile_id.clone(),
            claim_states: BTreeMap::new(),
            issues: Vec::new(),
        };

        if !self.validate() {
            evaluation.issues.push(AssuranceIssue::InvalidCase);
            for claim in &self.claims {
                evaluation
                    .claim_states
                    .insert(claim.id, ClaimState::Indeterminate);
            }
            return evaluation;
        }
        if !profile.validate() || profile.subject != self.subject {
            evaluation.issues.push(AssuranceIssue::InvalidProfile);
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
        let evidence: BTreeMap<_, _> = self.evidence.iter().map(|value| (value.id, value)).collect();
        let mut visiting = BTreeSet::new();

        for claim in &self.claims {
            evaluate_claim(
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

    /// Returns every claim whose conclusion may depend directly or transitively
    /// on the supplied evidence record. This is a requalification impact set,
    /// not proof that every returned claim is invalid.
    pub fn claims_affected_by_evidence(&self, evidence_id: Uuid) -> BTreeSet<Uuid> {
        let mut affected: BTreeSet<_> = self
            .claims
            .iter()
            .filter(|claim| claim.evidence_ids.contains(&evidence_id))
            .map(|claim| claim.id)
            .collect();
        close_over_dependents(&self.claims, &mut affected);
        affected
    }

    /// Returns every claim whose conclusion may depend directly or transitively
    /// on the supplied assumption.
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
}

#[allow(clippy::too_many_arguments)]
fn evaluate_claim(
    claim_id: Uuid,
    profile: &AssuranceProfile,
    claims: &BTreeMap<Uuid, &AssuranceClaim>,
    assumptions: &BTreeMap<Uuid, &Assumption>,
    evidence: &BTreeMap<Uuid, &EvidenceRecord>,
    visiting: &mut BTreeSet<Uuid>,
    evaluation: &mut AssuranceEvaluation,
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
            .push(AssuranceIssue::DependencyCycle(claim_id));
        return ClaimState::Indeterminate;
    }

    if claim.subject != profile.subject {
        evaluation.issues.push(AssuranceIssue::SubjectMismatch {
            claim_id,
            evidence_id: None,
        });
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
            evaluation.issues.push(AssuranceIssue::UnknownAssumption {
                claim_id,
                assumption_id: *assumption_id,
            });
            indeterminate = true;
            continue;
        };
        if assumption.subject != profile.subject {
            evaluation.issues.push(AssuranceIssue::SubjectMismatch {
                claim_id,
                evidence_id: None,
            });
            invalidated = true;
            continue;
        }
        match &assumption.status {
            AssumptionStatus::AcceptedUnderProfile(accepted)
                if accepted == &profile.profile_id => {}
            AssumptionStatus::AcceptedUnderProfile(accepted) => {
                evaluation
                    .issues
                    .push(AssuranceIssue::AssumptionProfileMismatch {
                        claim_id,
                        assumption_id: *assumption_id,
                        accepted_profile_id: accepted.clone(),
                    });
                invalidated = true;
            }
            AssumptionStatus::Open => {
                evaluation.issues.push(AssuranceIssue::OpenAssumption {
                    claim_id,
                    assumption_id: *assumption_id,
                });
                invalidated = true;
            }
            AssumptionStatus::Refuted => {
                evaluation.issues.push(AssuranceIssue::RefutedAssumption {
                    claim_id,
                    assumption_id: *assumption_id,
                });
                invalidated = true;
            }
            AssumptionStatus::Stale => {
                evaluation.issues.push(AssuranceIssue::StaleAssumption {
                    claim_id,
                    assumption_id: *assumption_id,
                });
                saw_stale = true;
            }
        }
    }

    for dependency_id in &claim.dependency_claim_ids {
        if !claims.contains_key(dependency_id) {
            evaluation.issues.push(AssuranceIssue::UnknownDependency {
                claim_id,
                dependency_claim_id: *dependency_id,
            });
            indeterminate = true;
            continue;
        }
        let state = evaluate_claim(
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
                .push(AssuranceIssue::DependencyNotQualified {
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

    let mut counterexample = false;
    let mut supported_properties = BTreeSet::new();

    for evidence_id in &claim.evidence_ids {
        let Some(record) = evidence.get(evidence_id).copied() else {
            evaluation.issues.push(AssuranceIssue::UnknownEvidence {
                claim_id,
                evidence_id: *evidence_id,
            });
            indeterminate = true;
            continue;
        };
        saw_evidence = true;
        if record.subject != profile.subject {
            evaluation.issues.push(AssuranceIssue::SubjectMismatch {
                claim_id,
                evidence_id: Some(*evidence_id),
            });
            continue;
        }
        if record.environment_digest != profile.environment_digest {
            evaluation
                .issues
                .push(AssuranceIssue::EnvironmentMismatch {
                    claim_id,
                    evidence_id: *evidence_id,
                });
            saw_stale = true;
            continue;
        }
        if record.validity == EvidenceValidity::Withdrawn {
            evaluation.issues.push(AssuranceIssue::WithdrawnEvidence {
                claim_id,
                evidence_id: *evidence_id,
            });
            saw_stale = true;
            continue;
        }
        if record.is_expired(profile.now_unix_s) {
            evaluation.issues.push(AssuranceIssue::ExpiredEvidence {
                claim_id,
                evidence_id: *evidence_id,
            });
            saw_stale = true;
            continue;
        }

        for property in &claim.required_properties {
            if record.establishes.contains(property) {
                match record.disposition {
                    EvidenceDisposition::Supports => {
                        supported_properties.insert(property.clone());
                    }
                    EvidenceDisposition::Contradicts => {
                        evaluation
                            .issues
                            .push(AssuranceIssue::ContradictoryEvidence {
                                claim_id,
                                evidence_id: *evidence_id,
                                property: property.clone(),
                            });
                        counterexample = true;
                    }
                }
            }
        }
    }

    let mut missing_property = false;
    for property in &claim.required_properties {
        if !supported_properties.contains(property) {
            evaluation
                .issues
                .push(AssuranceIssue::MissingPropertyEvidence {
                    claim_id,
                    property: property.clone(),
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

fn close_over_dependents(claims: &[AssuranceClaim], affected: &mut BTreeSet<Uuid>) {
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
    let set: BTreeSet<_> = values.iter().collect();
    set.len() == values.len()
}

fn unique_ids(mut values: impl Iterator<Item = Uuid>) -> bool {
    let mut ids = BTreeSet::new();
    values.all(|id| ids.insert(id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(digest: &str) -> SubjectRef {
        SubjectRef {
            namespace: "symthaea-test".into(),
            subject_id: "artifact-a".into(),
            digest: digest.into(),
        }
    }

    fn profile(subject: SubjectRef) -> AssuranceProfile {
        AssuranceProfile {
            profile_id: "security-profile-v1".into(),
            subject,
            environment_digest: "env-1".into(),
            now_unix_s: 100,
        }
    }

    fn supporting_evidence(subject: SubjectRef, property: &str) -> EvidenceRecord {
        EvidenceRecord {
            id: Uuid::new_v4(),
            subject,
            method: EvidenceMethod::PropertyTest,
            disposition: EvidenceDisposition::Supports,
            artifact_digest: "evidence-artifact-1".into(),
            environment_digest: "env-1".into(),
            establishes: vec![property.into()],
            does_not_establish: vec!["unrelated-property".into()],
            verifier_receipt: Some("receipt-1".into()),
            valid_until_unix_s: Some(200),
            validity: EvidenceValidity::Active,
        }
    }

    fn claim(subject: SubjectRef, property: &str, evidence_ids: Vec<Uuid>) -> AssuranceClaim {
        AssuranceClaim {
            id: Uuid::new_v4(),
            subject,
            proposition: "the requested security property holds under the profile".into(),
            required_properties: vec![property.into()],
            assumption_ids: Vec::new(),
            dependency_claim_ids: Vec::new(),
            evidence_ids,
        }
    }

    fn case(
        subject: SubjectRef,
        claims: Vec<AssuranceClaim>,
        assumptions: Vec<Assumption>,
        evidence: Vec<EvidenceRecord>,
    ) -> AssuranceCaseV2 {
        AssuranceCaseV2 {
            schema_version: ASSURANCE_CASE_SCHEMA_VERSION,
            case_id: Uuid::new_v4(),
            subject,
            claims,
            assumptions,
            evidence,
        }
    }

    #[test]
    fn evidence_reference_alone_does_not_discharge_claim() {
        let subject = subject("subject-1");
        let evidence = supporting_evidence(subject.clone(), "auth.request-bound");
        let claim = claim(subject.clone(), "auth.replay-resistant", vec![evidence.id]);
        let claim_id = claim.id;
        let evaluation = case(subject.clone(), vec![claim], vec![], vec![evidence])
            .evaluate(&profile(subject));

        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::EvidenceCollected)
        );
        assert!(!evaluation.is_qualified(claim_id));
    }

    #[test]
    fn exact_property_subject_and_environment_can_qualify() {
        let subject = subject("subject-1");
        let evidence = supporting_evidence(subject.clone(), "auth.request-bound");
        let claim = claim(subject.clone(), "auth.request-bound", vec![evidence.id]);
        let claim_id = claim.id;
        let evaluation = case(subject.clone(), vec![claim], vec![], vec![evidence])
            .evaluate(&profile(subject));

        assert!(evaluation.is_qualified(claim_id));
    }

    #[test]
    fn evidence_for_different_subject_cannot_qualify() {
        let case_subject = subject("subject-1");
        let evidence = supporting_evidence(subject("subject-2"), "auth.request-bound");
        let claim = claim(
            case_subject.clone(),
            "auth.request-bound",
            vec![evidence.id],
        );
        let claim_id = claim.id;
        let evaluation = case(case_subject.clone(), vec![claim], vec![], vec![evidence])
            .evaluate(&profile(case_subject));

        assert!(!evaluation.is_qualified(claim_id));
        assert!(evaluation.issues.iter().any(|issue| matches!(
            issue,
            AssuranceIssue::SubjectMismatch {
                claim_id: id,
                evidence_id: Some(_)
            } if *id == claim_id
        )));
    }

    #[test]
    fn expired_evidence_makes_claim_stale() {
        let subject = subject("subject-1");
        let mut evidence = supporting_evidence(subject.clone(), "auth.request-bound");
        evidence.valid_until_unix_s = Some(99);
        let claim = claim(subject.clone(), "auth.request-bound", vec![evidence.id]);
        let claim_id = claim.id;
        let evaluation = case(subject.clone(), vec![claim], vec![], vec![evidence])
            .evaluate(&profile(subject));

        assert_eq!(evaluation.claim_states.get(&claim_id), Some(&ClaimState::Stale));
    }

    #[test]
    fn active_counterexample_dominates_supporting_evidence() {
        let subject = subject("subject-1");
        let support = supporting_evidence(subject.clone(), "auth.request-bound");
        let mut contradiction = supporting_evidence(subject.clone(), "auth.request-bound");
        contradiction.id = Uuid::new_v4();
        contradiction.disposition = EvidenceDisposition::Contradicts;
        contradiction.artifact_digest = "counterexample-1".into();
        let claim = claim(
            subject.clone(),
            "auth.request-bound",
            vec![support.id, contradiction.id],
        );
        let claim_id = claim.id;
        let evaluation = case(
            subject.clone(),
            vec![claim],
            vec![],
            vec![support, contradiction],
        )
        .evaluate(&profile(subject));

        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::CounterexampleFound)
        );
    }

    #[test]
    fn assumption_acceptance_is_profile_bound() {
        let subject = subject("subject-1");
        let evidence = supporting_evidence(subject.clone(), "auth.request-bound");
        let assumption = Assumption {
            id: Uuid::new_v4(),
            subject: subject.clone(),
            statement: "trusted clock remains monotonic".into(),
            status: AssumptionStatus::AcceptedUnderProfile("different-profile".into()),
        };
        let mut claim = claim(subject.clone(), "auth.request-bound", vec![evidence.id]);
        claim.assumption_ids.push(assumption.id);
        let claim_id = claim.id;
        let evaluation = case(
            subject.clone(),
            vec![claim],
            vec![assumption],
            vec![evidence],
        )
        .evaluate(&profile(subject));

        assert_eq!(
            evaluation.claim_states.get(&claim_id),
            Some(&ClaimState::Invalidated)
        );
    }

    #[test]
    fn dependency_must_be_qualified_before_parent() {
        let subject = subject("subject-1");
        let child_evidence = supporting_evidence(subject.clone(), "auth.request-bound");
        let child = claim(
            subject.clone(),
            "auth.request-bound",
            vec![child_evidence.id],
        );
        let child_id = child.id;
        let parent_evidence = supporting_evidence(subject.clone(), "deploy.authorized");
        let mut parent = claim(
            subject.clone(),
            "deploy.authorized",
            vec![parent_evidence.id],
        );
        parent.dependency_claim_ids.push(child_id);
        let parent_id = parent.id;
        let evaluation = case(
            subject.clone(),
            vec![child, parent],
            vec![],
            vec![child_evidence, parent_evidence],
        )
        .evaluate(&profile(subject));

        assert!(evaluation.is_qualified(child_id));
        assert!(evaluation.is_qualified(parent_id));
    }

    #[test]
    fn dependency_cycle_is_indeterminate() {
        let subject = subject("subject-1");
        let first_evidence = supporting_evidence(subject.clone(), "a");
        let second_evidence = supporting_evidence(subject.clone(), "b");
        let mut first = claim(subject.clone(), "a", vec![first_evidence.id]);
        let mut second = claim(subject.clone(), "b", vec![second_evidence.id]);
        first.dependency_claim_ids.push(second.id);
        second.dependency_claim_ids.push(first.id);
        let first_id = first.id;
        let second_id = second.id;
        let evaluation = case(
            subject.clone(),
            vec![first, second],
            vec![],
            vec![first_evidence, second_evidence],
        )
        .evaluate(&profile(subject));

        assert_eq!(
            evaluation.claim_states.get(&first_id),
            Some(&ClaimState::Indeterminate)
        );
        assert_eq!(
            evaluation.claim_states.get(&second_id),
            Some(&ClaimState::Indeterminate)
        );
    }

    #[test]
    fn evidence_change_marks_transitive_dependents_for_requalification() {
        let subject = subject("subject-1");
        let child_evidence = supporting_evidence(subject.clone(), "auth.request-bound");
        let child = claim(
            subject.clone(),
            "auth.request-bound",
            vec![child_evidence.id],
        );
        let child_id = child.id;
        let parent_evidence = supporting_evidence(subject.clone(), "deploy.authorized");
        let mut parent = claim(
            subject.clone(),
            "deploy.authorized",
            vec![parent_evidence.id],
        );
        parent.dependency_claim_ids.push(child_id);
        let parent_id = parent.id;
        let assurance_case = case(
            subject,
            vec![child, parent],
            vec![],
            vec![child_evidence.clone(), parent_evidence],
        );

        let affected = assurance_case.claims_affected_by_evidence(child_evidence.id);
        assert_eq!(affected, BTreeSet::from([child_id, parent_id]));
    }
}
