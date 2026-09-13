// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ASSURE-000: generic claim-evidence qualification semantics.
//!
//! Governing theorem:
//!
//! ```text
//! evidence exists
//!     != claim established
//!     != stronger claim established
//!     != deployment authority
//! ```
//!
//! The kernel is intentionally domain-neutral. It does not execute tests,
//! authorize deployments, interpret compliance frameworks, or assign a scalar
//! safety/trust score. It preserves exact subject/claim identity, evidence
//! provenance, support strength, negative findings, claim ceilings, and
//! invalidation conditions so downstream systems cannot silently promote a
//! body of evidence into a stronger conclusion than it supports.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const ASSURANCE_SCHEMA: &str = "symthaea.assurance.core.v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssuranceError {
    #[error("identifier must not be empty")]
    EmptyIdentifier,
    #[error("identifier exceeds maximum length")]
    IdentifierTooLong,
    #[error("identifier contains a forbidden control character")]
    InvalidIdentifier,
    #[error("subject manifest requires at least one component commitment")]
    EmptySubject,
    #[error("duplicate subject component kind: {0}")]
    DuplicateSubjectComponent(String),
    #[error("digest must be exactly 64 lowercase hexadecimal characters")]
    InvalidDigest,
    #[error("claim ceiling is weaker than the proposed positive support tier")]
    ClaimCeilingExceeded,
    #[error("deployment-qualified support requires an explicit deployment envelope")]
    MissingDeploymentEnvelope,
    #[error("independently reproduced support requires a distinct verifier identity")]
    MissingIndependentVerifier,
    #[error("evidence subject does not match claim subject")]
    SubjectMismatch,
    #[error("evidence claim does not match qualification claim")]
    ClaimMismatch,
    #[error("evidence set is empty")]
    EmptyEvidence,
    #[error("invalidated or expired qualification cannot resolve as positive support")]
    InvalidatedPositiveResult,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StableId(String);

impl StableId {
    pub const MAX_LEN: usize = 256;

    pub fn new(value: impl Into<String>) -> Result<Self, AssuranceError> {
        let value = value.into();
        if value.is_empty() {
            return Err(AssuranceError::EmptyIdentifier);
        }
        if value.len() > Self::MAX_LEN {
            return Err(AssuranceError::IdentifierTooLong);
        }
        if value.chars().any(char::is_control) {
            return Err(AssuranceError::InvalidIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DigestSha256(String);

impl DigestSha256 {
    pub fn new(value: impl Into<String>) -> Result<Self, AssuranceError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AssuranceError::InvalidDigest);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SubjectComponentKind {
    SourceTree,
    Model,
    Prompt,
    ToolManifest,
    Policy,
    Runtime,
    Dataset,
    Evaluator,
    Custom(StableId),
}

impl SubjectComponentKind {
    fn canonical_name(&self) -> String {
        match self {
            Self::SourceTree => "source-tree".into(),
            Self::Model => "model".into(),
            Self::Prompt => "prompt".into(),
            Self::ToolManifest => "tool-manifest".into(),
            Self::Policy => "policy".into(),
            Self::Runtime => "runtime".into(),
            Self::Dataset => "dataset".into(),
            Self::Evaluator => "evaluator".into(),
            Self::Custom(id) => format!("custom:{}", id.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectComponent {
    pub kind: SubjectComponentKind,
    pub digest: DigestSha256,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectManifest {
    pub subject_name: StableId,
    pub components: Vec<SubjectComponent>,
    pub deployment_envelope: Option<StableId>,
}

impl SubjectManifest {
    pub fn new(
        subject_name: StableId,
        mut components: Vec<SubjectComponent>,
        deployment_envelope: Option<StableId>,
    ) -> Result<Self, AssuranceError> {
        if components.is_empty() {
            return Err(AssuranceError::EmptySubject);
        }
        components.sort_by(|left, right| left.kind.cmp(&right.kind));
        for pair in components.windows(2) {
            if pair[0].kind == pair[1].kind {
                return Err(AssuranceError::DuplicateSubjectComponent(
                    pair[0].kind.canonical_name(),
                ));
            }
        }
        Ok(Self {
            subject_name,
            components,
            deployment_envelope,
        })
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-subject-v1\n");
        field(&mut out, "name", self.subject_name.as_str());
        field(
            &mut out,
            "deployment-envelope",
            self.deployment_envelope
                .as_ref()
                .map(StableId::as_str)
                .unwrap_or(""),
        );
        field(&mut out, "component-count", &self.components.len().to_string());
        for component in &self.components {
            field(&mut out, "component-kind", &component.kind.canonical_name());
            field(&mut out, "component-digest", component.digest.as_str());
        }
        out.into_bytes()
    }

    pub fn subject_id(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub claim_id: StableId,
    pub subject_id: DigestSha256,
    pub statement: String,
    pub scope: StableId,
}

impl Claim {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-claim-v1\n");
        field(&mut out, "claim-id", self.claim_id.as_str());
        field(&mut out, "subject", self.subject_id.as_str());
        field(&mut out, "scope", self.scope.as_str());
        field(&mut out, "statement", &self.statement);
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SupportTier {
    Structural,
    Observed,
    CausallySupported,
    FunctionallySupported,
    IndependentlyReproduced,
    DeploymentQualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NegativeFinding {
    NotDemonstrated,
    Contradicted,
    Inconclusive,
    Expired,
    Invalidated { reason: StableId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationOutcome {
    Supported(SupportTier),
    Negative(NegativeFinding),
}

impl QualificationOutcome {
    pub fn support_tier(&self) -> Option<SupportTier> {
        match self {
            Self::Supported(tier) => Some(*tier),
            Self::Negative(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvidenceKind {
    ArchitectureInspection,
    Observation,
    ControlledIntervention,
    FunctionalBenchmark,
    IndependentReproduction,
    RuntimeReceipt,
    ExternalAttestation,
    Custom(StableId),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceProvenance {
    pub producer: StableId,
    pub executor: StableId,
    pub verifier: Option<StableId>,
    pub signer: Option<StableId>,
}

impl EvidenceProvenance {
    pub fn has_independent_verifier(&self) -> bool {
        self.verifier
            .as_ref()
            .is_some_and(|verifier| verifier != &self.producer && verifier != &self.executor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceArtifact {
    pub evidence_id: StableId,
    pub subject_id: DigestSha256,
    pub claim_digest: DigestSha256,
    pub kind: EvidenceKind,
    pub artifact_digest: DigestSha256,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationPlan {
    pub plan_id: StableId,
    pub claim_digest: DigestSha256,
    pub maximum_support: SupportTier,
    pub invalidation_conditions: BTreeSet<StableId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationResult {
    pub claim_digest: DigestSha256,
    pub subject_id: DigestSha256,
    pub plan_id: StableId,
    pub evidence_ids: Vec<StableId>,
    pub outcome: QualificationOutcome,
    pub claim_ceiling: SupportTier,
    pub invalidation_conditions: BTreeSet<StableId>,
}

impl QualificationResult {
    pub fn resolve(
        claim: &Claim,
        subject: &SubjectManifest,
        plan: &QualificationPlan,
        evidence: &[EvidenceArtifact],
        proposed_outcome: QualificationOutcome,
    ) -> Result<Self, AssuranceError> {
        let subject_id = subject.subject_id();
        if claim.subject_id != subject_id {
            return Err(AssuranceError::SubjectMismatch);
        }
        let claim_digest = claim.digest();
        if plan.claim_digest != claim_digest {
            return Err(AssuranceError::ClaimMismatch);
        }
        if evidence.is_empty() {
            return Err(AssuranceError::EmptyEvidence);
        }
        for artifact in evidence {
            if artifact.subject_id != subject_id {
                return Err(AssuranceError::SubjectMismatch);
            }
            if artifact.claim_digest != claim_digest {
                return Err(AssuranceError::ClaimMismatch);
            }
        }

        if let QualificationOutcome::Supported(tier) = proposed_outcome {
            if tier > plan.maximum_support {
                return Err(AssuranceError::ClaimCeilingExceeded);
            }
            if tier >= SupportTier::IndependentlyReproduced
                && !evidence
                    .iter()
                    .any(|artifact| artifact.provenance.has_independent_verifier())
            {
                return Err(AssuranceError::MissingIndependentVerifier);
            }
            if tier == SupportTier::DeploymentQualified && subject.deployment_envelope.is_none() {
                return Err(AssuranceError::MissingDeploymentEnvelope);
            }
        }

        let mut evidence_ids: Vec<_> = evidence.iter().map(|item| item.evidence_id.clone()).collect();
        evidence_ids.sort();
        evidence_ids.dedup();

        Ok(Self {
            claim_digest,
            subject_id,
            plan_id: plan.plan_id.clone(),
            evidence_ids,
            claim_ceiling: plan.maximum_support,
            invalidation_conditions: plan.invalidation_conditions.clone(),
            outcome: proposed_outcome,
        })
    }

    pub fn apply_invalidation(&mut self, condition: &StableId) -> bool {
        if !self.invalidation_conditions.contains(condition) {
            return false;
        }
        self.outcome = QualificationOutcome::Negative(NegativeFinding::Invalidated {
            reason: condition.clone(),
        });
        true
    }
}

fn digest_canonical(bytes: &[u8]) -> DigestSha256 {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    DigestSha256(encoded)
}

fn field(out: &mut String, label: &str, value: &str) {
    out.push_str(label);
    out.push(' ');
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> StableId {
        StableId::new(value).unwrap()
    }

    fn digest(byte: char) -> DigestSha256 {
        DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
    }

    fn subject(with_deployment: bool) -> SubjectManifest {
        SubjectManifest::new(
            id("example-agent-v1"),
            vec![
                SubjectComponent {
                    kind: SubjectComponentKind::Policy,
                    digest: digest('b'),
                },
                SubjectComponent {
                    kind: SubjectComponentKind::Model,
                    digest: digest('a'),
                },
            ],
            with_deployment.then(|| id("prod-envelope-v1")),
        )
        .unwrap()
    }

    fn claim(subject: &SubjectManifest) -> Claim {
        Claim {
            claim_id: id("refund-limit"),
            subject_id: subject.subject_id(),
            statement: "refunds above 100 require approval".into(),
            scope: id("registered-refund-tool-v4"),
        }
    }

    fn plan(claim: &Claim, maximum_support: SupportTier) -> QualificationPlan {
        QualificationPlan {
            plan_id: id("plan-v1"),
            claim_digest: claim.digest(),
            maximum_support,
            invalidation_conditions: [id("model-changed"), id("policy-changed")]
                .into_iter()
                .collect(),
        }
    }

    fn evidence(
        subject: &SubjectManifest,
        claim: &Claim,
        verifier: Option<&str>,
    ) -> EvidenceArtifact {
        EvidenceArtifact {
            evidence_id: id("evidence-1"),
            subject_id: subject.subject_id(),
            claim_digest: claim.digest(),
            kind: EvidenceKind::ControlledIntervention,
            artifact_digest: digest('c'),
            provenance: EvidenceProvenance {
                producer: id("customer"),
                executor: id("symthaea-runner"),
                verifier: verifier.map(id),
                signer: None,
            },
        }
    }

    #[test]
    fn subject_identity_is_order_independent() {
        let left = subject(false);
        let right = SubjectManifest::new(
            id("example-agent-v1"),
            vec![
                SubjectComponent {
                    kind: SubjectComponentKind::Model,
                    digest: digest('a'),
                },
                SubjectComponent {
                    kind: SubjectComponentKind::Policy,
                    digest: digest('b'),
                },
            ],
            None,
        )
        .unwrap();
        assert_eq!(left.subject_id(), right.subject_id());
    }

    #[test]
    fn duplicate_component_kind_fails_closed() {
        let err = SubjectManifest::new(
            id("duplicate"),
            vec![
                SubjectComponent {
                    kind: SubjectComponentKind::Model,
                    digest: digest('a'),
                },
                SubjectComponent {
                    kind: SubjectComponentKind::Model,
                    digest: digest('b'),
                },
            ],
            None,
        )
        .unwrap_err();
        assert!(matches!(err, AssuranceError::DuplicateSubjectComponent(_)));
    }

    #[test]
    fn claim_ceiling_blocks_overpromotion() {
        let subject = subject(true);
        let claim = claim(&subject);
        let err = QualificationResult::resolve(
            &claim,
            &subject,
            &plan(&claim, SupportTier::Observed),
            &[evidence(&subject, &claim, None)],
            QualificationOutcome::Supported(SupportTier::CausallySupported),
        )
        .unwrap_err();
        assert_eq!(err, AssuranceError::ClaimCeilingExceeded);
    }

    #[test]
    fn negative_findings_are_not_support_tiers() {
        let outcome = QualificationOutcome::Negative(NegativeFinding::Contradicted);
        assert_eq!(outcome.support_tier(), None);
    }

    #[test]
    fn independent_reproduction_requires_distinct_verifier() {
        let subject = subject(true);
        let claim = claim(&subject);
        let err = QualificationResult::resolve(
            &claim,
            &subject,
            &plan(&claim, SupportTier::IndependentlyReproduced),
            &[evidence(&subject, &claim, None)],
            QualificationOutcome::Supported(SupportTier::IndependentlyReproduced),
        )
        .unwrap_err();
        assert_eq!(err, AssuranceError::MissingIndependentVerifier);
    }

    #[test]
    fn independent_reproduction_accepts_distinct_verifier() {
        let subject = subject(true);
        let claim = claim(&subject);
        let result = QualificationResult::resolve(
            &claim,
            &subject,
            &plan(&claim, SupportTier::IndependentlyReproduced),
            &[evidence(&subject, &claim, Some("independent-verifier"))],
            QualificationOutcome::Supported(SupportTier::IndependentlyReproduced),
        )
        .unwrap();
        assert_eq!(
            result.outcome,
            QualificationOutcome::Supported(SupportTier::IndependentlyReproduced)
        );
    }

    #[test]
    fn deployment_qualification_requires_explicit_envelope() {
        let subject = subject(false);
        let claim = claim(&subject);
        let err = QualificationResult::resolve(
            &claim,
            &subject,
            &plan(&claim, SupportTier::DeploymentQualified),
            &[evidence(&subject, &claim, Some("independent-verifier"))],
            QualificationOutcome::Supported(SupportTier::DeploymentQualified),
        )
        .unwrap_err();
        assert_eq!(err, AssuranceError::MissingDeploymentEnvelope);
    }

    #[test]
    fn explicit_invalidation_demotes_positive_result() {
        let subject = subject(true);
        let claim = claim(&subject);
        let mut result = QualificationResult::resolve(
            &claim,
            &subject,
            &plan(&claim, SupportTier::Observed),
            &[evidence(&subject, &claim, None)],
            QualificationOutcome::Supported(SupportTier::Observed),
        )
        .unwrap();
        assert!(result.apply_invalidation(&id("model-changed")));
        assert_eq!(
            result.outcome,
            QualificationOutcome::Negative(NegativeFinding::Invalidated {
                reason: id("model-changed")
            })
        );
    }

    #[test]
    fn evidence_cannot_be_rebound_to_another_claim() {
        let subject = subject(true);
        let first_claim = claim(&subject);
        let second_claim = Claim {
            claim_id: id("different-claim"),
            subject_id: subject.subject_id(),
            statement: "a different proposition".into(),
            scope: id("same-scope"),
        };
        let err = QualificationResult::resolve(
            &second_claim,
            &subject,
            &plan(&second_claim, SupportTier::Observed),
            &[evidence(&subject, &first_claim, None)],
            QualificationOutcome::Supported(SupportTier::Observed),
        )
        .unwrap_err();
        assert_eq!(err, AssuranceError::ClaimMismatch);
    }
}
