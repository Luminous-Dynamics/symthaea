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
//! This crate is intentionally domain-neutral and non-authoritative. It does
//! not run evaluations, certify systems, authorize deployments, interpret
//! regulations, or assign scalar safety/trust scores. It provides exact,
//! deterministic semantics for subjects, claims, evidence provenance,
//! qualification strength, negative findings, claim ceilings, and explicit
//! invalidation conditions.

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const ASSURANCE_SCHEMA: &str = "symthaea.assurance.core.v1";
const MAX_ID_LEN: usize = 256;
const MAX_CLAIM_LEN: usize = 16 * 1024;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssuranceError {
    #[error("identifier must not be empty")]
    EmptyIdentifier,
    #[error("identifier exceeds 256 bytes")]
    IdentifierTooLong,
    #[error("identifier contains a control character")]
    InvalidIdentifier,
    #[error("digest must be exactly 64 lowercase hexadecimal characters")]
    InvalidDigest,
    #[error("claim statement must not be empty")]
    EmptyClaimStatement,
    #[error("claim statement exceeds 16384 bytes")]
    ClaimStatementTooLong,
    #[error("subject manifest requires at least one component commitment")]
    EmptySubject,
    #[error("duplicate subject component kind: {0}")]
    DuplicateSubjectComponent(String),
    #[error("evidence set is empty")]
    EmptyEvidence,
    #[error("duplicate evidence identity: {0}")]
    DuplicateEvidenceId(String),
    #[error("claim is bound to a different subject")]
    SubjectMismatch,
    #[error("qualification plan or evidence is bound to a different claim")]
    ClaimMismatch,
    #[error("claim ceiling is weaker than the proposed positive support tier")]
    ClaimCeilingExceeded,
    #[error("evidence does not satisfy the proposed support tier: {0}")]
    InsufficientEvidenceForTier(String),
    #[error("reproduced support requires a verifier identity distinct from producer and executor")]
    MissingDistinctVerifier,
    #[error("deployment-qualified support requires an explicit deployment envelope")]
    MissingDeploymentEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableId(String);

impl StableId {
    pub fn new(value: impl Into<String>) -> Result<Self, AssuranceError> {
        let value = value.into();
        if value.is_empty() {
            return Err(AssuranceError::EmptyIdentifier);
        }
        if value.len() > MAX_ID_LEN {
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectComponent {
    pub kind: SubjectComponentKind,
    pub digest: DigestSha256,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectManifest {
    subject_name: StableId,
    components: Vec<SubjectComponent>,
    deployment_envelope: Option<StableId>,
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

    pub fn subject_name(&self) -> &StableId {
        &self.subject_name
    }

    pub fn components(&self) -> &[SubjectComponent] {
        &self.components
    }

    pub fn deployment_envelope(&self) -> Option<&StableId> {
        self.deployment_envelope.as_ref()
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-subject-v1\n");
        field(&mut out, "name", self.subject_name.as_str());
        optional_id(&mut out, "deployment-envelope", self.deployment_envelope());
        field(
            &mut out,
            "component-count",
            &self.components.len().to_string(),
        );
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    claim_id: StableId,
    subject_id: DigestSha256,
    statement: String,
    scope: StableId,
}

impl Claim {
    pub fn new(
        claim_id: StableId,
        subject_id: DigestSha256,
        statement: impl Into<String>,
        scope: StableId,
    ) -> Result<Self, AssuranceError> {
        let statement = statement.into();
        if statement.is_empty() {
            return Err(AssuranceError::EmptyClaimStatement);
        }
        if statement.len() > MAX_CLAIM_LEN {
            return Err(AssuranceError::ClaimStatementTooLong);
        }
        Ok(Self {
            claim_id,
            subject_id,
            statement,
            scope,
        })
    }

    pub fn subject_id(&self) -> &DigestSha256 {
        &self.subject_id
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SupportTier {
    Structural,
    Observed,
    CausallySupported,
    FunctionallySupported,
    Reproduced,
    DeploymentQualified,
}

impl SupportTier {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Observed => "observed",
            Self::CausallySupported => "causally-supported",
            Self::FunctionallySupported => "functionally-supported",
            Self::Reproduced => "reproduced",
            Self::DeploymentQualified => "deployment-qualified",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegativeFinding {
    NotDemonstrated,
    Contradicted,
    Inconclusive,
    Expired,
    Invalidated { reason: StableId },
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceKind {
    ArchitectureInspection,
    Observation,
    ControlledIntervention,
    FunctionalBenchmark,
    Reproduction,
    RuntimeReceipt,
    ExternalAttestation,
    Custom(StableId),
}

impl EvidenceKind {
    fn canonical_name(&self) -> String {
        match self {
            Self::ArchitectureInspection => "architecture-inspection".into(),
            Self::Observation => "observation".into(),
            Self::ControlledIntervention => "controlled-intervention".into(),
            Self::FunctionalBenchmark => "functional-benchmark".into(),
            Self::Reproduction => "reproduction".into(),
            Self::RuntimeReceipt => "runtime-receipt".into(),
            Self::ExternalAttestation => "external-attestation".into(),
            Self::Custom(id) => format!("custom:{}", id.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProvenance {
    producer: StableId,
    executor: StableId,
    verifier: Option<StableId>,
    signer: Option<StableId>,
}

impl EvidenceProvenance {
    pub fn new(
        producer: StableId,
        executor: StableId,
        verifier: Option<StableId>,
        signer: Option<StableId>,
    ) -> Self {
        Self {
            producer,
            executor,
            verifier,
            signer,
        }
    }

    /// Returns true only when a verifier identity is present and differs from
    /// both producer and executor identities.
    ///
    /// This is identity separation, not evidence of common-cause independence.
    /// Organization, review-process, toolchain, and evidence-source diversity
    /// require a stronger verifier-diversity layer.
    pub fn has_distinct_verifier_identity(&self) -> bool {
        self.verifier
            .as_ref()
            .is_some_and(|verifier| verifier != &self.producer && verifier != &self.executor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceArtifact {
    evidence_id: StableId,
    subject_id: DigestSha256,
    claim_digest: DigestSha256,
    kind: EvidenceKind,
    artifact_digest: DigestSha256,
    provenance: EvidenceProvenance,
}

impl EvidenceArtifact {
    pub fn new(
        evidence_id: StableId,
        subject_id: DigestSha256,
        claim_digest: DigestSha256,
        kind: EvidenceKind,
        artifact_digest: DigestSha256,
        provenance: EvidenceProvenance,
    ) -> Self {
        Self {
            evidence_id,
            subject_id,
            claim_digest,
            kind,
            artifact_digest,
            provenance,
        }
    }

    pub fn evidence_id(&self) -> &StableId {
        &self.evidence_id
    }

    pub fn kind(&self) -> &EvidenceKind {
        &self.kind
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-evidence-v1\n");
        field(&mut out, "evidence-id", self.evidence_id.as_str());
        field(&mut out, "subject", self.subject_id.as_str());
        field(&mut out, "claim", self.claim_digest.as_str());
        field(&mut out, "kind", &self.kind.canonical_name());
        field(&mut out, "artifact", self.artifact_digest.as_str());
        field(&mut out, "producer", self.provenance.producer.as_str());
        field(&mut out, "executor", self.provenance.executor.as_str());
        optional_id(&mut out, "verifier", self.provenance.verifier.as_ref());
        optional_id(&mut out, "signer", self.provenance.signer.as_ref());
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationPlan {
    plan_id: StableId,
    claim_digest: DigestSha256,
    maximum_support: SupportTier,
    invalidation_conditions: BTreeSet<StableId>,
}

impl QualificationPlan {
    pub fn new(
        plan_id: StableId,
        claim_digest: DigestSha256,
        maximum_support: SupportTier,
        invalidation_conditions: BTreeSet<StableId>,
    ) -> Self {
        Self {
            plan_id,
            claim_digest,
            maximum_support,
            invalidation_conditions,
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-plan-v1\n");
        field(&mut out, "plan-id", self.plan_id.as_str());
        field(&mut out, "claim", self.claim_digest.as_str());
        field(
            &mut out,
            "maximum-support",
            self.maximum_support.canonical_name(),
        );
        field(
            &mut out,
            "invalidation-count",
            &self.invalidation_conditions.len().to_string(),
        );
        for condition in &self.invalidation_conditions {
            field(&mut out, "invalidation", condition.as_str());
        }
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationResult {
    claim_digest: DigestSha256,
    subject_id: DigestSha256,
    plan_id: StableId,
    plan_digest: DigestSha256,
    evidence_bindings: Vec<(StableId, DigestSha256)>,
    outcome: QualificationOutcome,
    claim_ceiling: SupportTier,
    invalidation_conditions: BTreeSet<StableId>,
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
        if claim.subject_id() != &subject_id {
            return Err(AssuranceError::SubjectMismatch);
        }
        let claim_digest = claim.digest();
        if plan.claim_digest != claim_digest {
            return Err(AssuranceError::ClaimMismatch);
        }
        if evidence.is_empty() {
            return Err(AssuranceError::EmptyEvidence);
        }

        let mut evidence_bindings = Vec::with_capacity(evidence.len());
        let mut seen_ids = BTreeSet::new();
        for artifact in evidence {
            if artifact.subject_id != subject_id {
                return Err(AssuranceError::SubjectMismatch);
            }
            if artifact.claim_digest != claim_digest {
                return Err(AssuranceError::ClaimMismatch);
            }
            if !seen_ids.insert(artifact.evidence_id.clone()) {
                return Err(AssuranceError::DuplicateEvidenceId(
                    artifact.evidence_id.as_str().to_owned(),
                ));
            }
            evidence_bindings.push((artifact.evidence_id.clone(), artifact.digest()));
        }
        evidence_bindings.sort_by(|left, right| left.0.cmp(&right.0));

        if let QualificationOutcome::Supported(tier) = &proposed_outcome {
            let tier = *tier;
            if tier > plan.maximum_support {
                return Err(AssuranceError::ClaimCeilingExceeded);
            }
            require_evidence_for_tier(tier, evidence)?;
            if tier >= SupportTier::Reproduced
                && !evidence.iter().any(|artifact| {
                    artifact.kind == EvidenceKind::Reproduction
                        && artifact.provenance.has_distinct_verifier_identity()
                })
            {
                return Err(AssuranceError::MissingDistinctVerifier);
            }
            if tier == SupportTier::DeploymentQualified && subject.deployment_envelope().is_none() {
                return Err(AssuranceError::MissingDeploymentEnvelope);
            }
        }

        Ok(Self {
            claim_digest,
            subject_id,
            plan_id: plan.plan_id.clone(),
            plan_digest: plan.digest(),
            evidence_bindings,
            outcome: proposed_outcome,
            claim_ceiling: plan.maximum_support,
            invalidation_conditions: plan.invalidation_conditions.clone(),
        })
    }

    pub fn outcome(&self) -> &QualificationOutcome {
        &self.outcome
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from("symthaea-assurance-result-v1\n");
        field(&mut out, "claim", self.claim_digest.as_str());
        field(&mut out, "subject", self.subject_id.as_str());
        field(&mut out, "plan-id", self.plan_id.as_str());
        field(&mut out, "plan-digest", self.plan_digest.as_str());
        field(
            &mut out,
            "claim-ceiling",
            self.claim_ceiling.canonical_name(),
        );
        append_outcome(&mut out, &self.outcome);
        field(
            &mut out,
            "evidence-count",
            &self.evidence_bindings.len().to_string(),
        );
        for (evidence_id, evidence_digest) in &self.evidence_bindings {
            field(&mut out, "evidence-id", evidence_id.as_str());
            field(&mut out, "evidence-digest", evidence_digest.as_str());
        }
        field(
            &mut out,
            "invalidation-count",
            &self.invalidation_conditions.len().to_string(),
        );
        for condition in &self.invalidation_conditions {
            field(&mut out, "invalidation", condition.as_str());
        }
        out.into_bytes()
    }

    pub fn digest(&self) -> DigestSha256 {
        digest_canonical(&self.canonical_bytes())
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

fn require_evidence_for_tier(
    tier: SupportTier,
    evidence: &[EvidenceArtifact],
) -> Result<(), AssuranceError> {
    let has = |kind: EvidenceKind| evidence.iter().any(|artifact| artifact.kind == kind);
    let enough = match tier {
        SupportTier::Structural => has(EvidenceKind::ArchitectureInspection),
        SupportTier::Observed => has(EvidenceKind::Observation),
        SupportTier::CausallySupported => has(EvidenceKind::ControlledIntervention),
        SupportTier::FunctionallySupported => {
            has(EvidenceKind::ControlledIntervention) && has(EvidenceKind::FunctionalBenchmark)
        }
        SupportTier::Reproduced => {
            has(EvidenceKind::ControlledIntervention)
                && has(EvidenceKind::FunctionalBenchmark)
                && has(EvidenceKind::Reproduction)
        }
        SupportTier::DeploymentQualified => {
            has(EvidenceKind::ControlledIntervention)
                && has(EvidenceKind::FunctionalBenchmark)
                && has(EvidenceKind::Reproduction)
                && has(EvidenceKind::RuntimeReceipt)
        }
    };
    if enough {
        Ok(())
    } else {
        Err(AssuranceError::InsufficientEvidenceForTier(
            tier.canonical_name().to_owned(),
        ))
    }
}

fn append_outcome(out: &mut String, outcome: &QualificationOutcome) {
    match outcome {
        QualificationOutcome::Supported(tier) => {
            field(out, "outcome", "supported");
            field(out, "support-tier", tier.canonical_name());
        }
        QualificationOutcome::Negative(finding) => {
            field(out, "outcome", "negative");
            match finding {
                NegativeFinding::NotDemonstrated => field(out, "finding", "not-demonstrated"),
                NegativeFinding::Contradicted => field(out, "finding", "contradicted"),
                NegativeFinding::Inconclusive => field(out, "finding", "inconclusive"),
                NegativeFinding::Expired => field(out, "finding", "expired"),
                NegativeFinding::Invalidated { reason } => {
                    field(out, "finding", "invalidated");
                    field(out, "reason", reason.as_str());
                }
            }
        }
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

fn optional_id(out: &mut String, label: &str, value: Option<&StableId>) {
    field(out, label, value.map(StableId::as_str).unwrap_or(""));
}

fn field(out: &mut String, label: &str, value: &str) {
    out.push_str(label);
    out.push(' ');
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('\n');
}
