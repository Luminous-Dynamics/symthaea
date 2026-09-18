// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative one-way projection into `symthaea-science-research`.
//!
//! Projection can preserve or lower authority. It never mints shared-science
//! qualification, replication, or independence authority from an aesthetic
//! domain release outcome.

use crate::{
    CausalAuthority, ScientificClaim as AestheticClaim, ScientificClaimKind,
    ScientificValidationOutcome, ScientificValidationReport,
};
use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_science_research::{
    AuthorityFacet, AuthorityLevel, AuthorityProfile, EvidenceKind, EvidenceRecord, EvidenceState,
    FrozenScientificSubject, ResearchId, ScientificClaim, ScientificSubject, Sha256Digest,
    SubjectKind,
};

#[derive(Debug, Clone, Serialize)]
pub struct ProjectedAestheticClaim {
    pub source_claim_id: String,
    pub source_claim_kind: ScientificClaimKind,
    pub subject: FrozenScientificSubject,
    pub evidence: EvidenceRecord,
    pub claim: ScientificClaim,
}

#[derive(Debug, Clone, Serialize)]
pub struct AestheticScienceProjection {
    pub source_report_sha256: Sha256Digest,
    pub source_outcome: ScientificValidationOutcome,
    pub claims: Vec<ProjectedAestheticClaim>,
}

#[derive(Debug)]
pub enum ProjectionError {
    SourceValidation(String),
    Serialization(serde_json::Error),
    CoreIdentity(String),
    CoreSubject(String),
    CoreEvidence(String),
    CoreClaim(String),
    MissingMetricEvidence(String),
    MissingCausalEvidence(String),
    MissingStudyQuality(String),
    MissingSemanticEvidence(String, String),
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceValidation(detail) => write!(formatter, "source validation failed: {detail}"),
            Self::Serialization(error) => write!(formatter, "projection serialization failed: {error}"),
            Self::CoreIdentity(detail) => write!(formatter, "shared-science identity failed: {detail}"),
            Self::CoreSubject(detail) => write!(formatter, "shared-science subject failed: {detail}"),
            Self::CoreEvidence(detail) => write!(formatter, "shared-science evidence failed: {detail}"),
            Self::CoreClaim(detail) => write!(formatter, "shared-science claim failed: {detail}"),
            Self::MissingMetricEvidence(metric) => write!(formatter, "claim references missing metric evidence {metric}"),
            Self::MissingCausalEvidence(study) => write!(formatter, "claim references missing causal evidence {study}"),
            Self::MissingStudyQuality(study) => write!(formatter, "claim references missing study-quality evidence {study}"),
            Self::MissingSemanticEvidence(baseline, candidate) => write!(
                formatter,
                "claim references missing semantic comparison {baseline} -> {candidate}"
            ),
        }
    }
}

impl std::error::Error for ProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(error) => Some(error),
            _ => None,
        }
    }
}

pub fn project_scientific_validation(
    report: &ScientificValidationReport,
) -> Result<AestheticScienceProjection, ProjectionError> {
    report
        .validate()
        .map_err(|error| ProjectionError::SourceValidation(error.to_string()))?;

    let source_report_sha256 = sha_json(report)?;
    let mut projected = Vec::with_capacity(report.bundle.claims.len());
    for source_claim in &report.bundle.claims {
        projected.push(project_claim(report, source_claim)?);
    }

    Ok(AestheticScienceProjection {
        source_report_sha256,
        source_outcome: report.outcome,
        claims: projected,
    })
}

fn project_claim(
    report: &ScientificValidationReport,
    source_claim: &AestheticClaim,
) -> Result<ProjectedAestheticClaim, ProjectionError> {
    let claim_bytes = serde_json::to_vec(source_claim).map_err(ProjectionError::Serialization)?;
    let content_sha256 = Sha256Digest::of_bytes(&claim_bytes);
    let subject_id = derived_id("AESTH-SUBJECT", &content_sha256)?;
    let subject = ScientificSubject {
        subject_id,
        kind: subject_kind(source_claim.kind),
        content_sha256,
        source_revision: format!(
            "aesthetic-release:{}:bundle:{}:report:{}",
            report.bundle.release_id, report.bundle.bundle_id, report.report_id
        ),
    }
    .freeze()
    .map_err(|issues| ProjectionError::CoreSubject(format!("{issues:?}")))?;

    let provenance_roots = provenance_roots(report, source_claim)?;
    let artifact_sha256 = sha_json(&(report, source_claim))?;
    let evidence_id = derived_id("AESTH-EVIDENCE", &artifact_sha256)?;
    let state = evidence_state(report.outcome);
    let kind = evidence_kind(source_claim.kind);
    let authority = if state == EvidenceState::Pass {
        bounded_authority(report, source_claim)
    } else {
        AuthorityProfile::empty()
    };
    let evidence = EvidenceRecord::new(
        evidence_id,
        subject.subject_sha256().clone(),
        kind,
        state,
        artifact_sha256,
        provenance_roots,
        None,
        authority,
    )
    .map_err(|issues| ProjectionError::CoreEvidence(format!("{issues:?}")))?;

    let claim_id = derived_id("AESTH-CLAIM", subject.subject_sha256())?;
    let claim = ScientificClaim::from_evidence(
        claim_id,
        subject.subject_sha256().clone(),
        std::slice::from_ref(&evidence),
    )
    .map_err(|issues| ProjectionError::CoreClaim(format!("{issues:?}")))?;

    Ok(ProjectedAestheticClaim {
        source_claim_id: source_claim.claim_id.clone(),
        source_claim_kind: source_claim.kind,
        subject,
        evidence,
        claim,
    })
}

fn provenance_roots(
    report: &ScientificValidationReport,
    claim: &AestheticClaim,
) -> Result<BTreeSet<Sha256Digest>, ProjectionError> {
    let mut roots = BTreeSet::new();

    let metric = report
        .bundle
        .metric_reports
        .iter()
        .find(|candidate| candidate.metric.metric_id == claim.metric_id)
        .ok_or_else(|| ProjectionError::MissingMetricEvidence(claim.metric_id.clone()))?;
    roots.insert(sha_json(metric)?);

    if let Some(study_id) = claim.causal_study_id.as_deref() {
        let causal = report
            .bundle
            .causal_reports
            .iter()
            .find(|candidate| candidate.study.study_id == study_id)
            .ok_or_else(|| ProjectionError::MissingCausalEvidence(study_id.to_owned()))?;
        roots.insert(sha_json(causal)?);

        let quality = report
            .bundle
            .study_quality_reports
            .iter()
            .find(|candidate| candidate.execution.study_id == study_id)
            .ok_or_else(|| ProjectionError::MissingStudyQuality(study_id.to_owned()))?;
        roots.insert(sha_json(quality)?);
    }

    if let (Some(baseline), Some(candidate)) = (
        claim.semantic_baseline_set_id.as_deref(),
        claim.semantic_candidate_set_id.as_deref(),
    ) {
        let semantic = report
            .bundle
            .semantic_drift_reports
            .iter()
            .find(|semantic| {
                semantic.baseline.set_id == baseline && semantic.candidate.set_id == candidate
            })
            .ok_or_else(|| {
                ProjectionError::MissingSemanticEvidence(
                    baseline.to_owned(),
                    candidate.to_owned(),
                )
            })?;
        roots.insert(sha_json(semantic)?);
    }

    Ok(roots)
}

fn bounded_authority(
    report: &ScientificValidationReport,
    claim: &AestheticClaim,
) -> AuthorityProfile {
    use AuthorityFacet as F;
    use AuthorityLevel as L;

    let mut authority = AuthorityProfile::empty().with(F::Provenance, L::Bound);
    match claim.kind {
        ScientificClaimKind::CausalBenefit => {
            authority = authority
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound)
                .with(F::Causal, L::Bound);
        }
        ScientificClaimKind::QuasiExperimentalBenefit => {
            authority = authority
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound)
                .with(F::Causal, L::Declared);
        }
        ScientificClaimKind::AssociationalSignal => {
            authority = authority
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound);
        }
        ScientificClaimKind::CrossContextComparability => {
            authority = authority
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound);
        }
        ScientificClaimKind::OperationalMetricUse => {
            authority = authority.with(F::Empirical, L::Bound);
        }
    }

    // A `Ready` source report may contain stronger causal evidence than the
    // source claim asks to assert. Never upgrade a quasi/associational claim
    // merely because a referenced study has stronger authority.
    if matches!(claim.kind, ScientificClaimKind::CausalBenefit) {
        let causal_is_causal = claim.causal_study_id.as_deref().is_some_and(|study_id| {
            report
                .bundle
                .causal_reports
                .iter()
                .any(|candidate| {
                    candidate.study.study_id == study_id
                        && candidate.authority == CausalAuthority::Causal
                })
        });
        if !causal_is_causal {
            authority = authority.with(F::Causal, L::None);
        }
    }
    authority
}

const fn evidence_state(outcome: ScientificValidationOutcome) -> EvidenceState {
    match outcome {
        ScientificValidationOutcome::Ready => EvidenceState::Pass,
        ScientificValidationOutcome::HumanReview => EvidenceState::Incomplete,
        ScientificValidationOutcome::Blocked => EvidenceState::Fail,
    }
}

const fn evidence_kind(kind: ScientificClaimKind) -> EvidenceKind {
    match kind {
        ScientificClaimKind::CausalBenefit
        | ScientificClaimKind::QuasiExperimentalBenefit
        | ScientificClaimKind::AssociationalSignal => EvidenceKind::CausalStudy,
        ScientificClaimKind::CrossContextComparability => EvidenceKind::RobustnessTest,
        ScientificClaimKind::OperationalMetricUse => EvidenceKind::DerivedMeasurement,
    }
}

const fn subject_kind(kind: ScientificClaimKind) -> SubjectKind {
    match kind {
        ScientificClaimKind::CausalBenefit | ScientificClaimKind::QuasiExperimentalBenefit => {
            SubjectKind::CausalHypothesis
        }
        ScientificClaimKind::AssociationalSignal
        | ScientificClaimKind::CrossContextComparability
        | ScientificClaimKind::OperationalMetricUse => SubjectKind::EmpiricalHypothesis,
    }
}

fn sha_json<T: Serialize>(value: &T) -> Result<Sha256Digest, ProjectionError> {
    let bytes = serde_json::to_vec(value).map_err(ProjectionError::Serialization)?;
    Ok(Sha256Digest::of_bytes(&bytes))
}

fn derived_id(prefix: &str, digest: &Sha256Digest) -> Result<ResearchId, ProjectionError> {
    ResearchId::parse(format!("{prefix}:{}", &digest.as_str()[..24]))
        .map_err(|error| ProjectionError::CoreIdentity(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quasi_claim_is_not_promoted_to_bound_causal_authority() {
        let mut profile = AuthorityProfile::empty()
            .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
            .with(AuthorityFacet::Execution, AuthorityLevel::Bound)
            .with(AuthorityFacet::Empirical, AuthorityLevel::Bound)
            .with(AuthorityFacet::Causal, AuthorityLevel::Declared);
        profile = profile.bounded_by(&EvidenceKind::CausalStudy.maximum_record_authority());
        assert_eq!(
            profile.get(AuthorityFacet::Causal),
            AuthorityLevel::Declared
        );
    }

    #[test]
    fn blocked_and_review_outcomes_cannot_carry_positive_authority() {
        assert_eq!(
            evidence_state(ScientificValidationOutcome::HumanReview),
            EvidenceState::Incomplete
        );
        assert_eq!(
            evidence_state(ScientificValidationOutcome::Blocked),
            EvidenceState::Fail
        );
    }

    #[test]
    fn compatibility_digest_and_crypto_digest_are_separate_namespaces() {
        let compatibility = crate::digest_bytes(b"same-bytes");
        let cryptographic = Sha256Digest::of_bytes(b"same-bytes");
        assert_eq!(compatibility.len(), 16);
        assert_eq!(cryptographic.as_str().len(), 64);
        assert_ne!(compatibility, cryptographic.as_str());
    }
}
