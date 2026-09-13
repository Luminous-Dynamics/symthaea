// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Non-destructive, multi-lineage evidence resolution over a base Butlin report.
//!
//! The historical [`ButlinIndicatorReport`] remains immutable. Independent
//! evidence methods are retained as an ordered lineage chain rather than being
//! collapsed into scalar `live_score` fields or overwriting one another.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::report::{ButlinIndicatorReport, EvidenceOutcome, SupportTier};

pub const BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V1: &str =
    "butlin-resolved-evidence-view-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLineageKindV1 {
    DirectQualification,
    CausalQualification,
}

impl EvidenceLineageKindV1 {
    const fn rank(self) -> u8 {
        match self {
            Self::DirectQualification => 10,
            Self::CausalQualification => 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceArtifactIdentityV1 {
    pub schema: String,
    pub digest_algorithm: String,
    pub digest: String,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAuthorityIdentityV1 {
    pub repository: String,
    pub workflow: String,
    pub workflow_sha: String,
    pub attestation_bundle_sha256: String,
    pub attestation_verification_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageIdentityV1 {
    pub kind: EvidenceLineageKindV1,
    pub method_id: String,
    pub policy_id: Option<String>,
    pub source_commit_sha: String,
    pub source_tree_sha: String,
    pub execution_run_id: String,
    pub toolchain: String,
    pub artifact: EvidenceArtifactIdentityV1,
    pub authority: Option<EvidenceAuthorityIdentityV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndicatorOutcomeOverlayV1 {
    pub indicator_id: String,
    /// Outcome immediately before this lineage is considered.
    pub base_outcome: EvidenceOutcome,
    /// What this evidence lineage itself established.
    pub lineage_outcome: EvidenceOutcome,
    /// Indicator outcome after applying this lineage conservatively.
    /// A negative higher-tier experiment does not erase a valid lower tier.
    pub resolved_outcome: EvidenceOutcome,
    pub lineage: EvidenceLineageIdentityV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EvidenceOutcomeCountsV1 {
    pub architectural_only: usize,
    pub observed: usize,
    pub causally_supported: usize,
    pub functionally_supported: usize,
    pub not_demonstrated: usize,
    pub contradicted: usize,
    pub inconclusive: usize,
}

impl EvidenceOutcomeCountsV1 {
    fn record(&mut self, outcome: EvidenceOutcome) {
        match outcome {
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly) => {
                self.architectural_only += 1;
            }
            EvidenceOutcome::Supported(SupportTier::Observed) => self.observed += 1,
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                self.causally_supported += 1;
            }
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported) => {
                self.functionally_supported += 1;
            }
            EvidenceOutcome::NotDemonstrated => self.not_demonstrated += 1,
            EvidenceOutcome::Contradicted => self.contradicted += 1,
            EvidenceOutcome::Inconclusive => self.inconclusive += 1,
        }
    }

    pub fn total(&self) -> usize {
        self.architectural_only
            + self.observed
            + self.causally_supported
            + self.functionally_supported
            + self.not_demonstrated
            + self.contradicted
            + self.inconclusive
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ButlinResolvedEvidenceViewV1 {
    pub schema: String,
    pub base_report_schema_version: u32,
    pub base_report_blake3: String,
    /// Ordered by indicator ID then evidence-lineage rank. Multiple entries for
    /// one indicator are intentional when stronger evidence builds on weaker.
    pub overlays: Vec<IndicatorOutcomeOverlayV1>,
    pub resolved_counts: EvidenceOutcomeCountsV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceResolutionViewErrorV1 {
    BaseSerialization(String),
    DuplicateBaseIndicator { indicator_id: String },
    UnknownIndicator { indicator_id: String },
    DuplicateLineage {
        indicator_id: String,
        kind: EvidenceLineageKindV1,
    },
    LineageInputMismatch {
        indicator_id: String,
        kind: EvidenceLineageKindV1,
        expected: EvidenceOutcome,
        observed: EvidenceOutcome,
    },
    InvalidLineageIdentity {
        indicator_id: String,
        field: String,
    },
    DirectEvidenceResolutionMismatch { indicator_id: String },
    CausalPrerequisiteNotObserved { observed: EvidenceOutcome },
}

impl std::fmt::Display for EvidenceResolutionViewErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BaseSerialization(error) => write!(f, "failed to serialize base report: {error}"),
            Self::DuplicateBaseIndicator { indicator_id } => {
                write!(f, "base report contains duplicate indicator {indicator_id:?}")
            }
            Self::UnknownIndicator { indicator_id } => {
                write!(f, "overlay targets unknown indicator {indicator_id:?}")
            }
            Self::DuplicateLineage { indicator_id, kind } => write!(
                f,
                "multiple {kind:?} overlays target indicator {indicator_id:?}"
            ),
            Self::LineageInputMismatch {
                indicator_id,
                kind,
                expected,
                observed,
            } => write!(
                f,
                "{kind:?} overlay for {indicator_id:?} expects prior outcome {expected:?}, but the chain currently has {observed:?}"
            ),
            Self::InvalidLineageIdentity { indicator_id, field } => write!(
                f,
                "overlay for {indicator_id:?} has invalid lineage field {field:?}"
            ),
            Self::DirectEvidenceResolutionMismatch { indicator_id } => write!(
                f,
                "direct evidence for {indicator_id:?} carries a cached resolution that does not match recomputation"
            ),
            Self::CausalPrerequisiteNotObserved { observed } => write!(
                f,
                "positive causal GWT-1 promotion requires an independent direct Observed lineage; direct outcome was {observed:?}"
            ),
        }
    }
}

impl std::error::Error for EvidenceResolutionViewErrorV1 {}

fn is_hex_len(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_lower_hex_len(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_lineage(indicator_id: &str, field: &str) -> EvidenceResolutionViewErrorV1 {
    EvidenceResolutionViewErrorV1::InvalidLineageIdentity {
        indicator_id: indicator_id.to_string(),
        field: field.to_string(),
    }
}

fn validate_artifact_identity(
    indicator_id: &str,
    artifact: &EvidenceArtifactIdentityV1,
) -> Result<(), EvidenceResolutionViewErrorV1> {
    if artifact.schema.trim().is_empty() {
        return Err(invalid_lineage(indicator_id, "artifact.schema"));
    }
    if !matches!(artifact.digest_algorithm.as_str(), "blake3" | "sha256") {
        return Err(invalid_lineage(indicator_id, "artifact.digest_algorithm"));
    }
    if !is_lower_hex_len(&artifact.digest, 64) {
        return Err(invalid_lineage(indicator_id, "artifact.digest"));
    }
    if artifact.byte_len == 0 {
        return Err(invalid_lineage(indicator_id, "artifact.byte_len"));
    }
    Ok(())
}

fn validate_authority_identity(
    indicator_id: &str,
    authority: &EvidenceAuthorityIdentityV1,
) -> Result<(), EvidenceResolutionViewErrorV1> {
    if authority.repository.trim().is_empty() {
        return Err(invalid_lineage(indicator_id, "authority.repository"));
    }
    if authority.workflow.trim().is_empty() {
        return Err(invalid_lineage(indicator_id, "authority.workflow"));
    }
    if !is_hex_len(&authority.workflow_sha, 40) {
        return Err(invalid_lineage(indicator_id, "authority.workflow_sha"));
    }
    if !is_lower_hex_len(&authority.attestation_bundle_sha256, 64) {
        return Err(invalid_lineage(
            indicator_id,
            "authority.attestation_bundle_sha256",
        ));
    }
    if !is_lower_hex_len(&authority.attestation_verification_sha256, 64) {
        return Err(invalid_lineage(
            indicator_id,
            "authority.attestation_verification_sha256",
        ));
    }
    Ok(())
}

fn validate_lineage(
    overlay: &IndicatorOutcomeOverlayV1,
) -> Result<(), EvidenceResolutionViewErrorV1> {
    if overlay.lineage.method_id.trim().is_empty() {
        return Err(invalid_lineage(&overlay.indicator_id, "method_id"));
    }
    if overlay
        .lineage
        .policy_id
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(invalid_lineage(&overlay.indicator_id, "policy_id"));
    }
    if !is_hex_len(&overlay.lineage.source_commit_sha, 40) {
        return Err(invalid_lineage(&overlay.indicator_id, "source_commit_sha"));
    }
    if !is_hex_len(&overlay.lineage.source_tree_sha, 40) {
        return Err(invalid_lineage(&overlay.indicator_id, "source_tree_sha"));
    }
    if overlay.lineage.execution_run_id.trim().is_empty() {
        return Err(invalid_lineage(&overlay.indicator_id, "execution_run_id"));
    }
    if overlay.lineage.toolchain.trim().is_empty() {
        return Err(invalid_lineage(&overlay.indicator_id, "toolchain"));
    }
    validate_artifact_identity(&overlay.indicator_id, &overlay.lineage.artifact)?;

    match overlay.lineage.kind {
        EvidenceLineageKindV1::DirectQualification => {
            if overlay.lineage.authority.is_some() {
                return Err(invalid_lineage(&overlay.indicator_id, "authority"));
            }
            match overlay.lineage_outcome {
                EvidenceOutcome::Supported(SupportTier::Observed) => {
                    if overlay.resolved_outcome != EvidenceOutcome::Supported(SupportTier::Observed) {
                        return Err(invalid_lineage(&overlay.indicator_id, "resolved_outcome"));
                    }
                }
                EvidenceOutcome::Supported(_) => {
                    return Err(invalid_lineage(&overlay.indicator_id, "lineage_outcome"));
                }
                EvidenceOutcome::NotDemonstrated
                | EvidenceOutcome::Contradicted
                | EvidenceOutcome::Inconclusive => {
                    if overlay.resolved_outcome != overlay.base_outcome {
                        return Err(invalid_lineage(&overlay.indicator_id, "resolved_outcome"));
                    }
                }
            }
        }
        EvidenceLineageKindV1::CausalQualification => {
            let authority = overlay
                .lineage
                .authority
                .as_ref()
                .ok_or_else(|| invalid_lineage(&overlay.indicator_id, "authority"))?;
            validate_authority_identity(&overlay.indicator_id, authority)?;

            match overlay.lineage_outcome {
                EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                    if overlay.resolved_outcome
                        != EvidenceOutcome::Supported(SupportTier::CausallySupported)
                    {
                        return Err(invalid_lineage(&overlay.indicator_id, "resolved_outcome"));
                    }
                }
                EvidenceOutcome::Supported(_) => {
                    return Err(invalid_lineage(&overlay.indicator_id, "lineage_outcome"));
                }
                EvidenceOutcome::NotDemonstrated
                | EvidenceOutcome::Contradicted
                | EvidenceOutcome::Inconclusive => {
                    if overlay.resolved_outcome != overlay.base_outcome {
                        return Err(invalid_lineage(&overlay.indicator_id, "resolved_outcome"));
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn base_report_blake3_v1(
    report: &ButlinIndicatorReport,
) -> Result<String, EvidenceResolutionViewErrorV1> {
    let bytes = serde_json::to_vec(report)
        .map_err(|error| EvidenceResolutionViewErrorV1::BaseSerialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn resolve_validated_overlays_v1(
    report: &ButlinIndicatorReport,
    mut overlays: Vec<IndicatorOutcomeOverlayV1>,
) -> Result<ButlinResolvedEvidenceViewV1, EvidenceResolutionViewErrorV1> {
    let mut current_outcomes = BTreeMap::new();
    for indicator in &report.indicators {
        if current_outcomes
            .insert(indicator.id.clone(), indicator.outcome)
            .is_some()
        {
            return Err(EvidenceResolutionViewErrorV1::DuplicateBaseIndicator {
                indicator_id: indicator.id.clone(),
            });
        }
    }

    overlays.sort_by(|a, b| {
        a.indicator_id
            .cmp(&b.indicator_id)
            .then_with(|| a.lineage.kind.rank().cmp(&b.lineage.kind.rank()))
    });

    let mut seen_lineages = BTreeSet::new();
    for overlay in &overlays {
        let key = (overlay.indicator_id.clone(), overlay.lineage.kind);
        if !seen_lineages.insert(key) {
            return Err(EvidenceResolutionViewErrorV1::DuplicateLineage {
                indicator_id: overlay.indicator_id.clone(),
                kind: overlay.lineage.kind,
            });
        }

        validate_lineage(overlay)?;
        let observed = current_outcomes
            .get(&overlay.indicator_id)
            .copied()
            .ok_or_else(|| EvidenceResolutionViewErrorV1::UnknownIndicator {
                indicator_id: overlay.indicator_id.clone(),
            })?;
        if observed != overlay.base_outcome {
            return Err(EvidenceResolutionViewErrorV1::LineageInputMismatch {
                indicator_id: overlay.indicator_id.clone(),
                kind: overlay.lineage.kind,
                expected: overlay.base_outcome,
                observed,
            });
        }
        current_outcomes.insert(overlay.indicator_id.clone(), overlay.resolved_outcome);
    }

    let mut resolved_counts = EvidenceOutcomeCountsV1::default();
    for indicator in &report.indicators {
        resolved_counts.record(
            current_outcomes
                .get(&indicator.id)
                .copied()
                .unwrap_or(indicator.outcome),
        );
    }

    Ok(ButlinResolvedEvidenceViewV1 {
        schema: BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V1.to_string(),
        base_report_schema_version: report.schema_version,
        base_report_blake3: base_report_blake3_v1(report)?,
        overlays,
        resolved_counts,
    })
}

#[cfg(feature = "symthaea-backend")]
fn gwt1_direct_overlay_v1(
    report: &ButlinIndicatorReport,
    evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
) -> Result<IndicatorOutcomeOverlayV1, EvidenceResolutionViewErrorV1> {
    use super::gwt1_evidence_envelope::resolve_gwt1_evidence_envelope_v1;
    use super::gwt1_promotion::promote_direct_gwt1_v1;

    let recomputed = resolve_gwt1_evidence_envelope_v1(
        &evidence.envelope,
        &evidence.raw_observation_bytes,
    );
    if recomputed != evidence.resolution {
        return Err(EvidenceResolutionViewErrorV1::DirectEvidenceResolutionMismatch {
            indicator_id: "GWT-1".to_string(),
        });
    }

    let mut matches = report
        .indicators
        .iter()
        .filter(|indicator| indicator.id == "GWT-1");
    let base = matches
        .next()
        .ok_or_else(|| EvidenceResolutionViewErrorV1::UnknownIndicator {
            indicator_id: "GWT-1".to_string(),
        })?;
    if matches.next().is_some() {
        return Err(EvidenceResolutionViewErrorV1::DuplicateBaseIndicator {
            indicator_id: "GWT-1".to_string(),
        });
    }

    let promotion = promote_direct_gwt1_v1(&recomputed);
    let receipt = &evidence.envelope.receipt;
    let raw = &evidence.envelope.raw_observations;
    let lineage_outcome = promotion.evidence_outcome;
    let resolved_outcome = if lineage_outcome == EvidenceOutcome::Supported(SupportTier::Observed) {
        lineage_outcome
    } else {
        base.outcome
    };

    let overlay = IndicatorOutcomeOverlayV1 {
        indicator_id: "GWT-1".to_string(),
        base_outcome: base.outcome,
        lineage_outcome,
        resolved_outcome,
        lineage: EvidenceLineageIdentityV1 {
            kind: EvidenceLineageKindV1::DirectQualification,
            method_id: receipt.schema.clone(),
            policy_id: Some(promotion.policy),
            source_commit_sha: receipt.source_commit_sha.clone(),
            source_tree_sha: receipt.source_tree_sha.clone(),
            execution_run_id: receipt.execution_run_id.clone(),
            toolchain: receipt.toolchain.clone(),
            artifact: EvidenceArtifactIdentityV1 {
                schema: raw.schema.clone(),
                digest_algorithm: "blake3".to_string(),
                digest: raw.blake3.clone(),
                byte_len: raw.byte_len,
            },
            authority: None,
        },
    };
    validate_lineage(&overlay)?;
    Ok(overlay)
}

#[cfg(feature = "symthaea-backend")]
fn gwt1_causal_overlay_v1(
    direct_overlay: &IndicatorOutcomeOverlayV1,
    verified: &super::gwt1_causal_verified_promotion::VerifiedGwt1CausalPromotionV1,
) -> Result<IndicatorOutcomeOverlayV1, EvidenceResolutionViewErrorV1> {
    use super::gwt1_causal_promotion_capsule::GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1;
    use super::gwt1_causal_resolution::Gwt1CausalQualificationOutcomeV1;
    use super::gwt1_causal_verified_promotion::{
        GWT1_CAUSAL_APPROVED_PROMOTION_WORKFLOW_SHA_V1, GWT1_CAUSAL_PROMOTION_WORKFLOW_V1,
    };

    let capsule = verified.capsule();
    if capsule.scientific_outcome == Gwt1CausalQualificationOutcomeV1::Qualified
        && direct_overlay.resolved_outcome != EvidenceOutcome::Supported(SupportTier::Observed)
    {
        return Err(EvidenceResolutionViewErrorV1::CausalPrerequisiteNotObserved {
            observed: direct_overlay.resolved_outcome,
        });
    }

    let lineage_outcome = capsule.eligible_evidence_outcome;
    let resolved_outcome = if lineage_outcome
        == EvidenceOutcome::Supported(SupportTier::CausallySupported)
    {
        lineage_outcome
    } else {
        direct_overlay.resolved_outcome
    };

    let overlay = IndicatorOutcomeOverlayV1 {
        indicator_id: "GWT-1".to_string(),
        base_outcome: direct_overlay.resolved_outcome,
        lineage_outcome,
        resolved_outcome,
        lineage: EvidenceLineageIdentityV1 {
            kind: EvidenceLineageKindV1::CausalQualification,
            method_id: GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1.to_string(),
            policy_id: Some(capsule.policy.clone()),
            source_commit_sha: capsule.evidence_subject.source_commit_sha.clone(),
            source_tree_sha: capsule.evidence_subject.source_tree_sha.clone(),
            execution_run_id: capsule.evidence_subject.execution_run_id.clone(),
            toolchain: capsule.evidence_subject.toolchain.clone(),
            artifact: EvidenceArtifactIdentityV1 {
                schema: capsule.schema.clone(),
                digest_algorithm: "sha256".to_string(),
                digest: verified.capsule_sha256().to_string(),
                byte_len: verified.capsule_byte_len(),
            },
            authority: Some(EvidenceAuthorityIdentityV1 {
                repository: capsule.repository.clone(),
                workflow: GWT1_CAUSAL_PROMOTION_WORKFLOW_V1.to_string(),
                workflow_sha: GWT1_CAUSAL_APPROVED_PROMOTION_WORKFLOW_SHA_V1.to_string(),
                attestation_bundle_sha256: verified
                    .promotion_attestation_bundle_sha256()
                    .to_string(),
                attestation_verification_sha256: verified
                    .promotion_attestation_verification_sha256()
                    .to_string(),
            }),
        },
    };
    validate_lineage(&overlay)?;
    Ok(overlay)
}

#[cfg(feature = "symthaea-backend")]
pub fn resolve_gwt1_evidence_view_v1(
    report: &ButlinIndicatorReport,
    evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
) -> Result<ButlinResolvedEvidenceViewV1, EvidenceResolutionViewErrorV1> {
    let overlay = gwt1_direct_overlay_v1(report, evidence)?;
    resolve_validated_overlays_v1(report, vec![overlay])
}

/// Resolve the complete GWT-1 V1 lineage without mutating the base report.
///
/// Positive causal support requires both an independently recomputed direct
/// `Observed` result and an opaque cryptographically verified causal promotion
/// token. Negative higher-tier results remain visible in `lineage_outcome` but
/// do not erase independently established lower-tier support.
#[cfg(feature = "symthaea-backend")]
pub fn resolve_gwt1_causal_evidence_view_v1(
    report: &ButlinIndicatorReport,
    direct_evidence: &super::gwt1_end_to_end::Gwt1EndToEndEvidenceV1,
    verified_causal: &super::gwt1_causal_verified_promotion::VerifiedGwt1CausalPromotionV1,
) -> Result<ButlinResolvedEvidenceViewV1, EvidenceResolutionViewErrorV1> {
    let direct = gwt1_direct_overlay_v1(report, direct_evidence)?;
    let causal = gwt1_causal_overlay_v1(&direct, verified_causal)?;
    resolve_validated_overlays_v1(report, vec![direct, causal])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::report::{EvidenceAnnotation, IndicatorEvidence};

    fn indicator(id: &str, outcome: EvidenceOutcome) -> IndicatorEvidence {
        IndicatorEvidence {
            id: id.to_string(),
            theory: "test theory".to_string(),
            description: "test indicator".to_string(),
            outcome,
            evidence: "test evidence".to_string(),
            architectural_score: 0.5,
            live_score: Some(0.5),
            probe_quality: None,
            causal_effect: None,
            functional_effect: None,
            annotations: Vec::<EvidenceAnnotation>::new(),
        }
    }

    fn report() -> ButlinIndicatorReport {
        ButlinIndicatorReport::from_indicators(vec![
            indicator(
                "GWT-1",
                EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            ),
            indicator("AE-1", EvidenceOutcome::NotDemonstrated),
        ])
    }

    fn artifact(algorithm: &str) -> EvidenceArtifactIdentityV1 {
        EvidenceArtifactIdentityV1 {
            schema: "artifact-v1".to_string(),
            digest_algorithm: algorithm.to_string(),
            digest: "a".repeat(64),
            byte_len: 128,
        }
    }

    fn overlay(
        kind: EvidenceLineageKindV1,
        base_outcome: EvidenceOutcome,
        lineage_outcome: EvidenceOutcome,
        resolved_outcome: EvidenceOutcome,
    ) -> IndicatorOutcomeOverlayV1 {
        IndicatorOutcomeOverlayV1 {
            indicator_id: "GWT-1".to_string(),
            base_outcome,
            lineage_outcome,
            resolved_outcome,
            lineage: EvidenceLineageIdentityV1 {
                kind,
                method_id: "method-v1".to_string(),
                policy_id: Some("policy-v1".to_string()),
                source_commit_sha: "b".repeat(40),
                source_tree_sha: "c".repeat(40),
                execution_run_id: "run-1".to_string(),
                toolchain: "rustc test".to_string(),
                artifact: artifact(if kind == EvidenceLineageKindV1::DirectQualification {
                    "blake3"
                } else {
                    "sha256"
                }),
                authority: if kind == EvidenceLineageKindV1::CausalQualification {
                    Some(EvidenceAuthorityIdentityV1 {
                        repository: "Luminous-Dynamics/symthaea".to_string(),
                        workflow: ".github/workflows/trusted.yml".to_string(),
                        workflow_sha: "d".repeat(40),
                        attestation_bundle_sha256: "e".repeat(64),
                        attestation_verification_sha256: "f".repeat(64),
                    })
                } else {
                    None
                },
            },
        }
    }

    #[test]
    fn positive_causal_chain_preserves_both_lineages() {
        let base = report();
        let original = base.clone();
        let direct = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let causal = overlay(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
        );
        let view = resolve_validated_overlays_v1(&base, vec![causal, direct])
            .expect("resolved chain");

        assert_eq!(base, original);
        assert_eq!(view.overlays.len(), 2);
        assert_eq!(view.overlays[0].lineage.kind, EvidenceLineageKindV1::DirectQualification);
        assert_eq!(view.overlays[1].lineage.kind, EvidenceLineageKindV1::CausalQualification);
        assert_eq!(view.resolved_counts.causally_supported, 1);
        assert_eq!(view.resolved_counts.total(), base.indicators.len());
    }

    #[test]
    fn direct_null_does_not_erase_architectural_support() {
        let direct = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
        );
        let view = resolve_validated_overlays_v1(&report(), vec![direct])
            .expect("resolved direct null");
        assert_eq!(view.resolved_counts.architectural_only, 1);
        assert_eq!(view.overlays[0].lineage_outcome, EvidenceOutcome::NotDemonstrated);
    }

    #[test]
    fn causal_null_does_not_erase_direct_observed() {
        let direct = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let causal = overlay(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::NotDemonstrated,
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let view = resolve_validated_overlays_v1(&report(), vec![direct, causal])
            .expect("resolved causal null");
        assert_eq!(view.resolved_counts.observed, 1);
        assert_eq!(view.overlays[1].lineage_outcome, EvidenceOutcome::NotDemonstrated);
    }

    #[test]
    fn direct_negative_cannot_overwrite_lower_tier() {
        let item = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Contradicted,
            EvidenceOutcome::Contradicted,
        );
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![item]),
            Err(EvidenceResolutionViewErrorV1::InvalidLineageIdentity { .. })
        ));
    }

    #[test]
    fn causal_negative_cannot_overwrite_lower_tier() {
        let direct = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        let causal = overlay(
            EvidenceLineageKindV1::CausalQualification,
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Contradicted,
            EvidenceOutcome::Contradicted,
        );
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![direct, causal]),
            Err(EvidenceResolutionViewErrorV1::InvalidLineageIdentity { .. })
        ));
    }

    #[test]
    fn direct_qualification_cannot_claim_causal_support() {
        let item = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
        );
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![item]),
            Err(EvidenceResolutionViewErrorV1::InvalidLineageIdentity { .. })
        ));
    }

    #[test]
    fn duplicate_same_lineage_is_rejected() {
        let item = overlay(
            EvidenceLineageKindV1::DirectQualification,
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly),
            EvidenceOutcome::Supported(SupportTier::Observed),
            EvidenceOutcome::Supported(SupportTier::Observed),
        );
        assert!(matches!(
            resolve_validated_overlays_v1(&report(), vec![item.clone(), item]),
            Err(EvidenceResolutionViewErrorV1::DuplicateLineage { .. })
        ));
    }

    #[test]
    fn base_report_digest_is_deterministic() {
        let base = report();
        assert_eq!(
            base_report_blake3_v1(&base).unwrap(),
            base_report_blake3_v1(&base).unwrap()
        );
    }
}

#[cfg(all(test, feature = "symthaea-backend"))]
mod backend_tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::benchmarks::butlin::gwt1_causal_verified_promotion::verified_gwt1_causal_promotion_for_test;
    use crate::benchmarks::butlin::{
        GWT1_CAUSAL_APPROVED_BUILDER_SHA_V1, GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1,
        GWT1_CAUSAL_PROMOTION_POLICY_V1, GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1,
        GWT1_CAUSAL_TRUSTED_REPOSITORY_V1, Gwt1CausalPromotionCapsuleV1,
        Gwt1CausalQualificationOutcomeV1, Gwt1ExecutionIdentityV1,
    };

    fn direct_overlay(lineage_outcome: EvidenceOutcome) -> IndicatorOutcomeOverlayV1 {
        let base_outcome = EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly);
        let resolved_outcome = if lineage_outcome == EvidenceOutcome::Supported(SupportTier::Observed) {
            lineage_outcome
        } else {
            base_outcome
        };
        IndicatorOutcomeOverlayV1 {
            indicator_id: "GWT-1".to_string(),
            base_outcome,
            lineage_outcome,
            resolved_outcome,
            lineage: EvidenceLineageIdentityV1 {
                kind: EvidenceLineageKindV1::DirectQualification,
                method_id: "direct-v1".to_string(),
                policy_id: Some("direct-policy-v1".to_string()),
                source_commit_sha: "a".repeat(40),
                source_tree_sha: "b".repeat(40),
                execution_run_id: "run-1".to_string(),
                toolchain: "rustc test".to_string(),
                artifact: EvidenceArtifactIdentityV1 {
                    schema: "raw-v1".to_string(),
                    digest_algorithm: "blake3".to_string(),
                    digest: "c".repeat(64),
                    byte_len: 128,
                },
                authority: None,
            },
        }
    }

    fn capsule(outcome: Gwt1CausalQualificationOutcomeV1) -> Gwt1CausalPromotionCapsuleV1 {
        let eligibility = match outcome {
            Gwt1CausalQualificationOutcomeV1::Qualified => {
                EvidenceOutcome::Supported(SupportTier::CausallySupported)
            }
            Gwt1CausalQualificationOutcomeV1::NotDemonstrated => EvidenceOutcome::NotDemonstrated,
            Gwt1CausalQualificationOutcomeV1::Contradicted => EvidenceOutcome::Contradicted,
            Gwt1CausalQualificationOutcomeV1::Inconclusive => EvidenceOutcome::Inconclusive,
        };
        Gwt1CausalPromotionCapsuleV1 {
            schema: GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1.to_string(),
            policy: GWT1_CAUSAL_PROMOTION_POLICY_V1.to_string(),
            indicator_id: "GWT-1".to_string(),
            repository: GWT1_CAUSAL_TRUSTED_REPOSITORY_V1.to_string(),
            trusted_builder_workflow: GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1.to_string(),
            trusted_builder_sha: GWT1_CAUSAL_APPROVED_BUILDER_SHA_V1.to_string(),
            trusted_builder_ref: "trusted-ref".to_string(),
            causal_archive_sha256: "1".repeat(64),
            archive_attestation_bundle_sha256: "2".repeat(64),
            archive_attestation_verification_sha256: "3".repeat(64),
            evidence_subject: Gwt1ExecutionIdentityV1 {
                source_commit_sha: "4".repeat(40),
                source_tree_sha: "5".repeat(40),
                execution_run_id: "123/1".to_string(),
                toolchain: "rustc test".to_string(),
                specialist_blob_shas: BTreeMap::from([
                    ("drive_manager".to_string(), "6".repeat(40)),
                    ("memory_manager".to_string(), "7".repeat(40)),
                    ("learning_manager".to_string(), "8".repeat(40)),
                    ("perception_manager".to_string(), "9".repeat(40)),
                ]),
            },
            scientific_outcome: outcome,
            eligible_evidence_outcome: eligibility,
            tier_ceiling: SupportTier::CausallySupported,
        }
    }

    #[test]
    fn positive_causal_overlay_requires_direct_observed() {
        let direct = direct_overlay(EvidenceOutcome::NotDemonstrated);
        let verified = verified_gwt1_causal_promotion_for_test(capsule(
            Gwt1CausalQualificationOutcomeV1::Qualified,
        ));
        assert!(matches!(
            gwt1_causal_overlay_v1(&direct, &verified),
            Err(EvidenceResolutionViewErrorV1::CausalPrerequisiteNotObserved { .. })
        ));
    }

    #[test]
    fn causal_null_preserves_observed_but_records_null_lineage() {
        let direct = direct_overlay(EvidenceOutcome::Supported(SupportTier::Observed));
        let verified = verified_gwt1_causal_promotion_for_test(capsule(
            Gwt1CausalQualificationOutcomeV1::NotDemonstrated,
        ));
        let causal = gwt1_causal_overlay_v1(&direct, &verified).expect("causal overlay");
        assert_eq!(causal.lineage_outcome, EvidenceOutcome::NotDemonstrated);
        assert_eq!(
            causal.resolved_outcome,
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
    }
}
