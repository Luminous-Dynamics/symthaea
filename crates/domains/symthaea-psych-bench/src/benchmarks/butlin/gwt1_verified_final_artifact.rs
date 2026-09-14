// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consumer-side authority for a fully reconstructed GWT-1 final artifact.
//!
//! This module deliberately does **not** verify GitHub attestations or admit tar
//! archives by itself. Those cryptographic/archive roots are supplied by the
//! separately frozen consumer verifier. Its responsibility is the final pure
//! theorem: persisted report bytes are authoritative only when they agree with
//! an independent reconstruction from the preserved direct and promotion source
//! material.
//!
//! The resulting [`VerifiedGwt1ResolvedArtifactV1`] is opaque and deliberately
//! non-serializable. There is no public constructor and no boolean-style
//! `verified` shortcut.

use super::gwt1_evidence_disposition::{
    Gwt1EvidenceDispositionErrorV1, Gwt1EvidenceDispositionSummaryV1,
    classify_gwt1_evidence_disposition_v1,
};
use super::gwt1_trusted_resolution::{
    GWT1_TRUSTED_RESOLUTION_CANDIDATE_SCHEMA_V1, Gwt1TrustedResolutionCandidateV1,
};
use super::report::{ButlinIndicatorReport, EvidenceOutcome, SupportTier};
use super::resolution_view_v2::ButlinResolvedEvidenceViewV2;

pub const GWT1_VERIFIED_FINAL_ARTIFACT_SCHEMA_V1: &str = "butlin-gwt1-verified-final-artifact-v1";

/// Opaque read-only authority token for one reconstructed final GWT-1 artifact.
///
/// Deliberately implements neither `Clone` nor serde traits. Authority is the
/// one verified in-memory capability produced by the trusted consumer path;
/// persisted authority remains the signed final archive and its verification
/// material.
#[derive(Debug)]
pub struct VerifiedGwt1ResolvedArtifactV1 {
    schema: &'static str,
    final_archive_sha256: String,
    base_report: ButlinIndicatorReport,
    resolved_view: ButlinResolvedEvidenceViewV2,
    disposition: Gwt1EvidenceDispositionSummaryV1,
}

impl VerifiedGwt1ResolvedArtifactV1 {
    pub fn schema(&self) -> &'static str {
        self.schema
    }

    pub fn final_archive_sha256(&self) -> &str {
        &self.final_archive_sha256
    }

    pub fn base_report(&self) -> &ButlinIndicatorReport {
        &self.base_report
    }

    pub fn resolved_view(&self) -> &ButlinResolvedEvidenceViewV2 {
        &self.resolved_view
    }

    pub fn disposition(&self) -> &Gwt1EvidenceDispositionSummaryV1 {
        &self.disposition
    }

    pub fn resolved_gwt1_outcome(&self) -> EvidenceOutcome {
        self.disposition.resolved_outcome
    }

    pub fn has_causal_contradiction(&self) -> bool {
        self.disposition.has_causal_contradiction
    }

    pub fn causal_follow_up_required(&self) -> bool {
        self.disposition.causal_follow_up_required
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gwt1FinalArtifactReconstructionErrorV1 {
    MalformedFinalArchiveSha256 { observed: String },
    WrongStoredCandidateSchema { observed: String },
    WrongRecomputedCandidateSchema { observed: String },
    EmptyRecomputedPromotionVerification,
    StoredCandidateBaseMismatch,
    StoredCandidateViewMismatch,
    StoredCandidateDispositionMismatch,
    RecomputedBaseMismatch,
    RecomputedViewMismatch,
    RecomputedDispositionMismatch,
    InternalPromotionVerificationMismatch,
    StoredDispositionInvalid(Gwt1EvidenceDispositionErrorV1),
    FunctionallySupportedForbidden,
}

impl std::fmt::Display for Gwt1FinalArtifactReconstructionErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedFinalArchiveSha256 { observed } => {
                write!(f, "malformed final archive SHA-256 {observed:?}")
            }
            Self::WrongStoredCandidateSchema { observed } => write!(
                f,
                "stored final-resolution candidate has unexpected schema {observed:?}"
            ),
            Self::WrongRecomputedCandidateSchema { observed } => write!(
                f,
                "recomputed final-resolution candidate has unexpected schema {observed:?}"
            ),
            Self::EmptyRecomputedPromotionVerification => write!(
                f,
                "independent reconstruction did not retain its inner promotion-verification transcript"
            ),
            Self::StoredCandidateBaseMismatch => {
                write!(
                    f,
                    "stored resolution candidate disagrees with stored base report"
                )
            }
            Self::StoredCandidateViewMismatch => {
                write!(
                    f,
                    "stored resolution candidate disagrees with stored V2 view"
                )
            }
            Self::StoredCandidateDispositionMismatch => write!(
                f,
                "stored resolution candidate disagrees with stored GWT-1 disposition"
            ),
            Self::RecomputedBaseMismatch => {
                write!(
                    f,
                    "stored base report differs from independent reconstruction"
                )
            }
            Self::RecomputedViewMismatch => {
                write!(
                    f,
                    "stored V2 resolved view differs from independent reconstruction"
                )
            }
            Self::RecomputedDispositionMismatch => write!(
                f,
                "stored GWT-1 disposition differs from independent reconstruction"
            ),
            Self::InternalPromotionVerificationMismatch => write!(
                f,
                "stored inner promotion-verification transcript differs from the reconstruction transcript"
            ),
            Self::StoredDispositionInvalid(error) => {
                write!(
                    f,
                    "stored GWT-1 disposition is not derivable from its V2 view: {error}"
                )
            }
            Self::FunctionallySupportedForbidden => write!(
                f,
                "GWT-1 final-artifact consumer forbids FunctionallySupported in this V1 authority lane"
            ),
        }
    }
}

impl std::error::Error for Gwt1FinalArtifactReconstructionErrorV1 {}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn gwt1_claims_functional_support(view: &ButlinResolvedEvidenceViewV2) -> bool {
    view.lineages.iter().any(|lineage| {
        lineage.indicator_id == "GWT-1"
            && (lineage.lineage_outcome
                == EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
                || lineage.resolved_outcome
                    == EvidenceOutcome::Supported(SupportTier::FunctionallySupported))
    })
}

/// Complete the pure reconstruction theorem after cryptographic verification,
/// bounded final-archive admission, and source replay have succeeded.
///
/// Visible only to sibling modules inside the Butlin authority implementation.
/// The trusted final-consumer adapter is the sole non-test caller and can invoke
/// it only after the workflow has established the external authority roots.
pub(super) fn verify_gwt1_final_reconstruction_v1(
    final_archive_sha256: &str,
    stored_base_report: &ButlinIndicatorReport,
    stored_resolved_view: &ButlinResolvedEvidenceViewV2,
    stored_disposition: &Gwt1EvidenceDispositionSummaryV1,
    stored_candidate: &Gwt1TrustedResolutionCandidateV1,
    stored_internal_promotion_verification: &[u8],
    recomputed_candidate: &Gwt1TrustedResolutionCandidateV1,
) -> Result<VerifiedGwt1ResolvedArtifactV1, Gwt1FinalArtifactReconstructionErrorV1> {
    if !is_lower_hex_64(final_archive_sha256) {
        return Err(
            Gwt1FinalArtifactReconstructionErrorV1::MalformedFinalArchiveSha256 {
                observed: final_archive_sha256.to_string(),
            },
        );
    }

    if stored_candidate.schema != GWT1_TRUSTED_RESOLUTION_CANDIDATE_SCHEMA_V1 {
        return Err(
            Gwt1FinalArtifactReconstructionErrorV1::WrongStoredCandidateSchema {
                observed: stored_candidate.schema.clone(),
            },
        );
    }
    if recomputed_candidate.schema != GWT1_TRUSTED_RESOLUTION_CANDIDATE_SCHEMA_V1 {
        return Err(
            Gwt1FinalArtifactReconstructionErrorV1::WrongRecomputedCandidateSchema {
                observed: recomputed_candidate.schema.clone(),
            },
        );
    }
    if recomputed_candidate
        .promotion_attestation_verification_bytes()
        .is_empty()
    {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::EmptyRecomputedPromotionVerification);
    }

    if &stored_candidate.base_report != stored_base_report {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::StoredCandidateBaseMismatch);
    }
    if &stored_candidate.resolved_view != stored_resolved_view {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::StoredCandidateViewMismatch);
    }
    if &stored_candidate.disposition != stored_disposition {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::StoredCandidateDispositionMismatch);
    }

    let rederived_disposition = classify_gwt1_evidence_disposition_v1(stored_resolved_view)
        .map_err(Gwt1FinalArtifactReconstructionErrorV1::StoredDispositionInvalid)?;
    if &rederived_disposition != stored_disposition {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::StoredCandidateDispositionMismatch);
    }

    if &recomputed_candidate.base_report != stored_base_report {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::RecomputedBaseMismatch);
    }
    if &recomputed_candidate.resolved_view != stored_resolved_view {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::RecomputedViewMismatch);
    }
    if &recomputed_candidate.disposition != stored_disposition {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::RecomputedDispositionMismatch);
    }
    if recomputed_candidate.promotion_attestation_verification_bytes()
        != stored_internal_promotion_verification
    {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::InternalPromotionVerificationMismatch);
    }

    if gwt1_claims_functional_support(stored_resolved_view)
        || stored_disposition.resolved_outcome
            == EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
    {
        return Err(Gwt1FinalArtifactReconstructionErrorV1::FunctionallySupportedForbidden);
    }

    Ok(VerifiedGwt1ResolvedArtifactV1 {
        schema: GWT1_VERIFIED_FINAL_ARTIFACT_SCHEMA_V1,
        final_archive_sha256: final_archive_sha256.to_string(),
        base_report: stored_base_report.clone(),
        resolved_view: stored_resolved_view.clone(),
        disposition: stored_disposition.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::ButlinIndicatorSuite;
    use crate::benchmarks::butlin::gwt1_trusted_resolution::trusted_resolution_candidate_for_test;
    use crate::benchmarks::butlin::resolution_view::{
        EvidenceArtifactIdentityV1, EvidenceLineageIdentityV1, EvidenceLineageKindV1,
        EvidenceOutcomeCountsV1, base_report_blake3_v1,
    };
    use crate::benchmarks::butlin::resolution_view_v2::{
        BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2, IndicatorEvidenceLineageV2,
    };
    use crate::harness::BenchmarkConfig;

    fn fixture() -> (
        ButlinIndicatorReport,
        ButlinResolvedEvidenceViewV2,
        Gwt1EvidenceDispositionSummaryV1,
        Gwt1TrustedResolutionCandidateV1,
        Gwt1TrustedResolutionCandidateV1,
        Vec<u8>,
    ) {
        let base_report = ButlinIndicatorSuite::evaluate(&BenchmarkConfig::default());
        let base_outcome = base_report
            .indicators
            .iter()
            .find(|indicator| indicator.id == "GWT-1")
            .expect("GWT-1 base indicator")
            .outcome;
        let direct = IndicatorEvidenceLineageV2 {
            indicator_id: "GWT-1".to_string(),
            base_outcome,
            lineage_outcome: EvidenceOutcome::Supported(SupportTier::Observed),
            resolved_outcome: EvidenceOutcome::Supported(SupportTier::Observed),
            lineage: EvidenceLineageIdentityV1 {
                kind: EvidenceLineageKindV1::DirectQualification,
                method_id: "fixture-direct-v1".to_string(),
                policy_id: Some("fixture-policy-v1".to_string()),
                source_commit_sha: "a".repeat(40),
                source_tree_sha: "b".repeat(40),
                execution_run_id: "123/1".to_string(),
                toolchain: "rustc fixture".to_string(),
                artifact: EvidenceArtifactIdentityV1 {
                    schema: "fixture-artifact-v1".to_string(),
                    digest_algorithm: "blake3".to_string(),
                    digest: "c".repeat(64),
                    byte_len: 1,
                },
                authority: None,
            },
        };
        let view = ButlinResolvedEvidenceViewV2 {
            schema: BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2.to_string(),
            base_report_schema_version: base_report.schema_version,
            base_report_blake3: base_report_blake3_v1(&base_report).expect("base digest"),
            lineages: vec![direct],
            resolved_counts: EvidenceOutcomeCountsV1 {
                architectural_only: base_report.indicators.len() - 1,
                observed: 1,
                ..EvidenceOutcomeCountsV1::default()
            },
        };
        let disposition = classify_gwt1_evidence_disposition_v1(&view).expect("disposition");
        let internal_verification =
            br#"[{"verificationResult":{"signature":{"certificate":"fixture"}}}]"#.to_vec();
        let stored_candidate = trusted_resolution_candidate_for_test(
            base_report.clone(),
            view.clone(),
            disposition.clone(),
            Vec::new(),
        );
        let recomputed_candidate = trusted_resolution_candidate_for_test(
            base_report.clone(),
            view.clone(),
            disposition.clone(),
            internal_verification.clone(),
        );
        (
            base_report,
            view,
            disposition,
            stored_candidate,
            recomputed_candidate,
            internal_verification,
        )
    }

    #[test]
    fn final_archive_digest_must_be_canonical_lower_hex() {
        assert!(is_lower_hex_64(&"a".repeat(64)));
        assert!(!is_lower_hex_64(&"A".repeat(64)));
        assert!(!is_lower_hex_64(&"a".repeat(63)));
        assert!(!is_lower_hex_64(&"g".repeat(64)));
    }

    #[test]
    fn verified_token_schema_is_frozen() {
        assert_eq!(
            GWT1_VERIFIED_FINAL_ARTIFACT_SCHEMA_V1,
            "butlin-gwt1-verified-final-artifact-v1"
        );
    }

    #[test]
    fn exact_reconstruction_mints_read_only_token() {
        let (base, view, disposition, stored, recomputed, transcript) = fixture();
        let token = verify_gwt1_final_reconstruction_v1(
            &"d".repeat(64),
            &base,
            &view,
            &disposition,
            &stored,
            &transcript,
            &recomputed,
        )
        .expect("exact reconstruction");
        let digest = "d".repeat(64);
        assert_eq!(token.final_archive_sha256(), digest.as_str());
        assert_eq!(
            token.resolved_gwt1_outcome(),
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
        assert!(!token.has_causal_contradiction());
    }

    #[test]
    fn inner_verification_transcript_mismatch_fails_closed() {
        let (base, view, disposition, stored, recomputed, _) = fixture();
        assert!(matches!(
            verify_gwt1_final_reconstruction_v1(
                &"d".repeat(64),
                &base,
                &view,
                &disposition,
                &stored,
                b"tampered",
                &recomputed,
            ),
            Err(Gwt1FinalArtifactReconstructionErrorV1::InternalPromotionVerificationMismatch)
        ));
    }

    #[test]
    fn empty_recomputed_verification_transcript_fails_closed() {
        let (base, view, disposition, stored, _, transcript) = fixture();
        let recomputed = trusted_resolution_candidate_for_test(
            base.clone(),
            view.clone(),
            disposition.clone(),
            Vec::new(),
        );
        assert!(matches!(
            verify_gwt1_final_reconstruction_v1(
                &"d".repeat(64),
                &base,
                &view,
                &disposition,
                &stored,
                &transcript,
                &recomputed,
            ),
            Err(Gwt1FinalArtifactReconstructionErrorV1::EmptyRecomputedPromotionVerification)
        ));
    }

    #[test]
    fn independently_recomputed_view_mismatch_fails_closed() {
        let (base, view, disposition, stored, _, transcript) = fixture();
        let mut recomputed_view = view.clone();
        recomputed_view.base_report_blake3 = "e".repeat(64);
        let recomputed = trusted_resolution_candidate_for_test(
            base.clone(),
            recomputed_view,
            disposition.clone(),
            transcript.clone(),
        );
        assert!(matches!(
            verify_gwt1_final_reconstruction_v1(
                &"d".repeat(64),
                &base,
                &view,
                &disposition,
                &stored,
                &transcript,
                &recomputed,
            ),
            Err(Gwt1FinalArtifactReconstructionErrorV1::RecomputedViewMismatch)
        ));
    }
}
