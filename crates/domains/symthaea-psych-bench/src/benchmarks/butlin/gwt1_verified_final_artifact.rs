// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authority sink for finalized GWT-1 reporting.
//!
//! Raw `resolved_view_v2.json`, disposition JSON, or a boolean are deliberately
//! insufficient to drive authoritative reporting. The renderer in this module
//! accepts only an opaque [`VerifiedGwt1ResolvedArtifactV1`].
//!
//! This tranche intentionally defines **no production constructor** for that
//! capability. The future verifier in #3411 must earn the capability by
//! verifying the complete final archive, its attestation, bounded admission,
//! internal checksums, retained authority bytes, and signer roots. Until that
//! verifier and its reviewed immutable signer root exist, authoritative GWT-1
//! reporting remains unmintable by construction.
//!
//! This module is gated by the consumer-only `trusted-resolution-consumer`
//! feature. That feature does not imply `trusted-resolution-authority` or
//! `symthaea-backend`, so a reporting process does not gain candidate-generation
//! or causal-promotion authority merely by consuming a finalized artifact.

use std::fmt;

use serde::Serialize;

use super::gwt1_evidence_disposition::{
    Gwt1EvidenceDispositionSummaryV1, Gwt1EvidenceDispositionV1,
};
use super::report::EvidenceOutcome;
use super::resolution_view_v2::ButlinResolvedEvidenceViewV2;

pub const GWT1_VERIFIED_RESOLVED_ARTIFACT_SCHEMA_V1: &str =
    "butlin-gwt1-verified-resolved-artifact-v1";
pub const GWT1_AUTHORITATIVE_REPORT_SCHEMA_V1: &str =
    "butlin-gwt1-authoritative-report-v1";

/// Process-local proof that a complete final GWT-1 artifact passed the frozen
/// consumer verification boundary.
///
/// Fields are private. The type is intentionally neither `Serialize` nor
/// `Deserialize`, and this tranche intentionally provides no production
/// constructor. Persisted authority remains the signed final archive and its
/// retained verification evidence, never an instance of this Rust type.
pub struct VerifiedGwt1ResolvedArtifactV1 {
    final_archive_sha256: String,
    final_attestation_bundle_sha256: String,
    final_attestation_verification_sha256: String,
    final_attestation_verification_bytes: Vec<u8>,
    final_resolution_workflow_sha: String,
    evidence_subject_sha: String,
    resolved_view: ButlinResolvedEvidenceViewV2,
    disposition: Gwt1EvidenceDispositionSummaryV1,
}

impl fmt::Debug for VerifiedGwt1ResolvedArtifactV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifiedGwt1ResolvedArtifactV1")
            .field("final_archive_sha256", &self.final_archive_sha256)
            .field(
                "final_attestation_bundle_sha256",
                &self.final_attestation_bundle_sha256,
            )
            .field(
                "final_attestation_verification_sha256",
                &self.final_attestation_verification_sha256,
            )
            .field(
                "final_resolution_workflow_sha",
                &self.final_resolution_workflow_sha,
            )
            .field("evidence_subject_sha", &self.evidence_subject_sha)
            .field("resolved_view", &self.resolved_view)
            .field("disposition", &self.disposition)
            .finish_non_exhaustive()
    }
}

impl VerifiedGwt1ResolvedArtifactV1 {
    pub fn final_archive_sha256(&self) -> &str {
        &self.final_archive_sha256
    }

    pub fn final_attestation_bundle_sha256(&self) -> &str {
        &self.final_attestation_bundle_sha256
    }

    pub fn final_attestation_verification_sha256(&self) -> &str {
        &self.final_attestation_verification_sha256
    }

    /// Exact bytes emitted by the final attestation-verifier invocation that
    /// will eventually mint this capability. Retaining the bytes prevents a
    /// digest from naming evidence that the reporting process cannot replay.
    pub fn final_attestation_verification_bytes(&self) -> &[u8] {
        &self.final_attestation_verification_bytes
    }

    pub fn final_resolution_workflow_sha(&self) -> &str {
        &self.final_resolution_workflow_sha
    }

    pub fn evidence_subject_sha(&self) -> &str {
        &self.evidence_subject_sha
    }

    pub fn resolved_view(&self) -> &ButlinResolvedEvidenceViewV2 {
        &self.resolved_view
    }

    pub fn disposition(&self) -> &Gwt1EvidenceDispositionSummaryV1 {
        &self.disposition
    }
}

/// Output-only projection that may be serialized only after rendering from an
/// opaque verified final-artifact capability. Its fields are private and it
/// deliberately does not implement `Deserialize`: external code can inspect or
/// serialize a report returned by the renderer, but cannot fabricate this type
/// from a struct literal or report-shaped JSON.
///
/// This is a reporting projection, not new scientific evidence. In particular,
/// `consciousness_or_sentience_established` is permanently false for this
/// artifact class: the Butlin GWT-1 evidence program does not establish either
/// proposition merely by resolving one indicator's evidence lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Gwt1AuthoritativeReportV1 {
    schema: String,
    authority_schema: String,
    final_archive_sha256: String,
    final_attestation_verification_sha256: String,
    final_resolution_workflow_sha: String,
    evidence_subject_sha: String,
    base_report_blake3: String,
    direct_outcome: EvidenceOutcome,
    causal_outcome: Option<EvidenceOutcome>,
    resolved_outcome: EvidenceOutcome,
    disposition: Gwt1EvidenceDispositionV1,
    has_causal_contradiction: bool,
    causal_follow_up_required: bool,
    consciousness_or_sentience_established: bool,
}

impl Gwt1AuthoritativeReportV1 {
    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn authority_schema(&self) -> &str {
        &self.authority_schema
    }

    pub fn final_archive_sha256(&self) -> &str {
        &self.final_archive_sha256
    }

    pub fn final_attestation_verification_sha256(&self) -> &str {
        &self.final_attestation_verification_sha256
    }

    pub fn final_resolution_workflow_sha(&self) -> &str {
        &self.final_resolution_workflow_sha
    }

    pub fn evidence_subject_sha(&self) -> &str {
        &self.evidence_subject_sha
    }

    pub fn base_report_blake3(&self) -> &str {
        &self.base_report_blake3
    }

    pub fn direct_outcome(&self) -> EvidenceOutcome {
        self.direct_outcome
    }

    pub fn causal_outcome(&self) -> Option<EvidenceOutcome> {
        self.causal_outcome
    }

    pub fn resolved_outcome(&self) -> EvidenceOutcome {
        self.resolved_outcome
    }

    pub fn disposition(&self) -> Gwt1EvidenceDispositionV1 {
        self.disposition
    }

    pub fn has_causal_contradiction(&self) -> bool {
        self.has_causal_contradiction
    }

    pub fn causal_follow_up_required(&self) -> bool {
        self.causal_follow_up_required
    }

    pub fn consciousness_or_sentience_established(&self) -> bool {
        self.consciousness_or_sentience_established
    }
}

/// Render authoritative GWT-1 reporting from a verified final artifact.
///
/// There is intentionally no overload accepting `ButlinResolvedEvidenceViewV2`,
/// `Gwt1EvidenceDispositionSummaryV1`, JSON bytes, or a verification boolean.
/// The opaque capability is the authority boundary.
pub fn render_verified_gwt1_authoritative_report_v1(
    verified: &VerifiedGwt1ResolvedArtifactV1,
) -> Gwt1AuthoritativeReportV1 {
    let disposition = verified.disposition();

    Gwt1AuthoritativeReportV1 {
        schema: GWT1_AUTHORITATIVE_REPORT_SCHEMA_V1.to_string(),
        authority_schema: GWT1_VERIFIED_RESOLVED_ARTIFACT_SCHEMA_V1.to_string(),
        final_archive_sha256: verified.final_archive_sha256().to_string(),
        final_attestation_verification_sha256: verified
            .final_attestation_verification_sha256()
            .to_string(),
        final_resolution_workflow_sha: verified.final_resolution_workflow_sha().to_string(),
        evidence_subject_sha: verified.evidence_subject_sha().to_string(),
        base_report_blake3: verified.resolved_view().base_report_blake3.clone(),
        direct_outcome: disposition.direct_outcome,
        causal_outcome: disposition.causal_outcome,
        resolved_outcome: disposition.resolved_outcome,
        disposition: disposition.disposition,
        has_causal_contradiction: disposition.has_causal_contradiction,
        causal_follow_up_required: disposition.causal_follow_up_required,
        consciousness_or_sentience_established: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::report::SupportTier;
    use super::super::resolution_view::EvidenceOutcomeCountsV1;
    use super::super::resolution_view_v2::BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2;

    fn verified_for_test() -> VerifiedGwt1ResolvedArtifactV1 {
        let observed = EvidenceOutcome::Supported(SupportTier::Observed);
        VerifiedGwt1ResolvedArtifactV1 {
            final_archive_sha256: "a".repeat(64),
            final_attestation_bundle_sha256: "b".repeat(64),
            final_attestation_verification_sha256: "c".repeat(64),
            final_attestation_verification_bytes: br#"[{"verified":true}]"#.to_vec(),
            final_resolution_workflow_sha: "d".repeat(40),
            evidence_subject_sha: "e".repeat(40),
            resolved_view: ButlinResolvedEvidenceViewV2 {
                schema: BUTLIN_RESOLVED_EVIDENCE_VIEW_SCHEMA_V2.to_string(),
                base_report_schema_version: 1,
                base_report_blake3: "f".repeat(64),
                lineages: Vec::new(),
                resolved_counts: EvidenceOutcomeCountsV1 {
                    observed: 1,
                    ..EvidenceOutcomeCountsV1::default()
                },
            },
            disposition: Gwt1EvidenceDispositionSummaryV1 {
                schema: super::super::gwt1_evidence_disposition::GWT1_EVIDENCE_DISPOSITION_SCHEMA_V1
                    .to_string(),
                direct_outcome: observed,
                causal_outcome: Some(EvidenceOutcome::Inconclusive),
                resolved_outcome: observed,
                disposition: Gwt1EvidenceDispositionV1::CausalInconclusiveRetainsObserved,
                has_causal_contradiction: false,
                causal_follow_up_required: true,
            },
        }
    }

    #[test]
    fn schemas_are_frozen() {
        assert_eq!(
            GWT1_VERIFIED_RESOLVED_ARTIFACT_SCHEMA_V1,
            "butlin-gwt1-verified-resolved-artifact-v1"
        );
        assert_eq!(
            GWT1_AUTHORITATIVE_REPORT_SCHEMA_V1,
            "butlin-gwt1-authoritative-report-v1"
        );
    }

    #[test]
    fn renderer_preserves_verified_authority_and_method_outcomes() {
        let verified = verified_for_test();
        let report = render_verified_gwt1_authoritative_report_v1(&verified);

        assert_eq!(report.schema(), GWT1_AUTHORITATIVE_REPORT_SCHEMA_V1);
        assert_eq!(report.authority_schema(), GWT1_VERIFIED_RESOLVED_ARTIFACT_SCHEMA_V1);
        assert_eq!(report.final_archive_sha256(), "a".repeat(64));
        assert_eq!(report.final_attestation_verification_sha256(), "c".repeat(64));
        assert_eq!(report.final_resolution_workflow_sha(), "d".repeat(40));
        assert_eq!(report.evidence_subject_sha(), "e".repeat(40));
        assert_eq!(report.base_report_blake3(), "f".repeat(64));
        assert_eq!(
            report.direct_outcome(),
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
        assert_eq!(report.causal_outcome(), Some(EvidenceOutcome::Inconclusive));
        assert_eq!(
            report.resolved_outcome(),
            EvidenceOutcome::Supported(SupportTier::Observed)
        );
        assert_eq!(
            report.disposition(),
            Gwt1EvidenceDispositionV1::CausalInconclusiveRetainsObserved
        );
        assert!(!report.has_causal_contradiction());
        assert!(report.causal_follow_up_required());
    }

    #[test]
    fn renderer_never_claims_consciousness_or_sentience() {
        let report = render_verified_gwt1_authoritative_report_v1(&verified_for_test());
        assert!(!report.consciousness_or_sentience_established());
    }
}