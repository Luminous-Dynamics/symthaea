// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Joins ingress use-time evidence revalidation to the signed model-context
//! exposure capture/verification seam.
//!
//! Each memory span must be revalidated at the independently pinned bitemporal
//! snapshot and separately pass a purpose/rights verifier. The resulting
//! in-process receipt digest must equal the lifecycle-revalidation digest
//! embedded in the observer-signed exposure span.

use broca_mace_ingress::{
    revalidate_for_use_v1, AsKnownQuery, EvidenceUseContextV1, EvidenceUseReceiptV1,
    EvidenceVerifier, IngestedExperience, RawInteractionV1, SourceTrustStatus,
    SourceLifecycleStatus, DependencyLineageStatusV1,
};
use broca_mace_policy::Scope;

use crate::mace_context_exposure_receipt::{
    MaceContextExposureRejectV1, MaceExposureAdmissionVerifierV1,
    MaceExpectedExposureBindingV1, MaceMemoryExposureSpanV1,
};

#[derive(Clone, Debug)]
pub struct MaceResolvedEvidenceUseStateV1 {
    /// Stable evidence identifier used by the signed exposure span.
    pub canonical_evidence_id: String,
    /// Canonical ingress event envelope and previously admitted in-process view.
    pub raw: RawInteractionV1,
    pub prior: IngestedExperience,
}

/// Resolves a candidate evidence ID to the source event and its original
/// verified ingress view. The resolver cannot confer trust: the EvidenceVerifier
/// below revalidates the event at the current decision snapshot.
pub trait MaceExposureUseStateResolverV1 {
    fn resolve_prior(
        &self,
        evidence_id: &str,
        expected: &MaceExpectedExposureBindingV1<'_>,
    ) -> Result<MaceResolvedEvidenceUseStateV1, MaceContextExposureRejectV1>;
}

/// Purpose/rights/consent check delegated to the existing Mycelix or deployment
/// rights authority. Ingress source trust is deliberately not treated as a
/// substitute for consent, purpose limitation, or user authorization.
pub trait MaceExposurePurposeRightsVerifierV1 {
    fn verify_current_use(
        &self,
        evidence_id: &str,
        payload_sha256: &str,
        source_root_digest_hex: &str,
        admissibility_receipt_sha256: &str,
        purpose_code: &str,
        scope: Scope,
        consumer_policy_sha256: &str,
        valid_at_ms: i64,
        known_at_ms: i64,
    ) -> Result<(), MaceContextExposureRejectV1>;
}

pub struct MaceIngressUseTimeAdmissionVerifierV1<'a, R, V, W> {
    pub state_resolver: &'a R,
    pub evidence_verifier: &'a V,
    pub rights_verifier: &'a W,
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

fn event_parent_ids(receipt: &EvidenceUseReceiptV1) -> Vec<String> {
    receipt.derived_from_event_ids.iter().map(u32::to_string).collect()
}

impl<R, V, W> MaceExposureAdmissionVerifierV1
    for MaceIngressUseTimeAdmissionVerifierV1<'_, R, V, W>
where
    R: MaceExposureUseStateResolverV1,
    V: EvidenceVerifier,
    W: MaceExposurePurposeRightsVerifierV1,
{
    fn verify_admitted_exposure(
        &self,
        exposure: &MaceMemoryExposureSpanV1,
        expected: &MaceExpectedExposureBindingV1<'_>,
    ) -> Result<(), MaceContextExposureRejectV1> {
        if exposure.admission_status != "admitted" {
            return Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected);
        }

        let state = self.state_resolver.resolve_prior(&exposure.evidence_id, expected)?;
        if state.canonical_evidence_id != exposure.evidence_id
            || state.raw.event_id != state.prior.experience().event_id
            || state.raw.scope != expected.use_scope
        {
            return Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected);
        }

        let query = AsKnownQuery {
            valid_at_ms: expected.use_valid_at_ms,
            known_at_ms: expected.use_known_at_ms,
        };
        let context = EvidenceUseContextV1 {
            purpose_code: expected.use_purpose_code.to_owned(),
            required_scope: expected.use_scope,
            consumer_policy_sha256: expected.use_consumer_policy_sha256.to_owned(),
            require_complete_dependency_lineage: expected.require_complete_dependency_lineage,
        };

        // The resolver supplies only the previously admitted event/view.
        // Current trust, lifecycle, validity, lineage, calibration, causal
        // assessment and outcome are independently reverified right here.
        let receipt = revalidate_for_use_v1(
            &state.raw,
            &state.prior,
            query,
            &context,
            self.evidence_verifier,
        )
        .map_err(|_| MaceContextExposureRejectV1::EvidenceAdmissionRejected)?;

        if !receipt.verify_integrity()
            || !receipt.has_verified_ingress_provenance()
            || receipt.event_id != state.raw.event_id
            || receipt.payload_sha256 != exposure.evidence_payload_sha256
            || receipt.current_raw_event_receipt_digest != exposure.source_receipt_sha256
            || receipt.current_authority_receipt_digest != exposure.authority_receipt_sha256
            || receipt.transformation_digest != exposure.transformation_receipt_sha256
            || receipt.canonical_sha256 != exposure.lifecycle_revalidation_receipt_sha256
            || receipt.valid_at_ms != expected.use_valid_at_ms
            || receipt.known_at_ms != expected.use_known_at_ms
            || receipt.purpose_code != expected.use_purpose_code
            || receipt.consumer_policy_sha256 != expected.use_consumer_policy_sha256
            || receipt.scope != expected.use_scope
            || receipt.source_trust_status != SourceTrustStatus::TrustedForScope
            || receipt.lifecycle_status_as_known != SourceLifecycleStatus::Active
        {
            return Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected);
        }

        if expected.require_complete_dependency_lineage
            && receipt.dependency_lineage_status != DependencyLineageStatusV1::Complete
        {
            return Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected);
        }

        if event_parent_ids(&receipt) != exposure.parent_event_ids {
            return Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected);
        }
        if let Some(manifest_digest) = receipt.dependency_manifest_sha256.as_deref() {
            if manifest_digest != exposure.dependency_closure_sha256 {
                return Err(MaceContextExposureRejectV1::EvidenceAdmissionRejected);
            }
        }

        self.rights_verifier.verify_current_use(
            &exposure.evidence_id,
            &receipt.payload_sha256,
            &receipt.source_root_digest.to_hex(),
            &exposure.admissibility_receipt_sha256,
            &expected.use_purpose_code,
            expected.use_scope,
            expected.use_consumer_policy_sha256,
            expected.use_valid_at_ms,
            expected.use_known_at_ms,
        )?;

        Ok(())
    }
}
