// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! One-way reporting projection from an opaque verified GWT-1 final artifact.
//!
//! This type is serializable so a trusted consumer workflow can persist and
//! separately attest a compact reporting artifact. It intentionally does not
//! implement `Deserialize` or expose public fields/constructors. A JSON document
//! with the same shape is therefore not an authority token: external authority
//! remains the trusted-consumer workflow attestation over the emitted bytes.

use serde::Serialize;

use super::gwt1_evidence_disposition::Gwt1EvidenceDispositionV1;
use super::gwt1_verified_final_artifact::VerifiedGwt1ResolvedArtifactV1;
use super::report::{EvidenceOutcome, SupportTier};

pub const GWT1_VERIFIED_REPORT_PROJECTION_SCHEMA_V1: &str =
    "butlin-gwt1-verified-report-projection-v1";

#[derive(Debug, Serialize)]
pub struct Gwt1VerifiedReportProjectionV1 {
    schema: &'static str,
    indicator_id: &'static str,
    final_archive_sha256: String,
    base_report_blake3: String,
    direct_outcome: EvidenceOutcome,
    causal_outcome: Option<EvidenceOutcome>,
    resolved_outcome: EvidenceOutcome,
    disposition: Gwt1EvidenceDispositionV1,
    has_causal_contradiction: bool,
    causal_follow_up_required: bool,
    tier_ceiling: SupportTier,
}

impl Gwt1VerifiedReportProjectionV1 {
    pub fn schema(&self) -> &'static str {
        self.schema
    }

    pub fn indicator_id(&self) -> &'static str {
        self.indicator_id
    }

    pub fn final_archive_sha256(&self) -> &str {
        &self.final_archive_sha256
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

    pub fn tier_ceiling(&self) -> SupportTier {
        self.tier_ceiling
    }
}

/// Project the minimum reporting state from an already verified opaque token.
///
/// This function cannot itself create authority because callers must already
/// possess `VerifiedGwt1ResolvedArtifactV1`, whose only constructor is private
/// to the trusted final-artifact verifier module.
pub fn project_gwt1_verified_report_v1(
    verified: &VerifiedGwt1ResolvedArtifactV1,
) -> Gwt1VerifiedReportProjectionV1 {
    let disposition = verified.disposition();
    Gwt1VerifiedReportProjectionV1 {
        schema: GWT1_VERIFIED_REPORT_PROJECTION_SCHEMA_V1,
        indicator_id: "GWT-1",
        final_archive_sha256: verified.final_archive_sha256().to_string(),
        base_report_blake3: verified.resolved_view().base_report_blake3.clone(),
        direct_outcome: disposition.direct_outcome,
        causal_outcome: disposition.causal_outcome,
        resolved_outcome: disposition.resolved_outcome,
        disposition: disposition.disposition,
        has_causal_contradiction: disposition.has_causal_contradiction,
        causal_follow_up_required: disposition.causal_follow_up_required,
        tier_ceiling: SupportTier::CausallySupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_schema_and_ceiling_are_frozen() {
        assert_eq!(
            GWT1_VERIFIED_REPORT_PROJECTION_SCHEMA_V1,
            "butlin-gwt1-verified-report-projection-v1"
        );
        assert_eq!(
            SupportTier::CausallySupported,
            SupportTier::CausallySupported
        );
    }
}
