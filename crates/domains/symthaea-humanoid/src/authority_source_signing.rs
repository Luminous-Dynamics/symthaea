// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-phase signing helper for verified humanoid authority sources.
//!
//! The authenticated statement binds the scheme ID but, correctly, not its own
//! signature bytes. This helper lets an upstream service obtain the exact canonical
//! statement digest first and attach authentication only after signing it.

use crate::evidence_digest::HumanoidEvidenceDigest;
use crate::qualification::HumanoidQualificationSubject;
use crate::verified_authority_source::{
    HumanoidAuthorityAuthenticationEvidence, HumanoidAuthoritySourceClaim,
    HumanoidVerifiedAuthorityKind,
};

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidAuthorityUnsignedClaim {
    subject: HumanoidQualificationSubject,
    source_kind: HumanoidVerifiedAuthorityKind,
    evidence_digest: HumanoidEvidenceDigest,
    scale: f32,
    evaluated_at_s: f64,
    valid_until_s: f64,
    issuer_id: String,
    key_id: String,
    revocation_epoch: u64,
    scheme_id: String,
    statement_digest: HumanoidEvidenceDigest,
}

impl HumanoidAuthorityUnsignedClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: HumanoidQualificationSubject,
        source_kind: HumanoidVerifiedAuthorityKind,
        evidence_digest: HumanoidEvidenceDigest,
        scale: f32,
        evaluated_at_s: f64,
        valid_until_s: f64,
        issuer_id: impl Into<String>,
        key_id: impl Into<String>,
        revocation_epoch: u64,
        scheme_id: impl Into<String>,
    ) -> Option<Self> {
        let issuer_id = issuer_id.into();
        let key_id = key_id.into();
        let scheme_id = scheme_id.into();

        // Signature bytes do not participate in the signed statement. A single
        // placeholder byte is therefore sufficient to ask the canonical claim
        // encoder for the exact digest the external signer must authenticate.
        let placeholder_auth =
            HumanoidAuthorityAuthenticationEvidence::new(scheme_id.clone(), vec![0x00])?;
        let preview = HumanoidAuthoritySourceClaim::new(
            &subject,
            source_kind,
            evidence_digest,
            scale,
            evaluated_at_s,
            valid_until_s,
            issuer_id.clone(),
            key_id.clone(),
            revocation_epoch,
            placeholder_auth,
        )?;
        let statement_digest = preview.statement_digest();
        if statement_digest.is_zero() {
            return None;
        }

        Some(Self {
            subject,
            source_kind,
            evidence_digest,
            scale,
            evaluated_at_s,
            valid_until_s,
            issuer_id,
            key_id,
            revocation_epoch,
            scheme_id,
            statement_digest,
        })
    }

    pub const fn statement_digest(&self) -> HumanoidEvidenceDigest {
        self.statement_digest
    }

    pub const fn source_kind(&self) -> HumanoidVerifiedAuthorityKind {
        self.source_kind
    }

    pub const fn evidence_digest(&self) -> HumanoidEvidenceDigest {
        self.evidence_digest
    }

    pub const fn scale(&self) -> f32 {
        self.scale
    }

    pub const fn evaluated_at_s(&self) -> f64 {
        self.evaluated_at_s
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.valid_until_s
    }

    pub fn scheme_id(&self) -> &str {
        &self.scheme_id
    }

    /// Attach the external signature and reconstruct the exact claim. The
    /// statement digest is checked again so signing cannot accidentally cross a
    /// changed scheme/subject/evidence/time lineage.
    pub fn attach_signature(self, signature: Vec<u8>) -> Option<HumanoidAuthoritySourceClaim> {
        let authentication =
            HumanoidAuthorityAuthenticationEvidence::new(self.scheme_id.clone(), signature)?;
        let claim = HumanoidAuthoritySourceClaim::new(
            &self.subject,
            self.source_kind,
            self.evidence_digest,
            self.scale,
            self.evaluated_at_s,
            self.valid_until_s,
            self.issuer_id,
            self.key_id,
            self.revocation_epoch,
            authentication,
        )?;
        (claim.statement_digest() == self.statement_digest).then_some(claim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "signing-test-backend",
        )
    }

    #[test]
    fn attached_signature_preserves_prepared_statement_digest() {
        let unsigned = HumanoidAuthorityUnsignedClaim::new(
            subject(),
            HumanoidVerifiedAuthorityKind::Epistemic,
            HumanoidEvidenceDigest::from_bytes([4; 32]),
            0.7,
            2.0,
            2.05,
            "state-estimator",
            "estimator-key-1",
            3,
            "ml-dsa-87",
        )
        .unwrap();
        let digest = unsigned.statement_digest();
        let claim = unsigned.attach_signature(vec![8; 64]).unwrap();
        assert_eq!(claim.statement_digest(), digest);
    }
}
