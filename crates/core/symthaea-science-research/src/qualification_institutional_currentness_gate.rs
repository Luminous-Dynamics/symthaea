// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public anti-circularity gate for institutional currentness authority.

use symthaea_trust_core::{AuthorizedTrustRoleAttestation, RootRoleQuorumProof, TrustedTime};

use crate::{
    InstitutionallyCurrentQualifiedScientificClaim, QualificationCurrentnessError,
    QualificationCurrentnessStatement, QualificationValidityReadinessAssessment,
    QualifiedScientificClaim,
};

pub fn authenticate_institutional_currentness(
    qualified: &QualifiedScientificClaim,
    readiness: &QualificationValidityReadinessAssessment,
    evaluation_time: &TrustedTime,
    statement: &QualificationCurrentnessStatement,
    lifecycle_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<InstitutionallyCurrentQualifiedScientificClaim, QualificationCurrentnessError> {
    if evaluation_time.bindings().iter().any(|binding| {
        binding.source_artifact_sha256() == readiness.assessment_sha256()
            || binding.source_artifact_sha256() == statement.statement_sha256()
            || binding.source_artifact_sha256() == lifecycle_authority.authority_sha256()
            || binding.source_artifact_sha256() == role_proof.proof_sha256()
    }) {
        return Err(QualificationCurrentnessError::TimeDependsOnCurrentnessDecision);
    }

    crate::qualification_institutional_currentness::authenticate_institutional_currentness(
        qualified,
        readiness,
        evaluation_time,
        statement,
        lifecycle_authority,
        role_proof,
    )
}
