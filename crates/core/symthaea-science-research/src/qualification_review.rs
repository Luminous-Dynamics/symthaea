// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Capability-bearing handoff into scientific qualification review.
//!
//! This follows Symthaea's existing capability pattern used by other domains:
//! downstream code receives a private-field capability rather than trusting a
//! caller-selected status enum. The capability proves only that the exact
//! `QualificationEligibilityManifest` reached `EligibleUnderDeclaredProfile`
//! under the exact frozen profile. It does **not** authenticate the profile,
//! qualify the claim, establish global independence, or grant scientific truth.

use serde::Serialize;

use crate::{
    AuthorityLevel, FrozenQualificationEligibilityProfile, QualificationEligibilityClosure,
    QualificationEligibilityManifest, Sha256Digest,
};

const REVIEW_READY_DIGEST_DOMAIN: &str =
    "symthaea.qualification-review-ready-capability.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationReviewReadinessError {
    ProfileIdentityMismatch,
    EligibilityNotSatisfied {
        closure: QualificationEligibilityClosure,
    },
    EligibilityProfileWasNotDeclared,
    ManifestAlreadyClaimsQualification,
    ManifestClaimsGlobalIndependence,
}

/// Non-forgeable outside this crate's validated constructor path.
///
/// The type is serializable for retained evidence, but intentionally not
/// deserializable. Rehydration must rerun the eligibility gates and mint a fresh
/// capability from the resulting manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationReviewReadyClaim {
    manifest: QualificationEligibilityManifest,
    profile_sha256: Sha256Digest,
    readiness_sha256: Sha256Digest,
}

impl QualificationReviewReadyClaim {
    pub fn try_new(
        manifest: QualificationEligibilityManifest,
        profile: &FrozenQualificationEligibilityProfile,
    ) -> Result<Self, QualificationReviewReadinessError> {
        if manifest.profile_sha256() != profile.profile_sha256() {
            return Err(QualificationReviewReadinessError::ProfileIdentityMismatch);
        }
        if manifest.closure() != QualificationEligibilityClosure::EligibleUnderDeclaredProfile {
            return Err(QualificationReviewReadinessError::EligibilityNotSatisfied {
                closure: manifest.closure(),
            });
        }
        if manifest.profile_binding_level() != AuthorityLevel::Declared {
            return Err(QualificationReviewReadinessError::EligibilityProfileWasNotDeclared);
        }
        if manifest.qualification_established() {
            return Err(QualificationReviewReadinessError::ManifestAlreadyClaimsQualification);
        }
        if manifest.global_independence_established() {
            return Err(QualificationReviewReadinessError::ManifestClaimsGlobalIndependence);
        }

        let readiness_sha256 = readiness_digest(manifest.manifest_sha256(), profile.profile_sha256());
        Ok(Self {
            manifest,
            profile_sha256: profile.profile_sha256().clone(),
            readiness_sha256,
        })
    }

    pub fn manifest(&self) -> &QualificationEligibilityManifest {
        &self.manifest
    }

    pub fn profile_sha256(&self) -> &Sha256Digest {
        &self.profile_sha256
    }

    pub fn readiness_sha256(&self) -> &Sha256Digest {
        &self.readiness_sha256
    }

    /// Hard-coded false by construction. Review readiness is not qualification.
    pub const fn qualification_established(&self) -> bool {
        false
    }
}

fn readiness_digest(
    manifest_sha256: &Sha256Digest,
    profile_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut bytes = Vec::new();
    append_frame(&mut bytes, REVIEW_READY_DIGEST_DOMAIN);
    append_frame(&mut bytes, manifest_sha256.as_str());
    append_frame(&mut bytes, profile_sha256.as_str());
    append_frame(&mut bytes, "eligible-under-declared-profile");
    append_frame(&mut bytes, "qualification-not-established");
    append_frame(&mut bytes, "global-independence-not-established");
    Sha256Digest::of_bytes(&bytes)
}

fn append_frame(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}
