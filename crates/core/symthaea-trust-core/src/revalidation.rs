// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Historical attestation validity under newly learned key-lifecycle facts.
//!
//! This deliberately does **not** ask whether the original signers are still
//! active today. Routine retirement or prospective revocation after the original
//! authority time does not erase legitimate historical authority. By contrast,
//! a later `CompromisedSince(T)` finding may invalidate an original authority
//! grant when T reaches back into the signing/evaluation interval.
//!
//! Lifecycle histories are content-addressed but are not yet authenticated by a
//! generic trust-root role in TRUST-005A. Therefore this module emits a
//! non-authorizing assessment. A later lifecycle-authority gate may wrap a
//! structurally supported assessment into a capability.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    AttestationPolicy, AttestationPolicyError, FramedDigest, KeyLifecycleHistory,
    Sha256Digest, SignatureAlgorithm, TemporalKeyStatus, TrustSnapshot,
    VerifiedAttestation,
};

const HISTORICAL_ATTESTATION_ASSESSMENT_DOMAIN: &str =
    "symthaea.historical-attestation-assessment.identity.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HistoricalAttestationClosure {
    StructurallySupported,
    HistoricalAuthorityInvalidated,
    Incomplete,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HistoricalSignerStatus {
    NotYetActive,
    Active,
    Retired,
    Revoked,
    Compromised,
    Replaced,
}

impl From<TemporalKeyStatus> for HistoricalSignerStatus {
    fn from(value: TemporalKeyStatus) -> Self {
        match value {
            TemporalKeyStatus::NotYetActive => Self::NotYetActive,
            TemporalKeyStatus::Active => Self::Active,
            TemporalKeyStatus::Retired => Self::Retired,
            TemporalKeyStatus::Revoked => Self::Revoked,
            TemporalKeyStatus::Compromised => Self::Compromised,
            TemporalKeyStatus::Replaced => Self::Replaced,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum HistoricalAttestationFinding {
    InvalidPolicy,
    PolicyIdentityMismatch,
    HistoricalTrustSnapshotInvalid,
    HistoricalTrustSnapshotIdentityMismatch,
    DuplicateLifecycleHistory {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    UnexpectedLifecycleHistory {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    MissingHistoricalKeyRecord {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    MissingLifecycleHistory {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    LifecycleIdentityMismatch {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    VerificationKeyIdentityMismatch {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    HistoricalSignerNotAuthoritative {
        algorithm: SignatureAlgorithm,
        key_id: String,
        status: HistoricalSignerStatus,
    },
    InsufficientHistoricallyValidSignatures {
        actual: usize,
        required: usize,
    },
    MissingRequiredAlgorithm {
        algorithm: SignatureAlgorithm,
    },
}

/// Non-authorizing reevaluation of one historical authority grant.
///
/// This type is serializable for retained evidence but intentionally carries
/// hard-coded false authority statements until lifecycle histories themselves
/// are authenticated under a generic lifecycle role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HistoricalAttestationAssessment {
    historical_authority_sha256: Sha256Digest,
    historical_policy_sha256: Sha256Digest,
    historical_trust_snapshot_sha256: Sha256Digest,
    historical_evaluation_time_unix_s: u64,
    lifecycle_history_sha256s: Vec<Sha256Digest>,
    surviving_historical_signers: Vec<(SignatureAlgorithm, String)>,
    findings: Vec<HistoricalAttestationFinding>,
    closure: HistoricalAttestationClosure,
    assessment_sha256: Sha256Digest,
}

impl HistoricalAttestationAssessment {
    pub fn historical_authority_sha256(&self) -> &Sha256Digest {
        &self.historical_authority_sha256
    }

    pub fn historical_policy_sha256(&self) -> &Sha256Digest {
        &self.historical_policy_sha256
    }

    pub fn historical_trust_snapshot_sha256(&self) -> &Sha256Digest {
        &self.historical_trust_snapshot_sha256
    }

    pub fn historical_evaluation_time_unix_s(&self) -> u64 {
        self.historical_evaluation_time_unix_s
    }

    pub fn lifecycle_history_sha256s(&self) -> &[Sha256Digest] {
        &self.lifecycle_history_sha256s
    }

    pub fn surviving_historical_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.surviving_historical_signers
    }

    pub fn findings(&self) -> &[HistoricalAttestationFinding] {
        &self.findings
    }

    pub fn closure(&self) -> HistoricalAttestationClosure {
        self.closure
    }

    pub fn assessment_sha256(&self) -> &Sha256Digest {
        &self.assessment_sha256
    }

    /// Content-addressed lifecycle history is not authenticated lifecycle
    /// authority. TRUST-005A intentionally keeps this false.
    pub const fn lifecycle_authority_established(&self) -> bool {
        false
    }

    /// Structural support under caller-presented lifecycle histories is not yet
    /// an authority capability.
    pub const fn historical_validity_established(&self) -> bool {
        false
    }
}

pub fn assess_historical_attestation(
    historical: &VerifiedAttestation,
    policy: &AttestationPolicy,
    historical_snapshot: &TrustSnapshot,
    lifecycle_histories: &[KeyLifecycleHistory],
) -> HistoricalAttestationAssessment {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut incomplete = false;
    let mut threshold_invalidated = false;

    let policy_sha256 = match policy.digest() {
        Ok(value) => value,
        Err(AttestationPolicyError::InvalidPolicy) => {
            findings.push(HistoricalAttestationFinding::InvalidPolicy);
            invalid = true;
            historical.policy_sha256().clone()
        }
    };
    if &policy_sha256 != historical.policy_sha256() {
        findings.push(HistoricalAttestationFinding::PolicyIdentityMismatch);
        invalid = true;
    }

    let historical_snapshot_sha256 = match historical_snapshot.digest() {
        Ok(value) => value,
        Err(_) => {
            findings.push(HistoricalAttestationFinding::HistoricalTrustSnapshotInvalid);
            invalid = true;
            historical.trust_snapshot_sha256().clone()
        }
    };
    if &historical_snapshot_sha256 != historical.trust_snapshot_sha256() {
        findings.push(HistoricalAttestationFinding::HistoricalTrustSnapshotIdentityMismatch);
        invalid = true;
    }

    let expected_identities: BTreeSet<_> = historical.valid_signers().iter().cloned().collect();
    let mut histories = BTreeMap::new();
    for history in lifecycle_histories {
        let identity = (history.algorithm().clone(), history.key_id().to_string());
        if !expected_identities.contains(&identity) {
            findings.push(HistoricalAttestationFinding::UnexpectedLifecycleHistory {
                algorithm: identity.0,
                key_id: identity.1,
            });
            invalid = true;
            continue;
        }
        if histories.insert(identity.clone(), history).is_some() {
            findings.push(HistoricalAttestationFinding::DuplicateLifecycleHistory {
                algorithm: identity.0,
                key_id: identity.1,
            });
            invalid = true;
        }
    }

    let historical_time = historical.evaluation_time_unix_s();
    let mut surviving = Vec::new();
    for (algorithm, key_id) in historical.valid_signers() {
        let identity = (algorithm.clone(), key_id.clone());
        let Some(snapshot_key) = historical_snapshot.key_record(algorithm, key_id) else {
            findings.push(HistoricalAttestationFinding::MissingHistoricalKeyRecord {
                algorithm: algorithm.clone(),
                key_id: key_id.clone(),
            });
            invalid = true;
            continue;
        };
        let Some(history) = histories.get(&identity).copied() else {
            findings.push(HistoricalAttestationFinding::MissingLifecycleHistory {
                algorithm: algorithm.clone(),
                key_id: key_id.clone(),
            });
            incomplete = true;
            continue;
        };
        if history.algorithm() != algorithm || history.key_id() != key_id {
            findings.push(HistoricalAttestationFinding::LifecycleIdentityMismatch {
                algorithm: algorithm.clone(),
                key_id: key_id.clone(),
            });
            invalid = true;
            continue;
        }
        if history.verification_key_sha256() != &snapshot_key.verification_key_sha256 {
            findings.push(HistoricalAttestationFinding::VerificationKeyIdentityMismatch {
                algorithm: algorithm.clone(),
                key_id: key_id.clone(),
            });
            invalid = true;
            continue;
        }
        let status = history.status_at(historical_time);
        if status.permits_historical_authority() {
            surviving.push(identity);
        } else {
            findings.push(HistoricalAttestationFinding::HistoricalSignerNotAuthoritative {
                algorithm: algorithm.clone(),
                key_id: key_id.clone(),
                status: status.into(),
            });
        }
    }

    surviving.sort();
    let surviving_algorithms: BTreeSet<_> =
        surviving.iter().map(|(algorithm, _)| algorithm.clone()).collect();
    if surviving.len() < policy.minimum_valid_signatures {
        findings.push(
            HistoricalAttestationFinding::InsufficientHistoricallyValidSignatures {
                actual: surviving.len(),
                required: policy.minimum_valid_signatures,
            },
        );
        threshold_invalidated = true;
    }
    for algorithm in &policy.required_algorithms {
        if !surviving_algorithms.contains(algorithm) {
            findings.push(HistoricalAttestationFinding::MissingRequiredAlgorithm {
                algorithm: algorithm.clone(),
            });
            threshold_invalidated = true;
        }
    }

    let mut lifecycle_history_sha256s: Vec<_> = histories
        .values()
        .map(|history| history.history_sha256().clone())
        .collect();
    lifecycle_history_sha256s.sort();

    findings.sort_by_key(|finding| format!("{finding:?}"));
    let closure = if invalid {
        HistoricalAttestationClosure::Invalid
    } else if incomplete {
        HistoricalAttestationClosure::Incomplete
    } else if threshold_invalidated {
        HistoricalAttestationClosure::HistoricalAuthorityInvalidated
    } else {
        HistoricalAttestationClosure::StructurallySupported
    };
    let assessment_sha256 = assessment_digest(
        historical.authority_sha256(),
        &policy_sha256,
        &historical_snapshot_sha256,
        historical_time,
        &lifecycle_history_sha256s,
        &surviving,
        closure,
    );

    HistoricalAttestationAssessment {
        historical_authority_sha256: historical.authority_sha256().clone(),
        historical_policy_sha256: policy_sha256,
        historical_trust_snapshot_sha256: historical_snapshot_sha256,
        historical_evaluation_time_unix_s: historical_time,
        lifecycle_history_sha256s,
        surviving_historical_signers: surviving,
        findings,
        closure,
        assessment_sha256,
    }
}

fn assessment_digest(
    historical_authority_sha256: &Sha256Digest,
    policy_sha256: &Sha256Digest,
    historical_trust_snapshot_sha256: &Sha256Digest,
    historical_evaluation_time_unix_s: u64,
    lifecycle_history_sha256s: &[Sha256Digest],
    surviving_signers: &[(SignatureAlgorithm, String)],
    closure: HistoricalAttestationClosure,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(HISTORICAL_ATTESTATION_ASSESSMENT_DOMAIN);
    digest.text(historical_authority_sha256.as_str());
    digest.text(policy_sha256.as_str());
    digest.text(historical_trust_snapshot_sha256.as_str());
    digest.text(&historical_evaluation_time_unix_s.to_string());
    for history in lifecycle_history_sha256s {
        digest.text("lifecycle-history");
        digest.text(history.as_str());
    }
    for (algorithm, key_id) in surviving_signers {
        digest.text("surviving-signer");
        digest_algorithm(&mut digest, algorithm);
        digest.text(key_id);
    }
    digest.text(match closure {
        HistoricalAttestationClosure::StructurallySupported => "structurally-supported",
        HistoricalAttestationClosure::HistoricalAuthorityInvalidated => {
            "historical-authority-invalidated"
        }
        HistoricalAttestationClosure::Incomplete => "incomplete",
        HistoricalAttestationClosure::Invalid => "invalid",
    });
    digest.text("lifecycle-authority-not-established");
    digest.text("historical-validity-not-established");
    digest.digest()
}

fn digest_algorithm(digest: &mut FramedDigest, algorithm: &SignatureAlgorithm) {
    match algorithm {
        SignatureAlgorithm::Ed25519 => digest.text("builtin:ed25519"),
        SignatureAlgorithm::MlDsa65 => digest.text("builtin:ml-dsa-65"),
        SignatureAlgorithm::MlDsa87 => digest.text("builtin:ml-dsa-87"),
        SignatureAlgorithm::Other(name) => {
            digest.text("other");
            digest.text(name);
        }
    }
}
