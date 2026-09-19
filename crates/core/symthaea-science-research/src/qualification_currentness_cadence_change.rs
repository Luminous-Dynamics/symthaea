// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Semantic safety proof for currentness-cadence policy evolution.
//!
//! Publishing policy changes makes weakening visible, but visibility alone does
//! not make weakening safe. The ordinary currentness-extension path therefore
//! accepts only an exact authorized cadence-policy lineage whose safety limits are
//! monotonically non-weakening from genesis through the latest policy.
//!
//! V1 has no ordinary relaxation path. A future explicit exceptional authority
//! (for example an independently governed emergency/recovery role) may define a
//! separately auditable relaxation theorem. Ordinary `QualificationLifecycle`
//! authority is insufficient by itself to weaken cadence.

use serde::Serialize;
use symthaea_trust_core::{FramedDigest, Sha256Digest as TrustSha256Digest};

use crate::{
    AuthorizedQualificationCurrentnessCadencePolicy, QualificationCurrentnessCadenceLimits,
    Sha256Digest,
};

const CADENCE_CHANGE_SAFETY_DOMAIN: &str =
    "symthaea.currentness-cadence-change-safety.identity.v1";
pub const MAX_CADENCE_CHANGE_POLICIES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QualificationCurrentnessCadenceChangeFinding {
    EmptyLineage,
    TooManyPolicies,
    QualificationMismatch { version: u64 },
    VersionMismatch { expected: u64, actual: u64 },
    GenesisHasPredecessor,
    PredecessorMismatch { version: u64 },
    CurrentnessAgeWeakened { version: u64 },
    HeadAgeWeakened { version: u64 },
    LifecycleCheckpointAgeWeakened { version: u64 },
    EvidenceSearchLagWeakened { version: u64 },
    ConvergedViewsWeakened { version: u64 },
    DistinctQuorumsWeakened { version: u64 },
    DistinctPrincipalsWeakened { version: u64 },
    DistinctOrganizationsWeakened { version: u64 },
    DistinctRegionsWeakened { version: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NonWeakeningQualificationCurrentnessCadenceLineage {
    qualification_sha256: Sha256Digest,
    latest_policy_version: u64,
    latest_policy_sha256: TrustSha256Digest,
    latest_policy_authority_sha256: TrustSha256Digest,
    policy_sha256s: Vec<TrustSha256Digest>,
    policy_authority_sha256s: Vec<TrustSha256Digest>,
    lineage_sha256: TrustSha256Digest,
}

impl NonWeakeningQualificationCurrentnessCadenceLineage {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn latest_policy_version(&self) -> u64 { self.latest_policy_version }
    pub fn latest_policy_sha256(&self) -> &TrustSha256Digest { &self.latest_policy_sha256 }
    pub fn latest_policy_authority_sha256(&self) -> &TrustSha256Digest {
        &self.latest_policy_authority_sha256
    }
    pub fn policy_sha256s(&self) -> &[TrustSha256Digest] { &self.policy_sha256s }
    pub fn policy_authority_sha256s(&self) -> &[TrustSha256Digest] {
        &self.policy_authority_sha256s
    }
    pub fn lineage_sha256(&self) -> &TrustSha256Digest { &self.lineage_sha256 }

    pub const fn non_weakening_lineage_established(&self) -> bool { true }
    pub const fn relaxation_authority_established(&self) -> bool { false }
    pub const fn currentness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn verify_non_weakening_currentness_cadence_lineage(
    policies: &[AuthorizedQualificationCurrentnessCadencePolicy],
) -> Result<NonWeakeningQualificationCurrentnessCadenceLineage, Vec<QualificationCurrentnessCadenceChangeFinding>> {
    let mut findings = Vec::new();
    if policies.is_empty() {
        return Err(vec![QualificationCurrentnessCadenceChangeFinding::EmptyLineage]);
    }
    if policies.len() > MAX_CADENCE_CHANGE_POLICIES {
        findings.push(QualificationCurrentnessCadenceChangeFinding::TooManyPolicies);
    }

    let qualification_sha256 = policies[0].qualification_sha256().clone();
    let mut previous: Option<&AuthorizedQualificationCurrentnessCadencePolicy> = None;
    let mut policy_sha256s = Vec::with_capacity(policies.len());
    let mut policy_authority_sha256s = Vec::with_capacity(policies.len());

    for (index, policy) in policies.iter().enumerate() {
        let expected_version = u64::try_from(index)
            .ok()
            .and_then(|value| value.checked_add(1))
            .unwrap_or(u64::MAX);
        if policy.qualification_sha256() != &qualification_sha256 {
            findings.push(QualificationCurrentnessCadenceChangeFinding::QualificationMismatch {
                version: policy.version(),
            });
        }
        if policy.version() != expected_version {
            findings.push(QualificationCurrentnessCadenceChangeFinding::VersionMismatch {
                expected: expected_version,
                actual: policy.version(),
            });
        }

        match previous {
            None => {
                if policy.policy().predecessor_policy_sha256().is_some() {
                    findings.push(QualificationCurrentnessCadenceChangeFinding::GenesisHasPredecessor);
                }
            }
            Some(previous_policy) => {
                if policy.policy().predecessor_policy_sha256()
                    != Some(previous_policy.policy_sha256())
                {
                    findings.push(QualificationCurrentnessCadenceChangeFinding::PredecessorMismatch {
                        version: policy.version(),
                    });
                }
                compare_limits(
                    previous_policy.policy().limits(),
                    policy.policy().limits(),
                    policy.version(),
                    &mut findings,
                );
            }
        }

        policy_sha256s.push(policy.policy_sha256().clone());
        policy_authority_sha256s.push(policy.authority_sha256().clone());
        previous = Some(policy);
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    let latest = policies
        .last()
        .expect("non-empty cadence lineage checked above");
    let lineage_sha256 = cadence_change_safety_digest(
        &qualification_sha256,
        &policy_sha256s,
        &policy_authority_sha256s,
    );
    Ok(NonWeakeningQualificationCurrentnessCadenceLineage {
        qualification_sha256,
        latest_policy_version: latest.version(),
        latest_policy_sha256: latest.policy_sha256().clone(),
        latest_policy_authority_sha256: latest.authority_sha256().clone(),
        policy_sha256s,
        policy_authority_sha256s,
        lineage_sha256,
    })
}

fn compare_limits(
    previous: &QualificationCurrentnessCadenceLimits,
    next: &QualificationCurrentnessCadenceLimits,
    version: u64,
    findings: &mut Vec<QualificationCurrentnessCadenceChangeFinding>,
) {
    if next.maximum_currentness_age_s > previous.maximum_currentness_age_s {
        findings.push(QualificationCurrentnessCadenceChangeFinding::CurrentnessAgeWeakened {
            version,
        });
    }
    if next.maximum_head_age_s > previous.maximum_head_age_s {
        findings.push(QualificationCurrentnessCadenceChangeFinding::HeadAgeWeakened { version });
    }
    if next.maximum_lifecycle_checkpoint_age_s
        > previous.maximum_lifecycle_checkpoint_age_s
    {
        findings.push(
            QualificationCurrentnessCadenceChangeFinding::LifecycleCheckpointAgeWeakened {
                version,
            },
        );
    }
    if next.maximum_evidence_search_lag_ms > previous.maximum_evidence_search_lag_ms {
        findings.push(QualificationCurrentnessCadenceChangeFinding::EvidenceSearchLagWeakened {
            version,
        });
    }
    if next.minimum_converged_views < previous.minimum_converged_views {
        findings.push(QualificationCurrentnessCadenceChangeFinding::ConvergedViewsWeakened {
            version,
        });
    }
    if next.minimum_distinct_quorums < previous.minimum_distinct_quorums {
        findings.push(QualificationCurrentnessCadenceChangeFinding::DistinctQuorumsWeakened {
            version,
        });
    }
    if next.minimum_distinct_principals < previous.minimum_distinct_principals {
        findings.push(QualificationCurrentnessCadenceChangeFinding::DistinctPrincipalsWeakened {
            version,
        });
    }
    if next.minimum_distinct_organizations < previous.minimum_distinct_organizations {
        findings.push(
            QualificationCurrentnessCadenceChangeFinding::DistinctOrganizationsWeakened {
                version,
            },
        );
    }
    if next.minimum_distinct_regions < previous.minimum_distinct_regions {
        findings.push(QualificationCurrentnessCadenceChangeFinding::DistinctRegionsWeakened {
            version,
        });
    }
}

fn cadence_change_safety_digest(
    qualification_sha256: &Sha256Digest,
    policy_sha256s: &[TrustSha256Digest],
    policy_authority_sha256s: &[TrustSha256Digest],
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CADENCE_CHANGE_SAFETY_DOMAIN);
    digest.text(qualification_sha256.as_str());
    for (policy, authority) in policy_sha256s.iter().zip(policy_authority_sha256s) {
        digest.text("policy");
        digest.text(policy.as_str());
        digest.text(authority.as_str());
    }
    digest.text("non-weakening-lineage-established");
    digest.text("relaxation-authority-not-established");
    digest.text("currentness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tighter_maxima_and_stronger_minima_are_non_weakening() {
        let previous = QualificationCurrentnessCadenceLimits {
            maximum_currentness_age_s: 3600,
            maximum_head_age_s: 300,
            maximum_lifecycle_checkpoint_age_s: 300,
            maximum_evidence_search_lag_ms: 60_000,
            minimum_converged_views: 2,
            minimum_distinct_quorums: 2,
            minimum_distinct_principals: 2,
            minimum_distinct_organizations: 2,
            minimum_distinct_regions: 1,
        };
        let next = QualificationCurrentnessCadenceLimits {
            maximum_currentness_age_s: 1800,
            maximum_head_age_s: 120,
            maximum_lifecycle_checkpoint_age_s: 120,
            maximum_evidence_search_lag_ms: 30_000,
            minimum_converged_views: 3,
            minimum_distinct_quorums: 3,
            minimum_distinct_principals: 3,
            minimum_distinct_organizations: 2,
            minimum_distinct_regions: 2,
        };
        let mut findings = Vec::new();
        compare_limits(&previous, &next, 2, &mut findings);
        assert!(findings.is_empty());
    }

    #[test]
    fn widening_currentness_age_is_detected_as_weakening() {
        let previous = QualificationCurrentnessCadenceLimits {
            maximum_currentness_age_s: 60,
            maximum_head_age_s: 60,
            maximum_lifecycle_checkpoint_age_s: 60,
            maximum_evidence_search_lag_ms: 1000,
            minimum_converged_views: 2,
            minimum_distinct_quorums: 2,
            minimum_distinct_principals: 2,
            minimum_distinct_organizations: 1,
            minimum_distinct_regions: 1,
        };
        let mut next = previous.clone();
        next.maximum_currentness_age_s = 61;
        let mut findings = Vec::new();
        compare_limits(&previous, &next, 2, &mut findings);
        assert!(findings.iter().any(|finding| matches!(
            finding,
            QualificationCurrentnessCadenceChangeFinding::CurrentnessAgeWeakened { .. }
        )));
    }
}
