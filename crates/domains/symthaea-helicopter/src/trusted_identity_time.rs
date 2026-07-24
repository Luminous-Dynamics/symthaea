// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trusted device identity and time evidence.
//!
//! Authentication and signature verification remain external cryptographic
//! responsibilities. This module verifies that already-authenticated identity,
//! boot, monotonic-counter, and time-source evidence is coherent and fresh.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustedTimeSourceKind {
    HardwareRtc,
    Gnss,
    NetworkAuthenticated,
    GroundStation,
    MonotonicDerived,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIdentityEvidence {
    pub aircraft_id: String,
    pub device_id: String,
    pub hardware_root_id: String,
    pub deployment_digest: String,
    pub secure_boot_verified: bool,
    pub identity_authenticity_evidence_id: Option<String>,
    pub boot_counter: u64,
    pub monotonic_counter: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedTimeObservation {
    pub source_id: String,
    pub source_kind: TrustedTimeSourceKind,
    pub failure_domain: String,
    pub authenticated: bool,
    pub observed_unix_ms: u64,
    pub received_monotonic_ms: u64,
    pub uncertainty_ms: u64,
    pub evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedIdentityTimePolicy {
    pub expected_aircraft_id: String,
    pub expected_device_id: String,
    pub expected_hardware_root_id: String,
    pub expected_deployment_digest: String,
    pub minimum_time_sources: usize,
    pub minimum_failure_domains: usize,
    pub maximum_source_age_ms: u64,
    pub maximum_pairwise_skew_ms: u64,
    pub maximum_uncertainty_ms: u64,
    pub require_secure_boot: bool,
    pub require_authenticated_time: bool,
}

impl Default for TrustedIdentityTimePolicy {
    fn default() -> Self {
        Self {
            expected_aircraft_id: String::new(),
            expected_device_id: String::new(),
            expected_hardware_root_id: String::new(),
            expected_deployment_digest: String::new(),
            minimum_time_sources: 2,
            minimum_failure_domains: 2,
            maximum_source_age_ms: 5_000,
            maximum_pairwise_skew_ms: 1_000,
            maximum_uncertainty_ms: 500,
            require_secure_boot: true,
            require_authenticated_time: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustedIdentityTimeStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustedIdentityTimeIssue {
    AircraftIdentityMismatch,
    DeviceIdentityMismatch,
    HardwareRootMismatch,
    DeploymentDigestMismatch,
    SecureBootNotVerified,
    MissingIdentityAuthenticityEvidence,
    BootCounterRollback {
        previous: u64,
        observed: u64,
    },
    MonotonicCounterRollback {
        previous: u64,
        observed: u64,
    },
    DuplicateTimeSource {
        source_id: String,
    },
    StaleTimeSource {
        source_id: String,
        age_ms: u64,
    },
    UnauthenticatedTimeSource {
        source_id: String,
    },
    ExcessiveTimeUncertainty {
        source_id: String,
        uncertainty_ms: u64,
    },
    MissingTimeEvidence {
        source_id: String,
    },
    InsufficientTimeSources {
        required: usize,
        observed: usize,
    },
    InsufficientFailureDomains {
        required: usize,
        observed: usize,
    },
    ExcessiveTimeSkew {
        source_a: String,
        source_b: String,
        skew_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedIdentityTimeReport {
    pub status: TrustedIdentityTimeStatus,
    pub accepted_time_ms: Option<u64>,
    pub accepted_uncertainty_ms: Option<u64>,
    pub accepted_source_ids: Vec<String>,
    pub issues: Vec<TrustedIdentityTimeIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustedIdentityTimeError {
    InvalidPolicy,
    EmptyIdentity,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedIdentityTimeState {
    pub last_boot_counter: Option<u64>,
    pub last_monotonic_counter: Option<u64>,
    pub last_accepted_time_ms: Option<u64>,
}

pub struct TrustedIdentityTimeVerifier {
    policy: TrustedIdentityTimePolicy,
    state: TrustedIdentityTimeState,
}

impl TrustedIdentityTimeVerifier {
    pub fn new(
        policy: TrustedIdentityTimePolicy,
        state: TrustedIdentityTimeState,
    ) -> Result<Self, TrustedIdentityTimeError> {
        if policy.expected_aircraft_id.trim().is_empty()
            || policy.expected_device_id.trim().is_empty()
            || policy.expected_hardware_root_id.trim().is_empty()
            || policy.expected_deployment_digest.trim().is_empty()
        {
            return Err(TrustedIdentityTimeError::EmptyIdentity);
        }
        if policy.minimum_time_sources == 0
            || policy.minimum_failure_domains == 0
            || policy.maximum_pairwise_skew_ms == 0
        {
            return Err(TrustedIdentityTimeError::InvalidPolicy);
        }
        Ok(Self { policy, state })
    }

    pub fn state(&self) -> &TrustedIdentityTimeState {
        &self.state
    }

    pub fn verify(
        &mut self,
        identity: &DeviceIdentityEvidence,
        observations: &[TrustedTimeObservation],
        now_monotonic_ms: u64,
    ) -> TrustedIdentityTimeReport {
        let mut issues = Vec::new();
        self.verify_identity(identity, &mut issues);

        let mut by_source = BTreeMap::<&str, &TrustedTimeObservation>::new();
        for observation in observations {
            if by_source
                .insert(observation.source_id.as_str(), observation)
                .is_some()
            {
                issues.push(TrustedIdentityTimeIssue::DuplicateTimeSource {
                    source_id: observation.source_id.clone(),
                });
            }
        }

        let mut accepted = Vec::<&TrustedTimeObservation>::new();
        for observation in by_source.values() {
            let age_ms = now_monotonic_ms.saturating_sub(observation.received_monotonic_ms);
            let mut usable = true;
            if age_ms > self.policy.maximum_source_age_ms {
                issues.push(TrustedIdentityTimeIssue::StaleTimeSource {
                    source_id: observation.source_id.clone(),
                    age_ms,
                });
                usable = false;
            }
            if self.policy.require_authenticated_time && !observation.authenticated {
                issues.push(TrustedIdentityTimeIssue::UnauthenticatedTimeSource {
                    source_id: observation.source_id.clone(),
                });
                usable = false;
            }
            if observation.uncertainty_ms > self.policy.maximum_uncertainty_ms {
                issues.push(TrustedIdentityTimeIssue::ExcessiveTimeUncertainty {
                    source_id: observation.source_id.clone(),
                    uncertainty_ms: observation.uncertainty_ms,
                });
                usable = false;
            }
            if observation
                .evidence_id
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
            {
                issues.push(TrustedIdentityTimeIssue::MissingTimeEvidence {
                    source_id: observation.source_id.clone(),
                });
                usable = false;
            }
            if usable {
                accepted.push(*observation);
            }
        }

        if accepted.len() < self.policy.minimum_time_sources {
            issues.push(TrustedIdentityTimeIssue::InsufficientTimeSources {
                required: self.policy.minimum_time_sources,
                observed: accepted.len(),
            });
        }
        let domains = accepted
            .iter()
            .map(|observation| observation.failure_domain.as_str())
            .collect::<BTreeSet<_>>();
        if domains.len() < self.policy.minimum_failure_domains {
            issues.push(TrustedIdentityTimeIssue::InsufficientFailureDomains {
                required: self.policy.minimum_failure_domains,
                observed: domains.len(),
            });
        }

        for (index, left) in accepted.iter().enumerate() {
            for right in accepted.iter().skip(index + 1) {
                let skew_ms = left.observed_unix_ms.abs_diff(right.observed_unix_ms);
                if skew_ms > self.policy.maximum_pairwise_skew_ms {
                    issues.push(TrustedIdentityTimeIssue::ExcessiveTimeSkew {
                        source_a: left.source_id.clone(),
                        source_b: right.source_id.clone(),
                        skew_ms,
                    });
                }
            }
        }

        let has_fail = issues.iter().any(issue_is_failure);
        let accepted_time_ms = if !has_fail
            && accepted.len() >= self.policy.minimum_time_sources
            && domains.len() >= self.policy.minimum_failure_domains
        {
            let mut times = accepted
                .iter()
                .map(|observation| observation.observed_unix_ms)
                .collect::<Vec<_>>();
            times.sort_unstable();
            Some(times[times.len() / 2])
        } else {
            None
        };
        let accepted_uncertainty_ms = accepted_time_ms.map(|time_ms| {
            accepted
                .iter()
                .map(|observation| {
                    observation.observed_unix_ms.abs_diff(time_ms) + observation.uncertainty_ms
                })
                .max()
                .unwrap_or(0)
        });

        if let Some(time_ms) = accepted_time_ms {
            self.state.last_boot_counter = Some(identity.boot_counter);
            self.state.last_monotonic_counter = Some(identity.monotonic_counter);
            self.state.last_accepted_time_ms = Some(time_ms);
        }

        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let status = if has_fail {
            TrustedIdentityTimeStatus::Fail
        } else if issues.is_empty() {
            TrustedIdentityTimeStatus::Pass
        } else {
            TrustedIdentityTimeStatus::Incomplete
        };
        let mut accepted_source_ids = accepted
            .iter()
            .map(|observation| observation.source_id.clone())
            .collect::<Vec<_>>();
        accepted_source_ids.sort();

        TrustedIdentityTimeReport {
            status,
            accepted_time_ms,
            accepted_uncertainty_ms,
            accepted_source_ids,
            issues,
        }
    }

    fn verify_identity(
        &self,
        identity: &DeviceIdentityEvidence,
        issues: &mut Vec<TrustedIdentityTimeIssue>,
    ) {
        if identity.aircraft_id != self.policy.expected_aircraft_id {
            issues.push(TrustedIdentityTimeIssue::AircraftIdentityMismatch);
        }
        if identity.device_id != self.policy.expected_device_id {
            issues.push(TrustedIdentityTimeIssue::DeviceIdentityMismatch);
        }
        if identity.hardware_root_id != self.policy.expected_hardware_root_id {
            issues.push(TrustedIdentityTimeIssue::HardwareRootMismatch);
        }
        if identity.deployment_digest != self.policy.expected_deployment_digest {
            issues.push(TrustedIdentityTimeIssue::DeploymentDigestMismatch);
        }
        if self.policy.require_secure_boot && !identity.secure_boot_verified {
            issues.push(TrustedIdentityTimeIssue::SecureBootNotVerified);
        }
        if identity
            .identity_authenticity_evidence_id
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            issues.push(TrustedIdentityTimeIssue::MissingIdentityAuthenticityEvidence);
        }
        if let Some(previous) = self.state.last_boot_counter {
            if identity.boot_counter < previous {
                issues.push(TrustedIdentityTimeIssue::BootCounterRollback {
                    previous,
                    observed: identity.boot_counter,
                });
            }
        }
        if let Some(previous) = self.state.last_monotonic_counter {
            if identity.monotonic_counter <= previous {
                issues.push(TrustedIdentityTimeIssue::MonotonicCounterRollback {
                    previous,
                    observed: identity.monotonic_counter,
                });
            }
        }
    }
}

fn issue_is_failure(issue: &TrustedIdentityTimeIssue) -> bool {
    matches!(
        issue,
        TrustedIdentityTimeIssue::AircraftIdentityMismatch
            | TrustedIdentityTimeIssue::DeviceIdentityMismatch
            | TrustedIdentityTimeIssue::HardwareRootMismatch
            | TrustedIdentityTimeIssue::DeploymentDigestMismatch
            | TrustedIdentityTimeIssue::SecureBootNotVerified
            | TrustedIdentityTimeIssue::BootCounterRollback { .. }
            | TrustedIdentityTimeIssue::MonotonicCounterRollback { .. }
            | TrustedIdentityTimeIssue::DuplicateTimeSource { .. }
            | TrustedIdentityTimeIssue::ExcessiveTimeSkew { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> TrustedIdentityTimePolicy {
        TrustedIdentityTimePolicy {
            expected_aircraft_id: "aircraft-1".into(),
            expected_device_id: "fcc-1".into(),
            expected_hardware_root_id: "tpm-1".into(),
            expected_deployment_digest: "sha256:deployment".into(),
            ..TrustedIdentityTimePolicy::default()
        }
    }

    fn identity() -> DeviceIdentityEvidence {
        DeviceIdentityEvidence {
            aircraft_id: "aircraft-1".into(),
            device_id: "fcc-1".into(),
            hardware_root_id: "tpm-1".into(),
            deployment_digest: "sha256:deployment".into(),
            secure_boot_verified: true,
            identity_authenticity_evidence_id: Some("attestation-1".into()),
            boot_counter: 7,
            monotonic_counter: 10,
        }
    }

    fn times() -> Vec<TrustedTimeObservation> {
        vec![
            TrustedTimeObservation {
                source_id: "gnss".into(),
                source_kind: TrustedTimeSourceKind::Gnss,
                failure_domain: "space".into(),
                authenticated: true,
                observed_unix_ms: 1_000_000,
                received_monotonic_ms: 900,
                uncertainty_ms: 20,
                evidence_id: Some("time-1".into()),
            },
            TrustedTimeObservation {
                source_id: "rtc".into(),
                source_kind: TrustedTimeSourceKind::HardwareRtc,
                failure_domain: "local".into(),
                authenticated: true,
                observed_unix_ms: 1_000_030,
                received_monotonic_ms: 910,
                uncertainty_ms: 50,
                evidence_id: Some("time-2".into()),
            },
        ]
    }

    #[test]
    fn coherent_identity_and_time_pass() {
        let mut verifier =
            TrustedIdentityTimeVerifier::new(policy(), TrustedIdentityTimeState::default())
                .unwrap();
        let report = verifier.verify(&identity(), &times(), 1_000);
        assert_eq!(report.status, TrustedIdentityTimeStatus::Pass);
        assert_eq!(report.accepted_time_ms, Some(1_000_030));
    }

    #[test]
    fn counter_rollback_fails() {
        let state = TrustedIdentityTimeState {
            last_boot_counter: Some(8),
            last_monotonic_counter: Some(11),
            last_accepted_time_ms: None,
        };
        let mut verifier = TrustedIdentityTimeVerifier::new(policy(), state).unwrap();
        let report = verifier.verify(&identity(), &times(), 1_000);
        assert_eq!(report.status, TrustedIdentityTimeStatus::Fail);
    }

    #[test]
    fn single_failure_domain_is_incomplete() {
        let mut observations = times();
        observations[1].failure_domain = "space".into();
        let mut verifier =
            TrustedIdentityTimeVerifier::new(policy(), TrustedIdentityTimeState::default())
                .unwrap();
        let report = verifier.verify(&identity(), &observations, 1_000);
        assert_eq!(report.status, TrustedIdentityTimeStatus::Incomplete);
    }
}
