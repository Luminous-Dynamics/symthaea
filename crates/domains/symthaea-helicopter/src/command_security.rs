// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated command freshness, sequencing, and replay protection.
//!
//! The hardware boundary already fails closed when transport or sensors are
//! unhealthy. This module closes the command-origin gap: a command is accepted
//! only when its origin, deployment, mission authority, timestamp, sequence,
//! payload digest, and external authenticity evidence are all valid. Signature
//! algorithms remain injected rather than being simulated here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedCommandEnvelope {
    pub schema_version: String,
    pub command_id: String,
    pub origin_id: String,
    pub deployment_id: String,
    pub mission_authority_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub payload_digest: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSecurityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub deployment_id: String,
    pub trusted_origins: Vec<String>,
    pub maximum_future_skew_ms: u64,
    pub maximum_command_age_ms: u64,
    pub maximum_sequence_gap: u64,
    pub require_monotonic_sequence: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionCommandAuthority {
    pub authority_id: String,
    pub deployment_id: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    pub permitted_origins: Vec<String>,
    pub revoked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandSecurityStatus {
    Accepted,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandSecurityIssue {
    InvalidEnvelope,
    InvalidDigest,
    DeploymentMismatch,
    UntrustedOrigin,
    AuthorityMismatch,
    AuthorityNotYetValid,
    AuthorityExpired,
    AuthorityRevoked,
    OriginNotPermitted,
    CommandFromFuture { skew_ms: u64, maximum_ms: u64 },
    CommandTooOld { age_ms: u64, maximum_ms: u64 },
    EnvelopeExpired,
    ReplayDetected { sequence: u64, last_accepted: u64 },
    SequenceGapTooLarge { gap: u64, maximum: u64 },
    AuthenticityRejected,
    AuthenticityUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSecurityDecision {
    pub command_id: String,
    pub origin_id: String,
    pub sequence: u64,
    pub status: CommandSecurityStatus,
    pub issues: Vec<CommandSecurityIssue>,
    pub accepted_at_ms: Option<u64>,
}

pub trait CommandAuthenticityVerifier {
    fn verify(
        &self,
        origin_id: &str,
        payload_digest: &str,
        signature: &[u8],
    ) -> Result<bool, CommandSecurityError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableCommandVerifier;

impl CommandAuthenticityVerifier for UnavailableCommandVerifier {
    fn verify(
        &self,
        _origin_id: &str,
        _payload_digest: &str,
        _signature: &[u8],
    ) -> Result<bool, CommandSecurityError> {
        Err(CommandSecurityError::VerifierUnavailable)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandSecurityError {
    InvalidPolicy,
    InvalidAuthority,
    VerifierUnavailable,
}

#[derive(Debug, Clone)]
pub struct CommandSecurityMonitor<V> {
    policy: CommandSecurityPolicy,
    verifier: V,
    last_sequence_by_origin: BTreeMap<String, u64>,
}

impl<V: CommandAuthenticityVerifier> CommandSecurityMonitor<V> {
    pub fn new(policy: CommandSecurityPolicy, verifier: V) -> Result<Self, CommandSecurityError> {
        let trusted: BTreeSet<_> = policy.trusted_origins.iter().collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.deployment_id.trim().is_empty()
            || policy.trusted_origins.is_empty()
            || trusted.len() != policy.trusted_origins.len()
            || policy.maximum_command_age_ms == 0
            || policy.maximum_sequence_gap == 0
        {
            return Err(CommandSecurityError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            verifier,
            last_sequence_by_origin: BTreeMap::new(),
        })
    }

    pub fn evaluate(
        &mut self,
        envelope: &AuthenticatedCommandEnvelope,
        authority: &MissionCommandAuthority,
        now_ms: u64,
    ) -> Result<CommandSecurityDecision, CommandSecurityError> {
        validate_authority(authority)?;
        let mut issues = Vec::new();
        if envelope.schema_version.trim().is_empty()
            || envelope.command_id.trim().is_empty()
            || envelope.origin_id.trim().is_empty()
            || envelope.mission_authority_id.trim().is_empty()
            || envelope.issued_at_ms > envelope.expires_at_ms
        {
            issues.push(CommandSecurityIssue::InvalidEnvelope);
        }
        if !valid_digest(&envelope.payload_digest) {
            issues.push(CommandSecurityIssue::InvalidDigest);
        }
        if envelope.deployment_id != self.policy.deployment_id
            || authority.deployment_id != self.policy.deployment_id
        {
            issues.push(CommandSecurityIssue::DeploymentMismatch);
        }
        if !self.policy.trusted_origins.contains(&envelope.origin_id) {
            issues.push(CommandSecurityIssue::UntrustedOrigin);
        }
        if envelope.mission_authority_id != authority.authority_id {
            issues.push(CommandSecurityIssue::AuthorityMismatch);
        }
        if authority.revoked {
            issues.push(CommandSecurityIssue::AuthorityRevoked);
        }
        if now_ms < authority.valid_from_ms {
            issues.push(CommandSecurityIssue::AuthorityNotYetValid);
        }
        if now_ms > authority.valid_until_ms {
            issues.push(CommandSecurityIssue::AuthorityExpired);
        }
        if !authority.permitted_origins.contains(&envelope.origin_id) {
            issues.push(CommandSecurityIssue::OriginNotPermitted);
        }
        if envelope.issued_at_ms > now_ms {
            let skew = envelope.issued_at_ms - now_ms;
            if skew > self.policy.maximum_future_skew_ms {
                issues.push(CommandSecurityIssue::CommandFromFuture {
                    skew_ms: skew,
                    maximum_ms: self.policy.maximum_future_skew_ms,
                });
            }
        } else {
            let age = now_ms - envelope.issued_at_ms;
            if age > self.policy.maximum_command_age_ms {
                issues.push(CommandSecurityIssue::CommandTooOld {
                    age_ms: age,
                    maximum_ms: self.policy.maximum_command_age_ms,
                });
            }
        }
        if now_ms > envelope.expires_at_ms {
            issues.push(CommandSecurityIssue::EnvelopeExpired);
        }
        if let Some(last) = self
            .last_sequence_by_origin
            .get(&envelope.origin_id)
            .copied()
        {
            if self.policy.require_monotonic_sequence && envelope.sequence <= last {
                issues.push(CommandSecurityIssue::ReplayDetected {
                    sequence: envelope.sequence,
                    last_accepted: last,
                });
            } else {
                let gap = envelope.sequence.saturating_sub(last);
                if gap > self.policy.maximum_sequence_gap {
                    issues.push(CommandSecurityIssue::SequenceGapTooLarge {
                        gap,
                        maximum: self.policy.maximum_sequence_gap,
                    });
                }
            }
        }

        match self.verifier.verify(
            &envelope.origin_id,
            &envelope.payload_digest,
            &envelope.signature,
        ) {
            Ok(true) => {}
            Ok(false) => issues.push(CommandSecurityIssue::AuthenticityRejected),
            Err(CommandSecurityError::VerifierUnavailable) => {
                issues.push(CommandSecurityIssue::AuthenticityUnavailable)
            }
            Err(error) => return Err(error),
        }

        let incomplete = issues
            .iter()
            .any(|issue| matches!(issue, CommandSecurityIssue::AuthenticityUnavailable));
        let status = if issues.is_empty() {
            CommandSecurityStatus::Accepted
        } else if incomplete && issues.len() == 1 {
            CommandSecurityStatus::Incomplete
        } else {
            CommandSecurityStatus::Rejected
        };
        if status == CommandSecurityStatus::Accepted {
            self.last_sequence_by_origin
                .insert(envelope.origin_id.clone(), envelope.sequence);
        }
        Ok(CommandSecurityDecision {
            command_id: envelope.command_id.clone(),
            origin_id: envelope.origin_id.clone(),
            sequence: envelope.sequence,
            status,
            issues,
            accepted_at_ms: (status == CommandSecurityStatus::Accepted).then_some(now_ms),
        })
    }

    pub fn reset_origin(&mut self, origin_id: &str) {
        self.last_sequence_by_origin.remove(origin_id);
    }
}

fn validate_authority(authority: &MissionCommandAuthority) -> Result<(), CommandSecurityError> {
    let origins: BTreeSet<_> = authority.permitted_origins.iter().collect();
    if authority.authority_id.trim().is_empty()
        || authority.deployment_id.trim().is_empty()
        || authority.valid_from_ms > authority.valid_until_ms
        || authority.permitted_origins.is_empty()
        || origins.len() != authority.permitted_origins.len()
    {
        return Err(CommandSecurityError::InvalidAuthority);
    }
    Ok(())
}

fn valid_digest(digest: &str) -> bool {
    let Some((algorithm, value)) = digest.split_once(':') else {
        return false;
    };
    !algorithm.trim().is_empty()
        && value.len() >= 16
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct AcceptVerifier;
    impl CommandAuthenticityVerifier for AcceptVerifier {
        fn verify(
            &self,
            _origin_id: &str,
            _payload_digest: &str,
            signature: &[u8],
        ) -> Result<bool, CommandSecurityError> {
            Ok(signature == b"valid")
        }
    }

    fn monitor() -> CommandSecurityMonitor<AcceptVerifier> {
        CommandSecurityMonitor::new(
            CommandSecurityPolicy {
                schema_version: "1".into(),
                policy_id: "command-security".into(),
                deployment_id: "aircraft-1".into(),
                trusted_origins: vec!["operator".into()],
                maximum_future_skew_ms: 100,
                maximum_command_age_ms: 500,
                maximum_sequence_gap: 10,
                require_monotonic_sequence: true,
            },
            AcceptVerifier,
        )
        .unwrap()
    }

    fn authority() -> MissionCommandAuthority {
        MissionCommandAuthority {
            authority_id: "mission-1".into(),
            deployment_id: "aircraft-1".into(),
            valid_from_ms: 0,
            valid_until_ms: 10_000,
            permitted_origins: vec!["operator".into()],
            revoked: false,
        }
    }

    fn envelope(sequence: u64) -> AuthenticatedCommandEnvelope {
        AuthenticatedCommandEnvelope {
            schema_version: "1".into(),
            command_id: format!("cmd-{sequence}"),
            origin_id: "operator".into(),
            deployment_id: "aircraft-1".into(),
            mission_authority_id: "mission-1".into(),
            sequence,
            issued_at_ms: 1_000,
            expires_at_ms: 2_000,
            payload_digest: "sha256:0123456789abcdef".into(),
            signature: b"valid".to_vec(),
        }
    }

    #[test]
    fn valid_command_is_accepted() {
        let decision = monitor()
            .evaluate(&envelope(1), &authority(), 1_100)
            .unwrap();
        assert_eq!(decision.status, CommandSecurityStatus::Accepted);
    }

    #[test]
    fn repeated_sequence_is_rejected() {
        let mut monitor = monitor();
        monitor.evaluate(&envelope(1), &authority(), 1_100).unwrap();
        let replay = monitor.evaluate(&envelope(1), &authority(), 1_101).unwrap();
        assert_eq!(replay.status, CommandSecurityStatus::Rejected);
        assert!(
            replay
                .issues
                .iter()
                .any(|issue| matches!(issue, CommandSecurityIssue::ReplayDetected { .. }))
        );
    }

    #[test]
    fn expired_authority_rejects_command() {
        let mut authority = authority();
        authority.valid_until_ms = 1_050;
        let decision = monitor().evaluate(&envelope(1), &authority, 1_100).unwrap();
        assert_eq!(decision.status, CommandSecurityStatus::Rejected);
    }
}
