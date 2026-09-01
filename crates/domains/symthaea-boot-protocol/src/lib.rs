// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed, privacy-preserving boot observability protocol.
//!
//! This crate deliberately carries normalized boot state rather than raw journal
//! lines, process metadata, network identifiers, or arbitrary strings. The OS
//! remains authoritative; consumers such as the boot renderer are presentation
//! layers only.

#![forbid(unsafe_code)]

pub mod prelude;
pub mod state;
pub mod wire;

use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_DETAIL_BYTES: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BootPhase {
    Kernel,
    Initrd,
    Storage,
    Filesystems,
    Security,
    Network,
    Services,
    Graphics,
    Session,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BootDomain {
    Kernel,
    Initrd,
    Storage,
    Filesystems,
    Security,
    Network,
    Services,
    Graphics,
    Session,
}

impl From<BootPhase> for Option<BootDomain> {
    fn from(value: BootPhase) -> Self {
        match value {
            BootPhase::Kernel => Some(BootDomain::Kernel),
            BootPhase::Initrd => Some(BootDomain::Initrd),
            BootPhase::Storage => Some(BootDomain::Storage),
            BootPhase::Filesystems => Some(BootDomain::Filesystems),
            BootPhase::Security => Some(BootDomain::Security),
            BootPhase::Network => Some(BootDomain::Network),
            BootPhase::Services => Some(BootDomain::Services),
            BootPhase::Graphics => Some(BootDomain::Graphics),
            BootPhase::Session => Some(BootDomain::Session),
            BootPhase::Ready => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BootHealth {
    Normal,
    Delayed,
    Degraded,
    Failed,
    Unknown,
}

impl BootHealth {
    /// Severity is used only by presentation reducers. `Unknown` is deliberately
    /// worse than `Normal` so absent telemetry is never rendered as healthy.
    pub const fn severity(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Unknown => 1,
            Self::Delayed => 2,
            Self::Degraded => 3,
            Self::Failed => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Criticality {
    Informational,
    NonCritical,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootSnapshot {
    pub protocol_version: u16,
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub phase: BootPhase,
    pub health: BootHealth,
    pub domains: Vec<DomainSnapshot>,
}

impl BootSnapshot {
    pub fn new(sequence: u64, elapsed: Duration, phase: BootPhase) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            sequence,
            elapsed_ms: saturating_millis(elapsed),
            phase,
            health: BootHealth::Unknown,
            domains: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion(self.protocol_version));
        }
        if self.domains.len() > BootDomain::COUNT {
            return Err(ProtocolError::TooManyDomains(self.domains.len()));
        }

        let mut seen = [false; BootDomain::COUNT];
        for domain in &self.domains {
            let index = domain.domain.index();
            if seen[index] {
                return Err(ProtocolError::DuplicateDomain(domain.domain));
            }
            seen[index] = true;

            if let Some(domain_elapsed_ms) = domain.elapsed_ms {
                if domain_elapsed_ms > self.elapsed_ms {
                    return Err(ProtocolError::DomainElapsedAfterSnapshot {
                        domain: domain.domain,
                        domain_elapsed_ms,
                        snapshot_elapsed_ms: self.elapsed_ms,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSnapshot {
    pub domain: BootDomain,
    pub state: DomainState,
    pub elapsed_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DomainState {
    Pending,
    Starting,
    Ready,
    Delayed,
    Degraded,
    Failed,
}

impl BootDomain {
    pub const COUNT: usize = 9;

    pub const fn index(self) -> usize {
        match self {
            Self::Kernel => 0,
            Self::Initrd => 1,
            Self::Storage => 2,
            Self::Filesystems => 3,
            Self::Security => 4,
            Self::Network => 5,
            Self::Services => 6,
            Self::Graphics => 7,
            Self::Session => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BootEvent {
    PhaseEntered {
        sequence: u64,
        elapsed_ms: u64,
        phase: BootPhase,
    },
    DomainStarting {
        sequence: u64,
        elapsed_ms: u64,
        domain: BootDomain,
    },
    DomainReady {
        sequence: u64,
        elapsed_ms: u64,
        domain: BootDomain,
    },
    DomainDelayed {
        sequence: u64,
        elapsed_ms: u64,
        domain: BootDomain,
    },
    DomainDegraded {
        sequence: u64,
        elapsed_ms: u64,
        domain: BootDomain,
        criticality: Criticality,
        detail: Option<BoundedDetail>,
    },
    DomainFailed {
        sequence: u64,
        elapsed_ms: u64,
        domain: BootDomain,
        criticality: Criticality,
        detail: Option<BoundedDetail>,
    },
    DomainRecovered {
        sequence: u64,
        elapsed_ms: u64,
        domain: BootDomain,
    },
    BootReady {
        sequence: u64,
        elapsed_ms: u64,
        health: BootHealth,
    },
}

impl BootEvent {
    pub fn sequence(&self) -> u64 {
        match self {
            Self::PhaseEntered { sequence, .. }
            | Self::DomainStarting { sequence, .. }
            | Self::DomainReady { sequence, .. }
            | Self::DomainDelayed { sequence, .. }
            | Self::DomainDegraded { sequence, .. }
            | Self::DomainFailed { sequence, .. }
            | Self::DomainRecovered { sequence, .. }
            | Self::BootReady { sequence, .. } => *sequence,
        }
    }

    /// Milliseconds elapsed since the start of the same boot-observation lineage.
    /// Consumers may use this alongside `sequence` to reject temporal regressions.
    pub fn elapsed_ms(&self) -> u64 {
        match self {
            Self::PhaseEntered { elapsed_ms, .. }
            | Self::DomainStarting { elapsed_ms, .. }
            | Self::DomainReady { elapsed_ms, .. }
            | Self::DomainDelayed { elapsed_ms, .. }
            | Self::DomainDegraded { elapsed_ms, .. }
            | Self::DomainFailed { elapsed_ms, .. }
            | Self::DomainRecovered { elapsed_ms, .. }
            | Self::BootReady { elapsed_ms, .. } => *elapsed_ms,
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if let Self::DomainDegraded { detail, .. } | Self::DomainFailed { detail, .. } = self {
            if let Some(detail) = detail {
                detail.validate()?;
            }
        }
        Ok(())
    }
}

/// A deliberately bounded, presentation-safe diagnostic hint. This is not a
/// journal line and must not contain secrets, command lines, paths, SSIDs, or
/// user content. Producers should prefer stable operator-facing identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundedDetail(String);

impl BoundedDetail {
    pub fn new(value: impl Into<String>) -> Result<Self, ProtocolError> {
        let value = value.into();
        if value.len() > MAX_DETAIL_BYTES {
            return Err(ProtocolError::DetailTooLong(value.len()));
        }
        if value.chars().any(char::is_control) {
            return Err(ProtocolError::ControlCharacter);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn validate(&self) -> Result<(), ProtocolError> {
        if self.0.len() > MAX_DETAIL_BYTES {
            return Err(ProtocolError::DetailTooLong(self.0.len()));
        }
        if self.0.chars().any(char::is_control) {
            return Err(ProtocolError::ControlCharacter);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    UnsupportedVersion(u16),
    TooManyDomains(usize),
    DuplicateDomain(BootDomain),
    DomainElapsedAfterSnapshot {
        domain: BootDomain,
        domain_elapsed_ms: u64,
        snapshot_elapsed_ms: u64,
    },
    ElapsedRegressed {
        previous_ms: u64,
        observed_ms: u64,
    },
    DetailTooLong(usize),
    ControlCharacter,
    WireTooLarge {
        bytes: usize,
        max: usize,
    },
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(v) => write!(f, "unsupported boot protocol version {v}"),
            Self::TooManyDomains(n) => write!(f, "boot snapshot contains too many domains: {n}"),
            Self::DuplicateDomain(domain) => write!(f, "boot snapshot repeats domain {domain:?}"),
            Self::DomainElapsedAfterSnapshot {
                domain,
                domain_elapsed_ms,
                snapshot_elapsed_ms,
            } => write!(
                f,
                "boot snapshot domain {domain:?} elapsed time {domain_elapsed_ms}ms exceeds snapshot elapsed time {snapshot_elapsed_ms}ms"
            ),
            Self::ElapsedRegressed {
                previous_ms,
                observed_ms,
            } => write!(
                f,
                "boot elapsed time regressed from {previous_ms}ms to {observed_ms}ms"
            ),
            Self::DetailTooLong(n) => write!(f, "boot detail exceeds {MAX_DETAIL_BYTES} bytes: {n}"),
            Self::ControlCharacter => write!(f, "boot detail contains a control character"),
            Self::WireTooLarge { bytes, max } => {
                write!(f, "boot protocol message exceeds wire budget: {bytes} > {max} bytes")
            }
        }
    }
}

impl std::error::Error for ProtocolError {}

fn saturating_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_snapshot_starts_unknown_not_healthy() {
        let snapshot = BootSnapshot::new(1, Duration::from_millis(5), BootPhase::Kernel);
        assert_eq!(snapshot.health, BootHealth::Unknown);
    }

    #[test]
    fn detail_is_bounded_and_single_line() {
        assert!(BoundedDetail::new("Network initialization delayed").is_ok());
        assert!(BoundedDetail::new("bad\nline").is_err());
        assert!(BoundedDetail::new("x".repeat(MAX_DETAIL_BYTES + 1)).is_err());
    }

    #[test]
    fn event_round_trip_is_stable() {
        let event = BootEvent::DomainFailed {
            sequence: 12,
            elapsed_ms: 2200,
            domain: BootDomain::Network,
            criticality: Criticality::NonCritical,
            detail: Some(BoundedDetail::new("network service unavailable").unwrap()),
        };
        let encoded = serde_json::to_vec(&event).unwrap();
        let decoded: BootEvent = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, event);
        assert_eq!(decoded.sequence(), 12);
        assert_eq!(decoded.elapsed_ms(), 2200);
        decoded.validate().unwrap();
    }

    #[test]
    fn snapshot_rejects_unknown_protocol_version() {
        let mut snapshot = BootSnapshot::new(1, Duration::from_millis(5), BootPhase::Kernel);
        snapshot.protocol_version = PROTOCOL_VERSION + 1;
        assert_eq!(
            snapshot.validate(),
            Err(ProtocolError::UnsupportedVersion(PROTOCOL_VERSION + 1))
        );
    }

    #[test]
    fn snapshot_rejects_duplicate_domains() {
        let mut snapshot = BootSnapshot::new(1, Duration::from_millis(5), BootPhase::Storage);
        snapshot.domains = vec![
            DomainSnapshot {
                domain: BootDomain::Storage,
                state: DomainState::Starting,
                elapsed_ms: Some(2),
            },
            DomainSnapshot {
                domain: BootDomain::Storage,
                state: DomainState::Ready,
                elapsed_ms: Some(4),
            },
        ];
        assert_eq!(
            snapshot.validate(),
            Err(ProtocolError::DuplicateDomain(BootDomain::Storage))
        );
    }

    #[test]
    fn snapshot_rejects_domain_timestamp_from_the_future() {
        let mut snapshot = BootSnapshot::new(1, Duration::from_millis(5), BootPhase::Storage);
        snapshot.domains = vec![DomainSnapshot {
            domain: BootDomain::Storage,
            state: DomainState::Ready,
            elapsed_ms: Some(6),
        }];
        assert!(matches!(
            snapshot.validate(),
            Err(ProtocolError::DomainElapsedAfterSnapshot { .. })
        ));
    }

    #[test]
    fn health_has_explicit_severity() {
        assert!(BootHealth::Failed.severity() > BootHealth::Degraded.severity());
        assert!(BootHealth::Degraded.severity() > BootHealth::Delayed.severity());
        assert!(BootHealth::Unknown.severity() > BootHealth::Normal.severity());
    }
}
