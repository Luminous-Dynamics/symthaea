// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Renderer-independent modulation derived from validated boot observations.
//!
//! This crate does not choose a `BootGenome`, render pixels, inspect journald,
//! or decide whether boot succeeded. It translates one already-authoritative
//! `BootSnapshot` lineage into bounded presentation facts that an exact visual
//! renderer may consume without inventing progress.

#![forbid(unsafe_code)]

use symthaea_boot_protocol::{BootDomain, BootHealth, BootPhase, BootSnapshot, DomainState};

pub const REVEAL_SCALE: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticBootAnchor {
    KernelActive,
    InitrdActive,
    StorageAvailable,
    FilesAvailable,
    SecurityReady,
    NetworkAvailable,
    ServicesAvailable,
    GraphicsAvailable,
    SessionStarting,
    SessionReady,
}

impl From<BootPhase> for SemanticBootAnchor {
    fn from(phase: BootPhase) -> Self {
        match phase {
            BootPhase::Kernel => Self::KernelActive,
            BootPhase::Initrd => Self::InitrdActive,
            BootPhase::Storage => Self::StorageAvailable,
            BootPhase::Filesystems => Self::FilesAvailable,
            BootPhase::Security => Self::SecurityReady,
            BootPhase::Network => Self::NetworkAvailable,
            BootPhase::Services => Self::ServicesAvailable,
            BootPhase::Graphics => Self::GraphicsAvailable,
            BootPhase::Session => Self::SessionStarting,
            BootPhase::Ready => Self::SessionReady,
        }
    }
}

impl SemanticBootAnchor {
    /// Fixed-point visual reveal floor earned by reaching this factual anchor.
    /// These are presentation constants, not percentages of Linux boot work.
    pub const fn reveal_floor(self) -> u32 {
        match self {
            Self::KernelActive => 50_000,
            Self::InitrdActive => 120_000,
            Self::StorageAvailable => 220_000,
            Self::FilesAvailable => 340_000,
            Self::SecurityReady => 450_000,
            Self::NetworkAvailable => 560_000,
            Self::ServicesAvailable => 680_000,
            Self::GraphicsAvailable => 820_000,
            Self::SessionStarting => 920_000,
            Self::SessionReady => REVEAL_SCALE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticFloor {
    Ambient,
    Status,
    Diagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DomainMask(u16);

impl DomainMask {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn insert(&mut self, domain: BootDomain) {
        self.0 |= 1u16 << domain.index();
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, domain: BootDomain) -> bool {
        (self.0 & (1u16 << domain.index())) != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveEcologyModulation {
    pub observation_sequence: u64,
    pub anchor: SemanticBootAnchor,
    pub health: BootHealth,
    pub reveal_floor: u32,
    pub delayed_domains: DomainMask,
    pub repair_domains: DomainMask,
    pub diagnostic_floor: DiagnosticFloor,
    pub pulse_token: u64,
    pub handoff_ready: bool,
}

impl LiveEcologyModulation {
    pub const fn validate(&self) -> bool {
        self.reveal_floor <= REVEAL_SCALE
            && (!self.handoff_ready || matches!(self.anchor, SemanticBootAnchor::SessionReady))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveAdapterError {
    InvalidSnapshot(String),
    SequenceRegressed { previous: u64, observed: u64 },
    AnchorRegressed {
        previous: SemanticBootAnchor,
        observed: SemanticBootAnchor,
    },
}

impl std::fmt::Display for LiveAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSnapshot(error) => write!(f, "invalid boot snapshot: {error}"),
            Self::SequenceRegressed { previous, observed } => write!(
                f,
                "boot observation sequence regressed: previous={previous}, observed={observed}"
            ),
            Self::AnchorRegressed { previous, observed } => write!(
                f,
                "semantic boot anchor regressed: previous={previous:?}, observed={observed:?}"
            ),
        }
    }
}

impl std::error::Error for LiveAdapterError {}

/// One reducer instance belongs to one already-validated boot observation lineage.
/// Callers must create/reset it when the protocol receiver adopts a new lineage.
#[derive(Debug, Default)]
pub struct LiveEcologyReducer {
    last_sequence: Option<u64>,
    last_anchor: Option<SemanticBootAnchor>,
}

impl LiveEcologyReducer {
    pub const fn new() -> Self {
        Self {
            last_sequence: None,
            last_anchor: None,
        }
    }

    pub fn reset(&mut self) {
        self.last_sequence = None;
        self.last_anchor = None;
    }

    pub fn reduce(
        &mut self,
        snapshot: &BootSnapshot,
    ) -> Result<LiveEcologyModulation, LiveAdapterError> {
        snapshot
            .validate()
            .map_err(|error| LiveAdapterError::InvalidSnapshot(error.to_string()))?;

        if let Some(previous) = self.last_sequence {
            if snapshot.sequence < previous {
                return Err(LiveAdapterError::SequenceRegressed {
                    previous,
                    observed: snapshot.sequence,
                });
            }
        }

        let anchor = SemanticBootAnchor::from(snapshot.phase);
        if let Some(previous) = self.last_anchor {
            if anchor < previous {
                return Err(LiveAdapterError::AnchorRegressed {
                    previous,
                    observed: anchor,
                });
            }
        }

        let mut delayed_domains = DomainMask::empty();
        let mut repair_domains = DomainMask::empty();
        for domain in &snapshot.domains {
            match domain.state {
                DomainState::Delayed => delayed_domains.insert(domain.domain),
                DomainState::Degraded | DomainState::Failed => {
                    repair_domains.insert(domain.domain)
                }
                DomainState::Pending | DomainState::Starting | DomainState::Ready => {}
            }
        }

        let diagnostic_floor = match snapshot.health {
            BootHealth::Normal | BootHealth::Unknown => DiagnosticFloor::Ambient,
            BootHealth::Delayed => DiagnosticFloor::Status,
            BootHealth::Degraded | BootHealth::Failed => DiagnosticFloor::Diagnostics,
        };

        let modulation = LiveEcologyModulation {
            observation_sequence: snapshot.sequence,
            anchor,
            health: snapshot.health,
            reveal_floor: anchor.reveal_floor(),
            delayed_domains,
            repair_domains,
            diagnostic_floor,
            pulse_token: snapshot.sequence,
            handoff_ready: matches!(snapshot.phase, BootPhase::Ready),
        };
        debug_assert!(modulation.validate());

        self.last_sequence = Some(snapshot.sequence);
        self.last_anchor = Some(anchor);
        Ok(modulation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use symthaea_boot_protocol::{DomainSnapshot, ProtocolError};

    fn snapshot(sequence: u64, phase: BootPhase, health: BootHealth) -> BootSnapshot {
        let mut snapshot = BootSnapshot::new(sequence, Duration::from_millis(sequence * 100), phase);
        snapshot.health = health;
        snapshot
    }

    #[test]
    fn anchors_and_reveal_floors_are_monotonic() {
        let phases = [
            BootPhase::Kernel,
            BootPhase::Initrd,
            BootPhase::Storage,
            BootPhase::Filesystems,
            BootPhase::Security,
            BootPhase::Network,
            BootPhase::Services,
            BootPhase::Graphics,
            BootPhase::Session,
            BootPhase::Ready,
        ];
        let mut previous = 0;
        for phase in phases {
            let anchor = SemanticBootAnchor::from(phase);
            assert!(anchor.reveal_floor() >= previous);
            previous = anchor.reveal_floor();
        }
        assert_eq!(previous, REVEAL_SCALE);
    }

    #[test]
    fn slow_boot_holds_last_factual_anchor() {
        let mut reducer = LiveEcologyReducer::new();
        let first = snapshot(7, BootPhase::Network, BootHealth::Delayed);
        let first = reducer.reduce(&first).unwrap();
        let later = snapshot(8, BootPhase::Network, BootHealth::Delayed);
        let later = reducer.reduce(&later).unwrap();

        assert_eq!(first.anchor, SemanticBootAnchor::NetworkAvailable);
        assert_eq!(later.anchor, first.anchor);
        assert_eq!(later.reveal_floor, first.reveal_floor);
        assert_eq!(later.diagnostic_floor, DiagnosticFloor::Status);
    }

    #[test]
    fn ready_is_the_only_handoff_ready_state() {
        let mut reducer = LiveEcologyReducer::new();
        let session = reducer
            .reduce(&snapshot(1, BootPhase::Session, BootHealth::Normal))
            .unwrap();
        assert!(!session.handoff_ready);

        let ready = reducer
            .reduce(&snapshot(2, BootPhase::Ready, BootHealth::Normal))
            .unwrap();
        assert!(ready.handoff_ready);
        assert_eq!(ready.reveal_floor, REVEAL_SCALE);
    }

    #[test]
    fn health_controls_only_presentation_floor() {
        let cases = [
            (BootHealth::Normal, DiagnosticFloor::Ambient),
            (BootHealth::Unknown, DiagnosticFloor::Ambient),
            (BootHealth::Delayed, DiagnosticFloor::Status),
            (BootHealth::Degraded, DiagnosticFloor::Diagnostics),
            (BootHealth::Failed, DiagnosticFloor::Diagnostics),
        ];

        for (health, expected) in cases {
            let mut reducer = LiveEcologyReducer::new();
            let modulation = reducer
                .reduce(&snapshot(1, BootPhase::Services, health))
                .unwrap();
            assert_eq!(modulation.diagnostic_floor, expected);
        }
    }

    #[test]
    fn delayed_and_repair_domains_are_bounded_masks() {
        let mut current = snapshot(4, BootPhase::Services, BootHealth::Degraded);
        current.domains = vec![
            DomainSnapshot {
                domain: BootDomain::Network,
                state: DomainState::Delayed,
                elapsed_ms: Some(300),
            },
            DomainSnapshot {
                domain: BootDomain::Services,
                state: DomainState::Failed,
                elapsed_ms: Some(350),
            },
        ];

        let mut reducer = LiveEcologyReducer::new();
        let modulation = reducer.reduce(&current).unwrap();
        assert!(modulation.delayed_domains.contains(BootDomain::Network));
        assert!(modulation.repair_domains.contains(BootDomain::Services));
        assert!(!modulation.repair_domains.contains(BootDomain::Network));
    }

    #[test]
    fn sequence_or_anchor_rewind_is_rejected_until_lineage_reset() {
        let mut reducer = LiveEcologyReducer::new();
        reducer
            .reduce(&snapshot(10, BootPhase::Graphics, BootHealth::Normal))
            .unwrap();

        assert!(matches!(
            reducer.reduce(&snapshot(9, BootPhase::Graphics, BootHealth::Normal)),
            Err(LiveAdapterError::SequenceRegressed { .. })
        ));
        assert!(matches!(
            reducer.reduce(&snapshot(11, BootPhase::Services, BootHealth::Normal)),
            Err(LiveAdapterError::AnchorRegressed { .. })
        ));

        reducer.reset();
        assert!(reducer
            .reduce(&snapshot(1, BootPhase::Kernel, BootHealth::Unknown))
            .is_ok());
    }

    #[test]
    fn invalid_snapshot_is_not_reinterpreted() {
        let mut current = snapshot(1, BootPhase::Storage, BootHealth::Unknown);
        current.domains = vec![
            DomainSnapshot {
                domain: BootDomain::Storage,
                state: DomainState::Starting,
                elapsed_ms: Some(10),
            },
            DomainSnapshot {
                domain: BootDomain::Storage,
                state: DomainState::Ready,
                elapsed_ms: Some(20),
            },
        ];
        assert_eq!(
            current.validate(),
            Err(ProtocolError::DuplicateDomain(BootDomain::Storage))
        );

        let mut reducer = LiveEcologyReducer::new();
        assert!(matches!(
            reducer.reduce(&current),
            Err(LiveAdapterError::InvalidSnapshot(_))
        ));
    }

    #[test]
    fn equal_sequence_is_idempotent_not_a_new_pulse_semantic() {
        let mut reducer = LiveEcologyReducer::new();
        let current = snapshot(3, BootPhase::Filesystems, BootHealth::Normal);
        let first = reducer.reduce(&current).unwrap();
        let second = reducer.reduce(&current).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.pulse_token, second.pulse_token);
    }
}
