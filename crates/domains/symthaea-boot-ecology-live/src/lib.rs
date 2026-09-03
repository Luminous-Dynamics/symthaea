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

const DOMAIN_MASK_CAPACITY: usize = u16::BITS as usize;
const _: [(); DOMAIN_MASK_CAPACITY - BootDomain::COUNT] =
    [(); DOMAIN_MASK_CAPACITY - BootDomain::COUNT];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticBootAnchor {
    KernelPhase,
    InitrdPhase,
    StoragePhase,
    FilesystemsPhase,
    SecurityPhase,
    NetworkPhase,
    ServicesPhase,
    GraphicsPhase,
    SessionPhase,
    SessionReady,
}

impl From<BootPhase> for SemanticBootAnchor {
    fn from(phase: BootPhase) -> Self {
        match phase {
            BootPhase::Kernel => Self::KernelPhase,
            BootPhase::Initrd => Self::InitrdPhase,
            BootPhase::Storage => Self::StoragePhase,
            BootPhase::Filesystems => Self::FilesystemsPhase,
            BootPhase::Security => Self::SecurityPhase,
            BootPhase::Network => Self::NetworkPhase,
            BootPhase::Services => Self::ServicesPhase,
            BootPhase::Graphics => Self::GraphicsPhase,
            BootPhase::Session => Self::SessionPhase,
            BootPhase::Ready => Self::SessionReady,
        }
    }
}

impl SemanticBootAnchor {
    /// Fixed-point visual reveal floor earned by reaching this factual phase.
    /// These constants order presentation; they are not percentages of Linux
    /// boot work and must never be described to the user as such.
    pub const fn reveal_floor(self) -> u32 {
        match self {
            Self::KernelPhase => 50_000,
            Self::InitrdPhase => 120_000,
            Self::StoragePhase => 220_000,
            Self::FilesystemsPhase => 340_000,
            Self::SecurityPhase => 450_000,
            Self::NetworkPhase => 560_000,
            Self::ServicesPhase => 680_000,
            Self::GraphicsPhase => 820_000,
            Self::SessionPhase => 920_000,
            Self::SessionReady => REVEAL_SCALE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

    pub const fn contains(self, domain: BootDomain) -> bool {
        (self.0 & (1u16 << domain.index())) != 0
    }

    const fn newly_set_from(self, previous: Self) -> bool {
        self.0 & !previous.0 != 0
    }

    const fn removed_from(self, previous: Self) -> bool {
        previous.0 & !self.0 != 0
    }
}

/// One bounded transient visual accent. Renderers should trigger it only when
/// `accent_token` changes, not once per received telemetry packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualAccent {
    None,
    Progress,
    Delay,
    Degraded,
    Failed,
    Recovery,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveEcologyModulation {
    pub observation_sequence: u64,
    pub anchor: SemanticBootAnchor,
    /// Authoritative coarse health copied from the validated snapshot. The live
    /// adapter never promotes or blesses this value.
    pub health: BootHealth,
    pub reveal_floor: u32,
    pub delayed_domains: DomainMask,
    pub degraded_domains: DomainMask,
    pub failed_domains: DomainMask,
    /// Minimum presentation visibility. This may defensively surface more detail
    /// than global health alone when domain state is inconsistent, but it never
    /// rewrites the authoritative `health` field.
    pub diagnostic_floor: DiagnosticFloor,
    /// Sequence-derived idempotency token for meaningful transient accents.
    /// It changes only when presentation-relevant semantics change.
    pub accent_token: u64,
    pub accent: VisualAccent,
    /// True only for the protocol's explicit `Ready` phase.
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
    SequenceRegressed {
        previous: u64,
        observed: u64,
    },
    SequenceEquivocated {
        sequence: u64,
    },
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
            Self::SequenceEquivocated { sequence } => write!(
                f,
                "boot observation sequence {sequence} changed presentation semantics"
            ),
            Self::AnchorRegressed { previous, observed } => write!(
                f,
                "semantic boot anchor regressed: previous={previous:?}, observed={observed:?}"
            ),
        }
    }
}

impl std::error::Error for LiveAdapterError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PresentationFingerprint {
    anchor: SemanticBootAnchor,
    health: BootHealth,
    delayed_domains: DomainMask,
    degraded_domains: DomainMask,
    failed_domains: DomainMask,
    diagnostic_floor: DiagnosticFloor,
    handoff_ready: bool,
}

impl PresentationFingerprint {
    fn from_snapshot(snapshot: &BootSnapshot) -> Self {
        let mut delayed_domains = DomainMask::empty();
        let mut degraded_domains = DomainMask::empty();
        let mut failed_domains = DomainMask::empty();

        for domain in &snapshot.domains {
            match domain.state {
                DomainState::Delayed => delayed_domains.insert(domain.domain),
                DomainState::Degraded => degraded_domains.insert(domain.domain),
                DomainState::Failed => failed_domains.insert(domain.domain),
                DomainState::Pending | DomainState::Starting | DomainState::Ready => {}
            }
        }

        let mut diagnostic_floor = match snapshot.health {
            BootHealth::Normal | BootHealth::Unknown => DiagnosticFloor::Ambient,
            BootHealth::Delayed => DiagnosticFloor::Status,
            BootHealth::Degraded | BootHealth::Failed => DiagnosticFloor::Diagnostics,
        };
        if !delayed_domains.is_empty() {
            diagnostic_floor = diagnostic_floor.max(DiagnosticFloor::Status);
        }
        if !degraded_domains.is_empty() || !failed_domains.is_empty() {
            diagnostic_floor = DiagnosticFloor::Diagnostics;
        }

        Self {
            anchor: SemanticBootAnchor::from(snapshot.phase),
            health: snapshot.health,
            delayed_domains,
            degraded_domains,
            failed_domains,
            diagnostic_floor,
            handoff_ready: matches!(snapshot.phase, BootPhase::Ready),
        }
    }

    const fn issues_removed_from(self, previous: Self) -> bool {
        self.delayed_domains.removed_from(previous.delayed_domains)
            || self.degraded_domains.removed_from(previous.degraded_domains)
            || self.failed_domains.removed_from(previous.failed_domains)
    }

    fn visual_issue_severity(self) -> u8 {
        let health = match self.health {
            BootHealth::Normal | BootHealth::Unknown => 0,
            BootHealth::Delayed => 1,
            BootHealth::Degraded => 2,
            BootHealth::Failed => 3,
        };
        let domains = if !self.failed_domains.is_empty() {
            3
        } else if !self.degraded_domains.is_empty() {
            2
        } else if !self.delayed_domains.is_empty() {
            1
        } else {
            0
        };
        health.max(domains)
    }
}

/// One reducer instance belongs to one already-validated boot observation lineage.
/// Callers must create/reset it when the protocol receiver adopts a new lineage.
#[derive(Debug, Default)]
pub struct LiveEcologyReducer {
    last_sequence: Option<u64>,
    last_fingerprint: Option<PresentationFingerprint>,
    last_modulation: Option<LiveEcologyModulation>,
}

impl LiveEcologyReducer {
    pub const fn new() -> Self {
        Self {
            last_sequence: None,
            last_fingerprint: None,
            last_modulation: None,
        }
    }

    pub fn reset(&mut self) {
        self.last_sequence = None;
        self.last_fingerprint = None;
        self.last_modulation = None;
    }

    pub fn reduce(
        &mut self,
        snapshot: &BootSnapshot,
    ) -> Result<LiveEcologyModulation, LiveAdapterError> {
        snapshot
            .validate()
            .map_err(|error| LiveAdapterError::InvalidSnapshot(error.to_string()))?;

        let fingerprint = PresentationFingerprint::from_snapshot(snapshot);

        if let Some(previous_sequence) = self.last_sequence {
            if snapshot.sequence < previous_sequence {
                return Err(LiveAdapterError::SequenceRegressed {
                    previous: previous_sequence,
                    observed: snapshot.sequence,
                });
            }
            if snapshot.sequence == previous_sequence {
                if self.last_fingerprint != Some(fingerprint) {
                    return Err(LiveAdapterError::SequenceEquivocated {
                        sequence: snapshot.sequence,
                    });
                }
                return Ok(self
                    .last_modulation
                    .expect("accepted sequence always has cached modulation"));
            }
        }

        if let Some(previous) = self.last_fingerprint
            && fingerprint.anchor < previous.anchor
        {
            return Err(LiveAdapterError::AnchorRegressed {
                previous: previous.anchor,
                observed: fingerprint.anchor,
            });
        }

        let accent = self
            .last_fingerprint
            .map(|previous| classify_accent(previous, fingerprint))
            .unwrap_or(VisualAccent::None);
        let previous_token = self.last_modulation.map(|item| item.accent_token).unwrap_or(0);
        let accent_token = if accent == VisualAccent::None {
            previous_token
        } else {
            snapshot.sequence
        };

        let modulation = LiveEcologyModulation {
            observation_sequence: snapshot.sequence,
            anchor: fingerprint.anchor,
            health: fingerprint.health,
            reveal_floor: fingerprint.anchor.reveal_floor(),
            delayed_domains: fingerprint.delayed_domains,
            degraded_domains: fingerprint.degraded_domains,
            failed_domains: fingerprint.failed_domains,
            diagnostic_floor: fingerprint.diagnostic_floor,
            accent_token,
            accent,
            handoff_ready: fingerprint.handoff_ready,
        };
        debug_assert!(modulation.validate());

        self.last_sequence = Some(snapshot.sequence);
        self.last_fingerprint = Some(fingerprint);
        self.last_modulation = Some(modulation);
        Ok(modulation)
    }
}

fn classify_accent(
    previous: PresentationFingerprint,
    current: PresentationFingerprint,
) -> VisualAccent {
    let previous_severity = previous.visual_issue_severity();
    let current_severity = current.visual_issue_severity();

    if current_severity > previous_severity {
        return accent_for_severity(current_severity);
    }
    if current.handoff_ready
        && !previous.handoff_ready
        && current.health == BootHealth::Normal
        && current_severity == 0
    {
        return VisualAccent::Ready;
    }
    if current_severity < previous_severity {
        return VisualAccent::Recovery;
    }

    if current.failed_domains.newly_set_from(previous.failed_domains) {
        return VisualAccent::Failed;
    }
    if current
        .degraded_domains
        .newly_set_from(previous.degraded_domains)
    {
        return VisualAccent::Degraded;
    }
    if current.delayed_domains.newly_set_from(previous.delayed_domains) {
        return VisualAccent::Delay;
    }
    if current.issues_removed_from(previous) {
        return VisualAccent::Recovery;
    }
    if current.anchor > previous.anchor {
        return VisualAccent::Progress;
    }
    VisualAccent::None
}

const fn accent_for_severity(severity: u8) -> VisualAccent {
    match severity {
        3.. => VisualAccent::Failed,
        2 => VisualAccent::Degraded,
        1 => VisualAccent::Delay,
        _ => VisualAccent::None,
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
    fn phase_anchor_does_not_overclaim_unit_readiness() {
        assert_eq!(
            SemanticBootAnchor::from(BootPhase::Network),
            SemanticBootAnchor::NetworkPhase
        );
        assert_eq!(
            SemanticBootAnchor::from(BootPhase::Graphics),
            SemanticBootAnchor::GraphicsPhase
        );
    }

    #[test]
    fn slow_boot_holds_last_factual_anchor_without_pulsing_on_telemetry_churn() {
        let mut reducer = LiveEcologyReducer::new();
        let first = reducer
            .reduce(&snapshot(7, BootPhase::Network, BootHealth::Delayed))
            .unwrap();
        let later = reducer
            .reduce(&snapshot(8, BootPhase::Network, BootHealth::Delayed))
            .unwrap();

        assert_eq!(first.anchor, SemanticBootAnchor::NetworkPhase);
        assert_eq!(later.anchor, first.anchor);
        assert_eq!(later.reveal_floor, first.reveal_floor);
        assert_eq!(later.diagnostic_floor, DiagnosticFloor::Status);
        assert_eq!(first.accent_token, 0);
        assert_eq!(later.accent_token, first.accent_token);
        assert_eq!(later.accent, VisualAccent::None);
    }

    #[test]
    fn meaningful_phase_advance_gets_one_progress_accent() {
        let mut reducer = LiveEcologyReducer::new();
        reducer
            .reduce(&snapshot(1, BootPhase::Filesystems, BootHealth::Normal))
            .unwrap();
        let progressed = reducer
            .reduce(&snapshot(2, BootPhase::Services, BootHealth::Normal))
            .unwrap();
        let unchanged = reducer
            .reduce(&snapshot(3, BootPhase::Services, BootHealth::Normal))
            .unwrap();

        assert_eq!(progressed.accent, VisualAccent::Progress);
        assert_eq!(progressed.accent_token, 2);
        assert_eq!(unchanged.accent, VisualAccent::None);
        assert_eq!(unchanged.accent_token, progressed.accent_token);
    }

    #[test]
    fn ready_is_the_only_handoff_ready_state_and_has_ready_accent_when_healthy() {
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
        assert_eq!(ready.accent, VisualAccent::Ready);
        assert_eq!(ready.accent_token, 2);
    }

    #[test]
    fn unknown_ready_is_not_celebrated_as_known_normal() {
        let mut reducer = LiveEcologyReducer::new();
        reducer
            .reduce(&snapshot(1, BootPhase::Session, BootHealth::Unknown))
            .unwrap();
        let ready = reducer
            .reduce(&snapshot(2, BootPhase::Ready, BootHealth::Unknown))
            .unwrap();
        assert!(ready.handoff_ready);
        assert_eq!(ready.health, BootHealth::Unknown);
        assert_ne!(ready.accent, VisualAccent::Ready);
    }

    #[test]
    fn degraded_ready_never_gets_a_celebratory_ready_accent() {
        let mut reducer = LiveEcologyReducer::new();
        reducer
            .reduce(&snapshot(1, BootPhase::Session, BootHealth::Degraded))
            .unwrap();
        let ready = reducer
            .reduce(&snapshot(2, BootPhase::Ready, BootHealth::Degraded))
            .unwrap();
        assert!(ready.handoff_ready);
        assert_eq!(ready.diagnostic_floor, DiagnosticFloor::Diagnostics);
        assert_ne!(ready.accent, VisualAccent::Ready);
    }

    #[test]
    fn health_sets_a_minimum_presentation_floor() {
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
    fn domain_state_cannot_be_visually_underreported() {
        let mut current = snapshot(4, BootPhase::Services, BootHealth::Normal);
        current.domains = vec![DomainSnapshot {
            domain: BootDomain::Services,
            state: DomainState::Failed,
            elapsed_ms: Some(350),
        }];

        let mut reducer = LiveEcologyReducer::new();
        let modulation = reducer.reduce(&current).unwrap();
        assert_eq!(modulation.health, BootHealth::Normal);
        assert_eq!(modulation.diagnostic_floor, DiagnosticFloor::Diagnostics);
        assert!(modulation.failed_domains.contains(BootDomain::Services));
    }

    #[test]
    fn delay_degraded_and_failed_domains_remain_separate_bounded_masks() {
        let mut current = snapshot(4, BootPhase::Services, BootHealth::Failed);
        current.domains = vec![
            DomainSnapshot {
                domain: BootDomain::Network,
                state: DomainState::Delayed,
                elapsed_ms: Some(300),
            },
            DomainSnapshot {
                domain: BootDomain::Services,
                state: DomainState::Degraded,
                elapsed_ms: Some(325),
            },
            DomainSnapshot {
                domain: BootDomain::Graphics,
                state: DomainState::Failed,
                elapsed_ms: Some(350),
            },
        ];

        let mut reducer = LiveEcologyReducer::new();
        let modulation = reducer.reduce(&current).unwrap();
        assert!(modulation.delayed_domains.contains(BootDomain::Network));
        assert!(modulation.degraded_domains.contains(BootDomain::Services));
        assert!(modulation.failed_domains.contains(BootDomain::Graphics));
        assert!(!modulation.failed_domains.contains(BootDomain::Network));
    }

    #[test]
    fn new_failure_outranks_other_transient_accents() {
        let mut reducer = LiveEcologyReducer::new();
        reducer
            .reduce(&snapshot(1, BootPhase::Network, BootHealth::Normal))
            .unwrap();

        let mut failed = snapshot(2, BootPhase::Services, BootHealth::Failed);
        failed.domains = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Failed,
            elapsed_ms: Some(200),
        }];
        let modulation = reducer.reduce(&failed).unwrap();
        assert_eq!(modulation.accent, VisualAccent::Failed);
        assert_eq!(modulation.accent_token, 2);
    }

    #[test]
    fn severity_downgrade_is_recovery_even_when_domain_moves_masks() {
        let mut failed = snapshot(4, BootPhase::Services, BootHealth::Failed);
        failed.domains = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Failed,
            elapsed_ms: Some(300),
        }];

        let mut degraded = snapshot(5, BootPhase::Services, BootHealth::Degraded);
        degraded.domains = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Degraded,
            elapsed_ms: Some(450),
        }];

        let mut reducer = LiveEcologyReducer::new();
        reducer.reduce(&failed).unwrap();
        let after = reducer.reduce(&degraded).unwrap();
        assert_eq!(after.accent, VisualAccent::Recovery);
        assert!(after.failed_domains.is_empty());
        assert!(after.degraded_domains.contains(BootDomain::Network));
    }

    #[test]
    fn recovery_clears_transient_visual_emphasis_without_rewinding_phase() {
        let mut degraded = snapshot(4, BootPhase::Services, BootHealth::Degraded);
        degraded.domains = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Degraded,
            elapsed_ms: Some(300),
        }];

        let mut recovered = snapshot(5, BootPhase::Services, BootHealth::Normal);
        recovered.domains = vec![DomainSnapshot {
            domain: BootDomain::Network,
            state: DomainState::Ready,
            elapsed_ms: Some(450),
        }];

        let mut reducer = LiveEcologyReducer::new();
        let before = reducer.reduce(&degraded).unwrap();
        let after = reducer.reduce(&recovered).unwrap();

        assert_eq!(before.anchor, after.anchor);
        assert_eq!(before.reveal_floor, after.reveal_floor);
        assert_eq!(before.diagnostic_floor, DiagnosticFloor::Diagnostics);
        assert_eq!(after.diagnostic_floor, DiagnosticFloor::Ambient);
        assert!(after.degraded_domains.is_empty());
        assert!(after.failed_domains.is_empty());
        assert_eq!(after.accent, VisualAccent::Recovery);
        assert_eq!(after.accent_token, 5);
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
    fn same_sequence_with_changed_semantics_is_rejected_as_equivocation() {
        let mut reducer = LiveEcologyReducer::new();
        reducer
            .reduce(&snapshot(7, BootPhase::Network, BootHealth::Normal))
            .unwrap();

        assert!(matches!(
            reducer.reduce(&snapshot(7, BootPhase::Services, BootHealth::Normal)),
            Err(LiveAdapterError::SequenceEquivocated { sequence: 7 })
        ));
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
    fn equal_sequence_is_exactly_idempotent() {
        let mut reducer = LiveEcologyReducer::new();
        let current = snapshot(3, BootPhase::Filesystems, BootHealth::Normal);
        let first = reducer.reduce(&current).unwrap();
        let second = reducer.reduce(&current).unwrap();
        assert_eq!(first, second);
    }
}
