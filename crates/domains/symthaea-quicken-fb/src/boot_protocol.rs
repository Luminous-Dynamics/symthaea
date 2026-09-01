// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-open consumer for normalized Symthaea boot telemetry.
//!
//! The framebuffer renderer is presentation only. Missing sockets, malformed
//! packets, stale sequences, foreign observation lineages, and snapshot I/O
//! failures are counted locally and never become boot authority.

#![forbid(unsafe_code)]

use std::fs;
use std::io;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

use symthaea_boot_protocol::wire::{
    MAX_WIRE_BYTES, ObservationId, WireApply, WireMessage, WireStateReducer,
    validate_datagram_size,
};
use symthaea_boot_protocol::{BootHealth, BootPhase, BootSnapshot};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PollReport {
    pub applied: u32,
    pub rejected: u32,
    pub lineage_resets: u32,
}

/// Small semantic projection used by the animation. It intentionally contains no
/// unit names, raw journal text, PIDs, paths, or other operator detail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootVisualState {
    pub phase: BootPhase,
    pub health: BootHealth,
    pub sequence: u64,
    pub growth_floor: f32,
}

impl BootVisualState {
    pub fn from_snapshot(snapshot: &BootSnapshot) -> Self {
        let phase_floor = match snapshot.phase {
            BootPhase::Kernel => 0.08,
            BootPhase::Initrd => 0.12,
            BootPhase::Storage => 0.22,
            BootPhase::Filesystems => 0.32,
            BootPhase::Security => 0.40,
            BootPhase::Network => 0.50,
            BootPhase::Services => 0.64,
            BootPhase::Graphics => 0.78,
            BootPhase::Session => 0.90,
            BootPhase::Ready => 1.00,
        };
        let health_factor = match snapshot.health {
            BootHealth::Normal => 1.0,
            BootHealth::Unknown => 0.85,
            BootHealth::Delayed => 0.65,
            BootHealth::Degraded => 0.45,
            BootHealth::Failed => 0.20,
        };
        Self {
            phase: snapshot.phase,
            health: snapshot.health,
            sequence: snapshot.sequence,
            growth_floor: phase_floor * health_factor,
        }
    }
}

pub struct BootTelemetry {
    socket: Option<UnixDatagram>,
    state_path: Option<PathBuf>,
    reducer: WireStateReducer,
}

impl BootTelemetry {
    /// Create a receiver. Failure to bind or load state disables only the failed
    /// capability; callers should continue rendering normally.
    pub fn new(socket_path: Option<&Path>, state_path: Option<&Path>) -> Self {
        let mut telemetry = Self {
            socket: None,
            state_path: state_path.map(Path::to_path_buf),
            reducer: WireStateReducer::default(),
        };

        if let Some(path) = socket_path {
            telemetry.socket = bind_nonblocking(path).ok();
        }
        let _ = telemetry.refresh_from_state(None);
        telemetry
    }

    pub fn snapshot(&self) -> Option<BootSnapshot> {
        self.reducer.snapshot()
    }

    pub fn visual_state(&self) -> Option<BootVisualState> {
        self.snapshot().as_ref().map(BootVisualState::from_snapshot)
    }

    pub fn poll(&mut self) -> PollReport {
        let mut report = PollReport::default();
        let Some(socket) = self.socket.as_ref().and_then(|socket| socket.try_clone().ok()) else {
            return report;
        };
        // MAX + 1 lets us distinguish an oversized datagram from an exact-budget
        // one before attempting deserialization.
        let mut buffer = [0_u8; MAX_WIRE_BYTES + 1];

        loop {
            match socket.recv(&mut buffer) {
                Ok(bytes) => {
                    let payload = &buffer[..bytes];
                    if validate_datagram_size(payload).is_err() {
                        report.rejected = report.rejected.saturating_add(1);
                        continue;
                    }
                    let message: WireMessage = match serde_json::from_slice(payload) {
                        Ok(message) => message,
                        Err(_) => {
                            report.rejected = report.rejected.saturating_add(1);
                            continue;
                        }
                    };
                    if message.validate().is_err() {
                        report.rejected = report.rejected.saturating_add(1);
                        continue;
                    }
                    self.apply_message(&message, &mut report);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => {
                    report.rejected = report.rejected.saturating_add(1);
                    break;
                }
            }
        }
        report
    }

    fn apply_message(&mut self, message: &WireMessage, report: &mut PollReport) {
        match self.reducer.apply(message) {
            Ok(WireApply::Applied) => {
                report.applied = report.applied.saturating_add(1);
            }
            Ok(WireApply::IgnoredStale) => {}
            Ok(WireApply::AwaitingSnapshot) | Ok(WireApply::ForeignObservation) => {
                // The persisted snapshot is the independently selected side
                // channel for authorizing a lineage reset. An event alone can
                // never reset the reducer.
                if self.refresh_from_state(Some(message.observation())) {
                    report.lineage_resets = report.lineage_resets.saturating_add(1);
                    if matches!(message, WireMessage::Event { .. }) {
                        match self.reducer.apply(message) {
                            Ok(WireApply::Applied) => {
                                report.applied = report.applied.saturating_add(1)
                            }
                            Ok(_) => {}
                            Err(_) => report.rejected = report.rejected.saturating_add(1),
                        }
                    } else {
                        report.applied = report.applied.saturating_add(1);
                    }
                }
            }
            Err(_) => {
                report.rejected = report.rejected.saturating_add(1);
            }
        }
    }

    /// Load only a validated snapshot. When `expected` is supplied, the file must
    /// belong to exactly that observation before it can reset lineage state.
    fn refresh_from_state(&mut self, expected: Option<ObservationId>) -> bool {
        let Some(path) = self.state_path.as_deref() else {
            return false;
        };
        let Ok(bytes) = fs::read(path) else {
            return false;
        };
        if validate_datagram_size(&bytes).is_err() {
            return false;
        }
        let Ok(message) = serde_json::from_slice::<WireMessage>(&bytes) else {
            return false;
        };
        if message.validate().is_err() {
            return false;
        }
        if !matches!(message, WireMessage::Snapshot { .. }) {
            return false;
        }
        if expected.is_some_and(|expected| message.observation() != expected) {
            return false;
        }
        self.reducer.reset_from_snapshot(&message).is_ok()
    }
}

fn bind_nonblocking(path: &Path) -> io::Result<UnixDatagram> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => {
            // Do not blindly unlink a live renderer's socket. Probe first; a
            // successful connect means another consumer owns the endpoint.
            let probe = UnixDatagram::unbound()?;
            match probe.connect(path) {
                Ok(()) => {
                    return Err(io::Error::new(
                        io::ErrorKind::AddrInUse,
                        "boot telemetry socket is already active",
                    ));
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound
                    ) =>
                {
                    fs::remove_file(path)?;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "boot telemetry path exists and is not a Unix socket",
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    let socket = UnixDatagram::bind(path)?;
    socket.set_nonblocking(true)?;
    Ok(socket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use symthaea_boot_protocol::wire::ObservationId;

    const OBS_A: ObservationId = ObservationId::from_bytes([0xA1; 16]);
    const OBS_B: ObservationId = ObservationId::from_bytes([0xB2; 16]);

    #[test]
    fn health_modulates_growth_without_claiming_authority() {
        let mut healthy = BootSnapshot::new(2, Duration::from_millis(10), BootPhase::Services);
        healthy.health = BootHealth::Normal;
        let mut failed = healthy.clone();
        failed.health = BootHealth::Failed;
        assert!(
            BootVisualState::from_snapshot(&healthy).growth_floor
                > BootVisualState::from_snapshot(&failed).growth_floor
        );
    }

    #[test]
    fn event_cannot_establish_initial_lineage() {
        let mut telemetry = BootTelemetry::new(None, None);
        let mut report = PollReport::default();
        telemetry.apply_message(
            &WireMessage::event(
                OBS_A,
                symthaea_boot_protocol::BootEvent::DomainReady {
                    sequence: 2,
                    elapsed_ms: 20,
                    domain: symthaea_boot_protocol::BootDomain::Storage,
                },
            ),
            &mut report,
        );
        assert!(telemetry.snapshot().is_none());
        assert_eq!(report.lineage_resets, 0);
    }

    #[test]
    fn foreign_event_cannot_reset_without_matching_snapshot_side_channel() {
        let mut telemetry = BootTelemetry::new(None, None);
        let initial = WireMessage::snapshot(
            OBS_A,
            BootSnapshot::new(1, Duration::from_millis(10), BootPhase::Kernel),
        );
        telemetry.reducer.reset_from_snapshot(&initial).unwrap();

        let mut report = PollReport::default();
        telemetry.apply_message(
            &WireMessage::event(
                OBS_B,
                symthaea_boot_protocol::BootEvent::DomainReady {
                    sequence: 99,
                    elapsed_ms: 11,
                    domain: symthaea_boot_protocol::BootDomain::Storage,
                },
            ),
            &mut report,
        );
        assert_eq!(telemetry.reducer.observation(), Some(OBS_A));
        assert_eq!(report.lineage_resets, 0);
    }
}
