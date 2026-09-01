// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::os::unix::net::UnixDatagram;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use symthaea_boot_protocol::wire::{ObservationId, WireMessage};
use symthaea_boot_protocol::{BootDomain, BootEvent, BootPhase, BootSnapshot};
use symthaea_quicken_fb::boot_protocol::BootTelemetry;

const OBS_A: ObservationId = ObservationId::from_bytes([0xA1; 16]);
const OBS_B: ObservationId = ObservationId::from_bytes([0xB2; 16]);

fn scratch_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-quicken-boot-protocol-{}-{nonce}",
        std::process::id()
    ))
}

fn write_wire(path: &std::path::Path, message: &WireMessage) {
    fs::write(path, serde_json::to_vec(message).unwrap()).unwrap();
}

#[test]
fn observer_restart_requires_matching_snapshot_before_foreign_event_applies() {
    let dir = scratch_dir();
    fs::create_dir_all(&dir).unwrap();
    let socket_path = dir.join("events.sock");
    let state_path = dir.join("state.json");

    let initial = WireMessage::snapshot(
        OBS_A,
        BootSnapshot::new(1, Duration::from_millis(10), BootPhase::Kernel),
    );
    write_wire(&state_path, &initial);

    {
        let mut telemetry = BootTelemetry::new(Some(&socket_path), Some(&state_path));
        assert_eq!(telemetry.snapshot().unwrap().sequence, 1);

        // A restarted authoritative observer publishes a fresh lineage snapshot
        // to the side channel before its live event arrives.
        let restarted = WireMessage::snapshot(
            OBS_B,
            BootSnapshot::new(1, Duration::from_millis(20), BootPhase::Storage),
        );
        write_wire(&state_path, &restarted);

        let event = WireMessage::event(
            OBS_B,
            BootEvent::DomainReady {
                sequence: 2,
                elapsed_ms: 30,
                domain: BootDomain::Filesystems,
            },
        );
        let sender = UnixDatagram::unbound().unwrap();
        sender
            .send_to(&serde_json::to_vec(&event).unwrap(), &socket_path)
            .unwrap();

        let report = telemetry.poll();
        assert_eq!(report.lineage_resets, 1);
        assert_eq!(report.applied, 1);
        let snapshot = telemetry.snapshot().unwrap();
        assert_eq!(snapshot.sequence, 2);
        assert_eq!(snapshot.phase, BootPhase::Storage);
        assert!(snapshot
            .domains
            .iter()
            .any(|domain| domain.domain == BootDomain::Filesystems));
    }

    let _ = fs::remove_file(&socket_path);
    let _ = fs::remove_file(&state_path);
    let _ = fs::remove_dir(&dir);
}

#[test]
fn oversized_datagram_is_rejected_without_losing_current_state() {
    let dir = scratch_dir();
    fs::create_dir_all(&dir).unwrap();
    let socket_path = dir.join("events.sock");
    let state_path = dir.join("state.json");

    let initial = WireMessage::snapshot(
        OBS_A,
        BootSnapshot::new(7, Duration::from_millis(50), BootPhase::Services),
    );
    write_wire(&state_path, &initial);

    {
        let mut telemetry = BootTelemetry::new(Some(&socket_path), Some(&state_path));
        let sender = UnixDatagram::unbound().unwrap();
        sender.send_to(&vec![b'x'; 4097], &socket_path).unwrap();

        let report = telemetry.poll();
        assert_eq!(report.rejected, 1);
        assert_eq!(telemetry.snapshot().unwrap().sequence, 7);
        assert_eq!(telemetry.snapshot().unwrap().phase, BootPhase::Services);
    }

    let _ = fs::remove_file(&socket_path);
    let _ = fs::remove_file(&state_path);
    let _ = fs::remove_dir(&dir);
}
