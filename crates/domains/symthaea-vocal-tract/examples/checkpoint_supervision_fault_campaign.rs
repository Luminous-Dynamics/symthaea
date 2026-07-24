// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic fault lane for supervised checkpoint services and audit export.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_vocal_tract::{
    CheckpointAuditRetentionPolicy, CheckpointFaultPoint, CheckpointKeyAgentAuditSink,
    CheckpointKeyAuditEvent, CheckpointKeyAuditKey, CheckpointKeyAuditOutcome,
    DurableCheckpointKeyAuditLog, ScriptedCheckpointFaults, SecureCheckpointAgentListener,
    UnixPeerPolicy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = temporary_root("checkpoint-supervision-faults");
    std::fs::create_dir_all(&root)?;

    let socket = root.join("services").join("key-agent.sock");
    let socket_faults =
        ScriptedCheckpointFaults::new([CheckpointFaultPoint::AfterSocketBindBeforeMode]);
    assert!(
        SecureCheckpointAgentListener::bind_with_fault_injector(&socket, &socket_faults,).is_err()
    );
    assert!(!socket.exists());

    let listener = SecureCheckpointAgentListener::bind(&socket)?;
    assert!(listener.socket_identity_is_current());
    let peer = UnixPeerPolicy::current_process_identity();

    let audit_root = root.join("audit");
    let audit =
        DurableCheckpointKeyAuditLog::new(&audit_root, CheckpointKeyAuditKey::new([0xA1; 32])?);
    audit.record(CheckpointKeyAuditEvent {
        request_id: [0xA2; 16],
        peer,
        operation: None,
        key_id: None,
        utterance_id: None,
        checkpoint_sequence: None,
        outcome: CheckpointKeyAuditOutcome::Denied,
    })?;

    let export = root.join("exports").join("audit-0001.bin");
    let export_faults =
        ScriptedCheckpointFaults::new([CheckpointFaultPoint::AfterAuditExportFileSync]);
    assert!(
        audit
            .export_verified_with_fault_injector(&export, &export_faults)
            .is_err()
    );
    assert!(!export.exists());

    let receipt = audit.export_verified(&export)?;
    let verified = audit.verify_export(&export)?;
    let retention =
        audit.retention_decision(CheckpointAuditRetentionPolicy::new(1, 64 * 1024 * 1024)?)?;
    assert_eq!(receipt.record_count, verified.records.len());
    assert!(retention.export_required);
    assert!(!retention.destructive_retention_permitted);

    println!(
        "socket_inode_guarded={}",
        listener.socket_identity_is_current()
    );
    println!("audit_export_records={}", receipt.record_count);
    println!("audit_export_bytes={}", receipt.artifact_bytes);
    println!("retention_requires_export={}", retention.export_required);
    println!(
        "destructive_retention_permitted={}",
        retention.destructive_retention_permitted,
    );

    drop(listener);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

fn temporary_root(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("symthaea-{label}-{}-{suffix}", std::process::id(),))
}
