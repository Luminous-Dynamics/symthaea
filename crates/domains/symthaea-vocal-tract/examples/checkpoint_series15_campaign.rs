// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Series 15 operational-durability delta campaign.
//!
//! This executable covers the new cross-process admission, retention-bound
//! compaction, and sudden child-process termination lanes. It models process
//! death without unwinding; it does not claim sudden device-power-loss safety.

use std::num::NonZeroU8;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use symthaea_vocal_tract::{
    CheckpointAuditArchiveAuthority, CheckpointAuditArchiveKey,
    CheckpointAuditCompactionDurability, CheckpointAuditRetentionRequirement, CheckpointFaultPoint,
    CheckpointKeyAgentAuditSink, CheckpointKeyAuditEvent, CheckpointKeyAuditKey,
    CheckpointKeyAuditOutcome, CheckpointKeyId, CheckpointKeyOperation,
    CheckpointOperationalTrustMetrics, CheckpointOperationalTrustRequirements,
    DurableCheckpointKeyAuditLog, ProcessExitCheckpointFaults,
    SharedCheckpointAgentConnectionLimiter, UnixPeerCredentials,
    assemble_checkpoint_operational_trust_evidence,
};

const AUDIT_KEY: [u8; 32] = [0xA1; 32];
const ARCHIVE_KEY: [u8; 32] = [0xA2; 32];
const REPOSITORY_BINDING: [u8; 32] = [0xA3; 32];
const STORAGE_CLASS_BINDING: [u8; 32] = [0xA4; 32];
const CHILD_MODE: &str = "SYMTHAEA_SERIES15_CHILD_MODE";
const CHILD_ROOT: &str = "SYMTHAEA_SERIES15_ROOT";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(mode) = std::env::var(CHILD_MODE) {
        let root = PathBuf::from(std::env::var(CHILD_ROOT)?);
        return child_main(&mode, &root);
    }

    let root = temporary_root("checkpoint-series15");
    std::fs::create_dir_all(&root)?;
    let mut metrics = CheckpointOperationalTrustMetrics::default();

    run_shared_admission_lane(&root, &mut metrics)?;
    run_retention_compaction_lane(&root, &mut metrics)?;
    run_process_crash_lanes(&root, &mut metrics)?;

    let report = assemble_checkpoint_operational_trust_evidence(
        metrics,
        CheckpointOperationalTrustRequirements::series_15_delta(),
    );
    println!("schema={}", report.schema);
    for gate in &report.gates {
        if gate.required {
            println!(
                "gate={} status={:?} detail={}",
                gate.name, gate.status, gate.detail
            );
        }
    }
    println!("series15_delta_passed={}", report.passed());
    std::fs::remove_dir_all(root)?;
    if !report.passed() {
        return Err("Series 15 operational delta campaign failed".into());
    }
    Ok(())
}

fn child_main(mode: &str, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        "hold-admission-slot" => {
            let limiter = SharedCheckpointAgentConnectionLimiter::new(root.join("slots"), 1)?;
            let _permit = limiter.try_acquire()?;
            std::fs::write(root.join("slot-ready"), b"ready")?;
            loop {
                thread::sleep(Duration::from_secs(60));
            }
        }
        "crash-before-compaction-write" => {
            run_compaction_child(root, CheckpointFaultPoint::BeforeAuditCompactionWrite)
        }
        "crash-after-compaction-publish" => {
            run_compaction_child(root, CheckpointFaultPoint::AfterAuditCompactionPublish)
        }
        _ => Err("unknown Series 15 child mode".into()),
    }
}

fn run_shared_admission_lane(
    root: &Path,
    metrics: &mut CheckpointOperationalTrustMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let lane = root.join("shared-admission");
    std::fs::create_dir_all(&lane)?;
    let mut child = spawn_child("hold-admission-slot", &lane)?;
    wait_for_file(&lane.join("slot-ready"))?;

    let competing = SharedCheckpointAgentConnectionLimiter::new(lane.join("slots"), 1)?;
    metrics.shared_connection_limit_exercised = true;
    metrics.shared_connection_limit_rejected = competing.try_acquire().is_err();
    child.kill()?;
    let _ = child.wait()?;
    let recovered = competing.try_acquire()?;
    drop(recovered);
    Ok(())
}

fn run_retention_compaction_lane(
    root: &Path,
    metrics: &mut CheckpointOperationalTrustMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let lane = root.join("retention-compaction");
    let artifacts = prepare_segment(&lane, 2)?;
    let authority = archive_authority()?;
    let receipt_bytes = std::fs::read(&artifacts.archive_receipt)?;
    let commitment = authority.seal_retention_commitment(
        &receipt_bytes,
        REPOSITORY_BINDING,
        [0xB1; 16],
        2_100_000_000,
        3,
        STORAGE_CLASS_BINDING,
    )?;
    let commitment_path = lane.join("archive").join("segment.commitment");
    let requirement = CheckpointAuditRetentionRequirement {
        minimum_retained_until_unix_seconds: 2_000_000_000,
        minimum_replicas: 2,
        expected_storage_class_binding: STORAGE_CLASS_BINDING,
    };
    authority.write_retention_commitment_no_overwrite(
        &commitment_path,
        &commitment,
        &receipt_bytes,
        REPOSITORY_BINDING,
        requirement,
    )?;
    metrics.audit_retention_commitment_exercised = true;
    metrics.audit_retention_commitment_verified = authority
        .verify_retention_commitment_file(
            &commitment_path,
            &artifacts.archive_receipt,
            REPOSITORY_BINDING,
            requirement,
        )
        .is_ok();

    let log = audit_log(&lane)?;
    let compacted = log.compact_live_segment_with_retention_commitment(
        &artifacts.export,
        &artifacts.archive_receipt,
        &commitment_path,
        &authority,
        REPOSITORY_BINDING,
        requirement,
        1_950_000_000,
    )?;
    metrics.audit_compaction_exercised = true;
    metrics.audit_compaction_succeeded =
        compacted.durability == CheckpointAuditCompactionDurability::Synced;
    log.record(audit_event(2))?;
    let records = log.verify()?;
    metrics.audit_compaction_continuity_verified = records.len() == 1
        && records[0].audit_sequence == 2
        && records[0].previous_record_digest == compacted.anchor.previous_head_record_digest;
    Ok(())
}

fn run_process_crash_lanes(
    root: &Path,
    metrics: &mut CheckpointOperationalTrustMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = root.join("crash-before-write");
    prepare_segment(&before, 1)?;
    let status = spawn_child_and_wait("crash-before-compaction-write", &before)?;
    metrics.process_crash_campaigns += 1;
    if exited_with(status, 97) && audit_log(&before)?.verify()?.len() == 1 {
        metrics.process_crash_recoveries += 1;
    }

    let after = root.join("crash-after-publish");
    prepare_segment(&after, 1)?;
    let status = spawn_child_and_wait("crash-after-compaction-publish", &after)?;
    metrics.process_crash_campaigns += 1;
    let log = audit_log(&after)?;
    let anchor = log.current_segment_anchor()?;
    metrics.audit_compaction_reconciliation_exercised = true;
    if exited_with(status, 97) {
        if let Some(anchor) = anchor {
            let digest = *blake3::hash(&postcard::to_stdvec(&anchor)?).as_bytes();
            metrics.audit_compaction_reconciled = log.reconcile_compaction(digest).is_ok();
            if metrics.audit_compaction_reconciled {
                metrics.process_crash_recoveries += 1;
            }
        }
    }
    Ok(())
}

fn run_compaction_child(
    root: &Path,
    point: CheckpointFaultPoint,
) -> Result<(), Box<dyn std::error::Error>> {
    let artifacts = SegmentArtifacts {
        export: root.join("exports").join("segment.bin"),
        archive_receipt: root.join("archive").join("segment.receipt"),
    };
    let faults = ProcessExitCheckpointFaults::new(point, NonZeroU8::new(97).unwrap());
    audit_log(root)?.compact_live_segment_with_fault_injector(
        artifacts.export,
        artifacts.archive_receipt,
        &archive_authority()?,
        REPOSITORY_BINDING,
        1_950_000_100,
        &faults,
    )?;
    Err("configured process-exit fault did not trigger".into())
}

struct SegmentArtifacts {
    export: PathBuf,
    archive_receipt: PathBuf,
}

fn prepare_segment(
    root: &Path,
    records: u64,
) -> Result<SegmentArtifacts, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(root)?;
    let log = audit_log(root)?;
    for sequence in 0..records {
        log.record(audit_event(sequence))?;
    }
    let export = root.join("exports").join("segment.bin");
    let export_receipt = log.export_verified(&export)?;
    let archive_receipt = root.join("archive").join("segment.receipt");
    let authority = archive_authority()?;
    let encoded = authority.seal_receipt(
        &export_receipt,
        [0xB2; 16],
        REPOSITORY_BINDING,
        1_900_000_000,
    )?;
    authority.write_receipt_no_overwrite(&archive_receipt, &encoded, REPOSITORY_BINDING)?;
    Ok(SegmentArtifacts {
        export,
        archive_receipt,
    })
}

fn audit_log(root: &Path) -> Result<DurableCheckpointKeyAuditLog, Box<dyn std::error::Error>> {
    Ok(DurableCheckpointKeyAuditLog::new(
        root.join("audit"),
        CheckpointKeyAuditKey::new(AUDIT_KEY)?,
    ))
}

fn archive_authority() -> Result<CheckpointAuditArchiveAuthority, Box<dyn std::error::Error>> {
    Ok(CheckpointAuditArchiveAuthority::new(
        CheckpointAuditArchiveKey::new(ARCHIVE_KEY)?,
    ))
}

fn audit_event(sequence: u64) -> CheckpointKeyAuditEvent {
    CheckpointKeyAuditEvent {
        request_id: [(sequence as u8).wrapping_add(1); 16],
        peer: UnixPeerCredentials {
            pid: std::process::id(),
            uid: 1000,
            gid: 1000,
        },
        operation: Some(CheckpointKeyOperation::Encrypt),
        key_id: Some(CheckpointKeyId([0xC1; 16])),
        utterance_id: Some(*b"series15-audit01"),
        checkpoint_sequence: Some(sequence),
        outcome: CheckpointKeyAuditOutcome::Allowed,
    }
}

fn spawn_child(mode: &str, root: &Path) -> Result<std::process::Child, Box<dyn std::error::Error>> {
    Ok(Command::new(std::env::current_exe()?)
        .env(CHILD_MODE, mode)
        .env(CHILD_ROOT, root)
        .spawn()?)
}

fn spawn_child_and_wait(mode: &str, root: &Path) -> Result<ExitStatus, Box<dyn std::error::Error>> {
    Ok(spawn_child(mode, root)?.wait()?)
}

fn wait_for_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..500 {
        if path.is_file() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    Err("child did not publish readiness marker".into())
}

fn exited_with(status: ExitStatus, code: i32) -> bool {
    status.code() == Some(code)
}

fn temporary_root(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("symthaea-{label}-{}-{suffix}", std::process::id()))
}
