// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local engineering lane for the checkpoint operational-trust campaign.
//!
//! This example exercises the real Unix protocols, durable audit log, and
//! cooperating-process rollback locks. It intentionally accepts
//! `SameTrustDomain` for the local monotonic server. Promotion must rerun the
//! same protocol against an `IndependentMonotonic` service.

use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_vocal_tract::{
    CheckpointAgentToken, CheckpointAuditError, CheckpointChainPosition, CheckpointEnvelopeError,
    CheckpointKey, CheckpointKeyAgentAuditSink, CheckpointKeyAgentPeerPolicy,
    CheckpointKeyAgentReplayGuard, CheckpointKeyAuditEvent, CheckpointKeyAuditKey,
    CheckpointKeyAuthorization, CheckpointKeyId, CheckpointKeyOperation, CheckpointKeyProvider,
    CheckpointKeyring, CheckpointMonotonicAgentToken, CheckpointMonotonicReplayGuard,
    CheckpointOperationalTrustMetrics, CheckpointOperationalTrustRequirements,
    CheckpointRollbackError, CheckpointRollbackProtector, DurableCheckpointKeyAuditLog,
    DurableRollbackProtector, RollbackProtectionLevel, RollbackStateKey, UnixCheckpointKeyAgent,
    UnixCheckpointMonotonicAgent, UnixPeerPolicy, assemble_checkpoint_operational_trust_evidence,
    effective_gid, effective_uid, serve_checkpoint_key_agent_connection_with_peer_policy,
    serve_checkpoint_key_agent_connection_with_security,
    serve_checkpoint_monotonic_agent_connection,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = temporary_root("checkpoint-operational-trust");
    std::fs::create_dir_all(&root)?;
    let mut metrics = CheckpointOperationalTrustMetrics::default();

    run_key_agent_lane(&root, &mut metrics)?;
    run_rollback_lane(&root, &mut metrics)?;
    run_monotonic_protocol_lane(&root, &mut metrics)?;

    let mut requirements = CheckpointOperationalTrustRequirements {
        minimum_monotonic_level: RollbackProtectionLevel::SameTrustDomain,
        ..CheckpointOperationalTrustRequirements::default()
    };
    // Series 15 has a separate child-process campaign because it intentionally
    // terminates subprocesses and compacts disposable audit segments.
    requirements.require_shared_connection_limit = false;
    requirements.require_audit_retention_commitment = false;
    requirements.require_audit_compaction = false;
    requirements.require_audit_compaction_reconciliation = false;
    requirements.minimum_process_crash_recoveries = 0;
    let report = assemble_checkpoint_operational_trust_evidence(metrics, requirements);
    println!("schema={}", report.schema);
    for gate in &report.gates {
        println!(
            "gate={} required={} status={:?} observed={:?} minimum={:?}",
            gate.name, gate.required, gate.status, gate.observed, gate.required_minimum,
        );
    }
    println!("local_engineering_lane_passed={}", report.passed());
    println!("promotion_note=rerun with an independently monotonic TPM/HSM/remote service");
    println!(
        "series15_note=run checkpoint_series15_campaign for shared admission and compaction crash lanes"
    );
    std::fs::remove_dir_all(root)?;
    if !report.passed() {
        return Err("checkpoint operational-trust engineering lane failed".into());
    }
    Ok(())
}

fn run_key_agent_lane(
    root: &Path,
    metrics: &mut CheckpointOperationalTrustMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket = root.join("key-agent.sock");
    let listener = UnixListener::bind(&socket)?;
    let token_bytes = [0x41; 32];
    let key_id = CheckpointKeyId([0x42; 16]);
    let audit_root = root.join("audit");
    let server_audit_root = audit_root.clone();
    let server = thread::spawn(move || -> Result<(), CheckpointEnvelopeError> {
        let ring = CheckpointKeyring::new(CheckpointKey::from_parts(key_id, [0x43; 32])?);
        let token = CheckpointAgentToken::new(token_bytes)?;
        let replay = CheckpointKeyAgentReplayGuard::with_default_window();
        let policy = CheckpointKeyAgentPeerPolicy::same_effective_user();
        let audit = DurableCheckpointKeyAuditLog::new(
            server_audit_root,
            CheckpointKeyAuditKey::new([0x44; 32])
                .map_err(|_| CheckpointEnvelopeError::KeyProviderUnavailable("audit key"))?,
        );
        for _ in 0..2 {
            let (mut stream, _) = listener
                .accept()
                .map_err(|_| CheckpointEnvelopeError::KeyProviderUnavailable("accept"))?;
            serve_checkpoint_key_agent_connection_with_security(
                &mut stream,
                &ring,
                &token,
                &replay,
                &policy,
                &audit,
            )?;
        }
        Ok(())
    });
    let client = UnixCheckpointKeyAgent::new(&socket, CheckpointAgentToken::new(token_bytes)?);
    assert_eq!(client.active_key_id()?, key_id);
    client.authorize_key_use(CheckpointKeyAuthorization {
        operation: CheckpointKeyOperation::Encrypt,
        key_id: Some(key_id),
        utterance_id: *b"ops-trust-test01",
        sequence: 0,
    })?;
    assert_eq!(client.active_encryption_key()?.id(), key_id);
    server.join().map_err(|_| "key-agent server panicked")??;

    let records =
        DurableCheckpointKeyAuditLog::new(&audit_root, CheckpointKeyAuditKey::new([0x44; 32])?)
            .verify()?;
    metrics.peer_credentials_exercised = true;
    metrics.durable_audit_records = records.len();
    metrics.durable_audit_chain_valid = records.len() == 2;

    let replay = CheckpointKeyAgentReplayGuard::new(4)?;
    replay.verify_and_record([0x51; 16])?;
    metrics.bearer_replay_rejected = replay.verify_and_record([0x51; 16]).is_err();

    let denied_socket = root.join("key-agent-denied.sock");
    let denied_listener = UnixListener::bind(&denied_socket)?;
    let denied_server = thread::spawn(move || {
        let ring = CheckpointKeyring::new(CheckpointKey::from_parts(key_id, [0x43; 32]).unwrap());
        let token = CheckpointAgentToken::new(token_bytes).unwrap();
        let replay = CheckpointKeyAgentReplayGuard::with_default_window();
        let policy = UnixPeerPolicy::new(
            vec![effective_uid().wrapping_add(1)],
            vec![effective_gid().wrapping_add(1)],
            true,
        )
        .unwrap();
        let (mut stream, _) = denied_listener.accept().unwrap();
        serve_checkpoint_key_agent_connection_with_peer_policy(
            &mut stream,
            &ring,
            &token,
            &replay,
            &policy,
        )
    });
    let denied_client =
        UnixCheckpointKeyAgent::new(&denied_socket, CheckpointAgentToken::new(token_bytes)?);
    metrics.unauthorized_peer_rejected = matches!(
        denied_client.active_key_id(),
        Err(CheckpointEnvelopeError::KeyUseDenied)
    );
    denied_server
        .join()
        .map_err(|_| "denied server panicked")??;

    struct FailingAudit;
    impl CheckpointKeyAgentAuditSink for FailingAudit {
        fn record(&self, _event: CheckpointKeyAuditEvent) -> Result<(), CheckpointAuditError> {
            Err(CheckpointAuditError::Unavailable(
                "injected campaign failure",
            ))
        }
    }
    let fail_socket = root.join("key-agent-audit-fail.sock");
    let fail_listener = UnixListener::bind(&fail_socket)?;
    let fail_server = thread::spawn(move || {
        let ring = CheckpointKeyring::new(CheckpointKey::from_parts(key_id, [0x43; 32]).unwrap());
        let token = CheckpointAgentToken::new(token_bytes).unwrap();
        let replay = CheckpointKeyAgentReplayGuard::with_default_window();
        let policy = CheckpointKeyAgentPeerPolicy::same_effective_user();
        let (mut stream, _) = fail_listener.accept().unwrap();
        serve_checkpoint_key_agent_connection_with_security(
            &mut stream,
            &ring,
            &token,
            &replay,
            &policy,
            &FailingAudit,
        )
    });
    let fail_client =
        UnixCheckpointKeyAgent::new(&fail_socket, CheckpointAgentToken::new(token_bytes)?);
    fail_client.authorize_key_use(CheckpointKeyAuthorization {
        operation: CheckpointKeyOperation::Encrypt,
        key_id: Some(key_id),
        utterance_id: *b"ops-audit-fail01",
        sequence: 0,
    })?;
    metrics.audit_failure_blocked_release = fail_client.active_encryption_key().is_err();
    assert!(
        fail_server
            .join()
            .map_err(|_| "audit server panicked")?
            .is_err()
    );
    Ok(())
}

fn run_rollback_lane(
    root: &Path,
    metrics: &mut CheckpointOperationalTrustMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let rollback_root = root.join("rollback");
    let first = Arc::new(DurableRollbackProtector::new(
        &rollback_root,
        RollbackStateKey::new([0x61; 32])?,
    ));
    let second = Arc::new(DurableRollbackProtector::new(
        &rollback_root,
        RollbackStateKey::new([0x61; 32])?,
    ));
    let barrier = Arc::new(Barrier::new(3));
    let spawn = |protector: Arc<DurableRollbackProtector>, barrier: Arc<Barrier>, digest: u8| {
        thread::spawn(move || {
            barrier.wait();
            protector.verify_and_advance(position(0, digest))
        })
    };
    let left = spawn(first, Arc::clone(&barrier), 1);
    let right = spawn(second, Arc::clone(&barrier), 2);
    barrier.wait();
    let results = [left.join().unwrap(), right.join().unwrap()];
    metrics.competing_rollback_writers = 2;
    metrics.competing_rollback_winners = results.iter().filter(|result| result.is_ok()).count();
    metrics.rollback_fork_rejected = results.iter().any(|result| {
        matches!(
            result,
            Err(CheckpointRollbackError::ForkDetected { sequence: 0 })
        )
    });
    let winner = DurableRollbackProtector::new(&rollback_root, RollbackStateKey::new([0x61; 32])?);
    metrics.rollback_gap_rejected = matches!(
        winner.verify_and_advance(position(2, 3)),
        Err(CheckpointRollbackError::SequenceGap { .. })
    );
    Ok(())
}

fn run_monotonic_protocol_lane(
    root: &Path,
    metrics: &mut CheckpointOperationalTrustMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket = root.join("monotonic-agent.sock");
    let listener = UnixListener::bind(&socket)?;
    let token_bytes = [0x71; 32];
    let state_root = root.join("monotonic-state");
    let server = thread::spawn(move || -> Result<(), CheckpointRollbackError> {
        let protector =
            DurableRollbackProtector::new(state_root, RollbackStateKey::new([0x72; 32])?);
        let token = CheckpointMonotonicAgentToken::new(token_bytes)?;
        let replay = CheckpointMonotonicReplayGuard::with_default_window();
        let policy = UnixPeerPolicy::same_effective_user();
        for _ in 0..4 {
            let (mut stream, _) = listener
                .accept()
                .map_err(|_| CheckpointRollbackError::Unavailable("accept"))?;
            serve_checkpoint_monotonic_agent_connection(
                &mut stream,
                &protector,
                &token,
                &replay,
                &policy,
            )?;
        }
        Ok(())
    });
    let client = UnixCheckpointMonotonicAgent::new(
        &socket,
        CheckpointMonotonicAgentToken::new(token_bytes)?,
        RollbackProtectionLevel::SameTrustDomain,
    )?;
    assert_eq!(client.current(position(0, 0).utterance_id)?, None);
    client.verify_and_advance(position(0, 4))?;
    assert_eq!(
        client.current(position(0, 0).utterance_id)?,
        Some(position(0, 4))
    );
    metrics.external_monotonic_requests = 4;
    metrics.external_monotonic_level = Some(client.protection_level());
    server.join().map_err(|_| "monotonic server panicked")??;

    let replay = CheckpointMonotonicReplayGuard::new(4)?;
    replay.verify_and_record([0x73; 16])?;
    metrics.external_monotonic_replay_rejected = replay.verify_and_record([0x73; 16]).is_err();
    Ok(())
}

fn position(sequence: u64, digest: u8) -> CheckpointChainPosition {
    CheckpointChainPosition {
        utterance_id: *b"ops-trust-test01",
        sequence,
        envelope_digest: [digest; 32],
    }
}

fn temporary_root(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("symthaea-{label}-{}-{suffix}", std::process::id(),))
}
