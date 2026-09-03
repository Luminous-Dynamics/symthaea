// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use std::fs::Permissions;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use symthaea_iot_actuation_guard::{
    ACTUATION_GUARD_SOCKET_MODE, ActuationGuardServer, ActuationGuardServerConfig,
    GuardIngressState, GuardPeerPolicy, GuardServerError,
};
use symthaea_iot_transport_receipt::{
    TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION, TransportAttestorKeyV1, TransportAttestorStatus,
    TransportTrustRegistry, TransportTrustSnapshotV1, XeniaReceiptPeerRoleV1,
};
use tokio::net::UnixStream;

fn current_uid_gid() -> (u32, u32) {
    let (left, _right) = UnixStream::pair().unwrap();
    let credentials = left.peer_cred().unwrap();
    (credentials.uid(), credentials.gid())
}

fn ingress_state(now_unix_ms: u64) -> GuardIngressState {
    let registry = TransportTrustRegistry::genesis(TransportTrustSnapshotV1 {
        schema_version: TRANSPORT_TRUST_SNAPSHOT_SCHEMA_VERSION,
        sequence: 1,
        issued_at_unix_ms: now_unix_ms.saturating_sub(1_000),
        expires_at_unix_ms: now_unix_ms + 60_000,
        previous_snapshot_digest: None,
        keys: vec![TransportAttestorKeyV1 {
            attestor_id: "xenia-gateway-a".into(),
            key_id: "transport-key-1".into(),
            ed25519_public_key: [0x21; 32],
            ml_dsa_public_key: vec![0x22; 1_952],
            status: TransportAttestorStatus::Active,
            not_before_unix_ms: now_unix_ms.saturating_sub(1_000),
            not_after_unix_ms: now_unix_ms + 60_000,
            max_receipt_lifetime_ms: 4_000,
            required_peer_role: XeniaReceiptPeerRoleV1::Viewer,
            allowed_peer_fingerprints: BTreeSet::from([[0x44; 32]]),
            require_input_control: true,
        }],
    })
    .unwrap();
    let head = registry.head();
    GuardIngressState::new(registry, head).unwrap()
}

fn now_ms() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

fn unique_runtime_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "symthaea-actuation-guard-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn prepare_dir(path: &Path, mode: u32) {
    std::fs::create_dir(path).unwrap();
    std::fs::set_permissions(path, Permissions::from_mode(mode)).unwrap();
}

fn cleanup_dir(path: &Path) {
    let socket = path.join("guard.sock");
    let _ = std::fs::remove_file(socket);
    let _ = std::fs::remove_dir(path);
}

#[tokio::test]
async fn bind_requires_protected_group_reachable_runtime_directory_and_exact_socket_mode() {
    let (uid, gid) = current_uid_gid();
    let runtime = unique_runtime_dir("positive");
    prepare_dir(&runtime, 0o750);
    let socket_path = runtime.join("guard.sock");

    let config = ActuationGuardServerConfig {
        socket_path: socket_path.clone(),
        expected_socket_gid: gid,
        peer_policy: GuardPeerPolicy::new(BTreeSet::from([uid])).unwrap(),
        request_timeout: Duration::from_millis(500),
    };
    let server = ActuationGuardServer::bind(config, ingress_state(now_ms()))
        .await
        .unwrap();

    let runtime_metadata = std::fs::symlink_metadata(&runtime).unwrap();
    assert_eq!(runtime_metadata.mode() & 0o022, 0);
    assert_ne!(runtime_metadata.mode() & 0o010, 0);
    assert_eq!(runtime_metadata.gid(), gid);

    let socket_metadata = std::fs::symlink_metadata(server.socket_path()).unwrap();
    assert_eq!(socket_metadata.mode() & 0o777, ACTUATION_GUARD_SOCKET_MODE);
    assert_eq!(socket_metadata.gid(), gid);

    drop(server);
    cleanup_dir(&runtime);
}

#[tokio::test]
async fn group_writable_runtime_directory_is_rejected_before_socket_creation() {
    let (uid, gid) = current_uid_gid();
    let runtime = unique_runtime_dir("group-write");
    prepare_dir(&runtime, 0o770);
    let socket_path = runtime.join("guard.sock");
    let config = ActuationGuardServerConfig {
        socket_path: socket_path.clone(),
        expected_socket_gid: gid,
        peer_policy: GuardPeerPolicy::new(BTreeSet::from([uid])).unwrap(),
        request_timeout: Duration::from_millis(500),
    };

    assert!(matches!(
        ActuationGuardServer::bind(config, ingress_state(now_ms())).await,
        Err(GuardServerError::RuntimeDirectoryWritableByNonOwner)
    ));
    assert!(!socket_path.exists());
    cleanup_dir(&runtime);
}

#[tokio::test]
async fn runtime_directory_without_group_search_is_rejected() {
    let (uid, gid) = current_uid_gid();
    let runtime = unique_runtime_dir("no-group-search");
    prepare_dir(&runtime, 0o740);
    let socket_path = runtime.join("guard.sock");
    let config = ActuationGuardServerConfig {
        socket_path: socket_path.clone(),
        expected_socket_gid: gid,
        peer_policy: GuardPeerPolicy::new(BTreeSet::from([uid])).unwrap(),
        request_timeout: Duration::from_millis(500),
    };

    assert!(matches!(
        ActuationGuardServer::bind(config, ingress_state(now_ms())).await,
        Err(GuardServerError::RuntimeDirectoryNotGroupSearchable)
    ));
    assert!(!socket_path.exists());
    cleanup_dir(&runtime);
}
