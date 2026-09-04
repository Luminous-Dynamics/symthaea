// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::SigningKey;
use serde_json::json;
use symthaea_authority::Digest32;
use symthaea_qualification_witness::{
    QualificationWitnessIdentityV1, QualificationWitnessPolicyV1, WITNESS_SCHEMA_VERSION,
};
use symthaea_qualification_witness_sequence::{
    verify_reserve_sign_persist_v1, DurableWitnessAttemptStateV1, SqliteWitnessSequenceStore,
    WitnessSequenceAttemptBindingV1,
};
use symthaea_qualification_witness_service::{
    QualificationVerifierRuntimePolicyV1, ReleaseEvidenceBindingsV1,
    VERIFIER_RUNTIME_SCHEMA_VERSION,
};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);
const FILE_DIGEST_DOMAIN: &[u8] = b"symthaea.qualification-verifier-file.v1\0";

struct Fixture {
    root: PathBuf,
    executable: PathBuf,
    verifier_script: PathBuf,
    archive: PathBuf,
    database: PathBuf,
}

impl Fixture {
    fn new(bindings: ReleaseEvidenceBindingsV1) -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "symthaea-witness-sequence-e2e-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();

        let executable = root.join("pinned-python-fixture");
        let verifier_script = root.join("pinned-evidence-verifier.py");
        let archive = root.join("qualification.tar.gz");
        let database = root.join("witness-sequence.sqlite3");

        let acceptance = json!({
            "schema": "symthaea.agency-tpm2-evidence-acceptance.v1",
            "accepted": true,
            "qualification_result": "PASS",
            "archive_sha256": hex_lower(&bindings.archive_sha256.0),
            "archive_hash_source": "caller",
            "manifest_sha256": "12".repeat(32),
            "head": hex_lower(&bindings.git_head),
            "tree": hex_lower(&bindings.git_tree),
            "external_head_bound": true,
            "external_tree_bound": true,
            "release_bound": true,
            "nixpkgs_locked": {
                "type": "github",
                "owner": "NixOS",
                "repo": "nixpkgs",
                "rev": "abc123",
                "narHash": "sha256-fixture"
            },
            "flake_lock_sha256": "15".repeat(32),
            "rust_toolchain_sha256": "16".repeat(32),
            "approved_pcr_profile": "17".repeat(32),
            "policy_digest": "18".repeat(32),
            "ak_public_digest": "19".repeat(32),
            "challenge_digest": "1a".repeat(32),
            "probe_sha256": "1b".repeat(32),
            "quote_wrapper_sha256": "1c".repeat(32),
            "checkquote_wrapper_sha256": "1d".repeat(32),
            "verifier_store": "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-symthaea-tpm2-verifier-v1"
        });
        let encoded = serde_json::to_string(&acceptance).unwrap();
        assert!(!encoded.contains('\''));
        let executable_bytes = format!("#!/bin/sh\nprintf '%s\\n' '{encoded}'\n");
        fs::write(&executable, executable_bytes).unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).unwrap();
        fs::write(&verifier_script, b"# fixture verifier identity\n").unwrap();
        fs::write(&archive, b"fixture archive bytes\n").unwrap();

        Self {
            root,
            executable,
            verifier_script,
            archive,
            database,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn file_digest(path: &Path) -> Digest32 {
    let bytes = fs::read(path).unwrap();
    let mut hasher = blake3::Hasher::new();
    hasher.update(FILE_DIGEST_DOMAIN);
    hasher.update(&bytes);
    Digest32(*hasher.finalize().as_bytes())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn bindings() -> ReleaseEvidenceBindingsV1 {
    ReleaseEvidenceBindingsV1 {
        archive_sha256: Digest32([0x11; 32]),
        git_head: [0x22; 20],
        git_tree: [0x33; 20],
    }
}

fn runtime(fixture: &Fixture) -> QualificationVerifierRuntimePolicyV1 {
    QualificationVerifierRuntimePolicyV1 {
        schema_version: VERIFIER_RUNTIME_SCHEMA_VERSION,
        runtime_policy_id: [0x31; 16],
        python_executable_path: fs::canonicalize(&fixture.executable)
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        python_executable_digest: file_digest(&fixture.executable),
        verifier_script_path: fs::canonicalize(&fixture.verifier_script)
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        verifier_script_digest: file_digest(&fixture.verifier_script),
        require_nix_store_paths: false,
        maximum_runtime_ms: 5_000,
        maximum_stdout_bytes: 64 * 1024,
        maximum_stderr_bytes: 64 * 1024,
    }
}

fn key() -> SigningKey {
    SigningKey::from_bytes(&[7; 32])
}

fn witness_policy(
    runtime: &QualificationVerifierRuntimePolicyV1,
    key: &SigningKey,
) -> QualificationWitnessPolicyV1 {
    QualificationWitnessPolicyV1 {
        schema_version: WITNESS_SCHEMA_VERSION,
        policy_id: [0x41; 16],
        witness_epoch: 9,
        threshold: 1,
        minimum_organizations: 1,
        minimum_services: 1,
        allowed_verifier_digests: vec![runtime.implementation_digest().unwrap()],
        witnesses: vec![QualificationWitnessIdentityV1 {
            witness_id: [1; 16],
            organization_id: [2; 16],
            service_id: [3; 16],
            public_key: key.verifying_key().to_bytes(),
        }],
    }
}

#[test]
fn same_attempt_retries_same_sequence_and_exact_signature() {
    let release = bindings();
    let fixture = Fixture::new(release);
    let runtime = runtime(&fixture);
    let key = key();
    let policy = witness_policy(&runtime, &key);
    let store = SqliteWitnessSequenceStore::open(&fixture.database).unwrap();
    let attempt_id = [9; 16];

    let first = verify_reserve_sign_persist_v1(
        &store,
        attempt_id,
        &runtime,
        &policy,
        [1; 16],
        &key,
        &fixture.archive,
        release,
    )
    .unwrap();
    let second = verify_reserve_sign_persist_v1(
        &store,
        attempt_id,
        &runtime,
        &policy,
        [1; 16],
        &key,
        &fixture.archive,
        release,
    )
    .unwrap();

    assert_eq!(first.sequence(), 1);
    assert_eq!(second.sequence(), 1);
    assert_eq!(first.reservation_digest(), second.reservation_digest());
    assert_eq!(first.attestation_digest(), second.attestation_digest());
    assert_eq!(
        first.verified.attestation.signature,
        second.verified.attestation.signature
    );
    assert_eq!(
        store.frontier([1; 16]).unwrap().unwrap().high_watermark,
        1
    );

    let durable = store
        .reserve_attempt(WitnessSequenceAttemptBindingV1 {
            attempt_id,
            witness_id: [1; 16],
            witness_epoch: policy.witness_epoch,
            archive_sha256: release.archive_sha256,
            git_head: release.git_head,
            git_tree: release.git_tree,
            verifier_digest: runtime.implementation_digest().unwrap(),
            witness_policy_digest: policy.digest().unwrap(),
        })
        .unwrap();
    assert_eq!(durable.sequence, 1);
    assert_eq!(durable.state, DurableWitnessAttemptStateV1::Signed);
    assert_eq!(durable.acceptance_digest, Some(first.verified.acceptance_digest()));
    assert_eq!(durable.attestation_digest, Some(first.attestation_digest()));

    let next = verify_reserve_sign_persist_v1(
        &store,
        [10; 16],
        &runtime,
        &policy,
        [1; 16],
        &key,
        &fixture.archive,
        release,
    )
    .unwrap();
    assert_eq!(next.sequence(), 2);
    assert_eq!(
        store.audit_witness([1; 16]).unwrap().unwrap().high_watermark,
        2
    );
}
