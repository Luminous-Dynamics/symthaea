// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(unix)]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_process_capture::ProcessSpec;
use symthaea_process_stdin::{
    BoundStdinProcessRequest, ContentAddressedStdinFile, StdinLauncherArtifact,
    capture_process_with_stdin_file,
};

fn unique_file(name: &str, bytes: &[u8]) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "symthaea-process-stdin-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    fs::write(&path, bytes).unwrap();
    path
}

fn launcher() -> StdinLauncherArtifact {
    StdinLauncherArtifact::observe(PathBuf::from(env!(
        "CARGO_BIN_EXE_symthaea-process-stdin-launcher"
    ))
    .as_path())
    .unwrap()
}

#[test]
fn exact_file_descriptor_becomes_target_stdin() {
    let payload = b"exact historical SQL bytes\nsecond line\n";
    let path = unique_file("stdin.sql", payload);
    let stdin = ContentAddressedStdinFile::observe(&path).unwrap();
    let request = BoundStdinProcessRequest {
        launcher: launcher(),
        stdin,
        target: ProcessSpec::new("/bin/cat").clear_environment(),
    };
    let capture = capture_process_with_stdin_file(&request).unwrap();
    assert!(capture.process_success());
    assert_eq!(capture.process_capture.stdout, payload);
    assert_eq!(capture.capture_sha256().unwrap().len(), 64);
    let _ = fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn wrong_preregistered_stdin_digest_is_captured_as_failure() {
    let path = unique_file("stdin.sql", b"actual bytes\n");
    let mut stdin = ContentAddressedStdinFile::observe(&path).unwrap();
    stdin.sha256 = "0".repeat(64);
    let request = BoundStdinProcessRequest {
        launcher: launcher(),
        stdin,
        target: ProcessSpec::new("/bin/cat").clear_environment(),
    };
    let capture = capture_process_with_stdin_file(&request).unwrap();
    assert!(!capture.process_success());
    assert!(capture.process_capture.stderr_lossy().contains("stdin identity mismatch"));
    let _ = fs::remove_dir_all(path.parent().unwrap());
}
