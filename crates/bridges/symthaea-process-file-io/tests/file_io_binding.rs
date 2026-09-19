// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![cfg(unix)]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use symthaea_process_capture::ProcessSpec;
use symthaea_process_file_io::{
    BoundFileIoProcessRequest, FileIoLauncherArtifact, NewStdoutFile,
    capture_process_with_file_io,
};
use symthaea_process_stdin::ContentAddressedStdinFile;

fn unique_dir() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "symthaea-process-file-io-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn launcher() -> FileIoLauncherArtifact {
    FileIoLauncherArtifact::observe(PathBuf::from(env!(
        "CARGO_BIN_EXE_symthaea-process-file-io-launcher"
    ))
    .as_path())
    .unwrap()
}

#[test]
fn verified_input_is_copied_to_create_new_output() {
    let dir = unique_dir();
    let input = dir.join("input.bin");
    let output = dir.join("output.bin");
    let payload = b"large-artifact fixture bytes\n";
    fs::write(&input, payload).unwrap();
    let request = BoundFileIoProcessRequest {
        launcher: launcher(),
        stdin: ContentAddressedStdinFile::observe(&input).unwrap(),
        stdout: NewStdoutFile::new(&output).unwrap(),
        target: ProcessSpec::new("/bin/cat").clear_environment(),
    };
    let capture = capture_process_with_file_io(&request).unwrap();
    assert!(capture.complete_output());
    assert_eq!(fs::read(&output).unwrap(), payload);
    assert!(capture.process_capture.stdout.is_empty());
    assert_eq!(capture.observed_stdout.as_ref().unwrap().bytes, payload.len() as u64);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn wrong_input_identity_never_creates_output() {
    let dir = unique_dir();
    let input = dir.join("input.bin");
    let output = dir.join("output.bin");
    fs::write(&input, b"actual bytes").unwrap();
    let mut stdin = ContentAddressedStdinFile::observe(&input).unwrap();
    stdin.sha256 = "0".repeat(64);
    let request = BoundFileIoProcessRequest {
        launcher: launcher(),
        stdin,
        stdout: NewStdoutFile::new(&output).unwrap(),
        target: ProcessSpec::new("/bin/cat").clear_environment(),
    };
    let capture = capture_process_with_file_io(&request).unwrap();
    assert!(!capture.process_capture.process_success());
    assert!(capture.observed_stdout.is_none());
    assert!(!output.exists());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn existing_output_fails_before_launch() {
    let dir = unique_dir();
    let input = dir.join("input.bin");
    let output = dir.join("output.bin");
    fs::write(&input, b"input").unwrap();
    fs::write(&output, b"do-not-overwrite").unwrap();
    let request = BoundFileIoProcessRequest {
        launcher: launcher(),
        stdin: ContentAddressedStdinFile::observe(&input).unwrap(),
        stdout: NewStdoutFile::new(&output).unwrap(),
        target: ProcessSpec::new("/bin/cat").clear_environment(),
    };
    assert!(capture_process_with_file_io(&request).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"do-not-overwrite");
    let _ = fs::remove_dir_all(dir);
}
