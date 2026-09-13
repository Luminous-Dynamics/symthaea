// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs::{self, OpenOptions};
use std::io;
use symthaea_sim_worker_filesystem::{
    WORKER_FILESYSTEM_PROFILE_V1, install_worker_filesystem_containment,
};

fn main() {
    let preexisting = format!(
        "/tmp/symthaea-landlock-existing-{}",
        std::process::id()
    );
    fs::write(&preexisting, b"created before containment")
        .expect("prepare existing file before containment");

    let status = install_worker_filesystem_containment()
        .expect("install seccomp + Landlock filesystem containment");

    expect_permission_denied("read /etc/passwd", || {
        fs::read("/etc/passwd").map(|_| ())
    });
    expect_permission_denied("iterate /", || {
        let mut entries = fs::read_dir("/")?;
        match entries.next() {
            Some(entry) => entry.map(|_| ()),
            None => Ok(()),
        }
    });

    let candidate = format!(
        "/tmp/symthaea-landlock-new-{}",
        std::process::id()
    );
    expect_permission_denied("create /tmp file", || {
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&candidate)
            .map(|_| ())
    });
    expect_permission_denied("write existing /tmp file", || {
        OpenOptions::new().write(true).open(&preexisting).map(|_| ())
    });
    expect_permission_denied("truncate existing /tmp file", || {
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&preexisting)
            .map(|_| ())
    });
    expect_permission_denied("remove existing /tmp file", || {
        fs::remove_file(&preexisting)
    });

    // Deliberate nonclaim: Landlock does not currently hide all path metadata.
    // Freeze that distinction so audit prose cannot drift into "filesystem
    // invisibility" while the actual policy is content/mutation confinement.
    fs::metadata("/etc/passwd").expect("metadata observation remains out of scope");
    fs::metadata(&preexisting).expect("existing-file metadata remains observable");

    println!("filesystem_profile={WORKER_FILESYSTEM_PROFILE_V1}");
    println!("landlock_abi={}", status.landlock_abi);
    println!("handled_access_fs=0x{:x}", status.handled_access_fs);
    println!("content_read=denied");
    println!("directory_read=denied");
    println!("file_create=denied");
    println!("existing_file_write=denied");
    println!("existing_file_truncate=denied");
    println!("existing_file_remove=denied");
    println!("metadata_visibility=not_restricted");
    println!("filesystem_probe=pass");
}

fn expect_permission_denied(name: &str, operation: impl FnOnce() -> io::Result<()>) {
    let error = operation().expect_err(name);
    assert_eq!(
        error.kind(),
        io::ErrorKind::PermissionDenied,
        "{name} returned unexpected error: {error}"
    );
}
