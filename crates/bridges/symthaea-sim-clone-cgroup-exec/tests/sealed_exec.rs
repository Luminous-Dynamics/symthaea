// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_clone_cgroup_exec::{
    spawn_sealed_image_in_cgroup, SealedExecveatCgroupError,
    SEALED_EXECVEAT_CGROUP_PROFILE_V1,
};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};
use symthaea_sim_worker_image::SealedWorkerImage;

fn lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-execveat-{}-{suffix}", std::process::id());
    CgroupV2Lease::create(
        root,
        &leaf,
        CgroupV2Limits {
            memory_max_bytes: 256 * 1024 * 1024,
            pids_max: 8,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        },
    )
    .unwrap()
}

fn wait_empty_then_remove(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().unwrap() {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("execveat cgroup remained populated after child exit");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

#[test]
#[ignore = "requires explicit cgroup-v2 resource root and Linux clone3/execveat"]
fn sealed_elf_executes_from_birth_cgroup_after_source_path_replacement() {
    let source = PathBuf::from(env!("CARGO_BIN_EXE_symthaea-sim-clone-exec-target"));
    let temp = std::env::temp_dir().join(format!(
        "symthaea-clone-exec-source-{}",
        std::process::id()
    ));
    fs::copy(&source, &temp).unwrap();

    let image = SealedWorkerImage::from_path(&temp).unwrap();
    let image_sha = image.image_sha256();
    fs::write(&temp, b"source pathname replaced after sealing").unwrap();

    let lease = lease("success");
    let mut child = spawn_sealed_image_in_cgroup(&image, &lease).unwrap();
    let evidence = child.evidence().clone();
    evidence.verify().unwrap();
    assert_eq!(evidence.profile, SEALED_EXECVEAT_CGROUP_PROFILE_V1);
    assert_eq!(evidence.image_sha256, hex_digest(image_sha));
    assert_eq!(evidence.proc_exe_sha256, hex_digest(image_sha));
    assert!(evidence.birth_membership_observed);
    assert!(evidence.exec_status_eof_observed);
    assert!(evidence.no_new_privs_before_exec);
    assert!(evidence.executable_fd_closed_on_exec);
    assert!(lease.contains_pid(child.pid()).unwrap());

    let mut ready = String::new();
    BufReader::new(child.stdout_mut())
        .read_line(&mut ready)
        .unwrap();
    assert_eq!(ready, "exec_ready\n");

    child.stdin_mut().unwrap().write_all(&[0x5a]).unwrap();
    child.close_stdin();
    let (status, released) = child.wait().unwrap();
    assert!(status.success());
    released.verify().unwrap();

    wait_empty_then_remove(lease);
    let _ = fs::remove_file(temp);
}

#[test]
#[ignore = "requires explicit cgroup-v2 resource root and Linux clone3/execveat"]
fn non_elf_sealed_image_reports_exec_failure_and_is_reaped() {
    let temp = std::env::temp_dir().join(format!(
        "symthaea-clone-exec-invalid-{}",
        std::process::id()
    ));
    fs::write(&temp, b"this is immutable but not an ELF executable").unwrap();
    let image = SealedWorkerImage::from_path(&temp).unwrap();
    let lease = lease("failure");

    let error = spawn_sealed_image_in_cgroup(&image, &lease)
        .expect_err("non-ELF memfd must fail through exec-status pipe");
    assert!(matches!(error, SealedExecveatCgroupError::ExecveatFailed(_)));

    wait_empty_then_remove(lease);
    let _ = fs::remove_file(temp);
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}
