// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_clone_cgroup_exec::SEALED_EXECVEAT_CGROUP_PROFILE_V1;
use symthaea_sim_clone_cgroup_exec_rlimit::{
    SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1, spawn_sealed_image_in_cgroup_with_rlimits,
};
use symthaea_sim_worker::{SUPERVISOR_PROFILE_V1, SupervisorLimits};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};
use symthaea_sim_worker_image::SealedWorkerImage;

fn lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-execveat-rlimit-{}-{suffix}", std::process::id());
    CgroupV2Lease::create(
        root,
        &leaf,
        CgroupV2Limits {
            memory_max_bytes: 512 * 1024 * 1024,
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
            panic!("rlimit exec cgroup remained populated after child exit");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

#[test]
#[ignore = "requires explicit cgroup-v2 resource root and Linux x86_64 clone3/execveat"]
fn exact_rlimits_session_and_no_new_privs_survive_exec() {
    let source = PathBuf::from(env!("CARGO_BIN_EXE_symthaea-sim-clone-exec-rlimit-target"));
    let temp = std::env::temp_dir().join(format!(
        "symthaea-clone-exec-rlimit-source-{}",
        std::process::id()
    ));
    fs::copy(&source, &temp).unwrap();

    let image = SealedWorkerImage::from_path(&temp).unwrap();
    let image_sha = image.image_sha256();
    fs::write(&temp, b"source pathname replaced after rlimit sealing").unwrap();

    let limits = SupervisorLimits::default();
    let lease = lease("success");
    let mut child = spawn_sealed_image_in_cgroup_with_rlimits(&image, &lease, limits).unwrap();
    let evidence = child.evidence().clone();
    evidence.verify().unwrap();
    assert_eq!(evidence.profile, SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1);
    assert_eq!(evidence.base.profile, SEALED_EXECVEAT_CGROUP_PROFILE_V1);
    assert_eq!(evidence.supervisor_profile, SUPERVISOR_PROFILE_V1);
    assert_eq!(evidence.base.image_sha256, hex_digest(image_sha));
    assert_eq!(evidence.base.proc_exe_sha256, hex_digest(image_sha));
    assert!(evidence.legacy_preexec_rlimits_applied);
    assert!(evidence.setsid_before_exec);
    assert!(evidence.no_new_privs_before_exec);
    assert!(evidence.umask_0077_before_exec);
    assert!(!evidence.parent_wall_time_enforced_by_this_launcher);
    assert!(!evidence.parent_output_limits_enforced_by_this_launcher);
    assert!(lease.contains_pid(child.pid()).unwrap());

    let mut observed = BTreeMap::<String, String>::new();
    {
        let mut reader = BufReader::new(child.stdout_mut());
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert!(!line.is_empty(), "target exited before exec_ready");
            let line = line.trim_end();
            if line == "exec_ready" {
                break;
            }
            let (key, value) = line.split_once('=').expect("key=value probe line");
            observed.insert(key.to_owned(), value.to_owned());
        }
    }

    let pid = child.pid().to_string();
    assert_eq!(observed.get("pid"), Some(&pid));
    assert_eq!(observed.get("sid"), Some(&pid));
    assert_eq!(observed.get("pgrp"), Some(&pid));
    assert_eq!(observed.get("no_new_privs").map(String::as_str), Some("1"));
    assert_eq!(observed.get("umask").map(String::as_str), Some("0077"));

    assert_limit(&observed, "as", limits.address_space_bytes);
    assert_limit(&observed, "cpu", limits.cpu_seconds);
    assert_limit(&observed, "fsize", limits.file_size_bytes);
    assert_limit(&observed, "nofile", limits.open_files);
    assert_limit(&observed, "nproc", limits.process_count);
    assert_limit(&observed, "core", 0);

    child.stdin_mut().unwrap().write_all(&[0x5a]).unwrap();
    child.close_stdin();
    let (status, released) = child.wait().unwrap();
    assert!(status.success());
    released.verify().unwrap();

    wait_empty_then_remove(lease);
    let _ = fs::remove_file(temp);
}

fn assert_limit(observed: &BTreeMap<String, String>, name: &str, expected: u64) {
    let expected = expected.to_string();
    assert_eq!(
        observed.get(&format!("rlimit_{name}_cur")),
        Some(&expected),
        "soft {name} limit differs"
    );
    assert_eq!(
        observed.get(&format!("rlimit_{name}_max")),
        Some(&expected),
        "hard {name} limit differs"
    );
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
