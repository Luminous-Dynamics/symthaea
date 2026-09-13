// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::io;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits, CGROUP_V2_PROFILE_V1};

const ROOT_ENV: &str = "SYMTHAEA_CGROUP_V2_DELEGATED_ROOT";

fn main() -> Result<(), Box<dyn Error>> {
    let root = match std::env::var(ROOT_ENV) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            println!("cgroup_v2_probe=unavailable:no_explicit_resource_root");
            std::process::exit(2);
        }
    };

    let leaf_name = format!("symthaea-probe-{}", std::process::id());
    let limits = CgroupV2Limits {
        memory_max_bytes: 256 * 1024 * 1024,
        pids_max: 16,
        cpu_quota_us: 50_000,
        cpu_period_us: 100_000,
    };
    let lease = CgroupV2Lease::create(&root, &leaf_name, limits)?;
    let evidence = lease.evidence()?;
    evidence.verify()?;

    let child = Command::new("/bin/sleep")
        .arg("30")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            lease.remove_empty()?;
            return Err(error.into());
        }
    };
    let child_pid = child.id();

    if let Err(error) = lease.place_pid(child_pid) {
        let _ = child.kill();
        let _ = child.wait();
        if !lease.populated().unwrap_or(true) {
            let _ = lease.remove_empty();
        }
        return Err(error.into());
    }
    if !lease.contains_pid(child_pid)? {
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::other("placed child is not visible in cgroup.procs").into());
    }

    child.kill()?;
    child.wait()?;
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated()? {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            return Err(io::Error::other("cgroup remained populated after child exit").into());
        }
        thread::sleep(Duration::from_millis(10));
    }

    println!("cgroup_v2_probe=pass");
    println!("profile={CGROUP_V2_PROFILE_V1}");
    println!("evidence_sha256={}", evidence.evidence_sha256);
    println!("root_device={}", evidence.delegated_root_device);
    println!("root_inode={}", evidence.delegated_root_inode);
    println!("leaf_device={}", evidence.leaf_device);
    println!("leaf_inode={}", evidence.leaf_inode);
    println!("memory_max_bytes={}", evidence.memory_max_bytes);
    println!("pids_max={}", evidence.pids_max);
    println!("cpu_max={} {}", evidence.cpu_quota_us, evidence.cpu_period_us);
    println!("memory_oom_group={}", evidence.memory_oom_group);

    lease.remove_empty()?;
    Ok(())
}
