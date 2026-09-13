// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_clone_cgroup::{
    spawn_blocked_in_cgroup, CloneIntoCgroupError, CLONE_INTO_CGROUP_PROFILE_V1,
};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")?);
    let leaf = format!("symthaea-clone-into-cgroup-{}", std::process::id());
    let lease = CgroupV2Lease::create(
        root,
        &leaf,
        CgroupV2Limits {
            memory_max_bytes: 256 * 1024 * 1024,
            pids_max: 8,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        },
    )?;

    let child = spawn_blocked_in_cgroup(&lease)?;
    let evidence = child.evidence().clone();
    evidence.verify()?;
    if evidence.profile != CLONE_INTO_CGROUP_PROFILE_V1 {
        return Err("unexpected clone-into-cgroup profile".into());
    }
    if !lease.contains_pid(child.pid())? {
        return Err("blocked clone3 child is not present in exact leaf".into());
    }

    match spawn_blocked_in_cgroup(&lease) {
        Err(CloneIntoCgroupError::TargetAlreadyPopulated) => {}
        Ok(second) => {
            let _ = second.terminate_and_wait();
            return Err("second child unexpectedly entered one-invocation leaf".into());
        }
        Err(error) => {
            return Err(format!("second spawn failed for wrong reason: {error}").into());
        }
    }

    let released = child.release_and_wait()?;
    released.verify()?;

    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated()? {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            return Err("clone-into-cgroup leaf remained populated after child exit".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty()?;

    println!("clone_into_cgroup_probe=pass");
    println!("profile={}", released.profile);
    println!("child_ready_observed={}", released.child_ready_observed);
    println!(
        "birth_membership_observed={}",
        released.birth_membership_observed
    );
    println!("second_spawn_rejected_populated_leaf=true");
    println!("evidence_sha256={}", released.evidence_sha256);
    Ok(())
}
