// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};

const MEMORY_LIMIT_BYTES: u64 = 128 * 1024 * 1024;
const MEMORY_PRESSURE_BYTES: usize = 512 * 1024 * 1024;
const CPU_QUOTA_US: u64 = 5_000;
const CPU_PERIOD_US: u64 = 100_000;

fn main() -> Result<(), Box<dyn Error>> {
    match std::env::args().nth(1).as_deref() {
        Some("--child-pids") => child_pids(),
        Some("--child-memory") => child_memory(),
        Some("--child-cpu") => child_cpu(),
        Some(other) => Err(io::Error::other(format!("unknown child mode: {other}")).into()),
        None => parent_probe(),
    }
}

fn parent_probe() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")?);
    probe_pids(&root)?;
    println!("pids_max_enforced=pass");
    probe_memory(&root)?;
    println!("memory_max_enforced=pass");
    probe_cpu(&root)?;
    println!("cpu_max_enforced=pass");
    println!("cgroup_v2_limits_probe=pass");
    Ok(())
}

fn probe_pids(root: &Path) -> Result<(), Box<dyn Error>> {
    let lease = lease(
        root,
        "pids",
        CgroupV2Limits {
            memory_max_bytes: MEMORY_LIMIT_BYTES,
            pids_max: 1,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        },
    )?;
    let mut child = spawn_blocked("--child-pids")?;
    place_or_terminate(&lease, &mut child)?;
    let pids_current = read_scalar_u64(&lease.leaf().join("pids.current"))?;
    if pids_current != 1 {
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::other(format!(
            "expected exactly one task before pids.max trigger, observed {pids_current}"
        ))
        .into());
    }
    trigger(&mut child)?;
    let status = child.wait()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "pids child did not observe EAGAIN: status={status}"
        ))
        .into());
    }
    cleanup(lease)?;
    Ok(())
}

fn probe_memory(root: &Path) -> Result<(), Box<dyn Error>> {
    let lease = lease(
        root,
        "memory",
        CgroupV2Limits {
            memory_max_bytes: MEMORY_LIMIT_BYTES,
            pids_max: 8,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        },
    )?;
    let mut child = spawn_blocked("--child-memory")?;
    place_or_terminate(&lease, &mut child)?;
    // Stateful memory charged before migration remains with the previous cgroup.
    // Baseline only after placement so this probe measures new post-placement
    // pressure triggered below, not pre-migration process startup memory.
    let before = read_counters(&lease.leaf().join("memory.events"))?;
    trigger(&mut child)?;
    let status = child.wait()?;
    let after = read_counters(&lease.leaf().join("memory.events"))?;
    let oom_before = counter(&before, "oom");
    let oom_after = counter(&after, "oom");
    let kill_before = counter(&before, "oom_kill");
    let kill_after = counter(&after, "oom_kill");
    if status.success() || (oom_after <= oom_before && kill_after <= kill_before) {
        return Err(io::Error::other(format!(
            "memory limit did not produce observable OOM enforcement: status={status}, before={before:?}, after={after:?}"
        ))
        .into());
    }
    cleanup(lease)?;
    Ok(())
}

fn probe_cpu(root: &Path) -> Result<(), Box<dyn Error>> {
    let lease = lease(
        root,
        "cpu",
        CgroupV2Limits {
            memory_max_bytes: MEMORY_LIMIT_BYTES,
            pids_max: 8,
            cpu_quota_us: CPU_QUOTA_US,
            cpu_period_us: CPU_PERIOD_US,
        },
    )?;
    let mut child = spawn_blocked("--child-cpu")?;
    place_or_terminate(&lease, &mut child)?;
    let before = read_counters(&lease.leaf().join("cpu.stat"))?;
    trigger(&mut child)?;
    let status = child.wait()?;
    if !status.success() {
        return Err(io::Error::other(format!("cpu pressure child failed: {status}")).into());
    }
    let after = read_counters(&lease.leaf().join("cpu.stat"))?;
    let throttled_before = counter(&before, "nr_throttled");
    let throttled_after = counter(&after, "nr_throttled");
    let usec_before = counter(&before, "throttled_usec");
    let usec_after = counter(&after, "throttled_usec");
    if throttled_after <= throttled_before && usec_after <= usec_before {
        return Err(io::Error::other(format!(
            "cpu.max did not produce observable throttling: before={before:?}, after={after:?}"
        ))
        .into());
    }
    cleanup(lease)?;
    Ok(())
}

fn child_pids() -> Result<(), Box<dyn Error>> {
    wait_for_trigger()?;
    match Command::new("/bin/true").status() {
        Err(error)
            if error.raw_os_error() == Some(libc::EAGAIN)
                || error.kind() == io::ErrorKind::WouldBlock =>
        {
            Ok(())
        }
        Err(error) => Err(io::Error::other(format!(
            "nested process creation failed for unexpected reason: {error}"
        ))
        .into()),
        Ok(status) => Err(io::Error::other(format!(
            "nested process unexpectedly escaped pids.max=1: {status}"
        ))
        .into()),
    }
}

fn child_memory() -> Result<(), Box<dyn Error>> {
    wait_for_trigger()?;
    let mut bytes = vec![0u8; MEMORY_PRESSURE_BYTES];
    for index in (0..bytes.len()).step_by(4096) {
        bytes[index] = bytes[index].wrapping_add(1);
        black_box(bytes[index]);
    }
    Err(io::Error::other("memory pressure unexpectedly completed above memory.max").into())
}

fn child_cpu() -> Result<(), Box<dyn Error>> {
    wait_for_trigger()?;
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    while Instant::now() < deadline {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        black_box(state);
    }
    Ok(())
}

fn wait_for_trigger() -> io::Result<()> {
    let mut byte = [0u8; 1];
    io::stdin().read_exact(&mut byte)?;
    if byte[0] != 1 {
        return Err(io::Error::other("invalid enforcement-probe trigger"));
    }
    Ok(())
}

fn lease(root: &Path, suffix: &str, limits: CgroupV2Limits) -> Result<CgroupV2Lease, Box<dyn Error>> {
    let name = format!("symthaea-enforcement-{}-{suffix}", std::process::id());
    Ok(CgroupV2Lease::create(root, &name, limits)?)
}

fn spawn_blocked(mode: &str) -> Result<Child, Box<dyn Error>> {
    Ok(Command::new(std::env::current_exe()?)
        .arg(mode)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()?)
}

fn place_or_terminate(lease: &CgroupV2Lease, child: &mut Child) -> Result<(), Box<dyn Error>> {
    if let Err(error) = lease.place_pid(child.id()) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error.into());
    }
    Ok(())
}

fn trigger(child: &mut Child) -> Result<(), Box<dyn Error>> {
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("child stdin pipe missing"))?;
    if let Err(error) = stdin.write_all(&[1]) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error.into());
    }
    drop(stdin);
    Ok(())
}

fn cleanup(lease: CgroupV2Lease) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated()? {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            return Err(io::Error::other("cgroup remained populated after child exit").into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty()?;
    Ok(())
}

fn read_scalar_u64(path: &Path) -> io::Result<u64> {
    fs::read_to_string(path)?
        .trim()
        .parse::<u64>()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn read_counters(path: &Path) -> io::Result<BTreeMap<String, u64>> {
    let contents = fs::read_to_string(path)?;
    let mut counters = BTreeMap::new();
    for line in contents.lines() {
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(value) = fields.next() else {
            continue;
        };
        if let Ok(value) = value.parse::<u64>() {
            counters.insert(name.to_owned(), value);
        }
    }
    Ok(counters)
}

fn counter(counters: &BTreeMap<String, u64>, name: &str) -> u64 {
    counters.get(name).copied().unwrap_or(0)
}
