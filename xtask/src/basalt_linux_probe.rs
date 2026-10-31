//! Read-only Linux capability inventory for BASALT observer-profile selection.
//!
//! This inventories exposed kernel interfaces and local tools. It does not
//! attach observers, load BPF programs, create namespaces, change policy,
//! execute a workload, or establish that any event was captured.

use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_TEXT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_HASHED_FILE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_MOUNT_ENTRIES: usize = 16_384;
const RELEVANT_MOUNTS: &[&str] = &[
    "bpf", "tracefs", "cgroup2",
];

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Collected,
    Absent,
    Inaccessible,
    Disabled,
    NotProbed,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatus {
    CandidateAvailable,
    NotAvailable,
    Unknown,
    Inaccessible,
}

#[derive(Debug, Serialize)]
pub struct LinuxProbeReport {
    pub report_version: u32,
    pub suite_id: &'static str,
    pub probe_id: String,
    pub observed_at_unix_ns: u64,
    pub kernel: KernelIdentity,
    pub mounts: Vec<MountEntry>,
    pub mount_inventory: ProbeEvidence,
    pub capabilities: Vec<CapabilityAssessment>,
    pub limitations: Vec<&'static str>,
    pub qualification_eligible: bool,
    pub observer_attached: bool,
    pub workload_executed: bool,
}

#[derive(Debug, Serialize)]
pub struct KernelIdentity {
    pub release: ProbeEvidence,
    pub architecture: String,
    pub active_lsm: ProbeEvidence,
    pub btf_vmlinux: ProbeEvidence,
    pub kernel_config: ProbeEvidence,
}

#[derive(Debug, Serialize)]
pub struct MountEntry {
    pub mount_id: u64,
    pub root: String,
    pub mount_point: String,
    pub filesystem: String,
}

#[derive(Debug, Serialize)]
pub struct CapabilityAssessment {
    pub capability_id: &'static str,
    pub status: CapabilityStatus,
    pub evidence: Vec<ProbeEvidence>,
    pub limitations: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProbeEvidence {
    pub key: String,
    pub status: EvidenceStatus,
    pub value: Option<String>,
    pub content_digest: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug)]
struct ConfigEvidence {
    evidence: ProbeEvidence,
    flags: BTreeMap<String, bool>,
}

pub fn run(out_path: &Path) -> anyhow::Result<LinuxProbeReport> {
    if !cfg!(target_os = "linux") {
        bail!("BASALT Linux capability inventory is supported only on Linux");
    }

    let report = probe_linux()?;
    let bytes = serde_json::to_vec_pretty(&report)?;
    write_new(out_path, &bytes)?;
    Ok(report)
}

pub fn probe_linux() -> anyhow::Result<LinuxProbeReport> {
    if !cfg!(target_os = "linux") {
        bail!("BASALT Linux capability inventory is supported only on Linux");
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?
        .as_nanos();
    let observed_at_unix_ns = u64::try_from(timestamp)
        .context("probe timestamp exceeds u64 nanoseconds")?;
    let probe_id = format!("basalt-linux-probe-{}-{observed_at_unix_ns}", std::process::id());

    let release = read_text_evidence("kernel_release", Path::new("/proc/sys/kernel/osrelease"));
    let active_lsm = read_text_evidence("active_lsm", Path::new("/sys/kernel/security/lsm"));
    let btf_vmlinux = digest_file_evidence("btf_vmlinux", Path::new("/sys/kernel/btf/vmlinux"));
    let config = read_kernel_config(release.value.as_deref().unwrap_or("unknown"));
    let mountinfo_raw = read_text_evidence("mountinfo_raw", Path::new("/proc/self/mountinfo"));
    let mounts = mountinfo_raw.value.as_deref().map(parse_relevant_mounts).unwrap_or_default();
    // Keep only a digest and bounded summary in the report; raw mountinfo can
    // contain unrelated host paths and is not needed for profile selection.
    let mount_text = ProbeEvidence {
        key: "mount_inventory".to_string(),
        status: mountinfo_raw.status,
        value: Some(format!("relevant_mounts={}", mounts.len())),
        content_digest: mountinfo_raw.content_digest.clone(),
        path: mountinfo_raw.path.clone(),
    };

    let (bpftool_evidence, bpftool_version) = match find_executable("bpftool") {
        Some(path) => (
            executable_evidence("bpftool", &path),
            command_version_evidence("bpftool_version", &path),
        ),
        None => (
            missing_evidence("bpftool", "bpftool"),
            not_probed_evidence("bpftool_version", "bpftool not found in PATH"),
        ),
    };

    let bpf_lsm_active = has_active_lsm(active_lsm.value.as_deref(), "bpf");
    let btf_available = btf_vmlinux.status == EvidenceStatus::Collected;
    let bpf_config = config.flags.get("CONFIG_BPF_LSM").copied();
    let bpf_syscall_config = config.flags.get("CONFIG_BPF_SYSCALL").copied();
    let bpf_status = assess_bpf_lsm(
        bpf_lsm_active,
        active_lsm.status,
        btf_available,
        bpf_config,
        bpf_syscall_config,
    );

    let tracefs_mount = mounts.iter().any(|mount| {
        mount.filesystem == "tracefs"
            || (mount.filesystem == "debugfs" && mount.mount_point == "/sys/kernel/debug")
    });
    let trace_events = trace_event_evidence();
    let trace_events_available = trace_events.iter()
        .all(|item| item.status == EvidenceStatus::Collected);
    let trace_status = if tracefs_mount && trace_events_available {
        CapabilityStatus::CandidateAvailable
    } else if !tracefs_mount && mount_text.status == EvidenceStatus::Collected {
        CapabilityStatus::NotAvailable
    } else if trace_events.iter().any(|item| item.status == EvidenceStatus::Inaccessible) {
        CapabilityStatus::Inaccessible
    } else {
        CapabilityStatus::Unknown
    };

    let fanotify_config = config.flags.get("CONFIG_FANOTIFY").copied();
    let fanotify_permissions = config.flags.get("CONFIG_FANOTIFY_ACCESS_PERMISSIONS").copied();
    let fanotify_status = match fanotify_config {
        Some(true) => CapabilityStatus::CandidateAvailable,
        Some(false) => CapabilityStatus::NotAvailable,
        None => CapabilityStatus::Unknown,
    };
    let fanotify_permission_status = match (fanotify_config, fanotify_permissions) {
        (Some(false), _) | (_, Some(false)) => CapabilityStatus::NotAvailable,
        (Some(true), Some(true)) => CapabilityStatus::CandidateAvailable,
        _ => CapabilityStatus::Unknown,
    };

    let landlock_config = config.flags.get("CONFIG_SECURITY_LANDLOCK").copied();
    let landlock_status = match landlock_config {
        Some(true) => CapabilityStatus::CandidateAvailable,
        Some(false) => CapabilityStatus::NotAvailable,
        None => CapabilityStatus::Unknown,
    };

    let cgroup_v2_mount = mounts.iter().any(|mount| mount.filesystem == "cgroup2");
    let cgroup_membership = read_text_evidence("cgroup_membership", Path::new("/proc/self/cgroup"));
    let cgroup_status = if cgroup_v2_mount && cgroup_membership.status == EvidenceStatus::Collected {
        CapabilityStatus::CandidateAvailable
    } else if !cgroup_v2_mount && mount_text.status == EvidenceStatus::Collected {
        CapabilityStatus::NotAvailable
    } else {
        CapabilityStatus::Unknown
    };

    let mut capabilities = vec![
        CapabilityAssessment {
            capability_id: "bpf_lsm",
            status: bpf_status,
            evidence: vec![
                active_lsm.clone(),
                btf_vmlinux.clone(),
                config.evidence.clone(),
                bpftool_evidence,
                bpftool_version,
            ],
            limitations: vec![
                "Interface presence does not prove the intended LSM hook set is attachable.",
                "No BPF program was loaded or attached by this inventory.",
                "BTF presence does not prove CO-RE relocation or verifier acceptance.",
                "Runtime event loss and observer lifecycle require a separate capture-time receipt.",
            ],
        },
        CapabilityAssessment {
            capability_id: "tracefs_process_events",
            status: trace_status,
            evidence: {
                let mut evidence = vec![mount_text.clone()];
                evidence.extend(trace_events);
                evidence
            },
            limitations: vec![
                "Tracepoint availability does not prove events were enabled or consumed.",
                "The inventory does not test per-process filtering, event loss, or event ordering.",
            ],
        },
        CapabilityAssessment {
            capability_id: "fanotify_notifications",
            status: fanotify_status,
            evidence: vec![
                config.evidence.clone(),
                not_probed_evidence("fanotify_runtime_api", "no fanotify_init/fanotify_mark syscall was executed"),
            ],
            limitations: vec![
                "Kernel configuration is not a runtime API or privilege probe.",
                "fanotify does not report mmap-based file access.",
                "fanotify does not report remote events on network filesystems.",
                "Recursive directory marking can race; event queues can overflow.",
            ],
        },
        CapabilityAssessment {
            capability_id: "fanotify_access_permissions",
            status: fanotify_permission_status,
            evidence: vec![config.evidence.clone()],
            limitations: vec![
                "Kernel configuration does not prove this process has permission to create a permission-event group.",
                "No permission event or fanotify response was generated by this inventory.",
            ],
        },
        CapabilityAssessment {
            capability_id: "landlock",
            status: landlock_status,
            evidence: vec![
                config.evidence.clone(),
                not_probed_evidence("landlock_abi", "no landlock_create_ruleset syscall was executed"),
            ],
            limitations: vec![
                "Kernel configuration is not a Landlock ABI/version probe.",
                "Landlock is an access-control mechanism, not an activity recorder.",
                "Some special filesystem objects remain outside explicit path restrictions.",
            ],
        },
        CapabilityAssessment {
            capability_id: "cgroup_v2_process_scope",
            status: cgroup_status,
            evidence: vec![mount_text.clone(), cgroup_membership],
            limitations: vec![
                "The inventory describes the current probe process scope, not a future workload cgroup.",
                "PIDs read from cgroup.procs can race with migration or PID reuse.",
            ],
        },
    ];

    capabilities.sort_by_key(|capability| capability.capability_id);

    Ok(LinuxProbeReport {
        report_version: 1,
        suite_id: "basalt-linux-capability-inventory-v1",
        probe_id,
        observed_at_unix_ns,
        kernel: KernelIdentity {
            release,
            architecture: env::consts::ARCH.to_string(),
            active_lsm,
            btf_vmlinux,
            kernel_config: config.evidence,
        },
        mounts,
        mount_inventory: mount_text,
        capabilities,
        limitations: vec![
            "This is a read-only interface inventory, not an observer execution.",
            "CandidateAvailable means interfaces/configuration were found, not that runtime capture was verified.",
            "No kernel programs, hooks, namespaces, cgroups, policies, or mount changes are created.",
            "No workload is executed and no dependency event is collected.",
            "Missing kernel configuration files produce Unknown, not an inferred Disabled result.",
        ],
        qualification_eligible: false,
        observer_attached: false,
        workload_executed: false,
    })
}

fn has_active_lsm(active_lsm: Option<&str>, target: &str) -> Option<bool> {
    active_lsm.map(|value| value.split(',').any(|name| name.trim() == target))
}

fn assess_bpf_lsm(
    active: Option<bool>,
    active_lsm_status: EvidenceStatus,
    btf_available: bool,
    bpf_lsm_config: Option<bool>,
    bpf_syscall_config: Option<bool>,
) -> CapabilityStatus {
    if bpf_syscall_config == Some(false) || bpf_lsm_config == Some(false) {
        return if active == Some(true) {
            CapabilityStatus::Unknown
        } else {
            CapabilityStatus::NotAvailable
        };
    }
    match active {
        Some(false) => CapabilityStatus::NotAvailable,
        Some(true) if btf_available => CapabilityStatus::CandidateAvailable,
        Some(true) => CapabilityStatus::Unknown,
        None if active_lsm_status == EvidenceStatus::Inaccessible => CapabilityStatus::Inaccessible,
        None => CapabilityStatus::Unknown,
    }
}

fn read_kernel_config(release: &str) -> ConfigEvidence {
    let candidates = [
        PathBuf::from(format!("/boot/config-{release}")),
        PathBuf::from(format!("/lib/modules/{release}/build/.config")),
        PathBuf::from(format!("/usr/lib/modules/{release}/build/.config")),
    ];

    let mut inaccessible = false;
    for path in candidates {
        match read_bounded_text(&path) {
            Ok(text) => {
                let mut flags = BTreeMap::new();
                for key in [
                    "CONFIG_BPF_SYSCALL",
                    "CONFIG_BPF_LSM",
                    "CONFIG_FANOTIFY",
                    "CONFIG_FANOTIFY_ACCESS_PERMISSIONS",
                    "CONFIG_SECURITY_LANDLOCK",
                ] {
                    if let Some(value) = parse_config_flag(&text, key) {
                        flags.insert(key.to_string(), value);
                    }
                }
                return ConfigEvidence {
                    evidence: ProbeEvidence {
                        key: "kernel_config".to_string(),
                        status: EvidenceStatus::Collected,
                        content_digest: Some(sha256_tagged(text.as_bytes())),
                        value: Some(format!("parsed_flags={}", flags.len())),
                        path: Some(path.display().to_string()),
                    },
                    flags,
                };
            }
            Err(error) => {
                if error.kind() == std::io::ErrorKind::PermissionDenied {
                    inaccessible = true;
                }
                continue;
            }
        }
    }

    ConfigEvidence {
        evidence: ProbeEvidence {
            key: "kernel_config".to_string(),
            status: if inaccessible { EvidenceStatus::Inaccessible } else { EvidenceStatus::NotProbed },
            value: None,
            content_digest: None,
            path: Some("checked /boot/config-<release>, /lib/modules/<release>/build/.config, /usr/lib/modules/<release>/build/.config".to_string()),
        },
        flags: BTreeMap::new(),
    }
}

fn parse_config_flag(config: &str, key: &str) -> Option<bool> {
    for line in config.lines() {
        if let Some(value) = line.strip_prefix(&format!("{key}=")) {
            return match value.trim() {
                "y" | "m" => Some(true),
                "n" => Some(false),
                _ => None,
            };
        }
        if line.trim() == format!("# {key} is not set") {
            return Some(false);
        }
    }
    None
}

fn trace_event_evidence() -> Vec<ProbeEvidence> {
    let bases = [
        Path::new("/sys/kernel/tracing/events/sched"),
        Path::new("/sys/kernel/debug/tracing/events/sched"),
    ];
    let events = [
        ("sched_process_fork", "process_fork_tracepoint"),
        ("sched_process_exec", "process_exec_tracepoint"),
        ("sched_process_exit", "process_exit_tracepoint"),
    ];
    events.iter().map(|(event, key)| {
        for base in bases {
            let path = base.join(event).join("id");
            if let Ok(text) = read_bounded_text(&path) {
                return ProbeEvidence {
                    key: (*key).to_string(),
                    status: EvidenceStatus::Collected,
                    value: Some(text.trim().to_string()),
                    content_digest: Some(sha256_tagged(text.as_bytes())),
                    path: Some(path.display().to_string()),
                };
            }
        }
        missing_evidence(key, event)
    }).collect()
}

fn read_text_evidence(key: &str, path: &Path) -> ProbeEvidence {
    match read_bounded_text(path) {
        Ok(text) => ProbeEvidence {
            key: key.to_string(),
            status: EvidenceStatus::Collected,
            content_digest: Some(sha256_tagged(text.as_bytes())),
            value: Some(text.trim().to_string()),
            path: Some(path.display().to_string()),
        },
        Err(error) => {
            let status = match error.kind() {
                std::io::ErrorKind::NotFound => EvidenceStatus::Absent,
                std::io::ErrorKind::PermissionDenied => EvidenceStatus::Inaccessible,
                _ => EvidenceStatus::NotProbed,
            };
            ProbeEvidence {
                key: key.to_string(),
                status,
                value: None,
                content_digest: None,
                path: Some(path.display().to_string()),
            }
        }
    }
}

fn digest_file_evidence(key: &str, path: &Path) -> ProbeEvidence {
    match sha256_file_bounded(path, MAX_HASHED_FILE_BYTES) {
        Ok(digest) => ProbeEvidence {
            key: key.to_string(),
            status: EvidenceStatus::Collected,
            value: Some("present".to_string()),
            content_digest: Some(digest),
            path: Some(path.display().to_string()),
        },
        Err(error) => {
            let status = match error.kind() {
                std::io::ErrorKind::NotFound => EvidenceStatus::Absent,
                std::io::ErrorKind::PermissionDenied => EvidenceStatus::Inaccessible,
                _ => EvidenceStatus::NotProbed,
            };
            ProbeEvidence {
                key: key.to_string(),
                status,
                value: None,
                content_digest: None,
                path: Some(path.display().to_string()),
            }
        }
    }
}

fn executable_evidence(key: &str, path: &Path) -> ProbeEvidence {
    match sha256_file_bounded(path, MAX_HASHED_FILE_BYTES) {
        Ok(digest) => ProbeEvidence {
            key: key.to_string(),
            status: EvidenceStatus::Collected,
            value: Some("executable_found".to_string()),
            content_digest: Some(digest),
            path: Some(path.display().to_string()),
        },
        Err(error) => ProbeEvidence {
            key: key.to_string(),
            status: if error.kind() == std::io::ErrorKind::PermissionDenied {
                EvidenceStatus::Inaccessible
            } else {
                EvidenceStatus::NotProbed
            },
            value: None,
            content_digest: None,
            path: Some(path.display().to_string()),
        },
    }
}

fn command_version_evidence(key: &str, executable: &Path) -> ProbeEvidence {
    match Command::new(executable).arg("--version").output() {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            ProbeEvidence {
                key: key.to_string(),
                status: EvidenceStatus::Collected,
                content_digest: Some(sha256_tagged(output.stdout.as_slice())),
                value: Some(text),
                path: Some(executable.display().to_string()),
            }
        }
        Ok(output) => ProbeEvidence {
            key: key.to_string(),
            status: EvidenceStatus::NotProbed,
            content_digest: Some(sha256_tagged(&output.stderr)),
            value: Some(format!("exit={}", output.status)),
            path: Some(executable.display().to_string()),
        },
        Err(_) => not_probed_evidence(key, "version invocation failed"),
    }
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(name);
        if candidate.is_file() {
            if let Ok(canonical) = candidate.canonicalize() {
                return Some(canonical);
            }
        }
    }
    None
}

fn missing_evidence(key: &str, path_or_name: &str) -> ProbeEvidence {
    ProbeEvidence {
        key: key.to_string(),
        status: EvidenceStatus::Absent,
        value: None,
        content_digest: None,
        path: Some(path_or_name.to_string()),
    }
}

fn not_probed_evidence(key: &str, reason: &str) -> ProbeEvidence {
    ProbeEvidence {
        key: key.to_string(),
        status: EvidenceStatus::NotProbed,
        value: Some(reason.to_string()),
        content_digest: None,
        path: None,
    }
}

fn parse_relevant_mounts(mountinfo: &str) -> Vec<MountEntry> {
    let mut mounts = Vec::new();
    for line in mountinfo.lines().take(MAX_MOUNT_ENTRIES) {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let left_fields: Vec<&str> = left.split_whitespace().collect();
        let right_fields: Vec<&str> = right.split_whitespace().collect();
        if left_fields.len() < 5 || right_fields.is_empty() {
            continue;
        }
        let filesystem = right_fields[0];
        let mount_point = unescape_mount_field(left_fields[4]);
        let root = unescape_mount_field(left_fields[3]);
        if RELEVANT_MOUNTS.contains(&filesystem)
            || mount_point == "/sys/kernel/tracing"
            || mount_point == "/sys/kernel/debug"
            || mount_point == "/sys/fs/bpf"
            || mount_point == "/sys/fs/cgroup"
        {
            if let Ok(mount_id) = left_fields[0].parse::<u64>() {
                mounts.push(MountEntry {
                    mount_id,
                    root,
                    mount_point,
                    filesystem: filesystem.to_string(),
                });
            }
        }
    }
    mounts.sort_by(|a, b| {
        (&a.filesystem, &a.mount_point, a.mount_id)
            .cmp(&(&b.filesystem, &b.mount_point, b.mount_id))
    });
    mounts
}

fn unescape_mount_field(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn read_bounded_text(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_TEXT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_TEXT_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "probe input exceeds 8 MiB",
        ));
    }
    String::from_utf8(bytes).map_err(|_| std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "probe input is not valid UTF-8",
    ))
}

fn sha256_file_bounded(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total.saturating_add(count as u64);
        if total > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "probe file exceeds hashing limit",
            ));
        }
        hasher.update(&buffer[..count]);
    }
    Ok(encode_digest(&hasher.finalize()))
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(encode_digest(&hasher.finalize()))
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_digest(&digest)
}

fn encode_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(71);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    format!("sha256:{output}")
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating probe report directory {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating new probe report {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_relevant_mounts_and_unescapes_paths() {
        let input = concat!(
            "21 1 0:20 / /sys/fs/bpf rw,nosuid,nodev - bpf bpf rw\n",
            "22 1 0:21 / /sys/kernel/tracing rw,nosuid,nodev - tracefs tracefs rw\n",
            "23 1 0:22 / /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup rw\n",
            "24 1 0:23 / /mnt/space\\040dir rw - ext4 /dev/sda rw\n"
        );
        let mounts = parse_relevant_mounts(input);
        assert_eq!(mounts.len(), 3);
        assert_eq!(mounts[0].mount_point, "/sys/fs/bpf");
        assert!(mounts.iter().any(|mount| mount.filesystem == "tracefs"));
        assert!(mounts.iter().any(|mount| mount.filesystem == "cgroup2"));
    }

    #[test]
    fn parses_kernel_configuration_enabled_disabled_and_module_states() {
        let config = concat!(
            "CONFIG_BPF_SYSCALL=y\n",
            "CONFIG_BPF_LSM=m\n",
            "# CONFIG_FANOTIFY is not set\n",
            "CONFIG_SECURITY_LANDLOCK=y\n"
        );
        assert_eq!(parse_config_flag(config, "CONFIG_BPF_SYSCALL"), Some(true));
        assert_eq!(parse_config_flag(config, "CONFIG_BPF_LSM"), Some(true));
        assert_eq!(parse_config_flag(config, "CONFIG_FANOTIFY"), Some(false));
        assert_eq!(parse_config_flag(config, "CONFIG_SECURITY_LANDLOCK"), Some(true));
        assert_eq!(parse_config_flag(config, "CONFIG_NOT_PRESENT"), None);
    }

    #[test]
    fn active_bpf_lsm_is_distinct_from_kernel_config_presence() {
        let modules = "capability,landlock,yama,apparmor,bpf";
        assert!(modules.split(',').any(|item| item == "bpf"));
        assert!(!"capability,apparmor".split(',').any(|item| item == "bpf"));
    }

    #[test]
    fn bpf_lsm_candidate_requires_active_module_and_btf() {
        assert_eq!(
            assess_bpf_lsm(Some(true), EvidenceStatus::Collected, true, Some(true), Some(true)),
            CapabilityStatus::CandidateAvailable
        );
        assert_eq!(
            assess_bpf_lsm(Some(true), EvidenceStatus::Collected, false, Some(true), Some(true)),
            CapabilityStatus::Unknown
        );
        assert_eq!(
            assess_bpf_lsm(Some(false), EvidenceStatus::Collected, true, Some(true), Some(true)),
            CapabilityStatus::NotAvailable
        );
    }

    #[test]
    fn contradictory_bpf_configuration_does_not_promote_a_candidate() {
        assert_eq!(
            assess_bpf_lsm(Some(true), EvidenceStatus::Collected, true, Some(false), Some(true)),
            CapabilityStatus::Unknown
        );
    }

    #[test]
    fn active_lsm_parser_distinguishes_present_absent_and_unknown() {
        assert_eq!(has_active_lsm(Some("capability,landlock,bpf"), "bpf"), Some(true));
        assert_eq!(has_active_lsm(Some("capability,apparmor"), "bpf"), Some(false));
        assert_eq!(has_active_lsm(None, "bpf"), None);
    }

    #[test]
    fn fanotify_limitations_are_explicit_not_hidden_by_config_detection() {
        let limitations = [
            "mmap-based access is not reported",
            "remote network filesystem events are not reported",
            "recursive directory marking can race",
            "event queues can overflow",
        ];
        assert_eq!(limitations.len(), 4);
    }

    #[test]
    fn missing_kernel_config_is_not_equivalent_to_disabled() {
        let absent = ConfigEvidence {
            evidence: not_probed_evidence("kernel_config", "config unavailable"),
            flags: BTreeMap::new(),
        };
        assert_eq!(absent.flags.get("CONFIG_FANOTIFY"), None);
        assert_eq!(absent.evidence.status, EvidenceStatus::NotProbed);
    }

    #[test]
    fn capability_status_is_never_a_qualification_claim() {
        let report = LinuxProbeReport {
            report_version: 1,
            suite_id: "basalt-linux-capability-inventory-v1",
            probe_id: "test".to_string(),
            observed_at_unix_ns: 1,
            kernel: KernelIdentity {
                release: not_probed_evidence("kernel", "test"),
                architecture: "x86_64".to_string(),
                active_lsm: not_probed_evidence("lsm", "test"),
                btf_vmlinux: not_probed_evidence("btf", "test"),
                kernel_config: not_probed_evidence("config", "test"),
            },
            mounts: Vec::new(),
            mount_inventory: not_probed_evidence("mountinfo", "test"),
            capabilities: Vec::new(),
            limitations: Vec::new(),
            qualification_eligible: false,
            observer_attached: false,
            workload_executed: false,
        };
        assert!(!report.qualification_eligible);
        assert!(!report.observer_attached);
        assert!(!report.workload_executed);
    }

    #[test]
    fn mount_inventory_persists_digest_and_count_not_unrelated_host_paths() {
        let raw = "1 0 0:1 / /workspace rw - ext4 /dev/root rw\n";
        let relevant = parse_relevant_mounts(raw);
        let evidence = ProbeEvidence {
            key: "mount_inventory".to_string(),
            status: EvidenceStatus::Collected,
            value: Some(format!("relevant_mounts={}", relevant.len())),
            content_digest: Some(sha256_tagged(raw.as_bytes())),
            path: Some("/proc/self/mountinfo".to_string()),
        };
        assert_eq!(evidence.value.as_deref(), Some("relevant_mounts=0"));
        assert_ne!(evidence.content_digest.as_deref(), Some(raw));
    }

    #[test]
    fn recognizes_debugfs_tracing_fallback_as_a_trace_mount() {
        let input = "22 1 0:21 / /sys/kernel/debug rw - debugfs debugfs rw\n";
        let mounts = parse_relevant_mounts(input);
        assert!(mounts.iter().any(|mount| {
            mount.filesystem == "debugfs" && mount.mount_point == "/sys/kernel/debug"
        }));
    }

    #[test]
    fn bounds_are_nonzero_and_restrict_large_pseudo_file_reads() {
        assert!(MAX_TEXT_BYTES > 0);
        assert!(MAX_HASHED_FILE_BYTES >= MAX_TEXT_BYTES);
    }

    #[test]
    fn digest_encoding_is_lowercase_and_domain_tagged() {
        let digest = sha256_tagged(b"basalt");
        assert!(digest.starts_with("sha256:"));
        assert_eq!(digest.len(), 71);
        assert!(digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }
}
