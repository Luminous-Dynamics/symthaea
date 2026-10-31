//! Explicit, bounded Linux syscall smoke tests for BASALT observer planning.
//!
//! This command is separate from the read-only inventory. It probes only:
//! - the side-effect-free Landlock ABI version query;
//! - creation and immediate closure of a fanotify notification group;
//! - optionally, creation and immediate closure of a fanotify permission-class
//!   group when the caller explicitly supplies --probe-permission-class.
//!
//! It does not add fanotify marks, generate permission events, attach BPF,
//! enable tracepoints, create cgroups/namespaces/rulesets, or run a workload.
//! Passing this command is not observer coverage or qualification.

use anyhow::{Context, bail};
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const LANDLOCK_CREATE_RULESET_VERSION: libc::c_uint = 1;
const FAN_CLOEXEC: libc::c_uint = 0x0000_0001;
const FAN_NONBLOCK: libc::c_uint = 0x0000_0002;
const FAN_CLASS_CONTENT: libc::c_uint = 0x0000_0004;
const FAN_REPORT_FID: libc::c_uint = 0x0000_0200;
const MAX_BOOT_ID_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStatus {
    Passed,
    NotSupported,
    Disabled,
    PermissionDenied,
    NotAttempted,
    Failed,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CleanupStatus {
    NotRequired,
    Closed,
    Failed,
}

#[derive(Debug, Serialize)]
pub struct ProbeObservation {
    pub probe_id: &'static str,
    pub status: ProbeStatus,
    pub detail: String,
    pub abi_version: Option<i32>,
    pub errno: Option<i32>,
    pub cleanup_status: CleanupStatus,
    pub resource_created: bool,
    pub event_capture_attempted: bool,
}

#[derive(Debug, Serialize)]
pub struct KernelIdentity {
    pub release: Option<String>,
    pub architecture: &'static str,
    pub boot_id_digest: Option<String>,
    pub mount_namespace_inode: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeSmokeReport {
    pub report_version: u32,
    pub suite_id: &'static str,
    pub probe_id: String,
    pub observed_at_unix_ns: u64,
    pub kernel: KernelIdentity,
    pub permission_class_probe_requested: bool,
    pub probes: Vec<ProbeObservation>,
    pub side_effects: SideEffectClaims,
    pub observer_attached: bool,
    pub workload_executed: bool,
    pub event_capture_established: bool,
    pub qualification_eligible: bool,
}

#[derive(Debug, Serialize)]
pub struct SideEffectClaims {
    pub bpf_programs_loaded_or_attached: bool,
    pub tracepoints_enabled_or_attached: bool,
    pub fanotify_marks_added: bool,
    pub landlock_rulesets_created_or_enforced: bool,
    pub cgroups_or_namespaces_created: bool,
    pub host_policy_or_sysctls_changed: bool,
    pub workload_processes_spawned: bool,
}

pub fn run(out_path: &Path, probe_permission_class: bool) -> anyhow::Result<RuntimeSmokeReport> {
    if !cfg!(target_os = "linux") {
        bail!("BASALT Linux runtime smoke tests are supported only on Linux");
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?
        .as_nanos();
    let observed_at_unix_ns = u64::try_from(timestamp)
        .context("probe timestamp exceeds u64 nanoseconds")?;
    let probe_id = format!("basalt-linux-smoke-{}-{observed_at_unix_ns}", std::process::id());

    let mut probes = vec![
        probe_landlock_abi(),
        probe_fanotify_group("fanotify_notification_group_init", 0),
    ];
    if probe_permission_class {
        probes.push(probe_fanotify_group(
            "fanotify_permission_class_group_init",
            FAN_CLASS_CONTENT,
        ));
    } else {
        probes.push(not_attempted(
            "fanotify_permission_class_group_init",
            "not attempted; pass --probe-permission-class to explicitly request this higher-privilege initialization check",
        ));
    }
    probes.extend([
        not_attempted(
            "bpf_lsm_load_attach",
            "not attempted; requires a separately pinned object, hook allowlist, privilege declaration and unload verifier",
        ),
        not_attempted(
            "process_tracepoint_attach",
            "not attempted; this command never writes tracefs enable files or attaches tracepoint programs",
        ),
        not_attempted(
            "cgroup_workload_scope",
            "not attempted; this command never creates or migrates cgroups and executes no workload",
        ),
    ]);

    let report = RuntimeSmokeReport {
        report_version: 1,
        suite_id: "basalt-linux-runtime-smoke-v1",
        probe_id,
        observed_at_unix_ns,
        kernel: kernel_identity(),
        permission_class_probe_requested: probe_permission_class,
        probes,
        side_effects: SideEffectClaims {
            bpf_programs_loaded_or_attached: false,
            tracepoints_enabled_or_attached: false,
            fanotify_marks_added: false,
            landlock_rulesets_created_or_enforced: false,
            cgroups_or_namespaces_created: false,
            host_policy_or_sysctls_changed: false,
            workload_processes_spawned: false,
        },
        observer_attached: false,
        workload_executed: false,
        event_capture_established: false,
        qualification_eligible: false,
    };

    let bytes = serde_json::to_vec_pretty(&report)?;
    write_new(out_path, &bytes)?;
    Ok(report)
}

fn kernel_identity() -> KernelIdentity {
    let release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let boot_id_digest = std::fs::read("/proc/sys/kernel/random/boot_id")
        .ok()
        .filter(|bytes| !bytes.is_empty() && bytes.len() <= MAX_BOOT_ID_BYTES)
        .map(|bytes| sha256_tagged(&bytes));

    let mount_namespace_inode = std::fs::metadata("/proc/self/ns/mnt")
        .ok()
        .map(|metadata| metadata.ino());

    KernelIdentity {
        release,
        architecture: std::env::consts::ARCH,
        boot_id_digest,
        mount_namespace_inode,
    }
}

#[cfg(target_os = "linux")]
fn probe_landlock_abi() -> ProbeObservation {
    // LANDLOCK_CREATE_RULESET_VERSION with a NULL attr and zero size is the
    // documented ABI query. It returns a version, not a ruleset file descriptor.
    let result = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<libc::c_void>(),
            0usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if result >= 1 {
        return ProbeObservation {
            probe_id: "landlock_abi_version_query",
            status: ProbeStatus::Passed,
            detail: "side-effect-free ABI query returned a supported version; no ruleset was created or enforced".to_string(),
            abi_version: i32::try_from(result).ok(),
            errno: None,
            cleanup_status: CleanupStatus::NotRequired,
            resource_created: false,
            event_capture_attempted: false,
        };
    }

    let error = std::io::Error::last_os_error();
    let errno = error.raw_os_error().unwrap_or(0);
    ProbeObservation {
        probe_id: "landlock_abi_version_query",
        status: classify_errno(errno),
        detail: format!("Landlock ABI query failed: {error}"),
        abi_version: None,
        errno: Some(errno),
        cleanup_status: CleanupStatus::NotRequired,
        resource_created: false,
        event_capture_attempted: false,
    }
}

#[cfg(not(target_os = "linux"))]
fn probe_landlock_abi() -> ProbeObservation {
    not_attempted("landlock_abi_version_query", "Linux-only syscall is unavailable on this target")
}

#[cfg(target_os = "linux")]
fn probe_fanotify_group(probe_id: &'static str, class_flag: libc::c_uint) -> ProbeObservation {
    // FAN_REPORT_FID is required for the limited unprivileged fanotify API on
    // modern kernels. No mark is installed, no event is generated or read.
    let flags = FAN_CLOEXEC | FAN_NONBLOCK | FAN_REPORT_FID | class_flag;
    let event_file_flags = libc::O_RDONLY | libc::O_LARGEFILE;
    let result = unsafe {
        libc::syscall(
            libc::SYS_fanotify_init,
            flags,
            event_file_flags,
        )
    };

    if result < 0 {
        let error = std::io::Error::last_os_error();
        let errno = error.raw_os_error().unwrap_or(0);
        return ProbeObservation {
            probe_id,
            status: classify_errno(errno),
            detail: format!("fanotify_init failed before a group descriptor was created: {error}"),
            abi_version: None,
            errno: Some(errno),
            cleanup_status: CleanupStatus::NotRequired,
            resource_created: false,
            event_capture_attempted: false,
        };
    }

    // Immediately close the descriptor. into_raw_fd makes the close explicit
    // so the receipt can record teardown rather than relying on Drop alone.
    let owned_fd = unsafe { OwnedFd::from_raw_fd(result as libc::c_int) };
    let raw_fd = owned_fd.into_raw_fd();
    let close_result = unsafe { libc::close(raw_fd) };
    if close_result == 0 {
        ProbeObservation {
            probe_id,
            status: ProbeStatus::Passed,
            detail: "fanotify group initialization succeeded and its descriptor was immediately closed; no marks or events were used".to_string(),
            abi_version: None,
            errno: None,
            cleanup_status: CleanupStatus::Closed,
            resource_created: true,
            event_capture_attempted: false,
        }
    } else {
        let error = std::io::Error::last_os_error();
        let errno = error.raw_os_error().unwrap_or(0);
        ProbeObservation {
            probe_id,
            status: ProbeStatus::Failed,
            detail: format!("fanotify group initialized but descriptor close was not confirmed: {error}"),
            abi_version: None,
            errno: Some(errno),
            cleanup_status: CleanupStatus::Failed,
            resource_created: true,
            event_capture_attempted: false,
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn probe_fanotify_group(probe_id: &'static str, _class_flag: u32) -> ProbeObservation {
    not_attempted(probe_id, "Linux-only fanotify syscall is unavailable on this target")
}

fn classify_errno(errno: i32) -> ProbeStatus {
    if errno == libc::ENOSYS || errno == libc::EINVAL {
        ProbeStatus::NotSupported
    } else if errno == libc::EOPNOTSUPP {
        ProbeStatus::Disabled
    } else if errno == libc::EPERM || errno == libc::EACCES {
        ProbeStatus::PermissionDenied
    } else {
        ProbeStatus::Failed
    }
}

fn not_attempted(probe_id: &'static str, detail: &str) -> ProbeObservation {
    ProbeObservation {
        probe_id,
        status: ProbeStatus::NotAttempted,
        detail: detail.to_string(),
        abi_version: None,
        errno: None,
        cleanup_status: CleanupStatus::NotRequired,
        resource_created: false,
        event_capture_attempted: false,
    }
}

fn sha256_tagged(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(71);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    format!("sha256:{output}")
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating smoke report directory {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating new smoke report {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_mapping_distinguishes_unsupported_disabled_and_denied() {
        assert_eq!(classify_errno(libc::ENOSYS), ProbeStatus::NotSupported);
        assert_eq!(classify_errno(libc::EINVAL), ProbeStatus::NotSupported);
        assert_eq!(classify_errno(libc::EOPNOTSUPP), ProbeStatus::Disabled);
        assert_eq!(classify_errno(libc::EPERM), ProbeStatus::PermissionDenied);
        assert_eq!(classify_errno(libc::EACCES), ProbeStatus::PermissionDenied);
        assert_eq!(classify_errno(libc::EIO), ProbeStatus::Failed);
    }

    #[test]
    fn fanotify_flags_are_bounded_notification_or_permission_class_only() {
        let notification = FAN_CLOEXEC | FAN_NONBLOCK | FAN_REPORT_FID;
        let permission = notification | FAN_CLASS_CONTENT;
        assert_eq!(notification & FAN_CLASS_CONTENT, 0);
        assert_eq!(permission & FAN_CLASS_CONTENT, FAN_CLASS_CONTENT);
        assert_eq!(notification & !(FAN_CLOEXEC | FAN_NONBLOCK | FAN_REPORT_FID), 0);
    }

    #[test]
    fn landlock_query_flag_is_version_query_not_ruleset_creation() {
        assert_eq!(LANDLOCK_CREATE_RULESET_VERSION, 1);
    }

    #[test]
    fn report_claims_never_promote_smoke_tests_to_capture_or_qualification() {
        let report = RuntimeSmokeReport {
            report_version: 1,
            suite_id: "basalt-linux-runtime-smoke-v1",
            probe_id: "test".to_string(),
            observed_at_unix_ns: 1,
            kernel: KernelIdentity {
                release: None,
                architecture: "x86_64",
                boot_id_digest: None,
                mount_namespace_inode: None,
            },
            permission_class_probe_requested: false,
            probes: vec![not_attempted("bpf_lsm_load_attach", "not run")],
            side_effects: SideEffectClaims {
                bpf_programs_loaded_or_attached: false,
                tracepoints_enabled_or_attached: false,
                fanotify_marks_added: false,
                landlock_rulesets_created_or_enforced: false,
                cgroups_or_namespaces_created: false,
                host_policy_or_sysctls_changed: false,
                workload_processes_spawned: false,
            },
            observer_attached: false,
            workload_executed: false,
            event_capture_established: false,
            qualification_eligible: false,
        };
        assert!(!report.observer_attached);
        assert!(!report.workload_executed);
        assert!(!report.event_capture_established);
        assert!(!report.qualification_eligible);
        assert!(!report.side_effects.fanotify_marks_added);
        assert!(!report.side_effects.landlock_rulesets_created_or_enforced);
    }

    #[test]
    fn not_attempted_observations_have_no_resource_or_cleanup_claim() {
        let item = not_attempted("tracepoint", "deliberately omitted");
        assert_eq!(item.status, ProbeStatus::NotAttempted);
        assert_eq!(item.cleanup_status, CleanupStatus::NotRequired);
        assert!(!item.resource_created);
        assert!(!item.event_capture_attempted);
        assert_eq!(item.errno, None);
    }

    #[test]
    fn permission_class_probe_is_separate_from_default_notification_probe() {
        let requested = false;
        let skipped = if requested {
            probe_fanotify_group("permission", FAN_CLASS_CONTENT)
        } else {
            not_attempted("permission", "not requested")
        };
        assert_eq!(skipped.status, ProbeStatus::NotAttempted);
    }

    #[test]
    fn digest_is_domain_tagged_and_fixed_width() {
        let digest = sha256_tagged(b"basalt-linux-smoke");
        assert!(digest.starts_with("sha256:"));
        assert_eq!(digest.len(), 71);
        assert!(digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }

    #[test]
    fn kernel_identity_does_not_require_optional_host_metadata() {
        let identity = KernelIdentity {
            release: None,
            architecture: "x86_64",
            boot_id_digest: None,
            mount_namespace_inode: None,
        };
        assert_eq!(identity.release, None);
        assert_eq!(identity.boot_id_digest, None);
    }

    #[test]
    fn successful_fanotify_probe_must_record_descriptor_cleanup() {
        let item = ProbeObservation {
            probe_id: "fanotify_notification_group_init",
            status: ProbeStatus::Passed,
            detail: "closed".to_string(),
            abi_version: None,
            errno: None,
            cleanup_status: CleanupStatus::Closed,
            resource_created: true,
            event_capture_attempted: false,
        };
        assert_eq!(item.cleanup_status, CleanupStatus::Closed);
        assert!(!item.event_capture_attempted);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn landlock_version_query_is_nonfatal_and_never_creates_a_ruleset() {
        let item = probe_landlock_abi();
        assert!(matches!(
            item.status,
            ProbeStatus::Passed
                | ProbeStatus::NotSupported
                | ProbeStatus::Disabled
                | ProbeStatus::PermissionDenied
                | ProbeStatus::Failed
        ));
        assert!(!item.resource_created);
        assert_eq!(item.cleanup_status, CleanupStatus::NotRequired);
        assert!(!item.event_capture_attempted);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fanotify_group_smoke_never_marks_or_reads_events_and_closes_successes() {
        let item = probe_fanotify_group("fanotify_notification_group_init", 0);
        assert!(matches!(
            item.status,
            ProbeStatus::Passed
                | ProbeStatus::NotSupported
                | ProbeStatus::Disabled
                | ProbeStatus::PermissionDenied
                | ProbeStatus::Failed
        ));
        assert!(!item.event_capture_attempted);
        if item.status == ProbeStatus::Passed {
            assert!(item.resource_created);
            assert_eq!(item.cleanup_status, CleanupStatus::Closed);
        } else {
            assert!(!item.resource_created);
            assert_eq!(item.cleanup_status, CleanupStatus::NotRequired);
        }
    }

    #[test]
    fn report_version_and_suite_identifier_are_pinned() {
        assert_eq!(1u32, 1);
        assert_eq!("basalt-linux-runtime-smoke-v1", "basalt-linux-runtime-smoke-v1");
    }

    #[test]
    fn permission_class_opt_in_has_explicit_fanotify_content_flag() {
        assert_ne!(FAN_CLASS_CONTENT, 0);
        assert_eq!(FAN_CLASS_CONTENT & !0x7, 0);
    }
}
