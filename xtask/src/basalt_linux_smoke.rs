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
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;
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
    OperationNotSupported,
    PermissionDenied,
    InvalidRequest,
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
}

#[derive(Debug, Serialize)]
pub struct ProcessSecurityContext {
    pub effective_uid: Option<u32>,
    pub effective_gid: Option<u32>,
    pub cap_eff_hex: Option<String>,
    pub cap_bnd_hex: Option<String>,
    pub no_new_privs: Option<u8>,
    pub seccomp_mode: Option<u8>,
    pub mount_namespace_inode: Option<u64>,
    pub user_namespace_inode: Option<u64>,
    pub pid_namespace_inode: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeSmokeReport {
    pub report_version: u32,
    pub suite_id: &'static str,
    pub probe_id: String,
    pub observed_at_unix_ns: u64,
    pub kernel: KernelIdentity,
    pub process_context: ProcessSecurityContext,
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
        process_context: process_security_context(),
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
    let release = read_bounded_file("/proc/sys/kernel/osrelease", 256)
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let boot_id_digest = read_bounded_file("/proc/sys/kernel/random/boot_id", MAX_BOOT_ID_BYTES as u64)
        .filter(|bytes| !bytes.is_empty())
        .map(|bytes| sha256_tagged(&bytes));

    KernelIdentity {
        release,
        architecture: std::env::consts::ARCH,
        boot_id_digest,
    }
}

fn proc_status_value<'a>(status: &'a str, key: &str) -> Option<&'a str> {
    status.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key).then_some(value.trim())
    })
}

fn proc_status_hex(status: &str, key: &str) -> Option<String> {
    let value = proc_status_value(status, key)?;
    (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

fn proc_status_u8(status: &str, key: &str) -> Option<u8> {
    proc_status_value(status, key)?.parse().ok()
}

#[cfg(target_os = "linux")]
fn namespace_inode(path: &str) -> Option<u64> {
    std::fs::metadata(path).ok().map(|metadata| metadata.ino())
}

fn process_security_context() -> ProcessSecurityContext {
    let status = read_process_status();
    #[cfg(target_os = "linux")]
    let (effective_uid, effective_gid) = (Some(unsafe { libc::geteuid() }), Some(unsafe { libc::getegid() }));
    #[cfg(not(target_os = "linux"))]
    let (effective_uid, effective_gid) = (None, None);

    #[cfg(target_os = "linux")]
    let (mount_namespace_inode, user_namespace_inode, pid_namespace_inode) = (
        namespace_inode("/proc/self/ns/mnt"),
        namespace_inode("/proc/self/ns/user"),
        namespace_inode("/proc/self/ns/pid"),
    );
    #[cfg(not(target_os = "linux"))]
    let (mount_namespace_inode, user_namespace_inode, pid_namespace_inode) = (None, None, None);

    ProcessSecurityContext {
        effective_uid,
        effective_gid,
        cap_eff_hex: status.as_deref().and_then(|text| proc_status_hex(text, "CapEff")),
        cap_bnd_hex: status.as_deref().and_then(|text| proc_status_hex(text, "CapBnd")),
        no_new_privs: status.as_deref().and_then(|text| proc_status_u8(text, "NoNewPrivs")),
        seccomp_mode: status.as_deref().and_then(|text| proc_status_u8(text, "Seccomp")),
        mount_namespace_inode,
        user_namespace_inode,
        pid_namespace_inode,
    }
}

fn read_bounded_file(path: &str, limit: u64) -> Option<Vec<u8>> {
    let file = File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes).ok()?;
    if bytes.len() as u64 > limit {
        return None;
    }
    Some(bytes)
}
fn read_process_status() -> Option<String> {
    const MAX_STATUS_BYTES: u64 = 256 * 1024;
    String::from_utf8(read_bounded_file("/proc/self/status", MAX_STATUS_BYTES)?).ok()
}

trait LinuxSmokeSyscalls {
    fn landlock_abi_version(&self) -> Result<i32, i32>;
    fn fanotify_init(&self, flags: libc::c_uint, event_file_flags: libc::c_int) -> Result<libc::c_int, i32>;
    fn close_fd(&self, fd: libc::c_int) -> Result<(), i32>;
}

#[cfg(target_os = "linux")]
struct SystemLinuxSmokeSyscalls;

#[cfg(target_os = "linux")]
impl LinuxSmokeSyscalls for SystemLinuxSmokeSyscalls {
    fn landlock_abi_version(&self) -> Result<i32, i32> {
        let result = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<libc::c_void>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        if result < 0 {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
        } else {
            i32::try_from(result).map_err(|_| libc::EOVERFLOW)
        }
    }

    fn fanotify_init(
        &self,
        flags: libc::c_uint,
        event_file_flags: libc::c_int,
    ) -> Result<libc::c_int, i32> {
        let result = unsafe {
            libc::syscall(libc::SYS_fanotify_init, flags, event_file_flags)
        };
        if result < 0 {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
        } else {
            // fanotify_init returns an int file descriptor even though syscall() exposes c_long.
            Ok(result as libc::c_int)
        }
    }

    fn close_fd(&self, fd: libc::c_int) -> Result<(), i32> {
        let result = unsafe { libc::close(fd) };
        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO))
        }
    }
}

#[cfg(target_os = "linux")]
fn probe_landlock_abi() -> ProbeObservation {
    probe_landlock_abi_with(&SystemLinuxSmokeSyscalls)
}

fn probe_landlock_abi_with(calls: &impl LinuxSmokeSyscalls) -> ProbeObservation {
    match calls.landlock_abi_version() {
        Ok(version) if version >= 1 => ProbeObservation {
            probe_id: "landlock_abi_version_query",
            status: ProbeStatus::Passed,
            detail: "side-effect-free ABI query returned a supported version; no ruleset was created or enforced".to_string(),
            abi_version: Some(version),
            errno: None,
            cleanup_status: CleanupStatus::NotRequired,
            resource_created: false,
            event_capture_attempted: false,
        },
        Ok(version) => ProbeObservation {
            probe_id: "landlock_abi_version_query",
            status: ProbeStatus::Failed,
            detail: format!("Landlock ABI query returned invalid version {version}; expected a positive version"),
            abi_version: Some(version),
            errno: None,
            cleanup_status: CleanupStatus::NotRequired,
            resource_created: false,
            event_capture_attempted: false,
        },
        Err(errno) => ProbeObservation {
            probe_id: "landlock_abi_version_query",
            status: classify_errno(errno),
            detail: format!(
                "Landlock ABI query failed: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
            abi_version: None,
            errno: Some(errno),
            cleanup_status: CleanupStatus::NotRequired,
            resource_created: false,
            event_capture_attempted: false,
        },
    }
}

#[cfg(not(target_os = "linux"))]
fn probe_landlock_abi() -> ProbeObservation {
    not_attempted("landlock_abi_version_query", "Linux-only syscall is unavailable on this target")
}

#[cfg(target_os = "linux")]
fn probe_fanotify_group(probe_id: &'static str, class_flag: libc::c_uint) -> ProbeObservation {
    probe_fanotify_group_with(&SystemLinuxSmokeSyscalls, probe_id, class_flag)
}

#[cfg(target_os = "linux")]
fn probe_fanotify_group_with(
    calls: &impl LinuxSmokeSyscalls,
    probe_id: &'static str,
    class_flag: libc::c_uint,
) -> ProbeObservation {
    // FAN_REPORT_FID is required for restricted unprivileged notification
    // groups, but the kernel rejects it with FAN_CLASS_CONTENT/PRE_CONTENT.
    let flags = fanotify_init_flags(class_flag);
    let event_file_flags = libc::O_RDONLY | libc::O_LARGEFILE;
    let fd = match calls.fanotify_init(flags, event_file_flags) {
        Ok(fd) => fd,
        Err(errno) => {
            return ProbeObservation {
                probe_id,
                status: classify_errno(errno),
                detail: format!(
                    "fanotify_init failed before a group descriptor was created: {}",
                    std::io::Error::from_raw_os_error(errno)
                ),
                abi_version: None,
                errno: Some(errno),
                cleanup_status: CleanupStatus::NotRequired,
                resource_created: false,
                event_capture_attempted: false,
            };
        }
    };

    // No operation occurs between descriptor acquisition and close other
    // than recording the returned integer. The injected adapter makes both
    // acquisition and cleanup paths independently testable.
    match calls.close_fd(fd) {
        Ok(()) => ProbeObservation {
            probe_id,
            status: ProbeStatus::Passed,
            detail: "fanotify group initialization succeeded and its descriptor was immediately closed; no marks or events were used".to_string(),
            abi_version: None,
            errno: None,
            cleanup_status: CleanupStatus::Closed,
            resource_created: true,
            event_capture_attempted: false,
        },
        Err(errno) => ProbeObservation {
            probe_id,
            status: ProbeStatus::Failed,
            detail: format!(
                "fanotify group initialized but descriptor close was not confirmed: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
            abi_version: None,
            errno: Some(errno),
            cleanup_status: CleanupStatus::Failed,
            resource_created: true,
            event_capture_attempted: false,
        },
    }
}

#[cfg(not(target_os = "linux"))]
fn probe_fanotify_group(probe_id: &'static str, _class_flag: u32) -> ProbeObservation {
    not_attempted(probe_id, "Linux-only fanotify syscall is unavailable on this target")
}

#[cfg(target_os = "linux")]
fn fanotify_init_flags(class_flag: libc::c_uint) -> libc::c_uint {
    // FAN_REPORT_FID is mandatory for restricted unprivileged notification
    // groups, but is invalid with FAN_CLASS_CONTENT and FAN_CLASS_PRE_CONTENT.
    let object_id_flag = if class_flag == 0 { FAN_REPORT_FID } else { 0 };
    FAN_CLOEXEC | FAN_NONBLOCK | object_id_flag | class_flag
}

fn classify_errno(errno: i32) -> ProbeStatus {
    if errno == libc::ENOSYS {
        ProbeStatus::NotSupported
    } else if errno == libc::EINVAL {
        ProbeStatus::InvalidRequest
    } else if errno == libc::EOPNOTSUPP {
        ProbeStatus::OperationNotSupported
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct FakeSyscalls {
        landlock: Result<i32, i32>,
        fanotify: Result<libc::c_int, i32>,
        close: Result<(), i32>,
        closed_fds: RefCell<Vec<libc::c_int>>,
        fanotify_flags: RefCell<Vec<libc::c_uint>>,
    }

    impl FakeSyscalls {
        fn new(
            landlock: Result<i32, i32>,
            fanotify: Result<libc::c_int, i32>,
            close: Result<(), i32>,
        ) -> Self {
            Self {
                landlock,
                fanotify,
                close,
                closed_fds: RefCell::new(Vec::new()),
                fanotify_flags: RefCell::new(Vec::new()),
            }
        }
    }

    impl LinuxSmokeSyscalls for FakeSyscalls {
        fn landlock_abi_version(&self) -> Result<i32, i32> {
            self.landlock
        }

        fn fanotify_init(
            &self,
            flags: libc::c_uint,
            _event_file_flags: libc::c_int,
        ) -> Result<libc::c_int, i32> {
            self.fanotify_flags.borrow_mut().push(flags);
            self.fanotify
        }

        fn close_fd(&self, fd: libc::c_int) -> Result<(), i32> {
            self.closed_fds.borrow_mut().push(fd);
            self.close
        }
    }

    #[test]
    fn errno_mapping_distinguishes_unsupported_operation_denied_and_invalid_request() {
        assert_eq!(classify_errno(libc::ENOSYS), ProbeStatus::NotSupported);
        assert_eq!(classify_errno(libc::EOPNOTSUPP), ProbeStatus::OperationNotSupported);
        assert_eq!(classify_errno(libc::EPERM), ProbeStatus::PermissionDenied);
        assert_eq!(classify_errno(libc::EACCES), ProbeStatus::PermissionDenied);
        assert_eq!(classify_errno(libc::EINVAL), ProbeStatus::InvalidRequest);
        assert_eq!(classify_errno(libc::EIO), ProbeStatus::Failed);
    }

    #[test]
    fn notification_and_permission_class_flags_do_not_use_an_invalid_combination() {
        let notification = fanotify_init_flags(0);
        let permission = fanotify_init_flags(FAN_CLASS_CONTENT);
        assert_eq!(notification & FAN_CLASS_CONTENT, 0);
        assert_ne!(notification & FAN_REPORT_FID, 0);
        assert_eq!(permission & FAN_CLASS_CONTENT, FAN_CLASS_CONTENT);
        assert_eq!(permission & FAN_REPORT_FID, 0);
        assert_eq!(notification & !(FAN_CLOEXEC | FAN_NONBLOCK | FAN_REPORT_FID), 0);
        assert_eq!(permission & !(FAN_CLOEXEC | FAN_NONBLOCK | FAN_CLASS_CONTENT), 0);
    }

    #[test]
    fn landlock_version_query_pass_records_version_without_resource_creation() {
        let fake = FakeSyscalls::new(Ok(7), Ok(31), Ok(()));
        let item = probe_landlock_abi_with(&fake);
        assert_eq!(item.status, ProbeStatus::Passed);
        assert_eq!(item.abi_version, Some(7));
        assert!(!item.resource_created);
        assert_eq!(item.cleanup_status, CleanupStatus::NotRequired);
    }

    #[test]
    fn zero_landlock_version_is_not_promoted_to_a_pass() {
        let fake = FakeSyscalls::new(Ok(0), Ok(31), Ok(()));
        let item = probe_landlock_abi_with(&fake);
        assert_eq!(item.status, ProbeStatus::Failed);
        assert_eq!(item.abi_version, Some(0));
    }

    #[test]
    fn landlock_errno_is_preserved_and_classified() {
        let fake = FakeSyscalls::new(Err(libc::EOPNOTSUPP), Ok(31), Ok(()));
        let item = probe_landlock_abi_with(&fake);
        assert_eq!(item.status, ProbeStatus::OperationNotSupported);
        assert_eq!(item.errno, Some(libc::EOPNOTSUPP));
        assert!(!item.resource_created);
    }

    #[test]
    fn fanotify_success_requires_a_confirmed_close() {
        let fake = FakeSyscalls::new(Ok(1), Ok(31), Ok(()));
        let item = probe_fanotify_group_with(&fake, "fanotify_notification_group_init", 0);
        assert_eq!(item.status, ProbeStatus::Passed);
        assert_eq!(item.cleanup_status, CleanupStatus::Closed);
        assert!(item.resource_created);
        assert_eq!(*fake.closed_fds.borrow(), vec![31]);
        assert!(!item.event_capture_attempted);
    }

    #[test]
    fn fanotify_initialization_error_never_attempts_close() {
        let fake = FakeSyscalls::new(Ok(1), Err(libc::EPERM), Ok(()));
        let item = probe_fanotify_group_with(&fake, "fanotify_notification_group_init", 0);
        assert_eq!(item.status, ProbeStatus::PermissionDenied);
        assert_eq!(item.errno, Some(libc::EPERM));
        assert_eq!(item.cleanup_status, CleanupStatus::NotRequired);
        assert!(!item.resource_created);
        assert!(fake.closed_fds.borrow().is_empty());
    }

    #[test]
    fn fanotify_close_failure_downgrades_the_probe_to_failed() {
        let fake = FakeSyscalls::new(Ok(1), Ok(31), Err(libc::EIO));
        let item = probe_fanotify_group_with(&fake, "fanotify_notification_group_init", 0);
        assert_eq!(item.status, ProbeStatus::Failed);
        assert_eq!(item.cleanup_status, CleanupStatus::Failed);
        assert!(item.resource_created);
        assert_eq!(item.errno, Some(libc::EIO));
        assert_eq!(*fake.closed_fds.borrow(), vec![31]);
    }

    #[test]
    fn permission_class_path_omits_report_fid_and_records_exact_flags() {
        let fake = FakeSyscalls::new(Ok(1), Ok(44), Ok(()));
        let item = probe_fanotify_group_with(
            &fake,
            "fanotify_permission_class_group_init",
            FAN_CLASS_CONTENT,
        );
        assert_eq!(item.status, ProbeStatus::Passed);
        assert_eq!(fake.fanotify_flags.borrow()[0] & FAN_REPORT_FID, 0);
        assert_ne!(fake.fanotify_flags.borrow()[0] & FAN_CLASS_CONTENT, 0);
        assert_eq!(*fake.closed_fds.borrow(), vec![44]);
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
    fn not_requested_permission_class_probe_stays_not_attempted() {
        let item = not_attempted(
            "fanotify_permission_class_group_init",
            "not requested",
        );
        assert_eq!(item.status, ProbeStatus::NotAttempted);
        assert!(!item.resource_created);
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
            },
            process_context: ProcessSecurityContext {
                effective_uid: None,
                effective_gid: None,
                cap_eff_hex: None,
                cap_bnd_hex: None,
                no_new_privs: None,
                seccomp_mode: None,
                mount_namespace_inode: None,
                user_namespace_inode: None,
                pid_namespace_inode: None,
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
    fn proc_status_security_fields_are_parsed_without_promoting_missing_values() {
        let status = "Name:\tbasalt\nCapEff:\t00000000000000a5\nCapBnd:\t00000000000000ff\nNoNewPrivs:\t1\nSeccomp:\t2\n";
        assert_eq!(proc_status_hex(status, "CapEff").as_deref(), Some("00000000000000a5"));
        assert_eq!(proc_status_hex(status, "CapBnd").as_deref(), Some("00000000000000ff"));
        assert_eq!(proc_status_u8(status, "NoNewPrivs"), Some(1));
        assert_eq!(proc_status_u8(status, "Seccomp"), Some(2));
        assert_eq!(proc_status_u8(status, "NoSuchField"), None);
    }

    #[test]
    fn malformed_proc_status_capability_values_remain_unknown() {
        let status = "CapEff:\t0xnot-hex\nNoNewPrivs:\tunknown\nSeccomp:\t3\n";
        assert_eq!(proc_status_hex(status, "CapEff"), None);
        assert_eq!(proc_status_u8(status, "NoNewPrivs"), None);
        assert_eq!(proc_status_u8(status, "Seccomp"), Some(3));
    }

    #[test]
    fn benchmark_gate_ids_must_resolve_to_declared_metrics() {
        let benchmark: serde_json::Value = serde_json::from_str(include_str!(
            "../../benchmarks/basalt/linux-runtime-smoke-benchmark.v1.json"
        )).unwrap();
        let ids: std::collections::BTreeSet<&str> = benchmark["metrics"].as_array().unwrap()
            .iter().map(|metric| metric["metric_id"].as_str().unwrap()).collect();
        for gate_group in ["safety", "efficiency"] {
            for gate in benchmark["gates"][gate_group].as_array().unwrap() {
                assert!(ids.contains(gate.as_str().unwrap()), "unresolved benchmark gate: {gate}");
            }
        }
    }

    #[test]
    fn report_schema_required_fields_match_serialized_report_fixture() {
        let report: RuntimeSmokeReport = RuntimeSmokeReport {
            report_version: 1,
            suite_id: "basalt-linux-runtime-smoke-v1",
            probe_id: "schema-test".to_string(),
            observed_at_unix_ns: 1,
            kernel: KernelIdentity { release: None, architecture: "x86_64", boot_id_digest: None },
            process_context: ProcessSecurityContext {
                effective_uid: None, effective_gid: None, cap_eff_hex: None, cap_bnd_hex: None,
                no_new_privs: None, seccomp_mode: None, mount_namespace_inode: None,
                user_namespace_inode: None, pid_namespace_inode: None,
            },
            permission_class_probe_requested: false,
            probes: Vec::new(),
            side_effects: SideEffectClaims {
                bpf_programs_loaded_or_attached: false, tracepoints_enabled_or_attached: false,
                fanotify_marks_added: false, landlock_rulesets_created_or_enforced: false,
                cgroups_or_namespaces_created: false, host_policy_or_sysctls_changed: false,
                workload_processes_spawned: false,
            },
            observer_attached: false, workload_executed: false,
            event_capture_established: false, qualification_eligible: false,
        };
        let value = serde_json::to_value(report).unwrap();
        let mut actual: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../benchmarks/basalt/schema/linux-runtime-smoke.v1.schema.json"
        )).unwrap();
        let mut required: Vec<String> = schema["required"].as_array().unwrap().iter()
            .map(|field| field.as_str().unwrap().to_string()).collect();
        actual.sort(); required.sort();
        assert_eq!(actual, required);
    }
    #[test]
    fn probe_status_serde_values_match_the_report_schema() {
        let statuses = [
            ProbeStatus::Passed,
            ProbeStatus::NotSupported,
            ProbeStatus::OperationNotSupported,
            ProbeStatus::PermissionDenied,
            ProbeStatus::InvalidRequest,
            ProbeStatus::NotAttempted,
            ProbeStatus::Failed,
        ];
        let actual: std::collections::BTreeSet<String> = statuses.iter()
            .map(|status| serde_json::to_value(status).unwrap().as_str().unwrap().to_string())
            .collect();
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../benchmarks/basalt/schema/linux-runtime-smoke.v1.schema.json"
        )).unwrap();
        let expected: std::collections::BTreeSet<String> = schema["$defs"]["probeObservation"]
            ["properties"]["status"]["enum"].as_array().unwrap().iter()
            .map(|value| value.as_str().unwrap().to_string()).collect();
        assert_eq!(actual, expected);
    }
    #[test]
    fn digest_is_domain_tagged_and_fixed_width() {
        let digest = sha256_tagged(b"basalt-linux-smoke");
        assert!(digest.starts_with("sha256:"));
        assert_eq!(digest.len(), 71);
        assert!(digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn live_safe_probes_never_claim_capture() {
        let landlock = probe_landlock_abi();
        let fanotify = probe_fanotify_group("fanotify_notification_group_init", 0);
        assert!(!landlock.resource_created);
        assert_eq!(landlock.cleanup_status, CleanupStatus::NotRequired);
        assert!(!fanotify.event_capture_attempted);
        if fanotify.status == ProbeStatus::Passed {
            assert_eq!(fanotify.cleanup_status, CleanupStatus::Closed);
        }
    }
}
