//! Pure, fail-closed planning for a Linux candidate sandbox.
//!
//! This module only validates inputs and builds exact argv vectors. It does not
//! execute Bubblewrap/systemd, deliver the seccomp FD, probe namespaces, or read
//! cgroup files. A plan is not evidence that the host enforced it.

use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const MAX_CLOSURE_ENTRIES: usize = 512;
const MIN_MEMORY_MAX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MEMORY_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_STDIO_BYTES: u64 = 1024 * 1024;
const MAX_TASKS: u32 = 128;

const NIX_BASE32: &str = "0123456789abcdfghijklmnpqrsvwxyz";

/// Fixed descriptors expected by the trusted launcher and Bubblewrap plan.
pub const SECCOMP_FD: i32 = 3;
/// Pre-seccomp staging barrier. EOF is NOT a deny signal in Bubblewrap; the
/// post-seccomp gate descriptor is the actual admission boundary.
pub const PRE_SECCOMP_BLOCK_FD: i32 = 4;
/// Bubblewrap JSON-lines child PID/namespace and exit-status channel.
pub const JSON_STATUS_FD: i32 = 5;
/// One-shot trusted gate channel. The gate must reject EOF and malformed tokens.
pub const GATE_CONTROL_FD: i32 = 6;
/// Held by Bubblewrap's monitor for the sandbox lifetime; not inherited by gate.
pub const SANDBOX_SYNC_FD: i32 = 7;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SandboxPlanError {
    InvalidPath(&'static str),
    InvalidStorePath(String),
    CandidateNotInClosure,
    GateNotInClosure,
    EmptyClosure,
    TooManyClosureEntries,
    DuplicateClosureEntry(String),
    ZeroDigest(&'static str),
    InvalidLimit(&'static str),
    InvalidSeccompDescriptor,
    InvalidBubblewrapPath,
    InvalidGatePath,
}

impl fmt::Display for SandboxPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(field) => write!(f, "{field} is not a safe absolute path"),
            Self::InvalidStorePath(path) => write!(f, "invalid Nix store entry: {path}"),
            Self::CandidateNotInClosure => {
                write!(f, "candidate executable is not inside a pinned runtime closure entry")
            }
            Self::GateNotInClosure => {
                write!(f, "trusted gate executable is not inside a pinned runtime closure entry")
            }
            Self::EmptyClosure => write!(f, "runtime closure must not be empty"),
            Self::TooManyClosureEntries => write!(f, "runtime closure exceeds the entry limit"),
            Self::DuplicateClosureEntry(path) => {
                write!(f, "duplicate runtime closure entry: {path}")
            }
            Self::ZeroDigest(field) => write!(f, "{field} digest must be nonzero"),
            Self::InvalidLimit(field) => write!(f, "invalid sandbox resource limit: {field}"),
            Self::InvalidSeccompDescriptor => {
                write!(f, "seccomp descriptor must be pre-opened and normalized to fd 3")
            }
            Self::InvalidBubblewrapPath => {
                write!(f, "Bubblewrap executable must be an absolute pinned Nix store path")
            }
            Self::InvalidGatePath => {
                write!(f, "trusted gate executable must be an absolute pinned Nix store path")
            }
        }
    }
}
impl Error for SandboxPlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeClosureEntry {
    /// Exact top-level Nix store path, not /nix/store or one of its parents.
    pub store_path: String,
    /// Digest of the store object/tree computed before launch.
    pub artifact_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateSandboxSpec {
    pub candidate_executable: String,
    pub candidate_sha256: [u8; 32],
    /// Trusted, pinned gate helper. Bubblewrap must execute this helper first;
    /// it must not execute candidate code until it receives the exact release token.
    pub gate_executable: String,
    pub gate_sha256: [u8; 32],
    pub runtime_closure: Vec<RuntimeClosureEntry>,
    pub bubblewrap_executable: String,
    pub bubblewrap_sha256: [u8; 32],
    /// Already-open, pinned architecture-specific seccomp BPF program.
    /// The launcher must ensure this exact descriptor survives exec as fd 3.
    pub seccomp_fd: i32,
    pub seccomp_sha256: [u8; 32],
    pub memory_max_bytes: u64,
    pub cpu_quota_percent: u32,
    pub tasks_max: u32,
    pub wall_timeout_ms: u64,
    pub scratch_max_bytes: u64,
    pub stdout_max_bytes: u64,
    pub stderr_max_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateSandboxPlan {
    pub bubblewrap_arguments: Vec<String>,
    pub systemd_run_arguments: Vec<String>,
    pub candidate_executable: String,
    pub gate_executable: String,
    pub runtime_closure_sha256: [u8; 32],
    pub plan_sha256: [u8; 32],
}

fn safe_absolute_path(value: &str) -> bool {
    if !value.starts_with('/')
        || value.starts_with("//")
        || value.contains('\0')
        || value.ends_with('/')
    {
        return false;
    }
    let mut parts = value.split('/');
    let _root = parts.next();
    parts.all(|part| !part.is_empty() && part != "." && part != "..")
}

fn nix_store_entry_name(path: &str) -> Option<&str> {
    if !safe_absolute_path(path) {
        return None;
    }
    let remainder = path.strip_prefix("/nix/store/")?;
    if remainder.contains('/') {
        return None;
    }
    let (hash, name) = remainder.split_once('-')?;
    if hash.len() != 32
        || !hash.chars().all(|character| NIX_BASE32.contains(character))
        || name.is_empty()
    {
        return None;
    }
    Some(name)
}

fn candidate_store_root(candidate_executable: &str) -> Option<String> {
    if !safe_absolute_path(candidate_executable) {
        return None;
    }
    let remainder = candidate_executable.strip_prefix("/nix/store/")?;
    let (entry, suffix) = match remainder.split_once('/') {
        Some((entry, suffix)) => (entry, Some(suffix)),
        None => (remainder, None),
    };
    if entry.is_empty() || suffix.is_some_and(|value| value.is_empty()) {
        return None;
    }
    let root = format!("/nix/store/{entry}");
    nix_store_entry_name(&root).map(|_| root)
}

fn validate_spec(spec: &CandidateSandboxSpec) -> Result<Vec<RuntimeClosureEntry>, SandboxPlanError> {
    if spec.candidate_sha256 == [0; 32] {
        return Err(SandboxPlanError::ZeroDigest("candidate"));
    }
    if spec.gate_sha256 == [0; 32] {
        return Err(SandboxPlanError::ZeroDigest("trusted gate"));
    }
    if spec.bubblewrap_sha256 == [0; 32] {
        return Err(SandboxPlanError::ZeroDigest("Bubblewrap"));
    }
    if spec.seccomp_sha256 == [0; 32] {
        return Err(SandboxPlanError::ZeroDigest("seccomp"));
    }
    if spec.seccomp_fd != SECCOMP_FD {
        return Err(SandboxPlanError::InvalidSeccompDescriptor);
    }
    if candidate_store_root(&spec.candidate_executable).is_none() {
        return Err(SandboxPlanError::InvalidPath("candidate_executable"));
    }
    if candidate_store_root(&spec.bubblewrap_executable).is_none() {
        return Err(SandboxPlanError::InvalidBubblewrapPath);
    }
    if candidate_store_root(&spec.gate_executable).is_none() {
        return Err(SandboxPlanError::InvalidGatePath);
    }
    if spec.runtime_closure.is_empty() {
        return Err(SandboxPlanError::EmptyClosure);
    }
    if spec.runtime_closure.len() > MAX_CLOSURE_ENTRIES {
        return Err(SandboxPlanError::TooManyClosureEntries);
    }
    if !(MIN_MEMORY_MAX_BYTES..=MAX_MEMORY_MAX_BYTES).contains(&spec.memory_max_bytes) {
        return Err(SandboxPlanError::InvalidLimit("memory_max_bytes"));
    }
    if !(1..=1_600).contains(&spec.cpu_quota_percent) {
        return Err(SandboxPlanError::InvalidLimit("cpu_quota_percent"));
    }
    if !(1..=MAX_TASKS).contains(&spec.tasks_max) {
        return Err(SandboxPlanError::InvalidLimit("tasks_max"));
    }
    if !(10..=3_600_000).contains(&spec.wall_timeout_ms) {
        return Err(SandboxPlanError::InvalidLimit("wall_timeout_ms"));
    }
    if !(4_096..=MAX_SCRATCH_BYTES).contains(&spec.scratch_max_bytes) {
        return Err(SandboxPlanError::InvalidLimit("scratch_max_bytes"));
    }
    if spec.stdout_max_bytes == 0 || spec.stdout_max_bytes > MAX_STDIO_BYTES {
        return Err(SandboxPlanError::InvalidLimit("stdout_max_bytes"));
    }
    if spec.stderr_max_bytes == 0 || spec.stderr_max_bytes > MAX_STDIO_BYTES {
        return Err(SandboxPlanError::InvalidLimit("stderr_max_bytes"));
    }

    let candidate_root = candidate_store_root(&spec.candidate_executable)
        .ok_or(SandboxPlanError::InvalidPath("candidate_executable"))?;
    let gate_root = candidate_store_root(&spec.gate_executable)
        .ok_or(SandboxPlanError::InvalidGatePath)?;
    let mut closure = spec.runtime_closure.clone();
    closure.sort_by(|left, right| left.store_path.cmp(&right.store_path));
    let mut previous: Option<&str> = None;
    let mut contains_candidate = false;
    let mut contains_gate = false;
    for entry in &closure {
        if nix_store_entry_name(&entry.store_path).is_none() {
            return Err(SandboxPlanError::InvalidStorePath(entry.store_path.clone()));
        }
        if entry.artifact_sha256 == [0; 32] {
            return Err(SandboxPlanError::ZeroDigest("runtime closure entry"));
        }
        if previous == Some(entry.store_path.as_str()) {
            return Err(SandboxPlanError::DuplicateClosureEntry(entry.store_path.clone()));
        }
        previous = Some(entry.store_path.as_str());
        if entry.store_path == candidate_root {
            contains_candidate = true;
        }
        if entry.store_path == gate_root {
            contains_gate = true;
        }
    }
    if !contains_candidate {
        return Err(SandboxPlanError::CandidateNotInClosure);
    }
    if !contains_gate {
        return Err(SandboxPlanError::GateNotInClosure);
    }
    Ok(closure)
}

fn closure_digest(entries: &[RuntimeClosureEntry]) -> [u8; 32] {
    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.candidate-runtime-closure.v1");
    hasher.usize(entries.len());
    for entry in entries {
        hasher.bytes(entry.store_path.as_bytes());
        hasher.bytes(&entry.artifact_sha256);
    }
    hasher.finish()
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut value = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut value, "{byte:02x}");
    }
    value
}

fn build_bubblewrap_arguments(
    spec: &CandidateSandboxSpec,
    closure: &[RuntimeClosureEntry],
) -> Vec<String> {
    let mut args = vec![
        "--unshare-user".into(),
        "--unshare-pid".into(),
        "--unshare-net".into(),
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--unshare-cgroup".into(),
        "--die-with-parent".into(),
        "--new-session".into(),
        "--cap-drop".into(),
        "ALL".into(),
        "--clearenv".into(),
        "--dir".into(),
        "/nix".into(),
        "--dir".into(),
        "/nix/store".into(),
    ];
    for entry in closure {
        args.push("--ro-bind".into());
        args.push(entry.store_path.clone());
        args.push(entry.store_path.clone());
    }
    args.extend([
        "--remount-ro".into(),
        "/nix".into(),
        "--proc".into(),
        "/proc".into(),
        "--dev".into(),
        "/dev".into(),
        "--size".into(),
        spec.scratch_max_bytes.to_string(),
        "--tmpfs".into(),
        "/tmp".into(),
        "--dir".into(),
        "/home".into(),
        "--dir".into(),
        "/home/candidate".into(),
        "--chdir".into(),
        "/tmp".into(),
        "--setenv".into(),
        "HOME".into(),
        "/home/candidate".into(),
        "--setenv".into(),
        "TMPDIR".into(),
        "/tmp".into(),
        "--setenv".into(),
        "LANG".into(),
        "C".into(),
        "--setenv".into(),
        "LC_ALL".into(),
        "C".into(),
        // Bubblewrap 0.11.x: block-fd is only a staging barrier. Its current
        // implementation ignores the read result, so EOF permits setup to continue.
        // The trusted gate on fd 6 is the hard admission boundary and must deny EOF.
        "--block-fd".into(),
        PRE_SECCOMP_BLOCK_FD.to_string(),
        "--json-status-fd".into(),
        JSON_STATUS_FD.to_string(),
        "--sync-fd".into(),
        SANDBOX_SYNC_FD.to_string(),
        "--seccomp".into(),
        SECCOMP_FD.to_string(),
        "--".into(),
        spec.gate_executable.clone(),
        "--gate-control-fd".into(),
        GATE_CONTROL_FD.to_string(),
        "--candidate-executable".into(),
        spec.candidate_executable.clone(),
        "--candidate-sha256".into(),
        hex_digest(&spec.candidate_sha256),
    ]);
    args
}

fn build_systemd_run_arguments(
    spec: &CandidateSandboxSpec,
    bwrap_args: &[String],
) -> Vec<String> {
    let mut args = vec![
        "--user".into(),
        "--scope".into(),
        // Scope mode runs as a child of systemd-run and inherits the launcher
        // environment/stdio. The trusted launcher must explicitly preserve fd 3
        // across fork/exec; --pipe is a transient-service mode and is unnecessary.
        format!("--property=MemoryMax={}", spec.memory_max_bytes),
        "--property=MemorySwapMax=0".into(),
        format!("--property=CPUQuota={}%", spec.cpu_quota_percent),
        format!("--property=TasksMax={}", spec.tasks_max),
        "--property=OOMPolicy=kill".into(),
        "--property=KillMode=control-group".into(),
        format!("--property=RuntimeMaxSec={}ms", spec.wall_timeout_ms),
        "--".into(),
        spec.bubblewrap_executable.clone(),
    ];
    args.extend_from_slice(bwrap_args);
    args
}

/// Validate the requested profile and create deterministic, exact argv vectors.
///
/// The caller must separately verify executable/store tree digests, inherit
/// fds 3-7 as pinned by this plan, launch under the system manager, verify
/// Bubblewrap JSON status plus live namespaces/cgroup state, and operate a
/// trusted post-seccomp gate. This plan is not a qualified execution boundary.
pub fn build_candidate_sandbox_plan(
    spec: &CandidateSandboxSpec,
) -> Result<CandidateSandboxPlan, SandboxPlanError> {
    let closure = validate_spec(spec)?;
    let closure_sha256 = closure_digest(&closure);
    let bwrap_args = build_bubblewrap_arguments(spec, &closure);
    let systemd_args = build_systemd_run_arguments(spec, &bwrap_args);

    let mut hasher = CanonicalHasher::new(b"symthaea.temporal.candidate-sandbox-plan.v1");
    hasher.bytes(spec.candidate_executable.as_bytes());
    hasher.bytes(&spec.candidate_sha256);
    hasher.bytes(spec.gate_executable.as_bytes());
    hasher.bytes(&spec.gate_sha256);
    hasher.bytes(spec.bubblewrap_executable.as_bytes());
    hasher.bytes(&spec.bubblewrap_sha256);
    hasher.bytes(&spec.seccomp_sha256);
    hasher.u64(spec.memory_max_bytes);
    hasher.u32(spec.cpu_quota_percent);
    hasher.u32(spec.tasks_max);
    hasher.u64(spec.wall_timeout_ms);
    hasher.u64(spec.scratch_max_bytes);
    hasher.u64(spec.stdout_max_bytes);
    hasher.u64(spec.stderr_max_bytes);
    hasher.bytes(&closure_sha256);
    hasher.usize(bwrap_args.len());
    for argument in &bwrap_args {
        hasher.bytes(argument.as_bytes());
    }
    hasher.usize(systemd_args.len());
    for argument in &systemd_args {
        hasher.bytes(argument.as_bytes());
    }

    Ok(CandidateSandboxPlan {
        bubblewrap_arguments: bwrap_args,
        systemd_run_arguments: systemd_args,
        candidate_executable: spec.candidate_executable.clone(),
        gate_executable: spec.gate_executable.clone(),
        runtime_closure_sha256: closure_sha256,
        plan_sha256: hasher.finish(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_path(name: &str, hash_character: char) -> String {
        format!("/nix/store/{}-{name}", hash_character.to_string().repeat(32))
    }

    fn spec() -> CandidateSandboxSpec {
        let candidate_store = store_path("candidate", '0');
        let bwrap_store = store_path("bubblewrap", '1');
        let gate_store = store_path("trusted-gate", '3');
        CandidateSandboxSpec {
            candidate_executable: format!("{candidate_store}/bin/edge-candidate"),
            candidate_sha256: [1; 32],
            gate_executable: format!("{gate_store}/bin/edge-candidate-gate"),
            gate_sha256: [7; 32],
            runtime_closure: vec![
                RuntimeClosureEntry {
                    store_path: store_path("libc", '2'),
                    artifact_sha256: [2; 32],
                },
                RuntimeClosureEntry {
                    store_path: candidate_store,
                    artifact_sha256: [3; 32],
                },
                RuntimeClosureEntry {
                    store_path: gate_store,
                    artifact_sha256: [8; 32],
                },
                RuntimeClosureEntry {
                    store_path: bwrap_store.clone(),
                    artifact_sha256: [4; 32],
                },
            ],
            bubblewrap_executable: format!("{bwrap_store}/bin/bwrap"),
            bubblewrap_sha256: [5; 32],
            seccomp_fd: 3,
            seccomp_sha256: [6; 32],
            memory_max_bytes: 128 * 1024 * 1024,
            cpu_quota_percent: 100,
            tasks_max: 16,
            wall_timeout_ms: 30_000,
            scratch_max_bytes: 1024 * 1024,
            stdout_max_bytes: 64 * 1024,
            stderr_max_bytes: 64 * 1024,
        }
    }

    #[test]
    fn profile_requires_explicit_mandatory_namespaces_and_never_try_flags() {
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        for required in [
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
        ] {
            assert!(plan.bubblewrap_arguments.iter().any(|arg| arg == required));
        }
        assert!(!plan.bubblewrap_arguments.iter().any(|arg| arg == "--unshare-all"));
        assert!(!plan.bubblewrap_arguments.iter().any(|arg| arg.ends_with("-try")));
    }

    #[test]
    fn only_exact_runtime_store_entries_are_individually_read_only_bound() {
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        let bind_count = plan.bubblewrap_arguments.iter()
            .filter(|arg| arg.as_str() == "--ro-bind")
            .count();
        assert_eq!(bind_count, 3);
        for (index, argument) in plan.bubblewrap_arguments.iter().enumerate() {
            if argument == "--ro-bind" {
                assert_ne!(plan.bubblewrap_arguments[index + 1], "/nix/store");
                assert_ne!(plan.bubblewrap_arguments[index + 1], "/home");
                assert!(plan.bubblewrap_arguments[index + 1].starts_with("/nix/store/"));
                assert_eq!(
                    plan.bubblewrap_arguments[index + 1],
                    plan.bubblewrap_arguments[index + 2]
                );
            }
        }
        assert!(plan.bubblewrap_arguments.iter().any(|arg| arg == "--clearenv"));
        assert!(plan.bubblewrap_arguments.iter().any(|arg| arg == "--seccomp"));
    }

    #[test]
    fn scope_mode_uses_inherited_stdio_without_pipe_or_wait_flags() {
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        assert!(plan.systemd_run_arguments.iter().any(|arg| arg == "--scope"));
        assert!(!plan.systemd_run_arguments.iter().any(|arg| arg == "--pipe"));
        assert!(!plan.systemd_run_arguments.iter().any(|arg| arg == "--wait"));
        for service_only in [
            "--property=NoNewPrivileges=yes",
            "--property=PrivateNetwork=yes",
            "--property=ProtectHome=yes",
            "--property=ProtectSystem=strict",
        ] {
            assert!(!plan.systemd_run_arguments.iter().any(|arg| arg == service_only));
        }
    }

    #[test]
    fn bubblewrap_runs_the_trusted_gate_not_candidate_directly() {
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        let separator = plan.bubblewrap_arguments.iter().position(|arg| arg == "--").unwrap();
        assert_eq!(plan.bubblewrap_arguments[separator + 1], spec().gate_executable);
        assert_ne!(plan.bubblewrap_arguments[separator + 1], spec().candidate_executable);
        let candidate_arg = plan.bubblewrap_arguments.iter()
            .position(|arg| arg == "--candidate-executable")
            .unwrap();
        assert_eq!(plan.bubblewrap_arguments[candidate_arg + 1], spec().candidate_executable);
    }

    #[test]
    fn bubblewrap_plan_pins_all_gate_and_status_descriptors() {
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        for (flag, descriptor) in [
            ("--seccomp", "3"),
            ("--block-fd", "4"),
            ("--json-status-fd", "5"),
            ("--sync-fd", "7"),
            ("--gate-control-fd", "6"),
        ] {
            let index = plan.bubblewrap_arguments.iter().position(|arg| arg == flag).unwrap();
            assert_eq!(plan.bubblewrap_arguments[index + 1], descriptor, "{flag}");
        }
    }

    #[test]
    fn trusted_gate_must_be_in_the_exact_pinned_runtime_closure() {
        let mut invalid = spec();
        invalid.runtime_closure.retain(|entry| !entry.store_path.contains("trusted-gate"));
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::GateNotInClosure
        );
    }

    #[test]
    fn changing_the_gate_artifact_changes_the_plan_commitment() {
        let first = build_candidate_sandbox_plan(&spec()).unwrap();
        let mut changed = spec();
        changed.gate_sha256[0] ^= 0xff;
        let second = build_candidate_sandbox_plan(&changed).unwrap();
        assert_ne!(first.plan_sha256, second.plan_sha256);
    }

    #[test]
    fn seccomp_descriptor_is_normalized_to_fd_three() {
        let mut invalid = spec();
        invalid.seccomp_fd = 4;
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::InvalidSeccompDescriptor
        );
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        let index = plan.bubblewrap_arguments.iter()
            .position(|arg| arg == "--seccomp")
            .unwrap();
        assert_eq!(plan.bubblewrap_arguments[index + 1], "3");
    }

    #[test]
    fn systemd_scope_plan_binds_memory_swap_cpu_task_and_runtime_limits() {
        let plan = build_candidate_sandbox_plan(&spec()).unwrap();
        for required in [
            "--property=MemoryMax=134217728",
            "--property=MemorySwapMax=0",
            "--property=CPUQuota=100%",
            "--property=TasksMax=16",
            "--property=OOMPolicy=kill",
            "--property=KillMode=control-group",
            "--property=RuntimeMaxSec=30000ms",
        ] {
            assert!(plan.systemd_run_arguments.iter().any(|arg| arg == required));
        }
    }

    #[test]
    fn runtime_closure_input_order_does_not_change_the_plan_digest() {
        let mut first = spec();
        let mut second = spec();
        second.runtime_closure.reverse();
        let a = build_candidate_sandbox_plan(&first).unwrap();
        let b = build_candidate_sandbox_plan(&second).unwrap();
        assert_eq!(a.plan_sha256, b.plan_sha256);
        assert_eq!(a.runtime_closure_sha256, b.runtime_closure_sha256);
    }

    #[test]
    fn changing_any_resource_limit_changes_the_plan_commitment() {
        let first = build_candidate_sandbox_plan(&spec()).unwrap();
        let mut changed = spec();
        changed.memory_max_bytes += 4096;
        let second = build_candidate_sandbox_plan(&changed).unwrap();
        assert_ne!(first.plan_sha256, second.plan_sha256);
    }

    #[test]
    fn candidate_must_be_inside_an_exact_pinned_store_closure_entry() {
        let mut invalid = spec();
        invalid.candidate_executable = "/nix/store/00000000000000000000000000000000-other/bin/candidate".into();
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::CandidateNotInClosure
        );
    }

    #[test]
    fn parent_traversal_and_the_whole_nix_store_are_rejected() {
        let mut invalid = spec();
        invalid.candidate_executable = "/nix/store/../etc/passwd".into();
        assert!(matches!(
            build_candidate_sandbox_plan(&invalid),
            Err(SandboxPlanError::InvalidPath(_))
        ));
        invalid = spec();
        invalid.runtime_closure[0].store_path = "/nix/store".into();
        assert!(matches!(
            build_candidate_sandbox_plan(&invalid),
            Err(SandboxPlanError::InvalidStorePath(_))
        ));
    }

    #[test]
    fn duplicate_store_entries_fail_closed() {
        let mut invalid = spec();
        invalid.runtime_closure.push(invalid.runtime_closure[0].clone());
        assert!(matches!(
            build_candidate_sandbox_plan(&invalid),
            Err(SandboxPlanError::DuplicateClosureEntry(_))
        ));
    }

    #[test]
    fn zero_seccomp_digest_and_invalid_fd_are_rejected() {
        let mut invalid = spec();
        invalid.seccomp_sha256 = [0; 32];
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::ZeroDigest("seccomp")
        );
        invalid = spec();
        invalid.seccomp_fd = 0;
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::InvalidSeccompDescriptor
        );
    }

    #[test]
    fn resource_limits_fail_closed_outside_declared_ranges() {
        let mut invalid = spec();
        invalid.memory_max_bytes = 1;
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::InvalidLimit("memory_max_bytes")
        );
        invalid = spec();
        invalid.stdout_max_bytes = 0;
        assert_eq!(
            build_candidate_sandbox_plan(&invalid).unwrap_err(),
            SandboxPlanError::InvalidLimit("stdout_max_bytes")
        );
    }
}
