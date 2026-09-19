use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "symthaea.cargo-adapter-semantics-input.v1";
const RECEIPT_SCHEMA: &str = "symthaea.cargo-adapter-semantics.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.cargo-adapter-semantics.v1\0";
const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const MAX_GRACE_MS: u64 = 5 * 60 * 1000;

/// V1 intentionally permits only ambient variables that cannot select a
/// compiler, wrapper, loader, credential source, network proxy, or mutable
/// execution directory. Exact values are execution-specific evidence and are
/// therefore not part of this reusable semantics receipt.
const SAFE_INHERITED_ENV_KEYS: &[&str] = &[
    "CARGO_TERM_COLOR",
    "LANG",
    "LC_ALL",
    "RUST_BACKTRACE",
    "SOURCE_DATE_EPOCH",
    "TERM",
    "TZ",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PlatformFamily {
    Linux,
    Macos,
    Windows,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RepositoryAccess {
    ReadOnlySource,
    IsolatedStagingTree,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkingDirectoryPolicy {
    RepositoryRoot,
    StagingRoot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NetworkPolicy {
    Denied,
    LoopbackOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EnvironmentInheritancePolicy {
    ClearThenAllowlist,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CargoHomePolicy {
    EphemeralEmpty,
    /// The prefetched cache is a bound read-only input. The adapter may not
    /// mutate or populate it during the admitted execution.
    ReadOnlyPrefetched,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EphemeralDirectoryPolicy {
    EphemeralEmpty,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BoundInputPolicy {
    ReadOnlyBoundClosures,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DescendantPolicy {
    SameSandbox,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StdinPolicy {
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapturePolicy {
    ExactBytesNoTruncation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProcessTreePolicy {
    TerminateThenKillEntireSandboxTree,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PointOfNoReturnPolicy {
    AfterEffectAdmissionBeforeSpawn,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct EnvironmentPolicy {
    pub inheritance: EnvironmentInheritancePolicy,
    pub allowed_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SandboxBackend {
    pub name: String,
    pub implementation_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoAdapterSemanticsSpec {
    pub schema: String,
    pub adapter_implementation_sha256: String,
    pub platform: PlatformFamily,
    pub sandbox_backend: SandboxBackend,
    pub repository_access: RepositoryAccess,
    pub working_directory: WorkingDirectoryPolicy,
    pub network: NetworkPolicy,
    pub environment: EnvironmentPolicy,
    pub home: EphemeralDirectoryPolicy,
    pub cargo_home: CargoHomePolicy,
    pub target_dir: EphemeralDirectoryPolicy,
    pub temp_dir: EphemeralDirectoryPolicy,
    pub external_inputs: BoundInputPolicy,
    pub descendants: DescendantPolicy,
    pub stdin: StdinPolicy,
    pub stdout: CapturePolicy,
    pub stderr: CapturePolicy,
    pub wall_clock_timeout_ms: u64,
    pub termination_grace_ms: u64,
    pub process_tree: ProcessTreePolicy,
    pub point_of_no_return: PointOfNoReturnPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoAdapterSemanticsReceipt {
    pub adapter_semantics_id: String,
    pub schema: String,
    pub adapter_implementation_sha256: String,
    pub platform: PlatformFamily,
    pub sandbox_backend: SandboxBackend,
    pub repository_access: RepositoryAccess,
    pub working_directory: WorkingDirectoryPolicy,
    pub network: NetworkPolicy,
    pub environment: EnvironmentPolicy,
    pub home: EphemeralDirectoryPolicy,
    pub cargo_home: CargoHomePolicy,
    pub target_dir: EphemeralDirectoryPolicy,
    pub temp_dir: EphemeralDirectoryPolicy,
    pub external_inputs: BoundInputPolicy,
    pub descendants: DescendantPolicy,
    pub stdin: StdinPolicy,
    pub stdout: CapturePolicy,
    pub stderr: CapturePolicy,
    pub wall_clock_timeout_ms: u64,
    pub termination_grace_ms: u64,
    pub process_tree: ProcessTreePolicy,
    pub point_of_no_return: PointOfNoReturnPolicy,
}

#[derive(Serialize)]
struct SemanticsIdentity<'a> {
    schema: &'static str,
    adapter_implementation_sha256: &'a str,
    platform: &'a PlatformFamily,
    sandbox_backend: &'a SandboxBackend,
    repository_access: &'a RepositoryAccess,
    working_directory: &'a WorkingDirectoryPolicy,
    network: &'a NetworkPolicy,
    environment: &'a EnvironmentPolicy,
    home: &'a EphemeralDirectoryPolicy,
    cargo_home: &'a CargoHomePolicy,
    target_dir: &'a EphemeralDirectoryPolicy,
    temp_dir: &'a EphemeralDirectoryPolicy,
    external_inputs: &'a BoundInputPolicy,
    descendants: &'a DescendantPolicy,
    stdin: &'a StdinPolicy,
    stdout: &'a CapturePolicy,
    stderr: &'a CapturePolicy,
    wall_clock_timeout_ms: u64,
    termination_grace_ms: u64,
    process_tree: &'a ProcessTreePolicy,
    point_of_no_return: &'a PointOfNoReturnPolicy,
}

pub(crate) fn run(spec_path: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo adapter semantics spec {}", spec_path.display()))?;
    let spec: CargoAdapterSemanticsSpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo adapter semantics spec {}", spec_path.display()))?;
    let receipt = build_receipt(spec)?;
    write_json(&receipt, output)
}

pub(crate) fn run_verify(receipt_path: &Path) -> anyhow::Result<()> {
    CargoAdapterSemanticsReceipt::load(receipt_path)?;
    Ok(())
}

pub(crate) fn build_receipt(
    mut spec: CargoAdapterSemanticsSpec,
) -> anyhow::Result<CargoAdapterSemanticsReceipt> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported Cargo adapter-semantics input schema: {}", spec.schema);
    }

    normalize_digest(
        "adapter_implementation_sha256",
        &mut spec.adapter_implementation_sha256,
    )?;
    normalize_backend(&mut spec.sandbox_backend)?;
    normalize_environment(&mut spec.environment)?;
    validate_cross_fields(&spec)?;

    let identity = SemanticsIdentity {
        schema: RECEIPT_SCHEMA,
        adapter_implementation_sha256: &spec.adapter_implementation_sha256,
        platform: &spec.platform,
        sandbox_backend: &spec.sandbox_backend,
        repository_access: &spec.repository_access,
        working_directory: &spec.working_directory,
        network: &spec.network,
        environment: &spec.environment,
        home: &spec.home,
        cargo_home: &spec.cargo_home,
        target_dir: &spec.target_dir,
        temp_dir: &spec.temp_dir,
        external_inputs: &spec.external_inputs,
        descendants: &spec.descendants,
        stdin: &spec.stdin,
        stdout: &spec.stdout,
        stderr: &spec.stderr,
        wall_clock_timeout_ms: spec.wall_clock_timeout_ms,
        termination_grace_ms: spec.termination_grace_ms,
        process_tree: &spec.process_tree,
        point_of_no_return: &spec.point_of_no_return,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize Cargo adapter-semantics identity")?;
    let adapter_semantics_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(CargoAdapterSemanticsReceipt {
        adapter_semantics_id,
        schema: RECEIPT_SCHEMA.into(),
        adapter_implementation_sha256: spec.adapter_implementation_sha256,
        platform: spec.platform,
        sandbox_backend: spec.sandbox_backend,
        repository_access: spec.repository_access,
        working_directory: spec.working_directory,
        network: spec.network,
        environment: spec.environment,
        home: spec.home,
        cargo_home: spec.cargo_home,
        target_dir: spec.target_dir,
        temp_dir: spec.temp_dir,
        external_inputs: spec.external_inputs,
        descendants: spec.descendants,
        stdin: spec.stdin,
        stdout: spec.stdout,
        stderr: spec.stderr,
        wall_clock_timeout_ms: spec.wall_clock_timeout_ms,
        termination_grace_ms: spec.termination_grace_ms,
        process_tree: spec.process_tree,
        point_of_no_return: spec.point_of_no_return,
    })
}

impl CargoAdapterSemanticsReceipt {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("read Cargo adapter-semantics receipt {}", path.display()))?;
        let mut receipt: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse Cargo adapter-semantics receipt {}", path.display()))?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub(crate) fn validate(&mut self) -> anyhow::Result<()> {
        if self.schema != RECEIPT_SCHEMA {
            bail!("unsupported Cargo adapter-semantics receipt schema: {}", self.schema);
        }
        normalize_digest("adapter_semantics_id", &mut self.adapter_semantics_id)?;
        normalize_digest(
            "adapter_implementation_sha256",
            &mut self.adapter_implementation_sha256,
        )?;
        normalize_backend(&mut self.sandbox_backend)?;
        normalize_environment(&mut self.environment)?;

        let spec = CargoAdapterSemanticsSpec {
            schema: INPUT_SCHEMA.into(),
            adapter_implementation_sha256: self.adapter_implementation_sha256.clone(),
            platform: self.platform.clone(),
            sandbox_backend: self.sandbox_backend.clone(),
            repository_access: self.repository_access.clone(),
            working_directory: self.working_directory.clone(),
            network: self.network.clone(),
            environment: self.environment.clone(),
            home: self.home.clone(),
            cargo_home: self.cargo_home.clone(),
            target_dir: self.target_dir.clone(),
            temp_dir: self.temp_dir.clone(),
            external_inputs: self.external_inputs.clone(),
            descendants: self.descendants.clone(),
            stdin: self.stdin.clone(),
            stdout: self.stdout.clone(),
            stderr: self.stderr.clone(),
            wall_clock_timeout_ms: self.wall_clock_timeout_ms,
            termination_grace_ms: self.termination_grace_ms,
            process_tree: self.process_tree.clone(),
            point_of_no_return: self.point_of_no_return.clone(),
        };
        validate_cross_fields(&spec)?;

        let rebuilt = build_receipt(spec)?;
        if rebuilt.adapter_semantics_id != self.adapter_semantics_id {
            bail!(
                "Cargo adapter-semantics identity mismatch: declared {}, computed {}",
                self.adapter_semantics_id,
                rebuilt.adapter_semantics_id
            );
        }
        Ok(())
    }
}

fn normalize_backend(backend: &mut SandboxBackend) -> anyhow::Result<()> {
    let trimmed = backend.name.trim();
    if trimmed.is_empty() || trimmed.len() > 64 {
        bail!("sandbox backend name must contain 1..=64 characters");
    }
    if !trimmed
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("sandbox backend name contains non-portable characters: {trimmed:?}");
    }
    backend.name = trimmed.to_string();
    normalize_digest(
        "sandbox_backend.implementation_sha256",
        &mut backend.implementation_sha256,
    )
}

fn normalize_environment(environment: &mut EnvironmentPolicy) -> anyhow::Result<()> {
    for key in &environment.allowed_keys {
        validate_env_key(key)?;
        if !SAFE_INHERITED_ENV_KEYS.contains(&key.as_str()) {
            bail!(
                "environment key {key:?} is not permitted for ambient inheritance in adapter-semantics v1"
            );
        }
    }
    environment.allowed_keys.sort();
    environment.allowed_keys.dedup();
    Ok(())
}

fn validate_env_key(key: &str) -> anyhow::Result<()> {
    let mut bytes = key.bytes();
    let Some(first) = bytes.next() else {
        bail!("environment allowlist contains an empty key");
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        bail!("environment key is not a portable identifier: {key:?}");
    }
    Ok(())
}

fn validate_cross_fields(spec: &CargoAdapterSemanticsSpec) -> anyhow::Result<()> {
    match (&spec.repository_access, &spec.working_directory) {
        (RepositoryAccess::ReadOnlySource, WorkingDirectoryPolicy::RepositoryRoot)
        | (RepositoryAccess::IsolatedStagingTree, WorkingDirectoryPolicy::StagingRoot) => {}
        _ => bail!("repository access and working-directory policy are inconsistent"),
    }

    if spec.wall_clock_timeout_ms == 0 || spec.wall_clock_timeout_ms > MAX_TIMEOUT_MS {
        bail!(
            "wall_clock_timeout_ms must be within 1..={MAX_TIMEOUT_MS}, got {}",
            spec.wall_clock_timeout_ms
        );
    }
    if spec.termination_grace_ms == 0 || spec.termination_grace_ms > MAX_GRACE_MS {
        bail!(
            "termination_grace_ms must be within 1..={MAX_GRACE_MS}, got {}",
            spec.termination_grace_ms
        );
    }
    if spec.termination_grace_ms >= spec.wall_clock_timeout_ms {
        bail!("termination_grace_ms must be smaller than wall_clock_timeout_ms");
    }
    Ok(())
}

fn write_json(receipt: &CargoAdapterSemanticsReceipt, output: Option<PathBuf>) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(receipt)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create adapter-semantics output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo adapter-semantics receipt {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn spec() -> CargoAdapterSemanticsSpec {
        CargoAdapterSemanticsSpec {
            schema: INPUT_SCHEMA.into(),
            adapter_implementation_sha256: digest('a'),
            platform: PlatformFamily::Linux,
            sandbox_backend: SandboxBackend {
                name: "bubblewrap".into(),
                implementation_sha256: digest('b'),
            },
            repository_access: RepositoryAccess::ReadOnlySource,
            working_directory: WorkingDirectoryPolicy::RepositoryRoot,
            network: NetworkPolicy::Denied,
            environment: EnvironmentPolicy {
                inheritance: EnvironmentInheritancePolicy::ClearThenAllowlist,
                allowed_keys: vec!["RUST_BACKTRACE".into(), "CARGO_TERM_COLOR".into()],
            },
            home: EphemeralDirectoryPolicy::EphemeralEmpty,
            cargo_home: CargoHomePolicy::ReadOnlyPrefetched,
            target_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
            temp_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
            external_inputs: BoundInputPolicy::ReadOnlyBoundClosures,
            descendants: DescendantPolicy::SameSandbox,
            stdin: StdinPolicy::Null,
            stdout: CapturePolicy::ExactBytesNoTruncation,
            stderr: CapturePolicy::ExactBytesNoTruncation,
            wall_clock_timeout_ms: 20 * 60 * 1000,
            termination_grace_ms: 5_000,
            process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
            point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
        }
    }

    #[test]
    fn environment_order_and_digest_spelling_canonicalize() {
        let a = build_receipt(spec()).unwrap();
        let mut other = spec();
        other.environment.allowed_keys = vec![
            "CARGO_TERM_COLOR".into(),
            "RUST_BACKTRACE".into(),
            "CARGO_TERM_COLOR".into(),
        ];
        other.adapter_implementation_sha256.make_ascii_uppercase();
        let b = build_receipt(other).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ambient_compilation_or_directory_controls_are_rejected() {
        for key in [
            "RUSTC_WRAPPER",
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "HOME",
            "CARGO_HOME",
            "CARGO_TARGET_DIR",
            "LD_PRELOAD",
            "HTTP_PROXY",
            "SSH_AUTH_SOCK",
        ] {
            let mut bad = spec();
            bad.environment.allowed_keys.push(key.into());
            assert!(build_receipt(bad).is_err(), "ambient key should reject: {key}");
        }
    }

    #[test]
    fn access_and_working_directory_must_match() {
        let mut bad = spec();
        bad.repository_access = RepositoryAccess::IsolatedStagingTree;
        assert!(build_receipt(bad).is_err());
    }

    #[test]
    fn unsafe_or_unbounded_timeout_shapes_reject() {
        let mut zero = spec();
        zero.wall_clock_timeout_ms = 0;
        assert!(build_receipt(zero).is_err());

        let mut inverted = spec();
        inverted.termination_grace_ms = inverted.wall_clock_timeout_ms;
        assert!(build_receipt(inverted).is_err());
    }

    #[test]
    fn backend_and_environment_inputs_are_portable() {
        let mut bad_backend = spec();
        bad_backend.sandbox_backend.name = "/usr/bin/bwrap".into();
        assert!(build_receipt(bad_backend).is_err());

        let mut bad_env = spec();
        bad_env.environment.allowed_keys.push("SECRET=VALUE".into());
        assert!(build_receipt(bad_env).is_err());
    }

    #[test]
    fn material_semantics_change_identity() {
        let denied = build_receipt(spec()).unwrap();

        let mut loopback = spec();
        loopback.network = NetworkPolicy::LoopbackOnly;
        let loopback = build_receipt(loopback).unwrap();
        assert_ne!(denied.adapter_semantics_id, loopback.adapter_semantics_id);

        let mut staged = spec();
        staged.repository_access = RepositoryAccess::IsolatedStagingTree;
        staged.working_directory = WorkingDirectoryPolicy::StagingRoot;
        let staged = build_receipt(staged).unwrap();
        assert_ne!(denied.adapter_semantics_id, staged.adapter_semantics_id);

        let mut ephemeral_cargo = spec();
        ephemeral_cargo.cargo_home = CargoHomePolicy::EphemeralEmpty;
        let ephemeral_cargo = build_receipt(ephemeral_cargo).unwrap();
        assert_ne!(denied.adapter_semantics_id, ephemeral_cargo.adapter_semantics_id);
    }

    #[test]
    fn stored_receipt_recomputes_to_declared_identity() {
        let mut receipt = build_receipt(spec()).unwrap();
        receipt.wall_clock_timeout_ms += 1;
        assert!(receipt.validate().is_err());
    }
}
