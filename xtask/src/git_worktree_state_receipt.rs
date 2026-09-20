use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path};

const STATE_SCHEMA: &str = "symthaea.git-worktree-state.v1";
const STATE_HASH_DOMAIN: &[u8] = b"symthaea.git-worktree-state.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ValidatedGitWorktreeState {
    pub state_id: String,
    pub schema: String,
    pub repository_source_snapshot_id: String,
    pub git_version: String,
    pub index_flags: Vec<GitIndexFlags>,
    pub sparse_checkout: SparseCheckoutState,
    pub ignore_environment: IgnoreEnvironment,
    pub fsmonitor_environment: FsmonitorEnvironment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct GitIndexFlags {
    pub path: String,
    pub skip_worktree: bool,
    pub assume_unchanged: bool,
    pub fsmonitor_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SparseCheckoutState {
    pub enabled: bool,
    pub cone_mode_configured: bool,
    pub sparse_index_configured: bool,
    pub specification_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct IgnoreEnvironment {
    pub info_exclude_present: bool,
    pub info_exclude_sha256: Option<String>,
    pub global_excludes_configured: bool,
    pub global_excludes_present: bool,
    pub global_excludes_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FsmonitorEnvironment {
    pub configured: bool,
    pub config_value_sha256: Option<String>,
}

#[derive(Serialize)]
struct StateIdentity<'a> {
    schema: &'a str,
    repository_source_snapshot_id: &'a str,
    git_version: &'a str,
    index_flags: &'a [GitIndexFlags],
    sparse_checkout: &'a SparseCheckoutState,
    ignore_environment: &'a IgnoreEnvironment,
    fsmonitor_environment: &'a FsmonitorEnvironment,
}

impl ValidatedGitWorktreeState {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read Git worktree-state receipt {}", path.display()))?;
        verify_bytes(&bytes)
    }

    pub(crate) fn validate(&mut self) -> anyhow::Result<()> {
        if self.schema != STATE_SCHEMA {
            bail!("unsupported Git worktree-state schema: {}", self.schema);
        }
        normalize_digest("state_id", &mut self.state_id)?;
        normalize_digest(
            "repository_source_snapshot_id",
            &mut self.repository_source_snapshot_id,
        )?;
        validate_canonical_text("git_version", &self.git_version)?;
        validate_index_flags(&self.index_flags)?;
        validate_sparse_checkout(&mut self.sparse_checkout)?;
        validate_ignore_environment(&mut self.ignore_environment)?;
        validate_fsmonitor_environment(&mut self.fsmonitor_environment)?;

        let computed = self.computed_state_id()?;
        if computed != self.state_id {
            bail!(
                "Git worktree-state identity mismatch: declared {}, computed {}",
                self.state_id,
                computed
            );
        }
        Ok(())
    }

    pub(crate) fn computed_state_id(&self) -> anyhow::Result<String> {
        let identity = StateIdentity {
            schema: &self.schema,
            repository_source_snapshot_id: &self.repository_source_snapshot_id,
            git_version: &self.git_version,
            index_flags: &self.index_flags,
            sparse_checkout: &self.sparse_checkout,
            ignore_environment: &self.ignore_environment,
            fsmonitor_environment: &self.fsmonitor_environment,
        };
        let bytes = serde_json::to_vec(&identity)
            .context("serialize canonical Git worktree-state payload")?;
        Ok(domain_sha256(STATE_HASH_DOMAIN, &bytes))
    }
}

pub(crate) fn verify_bytes(bytes: &[u8]) -> anyhow::Result<ValidatedGitWorktreeState> {
    let mut state: ValidatedGitWorktreeState =
        serde_json::from_slice(bytes).context("parse Git worktree-state receipt")?;
    state.validate()?;
    Ok(state)
}

pub(crate) fn verify_for_source(
    mut state: ValidatedGitWorktreeState,
    expected_source_snapshot_id: &str,
) -> anyhow::Result<ValidatedGitWorktreeState> {
    state.validate()?;
    let mut expected = expected_source_snapshot_id.to_string();
    normalize_digest("expected_source_snapshot_id", &mut expected)?;
    if state.repository_source_snapshot_id != expected {
        bail!(
            "Git worktree-state source {} does not match expected repository source {}",
            state.repository_source_snapshot_id,
            expected
        );
    }
    Ok(state)
}

fn validate_index_flags(values: &[GitIndexFlags]) -> anyhow::Result<()> {
    let mut previous: Option<&str> = None;
    for value in values {
        validate_canonical_relative_path(&value.path)?;
        if let Some(previous) = previous {
            if previous >= value.path.as_str() {
                bail!(
                    "Git index-flag paths must be strictly sorted and unique: {previous:?} then {:?}",
                    value.path
                );
            }
        }
        previous = Some(&value.path);
    }
    Ok(())
}

fn validate_sparse_checkout(value: &mut SparseCheckoutState) -> anyhow::Result<()> {
    normalize_optional_digest(
        "sparse_checkout.specification_sha256",
        &mut value.specification_sha256,
    )?;
    if value.enabled != value.specification_sha256.is_some() {
        bail!(
            "sparse checkout specification digest must be present iff sparse checkout is enabled"
        );
    }
    Ok(())
}

fn validate_ignore_environment(value: &mut IgnoreEnvironment) -> anyhow::Result<()> {
    normalize_optional_digest(
        "ignore_environment.info_exclude_sha256",
        &mut value.info_exclude_sha256,
    )?;
    normalize_optional_digest(
        "ignore_environment.global_excludes_sha256",
        &mut value.global_excludes_sha256,
    )?;

    if value.info_exclude_present != value.info_exclude_sha256.is_some() {
        bail!("info/exclude presence must exactly match its content digest presence");
    }
    if !value.global_excludes_configured
        && (value.global_excludes_present || value.global_excludes_sha256.is_some())
    {
        bail!("unconfigured global excludes must not claim a file or content digest");
    }
    if value.global_excludes_present != value.global_excludes_sha256.is_some() {
        bail!("global excludes file presence must exactly match its content digest presence");
    }
    Ok(())
}

fn validate_fsmonitor_environment(value: &mut FsmonitorEnvironment) -> anyhow::Result<()> {
    normalize_optional_digest(
        "fsmonitor_environment.config_value_sha256",
        &mut value.config_value_sha256,
    )?;
    if value.configured != value.config_value_sha256.is_some() {
        bail!("fsmonitor configured state must exactly match config-value digest presence");
    }
    Ok(())
}

fn validate_canonical_relative_path(value: &str) -> anyhow::Result<()> {
    if value.is_empty() {
        bail!("Git tracked path must not be empty");
    }
    let path = Path::new(value);
    if path.is_absolute() {
        bail!("Git tracked path must be repository-relative: {value:?}");
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(
                part.to_str()
                    .context("Git tracked path component is not valid UTF-8")?,
            ),
            Component::CurDir => bail!("Git tracked path must already be canonical: {value:?}"),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("Git tracked path escapes repository scope: {value:?}")
            }
        }
    }
    if parts.is_empty() {
        bail!("Git tracked path must identify an entry");
    }
    let canonical = parts.join("/");
    if canonical != value {
        bail!("Git tracked path must already be canonical: {value:?}");
    }
    if canonical == ".git" || canonical.starts_with(".git/") {
        bail!("Git tracked path must not address .git internals");
    }
    Ok(())
}

fn validate_canonical_text(name: &str, value: &str) -> anyhow::Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{name} must be non-empty canonical text without surrounding whitespace");
    }
    if value.bytes().any(|byte| matches!(byte, b'\r' | b'\n' | 0)) {
        bail!("{name} must not contain line separators or NUL");
    }
    if value.len() > 256 {
        bail!("{name} exceeds the 256-byte v1 limit");
    }
    Ok(())
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        normalize_digest(name, value)?;
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    validate_digest(name, value)?;
    value.make_ascii_lowercase();
    Ok(())
}

fn validate_digest(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
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
