use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};

use crate::repository_snapshot;
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const STATE_SCHEMA: &str = "symthaea.git-worktree-state.v1";
const STATE_HASH_DOMAIN: &[u8] = b"symthaea.git-worktree-state.v1\0";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GitWorktreeState {
    pub state_id: String,
    #[serde(flatten)]
    pub payload: GitWorktreeStatePayload,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GitWorktreeStatePayload {
    pub schema: &'static str,
    pub repository_source_snapshot_id: String,
    pub git_version: String,
    pub index_flags: Vec<GitIndexFlags>,
    pub sparse_checkout: SparseCheckoutState,
    pub ignore_environment: IgnoreEnvironment,
    pub fsmonitor_environment: FsmonitorEnvironment,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GitIndexFlags {
    pub path: String,
    pub skip_worktree: bool,
    pub assume_unchanged: bool,
    pub fsmonitor_valid: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SparseCheckoutState {
    pub enabled: bool,
    pub cone_mode_configured: bool,
    pub sparse_index_configured: bool,
    pub specification_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IgnoreEnvironment {
    pub info_exclude_present: bool,
    pub info_exclude_sha256: Option<String>,
    pub global_excludes_configured: bool,
    pub global_excludes_present: bool,
    pub global_excludes_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FsmonitorEnvironment {
    pub configured: bool,
    pub config_value_sha256: Option<String>,
}

#[derive(Debug, Clone)]
struct FlagSensor {
    path: String,
    flagged: bool,
}

#[derive(Debug, Clone, Copy)]
enum SensorKind {
    SkipWorktree,
    Lowercase,
}

pub fn run(root: &Path, source_path: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let source = ValidatedSnapshotReceipt::load(source_path)?;
    let current = repository_snapshot::build_snapshot(root, source.explicit_ignored_paths())?;
    if current.snapshot_id != source.snapshot_id {
        bail!(
            "repository source subject drift before Git-state capture: expected {}, current {}",
            source.snapshot_id,
            current.snapshot_id
        );
    }

    let state = build_state(root, &source.snapshot_id)?;
    let mut rendered = serde_json::to_string_pretty(&state)?;
    rendered.push('\n');
    if let Some(path) = output {
        reject_output_inside_repository(root, &path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("create Git-worktree-state output directory")?;
        }
        fs::write(&path, rendered).context("write Git-worktree-state receipt")?;
        println!("Git worktree state written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub fn build_state(root: &Path, source_snapshot_id: &str) -> anyhow::Result<GitWorktreeState> {
    verify_repo_root(root)?;
    validate_sha256("repository_source_snapshot_id", source_snapshot_id)?;
    let repository_source_snapshot_id = source_snapshot_id.to_ascii_lowercase();
    let git_version = git_text(root, &["--version"])?;

    let skip = parse_flag_sensor(
        &git_bytes(root, &["ls-files", "--cached", "-t", "-z", "--full-name"])?,
        SensorKind::SkipWorktree,
    )?;
    let assume = parse_flag_sensor(
        &git_bytes(root, &["ls-files", "--cached", "-v", "-z", "--full-name"])?,
        SensorKind::Lowercase,
    )?;
    let fsmonitor = parse_flag_sensor(
        &git_bytes(root, &["ls-files", "--cached", "-f", "-z", "--full-name"])?,
        SensorKind::Lowercase,
    )?;
    let index_flags = merge_flag_sensors(skip, assume, fsmonitor)?;

    let sparse_enabled = git_config_bool(root, "core.sparseCheckout")?;
    let cone_mode_configured = git_config_bool(root, "core.sparseCheckoutCone")?;
    let sparse_index_configured = git_config_bool(root, "index.sparse")?;
    let specification_sha256 = if sparse_enabled {
        Some(hash_required_file(
            &git_path(root, "info/sparse-checkout")?,
            "sparse-checkout specification",
        )?)
    } else {
        None
    };

    let (info_exclude_present, info_exclude_sha256) =
        hash_optional_file(&git_path(root, "info/exclude")?)?;
    let global_path = git_config_path(root, "core.excludesFile")?;
    let (global_excludes_configured, global_excludes_present, global_excludes_sha256) =
        match global_path {
            None => (false, false, None),
            Some(path) => {
                let (present, digest) = hash_optional_file(&path)?;
                (true, present, digest)
            }
        };

    let fsmonitor_config = git_config_raw(root, "core.fsmonitor")?;
    let payload = GitWorktreeStatePayload {
        schema: STATE_SCHEMA,
        repository_source_snapshot_id,
        git_version,
        index_flags,
        sparse_checkout: SparseCheckoutState {
            enabled: sparse_enabled,
            cone_mode_configured,
            sparse_index_configured,
            specification_sha256,
        },
        ignore_environment: IgnoreEnvironment {
            info_exclude_present,
            info_exclude_sha256,
            global_excludes_configured,
            global_excludes_present,
            global_excludes_sha256,
        },
        fsmonitor_environment: FsmonitorEnvironment {
            configured: fsmonitor_config.is_some(),
            config_value_sha256: fsmonitor_config.as_deref().map(|v| sha256(v.as_bytes())),
        },
    };

    let bytes = serde_json::to_vec(&payload).context("serialize Git worktree-state payload")?;
    let state_id = domain_sha256(STATE_HASH_DOMAIN, &bytes);
    Ok(GitWorktreeState { state_id, payload })
}

fn parse_flag_sensor(bytes: &[u8], kind: SensorKind) -> anyhow::Result<Vec<FlagSensor>> {
    let mut records = Vec::new();
    for raw in bytes.split(|b| *b == 0).filter(|record| !record.is_empty()) {
        if raw.len() < 3 || raw[1] != b' ' {
            bail!("unexpected git ls-files status record shape");
        }
        let status = raw[0] as char;
        let path = std::str::from_utf8(&raw[2..]).context("Git tracked path is not valid UTF-8")?;
        records.push(FlagSensor {
            path: normalize_relative_path(path)?,
            flagged: match kind {
                SensorKind::SkipWorktree => status == 'S',
                SensorKind::Lowercase => status.is_ascii_lowercase(),
            },
        });
    }
    records.sort_by(|a, b| a.path.cmp(&b.path));
    for pair in records.windows(2) {
        if pair[0].path == pair[1].path {
            bail!("duplicate tracked path reported by Git sensor: {}", pair[0].path);
        }
    }
    Ok(records)
}

fn merge_flag_sensors(
    skip: Vec<FlagSensor>,
    assume: Vec<FlagSensor>,
    fsmonitor: Vec<FlagSensor>,
) -> anyhow::Result<Vec<GitIndexFlags>> {
    let skip = sensor_map(skip);
    let assume = sensor_map(assume);
    let fsmonitor = sensor_map(fsmonitor);
    let skip_paths: Vec<_> = skip.keys().collect();
    if skip_paths != assume.keys().collect::<Vec<_>>()
        || skip_paths != fsmonitor.keys().collect::<Vec<_>>()
    {
        bail!("Git index-flag sensors disagree on the tracked path set");
    }
    Ok(skip
        .into_iter()
        .map(|(path, skip_worktree)| GitIndexFlags {
            assume_unchanged: *assume.get(&path).expect("validated path-set equality"),
            fsmonitor_valid: *fsmonitor.get(&path).expect("validated path-set equality"),
            path,
            skip_worktree,
        })
        .collect())
}

fn sensor_map(values: Vec<FlagSensor>) -> BTreeMap<String, bool> {
    values.into_iter().map(|v| (v.path, v.flagged)).collect()
}

fn git_config_bool(root: &Path, key: &str) -> anyhow::Result<bool> {
    let output = git_output(root, &["config", "--bool", "--get", key])?;
    if output.status.success() {
        return match utf8_trimmed(&output.stdout, "Git boolean config")? {
            "true" => Ok(true),
            "false" => Ok(false),
            other => bail!("Git config {key} returned non-boolean value {other:?}"),
        };
    }
    if output.status.code() == Some(1) && output.stdout.is_empty() {
        return Ok(false);
    }
    bail_git("read Git boolean config", output)
}

fn git_config_raw(root: &Path, key: &str) -> anyhow::Result<Option<String>> {
    let output = git_output(root, &["config", "--get", key])?;
    if output.status.success() {
        let value = std::str::from_utf8(&output.stdout)
            .context("Git config output is not valid UTF-8")?
            .trim_end_matches(|c| c == '\r' || c == '\n')
            .to_string();
        if value.is_empty() {
            bail!("Git config {key} returned an empty value");
        }
        return Ok(Some(value));
    }
    if output.status.code() == Some(1) && output.stdout.is_empty() {
        return Ok(None);
    }
    bail_git("read Git config", output)
}

fn git_config_path(root: &Path, key: &str) -> anyhow::Result<Option<PathBuf>> {
    let output = git_output(root, &["config", "--path", "--get", key])?;
    if output.status.success() {
        let value = utf8_trimmed(&output.stdout, "Git path config")?;
        if value.is_empty() {
            bail!("Git path config {key} returned an empty path");
        }
        let path = PathBuf::from(value);
        return Ok(Some(if path.is_absolute() { path } else { root.join(path) }));
    }
    if output.status.code() == Some(1) && output.stdout.is_empty() {
        return Ok(None);
    }
    bail_git("read Git path config", output)
}

fn git_path(root: &Path, suffix: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(git_text(root, &["rev-parse", "--git-path", suffix])?);
    Ok(if path.is_absolute() { path } else { root.join(path) })
}

fn hash_required_file(path: &Path, label: &str) -> anyhow::Result<String> {
    Ok(sha256(&fs::read(path).with_context(|| format!("read required {label}"))?))
}

fn hash_optional_file(path: &Path) -> anyhow::Result<(bool, Option<String>)> {
    match fs::read(path) {
        Ok(bytes) => Ok((true, Some(sha256(&bytes)))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok((false, None)),
        Err(error) => Err(error).context("read optional Git configuration input"),
    }
}

fn verify_repo_root(root: &Path) -> anyhow::Result<()> {
    let actual = fs::canonicalize(git_text(root, &["rev-parse", "--show-toplevel"])? )
        .context("canonicalize Git top-level")?;
    let expected = fs::canonicalize(root).context("canonicalize requested repository root")?;
    if actual != expected {
        bail!("requested repository root does not match Git top-level");
    }
    Ok(())
}

fn reject_output_inside_repository(root: &Path, output: &Path) -> anyhow::Result<()> {
    let root = fs::canonicalize(root).context("canonicalize repository root")?;
    let absolute = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    let parent = absolute.parent().unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    let candidate = parent.join(
        absolute.file_name().context("Git worktree-state output must identify a file")?,
    );
    if candidate.starts_with(&root) {
        bail!("Git worktree-state output must be outside the captured repository");
    }
    Ok(())
}

fn normalize_relative_path(value: &str) -> anyhow::Result<String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        bail!("Git tracked path must be non-empty and repository-relative");
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => parts.push(
                part.to_str().context("Git tracked path component is not valid UTF-8")?,
            ),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("Git tracked path escapes repository scope")
            }
        }
    }
    if parts.is_empty() {
        bail!("Git tracked path must identify an entry");
    }
    Ok(parts.join("/"))
}

fn git_text(root: &Path, args: &[&str]) -> anyhow::Result<String> {
    let output = git_output(root, args)?;
    if !output.status.success() {
        return bail_git("run Git command", output);
    }
    Ok(utf8_trimmed(&output.stdout, "Git output")?.to_string())
}

fn git_bytes(root: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let output = git_output(root, args)?;
    if !output.status.success() {
        return bail_git("run Git command", output);
    }
    Ok(output.stdout)
}

fn git_output(root: &Path, args: &[&str]) -> anyhow::Result<Output> {
    Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("execute git {}", args.join(" ")))
}

fn utf8_trimmed<'a>(bytes: &'a [u8], label: &str) -> anyhow::Result<&'a str> {
    Ok(std::str::from_utf8(bytes)
        .with_context(|| format!("{label} is not valid UTF-8"))?
        .trim())
}

fn bail_git<T>(context: &str, output: Output) -> anyhow::Result<T> {
    let stderr = String::from_utf8_lossy(&output.stderr);
    bail!("{context} failed with status {}: {}", output.status, stderr.trim())
}

fn validate_sha256(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_parser_detects_skip_worktree() {
        let values = parse_flag_sensor(b"S src/a.rs\0H src/b.rs\0", SensorKind::SkipWorktree).unwrap();
        assert_eq!(values.len(), 2);
        assert!(values[0].flagged);
        assert!(!values[1].flagged);
    }

    #[test]
    fn lowercase_sensor_detects_flag() {
        let values = parse_flag_sensor(b"h src/a.rs\0H src/b.rs\0", SensorKind::Lowercase).unwrap();
        assert!(values[0].flagged);
        assert!(!values[1].flagged);
    }

    #[test]
    fn mismatched_sensor_path_sets_fail_closed() {
        let skip = parse_flag_sensor(b"H a\0H b\0", SensorKind::SkipWorktree).unwrap();
        let assume = parse_flag_sensor(b"H a\0", SensorKind::Lowercase).unwrap();
        let fsmonitor = parse_flag_sensor(b"H a\0H b\0", SensorKind::Lowercase).unwrap();
        assert!(merge_flag_sensors(skip, assume, fsmonitor).is_err());
    }

    #[test]
    fn merged_flags_preserve_independent_bits() {
        let skip = parse_flag_sensor(b"S a\0H b\0", SensorKind::SkipWorktree).unwrap();
        let assume = parse_flag_sensor(b"H a\0h b\0", SensorKind::Lowercase).unwrap();
        let fsmonitor = parse_flag_sensor(b"h a\0H b\0", SensorKind::Lowercase).unwrap();
        let merged = merge_flag_sensors(skip, assume, fsmonitor).unwrap();
        assert!(merged[0].skip_worktree);
        assert!(!merged[0].assume_unchanged);
        assert!(merged[0].fsmonitor_valid);
        assert!(!merged[1].skip_worktree);
        assert!(merged[1].assume_unchanged);
        assert!(!merged[1].fsmonitor_valid);
    }
}
