use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

use crate::repository_snapshot;

const SNAPSHOT_SCHEMA: &str = "symthaea.repository-source-snapshot.v1";
const SNAPSHOT_HASH_DOMAIN: &[u8] = b"symthaea.repository-source-snapshot.v1\0";
const VERIFY_SCHEMA: &str = "symthaea.repository-source-verification.v1";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredSnapshot {
    snapshot_id: String,
    schema: String,
    git_head: String,
    git_head_tree: String,
    git_version: String,
    scope: StoredScope,
    entries: Vec<StoredEntry>,
    unknown_surfaces: Vec<StoredUnknownSurface>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredScope {
    tracked_worktree: bool,
    untracked_non_ignored: bool,
    explicit_ignored_inputs: Vec<String>,
    ignored_policy: String,
    symlink_policy: String,
    submodule_policy: String,
    external_input_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum StoredSourceClass {
    Tracked,
    UntrackedNonIgnored,
    ExplicitIgnored,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum StoredEntryKind {
    File,
    Symlink,
    MissingTracked,
    Gitlink,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredEntry {
    path: String,
    source_class: StoredSourceClass,
    kind: StoredEntryKind,
    content_sha256: Option<String>,
    size_bytes: Option<u64>,
    executable: Option<bool>,
    index_mode: Option<String>,
    index_blob: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct StoredUnknownSurface {
    kind: String,
    subject: Option<String>,
}

#[derive(Serialize)]
struct StoredPayload<'a> {
    schema: &'a str,
    git_head: &'a str,
    git_head_tree: &'a str,
    git_version: &'a str,
    scope: &'a StoredScope,
    entries: &'a [StoredEntry],
    unknown_surfaces: &'a [StoredUnknownSurface],
}

#[derive(Debug, Serialize)]
struct VerificationReport {
    schema: &'static str,
    expected_snapshot_id: String,
    current_snapshot_id: String,
    matches: bool,
    expected_git_head: String,
    current_git_head: String,
    expected_entry_count: usize,
    current_entry_count: usize,
    expected_unknown_surface_count: usize,
    current_unknown_surface_count: usize,
}

pub fn run(root: &Path, snapshot_path: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(snapshot_path)
        .with_context(|| format!("read repository source snapshot {}", snapshot_path.display()))?;
    let mut stored: StoredSnapshot = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse repository source snapshot {}", snapshot_path.display()))?;
    validate_stored_snapshot(&mut stored)?;

    let include_ignored = stored
        .scope
        .explicit_ignored_inputs
        .iter()
        .map(PathBuf::from)
        .collect();
    let current = repository_snapshot::build_snapshot(root, include_ignored)?;

    let matches = stored.snapshot_id == current.snapshot_id;
    let report = VerificationReport {
        schema: VERIFY_SCHEMA,
        expected_snapshot_id: stored.snapshot_id.clone(),
        current_snapshot_id: current.snapshot_id.clone(),
        matches,
        expected_git_head: stored.git_head.clone(),
        current_git_head: current.payload.git_head.clone(),
        expected_entry_count: stored.entries.len(),
        current_entry_count: current.payload.entries.len(),
        expected_unknown_surface_count: stored.unknown_surfaces.len(),
        current_unknown_surface_count: current.payload.unknown_surfaces.len(),
    };

    let mut rendered = serde_json::to_string_pretty(&report)?;
    rendered.push('\n');
    print!("{rendered}");

    if !matches {
        bail!(
            "repository source subject drift: expected {}, current {}",
            report.expected_snapshot_id,
            report.current_snapshot_id
        );
    }
    Ok(())
}

fn validate_stored_snapshot(snapshot: &mut StoredSnapshot) -> anyhow::Result<()> {
    if snapshot.schema != SNAPSHOT_SCHEMA {
        bail!("unsupported repository source snapshot schema: {}", snapshot.schema);
    }
    validate_sha256("snapshot_id", &snapshot.snapshot_id)?;
    snapshot.snapshot_id.make_ascii_lowercase();
    validate_git_object_id("git_head", &snapshot.git_head)?;
    validate_git_object_id("git_head_tree", &snapshot.git_head_tree)?;
    snapshot.git_head.make_ascii_lowercase();
    snapshot.git_head_tree.make_ascii_lowercase();
    if snapshot.git_version.trim().is_empty() {
        bail!("git_version must not be empty");
    }

    validate_strictly_sorted_unique(
        "scope.explicit_ignored_inputs",
        &snapshot.scope.explicit_ignored_inputs,
    )?;
    for path in &snapshot.scope.explicit_ignored_inputs {
        validate_relative_path(path)?;
    }

    for entry in &mut snapshot.entries {
        validate_relative_path(&entry.path)?;
        if let Some(digest) = &mut entry.content_sha256 {
            validate_sha256("entry.content_sha256", digest)?;
            digest.make_ascii_lowercase();
        }
        if let Some(mode) = &entry.index_mode {
            validate_git_mode(mode)?;
        }
        if let Some(object) = &mut entry.index_blob {
            validate_git_object_id("entry.index_blob", object)?;
            object.make_ascii_lowercase();
        }
        validate_entry_shape(entry)?;
    }
    validate_entry_order(&snapshot.entries)?;
    validate_unknown_order(&snapshot.unknown_surfaces)?;

    let payload = StoredPayload {
        schema: &snapshot.schema,
        git_head: &snapshot.git_head,
        git_head_tree: &snapshot.git_head_tree,
        git_version: &snapshot.git_version,
        scope: &snapshot.scope,
        entries: &snapshot.entries,
        unknown_surfaces: &snapshot.unknown_surfaces,
    };
    let bytes = serde_json::to_vec(&payload).context("serialize stored snapshot payload")?;
    let computed = domain_sha256(SNAPSHOT_HASH_DOMAIN, &bytes);
    if computed != snapshot.snapshot_id {
        bail!(
            "stored repository snapshot identity mismatch: declared {}, computed {}",
            snapshot.snapshot_id,
            computed
        );
    }
    Ok(())
}

fn validate_entry_shape(entry: &StoredEntry) -> anyhow::Result<()> {
    let tracked = entry.source_class == StoredSourceClass::Tracked;
    if tracked != entry.index_mode.is_some() || tracked != entry.index_blob.is_some() {
        bail!(
            "entry {} has inconsistent tracked/index identity fields",
            entry.path
        );
    }

    match entry.kind {
        StoredEntryKind::File | StoredEntryKind::Symlink => {
            if entry.content_sha256.is_none() || entry.size_bytes.is_none() {
                bail!("entry {} is missing content identity", entry.path);
            }
        }
        StoredEntryKind::MissingTracked => {
            if !tracked
                || entry.content_sha256.is_some()
                || entry.size_bytes.is_some()
                || entry.executable.is_some()
            {
                bail!("entry {} has invalid missing-tracked shape", entry.path);
            }
        }
        StoredEntryKind::Gitlink => {
            if !tracked
                || entry.index_mode.as_deref() != Some("160000")
                || entry.content_sha256.is_some()
                || entry.size_bytes.is_some()
                || entry.executable.is_some()
            {
                bail!("entry {} has invalid gitlink shape", entry.path);
            }
        }
    }
    Ok(())
}

fn validate_entry_order(entries: &[StoredEntry]) -> anyhow::Result<()> {
    for pair in entries.windows(2) {
        if pair[0].path >= pair[1].path {
            bail!(
                "snapshot entries must be strictly sorted and unique: {} then {}",
                pair[0].path,
                pair[1].path
            );
        }
    }
    Ok(())
}

fn validate_unknown_order(values: &[StoredUnknownSurface]) -> anyhow::Result<()> {
    for value in values {
        if value.kind.trim().is_empty() {
            bail!("unknown surface kind must not be empty");
        }
        if let Some(subject) = &value.subject {
            validate_relative_path(subject)?;
        }
    }
    for pair in values.windows(2) {
        if pair[0] >= pair[1] {
            bail!("unknown surfaces must be strictly sorted and unique");
        }
    }
    Ok(())
}

fn validate_strictly_sorted_unique(name: &str, values: &[String]) -> anyhow::Result<()> {
    for pair in values.windows(2) {
        if pair[0] >= pair[1] {
            bail!("{name} must be strictly sorted and unique");
        }
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> anyhow::Result<()> {
    if value.is_empty() {
        bail!("repository-relative path must not be empty");
    }
    let path = Path::new(value);
    if path.is_absolute() {
        bail!("repository-relative path must not be absolute: {value}");
    }
    let mut normal_count = 0usize;
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .with_context(|| format!("path is not valid UTF-8: {value}"))?;
                if part.is_empty() {
                    bail!("path contains an empty component: {value}");
                }
                normal_count += 1;
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("path escapes repository scope: {value}")
            }
        }
    }
    if normal_count == 0 {
        bail!("repository-relative path must identify an entry: {value}");
    }
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn validate_git_object_id(name: &str, value: &str) -> anyhow::Result<()> {
    if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 40- or 64-character Git object id");
    }
    Ok(())
}

fn validate_git_mode(value: &str) -> anyhow::Result<()> {
    if value.len() != 6 || !value.bytes().all(|byte| matches!(byte, b'0'..=b'7')) {
        bail!("invalid Git index mode: {value}");
    }
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
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

    fn stored() -> StoredSnapshot {
        let mut snapshot = StoredSnapshot {
            snapshot_id: String::new(),
            schema: SNAPSHOT_SCHEMA.into(),
            git_head: "a".repeat(40),
            git_head_tree: "b".repeat(40),
            git_version: "git version 2.50.0".into(),
            scope: StoredScope {
                tracked_worktree: true,
                untracked_non_ignored: true,
                explicit_ignored_inputs: vec![],
                ignored_policy: "git_exclude_standard_plus_explicit_ignored_inputs".into(),
                symlink_policy: "hash_link_target_bytes_do_not_follow_referent".into(),
                submodule_policy: "record_gitlink_index_identity_mark_contents_unknown".into(),
                external_input_policy: "not_captured".into(),
            },
            entries: vec![StoredEntry {
                path: "src/lib.rs".into(),
                source_class: StoredSourceClass::Tracked,
                kind: StoredEntryKind::File,
                content_sha256: Some("c".repeat(64)),
                size_bytes: Some(12),
                executable: Some(false),
                index_mode: Some("100644".into()),
                index_blob: Some("d".repeat(40)),
            }],
            unknown_surfaces: vec![StoredUnknownSurface {
                kind: "environment_network_and_external_build_inputs_not_captured".into(),
                subject: None,
            }],
        };
        snapshot.snapshot_id = stored_payload_id(&snapshot);
        snapshot
    }

    fn stored_payload_id(snapshot: &StoredSnapshot) -> String {
        let payload = StoredPayload {
            schema: &snapshot.schema,
            git_head: &snapshot.git_head,
            git_head_tree: &snapshot.git_head_tree,
            git_version: &snapshot.git_version,
            scope: &snapshot.scope,
            entries: &snapshot.entries,
            unknown_surfaces: &snapshot.unknown_surfaces,
        };
        domain_sha256(SNAPSHOT_HASH_DOMAIN, &serde_json::to_vec(&payload).unwrap())
    }

    #[test]
    fn validates_self_consistent_snapshot() {
        let mut snapshot = stored();
        validate_stored_snapshot(&mut snapshot).unwrap();
    }

    #[test]
    fn one_byte_identity_change_breaks_stored_receipt() {
        let mut snapshot = stored();
        snapshot.entries[0].content_sha256 = Some("e".repeat(64));
        assert!(validate_stored_snapshot(&mut snapshot).is_err());
    }

    #[test]
    fn rejects_noncanonical_entry_order() {
        let mut snapshot = stored();
        let mut other = snapshot.entries[0].clone();
        other.path = "a.rs".into();
        snapshot.entries.push(other);
        snapshot.snapshot_id = stored_payload_id(&snapshot);
        assert!(validate_stored_snapshot(&mut snapshot).is_err());
    }

    #[test]
    fn rejects_invalid_gitlink_shape() {
        let mut snapshot = stored();
        snapshot.entries[0].kind = StoredEntryKind::Gitlink;
        snapshot.snapshot_id = stored_payload_id(&snapshot);
        assert!(validate_stored_snapshot(&mut snapshot).is_err());
    }
}
