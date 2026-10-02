use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

const SNAPSHOT_SCHEMA: &str = "symthaea.repository-source-snapshot.v1";
const SNAPSHOT_HASH_DOMAIN: &[u8] = b"symthaea.repository-source-snapshot.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ValidatedSnapshotReceipt {
    pub snapshot_id: String,
    pub schema: String,
    pub git_head: String,
    pub git_head_tree: String,
    pub git_version: String,
    pub scope: SnapshotScope,
    pub entries: Vec<SnapshotEntry>,
    pub unknown_surfaces: Vec<UnknownSurface>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SnapshotScope {
    pub tracked_worktree: bool,
    pub untracked_non_ignored: bool,
    pub explicit_ignored_inputs: Vec<String>,
    pub ignored_policy: String,
    pub symlink_policy: String,
    pub submodule_policy: String,
    pub external_input_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SourceClass {
    Tracked,
    UntrackedNonIgnored,
    ExplicitIgnored,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EntryKind {
    File,
    Symlink,
    MissingTracked,
    Gitlink,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SnapshotEntry {
    pub path: String,
    pub source_class: SourceClass,
    pub kind: EntryKind,
    pub content_sha256: Option<String>,
    pub size_bytes: Option<u64>,
    pub executable: Option<bool>,
    pub index_mode: Option<String>,
    pub index_blob: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnknownSurface {
    pub kind: String,
    pub subject: Option<String>,
}

#[derive(Serialize)]
struct SnapshotPayload<'a> {
    schema: &'a str,
    git_head: &'a str,
    git_head_tree: &'a str,
    git_version: &'a str,
    scope: &'a SnapshotScope,
    entries: &'a [SnapshotEntry],
    unknown_surfaces: &'a [UnknownSurface],
}

impl ValidatedSnapshotReceipt {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read repository source snapshot {}", path.display()))?;
        let mut receipt: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse repository source snapshot {}", path.display()))?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub(crate) fn validate(&mut self) -> anyhow::Result<()> {
        if self.schema != SNAPSHOT_SCHEMA {
            bail!("unsupported repository source snapshot schema: {}", self.schema);
        }
        validate_sha256("snapshot_id", &self.snapshot_id)?;
        self.snapshot_id.make_ascii_lowercase();
        validate_git_object_id("git_head", &self.git_head)?;
        validate_git_object_id("git_head_tree", &self.git_head_tree)?;
        self.git_head.make_ascii_lowercase();
        self.git_head_tree.make_ascii_lowercase();
        if self.git_version.trim().is_empty() {
            bail!("git_version must not be empty");
        }

        validate_strictly_sorted_unique(
            "scope.explicit_ignored_inputs",
            &self.scope.explicit_ignored_inputs,
        )?;
        for path in &self.scope.explicit_ignored_inputs {
            validate_relative_path(path)?;
        }

        for entry in &mut self.entries {
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
        validate_entry_order(&self.entries)?;
        validate_unknown_order(&self.unknown_surfaces)?;

        let computed = self.computed_snapshot_id()?;
        if computed != self.snapshot_id {
            bail!(
                "stored repository snapshot identity mismatch: declared {}, computed {}",
                self.snapshot_id,
                computed
            );
        }
        Ok(())
    }

    pub(crate) fn explicit_ignored_paths(&self) -> Vec<PathBuf> {
        self.scope
            .explicit_ignored_inputs
            .iter()
            .map(PathBuf::from)
            .collect()
    }

    pub(crate) fn computed_snapshot_id(&self) -> anyhow::Result<String> {
        let payload = SnapshotPayload {
            schema: &self.schema,
            git_head: &self.git_head,
            git_head_tree: &self.git_head_tree,
            git_version: &self.git_version,
            scope: &self.scope,
            entries: &self.entries,
            unknown_surfaces: &self.unknown_surfaces,
        };
        let bytes = serde_json::to_vec(&payload).context("serialize stored snapshot payload")?;
        Ok(domain_sha256(SNAPSHOT_HASH_DOMAIN, &bytes))
    }
}

fn validate_entry_shape(entry: &SnapshotEntry) -> anyhow::Result<()> {
    let tracked = entry.source_class == SourceClass::Tracked;
    if tracked != entry.index_mode.is_some() || tracked != entry.index_blob.is_some() {
        bail!(
            "entry {} has inconsistent tracked/index identity fields",
            entry.path
        );
    }

    match entry.kind {
        EntryKind::File | EntryKind::Symlink => {
            if entry.content_sha256.is_none() || entry.size_bytes.is_none() {
                bail!("entry {} is missing content identity", entry.path);
            }
        }
        EntryKind::MissingTracked => {
            if !tracked
                || entry.content_sha256.is_some()
                || entry.size_bytes.is_some()
                || entry.executable.is_some()
            {
                bail!("entry {} has invalid missing-tracked shape", entry.path);
            }
        }
        EntryKind::Gitlink => {
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

fn validate_entry_order(entries: &[SnapshotEntry]) -> anyhow::Result<()> {
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

fn validate_unknown_order(values: &[UnknownSurface]) -> anyhow::Result<()> {
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

    fn stored() -> ValidatedSnapshotReceipt {
        let mut snapshot = ValidatedSnapshotReceipt {
            snapshot_id: String::new(),
            schema: SNAPSHOT_SCHEMA.into(),
            git_head: "a".repeat(40),
            git_head_tree: "b".repeat(40),
            git_version: "git version 2.50.0".into(),
            scope: SnapshotScope {
                tracked_worktree: true,
                untracked_non_ignored: true,
                explicit_ignored_inputs: vec![],
                ignored_policy: "git_exclude_standard_plus_explicit_ignored_inputs".into(),
                symlink_policy: "hash_link_target_bytes_do_not_follow_referent".into(),
                submodule_policy: "record_gitlink_index_identity_mark_contents_unknown".into(),
                external_input_policy: "not_captured".into(),
            },
            entries: vec![SnapshotEntry {
                path: "src/lib.rs".into(),
                source_class: SourceClass::Tracked,
                kind: EntryKind::File,
                content_sha256: Some("c".repeat(64)),
                size_bytes: Some(12),
                executable: Some(false),
                index_mode: Some("100644".into()),
                index_blob: Some("d".repeat(40)),
            }],
            unknown_surfaces: vec![UnknownSurface {
                kind: "environment_network_and_external_build_inputs_not_captured".into(),
                subject: None,
            }],
        };
        snapshot.snapshot_id = snapshot.computed_snapshot_id().unwrap();
        snapshot
    }

    #[test]
    fn validates_self_consistent_snapshot() {
        let mut snapshot = stored();
        snapshot.validate().unwrap();
    }

    #[test]
    fn one_byte_identity_change_breaks_stored_receipt() {
        let mut snapshot = stored();
        snapshot.entries[0].content_sha256 = Some("e".repeat(64));
        assert!(snapshot.validate().is_err());
    }

    #[test]
    fn rejects_noncanonical_entry_order() {
        let mut snapshot = stored();
        let mut other = snapshot.entries[0].clone();
        other.path = "a.rs".into();
        snapshot.entries.push(other);
        snapshot.snapshot_id = snapshot.computed_snapshot_id().unwrap();
        assert!(snapshot.validate().is_err());
    }

    #[test]
    fn rejects_invalid_gitlink_shape() {
        let mut snapshot = stored();
        snapshot.entries[0].kind = EntryKind::Gitlink;
        snapshot.snapshot_id = snapshot.computed_snapshot_id().unwrap();
        assert!(snapshot.validate().is_err());
    }
}
