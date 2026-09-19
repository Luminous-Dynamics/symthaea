use anyhow::Context;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::repository_snapshot_receipt::{SnapshotEntry, ValidatedSnapshotReceipt};

const DIFF_SCHEMA: &str = "symthaea.repository-source-diff.v1";
const DIFF_HASH_DOMAIN: &[u8] = b"symthaea.repository-source-diff.v1\0";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntryChangeKind {
    Added,
    Removed,
    Modified,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EntryField {
    SourceClass,
    Kind,
    Content,
    Size,
    Executable,
    IndexMode,
    IndexBlob,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EntryChange {
    pub path: String,
    pub kind: EntryChangeKind,
    pub changed_fields: Vec<EntryField>,
    pub source_bytes_changed: bool,
    pub index_state_changed: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SubjectField {
    GitHead,
    GitHeadTree,
    GitVersion,
    Scope,
    UnknownSurfaces,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RepositorySourceDiff {
    pub diff_id: String,
    pub schema: &'static str,
    pub base_snapshot_id: String,
    pub head_snapshot_id: String,
    pub identical: bool,
    pub source_bytes_changed: bool,
    pub index_state_changed: bool,
    pub subject_metadata_changed: bool,
    pub changed_subject_fields: Vec<SubjectField>,
    pub entry_changes: Vec<EntryChange>,
}

#[derive(Serialize)]
struct DiffIdentity<'a> {
    schema: &'static str,
    base_snapshot_id: &'a str,
    head_snapshot_id: &'a str,
    changed_subject_fields: &'a [SubjectField],
    entry_changes: &'a [EntryChange],
}

pub fn run(base_path: &Path, head_path: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let base = ValidatedSnapshotReceipt::load(base_path)?;
    let head = ValidatedSnapshotReceipt::load(head_path)?;
    let diff = compare(&base, &head)?;

    let mut rendered = serde_json::to_string_pretty(&diff)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create source-diff output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write repository source diff {}", path.display()))?;
        println!("Repository source diff written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub(crate) fn compare(
    base: &ValidatedSnapshotReceipt,
    head: &ValidatedSnapshotReceipt,
) -> anyhow::Result<RepositorySourceDiff> {
    let mut changed_subject_fields = BTreeSet::new();
    if base.git_head != head.git_head {
        changed_subject_fields.insert(SubjectField::GitHead);
    }
    if base.git_head_tree != head.git_head_tree {
        changed_subject_fields.insert(SubjectField::GitHeadTree);
    }
    if base.git_version != head.git_version {
        changed_subject_fields.insert(SubjectField::GitVersion);
    }
    if base.scope != head.scope {
        changed_subject_fields.insert(SubjectField::Scope);
    }
    if base.unknown_surfaces != head.unknown_surfaces {
        changed_subject_fields.insert(SubjectField::UnknownSurfaces);
    }

    let base_entries: BTreeMap<&str, &SnapshotEntry> = base
        .entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry))
        .collect();
    let head_entries: BTreeMap<&str, &SnapshotEntry> = head
        .entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry))
        .collect();

    let paths: BTreeSet<&str> = base_entries
        .keys()
        .chain(head_entries.keys())
        .copied()
        .collect();

    let mut entry_changes = Vec::new();
    for path in paths {
        match (base_entries.get(path), head_entries.get(path)) {
            (None, Some(head_entry)) => entry_changes.push(EntryChange {
                path: path.to_string(),
                kind: EntryChangeKind::Added,
                changed_fields: present_entry_fields(head_entry),
                source_bytes_changed: true,
                index_state_changed: has_index_identity(head_entry),
            }),
            (Some(base_entry), None) => entry_changes.push(EntryChange {
                path: path.to_string(),
                kind: EntryChangeKind::Removed,
                changed_fields: present_entry_fields(base_entry),
                source_bytes_changed: true,
                index_state_changed: has_index_identity(base_entry),
            }),
            (Some(base), Some(head)) if *base != *head => {
                entry_changes.push(compare_entry(base, head));
            }
            (Some(_), Some(_)) => {}
            (None, None) => unreachable!("path originates from union of base/head entry maps"),
        }
    }

    let source_bytes_changed = entry_changes.iter().any(|change| change.source_bytes_changed);
    let index_state_changed = entry_changes.iter().any(|change| change.index_state_changed);
    let changed_subject_fields: Vec<_> = changed_subject_fields.into_iter().collect();
    let subject_metadata_changed = !changed_subject_fields.is_empty();
    let identical = base.snapshot_id == head.snapshot_id;

    let identity = DiffIdentity {
        schema: DIFF_SCHEMA,
        base_snapshot_id: &base.snapshot_id,
        head_snapshot_id: &head.snapshot_id,
        changed_subject_fields: &changed_subject_fields,
        entry_changes: &entry_changes,
    };
    let bytes = serde_json::to_vec(&identity).context("serialize repository source diff identity")?;
    let diff_id = domain_sha256(DIFF_HASH_DOMAIN, &bytes);

    Ok(RepositorySourceDiff {
        diff_id,
        schema: DIFF_SCHEMA,
        base_snapshot_id: base.snapshot_id.clone(),
        head_snapshot_id: head.snapshot_id.clone(),
        identical,
        source_bytes_changed,
        index_state_changed,
        subject_metadata_changed,
        changed_subject_fields,
        entry_changes,
    })
}

fn compare_entry(base: &SnapshotEntry, head: &SnapshotEntry) -> EntryChange {
    let mut changed = BTreeSet::new();
    if base.source_class != head.source_class {
        changed.insert(EntryField::SourceClass);
    }
    if base.kind != head.kind {
        changed.insert(EntryField::Kind);
    }
    if base.content_sha256 != head.content_sha256 {
        changed.insert(EntryField::Content);
    }
    if base.size_bytes != head.size_bytes {
        changed.insert(EntryField::Size);
    }
    if base.executable != head.executable {
        changed.insert(EntryField::Executable);
    }
    if base.index_mode != head.index_mode {
        changed.insert(EntryField::IndexMode);
    }
    if base.index_blob != head.index_blob {
        changed.insert(EntryField::IndexBlob);
    }

    let source_bytes_changed = changed.contains(&EntryField::SourceClass)
        || changed.contains(&EntryField::Kind)
        || changed.contains(&EntryField::Content)
        || changed.contains(&EntryField::Size)
        || changed.contains(&EntryField::Executable);
    let index_state_changed = changed.contains(&EntryField::SourceClass)
        || changed.contains(&EntryField::IndexMode)
        || changed.contains(&EntryField::IndexBlob);

    EntryChange {
        path: base.path.clone(),
        kind: EntryChangeKind::Modified,
        changed_fields: changed.into_iter().collect(),
        source_bytes_changed,
        index_state_changed,
    }
}

fn has_index_identity(entry: &SnapshotEntry) -> bool {
    entry.index_mode.is_some() || entry.index_blob.is_some()
}

fn present_entry_fields(entry: &SnapshotEntry) -> Vec<EntryField> {
    let mut fields = vec![EntryField::SourceClass, EntryField::Kind];
    if entry.content_sha256.is_some() {
        fields.push(EntryField::Content);
    }
    if entry.size_bytes.is_some() {
        fields.push(EntryField::Size);
    }
    if entry.executable.is_some() {
        fields.push(EntryField::Executable);
    }
    if entry.index_mode.is_some() {
        fields.push(EntryField::IndexMode);
    }
    if entry.index_blob.is_some() {
        fields.push(EntryField::IndexBlob);
    }
    fields
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
    use crate::repository_snapshot_receipt::{
        EntryKind, SnapshotScope, SourceClass, UnknownSurface,
    };

    fn receipt(id_byte: char) -> ValidatedSnapshotReceipt {
        ValidatedSnapshotReceipt {
            snapshot_id: id_byte.to_string().repeat(64),
            schema: "symthaea.repository-source-snapshot.v1".into(),
            git_head: "a".repeat(40),
            git_head_tree: "b".repeat(40),
            git_version: "git version 2.55.0".into(),
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
                size_bytes: Some(10),
                executable: Some(false),
                index_mode: Some("100644".into()),
                index_blob: Some("d".repeat(40)),
            }],
            unknown_surfaces: vec![UnknownSurface {
                kind: "environment_network_and_external_build_inputs_not_captured".into(),
                subject: None,
            }],
        }
    }

    #[test]
    fn identical_receipts_have_no_changes() {
        let base = receipt('1');
        let head = base.clone();
        let diff = compare(&base, &head).unwrap();
        assert!(diff.identical);
        assert!(diff.entry_changes.is_empty());
        assert!(!diff.source_bytes_changed);
        assert!(!diff.index_state_changed);
        assert!(!diff.subject_metadata_changed);
    }

    #[test]
    fn content_change_is_source_not_index_change() {
        let base = receipt('1');
        let mut head = base.clone();
        head.snapshot_id = "2".repeat(64);
        head.entries[0].content_sha256 = Some("e".repeat(64));
        let diff = compare(&base, &head).unwrap();
        assert!(!diff.identical);
        assert!(diff.source_bytes_changed);
        assert!(!diff.index_state_changed);
        assert_eq!(diff.entry_changes[0].changed_fields, vec![EntryField::Content]);
    }

    #[test]
    fn staged_index_change_is_visible_without_source_byte_change() {
        let base = receipt('1');
        let mut head = base.clone();
        head.snapshot_id = "2".repeat(64);
        head.entries[0].index_blob = Some("e".repeat(40));
        let diff = compare(&base, &head).unwrap();
        assert!(!diff.source_bytes_changed);
        assert!(diff.index_state_changed);
        assert_eq!(diff.entry_changes[0].changed_fields, vec![EntryField::IndexBlob]);
    }

    #[test]
    fn untracked_addition_is_source_only() {
        let base = receipt('1');
        let mut head = base.clone();
        head.snapshot_id = "2".repeat(64);
        head.entries.push(SnapshotEntry {
            path: "scratch.rs".into(),
            source_class: SourceClass::UntrackedNonIgnored,
            kind: EntryKind::File,
            content_sha256: Some("e".repeat(64)),
            size_bytes: Some(4),
            executable: Some(false),
            index_mode: None,
            index_blob: None,
        });
        head.entries.sort_by(|a, b| a.path.cmp(&b.path));
        let diff = compare(&base, &head).unwrap();
        assert!(diff.source_bytes_changed);
        assert!(!diff.index_state_changed);
        let change = diff
            .entry_changes
            .iter()
            .find(|change| change.path == "scratch.rs")
            .unwrap();
        assert!(!change.changed_fields.contains(&EntryField::IndexMode));
        assert!(!change.changed_fields.contains(&EntryField::IndexBlob));
    }

    #[test]
    fn head_only_change_is_subject_metadata_change() {
        let base = receipt('1');
        let mut head = base.clone();
        head.snapshot_id = "2".repeat(64);
        head.git_head = "f".repeat(40);
        let diff = compare(&base, &head).unwrap();
        assert!(diff.entry_changes.is_empty());
        assert!(!diff.source_bytes_changed);
        assert!(!diff.index_state_changed);
        assert!(diff.subject_metadata_changed);
        assert_eq!(diff.changed_subject_fields, vec![SubjectField::GitHead]);
    }
}
