use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};

const SNAPSHOT_SCHEMA: &str = "symthaea.repository-source-snapshot.v1";
const SNAPSHOT_HASH_DOMAIN: &[u8] = b"symthaea.repository-source-snapshot.v1\0";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RepositorySourceSnapshot {
    pub snapshot_id: String,
    #[serde(flatten)]
    pub payload: RepositorySourceSnapshotPayload,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RepositorySourceSnapshotPayload {
    pub schema: &'static str,
    pub git_head: String,
    pub git_head_tree: String,
    pub git_version: String,
    pub scope: SnapshotScope,
    pub entries: Vec<SnapshotEntry>,
    pub unknown_surfaces: Vec<UnknownSurface>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SnapshotScope {
    pub tracked_worktree: bool,
    pub untracked_non_ignored: bool,
    pub explicit_ignored_inputs: Vec<String>,
    pub ignored_policy: &'static str,
    pub symlink_policy: &'static str,
    pub submodule_policy: &'static str,
    pub external_input_policy: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SourceClass {
    Tracked,
    UntrackedNonIgnored,
    ExplicitIgnored,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    File,
    Symlink,
    MissingTracked,
    Gitlink,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SnapshotEntry {
    pub path: String,
    pub source_class: SourceClass,
    pub kind: EntryKind,
    pub content_sha256: Option<String>,
    pub size_bytes: Option<u64>,
    pub executable: Option<bool>,
    pub index_mode: Option<String>,
    pub index_blob: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnknownSurface {
    pub kind: &'static str,
    pub subject: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexEntry {
    mode: String,
    blob: String,
    path: String,
}

pub fn run(
    root: &Path,
    include_ignored: Vec<PathBuf>,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    if let Some(path) = output.as_deref() {
        ensure_output_outside_repository(root, path)?;
    }

    let snapshot = build_snapshot(root, include_ignored)?;
    let mut rendered = serde_json::to_string_pretty(&snapshot)?;
    rendered.push('\n');

    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create snapshot output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write repository source snapshot {}", path.display()))?;
        println!("Repository source snapshot written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub fn build_snapshot(
    root: &Path,
    include_ignored: Vec<PathBuf>,
) -> anyhow::Result<RepositorySourceSnapshot> {
    verify_repo_root(root)?;

    let git_head = git_text(root, &["rev-parse", "HEAD"])?;
    let git_head_tree = git_text(root, &["rev-parse", "HEAD^{tree}"])?;
    let git_version = git_text(root, &["--version"])?;

    let tracked = parse_index_entries(&git_bytes(root, &["ls-files", "--stage", "-z"])?)?;
    let untracked = parse_nul_paths(&git_bytes(
        root,
        &["ls-files", "--others", "--exclude-standard", "-z"],
    )?)?;

    let mut explicit_ignored = Vec::with_capacity(include_ignored.len());
    for path in include_ignored {
        let normalized = normalize_relative_path(&path)?;
        if normalized == ".git" || normalized.starts_with(".git/") {
            bail!("explicit ignored input must not address .git internals: {normalized}");
        }
        if !git_check_ignored(root, &normalized)? {
            bail!("explicit ignored input is not ignored by Git: {normalized}");
        }
        explicit_ignored.push(normalized);
    }
    explicit_ignored.sort();
    explicit_ignored.dedup();

    let mut entries = BTreeMap::<String, SnapshotEntry>::new();
    let mut unknown_surfaces = BTreeSet::<UnknownSurface>::new();

    for index in tracked {
        if index.mode == "160000" {
            insert_unique(
                &mut entries,
                SnapshotEntry {
                    path: index.path.clone(),
                    source_class: SourceClass::Tracked,
                    kind: EntryKind::Gitlink,
                    content_sha256: None,
                    size_bytes: None,
                    executable: None,
                    index_mode: Some(index.mode),
                    index_blob: Some(index.blob),
                },
            )?;
            unknown_surfaces.insert(UnknownSurface {
                kind: "gitlink_contents_not_captured",
                subject: Some(index.path),
            });
            continue;
        }

        let entry = snapshot_path(
            root,
            &index.path,
            SourceClass::Tracked,
            Some(index.mode),
            Some(index.blob),
            true,
        )?;
        insert_with_unknowns(&mut entries, &mut unknown_surfaces, entry)?;
    }

    for path in untracked {
        let normalized = normalize_relative_path(Path::new(&path))?;
        let entry = snapshot_path(
            root,
            &normalized,
            SourceClass::UntrackedNonIgnored,
            None,
            None,
            false,
        )?;
        insert_with_unknowns(&mut entries, &mut unknown_surfaces, entry)?;
    }

    for path in &explicit_ignored {
        let entry = snapshot_path(
            root,
            path,
            SourceClass::ExplicitIgnored,
            None,
            None,
            false,
        )?;
        insert_with_unknowns(&mut entries, &mut unknown_surfaces, entry)?;
    }

    unknown_surfaces.insert(UnknownSurface {
        kind: "ignored_inputs_outside_explicit_set_not_captured",
        subject: None,
    });
    unknown_surfaces.insert(UnknownSurface {
        kind: "git_ignore_sources_outside_tracked_tree_not_bound",
        subject: None,
    });
    unknown_surfaces.insert(UnknownSurface {
        kind: "environment_network_and_external_build_inputs_not_captured",
        subject: None,
    });

    let payload = RepositorySourceSnapshotPayload {
        schema: SNAPSHOT_SCHEMA,
        git_head,
        git_head_tree,
        git_version,
        scope: SnapshotScope {
            tracked_worktree: true,
            untracked_non_ignored: true,
            explicit_ignored_inputs: explicit_ignored,
            ignored_policy: "git_exclude_standard_plus_explicit_ignored_inputs",
            symlink_policy: "hash_link_target_bytes_do_not_follow_referent",
            submodule_policy: "record_gitlink_index_identity_mark_contents_unknown",
            external_input_policy: "not_captured",
        },
        entries: entries.into_values().collect(),
        unknown_surfaces: unknown_surfaces.into_iter().collect(),
    };

    let payload_bytes =
        serde_json::to_vec(&payload).context("serialize repository source snapshot payload")?;
    let snapshot_id = domain_sha256(SNAPSHOT_HASH_DOMAIN, &payload_bytes);

    Ok(RepositorySourceSnapshot {
        snapshot_id,
        payload,
    })
}

fn verify_repo_root(root: &Path) -> anyhow::Result<()> {
    let actual = git_text(root, &["rev-parse", "--show-toplevel"])?;
    let actual = fs::canonicalize(&actual)
        .with_context(|| format!("canonicalize Git top-level {actual}"))?;
    let expected = fs::canonicalize(root)
        .with_context(|| format!("canonicalize requested repository root {}", root.display()))?;
    if actual != expected {
        bail!(
            "repository root mismatch: requested {}, Git reports {}",
            expected.display(),
            actual.display()
        );
    }
    Ok(())
}

fn ensure_output_outside_repository(root: &Path, output: &Path) -> anyhow::Result<()> {
    let root = fs::canonicalize(root)
        .with_context(|| format!("canonicalize repository root {}", root.display()))?;
    let absolute = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()
            .context("read current directory for snapshot output")?
            .join(output)
    };
    let resolved = resolve_existing_prefix(&absolute)?;
    if resolved.starts_with(&root) {
        bail!(
            "repository snapshot output must be outside the captured repository; {} resolves beneath {}",
            output.display(),
            root.display()
        );
    }
    Ok(())
}

fn resolve_existing_prefix(path: &Path) -> anyhow::Result<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .with_context(|| format!("cannot resolve output path {}", path.display()))?
            .to_os_string();
        suffix.push(name);
        existing.pop();
    }
    let mut resolved = fs::canonicalize(&existing)
        .with_context(|| format!("canonicalize existing output ancestor {}", existing.display()))?;
    for component in suffix.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn parse_index_entries(bytes: &[u8]) -> anyhow::Result<Vec<IndexEntry>> {
    let mut entries = Vec::new();
    for raw in bytes.split(|byte| *byte == 0).filter(|record| !record.is_empty()) {
        let tab = raw
            .iter()
            .position(|byte| *byte == b'\t')
            .context("git ls-files --stage record missing tab separator")?;
        let meta = std::str::from_utf8(&raw[..tab])
            .context("git index metadata is not valid UTF-8")?;
        let path = std::str::from_utf8(&raw[tab + 1..])
            .context("repository path is not valid UTF-8")?;
        let mut fields = meta.split_whitespace();
        let mode = fields.next().context("index record missing mode")?;
        let blob = fields.next().context("index record missing object id")?;
        let stage = fields.next().context("index record missing stage")?;
        if fields.next().is_some() {
            bail!("unexpected extra fields in index record for {path}");
        }
        if stage != "0" {
            bail!("unmerged index stage {stage} for {path}; source snapshot fails closed");
        }
        validate_git_mode(mode)?;
        validate_git_object_id(blob)?;
        let path = normalize_relative_path(Path::new(path))?;
        entries.push(IndexEntry {
            mode: mode.to_string(),
            blob: blob.to_ascii_lowercase(),
            path,
        });
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    for pair in entries.windows(2) {
        if pair[0].path == pair[1].path {
            bail!("duplicate tracked index path: {}", pair[0].path);
        }
    }
    Ok(entries)
}

fn parse_nul_paths(bytes: &[u8]) -> anyhow::Result<Vec<String>> {
    let mut paths = Vec::new();
    for raw in bytes.split(|byte| *byte == 0).filter(|record| !record.is_empty()) {
        let path = std::str::from_utf8(raw).context("repository path is not valid UTF-8")?;
        paths.push(normalize_relative_path(Path::new(path))?);
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn snapshot_path(
    root: &Path,
    relative: &str,
    source_class: SourceClass,
    index_mode: Option<String>,
    index_blob: Option<String>,
    allow_missing: bool,
) -> anyhow::Result<SnapshotEntry> {
    validate_no_symlink_ancestors(root, relative)?;
    let path = root.join(relative);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && allow_missing => {
            return Ok(SnapshotEntry {
                path: relative.to_string(),
                source_class,
                kind: EntryKind::MissingTracked,
                content_sha256: None,
                size_bytes: None,
                executable: None,
                index_mode,
                index_blob,
            });
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read worktree metadata for {relative}"));
        }
    };

    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        let target = fs::read_link(&path)
            .with_context(|| format!("read symlink target for {relative}"))?;
        let target = target
            .to_str()
            .with_context(|| format!("symlink target for {relative} is not valid UTF-8"))?;
        let bytes = target.as_bytes();
        return Ok(SnapshotEntry {
            path: relative.to_string(),
            source_class,
            kind: EntryKind::Symlink,
            content_sha256: Some(sha256(bytes)),
            size_bytes: Some(bytes.len() as u64),
            executable: None,
            index_mode,
            index_blob,
        });
    }

    if file_type.is_file() {
        let bytes = fs::read(&path).with_context(|| format!("read worktree file {relative}"))?;
        return Ok(SnapshotEntry {
            path: relative.to_string(),
            source_class,
            kind: EntryKind::File,
            content_sha256: Some(sha256(&bytes)),
            size_bytes: Some(bytes.len() as u64),
            executable: executable_bit(&metadata),
            index_mode,
            index_blob,
        });
    }

    bail!("unsupported worktree object type for {relative}; source snapshot fails closed")
}

fn insert_with_unknowns(
    entries: &mut BTreeMap<String, SnapshotEntry>,
    unknown_surfaces: &mut BTreeSet<UnknownSurface>,
    entry: SnapshotEntry,
) -> anyhow::Result<()> {
    if entry.kind == EntryKind::Symlink {
        unknown_surfaces.insert(UnknownSurface {
            kind: "symlink_referent_contents_not_captured",
            subject: Some(entry.path.clone()),
        });
    }
    insert_unique(entries, entry)
}

fn validate_no_symlink_ancestors(root: &Path, relative: &str) -> anyhow::Result<()> {
    let path = Path::new(relative);
    let mut current = root.to_path_buf();
    let components: Vec<_> = path.components().collect();
    for component in components.iter().take(components.len().saturating_sub(1)) {
        let Component::Normal(part) = component else {
            bail!("non-normal path component in {relative}");
        };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!(
                    "path {relative} traverses symlink ancestor {}; source snapshot will not follow it",
                    current.display()
                )
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect path ancestor {}", current.display()));
            }
        }
    }
    Ok(())
}

fn insert_unique(
    entries: &mut BTreeMap<String, SnapshotEntry>,
    entry: SnapshotEntry,
) -> anyhow::Result<()> {
    if let Some(existing) = entries.insert(entry.path.clone(), entry.clone()) {
        bail!(
            "snapshot path {} is present in multiple source classes ({:?}, {:?})",
            entry.path,
            existing.source_class,
            entry.source_class
        );
    }
    Ok(())
}

fn normalize_relative_path(path: &Path) -> anyhow::Result<String> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        bail!("snapshot path must be non-empty and repository-relative: {}", path.display());
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .with_context(|| format!("snapshot path is not valid UTF-8: {}", path.display()))?;
                if part.is_empty() {
                    bail!("snapshot path contains an empty component: {}", path.display());
                }
                parts.push(part);
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("snapshot path escapes the repository: {}", path.display())
            }
        }
    }
    if parts.is_empty() {
        bail!("snapshot path must identify a file: {}", path.display());
    }
    Ok(parts.join("/"))
}

fn validate_git_mode(mode: &str) -> anyhow::Result<()> {
    if mode.len() != 6 || !mode.bytes().all(|byte| matches!(byte, b'0'..=b'7')) {
        bail!("invalid Git index mode: {mode}");
    }
    Ok(())
}

fn validate_git_object_id(value: &str) -> anyhow::Result<()> {
    if value.len() < 40 || value.len() > 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("invalid Git object id: {value}");
    }
    Ok(())
}

fn git_check_ignored(root: &Path, relative: &str) -> anyhow::Result<bool> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["check-ignore", "--no-index", "--quiet", "--", relative])
        .output()
        .with_context(|| format!("run git check-ignore for {relative}"))?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(command_failure("git check-ignore", &output)),
    }
}

fn git_text(root: &Path, args: &[&str]) -> anyhow::Result<String> {
    let bytes = git_bytes(root, args)?;
    let text = String::from_utf8(bytes).context("Git output is not valid UTF-8")?;
    Ok(text.trim().to_string())
}

fn git_bytes(root: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    if !output.status.success() {
        return Err(command_failure(&format!("git {}", args.join(" ")), &output));
    }
    Ok(output.stdout)
}

fn command_failure(command: &str, output: &Output) -> anyhow::Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::anyhow!(
        "{command} failed with status {:?}: {}",
        output.status.code(),
        stderr.trim()
    )
}

#[cfg(unix)]
fn executable_bit(metadata: &fs::Metadata) -> Option<bool> {
    use std::os::unix::fs::PermissionsExt as _;
    Some(metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn executable_bit(_metadata: &fs::Metadata) -> Option<bool> {
    None
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
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_repository_relative_paths() {
        assert_eq!(
            normalize_relative_path(Path::new("./src/lib.rs")).unwrap(),
            "src/lib.rs"
        );
        assert!(normalize_relative_path(Path::new("../escape")).is_err());
        assert!(normalize_relative_path(Path::new("/")).is_err());
    }

    #[test]
    fn parses_stage_zero_index_records_and_sorts_them() {
        let input = b"100644 0123456789012345678901234567890123456789 0\tz.rs\0\
100755 abcdefabcdefabcdefabcdefabcdefabcdefabcd 0\ta.sh\0";
        let parsed = parse_index_entries(input).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].path, "a.sh");
        assert_eq!(parsed[1].path, "z.rs");
        assert_eq!(parsed[0].mode, "100755");
    }

    #[test]
    fn rejects_unmerged_index_stage() {
        let input = b"100644 0123456789012345678901234567890123456789 2\tsrc/lib.rs\0";
        assert!(parse_index_entries(input).is_err());
    }

    #[test]
    fn snapshot_hash_is_domain_separated() {
        assert_ne!(sha256(b"same"), domain_sha256(SNAPSHOT_HASH_DOMAIN, b"same"));
    }

    #[test]
    fn duplicate_path_classes_fail_closed() {
        let mut entries = BTreeMap::new();
        let first = SnapshotEntry {
            path: "a".into(),
            source_class: SourceClass::Tracked,
            kind: EntryKind::File,
            content_sha256: Some("00".repeat(32)),
            size_bytes: Some(1),
            executable: Some(false),
            index_mode: Some("100644".into()),
            index_blob: Some("0".repeat(40)),
        };
        insert_unique(&mut entries, first).unwrap();
        let second = SnapshotEntry {
            path: "a".into(),
            source_class: SourceClass::ExplicitIgnored,
            kind: EntryKind::File,
            content_sha256: Some("11".repeat(32)),
            size_bytes: Some(1),
            executable: Some(false),
            index_mode: None,
            index_blob: None,
        };
        assert!(insert_unique(&mut entries, second).is_err());
    }
}
