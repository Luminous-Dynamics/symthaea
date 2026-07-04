// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only, sandboxed tool surface the audit agent is allowed to call.
//!
//! Every tool routes through [`resolve_in_sandbox`] before touching the filesystem or
//! shelling out. There is no `write_file` tool and no unconditional command execution —
//! this crate audits arbitrary, potentially untrusted repositories, so the sandbox
//! boundary is the one piece of genuinely safety-critical logic in the whole tool.

use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// Why a requested path was rejected.
#[derive(Debug)]
pub enum SandboxViolation {
    /// The canonical path resolves outside the sandbox root.
    Escape {
        requested: PathBuf,
        canonical: PathBuf,
        root: PathBuf,
    },
    /// Neither the path nor any existing ancestor of it could be found.
    NotFound(PathBuf),
    /// Filesystem error while resolving the path (invalid bytes, path too long, etc).
    Io(std::io::Error),
}

impl std::fmt::Display for SandboxViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxViolation::Escape {
                requested,
                canonical,
                root,
            } => write!(
                f,
                "path {} (resolved: {}) escapes sandbox root {}",
                requested.display(),
                canonical.display(),
                root.display()
            ),
            SandboxViolation::NotFound(p) => write!(f, "path not found: {}", p.display()),
            SandboxViolation::Io(e) => write!(f, "io error resolving path: {e}"),
        }
    }
}

impl std::error::Error for SandboxViolation {}

/// Collapse `.`/`..` components lexically (no filesystem access) — matches the
/// normalization already proven at `symthaea/src/action/mod.rs:802-817`.
fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// Resolve `requested` against `root` and confirm the canonical result stays inside
/// `root`. `root` must already be canonicalized by the caller (done once, at startup).
///
/// Ported from the proven `SandboxRoot::validate` pattern at
/// `symthaea/src/action/mod.rs:286-334`: relative paths resolve against `root`;
/// canonicalization follows symlinks, so a symlink pointing outside the root is caught
/// here regardless of how deep in a walk it's encountered; if the path doesn't exist
/// yet, walk up to the nearest existing ancestor, canonicalize that, and rejoin the
/// non-existent suffix, so directory probes and not-yet-materialized paths fail cleanly
/// instead of erroring on `canonicalize()`.
pub fn resolve_in_sandbox(root: &Path, requested: &Path) -> Result<PathBuf, SandboxViolation> {
    let requested_owned = if requested.as_os_str().is_empty() {
        root.to_path_buf()
    } else if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    let requested_ref = requested_owned.as_path();

    let canonical = if requested_ref.exists() {
        requested_ref.canonicalize().map_err(SandboxViolation::Io)?
    } else {
        let mut ancestor = requested_ref.parent();
        let mut found = None;
        while let Some(parent) = ancestor {
            if parent.exists() {
                found = Some(parent);
                break;
            }
            ancestor = parent.parent();
        }
        let base = found.ok_or_else(|| SandboxViolation::NotFound(requested_owned.clone()))?;
        let base_canon = base.canonicalize().map_err(SandboxViolation::Io)?;
        let suffix = requested_ref.strip_prefix(base).unwrap_or(requested_ref);
        base_canon.join(suffix)
    };

    let canonical = normalize_path(&canonical);

    if !canonical.starts_with(root) {
        return Err(SandboxViolation::Escape {
            requested: requested_owned,
            canonical,
            root: root.to_path_buf(),
        });
    }

    Ok(canonical)
}

const READ_PREVIEW_BYTES: usize = 8_000;
const LIST_DIR_MAX_ENTRIES: usize = 200;

pub struct FilePreview {
    pub content: String,
    pub truncated: bool,
    pub total_bytes: usize,
}

pub struct DirListing {
    pub entries: Vec<String>,
    pub total_entries: usize,
}

/// The read-only tool surface, rooted at a canonicalized target directory.
pub struct Sandbox {
    root: PathBuf,
    allow_exec: Vec<String>,
    ripgrep_available: bool,
}

impl Sandbox {
    /// `root` must already exist; it is canonicalized once here.
    pub fn new(root: &Path, allow_exec: Vec<String>) -> std::io::Result<Self> {
        let root = root.canonicalize()?;
        let ripgrep_available = which_on_path("rg");
        Ok(Self {
            root,
            allow_exec,
            ripgrep_available,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn allow_exec(&self) -> &[String] {
        &self.allow_exec
    }

    fn resolve(&self, requested: &str) -> Result<PathBuf, SandboxViolation> {
        resolve_in_sandbox(&self.root, Path::new(requested))
    }

    pub fn read_file(&self, path: &str) -> Result<FilePreview, SandboxViolation> {
        let resolved = self.resolve(path)?;
        let bytes = std::fs::read(&resolved).map_err(SandboxViolation::Io)?;
        let total_bytes = bytes.len();
        let truncated = total_bytes > READ_PREVIEW_BYTES;
        let slice = &bytes[..total_bytes.min(READ_PREVIEW_BYTES)];
        let content = String::from_utf8_lossy(slice).into_owned();
        Ok(FilePreview {
            content,
            truncated,
            total_bytes,
        })
    }

    /// Lists a directory. When `recursive`, every descended-into subdirectory is
    /// individually re-validated against the sandbox before being read — a symlink
    /// planted inside the (possibly untrusted) target repo pointing outward must not
    /// let a recursive walk escape the sandbox just because the top-level path passed.
    pub fn list_dir(&self, path: &str, recursive: bool) -> Result<DirListing, SandboxViolation> {
        let resolved = self.resolve(path)?;
        let mut entries = Vec::new();
        if recursive {
            self.walk_recursive(&resolved, &mut entries)?;
        } else {
            let read_dir = std::fs::read_dir(&resolved).map_err(SandboxViolation::Io)?;
            for entry in read_dir {
                let entry = entry.map_err(SandboxViolation::Io)?;
                entries.push(entry.path().display().to_string());
            }
        }
        let total_entries = entries.len();
        entries.truncate(LIST_DIR_MAX_ENTRIES);
        Ok(DirListing {
            entries,
            total_entries,
        })
    }

    fn walk_recursive(&self, dir: &Path, out: &mut Vec<String>) -> Result<(), SandboxViolation> {
        if out.len() >= LIST_DIR_MAX_ENTRIES {
            return Ok(());
        }
        // Re-validate this directory itself (it may be a symlink target reached via an
        // earlier symlink hop) before reading its contents.
        let checked = resolve_in_sandbox(&self.root, dir)?;
        let read_dir = std::fs::read_dir(&checked).map_err(SandboxViolation::Io)?;
        for entry in read_dir {
            let entry = entry.map_err(SandboxViolation::Io)?;
            let path = entry.path();
            // Per-entry re-check: canonicalizes and rejects escapes even if `path` is a
            // symlink to outside the sandbox.
            let entry_checked = match resolve_in_sandbox(&self.root, &path) {
                Ok(p) => p,
                Err(SandboxViolation::Escape { .. }) => continue, // skip, don't abort the walk
                Err(e) => return Err(e),
            };
            out.push(entry_checked.display().to_string());
            if entry_checked.is_dir() && out.len() < LIST_DIR_MAX_ENTRIES {
                self.walk_recursive(&entry_checked, out)?;
            }
            if out.len() >= LIST_DIR_MAX_ENTRIES {
                break;
            }
        }
        Ok(())
    }

    /// Greps the repo for `pattern`, optionally restricted to files matching `glob`.
    /// Uses `rg` when present on `PATH`; otherwise falls back to a `.gitignore`-aware
    /// manual walk (the `ignore` crate) plus `regex` matching, so results are
    /// consistent either way and neither path descends into `.git`/`target`/etc by
    /// default.
    pub fn grep_repo(&self, pattern: &str, glob: Option<&str>) -> Result<String, SandboxViolation> {
        if self.ripgrep_available {
            let mut cmd = Command::new("rg");
            cmd.arg("--line-number")
                .arg("--max-columns")
                .arg("200")
                .arg("--heading");
            if let Some(g) = glob {
                cmd.arg("--glob").arg(g);
            }
            cmd.arg(pattern).current_dir(&self.root);
            let output = cmd.output().map_err(SandboxViolation::Io)?;
            return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
        }

        let re = regex::Regex::new(pattern)
            .map_err(|e| SandboxViolation::Io(std::io::Error::other(e.to_string())))?;
        let glob_matcher = glob.and_then(|g| glob_to_regex(g));
        let mut out = String::new();
        let walker = ignore::WalkBuilder::new(&self.root).build();
        for entry in walker.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(ref gm) = glob_matcher {
                let name = path.to_string_lossy();
                if !gm.is_match(&name) {
                    continue;
                }
            }
            let Ok(content) = std::fs::read_to_string(path) else {
                continue;
            };
            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    out.push_str(&format!("{}:{}:{}\n", path.display(), i + 1, line));
                }
            }
        }
        Ok(out)
    }

    pub fn git_log(&self, path: Option<&str>, limit: u32) -> Result<String, SandboxViolation> {
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.root)
            .arg("log")
            .arg("--oneline")
            .arg("-n")
            .arg(limit.to_string());
        if let Some(p) = path {
            let resolved = self.resolve(p)?;
            cmd.arg("--").arg(resolved);
        }
        run_git(cmd)
    }

    pub fn git_status(&self) -> Result<String, SandboxViolation> {
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.root)
            .arg("status")
            .arg("--porcelain");
        run_git(cmd)
    }

    pub fn git_diff(&self, path: Option<&str>) -> Result<String, SandboxViolation> {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(&self.root).arg("diff");
        if let Some(p) = path {
            let resolved = self.resolve(p)?;
            cmd.arg("--").arg(resolved);
        }
        run_git(cmd)
    }

    pub fn loc_count(&self, path: Option<&str>) -> Result<String, SandboxViolation> {
        let scope = self.resolve(path.unwrap_or(""))?;
        let mut by_ext: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let walker = ignore::WalkBuilder::new(&scope).build();
        for entry in walker.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().into_owned())
                .unwrap_or_else(|| "(none)".to_string());
            let lines = std::fs::read_to_string(path)
                .map(|s| s.lines().count())
                .unwrap_or(0);
            *by_ext.entry(ext).or_insert(0) += lines;
        }
        let mut out = String::new();
        for (ext, lines) in by_ext {
            out.push_str(&format!("{ext}: {lines} lines\n"));
        }
        Ok(out)
    }

    /// Runs `cmd` only if it exactly matches (after trim) an entry in `allow_exec`.
    /// The tool-call parser in `agent_loop` does not even advertise this tool's
    /// existence to the model when `allow_exec` is empty.
    pub fn run_check(&self, cmd: &str) -> Result<String, SandboxViolation> {
        let cmd = cmd.trim();
        if !self.allow_exec.iter().any(|allowed| allowed == cmd) {
            return Err(SandboxViolation::Io(std::io::Error::other(format!(
                "command not in --allow-exec whitelist: {cmd:?}"
            ))));
        }
        let mut parts = cmd.split_whitespace();
        let program = parts
            .next()
            .ok_or_else(|| SandboxViolation::Io(std::io::Error::other("empty command")))?;
        let args: Vec<&str> = parts.collect();
        let output = Command::new(program)
            .args(args)
            .current_dir(&self.root)
            .output()
            .map_err(SandboxViolation::Io)?;
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        Ok(combined)
    }
}

fn run_git(mut cmd: Command) -> Result<String, SandboxViolation> {
    let output = cmd.output().map_err(SandboxViolation::Io)?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn which_on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(program);
                candidate.is_file()
            })
        })
        .unwrap_or(false)
}

/// Extremely small glob-to-regex conversion for the fallback grep path: only handles
/// `*` (any run of non-separator chars) since that's the common case for `--glob`
/// filters like `*.rs`.
fn glob_to_regex(glob: &str) -> Option<regex::Regex> {
    let mut pattern = String::from("^.*");
    for part in glob.split('*') {
        pattern.push_str(&regex::escape(part));
        pattern.push_str(".*");
    }
    pattern.push('$');
    regex::Regex::new(&pattern).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Fixture {
        _tmp: tempfile::TempDir,
        root: PathBuf,
        outside: PathBuf,
    }

    fn build_fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("repo");
        fs::create_dir_all(root.join("src/sub")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/sub/inner.rs"), "// inner").unwrap();
        fs::write(root.join("README.md"), "# readme").unwrap();
        let outside = tmp.path().join("outside.txt");
        fs::write(&outside, "outside").unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&outside, root.join("link_out")).unwrap();
            symlink(root.join("src/main.rs"), root.join("link_in")).unwrap();
        }

        let root = root.canonicalize().unwrap();
        Fixture {
            _tmp: tmp,
            root,
            outside,
        }
    }

    fn allow(fixture: &Fixture, input: &str) {
        let result = resolve_in_sandbox(&fixture.root, Path::new(input));
        assert!(
            result.is_ok(),
            "expected allow for {input:?}, got {result:?}"
        );
    }

    fn deny_escape(fixture: &Fixture, input: &str) {
        let result = resolve_in_sandbox(&fixture.root, Path::new(input));
        assert!(
            matches!(result, Err(SandboxViolation::Escape { .. })),
            "expected Escape for {input:?}, got {result:?}"
        );
    }

    #[test]
    fn plain_in_tree_file_allowed() {
        allow(&build_fixture(), "src/main.rs");
    }

    #[test]
    fn dot_slash_prefix_allowed() {
        allow(&build_fixture(), "./src/main.rs");
    }

    #[test]
    fn nested_subdirectory_allowed() {
        allow(&build_fixture(), "src/sub/inner.rs");
    }

    #[test]
    fn empty_path_resolves_to_root() {
        let fixture = build_fixture();
        let resolved = resolve_in_sandbox(&fixture.root, Path::new("")).unwrap();
        assert_eq!(resolved, fixture.root);
    }

    #[test]
    fn parent_traversal_denied() {
        deny_escape(&build_fixture(), "../outside.txt");
    }

    #[test]
    fn buried_parent_traversal_denied() {
        deny_escape(&build_fixture(), "src/../../outside.txt");
    }

    #[test]
    fn absolute_path_outside_root_denied() {
        deny_escape(&build_fixture(), "/etc/passwd");
    }

    #[test]
    fn absolute_path_equal_to_root_allowed() {
        let fixture = build_fixture();
        let root_str = fixture.root.to_str().unwrap();
        allow(&fixture, root_str);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_denied() {
        deny_escape(&build_fixture(), "link_out");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_inside_root_allowed() {
        allow(&build_fixture(), "link_in");
    }

    #[test]
    fn nonexistent_path_with_missing_parent_is_not_found_not_panic() {
        let fixture = build_fixture();
        let result = resolve_in_sandbox(&fixture.root, Path::new("nope/also_nope/file.rs"));
        assert!(matches!(
            result,
            Err(SandboxViolation::NotFound(_)) | Err(SandboxViolation::Escape { .. })
        ));
    }

    #[test]
    fn nonexistent_file_with_existing_parent_allowed() {
        allow(&build_fixture(), "src/nonexistent.rs");
    }

    #[test]
    fn overlong_path_denied_not_panic() {
        let fixture = build_fixture();
        let long_name = "a".repeat(5000);
        let result = resolve_in_sandbox(&fixture.root, Path::new(&long_name));
        assert!(result.is_err(), "expected an error, got {result:?}");
    }

    #[test]
    fn directory_scope_path_allowed() {
        allow(&build_fixture(), "src/sub");
    }

    #[test]
    fn outside_fixture_file_is_reachable_only_via_root_not_via_relative_escape() {
        let fixture = build_fixture();
        // Sanity: the outside file really exists and really is outside root.
        assert!(fixture.outside.exists());
        assert!(!fixture.outside.starts_with(&fixture.root));
    }

    #[test]
    fn sandbox_read_file_rejects_traversal() {
        let fixture = build_fixture();
        let sandbox = Sandbox::new(&fixture.root, vec![]).unwrap();
        let result = sandbox.read_file("../../etc/passwd");
        assert!(matches!(result, Err(SandboxViolation::Escape { .. })));
    }

    #[test]
    fn run_check_rejected_when_not_whitelisted() {
        let fixture = build_fixture();
        let sandbox = Sandbox::new(&fixture.root, vec![]).unwrap();
        let result = sandbox.run_check("rm -rf /");
        assert!(result.is_err());
    }

    #[test]
    fn run_check_allowed_when_whitelisted() {
        let fixture = build_fixture();
        let sandbox = Sandbox::new(&fixture.root, vec!["true".to_string()]).unwrap();
        let result = sandbox.run_check("true");
        assert!(result.is_ok());
    }
}
