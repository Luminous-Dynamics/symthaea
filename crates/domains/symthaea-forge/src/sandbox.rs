// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-safe in-place mutation staging.
//!
//! `symthaea-forge` must actually write a candidate mutation to disk for
//! `cargo check`/`cargo test`/a benchmark binary to see it -- there is no
//! cheap way to make `rustc` compile a file that only exists in memory.
//! Rather than a full workspace copy (expensive, and this project's
//! standing rule is "no git worktrees" -- see
//! `memory/feedback_no_worktrees.md`), this module makes in-place mutation
//! *safe* instead of avoiding it:
//!
//! 1. Before mutating, the original file is copied to `<file>.forge-orig`.
//! 2. [`StagedMutation`] is an RAII guard: `Drop` restores the original
//!    content from the backup and deletes it, unless [`StagedMutation::commit`]
//!    was called first.
//! 3. On construction, [`Sandbox::new`] scans for orphaned `.forge-orig`
//!    files (evidence of a previous run that was killed mid-mutation --
//!    this project's session has hit exactly that failure mode repeatedly,
//!    see `memory/feedback_background_cargo_gets_killed_mystery.md`) and
//!    self-heals by restoring them before any new work starts.
//!
//! `Drop` cannot run after `SIGKILL`, so this is not a perfect guarantee --
//! but combined with the startup self-heal, an interrupted run leaves the
//! tree mutated for at most the lifetime of the next `Sandbox::new` call,
//! never permanently.

use std::path::{Path, PathBuf};

const BACKUP_SUFFIX: &str = ".forge-orig";

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("{path:?} resolves outside the allowed project root {root:?}")]
    OutsideProjectRoot { path: PathBuf, root: PathBuf },
    #[error("io error on {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub struct Sandbox {
    project_root: PathBuf,
}

impl Sandbox {
    /// `project_root` is canonicalized once and every staged file must
    /// resolve inside it (blocks absolute-path and `..`-traversal escapes,
    /// mirroring `self_optimization.rs`'s existing path guard).
    ///
    /// Also scans `search_roots` for orphaned `.forge-orig` backups left by
    /// a previous interrupted run and restores them.
    pub fn new(
        project_root: impl AsRef<Path>,
        search_roots: &[PathBuf],
    ) -> Result<Self, SandboxError> {
        let project_root = project_root
            .as_ref()
            .canonicalize()
            .map_err(|e| SandboxError::Io {
                path: project_root.as_ref().to_path_buf(),
                source: e,
            })?;
        let sandbox = Self { project_root };
        for root in search_roots {
            sandbox.heal_orphaned_backups(root)?;
        }
        Ok(sandbox)
    }

    fn heal_orphaned_backups(&self, dir: &Path) -> Result<(), SandboxError> {
        if !dir.is_dir() {
            return Ok(());
        }
        let entries = std::fs::read_dir(dir).map_err(|e| SandboxError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                self.heal_orphaned_backups(&path)?;
                continue;
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(orig_name) = name.strip_suffix(BACKUP_SUFFIX) {
                    let target = path.with_file_name(orig_name);
                    tracing_or_eprintln(&format!(
                        "forge: healing orphaned backup {path:?} -> restoring {target:?}"
                    ));
                    std::fs::copy(&path, &target).map_err(|e| SandboxError::Io {
                        path: target.clone(),
                        source: e,
                    })?;
                    std::fs::remove_file(&path).map_err(|e| SandboxError::Io {
                        path: path.clone(),
                        source: e,
                    })?;
                }
            }
        }
        Ok(())
    }

    fn require_within_root(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let canonical = path.canonicalize().map_err(|e| SandboxError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if !canonical.starts_with(&self.project_root) {
            return Err(SandboxError::OutsideProjectRoot {
                path: canonical,
                root: self.project_root.clone(),
            });
        }
        Ok(canonical)
    }

    /// Back up `file_path`'s current content and return an RAII guard.
    /// Restores automatically on drop unless [`StagedMutation::commit`] is
    /// called.
    pub fn stage(&self, file_path: &Path) -> Result<StagedMutation, SandboxError> {
        let canonical = self.require_within_root(file_path)?;
        let original = std::fs::read_to_string(&canonical).map_err(|e| SandboxError::Io {
            path: canonical.clone(),
            source: e,
        })?;
        let backup_path = backup_path_for(&canonical);
        std::fs::write(&backup_path, &original).map_err(|e| SandboxError::Io {
            path: backup_path.clone(),
            source: e,
        })?;
        Ok(StagedMutation {
            file_path: canonical,
            backup_path,
            original,
            committed: false,
        })
    }
}

fn backup_path_for(file_path: &Path) -> PathBuf {
    let mut s = file_path.as_os_str().to_owned();
    s.push(BACKUP_SUFFIX);
    PathBuf::from(s)
}

/// Try `tracing::warn!` if a subscriber is installed; always also print to
/// stderr so the healing message is visible even in a bare `cargo run`.
fn tracing_or_eprintln(msg: &str) {
    tracing::warn!(target: "symthaea_forge::sandbox", "{msg}");
    eprintln!("{msg}");
}

/// RAII guard over one staged mutation. Write the candidate content with
/// [`Self::write`], run gates against it, then either [`Self::commit`] (to
/// keep the mutation on disk -- used only for the *winning* candidate
/// pending human review, never for a losing one) or let the guard drop to
/// restore the original automatically.
#[derive(Debug)]
pub struct StagedMutation {
    file_path: PathBuf,
    backup_path: PathBuf,
    original: String,
    committed: bool,
}

impl StagedMutation {
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    pub fn original_content(&self) -> &str {
        &self.original
    }

    pub fn write(&self, new_content: &str) -> Result<(), SandboxError> {
        std::fs::write(&self.file_path, new_content).map_err(|e| SandboxError::Io {
            path: self.file_path.clone(),
            source: e,
        })
    }

    /// Restore the original content immediately (idempotent; also happens
    /// automatically on drop if this was never called).
    pub fn restore(&self) -> Result<(), SandboxError> {
        std::fs::write(&self.file_path, &self.original).map_err(|e| SandboxError::Io {
            path: self.file_path.clone(),
            source: e,
        })
    }

    /// Keep the current on-disk content and remove the backup -- the
    /// mutation survives this guard's drop. Only call this for a candidate
    /// that has cleared every gate and is ready for human review.
    ///
    /// Marks `committed` and lets `self` drop normally at the end of this
    /// call: `Drop::drop` always removes the backup file (for both the
    /// committed and non-committed paths -- see below), it only skips the
    /// restore-write when committed, so there is nothing left for this
    /// method to do afterward. (An earlier version also removed the
    /// backup file itself here, which raced with `Drop::drop` already
    /// having removed it and panicked on `NotFound`.)
    pub fn commit(mut self) -> Result<(), SandboxError> {
        self.committed = true;
        Ok(())
    }
}

impl Drop for StagedMutation {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::write(&self.file_path, &self.original);
        }
        let _ = std::fs::remove_file(&self.backup_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_project(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "symthaea-forge-sandbox-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        dir
    }

    #[test]
    fn stage_write_and_drop_restores_original() {
        let root = temp_project("restore");
        let file = root.join("src").join("lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();

        let sandbox = Sandbox::new(&root, &[]).unwrap();
        {
            let staged = sandbox.stage(&file).unwrap();
            staged.write("const X: i32 = 999;\n").unwrap();
            assert_eq!(
                std::fs::read_to_string(&file).unwrap(),
                "const X: i32 = 999;\n"
            );
            // guard drops here without commit()
        }
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "const X: i32 = 1;\n"
        );
        assert!(!backup_path_for(&file.canonicalize().unwrap()).exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn commit_keeps_the_mutation_on_disk() {
        let root = temp_project("commit");
        let file = root.join("src").join("lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();

        let sandbox = Sandbox::new(&root, &[]).unwrap();
        let staged = sandbox.stage(&file).unwrap();
        staged.write("const X: i32 = 2;\n").unwrap();
        staged.commit().unwrap();

        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "const X: i32 = 2;\n"
        );
        assert!(!backup_path_for(&file.canonicalize().unwrap()).exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn path_outside_project_root_is_rejected() {
        let root = temp_project("outside-root");
        let outside_dir = temp_project("outside-target");
        let outside_file = outside_dir.join("src").join("lib.rs");
        std::fs::write(&outside_file, "// not part of the project\n").unwrap();

        let sandbox = Sandbox::new(&root, &[]).unwrap();
        let err = sandbox.stage(&outside_file).unwrap_err();
        assert!(matches!(err, SandboxError::OutsideProjectRoot { .. }));

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn orphaned_backup_is_healed_on_construction() {
        let root = temp_project("heal");
        let file = root.join("src").join("lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();
        // Simulate a previous run that mutated the file and was killed
        // before its StagedMutation guard could drop: write the mutated
        // content to the real file, and the original to a `.forge-orig`
        // backup next to it (exactly what `stage()` + `write()` leaves
        // behind mid-flight).
        let canonical_file = file.canonicalize().unwrap();
        std::fs::write(&canonical_file, "const X: i32 = 999999;\n").unwrap();
        std::fs::write(backup_path_for(&canonical_file), "const X: i32 = 1;\n").unwrap();

        // Constructing a new Sandbox over this directory should notice and
        // restore the orphaned backup before any new work starts.
        let _sandbox = Sandbox::new(&root, &[root.clone()]).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "const X: i32 = 1;\n"
        );
        assert!(!backup_path_for(&canonical_file).exists());

        let _ = std::fs::remove_dir_all(&root);
    }
}
