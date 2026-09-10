// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-aware, exclusive in-place mutation staging.
//!
//! Forge currently has to place a candidate on disk while Cargo compiles/tests it. This module
//! keeps that exceptional mutation window narrow and fail-closed:
//!
//! 1. Every target and scan root is canonicalized beneath one project root.
//! 2. A `<file>.forge-orig` backup is created with `create_new`, acting as an exclusive per-file
//!    staging lease. A second Forge process cannot silently overwrite it.
//! 3. Pre-existing backups are reported as recovery-required evidence. They are **not**
//!    automatically restored because a backup may belong to another still-running Forge process.
//! 4. [`StagedMutation::restore`] consumes the guard and reports restoration/removal failures.
//!    `Drop` remains a best-effort crash/panic fallback only.
//! 5. There is deliberately no `commit()` capability: this sandbox cannot make a staged candidate
//!    survive as the canonical source file.
//!
//! `SIGKILL` can still interrupt the mutation window. The resulting backup makes that state
//! detectable on the next run, which must stop for explicit recovery rather than guessing that the
//! backup is orphaned.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

const BACKUP_SUFFIX: &str = ".forge-orig";

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("{path:?} resolves outside the allowed project root {root:?}")]
    OutsideProjectRoot { path: PathBuf, root: PathBuf },
    #[error("Forge staging/recovery backup already exists; explicit recovery is required: {0:?}")]
    RecoveryRequired(PathBuf),
    #[error("target changed while Forge was acquiring its staging lease: {0:?}")]
    ConcurrentModification(PathBuf),
    #[error("io error on {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug)]
pub struct Sandbox {
    project_root: PathBuf,
}

impl Sandbox {
    /// Create a sandbox rooted at one exact canonical project directory.
    ///
    /// `search_roots` are inspected only for stale/live staging markers. Every existing root must
    /// itself canonicalize beneath `project_root`; directory symlinks are never traversed.
    pub fn new(
        project_root: impl AsRef<Path>,
        search_roots: &[PathBuf],
    ) -> Result<Self, SandboxError> {
        let requested_root = project_root.as_ref();
        let project_root = requested_root
            .canonicalize()
            .map_err(|source| SandboxError::Io {
                path: requested_root.to_path_buf(),
                source,
            })?;
        let sandbox = Self { project_root };

        for root in search_roots {
            if !root.exists() {
                continue;
            }
            let canonical = sandbox.require_within_root(root)?;
            sandbox.detect_existing_backups(&canonical)?;
        }
        Ok(sandbox)
    }

    fn detect_existing_backups(&self, dir: &Path) -> Result<(), SandboxError> {
        if !dir.is_dir() {
            return Ok(());
        }
        let entries = std::fs::read_dir(dir).map_err(|source| SandboxError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| SandboxError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|source| SandboxError::Io {
                path: path.clone(),
                source,
            })?;
            if file_type.is_symlink() {
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(BACKUP_SUFFIX))
                {
                    return Err(SandboxError::RecoveryRequired(path));
                }
                continue;
            }
            if file_type.is_dir() {
                let canonical = self.require_within_root(&path)?;
                self.detect_existing_backups(&canonical)?;
                continue;
            }
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(BACKUP_SUFFIX))
            {
                return Err(SandboxError::RecoveryRequired(path));
            }
        }
        Ok(())
    }

    fn require_within_root(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        let canonical = path.canonicalize().map_err(|source| SandboxError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !canonical.starts_with(&self.project_root) {
            return Err(SandboxError::OutsideProjectRoot {
                path: canonical,
                root: self.project_root.clone(),
            });
        }
        Ok(canonical)
    }

    /// Acquire an exclusive staging lease and preserve the exact original UTF-8 source.
    pub fn stage(&self, file_path: &Path) -> Result<StagedMutation, SandboxError> {
        let canonical = self.require_within_root(file_path)?;
        let original = std::fs::read_to_string(&canonical).map_err(|source| SandboxError::Io {
            path: canonical.clone(),
            source,
        })?;
        let backup_path = backup_path_for(&canonical);

        let mut backup = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&backup_path)
        {
            Ok(file) => file,
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(SandboxError::RecoveryRequired(backup_path));
            }
            Err(source) => {
                return Err(SandboxError::Io {
                    path: backup_path,
                    source,
                });
            }
        };
        backup
            .write_all(original.as_bytes())
            .and_then(|_| backup.sync_all())
            .map_err(|source| SandboxError::Io {
                path: backup_path.clone(),
                source,
            })?;

        // Detect a non-Forge writer racing the lease acquisition window before any candidate bytes
        // are written. Keep the backup on error so explicit recovery has the known original.
        let after_lease = std::fs::read_to_string(&canonical).map_err(|source| SandboxError::Io {
            path: canonical.clone(),
            source,
        })?;
        if after_lease != original {
            return Err(SandboxError::ConcurrentModification(canonical));
        }

        Ok(StagedMutation {
            file_path: canonical,
            backup_path,
            original,
            restored: false,
        })
    }
}

fn backup_path_for(file_path: &Path) -> PathBuf {
    let mut name = file_path.as_os_str().to_owned();
    name.push(BACKUP_SUFFIX);
    PathBuf::from(name)
}

/// Exclusive guard over one temporary source mutation.
#[derive(Debug)]
pub struct StagedMutation {
    file_path: PathBuf,
    backup_path: PathBuf,
    original: String,
    restored: bool,
}

impl StagedMutation {
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    pub fn original_content(&self) -> &str {
        &self.original
    }

    /// Write candidate content only while the staging lease still exists and the canonical file
    /// still contains the original bytes observed when the lease was acquired.
    pub fn write(&self, new_content: &str) -> Result<(), SandboxError> {
        if !self.backup_path.is_file() {
            return Err(SandboxError::RecoveryRequired(self.backup_path.clone()));
        }
        let current = std::fs::read_to_string(&self.file_path).map_err(|source| SandboxError::Io {
            path: self.file_path.clone(),
            source,
        })?;
        if current != self.original {
            return Err(SandboxError::ConcurrentModification(self.file_path.clone()));
        }
        std::fs::write(&self.file_path, new_content).map_err(|source| SandboxError::Io {
            path: self.file_path.clone(),
            source,
        })
    }

    /// Restore the exact original source and release the staging lease.
    ///
    /// Search code should call this explicitly so restoration failures propagate. `Drop` is only
    /// an emergency fallback for early-return/panic paths.
    pub fn restore(mut self) -> Result<(), SandboxError> {
        std::fs::write(&self.file_path, &self.original).map_err(|source| SandboxError::Io {
            path: self.file_path.clone(),
            source,
        })?;
        std::fs::remove_file(&self.backup_path).map_err(|source| SandboxError::Io {
            path: self.backup_path.clone(),
            source,
        })?;
        self.restored = true;
        Ok(())
    }
}

impl Drop for StagedMutation {
    fn drop(&mut self) {
        if self.restored {
            return;
        }
        if let Err(error) = std::fs::write(&self.file_path, &self.original) {
            eprintln!(
                "forge: emergency restore failed for {:?}: {error}; backup retained at {:?}",
                self.file_path, self.backup_path
            );
            return;
        }
        if let Err(error) = std::fs::remove_file(&self.backup_path) {
            eprintln!(
                "forge: restored {:?} but failed to remove staging backup {:?}: {error}",
                self.file_path, self.backup_path
            );
        }
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
    fn stage_write_and_explicit_restore_restores_original() {
        let root = temp_project("restore");
        let file = root.join("src/lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();
        let sandbox = Sandbox::new(&root, &[]).unwrap();
        let staged = sandbox.stage(&file).unwrap();
        staged.write("const X: i32 = 999;\n").unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "const X: i32 = 999;\n");
        staged.restore().unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "const X: i32 = 1;\n");
        assert!(!backup_path_for(&file.canonicalize().unwrap()).exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn drop_remains_best_effort_fallback() {
        let root = temp_project("drop-restore");
        let file = root.join("src/lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();
        let sandbox = Sandbox::new(&root, &[]).unwrap();
        {
            let staged = sandbox.stage(&file).unwrap();
            staged.write("const X: i32 = 2;\n").unwrap();
        }
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "const X: i32 = 1;\n");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn second_stager_cannot_overwrite_live_backup() {
        let root = temp_project("exclusive");
        let file = root.join("src/lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();
        let first = Sandbox::new(&root, &[]).unwrap();
        let guard = first.stage(&file).unwrap();
        let second = Sandbox::new(&root, &[]).unwrap();
        assert!(matches!(
            second.stage(&file),
            Err(SandboxError::RecoveryRequired(_))
        ));
        guard.restore().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn existing_backup_stops_startup_instead_of_auto_healing() {
        let root = temp_project("recovery-required");
        let file = root.join("src/lib.rs");
        std::fs::write(&file, "mutated\n").unwrap();
        let canonical = file.canonicalize().unwrap();
        std::fs::write(backup_path_for(&canonical), "original\n").unwrap();
        let error = Sandbox::new(&root, &[root.join("src")]).unwrap_err();
        assert!(matches!(error, SandboxError::RecoveryRequired(_)));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "mutated\n");
        assert_eq!(
            std::fs::read_to_string(backup_path_for(&canonical)).unwrap(),
            "original\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn search_root_outside_project_is_rejected_before_scanning() {
        let root = temp_project("root");
        let outside = temp_project("outside-scan");
        assert!(matches!(
            Sandbox::new(&root, &[outside.clone()]),
            Err(SandboxError::OutsideProjectRoot { .. })
        ));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn path_outside_project_root_is_rejected() {
        let root = temp_project("outside-root");
        let outside_dir = temp_project("outside-target");
        let outside_file = outside_dir.join("src/lib.rs");
        std::fs::write(&outside_file, "// not part of the project\n").unwrap();
        let sandbox = Sandbox::new(&root, &[]).unwrap();
        assert!(matches!(
            sandbox.stage(&outside_file),
            Err(SandboxError::OutsideProjectRoot { .. })
        ));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn write_detects_non_forge_modification_after_lease() {
        let root = temp_project("concurrent-write");
        let file = root.join("src/lib.rs");
        std::fs::write(&file, "const X: i32 = 1;\n").unwrap();
        let sandbox = Sandbox::new(&root, &[]).unwrap();
        let staged = sandbox.stage(&file).unwrap();
        std::fs::write(&file, "external edit\n").unwrap();
        assert!(matches!(
            staged.write("candidate\n"),
            Err(SandboxError::ConcurrentModification(_))
        ));
        drop(staged);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "const X: i32 = 1;\n");
        let _ = std::fs::remove_dir_all(&root);
    }
}