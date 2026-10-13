// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Morphological Bridge — candidate staging for autonomous body-design research.
//!
//! Generated morphology is untrusted input. This module stages immutable,
//! content-addressed candidates and never replaces the active model. Parsing,
//! physics compilation, qualification, and promotion belong to separate,
//! independently checked pipeline stages.

use anyhow::{bail, Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const MAX_MODEL_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct MorphologicalBridge {
    pub soma_root: PathBuf,
}

/// A staged candidate, not evidence that the model is physically valid or safe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedMorphology {
    pub model_name: String,
    pub candidate_id: String,
    pub candidate_path: PathBuf,
    /// BLAKE3 digest of the exact model bytes. This establishes byte identity,
    /// not physical validity, provenance, or qualification.
    pub content_digest: String,
    pub bytes: u64,
}

impl MorphologicalBridge {
    pub fn new(root: &str) -> Self {
        Self {
            soma_root: PathBuf::from(root).join("crates/symthaea-soma"),
        }
    }

    /// Compatibility entry point for morphology generation.
    ///
    /// This method now stages a content-addressed candidate and deliberately
    /// does not overwrite or activate a live model. Call
    /// `stage_candidate_body_model` when the candidate path/receipt metadata
    /// is needed. Physics validation and promotion remain separate operations.
    pub fn refactor_body_model(&self, model_name: &str, new_xml: &str) -> Result<()> {
        let candidate_id = blake3::hash(new_xml.as_bytes()).to_hex().to_string();
        let staged = self.stage_candidate_body_model(model_name, &candidate_id, new_xml)?;
        println!(
            "Morphology candidate staged: {} (BLAKE3 {})",
            staged.candidate_path.display(),
            staged.content_digest
        );
        Ok(())
    }

    /// Stage one named morphology candidate without activating it.
    ///
    /// The candidate ID must be a safe, stable slug. The final file is created
    /// without clobbering an existing ID; a same-filesystem hard-link makes
    /// the complete, fsynced temporary file visible atomically.
    ///
    /// Directory canonicalization and symlink checks reject pre-existing
    /// escapes, but are not a race-free defense against a concurrent writer
    /// replacing path components between checks and filesystem operations.
    /// Callers must keep the staging parent owner-controlled; hostile concurrent
    /// filesystem actors require descriptor-relative no-follow operations.
    pub fn stage_candidate_body_model(
        &self,
        model_name: &str,
        candidate_id: &str,
        model_bytes: &str,
    ) -> Result<StagedMorphology> {
        validate_model_name(model_name)?;
        validate_candidate_id(candidate_id)?;
        validate_model_bytes(model_bytes)?;

        let soma_root = &self.soma_root;
        let models_dir = soma_root.join("models");
        fs::create_dir_all(&models_dir)
            .with_context(|| format!("create model directory {}", models_dir.display()))?;

        let soma_root_canonical = soma_root
            .canonicalize()
            .with_context(|| format!("canonicalize Soma root {}", soma_root.display()))?;
        let models_canonical = models_dir
            .canonicalize()
            .with_context(|| format!("canonicalize models directory {}", models_dir.display()))?;
        if !models_canonical.starts_with(&soma_root_canonical) {
            bail!("models directory resolves outside the Soma root");
        }

        // Inspect each component before creating descendants. In particular,
        // do not call create_dir_all through a pre-existing candidates symlink:
        // that could create directories outside the model root before we notice.
        let candidates_root = create_contained_directory(&models_canonical, "candidates")?;
        let candidate_dir = create_contained_directory(&candidates_root, model_name)?;

        let candidate_path = candidate_dir.join(format!("{candidate_id}.xml"));
        let digest = blake3::hash(model_bytes.as_bytes()).to_hex().to_string();

        if path_exists_without_following_symlinks(&candidate_path)? {
            return existing_candidate(
                &candidate_path,
                model_name,
                candidate_id,
                &digest,
            );
        }

        let temp_path = write_synced_temp(&candidate_dir, candidate_id, model_bytes.as_bytes())?;
        match fs::hard_link(&temp_path, &candidate_path) {
            Ok(()) => {
                fs::remove_file(&temp_path)
                    .context("remove temporary morphology candidate after staging")?;
                File::open(&candidate_dir)
                    .and_then(|directory| directory.sync_all())
                    .context("sync morphology candidate directory")?;
                Ok(StagedMorphology {
                    model_name: model_name.to_string(),
                    candidate_id: candidate_id.to_string(),
                    candidate_path,
                    content_digest: digest,
                    bytes: model_bytes.len() as u64,
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = fs::remove_file(&temp_path);
                existing_candidate(
                    &candidate_path,
                    model_name,
                    candidate_id,
                    &digest,
                )
            }
            Err(error) => {
                let _ = fs::remove_file(&temp_path);
                Err(error).with_context(|| {
                    format!("publish staged candidate {}", candidate_path.display())
                })
            }
        }
    }
}

fn create_contained_directory(parent: &Path, component: &str) -> Result<PathBuf> {
    let path = parent.join(component);
    match fs::symlink_metadata(&path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("directory component must not be a symlink or non-directory: {}", path.display());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match fs::create_dir(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let metadata = fs::symlink_metadata(&path)
                        .with_context(|| format!("inspect raced directory {}", path.display()))?;
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        bail!("directory component raced to a symlink or non-directory: {}", path.display());
                    }
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("create directory {}", path.display()));
                }
            }
        }
        Err(error) => return Err(error).with_context(|| format!("inspect directory {}", path.display())),
    }

    let canonical = path.canonicalize()
        .with_context(|| format!("canonicalize directory {}", path.display()))?;
    if !canonical.starts_with(parent) {
        bail!("directory component resolves outside its parent: {}", path.display());
    }
    Ok(canonical)
}

fn path_exists_without_following_symlinks(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                bail!("candidate path exists and is not a regular file: {}", path.display());
            }
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
}

fn existing_candidate(
    path: &Path,
    model_name: &str,
    candidate_id: &str,
    expected_digest: &str,
) -> Result<StagedMorphology> {
    let bytes = fs::read(path).with_context(|| format!("read existing candidate {}", path.display()))?;
    let actual_digest = blake3::hash(&bytes).to_hex().to_string();
    if actual_digest != expected_digest {
        bail!("candidate ID already exists with different content: {candidate_id}");
    }
    Ok(StagedMorphology {
        model_name: model_name.to_string(),
        candidate_id: candidate_id.to_string(),
        candidate_path: path.to_path_buf(),
        content_digest: actual_digest,
        bytes: bytes.len() as u64,
    })
}

fn write_synced_temp(directory: &Path, candidate_id: &str, bytes: &[u8]) -> Result<PathBuf> {
    for _ in 0..32 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp_path = directory.join(format!(
            ".{candidate_id}.{}.{}.tmp",
            std::process::id(),
            sequence
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(mut file) => {
                if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
                    let _ = fs::remove_file(&temp_path);
                    return Err(error).context("write and sync staged morphology candidate");
                }
                return Ok(temp_path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create temporary file in {}", directory.display())
                });
            }
        }
    }
    bail!("could not allocate a unique temporary morphology candidate path")
}

fn validate_model_bytes(bytes: &str) -> Result<()> {
    if bytes.trim().is_empty() {
        bail!("morphology candidate must not be empty");
    }
    if bytes.len() > MAX_MODEL_BYTES {
        bail!("morphology candidate exceeds the {} byte size limit", MAX_MODEL_BYTES);
    }
    Ok(())
}

fn validate_slug(value: &str, field: &str, max_len: usize) -> Result<()> {
    let valid = !value.is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_');
    if valid {
        Ok(())
    } else {
        bail!("invalid {field} {value:?}")
    }
}

fn validate_model_name(name: &str) -> Result<()> {
    validate_slug(name, "model name", 64)
}

fn validate_candidate_id(candidate_id: &str) -> Result<()> {
    validate_slug(candidate_id, "candidate ID", 96)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "symthaea-morphology-{}-{}",
            std::process::id(),
            TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn rejects_path_traversal_in_model_and_candidate_ids() {
        let root = test_root();
        let bridge = MorphologicalBridge::new(root.to_str().unwrap());
        assert!(bridge.stage_candidate_body_model("../escape", "valid", "<mujoco/>").is_err());
        assert!(bridge.stage_candidate_body_model("arm", "../escape", "<mujoco/>").is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn staging_preserves_active_model_and_returns_content_identity() {
        let root = test_root();
        let bridge = MorphologicalBridge::new(root.to_str().unwrap());
        let models = bridge.soma_root.join("models");
        fs::create_dir_all(&models).unwrap();
        let active = models.join("arm.xml");
        fs::write(&active, "<mujoco><worldbody/></mujoco>").unwrap();
        let original = fs::read(&active).unwrap();

        let candidate_xml = "<mujoco><worldbody><body name=\"candidate\"/></worldbody></mujoco>";
        let staged = bridge
            .stage_candidate_body_model("arm", "candidate-v1", candidate_xml)
            .unwrap();

        assert_eq!(fs::read(&active).unwrap(), original);
        assert!(staged.candidate_path.starts_with(models.join("candidates")));
        assert_eq!(
            fs::read(&staged.candidate_path).unwrap(),
            b"<mujoco><worldbody><body name=\"candidate\"/></worldbody></mujoco>"
        );
        assert_eq!(
            staged.content_digest,
            blake3::hash(&fs::read(&staged.candidate_path).unwrap())
                .to_hex()
                .to_string()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn candidate_ids_are_immutable() {
        let root = test_root();
        let bridge = MorphologicalBridge::new(root.to_str().unwrap());
        let first = bridge.stage_candidate_body_model("arm", "candidate-1", "<mujoco/>").unwrap();
        let second = bridge.stage_candidate_body_model("arm", "candidate-1", "<mujoco/>").unwrap();
        assert_eq!(first, second);
        assert!(bridge.stage_candidate_body_model("arm", "candidate-1", "<mujoco><worldbody/></mujoco>").is_err());
        assert_eq!(fs::read(&first.candidate_path).unwrap(), b"<mujoco/>");
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_candidate_directory_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = test_root();
        let outside = root.with_extension("outside");
        let bridge = MorphologicalBridge::new(root.to_str().unwrap());
        let models = bridge.soma_root.join("models");
        fs::create_dir_all(&models).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, models.join("candidates")).unwrap();

        assert!(bridge.stage_candidate_body_model("arm", "candidate-1", "<mujoco/>").is_err());
        assert!(!outside.join("arm").exists(), "symlink target must not be modified");
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(outside);
    }
}
