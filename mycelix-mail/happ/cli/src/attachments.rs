// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use blake2::{Blake2b512, Digest};
use mime_guess::MimeGuess;
use std::fs;
use std::path::{Path, PathBuf};

use crate::types::AttachmentRef;

/// Local attachment store (Iroh-compatible ticket placeholder).
///
/// Stores attachment bytes on disk and returns a deterministic ticket derived
/// from the content hash. This provides a clean integration seam for a future
/// Iroh-backed transport while keeping the CLI functional today.
pub struct AttachmentStore {
    base_dir: PathBuf,
    upload_command: Option<String>,
}

impl AttachmentStore {
    pub fn new(base_dir: PathBuf, upload_command: Option<String>) -> Result<Self> {
        fs::create_dir_all(&base_dir)
            .with_context(|| format!("Failed to create attachment store dir: {:?}", base_dir))?;
        Ok(Self {
            base_dir,
            upload_command,
        })
    }

    pub fn store_paths(&self, paths: &[String]) -> Result<Vec<AttachmentRef>> {
        let mut attachments = Vec::new();
        for path_str in paths {
            let attachment = self.store_path(Path::new(path_str))?;
            attachments.push(attachment);
        }
        Ok(attachments)
    }

    fn store_path(&self, path: &Path) -> Result<AttachmentRef> {
        let content = fs::read(path)
            .with_context(|| format!("Failed to read attachment: {:?}", path))?;
        let size_bytes = content.len() as u64;

        let mut hasher = Blake2b512::new();
        hasher.update(&content);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);

        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("attachment")
            .to_string();

        let content_type = MimeGuess::from_path(path)
            .first_or_octet_stream()
            .essence_str()
            .to_string();

        let storage_path = self.base_dir.join(&hash_hex);
        if !storage_path.exists() {
            fs::write(&storage_path, &content)
                .with_context(|| format!("Failed to write attachment to {:?}", storage_path))?;
        }

        let ticket = if let Some(cmd_template) = &self.upload_command {
            self.upload_via_command(cmd_template, path, &hash_hex, &filename)?
        } else {
            format!("iroh:{}", hash_hex)
        };

        Ok(AttachmentRef {
            filename,
            content_type,
            size_bytes,
            hash: hash_hex.clone(),
            ticket,
        })
    }

    fn upload_via_command(
        &self,
        template: &str,
        path: &Path,
        hash: &str,
        filename: &str,
    ) -> Result<String> {
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Attachment path is not valid UTF-8"))?;

        let cmd = template
            .replace("{path}", path_str)
            .replace("{hash}", hash)
            .replace("{filename}", filename);

        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .context("Failed to execute attachment upload command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(
                "Attachment upload command failed: {}",
                stderr.trim()
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let ticket = stdout
            .lines()
            .map(|l| l.trim())
            .find(|l| !l.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Upload command returned empty ticket"))?;

        Ok(ticket.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn stores_attachment_and_returns_ticket() {
        let dir = tempdir().unwrap();
        let store = AttachmentStore::new(dir.path().join("store"), None).unwrap();

        let file_path = dir.path().join("hello.txt");
        fs::write(&file_path, b"hello world").unwrap();

        let attachments = store
            .store_paths(&[file_path.to_string_lossy().to_string()])
            .unwrap();

        assert_eq!(attachments.len(), 1);
        let attachment = &attachments[0];
        assert_eq!(attachment.filename, "hello.txt");
        assert!(attachment.ticket.starts_with("iroh:"));
        assert_eq!(attachment.size_bytes, 11);
    }
}
