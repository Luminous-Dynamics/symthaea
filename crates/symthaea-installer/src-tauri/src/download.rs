// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ISO download with SHA-256 verification and progress events.
//!
//! Downloads the NixForHumanity ISO from GitHub Releases, emitting
//! `download-progress` events to the Tauri frontend for UI updates.

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::Emitter;

/// GitHub Releases URL for the NixForHumanity ISO.
const ISO_URL: &str = "https://github.com/Luminous-Dynamics/nixforhumanity/releases/download/v0.1.0/nixos-minimal-26.05pre-git-x86_64-linux.iso";

/// Expected SHA-256 hash of the ISO (set after first verified build).
/// When empty, hash verification is skipped (download-only mode).
const ISO_SHA256: &str = "";

/// Download the NixForHumanity ISO to `dest_dir`, returning the final path.
///
/// Emits `download-progress` events with `{ downloaded, total, percent }` payload.
/// If the ISO already exists and its hash matches, the download is skipped.
pub async fn download_iso(app: tauri::AppHandle, dest_dir: &Path) -> Result<PathBuf, String> {
    let dest = dest_dir.join("nixforhumanity.iso");

    // If already downloaded, optionally verify hash and skip re-download
    if dest.exists() {
        if !ISO_SHA256.is_empty() {
            if verify_hash(&dest, ISO_SHA256)? {
                let _ = app.emit(
                    "download-progress",
                    serde_json::json!({
                        "downloaded": 0, "total": 0, "percent": 100,
                        "status": "cached"
                    }),
                );
                return Ok(dest);
            }
            // Hash mismatch — re-download
        } else {
            // No hash to verify; assume cached ISO is good
            let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            if size > 100_000_000 {
                let _ = app.emit(
                    "download-progress",
                    serde_json::json!({
                        "downloaded": size, "total": size, "percent": 100,
                        "status": "cached"
                    }),
                );
                return Ok(dest);
            }
        }
    }

    let response = reqwest::get(ISO_URL)
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed: HTTP {}", response.status()));
    }

    let total = response.content_length().unwrap_or(0);

    let _ = app.emit(
        "download-progress",
        serde_json::json!({
            "downloaded": 0_u64, "total": total, "percent": 0_u32,
            "status": "downloading"
        }),
    );

    let mut file =
        std::fs::File::create(&dest).map_err(|e| format!("Cannot create file: {}", e))?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_emitted: u64 = 0;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
        file.write_all(&chunk)
            .map_err(|e| format!("Write error: {}", e))?;
        hasher.update(&chunk);
        downloaded += chunk.len() as u64;

        // Emit progress every ~1 MB to avoid flooding the event bus
        if downloaded - last_emitted >= 1_048_576 {
            last_emitted = downloaded;
            let percent = if total > 0 {
                (downloaded as f64 / total as f64 * 100.0) as u32
            } else {
                0
            };
            let _ = app.emit(
                "download-progress",
                serde_json::json!({
                    "downloaded": downloaded,
                    "total": total,
                    "percent": percent,
                    "status": "downloading"
                }),
            );
        }
    }

    // Final progress event
    let _ = app.emit(
        "download-progress",
        serde_json::json!({
            "downloaded": downloaded, "total": total, "percent": 100_u32,
            "status": "complete"
        }),
    );

    // Verify hash if we have one
    if !ISO_SHA256.is_empty() {
        let hash = format!("{:x}", hasher.finalize());
        if hash != ISO_SHA256 {
            // Remove corrupt download
            let _ = std::fs::remove_file(&dest);
            return Err(format!(
                "SHA-256 mismatch: expected {}, got {}",
                ISO_SHA256, hash
            ));
        }
    }

    Ok(dest)
}

/// Verify that a file matches the expected SHA-256 hash.
fn verify_hash(path: &Path, expected: &str) -> Result<bool, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open for hash: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Read error during hash: {}", e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash = format!("{:x}", hasher.finalize());
    Ok(hash == expected)
}