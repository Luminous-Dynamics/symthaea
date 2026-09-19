// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic Phase-A workspace inventory and exact target-file leak canary.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{env, fs, path::{Path, PathBuf}, process::ExitCode};

#[derive(Debug, Serialize)]
struct WorkspaceInventory {
    schema_version: u32,
    root_label: String,
    entries: Vec<WorkspaceEntry>,
}

#[derive(Debug, Serialize)]
struct WorkspaceEntry {
    relative_path: String,
    size_bytes: u64,
    sha256: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("MAG-QUAL-001 inventory error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
        return Err(
            "usage: mag_qual_inventory <workspace-root> <root-label> <sealed-target-sha256> <output.json>"
                .into(),
        );
    }
    let root = fs::canonicalize(&args[0])?;
    let root_label = args[1].trim();
    if root_label.is_empty() {
        return Err("root-label cannot be empty".into());
    }
    let sealed_target_sha = args[2].to_ascii_lowercase();
    validate_sha256(&sealed_target_sha)?;
    let output = PathBuf::from(&args[3]);

    // Avoid accidentally inventorying an old copy of the output artifact.
    if output.exists() {
        fs::remove_file(&output)?;
    }

    let mut entries = Vec::new();
    walk(&root, &root, &sealed_target_sha, &mut entries)?;
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let inventory = WorkspaceInventory {
        schema_version: 1,
        root_label: root_label.to_string(),
        entries,
    };
    let mut bytes = serde_json::to_vec_pretty(&inventory)?;
    bytes.push(b'\n');
    let inventory_sha = sha256_hex(&bytes);
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, bytes)?;

    println!("workspace_inventory_sha256={inventory_sha}");
    println!("workspace_inventory_path={}", output.display());
    Ok(())
}

fn walk(
    root: &Path,
    directory: &Path,
    sealed_target_sha: &str,
    entries: &mut Vec<WorkspaceEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut children = fs::read_dir(directory)?
        .collect::<Result<Vec<_>, _>>()?;
    children.sort_by_key(|entry| entry.file_name());

    for child in children {
        let path = child.path();
        let metadata = fs::symlink_metadata(&path)?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(format!(
                "symlink forbidden in strong Phase-A workspace inventory: {}",
                relative(root, &path)?
            )
            .into());
        }
        if file_type.is_dir() {
            walk(root, &path, sealed_target_sha, entries)?;
            continue;
        }
        if !file_type.is_file() {
            return Err(format!(
                "special file forbidden in strong Phase-A workspace inventory: {}",
                relative(root, &path)?
            )
            .into());
        }

        let bytes = fs::read(&path)?;
        let digest = sha256_hex(&bytes);
        if digest.eq_ignore_ascii_case(sealed_target_sha) {
            return Err(format!(
                "sealed target bytes are present in Phase-A workspace as {}",
                relative(root, &path)?
            )
            .into());
        }
        entries.push(WorkspaceEntry {
            relative_path: relative(root, &path)?,
            size_bytes: metadata.len(),
            sha256: digest,
        });
    }
    Ok(())
}

fn relative(root: &Path, path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let relative = path.strip_prefix(root)?;
    let text = relative
        .to_str()
        .ok_or("non-UTF-8 paths are forbidden in qualification workspace")?;
    Ok(text.replace('\\', "/"))
}

fn validate_sha256(value: &str) -> Result<(), Box<dyn std::error::Error>> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err("sealed-target SHA-256 must be exactly 64 hexadecimal characters".into())
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
