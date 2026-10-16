// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verify local asset bytes against a bioacoustic corpus manifest.
//!
//! This is the byte-integrity stage after metadata preflight. It does not
//! decode audio, authenticate human rights decisions, or evaluate a detector.

use clap::Parser;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use symthaea_communication::bioacoustic_corpus::{
    BioacousticCorpusManifest, validate_bioacoustic_corpus,
};

#[derive(Debug, Parser)]
#[command(
    name = "bioacoustic-asset-audit",
    about = "Hash local bioacoustic assets against an admitted corpus manifest"
)]
struct Args {
    /// JSON BioacousticCorpusManifest file.
    manifest: PathBuf,
    /// Directory containing root-relative paths in each recording's asset_uri.
    #[arg(long)]
    asset_root: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("BIOACOUSTIC_ASSET_AUDIT=FAIL");
        eprintln!("REASON={error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.manifest)
        .map_err(|error| format!("cannot read manifest {}: {error}", args.manifest.display()))?;
    let manifest: BioacousticCorpusManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse bioacoustic corpus manifest: {error}"))?;

    // Re-run admission checks before reading corpus assets. An asserted
    // approval is required, but this program cannot prove that it is lawful.
    validate_bioacoustic_corpus(
        &manifest,
        Default::default(),
    )?;
    let manifest_identity = manifest.computed_manifest_blake3()?;
    let asset_root = args.asset_root.canonicalize().map_err(|error| {
        format!(
            "cannot resolve asset root {}: {error}",
            args.asset_root.display()
        )
    })?;
    if !asset_root.is_dir() {
        return Err(format!("asset root is not a directory: {}", asset_root.display()));
    }

    let mut checked = 0usize;
    let mut failed = 0usize;
    for recording in &manifest.recordings {
        match verify_recording(&asset_root, &recording.asset_uri, &recording.content_sha256) {
            Ok(actual_hash) => {
                checked += 1;
                println!(
                    "ASSET_SHA256=PASS sample_id={} sha256={actual_hash}",
                    recording.sample_id
                );
            }
            Err(error) => {
                failed += 1;
                eprintln!(
                    "ASSET_SHA256=FAIL sample_id={} reason={error}",
                    recording.sample_id
                );
            }
        }
    }

    if failed > 0 {
        return Err(format!(
            "{failed} of {} recording assets failed local byte-integrity checks",
            manifest.recordings.len()
        ));
    }

    println!("BIOACOUSTIC_ASSET_AUDIT=PASS");
    println!("CORPUS_SCHEMA_VERSION={}", manifest.schema_version);
    println!("DATASET_ID={}", manifest.dataset_id);
    println!("DATASET_REVISION={}", manifest.dataset_revision);
    println!("MANIFEST_BLAKE3={manifest_identity}");
    println!("ASSET_COUNT={checked}");
    println!("ASSET_BYTES_VERIFIED=true");
    println!("AUDIO_HEADER_METADATA_VERIFIED=false");
    println!("RIGHTS_DECISION_AUTHENTICATED=false");
    println!("MODEL_OR_DETECTION_PERFORMANCE_EVALUATED=false");
    Ok(())
}

fn verify_recording(root: &Path, asset_uri: &str, expected_sha256: &str) -> Result<String, String> {
    let relative_path = validate_relative_asset_uri(asset_uri)?;
    let unresolved_path = root.join(relative_path);
    let resolved_path = unresolved_path
        .canonicalize()
        .map_err(|error| format!("cannot resolve asset path {}: {error}", unresolved_path.display()))?;

    // Canonicalization also prevents a symlink inside asset_root from escaping it.
    if !resolved_path.starts_with(root) {
        return Err("resolved asset path escapes --asset-root".into());
    }
    if !resolved_path.is_file() {
        return Err("asset path is not a regular file".into());
    }

    let file = File::open(&resolved_path)
        .map_err(|error| format!("cannot open asset {}: {error}", resolved_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("cannot read asset {}: {error}", resolved_path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    let actual_sha256 = format!("{:x}", hasher.finalize());
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(format!(
            "SHA-256 mismatch: manifest={} actual={actual_sha256}",
            expected_sha256
        ));
    }
    Ok(actual_sha256)
}

fn validate_relative_asset_uri(asset_uri: &str) -> Result<PathBuf, String> {
    if asset_uri.trim().is_empty()
        || asset_uri.trim() != asset_uri
        || asset_uri.contains("://")
        || asset_uri.contains('\\')
        || asset_uri.contains(':')
        || asset_uri.starts_with('/')
    {
        return Err("asset_uri must be a clean, root-relative local path".into());
    }

    let path = Path::new(asset_uri);
    if asset_uri
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
        || path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(
            "asset_uri must use normal relative path components without empty, '.', or '..' segments"
                .into(),
        );
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_nested_relative_asset_paths() {
        assert_eq!(
            validate_relative_asset_uri("recordings/site-a/session-01.wav").unwrap(),
            PathBuf::from("recordings/site-a/session-01.wav")
        );
    }

    fn temporary_directory() -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "bioacoustic-asset-audit-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn verifies_exact_bytes_and_rejects_a_mismatched_hash() {
        let root = temporary_directory();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("sample.wav"), b"abc").unwrap();
        let canonical_root = root.canonicalize().unwrap();
        let expected = format!("{:x}", Sha256::digest(b"abc"));

        assert_eq!(
            verify_recording(&canonical_root, "sample.wav", &expected).unwrap(),
            expected
        );
        assert!(verify_recording(&canonical_root, "sample.wav", &"0".repeat(64)).is_err());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_that_escape_the_asset_root() {
        use std::os::unix::fs::symlink;

        let parent = temporary_directory();
        let root = parent.join("assets");
        std::fs::create_dir_all(&root).unwrap();
        let external = parent.join("outside.wav");
        std::fs::write(&external, b"outside").unwrap();
        symlink(&external, root.join("escape.wav")).unwrap();
        let canonical_root = root.canonicalize().unwrap();

        let result = verify_recording(
            &canonical_root,
            "escape.wav",
            &format!("{:x}", Sha256::digest(b"outside")),
        );
        assert!(result.is_err_and(|error| error.contains("escapes --asset-root")));

        std::fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn rejects_remote_absolute_and_traversal_paths() {
        for path in [
            "",
            "https://example.org/audio.wav",
            "/tmp/audio.wav",
            "../outside.wav",
            "recordings/../../outside.wav",
            "recordings/./audio.wav",
            "recordings//audio.wav",
            "recordings/",
            "./audio.wav",
            "recordings\\audio.wav",
            "C:/audio.wav",
            " audio.wav",
        ] {
            assert!(
                validate_relative_asset_uri(path).is_err(),
                "path must be rejected: {path:?}"
            );
        }
    }
}
