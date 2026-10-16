// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inspect an SFZ numeric key/velocity mapping and verify local sample references.

use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use symthaea_muse::sfz_manifest::{inspect_sample_reference, parse_sfz_mapping};

fn main() {
    if let Err(error) = run() {
        eprintln!("muse_sfz_inspect: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut sfz_file = None;
    let mut sample_root = None;
    let mut source_label = None;
    let mut source_revision = None;
    let mut hash_samples = false;
    let mut args = std::env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--sfz" => {
                sfz_file = Some(PathBuf::from(required_value(&mut args, "--sfz")?));
            }
            "--sample-root" => {
                sample_root = Some(PathBuf::from(required_value(&mut args, "--sample-root")?));
            }
            "--source-label" => {
                source_label = Some(required_value(&mut args, "--source-label")?);
            }
            "--source-revision" => {
                source_revision = Some(required_value(&mut args, "--source-revision")?);
            }
            "--hash-samples" => hash_samples = true,
            "--help" | "-h" => {
                println!(
                    "Usage: muse_sfz_inspect --sfz FILE [--sample-root DIR] [--source-label LABEL] [--source-revision REV] [--hash-samples]\nParses a mapping subset only; does not render or decode audio."
                );
                return Ok(());
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {other}"),
                )
                .into());
            }
        }
    }

    let sfz_file = sfz_file.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "--sfz is required (see --help)")
    })?;
    let source = std::fs::read_to_string(&sfz_file)?;
    let mut manifest = parse_sfz_mapping(&source)?;
    manifest.declared_source_label = source_label;
    manifest.declared_source_revision = source_revision;

    let root = sample_root.unwrap_or_else(|| {
        sfz_file.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
    });
    let mut present = 0usize;
    let mut missing = 0usize;
    let mut symlink_or_non_file = 0usize;

    for region in &mut manifest.regions {
        match inspect_sample_reference(&root, &region.sample_path, hash_samples)? {
            Some((byte_len, sha256)) => {
                region.sample_asset_state = if hash_samples {
                    "present_hashed".to_owned()
                } else {
                    "present_unhashed".to_owned()
                };
                region.sample_byte_len = Some(byte_len);
                region.sample_sha256 = sha256;
                present += 1;
            }
            None => {
                let direct = root.join(&region.sample_path);
                let exists = std::fs::symlink_metadata(&direct).is_ok();
                region.sample_asset_state = if exists {
                    "symlink_or_non_regular_path_rejected".to_owned()
                } else {
                    "missing".to_owned()
                };
                if exists {
                    symlink_or_non_file += 1;
                } else {
                    missing += 1;
                }
            }
        }
    }
    manifest.unhandled_opcodes.sort();
    manifest.unhandled_opcodes.dedup();

    eprintln!(
        "regions={} present={} missing={} symlink_or_non_file={} audio_decode_evidence=not_performed",
        manifest.region_count, present, missing, symlink_or_non_file
    );
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn required_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, io::Error> {
    args.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{option} requires a value"),
        )
    })
}
