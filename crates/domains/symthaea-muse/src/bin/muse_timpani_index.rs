// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emit a deterministic metadata inventory for an unpacked timpani sample root.

use std::error::Error;
use std::io;
use std::path::PathBuf;
use symthaea_muse::percussion_samples::{index_timpani_samples, TimpaniIndexOptions};

fn main() {
    if let Err(error) = run() {
        eprintln!("muse_timpani_index: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut root = None;
    let mut source_label = None;
    let mut source_revision = None;
    let mut hash_content = false;
    let mut args = std::env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--root" => {
                root = Some(PathBuf::from(args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--root requires a path")
                })?));
            }
            "--source-label" => {
                source_label = Some(args.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--source-label requires a label",
                    )
                })?);
            }
            "--source-revision" => {
                source_revision = Some(args.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--source-revision requires a hexadecimal revision",
                    )
                })?);
            }
            "--hash-content" => hash_content = true,
            "--help" | "-h" => {
                println!(
                    "Usage: muse_timpani_index --root DIR [--source-label LABEL]\n       [--source-revision SHA] [--hash-content]\nIndexes filenames only; it does not decode audio or infer MIDI pitches."
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

    let root = root.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "--root is required (see --help)")
    })?;
    let options = TimpaniIndexOptions {
        source_label,
        source_revision,
        hash_content,
    };
    let report = index_timpani_samples(&root, &options)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
