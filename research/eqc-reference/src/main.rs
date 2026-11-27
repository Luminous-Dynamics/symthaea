// SPDX-License-Identifier: Apache-2.0
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use symthaea_eqc_reference::validate_corpus_file;

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 2 {
        bail!("usage: symthaea-eqc-reference [corpus-path] [output-json-path]");
    }

    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("standalone EQC package must live under research/")?
        .canonicalize()
        .context("resolve repository root")?;

    let corpus_arg = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/research/eqc-seed-corpus-v0.1.json"));
    let corpus_path = if corpus_arg.is_absolute() {
        corpus_arg
    } else {
        repository_root.join(corpus_arg)
    };

    let report = validate_corpus_file(&corpus_path, &repository_root)?;
    let encoded = serde_json::to_vec_pretty(&report)?;

    if let Some(output_arg) = args.get(1) {
        let output_arg = PathBuf::from(output_arg);
        let output_path = if output_arg.is_absolute() {
            output_arg
        } else {
            repository_root.join(output_arg)
        };
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&output_path, &encoded)
            .with_context(|| format!("write EQC report {}", output_path.display()))?;
        println!("EQC reference report written to {}", output_path.display());
    } else {
        println!("{}", String::from_utf8(encoded)?);
    }
    Ok(())
}
