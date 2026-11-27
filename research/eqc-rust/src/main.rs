// SPDX-License-Identifier: AGPL-3.0-or-later

mod eqc;

use anyhow::{Context, bail};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn repository_root() -> anyhow::Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("standalone EQC crate must remain under research/")?;
    root.canonicalize()
        .with_context(|| format!("resolve repository root {}", root.display()))
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args_os().skip(1);
    let command = args
        .next()
        .and_then(|arg| arg.into_string().ok())
        .unwrap_or_default();

    if command != "eqc-validate" {
        bail!("usage: eqc-reference eqc-validate [--corpus PATH] [--out PATH]");
    }

    let root = repository_root()?;
    let mut corpus = PathBuf::from("docs/research/eqc-seed-corpus-v0.1.json");
    let mut output: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        let flag = arg
            .into_string()
            .map_err(|_| anyhow::anyhow!("CLI flags must be valid UTF-8"))?;
        let value = args
            .next()
            .context("each CLI flag requires a path value")?;
        match flag.as_str() {
            "--corpus" => corpus = PathBuf::from(value),
            "--out" => output = Some(PathBuf::from(value)),
            _ => bail!("unknown argument: {flag}"),
        }
    }

    let corpus_path = if corpus.is_absolute() {
        corpus
    } else {
        root.join(corpus)
    };
    let report = eqc::validate_corpus_file(&corpus_path, &root)?;
    let bytes = serde_json::to_vec_pretty(&report)?;

    if let Some(output) = output {
        let output_path = if output.is_absolute() {
            output
        } else {
            root.join(output)
        };
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        fs::write(&output_path, &bytes)
            .with_context(|| format!("write EQC report {}", output_path.display()))?;
        println!("EQC reference report written to {}", output_path.display());
    } else {
        println!("{}", String::from_utf8(bytes)?);
    }
    Ok(())
}
