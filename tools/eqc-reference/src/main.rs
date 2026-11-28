// SPDX-License-Identifier: AGPL-3.0-or-later
use anyhow::{Context, bail};
use luminous_eqc_reference::validate_corpus_file;
use std::path::{Path, PathBuf};

fn parse_path(args: &[String], flag: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn resolve(root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .context("resolve repository root from standalone EQC workspace")?;

    let corpus = resolve(
        &root,
        parse_path(
            &args,
            "--corpus",
            "docs/research/eqc-seed-corpus-v0.1.json",
        ),
    );
    let report = validate_corpus_file(&corpus, &root)
        .with_context(|| format!("validate EQC corpus {}", corpus.display()))?;
    let output = serde_json::to_vec_pretty(&report)?;

    if let Some(path) = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|index| args.get(index + 1))
    {
        let output_path = resolve(&root, PathBuf::from(path));
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
        std::fs::write(&output_path, &output)
            .with_context(|| format!("write EQC report {}", output_path.display()))?;
        println!("EQC reference report written to {}", output_path.display());
    } else if args.iter().any(|arg| arg == "--out") {
        bail!("--out requires a path");
    } else {
        println!("{}", String::from_utf8(output)?);
    }
    Ok(())
}
