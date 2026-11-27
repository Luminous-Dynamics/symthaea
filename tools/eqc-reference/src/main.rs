mod eqc;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let mut corpus = PathBuf::from("docs/research/eqc-seed-corpus-v0.1.json");
    let mut output = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--corpus" => {
                corpus = PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--corpus requires a path"))?,
                );
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--out requires a path"))?,
                ));
            }
            "--help" | "-h" => {
                println!(
                    "eqc-reference [--corpus PATH] [--out PATH]\n\
                     Validates the synthetic EQC v0.1 seed against the independent reference policy.\n\
                     This is reference-contract validation, not a system-under-test qualification."
                );
                return Ok(());
            }
            unknown => anyhow::bail!("unknown argument: {unknown}"),
        }
    }

    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let corpus_path = if corpus.is_absolute() {
        corpus
    } else {
        repository_root.join(corpus)
    };

    let report = eqc::validate_corpus_file(&corpus_path, &repository_root)?;
    let encoded = serde_json::to_vec_pretty(&report)?;

    if let Some(output) = output {
        let output_path = if output.is_absolute() {
            output
        } else {
            repository_root.join(output)
        };
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output_path, &encoded)?;
        println!("EQC reference report written to {}", output_path.display());
    } else {
        println!("{}", String::from_utf8(encoded)?);
    }

    Ok(())
}
