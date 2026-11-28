mod eqc;
pub mod observer;
pub mod qualification;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let mut corpus = PathBuf::from("docs/research/eqc-seed-corpus-v0.1.json");
    let mut output = None;
    let mut observation = None;
    let mut task_id = None;

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
            "--observation" => {
                observation =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--observation requires a path")
                    })?));
            }
            "--task-id" => {
                task_id = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--task-id requires a value"))?,
                );
            }
            "--help" | "-h" => {
                println!(
                    "eqc-reference [--corpus PATH] [--out PATH] [--observation PATH --task-id ID]\n\
                     Validates the synthetic EQC seed or an adapter observation against a frozen task.\n\
                     Observation qualification requires capability, authority, input integrity, provenance, replay, and failure containment."
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

    if let Some(observation_path) = observation {
        let task_id = task_id.ok_or_else(|| anyhow::anyhow!("--observation requires --task-id"))?;
        // Never trust an adapter's comparison against an arbitrary or mutated
        // task definition. Re-validate the frozen corpus and all fixture
        // digests before selecting the task for system-observation grading.
        let _validated_seed = eqc::validate_corpus_file(&corpus_path, &repository_root)?;
        let corpus_bytes = fs::read(&corpus_path)?;
        let corpus_value: serde_json::Value = serde_json::from_slice(&corpus_bytes)?;
        let task = corpus_value
            .get("tasks")
            .and_then(serde_json::Value::as_array)
            .and_then(|tasks| {
                tasks.iter().find(|task| {
                    task.get("task_id").and_then(serde_json::Value::as_str)
                        == Some(task_id.as_str())
                })
            })
            .ok_or_else(|| anyhow::anyhow!("task_id not found in frozen corpus: {task_id}"))?;
        let observation_path = if observation_path.is_absolute() {
            observation_path
        } else {
            repository_root.join(observation_path)
        };
        let observation_bytes = fs::read(&observation_path)?;
        let observation_value: serde_json::Value = serde_json::from_slice(&observation_bytes)?;
        qualification::validate_observation_schema(&observation_value, &repository_root)?;
        let report = qualification::qualify_observation(task, &observation_value)?;
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
            println!(
                "EQC qualification report written to {}",
                output_path.display()
            );
        } else {
            println!("{}", String::from_utf8(encoded)?);
        }
        if !report.qualified {
            std::process::exit(2);
        }
        return Ok(());
    }

    if task_id.is_some() {
        anyhow::bail!("--task-id is only valid together with --observation");
    }

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
