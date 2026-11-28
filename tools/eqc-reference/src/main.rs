mod eqc;
pub mod observer;
pub mod qualification;

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;

fn next_path(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> anyhow::Result<PathBuf> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a path"))
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> anyhow::Result<String> {
    args.next()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a non-empty value"))
}

fn write_report(
    report: &qualification::QualificationReport,
    output: Option<PathBuf>,
    repository_root: &std::path::Path,
) -> anyhow::Result<()> {
    let encoded = serde_json::to_vec_pretty(report)?;
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
        println!("EQC qualification report written to {}", output_path.display());
    } else {
        println!("{}", String::from_utf8(encoded)?);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let mut corpus = PathBuf::from("docs/research/eqc-seed-corpus-v0.1.json");
    let mut output = None;
    let mut observation = None;
    let mut task_id = None;
    let mut observer_envelope = None;
    let mut observer_trust_store = None;
    let mut expected_trust_store_sha256 = None;
    let mut run_id = None;
    let mut challenge_nonce = None;
    let mut verification_time = None;
    let mut challenge_ledger = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--corpus" => corpus = next_path(&mut args, "--corpus")?,
            "--out" => output = Some(next_path(&mut args, "--out")?),
            "--observation" => observation = Some(next_path(&mut args, "--observation")?),
            "--task-id" => task_id = Some(next_value(&mut args, "--task-id")?),
            "--observer-envelope" => {
                observer_envelope = Some(next_path(&mut args, "--observer-envelope")?)
            }
            "--observer-trust-store" => {
                observer_trust_store = Some(next_path(&mut args, "--observer-trust-store")?)
            }
            "--trust-store-sha256" => {
                expected_trust_store_sha256 =
                    Some(next_value(&mut args, "--trust-store-sha256")?)
            }
            "--run-id" => run_id = Some(next_value(&mut args, "--run-id")?),
            "--challenge-nonce" => {
                challenge_nonce = Some(next_value(&mut args, "--challenge-nonce")?)
            }
            "--verification-time" => {
                verification_time = Some(next_value(&mut args, "--verification-time")?)
            }
            "--challenge-ledger" => {
                challenge_ledger = Some(next_path(&mut args, "--challenge-ledger")?)
            }
            "--help" | "-h" => {
                println!(
                    "eqc-reference [--corpus PATH] [--out PATH] [--observation PATH --task-id ID]\n\
                     Optional independent qualification requires all of:\n\
                     --observer-envelope PATH --observer-trust-store PATH\n\
                     --trust-store-sha256 SHA256 --run-id ID --challenge-nonce HEX\n\
                     --verification-time UTC --challenge-ledger PATH\n\
                     Trust-store digest, run challenge and verification time must be supplied by the evaluator out-of-band."
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

    let observer_arguments = [
        observer_envelope.is_some(),
        observer_trust_store.is_some(),
        expected_trust_store_sha256.is_some(),
        run_id.is_some(),
        challenge_nonce.is_some(),
        verification_time.is_some(),
        challenge_ledger.is_some(),
    ];
    let observer_argument_count = observer_arguments.iter().filter(|present| **present).count();

    if let Some(observation_path) = observation {
        let task_id = task_id.ok_or_else(|| anyhow::anyhow!("--observation requires --task-id"))?;
        // Validate the frozen corpus and its local fixtures before selecting
        // the task. An adapter cannot substitute or mutate the graded task.
        let _validated_seed = eqc::validate_corpus_file(&corpus_path, &repository_root)?;
        let corpus_bytes = fs::read(&corpus_path)?;
        let corpus_sha256 = format!("{:x}", Sha256::digest(&corpus_bytes));
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

        let report = match observer_argument_count {
            0 => qualification::qualify_observation(task, &observation_value)?,
            7 => {
                let envelope_path = observer_envelope.expect("count established all observer args");
                let trust_store_path =
                    observer_trust_store.expect("count established all observer args");
                let trust_store_pin = expected_trust_store_sha256
                    .expect("count established all observer args");
                let run_id = run_id.expect("count established all observer args");
                let challenge_nonce =
                    challenge_nonce.expect("count established all observer args");
                let verification_time =
                    verification_time.expect("count established all observer args");
                let challenge_ledger =
                    challenge_ledger.expect("count established all observer args");

                let envelope_bytes = fs::read(&envelope_path)
                    .with_context(|| format!("read observer envelope {}", envelope_path.display()))?;
                let envelope: serde_json::Value = serde_json::from_slice(&envelope_bytes)
                    .context("parse observer envelope")?;
                let trust_store_bytes = fs::read(&trust_store_path).with_context(|| {
                    format!("read evaluator trust store {}", trust_store_path.display())
                })?;
                let trust_store: serde_json::Value =
                    serde_json::from_slice(&trust_store_bytes).context("parse evaluator trust store")?;

                let context = observer::ObserverVerificationContext {
                    task,
                    observation: &observation_value,
                    run_id: &run_id,
                    challenge_nonce: &challenge_nonce,
                    verification_time: &verification_time,
                    task_corpus_sha256: &corpus_sha256,
                    expected_trust_store_sha256: &trust_store_pin,
                    observation_bytes: &observation_bytes,
                };
                let verified = observer::verify_and_consume(
                    &envelope,
                    &trust_store,
                    &trust_store_bytes,
                    &context,
                    &repository_root,
                    &challenge_ledger,
                )?;
                qualification::qualify_observation_with_observer(
                    task,
                    &observation_value,
                    verified,
                )?
            }
            count => anyhow::bail!(
                "independent observer verification requires all seven observer arguments; received {count}"
            ),
        };

        write_report(&report, output, &repository_root)?;
        if !report.qualified {
            std::process::exit(2);
        }
        return Ok(());
    }

    if task_id.is_some() {
        anyhow::bail!("--task-id is only valid together with --observation");
    }
    if observer_argument_count != 0 {
        anyhow::bail!("observer verification flags require --observation and --task-id");
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
