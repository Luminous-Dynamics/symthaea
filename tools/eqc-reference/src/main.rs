mod eqc;
pub mod observer;
pub mod qualification;

use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NonceLedgerDocument {
    schema: String,
    nonces: Vec<String>,
}

struct NonceLedgerLock {
    path: PathBuf,
}

fn non_empty_parent(path: &Path) -> Option<&Path> {
    path.parent().filter(|parent| !parent.as_os_str().is_empty())
}

impl Drop for NonceLedgerLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn acquire_nonce_ledger_lock(path: &Path) -> anyhow::Result<NonceLedgerLock> {
    let lock_path = PathBuf::from(format!("{}.lock", path.display()));
    if let Some(parent) = non_empty_parent(&lock_path) {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .with_context(|| {
            format!(
                "acquire exclusive observer nonce ledger lock {}; stale locks require operator review",
                lock_path.display()
            )
        })?;
    writeln!(file, "pid={}", std::process::id())?;
    file.sync_all()?;
    Ok(NonceLedgerLock { path: lock_path })
}

fn read_nonce_ledger(path: &Path) -> anyhow::Result<observer::ObserverNonceLedger> {
    if !path.exists() {
        return Ok(observer::ObserverNonceLedger::default());
    }
    let bytes = fs::read(path).with_context(|| format!("read nonce ledger {}", path.display()))?;
    let document: NonceLedgerDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse nonce ledger {}", path.display()))?;
    if document.schema != "luminous.eqc.observer-nonce-ledger.v0.1" {
        anyhow::bail!("unsupported observer nonce ledger schema");
    }
    observer::ObserverNonceLedger::from_nonces(document.nonces)
}

fn write_nonce_ledger(
    path: &Path,
    ledger: &observer::ObserverNonceLedger,
) -> anyhow::Result<()> {
    if let Some(parent) = non_empty_parent(path) {
        fs::create_dir_all(parent)?;
    }
    let document = NonceLedgerDocument {
        schema: "luminous.eqc.observer-nonce-ledger.v0.1".to_owned(),
        nonces: ledger.sorted_nonces(),
    };
    let bytes = serde_json::to_vec_pretty(&document)?;
    let temp_path = PathBuf::from(format!("{}.tmp-{}", path.display(), std::process::id()));
    let mut temp = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .with_context(|| format!("create temporary nonce ledger {}", temp_path.display()))?;
    temp.write_all(&bytes)?;
    temp.sync_all()?;
    drop(temp);
    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "atomically replace nonce ledger {} from {}",
            path.display(),
            temp_path.display()
        )
    })?;
    let parent = non_empty_parent(path).unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn write_output_or_stdout(output: Option<PathBuf>, repository_root: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(output) = output {
        let output_path = if output.is_absolute() {
            output
        } else {
            repository_root.join(output)
        };
        if let Some(parent) = non_empty_parent(&output_path) {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output_path, bytes)?;
        println!("EQC report written to {}", output_path.display());
    } else {
        println!("{}", String::from_utf8(bytes.to_vec())?);
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
    let mut observer_trust_policy = None;
    let mut observer_expectation = None;
    let mut nonce_ledger = None;

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
            "--observer-envelope" => {
                observer_envelope = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--observer-envelope requires a path"))?,
                ));
            }
            "--observer-trust-policy" => {
                observer_trust_policy = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--observer-trust-policy requires a path"))?,
                ));
            }
            "--observer-expectation" => {
                observer_expectation = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--observer-expectation requires a path"))?,
                ));
            }
            "--nonce-ledger" => {
                nonce_ledger = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--nonce-ledger requires a path"))?,
                ));
            }
            "--help" | "-h" => {
                println!(
                    "eqc-reference [--corpus PATH] [--out PATH] [--observation PATH --task-id ID]\n\
                     or --observer-envelope PATH --observer-trust-policy PATH --observer-expectation PATH --nonce-ledger PATH\n\
                     Validates the synthetic seed, an adapter observation, or a signed observer envelope.\n\
                     Observer trust policy and run expectation must be supplied independently of the envelope. Observer verification does not qualify a system until the host measurement runtime is independently audited."
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

    if observer_envelope.is_some()
        || observer_trust_policy.is_some()
        || observer_expectation.is_some()
        || nonce_ledger.is_some()
    {
        if observation.is_some() || task_id.is_some() {
            anyhow::bail!("signed observer verification cannot be combined with --observation or --task-id");
        }
        let envelope_path = observer_envelope
            .ok_or_else(|| anyhow::anyhow!("--observer-envelope is required"))?;
        let trust_path = observer_trust_policy
            .ok_or_else(|| anyhow::anyhow!("--observer-trust-policy is required"))?;
        let expectation_path = observer_expectation
            .ok_or_else(|| anyhow::anyhow!("--observer-expectation is required"))?;
        let ledger_path = nonce_ledger
            .ok_or_else(|| anyhow::anyhow!("--nonce-ledger is required"))?;

        let envelope_bytes = fs::read(&envelope_path)?;
        let trust_bytes = fs::read(&trust_path)?;
        let expectation_bytes = fs::read(&expectation_path)?;
        let envelope_value: serde_json::Value = serde_json::from_slice(&envelope_bytes)?;
        let trust_value: serde_json::Value = serde_json::from_slice(&trust_bytes)?;
        let expectation_value: serde_json::Value = serde_json::from_slice(&expectation_bytes)?;
        observer::validate_observer_contract_schemas(
            &envelope_value,
            &trust_value,
            &expectation_value,
            &repository_root,
        )?;
        let envelope: observer::SignedObserverEnvelope = serde_json::from_value(envelope_value)?;
        let trust: observer::ObserverTrustPolicy = serde_json::from_value(trust_value)?;
        let expectation: observer::ObserverRunExpectation =
            serde_json::from_value(expectation_value)?;

        // The exclusive lock makes this file-backed ledger safe for one
        // machine's concurrent evaluator processes. A distributed deployment
        // still needs a durable transactional service, not a shared NFS file.
        let _ledger_lock = acquire_nonce_ledger_lock(&ledger_path)?;
        let mut ledger = read_nonce_ledger(&ledger_path)?;
        let report = observer::verify_envelope(&envelope, &trust, &expectation, &mut ledger)?;
        write_nonce_ledger(&ledger_path, &ledger)?;
        let encoded = serde_json::to_vec_pretty(&report)?;
        write_output_or_stdout(output, &repository_root, &encoded)?;
        if !report.qualified {
            std::process::exit(2);
        }
        return Ok(());
    }

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
