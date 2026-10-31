//! Minimal, separately compiled BASALT fixture task.
//!
//! This executable has no catalog/oracle/receipt API. It can consume an input,
//! read and discard it, or not access it. The parent harness pins this binary's
//! SHA-256 before spawning it.

use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;

const INDEPENDENT_TASK_OUTPUT: &[u8] = b"basalt-fixture-task-output:v1:input-independent";

#[derive(Debug, Serialize)]
struct WorkerOutput {
    format_version: u32,
    task_mode: String,
    output_digest: String,
    bytes_read: u64,
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args_os().skip(1);
    let mut mode: Option<String> = None;
    let mut input: Option<PathBuf> = None;

    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--mode") => {
                let value = args.next().context("missing value after --mode")?;
                mode = Some(value.into_string().map_err(|_| anyhow::anyhow!("--mode must be UTF-8"))?);
            }
            Some("--input") => {
                let value = args.next().context("missing value after --input")?;
                input = Some(PathBuf::from(value));
            }
            _ => bail!("unsupported fixture-worker argument"),
        }
    }

    let mode = mode.context("--mode is required")?;
    let input = input.context("--input is required")?;
    let output = run(&mode, &input)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn run(mode: &str, input: &std::path::Path) -> anyhow::Result<WorkerOutput> {
    if !matches!(mode, "consume" | "read_discard" | "ignore") {
        bail!("worker mode must be consume, read_discard, or ignore");
    }

    let (output_digest, bytes_read) = match mode {
        "consume" => {
            let bytes = fs::read(input)
                .with_context(|| format!("fixture worker reading {}", input.display()))?;
            (sha256_tagged(&bytes), bytes.len() as u64)
        }
        "read_discard" => {
            let bytes = fs::read(input)
                .with_context(|| format!("fixture worker reading and discarding {}", input.display()))?;
            // The access is real, but the data-derived value is intentionally unused.
            (sha256_tagged(INDEPENDENT_TASK_OUTPUT), bytes.len() as u64)
        }
        "ignore" => {
            // Deliberately do not open, stat, canonicalize, or inspect the fixture path.
            (sha256_tagged(INDEPENDENT_TASK_OUTPUT), 0)
        }
        _ => unreachable!("worker mode was validated above"),
    };

    Ok(WorkerOutput {
        format_version: 1,
        task_mode: mode.to_string(),
        output_digest,
        bytes_read,
    })
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("sha256:{encoded}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock follows UNIX epoch")
            .as_nanos();
        env::temp_dir().join(format!("basalt-worker-{}-{nanos}-{label}", std::process::id()))
    }

    #[test]
    fn ignore_mode_does_not_open_or_stat_the_resource() {
        let missing = test_path("absent");
        let output = run("ignore", &missing).expect("ignore mode does not touch path");
        assert_eq!(output.bytes_read, 0);
        assert_eq!(output.output_digest, sha256_tagged(INDEPENDENT_TASK_OUTPUT));
        assert!(!missing.exists());
    }

    #[test]
    fn consume_mode_reads_and_hashes_exact_bytes() {
        let path = test_path("consume");
        let bytes = b"separate worker payload";
        fs::write(&path, bytes).expect("fixture file writes");
        let output = run("consume", &path).expect("consume mode reads file");
        assert_eq!(output.bytes_read, bytes.len() as u64);
        assert_eq!(output.output_digest, sha256_tagged(bytes));
        fs::remove_file(path).expect("fixture cleanup");
    }

    #[test]
    fn read_discard_mode_reads_input_but_keeps_output_independent() {
        let path = test_path("read-discard");
        fs::write(&path, b"before state").expect("baseline fixture writes");
        let before = run("read_discard", &path).expect("baseline is read and discarded");
        fs::write(&path, b"after---state").expect("mutated fixture writes");
        let after = run("read_discard", &path).expect("mutated input is read and discarded");
        assert_eq!(before.bytes_read, b"before state".len() as u64);
        assert_eq!(after.bytes_read, b"after---state".len() as u64);
        assert_eq!(before.output_digest, after.output_digest);
        assert_eq!(after.output_digest, sha256_tagged(INDEPENDENT_TASK_OUTPUT));
        fs::remove_file(path).expect("fixture cleanup");
    }
}
