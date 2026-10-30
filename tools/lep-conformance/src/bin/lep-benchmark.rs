use lep_conformance::{parse_json_unique_keys, validate_schema_only, verify_statement, PREDICATE_TYPE, SCHEMA_VERSION, STATEMENT_TYPE};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::hint::black_box;
use std::process::Command;
use std::process::ExitCode;
use std::time::Instant;

fn usage() -> &'static str {
    "Usage: lep-benchmark [target-bytes=1024] [iterations=1000]"
}

fn synthetic_statement() -> Value {
    let target = json!({
        "source": {
            "repository": "https://github.com/Luminous-Dynamics/symthaea",
            "revision": "754589fa14a63070cf75d8f9ec90078e673d626c"
        },
        "configurationDigest": { "sha256": "a".repeat(64) },
        "inputs": []
    });

    json!({
        "_type": STATEMENT_TYPE,
        "subject": [{
            "name": "synthetic-benchmark-report.json",
            "digest": { "sha256": "b".repeat(64) }
        }],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "schemaVersion": SCHEMA_VERSION,
            "claim": {
                "id": "LEP-PERF-SYNTHETIC-001",
                "kind": "test_result",
                "statement": "Synthetic validator performance workload; not a production measurement.",
                "scope": ["validator-performance"]
            },
            "target": target,
            "execution": {
                "status": "completed",
                "result": "passed",
                "runId": "synthetic-performance-run"
            },
            "verification": {
                "outcome": "passed",
                "verifier": { "id": "lep-benchmark", "version": "0.1.0" },
                "checkedTarget": target,
                "checks": [{ "id": "synthetic-control", "outcome": "passed" }]
            },
            "evidence": [],
            "limitations": []
        }
    })
}

fn sized_input(target_bytes: usize) -> Result<Vec<u8>, String> {
    let mut statement = synthetic_statement();
    let initial = serde_json::to_vec(&statement).map_err(|error| error.to_string())?;
    if target_bytes > initial.len() + 2 {
        let padding_len = target_bytes - initial.len() - 2;
        statement["predicate"]["limitations"] = json!([ "x".repeat(padding_len) ]);
    }

    serde_json::to_vec(&statement).map_err(|error| error.to_string())
}

fn measure<F>(iterations: usize, mut operation: F) -> Vec<u128>
where
    F: FnMut(),
{
    // Warm the process-local schema cache and allocator before recording latency.
    for _ in 0..20 {
        operation();
    }

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples
}

fn percentile(samples: &[u128], percentage: usize) -> u128 {
    let index = ((samples.len() - 1) * percentage) / 100;
    samples[index]
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn summarize(samples: &[u128]) -> Value {
    let p50 = percentile(samples, 50).max(1);
    let p95 = percentile(samples, 95);
    let mean = samples.iter().copied().sum::<u128>() / samples.len() as u128;

    json!({
        "p50_ns": p50,
        "p95_ns": p95,
        "mean_ns": mean,
        "throughput_records_per_second_at_p50": 1_000_000_000.0_f64 / p50 as f64
    })
}

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 2 {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }

    let target_bytes = match args.first().map(|value| value.parse::<usize>()).transpose() {
        Ok(Some(value)) if value > 0 => value,
        Ok(None) => 1024,
        _ => {
            eprintln!("target-bytes must be a positive integer. {}", usage());
            return ExitCode::from(2);
        }
    };
    let iterations = match args.get(1).map(|value| value.parse::<usize>()).transpose() {
        Ok(Some(value)) if value > 0 => value,
        Ok(None) => 1000,
        _ => {
            eprintln!("iterations must be a positive integer. {}", usage());
            return ExitCode::from(2);
        }
    };

    let bytes = match sized_input(target_bytes) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to create benchmark input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let input = match parse_json_unique_keys(&bytes) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("generated benchmark input failed strict parsing: {error}");
            return ExitCode::FAILURE;
        }
    };
    let predicate = match input.get("predicate") {
        Some(predicate) => predicate,
        None => {
            eprintln!("generated benchmark input has no predicate");
            return ExitCode::FAILURE;
        }
    };
    let schema_report = validate_schema_only(predicate);
    let full_report = verify_statement(&input);
    if !schema_report.conforms || !full_report.conforms {
        eprintln!(
            "generated benchmark input is not valid: schema={:?}; full={:?}",
            schema_report.violations, full_report.violations
        );
        return ExitCode::FAILURE;
    }

    let schema_input = bytes.clone();
    let schema_samples = measure(iterations, || {
        let statement = parse_json_unique_keys(&schema_input)
            .expect("benchmark input must remain valid JSON");
        let predicate = statement.get("predicate").expect("predicate must exist");
        black_box(validate_schema_only(predicate));
    });

    let full_input = bytes.clone();
    let full_samples = measure(iterations, || {
        let statement = parse_json_unique_keys(&full_input)
            .expect("benchmark input must remain valid JSON");
        black_box(verify_statement(&statement));
    });

    let report = json!({
        "suite": "LEP-PERF-0.1",
        "scope": "Synthetic local parsing + validation only; excludes signature verification, artifact I/O, and domain execution.",
        "requested_target_bytes": target_bytes,
        "actual_input_bytes": bytes.len(),
        "iterations": iterations,
        "warmup_iterations_per_path": 20,
        "environment": {
            "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "rustc_version": rustc_version()
        },
        "measurements": {
            "strict_parse_plus_schema": summarize(&schema_samples),
            "strict_parse_plus_schema_and_semantics": summarize(&full_samples)
        }
    });

    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("failed to serialize benchmark result: {error}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
