//! Replay and microbenchmark a versioned Symthaea embodiment trace.
//!
//! Usage:
//!   cargo run -p symthaea-canvas --example embodiment_trace_replay -- --trace trace.json
//!   cargo run -p symthaea-canvas --example embodiment_trace_replay -- --trace trace.json --iterations 500
//!
//! This measures semantic reduction only. It does not measure SVG rasterization,
//! browser presentation, audio playback, GPU rendering, or human perception.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use symthaea_canvas::{EmbodimentTrace, EMBODIMENT_TRACE_SCHEMA_VERSION};

const MAX_TRACE_FILE_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_ITERATIONS: usize = 100;
const MAX_ITERATIONS: usize = 1_000_000;

#[derive(Serialize)]
struct BenchmarkReport {
    trace_schema_version: u16,
    session_id: String,
    event_count: usize,
    iterations: usize,
    replay_latency_min_ns: u64,
    replay_latency_p50_ns: u64,
    replay_latency_p95_ns: u64,
    replay_latency_max_ns: u64,
    replay_throughput_per_second: f64,
    summary: symthaea_canvas::EmbodimentTraceSummary,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("embodiment trace replay failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut trace_path: Option<PathBuf> = None;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut args = std::env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--trace" => {
                let value = args.next().ok_or("--trace requires a path")?;
                trace_path = Some(PathBuf::from(value));
            }
            "--iterations" => {
                let value = args.next().ok_or("--iterations requires an integer")?;
                iterations = value.parse::<usize>()?;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }

    if iterations == 0 || iterations > MAX_ITERATIONS {
        return Err(format!(
            "--iterations must be within 1..={MAX_ITERATIONS}"
        )
        .into());
    }
    let trace_path = trace_path.ok_or("--trace is required")?;
    let metadata = fs::metadata(&trace_path)?;
    if metadata.len() > MAX_TRACE_FILE_BYTES {
        return Err(format!(
            "trace file is {} bytes; maximum is {} bytes",
            metadata.len(),
            MAX_TRACE_FILE_BYTES
        )
        .into());
    }

    let bytes = fs::read(&trace_path)?;
    let trace: EmbodimentTrace = serde_json::from_slice(&bytes)?;
    if trace.schema_version != EMBODIMENT_TRACE_SCHEMA_VERSION {
        return Err(format!(
            "trace schema {} is unsupported; this binary supports {}",
            trace.schema_version, EMBODIMENT_TRACE_SCHEMA_VERSION
        )
        .into());
    }

    // Validate and summarize before timing so malformed input never produces
    // a benchmark report that could be mistaken for a valid run.
    let summary = trace.summarize()?;

    // Warm the allocator and code paths outside the reported sample window.
    for _ in 0..3 {
        std::hint::black_box(trace.replay()?);
    }

    let mut samples_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        std::hint::black_box(trace.replay()?);
        samples_ns.push(start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
    }

    samples_ns.sort_unstable();
    let total_ns: u128 = samples_ns.iter().map(|sample| u128::from(*sample)).sum();
    let elapsed_seconds = (total_ns as f64) / 1_000_000_000.0;
    let throughput = if elapsed_seconds > 0.0 {
        iterations as f64 / elapsed_seconds
    } else {
        0.0
    };

    let report = BenchmarkReport {
        trace_schema_version: trace.schema_version,
        session_id: trace.session_id,
        event_count: trace.events.len(),
        iterations,
        replay_latency_min_ns: samples_ns[0],
        replay_latency_p50_ns: percentile(&samples_ns, 50),
        replay_latency_p95_ns: percentile(&samples_ns, 95),
        replay_latency_max_ns: *samples_ns.last().unwrap_or(&0),
        replay_throughput_per_second: throughput,
        summary,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn percentile(sorted_samples: &[u64], percentile: usize) -> u64 {
    if sorted_samples.is_empty() {
        return 0;
    }
    let bounded = percentile.min(100);
    let index = ((sorted_samples.len() - 1) * bounded).div_ceil(100);
    sorted_samples[index.min(sorted_samples.len() - 1)]
}

fn print_usage() {
    println!(
        "Usage: embodiment_trace_replay --trace PATH [--iterations N]\n\
         Replays one valid trace and reports semantic-reducer latency (P50/P95).\n\
         This excludes SVG/browser/GPU/audio/human-perception latency."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank_for_sorted_samples() {
        let samples = [10, 20, 30, 40, 50];
        assert_eq!(percentile(&samples, 0), 10);
        assert_eq!(percentile(&samples, 50), 30);
        assert_eq!(percentile(&samples, 95), 50);
        assert_eq!(percentile(&samples, 100), 50);
    }

    #[test]
    fn empty_percentile_is_zero() {
        assert_eq!(percentile(&[], 95), 0);
    }
}
