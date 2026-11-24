//! Render a Mycelial Presence SVG from a versioned embodiment event trace.
//!
//! Usage:
//!   cargo run -p symthaea-canvas --example mycelial_presence_demo --
//!     --trace crates/domains/symthaea-canvas/tests/fixtures/embodiment_trace_v1.json
//!     --output /tmp/symthaea-presence.svg
//!
//! The SVG is a renderer output, not a claim that any cognitive activity or
//! subjective experience exists beyond the signals in the input trace.

use std::fs;
use std::path::PathBuf;

use symthaea_canvas::{
    EmbodimentTrace, MotionPreference, SvgRenderOptions, render_mycelial_presence,
};

const MAX_TRACE_FILE_BYTES: u64 = 16 * 1024 * 1024;

fn main() {
    if let Err(error) = run() {
        eprintln!("mycelial presence render failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut trace_path: Option<PathBuf> = None;
    let mut output_path = PathBuf::from("symthaea-presence.svg");
    let mut through_sequence: Option<u64> = None;
    let mut reduced_motion = false;
    let mut args = std::env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--trace" => {
                trace_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--trace requires a path".to_string())?,
                ));
            }
            "--output" => {
                output_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--output requires a path".to_string())?,
                );
            }
            "--through-sequence" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--through-sequence requires an integer".to_string())?;
                through_sequence = Some(value.parse::<u64>().map_err(|error| error.to_string())?);
            }
            "--reduced-motion" => reduced_motion = true,
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    let trace_path = trace_path.ok_or_else(|| "--trace is required".to_string())?;
    let metadata = fs::metadata(&trace_path).map_err(|error| error.to_string())?;
    if metadata.len() > MAX_TRACE_FILE_BYTES {
        return Err(format!(
            "trace file is {} bytes; maximum is {} bytes",
            metadata.len(),
            MAX_TRACE_FILE_BYTES
        ));
    }

    let bytes = fs::read(&trace_path).map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_TRACE_FILE_BYTES {
        return Err("trace grew beyond the configured size limit while reading".to_string());
    }
    let mut trace: EmbodimentTrace =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if let Some(sequence) = through_sequence {
        trace.events.retain(|event| event.sequence <= sequence);
        if trace.events.is_empty() {
            return Err(format!("no trace events have sequence <= {sequence}"));
        }
    }
    let snapshot = trace.replay().map_err(|error| error.to_string())?;

    let options = SvgRenderOptions {
        elapsed_seconds: 0.0,
        motion: if reduced_motion {
            MotionPreference::Reduced
        } else {
            MotionPreference::Full
        },
        instance_id: None,
    };
    let svg = render_mycelial_presence(&snapshot, options);
    fs::write(&output_path, svg.as_bytes()).map_err(|error| error.to_string())?;

    println!(
        "rendered session '{}' through sequence {:?} to {} ({} SVG bytes; reduced_motion={})",
        trace.session_id,
        through_sequence,
        output_path.display(),
        svg.len(),
        reduced_motion
    );
    Ok(())
}

fn print_usage() {
    println!(concat!(
        "Usage: mycelial_presence_demo --trace PATH [--through-sequence N] ",
        "[--output PATH] [--reduced-motion]\n",
        "Replays a validated trace prefix and exports a self-contained Mycelial Presence SVG."
    ));
}
