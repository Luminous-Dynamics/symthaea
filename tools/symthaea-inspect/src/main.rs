/*!
 * Symthaea Inspector - Observability tool for consciousness-first AI
 *
 * This tool provides visibility into Symthaea's internal dynamics:
 * - Router selection decisions
 * - Workspace ignition events
 * - Φ (integrated information) measurements
 * - Primitive activation patterns
 * - Security authorization decisions
 */

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

mod trace;
mod export;
mod stats;

use trace::{Trace, Event};

/// Symthaea Inspector - Observability for consciousness-first AI
#[derive(Parser)]
#[command(name = "symthaea-inspect")]
#[command(about = "Inspect, replay, and analyze Symthaea execution traces", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Capture a trace from a running Symthaea instance
    Capture {
        /// Output file path
        #[arg(short, long, default_value = "trace.json")]
        output: PathBuf,

        /// Session ID to capture (if multiple instances running)
        #[arg(short, long)]
        session: Option<String>,
    },

    /// Replay a captured trace
    Replay {
        /// Trace file to replay
        trace: PathBuf,

        /// Interactive mode (step through events)
        #[arg(short, long)]
        interactive: bool,

        /// Start from event number
        #[arg(long, default_value = "0")]
        from: usize,

        /// Stop at event number (default: all)
        #[arg(long)]
        to: Option<usize>,
    },

    /// Export trace data to various formats
    Export {
        /// Trace file
        trace: PathBuf,

        /// Metric to export (phi, free_energy, confidence, etc.)
        #[arg(short, long)]
        metric: String,

        /// Output format (csv, json, jsonl)
        #[arg(short, long, default_value = "csv")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show statistics about a trace
    Stats {
        /// Trace file
        trace: PathBuf,

        /// Show detailed breakdown
        #[arg(short, long)]
        detailed: bool,
    },

    /// Monitor a running Symthaea instance (live mode)
    #[cfg(feature = "live")]
    Monitor {
        /// Trace file to watch (written by Symthaea)
        #[arg(short, long, default_value = "trace.json")]
        trace: PathBuf,

        /// Update interval in milliseconds
        #[arg(short, long, default_value = "100")]
        interval: u64,
    },

    /// Validate a trace file format
    Validate {
        /// Trace file to validate
        trace: PathBuf,

        /// Show detailed validation errors
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Capture { output, session } => {
            capture_trace(output, session)
        }
        Commands::Replay { trace, interactive, from, to } => {
            replay_trace(trace, interactive, from, to)
        }
        Commands::Export { trace, metric, format, output } => {
            export::export_metric(&trace, &metric, &format, output.as_ref())
        }
        Commands::Stats { trace, detailed } => {
            stats::show_stats(&trace, detailed)
        }
        #[cfg(feature = "live")]
        Commands::Monitor { trace, interval } => {
            monitor_live(trace, interval)
        }
        Commands::Validate { trace, verbose } => {
            validate_trace(trace, verbose)
        }
    }
}

fn capture_trace(output: PathBuf, _session: Option<String>) -> Result<()> {
    println!("📹 Capturing trace to: {}", output.display());
    println!();
    println!("⚠️  Note: Trace capture requires integration with Symthaea core.");
    println!("   This feature will be implemented in Sprint 1.");
    println!();
    println!("Expected usage:");
    println!("  1. Enable tracing in Symthaea: SYMTHAEA_TRACE=1");
    println!("  2. Run your Symthaea application");
    println!("  3. Trace will be written to {}", output.display());

    Ok(())
}

fn replay_trace(trace_path: PathBuf, interactive: bool, from: usize, to: Option<usize>) -> Result<()> {
    println!("▶️  Replaying trace: {}", trace_path.display());
    println!();

    let trace = Trace::load(&trace_path)
        .context("Failed to load trace file")?;

    println!("Session: {}", trace.session_id);
    println!("Started: {}", trace.timestamp_start);
    println!("Events: {}", trace.events.len());
    println!();

    let end = to.unwrap_or(trace.events.len());
    let events = &trace.events[from..end.min(trace.events.len())];

    if interactive {
        replay_interactive(events)?;
    } else {
        replay_sequential(events)?;
    }

    Ok(())
}

fn replay_sequential(events: &[Event]) -> Result<()> {
    for (idx, event) in events.iter().enumerate() {
        println!("[{}] {} - {}", idx, event.timestamp, event.event_type);

        match &event.event_type {
            trace::EventType::RouterSelection => {
                if let Some(data) = &event.data {
                    if let Ok(router_data) = serde_json::from_value::<trace::RouterSelectionData>(data.clone()) {
                        println!("  Input: {}", router_data.input);
                        println!("  Selected: {} (confidence: {:.2}%)",
                            router_data.selected_router, router_data.confidence * 100.0);
                    }
                }
            }
            trace::EventType::WorkspaceIgnition => {
                if let Some(data) = &event.data {
                    if let Ok(ws_data) = serde_json::from_value::<trace::WorkspaceIgnitionData>(data.clone()) {
                        println!("  Φ: {:.3}", ws_data.phi);
                        println!("  Coalition size: {}", ws_data.coalition_size);
                        println!("  Active primitives: {:?}", ws_data.active_primitives);
                    }
                }
            }
            trace::EventType::ResponseGenerated => {
                if let Some(data) = &event.data {
                    if let Ok(resp_data) = serde_json::from_value::<trace::ResponseData>(data.clone()) {
                        println!("  Response: {}", resp_data.content);
                        println!("  Confidence: {:.2}%", resp_data.confidence * 100.0);
                    }
                }
            }
            _ => {}
        }
        println!();
    }

    Ok(())
}

fn replay_interactive(events: &[Event]) -> Result<()> {
    use std::io::{self, Write};

    println!("Interactive mode: Press ENTER to step through events, 'q' to quit");
    println!();

    for (idx, event) in events.iter().enumerate() {
        println!("[{}] {} - {}", idx, event.timestamp, event.event_type);

        // Show event details
        if let Some(data) = &event.data {
            println!("{}", serde_json::to_string_pretty(data)?);
        }
        println!();

        // Wait for user input
        print!("Press ENTER to continue (or 'q' to quit): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "q" {
            break;
        }
    }

    Ok(())
}

#[cfg(feature = "live")]
fn monitor_live(trace_path: PathBuf, interval: u64) -> Result<()> {
    use notify::{Watcher, RecursiveMode, watcher};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    println!("👁️  Monitoring: {}", trace_path.display());
    println!("   Update interval: {}ms", interval);
    println!();

    let (tx, rx) = channel();
    let mut watcher = watcher(tx, Duration::from_millis(interval))?;
    watcher.watch(&trace_path, RecursiveMode::NonRecursive)?;

    let mut last_event_count = 0;

    loop {
        match rx.recv_timeout(Duration::from_millis(interval)) {
            Ok(_) => {
                // File changed, reload and show new events
                if let Ok(trace) = Trace::load(&trace_path) {
                    if trace.events.len() > last_event_count {
                        let new_events = &trace.events[last_event_count..];
                        replay_sequential(new_events)?;
                        last_event_count = trace.events.len();
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Normal timeout, continue
            }
            Err(e) => {
                eprintln!("Watch error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

fn validate_trace(trace_path: PathBuf, verbose: bool) -> Result<()> {
    println!("🔍 Validating trace: {}", trace_path.display());
    println!();

    match Trace::load(&trace_path) {
        Ok(trace) => {
            println!("✅ Trace is valid");
            println!();
            println!("Details:");
            println!("  Version: {}", trace.version);
            println!("  Session: {}", trace.session_id);
            println!("  Events: {}", trace.events.len());
            println!("  Duration: {} - {}", trace.timestamp_start,
                trace.timestamp_end.as_ref().unwrap_or(&"(ongoing)".to_string()));

            if verbose {
                println!();
                println!("Event types:");
                let mut type_counts = std::collections::HashMap::new();
                for event in &trace.events {
                    *type_counts.entry(event.event_type.clone()).or_insert(0) += 1;
                }
                for (event_type, count) in type_counts {
                    println!("  {}: {}", event_type, count);
                }
            }

            Ok(())
        }
        Err(e) => {
            println!("❌ Trace validation failed");
            println!();
            println!("Error: {}", e);

            if verbose {
                println!();
                println!("Detailed error:");
                println!("{:?}", e);
            }

            Err(e)
        }
    }
}
