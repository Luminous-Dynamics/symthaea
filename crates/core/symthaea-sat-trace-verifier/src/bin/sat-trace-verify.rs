use std::env;
use std::fs;
use std::io::{self, Read};

use symthaea_sat_trace_verifier::{compute_manifest_digest, verify_trace};

fn main() {
    if let Err(error) = run() {
        eprintln!("SAT trace verification failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let run_id = args
        .next()
        .ok_or("usage: sat-trace-verify <run-id> <manifest.json> <events.jsonl|->")?;
    let manifest_path = args.next().ok_or("missing frozen manifest path")?;
    let input = args
        .next()
        .ok_or("missing events.jsonl path (or - for stdin)")?;
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let manifest_json = fs::read_to_string(manifest_path)?;
    let manifest_digest = compute_manifest_digest(&manifest_json)?;

    let jsonl = if input == "-" {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        fs::read_to_string(input)?
    };

    let report = verify_trace(&jsonl, &run_id, &manifest_digest)?;
    println!("run_id={}", report.run_id);
    println!("manifest_digest={}", report.manifest_digest);
    println!("events={}", report.event_count);
    println!("event_chain_verified={}", report.event_chain_verified);
    println!("authority_bindings_checked={}", report.authority_bindings_checked);
    println!("disclosure_bindings_checked={}", report.disclosure_bindings_checked);
    println!("sink_projections_checked={}", report.sink_projections_checked);
    println!("effect_submissions={}", report.effect_submissions);
    println!("terminal_effect_observations={}", report.terminal_effect_observations);
    println!("hold_required={}", report.hold_required);
    println!(
        "unresolved_idempotency_keys={}",
        report.unresolved_idempotency_keys.join(",")
    );
    println!("signatures_verified={}", report.signatures_verified);
    println!("system_qualified={}", report.system_qualified);
    Ok(())
}
