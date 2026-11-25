use std::env;
use std::fs;
use std::io::{self, Read};

use symthaea_sat_trace_verifier::{compute_manifest_digest, verify_trace, verify_trace_with_anchor};

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
        .ok_or("usage: sat-trace-verify <run-id> <manifest.json> <events.jsonl|-> [expected-head-digest]")?;
    let manifest_path = args.next().ok_or("missing frozen manifest path")?;
    let input = args
        .next()
        .ok_or("missing events.jsonl path (or - for stdin)")?;
    let expected_head_digest = args.next();
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

    let report = match expected_head_digest {
        Some(anchor) => verify_trace_with_anchor(
            &jsonl,
            &run_id,
            &manifest_digest,
            &anchor,
        )?,
        None => verify_trace(&jsonl, &run_id, &manifest_digest)?,
    };
    println!("run_id={}", report.run_id);
    println!("manifest_digest={}", report.manifest_digest);
    println!("events={}", report.event_count);
    println!("event_chain_consistent={}", report.event_chain_consistent);
    println!("head_digest={}", report.head_digest);
    println!(
        "head_digest_matches_supplied_anchor={}",
        report.head_digest_matches_supplied_anchor
    );
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
