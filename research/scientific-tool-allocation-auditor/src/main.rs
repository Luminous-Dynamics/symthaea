use scientific_tool_allocation_auditor::audit;
use std::{env, fs, process};

fn main() {
    if let Err(error) = run() {
        eprintln!("scientific-tool-allocation-auditor: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: scientific-tool-allocation-auditor MANIFEST.json TRIALS.jsonl".into());
    }
    let manifest = fs::read(&args[0])?;
    let trials = fs::read(&args[1])?;
    let report = audit(&manifest, &trials)?;
    if !report.verify_receipt() {
        return Err("generated receipt did not verify".into());
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
