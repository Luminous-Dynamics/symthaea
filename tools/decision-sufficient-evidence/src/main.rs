use std::{env, fs, process};

use decision_sufficient_evidence::plan_evidence_json;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: decision-sufficient-evidence <problem.json>");
        process::exit(2);
    });
    let input = fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("ERROR D3-IO-001 {path}: {error}");
        process::exit(2);
    });
    match plan_evidence_json(&input) {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).expect("serialize planner report")),
        Err(error) => {
            eprintln!("ERROR D3-PLAN-001 {error}");
            process::exit(1);
        }
    }
}
