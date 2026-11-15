use std::{env, fs, process};

use temporal_replay_oracle::{compare_expected, evaluate_json};

fn usage() {
    eprintln!("Usage:");
    eprintln!("  temporal-replay-oracle <ledger.json> <query.json> [expected.json <query-key>]");
    eprintln!("This executable independently verifies the D2 ledger and reconstructs the requested actor/time view.");
    eprintln!("It does not import development-map-validator or execute repository code.");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage();
        if args.is_empty() { process::exit(2); }
        return;
    }
    if args.len() != 2 && args.len() != 4 {
        usage();
        process::exit(2);
    }
    let ledger = fs::read_to_string(&args[0]).unwrap_or_else(|error| {
        eprintln!("ERROR ORACLE-IO-001 {}: {error}", args[0]);
        process::exit(2);
    });
    let query = fs::read_to_string(&args[1]).unwrap_or_else(|error| {
        eprintln!("ERROR ORACLE-IO-002 {}: {error}", args[1]);
        process::exit(2);
    });
    match evaluate_json(&ledger, &query) {
        Err(error) => {
            eprintln!("REJECT TEMPORAL_LEDGER {error}");
            process::exit(1);
        }
        Ok(actual) => {
            if args.len() == 4 {
                let expected_text = fs::read_to_string(&args[2]).unwrap_or_else(|error| {
                    eprintln!("ERROR ORACLE-IO-003 {}: {error}", args[2]);
                    process::exit(2);
                });
                let expected_root: serde_json::Value = serde_json::from_str(&expected_text).unwrap_or_else(|error| {
                    eprintln!("ERROR ORACLE-JSON-001 {}: {error}", args[2]);
                    process::exit(2);
                });
                let expected = &expected_root["queries"][&args[3]];
                if let Err(error) = compare_expected(&actual, expected) {
                    eprintln!("MISMATCH EXPECTED_STATE {error}");
                    process::exit(1);
                }
                if actual["ledger_head_hash"] != expected_root["expected_ledger"]["head_hash"] {
                    eprintln!("MISMATCH LEDGER_HEAD_HASH");
                    process::exit(1);
                }
                println!("MATCH INDEPENDENT_TEMPORAL_ORACLE");
            } else {
                println!("{}", serde_json::to_string_pretty(&actual).expect("serialize result"));
            }
        }
    }
}
