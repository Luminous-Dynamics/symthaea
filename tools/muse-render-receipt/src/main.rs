use muse_render_receipt::{receipt_only_report, verify_bundle, verify_receipt_bytes};
use std::env;
use std::fs;
use std::path::Path;
use std::process;

fn usage() {
    eprintln!(
        "Usage:\n  muse-render-receipt verify RECEIPT.json [BUNDLE_DIR]\n\n\
         Receipt-only mode verifies the JSON seal and internal receipt invariants.\n\
         Bundle mode also verifies exact source/edit/MIDI/renderer/WAV artifacts."
    );
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        usage();
        return;
    }
    if args[0] != "verify" || !(2..=3).contains(&args.len()) {
        usage();
        process::exit(2);
    }

    let receipt_bytes = match fs::read(&args[1]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read receipt {}: {error}", args[1]);
            process::exit(2);
        }
    };
    let receipt = match verify_receipt_bytes(&receipt_bytes) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!("receipt verification failed: {error}");
            process::exit(1);
        }
    };

    let report = if args.len() == 3 {
        match verify_bundle(Path::new(&args[2]), &receipt) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("bundle verification failed: {error}");
                process::exit(1);
            }
        }
    } else {
        match receipt_only_report(&receipt) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("receipt-only verification failed: {error}");
                process::exit(1);
            }
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("could not serialize verification report: {error}");
            process::exit(1);
        }
    }

    if report.attempt_state == "outcome_unknown" {
        process::exit(3);
    }
}
