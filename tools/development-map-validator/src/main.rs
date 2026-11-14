use std::{env, fs, process};

use development_map_validator::validate_document_json;

fn usage() {
    eprintln!("Usage: development-map-validator <snapshot.json>");
    eprintln!("Validates EGDDM v0.1 map snapshots, federation manifests, or DYNPLAN case/oracle fixtures.");
    eprintln!("This is a read-only structural validator, not a full JSON Schema validator or signature verifier.");
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        usage();
        process::exit(2);
    };
    if path == "--help" || path == "-h" {
        usage();
        return;
    }
    if args.next().is_some() {
        usage();
        process::exit(2);
    }

    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("ERROR EGDDM-IO-001 {path}: {error}");
            process::exit(2);
        }
    };

    match validate_document_json(&input) {
        Err(error) => {
            eprintln!("ERROR EGDDM-JSON-001 {path}: {error}");
            process::exit(2);
        }
        Ok(report) if report.is_valid() => {
            println!("VALID STRUCTURAL_CONTRACT path={path}");
            println!("NOTE no source URI, Git object, signature, build, test, CI or runtime evidence was independently verified");
        }
        Ok(report) => {
            for diagnostic in &report.diagnostics {
                eprintln!(
                    "ERROR {} {} {}",
                    diagnostic.code, diagnostic.path, diagnostic.message
                );
            }
            eprintln!("INVALID diagnostics={}", report.diagnostics.len());
            process::exit(1);
        }
    }
}
