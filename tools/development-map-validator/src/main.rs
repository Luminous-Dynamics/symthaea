use std::{env, fs, process};

use development_map_validator::{
    cargo_manifest::{project_cargo_manifest, ManifestAnchor},
    validate_document_json,
};

fn usage() {
    eprintln!("Usage:");
    eprintln!("  development-map-validator <snapshot.json>");
    eprintln!("  development-map-validator cargo-manifest <Cargo.toml> <repository> <revision> <tree-oid> <manifest-path> <blob-oid>");
    eprintln!();
    eprintln!("Validates EGDDM snapshots/federation/planning/retrieval fixtures, or");
    eprintln!("projects one Cargo.toml as read-only manifest-declared evidence.");
    eprintln!("The Cargo mode does not invoke Cargo, execute build scripts, resolve");
    eprintln!("workspace globs, expand features, or authenticate supplied Git IDs.");
}

fn validate_document(path: &str) -> i32 {
    let input = match fs::read_to_string(path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("ERROR EGDDM-IO-001 {path}: {error}");
            return 2;
        }
    };

    match validate_document_json(&input) {
        Err(error) => {
            eprintln!("ERROR EGDDM-JSON-001 {path}: {error}");
            2
        }
        Ok(report) if report.is_valid() => {
            println!("VALID STRUCTURAL_CONTRACT path={path}");
            println!("NOTE no source URI, Git object, signature, build, test, CI or runtime evidence was independently verified");
            0
        }
        Ok(report) => {
            for diagnostic in &report.diagnostics {
                eprintln!(
                    "ERROR {} {} {}",
                    diagnostic.code, diagnostic.path, diagnostic.message
                );
            }
            eprintln!("INVALID diagnostics={}", report.diagnostics.len());
            1
        }
    }
}

fn project_cargo_manifest_file(args: &[String]) -> i32 {
    if args.len() != 7 {
        usage();
        return 2;
    }

    let manifest_file = &args[1];
    let input = match fs::read_to_string(manifest_file) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("ERROR EGDDM-CARGO-IO-001 {manifest_file}: {error}");
            return 2;
        }
    };

    let source = ManifestAnchor {
        repository_full_name: args[2].clone(),
        revision: args[3].clone(),
        tree_oid: args[4].clone(),
        manifest_path: args[5].clone(),
        blob_oid: args[6].clone(),
    };

    match project_cargo_manifest(&input, source) {
        Err(error) => {
            eprintln!("ERROR EGDDM-CARGO-001 {manifest_file}: {error}");
            1
        }
        Ok(projection) => match serde_json::to_string_pretty(&projection) {
            Ok(output) => {
                println!("{output}");
                0
            }
            Err(error) => {
                eprintln!("ERROR EGDDM-CARGO-JSON-001 {manifest_file}: {error}");
                2
            }
        },
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        usage();
        if args.is_empty() {
            process::exit(2);
        }
        return;
    }

    let code = if args[0] == "cargo-manifest" {
        project_cargo_manifest_file(&args)
    } else if args.len() == 1 {
        validate_document(&args[0])
    } else {
        usage();
        2
    };

    if code != 0 {
        process::exit(code);
    }
}
