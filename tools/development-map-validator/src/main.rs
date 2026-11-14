use std::{env, fs, process};

use development_map_validator::{
    cargo_manifest::{project_cargo_manifest, ManifestAnchor},
    rust_syntax::{project_rust_file, RustFileAnchor},
    tree_projection::{collect_github_tree_inventory_json, compose_repository_tree_json, RepositoryTreeAnchor},
    dynamapbench::validate_dynamapbench_manifest_json,
    validate_document_json,
};

fn usage() {
    eprintln!("Usage:");
    eprintln!("  development-map-validator <snapshot.json>");
    eprintln!("  development-map-validator cargo-manifest <Cargo.toml> <repository> <revision> <tree-oid> <manifest-path> <blob-oid>");
    eprintln!("  development-map-validator rust-file <source.rs> <repository> <revision> <tree-oid> <source-path> <blob-oid>");
    eprintln!("  development-map-validator tree-inventory <github-tree.json> <repository> <revision> <tree-oid>");
    eprintln!("  development-map-validator tree-compose <request.json>");
    eprintln!("  development-map-validator dynamapbench-manifest <manifest.json>");
    eprintln!();
    eprintln!("Validates EGDDM snapshots/federation/planning/retrieval fixtures, or");
    eprintln!("projects a Cargo manifest or Rust source file, or composes exact source projections into a Git tree snapshot.");
    eprintln!("No mode invokes Cargo, rustc, build scripts, tests or repository code.");
    eprintln!("Git identities are recorded and syntax-checked, not authenticated.");
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

fn read_input(path: &str, diagnostic_code: &str) -> Result<String, i32> {
    fs::read_to_string(path).map_err(|error| {
        eprintln!("ERROR {diagnostic_code} {path}: {error}");
        2
    })
}

fn project_cargo_manifest_file(args: &[String]) -> i32 {
    if args.len() != 7 {
        usage();
        return 2;
    }

    let manifest_file = &args[1];
    let input = match read_input(manifest_file, "EGDDM-CARGO-IO-001") {
        Ok(input) => input,
        Err(code) => return code,
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

fn project_rust_source_file(args: &[String]) -> i32 {
    if args.len() != 7 {
        usage();
        return 2;
    }

    let source_file = &args[1];
    let input = match read_input(source_file, "EGDDM-RUST-IO-001") {
        Ok(input) => input,
        Err(code) => return code,
    };
    let source = RustFileAnchor {
        repository_full_name: args[2].clone(),
        revision: args[3].clone(),
        tree_oid: args[4].clone(),
        source_path: args[5].clone(),
        blob_oid: args[6].clone(),
    };

    match project_rust_file(&input, source) {
        Err(error) => {
            eprintln!("ERROR EGDDM-RUST-001 {source_file}: {error}");
            1
        }
        Ok(projection) => match serde_json::to_string_pretty(&projection) {
            Ok(output) => {
                println!("{output}");
                0
            }
            Err(error) => {
                eprintln!("ERROR EGDDM-RUST-JSON-001 {source_file}: {error}");
                2
            }
        },
    }
}



fn collect_github_tree_inventory_file(args: &[String]) -> i32 {
    if args.len() != 5 {
        usage();
        return 2;
    }
    let request_file = &args[1];
    let input = match read_input(request_file, "EGDDM-TREE-IO-001") {
        Ok(input) => input,
        Err(code) => return code,
    };
    let source = RepositoryTreeAnchor {
        repository_full_name: args[2].clone(),
        revision: args[3].clone(),
        tree_oid: args[4].clone(),
    };
    match collect_github_tree_inventory_json(&input, source) {
        Err(error) => {
            eprintln!("ERROR EGDDM-TREE-COLLECT-001 {request_file}: {error}");
            1
        }
        Ok(inventory) => match serde_json::to_string_pretty(&inventory) {
            Ok(output) => {
                println!("{output}");
                0
            }
            Err(error) => {
                eprintln!("ERROR EGDDM-TREE-COLLECT-JSON-001 {request_file}: {error}");
                2
            }
        },
    }
}

fn compose_repository_tree_file(args: &[String]) -> i32 {
    if args.len() != 2 {
        usage();
        return 2;
    }
    let request_file = &args[1];
    let input = match read_input(request_file, "EGDDM-TREE-IO-001") {
        Ok(input) => input,
        Err(code) => return code,
    };
    match compose_repository_tree_json(&input) {
        Err(error) => {
            eprintln!("ERROR EGDDM-TREE-001 {request_file}: {error}");
            1
        }
        Ok(composition) => match serde_json::to_string_pretty(&composition) {
            Ok(output) => {
                println!("{output}");
                0
            }
            Err(error) => {
                eprintln!("ERROR EGDDM-TREE-JSON-001 {request_file}: {error}");
                2
            }
        },
    }
}


fn validate_dynamapbench_manifest_file(args: &[String]) -> i32 {
    if args.len() != 2 {
        usage();
        return 2;
    }
    let path = &args[1];
    let input = match read_input(path, "DMB-IO-001") {
        Ok(input) => input,
        Err(code) => return code,
    };
    match validate_dynamapbench_manifest_json(&input) {
        Err(error) => {
            eprintln!("ERROR DMB-JSON-001 {path}: {error}");
            2
        }
        Ok(report) if report.is_valid() => {
            println!("VALID DYNAMAPBENCH_MANIFEST path={path}");
            println!("NOTE no Git object, corpus artifact, evaluator binary, signature, benchmark run or source content was independently verified");
            0
        }
        Ok(report) => {
            for diagnostic in &report.diagnostics {
                eprintln!("ERROR {} {} {}", diagnostic.code, diagnostic.path, diagnostic.message);
            }
            eprintln!("INVALID diagnostics={}", report.diagnostics.len());
            1
        }
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

    let code = match args[0].as_str() {
        "cargo-manifest" => project_cargo_manifest_file(&args),
        "rust-file" => project_rust_source_file(&args),
        "tree-inventory" => collect_github_tree_inventory_file(&args),
        "tree-compose" => compose_repository_tree_file(&args),
        "dynamapbench-manifest" => validate_dynamapbench_manifest_file(&args),
        _ if args.len() == 1 => validate_document(&args[0]),
        _ => {
            usage();
            2
        }
    };

    if code != 0 {
        process::exit(code);
    }
}
