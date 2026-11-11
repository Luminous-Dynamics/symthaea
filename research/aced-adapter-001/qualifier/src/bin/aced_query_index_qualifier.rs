use aced_repair_index_qualifier::query_inventory::qualify_query_inventory;
use aced_repair_index_qualifier::query_protocol::qualify_query_comparison_protocol;
use serde_json::Value;
use std::env;
use std::fs;
use std::process::ExitCode;

fn read_json(path: &str) -> Result<(Value, Vec<u8>), String> {
    let bytes = fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{path}: invalid JSON: {error}"))?;
    Ok((value, bytes))
}

fn read_bytes(path: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("{path}: {error}"))
}

fn main() -> ExitCode {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 6 {
        eprintln!(
            "usage: aced-query-index-qualifier <inventory.json> <source-manifest.json> <SHA256SUMS> <comparison-protocol.json> <comparison-schema.json>"
        );
        return ExitCode::from(2);
    }

    let (inventory, inventory_bytes) = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let (manifest, _) = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let checksums = match read_bytes(&args[3]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let (comparison_protocol, _) = match read_json(&args[4]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let (comparison_schema, _) = match read_json(&args[5]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };

    match qualify_query_inventory(
        &inventory,
        &manifest,
        &inventory_bytes,
        &checksums,
    ) {
        Ok(report) => {
            println!(
                "ACED-QUERY METADATA INVENTORY QUALIFIED: tracks={} entries={} unique_payloads={} core_struct_exact_overlap={} core_decision_exact_overlap={} decision_filename_collisions={} checksum_entries={}",
                report.per_track_counts.len(),
                report.total_track_entries,
                report.unique_payload_sha256_count,
                report.core_struct_guess_struct_exact_content_overlap,
                report.core_decision_guess_decision_exact_content_overlap,
                report.core_decision_guess_decision_filename_collision_count,
                report.checksum_entries
            );
            for (track, count) in report.per_track_counts {
                println!("  {track}: {count}");
            }
            match qualify_query_comparison_protocol(
                &comparison_protocol,
                &comparison_schema,
                &manifest,
            ) {
                Ok(()) => {
                    println!(
                        "ACED-QUERY COMPARISON PROTOCOL QUALIFIED: task projection, upstream identity, parser, official baselines, resource budgets and closed schema."
                    );
                }
                Err(protocol_errors) => {
                    eprintln!(
                        "ACED-QUERY COMPARISON PROTOCOL QUALIFICATION FAILED ({} findings):",
                        protocol_errors.len()
                    );
                    for error in protocol_errors {
                        eprintln!(" - {error}");
                    }
                    return ExitCode::FAILURE;
                }
            }
            println!(
                "CLAIM CEILING: metadata, source digests, taxonomy distributions, task mapping and protocol structure only; no world payload was copied or executed and no model was run."
            );
            ExitCode::SUCCESS
        }
        Err(errors) => {
            eprintln!(
                "ACED-QUERY METADATA INVENTORY QUALIFICATION FAILED ({} findings):",
                errors.len()
            );
            for error in errors {
                eprintln!(" - {error}");
            }
            ExitCode::FAILURE
        }
    }
}
