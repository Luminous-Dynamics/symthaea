// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;

use symthaea_cosmology_bench::{ChainTable, CHAIN_AUDIT_PROTOCOL, CHAIN_AUDIT_VERSION};

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(format!(
            "usage: {} CHAIN_TXT COLUMN_NAME",
            args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
        ));
    }

    let path = &args[1];
    let requested_column = &args[2];
    let file = File::open(path).map_err(|error| format!("cannot open {path}: {error}"))?;
    let chain = ChainTable::parse(BufReader::new(file)).map_err(|error| error.to_string())?;
    let summary = chain.summarize(requested_column).map_err(|error| error.to_string())?;

    println!("{{");
    println!("  \"protocol\": {},", json_string(CHAIN_AUDIT_PROTOCOL));
    println!("  \"version\": {},", CHAIN_AUDIT_VERSION);
    println!("  \"input_path\": {},", json_string(path));
    println!("  \"selected_column\": {},", json_string(&summary.name));
    println!("  \"sample_rows\": {},", summary.sample_rows);
    println!("  \"positive_weight_rows\": {},", summary.positive_weight_rows);
    println!("  \"total_weight\": {:.17},", summary.total_weight);
    println!("  \"weight_effective_sample_size\": {:.17},", summary.weight_effective_sample_size);
    println!("  \"weighted_mean\": {:.17},", summary.weighted_mean);
    println!("  \"weighted_variance\": {:.17},", summary.weighted_variance);
    println!("  \"weighted_empirical_q16\": {:.17},", summary.q16);
    println!("  \"weighted_empirical_q50\": {:.17},", summary.q50);
    println!("  \"weighted_empirical_q84\": {:.17}", summary.q84);
    println!("}}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
