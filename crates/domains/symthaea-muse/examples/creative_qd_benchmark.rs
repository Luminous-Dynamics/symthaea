// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the frozen, deterministic creative-intent / descriptor-diversity corpus.
//!
//! Usage:
//! cargo run -p symthaea-muse --example creative_qd_benchmark -- [source-commit-sha]

use std::error::Error;

use symthaea_muse::creative_qd_bench::run_corpus_bytes;

const CORPUS: &[u8] = include_bytes!("../fixtures/creative_intent_contract_v1.json");

fn main() -> Result<(), Box<dyn Error>> {
    let source_commit = std::env::args().nth(1);
    let receipt = run_corpus_bytes(CORPUS, source_commit.as_deref())?;
    let mismatches = receipt.expected_outcome_mismatches.len();
    println!("{}", serde_json::to_string_pretty(&receipt)?);

    if mismatches != 0 {
        return Err(format!(
            "{mismatches} frozen candidate expectation(s) did not match the audit"
        )
        .into());
    }
    Ok(())
}
