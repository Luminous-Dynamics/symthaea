//! CLI for independently verifying signed PITB component-coverage attestations.

use std::{env, fs};

use anyhow::{Context, Result, bail};

#[path = "../coverage_receipt.rs"]
mod coverage_receipt;
use coverage_receipt::{EXPECTED_PARTITION_DIGEST, validate_frozen_partition, verify_receipt};

fn run(args: &[String]) -> Result<()> {
    if args.len() != 8 {
        bail!(
            concat!(
                "usage: verify_component_coverage_receipt <splits.json> <edges.json> ",
                "<receipt.json> <trust-store.json> <review-methodology.md> ",
                "<expected-trust-store-sha256> <expected-methodology-sha256> <report.json>"
            )
        );
    }
    let split_bytes = fs::read(&args[0]).with_context(|| format!("read {}", args[0]))?;
    let inventory_bytes = fs::read(&args[1]).with_context(|| format!("read {}", args[1]))?;
    let receipt_bytes = fs::read(&args[2]).with_context(|| format!("read {}", args[2]))?;
    let trust_bytes = fs::read(&args[3]).with_context(|| format!("read {}", args[3]))?;
    let methodology_bytes = fs::read(&args[4]).with_context(|| format!("read {}", args[4]))?;
    let families = validate_frozen_partition(&split_bytes)?;
    let report = verify_receipt(
        &inventory_bytes,
        &receipt_bytes,
        &trust_bytes,
        &methodology_bytes,
        &args[5],
        &args[6],
        EXPECTED_PARTITION_DIGEST,
        &families,
    )?;
    fs::write(&args[7], serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args[7]))?;
    println!(
        "PITB_COVERAGE_RECEIPT_VERIFIED: signers={} organizations={} trust_store_sha256={}",
        report.verified_reviewer_ids.len(),
        report.distinct_reviewer_organization_count,
        report.trust_store_sha256
    );
    Ok(())
}

fn main() -> Result<()> {
    run(&env::args().skip(1).collect::<Vec<_>>())
}
