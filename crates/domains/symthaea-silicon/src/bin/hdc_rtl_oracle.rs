// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Generate exact RTL word fixtures from Symthaea's canonical BinaryHV oracle.
// This generator is a software-reference fixture tool; it does not run or
// qualify any RTL/EDA hardware implementation.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use symthaea_core::hdc::BinaryHV;

const WIDTHS_BITS: [usize; 4] = [64, 128, 256, 512];
const HDC_BYTES: usize = BinaryHV::BYTES;
const HDC_BITS: usize = BinaryHV::DIM;

struct Scenario {
    id: &'static str,
    a: BinaryHV,
    b: BinaryHV,
}

#[derive(Serialize)]
struct WidthReceipt {
    datapath_width_bits: usize,
    hamming_count_width_bits: usize,
    words_per_vector: usize,
    scenarios: usize,
    word_cases: usize,
    xor_mem_file: String,
    xor_mem_sha256: String,
    hamming_mem_file: String,
    hamming_mem_sha256: String,
}

#[derive(Serialize)]
struct OracleReceipt {
    schema_version: &'static str,
    status: &'static str,
    generator: &'static str,
    generator_package_version: &'static str,
    git_head: String,
    canonical_type: &'static str,
    canonical_source_path: &'static str,
    canonical_source_sha256: String,
    cargo_lock_sha256: String,
    dimension_bits: usize,
    storage_bytes: usize,
    byte_order: &'static str,
    scenario_ids: Vec<&'static str>,
    widths: Vec<WidthReceipt>,
    claim_ceiling: &'static str,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_file(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(sha256_hex(&fs::read(path)?))
}

fn find_repo_root() -> Result<PathBuf, Box<dyn Error>> {
    let mut current = env::current_dir()?;
    loop {
        let source = current.join("crates/core/symthaea-core/src/hdc/binary_hv.rs");
        let lock = current.join("Cargo.lock");
        let flake = current.join("flake.nix");
        if source.is_file() && lock.is_file() && flake.is_file() {
            return Ok(current);
        }
        if !current.pop() {
            return Err("could not locate repository root (Cargo.lock, flake.nix and canonical BinaryHV source required)".into());
        }
    }
}

fn git_head(repo_root: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed; cannot bind oracle fixtures to source revision".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn parse_args() -> Result<(PathBuf, Option<String>), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut output_dir: Option<PathBuf> = None;
    let mut expected_git_head: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => {
                output_dir = Some(args.next().ok_or("--output-dir requires a path")?.into());
            }
            "--expected-git-head" => {
                expected_git_head = Some(args.next().ok_or("--expected-git-head requires a SHA")?);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: hdc_rtl_oracle --output-dir PATH [--expected-git-head SHA]\n\
                     Writes canonical BinaryHV-derived XOR and per-word Hamming fixtures for 64/128/256/512-bit words."
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}").into()),
        }
    }

    let output_dir = output_dir.ok_or("--output-dir is required")?;
    Ok((output_dir, expected_git_head))
}

fn scenarios() -> Vec<Scenario> {
    let zero = BinaryHV::zero();
    let ones = BinaryHV::ones();
    let seed_1 = BinaryHV::random(1);
    let seed_42 = BinaryHV::random(42);
    let seed_2026 = BinaryHV::random(2026);
    let seed_0x5a17 = BinaryHV::random(0x5a17);

    vec![
        Scenario { id: "zero-zero", a: zero, b: zero },
        Scenario { id: "zero-ones", a: zero, b: ones },
        Scenario { id: "ones-zero", a: ones, b: zero },
        Scenario { id: "same-seed-42", a: seed_42, b: seed_42 },
        Scenario { id: "seed-1-vs-42", a: seed_1, b: seed_42 },
        Scenario { id: "seed-2026-vs-5a17", a: seed_2026, b: seed_0x5a17 },
    ]
}

fn word_hex(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push_str(&format!("{byte:02x}"));
    }
    result
}

fn popcount_bytes(bytes: &[u8]) -> u32 {
    bytes.iter().map(|byte| byte.count_ones()).sum()
}

fn generate_width(
    out_dir: &Path,
    width_bits: usize,
    scenarios: &[Scenario],
) -> Result<WidthReceipt, Box<dyn Error>> {
    if width_bits == 0 || width_bits % 8 != 0 || HDC_BITS % width_bits != 0 {
        return Err(format!("unsupported datapath width: {width_bits}").into());
    }

    let word_bytes = width_bits / 8;
    let words_per_vector = HDC_BYTES / word_bytes;
    let count_width_bits = (usize::BITS - width_bits.leading_zeros()) as usize;
    let count_hex_digits = (count_width_bits + 3) / 4;
    let mut xor_mem = String::new();
    let mut hamming_mem = String::new();
    let mut word_cases = 0usize;

    for scenario in scenarios {
        let bound = scenario.a.bind(&scenario.b);
        let whole_vector_distance = scenario.a.hamming_distance(&scenario.b);
        let mut summed_word_distance = 0u32;

        for word_index in 0..words_per_vector {
            let start = word_index * word_bytes;
            let end = start + word_bytes;
            let a_word = &scenario.a.0[start..end];
            let b_word = &scenario.b.0[start..end];
            let xor_word = &bound.0[start..end];

            // Independently check that the canonical bind result agrees with
            // bytewise XOR before we emit any expected value.
            for index in 0..word_bytes {
                if xor_word[index] != (a_word[index] ^ b_word[index]) {
                    return Err(format!(
                        "canonical BinaryHV::bind disagrees with byte XOR: scenario={} width={} word={}",
                        scenario.id, width_bits, word_index
                    ).into());
                }
            }

            let word_distance = popcount_bytes(xor_word);
            summed_word_distance += word_distance;
            xor_mem.push_str(&word_hex(a_word));
            xor_mem.push_str(&word_hex(b_word));
            xor_mem.push_str(&word_hex(xor_word));
            xor_mem.push('\n');

            // The count field is nibble padded for $readmemh. The testbench
            // consumes only the low COUNT_W bits. Operand byte order is the
            // same as the XOR fixture, with BinaryHV byte zero at the literal's
            // most-significant byte position.
            hamming_mem.push_str(&word_hex(a_word));
            hamming_mem.push_str(&word_hex(b_word));
            hamming_mem.push_str(&format!("{word_distance:0count_hex_digits$x}"));
            hamming_mem.push('\n');
            word_cases += 1;
        }

        if summed_word_distance != whole_vector_distance {
            return Err(format!(
                "sum of per-word Hamming counts differs from canonical full-vector distance: scenario={} width={} words_sum={} full={}",
                scenario.id, width_bits, summed_word_distance, whole_vector_distance
            ).into());
        }
    }

    let xor_name = format!("hdc_oracle_xor_w{width_bits}.mem");
    let hamming_name = format!("hdc_oracle_hamming_w{width_bits}.mem");
    fs::write(out_dir.join(&xor_name), xor_mem.as_bytes())?;
    fs::write(out_dir.join(&hamming_name), hamming_mem.as_bytes())?;

    Ok(WidthReceipt {
        datapath_width_bits: width_bits,
        hamming_count_width_bits: count_width_bits,
        words_per_vector,
        scenarios: scenarios.len(),
        word_cases,
        xor_mem_sha256: hash_file(&out_dir.join(&xor_name))?,
        hamming_mem_sha256: hash_file(&out_dir.join(&hamming_name))?,
        xor_mem_file: xor_name,
        hamming_mem_file: hamming_name,
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let (output_dir, expected_git_head) = parse_args()?;
    let repo_root = find_repo_root()?;
    let current_git_head = git_head(&repo_root)?;

    if let Some(expected) = expected_git_head {
        if expected != current_git_head {
            return Err(format!(
                "repository HEAD mismatch: expected {expected}, found {current_git_head}; refusing to emit fixtures"
            ).into());
        }
    }

    fs::create_dir_all(&output_dir)?;
    let canonical_relative = "crates/core/symthaea-core/src/hdc/binary_hv.rs";
    let canonical_path = repo_root.join(canonical_relative);
    let cargo_lock_path = repo_root.join("Cargo.lock");
    let canonical_sha = hash_file(&canonical_path)?;
    let lock_sha = hash_file(&cargo_lock_path)?;
    let cases = scenarios();
    let scenario_ids = cases.iter().map(|scenario| scenario.id).collect();
    let mut widths = Vec::new();

    for width_bits in WIDTHS_BITS {
        widths.push(generate_width(&output_dir, width_bits, &cases)?);
    }

    let receipt = OracleReceipt {
        schema_version: "1.0",
        status: "software_oracle_fixtures_generated_not_rtl_tested",
        generator: "hdc_rtl_oracle",
        generator_package_version: env!("CARGO_PKG_VERSION"),
        git_head: current_git_head,
        canonical_type: "symthaea_core::hdc::BinaryHV",
        canonical_source_path: canonical_relative,
        canonical_source_sha256: canonical_sha,
        cargo_lock_sha256: lock_sha,
        dimension_bits: HDC_BITS,
        storage_bytes: HDC_BYTES,
        byte_order: "BinaryHV byte index 0 occupies the most-significant byte position in each emitted word literal; identical ordering is used for operands and expected XOR.",
        scenario_ids,
        widths,
        claim_ceiling: "Generated rows come from canonical software plus local algebra checks. They do not show that RTL passed simulation, formal verification, synthesis/PPA or physical qualification.",
    };

    let metadata = serde_json::to_vec_pretty(&receipt)?;
    fs::write(output_dir.join("oracle_receipt.json"), &metadata)?;
    println!("{}", output_dir.join("oracle_receipt.json").display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("hdc_rtl_oracle: {error}");
        std::process::exit(2);
    }
}
