// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CLI: Generate ZK proofs for tax compliance.
//!
//! A comprehensive command-line interface for generating all types of
//! zero-knowledge tax proofs.
//!
//! # Usage
//!
//! ```bash
//! # Tax bracket proof
//! cargo run --example cli -- bracket --income 85000 --jurisdiction US --status single --year 2024
//!
//! # Income range proof
//! cargo run --example cli -- range --income 95000 --min 90000 --year 2024
//!
//! # Effective tax rate proof
//! cargo run --example cli -- effective --income 85000 --jurisdiction US --status single --year 2024
//!
//! # Batch multi-year proof
//! cargo run --example cli -- batch --jurisdiction US --status single --years 2022,2023,2024 --incomes 75000,80000,85000
//!
//! # Cross-jurisdiction comparison
//! cargo run --example cli -- cross --income 100000 --jurisdictions US,UK,DE --status single --year 2024
//!
//! # OECD comparison (7 major economies)
//! cargo run --example cli -- oecd --income 100000 --status single --year 2024
//!
//! # G20 comparison (all 19 countries)
//! cargo run --example cli -- g20 --income 100000 --status single --year 2024
//!
//! # Deduction proof
//! cargo run --example cli -- deduction --charitable 5000 --mortgage 12000 --salt 10000 --status single --year 2024
//!
//! # Composite proof (all-in-one)
//! cargo run --example cli -- composite --income 85000 --jurisdiction US --status single --year 2024
//!
//! # Proof chain (audit trail)
//! cargo run --example cli -- chain --jurisdiction US --status single --years 2022,2023,2024 --incomes 75000,80000,85000
//!
//! # US State proof
//! cargo run --example cli -- state --income 150000 --state CA --status single --year 2024
//!
//! # Swiss Canton proof
//! cargo run --example cli -- canton --income 200000 --canton ZH --status single --year 2024
//! ```

use mycelix_zk_tax::{
    FilingStatus, Jurisdiction, TaxBracketProver,
    RangeProofBuilder, BatchProofBuilder,
    EffectiveTaxRateProof, CrossJurisdictionProofBuilder,
    DeductionProofBuilder, DeductionCategory,
    CompositeProofBuilder, ProofChain,
};
use mycelix_zk_tax::subnational::{
    USState, SwissCanton, get_state_brackets, get_cantonal_brackets,
};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "bracket" => cmd_bracket(&args[2..]),
        "range" => cmd_range(&args[2..]),
        "effective" => cmd_effective(&args[2..]),
        "batch" => cmd_batch(&args[2..]),
        "cross" => cmd_cross(&args[2..]),
        "oecd" => cmd_oecd(&args[2..]),
        "g20" => cmd_g20(&args[2..]),
        "deduction" => cmd_deduction(&args[2..]),
        "composite" => cmd_composite(&args[2..]),
        "chain" => cmd_chain(&args[2..]),
        "state" => cmd_state(&args[2..]),
        "canton" => cmd_canton(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!(r#"
Mycelix ZK Tax CLI - Zero-Knowledge Tax Proof Generator

USAGE:
    cargo run --example cli -- <COMMAND> [OPTIONS]

COMMANDS:
    bracket     Generate a tax bracket proof
    range       Generate an income range proof
    effective   Generate an effective tax rate proof
    batch       Generate a multi-year batch proof
    cross       Generate a cross-jurisdiction proof
    oecd        Generate OECD countries comparison (7 countries)
    g20         Generate G20 countries comparison (19 countries)
    deduction   Generate a deduction proof
    composite   Generate a composite proof (all-in-one)
    chain       Generate a proof chain (audit trail)
    state       Generate a US state tax proof
    canton      Generate a Swiss canton tax proof
    help        Show this help message

EXAMPLES:
    # Prove tax bracket without revealing income
    cargo run --example cli -- bracket --income 85000 --jurisdiction US --status single --year 2024

    # Prove income >= $90,000 (for rental applications)
    cargo run --example cli -- range --income 95000 --min 90000 --year 2024

    # Show effective vs marginal tax rate
    cargo run --example cli -- effective --income 150000 --jurisdiction US --status mfj --year 2024

    # 3-year income history (for mortgages)
    cargo run --example cli -- batch --jurisdiction US --status single --years 2022,2023,2024 --incomes 75000,80000,85000

    # Compare tax brackets across countries
    cargo run --example cli -- cross --income 100000 --jurisdictions US,UK,DE,FR --status single --year 2024

    # Compare all OECD major economies
    cargo run --example cli -- oecd --income 100000 --status single --year 2024

    # Compare all G20 countries
    cargo run --example cli -- g20 --income 100000 --status single --year 2024

    # Prove deductions without revealing amounts
    cargo run --example cli -- deduction --charitable 5000 --mortgage 12000 --salt 10000 --year 2024

    # All-in-one composite proof
    cargo run --example cli -- composite --income 85000 --jurisdiction US --status single --year 2024

    # Multi-year audit chain
    cargo run --example cli -- chain --jurisdiction US --status single --years 2022,2023,2024 --incomes 75000,80000,85000

    # California state tax
    cargo run --example cli -- state --income 150000 --state CA --status single --year 2024

    # Swiss canton (Zurich)
    cargo run --example cli -- canton --income 200000 --canton ZH --status single --year 2024

JURISDICTIONS:
    US, CA, MX, BR, AR (Americas)
    UK, DE, FR, IT, RU, TR (Europe)
    JP, CN, IN, KR, ID, AU (Asia-Pacific)
    SA, ZA (Middle East & Africa)

US STATES:
    CA, NY, TX, FL, IL, PA, WA, NV, etc.

SWISS CANTONS:
    ZH (Zurich), ZG (Zug), GE (Geneva), BS (Basel), etc.

FILING STATUS:
    single, mfj (married filing jointly), mfs (married filing separately), hoh (head of household)
"#);
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().ok();
        }
    }
    None
}

fn parse_arg_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(&args[i + 1]);
        }
    }
    None
}

fn cmd_bracket(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let jurisdiction_str = parse_arg_str(args, "--jurisdiction").unwrap_or("US");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let jurisdiction: Jurisdiction = jurisdiction_str.parse().unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let filing_status: FilingStatus = status_str.parse().unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    println!("=== Tax Bracket Proof ===\n");
    println!("Input (PRIVATE - never revealed):");
    println!("  Income: ${}", format_number(income));
    println!("  Jurisdiction: {}", jurisdiction.name());
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    let prover = TaxBracketProver::dev_mode();
    let start = Instant::now();
    let proof = prover.prove(income, jurisdiction, filing_status, year)
        .expect("Proof generation failed");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC - safe to share):");
    println!("  Bracket: {} ({}%)", proof.bracket_index, proof.rate_bps as f64 / 100.0);
    println!("  Range: ${} - {}", format_number(proof.bracket_lower),
        if proof.bracket_upper == u64::MAX { "+".to_string() }
        else { format!("${}", format_number(proof.bracket_upper)) });
    println!("  Commitment: {}", proof.commitment);
    println!("\nGenerated in {:?}", elapsed);
    println!("\nThis proves: 'My income falls in the {}% bracket'", proof.rate_bps as f64 / 100.0);
    println!("WITHOUT revealing the actual amount!");
}

fn cmd_range(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let min: u64 = parse_arg(args, "--min").unwrap_or(0);
    let max: u64 = parse_arg(args, "--max").unwrap_or(u64::MAX);
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    println!("=== Income Range Proof ===\n");
    println!("Input (PRIVATE):");
    println!("  Income: ${}", format_number(income));
    println!("  Proving: {} <= income <= {}",
        if min == 0 { "0".to_string() } else { format!("${}", format_number(min)) },
        if max == u64::MAX { "+".to_string() } else { format!("${}", format_number(max)) });
    println!("  Tax Year: {}\n", year);

    let start = Instant::now();
    let proof = if max == u64::MAX {
        RangeProofBuilder::new(income, year).prove_above(min)
    } else if min == 0 {
        RangeProofBuilder::new(income, year).prove_below(max)
    } else {
        RangeProofBuilder::new(income, year).prove_between(min, max)
    }.expect("Range proof failed");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  {}", proof.summary());
    println!("  Commitment: {}", proof.commitment);
    println!("\nGenerated in {:?}", elapsed);

    if min > 0 && max == u64::MAX {
        println!("\nUse case: Rental application proving income >= ${}", format_number(min));
    } else if max < u64::MAX && min == 0 {
        println!("\nUse case: Benefits eligibility proving income <= ${}", format_number(max));
    }
}

fn cmd_effective(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let jurisdiction_str = parse_arg_str(args, "--jurisdiction").unwrap_or("US");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let jurisdiction: Jurisdiction = jurisdiction_str.parse().unwrap();
    let filing_status: FilingStatus = status_str.parse().unwrap();

    println!("=== Effective Tax Rate Proof ===\n");
    println!("Input (PRIVATE):");
    println!("  Income: ${}", format_number(income));
    println!("  Jurisdiction: {}", jurisdiction.name());
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    let start = Instant::now();
    let proof = EffectiveTaxRateProof::prove_dev(income, jurisdiction, filing_status, year)
        .expect("Effective rate proof failed");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  Marginal Rate: {:.0}% (bracket {})", proof.marginal_rate_percent(), proof.marginal_bracket);
    println!("  Effective Rate: {:.1}%", proof.effective_rate_percent());
    println!("  Rate Range: {:.1}% - {:.1}%",
        proof.effective_rate_lower_bps as f64 / 100.0,
        proof.effective_rate_upper_bps as f64 / 100.0);
    println!("  Progressive Savings: {:.1}% lower than marginal!",
        proof.progressive_tax_savings_bps() as f64 / 100.0);
    println!("  Commitment: {}", proof.commitment);
    println!("\nGenerated in {:?}", elapsed);
    println!("\nKey insight: Your ACTUAL tax burden is {:.1}%, not {:.0}%!",
        proof.effective_rate_percent(), proof.marginal_rate_percent());
}

fn cmd_batch(args: &[String]) {
    let jurisdiction_str = parse_arg_str(args, "--jurisdiction").unwrap_or("US");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let years_str = parse_arg_str(args, "--years").unwrap_or("2022,2023,2024");
    let incomes_str = parse_arg_str(args, "--incomes").unwrap_or("75000,80000,85000");

    let jurisdiction: Jurisdiction = jurisdiction_str.parse().unwrap();
    let filing_status: FilingStatus = status_str.parse().unwrap();

    let years: Vec<u32> = years_str.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let incomes: Vec<u64> = incomes_str.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    if years.len() != incomes.len() {
        eprintln!("Error: years and incomes must have same length");
        std::process::exit(1);
    }

    println!("=== Multi-Year Batch Proof ===\n");
    println!("Input (PRIVATE):");
    for (y, i) in years.iter().zip(incomes.iter()) {
        println!("  {}: ${}", y, format_number(*i));
    }
    println!("  Jurisdiction: {}", jurisdiction.name());
    println!("  Filing Status: {}\n", filing_status.name());

    let start = Instant::now();
    let mut builder = BatchProofBuilder::new(jurisdiction, filing_status);
    for (year, income) in years.iter().zip(incomes.iter()) {
        builder = builder.add_year(*year, *income);
    }
    let proof = builder.build_dev().expect("Batch proof failed");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  {}", proof.summary());
    for yp in &proof.year_proofs {
        println!("    {}: Bracket {} ({}%)", yp.tax_year, yp.bracket_index, yp.rate_bps as f64 / 100.0);
    }
    println!("  Consistent bracket: {}", proof.consistent_bracket().map_or("No".to_string(), |b| format!("Yes ({})", b)));
    println!("  Commitment: {}", proof.batch_commitment);
    println!("\nGenerated in {:?}", elapsed);
    println!("\nUse case: Mortgage application proving {}-year income history", years.len());
}

fn cmd_cross(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let jurisdictions_str = parse_arg_str(args, "--jurisdictions").unwrap_or("US,UK");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let filing_status: FilingStatus = status_str.parse().unwrap();

    let jurisdictions: Vec<Jurisdiction> = jurisdictions_str.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    println!("=== Cross-Jurisdiction Proof ===\n");
    println!("Input (PRIVATE):");
    println!("  Income: ${} USD equivalent", format_number(income));
    println!("  Countries: {:?}", jurisdictions.iter().map(|j| j.name()).collect::<Vec<_>>());
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    let start = Instant::now();
    let mut builder = CrossJurisdictionProofBuilder::new(income, year);
    for j in &jurisdictions {
        builder = builder.add_jurisdiction(*j, filing_status);
    }
    let proof = builder.build_dev().expect("Cross-jurisdiction proof failed");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  {}", proof.summary());
    for info in &proof.jurisdiction_brackets {
        let rate = info.rate_bps as f64 / 100.0;
        println!("    {}: {:.0}% (bracket {})", info.jurisdiction.name(), rate, info.bracket_index);
    }
    let (min, max) = proof.rate_range_bps().unwrap();
    println!("\n  Rate Range: {:.0}% - {:.0}%", min as f64 / 100.0, max as f64 / 100.0);
    println!("  Average Rate: {:.1}%", proof.average_rate_bps() as f64 / 100.0);
    println!("  Commitment: {}", proof.combined_commitment);
    println!("\nGenerated in {:?}", elapsed);
}

fn cmd_oecd(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or(100_000);
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let filing_status: FilingStatus = status_str.parse().unwrap();

    println!("=== OECD Major Economies Comparison ===\n");
    println!("Input: ${} USD equivalent, {} {}\n", format_number(income), filing_status.name(), year);

    let start = Instant::now();
    let proof = CrossJurisdictionProofBuilder::new(income, year)
        .add_oecd_common(filing_status)
        .build_dev()
        .expect("OECD proof failed");
    let elapsed = start.elapsed();

    println!("Tax Bracket Comparison:");
    for info in &proof.jurisdiction_brackets {
        let rate = info.rate_bps as f64 / 100.0;
        let indicator = if rate == 0.0 { " (no income tax)" }
            else if rate >= 40.0 { " (high)" }
            else if rate >= 25.0 { "" }
            else { " (low)" };
        println!("  {:15} {:5.0}%{}", info.jurisdiction.name(), rate, indicator);
    }

    let (min, max) = proof.rate_range_bps().unwrap();
    println!("\nStatistics:");
    println!("  Lowest: {:.0}%", min as f64 / 100.0);
    println!("  Highest: {:.0}%", max as f64 / 100.0);
    println!("  Average: {:.1}%", proof.average_rate_bps() as f64 / 100.0);
    println!("  Spread: {:.0} percentage points", (max - min) as f64 / 100.0);
    println!("\nGenerated in {:?}", elapsed);
}

fn cmd_g20(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or(100_000);
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let filing_status: FilingStatus = status_str.parse().unwrap();

    println!("=== G20 Countries Comparison ===\n");
    println!("Input: ${} USD equivalent, {} {}\n", format_number(income), filing_status.name(), year);

    let start = Instant::now();
    let proof = CrossJurisdictionProofBuilder::new(income, year)
        .add_jurisdictions(&[
            Jurisdiction::US, Jurisdiction::CA, Jurisdiction::MX,
            Jurisdiction::BR, Jurisdiction::AR, Jurisdiction::UK,
            Jurisdiction::DE, Jurisdiction::FR, Jurisdiction::IT,
            Jurisdiction::RU, Jurisdiction::TR, Jurisdiction::JP,
            Jurisdiction::CN, Jurisdiction::IN, Jurisdiction::KR,
            Jurisdiction::ID, Jurisdiction::AU, Jurisdiction::SA,
            Jurisdiction::ZA,
        ], filing_status)
        .build_dev()
        .expect("G20 proof failed");
    let elapsed = start.elapsed();

    // Sort by rate for display
    let mut sorted: Vec<_> = proof.jurisdiction_brackets.iter().collect();
    sorted.sort_by_key(|info| info.rate_bps);

    println!("Tax Bracket Comparison (sorted by rate):");
    for info in sorted {
        let rate = info.rate_bps as f64 / 100.0;
        let bar_len = (rate / 2.0) as usize;
        let bar: String = "".repeat(bar_len.min(25));
        println!("  {:15} {:5.0}% {}", info.jurisdiction.name(), rate, bar);
    }

    let (min, max) = proof.rate_range_bps().unwrap();
    println!("\nStatistics:");
    println!("  Countries: {}", proof.jurisdiction_count());
    println!("  Lowest: {:.0}%", min as f64 / 100.0);
    println!("  Highest: {:.0}%", max as f64 / 100.0);
    println!("  Average: {:.1}%", proof.average_rate_bps() as f64 / 100.0);
    println!("\nGenerated in {:?}", elapsed);
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn cmd_deduction(args: &[String]) {
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);
    let filing_status: FilingStatus = status_str.parse().unwrap();

    // Parse deduction amounts
    let charitable: u64 = parse_arg(args, "--charitable").unwrap_or(0);
    let mortgage: u64 = parse_arg(args, "--mortgage").unwrap_or(0);
    let salt: u64 = parse_arg(args, "--salt").unwrap_or(0);
    let medical: u64 = parse_arg(args, "--medical").unwrap_or(0);
    let retirement: u64 = parse_arg(args, "--retirement").unwrap_or(0);

    println!("=== Deduction Proof ===\n");
    println!("Input (PRIVATE - amounts never revealed):");
    if charitable > 0 { println!("  Charitable: ${}", format_number(charitable)); }
    if mortgage > 0 { println!("  Mortgage Interest: ${}", format_number(mortgage)); }
    if salt > 0 { println!("  State & Local Taxes: ${}", format_number(salt)); }
    if medical > 0 { println!("  Medical: ${}", format_number(medical)); }
    if retirement > 0 { println!("  Retirement: ${}", format_number(retirement)); }
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    let start = Instant::now();
    let mut builder = DeductionProofBuilder::new(filing_status, year);
    if charitable > 0 { builder = builder.add(DeductionCategory::Charitable, charitable); }
    if mortgage > 0 { builder = builder.add(DeductionCategory::MortgageInterest, mortgage); }
    if salt > 0 { builder = builder.add(DeductionCategory::StateLocalTaxes, salt); }
    if medical > 0 { builder = builder.add(DeductionCategory::Medical, medical); }
    if retirement > 0 { builder = builder.add(DeductionCategory::Retirement, retirement); }

    let proof = builder.build().expect("Deduction proof failed");
    let elapsed = start.elapsed();

    let standard_deduction = if filing_status == FilingStatus::MarriedFilingJointly { 29_200 } else { 14_600 };

    println!("Output (PUBLIC):");
    println!("  Total deductions: ${} - ${}",
        format_number(proof.total_lower), format_number(proof.total_upper));
    println!("  Categories claimed: {}", proof.category_count);
    println!("  Standard deduction ({}): ${}", year, format_number(standard_deduction));
    println!("  Should itemize? {}", if proof.total_lower > standard_deduction { "YES" } else { "NO" });
    println!("  Commitment: {}", proof.commitment);
    println!("\nGenerated in {:?}", elapsed);
    println!("\nThis proves: 'I have at least ${} in deductions'", format_number(proof.total_lower));
    println!("WITHOUT revealing exact amounts per category!");
}

fn cmd_composite(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let jurisdiction_str = parse_arg_str(args, "--jurisdiction").unwrap_or("US");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let jurisdiction: Jurisdiction = jurisdiction_str.parse().unwrap();
    let filing_status: FilingStatus = status_str.parse().unwrap();

    println!("=== Composite Proof (All-in-One) ===\n");
    println!("Input (PRIVATE):");
    println!("  Income: ${}", format_number(income));
    println!("  Jurisdiction: {}", jurisdiction.name());
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    let start = Instant::now();
    let proof = CompositeProofBuilder::new(income, jurisdiction, filing_status, year)
        .with_bracket()
        .with_effective_rate()
        .with_range(0, income * 2)  // Prove income is below 2x stated
        .build_dev()
        .expect("Composite proof failed");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  Components included: {}", proof.component_count());

    if proof.has_bracket() {
        if let Some(ref bp) = proof.bracket_proof {
            println!("  [Bracket] {} ({}%)", bp.bracket_index, bp.rate_bps as f64 / 100.0);
        }
    }
    if proof.has_effective_rate() {
        if let Some(ref ep) = proof.effective_rate_proof {
            println!("  [Effective] {:.1}% (marginal: {:.0}%)",
                ep.effective_rate_percent(), ep.marginal_rate_percent());
        }
    }
    if proof.has_range() {
        if let Some(ref rp) = proof.range_proof {
            println!("  [Range] ${} - ${}",
                format_number(rp.range_lower), format_number(rp.range_upper));
        }
    }

    println!("  Verified: {}", proof.verify().is_ok());
    println!("\nGenerated in {:?}", elapsed);
    println!("\nThis single proof attests to bracket, effective rate, AND income range!");
}

fn cmd_chain(args: &[String]) {
    let jurisdiction_str = parse_arg_str(args, "--jurisdiction").unwrap_or("US");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let years_str = parse_arg_str(args, "--years").unwrap_or("2022,2023,2024");
    let incomes_str = parse_arg_str(args, "--incomes").unwrap_or("75000,80000,85000");

    let jurisdiction: Jurisdiction = jurisdiction_str.parse().unwrap();
    let filing_status: FilingStatus = status_str.parse().unwrap();

    let years: Vec<u32> = years_str.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let incomes: Vec<u64> = incomes_str.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    if years.len() != incomes.len() {
        eprintln!("Error: years and incomes must have same length");
        std::process::exit(1);
    }

    println!("=== Proof Chain (Immutable Audit Trail) ===\n");
    println!("Input (PRIVATE):");
    for (y, i) in years.iter().zip(incomes.iter()) {
        println!("  {}: ${}", y, format_number(*i));
    }
    println!("  Jurisdiction: {}", jurisdiction.name());
    println!("  Filing Status: {}\n", filing_status.name());

    let start = Instant::now();
    let prover = TaxBracketProver::dev_mode();
    let mut chain = ProofChain::new("cli-user");

    for (year, income) in years.iter().zip(incomes.iter()) {
        let proof = prover.prove(*income, jurisdiction, filing_status, *year)
            .expect("Proof generation failed");
        chain.add_bracket_proof(&proof);
    }
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  Chain length: {} proofs", chain.len());
    println!("  Chain integrity: {}", if chain.verify().is_ok() { "VALID" } else { "INVALID" });

    println!("\n  Proof links:");
    for (i, link) in chain.links.iter().enumerate() {
        let link_hex = link.link_hash.to_hex();
        let prev_hex = link.previous_hash.to_hex();
        println!("    [{}] {}... -> prev: {}...",
            i,
            &link_hex[..16],
            &prev_hex[..8]);
    }

    let final_hash = chain.links.last()
        .map(|l| l.link_hash.to_hex())
        .unwrap_or_else(|| "empty".to_string());
    println!("\n  Final chain hash: {}...", &final_hash[..32.min(final_hash.len())]);
    println!("\nGenerated in {:?}", elapsed);
    println!("\nEach proof is cryptographically linked via SHA3-256!");
    println!("Tampering with ANY proof invalidates the entire chain.");
}

fn cmd_state(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let state_str = parse_arg_str(args, "--state").unwrap_or("CA");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let state: USState = state_str.parse().unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let filing_status: FilingStatus = status_str.parse().unwrap();

    println!("=== US State Tax Proof ===\n");
    println!("Input (PRIVATE):");
    println!("  Income: ${}", format_number(income));
    println!("  State: {} ({})", state.name(), state_str.to_uppercase());
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    if !state.has_income_tax() {
        println!("Output:");
        println!("  {} has NO state income tax!", state.name());
        println!("  Rate: 0%");
        return;
    }

    let start = Instant::now();
    let brackets = get_state_brackets(state, year, filing_status)
        .expect("Failed to get state brackets");

    let bracket = brackets.iter()
        .find(|b| income >= b.lower && income < b.upper)
        .expect("No bracket found");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  Bracket: {} ({}%)", bracket.index, bracket.rate_bps as f64 / 100.0);
    println!("  Range: ${} - {}",
        format_number(bracket.lower),
        if bracket.upper == u64::MAX { "+".to_string() }
        else { format!("${}", format_number(bracket.upper)) });
    println!("  Tax type: {}", if state.is_flat_tax() { "Flat" } else { "Progressive" });
    println!("\nGenerated in {:?}", elapsed);

    // Show comparison with federal
    let prover = TaxBracketProver::dev_mode();
    if let Ok(federal) = prover.prove(income, Jurisdiction::US, filing_status, year) {
        let combined = bracket.rate_bps + federal.rate_bps;
        println!("\nCombined with federal:");
        println!("  Federal marginal: {}%", federal.rate_bps as f64 / 100.0);
        println!("  State marginal: {}%", bracket.rate_bps as f64 / 100.0);
        println!("  Combined marginal: {}%", combined as f64 / 100.0);
    }
}

fn cmd_canton(args: &[String]) {
    let income: u64 = parse_arg(args, "--income").unwrap_or_else(|| {
        eprintln!("Error: --income required");
        std::process::exit(1);
    });
    let canton_str = parse_arg_str(args, "--canton").unwrap_or("ZH");
    let status_str = parse_arg_str(args, "--status").unwrap_or("single");
    let year: u32 = parse_arg(args, "--year").unwrap_or(2024);

    let canton: SwissCanton = canton_str.parse().unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let filing_status: FilingStatus = status_str.parse().unwrap();

    println!("=== Swiss Canton Tax Proof ===\n");
    println!("Input (PRIVATE):");
    println!("  Income: CHF {}", format_number(income));
    println!("  Canton: {} ({})", canton.name(), canton_str.to_uppercase());
    println!("  Language: {}", canton.language());
    println!("  Filing Status: {}", filing_status.name());
    println!("  Tax Year: {}\n", year);

    let start = Instant::now();
    let brackets = get_cantonal_brackets(canton, year, filing_status)
        .expect("Failed to get cantonal brackets");

    let bracket = brackets.iter()
        .find(|b| income >= b.lower && income < b.upper)
        .expect("No bracket found");
    let elapsed = start.elapsed();

    println!("Output (PUBLIC):");
    println!("  Bracket: {} ({}%)", bracket.index, bracket.rate_bps as f64 / 100.0);
    println!("  Range: CHF {} - {}",
        format_number(bracket.lower),
        if bracket.upper == u64::MAX { "+".to_string() }
        else { format!("CHF {}", format_number(bracket.upper)) });
    println!("\nGenerated in {:?}", elapsed);

    // Low tax canton note
    let low_tax_cantons = ["ZG", "SZ", "NW", "OW", "AI", "AR", "UR"];
    if low_tax_cantons.contains(&canton_str.to_uppercase().as_str()) {
        println!("\nNote: {} is known as a low-tax canton!", canton.name());
    }
}
