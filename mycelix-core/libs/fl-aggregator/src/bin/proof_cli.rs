//! Proof CLI Tool
//!
//! Command-line interface for generating and verifying zkSTARK proofs.
//!
//! ## Usage
//!
//! ```bash
//! # Generate a range proof
//! proof-cli range --value 42 --min 0 --max 100
//!
//! # Verify identity assurance
//! proof-cli identity --did "did:mycelix:alice" --level E2 --factors "0.5:0:true,0.3:1:true"
//!
//! # Check vote eligibility
//! proof-cli vote --did "did:mycelix:voter" --proposal constitutional
//! ```

use clap::{Parser, Subcommand};
use fl_aggregator::proofs::{
    RangeProof, GradientIntegrityProof, IdentityAssuranceProof, VoteEligibilityProof,
    ProofConfig, SecurityLevel, ProofAssuranceLevel,
    ProofIdentityFactor, ProofProposalType, ProofVoterProfile,
};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "proof-cli")]
#[command(about = "zkSTARK proof generation and verification CLI")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Security level (96, 128, 256)
    #[arg(long, default_value = "128")]
    security: u16,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    format: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate and verify a range proof
    Range {
        /// Value to prove is in range
        #[arg(long)]
        value: u64,

        /// Minimum value (inclusive)
        #[arg(long)]
        min: u64,

        /// Maximum value (inclusive)
        #[arg(long)]
        max: u64,
    },

    /// Generate and verify a gradient integrity proof
    Gradient {
        /// Comma-separated gradient values
        #[arg(long)]
        values: String,

        /// Maximum allowed L2 norm
        #[arg(long, default_value = "10.0")]
        max_norm: f32,
    },

    /// Generate and verify an identity assurance proof
    Identity {
        /// Decentralized identifier
        #[arg(long)]
        did: String,

        /// Minimum assurance level (E0, E1, E2, E3, E4)
        #[arg(long, default_value = "E1")]
        level: String,

        /// Factors as "contribution:category:active,..." (e.g., "0.5:0:true,0.3:1:true")
        #[arg(long)]
        factors: String,
    },

    /// Generate and verify a vote eligibility proof
    Vote {
        /// Voter DID
        #[arg(long)]
        did: String,

        /// Proposal type (standard, constitutional, model, emergency, treasury, membership)
        #[arg(long, default_value = "standard")]
        proposal: String,

        /// Assurance level (0-4)
        #[arg(long, default_value = "2")]
        assurance: u8,

        /// MATL score (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        matl: f32,

        /// Stake amount
        #[arg(long, default_value = "100.0")]
        stake: f32,

        /// Account age in days
        #[arg(long, default_value = "30")]
        age: u32,

        /// Participation rate (0.0-1.0)
        #[arg(long, default_value = "0.2")]
        participation: f32,

        /// Has humanity proof
        #[arg(long)]
        humanity: bool,

        /// FL contributions count
        #[arg(long, default_value = "0")]
        fl_contributions: u32,
    },

    /// Run benchmarks
    Bench {
        /// Number of iterations
        #[arg(long, default_value = "5")]
        iterations: usize,
    },
}

fn get_config(security: u16) -> ProofConfig {
    let level = match security {
        96 => SecurityLevel::Standard96,
        256 => SecurityLevel::High256,
        _ => SecurityLevel::Standard128,
    };

    ProofConfig {
        security_level: level,
        parallel: false,
        max_proof_size: 0,
    }
}

fn parse_assurance_level(s: &str) -> Option<ProofAssuranceLevel> {
    match s.to_uppercase().as_str() {
        "E0" => Some(ProofAssuranceLevel::E0),
        "E1" => Some(ProofAssuranceLevel::E1),
        "E2" => Some(ProofAssuranceLevel::E2),
        "E3" => Some(ProofAssuranceLevel::E3),
        "E4" => Some(ProofAssuranceLevel::E4),
        _ => None,
    }
}

fn parse_proposal_type(s: &str) -> Option<ProofProposalType> {
    match s.to_lowercase().as_str() {
        "standard" => Some(ProofProposalType::Standard),
        "constitutional" => Some(ProofProposalType::Constitutional),
        "model" | "modelgovernance" => Some(ProofProposalType::ModelGovernance),
        "emergency" => Some(ProofProposalType::Emergency),
        "treasury" => Some(ProofProposalType::Treasury),
        "membership" => Some(ProofProposalType::Membership),
        _ => None,
    }
}

fn parse_factors(s: &str) -> Vec<ProofIdentityFactor> {
    s.split(',')
        .filter_map(|factor_str| {
            let parts: Vec<&str> = factor_str.trim().split(':').collect();
            if parts.len() >= 3 {
                let contribution = parts[0].parse::<f32>().ok()?;
                let category = parts[1].parse::<u8>().ok()?;
                let active = parts[2].parse::<bool>().ok().unwrap_or(true);
                Some(ProofIdentityFactor::new(contribution, category, active))
            } else {
                None
            }
        })
        .collect()
}

fn main() {
    let cli = Cli::parse();
    let config = get_config(cli.security);
    let is_json = cli.format == "json";

    match cli.command {
        Commands::Range { value, min, max } => {
            println!("Generating RangeProof...");
            println!("  Value: {} in [{}, {}]", value, min, max);

            let start = Instant::now();
            match RangeProof::generate(value, min, max, config) {
                Ok(proof) => {
                    let gen_time = start.elapsed();
                    println!("  Generation time: {:?}", gen_time);
                    println!("  Proof size: {} bytes", proof.size());

                    let verify_start = Instant::now();
                    match proof.verify() {
                        Ok(result) => {
                            let verify_time = verify_start.elapsed();
                            println!("  Verification time: {:?}", verify_time);
                            println!("  Valid: {}", result.valid);
                            if let Some(details) = &result.details {
                                println!("  Details: {}", details);
                            }
                        }
                        Err(e) => eprintln!("Verification error: {:?}", e),
                    }
                }
                Err(e) => eprintln!("Generation error: {:?}", e),
            }
        }

        Commands::Gradient { values, max_norm } => {
            let gradient: Vec<f32> = values
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            println!("Generating GradientIntegrityProof...");
            println!("  Elements: {}", gradient.len());
            println!("  Max norm: {}", max_norm);

            let start = Instant::now();
            match GradientIntegrityProof::generate(&gradient, max_norm, config) {
                Ok(proof) => {
                    let gen_time = start.elapsed();
                    println!("  Generation time: {:?}", gen_time);
                    println!("  Proof size: {} bytes", proof.size());

                    let verify_start = Instant::now();
                    match proof.verify() {
                        Ok(result) => {
                            let verify_time = verify_start.elapsed();
                            println!("  Verification time: {:?}", verify_time);
                            println!("  Valid: {}", result.valid);
                        }
                        Err(e) => eprintln!("Verification error: {:?}", e),
                    }
                }
                Err(e) => eprintln!("Generation error: {:?}", e),
            }
        }

        Commands::Identity { did, level, factors } => {
            let assurance_level = parse_assurance_level(&level)
                .expect("Invalid assurance level. Use E0, E1, E2, E3, or E4");

            let identity_factors = parse_factors(&factors);

            println!("Generating IdentityAssuranceProof...");
            println!("  DID: {}", did);
            println!("  Required level: {:?}", assurance_level);
            println!("  Factors: {}", identity_factors.len());

            let start = Instant::now();
            match IdentityAssuranceProof::generate(&did, &identity_factors, assurance_level, config) {
                Ok(proof) => {
                    let gen_time = start.elapsed();
                    println!("  Generation time: {:?}", gen_time);
                    println!("  Proof size: {} bytes", proof.size());
                    println!("  Final score: {}", proof.final_score());
                    println!("  Meets threshold: {}", proof.meets_threshold());

                    let verify_start = Instant::now();
                    match proof.verify() {
                        Ok(result) => {
                            let verify_time = verify_start.elapsed();
                            println!("  Verification time: {:?}", verify_time);
                            println!("  Valid: {}", result.valid);
                        }
                        Err(e) => eprintln!("Verification error: {:?}", e),
                    }
                }
                Err(e) => eprintln!("Generation error: {:?}", e),
            }
        }

        Commands::Vote {
            did, proposal, assurance, matl, stake,
            age, participation, humanity, fl_contributions,
        } => {
            let proposal_type = parse_proposal_type(&proposal)
                .expect("Invalid proposal type");

            let voter = ProofVoterProfile {
                did: did.clone(),
                assurance_level: assurance,
                matl_score: matl,
                stake,
                account_age_days: age,
                participation_rate: participation,
                has_humanity_proof: humanity,
                fl_contributions,
            };

            println!("Generating VoteEligibilityProof...");
            println!("  Voter: {}", did);
            println!("  Proposal type: {:?}", proposal_type);

            let start = Instant::now();
            match VoteEligibilityProof::generate(&voter, proposal_type, config) {
                Ok(proof) => {
                    let gen_time = start.elapsed();
                    println!("  Generation time: {:?}", gen_time);
                    println!("  Proof size: {} bytes", proof.size());
                    println!("  Eligible: {}", proof.is_eligible());
                    println!("  Requirements met: {}/{}",
                        proof.requirements_met(), proof.active_requirements());

                    let verify_start = Instant::now();
                    match proof.verify() {
                        Ok(result) => {
                            let verify_time = verify_start.elapsed();
                            println!("  Verification time: {:?}", verify_time);
                            println!("  Valid: {}", result.valid);
                        }
                        Err(e) => eprintln!("Verification error: {:?}", e),
                    }
                }
                Err(e) => eprintln!("Generation error: {:?}", e),
            }
        }

        Commands::Bench { iterations } => {
            println!("Running benchmarks ({} iterations)...\n", iterations);

            // RangeProof
            let mut total_gen = std::time::Duration::ZERO;
            let mut total_verify = std::time::Duration::ZERO;
            for _ in 0..iterations {
                let start = Instant::now();
                let proof = RangeProof::generate(500, 0, 1000, config.clone()).unwrap();
                total_gen += start.elapsed();

                let start = Instant::now();
                proof.verify().unwrap();
                total_verify += start.elapsed();
            }
            println!("RangeProof:");
            println!("  Avg generation: {:?}", total_gen / iterations as u32);
            println!("  Avg verification: {:?}", total_verify / iterations as u32);

            // IdentityProof
            let factors = vec![
                ProofIdentityFactor::new(0.5, 0, true),
                ProofIdentityFactor::new(0.3, 1, true),
            ];
            let mut total_gen = std::time::Duration::ZERO;
            let mut total_verify = std::time::Duration::ZERO;
            for _ in 0..iterations {
                let start = Instant::now();
                let proof = IdentityAssuranceProof::generate(
                    "did:bench", &factors, ProofAssuranceLevel::E2, config.clone()
                ).unwrap();
                total_gen += start.elapsed();

                let start = Instant::now();
                proof.verify().unwrap();
                total_verify += start.elapsed();
            }
            println!("\nIdentityAssuranceProof:");
            println!("  Avg generation: {:?}", total_gen / iterations as u32);
            println!("  Avg verification: {:?}", total_verify / iterations as u32);

            // VoteProof
            let voter = ProofVoterProfile {
                did: "did:bench".to_string(),
                assurance_level: 2,
                matl_score: 0.7,
                stake: 500.0,
                account_age_days: 100,
                participation_rate: 0.5,
                has_humanity_proof: true,
                fl_contributions: 20,
            };
            let mut total_gen = std::time::Duration::ZERO;
            let mut total_verify = std::time::Duration::ZERO;
            for _ in 0..iterations {
                let start = Instant::now();
                let proof = VoteEligibilityProof::generate(
                    &voter, ProofProposalType::Constitutional, config.clone()
                ).unwrap();
                total_gen += start.elapsed();

                let start = Instant::now();
                proof.verify().unwrap();
                total_verify += start.elapsed();
            }
            println!("\nVoteEligibilityProof:");
            println!("  Avg generation: {:?}", total_gen / iterations as u32);
            println!("  Avg verification: {:?}", total_verify / iterations as u32);

            println!("\nBenchmarks complete!");
        }
    }
}
