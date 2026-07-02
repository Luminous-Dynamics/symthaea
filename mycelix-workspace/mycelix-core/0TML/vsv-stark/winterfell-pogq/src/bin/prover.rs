// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Winterfell PoGQ Prover CLI
//!
//! Drop-in replacement for RISC Zero host binary with compatible interface.
//!
//! Usage:
//!   winterfell-prover --public <public.json> --witness <witness.json> --output <proof.bin>
//!   winterfell-prover --verify --proof <proof.bin> --public <public.json>

use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use winterfell_pogq::{AirPublicInputs, PoGQProver, AIR_SCHEMA_REV};

#[derive(Parser)]
#[command(name = "winterfell-prover")]
#[command(about = "Winterfell-based PoGQ prover (3-10x faster than zkVM)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate proof
    Prove {
        /// Public inputs JSON file
        #[arg(short, long)]
        public: PathBuf,

        /// Witness (private) JSON file
        #[arg(short, long)]
        witness: PathBuf,

        /// Output proof file
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Verify proof
    Verify {
        /// Proof file to verify
        #[arg(short, long)]
        proof: PathBuf,

        /// Public inputs JSON file
        #[arg(short, long)]
        public: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prove {
            public,
            witness,
            output,
        } => {
            println!("🔬 Winterfell PoGQ Prover");
            println!("========================");

            // Load public inputs
            let public_json = fs::read_to_string(&public)?;
            let public_inputs: PublicInputsJson = serde_json::from_str(&public_json)?;

            // Load witness
            let witness_json = fs::read_to_string(&witness)?;
            let witness_data: WitnessJson = serde_json::from_str(&witness_json)?;

            // Convert to AIR inputs (Q16.16 scaling)
            let public = public_inputs.to_air_inputs();
            let witness_scores = witness_data.to_witness_scores();

            // Generate proof
            println!("\n📊 Inputs:");
            println!("  β = {:.2}", public.beta as f64 / 65536.0);
            println!("  w = {}", public.w);
            println!("  k = {}", public.k);
            println!("  m = {}", public.m);
            println!("  threshold = {:.2}", public.threshold as f64 / 65536.0);
            println!("  trace_length = {}", public.trace_length);

            println!("\n⚡ Generating proof...");
            let prover = PoGQProver::new();
            let result = prover.prove_exec(public.clone(), witness_scores)?;

            println!("\n✅ Proof generated!");
            println!("  Trace build: {}ms", result.trace_build_ms);
            println!("  Proving: {}ms", result.prove_ms);
            println!("  Serialization: {}ms", result.serialize_ms);
            println!("  Total: {}ms", result.total_ms);
            println!("  Proof size: {} bytes ({:.1} KB)", result.proof_bytes.len(), result.proof_bytes.len() as f64 / 1024.0);

            // Save proof
            fs::write(&output, &result.proof_bytes)?;
            println!("\n💾 Proof saved to: {}", output.display());

            // Auto-verify
            println!("\n🔍 Auto-verifying...");
            let valid = prover.verify_proof(&result.proof_bytes, public)?;
            if valid {
                println!("✅ Proof verified successfully!");
            } else {
                eprintln!("❌ Verification failed!");
                std::process::exit(1);
            }
        }
        Commands::Verify { proof, public } => {
            println!("🔍 Verifying Winterfell PoGQ Proof");
            println!("==================================");

            // Load proof
            let proof_bytes = fs::read(&proof)?;
            println!("📄 Proof: {} bytes", proof_bytes.len());

            // Load public inputs
            let public_json = fs::read_to_string(&public)?;
            let public_inputs: PublicInputsJson = serde_json::from_str(&public_json)?;
            let public = public_inputs.to_air_inputs();

            // Verify
            println!("\n⚡ Verifying...");
            let start = std::time::Instant::now();
            let prover = PoGQProver::new();
            let valid = prover.verify_proof(&proof_bytes, public)?;
            let verify_ms = start.elapsed().as_millis();

            if valid {
                println!("\n✅ Proof verified successfully in {}ms!", verify_ms);
            } else {
                eprintln!("\n❌ Verification failed!");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

// JSON input formats (matching vsv-stark/host interface)

#[derive(serde::Deserialize)]
struct PublicInputsJson {
    beta: f64,
    w: u64,
    k: u64,
    m: u64,
    threshold: f64,
    ema_init: f64,
    viol_init: u64,
    clear_init: u64,
    quar_init: u64,
    round_init: u64,
    quar_out: u64,
    trace_length: usize,
}

impl PublicInputsJson {
    fn to_air_inputs(&self) -> AirPublicInputs {
        let scale = |val: f64| (val * 65536.0) as u64;

        AirPublicInputs {
            beta: scale(self.beta),
            w: self.w,
            k: self.k,
            m: self.m,
            threshold: scale(self.threshold),
            ema_init: scale(self.ema_init),
            viol_init: self.viol_init,
            clear_init: self.clear_init,
            quar_init: self.quar_init,
            round_init: self.round_init,
            quar_out: self.quar_out,
            trace_length: self.trace_length,
            // Provenance (prover will populate automatically)
            prov_hash: [0, 0, 0, 0],
            profile_id: 128,
            air_rev: AIR_SCHEMA_REV,
        }
    }
}

#[derive(serde::Deserialize)]
struct WitnessJson {
    scores: Vec<f64>,
}

impl WitnessJson {
    fn to_witness_scores(&self) -> Vec<u64> {
        self.scores
            .iter()
            .map(|&val| (val * 65536.0) as u64)
            .collect()
    }
}
