//! Proof Debugger CLI
//!
//! Interactive tool for inspecting, validating, and debugging zkSTARK proofs.
//!
//! ## Usage
//!
//! ```bash
//! # Inspect a proof envelope
//! proof-debugger inspect proof.bin
//!
//! # Verify a proof
//! proof-debugger verify proof.bin
//!
//! # Compare two proofs
//! proof-debugger compare proof1.bin proof2.bin
//!
//! # Benchmark verification
//! proof-debugger benchmark proof.bin --iterations 100
//! ```

use clap::{Parser, Subcommand};
use colored::Colorize;
use fl_aggregator::proofs::{ProofType, SecurityLevel, RangeProof, GradientIntegrityProof};
use fl_aggregator::proofs::integration::{
    ProofEnvelope, MAGIC_BYTES, SERIALIZATION_VERSION,
};
use fl_aggregator::proofs::gpu;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};

#[derive(Parser)]
#[command(name = "proof-debugger")]
#[command(about = "Debug and inspect zkSTARK proofs", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect a proof envelope
    Inspect {
        /// Path to proof file
        path: PathBuf,
        /// Show raw bytes
        #[arg(long)]
        raw: bool,
    },

    /// Verify a proof
    Verify {
        /// Path to proof file
        path: PathBuf,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Compare two proofs
    Compare {
        /// First proof file
        proof1: PathBuf,
        /// Second proof file
        proof2: PathBuf,
    },

    /// Benchmark proof verification
    Benchmark {
        /// Path to proof file
        path: PathBuf,
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
    },

    /// Generate a test proof
    Generate {
        /// Proof type (range, gradient, identity, vote)
        #[arg(short, long, default_value = "range")]
        proof_type: String,
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Validate proof format
    Validate {
        /// Path to proof file
        path: PathBuf,
    },

    /// Show detailed proof statistics
    Stats {
        /// Path to proof file
        path: PathBuf,
    },

    /// Show GPU acceleration status
    Gpu,

    /// Hex dump of proof bytes
    Hexdump {
        /// Path to proof file
        path: PathBuf,
        /// Number of bytes to show (default: 512)
        #[arg(short, long, default_value = "512")]
        bytes: usize,
        /// Offset to start from
        #[arg(short, long, default_value = "0")]
        offset: usize,
    },

    /// Analyze proof structure and constraints
    Analyze {
        /// Path to proof file
        path: PathBuf,
    },
}

#[derive(Tabled)]
struct ProofInfo {
    #[tabled(rename = "Field")]
    field: String,
    #[tabled(rename = "Value")]
    value: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { path, raw } => inspect_proof(&path, raw),
        Commands::Verify { path, verbose } => verify_proof(&path, verbose),
        Commands::Compare { proof1, proof2 } => compare_proofs(&proof1, &proof2),
        Commands::Benchmark { path, iterations } => benchmark_proof(&path, iterations),
        Commands::Generate { proof_type, output } => generate_proof(&proof_type, &output),
        Commands::Validate { path } => validate_proof(&path),
        Commands::Stats { path } => show_stats(&path),
        Commands::Gpu => show_gpu_status(),
        Commands::Hexdump { path, bytes, offset } => hexdump(&path, bytes, offset),
        Commands::Analyze { path } => analyze_proof(&path),
    }
}

fn inspect_proof(path: &PathBuf, show_raw: bool) {
    println!("{}", "Proof Inspection".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: Failed to read file: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    println!("File: {}", path.display().to_string().yellow());
    println!("Size: {} bytes", bytes.len().to_string().green());

    // Check magic bytes
    if bytes.len() >= 4 {
        let magic = &bytes[0..4];
        let magic_valid = magic == MAGIC_BYTES;
        println!(
            "Magic: {} ({})",
            format!("{:02X} {:02X} {:02X} {:02X}", magic[0], magic[1], magic[2], magic[3]).blue(),
            if magic_valid { "valid".green() } else { "invalid".red() }
        );
    }

    // Parse envelope
    match ProofEnvelope::from_bytes(&bytes) {
        Ok(envelope) => {
            println!();
            println!("{}", "Envelope Contents".bold());
            println!("{}", "-".repeat(50));

            let info = vec![
                ProofInfo { field: "Version".to_string(), value: envelope.version.to_string() },
                ProofInfo { field: "Proof Type".to_string(), value: format!("{:?}", envelope.proof_type) },
                ProofInfo { field: "Security Level".to_string(), value: format!("{:?}", envelope.security_level) },
                ProofInfo { field: "Proof Size".to_string(), value: format!("{} bytes", envelope.proof_bytes.len()) },
                ProofInfo { field: "Public Inputs Size".to_string(), value: format!("{} bytes", envelope.public_inputs.len()) },
                ProofInfo { field: "Compressed".to_string(), value: envelope.compressed.to_string() },
                ProofInfo { field: "Checksum".to_string(), value: format!("{:02X}{:02X}{:02X}{:02X}",
                    envelope.checksum[0], envelope.checksum[1], envelope.checksum[2], envelope.checksum[3]) },
            ];

            let table = Table::new(info).to_string();
            println!("{}", table);

            // Show public inputs
            if let Ok(inputs) = String::from_utf8(envelope.public_inputs.clone()) {
                println!();
                println!("{}", "Public Inputs (JSON)".bold());
                println!("{}", inputs.dimmed());
            }

            if show_raw {
                println!();
                println!("{}", "Raw Proof Bytes (first 256)".bold());
                let display_bytes = &envelope.proof_bytes[..envelope.proof_bytes.len().min(256)];
                for (i, chunk) in display_bytes.chunks(16).enumerate() {
                    print!("{:04X}: ", i * 16);
                    for byte in chunk {
                        print!("{:02X} ", byte);
                    }
                    println!();
                }
            }
        }
        Err(e) => {
            eprintln!("{}: Failed to parse envelope: {}", "Error".red(), e);
        }
    }
}

fn verify_proof(path: &PathBuf, verbose: bool) {
    println!("{}", "Proof Verification".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: Failed to read file: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    let envelope = match ProofEnvelope::from_bytes(&bytes) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{}: {}", "Parse Error".red(), e);
            std::process::exit(1);
        }
    };

    println!("Proof Type: {:?}", envelope.proof_type);
    println!();

    let start = Instant::now();

    // Envelope-level validation:
    // 1. Magic bytes verified during parse
    // 2. Checksum verified during parse
    // 3. Non-empty proof bytes
    // 4. Valid proof type
    let valid = !envelope.proof_bytes.is_empty() && matches!(
        envelope.proof_type,
        ProofType::Range |
        ProofType::GradientIntegrity |
        ProofType::IdentityAssurance |
        ProofType::VoteEligibility |
        ProofType::Membership
    );

    let duration = start.elapsed();

    if valid {
        println!("{}", "Verification: PASSED".green().bold());
        println!("(Envelope-level validation - format, checksum, structure)");
    } else {
        println!("{}", "Verification: FAILED".red().bold());
        if envelope.proof_bytes.is_empty() {
            println!("Reason: Empty proof bytes");
        } else {
            println!("Reason: Unsupported proof type");
        }
    }
    println!("Time: {:?}", duration);

    if verbose {
        println!();
        println!("{}", "Verification Details".bold());
        println!("Security: {:?}", envelope.security_level);
        println!("Proof bytes: {} bytes", envelope.proof_bytes.len());
        println!("Public inputs: {} bytes", envelope.public_inputs.len());
        println!("Compressed: {}", envelope.compressed);
    }
}

fn compare_proofs(path1: &PathBuf, path2: &PathBuf) {
    println!("{}", "Proof Comparison".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes1 = fs::read(path1).expect("Failed to read first file");
    let bytes2 = fs::read(path2).expect("Failed to read second file");

    let env1 = ProofEnvelope::from_bytes(&bytes1).expect("Failed to parse first proof");
    let env2 = ProofEnvelope::from_bytes(&bytes2).expect("Failed to parse second proof");

    println!("| {:20} | {:20} | {:20} |", "Field", "Proof 1", "Proof 2");
    println!("|{:-<22}|{:-<22}|{:-<22}|", "", "", "");

    let type_match = env1.proof_type == env2.proof_type;
    println!("| {:20} | {:20} | {:20} |",
        "Type",
        format!("{:?}", env1.proof_type),
        if type_match { format!("{:?}", env2.proof_type).green() } else { format!("{:?}", env2.proof_type).red() }
    );

    let sec_match = env1.security_level == env2.security_level;
    println!("| {:20} | {:20} | {:20} |",
        "Security Level",
        format!("{:?}", env1.security_level),
        if sec_match { format!("{:?}", env2.security_level).green() } else { format!("{:?}", env2.security_level).red() }
    );

    println!("| {:20} | {:20} | {:20} |",
        "Proof Size",
        format!("{} bytes", env1.proof_bytes.len()),
        format!("{} bytes", env2.proof_bytes.len())
    );

    let checksum_match = env1.checksum == env2.checksum;
    println!("| {:20} | {:20} | {:20} |",
        "Checksum Match",
        "N/A",
        if checksum_match { "Yes".green() } else { "No".yellow() }
    );
}

fn benchmark_proof(path: &PathBuf, iterations: usize) {
    println!("{}", "Proof Benchmark".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = fs::read(path).expect("Failed to read file");
    let envelope = ProofEnvelope::from_bytes(&bytes).expect("Failed to parse proof");

    println!("Proof Type: {:?}", envelope.proof_type);
    println!("Iterations: {}", iterations);
    println!("Benchmarking: envelope parsing + validation");
    println!();

    let mut times: Vec<Duration> = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        // Benchmark envelope parsing and validation
        let _ = ProofEnvelope::from_bytes(&bytes);
        times.push(start.elapsed());

        if (i + 1) % 10 == 0 {
            print!("\rProgress: {}/{}", i + 1, iterations);
        }
    }
    println!();

    if times.is_empty() {
        eprintln!("No successful verifications");
        return;
    }

    times.sort();

    let total: Duration = times.iter().sum();
    let avg = total / times.len() as u32;
    let min = times.first().unwrap();
    let max = times.last().unwrap();
    let p50 = times[times.len() / 2];
    let p95 = times[times.len() * 95 / 100];
    let p99 = times[times.len() * 99 / 100];

    println!();
    println!("{}", "Results".bold());
    println!("{}", "-".repeat(30));
    println!("Total:   {:?}", total);
    println!("Average: {:?}", avg);
    println!("Min:     {:?}", min);
    println!("Max:     {:?}", max);
    println!("P50:     {:?}", p50);
    println!("P95:     {:?}", p95);
    println!("P99:     {:?}", p99);
    println!("Throughput: {:.2} proofs/sec", iterations as f64 / total.as_secs_f64());
}

fn generate_proof(proof_type: &str, output: &PathBuf) {
    use fl_aggregator::proofs::ProofConfig;

    println!("{}", "Generating Test Proof".bold().cyan());

    let config = ProofConfig {
        security_level: SecurityLevel::Standard96,
        parallel: false,
        max_proof_size: 0,
    };

    let envelope = match proof_type.to_lowercase().as_str() {
        "range" => {
            let proof = RangeProof::generate(50, 0, 100, config).expect("Failed to generate");
            ProofEnvelope::from_range_proof(&proof).expect("Failed to create envelope")
        }
        "gradient" => {
            let gradient = vec![0.1, -0.2, 0.3, -0.4, 0.5];
            let proof = GradientIntegrityProof::generate(&gradient, 5.0, config)
                .expect("Failed to generate");
            ProofEnvelope::from_gradient_proof(&proof).expect("Failed to create envelope")
        }
        _ => {
            eprintln!("Unknown proof type: {}", proof_type);
            std::process::exit(1);
        }
    };

    let bytes = envelope.to_bytes().expect("Failed to serialize");
    fs::write(output, &bytes).expect("Failed to write file");

    println!("{}: {}", "Generated".green(), output.display());
    println!("Size: {} bytes", bytes.len());
}

fn validate_proof(path: &PathBuf) {
    println!("{}", "Proof Validation".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("{} File read: {}", "FAIL".red(), e);
            std::process::exit(1);
        }
    };

    // Check 1: File size
    if bytes.len() < 8 {
        println!("{} File too small ({} bytes)", "FAIL".red(), bytes.len());
        std::process::exit(1);
    }
    println!("{} File size: {} bytes", "PASS".green(), bytes.len());

    // Check 2: Magic bytes
    if &bytes[0..4] != MAGIC_BYTES {
        println!("{} Invalid magic bytes", "FAIL".red());
        std::process::exit(1);
    }
    println!("{} Magic bytes", "PASS".green());

    // Check 3: Parse envelope
    let envelope = match ProofEnvelope::from_bytes(&bytes) {
        Ok(e) => e,
        Err(e) => {
            println!("{} Envelope parse: {}", "FAIL".red(), e);
            std::process::exit(1);
        }
    };
    println!("{} Envelope structure", "PASS".green());

    // Check 4: Version
    if envelope.version > SERIALIZATION_VERSION {
        println!("{} Version {} > current {}", "WARN".yellow(), envelope.version, SERIALIZATION_VERSION);
    } else {
        println!("{} Version: {}", "PASS".green(), envelope.version);
    }

    // Check 5: Proof type
    println!("{} Proof type: {:?}", "PASS".green(), envelope.proof_type);

    // Check 6: Non-empty proof
    if envelope.proof_bytes.is_empty() {
        println!("{} Empty proof bytes", "FAIL".red());
        std::process::exit(1);
    }
    println!("{} Proof data present", "PASS".green());

    println!();
    println!("{}", "All validations passed!".green().bold());
}

fn show_stats(path: &PathBuf) {
    println!("{}", "Proof Statistics".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: Failed to read file: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    let envelope = match ProofEnvelope::from_bytes(&bytes) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{}: Failed to parse: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    println!("{}", "File Statistics".bold());
    println!("  Total file size:     {} bytes", bytes.len().to_string().green());
    println!("  Proof data size:     {} bytes", envelope.proof_bytes.len().to_string().green());
    println!("  Public inputs size:  {} bytes", envelope.public_inputs.len());
    println!("  Overhead:            {} bytes ({:.1}%)",
        bytes.len() - envelope.proof_bytes.len(),
        ((bytes.len() - envelope.proof_bytes.len()) as f64 / bytes.len() as f64) * 100.0
    );

    println!();
    println!("{}", "Proof Properties".bold());
    println!("  Type:            {:?}", envelope.proof_type);
    println!("  Security Level:  {:?}", envelope.security_level);
    println!("  Compressed:      {}", if envelope.compressed { "Yes".green() } else { "No".yellow() });
    println!("  Version:         {}", envelope.version);

    // Estimate security bits
    let security_bits = match envelope.security_level {
        SecurityLevel::Standard96 => 96,
        SecurityLevel::Standard128 => 128,
        SecurityLevel::High256 => 256,
    };
    println!("  Security bits:   ~{} bits", security_bits);

    println!();
    println!("{}", "Size Analysis".bold());

    // Estimate compression ratio if compressed
    if envelope.compressed {
        println!("  Compression:     Enabled (zstd)");
        println!("  Original size:   Unknown (compressed)");
    } else {
        println!("  Compression:     Disabled");
        println!("  Potential savings with compression: ~30-50%");
    }

    // Size per security bit
    let bytes_per_bit = envelope.proof_bytes.len() as f64 / security_bits as f64;
    println!("  Bytes/security bit: {:.1}", bytes_per_bit);

    // Efficiency rating
    let efficiency = if bytes_per_bit < 100.0 {
        "Excellent".green()
    } else if bytes_per_bit < 200.0 {
        "Good".green()
    } else if bytes_per_bit < 400.0 {
        "Average".yellow()
    } else {
        "Could be optimized".red()
    };
    println!("  Efficiency:      {}", efficiency);
}

fn show_gpu_status() {
    println!("{}", "GPU Acceleration Status".bold().cyan());
    println!("{}", "=".repeat(50));

    let status = gpu::check_gpu_availability();

    if status.available {
        println!("{} GPU acceleration available", "OK".green().bold());
        if let Some(device) = &status.device {
            println!();
            println!("{}", "Device Information".bold());
            println!("  Name:           {}", device.name.yellow());
            println!("  Backend:        {}", device.backend);
            println!("  Compute Units:  {}", device.compute_units);
            if device.total_memory > 0 {
                println!("  Total Memory:   {} MB", device.total_memory / 1024 / 1024);
                println!("  Free Memory:    {} MB", device.available_memory / 1024 / 1024);
            }
            println!("  Suitable:       {}", if device.suitable { "Yes".green() } else { "No".red() });
        }
    } else {
        println!("{} GPU acceleration not available", "WARN".yellow().bold());
        if let Some(error) = &status.error {
            println!("  Reason: {}", error.dimmed());
        }
    }

    println!();
    println!("{}", "Available Backends".bold());

    #[cfg(feature = "proofs-gpu-wgpu")]
    println!("  {} wgpu (WebGPU)", "Enabled ".green());
    #[cfg(not(feature = "proofs-gpu-wgpu"))]
    println!("  {} wgpu (WebGPU)", "Disabled".dimmed());

    #[cfg(feature = "proofs-gpu-cuda")]
    println!("  {} CUDA (NVIDIA)", "Enabled ".green());
    #[cfg(not(feature = "proofs-gpu-cuda"))]
    println!("  {} CUDA (NVIDIA)", "Disabled".dimmed());

    #[cfg(feature = "proofs-gpu-metal")]
    println!("  {} Metal (Apple)", "Enabled ".green());
    #[cfg(not(feature = "proofs-gpu-metal"))]
    println!("  {} Metal (Apple)", "Disabled".dimmed());

    #[cfg(feature = "proofs-gpu-opencl")]
    println!("  {} OpenCL", "Enabled ".green());
    #[cfg(not(feature = "proofs-gpu-opencl"))]
    println!("  {} OpenCL", "Disabled".dimmed());

    println!();
    println!("{}", "Performance Estimates".bold());
    println!("  Expected NTT speedup:        ~{}x", gpu::EXPECTED_NTT_SPEEDUP);
    println!("  Expected Merkle speedup:     ~{}x", gpu::EXPECTED_MERKLE_SPEEDUP);
    println!("  Expected Polynomial speedup: ~{}x", gpu::EXPECTED_POLYNOMIAL_SPEEDUP);

    println!();
    println!("{}", "Roadmap".bold());
    println!("  Current Phase: {}", gpu::roadmap::CURRENT_PHASE);
    println!("  Completed:");
    for item in gpu::roadmap::COMPLETED {
        println!("    {} {}", "✓".green(), item);
    }
    println!("  Next Steps:");
    for item in gpu::roadmap::NEXT_STEPS {
        println!("    {} {}", "○".dimmed(), item);
    }
}

fn hexdump(path: &PathBuf, num_bytes: usize, offset: usize) {
    println!("{}", "Proof Hex Dump".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: Failed to read file: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    println!("File: {}", path.display().to_string().yellow());
    println!("Total size: {} bytes", bytes.len());
    println!("Showing: {} bytes from offset {}", num_bytes.min(bytes.len() - offset), offset);
    println!();

    if offset >= bytes.len() {
        eprintln!("{}: Offset {} exceeds file size {}", "Error".red(), offset, bytes.len());
        std::process::exit(1);
    }

    let end = (offset + num_bytes).min(bytes.len());
    let display_bytes = &bytes[offset..end];

    // Header
    print!("{:8} ", "Offset".dimmed());
    for i in 0..16 {
        print!("{:02X} ", i);
    }
    print!(" ");
    println!("{}", "ASCII".dimmed());
    println!("{}", "-".repeat(76));

    // Hex dump with ASCII
    for (i, chunk) in display_bytes.chunks(16).enumerate() {
        let addr = offset + i * 16;
        print!("{:08X} ", addr);

        // Hex bytes
        for (j, byte) in chunk.iter().enumerate() {
            // Color code different sections
            let colored = if addr + j < 4 {
                format!("{:02X}", byte).red() // Magic bytes
            } else if addr + j < 8 {
                format!("{:02X}", byte).yellow() // Header
            } else {
                format!("{:02X}", byte).normal()
            };
            print!("{} ", colored);
        }

        // Padding for incomplete lines
        for _ in chunk.len()..16 {
            print!("   ");
        }

        print!(" ");

        // ASCII representation
        for byte in chunk {
            if *byte >= 32 && *byte < 127 {
                print!("{}", (*byte as char).to_string().green());
            } else {
                print!("{}", ".".dimmed());
            }
        }
        println!();
    }

    println!();
    println!("{}", "Legend".bold());
    println!("  {} Magic bytes (first 4 bytes)", "Red".red());
    println!("  {} Header bytes", "Yellow".yellow());
    println!("  {} Proof data", "White".normal());
}

fn analyze_proof(path: &PathBuf) {
    println!("{}", "Proof Analysis".bold().cyan());
    println!("{}", "=".repeat(50));

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: Failed to read file: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    let envelope = match ProofEnvelope::from_bytes(&bytes) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{}: Failed to parse: {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    println!("{}", "1. Format Analysis".bold());
    println!("   Magic bytes valid: {}", "Yes".green());
    println!("   Checksum valid:    {}", "Yes".green());
    println!("   Structure valid:   {}", "Yes".green());

    println!();
    println!("{}", "2. Proof Type Analysis".bold());
    match envelope.proof_type {
        ProofType::Range => {
            println!("   Type: Range Proof");
            println!("   Purpose: Proves value lies within [min, max] without revealing value");
            println!("   AIR: 3 columns (bit, range_flag, accumulator), 64 rows");
            println!("   Constraint degree: 2");
        }
        ProofType::GradientIntegrity => {
            println!("   Type: Gradient Integrity Proof");
            println!("   Purpose: Proves gradient is valid and within norm bounds");
            println!("   AIR: 5 columns, N rows (gradient length)");
            println!("   Constraint degree: 2");
        }
        ProofType::IdentityAssurance => {
            println!("   Type: Identity Assurance Proof");
            println!("   Purpose: Proves identity meets assurance level without revealing factors");
            println!("   AIR: 8 columns, 9 rows (one per factor type)");
            println!("   Constraint degree: 2");
        }
        ProofType::VoteEligibility => {
            println!("   Type: Vote Eligibility Proof");
            println!("   Purpose: Proves voter meets requirements without revealing specifics");
            println!("   AIR: 10 columns, 8 rows");
            println!("   Constraint degree: 2");
        }
        ProofType::Membership => {
            println!("   Type: Membership Proof");
            println!("   Purpose: Proves element exists in Merkle tree");
            println!("   AIR: 8 columns (hash path), variable rows (tree depth)");
            println!("   Constraint degree: 2");
        }
    }

    println!();
    println!("{}", "3. Security Analysis".bold());
    match envelope.security_level {
        SecurityLevel::Standard96 => {
            println!("   Level: Standard (96-bit)");
            println!("   Blowup factor: 8");
            println!("   FRI queries: 27");
            println!("   Suitable for: Most applications");
        }
        SecurityLevel::Standard128 => {
            println!("   Level: Standard (128-bit)");
            println!("   Blowup factor: 16");
            println!("   FRI queries: 40");
            println!("   Suitable for: High-value transactions");
        }
        SecurityLevel::High256 => {
            println!("   Level: High (256-bit)");
            println!("   Blowup factor: 32");
            println!("   FRI queries: 80");
            println!("   Suitable for: Critical infrastructure");
        }
    }

    println!();
    println!("{}", "4. Size Analysis".bold());
    let proof_kb = envelope.proof_bytes.len() as f64 / 1024.0;
    let size_rating = if proof_kb < 10.0 {
        "Compact".green()
    } else if proof_kb < 50.0 {
        "Normal".green()
    } else if proof_kb < 100.0 {
        "Large".yellow()
    } else {
        "Very Large".red()
    };
    println!("   Proof size: {:.2} KB ({})", proof_kb, size_rating);

    if envelope.compressed {
        println!("   Compression: Enabled (zstd)");
    } else {
        println!("   Compression: {} - enable for ~30-50% reduction", "Disabled".yellow());
    }

    println!();
    println!("{}", "5. Public Inputs".bold());
    if let Ok(json) = String::from_utf8(envelope.public_inputs.clone()) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
            println!("   Format: JSON");
            if let Some(obj) = parsed.as_object() {
                for (key, value) in obj {
                    println!("   {}: {}", key.cyan(), value);
                }
            } else {
                println!("   {}", json.dimmed());
            }
        } else {
            println!("   Raw: {}", json.dimmed());
        }
    } else {
        println!("   Format: Binary ({} bytes)", envelope.public_inputs.len());
    }

    println!();
    println!("{}", "6. Recommendations".bold());
    let mut recs = Vec::new();

    if !envelope.compressed && envelope.proof_bytes.len() > 20_000 {
        recs.push("Enable compression for large proofs");
    }

    if matches!(envelope.security_level, SecurityLevel::Standard96) && envelope.proof_bytes.len() > 50_000 {
        recs.push("Consider if Standard security is sufficient for this proof size");
    }

    if recs.is_empty() {
        println!("   {} No issues detected", "✓".green());
    } else {
        for rec in recs {
            println!("   {} {}", "○".yellow(), rec);
        }
    }
}
