//! Emit a deterministic, canonical DIMACS edge graph from the QCOMP regular-graph generator.
//!
//! Example:
//!   cargo run -p symthaea-quantum-comp --bin qcomp_rrg_generate -- 1000 3 12345 > rrg.dimacs
//!
//! Use the exact generated file as the shared input to all solver arms. Record a SHA-256
//! of the file with the system's standard sha256sum/shasum tooling; this binary does not
//! claim to compute a cryptographic digest itself.

use std::env;
use std::process::ExitCode;

use symthaea_quantum_comp::{MisRandomRegularConfig, generate_mis_random_regular};

fn parse_seed(value: &str) -> Result<u64, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|error| format!("invalid hexadecimal seed {value:?}: {error}"))
    } else {
        value.parse::<u64>().map_err(|error| format!("invalid decimal seed {value:?}: {error}"))
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() < 3 || args.len() > 4 {
        return Err("expected VERTICES DEGREE SEED [MAX_ATTEMPTS]".to_owned());
    }
    let vertex_count = args[0].parse::<usize>()
        .map_err(|error| format!("invalid vertex count {:?}: {error}", args[0]))?;
    let degree = args[1].parse::<usize>()
        .map_err(|error| format!("invalid degree {:?}: {error}", args[1]))?;
    let seed = parse_seed(&args[2])?;
    let max_attempts = if args.len() == 4 {
        args[3].parse::<usize>()
            .map_err(|error| format!("invalid max attempts {:?}: {error}", args[3]))?
    } else {
        100_000
    };
    let config = MisRandomRegularConfig {
        vertex_count, degree, seed, max_attempts,
    };
    let (graph, report) = generate_mis_random_regular(config)
        .map_err(|error| error.to_string())?;

    // Keep stdout as parseable graph data only. Metadata is carried in DIMACS comments.
    println!("c generator symthaea-qcomp-random-regular-v1");
    println!("c seed {}", report.config.seed);
    println!("c rejection_attempts {}", report.attempts);
    println!("c max_attempts {}", report.config.max_attempts);
    println!("c construction_model configuration-model-conditioned-on-simple-graph");
    println!("c prng seeded-xorshift64star-noncryptographic");
    println!("p edge {} {}", graph.vertex_count(), graph.edges().len());
    for &(u, v) in graph.edges() {
        println!("e {} {}", u + 1, v + 1);
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("Usage: qcomp_rrg_generate VERTICES DEGREE SEED [MAX_ATTEMPTS]");
        println!("Emits canonical DIMACS; redirect it to a file and record sha256sum for provenance.");
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("qcomp_rrg_generate: {error}");
            eprintln!("Usage: qcomp_rrg_generate VERTICES DEGREE SEED [MAX_ATTEMPTS]");
            ExitCode::from(2)
        }
    }
}
