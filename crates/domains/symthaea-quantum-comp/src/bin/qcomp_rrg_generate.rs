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


fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch <= '\u{001f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn json_string_array(values: &[String]) -> String {
    let items = values.iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

/// Structured failure receipt for a run that did not produce a graph.
/// DIMACS stdout remains empty on failure; this JSON is emitted on stderr.
fn failure_receipt_json(args: &[String], error: &str) -> String {
    format!(
        concat!(
            "{{",
            "\"schema\":\"org.luminousdynamics.symthaea.qcomp.random-regular-generation.v1\",",
            "\"status\":\"failed\",",
            "\"operation\":\"generate_simple_regular_graph\",",
            "\"arguments\":{},",
            "\"error\":\"{}\",",
            "\"output_graph_emitted\":false,",
            "\"limitations\":[\"failure receipt is not a generated workload\",",
            "\"no graph bytes or SHA-256 are available for a failed generation\"]",
            "}}"
        ),
        json_string_array(args),
        json_escape(error)
    )
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
            eprintln!("{}", failure_receipt_json(&args, &error));
            ExitCode::from(2)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_receipt_is_valid_json_shaped_and_escapes_untrusted_arguments() {
        let args = vec!["not-a-number".to_owned(), "3\n4".to_owned(), "seed\"value".to_owned()];
        let receipt = failure_receipt_json(&args, "invalid value: \"bad\"\nnext line");
        assert!(receipt.contains("\"status\":\"failed\""));
        assert!(receipt.contains("\"output_graph_emitted\":false"));
        assert!(receipt.contains("\"arguments\":[\"not-a-number\",\"3\\n4\",\"seed\\\"value\"]"));
        assert!(receipt.contains("\"error\":\"invalid value: \\\"bad\\\"\\nnext line\""));
    }

    #[test]
    fn retry_exhaustion_receipt_keeps_all_user_arguments() {
        let args = vec!["6".to_owned(), "3".to_owned(), "1".to_owned(), "1".to_owned()];
        let receipt = failure_receipt_json(
            &args,
            "no simple pairing found in 1 attempts (n=6, degree=3, seed=1)",
        );
        assert!(receipt.contains("\"arguments\":[\"6\",\"3\",\"1\",\"1\"]"));
        assert!(receipt.contains("\"status\":\"failed\""));
    }
}
