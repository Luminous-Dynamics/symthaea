use std::env;
use std::process::ExitCode;

use symthaea_chaos_reference_suite::{digest_non_security, generate, FixtureId};

fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T, String> {
    match args.get(index) {
        Some(value) => value.parse::<T>().map_err(|_| format!("invalid argument at position {index}: {value}")),
        None => Ok(default),
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let sample_count: usize = parse_arg(&args, 1, 2048)?;
    let seed: u64 = parse_arg(&args, 2, 20_261_116)?;

    println!("# suite=symthaea-chaos-reference-suite-v0");
    println!("# seed={seed}");
    println!("# samples_per_fixture={sample_count}");
    println!("fixture,regime,step,x,y,z");

    for id in FixtureId::ALL {
        let samples = generate(id, sample_count, seed)?;
        println!("# fixture_id={}", id.as_str());
        println!("# regime={}", id.regime());
        println!("# burn_in={}", id.burn_in());
        println!("# digest_non_security={:016x}", digest_non_security(&samples));
        for sample in samples {
            println!(
                "{},{},{},{:.17e},{:.17e},{:.17e}",
                id.as_str(), id.regime(), sample.step, sample.x, sample.y, sample.z
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("chaos fixture generation failed: {error}");
            ExitCode::FAILURE
        }
    }
}
