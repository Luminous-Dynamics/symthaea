//! Emit the frozen factorial case matrix as deterministic CSV.
//!
//! Usage:
//!   cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- discovery
//!   cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- confirmation
//!   cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- both

use std::io::{self, Write};
use std::process::ExitCode;

use muse_factorial_audit::{Cohort, generate_cases};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse-factorial-audit: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let cohort_arg = args.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing cohort; expected discovery, confirmation, or both",
        )
    })?;
    if args.next().is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "unexpected extra arguments").into());
    }

    let cohorts: Vec<Cohort> = match cohort_arg.as_str() {
        "discovery" => vec![Cohort::Discovery],
        "confirmation" => vec![Cohort::Confirmation],
        "both" => vec![Cohort::Discovery, Cohort::Confirmation],
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown cohort; expected discovery, confirmation, or both",
            )
            .into());
        }
    };

    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    writeln!(
        output,
        "case_id,cohort,style,seed,valence_code,arousal_code,energy_code,bars_code"
    )?;
    for cohort in cohorts {
        for case in generate_cases(cohort) {
            writeln!(
                output,
                "{},{},{},{},{},{},{},{}",
                case.case_id,
                cohort.as_str(),
                case.style,
                case.seed,
                case.cell.valence,
                case.cell.arousal,
                case.cell.energy,
                case.cell.bars
            )?;
        }
    }
    output.flush()?;
    Ok(())
}
