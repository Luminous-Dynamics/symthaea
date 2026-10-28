use science_bench_refresh_validator::validate_manifest_bytes;
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage(program: &str) {
    eprintln!("usage: {program} MANIFEST.json [--require-scoring|--require-sealed]");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        usage(args.first().map(String::as_str).unwrap_or("science-bench-refresh-validator"));
        return ExitCode::from(2);
    }

    let mode = if args.len() == 3 {
        match args[2].as_str() {
            "--require-scoring" => "scoring",
            "--require-sealed" => "sealed",
            other => {
                eprintln!("unknown option: {other}");
                usage(&args[0]);
                return ExitCode::from(2);
            }
        }
    } else {
        "report"
    };

    let bytes = match fs::read(&args[1]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read manifest {}: {error}", args[1]);
            return ExitCode::from(2);
        }
    };

    let report = match validate_manifest_bytes(&bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("cannot serialize validation report: {error}");
            return ExitCode::from(2);
        }
    }

    if !report.valid {
        return ExitCode::from(2);
    }
    if mode == "scoring" && !report.admissible_for_scoring {
        return ExitCode::from(3);
    }
    if mode == "sealed" && !report.eligible_for_sealed_generalization_claim {
        return ExitCode::from(4);
    }
    ExitCode::SUCCESS
}
