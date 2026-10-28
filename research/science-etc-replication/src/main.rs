use science_etc_replication_validator::validate_manifest_bytes;
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage(program: &str) {
    eprintln!(
        "usage:\n  {program} MANIFEST.json\n  {program} MANIFEST.json --require-qualified"
    );
}

fn run(args: &[String]) -> ExitCode {
    if args.is_empty() || args.len() > 2 {
        usage("science-etc-replication-validator");
        return ExitCode::from(2);
    }

    let require_qualified = if args.len() == 2 {
        match args[1].as_str() {
            "--require-qualified" => true,
            other => {
                eprintln!("unknown option: {other}");
                usage("science-etc-replication-validator");
                return ExitCode::from(2);
            }
        }
    } else {
        false
    };

    let bytes = match fs::read(&args[0]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read manifest {}: {error}", args[0]);
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

    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("validation report serializes")
    );

    if !report.valid {
        return ExitCode::from(2);
    }
    if require_qualified && !report.manifest_declares_scoring_eligible {
        return ExitCode::from(3);
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    run(&args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_arguments_are_rejected() {
        assert_eq!(run(&[]), ExitCode::from(2));
    }

    #[test]
    fn unknown_flag_is_rejected_before_file_read() {
        assert_eq!(
            run(&["missing-manifest.json".to_owned(), "--force-score".to_owned()]),
            ExitCode::from(2)
        );
    }
}
