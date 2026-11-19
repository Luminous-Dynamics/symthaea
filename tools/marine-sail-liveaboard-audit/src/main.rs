use marine_sail_liveaboard_audit::{audit_case, compare_pair, print_json, read_json};
use std::env;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "Usage:\n  marine-sail-liveaboard-audit audit-case <case.json>\n  marine-sail-liveaboard-audit compare-pair <baseline.json> <candidate.json>"
    );
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    match arguments.get(1).map(String::as_str) {
        Some("audit-case") if arguments.len() == 3 => {
            let case = read_json(&arguments[2])?;
            print_json(&audit_case(&case).to_value())
        }
        Some("compare-pair") if arguments.len() == 4 => {
            let baseline = read_json(&arguments[2])?;
            let candidate = read_json(&arguments[3])?;
            print_json(&compare_pair(&baseline, &candidate))
        }
        _ => {
            usage();
            Err("invalid command line".to_owned())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
