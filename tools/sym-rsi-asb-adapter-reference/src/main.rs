use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_asb_adapter_reference::parse_and_validate;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: sym-rsi-asb-adapter-reference <manifest.json>");
        return ExitCode::from(64);
    };

    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };

    match parse_and_validate(&contents) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("serializable validation report")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Err(error) => {
            eprintln!("invalid compatibility manifest JSON: {error}");
            ExitCode::from(65)
        }
    }
}
