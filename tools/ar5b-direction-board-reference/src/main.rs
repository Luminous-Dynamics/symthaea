use std::env;
use std::fs;
use std::process::ExitCode;

use ar5b_direction_board_reference::parse_and_allocate;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: ar5b-direction-board-reference <campaign.json>");
        return ExitCode::from(64);
    };

    let input = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(66);
        }
    };

    let report = match parse_and_allocate(&input) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("campaign rejected: {error}");
            return ExitCode::from(2);
        }
    };

    match serde_json::to_string_pretty(&report) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("cannot serialize allocation report: {error}");
            return ExitCode::from(70);
        }
    }

    ExitCode::SUCCESS
}
