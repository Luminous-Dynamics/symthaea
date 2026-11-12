use rea_proof_obligation_fixture_check::score::{
    parse_unique_json, score_policy_submission,
};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;

fn read_bounded(path: &PathBuf, label: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("{label}: {error}"))?;
    let mut bytes = Vec::new();
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{label}: {error}"))?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(format!("{label} exceeds {MAX_INPUT_BYTES} bytes"));
    }
    Ok(bytes)
}

fn run() -> Result<i32, String> {
    let mut args = env::args_os();
    let _program = args.next();
    let agent_path = args.next().map(PathBuf::from);
    let grader_path = args.next().map(PathBuf::from);
    let submission_path = args.next().map(PathBuf::from);
    if args.next().is_some()
        || agent_path.is_none()
        || grader_path.is_none()
        || submission_path.is_none()
    {
        return Err(
            "usage: rea-proof-obligation-score <agent-view.json> <grader-view.json> <policy-run.json>"
                .to_owned(),
        );
    }

    let agent_bytes = read_bounded(&agent_path.unwrap(), "agent view")?;
    let grader_bytes = read_bounded(&grader_path.unwrap(), "grader view")?;
    let submission_bytes = read_bounded(&submission_path.unwrap(), "policy submission")?;

    let agent = parse_unique_json(&agent_bytes, "agent view").map_err(|error| error.to_string())?;
    let grader = parse_unique_json(&grader_bytes, "grader view").map_err(|error| error.to_string())?;
    let submission = parse_unique_json(&submission_bytes, "policy submission")
        .map_err(|error| error.to_string())?;

    let report = score_policy_submission(&agent, &grader, &submission)
        .map_err(|error| error.to_string())?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report).map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("rea-proof-obligation-score: {error}");
            std::process::exit(2);
        }
    }
}
