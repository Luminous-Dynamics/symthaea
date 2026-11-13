use rea_behavior_observation_check::validate_profile_bytes;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

const MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;

fn read_bounded(path: &PathBuf) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(format!(
            "{} exceeds {MAX_INPUT_BYTES} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let mut args = env::args_os();
    let _program = args.next();
    let Some(path) = args.next().map(PathBuf::from) else {
        return Err("usage: rea-behavior-observation-check <profile.json>".to_owned());
    };
    if args.next().is_some() {
        return Err("usage: rea-behavior-observation-check <profile.json>".to_owned());
    }

    let bytes = read_bounded(&path)?;
    let report = validate_profile_bytes(&bytes).map_err(|error| error.to_string())?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report).map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rea-behavior-observation-check: {error}");
        std::process::exit(2);
    }
}
