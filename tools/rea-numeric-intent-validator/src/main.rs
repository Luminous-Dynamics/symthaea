use rea_numeric_intent_validator::validate_inputs;
use rea_numeric_lexeme_audit::MAX_INPUT_BYTES;
use std::env;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;

fn read_bounded(path: &PathBuf, label: &str) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("{label}: {e}"))?;
    let mut bytes = Vec::new();
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{label}: {e}"))?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(format!("{label} exceeds {MAX_INPUT_BYTES} bytes"));
    }
    Ok(bytes)
}

fn run() -> Result<i32, String> {
    let mut args = env::args_os();
    let _program = args.next();
    let profile_path = args.next().map(PathBuf::from);
    let schema_path = args.next().map(PathBuf::from);
    let document_path = args.next().map(PathBuf::from);
    if args.next().is_some() || profile_path.is_none() || schema_path.is_none() || document_path.is_none() {
        return Err(
            "usage: rea-numeric-intent-validate <profile.json> <value-schema.json> <document.json>"
                .to_owned(),
        );
    }

    let profile = read_bounded(&profile_path.unwrap(), "profile")?;
    let value_schema = read_bounded(&schema_path.unwrap(), "value schema")?;
    let document = read_bounded(&document_path.unwrap(), "document")?;
    let report = validate_inputs(&profile, &value_schema, &document)
        .map_err(|e| e.to_string())?;

    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report).map_err(|e| e.to_string())?;
    output.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(if report.overall_status == "conformant" { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("rea-numeric-intent-validate: {error}");
            std::process::exit(2);
        }
    }
}
