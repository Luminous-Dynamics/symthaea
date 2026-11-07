use rea_numeric_lexeme_audit::{MAX_INPUT_BYTES, audit_json};
use std::env;
use std::fs::File;
use std::io::{Read, Write};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let path = args
        .next()
        .ok_or("usage: rea-numeric-lexeme-audit <input.json>")?;
    if args.next().is_some() {
        return Err("expected exactly one input path".into());
    }

    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    let report = audit_json(&bytes)?;

    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rea-numeric-lexeme-audit: {error}");
        std::process::exit(2);
    }
}
