use std::{env, fs, path::PathBuf, process};

use symthaea_marine_geometry::full_envelope::{
    FullEnvelopeDisposition, FullEnvelopeSpec, validate_full_envelope_spec,
};

const USAGE: &str = concat!(
    "Usage: marine-full-envelope-validate <full-envelope-spec.json> [--output <report.json>]\n",
    "Validates the G2f-a full external hull input contract. It does not generate a surface or approve a vessel.\n",
    "Exit codes: 0 = valid input contract; 1 = incomplete/unsupported; 2 = rejected specification; 3 = CLI, I/O, or parse error."
);

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(path));
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(0);
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option: {value}\n{USAGE}"));
            }
            value => {
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("exactly one input file is accepted\n{USAGE}"));
                }
            }
        }
    }

    let input = input.ok_or_else(|| USAGE.to_string())?;
    let bytes =
        fs::read(&input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let specification: FullEnvelopeSpec = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid full-envelope specification: {error}"))?;
    let report = validate_full_envelope_spec(&specification);
    let disposition = report.disposition;
    let serialized = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize validation report: {error}"))?;

    if let Some(path) = output {
        fs::write(&path, serialized)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8_lossy(&serialized));
    }

    Ok(match disposition {
        FullEnvelopeDisposition::ValidInputContract => 0,
        FullEnvelopeDisposition::Incomplete => 1,
        FullEnvelopeDisposition::Rejected => 2,
    })
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(3);
        }
    }
}
