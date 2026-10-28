use std::env;
use std::fs;
use std::process::ExitCode;

use vleb_cobol_index::{IndexOptions, index_fixed_format};

fn usage() -> &'static str {
    "Usage: vleb-cobol-index --format fixed [--include-debug] <source.cbl>"
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let mut explicit_fixed_format = false;
    let mut include_debug_lines = false;
    let mut source_path: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--format" => {
                let value = args.next().ok_or_else(|| usage().to_owned())?;
                if value != "fixed" {
                    return Err(format!(
                        "unsupported source format {value:?}; only explicit fixed format is supported\n{}",
                        usage()
                    ));
                }
                explicit_fixed_format = true;
            }
            "--include-debug" => include_debug_lines = true,
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(0);
            }
            option if option.starts_with('-') => {
                return Err(format!("unknown option {option:?}\n{}", usage()));
            }
            path => {
                if source_path.replace(path.to_owned()).is_some() {
                    return Err(format!("exactly one source path is accepted\n{}", usage()));
                }
            }
        }
    }

    if !explicit_fixed_format {
        return Err(format!(
            "source format must be declared explicitly; refusing to infer fixed/free format\n{}",
            usage()
        ));
    }
    let source_path = source_path.ok_or_else(|| usage().to_owned())?;
    let source = fs::read_to_string(&source_path).map_err(|error| {
        format!(
            "cannot read {source_path:?} as UTF-8 source text: {error}; decode the original encoding explicitly before indexing"
        )
    })?;

    let result = index_fixed_format(
        &source,
        IndexOptions {
            include_debug_lines,
        },
    );
    println!("{}", result.to_json());

    if result.diagnostics.is_empty() {
        Ok(0)
    } else {
        Ok(2)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(64)
        }
    }
}
