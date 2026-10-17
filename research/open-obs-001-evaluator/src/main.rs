use open_obs_evaluator::{evaluate, parse_manifest};
use std::env;
use std::fs;
use std::process;

fn usage() -> &'static str {
    "usage: open-obs-evaluator MANIFEST.tsv --source-revision FULL_GIT_COMMIT_SHA"
}

fn valid_revision(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[1] != "--source-revision" {
        return Err(usage().to_owned());
    }
    let manifest_path = &args[0];
    let revision = &args[2];
    if !valid_revision(revision) {
        return Err("source revision must be a full 40- or 64-character hexadecimal commit identifier".to_owned());
    }

    let bytes = fs::read(manifest_path)
        .map_err(|error| format!("cannot read manifest {manifest_path:?}: {error}"))?;
    let parsed = parse_manifest(&bytes)?;
    let report = evaluate(&parsed, revision);
    print!("{}", report.render_tsv());

    // Qualification failures and insufficient evidence both fail closed for CI.
    // Exploratory users still receive the complete report on stdout.
    Ok(if report.overall_gate == open_obs_evaluator::Gate::Pass {
        0
    } else {
        1
    })
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}
