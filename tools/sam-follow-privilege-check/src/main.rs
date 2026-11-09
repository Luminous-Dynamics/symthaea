use std::env;
use std::path::PathBuf;

use sam_follow_privilege_check::validate_files;

fn main() {
    let mut args = env::args_os().skip(1);
    let schema = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("docs/autonomy/SAM_FOLLOW_PRIVILEGE_BENCHMARK_SCHEMA_V0_1.json")
    });
    let report = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("docs/autonomy/fixtures/SAM_FOLLOW_PRIVILEGE_BENCHMARK_V0_1.json")
    });
    if args.next().is_some() {
        eprintln!("usage: sam-follow-privilege-check [SCHEMA.json] [REPORT.json]");
        std::process::exit(2);
    }
    match validate_files(&schema, &report) {
        Ok(()) => println!("PASS: {} conforms to {} (Draft 2020-12 + SAM privilege semantics)", report.display(), schema.display()),
        Err(errors) => {
            for error in errors { eprintln!("ERROR: {error}"); }
            std::process::exit(1);
        }
    }
}
