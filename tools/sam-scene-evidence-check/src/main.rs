use std::env;
use std::path::PathBuf;

use sam_scene_evidence_check::validate_files;

fn main() {
    let mut args = env::args_os().skip(1);
    let schema = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/autonomy/SAM_SCENE_EVIDENCE_SCHEMA_V0_1.json"));
    let fixture = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("docs/autonomy/fixtures/SAM_SCENE_EVIDENCE_CONTRACT_V0_1.json")
    });

    if args.next().is_some() {
        eprintln!("usage: sam-scene-evidence-check [SCHEMA.json] [FIXTURE.json]");
        std::process::exit(2);
    }

    match validate_files(&schema, &fixture) {
        Ok(()) => {
            println!(
                "PASS: {} conforms to {} (Draft 2020-12 + SAM semantic checks)",
                fixture.display(),
                schema.display()
            );
        }
        Err(errors) => {
            for error in &errors {
                eprintln!("ERROR: {error}");
            }
            eprintln!("FAIL: {} validation error(s)", errors.len());
            std::process::exit(1);
        }
    }
}
