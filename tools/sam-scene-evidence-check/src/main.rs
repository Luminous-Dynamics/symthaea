use std::env;
use std::path::PathBuf;

use sam_scene_evidence_check::{validate_bundle_files, validate_files};

fn main() {
    let mut args = env::args_os().skip(1);
    let schema = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/autonomy/SAM_SCENE_EVIDENCE_SCHEMA_V0_1.json"));
    let fixture = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("docs/autonomy/fixtures/SAM_SCENE_EVIDENCE_CONTRACT_V0_1.json")
    });
    let traffic_schema = args.next().map(PathBuf::from);
    let traffic_fixture = args.next().map(PathBuf::from);

    if args.next().is_some() || traffic_schema.is_some() != traffic_fixture.is_some() {
        eprintln!(
            "usage: sam-scene-evidence-check [SCENE_SCHEMA.json] [SCENE_FIXTURE.json] [TRAFFIC_SCHEMA.json TRAFFIC_FIXTURE.json]"
        );
        std::process::exit(2);
    }

    let has_traffic = traffic_schema.is_some();
    let result = match (traffic_schema, traffic_fixture) {
        (Some(traffic_schema), Some(traffic_fixture)) => {
            validate_bundle_files(&schema, &fixture, &traffic_schema, &traffic_fixture)
        }
        (None, None) => validate_files(&schema, &fixture),
        _ => unreachable!("argument pairing checked above"),
    };

    match result {
        Ok(()) => {
            println!(
                "PASS: {} conforms to {} (scene schema, traffic profile and cross-contract identity checks)",
                fixture.display(),
                schema.display()
            );
            if has_traffic {
                println!(
                    "PASS: scene traffic_convention_profile_id resolves in the validated traffic profile registry"
                );
            }
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
