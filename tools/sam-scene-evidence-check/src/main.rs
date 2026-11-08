use std::env;
use std::path::PathBuf;

use sam_scene_evidence_check::{validate_files, validate_traffic_files};

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

    if let Err(errors) = validate_files(&schema, &fixture) {
        for error in &errors {
            eprintln!("ERROR: {error}");
        }
        eprintln!("FAIL: {} scene validation error(s)", errors.len());
        std::process::exit(1);
    }
    println!(
        "PASS: {} conforms to {} (Draft 2020-12 + SAM semantic checks)",
        fixture.display(),
        schema.display()
    );

    if let (Some(traffic_schema), Some(traffic_fixture)) = (traffic_schema, traffic_fixture) {
        if let Err(errors) = validate_traffic_files(&traffic_schema, &traffic_fixture) {
            for error in &errors {
                eprintln!("ERROR: {error}");
            }
            eprintln!(
                "FAIL: {} traffic convention validation error(s)",
                errors.len()
            );
            std::process::exit(1);
        }
        println!(
            "PASS: {} conforms to {} (Draft 2020-12 + traffic-side mirror semantics)",
            traffic_fixture.display(),
            traffic_schema.display()
        );
    }
}
