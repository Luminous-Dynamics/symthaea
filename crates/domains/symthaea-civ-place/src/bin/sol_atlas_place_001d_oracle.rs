use std::{env, fs};

use serde_json::json;
use symthaea_civ_place::sol_atlas::{evaluate, project, SolAtlasQualificationInputV1};
use symthaea_civ_place::{canonical_json, sha256_hex};

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "docs/engineering/fixtures/sol-atlas-place-001d.json".into());
    let raw = fs::read(&path).expect("Sol Atlas fixture");
    let input: SolAtlasQualificationInputV1 =
        serde_json::from_slice(&raw).expect("Sol Atlas fixture JSON");

    let (nodes, deps, projections, services, receipt) = project(&input).expect("Sol Atlas projection");
    let evaluations = evaluate(&input).expect("CIV-PLACE evaluation");

    let output = json!({
        "profile": input.profile,
        "schema_version": input.plan.schema_version,
        "fixture_sha256": sha256_hex(&raw),
        "plan_sha256": receipt.plan_sha256,
        "projection": {
            "nodes": nodes,
            "dependencies": deps,
            "projections": projections,
            "services": services,
            "receipt": receipt,
        },
        "evaluations": evaluations,
    });

    println!(
        "{}",
        String::from_utf8(canonical_json(&output).expect("canonical output")).expect("utf8")
    );
}
