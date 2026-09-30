use std::collections::BTreeMap;
use std::fs;

use symthaea_civ_place::{canonical_json, derive_hostile_case, evaluate_fixture, sha256_hex, FixtureV1};

fn main() {
    let path = "docs/engineering/fixtures/civ-place-001b.json";
    let raw = fs::read(path).expect("fixture");
    let fixture: FixtureV1 = serde_json::from_slice(&raw).expect("fixture JSON");
    let evaluations = evaluate_fixture(&fixture).expect("evaluation");
    let hostile_cases = fixture.hostile_cases.iter().map(|case| {
        let value = derive_hostile_case(case, &fixture, &evaluations).expect("hostile case");
        (case.id.clone(), value)
    }).collect::<BTreeMap<_,_>>();

    let output = serde_json::json!({
        "profile": fixture.profile,
        "schema_version": fixture.schema_version,
        "fixture_sha256": sha256_hex(&raw),
        "services": evaluations,
        "hostile_cases": hostile_cases,
    });
    println!("{}", String::from_utf8(canonical_json(&output).expect("canonical")).unwrap());
}
