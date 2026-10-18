//! Conformance tests for CulinaryRegionalPackV1.
//!
//! Fixture packs are intentionally synthetic and configure no real regional
//! food, price, nutrition, cultural, or regulatory facts.

use serde_json::Value;
use symthaea_culinary::regionalization::validate_regional_pack_v1;

const SCHEMA: &str =
    include_str!("../schemas/CulinaryRegionalPackV1.schema.json");
const ZA: &str =
    include_str!("../fixtures/culinary-regionalization-v1/synthetic-za.json");
const IN: &str =
    include_str!("../fixtures/culinary-regionalization-v1/synthetic-in.json");
const BR: &str =
    include_str!("../fixtures/culinary-regionalization-v1/synthetic-br.json");

fn parse(raw: &str) -> Value {
    serde_json::from_str(raw).expect("fixture/schema must be valid JSON")
}

#[test]
fn schema_is_parseable_and_versioned() {
    let schema = parse(SCHEMA);
    assert_eq!(schema["title"], "CulinaryRegionalPackV1");
    assert_eq!(schema["properties"]["schema_version"]["const"], "1.0.0");
}

#[test]
fn diverse_locale_fixtures_validate_without_claiming_regional_data_coverage() {
    for raw in [ZA, IN, BR] {
        let pack = parse(raw);
        assert_eq!(pack["profile_status"], "synthetic_example");
        assert_eq!(pack["sources"].as_array().unwrap().len(), 0);
        assert!(pack["verticals"].as_array().unwrap().iter().all(|vertical| {
            vertical["status"] == "not_configured"
                && vertical["source_refs"].as_array().unwrap().is_empty()
        }));
        assert!(validate_regional_pack_v1(&pack).is_ok());
    }
}

#[test]
fn restricted_sources_cannot_claim_public_redistribution() {
    let mut pack = parse(ZA);
    pack["sources"] = serde_json::json!([{
        "source_id":"restricted-source",
        "publisher":"Synthetic example only",
        "reference":"synthetic:restricted-source",
        "source_kind":"synthetic_test",
        "access_class":"permission_required",
        "license_reference":"synthetic:permission-required",
        "redistribution_permitted":true,
        "status":"active",
        "known_at":"2026-01-01T00:00:00Z",
        "valid_from":"2026-01-01T00:00:00Z"
    }]);
    let errors = validate_regional_pack_v1(&pack).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("redistribution")));
}

#[test]
fn supported_vertical_without_source_is_rejected() {
    let mut pack = parse(IN);
    pack["verticals"][0]["status"] = serde_json::json!("supported");
    let errors = validate_regional_pack_v1(&pack).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("no sources")));
}

#[test]
fn a_vertical_cannot_reference_an_undeclared_source() {
    let mut pack = parse(BR);
    pack["verticals"][0]["status"] = serde_json::json!("partial");
    pack["verticals"][0]["source_refs"] = serde_json::json!(["source-that-does-not-exist"]);
    let errors = validate_regional_pack_v1(&pack).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("undeclared source_id")));
}

#[test]
fn country_code_must_not_stand_in_for_cultural_preference() {
    let mut pack = parse(ZA);
    pack["fallback_policy"]["infer_food_culture_from_country_code"] = serde_json::json!(true);
    let errors = validate_regional_pack_v1(&pack).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("infer food culture")));
}

#[test]
fn unit_conversions_must_have_explicit_evidence_and_basis() {
    let mut pack = parse(IN);
    pack["conventions"]["density_required_for_mass_volume_conversion"] = serde_json::json!(false);
    let errors = validate_regional_pack_v1(&pack).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("mass-volume conversion")));
}
