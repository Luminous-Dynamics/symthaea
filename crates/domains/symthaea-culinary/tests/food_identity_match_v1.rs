//! Integration tests for context-aware global food identity matching.
//!
//! Fixtures are synthetic. These tests validate matching-contract semantics,
//! not the accuracy of external source records or a food-matching model.

use serde_json::Value;
use symthaea_culinary::food_matching::validate_food_identity_match_v1;

const SCHEMA: &str =
    include_str!("../schemas/FoodIdentityMatchV1.schema.json");
const EXACT_MULTILINGUAL: &str =
    include_str!("../fixtures/food-identity-match-v1/exact-multilingual-potato.json");

fn fixture() -> Value {
    serde_json::from_str(EXACT_MULTILINGUAL).expect("food matching fixture is valid JSON")
}

fn violations(value: &Value) -> Vec<String> {
    validate_food_identity_match_v1(value).err().unwrap_or_default()
}

#[test]
fn schema_is_parseable_and_versioned() {
    let schema: Value = serde_json::from_str(SCHEMA).expect("schema must be valid JSON");
    assert_eq!(schema["title"], "FoodIdentityMatchV1");
    assert_eq!(schema["properties"]["schema_version"]["const"], "1.0.0");
}

#[test]
fn multilingual_aliases_can_refer_to_the_same_structured_food_identity() {
    let value = fixture();
    assert_eq!(value["query_record"]["names"][1]["language_tag"], "zu-ZA");
    assert_eq!(value["candidate_record"]["names"][0]["language_tag"], "en-US");
    assert!(validate_food_identity_match_v1(&value).is_ok());
}

#[test]
fn preparation_state_mismatch_cannot_be_exact_match() {
    let mut value = fixture();
    value["candidate_record"]["identity"]["preparation_state"] = serde_json::json!("boiled");
    value["attribute_checks"][3]["candidate_value"] = serde_json::json!("boiled");
    value["attribute_checks"][3]["relation"] = serde_json::json!("different");
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("exact_match requires \"preparation_state\"")));
    assert!(errors.iter().any(|error| error.contains("hard-block differences")));
}

#[test]
fn shared_or_similar_name_does_not_override_taxon_mismatch() {
    let mut value = fixture();
    value["candidate_record"]["identity"]["taxon_id"] = serde_json::json!("taxon:solanum-lycopersicum");
    value["attribute_checks"][0]["candidate_value"] = serde_json::json!("taxon:solanum-lycopersicum");
    value["attribute_checks"][0]["relation"] = serde_json::json!("different");
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("exact_match requires \"taxon_id\"")));
}

#[test]
fn missing_taxon_is_explicit_review_not_a_false_exact_match() {
    let mut value = fixture();
    value["outcome"] = serde_json::json!("review_required");
    value["candidate_record"]["identity"]["taxon_id"] = Value::Null;
    value["attribute_checks"][0]["candidate_value"] = Value::Null;
    value["attribute_checks"][0]["relation"] = serde_json::json!("unknown");
    assert!(validate_food_identity_match_v1(&value).is_ok());
}

#[test]
fn a_qualified_recipe_substitution_must_record_its_deviation() {
    let mut value = fixture();
    value["use_scope"] = serde_json::json!("recipe_substitution");
    value["outcome"] = serde_json::json!("qualified_approximation");
    value["query_record"]["identity"]["cultivar"] = serde_json::json!("cultivar:a");
    value["candidate_record"]["identity"]["cultivar"] = serde_json::json!("cultivar:b");
    value["attribute_checks"].as_array_mut().unwrap().push(serde_json::json!({
        "attribute_id":"cultivar",
        "query_value":"cultivar:a",
        "candidate_value":"cultivar:b",
        "relation":"different"
    }));
    value["deviations"] = serde_json::json!([{
        "attribute_id":"cultivar",
        "reason":"Synthetic substitution example; sensory and nutrient equivalence are not asserted."
    }]);
    value["rationale"] = serde_json::json!("Synthetic candidate substitution for a recipe; not a claim of compositional identity.");
    assert!(validate_food_identity_match_v1(&value).is_ok());
}

#[test]
fn approximate_identity_is_not_allowed_for_allergen_screening() {
    let mut value = fixture();
    value["use_scope"] = serde_json::json!("allergen_screening");
    value["outcome"] = serde_json::json!("qualified_approximation");
    value["query_record"]["identity"]["cultivar"] = serde_json::json!("cultivar:a");
    value["candidate_record"]["identity"]["cultivar"] = serde_json::json!("cultivar:b");
    value["attribute_checks"].as_array_mut().unwrap().push(serde_json::json!({
        "attribute_id":"cultivar",
        "query_value":"cultivar:a",
        "candidate_value":"cultivar:b",
        "relation":"different"
    }));
    value["deviations"] = serde_json::json!([{
        "attribute_id":"cultivar",
        "reason":"Synthetic mismatch."
    }]);
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("not permitted for product_lookup, price_comparison, or allergen_screening")));
}

#[test]
fn component_bases_require_a_real_conversion_reference_when_different() {
    let mut value = fixture();
    value["component_claims"][0]["candidate_basis"] = serde_json::json!("g/100g dry matter");
    value["component_claims"][0]["comparability"] = serde_json::json!("converted");
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("requires conversion_ref")));
}

#[test]
fn attribute_check_cannot_rewrite_source_record_identity() {
    let mut value = fixture();
    value["candidate_record"]["identity"]["taxon_id"] = serde_json::json!("taxon:solanum-lycopersicum");
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("does not match the referenced record identity fields")));
}

#[test]
fn duplicate_attribute_checks_are_rejected() {
    let mut value = fixture();
    let first_check = value["attribute_checks"][0].clone();
    value["attribute_checks"].as_array_mut().unwrap().push(first_check);
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("duplicate attribute check")));
}

#[test]
fn malformed_language_tags_are_rejected() {
    let mut value = fixture();
    value["query_record"]["names"][0]["language_tag"] = serde_json::json!("en--ZA");
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("not a valid-looking BCP 47 tag")));
}

#[test]
fn exact_match_requires_explicit_assessment_of_fortification_and_brand_identity() {
    let mut value = fixture();
    let retained_checks: Vec<Value> = value["attribute_checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|check| check["attribute_id"] != "fortification")
        .cloned()
        .collect();
    value["attribute_checks"] = Value::Array(retained_checks);
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("hard-block identity attribute \"fortification\"")));
}

#[test]
fn qualified_approximation_cannot_hide_an_unknown_taxon() {
    let mut value = fixture();
    value["use_scope"] = serde_json::json!("recipe_substitution");
    value["outcome"] = serde_json::json!("qualified_approximation");
    value["candidate_record"]["identity"]["taxon_id"] = Value::Null;
    value["attribute_checks"][0]["candidate_value"] = Value::Null;
    value["attribute_checks"][0]["relation"] = serde_json::json!("unknown");
    value["deviations"] = serde_json::json!([{
        "attribute_id":"taxon_id",
        "reason":"Taxon is unknown and cannot support this approximation."
    }]);
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("cannot leave required identity attributes unknown/different")));
}

#[test]
fn approximate_match_cannot_hide_unknown_preparation_state() {
    let mut value = fixture();
    value["use_scope"] = serde_json::json!("recipe_substitution");
    value["outcome"] = serde_json::json!("qualified_approximation");
    value["candidate_record"]["identity"]["preparation_state"] = Value::Null;
    value["attribute_checks"][3]["candidate_value"] = Value::Null;
    value["attribute_checks"][3]["relation"] = serde_json::json!("unknown");
    value["deviations"] = serde_json::json!([{
        "attribute_id":"preparation_state",
        "reason":"Preparation state is unknown."
    }]);
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("cannot leave hard-block identity attributes unknown")));
}

#[test]
fn branded_product_cannot_claim_exact_identity_without_brand_id() {
    let mut value = fixture();
    value["query_record"]["identity"]["product_kind"] = serde_json::json!("branded_product");
    value["candidate_record"]["identity"]["product_kind"] = serde_json::json!("branded_product");
    value["attribute_checks"][4]["query_value"] = serde_json::json!("branded_product");
    value["attribute_checks"][4]["candidate_value"] = serde_json::json!("branded_product");
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("requires a known brand_id")));
}

#[test]
fn approximation_cannot_invent_a_deviation_without_an_attribute_conflict() {
    let mut value = fixture();
    value["use_scope"] = serde_json::json!("recipe_substitution");
    value["outcome"] = serde_json::json!("qualified_approximation");
    value["deviations"] = serde_json::json!([{
        "attribute_id":"cultivar",
        "reason":"Claimed cultivar difference is not present in the attribute assessment."
    }]);
    let errors = violations(&value);
    assert!(errors.iter().any(|error| error.contains("has no matching different/unknown attribute relation")));
}
