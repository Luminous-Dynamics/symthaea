//! Context-aware food identity matching for global culinary data exchange.
//!
//! This module validates an externally supplied match assessment. It does not
//! discover matches, translate food names, validate external ontology IDs, or
//! prove that a cited source is accurate. Names are not treated as identities:
//! critical preparation/processing/product attributes must agree for an exact match.

use serde_json::Value;
use std::collections::BTreeSet;

const REQUIRED_IDENTITY_ATTRIBUTES: &[&str] = &[
    "taxon_id",
    "food_part",
    "processing_state",
    "preparation_state",
    "product_kind",
    "edible_basis",
];

const ASSESSED_HARD_BLOCK_ATTRIBUTES: &[&str] = &[
    "taxon_id",
    "food_part",
    "processing_state",
    "preparation_state",
    "preservation",
    "product_kind",
    "brand_id",
    "fortification",
    "edible_basis",
];

const HARD_BLOCK_ATTRIBUTES: &[&str] = &[
    "taxon_id",
    "food_part",
    "processing_state",
    "preparation_state",
    "preservation",
    "product_kind",
    "brand_id",
    "fortification",
    "edible_basis",
];

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn language_tag_is_well_formed(tag: &str) -> bool {
    let mut parts = tag.split('-');
    let Some(language) = parts.next() else {
        return false;
    };
    (2..=8).contains(&language.len())
        && language.bytes().all(|byte| byte.is_ascii_alphabetic())
        && parts.all(|part| {
            !part.is_empty()
                && part.len() <= 8
                && part.bytes().all(|byte| byte.is_ascii_alphanumeric())
        })
}
fn identity_record_errors(record: &Value, label: &str, errors: &mut Vec<String>) {
    for field in ["record_id", "source_ref", "source_version"] {
        if non_empty(record, field).is_none() {
            errors.push(format!("{label}.{field} must be a non-empty string"));
        }
    }
    match record.get("names").and_then(Value::as_array) {
        Some(names) if !names.is_empty() => {
            for (index, name) in names.iter().enumerate() {
                if non_empty(name, "text").is_none() || non_empty(name, "language_tag").is_none() {
                    errors.push(format!("{label}.names[{index}] requires text and language_tag"));
                } else if !language_tag_is_well_formed(non_empty(name, "language_tag").unwrap()) {
                    errors.push(format!(
                        "{label}.names[{index}].language_tag is not a valid-looking BCP 47 tag"
                    ));
                }
            }
        }
        _ => errors.push(format!("{label}.names must be a non-empty array")),
    }
    let null = Value::Null;
    let identity = record.get("identity").unwrap_or(&null);
    if !identity.is_object() {
        errors.push(format!("{label}.identity must be an object"));
        return;
    }
    for attribute in [
        "taxon_id", "food_part", "processing_state", "preparation_state",
        "product_kind", "edible_basis", "preservation", "brand_id",
        "fortification", "cultivar", "origin_region"
    ] {
        if value_string_or_null(identity.get(attribute)).is_none() {
            errors.push(format!("{label}.identity.{attribute} must be a string or null"));
        }
    }
}

fn value_string_or_null(value: Option<&Value>) -> Option<Option<&str>> {
    match value {
        Some(Value::String(text)) if !text.trim().is_empty() => Some(Some(text.as_str())),
        Some(Value::Null) => Some(None),
        _ => None,
    }
}

fn relation_is_consistent(check: &Value) -> bool {
    let Some(query_value) = value_string_or_null(check.get("query_value")) else {
        return false;
    };
    let Some(candidate_value) = value_string_or_null(check.get("candidate_value")) else {
        return false;
    };
    let relation = non_empty(check, "relation").unwrap_or("");
    match relation {
        "match" => {
            query_value.is_some() && candidate_value.is_some() && query_value == candidate_value
        },
        "different" => {
            query_value.is_some() && candidate_value.is_some() && query_value != candidate_value
        },
        "unknown" => query_value.is_none() || candidate_value.is_none(),
        "not_applicable" => query_value.is_none() && candidate_value.is_none(),
        _ => false,
    }
}

fn match_scope_allows_approximation(scope: &str) -> bool {
    matches!(scope, "recipe_substitution" | "nutrition_estimation")
}

/// Validate the semantic invariants of a FoodIdentityMatchV1 assessment.
///
/// This is deliberately not a full JSON Schema implementation. Consumers must
/// also validate the wire object against FoodIdentityMatchV1.schema.json and
/// qualify the underlying records, rights, taxonomy mappings and sources.
pub fn validate_food_identity_match_v1(assessment: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if assessment.get("schema_version").and_then(Value::as_str) != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty(assessment, "assessment_id").is_none() {
        errors.push("assessment_id must be a non-empty string".to_owned());
    }
    let scope = non_empty(assessment, "use_scope");
    if !matches!(
        scope,
        Some(
            "nutrition_estimation"
                | "recipe_substitution"
                | "product_lookup"
                | "allergen_screening"
                | "price_comparison"
        )
    ) {
        errors.push("use_scope must be an explicitly supported comparison context".to_owned());
    }
    let outcome = non_empty(assessment, "outcome");
    if !matches!(
        outcome,
        Some("exact_match" | "qualified_approximation" | "review_required" | "no_match")
    ) {
        errors.push("outcome is invalid".to_owned());
    }

    let null = Value::Null;
    let query = assessment.get("query_record").unwrap_or(&null);
    let candidate = assessment.get("candidate_record").unwrap_or(&null);
    identity_record_errors(query, "query_record", &mut errors);
    identity_record_errors(candidate, "candidate_record", &mut errors);

    let mut check_ids = BTreeSet::new();
    let mut relation_by_attribute = std::collections::BTreeMap::new();
    let checks: &[Value] = match assessment.get("attribute_checks").and_then(Value::as_array) {
        Some(checks) => checks.as_slice(),
        None => {
            errors.push("attribute_checks must be an array".to_owned());
            &[]
        }
    };
    for (index, check) in checks.iter().enumerate() {
        let Some(attribute_id) = non_empty(check, "attribute_id") else {
            errors.push(format!("attribute_checks[{index}].attribute_id is required"));
            continue;
        };
        if !check_ids.insert(attribute_id.to_owned()) {
            errors.push(format!("duplicate attribute check {attribute_id:?}"));
        }
        if !relation_is_consistent(check) {
            errors.push(format!(
                "attribute check {attribute_id:?} has an inconsistent relation/value pair"
            ));
        }
        if ![
            "taxon_id", "food_part", "processing_state", "preparation_state",
            "product_kind", "edible_basis", "preservation", "brand_id",
            "fortification", "cultivar", "origin_region"
        ].contains(&attribute_id) {
            errors.push(format!("unsupported attribute_id {attribute_id:?}"));
        }
        let query_identity_value =
            value_string_or_null(query.get("identity").and_then(|v| v.get(attribute_id)));
        let candidate_identity_value =
            value_string_or_null(candidate.get("identity").and_then(|v| v.get(attribute_id)));
        let query_check_value = value_string_or_null(check.get("query_value"));
        let candidate_check_value = value_string_or_null(check.get("candidate_value"));
        if query_identity_value != query_check_value
            || candidate_identity_value != candidate_check_value
        {
            errors.push(format!(
                "attribute check {attribute_id:?} does not match the referenced record identity fields"
            ));
        }
        if let Some(relation) = non_empty(check, "relation") {
            relation_by_attribute.insert(attribute_id.to_owned(), relation.to_owned());
        }
    }

    for required in REQUIRED_IDENTITY_ATTRIBUTES {
        if !relation_by_attribute.contains_key(*required) {
            errors.push(format!("required identity attribute {required:?} is not assessed"));
        }
    }
    for required in ASSESSED_HARD_BLOCK_ATTRIBUTES {
        if !relation_by_attribute.contains_key(*required) {
            errors.push(format!(
                "hard-block identity attribute {required:?} must be explicitly assessed"
            ));
        }
    }

    let rationale = non_empty(assessment, "rationale");
    let deviations = assessment.get("deviations").and_then(Value::as_array);
    let mut deviation_ids = BTreeSet::new();
    if let Some(deviations) = deviations {
        for deviation in deviations {
            if let Some(id) = non_empty(deviation, "attribute_id") {
                deviation_ids.insert(id.to_owned());
                if non_empty(deviation, "reason").is_none() {
                    errors.push(format!("deviation {id:?} requires a reason"));
                }
            } else {
                errors.push("every deviation requires attribute_id".to_owned());
            }
        }
    } else {
        errors.push("deviations must be an array".to_owned());
    }

    let hard_differences: BTreeSet<String> = relation_by_attribute
        .iter()
        .filter(|(attribute, relation)| {
            HARD_BLOCK_ATTRIBUTES.contains(&attribute.as_str()) && relation.as_str() == "different"
        })
        .map(|(attribute, _)| attribute.clone())
        .collect();
    let hard_unknowns: BTreeSet<String> = relation_by_attribute
        .iter()
        .filter(|(attribute, relation)| {
            REQUIRED_IDENTITY_ATTRIBUTES.contains(&attribute.as_str())
                && relation.as_str() != "match"
        })
        .map(|(attribute, _)| attribute.clone())
        .collect();
    let hard_block_unknowns: BTreeSet<String> = relation_by_attribute
        .iter()
        .filter(|(attribute, relation)| {
            HARD_BLOCK_ATTRIBUTES.contains(&attribute.as_str()) && relation.as_str() == "unknown"
        })
        .map(|(attribute, _)| attribute.clone())
        .collect();
    let all_differences_or_unknowns: BTreeSet<String> = relation_by_attribute
        .iter()
        .filter(|(_, relation)| matches!(relation.as_str(), "different" | "unknown"))
        .map(|(attribute, _)| attribute.clone())
        .collect();
    let query_identity = query.get("identity");
    let candidate_identity = candidate.get("identity");
    let branded_without_identity = [
        query_identity,
        candidate_identity,
    ].iter().any(|identity| {
        identity
            .and_then(|value| value.get("product_kind"))
            .and_then(Value::as_str)
            == Some("branded_product")
            && identity.and_then(|value| value.get("brand_id")).and_then(Value::as_str).is_none()
    });
    if matches!(outcome, Some("exact_match" | "qualified_approximation"))
        && branded_without_identity
    {
        errors.push(
            "exact/approximate identity of a branded product requires a known brand_id".to_owned()
        );
    }

    if outcome == Some("exact_match") {
        for required in REQUIRED_IDENTITY_ATTRIBUTES {
            if relation_by_attribute.get(*required).map(String::as_str) != Some("match") {
                errors.push(format!("exact_match requires {required:?} to be known and matching"));
            }
        }
        if !hard_differences.is_empty() {
            errors.push(format!(
                "exact_match cannot claim identity across hard-block differences: {hard_differences:?}"
            ));
        }
        if !hard_unknowns.is_empty() {
            errors.push(format!(
                "exact_match cannot leave required identity attributes unknown/different: {hard_unknowns:?}"
            ));
        }
        for (attribute, relation) in &relation_by_attribute {
            if relation != "match" && relation != "not_applicable" {
                errors.push(format!(
                    "exact_match cannot accept {attribute:?} relation {relation:?}"
                ));
            }
        }
        if !deviation_ids.is_empty() {
            errors.push("exact_match cannot declare identity deviations".to_owned());
        }
    }

    if outcome == Some("qualified_approximation") {
        if !scope.is_some_and(match_scope_allows_approximation) {
            errors.push(
                "qualified_approximation is not permitted for product_lookup, price_comparison, or allergen_screening"
                    .to_owned(),
            );
        }
        if rationale.is_none() {
            errors.push("qualified_approximation requires a non-empty rationale".to_owned());
        }
        if deviation_ids.is_empty() {
            errors.push("qualified_approximation requires explicit deviations".to_owned());
        }
        for attribute in &all_differences_or_unknowns {
            if !deviation_ids.contains(attribute) {
                errors.push(format!(
                    "difference/unknown for {attribute:?} must be explicitly declared as a deviation"
                ));
            }
        }
        if !hard_unknowns.is_empty() {
            errors.push(format!(
                "qualified_approximation cannot leave required identity attributes unknown/different: {hard_unknowns:?}"
            ));
        }
        if !hard_block_unknowns.is_empty() {
            errors.push(format!(
                "qualified_approximation cannot leave hard-block identity attributes unknown: {hard_block_unknowns:?}"
            ));
        }
        if scope == Some("nutrition_estimation") && !hard_differences.is_empty() {
            errors.push(format!(
                "nutrition_estimation cannot silently use hard-block identity differences as a nutrient proxy: {hard_differences:?}"
            ));
        }
    }

    if scope == Some("allergen_screening") && outcome == Some("qualified_approximation") {
        errors.push("approximate food identity cannot support allergen screening".to_owned());
    }

    if outcome == Some("no_match")
        && hard_differences.is_empty()
        && hard_unknowns.is_empty()
        && rationale.is_none()
    {
        errors.push(
            "no_match requires a known conflict/unknown or an explicit rationale".to_owned()
        );
    }

    let claims = match assessment.get("component_claims").and_then(Value::as_array) {
        Some(claims) => claims,
        None => {
            errors.push("component_claims must be an array (empty is allowed)".to_owned());
            &[]
        }
    };
    for (index, claim) in claims.iter().enumerate() {
        let component = non_empty(claim, "component_id").unwrap_or("<missing>");
        let query_basis = non_empty(claim, "query_basis");
        let candidate_basis = non_empty(claim, "candidate_basis");
        let comparability = non_empty(claim, "comparability").unwrap_or("");
        match comparability {
            "same_basis" => {
                if query_basis.is_none() || query_basis != candidate_basis {
                    errors.push(format!(
                        "component_claims[{index}] {component:?} claims same_basis but bases differ or are absent"
                    ));
                }
            }
            "converted" => {
                if non_empty(claim, "conversion_ref").is_none() {
                    errors.push(format!(
                        "component_claims[{index}] {component:?} requires conversion_ref"
                    ));
                }
            }
            "not_comparable" => {}
            _ => errors.push(format!(
                "component_claims[{index}] {component:?} has invalid comparability"
            )),
        }
        if outcome == Some("exact_match") && comparability == "not_comparable" {
            errors.push(format!(
                "exact_match cannot claim component {component:?} is comparable when it is not"
            ));
        }
        if outcome == Some("qualified_approximation")
            && comparability == "not_comparable"
            && scope == Some("nutrition_estimation")
        {
            errors.push(format!(
                "nutrition_estimation approximation cannot claim uncomparable component {component:?} as an input"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assessment() -> Value {
        serde_json::from_str(include_str!(
            "../fixtures/food-identity-match-v1/exact-multilingual-potato.json"
        ))
        .expect("synthetic food identity fixture must parse")
    }

    #[test]
    fn exact_food_match_requires_critical_attributes_and_consistent_relations() {
        assert!(validate_food_identity_match_v1(&assessment()).is_ok());
    }

    #[test]
    fn raw_and_cooked_foods_cannot_be_declared_exact() {
        let mut value = assessment();
        value["outcome"] = json!("exact_match");
        value["attribute_checks"][3]["candidate_value"] = json!("boiled");
        value["attribute_checks"][3]["relation"] = json!("different");
        let errors = validate_food_identity_match_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("exact_match requires \"preparation_state\"")));
    }

    #[test]
    fn a_relation_must_agree_with_the_declared_attribute_values() {
        let mut value = assessment();
        value["attribute_checks"][0]["candidate_value"] = json!("taxon:solanum-lycopersicum");
        let errors = validate_food_identity_match_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("inconsistent relation/value pair")));
    }

    #[test]
    fn nutrient_bases_cannot_be_compared_without_conversion_evidence() {
        let mut value = assessment();
        value["component_claims"][0]["candidate_basis"] = json!("g/100g dry matter");
        let errors = validate_food_identity_match_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("claims same_basis but bases differ")));
    }

    #[test]
    fn approximate_matches_need_declared_context_and_deviations() {
        let mut value = assessment();
        value["outcome"] = json!("qualified_approximation");
        value["use_scope"] = json!("product_lookup");
        value["deviations"] = json!([]);
        let errors = validate_food_identity_match_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("not permitted for product_lookup")));
        assert!(errors.iter().any(|e| e.contains("requires explicit deviations")));
    }

    #[test]
    fn allergen_screening_never_accepts_approximate_identity() {
        let mut value = assessment();
        value["outcome"] = json!("qualified_approximation");
        value["use_scope"] = json!("allergen_screening");
        value["deviations"] = json!([{"attribute_id":"cultivar","reason":"synthetic example"}]);
        let errors = validate_food_identity_match_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("not permitted")));
    }

    #[test]
    fn unknown_required_identity_attribute_forces_non_exact_outcome() {
        let mut value = assessment();
        value["outcome"] = json!("review_required");
        value["attribute_checks"][0]["candidate_value"] = Value::Null;
        value["candidate_record"]["identity"]["taxon_id"] = Value::Null;
        value["attribute_checks"][0]["relation"] = json!("unknown");
        assert!(validate_food_identity_match_v1(&value).is_ok());
    }

    #[test]
    fn branded_food_identity_requires_explicit_brand_and_fortification_checks_when_present() {
        let mut value = assessment();
        value["attribute_checks"][0]["relation"] = json!("match");
        value["attribute_checks"].as_array_mut().unwrap().push(json!({
            "attribute_id":"brand_id","query_value":"brand:alpha","candidate_value":"brand:beta","relation":"different"
        }));
        let errors = validate_food_identity_match_v1(&value).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("duplicate attribute check \"brand_id\"")));
        assert!(errors.iter().any(|e| e.contains("hard-block differences")));
    }
}
