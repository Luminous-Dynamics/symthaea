//! Versioned regional capability packs for globally portable culinary decisions.
//!
//! A pack describes the scope and evidence capabilities available to a deployment.
//! It does not itself establish the truth of referenced sources, define a clinical
//! diet, or infer a user's food culture from their country or language.

use serde_json::Value;
use std::collections::BTreeSet;

const REQUIRED_VERTICALS: &[&str] = &[
    "ingredient_composition",
    "dietary_guidance",
    "price_affordability",
    "product_identity",
    "food_safety_requirements",
    "cultural_acceptability",
    "seasonality",
    "availability_logistics",
    "cooking_measurements",
    "energy_and_waste",
];

fn non_empty_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
}

fn is_upper_ascii_code(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_uppercase())
}

fn is_bcp47_like_tag(value: &str) -> bool {
    let mut parts = value.split('-');
    let Some(language) = parts.next() else {
        return false;
    };
    if !(2..=8).contains(&language.len()) || !language.bytes().all(|b| b.is_ascii_alphabetic()) {
        return false;
    }
    parts.all(|part| {
        !part.is_empty()
            && part.len() <= 8
            && part.bytes().all(|b| b.is_ascii_alphanumeric())
    })
}

/// Validate the semantic invariants of a regional pack.
///
/// This is deliberately **not** a full JSON Schema implementation. Consumers
/// should validate the wire format against `CulinaryRegionalPackV1.schema.json`
/// in addition to calling this function, and must independently qualify sources.
pub fn validate_regional_pack_v1(pack: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let null = Value::Null;

    if pack.get("schema_version").and_then(Value::as_str) != Some("1.0.0") {
        errors.push("schema_version must be 1.0.0".to_owned());
    }
    if non_empty_string(pack, "pack_id").is_none() {
        errors.push("pack_id must be a non-empty string".to_owned());
    }
    if !matches!(
        pack.get("profile_status").and_then(Value::as_str),
        Some("synthetic_example" | "draft" | "qualified")
    ) {
        errors.push("profile_status must be synthetic_example, draft, or qualified".to_owned());
    }

    let region = pack.get("region").unwrap_or(&null);
    let country_code = non_empty_string(region, "country_code");
    if !country_code.is_some_and(|code| is_upper_ascii_code(code, 2)) {
        errors.push("region.country_code must be a two-letter uppercase code".to_owned());
    }
    if non_empty_string(region, "market_scope").is_none() {
        errors.push("region.market_scope must identify the actual market/geographic scope".to_owned());
    }
    if !non_empty_string(region, "locale_tag").is_some_and(is_bcp47_like_tag) {
        errors.push("region.locale_tag must be a valid-looking BCP 47 language tag".to_owned());
    }
    let mut language_tags = BTreeSet::new();
    match region.get("language_tags").and_then(Value::as_array) {
        Some(tags) if !tags.is_empty() => {
            for tag in tags {
                let Some(tag) = tag.as_str() else {
                    errors.push("region.language_tags must contain strings".to_owned());
                    continue;
                };
                if !is_bcp47_like_tag(tag) {
                    errors.push(format!("region.language_tags contains invalid-looking tag {tag:?}"));
                }
                if !language_tags.insert(tag.to_owned()) {
                    errors.push(format!("region.language_tags contains duplicate tag {tag:?}"));
                }
            }
        }
        _ => errors.push("region.language_tags must be a non-empty array".to_owned()),
    }

    if let Some(locale) = non_empty_string(region, "locale_tag") {
        if !language_tags.contains(locale) {
            errors.push("region.locale_tag must also appear in region.language_tags".to_owned());
        }
    }

    let conventions = pack.get("conventions").unwrap_or(&null);
    if !non_empty_string(conventions, "currency").is_some_and(|s| is_upper_ascii_code(s, 3)) {
        errors.push("conventions.currency must be a three-letter uppercase code".to_owned());
    }
    if !matches!(
        non_empty_string(conventions, "mass_unit"),
        Some("g" | "kg" | "oz" | "lb")
    ) {
        errors.push("conventions.mass_unit must be g, kg, oz, or lb".to_owned());
    }
    if !matches!(
        non_empty_string(conventions, "volume_unit"),
        Some("mL" | "L" | "US_fl_oz" | "US_cup" | "metric_cup" | "imperial_cup")
    ) {
        errors.push("conventions.volume_unit must be an explicitly supported volume unit".to_owned());
    }
    if !matches!(
        non_empty_string(conventions, "temperature_unit"),
        Some("C" | "F")
    ) {
        errors.push("conventions.temperature_unit must be C or F".to_owned());
    }
    if conventions.get("conversion_evidence_required").and_then(Value::as_bool) != Some(true) {
        errors.push("unit conversions must require explicit conversion evidence".to_owned());
    }
    if conventions.get("density_required_for_mass_volume_conversion").and_then(Value::as_bool)
        != Some(true)
    {
        errors.push("mass-volume conversion must require an explicit density or conversion basis".to_owned());
    }
    if conventions.get("source_native_nutrient_basis_required").and_then(Value::as_bool) != Some(true) {
        errors.push("nutrient values must preserve their source-native basis".to_owned());
    }

    let mut source_ids = BTreeSet::new();
    let sources: &[Value] = match pack.get("sources").and_then(Value::as_array) {
        Some(sources) => sources.as_slice(),
        None => {
            errors.push("sources must be an array".to_owned());
            &[]
        }
    };
    for source in sources {
        let source_id = non_empty_string(source, "source_id");
        if source_id.is_none() {
            errors.push("each source needs a non-empty source_id".to_owned());
            continue;
        }
        let source_id = source_id.unwrap();
        if !source_ids.insert(source_id.to_owned()) {
            errors.push(format!("duplicate source_id {source_id:?}"));
        }
        for field in ["publisher", "reference", "source_kind", "license_reference", "known_at", "valid_from"] {
            if non_empty_string(source, field).is_none() {
                errors.push(format!("source {source_id:?} needs non-empty {field}"));
            }
        }
        if !matches!(
            non_empty_string(source, "access_class"),
            Some("open_reuse" | "attribution_required" | "permission_required" | "restricted" | "unknown")
        ) {
            errors.push(format!("source {source_id:?} has an unknown access_class"));
        }
        if !matches!(
            non_empty_string(source, "status"),
            Some("active" | "expired" | "withdrawn" | "unknown")
        ) {
            errors.push(format!("source {source_id:?} has an unknown status"));
        }
        if source.get("redistribution_permitted").and_then(Value::as_bool) == Some(true)
            && !matches!(
                non_empty_string(source, "access_class"),
                Some("open_reuse" | "attribution_required")
            )
        {
            errors.push(format!(
                "source {source_id:?} claims redistribution without a compatible access_class"
            ));
        }
    }

    let verticals: &[Value] = match pack.get("verticals").and_then(Value::as_array) {
        Some(verticals) => verticals.as_slice(),
        None => {
            errors.push("verticals must be an array".to_owned());
            &[]
        }
    };
    let mut vertical_ids = BTreeSet::new();
    for vertical in verticals {
        let vertical_id = non_empty_string(vertical, "vertical_id");
        if vertical_id.is_none() {
            errors.push("each vertical needs a non-empty vertical_id".to_owned());
            continue;
        }
        let vertical_id = vertical_id.unwrap();
        if !vertical_ids.insert(vertical_id.to_owned()) {
            errors.push(format!("duplicate vertical_id {vertical_id:?}"));
        }
        let status = non_empty_string(vertical, "status");
        if !matches!(
            status,
            Some("supported" | "partial" | "not_configured" | "license_blocked")
        ) {
            errors.push(format!("vertical {vertical_id:?} has an invalid status"));
        }
        if non_empty_string(vertical, "coverage_notes").is_none() {
            errors.push(format!("vertical {vertical_id:?} must describe its coverage/limitations"));
        }
        let refs = match vertical.get("source_refs").and_then(Value::as_array) {
            Some(refs) => refs,
            None => {
                errors.push(format!("vertical {vertical_id:?} source_refs must be an array"));
                continue;
            }
        };
        if matches!(status, Some("supported" | "partial")) && refs.is_empty() {
            errors.push(format!("vertical {vertical_id:?} cannot be supported/partial with no sources"));
        }
        for source_ref in refs {
            match source_ref.as_str() {
                Some(id) if source_ids.contains(id) => {}
                Some(id) => errors.push(format!(
                    "vertical {vertical_id:?} references undeclared source_id {id:?}"
                )),
                None => errors.push(format!(
                    "vertical {vertical_id:?} source_refs must contain strings"
                )),
            }
        }
    }
    if pack.get("profile_status").and_then(Value::as_str) == Some("qualified")
        && verticals.iter().any(|vertical| {
            matches!(
                non_empty_string(vertical, "status"),
                Some("not_configured" | "license_blocked")
            )
        })
    {
        errors.push("a qualified regional pack cannot include not_configured or license_blocked verticals".to_owned());
    }

    for required in REQUIRED_VERTICALS {
        if !vertical_ids.contains(*required) {
            errors.push(format!("regional pack must explicitly declare vertical {required:?}"));
        }
    }

    let fallback = pack.get("fallback_policy").unwrap_or(&null);
    if non_empty_string(fallback, "missing_values") != Some("unknown_not_zero") {
        errors.push("fallback_policy.missing_values must preserve missing as unknown_not_zero".to_owned());
    }
    if non_empty_string(fallback, "cross_region_substitution")
        != Some("explicitly_label_source_region_and_uncertainty")
    {
        errors.push("cross-region substitutions must be explicitly labelled with source region and uncertainty".to_owned());
    }
    if fallback.get("infer_food_culture_from_country_code").and_then(Value::as_bool) != Some(false) {
        errors.push("country code must not be used to infer food culture".to_owned());
    }
    if fallback.get("infer_probabilities_from_scenario_counts").and_then(Value::as_bool) != Some(false) {
        errors.push("scenario counts must not be treated as probabilities".to_owned());
    }
    if fallback.get("allow_restricted_data_in_public_pack").and_then(Value::as_bool) != Some(false) {
        errors.push("restricted data must be disabled in public packs".to_owned());
    }
    if fallback.get("require_known_at_for_replay").and_then(Value::as_bool) != Some(true) {
        errors.push("historical replay must require known_at".to_owned());
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

    fn valid_pack() -> Value {
        json!({
            "schema_version": "1.0.0",
            "pack_id": "fixture:global-region-v1",
            "profile_status": "synthetic_example",
            "region": {
                "country_code": "ZA",
                "market_scope": "synthetic test market; not a real market profile",
                "locale_tag": "en-ZA",
                "language_tags": ["en-ZA", "zu-ZA"]
            },
            "conventions": {
                "currency": "ZAR",
                "mass_unit": "g",
                "volume_unit": "mL",
                "temperature_unit": "C",
                "conversion_evidence_required": true,
                "density_required_for_mass_volume_conversion": true,
                "source_native_nutrient_basis_required": true
            },
            "sources": [{
                "source_id": "synthetic-source",
                "publisher": "Synthetic example only",
                "reference": "synthetic:global-regionalization-fixture",
                "source_kind": "illustrative_only",
                "access_class": "unknown",
                "license_reference": "synthetic:no-data-license",
                "redistribution_permitted": false,
                "status": "unknown",
                "known_at": "2026-01-01T00:00:00Z",
                "valid_from": "2026-01-01T00:00:00Z"
            }],
            "verticals": [
                {"vertical_id":"ingredient_composition","status":"not_configured","source_refs":[],"coverage_notes":"No composition source configured."},
                {"vertical_id":"dietary_guidance","status":"not_configured","source_refs":[],"coverage_notes":"No dietary guidance source configured."},
                {"vertical_id":"price_affordability","status":"not_configured","source_refs":[],"coverage_notes":"No current price source configured."},
                {"vertical_id":"product_identity","status":"not_configured","source_refs":[],"coverage_notes":"No product identity source configured."},
                {"vertical_id":"food_safety_requirements","status":"not_configured","source_refs":[],"coverage_notes":"No source configured; do not certify safety."},
                {"vertical_id":"cultural_acceptability","status":"not_configured","source_refs":[],"coverage_notes":"Preferences require user/community input."},
                {"vertical_id":"seasonality","status":"not_configured","source_refs":[],"coverage_notes":"No seasonal source configured."},
                {"vertical_id":"availability_logistics","status":"not_configured","source_refs":[],"coverage_notes":"No live availability source configured."},
                {"vertical_id":"cooking_measurements","status":"not_configured","source_refs":[],"coverage_notes":"No conversion table configured."},
                {"vertical_id":"energy_and_waste","status":"not_configured","source_refs":[],"coverage_notes":"No environmental source configured."}
            ],
            "fallback_policy": {
                "missing_values":"unknown_not_zero",
                "cross_region_substitution":"explicitly_label_source_region_and_uncertainty",
                "infer_food_culture_from_country_code":false,
                "infer_probabilities_from_scenario_counts":false,
                "allow_restricted_data_in_public_pack":false,
                "require_known_at_for_replay":true
            }
        })
    }

    #[test]
    fn synthetic_profile_with_all_verticals_explicitly_unconfigured_is_valid() {
        assert!(validate_regional_pack_v1(&valid_pack()).is_ok());
    }

    #[test]
    fn regional_pack_rejects_missing_vertical_inventory() {
        let mut pack = valid_pack();
        pack["verticals"] = json!([]);
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("ingredient_composition")));
        assert!(errors.iter().any(|error| error.contains("energy_and_waste")));
    }

    #[test]
    fn regional_pack_rejects_unsupported_claim_of_data_redistribution() {
        let mut pack = valid_pack();
        pack["sources"][0]["access_class"] = json!("permission_required");
        pack["sources"][0]["redistribution_permitted"] = json!(true);
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("redistribution")));
    }

    #[test]
    fn regional_pack_rejects_implicit_cross_region_cultural_inference() {
        let mut pack = valid_pack();
        pack["fallback_policy"]["infer_food_culture_from_country_code"] = json!(true);
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("must not be used to infer food culture")));
    }

    #[test]
    fn supported_vertical_must_reference_a_declared_source() {
        let mut pack = valid_pack();
        pack["verticals"][0]["status"] = json!("supported");
        pack["verticals"][0]["source_refs"] = json!(["missing-source"]);
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("undeclared source_id")));
    }
    #[test]
    fn qualified_pack_cannot_hide_unconfigured_verticals() {
        let mut pack = valid_pack();
        pack["profile_status"] = json!("qualified");
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("qualified regional pack")));
    }

    #[test]
    fn default_locale_must_be_in_the_declared_language_tags() {
        let mut pack = valid_pack();
        pack["region"]["language_tags"] = json!(["zu-ZA"]);
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("locale_tag must also appear")));
    }

    #[test]
    fn fallbacks_cannot_silently_guess_missing_values_or_foreign_sources() {
        let mut pack = valid_pack();
        pack["fallback_policy"]["missing_values"] = json!("use_zero");
        pack["fallback_policy"]["cross_region_substitution"] = json!("automatic");
        let errors = validate_regional_pack_v1(&pack).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("unknown_not_zero")));
        assert!(errors.iter().any(|error| error.contains("source region and uncertainty")));
    }

}
