// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Regression gates for source-grounded Melothaea research profiles.
//!
//! These tests deliberately validate documentation records only. They do not
//! deserialize them into runtime recipes or promote any profile to an
//! executable/culturally qualified style.

use crate::catalog::catalog_entry;
use serde_json::Value;
use std::collections::BTreeSet;

const AMAPIANO_JSON: &str = include_str!(
    "../../../../docs/research/melothaea/southern-african-style-profiles/amapiano-research-profile-v0.1.json"
);
const GQOM_JSON: &str = include_str!(
    "../../../../docs/research/melothaea/southern-african-style-profiles/gqom-research-profile-v0.1.json"
);
const PROFILE_SCHEMA_JSON: &str = include_str!(
    "../../../../docs/research/melothaea/southern-african-style-profiles/research-style-profile-v1.schema.json"
);

fn string_field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{path}.{key} must be a non-empty string"))
}

fn string_set(value: &Value, path: &str) -> Result<BTreeSet<String>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{path} must be an array"))?;
    let mut result = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let text = item
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{path}[{index}] must be a non-empty string"))?;
        if !result.insert(text.to_owned()) {
            return Err(format!("{path} contains duplicate value {text:?}"));
        }
    }
    Ok(result)
}

fn check_source_refs(
    value: &Value,
    path: &str,
    known_sources: &BTreeSet<String>,
) -> Result<(), String> {
    let ids = value
        .get("source_ids")
        .ok_or_else(|| format!("{path}.source_ids is required"))?;
    let ids = string_set(ids, &format!("{path}.source_ids"))?;
    if ids.is_empty() {
        return Err(format!("{path}.source_ids must not be empty"));
    }
    for id in ids {
        if !known_sources.contains(&id) {
            return Err(format!("{path} references unknown source ID {id:?}"));
        }
    }
    Ok(())
}

fn validate_research_profile(
    profile: &Value,
    expected_profile_id: &str,
    expected_catalog_id: u16,
    expected_name: &str,
) -> Result<(), String> {
    if string_field(profile, "schema_id", "$")?
        != "org.luminousdynamics.melothaea.research-style-profile"
    {
        return Err("$.schema_id is not the Melothaea research-profile schema".into());
    }
    if profile.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("$.schema_version must be 1".into());
    }
    if string_field(profile, "record_kind", "$")? != "research_profile_not_runtime_recipe" {
        return Err("$.record_kind must preserve the research-only boundary".into());
    }
    if string_field(profile, "profile_id", "$")? != expected_profile_id {
        return Err("$.profile_id does not match the expected stable identity".into());
    }
    if string_field(profile, "display_name", "$")? != expected_name {
        return Err("$.display_name does not match the canonical style name".into());
    }
    if string_field(profile, "status", "$")? != "research_only" {
        return Err("$.status must remain research_only".into());
    }
    if string_field(profile, "scope", "$")?.len() < 20 {
        return Err("$.scope must disclose the profile's bounded scope".into());
    }

    let catalog_ref = profile
        .get("current_catalog_reference")
        .ok_or_else(|| "$.current_catalog_reference is required".to_string())?;
    let actual_id = catalog_ref
        .get("catalog_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| "$.current_catalog_reference.catalog_id must be an integer".to_string())?;
    if actual_id != u64::from(expected_catalog_id) {
        return Err("catalogue ID does not match the expected existing identity".into());
    }
    let meaning = string_field(catalog_ref, "meaning", "$.current_catalog_reference")?;
    if !meaning.contains("not an executable composer route") {
        return Err("catalogue reference must disclose the non-executable route boundary".into());
    }
    let catalog_entry = catalog_entry(expected_catalog_id)
        .ok_or_else(|| format!("catalogue entry {expected_catalog_id} is missing"))?;
    if catalog_entry.name != expected_name {
        return Err(format!(
            "catalogue entry {expected_catalog_id} is {:?}, expected {:?}",
            catalog_entry.name, expected_name
        ));
    }
    if catalog_entry.composer_style.is_some() {
        return Err(format!(
            "catalogue entry {expected_catalog_id} unexpectedly has a composer route"
        ));
    }

    let sources = profile
        .get("sources")
        .and_then(Value::as_array)
        .filter(|sources| !sources.is_empty())
        .ok_or_else(|| "$.sources must be a non-empty array".to_string())?;
    let mut known_sources = BTreeSet::new();
    for (index, source) in sources.iter().enumerate() {
        let path = format!("$.sources[{index}]");
        let id = string_field(source, "id", &path)?;
        string_field(source, "title", &path)?;
        let url = string_field(source, "url", &path)?;
        if !url.starts_with("https://") {
            return Err(format!("{path}.url must use HTTPS"));
        }
        string_field(source, "source_kind", &path)?;
        for field in ["supports", "limitations"] {
            let values = source
                .get(field)
                .and_then(Value::as_array)
                .filter(|values| !values.is_empty())
                .ok_or_else(|| format!("{path}.{field} must be a non-empty array"))?;
            if values.iter().any(|item| item.as_str().is_none_or(str::is_empty)) {
                return Err(format!("{path}.{field} contains a non-string or empty item"));
            }
        }
        if !known_sources.insert(id.to_owned()) {
            return Err(format!("duplicate source ID {id:?}"));
        }
    }

    let declared_source_ids = string_set(
        profile
            .get("source_ids")
            .ok_or_else(|| "$.source_ids is required".to_string())?,
        "$.source_ids",
    )?;
    if declared_source_ids != known_sources {
        return Err("$.source_ids must exactly match the source records".into());
    }

    let historical = profile
        .get("historical_context")
        .ok_or_else(|| "$.historical_context is required".to_string())?;
    string_field(historical, "summary", "$.historical_context")?;
    let claims = historical
        .get("claims")
        .and_then(Value::as_array)
        .ok_or_else(|| "$.historical_context.claims must be an array".to_string())?;
    for (index, claim) in claims.iter().enumerate() {
        let path = format!("$.historical_context.claims[{index}]");
        string_field(claim, "id", &path)?;
        string_field(claim, "statement", &path)?;
        string_field(claim, "limit", &path)?;
        check_source_refs(claim, &path, &known_sources)?;
    }

    let axes = profile
        .get("candidate_identity_axes")
        .and_then(Value::as_array)
        .filter(|axes| !axes.is_empty())
        .ok_or_else(|| "$.candidate_identity_axes must be a non-empty array".to_string())?;
    let mut axis_ids = BTreeSet::new();
    for (index, axis) in axes.iter().enumerate() {
        let path = format!("$.candidate_identity_axes[{index}]");
        let id = string_field(axis, "id", &path)?;
        if !axis_ids.insert(id.to_owned()) {
            return Err(format!("duplicate candidate identity axis {id:?}"));
        }
        for field in [
            "axis",
            "observation",
            "operational_hypothesis",
            "status",
            "disconfirming_case",
        ] {
            string_field(axis, field, &path)?;
        }
        check_source_refs(axis, &path, &known_sources)?;
    }

    let neighbours = profile
        .get("nearest_neighbour_questions")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .ok_or_else(|| "$.nearest_neighbour_questions must be non-empty".to_string())?;
    for (index, item) in neighbours.iter().enumerate() {
        let path = format!("$.nearest_neighbour_questions[{index}]");
        string_field(item, "comparison", &path)?;
        string_field(item, "question", &path)?;
    }

    for field in ["non_inferences", "required_next_evidence"] {
        let values = profile
            .get(field)
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("$.{field} must be a non-empty array"))?;
        if values.iter().any(|item| item.as_str().is_none_or(str::is_empty)) {
            return Err(format!("$.{field} contains a non-string or empty item"));
        }
    }

    let readiness = profile
        .get("implementation_readiness")
        .ok_or_else(|| "$.implementation_readiness is required".to_string())?;
    for field in [
        "executable_composer_route",
        "runtime_style_profile",
        "qualified_renderer_profile",
        "licensed_audio_corpus",
        "practitioner_review",
        "human_listening_qualification",
    ] {
        if readiness.get(field).and_then(Value::as_bool) != Some(false) {
            return Err(format!(
                "$.implementation_readiness.{field} must remain false until separately qualified"
            ));
        }
    }
    for field in ["allowed_now", "prohibited_claims"] {
        let values = readiness
            .get(field)
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| format!("$.implementation_readiness.{field} must be non-empty"))?;
        if values.iter().any(|item| item.as_str().is_none_or(str::is_empty)) {
            return Err(format!(
                "$.implementation_readiness.{field} contains a non-string or empty item"
            ));
        }
    }

    if profile.get("composer_style").is_some() || profile.get("composer_route").is_some() {
        return Err("research profile must not declare an executable composer route".into());
    }

    Ok(())
}

fn parse_profile(raw: &str, expected_id: &str, catalog_id: u16, name: &str) -> Value {
    let value: Value = serde_json::from_str(raw).expect("research profile JSON must parse");
    validate_research_profile(&value, expected_id, catalog_id, name)
        .unwrap_or_else(|error| panic!("{expected_id}: {error}"));
    value
}

#[test]
fn amapiano_profile_is_source_complete_and_stays_research_only() {
    parse_profile(AMAPIANO_JSON, "south-african-amapiano", 109, "Amapiano");
}

#[test]
fn gqom_profile_is_source_complete_and_stays_research_only() {
    parse_profile(GQOM_JSON, "south-african-gqom", 110, "Gqom");
}

#[test]
fn schema_is_versioned_and_requires_evidence_and_readiness_boundaries() {
    let schema: Value =
        serde_json::from_str(PROFILE_SCHEMA_JSON).expect("profile schema JSON must parse");
    assert_eq!(
        schema.get("$schema").and_then(Value::as_str),
        Some("https://json-schema.org/draft/2020-12/schema")
    );
    assert_eq!(
        schema.pointer("/properties/schema_version/const").and_then(Value::as_u64),
        Some(1)
    );
    let required: BTreeSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .expect("schema required array")
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    for field in [
        "record_kind",
        "current_catalog_reference",
        "historical_context",
        "candidate_identity_axes",
        "implementation_readiness",
        "sources",
        "source_ids",
    ] {
        assert!(required.contains(field), "schema omits required field {field}");
    }
}

#[test]
fn source_reference_mutation_fails_closed() {
    let mut profile: Value = serde_json::from_str(AMAPIANO_JSON).unwrap();
    profile["candidate_identity_axes"][0]["source_ids"][0] =
        Value::String("UNREGISTERED-SOURCE".into());
    assert!(
        validate_research_profile(&profile, "south-african-amapiano", 109, "Amapiano").is_err()
    );
}

#[test]
fn executable_capability_mutation_fails_closed() {
    let mut profile: Value = serde_json::from_str(GQOM_JSON).unwrap();
    profile["implementation_readiness"]["executable_composer_route"] = Value::Bool(true);
    assert!(validate_research_profile(&profile, "south-african-gqom", 110, "Gqom").is_err());
}

#[test]
fn catalogue_identity_mutation_fails_closed() {
    let mut profile: Value = serde_json::from_str(AMAPIANO_JSON).unwrap();
    profile["current_catalog_reference"]["catalog_id"] = Value::from(110);
    assert!(
        validate_research_profile(&profile, "south-african-amapiano", 109, "Amapiano").is_err()
    );
}

#[test]
fn promotion_record_cannot_silently_become_a_runtime_recipe() {
    let mut profile: Value = serde_json::from_str(GQOM_JSON).unwrap();
    profile["record_kind"] = Value::String("runtime_style_recipe".into());
    assert!(validate_research_profile(&profile, "south-african-gqom", 110, "Gqom").is_err());
}
