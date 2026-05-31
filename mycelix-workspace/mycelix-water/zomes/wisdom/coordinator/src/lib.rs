// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Wisdom Coordinator Zome
//! Business logic for traditional water knowledge, conservation, and climate patterns

use hdk::prelude::*;
use wisdom_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// TRADITIONAL PRACTICES
// ============================================================================

/// Record a traditional water management practice
#[hdk_extern]
pub fn record_practice(practice: TraditionalPractice) -> ExternResult<Record> {
    if practice.title.is_empty() || practice.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if practice.description.is_empty() || practice.description.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-8192 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::TraditionalPractice(practice.clone()))?;

    // Link to all practices
    create_entry(&EntryTypes::Anchor(Anchor("all_practices".to_string())))?;
    create_link(
        anchor_hash("all_practices")?,
        action_hash.clone(),
        LinkTypes::AllPractices,
        (),
    )?;

    // Link practice type to practice
    let type_anchor = format!("practice_type:{:?}", practice.practice_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::PracticeTypeToEntry,
        (),
    )?;

    // Link recorder to practice
    create_link(
        practice.recorded_by.clone(),
        action_hash.clone(),
        LinkTypes::RecorderToPractice,
        (),
    )?;

    // If public, also link to public practices anchor
    if practice.access_level == AccessLevel::Public {
        create_entry(&EntryTypes::Anchor(Anchor("public_practices".to_string())))?;
        create_link(
            anchor_hash("public_practices")?,
            action_hash.clone(),
            LinkTypes::PublicPractices,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created practice".into()
    )))
}

/// Get practices by type
#[hdk_extern]
pub fn get_practices_by_type(practice_type: PracticeType) -> ExternResult<Vec<Record>> {
    let type_anchor = format!("practice_type:{:?}", practice_type);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::PracticeTypeToEntry)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all public practices
#[hdk_extern]
pub fn get_public_practices(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("public_practices")?, LinkTypes::PublicPractices)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all practices (regardless of access level -- caller is responsible for filtering)
#[hdk_extern]
pub fn get_all_practices(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_practices")?, LinkTypes::AllPractices)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CONSERVATION METHODS
// ============================================================================

/// Share a conservation method
#[hdk_extern]
pub fn share_conservation_method(method: ConservationMethod) -> ExternResult<Record> {
    if method.title.is_empty() || method.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if method.description.is_empty() || method.description.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-8192 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::ConservationMethod(method.clone()))?;

    // Link to all conservation methods
    create_entry(&EntryTypes::Anchor(Anchor(
        "all_conservation_methods".to_string(),
    )))?;
    create_link(
        anchor_hash("all_conservation_methods")?,
        action_hash.clone(),
        LinkTypes::AllConservationMethods,
        (),
    )?;

    // Link cost level to method
    let cost_anchor = format!("cost_level:{:?}", method.cost_level);
    create_entry(&EntryTypes::Anchor(Anchor(cost_anchor.clone())))?;
    create_link(
        anchor_hash(&cost_anchor)?,
        action_hash.clone(),
        LinkTypes::CostLevelToMethod,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created conservation method".into()
    )))
}

/// Get all conservation methods
#[hdk_extern]
pub fn get_conservation_methods(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("all_conservation_methods")?,
            LinkTypes::AllConservationMethods,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// CLIMATE WATER PATTERNS
// ============================================================================

/// Record an observed climate-water pattern
#[hdk_extern]
pub fn record_climate_pattern(pattern: ClimateWaterPattern) -> ExternResult<Record> {
    if pattern.region.is_empty() || pattern.region.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Region must be 1-256 characters".into()
        )));
    }
    if pattern.description.is_empty() || pattern.description.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-4096 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::ClimateWaterPattern(pattern.clone()))?;

    // Link to all climate patterns
    create_entry(&EntryTypes::Anchor(Anchor(
        "all_climate_patterns".to_string(),
    )))?;
    create_link(
        anchor_hash("all_climate_patterns")?,
        action_hash.clone(),
        LinkTypes::AllClimatePatterns,
        (),
    )?;

    // Link region to pattern
    let region_anchor = format!("region:{}", pattern.region);
    create_entry(&EntryTypes::Anchor(Anchor(region_anchor.clone())))?;
    create_link(
        anchor_hash(&region_anchor)?,
        action_hash.clone(),
        LinkTypes::RegionToPattern,
        (),
    )?;

    // Link pattern type to pattern
    let pattern_type_anchor = format!("pattern_type:{:?}", pattern.pattern_type);
    create_entry(&EntryTypes::Anchor(Anchor(pattern_type_anchor.clone())))?;
    create_link(
        anchor_hash(&pattern_type_anchor)?,
        action_hash.clone(),
        LinkTypes::PatternTypeToPattern,
        (),
    )?;

    // Link observer to pattern
    create_link(
        pattern.observed_by.clone(),
        action_hash.clone(),
        LinkTypes::ObserverToPattern,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created climate pattern".into()
    )))
}

/// Get climate patterns for a specific region
#[hdk_extern]
pub fn get_regional_patterns(region: String) -> ExternResult<Vec<Record>> {
    let region_anchor = format!("region:{}", region);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&region_anchor)?, LinkTypes::RegionToPattern)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(records)
}

/// Get all climate patterns
#[hdk_extern]
pub fn get_all_climate_patterns(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("all_climate_patterns")?,
            LinkTypes::AllClimatePatterns,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
