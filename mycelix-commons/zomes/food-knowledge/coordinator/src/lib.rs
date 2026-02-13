//! Food Knowledge Coordinator Zome
//! Business logic for seed catalogs, traditional practices, and recipe sharing.

use food_knowledge_integrity::*;
use hdk::prelude::*;

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
// SEED CATALOG
// ============================================================================

#[hdk_extern]
pub fn catalog_seed(seed: SeedVariety) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::SeedVariety(seed.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_seeds".to_string())))?;
    create_link(anchor_hash("all_seeds")?, action_hash.clone(), LinkTypes::AllSeeds, ())?;

    let species_anchor = format!("species:{}", seed.species);
    create_entry(&EntryTypes::Anchor(Anchor(species_anchor.clone())))?;
    create_link(anchor_hash(&species_anchor)?, action_hash.clone(), LinkTypes::SpeciesToSeed, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created seed".into())))
}

#[hdk_extern]
pub fn get_seed(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn get_seeds_by_species(species: String) -> ExternResult<Vec<Record>> {
    let species_anchor = format!("species:{}", species);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&species_anchor)?, LinkTypes::SpeciesToSeed)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// TRADITIONAL PRACTICES
// ============================================================================

#[hdk_extern]
pub fn share_practice(practice: TraditionalPractice) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::TraditionalPractice(practice.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_practices".to_string())))?;
    create_link(anchor_hash("all_practices")?, action_hash.clone(), LinkTypes::AllPractices, ())?;

    let cat_anchor = format!("category:{:?}", practice.category);
    create_entry(&EntryTypes::Anchor(Anchor(cat_anchor.clone())))?;
    create_link(anchor_hash(&cat_anchor)?, action_hash.clone(), LinkTypes::CategoryToPractice, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created practice".into())))
}

#[hdk_extern]
pub fn get_practices_by_category(category: String) -> ExternResult<Vec<Record>> {
    let cat_anchor = format!("category:{}", category);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&cat_anchor)?, LinkTypes::CategoryToPractice)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// RECIPES
// ============================================================================

#[hdk_extern]
pub fn share_recipe(recipe: Recipe) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let action_hash = create_entry(&EntryTypes::Recipe(recipe.clone()))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_recipes".to_string())))?;
    create_link(anchor_hash("all_recipes")?, action_hash.clone(), LinkTypes::AllRecipes, ())?;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToRecipe, ())?;

    for tag in &recipe.tags {
        let tag_anchor = format!("tag:{}", tag);
        create_entry(&EntryTypes::Anchor(Anchor(tag_anchor.clone())))?;
        create_link(anchor_hash(&tag_anchor)?, action_hash.clone(), LinkTypes::TagToRecipe, ())?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created recipe".into())))
}

#[hdk_extern]
pub fn get_recipes_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let tag_anchor = format!("tag:{}", tag);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&tag_anchor)?, LinkTypes::TagToRecipe)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn search_knowledge(query: String) -> ExternResult<Vec<Record>> {
    // Simple search: try as species, category, or tag
    let mut results = Vec::new();

    if let Ok(seeds) = get_seeds_by_species(query.clone()) {
        results.extend(seeds);
    }
    if let Ok(practices) = get_practices_by_category(query.clone()) {
        results.extend(practices);
    }
    if let Ok(recipes) = get_recipes_by_tag(query) {
        results.extend(recipes);
    }

    Ok(results)
}
