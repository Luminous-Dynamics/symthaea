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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // SeedVariety serde roundtrip tests
    // ========================================================================

    #[test]
    fn seed_variety_serde_roundtrip() {
        let seed = SeedVariety {
            name: "Cherokee Purple".to_string(),
            species: "Solanum lycopersicum".to_string(),
            origin: Some("Tennessee".to_string()),
            days_to_maturity: 80,
            companion_plants: vec!["Basil".to_string(), "Marigold".to_string()],
            avoid_plants: vec!["Fennel".to_string()],
            seed_saving_notes: Some("Open-pollinated, save from ripe fruit".to_string()),
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Cherokee Purple");
        assert_eq!(decoded.species, "Solanum lycopersicum");
        assert_eq!(decoded.origin, Some("Tennessee".to_string()));
        assert_eq!(decoded.days_to_maturity, 80);
        assert_eq!(decoded.companion_plants.len(), 2);
        assert_eq!(decoded.avoid_plants, vec!["Fennel".to_string()]);
    }

    #[test]
    fn seed_variety_none_fields_roundtrip() {
        let seed = SeedVariety {
            name: "Roma".to_string(),
            species: "Solanum lycopersicum".to_string(),
            origin: None,
            days_to_maturity: 75,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.origin, None);
        assert_eq!(decoded.seed_saving_notes, None);
        assert!(decoded.companion_plants.is_empty());
    }

    // ========================================================================
    // PracticeCategory enum serde tests
    // ========================================================================

    #[test]
    fn practice_category_all_variants_serde() {
        let variants = vec![
            PracticeCategory::Planting,
            PracticeCategory::Harvest,
            PracticeCategory::Soil,
            PracticeCategory::Pest,
            PracticeCategory::Water,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: PracticeCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    // ========================================================================
    // TraditionalPractice serde roundtrip tests
    // ========================================================================

    #[test]
    fn traditional_practice_serde_roundtrip() {
        let practice = TraditionalPractice {
            name: "Three Sisters Planting".to_string(),
            description: "Corn, beans, and squash planted together".to_string(),
            region: Some("Haudenosaunee territory".to_string()),
            season: Some("Spring".to_string()),
            category: PracticeCategory::Planting,
        };
        let json = serde_json::to_string(&practice).unwrap();
        let decoded: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Three Sisters Planting");
        assert_eq!(decoded.category, PracticeCategory::Planting);
        assert_eq!(decoded.region, Some("Haudenosaunee territory".to_string()));
    }

    // ========================================================================
    // Recipe serde roundtrip tests
    // ========================================================================

    #[test]
    fn recipe_serde_roundtrip() {
        let recipe = Recipe {
            name: "Tomato Sauce".to_string(),
            ingredients: vec!["Tomatoes".to_string(), "Garlic".to_string()],
            instructions: "Blend and simmer".to_string(),
            servings: 4,
            prep_time_min: 45,
            tags: vec!["sauce".to_string(), "preserving".to_string()],
            source_attribution: Some("Community cookbook".to_string()),
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Tomato Sauce");
        assert_eq!(decoded.servings, 4);
        assert_eq!(decoded.prep_time_min, 45);
        assert_eq!(decoded.tags, vec!["sauce", "preserving"]);
        assert_eq!(decoded.source_attribution, Some("Community cookbook".to_string()));
    }

    #[test]
    fn recipe_no_attribution_roundtrip() {
        let recipe = Recipe {
            name: "Simple Bread".to_string(),
            ingredients: vec!["Flour".to_string(), "Water".to_string(), "Yeast".to_string()],
            instructions: "Mix, knead, rise, bake".to_string(),
            servings: 1,
            prep_time_min: 120,
            tags: vec![],
            source_attribution: None,
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.source_attribution, None);
        assert!(decoded.tags.is_empty());
    }
}
