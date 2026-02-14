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

// ============================================================================
// UPDATE FUNCTIONS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateSeedVarietyInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: SeedVariety,
}

#[hdk_extern]
pub fn update_seed_variety(input: UpdateSeedVarietyInput) -> ExternResult<ActionHash> {
    update_entry(input.original_action_hash, &EntryTypes::SeedVariety(input.updated_entry))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateTraditionalPracticeInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: TraditionalPractice,
}

#[hdk_extern]
pub fn update_traditional_practice(input: UpdateTraditionalPracticeInput) -> ExternResult<ActionHash> {
    update_entry(input.original_action_hash, &EntryTypes::TraditionalPractice(input.updated_entry))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateRecipeInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: Recipe,
}

#[hdk_extern]
pub fn update_recipe(input: UpdateRecipeInput) -> ExternResult<ActionHash> {
    update_entry(input.original_action_hash, &EntryTypes::Recipe(input.updated_entry))
}

// ============================================================================
// SEARCH
// ============================================================================

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

    // ========================================================================
    // Anchor serde roundtrip tests
    // ========================================================================

    #[test]
    fn anchor_serde_roundtrip() {
        let anchor = food_knowledge_integrity::Anchor("all_seeds".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let decoded: food_knowledge_integrity::Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, anchor);
    }

    #[test]
    fn anchor_empty_string_serde() {
        let anchor = food_knowledge_integrity::Anchor("".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let decoded: food_knowledge_integrity::Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, anchor);
    }

    #[test]
    fn anchor_unicode_serde() {
        let anchor = food_knowledge_integrity::Anchor("species:\u{7389}\u{7c73}".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let decoded: food_knowledge_integrity::Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, anchor);
    }

    // ========================================================================
    // Struct clone and equality tests
    // ========================================================================

    #[test]
    fn seed_variety_clone_is_equal() {
        let seed = SeedVariety {
            name: "Brandywine".to_string(),
            species: "Solanum lycopersicum".to_string(),
            origin: Some("Amish community, Chester County PA".to_string()),
            days_to_maturity: 90,
            companion_plants: vec!["Basil".to_string()],
            avoid_plants: vec!["Brassicas".to_string()],
            seed_saving_notes: Some("Heirloom, open-pollinated".to_string()),
        };
        let cloned = seed.clone();
        assert_eq!(seed, cloned);
    }

    #[test]
    fn traditional_practice_clone_is_equal() {
        let practice = TraditionalPractice {
            name: "Milpa".to_string(),
            description: "Mesoamerican crop-growing system".to_string(),
            region: Some("Central America".to_string()),
            season: Some("Spring".to_string()),
            category: PracticeCategory::Planting,
        };
        let cloned = practice.clone();
        assert_eq!(practice, cloned);
    }

    #[test]
    fn recipe_clone_is_equal() {
        let recipe = Recipe {
            name: "Pozole".to_string(),
            ingredients: vec!["Hominy".to_string(), "Pork".to_string(), "Chili".to_string()],
            instructions: "Simmer all ingredients for hours".to_string(),
            servings: 8,
            prep_time_min: 240,
            tags: vec!["Mexican".to_string(), "soup".to_string()],
            source_attribution: Some("Grandmother's recipe".to_string()),
        };
        let cloned = recipe.clone();
        assert_eq!(recipe, cloned);
    }

    // ========================================================================
    // Edge case: large data and boundary values
    // ========================================================================

    #[test]
    fn seed_variety_max_maturity_days() {
        let seed = SeedVariety {
            name: "Century Plant".to_string(),
            species: "Agave americana".to_string(),
            origin: None,
            days_to_maturity: u32::MAX,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.days_to_maturity, u32::MAX);
    }

    #[test]
    fn recipe_many_ingredients_serde() {
        let recipe = Recipe {
            name: "Complex Stew".to_string(),
            ingredients: (0..100).map(|i| format!("ingredient-{}", i)).collect(),
            instructions: "Combine all ingredients".to_string(),
            servings: 20,
            prep_time_min: 300,
            tags: (0..50).map(|i| format!("tag-{}", i)).collect(),
            source_attribution: Some("Community potluck".to_string()),
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ingredients.len(), 100);
        assert_eq!(decoded.tags.len(), 50);
    }

    #[test]
    fn traditional_practice_none_fields_serde() {
        let practice = TraditionalPractice {
            name: "Companion Planting".to_string(),
            description: "Growing compatible plants near each other".to_string(),
            region: None,
            season: None,
            category: PracticeCategory::Pest,
        };
        let json = serde_json::to_string(&practice).unwrap();
        let decoded: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert!(decoded.region.is_none());
        assert!(decoded.season.is_none());
    }

    #[test]
    fn seed_variety_unicode_fields_serde() {
        let seed = SeedVariety {
            name: "\u{5c3e}\u{5f35}\u{5927}\u{6839}".to_string(),
            species: "Raphanus sativus var. longipinnatus".to_string(),
            origin: Some("\u{65e5}\u{672c}".to_string()),
            days_to_maturity: 60,
            companion_plants: vec!["\u{4eba}\u{53c2}".to_string()],
            avoid_plants: vec![],
            seed_saving_notes: Some("\u{79cb}\u{306b}\u{7a2e}\u{3092}\u{53d6}\u{308b}".to_string()),
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, seed.name);
        assert_eq!(decoded.origin, seed.origin);
    }

    // ── Additional edge-case and boundary tests ─────────────────────────

    #[test]
    fn seed_variety_zero_maturity_serde_roundtrip() {
        // Zero maturity is invalid for validation but serde must still roundtrip
        let seed = SeedVariety {
            name: "Test".to_string(),
            species: "Test species".to_string(),
            origin: None,
            days_to_maturity: 0,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.days_to_maturity, 0);
    }

    #[test]
    fn seed_variety_many_companion_and_avoid_plants() {
        let seed = SeedVariety {
            name: "Social Tomato".to_string(),
            species: "Solanum lycopersicum".to_string(),
            origin: Some("Community garden".to_string()),
            days_to_maturity: 90,
            companion_plants: (0..50).map(|i| format!("companion-{}", i)).collect(),
            avoid_plants: (0..30).map(|i| format!("avoid-{}", i)).collect(),
            seed_saving_notes: Some("Lots of companions".to_string()),
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.companion_plants.len(), 50);
        assert_eq!(decoded.avoid_plants.len(), 30);
    }

    #[test]
    fn traditional_practice_all_categories_roundtrip() {
        for cat in [
            PracticeCategory::Planting,
            PracticeCategory::Harvest,
            PracticeCategory::Soil,
            PracticeCategory::Pest,
            PracticeCategory::Water,
        ] {
            let practice = TraditionalPractice {
                name: format!("Practice for {:?}", cat),
                description: "Test description.".to_string(),
                region: Some("Global".to_string()),
                season: Some("All year".to_string()),
                category: cat.clone(),
            };
            let json = serde_json::to_string(&practice).unwrap();
            let decoded: TraditionalPractice = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.category, cat);
        }
    }

    #[test]
    fn recipe_zero_servings_serde_roundtrip() {
        // Zero servings is invalid for validation but serde must still roundtrip
        let recipe = Recipe {
            name: "Nothing Soup".to_string(),
            ingredients: vec!["Air".to_string()],
            instructions: "Just wait".to_string(),
            servings: 0,
            prep_time_min: 0,
            tags: vec![],
            source_attribution: None,
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.servings, 0);
        assert_eq!(decoded.prep_time_min, 0);
    }

    #[test]
    fn recipe_max_servings_and_prep_time() {
        let recipe = Recipe {
            name: "Marathon Feast".to_string(),
            ingredients: vec!["Everything".to_string()],
            instructions: "Cook for a very long time".to_string(),
            servings: u32::MAX,
            prep_time_min: u32::MAX,
            tags: vec!["extreme".to_string()],
            source_attribution: Some("Legend".to_string()),
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.servings, u32::MAX);
        assert_eq!(decoded.prep_time_min, u32::MAX);
    }

    #[test]
    fn recipe_unicode_name_and_tags_roundtrip() {
        let recipe = Recipe {
            name: "\u{5473}\u{564c}\u{6c41}".to_string(), // Miso soup in Japanese
            ingredients: vec!["\u{8c46}\u{8150}".to_string(), "\u{308f}\u{304b}\u{3081}".to_string()],
            instructions: "\u{716e}\u{308b}".to_string(),
            servings: 2,
            prep_time_min: 15,
            tags: vec!["\u{548c}\u{98df}".to_string(), "\u{6c41}\u{7269}".to_string()],
            source_attribution: Some("\u{304a}\u{3070}\u{3042}\u{3061}\u{3083}\u{3093}".to_string()),
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, recipe.name);
        assert_eq!(decoded.tags.len(), 2);
        assert_eq!(decoded.source_attribution, recipe.source_attribution);
    }

    #[test]
    fn seed_variety_one_day_maturity_roundtrip() {
        let seed = SeedVariety {
            name: "Quick Sprout".to_string(),
            species: "Lepidium sativum".to_string(),
            origin: None,
            days_to_maturity: 1,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        let json = serde_json::to_string(&seed).unwrap();
        let decoded: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.days_to_maturity, 1);
    }

    // ========================================================================
    // UpdateSeedVarietyInput tests
    // ========================================================================

    #[test]
    fn update_seed_variety_input_struct_construction() {
        let input = UpdateSeedVarietyInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: SeedVariety {
                name: "Updated Cherokee Purple".to_string(),
                species: "Solanum lycopersicum".to_string(),
                origin: Some("Tennessee".to_string()),
                days_to_maturity: 85,
                companion_plants: vec!["Basil".to_string()],
                avoid_plants: vec![],
                seed_saving_notes: Some("Updated notes".to_string()),
            },
        };
        assert_eq!(input.original_action_hash, ActionHash::from_raw_36(vec![0xdb; 36]));
        assert_eq!(input.updated_entry.name, "Updated Cherokee Purple");
        assert_eq!(input.updated_entry.days_to_maturity, 85);
    }

    #[test]
    fn update_seed_variety_input_serde_roundtrip() {
        let input = UpdateSeedVarietyInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            updated_entry: SeedVariety {
                name: "Roma".to_string(),
                species: "Solanum lycopersicum".to_string(),
                origin: None,
                days_to_maturity: 75,
                companion_plants: vec![],
                avoid_plants: vec![],
                seed_saving_notes: None,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateSeedVarietyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.name, "Roma");
        assert_eq!(decoded.updated_entry.origin, None);
    }

    #[test]
    fn update_seed_variety_input_clone_is_equal() {
        let input = UpdateSeedVarietyInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xcd; 36]),
            updated_entry: SeedVariety {
                name: "Brandywine".to_string(),
                species: "Solanum lycopersicum".to_string(),
                origin: Some("PA".to_string()),
                days_to_maturity: 90,
                companion_plants: vec!["Basil".to_string()],
                avoid_plants: vec!["Fennel".to_string()],
                seed_saving_notes: None,
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }

    // ========================================================================
    // UpdateTraditionalPracticeInput tests
    // ========================================================================

    #[test]
    fn update_traditional_practice_input_struct_construction() {
        let input = UpdateTraditionalPracticeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: TraditionalPractice {
                name: "Updated Three Sisters".to_string(),
                description: "Corn, beans, squash, and sunflowers".to_string(),
                region: Some("North America".to_string()),
                season: Some("Late Spring".to_string()),
                category: PracticeCategory::Planting,
            },
        };
        assert_eq!(input.original_action_hash, ActionHash::from_raw_36(vec![0xdb; 36]));
        assert_eq!(input.updated_entry.name, "Updated Three Sisters");
    }

    #[test]
    fn update_traditional_practice_input_serde_roundtrip() {
        let input = UpdateTraditionalPracticeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xef; 36]),
            updated_entry: TraditionalPractice {
                name: "Companion Planting".to_string(),
                description: "Growing compatible plants near each other".to_string(),
                region: None,
                season: None,
                category: PracticeCategory::Pest,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTraditionalPracticeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.category, PracticeCategory::Pest);
        assert!(decoded.updated_entry.region.is_none());
    }

    #[test]
    fn update_traditional_practice_input_clone_is_equal() {
        let input = UpdateTraditionalPracticeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0x11; 36]),
            updated_entry: TraditionalPractice {
                name: "Milpa".to_string(),
                description: "Mesoamerican system".to_string(),
                region: Some("Central America".to_string()),
                season: Some("Spring".to_string()),
                category: PracticeCategory::Soil,
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }

    // ========================================================================
    // UpdateRecipeInput tests
    // ========================================================================

    #[test]
    fn update_recipe_input_struct_construction() {
        let input = UpdateRecipeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: Recipe {
                name: "Updated Tomato Sauce".to_string(),
                ingredients: vec!["Tomatoes".to_string(), "Garlic".to_string(), "Onion".to_string()],
                instructions: "Blend, season, and simmer".to_string(),
                servings: 6,
                prep_time_min: 60,
                tags: vec!["sauce".to_string(), "Italian".to_string()],
                source_attribution: Some("Updated cookbook".to_string()),
            },
        };
        assert_eq!(input.original_action_hash, ActionHash::from_raw_36(vec![0xdb; 36]));
        assert_eq!(input.updated_entry.servings, 6);
        assert_eq!(input.updated_entry.ingredients.len(), 3);
    }

    #[test]
    fn update_recipe_input_serde_roundtrip() {
        let input = UpdateRecipeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xaa; 36]),
            updated_entry: Recipe {
                name: "Simple Bread".to_string(),
                ingredients: vec!["Flour".to_string(), "Water".to_string()],
                instructions: "Mix and bake".to_string(),
                servings: 1,
                prep_time_min: 90,
                tags: vec![],
                source_attribution: None,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateRecipeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.name, "Simple Bread");
        assert!(decoded.updated_entry.tags.is_empty());
        assert!(decoded.updated_entry.source_attribution.is_none());
    }

    #[test]
    fn update_recipe_input_clone_is_equal() {
        let input = UpdateRecipeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xbb; 36]),
            updated_entry: Recipe {
                name: "Pozole".to_string(),
                ingredients: vec!["Hominy".to_string(), "Pork".to_string()],
                instructions: "Simmer for hours".to_string(),
                servings: 8,
                prep_time_min: 240,
                tags: vec!["Mexican".to_string()],
                source_attribution: Some("Family recipe".to_string()),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }

    // ========================================================================
    // Existing tests continue below
    // ========================================================================

    #[test]
    fn recipe_single_ingredient_no_tags_roundtrip() {
        let recipe = Recipe {
            name: "Water".to_string(),
            ingredients: vec!["H2O".to_string()],
            instructions: "Pour.".to_string(),
            servings: 1,
            prep_time_min: 1,
            tags: vec![],
            source_attribution: None,
        };
        let json = serde_json::to_string(&recipe).unwrap();
        let decoded: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ingredients.len(), 1);
        assert!(decoded.tags.is_empty());
        assert_eq!(decoded.source_attribution, None);
    }
}
