//! Food Knowledge Integrity Zome
//! Entry types and validation for seed varieties, traditional practices, and recipes.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// SEED VARIETY
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SeedVariety {
    pub name: String,
    pub species: String,
    pub origin: Option<String>,
    pub days_to_maturity: u32,
    pub companion_plants: Vec<String>,
    pub avoid_plants: Vec<String>,
    pub seed_saving_notes: Option<String>,
}

// ============================================================================
// TRADITIONAL PRACTICE
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PracticeCategory {
    Planting,
    Harvest,
    Soil,
    Pest,
    Water,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TraditionalPractice {
    pub name: String,
    pub description: String,
    pub region: Option<String>,
    pub season: Option<String>,
    pub category: PracticeCategory,
}

// ============================================================================
// RECIPE
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Recipe {
    pub name: String,
    pub ingredients: Vec<String>,
    pub instructions: String,
    pub servings: u32,
    pub prep_time_min: u32,
    pub tags: Vec<String>,
    pub source_attribution: Option<String>,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    SeedVariety(SeedVariety),
    TraditionalPractice(TraditionalPractice),
    Recipe(Recipe),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllSeeds,
    AllPractices,
    AllRecipes,
    SpeciesToSeed,
    CategoryToPractice,
    TagToRecipe,
    AgentToRecipe,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SeedVariety(s) => validate_seed(s),
                EntryTypes::TraditionalPractice(p) => validate_practice(p),
                EntryTypes::Recipe(r) => validate_recipe(r),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_seed(s: SeedVariety) -> ExternResult<ValidateCallbackResult> {
    if s.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Seed name cannot be empty".into()));
    }
    if s.species.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Species cannot be empty".into()));
    }
    if s.days_to_maturity == 0 {
        return Ok(ValidateCallbackResult::Invalid("Days to maturity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_practice(p: TraditionalPractice) -> ExternResult<ValidateCallbackResult> {
    if p.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Practice name cannot be empty".into()));
    }
    if p.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Description cannot be empty".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_recipe(r: Recipe) -> ExternResult<ValidateCallbackResult> {
    if r.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Recipe name cannot be empty".into()));
    }
    if r.ingredients.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Must have at least one ingredient".into()));
    }
    if r.instructions.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Instructions cannot be empty".into()));
    }
    if r.servings == 0 {
        return Ok(ValidateCallbackResult::Invalid("Servings must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn valid_seed() -> SeedVariety {
        SeedVariety {
            name: "Cherokee Purple".into(),
            species: "Solanum lycopersicum".into(),
            origin: Some("Tennessee".into()),
            days_to_maturity: 80,
            companion_plants: vec!["Basil".into(), "Marigold".into()],
            avoid_plants: vec!["Fennel".into()],
            seed_saving_notes: Some("Open-pollinated, save from ripe fruit".into()),
        }
    }

    fn valid_practice() -> TraditionalPractice {
        TraditionalPractice {
            name: "Three Sisters Planting".into(),
            description: "Corn, beans, and squash planted together".into(),
            region: Some("Haudenosaunee territory".into()),
            season: Some("Spring".into()),
            category: PracticeCategory::Planting,
        }
    }

    fn valid_recipe() -> Recipe {
        Recipe {
            name: "Tomato Sauce".into(),
            ingredients: vec!["Tomatoes".into(), "Garlic".into(), "Basil".into()],
            instructions: "Blend tomatoes, simmer with garlic and basil for 30 min".into(),
            servings: 4,
            prep_time_min: 45,
            tags: vec!["sauce".into(), "preserving".into()],
            source_attribution: Some("Community cookbook".into()),
        }
    }

    fn assert_valid(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    // ── Serde roundtrip tests ───────────────────────────────────────────

    #[test]
    fn serde_roundtrip_practice_category() {
        let cats = vec![
            PracticeCategory::Planting, PracticeCategory::Harvest,
            PracticeCategory::Soil, PracticeCategory::Pest, PracticeCategory::Water,
        ];
        for c in &cats {
            let json = serde_json::to_string(c).unwrap();
            let back: PracticeCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, c);
        }
    }

    #[test]
    fn serde_roundtrip_seed_variety() {
        let s = valid_seed();
        let json = serde_json::to_string(&s).unwrap();
        let back: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn serde_roundtrip_seed_minimal() {
        let s = SeedVariety {
            name: "Test".into(),
            species: "Test".into(),
            origin: None,
            days_to_maturity: 1,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: SeedVariety = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn serde_roundtrip_traditional_practice() {
        let p = valid_practice();
        let json = serde_json::to_string(&p).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn serde_roundtrip_practice_minimal() {
        let p = TraditionalPractice {
            name: "Test".into(),
            description: "Desc".into(),
            region: None,
            season: None,
            category: PracticeCategory::Soil,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn serde_roundtrip_recipe() {
        let r = valid_recipe();
        let json = serde_json::to_string(&r).unwrap();
        let back: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_recipe_minimal() {
        let r = Recipe {
            name: "Simple".into(),
            ingredients: vec!["Water".into()],
            instructions: "Boil".into(),
            servings: 1,
            prep_time_min: 0,
            tags: vec![],
            source_attribution: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Recipe = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ── validate_seed: name ─────────────────────────────────────────────

    #[test]
    fn valid_seed_passes() {
        assert_valid(validate_seed(valid_seed()));
    }

    #[test]
    fn seed_empty_name_rejected() {
        let mut s = valid_seed();
        s.name = String::new();
        assert_invalid(validate_seed(s), "Seed name cannot be empty");
    }

    #[test]
    fn seed_whitespace_name_rejected() {
        let mut s = valid_seed();
        s.name = " ".into();
        assert_invalid(validate_seed(s), "Seed name cannot be empty");
    }

    // ── validate_seed: species ──────────────────────────────────────────

    #[test]
    fn seed_empty_species_rejected() {
        let mut s = valid_seed();
        s.species = String::new();
        assert_invalid(validate_seed(s), "Species cannot be empty");
    }

    #[test]
    fn seed_whitespace_species_rejected() {
        let mut s = valid_seed();
        s.species = "  ".into();
        assert_invalid(validate_seed(s), "Species cannot be empty");
    }

    // ── validate_seed: days_to_maturity ─────────────────────────────────

    #[test]
    fn seed_zero_maturity_rejected() {
        let mut s = valid_seed();
        s.days_to_maturity = 0;
        assert_invalid(validate_seed(s), "Days to maturity must be positive");
    }

    #[test]
    fn seed_one_day_maturity_valid() {
        let mut s = valid_seed();
        s.days_to_maturity = 1;
        assert_valid(validate_seed(s));
    }

    #[test]
    fn seed_large_maturity_valid() {
        let mut s = valid_seed();
        s.days_to_maturity = 365;
        assert_valid(validate_seed(s));
    }

    #[test]
    fn seed_very_large_maturity_valid() {
        let mut s = valid_seed();
        s.days_to_maturity = u32::MAX;
        assert_valid(validate_seed(s));
    }

    // ── validate_seed: optional fields ──────────────────────────────────

    #[test]
    fn seed_no_origin_valid() {
        let mut s = valid_seed();
        s.origin = None;
        assert_valid(validate_seed(s));
    }

    #[test]
    fn seed_empty_companion_plants_valid() {
        let mut s = valid_seed();
        s.companion_plants = vec![];
        assert_valid(validate_seed(s));
    }

    #[test]
    fn seed_empty_avoid_plants_valid() {
        let mut s = valid_seed();
        s.avoid_plants = vec![];
        assert_valid(validate_seed(s));
    }

    #[test]
    fn seed_no_saving_notes_valid() {
        let mut s = valid_seed();
        s.seed_saving_notes = None;
        assert_valid(validate_seed(s));
    }

    // ── validate_seed: combined invalid ─────────────────────────────────

    #[test]
    fn seed_empty_name_with_empty_species_rejects_name_first() {
        let mut s = valid_seed();
        s.name = String::new();
        s.species = String::new();
        assert_invalid(validate_seed(s), "Seed name cannot be empty");
    }

    #[test]
    fn seed_empty_species_with_zero_maturity_rejects_species_first() {
        let mut s = valid_seed();
        s.species = String::new();
        s.days_to_maturity = 0;
        assert_invalid(validate_seed(s), "Species cannot be empty");
    }

    // ── validate_practice: name ─────────────────────────────────────────

    #[test]
    fn valid_practice_passes() {
        assert_valid(validate_practice(valid_practice()));
    }

    #[test]
    fn practice_empty_name_rejected() {
        let mut p = valid_practice();
        p.name = String::new();
        assert_invalid(validate_practice(p), "Practice name cannot be empty");
    }

    #[test]
    fn practice_whitespace_name_rejected() {
        let mut p = valid_practice();
        p.name = " ".into();
        assert_invalid(validate_practice(p), "Practice name cannot be empty");
    }

    // ── validate_practice: description ──────────────────────────────────

    #[test]
    fn practice_empty_description_rejected() {
        let mut p = valid_practice();
        p.description = String::new();
        assert_invalid(validate_practice(p), "Description cannot be empty");
    }

    #[test]
    fn practice_whitespace_description_rejected() {
        let mut p = valid_practice();
        p.description = "  ".into();
        assert_invalid(validate_practice(p), "Description cannot be empty");
    }

    // ── validate_practice: category variants ────────────────────────────

    #[test]
    fn all_practice_categories_valid() {
        for cat in [PracticeCategory::Planting, PracticeCategory::Harvest,
                     PracticeCategory::Soil, PracticeCategory::Pest, PracticeCategory::Water] {
            let mut p = valid_practice();
            p.category = cat;
            assert_valid(validate_practice(p));
        }
    }

    // ── validate_practice: optional fields ──────────────────────────────

    #[test]
    fn practice_no_region_valid() {
        let mut p = valid_practice();
        p.region = None;
        assert_valid(validate_practice(p));
    }

    #[test]
    fn practice_no_season_valid() {
        let mut p = valid_practice();
        p.season = None;
        assert_valid(validate_practice(p));
    }

    #[test]
    fn practice_no_optionals_valid() {
        let mut p = valid_practice();
        p.region = None;
        p.season = None;
        assert_valid(validate_practice(p));
    }

    // ── validate_practice: combined invalid ─────────────────────────────

    #[test]
    fn practice_empty_name_and_description_rejects_name_first() {
        let mut p = valid_practice();
        p.name = String::new();
        p.description = String::new();
        assert_invalid(validate_practice(p), "Practice name cannot be empty");
    }

    // ── validate_recipe: name ───────────────────────────────────────────

    #[test]
    fn valid_recipe_passes() {
        assert_valid(validate_recipe(valid_recipe()));
    }

    #[test]
    fn recipe_empty_name_rejected() {
        let mut r = valid_recipe();
        r.name = String::new();
        assert_invalid(validate_recipe(r), "Recipe name cannot be empty");
    }

    #[test]
    fn recipe_whitespace_name_rejected() {
        let mut r = valid_recipe();
        r.name = " ".into();
        assert_invalid(validate_recipe(r), "Recipe name cannot be empty");
    }

    // ── validate_recipe: ingredients ────────────────────────────────────

    #[test]
    fn recipe_no_ingredients_rejected() {
        let mut r = valid_recipe();
        r.ingredients = vec![];
        assert_invalid(validate_recipe(r), "Must have at least one ingredient");
    }

    #[test]
    fn recipe_one_ingredient_valid() {
        let mut r = valid_recipe();
        r.ingredients = vec!["Water".into()];
        assert_valid(validate_recipe(r));
    }

    #[test]
    fn recipe_many_ingredients_valid() {
        let mut r = valid_recipe();
        r.ingredients = (0..50).map(|i| format!("ingredient_{i}")).collect();
        assert_valid(validate_recipe(r));
    }

    // ── validate_recipe: instructions ───────────────────────────────────

    #[test]
    fn recipe_empty_instructions_rejected() {
        let mut r = valid_recipe();
        r.instructions = String::new();
        assert_invalid(validate_recipe(r), "Instructions cannot be empty");
    }

    #[test]
    fn recipe_whitespace_instructions_rejected() {
        let mut r = valid_recipe();
        r.instructions = " ".into();
        assert_invalid(validate_recipe(r), "Instructions cannot be empty");
    }

    // ── validate_recipe: servings ───────────────────────────────────────

    #[test]
    fn recipe_zero_servings_rejected() {
        let mut r = valid_recipe();
        r.servings = 0;
        assert_invalid(validate_recipe(r), "Servings must be positive");
    }

    #[test]
    fn recipe_one_serving_valid() {
        let mut r = valid_recipe();
        r.servings = 1;
        assert_valid(validate_recipe(r));
    }

    #[test]
    fn recipe_large_servings_valid() {
        let mut r = valid_recipe();
        r.servings = 1000;
        assert_valid(validate_recipe(r));
    }

    // ── validate_recipe: optional fields ────────────────────────────────

    #[test]
    fn recipe_zero_prep_time_valid() {
        let mut r = valid_recipe();
        r.prep_time_min = 0;
        assert_valid(validate_recipe(r));
    }

    #[test]
    fn recipe_empty_tags_valid() {
        let mut r = valid_recipe();
        r.tags = vec![];
        assert_valid(validate_recipe(r));
    }

    #[test]
    fn recipe_no_attribution_valid() {
        let mut r = valid_recipe();
        r.source_attribution = None;
        assert_valid(validate_recipe(r));
    }

    // ── validate_recipe: combined invalid ───────────────────────────────

    #[test]
    fn recipe_empty_name_with_no_ingredients_rejects_name_first() {
        let mut r = valid_recipe();
        r.name = String::new();
        r.ingredients = vec![];
        assert_invalid(validate_recipe(r), "Recipe name cannot be empty");
    }

    #[test]
    fn recipe_no_ingredients_with_empty_instructions_rejects_ingredients_first() {
        let mut r = valid_recipe();
        r.ingredients = vec![];
        r.instructions = String::new();
        assert_invalid(validate_recipe(r), "Must have at least one ingredient");
    }

    #[test]
    fn recipe_empty_instructions_with_zero_servings_rejects_instructions_first() {
        let mut r = valid_recipe();
        r.instructions = String::new();
        r.servings = 0;
        assert_invalid(validate_recipe(r), "Instructions cannot be empty");
    }

    // ── Anchor test ─────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_seeds".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }
}
