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
    if s.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Seed name cannot be empty".into()));
    }
    if s.species.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Species cannot be empty".into()));
    }
    if s.days_to_maturity == 0 {
        return Ok(ValidateCallbackResult::Invalid("Days to maturity must be positive".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_practice(p: TraditionalPractice) -> ExternResult<ValidateCallbackResult> {
    if p.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Practice name cannot be empty".into()));
    }
    if p.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Description cannot be empty".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_recipe(r: Recipe) -> ExternResult<ValidateCallbackResult> {
    if r.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Recipe name cannot be empty".into()));
    }
    if r.ingredients.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Must have at least one ingredient".into()));
    }
    if r.instructions.is_empty() {
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

    #[test]
    fn valid_seed_passes() {
        let s = SeedVariety {
            name: "Cherokee Purple".into(),
            species: "Solanum lycopersicum".into(),
            origin: Some("Tennessee".into()),
            days_to_maturity: 80,
            companion_plants: vec!["Basil".into(), "Marigold".into()],
            avoid_plants: vec!["Fennel".into()],
            seed_saving_notes: Some("Open-pollinated, save from ripe fruit".into()),
        };
        assert_eq!(validate_seed(s).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn seed_empty_name_rejected() {
        let s = SeedVariety {
            name: String::new(),
            species: "Solanum lycopersicum".into(),
            origin: None,
            days_to_maturity: 80,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        assert!(matches!(validate_seed(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn seed_empty_species_rejected() {
        let s = SeedVariety {
            name: "Tomato".into(),
            species: String::new(),
            origin: None,
            days_to_maturity: 80,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        assert!(matches!(validate_seed(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn seed_zero_maturity_rejected() {
        let s = SeedVariety {
            name: "Tomato".into(),
            species: "Solanum lycopersicum".into(),
            origin: None,
            days_to_maturity: 0,
            companion_plants: vec![],
            avoid_plants: vec![],
            seed_saving_notes: None,
        };
        assert!(matches!(validate_seed(s).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn valid_practice_passes() {
        let p = TraditionalPractice {
            name: "Three Sisters Planting".into(),
            description: "Corn, beans, and squash planted together".into(),
            region: Some("Haudenosaunee territory".into()),
            season: Some("Spring".into()),
            category: PracticeCategory::Planting,
        };
        assert_eq!(validate_practice(p).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn practice_empty_name_rejected() {
        let p = TraditionalPractice {
            name: String::new(),
            description: "A practice".into(),
            region: None,
            season: None,
            category: PracticeCategory::Soil,
        };
        assert!(matches!(validate_practice(p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn practice_empty_description_rejected() {
        let p = TraditionalPractice {
            name: "Mulching".into(),
            description: String::new(),
            region: None,
            season: None,
            category: PracticeCategory::Soil,
        };
        assert!(matches!(validate_practice(p).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn all_practice_categories_valid() {
        for cat in [PracticeCategory::Planting, PracticeCategory::Harvest,
                     PracticeCategory::Soil, PracticeCategory::Pest, PracticeCategory::Water] {
            let p = TraditionalPractice {
                name: "Test".into(),
                description: "Test desc".into(),
                region: None,
                season: None,
                category: cat,
            };
            assert_eq!(validate_practice(p).unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn valid_recipe_passes() {
        let r = Recipe {
            name: "Tomato Sauce".into(),
            ingredients: vec!["Tomatoes".into(), "Garlic".into(), "Basil".into()],
            instructions: "Blend tomatoes, simmer with garlic and basil for 30 min".into(),
            servings: 4,
            prep_time_min: 45,
            tags: vec!["sauce".into(), "preserving".into()],
            source_attribution: Some("Community cookbook".into()),
        };
        assert_eq!(validate_recipe(r).unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn recipe_empty_name_rejected() {
        let r = Recipe {
            name: String::new(),
            ingredients: vec!["Flour".into()],
            instructions: "Mix".into(),
            servings: 1,
            prep_time_min: 5,
            tags: vec![],
            source_attribution: None,
        };
        assert!(matches!(validate_recipe(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn recipe_no_ingredients_rejected() {
        let r = Recipe {
            name: "Empty Dish".into(),
            ingredients: vec![],
            instructions: "Do nothing".into(),
            servings: 1,
            prep_time_min: 0,
            tags: vec![],
            source_attribution: None,
        };
        assert!(matches!(validate_recipe(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn recipe_empty_instructions_rejected() {
        let r = Recipe {
            name: "Mystery".into(),
            ingredients: vec!["Something".into()],
            instructions: String::new(),
            servings: 1,
            prep_time_min: 0,
            tags: vec![],
            source_attribution: None,
        };
        assert!(matches!(validate_recipe(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn recipe_zero_servings_rejected() {
        let r = Recipe {
            name: "Zero Serve".into(),
            ingredients: vec!["Flour".into()],
            instructions: "Make something".into(),
            servings: 0,
            prep_time_min: 10,
            tags: vec![],
            source_attribution: None,
        };
        assert!(matches!(validate_recipe(r).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
}
