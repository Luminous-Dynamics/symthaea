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
