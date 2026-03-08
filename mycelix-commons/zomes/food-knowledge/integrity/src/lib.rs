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
// SEED STOCK (Exchange)
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SeedStock {
    pub variety_hash: ActionHash,
    pub grower: AgentPubKey,
    pub quantity_grams: f64,
    pub location: String,
    pub germination_rate_pct: Option<f64>,
    pub available_for_exchange: bool,
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SeedRequestStatus {
    Open,
    Matched,
    Fulfilled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SeedRequest {
    pub wanted_variety: String,
    pub quantity_grams: f64,
    pub requester: AgentPubKey,
    pub status: SeedRequestStatus,
    pub deadline: Option<u64>,
}

// ============================================================================
// NUTRIENT PROFILE
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct NutrientProfile {
    pub crop_name: String,
    pub calories_per_100g: f64,
    pub protein_g: f64,
    pub carbs_g: f64,
    pub fat_g: f64,
    pub fiber_g: f64,
    pub key_vitamins: Vec<String>,
    pub key_minerals: Vec<String>,
}

// ============================================================================
// SEED QUALITY RATING
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SeedQualityRating {
    pub exchange_hash: ActionHash,
    pub rater: AgentPubKey,
    pub rating: u8,
    pub germination_observed_pct: Option<f64>,
    pub comment: Option<String>,
    pub rated_at: u64,
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
    SeedStock(SeedStock),
    SeedRequest(SeedRequest),
    NutrientProfile(NutrientProfile),
    SeedQualityRating(SeedQualityRating),
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
    VarietyToStocks,
    AllSeedRequests,
    CropToNutrients,
    ExchangeToRatings,
    GrowerToRatings,
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
                EntryTypes::SeedStock(s) => validate_seed_stock(s),
                EntryTypes::SeedRequest(r) => validate_seed_request(r),
                EntryTypes::NutrientProfile(n) => validate_nutrient_profile(n),
                EntryTypes::SeedQualityRating(r) => validate_seed_quality_rating(r),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SeedVariety(s) => validate_seed(s),
                EntryTypes::TraditionalPractice(p) => validate_practice(p),
                EntryTypes::Recipe(r) => validate_recipe(r),
                EntryTypes::SeedStock(s) => validate_seed_stock(s),
                EntryTypes::SeedRequest(r) => validate_seed_request(r),
                EntryTypes::NutrientProfile(n) => validate_nutrient_profile(n),
                EntryTypes::SeedQualityRating(r) => validate_seed_quality_rating(r),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::AllSeeds => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllSeeds link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllPractices => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllPractices link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllRecipes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllRecipes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SpeciesToSeed => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SpeciesToSeed link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CategoryToPractice => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CategoryToPractice link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TagToRecipe => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TagToRecipe link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToRecipe => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToRecipe link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::VarietyToStocks => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "VarietyToStocks link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllSeedRequests => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllSeedRequests link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CropToNutrients => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CropToNutrients link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ExchangeToRatings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ExchangeToRatings link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::GrowerToRatings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "GrowerToRatings link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_seed(s: SeedVariety) -> ExternResult<ValidateCallbackResult> {
    if s.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Seed name cannot be empty".into(),
        ));
    }
    if s.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Seed name too long (max 256 chars)".into(),
        ));
    }
    if s.species.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Species cannot be empty".into(),
        ));
    }
    if s.species.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Species too long (max 128 chars)".into(),
        ));
    }
    if s.days_to_maturity == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Days to maturity must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_practice(p: TraditionalPractice) -> ExternResult<ValidateCallbackResult> {
    if p.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice name cannot be empty".into(),
        ));
    }
    if p.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice name too long (max 256 chars)".into(),
        ));
    }
    if p.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Description cannot be empty".into(),
        ));
    }
    if p.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description too long (max 4096 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_recipe(r: Recipe) -> ExternResult<ValidateCallbackResult> {
    if r.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipe name cannot be empty".into(),
        ));
    }
    if r.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recipe name too long (max 256 chars)".into(),
        ));
    }
    if r.ingredients.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have at least one ingredient".into(),
        ));
    }
    if r.instructions.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Instructions cannot be empty".into(),
        ));
    }
    if r.instructions.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Instructions too long (max 4096 chars)".into(),
        ));
    }
    if r.servings == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Servings must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_seed_stock(s: SeedStock) -> ExternResult<ValidateCallbackResult> {
    if !s.quantity_grams.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a finite number".into(),
        ));
    }
    if s.quantity_grams <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedStock quantity must be positive".into(),
        ));
    }
    if s.location.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedStock location cannot be empty".into(),
        ));
    }
    if s.location.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedStock location too long (max 512 chars)".into(),
        ));
    }
    if let Some(rate) = s.germination_rate_pct {
        if !rate.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Germination rate must be a finite number".into(),
            ));
        }
        if !(0.0..=100.0).contains(&rate) {
            return Ok(ValidateCallbackResult::Invalid(
                "Germination rate must be between 0 and 100".into(),
            ));
        }
    }
    if let Some(ref notes) = s.notes {
        if notes.len() > 2048 {
            return Ok(ValidateCallbackResult::Invalid(
                "SeedStock notes too long (max 2048 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_seed_request(r: SeedRequest) -> ExternResult<ValidateCallbackResult> {
    if r.wanted_variety.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedRequest wanted_variety cannot be empty".into(),
        ));
    }
    if r.wanted_variety.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedRequest wanted_variety too long (max 256 chars)".into(),
        ));
    }
    if !r.quantity_grams.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be a finite number".into(),
        ));
    }
    if r.quantity_grams <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedRequest quantity must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_seed_quality_rating(r: SeedQualityRating) -> ExternResult<ValidateCallbackResult> {
    if r.rating < 1 || r.rating > 5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rating must be between 1 and 5".into(),
        ));
    }
    if r.rated_at == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "SeedQualityRating rated_at cannot be zero".into(),
        ));
    }
    if let Some(pct) = r.germination_observed_pct {
        if !(0.0..=100.0).contains(&pct) {
            return Ok(ValidateCallbackResult::Invalid(
                "Observed germination rate must be between 0 and 100".into(),
            ));
        }
    }
    if let Some(ref comment) = r.comment {
        if comment.len() > 2048 {
            return Ok(ValidateCallbackResult::Invalid(
                "Rating comment too long (max 2048 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_nutrient_profile(n: NutrientProfile) -> ExternResult<ValidateCallbackResult> {
    if n.crop_name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "NutrientProfile crop_name cannot be empty".into(),
        ));
    }
    if n.crop_name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "NutrientProfile crop_name too long (max 256 chars)".into(),
        ));
    }
    if !n.calories_per_100g.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Calories must be a finite number".into(),
        ));
    }
    if n.calories_per_100g < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Calories cannot be negative".into(),
        ));
    }
    if !n.protein_g.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Protein must be a finite number".into(),
        ));
    }
    if n.protein_g < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Protein cannot be negative".into(),
        ));
    }
    if !n.carbs_g.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Carbs must be a finite number".into(),
        ));
    }
    if n.carbs_g < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Carbs cannot be negative".into(),
        ));
    }
    if !n.fat_g.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Fat must be a finite number".into(),
        ));
    }
    if n.fat_g < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fat cannot be negative".into(),
        ));
    }
    if !n.fiber_g.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Fiber must be a finite number".into(),
        ));
    }
    if n.fiber_g < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fiber cannot be negative".into(),
        ));
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
            PracticeCategory::Planting,
            PracticeCategory::Harvest,
            PracticeCategory::Soil,
            PracticeCategory::Pest,
            PracticeCategory::Water,
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

    // ── validate_seed: name length ────────────────────────────────────

    #[test]
    fn seed_name_too_long_rejected() {
        let mut s = valid_seed();
        s.name = "x".repeat(257);
        assert_invalid(validate_seed(s), "Seed name too long (max 256 chars)");
    }

    #[test]
    fn seed_name_at_max_valid() {
        let mut s = valid_seed();
        s.name = "x".repeat(256);
        assert_valid(validate_seed(s));
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

    // ── validate_seed: species length ─────────────────────────────────

    #[test]
    fn seed_species_too_long_rejected() {
        let mut s = valid_seed();
        s.species = "x".repeat(129);
        assert_invalid(validate_seed(s), "Species too long (max 128 chars)");
    }

    #[test]
    fn seed_species_at_max_valid() {
        let mut s = valid_seed();
        s.species = "x".repeat(128);
        assert_valid(validate_seed(s));
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

    // ── validate_practice: name length ────────────────────────────────

    #[test]
    fn practice_name_too_long_rejected() {
        let mut p = valid_practice();
        p.name = "x".repeat(257);
        assert_invalid(
            validate_practice(p),
            "Practice name too long (max 256 chars)",
        );
    }

    #[test]
    fn practice_name_at_max_valid() {
        let mut p = valid_practice();
        p.name = "x".repeat(256);
        assert_valid(validate_practice(p));
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

    // ── validate_practice: description length ─────────────────────────

    #[test]
    fn practice_description_too_long_rejected() {
        let mut p = valid_practice();
        p.description = "x".repeat(4097);
        assert_invalid(
            validate_practice(p),
            "Description too long (max 4096 chars)",
        );
    }

    #[test]
    fn practice_description_at_max_valid() {
        let mut p = valid_practice();
        p.description = "x".repeat(4096);
        assert_valid(validate_practice(p));
    }

    // ── validate_practice: category variants ────────────────────────────

    #[test]
    fn all_practice_categories_valid() {
        for cat in [
            PracticeCategory::Planting,
            PracticeCategory::Harvest,
            PracticeCategory::Soil,
            PracticeCategory::Pest,
            PracticeCategory::Water,
        ] {
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

    // ── validate_recipe: name length ──────────────────────────────────

    #[test]
    fn recipe_name_too_long_rejected() {
        let mut r = valid_recipe();
        r.name = "x".repeat(257);
        assert_invalid(validate_recipe(r), "Recipe name too long (max 256 chars)");
    }

    #[test]
    fn recipe_name_at_max_valid() {
        let mut r = valid_recipe();
        r.name = "x".repeat(256);
        assert_valid(validate_recipe(r));
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

    // ── validate_recipe: instructions length ──────────────────────────

    #[test]
    fn recipe_instructions_too_long_rejected() {
        let mut r = valid_recipe();
        r.instructions = "x".repeat(4097);
        assert_invalid(validate_recipe(r), "Instructions too long (max 4096 chars)");
    }

    #[test]
    fn recipe_instructions_at_max_valid() {
        let mut r = valid_recipe();
        r.instructions = "x".repeat(4096);
        assert_valid(validate_recipe(r));
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

    // ── Update validation: SeedVariety ───────────────────────────────────

    #[test]
    fn test_update_seed_invalid_name_rejected() {
        let mut s = valid_seed();
        s.name = String::new();
        assert_invalid(validate_seed(s), "Seed name cannot be empty");
    }

    #[test]
    fn test_update_seed_invalid_species_rejected() {
        let mut s = valid_seed();
        s.species = "  ".into();
        assert_invalid(validate_seed(s), "Species cannot be empty");
    }

    #[test]
    fn test_update_seed_invalid_maturity_rejected() {
        let mut s = valid_seed();
        s.days_to_maturity = 0;
        assert_invalid(validate_seed(s), "Days to maturity must be positive");
    }

    // ── Update validation: TraditionalPractice ───────────────────────────

    #[test]
    fn test_update_practice_invalid_name_rejected() {
        let mut p = valid_practice();
        p.name = String::new();
        assert_invalid(validate_practice(p), "Practice name cannot be empty");
    }

    #[test]
    fn test_update_practice_invalid_description_rejected() {
        let mut p = valid_practice();
        p.description = "  ".into();
        assert_invalid(validate_practice(p), "Description cannot be empty");
    }

    // ── Update validation: Recipe ────────────────────────────────────────

    #[test]
    fn test_update_recipe_invalid_name_rejected() {
        let mut r = valid_recipe();
        r.name = String::new();
        assert_invalid(validate_recipe(r), "Recipe name cannot be empty");
    }

    #[test]
    fn test_update_recipe_invalid_ingredients_rejected() {
        let mut r = valid_recipe();
        r.ingredients = vec![];
        assert_invalid(validate_recipe(r), "Must have at least one ingredient");
    }

    #[test]
    fn test_update_recipe_invalid_servings_rejected() {
        let mut r = valid_recipe();
        r.servings = 0;
        assert_invalid(validate_recipe(r), "Servings must be positive");
    }

    // ── Link tag length validation tests ────────────────────────────────

    fn validate_link_tag(link_type: &LinkTypes, tag_len: usize) -> ValidateCallbackResult {
        let tag = LinkTag(vec![0u8; tag_len]);
        let max = match link_type {
            LinkTypes::AllSeeds
            | LinkTypes::AllPractices
            | LinkTypes::AllRecipes
            | LinkTypes::SpeciesToSeed
            | LinkTypes::CategoryToPractice
            | LinkTypes::AgentToRecipe
            | LinkTypes::VarietyToStocks
            | LinkTypes::AllSeedRequests
            | LinkTypes::CropToNutrients
            | LinkTypes::ExchangeToRatings
            | LinkTypes::GrowerToRatings => 256,
            LinkTypes::TagToRecipe => 512,
        };
        let name = match link_type {
            LinkTypes::AllSeeds => "AllSeeds",
            LinkTypes::AllPractices => "AllPractices",
            LinkTypes::AllRecipes => "AllRecipes",
            LinkTypes::SpeciesToSeed => "SpeciesToSeed",
            LinkTypes::CategoryToPractice => "CategoryToPractice",
            LinkTypes::TagToRecipe => "TagToRecipe",
            LinkTypes::AgentToRecipe => "AgentToRecipe",
            LinkTypes::VarietyToStocks => "VarietyToStocks",
            LinkTypes::AllSeedRequests => "AllSeedRequests",
            LinkTypes::CropToNutrients => "CropToNutrients",
            LinkTypes::ExchangeToRatings => "ExchangeToRatings",
            LinkTypes::GrowerToRatings => "GrowerToRatings",
        };
        if tag.0.len() > max {
            ValidateCallbackResult::Invalid(format!(
                "{} link tag too long (max {} bytes)",
                name, max
            ))
        } else {
            ValidateCallbackResult::Valid
        }
    }

    #[test]
    fn test_link_all_seeds_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllSeeds, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_seeds_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllSeeds, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_all_practices_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllPractices, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_practices_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllPractices, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_all_recipes_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllRecipes, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_recipes_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllRecipes, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_species_to_seed_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::SpeciesToSeed, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_species_to_seed_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::SpeciesToSeed, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_category_to_practice_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::CategoryToPractice, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_category_to_practice_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::CategoryToPractice, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_to_recipe_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::TagToRecipe, 512);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_to_recipe_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::TagToRecipe, 513);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_agent_to_recipe_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToRecipe, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_agent_to_recipe_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToRecipe, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── New type test helpers ───────────────────────────────────────────

    fn valid_seed_stock() -> SeedStock {
        SeedStock {
            variety_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            grower: AgentPubKey::from_raw_36(vec![0xab; 36]),
            quantity_grams: 50.0,
            location: "Community garden shed".into(),
            germination_rate_pct: Some(85.0),
            available_for_exchange: true,
            notes: Some("Harvested fall 2025".into()),
        }
    }

    fn valid_seed_request() -> SeedRequest {
        SeedRequest {
            wanted_variety: "Cherokee Purple".into(),
            quantity_grams: 10.0,
            requester: AgentPubKey::from_raw_36(vec![0xcd; 36]),
            status: SeedRequestStatus::Open,
            deadline: Some(1700000000),
        }
    }

    fn valid_nutrient_profile() -> NutrientProfile {
        NutrientProfile {
            crop_name: "Kale".into(),
            calories_per_100g: 49.0,
            protein_g: 4.3,
            carbs_g: 8.8,
            fat_g: 0.9,
            fiber_g: 3.6,
            key_vitamins: vec!["A".into(), "C".into(), "K".into()],
            key_minerals: vec!["Calcium".into(), "Iron".into()],
        }
    }

    // ── Serde roundtrip: SeedStock ──────────────────────────────────────

    #[test]
    fn serde_roundtrip_seed_stock() {
        let s = valid_seed_stock();
        let json = serde_json::to_string(&s).unwrap();
        let back: SeedStock = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn serde_roundtrip_seed_stock_minimal() {
        let s = SeedStock {
            variety_hash: ActionHash::from_raw_36(vec![0xaa; 36]),
            grower: AgentPubKey::from_raw_36(vec![0xbb; 36]),
            quantity_grams: 1.0,
            location: "A".into(),
            germination_rate_pct: None,
            available_for_exchange: false,
            notes: None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: SeedStock = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    // ── Serde roundtrip: SeedRequest ────────────────────────────────────

    #[test]
    fn serde_roundtrip_seed_request_open() {
        let r = valid_seed_request();
        let json = serde_json::to_string(&r).unwrap();
        let back: SeedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_seed_request_matched() {
        let mut r = valid_seed_request();
        r.status = SeedRequestStatus::Matched;
        let json = serde_json::to_string(&r).unwrap();
        let back: SeedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_seed_request_fulfilled() {
        let mut r = valid_seed_request();
        r.status = SeedRequestStatus::Fulfilled;
        r.deadline = None;
        let json = serde_json::to_string(&r).unwrap();
        let back: SeedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ── Serde roundtrip: SeedRequestStatus ──────────────────────────────

    #[test]
    fn serde_roundtrip_seed_request_status_all_variants() {
        let variants = vec![
            SeedRequestStatus::Open,
            SeedRequestStatus::Matched,
            SeedRequestStatus::Fulfilled,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: SeedRequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }

    // ── Serde roundtrip: NutrientProfile ────────────────────────────────

    #[test]
    fn serde_roundtrip_nutrient_profile() {
        let n = valid_nutrient_profile();
        let json = serde_json::to_string(&n).unwrap();
        let back: NutrientProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(back, n);
    }

    #[test]
    fn serde_roundtrip_nutrient_profile_minimal() {
        let n = NutrientProfile {
            crop_name: "Rice".into(),
            calories_per_100g: 130.0,
            protein_g: 2.7,
            carbs_g: 28.2,
            fat_g: 0.3,
            fiber_g: 0.4,
            key_vitamins: vec![],
            key_minerals: vec![],
        };
        let json = serde_json::to_string(&n).unwrap();
        let back: NutrientProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(back, n);
    }

    // ── validate_seed_stock ─────────────────────────────────────────────

    #[test]
    fn valid_seed_stock_passes() {
        assert_valid(validate_seed_stock(valid_seed_stock()));
    }

    #[test]
    fn seed_stock_zero_quantity_rejected() {
        let mut s = valid_seed_stock();
        s.quantity_grams = 0.0;
        assert_invalid(
            validate_seed_stock(s),
            "SeedStock quantity must be positive",
        );
    }

    #[test]
    fn seed_stock_negative_quantity_rejected() {
        let mut s = valid_seed_stock();
        s.quantity_grams = -1.0;
        assert_invalid(
            validate_seed_stock(s),
            "SeedStock quantity must be positive",
        );
    }

    #[test]
    fn seed_stock_empty_location_rejected() {
        let mut s = valid_seed_stock();
        s.location = String::new();
        assert_invalid(validate_seed_stock(s), "SeedStock location cannot be empty");
    }

    #[test]
    fn seed_stock_whitespace_location_rejected() {
        let mut s = valid_seed_stock();
        s.location = "   ".into();
        assert_invalid(validate_seed_stock(s), "SeedStock location cannot be empty");
    }

    #[test]
    fn seed_stock_location_too_long_rejected() {
        let mut s = valid_seed_stock();
        s.location = "x".repeat(513);
        assert_invalid(
            validate_seed_stock(s),
            "SeedStock location too long (max 512 chars)",
        );
    }

    #[test]
    fn seed_stock_location_at_max_valid() {
        let mut s = valid_seed_stock();
        s.location = "x".repeat(512);
        assert_valid(validate_seed_stock(s));
    }

    #[test]
    fn seed_stock_germination_negative_rejected() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = Some(-0.1);
        assert_invalid(
            validate_seed_stock(s),
            "Germination rate must be between 0 and 100",
        );
    }

    #[test]
    fn seed_stock_germination_over_100_rejected() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = Some(100.1);
        assert_invalid(
            validate_seed_stock(s),
            "Germination rate must be between 0 and 100",
        );
    }

    #[test]
    fn seed_stock_germination_zero_valid() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = Some(0.0);
        assert_valid(validate_seed_stock(s));
    }

    #[test]
    fn seed_stock_germination_100_valid() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = Some(100.0);
        assert_valid(validate_seed_stock(s));
    }

    #[test]
    fn seed_stock_germination_none_valid() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = None;
        assert_valid(validate_seed_stock(s));
    }

    #[test]
    fn seed_stock_notes_too_long_rejected() {
        let mut s = valid_seed_stock();
        s.notes = Some("x".repeat(2049));
        assert_invalid(
            validate_seed_stock(s),
            "SeedStock notes too long (max 2048 chars)",
        );
    }

    #[test]
    fn seed_stock_notes_at_max_valid() {
        let mut s = valid_seed_stock();
        s.notes = Some("x".repeat(2048));
        assert_valid(validate_seed_stock(s));
    }

    #[test]
    fn seed_stock_notes_none_valid() {
        let mut s = valid_seed_stock();
        s.notes = None;
        assert_valid(validate_seed_stock(s));
    }

    #[test]
    fn seed_stock_not_available_valid() {
        let mut s = valid_seed_stock();
        s.available_for_exchange = false;
        assert_valid(validate_seed_stock(s));
    }

    // ── validate_seed_request ───────────────────────────────────────────

    #[test]
    fn valid_seed_request_passes() {
        assert_valid(validate_seed_request(valid_seed_request()));
    }

    #[test]
    fn seed_request_empty_variety_rejected() {
        let mut r = valid_seed_request();
        r.wanted_variety = String::new();
        assert_invalid(
            validate_seed_request(r),
            "SeedRequest wanted_variety cannot be empty",
        );
    }

    #[test]
    fn seed_request_whitespace_variety_rejected() {
        let mut r = valid_seed_request();
        r.wanted_variety = "  ".into();
        assert_invalid(
            validate_seed_request(r),
            "SeedRequest wanted_variety cannot be empty",
        );
    }

    #[test]
    fn seed_request_variety_too_long_rejected() {
        let mut r = valid_seed_request();
        r.wanted_variety = "x".repeat(257);
        assert_invalid(
            validate_seed_request(r),
            "SeedRequest wanted_variety too long (max 256 chars)",
        );
    }

    #[test]
    fn seed_request_variety_at_max_valid() {
        let mut r = valid_seed_request();
        r.wanted_variety = "x".repeat(256);
        assert_valid(validate_seed_request(r));
    }

    #[test]
    fn seed_request_zero_quantity_rejected() {
        let mut r = valid_seed_request();
        r.quantity_grams = 0.0;
        assert_invalid(
            validate_seed_request(r),
            "SeedRequest quantity must be positive",
        );
    }

    #[test]
    fn seed_request_negative_quantity_rejected() {
        let mut r = valid_seed_request();
        r.quantity_grams = -5.0;
        assert_invalid(
            validate_seed_request(r),
            "SeedRequest quantity must be positive",
        );
    }

    #[test]
    fn seed_request_no_deadline_valid() {
        let mut r = valid_seed_request();
        r.deadline = None;
        assert_valid(validate_seed_request(r));
    }

    #[test]
    fn seed_request_all_statuses_valid() {
        for status in [
            SeedRequestStatus::Open,
            SeedRequestStatus::Matched,
            SeedRequestStatus::Fulfilled,
        ] {
            let mut r = valid_seed_request();
            r.status = status;
            assert_valid(validate_seed_request(r));
        }
    }

    // ── validate_nutrient_profile ───────────────────────────────────────

    #[test]
    fn valid_nutrient_profile_passes() {
        assert_valid(validate_nutrient_profile(valid_nutrient_profile()));
    }

    #[test]
    fn nutrient_profile_empty_crop_name_rejected() {
        let mut n = valid_nutrient_profile();
        n.crop_name = String::new();
        assert_invalid(
            validate_nutrient_profile(n),
            "NutrientProfile crop_name cannot be empty",
        );
    }

    #[test]
    fn nutrient_profile_whitespace_crop_name_rejected() {
        let mut n = valid_nutrient_profile();
        n.crop_name = "  ".into();
        assert_invalid(
            validate_nutrient_profile(n),
            "NutrientProfile crop_name cannot be empty",
        );
    }

    #[test]
    fn nutrient_profile_crop_name_too_long_rejected() {
        let mut n = valid_nutrient_profile();
        n.crop_name = "x".repeat(257);
        assert_invalid(
            validate_nutrient_profile(n),
            "NutrientProfile crop_name too long (max 256 chars)",
        );
    }

    #[test]
    fn nutrient_profile_crop_name_at_max_valid() {
        let mut n = valid_nutrient_profile();
        n.crop_name = "x".repeat(256);
        assert_valid(validate_nutrient_profile(n));
    }

    #[test]
    fn nutrient_profile_negative_calories_rejected() {
        let mut n = valid_nutrient_profile();
        n.calories_per_100g = -1.0;
        assert_invalid(validate_nutrient_profile(n), "Calories cannot be negative");
    }

    #[test]
    fn nutrient_profile_zero_calories_valid() {
        let mut n = valid_nutrient_profile();
        n.calories_per_100g = 0.0;
        assert_valid(validate_nutrient_profile(n));
    }

    #[test]
    fn nutrient_profile_negative_protein_rejected() {
        let mut n = valid_nutrient_profile();
        n.protein_g = -0.1;
        assert_invalid(validate_nutrient_profile(n), "Protein cannot be negative");
    }

    #[test]
    fn nutrient_profile_negative_carbs_rejected() {
        let mut n = valid_nutrient_profile();
        n.carbs_g = -0.1;
        assert_invalid(validate_nutrient_profile(n), "Carbs cannot be negative");
    }

    #[test]
    fn nutrient_profile_negative_fat_rejected() {
        let mut n = valid_nutrient_profile();
        n.fat_g = -0.1;
        assert_invalid(validate_nutrient_profile(n), "Fat cannot be negative");
    }

    #[test]
    fn nutrient_profile_negative_fiber_rejected() {
        let mut n = valid_nutrient_profile();
        n.fiber_g = -0.1;
        assert_invalid(validate_nutrient_profile(n), "Fiber cannot be negative");
    }

    #[test]
    fn nutrient_profile_all_zero_macros_valid() {
        let n = NutrientProfile {
            crop_name: "Water".into(),
            calories_per_100g: 0.0,
            protein_g: 0.0,
            carbs_g: 0.0,
            fat_g: 0.0,
            fiber_g: 0.0,
            key_vitamins: vec![],
            key_minerals: vec![],
        };
        assert_valid(validate_nutrient_profile(n));
    }

    // ── NaN/Infinity bypass hardening tests ────────────────────────────

    #[test]
    fn seed_stock_nan_quantity_rejected() {
        let mut s = valid_seed_stock();
        s.quantity_grams = f64::NAN;
        assert_invalid(validate_seed_stock(s), "Quantity must be a finite number");
    }

    #[test]
    fn seed_stock_infinity_quantity_rejected() {
        let mut s = valid_seed_stock();
        s.quantity_grams = f64::INFINITY;
        assert_invalid(validate_seed_stock(s), "Quantity must be a finite number");
    }

    #[test]
    fn seed_stock_nan_germination_rejected() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = Some(f64::NAN);
        assert_invalid(
            validate_seed_stock(s),
            "Germination rate must be a finite number",
        );
    }

    #[test]
    fn seed_stock_infinity_germination_rejected() {
        let mut s = valid_seed_stock();
        s.germination_rate_pct = Some(f64::INFINITY);
        assert_invalid(
            validate_seed_stock(s),
            "Germination rate must be a finite number",
        );
    }

    #[test]
    fn seed_request_nan_quantity_rejected() {
        let mut r = valid_seed_request();
        r.quantity_grams = f64::NAN;
        assert_invalid(validate_seed_request(r), "Quantity must be a finite number");
    }

    #[test]
    fn seed_request_infinity_quantity_rejected() {
        let mut r = valid_seed_request();
        r.quantity_grams = f64::INFINITY;
        assert_invalid(validate_seed_request(r), "Quantity must be a finite number");
    }

    #[test]
    fn nutrient_profile_nan_calories_rejected() {
        let mut n = valid_nutrient_profile();
        n.calories_per_100g = f64::NAN;
        assert_invalid(
            validate_nutrient_profile(n),
            "Calories must be a finite number",
        );
    }

    #[test]
    fn nutrient_profile_infinity_calories_rejected() {
        let mut n = valid_nutrient_profile();
        n.calories_per_100g = f64::INFINITY;
        assert_invalid(
            validate_nutrient_profile(n),
            "Calories must be a finite number",
        );
    }

    #[test]
    fn nutrient_profile_nan_protein_rejected() {
        let mut n = valid_nutrient_profile();
        n.protein_g = f64::NAN;
        assert_invalid(
            validate_nutrient_profile(n),
            "Protein must be a finite number",
        );
    }

    #[test]
    fn nutrient_profile_nan_carbs_rejected() {
        let mut n = valid_nutrient_profile();
        n.carbs_g = f64::NAN;
        assert_invalid(
            validate_nutrient_profile(n),
            "Carbs must be a finite number",
        );
    }

    #[test]
    fn nutrient_profile_nan_fat_rejected() {
        let mut n = valid_nutrient_profile();
        n.fat_g = f64::NAN;
        assert_invalid(validate_nutrient_profile(n), "Fat must be a finite number");
    }

    #[test]
    fn nutrient_profile_nan_fiber_rejected() {
        let mut n = valid_nutrient_profile();
        n.fiber_g = f64::NAN;
        assert_invalid(
            validate_nutrient_profile(n),
            "Fiber must be a finite number",
        );
    }

    #[test]
    fn nutrient_profile_empty_vitamins_minerals_valid() {
        let mut n = valid_nutrient_profile();
        n.key_vitamins = vec![];
        n.key_minerals = vec![];
        assert_valid(validate_nutrient_profile(n));
    }

    // ── Link tag tests: new link types ──────────────────────────────────

    #[test]
    fn test_link_variety_to_stocks_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::VarietyToStocks, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_variety_to_stocks_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::VarietyToStocks, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_all_seed_requests_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AllSeedRequests, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_all_seed_requests_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AllSeedRequests, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_crop_to_nutrients_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::CropToNutrients, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_crop_to_nutrients_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::CropToNutrients, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_exchange_to_ratings_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::ExchangeToRatings, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_exchange_to_ratings_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::ExchangeToRatings, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_grower_to_ratings_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::GrowerToRatings, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_grower_to_ratings_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::GrowerToRatings, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── SeedQualityRating helpers ────────────────────────────────────────

    fn valid_seed_quality_rating() -> SeedQualityRating {
        SeedQualityRating {
            exchange_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            rater: AgentPubKey::from_raw_36(vec![0xab; 36]),
            rating: 4,
            germination_observed_pct: Some(90.0),
            comment: Some("Excellent germination rate".into()),
            rated_at: 1700000000,
        }
    }

    // ── Serde roundtrip: SeedQualityRating ──────────────────────────────

    #[test]
    fn serde_roundtrip_seed_quality_rating() {
        let r = valid_seed_quality_rating();
        let json = serde_json::to_string(&r).unwrap();
        let back: SeedQualityRating = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_seed_quality_rating_minimal() {
        let r = SeedQualityRating {
            exchange_hash: ActionHash::from_raw_36(vec![0xaa; 36]),
            rater: AgentPubKey::from_raw_36(vec![0xbb; 36]),
            rating: 1,
            germination_observed_pct: None,
            comment: None,
            rated_at: 1,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: SeedQualityRating = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ── validate_seed_quality_rating ────────────────────────────────────

    #[test]
    fn valid_seed_quality_rating_passes() {
        assert_valid(validate_seed_quality_rating(valid_seed_quality_rating()));
    }

    #[test]
    fn seed_quality_rating_zero_rejected() {
        let mut r = valid_seed_quality_rating();
        r.rating = 0;
        assert_invalid(
            validate_seed_quality_rating(r),
            "Rating must be between 1 and 5",
        );
    }

    #[test]
    fn seed_quality_rating_six_rejected() {
        let mut r = valid_seed_quality_rating();
        r.rating = 6;
        assert_invalid(
            validate_seed_quality_rating(r),
            "Rating must be between 1 and 5",
        );
    }

    #[test]
    fn seed_quality_rating_one_valid() {
        let mut r = valid_seed_quality_rating();
        r.rating = 1;
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_five_valid() {
        let mut r = valid_seed_quality_rating();
        r.rating = 5;
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_zero_rated_at_rejected() {
        let mut r = valid_seed_quality_rating();
        r.rated_at = 0;
        assert_invalid(
            validate_seed_quality_rating(r),
            "SeedQualityRating rated_at cannot be zero",
        );
    }

    #[test]
    fn seed_quality_rating_negative_germination_rejected() {
        let mut r = valid_seed_quality_rating();
        r.germination_observed_pct = Some(-0.1);
        assert_invalid(
            validate_seed_quality_rating(r),
            "Observed germination rate must be between 0 and 100",
        );
    }

    #[test]
    fn seed_quality_rating_germination_over_100_rejected() {
        let mut r = valid_seed_quality_rating();
        r.germination_observed_pct = Some(100.1);
        assert_invalid(
            validate_seed_quality_rating(r),
            "Observed germination rate must be between 0 and 100",
        );
    }

    #[test]
    fn seed_quality_rating_germination_zero_valid() {
        let mut r = valid_seed_quality_rating();
        r.germination_observed_pct = Some(0.0);
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_germination_100_valid() {
        let mut r = valid_seed_quality_rating();
        r.germination_observed_pct = Some(100.0);
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_germination_none_valid() {
        let mut r = valid_seed_quality_rating();
        r.germination_observed_pct = None;
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_comment_too_long_rejected() {
        let mut r = valid_seed_quality_rating();
        r.comment = Some("x".repeat(2049));
        assert_invalid(
            validate_seed_quality_rating(r),
            "Rating comment too long (max 2048 chars)",
        );
    }

    #[test]
    fn seed_quality_rating_comment_at_max_valid() {
        let mut r = valid_seed_quality_rating();
        r.comment = Some("x".repeat(2048));
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_comment_none_valid() {
        let mut r = valid_seed_quality_rating();
        r.comment = None;
        assert_valid(validate_seed_quality_rating(r));
    }

    #[test]
    fn seed_quality_rating_all_ratings_valid() {
        for rating in 1..=5 {
            let mut r = valid_seed_quality_rating();
            r.rating = rating;
            assert_valid(validate_seed_quality_rating(r));
        }
    }
}
