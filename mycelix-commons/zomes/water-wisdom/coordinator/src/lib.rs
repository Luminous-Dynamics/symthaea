// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wisdom Coordinator Zome
//! Business logic for traditional water knowledge, conservation, and climate patterns

use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;
use water_wisdom_integrity::*;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

// ============================================================================
// TRADITIONAL PRACTICES
// ============================================================================

/// Record a traditional water management practice
#[hdk_extern]
pub fn record_practice(practice: TraditionalPractice) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_practice")?;
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
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "share_conservation_method")?;
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
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_climate_pattern")?;
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
    records.sort_by_key(|a| a.action().timestamp());
    Ok(records)
}

// ============================================================================
// UPDATE FUNCTIONS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateTraditionalPracticeInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: TraditionalPractice,
}

/// Update a traditional water management practice
#[hdk_extern]
pub fn update_practice(input: UpdateTraditionalPracticeInput) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_practice")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::TraditionalPractice(input.updated_entry),
    )
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateConservationMethodInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: ConservationMethod,
}

/// Update a conservation method
#[hdk_extern]
pub fn update_conservation_method(
    input: UpdateConservationMethodInput,
) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_conservation_method")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::ConservationMethod(input.updated_entry),
    )
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateClimateWaterPatternInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: ClimateWaterPattern,
}

/// Update an observed climate-water pattern
#[hdk_extern]
pub fn update_climate_pattern(input: UpdateClimateWaterPatternInput) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_climate_pattern")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::ClimateWaterPattern(input.updated_entry),
    )
}

// ============================================================================
// QUERIES (continued)
// ============================================================================

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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // INTEGRITY ENUM SERDE ROUNDTRIP TESTS (via coordinator re-export)
    // ========================================================================

    #[test]
    fn water_classification_all_variants_serde() {
        for variant in [
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: WaterClassification = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn practice_type_all_variants_serde() {
        for variant in [
            PracticeType::Irrigation,
            PracticeType::Conservation,
            PracticeType::Purification,
            PracticeType::Harvesting,
            PracticeType::Divining,
            PracticeType::Ceremony,
            PracticeType::Seasonal,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: PracticeType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn access_level_all_variants_serde() {
        for variant in [
            AccessLevel::Public,
            AccessLevel::CommunityOnly,
            AccessLevel::ElderApproved,
            AccessLevel::Sacred,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: AccessLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn cost_level_all_variants_serde() {
        for variant in [
            CostLevel::Free,
            CostLevel::Low,
            CostLevel::Medium,
            CostLevel::High,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: CostLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn difficulty_level_all_variants_serde() {
        for variant in [
            DifficultyLevel::Beginner,
            DifficultyLevel::Intermediate,
            DifficultyLevel::Advanced,
            DifficultyLevel::Expert,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: DifficultyLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn pattern_type_all_variants_serde() {
        for variant in [
            PatternType::Drought,
            PatternType::Flood,
            PatternType::SeasonalShift,
            PatternType::QualityChange,
            PatternType::LevelChange,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: PatternType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    // ========================================================================
    // STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn traditional_practice_serde_roundtrip() {
        let practice = TraditionalPractice {
            id: "tp-001".to_string(),
            title: "Qanat System".to_string(),
            description: "Ancient underground water channel system for irrigation".to_string(),
            practice_type: PracticeType::Irrigation,
            region: "Middle East".to_string(),
            culture_or_community: "Persian".to_string(),
            recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            access_level: AccessLevel::Public,
            effectiveness_rating: Some(8),
        };
        let json = serde_json::to_string(&practice).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "tp-001");
        assert_eq!(back.title, "Qanat System");
        assert_eq!(back.practice_type, PracticeType::Irrigation);
        assert_eq!(back.region, "Middle East");
        assert_eq!(back.culture_or_community, "Persian");
        assert_eq!(back.access_level, AccessLevel::Public);
        assert_eq!(back.effectiveness_rating, Some(8));
    }

    #[test]
    fn traditional_practice_no_rating_serde() {
        let practice = TraditionalPractice {
            id: "tp-002".to_string(),
            title: "Rain Ceremony".to_string(),
            description: "Traditional ceremony to invite rain".to_string(),
            practice_type: PracticeType::Ceremony,
            region: "Southern Africa".to_string(),
            culture_or_community: "Zulu".to_string(),
            recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            access_level: AccessLevel::Sacred,
            effectiveness_rating: None,
        };
        let json = serde_json::to_string(&practice).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert!(back.effectiveness_rating.is_none());
        assert_eq!(back.access_level, AccessLevel::Sacred);
    }

    #[test]
    fn traditional_practice_clone_is_equal() {
        let practice = TraditionalPractice {
            id: "tp-003".to_string(),
            title: "Fogwater Collection".to_string(),
            description: "Using mesh nets to capture fog".to_string(),
            practice_type: PracticeType::Harvesting,
            region: "Atacama Desert".to_string(),
            culture_or_community: "Chilean coastal communities".to_string(),
            recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            access_level: AccessLevel::CommunityOnly,
            effectiveness_rating: Some(7),
        };
        let cloned = practice.clone();
        assert_eq!(practice, cloned);
    }

    #[test]
    fn conservation_method_serde_roundtrip() {
        let method = ConservationMethod {
            id: "cm-001".to_string(),
            title: "Drip Irrigation".to_string(),
            description: "Low-volume water delivery directly to plant roots".to_string(),
            water_saved_percent: Some(60),
            applicable_to: vec![
                WaterClassification::Irrigation,
                WaterClassification::Cooking,
            ],
            cost_level: CostLevel::Medium,
            difficulty: DifficultyLevel::Intermediate,
        };
        let json = serde_json::to_string(&method).unwrap();
        let back: ConservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "cm-001");
        assert_eq!(back.title, "Drip Irrigation");
        assert_eq!(back.water_saved_percent, Some(60));
        assert_eq!(back.applicable_to.len(), 2);
        assert_eq!(back.cost_level, CostLevel::Medium);
        assert_eq!(back.difficulty, DifficultyLevel::Intermediate);
    }

    #[test]
    fn conservation_method_no_savings_serde() {
        let method = ConservationMethod {
            id: "cm-002".to_string(),
            title: "Water Quality Monitoring".to_string(),
            description: "Regular testing of water sources".to_string(),
            water_saved_percent: None,
            applicable_to: vec![WaterClassification::Potable],
            cost_level: CostLevel::Low,
            difficulty: DifficultyLevel::Beginner,
        };
        let json = serde_json::to_string(&method).unwrap();
        let back: ConservationMethod = serde_json::from_str(&json).unwrap();
        assert!(back.water_saved_percent.is_none());
        assert_eq!(back.applicable_to.len(), 1);
    }

    #[test]
    fn conservation_method_clone_is_equal() {
        let method = ConservationMethod {
            id: "cm-003".to_string(),
            title: "Rainwater Harvesting".to_string(),
            description: "Collecting and storing rainwater from rooftops".to_string(),
            water_saved_percent: Some(40),
            applicable_to: vec![
                WaterClassification::Irrigation,
                WaterClassification::Hygiene,
                WaterClassification::Greywater,
            ],
            cost_level: CostLevel::Low,
            difficulty: DifficultyLevel::Beginner,
        };
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    #[test]
    fn climate_water_pattern_serde_roundtrip() {
        let pattern = ClimateWaterPattern {
            region: "Southwest US".to_string(),
            season: "Summer".to_string(),
            pattern_type: PatternType::Drought,
            description: "Extended drought conditions with below-average rainfall".to_string(),
            observed_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            observed_at: Timestamp::from_micros(1_700_000_000),
            indicators: vec![
                "Reduced streamflow".to_string(),
                "Lowered aquifer levels".to_string(),
                "Increased wildfire risk".to_string(),
            ],
        };
        let json = serde_json::to_string(&pattern).unwrap();
        let back: ClimateWaterPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(back.region, "Southwest US");
        assert_eq!(back.season, "Summer");
        assert_eq!(back.pattern_type, PatternType::Drought);
        assert_eq!(back.indicators.len(), 3);
        assert!(back
            .indicators
            .contains(&"Increased wildfire risk".to_string()));
    }

    #[test]
    fn climate_water_pattern_no_indicators_serde() {
        let pattern = ClimateWaterPattern {
            region: "Northern Europe".to_string(),
            season: "Autumn".to_string(),
            pattern_type: PatternType::SeasonalShift,
            description: "Delayed onset of autumn rains".to_string(),
            observed_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            observed_at: Timestamp::from_micros(1_600_000_000),
            indicators: vec![],
        };
        let json = serde_json::to_string(&pattern).unwrap();
        let back: ClimateWaterPattern = serde_json::from_str(&json).unwrap();
        assert!(back.indicators.is_empty());
        assert_eq!(back.pattern_type, PatternType::SeasonalShift);
    }

    #[test]
    fn climate_water_pattern_clone_is_equal() {
        let pattern = ClimateWaterPattern {
            region: "Mekong Delta".to_string(),
            season: "Monsoon".to_string(),
            pattern_type: PatternType::Flood,
            description: "Severe monsoon flooding above historical norms".to_string(),
            observed_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            observed_at: Timestamp::from_micros(1_500_000_000),
            indicators: vec!["Rising river levels".to_string()],
        };
        let cloned = pattern.clone();
        assert_eq!(pattern, cloned);
    }

    #[test]
    fn traditional_practice_unicode_fields_serde() {
        let practice = TraditionalPractice {
            id: "tp-unicode".to_string(),
            title: "\u{0642}\u{0646}\u{0627}\u{0629}".to_string(),
            description: "\u{53e4}\u{4ee3}\u{306e}\u{6c34}\u{7ba1}\u{7406}\u{6280}\u{8853}"
                .to_string(),
            practice_type: PracticeType::Irrigation,
            region: "\u{4e2d}\u{4e1c}".to_string(),
            culture_or_community: "\u{0641}\u{0627}\u{0631}\u{0633}\u{06cc}".to_string(),
            recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            access_level: AccessLevel::ElderApproved,
            effectiveness_rating: Some(9),
        };
        let json = serde_json::to_string(&practice).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(practice, back);
    }

    #[test]
    fn conservation_method_all_classifications_serde() {
        let method = ConservationMethod {
            id: "cm-all".to_string(),
            title: "Comprehensive Water Reuse".to_string(),
            description: "System covering all water classifications".to_string(),
            water_saved_percent: Some(80),
            applicable_to: vec![
                WaterClassification::Potable,
                WaterClassification::Cooking,
                WaterClassification::Hygiene,
                WaterClassification::Irrigation,
                WaterClassification::Industrial,
                WaterClassification::Recreation,
                WaterClassification::Greywater,
            ],
            cost_level: CostLevel::High,
            difficulty: DifficultyLevel::Expert,
        };
        let json = serde_json::to_string(&method).unwrap();
        let back: ConservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(back.applicable_to.len(), 7);
        assert_eq!(back.cost_level, CostLevel::High);
        assert_eq!(back.difficulty, DifficultyLevel::Expert);
    }

    #[test]
    fn anchor_serde_roundtrip() {
        let anchor = water_wisdom_integrity::Anchor("all_practices".to_string());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: water_wisdom_integrity::Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn climate_water_pattern_many_indicators_serde() {
        let pattern = ClimateWaterPattern {
            region: "Amazon Basin".to_string(),
            season: "Dry Season".to_string(),
            pattern_type: PatternType::LevelChange,
            description: "Unprecedented low water levels across tributaries".to_string(),
            observed_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            observed_at: Timestamp::from_micros(1_800_000_000),
            indicators: (0..20)
                .map(|i| format!("Indicator {}: measurement data", i))
                .collect(),
        };
        let json = serde_json::to_string(&pattern).unwrap();
        let back: ClimateWaterPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(back.indicators.len(), 20);
        assert_eq!(back.pattern_type, PatternType::LevelChange);
    }

    // ========================================================================
    // BOUNDARY AND EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn traditional_practice_max_effectiveness_rating_serde() {
        let practice = TraditionalPractice {
            id: "tp-max".to_string(),
            title: "Perfect technique".to_string(),
            description: "Achieves maximum effectiveness".to_string(),
            practice_type: PracticeType::Conservation,
            region: "Global".to_string(),
            culture_or_community: "Universal".to_string(),
            recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
            access_level: AccessLevel::Public,
            effectiveness_rating: Some(u8::MAX),
        };
        let json = serde_json::to_string(&practice).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(back.effectiveness_rating, Some(u8::MAX));
        assert_eq!(back.practice_type, PracticeType::Conservation);
    }

    #[test]
    fn conservation_method_empty_applicable_to_serde() {
        let method = ConservationMethod {
            id: "cm-empty".to_string(),
            title: "Theoretical Method".to_string(),
            description: "Not yet applied to any classification".to_string(),
            water_saved_percent: Some(0),
            applicable_to: vec![],
            cost_level: CostLevel::Free,
            difficulty: DifficultyLevel::Beginner,
        };
        let json = serde_json::to_string(&method).unwrap();
        let back: ConservationMethod = serde_json::from_str(&json).unwrap();
        assert!(back.applicable_to.is_empty());
        assert_eq!(back.water_saved_percent, Some(0));
    }

    #[test]
    fn climate_water_pattern_quality_change_type_serde() {
        let pattern = ClimateWaterPattern {
            region: "Great Lakes".to_string(),
            season: "Spring".to_string(),
            pattern_type: PatternType::QualityChange,
            description: "Algal bloom affecting water quality in nearshore areas".to_string(),
            observed_by: AgentPubKey::from_raw_36(vec![0xcd; 36]),
            observed_at: Timestamp::from_micros(1_750_000_000),
            indicators: vec![
                "Elevated phosphorus".to_string(),
                "Cyanobacteria detected".to_string(),
            ],
        };
        let json = serde_json::to_string(&pattern).unwrap();
        let back: ClimateWaterPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pattern_type, PatternType::QualityChange);
        assert_eq!(back.region, "Great Lakes");
        assert_eq!(back.indicators.len(), 2);
    }

    #[test]
    fn conservation_method_max_water_saved_percent_serde() {
        let method = ConservationMethod {
            id: "cm-max".to_string(),
            title: "Perfect Conservation".to_string(),
            description: "Saves maximum percentage of water".to_string(),
            water_saved_percent: Some(u8::MAX),
            applicable_to: vec![WaterClassification::Irrigation],
            cost_level: CostLevel::High,
            difficulty: DifficultyLevel::Expert,
        };
        let json = serde_json::to_string(&method).unwrap();
        let back: ConservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(back.water_saved_percent, Some(u8::MAX));
        assert_eq!(back.difficulty, DifficultyLevel::Expert);
    }

    // ========================================================================
    // UpdateTraditionalPracticeInput tests
    // ========================================================================

    #[test]
    fn update_traditional_practice_input_struct_construction() {
        let input = UpdateTraditionalPracticeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: TraditionalPractice {
                id: "tp-updated".to_string(),
                title: "Updated Qanat System".to_string(),
                description: "Improved underground water channel system".to_string(),
                practice_type: PracticeType::Irrigation,
                region: "Middle East".to_string(),
                culture_or_community: "Persian".to_string(),
                recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
                access_level: AccessLevel::Public,
                effectiveness_rating: Some(9),
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.title, "Updated Qanat System");
        assert_eq!(input.updated_entry.effectiveness_rating, Some(9));
    }

    #[test]
    fn update_traditional_practice_input_serde_roundtrip() {
        let input = UpdateTraditionalPracticeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xef; 36]),
            updated_entry: TraditionalPractice {
                id: "tp-serde".to_string(),
                title: "Rain Ceremony".to_string(),
                description: "Traditional ceremony to invite rain".to_string(),
                practice_type: PracticeType::Ceremony,
                region: "Southern Africa".to_string(),
                culture_or_community: "Zulu".to_string(),
                recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
                access_level: AccessLevel::Sacred,
                effectiveness_rating: None,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTraditionalPracticeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.title, "Rain Ceremony");
        assert_eq!(decoded.updated_entry.access_level, AccessLevel::Sacred);
        assert!(decoded.updated_entry.effectiveness_rating.is_none());
    }

    #[test]
    fn update_traditional_practice_input_clone_is_equal() {
        let input = UpdateTraditionalPracticeInput {
            original_action_hash: ActionHash::from_raw_36(vec![0x11; 36]),
            updated_entry: TraditionalPractice {
                id: "tp-clone".to_string(),
                title: "Fog Collection".to_string(),
                description: "Using nets to collect fog".to_string(),
                practice_type: PracticeType::Harvesting,
                region: "Chile".to_string(),
                culture_or_community: "Coastal".to_string(),
                recorded_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
                access_level: AccessLevel::CommunityOnly,
                effectiveness_rating: Some(7),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }

    // ========================================================================
    // UpdateConservationMethodInput tests
    // ========================================================================

    #[test]
    fn update_conservation_method_input_struct_construction() {
        let input = UpdateConservationMethodInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: ConservationMethod {
                id: "cm-updated".to_string(),
                title: "Updated Drip Irrigation".to_string(),
                description: "Improved low-volume water delivery".to_string(),
                water_saved_percent: Some(70),
                applicable_to: vec![WaterClassification::Irrigation],
                cost_level: CostLevel::Medium,
                difficulty: DifficultyLevel::Intermediate,
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.title, "Updated Drip Irrigation");
        assert_eq!(input.updated_entry.water_saved_percent, Some(70));
    }

    #[test]
    fn update_conservation_method_input_serde_roundtrip() {
        let input = UpdateConservationMethodInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xaa; 36]),
            updated_entry: ConservationMethod {
                id: "cm-serde".to_string(),
                title: "Water Quality Monitoring".to_string(),
                description: "Regular testing of water sources".to_string(),
                water_saved_percent: None,
                applicable_to: vec![WaterClassification::Potable, WaterClassification::Cooking],
                cost_level: CostLevel::Low,
                difficulty: DifficultyLevel::Beginner,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateConservationMethodInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.title, "Water Quality Monitoring");
        assert!(decoded.updated_entry.water_saved_percent.is_none());
        assert_eq!(decoded.updated_entry.applicable_to.len(), 2);
    }

    #[test]
    fn update_conservation_method_input_clone_is_equal() {
        let input = UpdateConservationMethodInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xbb; 36]),
            updated_entry: ConservationMethod {
                id: "cm-clone".to_string(),
                title: "Rainwater Harvesting".to_string(),
                description: "Collecting rainwater".to_string(),
                water_saved_percent: Some(40),
                applicable_to: vec![WaterClassification::Greywater],
                cost_level: CostLevel::Free,
                difficulty: DifficultyLevel::Beginner,
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }

    // ========================================================================
    // UpdateClimateWaterPatternInput tests
    // ========================================================================

    #[test]
    fn update_climate_water_pattern_input_struct_construction() {
        let input = UpdateClimateWaterPatternInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: ClimateWaterPattern {
                region: "Southwest US".to_string(),
                season: "Summer".to_string(),
                pattern_type: PatternType::Drought,
                description: "Updated drought assessment".to_string(),
                observed_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
                observed_at: Timestamp::from_micros(1_700_000_000),
                indicators: vec!["Reduced streamflow".to_string(), "Low aquifer".to_string()],
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.region, "Southwest US");
        assert_eq!(input.updated_entry.indicators.len(), 2);
    }

    #[test]
    fn update_climate_water_pattern_input_serde_roundtrip() {
        let input = UpdateClimateWaterPatternInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xcc; 36]),
            updated_entry: ClimateWaterPattern {
                region: "Mekong Delta".to_string(),
                season: "Monsoon".to_string(),
                pattern_type: PatternType::Flood,
                description: "Severe monsoon flooding".to_string(),
                observed_by: AgentPubKey::from_raw_36(vec![0xcd; 36]),
                observed_at: Timestamp::from_micros(1_500_000_000),
                indicators: vec![],
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateClimateWaterPatternInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.region, "Mekong Delta");
        assert_eq!(decoded.updated_entry.pattern_type, PatternType::Flood);
        assert!(decoded.updated_entry.indicators.is_empty());
    }

    #[test]
    fn update_climate_water_pattern_input_clone_is_equal() {
        let input = UpdateClimateWaterPatternInput {
            original_action_hash: ActionHash::from_raw_36(vec![0x33; 36]),
            updated_entry: ClimateWaterPattern {
                region: "Great Lakes".to_string(),
                season: "Spring".to_string(),
                pattern_type: PatternType::QualityChange,
                description: "Algal bloom".to_string(),
                observed_by: AgentPubKey::from_raw_36(vec![0xab; 36]),
                observed_at: Timestamp::from_micros(1_750_000_000),
                indicators: vec!["Elevated phosphorus".to_string()],
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry, input.updated_entry);
    }
}
