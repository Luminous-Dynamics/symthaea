// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wisdom Integrity Zome
//! Traditional water knowledge, conservation methods, and climate patterns

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// WATER CLASSIFICATION (shared type for cross-zome reference)
// ============================================================================

/// Classification of water use
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WaterClassification {
    Potable,
    Cooking,
    Hygiene,
    Irrigation,
    Industrial,
    Recreation,
    Greywater,
}

// ============================================================================
// TRADITIONAL PRACTICES
// ============================================================================

/// Type of traditional water practice
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PracticeType {
    Irrigation,
    Conservation,
    Purification,
    Harvesting,
    Divining,
    Ceremony,
    Seasonal,
}

/// Access level for traditional knowledge
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AccessLevel {
    /// Visible to everyone
    Public,
    /// Visible only to community members
    CommunityOnly,
    /// Requires elder approval to view
    ElderApproved,
    /// Sacred knowledge, highest restriction
    Sacred,
}

/// A traditional water management practice
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TraditionalPractice {
    /// Unique identifier
    pub id: String,
    /// Title of the practice
    pub title: String,
    /// Full description
    pub description: String,
    /// Type of practice
    pub practice_type: PracticeType,
    /// Geographic region where this practice originates
    pub region: String,
    /// Culture or community that developed this practice
    pub culture_or_community: String,
    /// Agent who recorded this practice
    pub recorded_by: AgentPubKey,
    /// Access restrictions
    pub access_level: AccessLevel,
    /// Effectiveness rating (1-10, optional)
    pub effectiveness_rating: Option<u8>,
}

// ============================================================================
// CONSERVATION METHODS
// ============================================================================

/// Cost level for implementing a conservation method
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CostLevel {
    Free,
    Low,
    Medium,
    High,
}

/// Difficulty level for implementing a conservation method
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// A water conservation method that can be shared and adopted
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConservationMethod {
    /// Unique identifier
    pub id: String,
    /// Title of the method
    pub title: String,
    /// Full description with instructions
    pub description: String,
    /// Estimated water savings percentage (0-100)
    pub water_saved_percent: Option<u8>,
    /// Water use categories this method applies to
    pub applicable_to: Vec<WaterClassification>,
    /// Cost to implement
    pub cost_level: CostLevel,
    /// Difficulty to implement
    pub difficulty: DifficultyLevel,
}

// ============================================================================
// CLIMATE WATER PATTERNS
// ============================================================================

/// Type of climate-related water pattern
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PatternType {
    Drought,
    Flood,
    SeasonalShift,
    QualityChange,
    LevelChange,
}

/// An observed climate-related water pattern
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClimateWaterPattern {
    /// Geographic region
    pub region: String,
    /// Season when observed
    pub season: String,
    /// Type of pattern
    pub pattern_type: PatternType,
    /// Description of the observation
    pub description: String,
    /// Agent who made the observation
    pub observed_by: AgentPubKey,
    /// When the observation was made
    pub observed_at: Timestamp,
    /// Environmental indicators noted
    pub indicators: Vec<String>,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    TraditionalPractice(TraditionalPractice),
    ConservationMethod(ConservationMethod),
    ClimateWaterPattern(ClimateWaterPattern),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all practices
    AllPractices,
    /// Practice type to practices
    PracticeTypeToEntry,
    /// Public practices anchor
    PublicPractices,
    /// Recorder to their practices
    RecorderToPractice,
    /// Anchor to all conservation methods
    AllConservationMethods,
    /// Cost level to conservation methods
    CostLevelToMethod,
    /// Anchor to all climate patterns
    AllClimatePatterns,
    /// Region to climate patterns
    RegionToPattern,
    /// Pattern type to patterns
    PatternTypeToPattern,
    /// Observer to their patterns
    ObserverToPattern,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TraditionalPractice(practice) => {
                    validate_create_practice(action, practice)
                }
                EntryTypes::ConservationMethod(method) => {
                    validate_create_conservation_method(action, method)
                }
                EntryTypes::ClimateWaterPattern(pattern) => {
                    validate_create_climate_pattern(action, pattern)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TraditionalPractice(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ConservationMethod(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ClimateWaterPattern(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllPractices => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PracticeTypeToEntry => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicPractices => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecorderToPractice => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllConservationMethods => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CostLevelToMethod => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllClimatePatterns => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RegionToPattern => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PatternTypeToPattern => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ObserverToPattern => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_practice(
    _action: Create,
    practice: TraditionalPractice,
) -> ExternResult<ValidateCallbackResult> {
    if practice.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice ID cannot be empty".into(),
        ));
    }
    if practice.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice ID must be 256 characters or fewer".into(),
        ));
    }
    if practice.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice title cannot be empty".into(),
        ));
    }
    if practice.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice title must be 512 characters or fewer".into(),
        ));
    }
    if practice.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice description cannot be empty".into(),
        ));
    }
    if practice.description.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice description must be 16384 characters or fewer".into(),
        ));
    }
    if practice.region.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Region cannot be empty".into(),
        ));
    }
    if practice.region.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Region must be 512 characters or fewer".into(),
        ));
    }
    if practice.culture_or_community.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Culture or community cannot be empty".into(),
        ));
    }
    if practice.culture_or_community.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Culture or community must be 512 characters or fewer".into(),
        ));
    }
    if let Some(rating) = practice.effectiveness_rating {
        if rating == 0 || rating > 10 {
            return Ok(ValidateCallbackResult::Invalid(
                "Effectiveness rating must be between 1 and 10".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_conservation_method(
    _action: Create,
    method: ConservationMethod,
) -> ExternResult<ValidateCallbackResult> {
    if method.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method ID cannot be empty".into(),
        ));
    }
    if method.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method ID must be 256 characters or fewer".into(),
        ));
    }
    if method.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method title cannot be empty".into(),
        ));
    }
    if method.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method title must be 256 characters or fewer".into(),
        ));
    }
    if method.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method description cannot be empty".into(),
        ));
    }
    if method.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method description must be 4096 characters or fewer".into(),
        ));
    }
    if let Some(pct) = method.water_saved_percent {
        if pct > 100 {
            return Ok(ValidateCallbackResult::Invalid(
                "Water saved percent cannot exceed 100".into(),
            ));
        }
    }
    if method.applicable_to.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Must specify at least one applicable water classification".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_climate_pattern(
    _action: Create,
    pattern: ClimateWaterPattern,
) -> ExternResult<ValidateCallbackResult> {
    if pattern.region.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Region cannot be empty".into(),
        ));
    }
    if pattern.region.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Region must be 256 characters or fewer".into(),
        ));
    }
    if pattern.season.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Season cannot be empty".into(),
        ));
    }
    if pattern.season.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Season must be 128 characters or fewer".into(),
        ));
    }
    if pattern.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Description cannot be empty".into(),
        ));
    }
    if pattern.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be 4096 characters or fewer".into(),
        ));
    }
    if pattern.indicators.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 indicators".into(),
        ));
    }
    for indicator in &pattern.indicators {
        if indicator.trim().is_empty() || indicator.len() > 512 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each indicator must be 1-512 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    fn make_practice() -> TraditionalPractice {
        TraditionalPractice {
            id: "tp-001".into(),
            title: "Qanat System".into(),
            description: "Ancient underground water channel system".into(),
            practice_type: PracticeType::Irrigation,
            region: "Middle East".into(),
            culture_or_community: "Persian".into(),
            recorded_by: fake_agent(),
            access_level: AccessLevel::Public,
            effectiveness_rating: Some(8),
        }
    }

    fn make_conservation_method() -> ConservationMethod {
        ConservationMethod {
            id: "cm-001".into(),
            title: "Drip Irrigation".into(),
            description: "Low-volume water delivery directly to roots".into(),
            water_saved_percent: Some(60),
            applicable_to: vec![WaterClassification::Irrigation],
            cost_level: CostLevel::Medium,
            difficulty: DifficultyLevel::Intermediate,
        }
    }

    fn make_climate_pattern() -> ClimateWaterPattern {
        ClimateWaterPattern {
            region: "Southwest US".into(),
            season: "Summer".into(),
            pattern_type: PatternType::Drought,
            description: "Extended drought conditions with below-average rainfall".into(),
            observed_by: fake_agent(),
            observed_at: Timestamp::from_micros(1_000_000),
            indicators: vec!["Reduced streamflow".into(), "Lowered aquifer levels".into()],
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_water_classification() {
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
    fn serde_roundtrip_practice_type() {
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
    fn serde_roundtrip_access_level() {
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
    fn serde_roundtrip_cost_level() {
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
    fn serde_roundtrip_difficulty_level() {
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
    fn serde_roundtrip_pattern_type() {
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

    #[test]
    fn serde_roundtrip_traditional_practice() {
        let practice = make_practice();
        let json = serde_json::to_string(&practice).unwrap();
        let back: TraditionalPractice = serde_json::from_str(&json).unwrap();
        assert_eq!(practice, back);
    }

    #[test]
    fn serde_roundtrip_conservation_method() {
        let method = make_conservation_method();
        let json = serde_json::to_string(&method).unwrap();
        let back: ConservationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(method, back);
    }

    #[test]
    fn serde_roundtrip_climate_water_pattern() {
        let pattern = make_climate_pattern();
        let json = serde_json::to_string(&pattern).unwrap();
        let back: ClimateWaterPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(pattern, back);
    }

    #[test]
    fn serde_roundtrip_anchor() {
        let anchor = Anchor("water-wisdom".into());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(anchor, back);
    }

    // ========================================================================
    // TRADITIONAL PRACTICE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_practice_passes() {
        let result = validate_create_practice(fake_create(), make_practice());
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_empty_id_rejected() {
        let mut p = make_practice();
        p.id = "".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Practice ID cannot be empty");
    }

    #[test]
    fn practice_empty_title_rejected() {
        let mut p = make_practice();
        p.title = "".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Practice title cannot be empty");
    }

    #[test]
    fn practice_empty_description_rejected() {
        let mut p = make_practice();
        p.description = "".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Practice description cannot be empty");
    }

    #[test]
    fn practice_empty_region_rejected() {
        let mut p = make_practice();
        p.region = "".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Region cannot be empty");
    }

    #[test]
    fn practice_empty_culture_rejected() {
        let mut p = make_practice();
        p.culture_or_community = "".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Culture or community cannot be empty");
    }

    #[test]
    fn practice_effectiveness_rating_zero_rejected() {
        let mut p = make_practice();
        p.effectiveness_rating = Some(0);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Effectiveness rating must be between 1 and 10"
        );
    }

    #[test]
    fn practice_effectiveness_rating_11_rejected() {
        let mut p = make_practice();
        p.effectiveness_rating = Some(11);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Effectiveness rating must be between 1 and 10"
        );
    }

    #[test]
    fn practice_effectiveness_rating_255_rejected() {
        let mut p = make_practice();
        p.effectiveness_rating = Some(255);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Effectiveness rating must be between 1 and 10"
        );
    }

    #[test]
    fn practice_effectiveness_rating_1_accepted() {
        let mut p = make_practice();
        p.effectiveness_rating = Some(1);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_effectiveness_rating_10_accepted() {
        let mut p = make_practice();
        p.effectiveness_rating = Some(10);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_effectiveness_rating_5_accepted() {
        let mut p = make_practice();
        p.effectiveness_rating = Some(5);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_effectiveness_rating_none_accepted() {
        let mut p = make_practice();
        p.effectiveness_rating = None;
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_all_practice_types_accepted() {
        for practice_type in [
            PracticeType::Irrigation,
            PracticeType::Conservation,
            PracticeType::Purification,
            PracticeType::Harvesting,
            PracticeType::Divining,
            PracticeType::Ceremony,
            PracticeType::Seasonal,
        ] {
            let mut p = make_practice();
            p.practice_type = practice_type;
            let result = validate_create_practice(fake_create(), p);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn practice_all_access_levels_accepted() {
        for access_level in [
            AccessLevel::Public,
            AccessLevel::CommunityOnly,
            AccessLevel::ElderApproved,
            AccessLevel::Sacred,
        ] {
            let mut p = make_practice();
            p.access_level = access_level;
            let result = validate_create_practice(fake_create(), p);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn practice_unicode_title_accepted() {
        let mut p = make_practice();
        p.title = "Qanat \u{0642}\u{0646}\u{0627}\u{0629}".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_unicode_region_accepted() {
        let mut p = make_practice();
        p.region = "\u{4e2d}\u{56fd}\u{4e91}\u{5357}".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_unicode_culture_accepted() {
        let mut p = make_practice();
        p.culture_or_community = "\u{0420}\u{0443}\u{0441}\u{0441}\u{043a}\u{0438}\u{0435}".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_long_description_accepted() {
        let mut p = make_practice();
        p.description = "A".repeat(10_000);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_whitespace_only_id_rejected() {
        let mut p = make_practice();
        p.id = "   ".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Practice ID cannot be empty");
    }

    // ========================================================================
    // CONSERVATION METHOD VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_conservation_method_passes() {
        let result = validate_create_conservation_method(fake_create(), make_conservation_method());
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_empty_id_rejected() {
        let mut m = make_conservation_method();
        m.id = "".into();
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Conservation method ID cannot be empty"
        );
    }

    #[test]
    fn conservation_method_empty_title_rejected() {
        let mut m = make_conservation_method();
        m.title = "".into();
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Conservation method title cannot be empty"
        );
    }

    #[test]
    fn conservation_method_empty_description_rejected() {
        let mut m = make_conservation_method();
        m.description = "".into();
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Conservation method description cannot be empty"
        );
    }

    #[test]
    fn conservation_method_water_saved_percent_101_rejected() {
        let mut m = make_conservation_method();
        m.water_saved_percent = Some(101);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Water saved percent cannot exceed 100"
        );
    }

    #[test]
    fn conservation_method_water_saved_percent_255_rejected() {
        let mut m = make_conservation_method();
        m.water_saved_percent = Some(255);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Water saved percent cannot exceed 100"
        );
    }

    #[test]
    fn conservation_method_water_saved_percent_100_accepted() {
        let mut m = make_conservation_method();
        m.water_saved_percent = Some(100);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_water_saved_percent_0_accepted() {
        let mut m = make_conservation_method();
        m.water_saved_percent = Some(0);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_water_saved_percent_50_accepted() {
        let mut m = make_conservation_method();
        m.water_saved_percent = Some(50);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_water_saved_percent_none_accepted() {
        let mut m = make_conservation_method();
        m.water_saved_percent = None;
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_empty_applicable_to_rejected() {
        let mut m = make_conservation_method();
        m.applicable_to = vec![];
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Must specify at least one applicable water classification"
        );
    }

    #[test]
    fn conservation_method_single_applicable_to_accepted() {
        let mut m = make_conservation_method();
        m.applicable_to = vec![WaterClassification::Potable];
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_all_classifications_accepted() {
        let mut m = make_conservation_method();
        m.applicable_to = vec![
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ];
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_duplicate_classifications_accepted() {
        // Validation does not check for duplicates
        let mut m = make_conservation_method();
        m.applicable_to = vec![WaterClassification::Potable, WaterClassification::Potable];
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_all_cost_levels_accepted() {
        for cost_level in [
            CostLevel::Free,
            CostLevel::Low,
            CostLevel::Medium,
            CostLevel::High,
        ] {
            let mut m = make_conservation_method();
            m.cost_level = cost_level;
            let result = validate_create_conservation_method(fake_create(), m);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn conservation_method_all_difficulty_levels_accepted() {
        for difficulty in [
            DifficultyLevel::Beginner,
            DifficultyLevel::Intermediate,
            DifficultyLevel::Advanced,
            DifficultyLevel::Expert,
        ] {
            let mut m = make_conservation_method();
            m.difficulty = difficulty;
            let result = validate_create_conservation_method(fake_create(), m);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn conservation_method_unicode_title_accepted() {
        let mut m = make_conservation_method();
        m.title = "\u{6ef4}\u{704c}\u{7cfb}\u{7edf}".into();
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_long_description_accepted() {
        let mut m = make_conservation_method();
        m.description = "B".repeat(4096);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // CLIMATE WATER PATTERN VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_climate_pattern_passes() {
        let result = validate_create_climate_pattern(fake_create(), make_climate_pattern());
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_empty_region_rejected() {
        let mut cp = make_climate_pattern();
        cp.region = "".into();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Region cannot be empty");
    }

    #[test]
    fn climate_pattern_empty_season_rejected() {
        let mut cp = make_climate_pattern();
        cp.season = "".into();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Season cannot be empty");
    }

    #[test]
    fn climate_pattern_empty_description_rejected() {
        let mut cp = make_climate_pattern();
        cp.description = "".into();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Description cannot be empty");
    }

    #[test]
    fn climate_pattern_51_indicators_rejected() {
        let mut cp = make_climate_pattern();
        cp.indicators = (0..51).map(|i| format!("Indicator {}", i)).collect();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Cannot have more than 50 indicators");
    }

    #[test]
    fn climate_pattern_50_indicators_accepted() {
        let mut cp = make_climate_pattern();
        cp.indicators = (0..50).map(|i| format!("Indicator {}", i)).collect();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_zero_indicators_accepted() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec![];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_one_indicator_accepted() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["Soil moisture".into()];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_empty_indicator_string_rejected() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["Valid indicator".into(), "".into()];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Each indicator must be 1-512 characters"
        );
    }

    #[test]
    fn climate_pattern_first_indicator_empty_rejected() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["".into()];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Each indicator must be 1-512 characters"
        );
    }

    #[test]
    fn climate_pattern_indicator_513_chars_rejected() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["X".repeat(513)];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Each indicator must be 1-512 characters"
        );
    }

    #[test]
    fn climate_pattern_indicator_512_chars_accepted() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["X".repeat(512)];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_indicator_1_char_accepted() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["A".into()];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_indicator_over_512_in_middle_rejected() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec![
            "Valid".into(),
            "Also valid".into(),
            "X".repeat(513),
            "Would be valid".into(),
        ];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Each indicator must be 1-512 characters"
        );
    }

    #[test]
    fn climate_pattern_all_pattern_types_accepted() {
        for pattern_type in [
            PatternType::Drought,
            PatternType::Flood,
            PatternType::SeasonalShift,
            PatternType::QualityChange,
            PatternType::LevelChange,
        ] {
            let mut cp = make_climate_pattern();
            cp.pattern_type = pattern_type;
            let result = validate_create_climate_pattern(fake_create(), cp);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn climate_pattern_unicode_region_accepted() {
        let mut cp = make_climate_pattern();
        cp.region = "\u{0410}\u{043b}\u{0442}\u{0430}\u{0439}".into();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_unicode_season_accepted() {
        let mut cp = make_climate_pattern();
        cp.season = "\u{51ac}\u{5929}".into();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_unicode_indicator_accepted() {
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["\u{6c34}\u{4f4d}\u{4e0b}\u{964d}".into()];
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_unicode_indicator_length_checked_by_bytes() {
        // Multi-byte unicode: each CJK char is 3 bytes in UTF-8.
        // .len() returns byte count, so 171 CJK chars = 513 bytes.
        let mut cp = make_climate_pattern();
        let cjk_char = "\u{6c34}"; // 3 bytes
        cp.indicators = vec![cjk_char.repeat(171)]; // 513 bytes
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Each indicator must be 1-512 characters"
        );
    }

    #[test]
    fn climate_pattern_long_description_accepted() {
        let mut cp = make_climate_pattern();
        cp.description = "C".repeat(4096);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_100_indicators_rejected() {
        let mut cp = make_climate_pattern();
        cp.indicators = (0..100).map(|i| format!("Ind {}", i)).collect();
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Cannot have more than 50 indicators");
    }

    // ========================================================================
    // EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn practice_all_fields_minimal_valid() {
        let p = TraditionalPractice {
            id: "x".into(),
            title: "y".into(),
            description: "z".into(),
            practice_type: PracticeType::Ceremony,
            region: "a".into(),
            culture_or_community: "b".into(),
            recorded_by: fake_agent(),
            access_level: AccessLevel::Sacred,
            effectiveness_rating: None,
        };
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_all_fields_minimal_valid() {
        let m = ConservationMethod {
            id: "x".into(),
            title: "y".into(),
            description: "z".into(),
            water_saved_percent: None,
            applicable_to: vec![WaterClassification::Greywater],
            cost_level: CostLevel::Free,
            difficulty: DifficultyLevel::Beginner,
        };
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_all_fields_minimal_valid() {
        let cp = ClimateWaterPattern {
            region: "x".into(),
            season: "y".into(),
            pattern_type: PatternType::Flood,
            description: "z".into(),
            observed_by: fake_agent(),
            observed_at: Timestamp::from_micros(0),
            indicators: vec![],
        };
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_effectiveness_rating_boundary_1_and_10() {
        for rating in [1u8, 2, 5, 9, 10] {
            let mut p = make_practice();
            p.effectiveness_rating = Some(rating);
            let result = validate_create_practice(fake_create(), p);
            assert!(is_valid(&result), "Rating {} should be valid", rating);
        }
    }

    #[test]
    fn practice_effectiveness_rating_all_invalid_boundaries() {
        for rating in [0u8, 11, 50, 100, 255] {
            let mut p = make_practice();
            p.effectiveness_rating = Some(rating);
            let result = validate_create_practice(fake_create(), p);
            assert!(is_invalid(&result), "Rating {} should be invalid", rating);
        }
    }

    #[test]
    fn conservation_method_water_saved_all_valid_boundaries() {
        for pct in [0u8, 1, 50, 99, 100] {
            let mut m = make_conservation_method();
            m.water_saved_percent = Some(pct);
            let result = validate_create_conservation_method(fake_create(), m);
            assert!(is_valid(&result), "Percent {} should be valid", pct);
        }
    }

    #[test]
    fn conservation_method_water_saved_all_invalid_boundaries() {
        for pct in [101u8, 150, 200, 255] {
            let mut m = make_conservation_method();
            m.water_saved_percent = Some(pct);
            let result = validate_create_conservation_method(fake_create(), m);
            assert!(is_invalid(&result), "Percent {} should be invalid", pct);
        }
    }

    #[test]
    fn climate_pattern_indicator_boundary_lengths() {
        // Exactly 1 byte: valid
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["A".into()];
        assert!(is_valid(&validate_create_climate_pattern(
            fake_create(),
            cp
        )));

        // Exactly 512 bytes: valid
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["A".repeat(512)];
        assert!(is_valid(&validate_create_climate_pattern(
            fake_create(),
            cp
        )));

        // Exactly 513 bytes: invalid
        let mut cp = make_climate_pattern();
        cp.indicators = vec!["A".repeat(513)];
        assert!(is_invalid(&validate_create_climate_pattern(
            fake_create(),
            cp
        )));
    }

    #[test]
    fn climate_pattern_indicator_count_boundaries() {
        // 49 indicators: valid
        let mut cp = make_climate_pattern();
        cp.indicators = (0..49).map(|i| format!("I{}", i)).collect();
        assert!(is_valid(&validate_create_climate_pattern(
            fake_create(),
            cp
        )));

        // 50 indicators: valid
        let mut cp = make_climate_pattern();
        cp.indicators = (0..50).map(|i| format!("I{}", i)).collect();
        assert!(is_valid(&validate_create_climate_pattern(
            fake_create(),
            cp
        )));

        // 51 indicators: invalid
        let mut cp = make_climate_pattern();
        cp.indicators = (0..51).map(|i| format!("I{}", i)).collect();
        assert!(is_invalid(&validate_create_climate_pattern(
            fake_create(),
            cp
        )));
    }

    #[test]
    fn practice_emoji_in_title_accepted() {
        let mut p = make_practice();
        p.title = "Water Ceremony \u{1f4a7}\u{1f30a}".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_newlines_in_description_accepted() {
        let mut p = make_practice();
        p.description = "Line 1\nLine 2\nLine 3".into();
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_many_applicable_to_accepted() {
        let mut m = make_conservation_method();
        // 100 items (with duplicates) should be valid since no cap is enforced
        m.applicable_to = (0..100).map(|_| WaterClassification::Potable).collect();
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    // --- String length limit hardening tests ---

    #[test]
    fn practice_id_too_long_rejected() {
        let mut p = make_practice();
        p.id = "A".repeat(257);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn practice_id_at_limit_accepted() {
        let mut p = make_practice();
        p.id = "A".repeat(256);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_title_too_long_rejected() {
        let mut p = make_practice();
        p.title = "A".repeat(513);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn practice_title_at_limit_accepted() {
        let mut p = make_practice();
        p.title = "A".repeat(512);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_description_too_long_rejected() {
        let mut p = make_practice();
        p.description = "A".repeat(16385);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn practice_description_at_limit_accepted() {
        let mut p = make_practice();
        p.description = "A".repeat(16384);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_region_too_long_rejected() {
        let mut p = make_practice();
        p.region = "A".repeat(513);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn practice_culture_too_long_rejected() {
        let mut p = make_practice();
        p.culture_or_community = "A".repeat(513);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn practice_sacred_access_level_accepted() {
        let mut p = make_practice();
        p.access_level = AccessLevel::Sacred;
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_all_access_levels_valid() {
        for level in [
            AccessLevel::Public,
            AccessLevel::CommunityOnly,
            AccessLevel::ElderApproved,
            AccessLevel::Sacred,
        ] {
            let mut p = make_practice();
            p.access_level = level;
            let result = validate_create_practice(fake_create(), p);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn practice_multibyte_unicode_title_at_byte_limit() {
        let mut p = make_practice();
        // 4-byte emoji x 128 = 512 bytes
        p.title = "\u{1f4a7}".repeat(128);
        assert_eq!(p.title.len(), 512);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_valid(&result));
    }

    #[test]
    fn practice_multibyte_unicode_title_over_byte_limit() {
        let mut p = make_practice();
        // 4-byte emoji x 129 = 516 bytes > 512
        p.title = "\u{1f4a7}".repeat(129);
        assert!(p.title.len() > 512);
        let result = validate_create_practice(fake_create(), p);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS — CONSERVATION METHOD
    // ========================================================================

    #[test]
    fn conservation_method_id_at_limit_accepted() {
        let mut m = make_conservation_method();
        m.id = "A".repeat(64);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_id_over_limit_rejected() {
        let mut m = make_conservation_method();
        m.id = "A".repeat(257);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Conservation method ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn conservation_method_title_at_limit_accepted() {
        let mut m = make_conservation_method();
        m.title = "A".repeat(256);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_title_over_limit_rejected() {
        let mut m = make_conservation_method();
        m.title = "A".repeat(257);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Conservation method title must be 256 characters or fewer"
        );
    }

    #[test]
    fn conservation_method_description_at_limit_accepted() {
        let mut m = make_conservation_method();
        m.description = "A".repeat(4096);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_valid(&result));
    }

    #[test]
    fn conservation_method_description_over_limit_rejected() {
        let mut m = make_conservation_method();
        m.description = "A".repeat(4097);
        let result = validate_create_conservation_method(fake_create(), m);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Conservation method description must be 4096 characters or fewer"
        );
    }

    // ========================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS — CLIMATE PATTERN
    // ========================================================================

    #[test]
    fn climate_pattern_region_at_limit_accepted() {
        let mut cp = make_climate_pattern();
        cp.region = "A".repeat(256);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_region_over_limit_rejected() {
        let mut cp = make_climate_pattern();
        cp.region = "A".repeat(257);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Region must be 256 characters or fewer"
        );
    }

    #[test]
    fn climate_pattern_season_at_limit_accepted() {
        let mut cp = make_climate_pattern();
        cp.season = "A".repeat(128);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_season_over_limit_rejected() {
        let mut cp = make_climate_pattern();
        cp.season = "A".repeat(129);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Season must be 128 characters or fewer"
        );
    }

    #[test]
    fn climate_pattern_description_at_limit_accepted() {
        let mut cp = make_climate_pattern();
        cp.description = "A".repeat(4096);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_valid(&result));
    }

    #[test]
    fn climate_pattern_description_over_limit_rejected() {
        let mut cp = make_climate_pattern();
        cp.description = "A".repeat(4097);
        let result = validate_create_climate_pattern(fake_create(), cp);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Description must be 4096 characters or fewer"
        );
    }
}
