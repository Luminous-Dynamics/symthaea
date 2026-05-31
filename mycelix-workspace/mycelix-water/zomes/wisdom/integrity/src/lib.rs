// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Wisdom Integrity Zome
//! Traditional water knowledge, conservation methods, and climate patterns

use hdi::prelude::*;

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
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_practice(
    _action: Create,
    practice: TraditionalPractice,
) -> ExternResult<ValidateCallbackResult> {
    if practice.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice ID cannot be empty".into(),
        ));
    }
    if practice.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice title cannot be empty".into(),
        ));
    }
    if practice.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Practice description cannot be empty".into(),
        ));
    }
    if practice.region.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Region cannot be empty".into(),
        ));
    }
    if practice.culture_or_community.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Culture or community cannot be empty".into(),
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
    if method.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method ID cannot be empty".into(),
        ));
    }
    if method.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method title cannot be empty".into(),
        ));
    }
    if method.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conservation method description cannot be empty".into(),
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
    if pattern.region.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Region cannot be empty".into(),
        ));
    }
    if pattern.season.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Season cannot be empty".into(),
        ));
    }
    if pattern.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Description cannot be empty".into(),
        ));
    }
    if pattern.indicators.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 indicators".into(),
        ));
    }
    for indicator in &pattern.indicators {
        if indicator.is_empty() || indicator.len() > 512 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each indicator must be 1-512 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}
