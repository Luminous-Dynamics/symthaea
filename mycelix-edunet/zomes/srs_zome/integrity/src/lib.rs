// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spaced Repetition System (SRS) Integrity Zome
//!
//! Implements the SuperMemo SM-2 algorithm for optimal learning retention.
//!
//! ## The Science Behind SRS
//!
//! Spaced repetition leverages the "spacing effect" - the phenomenon where
//! learning is more effective when study sessions are spaced out over time.
//! This system can improve long-term retention by 200-400% compared to
//! traditional study methods.
//!
//! ## The SM-2 Algorithm
//!
//! The SM-2 algorithm calculates optimal review intervals based on:
//! - **Ease Factor (EF)**: How easy/difficult the item is for the learner
//! - **Interval**: Days until next review
//! - **Repetitions**: Number of successful reviews
//! - **Quality**: Self-rated recall quality (0-5)
//!
//! Formula:
//! - EF' = EF + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
//! - If quality >= 3: interval = previous_interval * EF
//! - If quality < 3: restart from interval = 1
//!
//! ## Integration with Knowledge Roots
//!
//! Each ReviewCard links to a KnowledgeNode, enabling:
//! - Prerequisite-aware scheduling (review fundamentals first)
//! - Skill-aligned review priorities
//! - Pod-based review challenges

use hdi::prelude::*;

/// Quality of recall during a review session (0-5 scale)
/// This is the core input to the SM-2 algorithm
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecallQuality {
    /// Complete blackout, no memory at all
    Blackout = 0,
    /// Incorrect response, but remembered upon seeing answer
    Incorrect = 1,
    /// Incorrect response, but answer seemed easy to recall
    IncorrectEasy = 2,
    /// Correct response with serious difficulty
    CorrectHard = 3,
    /// Correct response after hesitation
    CorrectHesitant = 4,
    /// Perfect response with no hesitation
    Perfect = 5,
}

impl RecallQuality {
    pub fn as_u8(&self) -> u8 {
        match self {
            RecallQuality::Blackout => 0,
            RecallQuality::Incorrect => 1,
            RecallQuality::IncorrectEasy => 2,
            RecallQuality::CorrectHard => 3,
            RecallQuality::CorrectHesitant => 4,
            RecallQuality::Perfect => 5,
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => RecallQuality::Blackout,
            1 => RecallQuality::Incorrect,
            2 => RecallQuality::IncorrectEasy,
            3 => RecallQuality::CorrectHard,
            4 => RecallQuality::CorrectHesitant,
            _ => RecallQuality::Perfect,
        }
    }

    /// Returns true if the response was correct (quality >= 3)
    pub fn is_correct(&self) -> bool {
        self.as_u8() >= 3
    }
}

impl Default for RecallQuality {
    fn default() -> Self {
        RecallQuality::CorrectHesitant
    }
}

/// Card status in the SRS system
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CardStatus {
    /// New card, never reviewed
    New,
    /// Currently in learning phase (short intervals)
    Learning,
    /// In review phase (spaced intervals)
    Review,
    /// Card has been suspended (won't appear in reviews)
    Suspended,
    /// Card is buried until tomorrow
    Buried,
    /// Card has been "graduated" (very long intervals, well-learned)
    Graduated,
}

impl Default for CardStatus {
    fn default() -> Self {
        CardStatus::New
    }
}

/// A review card linked to a knowledge node
/// This is the core entity of the SRS system
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct ReviewCard {
    /// The knowledge node this card is for
    pub node_hash: ActionHash,
    /// The learner who owns this card
    pub learner: AgentPubKey,

    // === SM-2 Algorithm State ===

    /// Ease factor (stored as permille: 2500 = 2.5)
    /// Range: 1300 (hard) to 3000+ (easy)
    /// Default: 2500 (2.5)
    pub ease_factor_permille: u16,

    /// Current interval in minutes (allows sub-day intervals for learning)
    /// 1440 = 1 day, 10080 = 1 week
    pub interval_minutes: u32,

    /// Number of consecutive successful reviews
    pub repetitions: u32,

    /// Current card status
    pub status: CardStatus,

    // === Scheduling ===

    /// When this card is due for review (Unix timestamp)
    pub due_at: i64,

    /// When this card was last reviewed
    pub last_reviewed_at: Option<i64>,

    // === Statistics ===

    /// Total number of reviews
    pub total_reviews: u32,

    /// Number of times the card was answered correctly
    pub correct_count: u32,

    /// Number of times the card was answered incorrectly
    pub incorrect_count: u32,

    /// Number of times the card "lapsed" (forgotten after learning)
    pub lapses: u32,

    /// Average time to answer in seconds (stored as u32)
    pub avg_time_seconds: u32,

    // === Metadata ===

    /// Optional custom front content (overrides node content)
    pub custom_front: Option<String>,

    /// Optional custom back content (overrides node content)
    pub custom_back: Option<String>,

    /// Tags for organization
    pub tags: Vec<String>,

    /// Card creation timestamp
    pub created_at: i64,

    /// Card modification timestamp
    pub modified_at: i64,
}

/// A single review event
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct ReviewEvent {
    /// The card that was reviewed
    pub card_hash: ActionHash,
    /// The learner who reviewed
    pub learner: AgentPubKey,
    /// Quality of recall (0-5)
    pub quality: u8,
    /// Time taken to respond in milliseconds
    pub response_time_ms: u32,
    /// Ease factor before this review (permille)
    pub ease_before_permille: u16,
    /// Ease factor after this review (permille)
    pub ease_after_permille: u16,
    /// Interval before this review (minutes)
    pub interval_before_minutes: u32,
    /// Interval after this review (minutes)
    pub interval_after_minutes: u32,
    /// When the review occurred
    pub reviewed_at: i64,
}

/// A review session containing multiple reviews
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct ReviewSession {
    /// The learner who conducted the session
    pub learner: AgentPubKey,
    /// Cards reviewed in this session
    pub cards_reviewed: Vec<ActionHash>,
    /// Total time spent in seconds
    pub duration_seconds: u32,
    /// Number of cards answered correctly
    pub correct_count: u32,
    /// Number of cards answered incorrectly
    pub incorrect_count: u32,
    /// Session start time
    pub started_at: i64,
    /// Session end time
    pub ended_at: i64,
    /// Optional pod this session was part of
    pub pod_hash: Option<ActionHash>,
}

/// Daily study statistics
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct DailyStats {
    /// The learner
    pub learner: AgentPubKey,
    /// Date (as YYYYMMDD integer for easy querying)
    pub date: u32,
    /// Number of new cards studied
    pub new_cards: u32,
    /// Number of cards reviewed
    pub reviews: u32,
    /// Number of cards relearned (lapses)
    pub relearns: u32,
    /// Total study time in seconds
    pub study_time_seconds: u32,
    /// Retention rate as permille (850 = 85%)
    pub retention_permille: u16,
    /// Current streak (consecutive days of study)
    pub streak_days: u32,
}

/// SRS configuration for a learner (personalization)
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct SrsConfig {
    /// The learner this config belongs to
    pub learner: AgentPubKey,

    // === New Card Settings ===

    /// Maximum new cards per day
    pub new_cards_per_day: u32,

    /// Learning steps in minutes (e.g., [1, 10] = 1 min, then 10 min)
    pub learning_steps_minutes: Vec<u32>,

    /// Graduating interval in days (when card moves from learning to review)
    pub graduating_interval_days: u32,

    /// Easy interval in days (when user presses "Easy" on new card)
    pub easy_interval_days: u32,

    // === Review Settings ===

    /// Maximum reviews per day (0 = unlimited)
    pub max_reviews_per_day: u32,

    /// Easy bonus as permille (1300 = 1.3x multiplier)
    pub easy_bonus_permille: u16,

    /// Interval modifier as permille (1000 = 100%, no modification)
    pub interval_modifier_permille: u16,

    /// Maximum interval in days
    pub max_interval_days: u32,

    // === Lapse Settings ===

    /// Relearning steps in minutes after a lapse
    pub relearning_steps_minutes: Vec<u32>,

    /// Minimum interval in days after a lapse
    pub lapse_min_interval_days: u32,

    /// Leech threshold (number of lapses before card is a "leech")
    pub leech_threshold: u32,

    /// What to do with leeches
    pub leech_action: LeechAction,

    // === Timing ===

    /// Hour when the next day starts (for streak calculation)
    pub day_start_hour: u8,

    /// Timezone offset in minutes from UTC
    pub timezone_offset_minutes: i16,

    /// Config creation timestamp
    pub created_at: i64,

    /// Config modification timestamp
    pub modified_at: i64,
}

/// What to do when a card becomes a "leech" (repeatedly forgotten)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeechAction {
    /// Suspend the card
    Suspend,
    /// Just tag the card as a leech
    TagOnly,
    /// Convert to simpler card format
    Simplify,
}

impl Default for LeechAction {
    fn default() -> Self {
        LeechAction::TagOnly
    }
}

/// A deck organizing cards by topic
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Deck {
    /// Deck name
    pub name: String,
    /// Deck description
    pub description: String,
    /// Owner of the deck
    pub owner: AgentPubKey,
    /// Parent deck (for nested decks)
    pub parent: Option<ActionHash>,
    /// Custom config overrides (if different from learner defaults)
    pub config_override: Option<ActionHash>,
    /// Whether this deck is shared publicly
    pub is_public: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: i64,
}

/// Forecast data for upcoming reviews
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewForecast {
    /// Date (YYYYMMDD)
    pub date: u32,
    /// Number of cards due for review
    pub review_count: u32,
    /// Number of new cards available
    pub new_count: u32,
    /// Estimated time in minutes
    pub estimated_minutes: u32,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 1, visibility = "private")]
    ReviewCard(ReviewCard),
    #[entry_type(required_validations = 1, visibility = "private")]
    ReviewEvent(ReviewEvent),
    #[entry_type(required_validations = 1, visibility = "private")]
    ReviewSession(ReviewSession),
    #[entry_type(required_validations = 1, visibility = "private")]
    DailyStats(DailyStats),
    #[entry_type(required_validations = 1, visibility = "private")]
    SrsConfig(SrsConfig),
    #[entry_type(required_validations = 3, visibility = "public")]
    Deck(Deck),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Learner -> Their cards
    LearnerToCards,
    /// Learner -> Their config
    LearnerToConfig,
    /// Learner -> Their daily stats
    LearnerToStats,
    /// Learner -> Their review sessions
    LearnerToSessions,
    /// Deck -> Cards in deck
    DeckToCards,
    /// Card -> Review events
    CardToEvents,
    /// Node -> Cards for that node
    NodeToCards,
    /// All public decks anchor
    AllDecks,
    /// Learner -> Decks they own
    LearnerToDecks,
}

// ============== Validation Functions ==============

/// Validate review card creation
pub fn validate_create_card(card: &ReviewCard) -> ExternResult<ValidateCallbackResult> {
    // Ease factor should be reasonable (1.0 to 5.0, stored as permille)
    if card.ease_factor_permille < 1000 || card.ease_factor_permille > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Ease factor must be between 1000 and 5000 (1.0 to 5.0)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate SRS config
pub fn validate_srs_config(config: &SrsConfig) -> ExternResult<ValidateCallbackResult> {
    // Learning steps must not be empty
    if config.learning_steps_minutes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Learning steps cannot be empty".to_string(),
        ));
    }

    // Max interval must be reasonable
    if config.max_interval_days > 36500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum interval cannot exceed 100 years".to_string(),
        ));
    }

    // Day start hour must be valid
    if config.day_start_hour >= 24 {
        return Ok(ValidateCallbackResult::Invalid(
            "Day start hour must be 0-23".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate callback dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::ReviewCard(card) => validate_create_card(&card),
                EntryTypes::SrsConfig(config) => validate_srs_config(&config),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::ReviewCard(card) => validate_create_card(&card),
                EntryTypes::SrsConfig(config) => validate_srs_config(&config),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_quality_conversions() {
        assert_eq!(RecallQuality::Blackout.as_u8(), 0);
        assert_eq!(RecallQuality::Perfect.as_u8(), 5);
        assert_eq!(RecallQuality::from_u8(3), RecallQuality::CorrectHard);
        assert!(RecallQuality::CorrectHard.is_correct());
        assert!(!RecallQuality::IncorrectEasy.is_correct());
    }

    #[test]
    fn test_card_status_default() {
        assert_eq!(CardStatus::default(), CardStatus::New);
    }

    #[test]
    fn test_leech_action_default() {
        assert_eq!(LeechAction::default(), LeechAction::TagOnly);
    }
}
