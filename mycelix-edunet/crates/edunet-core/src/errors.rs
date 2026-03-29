// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Structured Error Handling for EduNet
//!
//! Provides descriptive, user-friendly error types with:
//! - Error codes for programmatic handling
//! - Context about what entity/action failed
//! - Resolution hints where possible
//! - Consistent error patterns across all zomes

use serde::{Deserialize, Serialize};

/// Error category codes for programmatic handling
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorCode {
    // Entity errors (1xx)
    EntityNotFound = 100,
    EntityAlreadyExists = 101,
    InvalidEntityState = 102,
    EntityExpired = 103,

    // Validation errors (2xx)
    ValidationFailed = 200,
    InvalidInput = 201,
    OutOfRange = 202,
    InvalidFormat = 203,
    MissingRequired = 204,

    // Authorization errors (3xx)
    Unauthorized = 300,
    InsufficientPermissions = 301,
    AuthorMismatch = 302,

    // State errors (4xx)
    InvalidStateTransition = 400,
    ConcurrentModification = 401,
    StaleData = 402,

    // External errors (5xx)
    CrossZomeCallFailed = 500,
    NetworkError = 501,
    StorageError = 502,

    // Resource errors (6xx)
    ResourceExhausted = 600,
    LimitExceeded = 601,
    QuotaExceeded = 602,
}

/// Structured error with context and hints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EduNetError {
    /// Error code for programmatic handling
    pub code: ErrorCode,
    /// Entity type that the error relates to (e.g., "ReviewCard", "LearnerProfile")
    pub entity_type: String,
    /// Specific action that failed (e.g., "get", "create", "update", "delete")
    pub action: String,
    /// Human-readable error message
    pub message: String,
    /// Optional context (e.g., entity ID, parameter name)
    pub context: Option<String>,
    /// Optional hint for resolution
    pub hint: Option<String>,
}

impl EduNetError {
    /// Create a new EduNet error
    pub fn new(code: ErrorCode, entity_type: &str, action: &str, message: &str) -> Self {
        Self {
            code,
            entity_type: entity_type.to_string(),
            action: action.to_string(),
            message: message.to_string(),
            context: None,
            hint: None,
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }

    /// Add a resolution hint
    pub fn with_hint(mut self, hint: &str) -> Self {
        self.hint = Some(hint.to_string());
        self
    }

    /// Format as a descriptive error message
    pub fn to_message(&self) -> String {
        let mut msg = format!(
            "[E{:03}] {} {} failed: {}",
            self.code.clone() as u16,
            self.entity_type,
            self.action,
            self.message
        );

        if let Some(ref ctx) = self.context {
            msg.push_str(&format!(" ({})", ctx));
        }

        if let Some(ref hint) = self.hint {
            msg.push_str(&format!(" → Hint: {}", hint));
        }

        msg
    }
}

impl std::fmt::Display for EduNetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_message())
    }
}

// ============================================================================
// Common Error Constructors
// ============================================================================

/// Helper functions for common error patterns
pub mod errors {
    use super::*;

    /// Entity not found in DHT
    pub fn not_found(entity_type: &str, context: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::EntityNotFound,
            entity_type,
            "get",
            "Entry not found in DHT"
        )
        .with_context(context)
        .with_hint("The entry may have been deleted or never created")
    }

    /// Entity already exists (duplicate)
    pub fn already_exists(entity_type: &str, context: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::EntityAlreadyExists,
            entity_type,
            "create",
            "Entry already exists"
        )
        .with_context(context)
        .with_hint("Use update instead of create, or delete the existing entry first")
    }

    /// Failed to decode/deserialize entry
    pub fn invalid_entry(entity_type: &str, context: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidFormat,
            entity_type,
            "decode",
            "Failed to decode entry from DHT"
        )
        .with_context(context)
        .with_hint("The entry may be corrupted or from an incompatible version")
    }

    /// Failed to create entry
    pub fn create_failed(entity_type: &str, context: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::StorageError,
            entity_type,
            "create",
            "Failed to store entry in DHT"
        )
        .with_context(context)
    }

    /// Failed to update entry
    pub fn update_failed(entity_type: &str, context: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::StorageError,
            entity_type,
            "update",
            "Failed to update entry in DHT"
        )
        .with_context(context)
    }

    /// Input validation failed
    pub fn validation_failed(entity_type: &str, field: &str, reason: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::ValidationFailed,
            entity_type,
            "validate",
            reason
        )
        .with_context(&format!("field: {}", field))
    }

    /// Value out of valid range
    pub fn out_of_range(entity_type: &str, field: &str, value: &str, range: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::OutOfRange,
            entity_type,
            "validate",
            &format!("Value {} is out of valid range", value)
        )
        .with_context(&format!("field: {}, valid range: {}", field, range))
    }

    /// Missing required field
    pub fn missing_required(entity_type: &str, field: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::MissingRequired,
            entity_type,
            "validate",
            "Required field is missing or empty"
        )
        .with_context(&format!("field: {}", field))
    }

    /// Unauthorized action
    pub fn unauthorized(entity_type: &str, action: &str, reason: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::Unauthorized,
            entity_type,
            action,
            reason
        )
        .with_hint("Ensure you are the author of the entry or have appropriate permissions")
    }

    /// Invalid state transition
    pub fn invalid_state(entity_type: &str, current: &str, target: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidStateTransition,
            entity_type,
            "transition",
            &format!("Cannot transition from {} to {}", current, target)
        )
        .with_hint("Check the valid state transition rules for this entity type")
    }

    /// Cross-zome call failed
    pub fn cross_zome_failed(target_zome: &str, function: &str, reason: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::CrossZomeCallFailed,
            target_zome,
            function,
            reason
        )
        .with_hint("Check that the target zome is installed and the function exists")
    }

    /// Limit exceeded
    pub fn limit_exceeded(entity_type: &str, limit_name: &str, limit_value: u32) -> EduNetError {
        EduNetError::new(
            ErrorCode::LimitExceeded,
            entity_type,
            "create",
            &format!("{} limit of {} exceeded", limit_name, limit_value)
        )
    }
}

// ============================================================================
// SRS-Specific Errors
// ============================================================================

pub mod srs_errors {
    use super::*;

    pub fn card_not_found(hash: &str) -> EduNetError {
        errors::not_found("ReviewCard", &format!("hash: {}", hash))
    }

    pub fn deck_not_found(hash: &str) -> EduNetError {
        errors::not_found("Deck", &format!("hash: {}", hash))
    }

    pub fn session_not_found(hash: &str) -> EduNetError {
        errors::not_found("ReviewSession", &format!("hash: {}", hash))
    }

    pub fn invalid_recall_quality(quality: u8) -> EduNetError {
        errors::out_of_range("ReviewEvent", "quality", &quality.to_string(), "0-5")
            .with_hint("0=complete blackout, 1-2=incorrect, 3=difficult, 4=good, 5=perfect")
    }

    pub fn card_is_leech(hash: &str, lapse_count: u32) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidEntityState,
            "ReviewCard",
            "review",
            &format!("Card has been marked as leech after {} lapses", lapse_count)
        )
        .with_context(&format!("hash: {}", hash))
        .with_hint("Consider editing this card to make it easier to remember")
    }

    pub fn card_not_due(hash: &str, due_at: i64) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidEntityState,
            "ReviewCard",
            "review",
            "Card is not due for review yet"
        )
        .with_context(&format!("hash: {}, due_at: {}", hash, due_at))
        .with_hint("Use 'get_due_cards' to find cards ready for review")
    }
}

// ============================================================================
// Gamification-Specific Errors
// ============================================================================

pub mod gamification_errors {
    use super::*;

    pub fn badge_already_earned(badge_id: &str) -> EduNetError {
        errors::already_exists("EarnedBadge", &format!("badge_id: {}", badge_id))
    }

    pub fn badge_not_found(hash: &str) -> EduNetError {
        errors::not_found("BadgeDefinition", &format!("hash: {}", hash))
    }

    pub fn streak_broken(last_activity: i64, gap_days: u32) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidEntityState,
            "LearnerStreak",
            "continue",
            &format!("Streak broken after {} days of inactivity", gap_days)
        )
        .with_context(&format!("last_activity: {}", last_activity))
        .with_hint("Start a new streak by completing an activity today")
    }

    pub fn no_streak_freezes(freezes_used: u8, max_freezes: u8) -> EduNetError {
        EduNetError::new(
            ErrorCode::ResourceExhausted,
            "LearnerStreak",
            "freeze",
            "No streak freezes remaining"
        )
        .with_context(&format!("used: {}, max: {}", freezes_used, max_freezes))
        .with_hint("Complete activities to earn more streak freezes")
    }

    pub fn invalid_xp_amount(amount: i64) -> EduNetError {
        EduNetError::new(
            ErrorCode::OutOfRange,
            "XpTransaction",
            "award",
            "XP amount must be positive"
        )
        .with_context(&format!("amount: {}", amount))
    }

    pub fn leaderboard_not_found(period: &str, scope: &str) -> EduNetError {
        errors::not_found("Leaderboard", &format!("period: {}, scope: {}", period, scope))
    }
}

// ============================================================================
// Adaptive Learning Errors
// ============================================================================

pub mod adaptive_errors {
    use super::*;

    pub fn profile_not_found(agent: &str) -> EduNetError {
        errors::not_found("LearnerProfile", &format!("agent: {}", agent))
    }

    pub fn skill_not_found(skill_id: &str) -> EduNetError {
        errors::not_found("SkillMastery", &format!("skill_id: {}", skill_id))
    }

    pub fn goal_not_found(hash: &str) -> EduNetError {
        errors::not_found("LearningGoal", &format!("hash: {}", hash))
    }

    pub fn invalid_mastery_level(permille: u16) -> EduNetError {
        errors::out_of_range("SkillMastery", "mastery_permille", &permille.to_string(), "0-1000")
            .with_hint("Mastery is measured in permille (0-1000, where 1000 = 100%)")
    }

    pub fn difficulty_mismatch(skill_difficulty: u16, content_difficulty: u16) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidEntityState,
            "Recommendation",
            "generate",
            "Content difficulty outside learner's ZPD"
        )
        .with_context(&format!(
            "skill_level: {}, content_difficulty: {}",
            skill_difficulty, content_difficulty
        ))
        .with_hint("Select content within ±150 permille of current skill level")
    }

    pub fn goal_already_completed(hash: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidStateTransition,
            "LearningGoal",
            "complete",
            "Goal is already marked as completed"
        )
        .with_context(&format!("hash: {}", hash))
    }

    pub fn invalid_learning_style(style: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidInput,
            "LearnerProfile",
            "update",
            "Invalid learning style"
        )
        .with_context(&format!("style: {}", style))
        .with_hint("Valid styles: Visual, Auditory, ReadWrite, Kinesthetic")
    }
}

// ============================================================================
// Integration Layer Errors
// ============================================================================

pub mod integration_errors {
    use super::*;

    pub fn session_not_found(hash: &str) -> EduNetError {
        errors::not_found("OrchestratedSession", &format!("hash: {}", hash))
    }

    pub fn session_already_active(existing_hash: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::EntityAlreadyExists,
            "OrchestratedSession",
            "start",
            "An active session already exists"
        )
        .with_context(&format!("existing_session: {}", existing_hash))
        .with_hint("End the current session before starting a new one")
    }

    pub fn session_not_active(hash: &str, state: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidStateTransition,
            "OrchestratedSession",
            "update",
            &format!("Session is not active (current state: {})", state)
        )
        .with_context(&format!("hash: {}", hash))
    }

    pub fn invalid_session_transition(current: &str, target: &str) -> EduNetError {
        errors::invalid_state("OrchestratedSession", current, target)
    }

    pub fn trigger_not_found(hash: &str) -> EduNetError {
        errors::not_found("AchievementTrigger", &format!("hash: {}", hash))
    }

    pub fn invalid_event_type(event_type: &str) -> EduNetError {
        EduNetError::new(
            ErrorCode::InvalidInput,
            "LearningEvent",
            "record",
            "Invalid event type"
        )
        .with_context(&format!("type: {}", event_type))
        .with_hint("Valid types: SrsReview, SkillPractice, CourseProgress, GoalMilestone, SessionComplete")
    }

    pub fn cross_zome_srs_failed(function: &str, reason: &str) -> EduNetError {
        errors::cross_zome_failed("srs_coordinator", function, reason)
    }

    pub fn cross_zome_gamification_failed(function: &str, reason: &str) -> EduNetError {
        errors::cross_zome_failed("gamification_coordinator", function, reason)
    }

    pub fn cross_zome_adaptive_failed(function: &str, reason: &str) -> EduNetError {
        errors::cross_zome_failed("adaptive_coordinator", function, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_formatting() {
        let err = errors::not_found("ReviewCard", "hash: abc123");
        let msg = err.to_message();

        assert!(msg.contains("[E100]"));
        assert!(msg.contains("ReviewCard"));
        assert!(msg.contains("not found"));
        assert!(msg.contains("abc123"));
        assert!(msg.contains("Hint:"));
    }

    #[test]
    fn test_srs_card_not_found() {
        let err = srs_errors::card_not_found("Qm123");
        assert_eq!(err.code, ErrorCode::EntityNotFound);
        assert_eq!(err.entity_type, "ReviewCard");
        assert!(err.context.unwrap().contains("Qm123"));
    }

    #[test]
    fn test_validation_error() {
        let err = errors::out_of_range("ReviewEvent", "quality", "7", "0-5");
        let msg = err.to_message();

        assert!(msg.contains("[E202]"));
        assert!(msg.contains("quality"));
        assert!(msg.contains("7"));
        assert!(msg.contains("0-5"));
    }

    #[test]
    fn test_cross_zome_error() {
        let err = integration_errors::cross_zome_srs_failed("get_due_cards", "Network timeout");
        let msg = err.to_message();

        assert!(msg.contains("[E500]"));
        assert!(msg.contains("srs_coordinator"));
        assert!(msg.contains("Network timeout"));
    }

    #[test]
    fn test_gamification_streak_error() {
        let err = gamification_errors::streak_broken(1234567890, 3);
        assert_eq!(err.code, ErrorCode::InvalidEntityState);
        assert!(err.message.contains("3 days"));
        assert!(err.hint.is_some());
    }

    #[test]
    fn test_adaptive_zpd_error() {
        let err = adaptive_errors::difficulty_mismatch(500, 900);
        let msg = err.to_message();

        assert!(msg.contains("ZPD"));
        assert!(msg.contains("500"));
        assert!(msg.contains("900"));
    }

    #[test]
    fn test_error_chaining() {
        let err = EduNetError::new(
            ErrorCode::ValidationFailed,
            "TestEntity",
            "create",
            "Test failure"
        )
        .with_context("test context")
        .with_hint("test hint");

        let msg = err.to_message();
        assert!(msg.contains("test context"));
        assert!(msg.contains("test hint"));
    }
}
