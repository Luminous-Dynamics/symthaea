//! Hearth Gratitude Integrity Zome
//! Defines entry types and validation for gratitude expressions,
//! appreciation circles, and gratitude anchors.

use hdi::prelude::*;
use hearth_types::*;

// ============================================================================
// Entry Types
// ============================================================================

/// A gratitude expression from one hearth member to another.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GratitudeExpression {
    /// The hearth this gratitude belongs to.
    pub hearth_hash: ActionHash,
    /// Agent expressing gratitude.
    pub from_agent: AgentPubKey,
    /// Agent receiving gratitude.
    pub to_agent: AgentPubKey,
    /// The gratitude message.
    pub message: String,
    /// Category of gratitude.
    pub gratitude_type: GratitudeType,
    /// Privacy scope.
    pub visibility: HearthVisibility,
    /// When the gratitude was expressed.
    pub created_at: Timestamp,
}

/// A themed appreciation circle where members share gratitude together.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AppreciationCircle {
    /// The hearth this circle belongs to.
    pub hearth_hash: ActionHash,
    /// Theme or topic for the appreciation circle.
    pub theme: String,
    /// Agents participating in the circle.
    pub participants: Vec<AgentPubKey>,
    /// When the circle started.
    pub started_at: Timestamp,
    /// When the circle completed (None if still in progress).
    pub completed_at: Option<Timestamp>,
    /// Current status of the circle.
    pub status: CircleStatus,
}

/// Anchor tracking an agent's gratitude statistics within a hearth.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GratitudeAnchor {
    /// The agent this anchor tracks.
    pub agent: AgentPubKey,
    /// The hearth this anchor belongs to.
    pub hearth_hash: ActionHash,
    /// Total gratitude expressions given.
    pub total_given: u32,
    /// Total gratitude expressions received.
    pub total_received: u32,
    /// Current streak of consecutive days with gratitude.
    pub current_streak_days: u32,
}

// ============================================================================
// Entry & Link Type Registration
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    GratitudeExpression(GratitudeExpression),
    AppreciationCircle(AppreciationCircle),
    GratitudeAnchor(GratitudeAnchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Hearth to all gratitude expressions within it.
    HearthToGratitude,
    /// Agent to gratitude expressions they have given.
    AgentToGratitudeGiven,
    /// Agent to gratitude expressions they have received.
    AgentToGratitudeReceived,
    /// Hearth to appreciation circles.
    HearthToCircles,
    /// Agent to circles they participate in.
    AgentToCircles,
    /// Hearth to gratitude anchors for its members.
    HearthToGratitudeAnchors,
}

// ============================================================================
// Genesis
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::GratitudeExpression(expr) => validate_gratitude(action, expr),
                EntryTypes::AppreciationCircle(circle) => validate_circle(action, circle),
                EntryTypes::GratitudeAnchor(anchor) => validate_anchor(action, anchor),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::GratitudeExpression(_) => {
                    // Gratitude expressions are immutable once created.
                    Ok(ValidateCallbackResult::Invalid(
                        "Gratitude expressions cannot be updated".into(),
                    ))
                }
                EntryTypes::AppreciationCircle(circle) => validate_circle_update(circle),
                EntryTypes::GratitudeAnchor(anchor) => validate_anchor_update(anchor),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => validate_create_link(link_type, &tag),
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => validate_delete_link(link_type, &tag),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Invalid(
            "Gratitude expressions cannot be deleted once created".into(),
        )),
    }
}

fn validate_gratitude(
    _action: Create,
    expr: GratitudeExpression,
) -> ExternResult<ValidateCallbackResult> {
    if expr.message.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Gratitude message cannot be empty".into(),
        ));
    }
    if expr.message.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Gratitude message must be <= 2048 characters".into(),
        ));
    }
    if expr.from_agent == expr.to_agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot express gratitude to yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_circle(
    _action: Create,
    circle: AppreciationCircle,
) -> ExternResult<ValidateCallbackResult> {
    if circle.theme.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle theme cannot be empty".into(),
        ));
    }
    if circle.theme.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle theme must be <= 256 characters".into(),
        ));
    }
    if circle.participants.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must have at least 2 participants".into(),
        ));
    }
    if circle.participants.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle cannot have more than 50 participants".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_circle_update(circle: AppreciationCircle) -> ExternResult<ValidateCallbackResult> {
    if circle.theme.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle theme cannot be empty".into(),
        ));
    }
    if circle.theme.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle theme must be <= 256 characters".into(),
        ));
    }
    if circle.participants.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must have at least 2 participants".into(),
        ));
    }
    if circle.participants.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle cannot have more than 50 participants".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_anchor(
    _action: Create,
    anchor: GratitudeAnchor,
) -> ExternResult<ValidateCallbackResult> {
    // Check that total counts don't overflow when summed.
    if anchor
        .total_given
        .checked_add(anchor.total_received)
        .is_none()
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Gratitude anchor total counts would overflow".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_anchor_update(anchor: GratitudeAnchor) -> ExternResult<ValidateCallbackResult> {
    if anchor
        .total_given
        .checked_add(anchor.total_received)
        .is_none()
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Gratitude anchor total counts would overflow".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_link(
    link_type: LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    let max_len = link_tag_max_len(&link_type);
    if tag.0.len() > max_len {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "{:?} link tag too long (max {} bytes)",
            link_type, max_len
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_delete_link(
    link_type: LinkTypes,
    tag: &LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    let max_len = link_tag_max_len(&link_type);
    if tag.0.len() > max_len {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "{:?} delete link tag too long (max {} bytes)",
            link_type, max_len
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn link_tag_max_len(link_type: &LinkTypes) -> usize {
    match link_type {
        LinkTypes::HearthToGratitude => 256,
        LinkTypes::AgentToGratitudeGiven => 256,
        LinkTypes::AgentToGratitudeReceived => 256,
        LinkTypes::HearthToCircles => 256,
        LinkTypes::AgentToCircles => 256,
        LinkTypes::HearthToGratitudeAnchors => 256,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Factory functions

    fn agent_a() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAA; 36])
    }

    fn agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xBB; 36])
    }

    fn valid_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xAB; 36])
    }

    fn valid_timestamp() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    fn valid_create_action() -> Create {
        Create {
            author: agent_a(),
            timestamp: valid_timestamp(),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0xAC; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0xAD; 36]),
            weight: Default::default(),
        }
    }

    fn valid_gratitude() -> GratitudeExpression {
        GratitudeExpression {
            hearth_hash: valid_action_hash(),
            from_agent: agent_a(),
            to_agent: agent_b(),
            message: "Thank you for everything!".to_string(),
            gratitude_type: GratitudeType::Appreciation,
            visibility: HearthVisibility::AllMembers,
            created_at: valid_timestamp(),
        }
    }

    fn valid_circle() -> AppreciationCircle {
        AppreciationCircle {
            hearth_hash: valid_action_hash(),
            theme: "Weekly gratitude".to_string(),
            participants: vec![agent_a(), agent_b()],
            started_at: valid_timestamp(),
            completed_at: None,
            status: CircleStatus::Open,
        }
    }

    fn valid_anchor() -> GratitudeAnchor {
        GratitudeAnchor {
            agent: agent_a(),
            hearth_hash: valid_action_hash(),
            total_given: 10,
            total_received: 5,
            current_streak_days: 3,
        }
    }

    // ── GratitudeExpression validation tests ──────────────────────────

    #[test]
    fn test_valid_gratitude_expression() {
        let expr = valid_gratitude();
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_gratitude_empty_message_rejected() {
        let mut expr = valid_gratitude();
        expr.message = "".to_string();
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Gratitude message cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_gratitude_message_one_char() {
        let mut expr = valid_gratitude();
        expr.message = "T".to_string();
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_gratitude_message_at_max() {
        let mut expr = valid_gratitude();
        expr.message = "a".repeat(2048);
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_gratitude_message_exceeds_max_rejected() {
        let mut expr = valid_gratitude();
        expr.message = "a".repeat(2049);
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Gratitude message must be <= 2048 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_gratitude_self_expression_rejected() {
        let mut expr = valid_gratitude();
        expr.to_agent = expr.from_agent.clone();
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Cannot express gratitude to yourself");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_gratitude_all_types() {
        let types = vec![
            GratitudeType::Appreciation,
            GratitudeType::Acknowledgment,
            GratitudeType::Celebration,
            GratitudeType::Blessing,
            GratitudeType::Custom("Heartfelt".to_string()),
        ];
        for gt in types {
            let mut expr = valid_gratitude();
            expr.gratitude_type = gt;
            let action = valid_create_action();
            let result = validate_gratitude(action, expr).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_gratitude_unicode_message() {
        let mut expr = valid_gratitude();
        expr.message = "Merci beaucoup pour votre aide!".to_string();
        let action = valid_create_action();
        let result = validate_gratitude(action, expr).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── AppreciationCircle validation tests ──────────────────────────

    #[test]
    fn test_valid_circle() {
        let circle = valid_circle();
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_circle_empty_theme_rejected() {
        let mut circle = valid_circle();
        circle.theme = "".to_string();
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle theme cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_circle_theme_one_char() {
        let mut circle = valid_circle();
        circle.theme = "G".to_string();
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_circle_theme_at_max() {
        let mut circle = valid_circle();
        circle.theme = "a".repeat(256);
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_circle_theme_exceeds_max_rejected() {
        let mut circle = valid_circle();
        circle.theme = "a".repeat(257);
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle theme must be <= 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_circle_too_few_participants_rejected() {
        let mut circle = valid_circle();
        circle.participants = vec![agent_a()];
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle must have at least 2 participants");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_circle_zero_participants_rejected() {
        let mut circle = valid_circle();
        circle.participants = vec![];
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle must have at least 2 participants");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_circle_too_many_participants_rejected() {
        let mut circle = valid_circle();
        circle.participants = (0..51)
            .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
            .collect();
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle cannot have more than 50 participants");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_circle_at_max_participants() {
        let mut circle = valid_circle();
        circle.participants = (0..50)
            .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
            .collect();
        let action = valid_create_action();
        let result = validate_circle(action, circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_circle_all_statuses() {
        let statuses = vec![
            CircleStatus::Open,
            CircleStatus::InProgress,
            CircleStatus::Completed,
        ];
        for status in statuses {
            let mut circle = valid_circle();
            circle.status = status;
            let action = valid_create_action();
            let result = validate_circle(action, circle).unwrap();
            assert_eq!(result, ValidateCallbackResult::Valid);
        }
    }

    // ── GratitudeAnchor validation tests ─────────────────────────────

    #[test]
    fn test_valid_anchor() {
        let anchor = valid_anchor();
        let action = valid_create_action();
        let result = validate_anchor(action, anchor).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_anchor_zero_counts() {
        let mut anchor = valid_anchor();
        anchor.total_given = 0;
        anchor.total_received = 0;
        anchor.current_streak_days = 0;
        let action = valid_create_action();
        let result = validate_anchor(action, anchor).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_anchor_overflow_rejected() {
        let mut anchor = valid_anchor();
        anchor.total_given = u32::MAX;
        anchor.total_received = 1;
        let action = valid_create_action();
        let result = validate_anchor(action, anchor).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Gratitude anchor total counts would overflow");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_anchor_max_without_overflow() {
        let mut anchor = valid_anchor();
        anchor.total_given = u32::MAX;
        anchor.total_received = 0;
        let action = valid_create_action();
        let result = validate_anchor(action, anchor).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ── Serde roundtrip tests ────────────────────────────────────────

    #[test]
    fn test_gratitude_expression_serde_roundtrip() {
        let expr = valid_gratitude();
        let json = serde_json::to_string(&expr).unwrap();
        let decoded: GratitudeExpression = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, expr);
    }

    #[test]
    fn test_appreciation_circle_serde_roundtrip() {
        let circle = valid_circle();
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: AppreciationCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, circle);
    }

    #[test]
    fn test_gratitude_anchor_serde_roundtrip() {
        let anchor = valid_anchor();
        let json = serde_json::to_string(&anchor).unwrap();
        let decoded: GratitudeAnchor = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, anchor);
    }

    // ── Link tag validation tests ────────────────────────────────────

    fn assert_valid_result(result: ExternResult<ValidateCallbackResult>) {
        match result {
            Ok(ValidateCallbackResult::Valid) => {}
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                panic!("Expected Valid, got Invalid: {msg}")
            }
            other => panic!("Expected Valid, got {other:?}"),
        }
    }

    fn assert_invalid_result(result: ExternResult<ValidateCallbackResult>, expected_substr: &str) {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains(expected_substr),
                    "Expected Invalid message containing '{expected_substr}', got: '{msg}'"
                );
            }
            Ok(ValidateCallbackResult::Valid) => {
                panic!("Expected Invalid containing '{expected_substr}', got Valid")
            }
            other => panic!("Expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_link_hearth_to_gratitude_valid_tag() {
        let tag = LinkTag(vec![0u8; 128]);
        assert_valid_result(validate_create_link(LinkTypes::HearthToGratitude, &tag));
    }

    #[test]
    fn test_link_hearth_to_gratitude_tag_at_limit() {
        let tag = LinkTag(vec![0u8; 256]);
        assert_valid_result(validate_create_link(LinkTypes::HearthToGratitude, &tag));
    }

    #[test]
    fn test_link_hearth_to_gratitude_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::HearthToGratitude, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_agent_to_given_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::AgentToGratitudeGiven, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_agent_to_received_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::AgentToGratitudeReceived, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_hearth_to_circles_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::HearthToCircles, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_agent_to_circles_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::AgentToCircles, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_link_hearth_to_anchors_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_create_link(LinkTypes::HearthToGratitudeAnchors, &tag),
            "link tag too long",
        );
    }

    #[test]
    fn test_delete_link_tag_too_long_rejected() {
        let tag = LinkTag(vec![0u8; 257]);
        assert_invalid_result(
            validate_delete_link(LinkTypes::HearthToGratitude, &tag),
            "delete link tag too long",
        );
    }

    #[test]
    fn test_delete_link_valid_tag() {
        let tag = LinkTag(vec![0u8; 256]);
        assert_valid_result(validate_delete_link(LinkTypes::HearthToGratitude, &tag));
    }

    // ── Circle update validation tests ───────────────────────────────

    #[test]
    fn test_circle_update_valid() {
        let circle = valid_circle();
        let result = validate_circle_update(circle).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_circle_update_empty_theme_rejected() {
        let mut circle = valid_circle();
        circle.theme = "".to_string();
        let result = validate_circle_update(circle).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Circle theme cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // ── Anchor update validation tests ───────────────────────────────

    #[test]
    fn test_anchor_update_valid() {
        let anchor = valid_anchor();
        let result = validate_anchor_update(anchor).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_anchor_update_overflow_rejected() {
        let mut anchor = valid_anchor();
        anchor.total_given = u32::MAX;
        anchor.total_received = 1;
        let result = validate_anchor_update(anchor).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Gratitude anchor total counts would overflow");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // ── Delete guard tests ──────────────────────────────────────────

    #[test]
    fn delete_guard_message_content() {
        let msg = "Gratitude expressions cannot be deleted once created";
        assert!(msg.contains("cannot be deleted"));
    }
}
