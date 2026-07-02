// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Support Knowledge Integrity Zome
//! Entry types and validation for knowledge articles, resolutions, flags, and reputation.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};
use support_types::*;

// ============================================================================
// ANCHOR
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// KNOWLEDGE ARTICLE
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KnowledgeArticle {
    pub title: String,
    pub content: String,
    pub category: SupportCategory,
    pub tags: Vec<String>,
    pub author: AgentPubKey,
    pub source: ArticleSource,
    pub difficulty_level: DifficultyLevel,
    pub upvotes: u32,
    pub verified: bool,
    pub deprecated: bool,
    pub deprecation_reason: Option<String>,
    pub version: u32,
}

// ============================================================================
// RESOLUTION
// ============================================================================

/// A verified resolution linking a ticket to its solution steps.
///
/// ## Signature Design: Manual Ed25519 (not Holochain countersigning)
///
/// Resolutions carry `helper_signature` and `requester_signature` as raw
/// Ed25519 byte vectors rather than using Holochain's built-in countersigning
/// API. This is intentional:
///
/// - **Countersigning requires both parties online simultaneously** to complete
///   the preflight handshake. Support workflows are inherently async — a helper
///   may resolve a ticket hours before the requester confirms.
/// - **Manual signatures are verifiable offline**: any agent can verify
///   `ed25519_verify(agent_pubkey, canonical_bytes, signature)` without
///   needing network round-trips.
/// - **Future TODO**: When Holochain ships async countersigning (RFC pending),
///   migrate to native countersigned entries for stronger DHT consistency
///   guarantees. The manual pattern remains correct for the async case.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Resolution {
    pub ticket_hash: ActionHash,
    pub steps: Vec<String>,
    pub root_cause: Option<String>,
    pub time_to_resolve_mins: Option<u32>,
    pub effectiveness_rating: Option<u8>,
    pub helper: AgentPubKey,
    pub requester: AgentPubKey,
    pub anonymized: bool,
    pub helper_signature: Vec<u8>,
    pub requester_signature: Vec<u8>,
}

// ============================================================================
// ARTICLE FLAG
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArticleFlag {
    pub article_hash: ActionHash,
    pub flagger: AgentPubKey,
    pub reason: FlagReason,
    pub description: String,
    pub created_at: Timestamp,
}

// ============================================================================
// REPUTATION RECORD
// ============================================================================

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReputationRecord {
    pub agent: AgentPubKey,
    pub event: ReputationEvent,
    pub reference_hash: ActionHash,
    pub points: i32,
    pub created_at: Timestamp,
}

// ============================================================================
// ARTICLE-TICKET LINKING
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LinkReason {
    SuggestedFAQ,
    DuplicateResolution,
    RelatedKnowledge,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArticleTicketLink {
    pub article_hash: ActionHash,
    pub ticket_hash: ActionHash,
    pub linked_by: AgentPubKey,
    pub link_reason: LinkReason,
    pub created_at: Timestamp,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    KnowledgeArticle(KnowledgeArticle),
    Resolution(Resolution),
    ArticleFlag(ArticleFlag),
    ReputationRecord(ReputationRecord),
    ArticleTicketLink(ArticleTicketLink),
}

#[hdk_link_types]
pub enum LinkTypes {
    ShardedArticles,
    CategoryToArticle,
    TagToArticle,
    AgentToArticle,
    ArticleToResolution,
    ArticleToFlag,
    AgentToReputation,
    ArticleToTickets,
    TicketToArticles,
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
                EntryTypes::KnowledgeArticle(a) => validate_article(a),
                EntryTypes::Resolution(r) => validate_resolution(r),
                EntryTypes::ArticleFlag(f) => validate_flag(f),
                EntryTypes::ReputationRecord(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ArticleTicketLink(l) => validate_article_ticket_link(l),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::KnowledgeArticle(a) => validate_article(a),
                EntryTypes::Resolution(r) => validate_resolution(r),
                EntryTypes::ArticleFlag(f) => validate_flag(f),
                EntryTypes::ReputationRecord(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ArticleTicketLink(l) => validate_article_ticket_link(l),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::ShardedArticles => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ShardedArticles link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::CategoryToArticle => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "CategoryToArticle link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TagToArticle => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TagToArticle link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToArticle => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToArticle link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ArticleToResolution => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ArticleToResolution link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ArticleToFlag => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ArticleToFlag link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToReputation => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToReputation link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ArticleToTickets => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ArticleToTickets link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::TicketToArticles => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "TicketToArticles link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
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

fn validate_article(a: KnowledgeArticle) -> ExternResult<ValidateCallbackResult> {
    if a.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Article title cannot be empty".into(),
        ));
    }
    if a.title.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Article title too long (max 256 chars)".into(),
        ));
    }
    if a.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Article content cannot be empty".into(),
        ));
    }
    if a.content.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "Article content too long (max 65536 chars)".into(),
        ));
    }
    if a.tags.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Article cannot have more than 20 tags".into(),
        ));
    }
    for tag in &a.tags {
        if tag.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Tag too long (max 256 chars per item)".into(),
            ));
        }
    }
    if let Some(ref reason) = a.deprecation_reason {
        if reason.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Deprecation reason too long (max 4096 chars)".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_resolution(r: Resolution) -> ExternResult<ValidateCallbackResult> {
    if r.steps.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resolution must have at least one step".into(),
        ));
    }
    for step in &r.steps {
        if step.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Resolution step too long (max 4096 chars per item)".into(),
            ));
        }
    }
    if let Some(ref root_cause) = r.root_cause {
        if root_cause.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Root cause too long (max 4096 chars)".into(),
            ));
        }
    }
    if let Some(rating) = r.effectiveness_rating {
        if !(1..=5).contains(&rating) {
            return Ok(ValidateCallbackResult::Invalid(
                "Effectiveness rating must be between 1 and 5".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_flag(f: ArticleFlag) -> ExternResult<ValidateCallbackResult> {
    if f.description.len() < 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Flag description must be at least 10 characters".into(),
        ));
    }
    if f.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Flag description too long (max 4096 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_article_ticket_link(_l: ArticleTicketLink) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn test_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xab; 36])
    }

    fn test_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xdb; 36])
    }

    fn test_timestamp() -> Timestamp {
        Timestamp::from_micros(1_700_000_000_000_000)
    }

    fn valid_article() -> KnowledgeArticle {
        KnowledgeArticle {
            title: "How to configure Holochain networking".into(),
            content: "## Step 1\nInstall the conductor...".into(),
            category: SupportCategory::Holochain,
            tags: vec!["holochain".into(), "networking".into()],
            author: test_agent(),
            source: ArticleSource::Community,
            difficulty_level: DifficultyLevel::Intermediate,
            upvotes: 0,
            verified: false,
            deprecated: false,
            deprecation_reason: None,
            version: 1,
        }
    }

    fn valid_resolution() -> Resolution {
        Resolution {
            ticket_hash: test_action_hash(),
            steps: vec![
                "Restart the conductor".into(),
                "Clear the lair keystore cache".into(),
            ],
            root_cause: Some("Stale keystore lock file".into()),
            time_to_resolve_mins: Some(15),
            effectiveness_rating: Some(4),
            helper: test_agent(),
            requester: AgentPubKey::from_raw_36(vec![0xbc; 36]),
            anonymized: false,
            helper_signature: vec![0x01; 64],
            requester_signature: vec![0x02; 64],
        }
    }

    fn valid_flag() -> ArticleFlag {
        ArticleFlag {
            article_hash: test_action_hash(),
            flagger: test_agent(),
            reason: FlagReason::Outdated,
            description: "This article references Holochain 0.1 which is no longer supported"
                .into(),
            created_at: test_timestamp(),
        }
    }

    fn valid_reputation() -> ReputationRecord {
        ReputationRecord {
            agent: test_agent(),
            event: ReputationEvent::HelpProvided,
            reference_hash: test_action_hash(),
            points: 10,
            created_at: test_timestamp(),
        }
    }

    fn valid_article_ticket_link() -> ArticleTicketLink {
        ArticleTicketLink {
            article_hash: test_action_hash(),
            ticket_hash: ActionHash::from_raw_36(vec![0xcc; 36]),
            linked_by: test_agent(),
            link_reason: LinkReason::SuggestedFAQ,
            created_at: test_timestamp(),
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
    fn serde_roundtrip_anchor() {
        let a = Anchor("all_articles".to_string());
        let json = serde_json::to_string(&a).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn serde_roundtrip_knowledge_article() {
        let a = valid_article();
        let json = serde_json::to_string(&a).unwrap();
        let back: KnowledgeArticle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn serde_roundtrip_knowledge_article_minimal() {
        let a = KnowledgeArticle {
            title: "Test".into(),
            content: "Content".into(),
            category: SupportCategory::General,
            tags: vec![],
            author: test_agent(),
            source: ArticleSource::PreSeeded,
            difficulty_level: DifficultyLevel::Beginner,
            upvotes: 0,
            verified: false,
            deprecated: false,
            deprecation_reason: None,
            version: 1,
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: KnowledgeArticle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn serde_roundtrip_knowledge_article_deprecated() {
        let mut a = valid_article();
        a.deprecated = true;
        a.deprecation_reason = Some("Superseded by new article".into());
        let json = serde_json::to_string(&a).unwrap();
        let back: KnowledgeArticle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
        assert!(back.deprecated);
        assert_eq!(
            back.deprecation_reason,
            Some("Superseded by new article".into())
        );
    }

    #[test]
    fn serde_roundtrip_resolution() {
        let r = valid_resolution();
        let json = serde_json::to_string(&r).unwrap();
        let back: Resolution = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_resolution_minimal() {
        let r = Resolution {
            ticket_hash: test_action_hash(),
            steps: vec!["Reboot".into()],
            root_cause: None,
            time_to_resolve_mins: None,
            effectiveness_rating: None,
            helper: test_agent(),
            requester: test_agent(),
            anonymized: true,
            helper_signature: vec![],
            requester_signature: vec![],
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: Resolution = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_article_flag() {
        let f = valid_flag();
        let json = serde_json::to_string(&f).unwrap();
        let back: ArticleFlag = serde_json::from_str(&json).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn serde_roundtrip_reputation_record() {
        let r = valid_reputation();
        let json = serde_json::to_string(&r).unwrap();
        let back: ReputationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_reputation_negative_points() {
        let mut r = valid_reputation();
        r.points = -5;
        r.event = ReputationEvent::ArticleFlagged;
        let json = serde_json::to_string(&r).unwrap();
        let back: ReputationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.points, -5);
    }

    // ── validate_article: title ─────────────────────────────────────────

    #[test]
    fn valid_article_passes() {
        assert_valid(validate_article(valid_article()));
    }

    #[test]
    fn article_empty_title_rejected() {
        let mut a = valid_article();
        a.title = String::new();
        assert_invalid(validate_article(a), "Article title cannot be empty");
    }

    #[test]
    fn article_whitespace_title_rejected() {
        let mut a = valid_article();
        a.title = "   ".into();
        assert_invalid(validate_article(a), "Article title cannot be empty");
    }

    // ── validate_article: content ───────────────────────────────────────

    #[test]
    fn article_empty_content_rejected() {
        let mut a = valid_article();
        a.content = String::new();
        assert_invalid(validate_article(a), "Article content cannot be empty");
    }

    #[test]
    fn article_whitespace_content_rejected() {
        let mut a = valid_article();
        a.content = "  ".into();
        assert_invalid(validate_article(a), "Article content cannot be empty");
    }

    // ── validate_article: tags ──────────────────────────────────────────

    #[test]
    fn article_20_tags_valid() {
        let mut a = valid_article();
        a.tags = (0..20).map(|i| format!("tag-{i}")).collect();
        assert_valid(validate_article(a));
    }

    #[test]
    fn article_21_tags_rejected() {
        let mut a = valid_article();
        a.tags = (0..21).map(|i| format!("tag-{i}")).collect();
        assert_invalid(validate_article(a), "Article cannot have more than 20 tags");
    }

    #[test]
    fn article_zero_tags_valid() {
        let mut a = valid_article();
        a.tags = vec![];
        assert_valid(validate_article(a));
    }

    // ── validate_article: combined invalid ──────────────────────────────

    #[test]
    fn article_empty_title_and_content_rejects_title_first() {
        let mut a = valid_article();
        a.title = String::new();
        a.content = String::new();
        assert_invalid(validate_article(a), "Article title cannot be empty");
    }

    #[test]
    fn article_empty_content_and_too_many_tags_rejects_content_first() {
        let mut a = valid_article();
        a.content = String::new();
        a.tags = (0..21).map(|i| format!("tag-{i}")).collect();
        assert_invalid(validate_article(a), "Article content cannot be empty");
    }

    // ── validate_article: optional / misc fields ────────────────────────

    #[test]
    fn article_all_sources_valid() {
        for source in [
            ArticleSource::Community,
            ArticleSource::PreSeeded,
            ArticleSource::SymthaeaGenerated,
        ] {
            let mut a = valid_article();
            a.source = source;
            assert_valid(validate_article(a));
        }
    }

    #[test]
    fn article_all_difficulty_levels_valid() {
        for level in [
            DifficultyLevel::Beginner,
            DifficultyLevel::Intermediate,
            DifficultyLevel::Advanced,
        ] {
            let mut a = valid_article();
            a.difficulty_level = level;
            assert_valid(validate_article(a));
        }
    }

    #[test]
    fn article_all_categories_valid() {
        for cat in [
            SupportCategory::Network,
            SupportCategory::Hardware,
            SupportCategory::Software,
            SupportCategory::Holochain,
            SupportCategory::Mycelix,
            SupportCategory::Security,
            SupportCategory::General,
        ] {
            let mut a = valid_article();
            a.category = cat;
            assert_valid(validate_article(a));
        }
    }

    #[test]
    fn article_verified_and_deprecated_valid() {
        let mut a = valid_article();
        a.verified = true;
        a.deprecated = true;
        a.deprecation_reason = Some("Replaced".into());
        a.upvotes = 42;
        a.version = 5;
        assert_valid(validate_article(a));
    }

    // ── validate_resolution: steps ──────────────────────────────────────

    #[test]
    fn valid_resolution_passes() {
        assert_valid(validate_resolution(valid_resolution()));
    }

    #[test]
    fn resolution_empty_steps_rejected() {
        let mut r = valid_resolution();
        r.steps = vec![];
        assert_invalid(
            validate_resolution(r),
            "Resolution must have at least one step",
        );
    }

    #[test]
    fn resolution_one_step_valid() {
        let mut r = valid_resolution();
        r.steps = vec!["Fix it".into()];
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_many_steps_valid() {
        let mut r = valid_resolution();
        r.steps = (0..50).map(|i| format!("Step {i}")).collect();
        assert_valid(validate_resolution(r));
    }

    // ── validate_resolution: effectiveness_rating ───────────────────────

    #[test]
    fn resolution_rating_none_valid() {
        let mut r = valid_resolution();
        r.effectiveness_rating = None;
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_rating_1_valid() {
        let mut r = valid_resolution();
        r.effectiveness_rating = Some(1);
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_rating_5_valid() {
        let mut r = valid_resolution();
        r.effectiveness_rating = Some(5);
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_rating_0_rejected() {
        let mut r = valid_resolution();
        r.effectiveness_rating = Some(0);
        assert_invalid(
            validate_resolution(r),
            "Effectiveness rating must be between 1 and 5",
        );
    }

    #[test]
    fn resolution_rating_6_rejected() {
        let mut r = valid_resolution();
        r.effectiveness_rating = Some(6);
        assert_invalid(
            validate_resolution(r),
            "Effectiveness rating must be between 1 and 5",
        );
    }

    #[test]
    fn resolution_rating_255_rejected() {
        let mut r = valid_resolution();
        r.effectiveness_rating = Some(255);
        assert_invalid(
            validate_resolution(r),
            "Effectiveness rating must be between 1 and 5",
        );
    }

    // ── validate_resolution: combined invalid ───────────────────────────

    #[test]
    fn resolution_empty_steps_and_bad_rating_rejects_steps_first() {
        let mut r = valid_resolution();
        r.steps = vec![];
        r.effectiveness_rating = Some(0);
        assert_invalid(
            validate_resolution(r),
            "Resolution must have at least one step",
        );
    }

    // ── validate_resolution: optional fields ────────────────────────────

    #[test]
    fn resolution_no_root_cause_valid() {
        let mut r = valid_resolution();
        r.root_cause = None;
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_no_time_valid() {
        let mut r = valid_resolution();
        r.time_to_resolve_mins = None;
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_anonymized_valid() {
        let mut r = valid_resolution();
        r.anonymized = true;
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_empty_signatures_valid() {
        let mut r = valid_resolution();
        r.helper_signature = vec![];
        r.requester_signature = vec![];
        assert_valid(validate_resolution(r));
    }

    // ── validate_flag: description ──────────────────────────────────────

    #[test]
    fn valid_flag_passes() {
        assert_valid(validate_flag(valid_flag()));
    }

    #[test]
    fn flag_short_description_rejected() {
        let mut f = valid_flag();
        f.description = "Too short".into(); // 9 chars
        assert_invalid(
            validate_flag(f),
            "Flag description must be at least 10 characters",
        );
    }

    #[test]
    fn flag_exactly_10_chars_valid() {
        let mut f = valid_flag();
        f.description = "0123456789".into(); // exactly 10
        assert_valid(validate_flag(f));
    }

    #[test]
    fn flag_9_chars_rejected() {
        let mut f = valid_flag();
        f.description = "012345678".into(); // 9 chars
        assert_invalid(
            validate_flag(f),
            "Flag description must be at least 10 characters",
        );
    }

    #[test]
    fn flag_empty_description_rejected() {
        let mut f = valid_flag();
        f.description = String::new();
        assert_invalid(
            validate_flag(f),
            "Flag description must be at least 10 characters",
        );
    }

    #[test]
    fn flag_long_description_valid() {
        let mut f = valid_flag();
        f.description = "x".repeat(1000);
        assert_valid(validate_flag(f));
    }

    // ── validate_flag: all reasons ──────────────────────────────────────

    #[test]
    fn flag_all_reasons_valid() {
        for reason in [
            FlagReason::Harmful,
            FlagReason::Incorrect,
            FlagReason::Outdated,
            FlagReason::Spam,
        ] {
            let mut f = valid_flag();
            f.reason = reason;
            assert_valid(validate_flag(f));
        }
    }

    // ── ReputationRecord: validation (always valid) ─────────────────────

    #[test]
    fn reputation_record_valid() {
        // ReputationRecord has no custom validation rules, always valid
        let r = valid_reputation();
        let json = serde_json::to_string(&r).unwrap();
        let back: ReputationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ── Update validation tests ─────────────────────────────────────────

    #[test]
    fn update_article_invalid_title_rejected() {
        let mut a = valid_article();
        a.title = String::new();
        assert_invalid(validate_article(a), "Article title cannot be empty");
    }

    #[test]
    fn update_article_invalid_content_rejected() {
        let mut a = valid_article();
        a.content = "  ".into();
        assert_invalid(validate_article(a), "Article content cannot be empty");
    }

    #[test]
    fn update_article_too_many_tags_rejected() {
        let mut a = valid_article();
        a.tags = (0..21).map(|i| format!("tag-{i}")).collect();
        assert_invalid(validate_article(a), "Article cannot have more than 20 tags");
    }

    #[test]
    fn update_resolution_empty_steps_rejected() {
        let mut r = valid_resolution();
        r.steps = vec![];
        assert_invalid(
            validate_resolution(r),
            "Resolution must have at least one step",
        );
    }

    #[test]
    fn update_resolution_bad_rating_rejected() {
        let mut r = valid_resolution();
        r.effectiveness_rating = Some(0);
        assert_invalid(
            validate_resolution(r),
            "Effectiveness rating must be between 1 and 5",
        );
    }

    #[test]
    fn update_flag_short_description_rejected() {
        let mut f = valid_flag();
        f.description = "short".into();
        assert_invalid(
            validate_flag(f),
            "Flag description must be at least 10 characters",
        );
    }

    // ── Link tag length validation tests ────────────────────────────────

    fn validate_link_tag(link_type: &LinkTypes, tag_len: usize) -> ValidateCallbackResult {
        let tag = LinkTag(vec![0u8; tag_len]);
        let max = match link_type {
            LinkTypes::ShardedArticles
            | LinkTypes::CategoryToArticle
            | LinkTypes::AgentToArticle
            | LinkTypes::ArticleToResolution
            | LinkTypes::ArticleToFlag
            | LinkTypes::AgentToReputation
            | LinkTypes::ArticleToTickets
            | LinkTypes::TicketToArticles => 256,
            LinkTypes::TagToArticle => 512,
        };
        let name = match link_type {
            LinkTypes::ShardedArticles => "ShardedArticles",
            LinkTypes::CategoryToArticle => "CategoryToArticle",
            LinkTypes::TagToArticle => "TagToArticle",
            LinkTypes::AgentToArticle => "AgentToArticle",
            LinkTypes::ArticleToResolution => "ArticleToResolution",
            LinkTypes::ArticleToFlag => "ArticleToFlag",
            LinkTypes::AgentToReputation => "AgentToReputation",
            LinkTypes::ArticleToTickets => "ArticleToTickets",
            LinkTypes::TicketToArticles => "TicketToArticles",
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
    fn link_sharded_articles_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::ShardedArticles, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_sharded_articles_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::ShardedArticles, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_category_to_article_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::CategoryToArticle, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_category_to_article_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::CategoryToArticle, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_tag_to_article_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::TagToArticle, 512);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_tag_to_article_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::TagToArticle, 513);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_agent_to_article_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToArticle, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_agent_to_article_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToArticle, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_article_to_resolution_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::ArticleToResolution, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_article_to_resolution_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::ArticleToResolution, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_article_to_flag_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::ArticleToFlag, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_article_to_flag_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::ArticleToFlag, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn link_agent_to_reputation_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::AgentToReputation, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_agent_to_reputation_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::AgentToReputation, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Serde roundtrip tests: LinkReason ──────────────────────────────

    #[test]
    fn serde_roundtrip_link_reason_suggested_faq() {
        let r = LinkReason::SuggestedFAQ;
        let json = serde_json::to_string(&r).unwrap();
        let back: LinkReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_link_reason_duplicate_resolution() {
        let r = LinkReason::DuplicateResolution;
        let json = serde_json::to_string(&r).unwrap();
        let back: LinkReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn serde_roundtrip_link_reason_related_knowledge() {
        let r = LinkReason::RelatedKnowledge;
        let json = serde_json::to_string(&r).unwrap();
        let back: LinkReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ── Serde roundtrip tests: ArticleTicketLink ───────────────────────

    #[test]
    fn serde_roundtrip_article_ticket_link() {
        let l = valid_article_ticket_link();
        let json = serde_json::to_string(&l).unwrap();
        let back: ArticleTicketLink = serde_json::from_str(&json).unwrap();
        assert_eq!(back, l);
    }

    // ── validate_article_ticket_link ───────────────────────────────────

    #[test]
    fn valid_article_ticket_link_passes() {
        assert_valid(validate_article_ticket_link(valid_article_ticket_link()));
    }

    // ── Link tag tests: ArticleToTickets ───────────────────────────────

    #[test]
    fn link_article_to_tickets_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::ArticleToTickets, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_article_to_tickets_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::ArticleToTickets, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── Link tag tests: TicketToArticles ───────────────────────────────

    #[test]
    fn link_ticket_to_articles_tag_at_max_accepted() {
        let result = validate_link_tag(&LinkTypes::TicketToArticles, 256);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_ticket_to_articles_tag_over_max_rejected() {
        let result = validate_link_tag(&LinkTypes::TicketToArticles, 257);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ── String length boundary tests ──────────────────────────────────

    #[test]
    fn article_title_at_limit_accepted() {
        let mut a = valid_article();
        a.title = "x".repeat(256);
        assert_valid(validate_article(a));
    }

    #[test]
    fn article_title_over_limit_rejected() {
        let mut a = valid_article();
        a.title = "x".repeat(257);
        assert_invalid(validate_article(a), "Article title too long");
    }

    #[test]
    fn article_content_at_limit_accepted() {
        let mut a = valid_article();
        a.content = "x".repeat(65536);
        assert_valid(validate_article(a));
    }

    #[test]
    fn article_content_over_limit_rejected() {
        let mut a = valid_article();
        a.content = "x".repeat(65537);
        assert_invalid(validate_article(a), "Article content too long");
    }

    #[test]
    fn article_tag_at_limit_accepted() {
        let mut a = valid_article();
        a.tags = vec!["x".repeat(256)];
        assert_valid(validate_article(a));
    }

    #[test]
    fn article_tag_over_limit_rejected() {
        let mut a = valid_article();
        a.tags = vec!["x".repeat(257)];
        assert_invalid(validate_article(a), "Tag too long");
    }

    #[test]
    fn article_deprecation_reason_at_limit_accepted() {
        let mut a = valid_article();
        a.deprecated = true;
        a.deprecation_reason = Some("x".repeat(4096));
        assert_valid(validate_article(a));
    }

    #[test]
    fn article_deprecation_reason_over_limit_rejected() {
        let mut a = valid_article();
        a.deprecated = true;
        a.deprecation_reason = Some("x".repeat(4097));
        assert_invalid(validate_article(a), "Deprecation reason too long");
    }

    #[test]
    fn resolution_step_at_limit_accepted() {
        let mut r = valid_resolution();
        r.steps = vec!["x".repeat(4096)];
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_step_over_limit_rejected() {
        let mut r = valid_resolution();
        r.steps = vec!["x".repeat(4097)];
        assert_invalid(validate_resolution(r), "Resolution step too long");
    }

    #[test]
    fn resolution_root_cause_at_limit_accepted() {
        let mut r = valid_resolution();
        r.root_cause = Some("x".repeat(4096));
        assert_valid(validate_resolution(r));
    }

    #[test]
    fn resolution_root_cause_over_limit_rejected() {
        let mut r = valid_resolution();
        r.root_cause = Some("x".repeat(4097));
        assert_invalid(validate_resolution(r), "Root cause too long");
    }

    #[test]
    fn flag_description_at_limit_accepted() {
        let mut f = valid_flag();
        f.description = "x".repeat(4096);
        assert_valid(validate_flag(f));
    }

    #[test]
    fn flag_description_over_limit_rejected() {
        let mut f = valid_flag();
        f.description = "x".repeat(4097);
        assert_invalid(validate_flag(f), "Flag description too long");
    }
}
