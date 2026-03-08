//! Support Knowledge Coordinator Zome
//! Business logic for knowledge articles, resolutions, flags, and reputation.

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use support_knowledge_integrity::*;
use support_types::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

// ============================================================================
// BRIDGE SIGNAL (for cross-domain UI notification)
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BridgeEventSignal {
    pub event_type: String,
    pub source_zome: String,
    pub payload: String,
}

// ============================================================================
// INPUT TYPES
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateArticleInput {
    pub original_hash: ActionHash,
    pub updated: KnowledgeArticle,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeprecateInput {
    pub article_hash: ActionHash,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinkArticleInput {
    pub article_hash: ActionHash,
    pub ticket_hash: ActionHash,
    pub link_reason: LinkReason,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FindSimilarInput {
    pub keywords: Vec<String>,
    pub max_results: Option<usize>,
}

// ============================================================================
// HELPERS
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

fn extract_article(record: &Record) -> ExternResult<KnowledgeArticle> {
    record
        .entry()
        .to_app_option::<KnowledgeArticle>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Not a KnowledgeArticle".into()
        )))
}

// ============================================================================
// ARTICLE MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn create_article(article: KnowledgeArticle) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_article")?;
    let action_hash = create_entry(&EntryTypes::KnowledgeArticle(article.clone()))?;

    // Hash-sharded anchor for all_articles
    let entry_hash = hash_entry(&EntryTypes::KnowledgeArticle(article.clone()))?;
    let shard_anchor = hash_sharded_anchor("support", "articles", &entry_hash);
    create_entry(&EntryTypes::Anchor(Anchor(shard_anchor.clone())))?;
    create_link(
        anchor_hash(&shard_anchor)?,
        action_hash.clone(),
        LinkTypes::ShardedArticles,
        (),
    )?;

    // Category anchor
    let category_str = format!("support:category:{:?}", article.category);
    create_entry(&EntryTypes::Anchor(Anchor(category_str.clone())))?;
    create_link(
        anchor_hash(&category_str)?,
        action_hash.clone(),
        LinkTypes::CategoryToArticle,
        (),
    )?;

    // Tag anchors
    for tag in &article.tags {
        let tag_anchor = format!("support:tag:{}", tag);
        create_entry(&EntryTypes::Anchor(Anchor(tag_anchor.clone())))?;
        create_link(
            anchor_hash(&tag_anchor)?,
            action_hash.clone(),
            LinkTypes::TagToArticle,
            (),
        )?;
    }

    // Agent anchor
    create_link(
        article.author,
        action_hash.clone(),
        LinkTypes::AgentToArticle,
        (),
    )?;

    // Emit bridge signal
    let _ = emit_signal(&BridgeEventSignal {
        event_type: "article_created".to_string(),
        source_zome: "support_knowledge".to_string(),
        payload: format!(
            r#"{{"article_hash":"{}","title":"{}","category":"{:?}"}}"#,
            action_hash, article.title, article.category,
        ),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created article".into()
    )))
}

#[hdk_extern]
pub fn update_article(input: UpdateArticleInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "update_article")?;
    let _original = get(input.original_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Original article not found".into())
    ))?;

    let action_hash = update_entry(
        input.original_hash,
        &EntryTypes::KnowledgeArticle(input.updated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated article".into()
    )))
}

#[hdk_extern]
pub fn deprecate_article(input: DeprecateInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "deprecate_article")?;
    let original = get(input.article_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Article not found".into())
    ))?;

    let mut article = extract_article(&original)?;
    article.deprecated = true;
    article.deprecation_reason = Some(input.reason);

    let action_hash = update_entry(input.article_hash, &EntryTypes::KnowledgeArticle(article))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find deprecated article".into()
    )))
}

#[hdk_extern]
pub fn get_article(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

#[hdk_extern]
pub fn search_by_category(category: SupportCategory) -> ExternResult<Vec<Record>> {
    let category_str = format!("support:category:{:?}", category);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&category_str)?, LinkTypes::CategoryToArticle)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn search_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let tag_anchor = format!("support:tag:{}", tag);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&tag_anchor)?, LinkTypes::TagToArticle)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn list_recent_articles(_: ()) -> ExternResult<Vec<Record>> {
    // Collect from all 256 hash-sharded buckets
    let mut all_records = Vec::new();
    for byte in 0..=255u8 {
        let shard_anchor = format!("support:articles:{:02x}", byte);
        let links = get_links(
            LinkQuery::try_new(anchor_hash(&shard_anchor)?, LinkTypes::ShardedArticles)?,
            GetStrategy::default(),
        )?;
        let mut records = records_from_links(links)?;
        all_records.append(&mut records);
    }
    Ok(all_records)
}

#[hdk_extern]
pub fn upvote_article(action_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "upvote_article")?;
    let original = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Article not found".into())
    ))?;

    let mut article = extract_article(&original)?;
    article.upvotes += 1;

    let new_hash = update_entry(action_hash, &EntryTypes::KnowledgeArticle(article))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find upvoted article".into()
    )))
}

#[hdk_extern]
pub fn verify_article(action_hash: ActionHash) -> ExternResult<Record> {
    require_consciousness(&requirement_for_proposal(), "verify_article")?;
    let original = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Article not found".into())
    ))?;

    let mut article = extract_article(&original)?;
    article.verified = true;

    let new_hash = update_entry(action_hash, &EntryTypes::KnowledgeArticle(article))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find verified article".into()
    )))
}

// ============================================================================
// RESOLUTION MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn create_resolution(resolution: Resolution) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "create_resolution")?;
    let action_hash = create_entry(&EntryTypes::Resolution(resolution.clone()))?;
    create_link(
        resolution.ticket_hash,
        action_hash.clone(),
        LinkTypes::ArticleToResolution,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created resolution".into()
    )))
}

// ============================================================================
// FLAG MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn flag_article(flag: ArticleFlag) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "flag_article")?;
    let action_hash = create_entry(&EntryTypes::ArticleFlag(flag.clone()))?;
    create_link(
        flag.article_hash,
        action_hash.clone(),
        LinkTypes::ArticleToFlag,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created flag".into()
    )))
}

#[hdk_extern]
pub fn get_flags(article_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(article_hash, LinkTypes::ArticleToFlag)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// REPUTATION MANAGEMENT
// ============================================================================

#[hdk_extern]
pub fn get_agent_reputation(agent: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToReputation)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// ARTICLE-TICKET LINKING
// ============================================================================

#[hdk_extern]
pub fn link_article_to_ticket(input: LinkArticleInput) -> ExternResult<Record> {
    require_consciousness(&requirement_for_basic(), "link_article_to_ticket")?;
    let agent = agent_info()?.agent_initial_pubkey;
    let link = ArticleTicketLink {
        article_hash: input.article_hash.clone(),
        ticket_hash: input.ticket_hash.clone(),
        linked_by: agent,
        link_reason: input.link_reason,
        created_at: sys_time()?,
    };
    let action_hash = create_entry(&EntryTypes::ArticleTicketLink(link))?;
    create_link(
        input.article_hash,
        action_hash.clone(),
        LinkTypes::ArticleToTickets,
        (),
    )?;
    create_link(
        input.ticket_hash,
        action_hash.clone(),
        LinkTypes::TicketToArticles,
        (),
    )?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created article-ticket link".into()
    )))
}

#[hdk_extern]
pub fn get_suggested_articles(ticket_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(ticket_hash, LinkTypes::TicketToArticles)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[hdk_extern]
pub fn get_tickets_for_article(article_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(article_hash, LinkTypes::ArticleToTickets)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_event_signal_serde_roundtrip() {
        let signal = BridgeEventSignal {
            event_type: "article_created".to_string(),
            source_zome: "support_knowledge".to_string(),
            payload: r#"{"article_hash":"abc","title":"Test","category":"General"}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type, "article_created");
        assert_eq!(decoded.source_zome, "support_knowledge");
        assert!(decoded.payload.contains("Test"));
    }

    #[test]
    fn bridge_event_signal_empty_payload() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            source_zome: "support_knowledge".to_string(),
            payload: String::new(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.payload, "");
    }

    #[test]
    fn bridge_event_signal_json_structure() {
        let signal = BridgeEventSignal {
            event_type: "article_created".to_string(),
            source_zome: "support_knowledge".to_string(),
            payload: "{}".to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("\"event_type\""));
        assert!(json.contains("\"source_zome\""));
        assert!(json.contains("\"payload\""));
    }

    #[test]
    fn bridge_event_signal_unicode_payload() {
        let signal = BridgeEventSignal {
            event_type: "article_created".to_string(),
            source_zome: "support_knowledge".to_string(),
            payload: "{\"title\":\"\u{77e5}\u{8b58}\u{30d9}\u{30fc}\u{30b9}\"}".to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: BridgeEventSignal = serde_json::from_str(&json).unwrap();
        assert!(decoded.payload.contains("\u{77e5}"));
    }

    #[test]
    fn bridge_event_signal_clone_is_equal() {
        let signal = BridgeEventSignal {
            event_type: "test".to_string(),
            source_zome: "support_knowledge".to_string(),
            payload: "{}".to_string(),
        };
        let cloned = signal.clone();
        assert_eq!(cloned.event_type, signal.event_type);
        assert_eq!(cloned.source_zome, signal.source_zome);
        assert_eq!(cloned.payload, signal.payload);
    }

    #[test]
    fn update_article_input_serde_roundtrip() {
        let input = UpdateArticleInput {
            original_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated: KnowledgeArticle {
                title: "Updated title".into(),
                content: "Updated content".into(),
                category: SupportCategory::Network,
                tags: vec!["updated".into()],
                author: AgentPubKey::from_raw_36(vec![0xab; 36]),
                source: ArticleSource::Community,
                difficulty_level: DifficultyLevel::Advanced,
                upvotes: 5,
                verified: true,
                deprecated: false,
                deprecation_reason: None,
                version: 2,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateArticleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.updated.title, "Updated title");
        assert_eq!(decoded.updated.version, 2);
        assert!(decoded.updated.verified);
    }

    #[test]
    fn update_article_input_deprecated_article_roundtrip() {
        let input = UpdateArticleInput {
            original_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated: KnowledgeArticle {
                title: "Old article".into(),
                content: "Legacy content".into(),
                category: SupportCategory::Software,
                tags: vec![],
                author: AgentPubKey::from_raw_36(vec![0xab; 36]),
                source: ArticleSource::PreSeeded,
                difficulty_level: DifficultyLevel::Beginner,
                upvotes: 0,
                verified: false,
                deprecated: true,
                deprecation_reason: Some("Replaced by v2".into()),
                version: 1,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateArticleInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.updated.deprecated);
        assert_eq!(
            decoded.updated.deprecation_reason,
            Some("Replaced by v2".into())
        );
    }

    #[test]
    fn deprecate_input_serde_roundtrip() {
        let input = DeprecateInput {
            article_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            reason: "Superseded by newer article on the same topic".into(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeprecateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.reason,
            "Superseded by newer article on the same topic"
        );
    }

    #[test]
    fn deprecate_input_empty_reason_roundtrip() {
        let input = DeprecateInput {
            article_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            reason: String::new(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DeprecateInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.reason, "");
    }

    #[test]
    fn link_article_input_serde_roundtrip() {
        let input = LinkArticleInput {
            article_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            ticket_hash: ActionHash::from_raw_36(vec![0xcc; 36]),
            link_reason: LinkReason::SuggestedFAQ,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LinkArticleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.article_hash, input.article_hash);
        assert_eq!(decoded.ticket_hash, input.ticket_hash);
    }

    #[test]
    fn find_similar_input_serde_roundtrip() {
        let input = FindSimilarInput {
            keywords: vec!["holochain".into(), "networking".into()],
            max_results: Some(10),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FindSimilarInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.keywords, vec!["holochain", "networking"]);
        assert_eq!(decoded.max_results, Some(10));
    }

    #[test]
    fn find_similar_input_no_max_results_roundtrip() {
        let input = FindSimilarInput {
            keywords: vec!["test".into()],
            max_results: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FindSimilarInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.keywords, vec!["test"]);
        assert_eq!(decoded.max_results, None);
    }
}
