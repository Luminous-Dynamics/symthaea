//! Publication Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Publication {
    pub id: String,
    pub title: String,
    pub content_hash: String,
    pub content_type: ContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: License,
    pub encrypted: bool,
    pub published: Timestamp,
    pub updated: Option<Timestamp>,
    pub version: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContentType {
    Article,
    Opinion,
    Investigation,
    Review,
    Analysis,
    Interview,
    Report,
    Editorial,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct License {
    pub license_type: LicenseType,
    pub attribution_required: bool,
    pub commercial_use: bool,
    pub derivative_works: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LicenseType {
    CC0,
    CCBY,
    CCBYSA,
    CCBYNC,
    CCBYNCSA,
    AllRightsReserved,
    Custom(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContentBlock {
    pub publication_id: String,
    pub block_index: u32,
    pub content: String,
    pub encrypted_content: Option<String>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PublicationVersion {
    pub publication_id: String,
    pub version: u32,
    pub content_hash: String,
    pub change_summary: String,
    pub created: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    Publication(Publication),
    ContentBlock(ContentBlock),
    PublicationVersion(PublicationVersion),
}

#[hdk_link_types]
pub enum LinkTypes {
    AuthorToPublications,
    TagToPublications,
    PublicationToBlocks,
    PublicationToVersions,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Publication(publication) => {
                    validate_create_publication(EntryCreationAction::Create(action), publication)
                }
                EntryTypes::ContentBlock(block) => {
                    validate_create_content_block(EntryCreationAction::Create(action), block)
                }
                EntryTypes::PublicationVersion(version) => validate_create_publication_version(
                    EntryCreationAction::Create(action),
                    version,
                ),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Publication(publication) => {
                    validate_update_publication(action, publication)
                }
                EntryTypes::ContentBlock(_) => Ok(ValidateCallbackResult::Invalid(
                    "Content blocks cannot be updated".into(),
                )),
                EntryTypes::PublicationVersion(_) => Ok(ValidateCallbackResult::Invalid(
                    "Publication versions cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::AuthorToPublications
                | LinkTypes::TagToPublications
                | LinkTypes::PublicationToBlocks
                | LinkTypes::PublicationToVersions => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { action, tag, .. } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            let original_action = must_get_action(action.link_add_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_create_publication(
    _action: EntryCreationAction,
    publication: Publication,
) -> ExternResult<ValidateCallbackResult> {
    if !publication.author_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Author must be a valid DID".into(),
        ));
    }
    if publication.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Title cannot be empty".into(),
        ));
    }
    if publication.content_hash.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash required".into(),
        ));
    }

    // String length limits
    if publication.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication ID too long (max 256)".into(),
        ));
    }
    if publication.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication title too long (max 512)".into(),
        ));
    }
    if publication.content_hash.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash too long (max 256)".into(),
        ));
    }
    if publication.author_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Author DID too long (max 256)".into(),
        ));
    }
    if publication.language.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Language too long (max 128)".into(),
        ));
    }

    // Vec length limits
    if publication.co_authors.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many co-authors (max 50)".into(),
        ));
    }
    if publication.tags.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many tags (max 100)".into(),
        ));
    }

    // Validate co-author DIDs
    for co_author in &publication.co_authors {
        if co_author.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Co-author DID too long (max 256)".into(),
            ));
        }
    }

    // Validate tag lengths
    for tag in &publication.tags {
        if tag.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Tag too long (max 256)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_publication(
    _action: Update,
    publication: Publication,
) -> ExternResult<ValidateCallbackResult> {
    if publication.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Title cannot be empty".into(),
        ));
    }

    // String length limits
    if publication.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication ID too long (max 256)".into(),
        ));
    }
    if publication.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication title too long (max 512)".into(),
        ));
    }
    if publication.content_hash.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash too long (max 256)".into(),
        ));
    }
    if publication.author_did.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Author DID too long (max 256)".into(),
        ));
    }
    if publication.language.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Language too long (max 128)".into(),
        ));
    }

    // Vec length limits
    if publication.co_authors.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many co-authors (max 50)".into(),
        ));
    }
    if publication.tags.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many tags (max 100)".into(),
        ));
    }

    // Validate co-author DIDs
    for co_author in &publication.co_authors {
        if co_author.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Co-author DID too long (max 256)".into(),
            ));
        }
    }

    // Validate tag lengths
    for tag in &publication.tags {
        if tag.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Tag too long (max 256)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_content_block(
    _action: EntryCreationAction,
    block: ContentBlock,
) -> ExternResult<ValidateCallbackResult> {
    if block.publication_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content block publication_id cannot be empty".into(),
        ));
    }
    if block.content.trim().is_empty() && block.encrypted_content.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content block must have content or encrypted_content".into(),
        ));
    }

    // String length limits
    if block.publication_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Content block publication_id too long (max 256)".into(),
        ));
    }
    if block.content.len() > 1_048_576 {
        return Ok(ValidateCallbackResult::Invalid(
            "Content block content too long (max 1048576)".into(),
        ));
    }
    if let Some(ref encrypted) = block.encrypted_content {
        if encrypted.len() > 1_048_576 {
            return Ok(ValidateCallbackResult::Invalid(
                "Content block encrypted_content too long (max 1048576)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_publication_version(
    _action: EntryCreationAction,
    version: PublicationVersion,
) -> ExternResult<ValidateCallbackResult> {
    if version.publication_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication version publication_id cannot be empty".into(),
        ));
    }
    if version.content_hash.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication version content_hash cannot be empty".into(),
        ));
    }
    if version.version == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication version number must be at least 1".into(),
        ));
    }

    // String length limits
    if version.publication_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication version publication_id too long (max 256)".into(),
        ));
    }
    if version.content_hash.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication version content_hash too long (max 256)".into(),
        ));
    }
    if version.change_summary.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Publication version change_summary too long (max 4096)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helpers
    fn fake_create(entry_hash: EntryHash, author: AgentPubKey) -> Create {
        Create {
            author,
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash,
            weight: Default::default(),
        }
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn is_valid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn get_invalid_reason(result: ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(reason)) => reason,
            _ => panic!("Expected Invalid result"),
        }
    }

    // Factory functions
    fn valid_publication() -> Publication {
        Publication {
            id: "pub-123".into(),
            title: "Test Publication".into(),
            content_hash: "QmHash123".into(),
            content_type: ContentType::Article,
            author_did: "did:mycelix:author123".into(),
            co_authors: vec![],
            language: "en".into(),
            tags: vec!["test".into()],
            license: License {
                license_type: LicenseType::CCBY,
                attribution_required: true,
                commercial_use: true,
                derivative_works: true,
            },
            encrypted: false,
            published: ts(),
            updated: None,
            version: 1,
        }
    }

    fn valid_content_block() -> ContentBlock {
        ContentBlock {
            publication_id: "pub-123".into(),
            block_index: 0,
            content: "Test content".into(),
            encrypted_content: None,
        }
    }

    fn valid_publication_version() -> PublicationVersion {
        PublicationVersion {
            publication_id: "pub-123".into(),
            version: 2,
            content_hash: "QmNewHash456".into(),
            change_summary: "Updated content".into(),
            created: ts(),
        }
    }

    fn valid_anchor() -> Anchor {
        Anchor("test_anchor".into())
    }

    // Publication validation tests
    #[test]
    fn valid_publication_passes() {
        let pub_data = valid_publication();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_requires_did_prefix() {
        let mut pub_data = valid_publication();
        pub_data.author_did = "not-a-did".into();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result.clone()));
        assert_eq!(get_invalid_reason(result), "Author must be a valid DID");
    }

    #[test]
    fn publication_requires_non_empty_title() {
        let mut pub_data = valid_publication();
        pub_data.title = "".into();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result.clone()));
        assert_eq!(get_invalid_reason(result), "Title cannot be empty");
    }

    #[test]
    fn publication_requires_content_hash() {
        let mut pub_data = valid_publication();
        pub_data.content_hash = "".into();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result.clone()));
        assert_eq!(get_invalid_reason(result), "Content hash required");
    }

    #[test]
    fn publication_accepts_all_did_formats() {
        let test_dids = vec![
            "did:mycelix:123",
            "did:key:z6Mk",
            "did:web:example.com",
            "did:ethr:0x123",
        ];

        for did in test_dids {
            let mut pub_data = valid_publication();
            pub_data.author_did = did.into();
            let result = validate_create_publication(
                EntryCreationAction::Create(fake_create(
                    EntryHash::from_raw_36(vec![0; 36]),
                    AgentPubKey::from_raw_36(vec![1; 36]),
                )),
                pub_data,
            );
            assert!(is_valid(result), "Failed for DID: {}", did);
        }
    }

    #[test]
    fn publication_with_multiple_co_authors() {
        let mut pub_data = valid_publication();
        pub_data.co_authors = vec![
            "did:mycelix:coauthor1".into(),
            "did:mycelix:coauthor2".into(),
        ];
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_with_encrypted_flag() {
        let mut pub_data = valid_publication();
        pub_data.encrypted = true;
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_with_updated_timestamp() {
        let mut pub_data = valid_publication();
        pub_data.updated = Some(Timestamp::from_micros(1_000_000));
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    // Publication update tests
    #[test]
    fn valid_publication_update_passes() {
        let pub_data = valid_publication();
        let update_action = Update {
            author: AgentPubKey::from_raw_36(vec![1; 36]),
            timestamp: Timestamp::from_micros(1),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        };
        let result = validate_update_publication(update_action, pub_data);
        assert!(is_valid(result));
    }

    #[test]
    fn publication_update_requires_non_empty_title() {
        let mut pub_data = valid_publication();
        pub_data.title = "".into();
        let update_action = Update {
            author: AgentPubKey::from_raw_36(vec![1; 36]),
            timestamp: Timestamp::from_micros(1),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        };
        let result = validate_update_publication(update_action, pub_data);
        assert!(is_invalid(result.clone()));
        assert_eq!(get_invalid_reason(result), "Title cannot be empty");
    }

    // ContentBlock validation tests
    #[test]
    fn valid_content_block_passes() {
        let block = valid_content_block();
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn content_block_with_encrypted_content() {
        let mut block = valid_content_block();
        block.encrypted_content = Some("encrypted_data".into());
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn content_block_with_empty_content_no_encrypted_rejected() {
        let mut block = valid_content_block();
        block.content = "".into();
        block.encrypted_content = None;
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn content_block_with_empty_content_but_encrypted_passes() {
        let mut block = valid_content_block();
        block.content = "".into();
        block.encrypted_content = Some("encrypted_data_here".into());
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn content_block_empty_publication_id_rejected() {
        let mut block = valid_content_block();
        block.publication_id = "".into();
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_invalid(result));
    }

    // PublicationVersion validation tests
    #[test]
    fn valid_publication_version_passes() {
        let version = valid_publication_version();
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_version_with_empty_summary_passes() {
        // change_summary is not validated (can be empty)
        let mut version = valid_publication_version();
        version.change_summary = "".into();
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_version_empty_publication_id_rejected() {
        let mut version = valid_publication_version();
        version.publication_id = "".into();
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_version_empty_content_hash_rejected() {
        let mut version = valid_publication_version();
        version.content_hash = "".into();
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_version_zero_rejected() {
        let mut version = valid_publication_version();
        version.version = 0;
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_invalid(result));
    }

    // Anchor validation tests
    #[test]
    fn valid_anchor_passes() {
        let _anchor = valid_anchor();
        // Anchors are always valid in current implementation
    }

    // Serde roundtrip tests
    #[test]
    fn content_type_serde_roundtrip() {
        let variants = vec![
            ContentType::Article,
            ContentType::Opinion,
            ContentType::Investigation,
            ContentType::Review,
            ContentType::Analysis,
            ContentType::Interview,
            ContentType::Report,
            ContentType::Editorial,
            ContentType::Other("custom".into()),
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: ContentType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn license_type_serde_roundtrip() {
        let variants = vec![
            LicenseType::CC0,
            LicenseType::CCBY,
            LicenseType::CCBYSA,
            LicenseType::CCBYNC,
            LicenseType::CCBYNCSA,
            LicenseType::AllRightsReserved,
            LicenseType::Custom("custom-license".into()),
        ];

        for variant in variants {
            let serialized = serde_json::to_string(&variant).unwrap();
            let deserialized: LicenseType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn license_serde_roundtrip() {
        let license = License {
            license_type: LicenseType::CCBYSA,
            attribution_required: true,
            commercial_use: false,
            derivative_works: true,
        };
        let serialized = serde_json::to_string(&license).unwrap();
        let deserialized: License = serde_json::from_str(&serialized).unwrap();
        assert_eq!(license, deserialized);
    }

    #[test]
    fn publication_serde_roundtrip() {
        let pub_data = valid_publication();
        let serialized = serde_json::to_string(&pub_data).unwrap();
        let deserialized: Publication = serde_json::from_str(&serialized).unwrap();
        assert_eq!(pub_data, deserialized);
    }

    #[test]
    fn content_block_serde_roundtrip() {
        let block = valid_content_block();
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: ContentBlock = serde_json::from_str(&serialized).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn publication_version_serde_roundtrip() {
        let version = valid_publication_version();
        let serialized = serde_json::to_string(&version).unwrap();
        let deserialized: PublicationVersion = serde_json::from_str(&serialized).unwrap();
        assert_eq!(version, deserialized);
    }

    #[test]
    fn anchor_serde_roundtrip() {
        let anchor = valid_anchor();
        let serialized = serde_json::to_string(&anchor).unwrap();
        let deserialized: Anchor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(anchor, deserialized);
    }

    // Edge case tests
    #[test]
    fn publication_with_whitespace_only_title_rejected() {
        let mut pub_data = valid_publication();
        pub_data.title = "   ".into();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_with_very_long_title_rejected() {
        let mut pub_data = valid_publication();
        pub_data.title = "a".repeat(10000);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_with_many_tags() {
        let mut pub_data = valid_publication();
        pub_data.tags = (0..100).map(|i| format!("tag{}", i)).collect();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_with_version_zero_passes() {
        // Publication's own version field is not validated (only PublicationVersion is)
        let mut pub_data = valid_publication();
        pub_data.version = 0;
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn content_block_with_high_index() {
        let mut block = valid_content_block();
        block.block_index = u32::MAX;
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_version_with_version_number() {
        let mut version = valid_publication_version();
        version.version = 999;
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_valid(result));
    }

    // Tests for all ContentType variants
    #[test]
    fn publication_with_each_content_type() {
        let content_types = vec![
            ContentType::Article,
            ContentType::Opinion,
            ContentType::Investigation,
            ContentType::Review,
            ContentType::Analysis,
            ContentType::Interview,
            ContentType::Report,
            ContentType::Editorial,
            ContentType::Other("podcast".into()),
        ];

        for content_type in content_types {
            let mut pub_data = valid_publication();
            pub_data.content_type = content_type.clone();
            let result = validate_create_publication(
                EntryCreationAction::Create(fake_create(
                    EntryHash::from_raw_36(vec![0; 36]),
                    AgentPubKey::from_raw_36(vec![1; 36]),
                )),
                pub_data,
            );
            assert!(
                is_valid(result),
                "Failed for content_type: {:?}",
                content_type
            );
        }
    }

    // Tests for all License types
    #[test]
    fn publication_with_each_license_type() {
        let license_types = vec![
            LicenseType::CC0,
            LicenseType::CCBY,
            LicenseType::CCBYSA,
            LicenseType::CCBYNC,
            LicenseType::CCBYNCSA,
            LicenseType::AllRightsReserved,
            LicenseType::Custom("proprietary".into()),
        ];

        for license_type in license_types {
            let mut pub_data = valid_publication();
            pub_data.license.license_type = license_type.clone();
            let result = validate_create_publication(
                EntryCreationAction::Create(fake_create(
                    EntryHash::from_raw_36(vec![0; 36]),
                    AgentPubKey::from_raw_36(vec![1; 36]),
                )),
                pub_data,
            );
            assert!(
                is_valid(result),
                "Failed for license_type: {:?}",
                license_type
            );
        }
    }

    #[test]
    fn publication_with_all_license_flags_false() {
        let mut pub_data = valid_publication();
        pub_data.license = License {
            license_type: LicenseType::AllRightsReserved,
            attribution_required: false,
            commercial_use: false,
            derivative_works: false,
        };
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_with_special_characters_in_id() {
        let mut pub_data = valid_publication();
        pub_data.id = "pub-123-!@#$%^&*()".into();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_with_unicode_content() {
        let mut pub_data = valid_publication();
        pub_data.title = "日本語のタイトル 🌸".into();
        pub_data.language = "ja".into();
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    // ========================================================================
    // STRING LENGTH BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn publication_title_at_limit_passes() {
        let mut pub_data = valid_publication();
        pub_data.title = "a".repeat(512);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_title_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.title = "a".repeat(513);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_id_at_limit_passes() {
        let mut pub_data = valid_publication();
        pub_data.id = "x".repeat(256);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_id_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.id = "x".repeat(257);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_content_hash_at_limit_passes() {
        let mut pub_data = valid_publication();
        pub_data.content_hash = "h".repeat(256);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_content_hash_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.content_hash = "h".repeat(257);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_language_at_limit_passes() {
        let mut pub_data = valid_publication();
        pub_data.language = "l".repeat(128);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_language_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.language = "l".repeat(129);
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_tag_at_limit_passes() {
        let mut pub_data = valid_publication();
        pub_data.tags = vec!["t".repeat(256)];
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_tag_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.tags = vec!["t".repeat(257)];
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_co_author_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.co_authors = vec!["d".repeat(257)];
        let result = validate_create_publication(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            pub_data,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn content_block_publication_id_at_limit_passes() {
        let mut block = valid_content_block();
        block.publication_id = "p".repeat(256);
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn content_block_publication_id_over_limit_rejected() {
        let mut block = valid_content_block();
        block.publication_id = "p".repeat(257);
        let result = validate_create_content_block(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            block,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_version_change_summary_at_limit_passes() {
        let mut version = valid_publication_version();
        version.change_summary = "s".repeat(4096);
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_valid(result));
    }

    #[test]
    fn publication_version_change_summary_over_limit_rejected() {
        let mut version = valid_publication_version();
        version.change_summary = "s".repeat(4097);
        let result = validate_create_publication_version(
            EntryCreationAction::Create(fake_create(
                EntryHash::from_raw_36(vec![0; 36]),
                AgentPubKey::from_raw_36(vec![1; 36]),
            )),
            version,
        );
        assert!(is_invalid(result));
    }

    #[test]
    fn publication_update_title_over_limit_rejected() {
        let mut pub_data = valid_publication();
        pub_data.title = "a".repeat(513);
        let update_action = Update {
            author: AgentPubKey::from_raw_36(vec![1; 36]),
            timestamp: Timestamp::from_micros(1),
            action_seq: 1,
            prev_action: ActionHash::from_raw_36(vec![0; 36]),
            original_action_address: ActionHash::from_raw_36(vec![0; 36]),
            original_entry_address: EntryHash::from_raw_36(vec![0; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0; 36]),
            weight: Default::default(),
        };
        let result = validate_update_publication(update_action, pub_data);
        assert!(is_invalid(result));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::AuthorToPublications
            | LinkTypes::TagToPublications
            | LinkTypes::PublicationToBlocks
            | LinkTypes::PublicationToVersions => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- AuthorToPublications (256-byte limit) boundary tests --

    #[test]
    fn link_tag_author_to_publications_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::AuthorToPublications, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_author_to_publications_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::AuthorToPublications, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_author_to_publications_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::AuthorToPublications, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- PublicationToVersions (256-byte limit) boundary tests --

    #[test]
    fn link_tag_pub_to_versions_at_limit_valid() {
        let tag = LinkTag::new(vec![0xEE; 256]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToVersions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_pub_to_versions_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xEE; 257]);
        let result = validate_create_link_tag(&LinkTypes::PublicationToVersions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::AuthorToPublications,
            LinkTypes::TagToPublications,
            LinkTypes::PublicationToBlocks,
            LinkTypes::PublicationToVersions,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
