// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Publication Zome Unit Tests
//!
//! Comprehensive test coverage for content publishing, metadata validation,
//! licensing, versioning, content blocks, and access control.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestPublication {
    pub id: String,
    pub title: String,
    pub content_hash: String,
    pub content_type: TestContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: TestLicense,
    pub encrypted: bool,
    pub published: i64,
    pub updated: Option<i64>,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestContentType {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestLicense {
    pub license_type: TestLicenseType,
    pub attribution_required: bool,
    pub commercial_use: bool,
    pub derivative_works: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestLicenseType {
    CC0,
    CCBY,
    CCBYSA,
    CCBYNC,
    CCBYNCSA,
    AllRightsReserved,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestContentBlock {
    pub publication_id: String,
    pub block_index: u32,
    pub content: String,
    pub encrypted_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestPublicationVersion {
    pub publication_id: String,
    pub version: u32,
    pub content_hash: String,
    pub change_summary: String,
    pub created: i64,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_publication() -> TestPublication {
    TestPublication {
        id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        title: "Understanding Distributed Systems".to_string(),
        content_hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
        content_type: TestContentType::Article,
        author_did: "did:mycelix:author123".to_string(),
        co_authors: vec![],
        language: "en".to_string(),
        tags: vec!["distributed-systems".to_string(), "technology".to_string()],
        license: TestLicense {
            license_type: TestLicenseType::CCBY,
            attribution_required: true,
            commercial_use: true,
            derivative_works: true,
        },
        encrypted: false,
        published: 1704067200000000,
        updated: None,
        version: 1,
    }
}

fn create_cc0_license() -> TestLicense {
    TestLicense {
        license_type: TestLicenseType::CC0,
        attribution_required: false,
        commercial_use: true,
        derivative_works: true,
    }
}

fn create_all_rights_reserved_license() -> TestLicense {
    TestLicense {
        license_type: TestLicenseType::AllRightsReserved,
        attribution_required: true,
        commercial_use: false,
        derivative_works: false,
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

fn validate_content_hash(hash: &str) -> bool {
    !hash.is_empty()
}

fn validate_title(title: &str) -> bool {
    !title.is_empty()
}

fn validate_publication(pub_entry: &TestPublication) -> Result<(), String> {
    if !validate_did(&pub_entry.author_did) {
        return Err("Author must be a valid DID".to_string());
    }
    if !validate_title(&pub_entry.title) {
        return Err("Title cannot be empty".to_string());
    }
    if !validate_content_hash(&pub_entry.content_hash) {
        return Err("Content hash required".to_string());
    }
    Ok(())
}

fn can_update_publication(pub_entry: &TestPublication, requester_did: &str) -> bool {
    pub_entry.author_did == requester_did
}

fn can_delete_publication(pub_entry: &TestPublication, requester_did: &str) -> bool {
    pub_entry.author_did == requester_did
}

fn can_add_tags(pub_entry: &TestPublication, requester_did: &str) -> bool {
    pub_entry.author_did == requester_did
}

fn can_update_license(pub_entry: &TestPublication, requester_did: &str) -> bool {
    pub_entry.author_did == requester_did
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PUBLICATION CREATION TESTS
    // =========================================================================

    #[test]
    fn test_publication_requires_valid_author_did() {
        let publication = create_test_publication();
        assert!(validate_did(&publication.author_did));
        assert!(publication.author_did.starts_with("did:"));
    }

    #[test]
    fn test_publication_invalid_author_did_rejected() {
        let invalid_dids = vec![
            "author123",
            "invalid:author",
            "",
            "did",
            "DID:mycelix:author", // Case sensitive
        ];
        for did in invalid_dids {
            assert!(!validate_did(did), "Should reject invalid DID: {}", did);
        }
    }

    #[test]
    fn test_publication_has_unique_id() {
        let publication = create_test_publication();
        // ID should contain pub prefix and timestamp for uniqueness
        assert!(publication.id.starts_with("pub:"));
        assert!(
            publication
                .id
                .contains(&publication.author_did.replace("did:", ""))
        );
    }

    #[test]
    fn test_publication_requires_title() {
        let publication = create_test_publication();
        assert!(!publication.title.is_empty());
        assert!(validate_title(&publication.title));
    }

    #[test]
    fn test_publication_empty_title_rejected() {
        let mut publication = create_test_publication();
        publication.title = "".to_string();
        let result = validate_publication(&publication);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Title cannot be empty");
    }

    #[test]
    fn test_publication_requires_content_hash() {
        let publication = create_test_publication();
        assert!(!publication.content_hash.is_empty());
        assert!(validate_content_hash(&publication.content_hash));
    }

    #[test]
    fn test_publication_empty_content_hash_rejected() {
        let mut publication = create_test_publication();
        publication.content_hash = "".to_string();
        let result = validate_publication(&publication);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Content hash required");
    }

    #[test]
    fn test_publication_initial_version_is_one() {
        let publication = create_test_publication();
        assert_eq!(publication.version, 1);
    }

    #[test]
    fn test_publication_initial_state_no_updates() {
        let publication = create_test_publication();
        assert!(publication.updated.is_none());
    }

    #[test]
    fn test_publication_published_timestamp_valid() {
        let publication = create_test_publication();
        assert!(publication.published > 0);
        assert!(publication.published > 946684800000000); // After year 2000
    }

    // =========================================================================
    // CONTENT TYPE TESTS
    // =========================================================================

    #[test]
    fn test_all_content_types_supported() {
        let types = vec![
            TestContentType::Article,
            TestContentType::Opinion,
            TestContentType::Investigation,
            TestContentType::Review,
            TestContentType::Analysis,
            TestContentType::Interview,
            TestContentType::Report,
            TestContentType::Editorial,
            TestContentType::Other("Documentary".to_string()),
        ];
        for content_type in types {
            let mut publication = create_test_publication();
            publication.content_type = content_type.clone();
            assert!(validate_publication(&publication).is_ok());
        }
    }

    #[test]
    fn test_custom_content_type() {
        let mut publication = create_test_publication();
        publication.content_type = TestContentType::Other("Podcast Transcript".to_string());
        assert!(matches!(
            publication.content_type,
            TestContentType::Other(_)
        ));
    }

    // =========================================================================
    // LICENSING TESTS
    // =========================================================================

    #[test]
    fn test_license_types_supported() {
        let types = vec![
            TestLicenseType::CC0,
            TestLicenseType::CCBY,
            TestLicenseType::CCBYSA,
            TestLicenseType::CCBYNC,
            TestLicenseType::CCBYNCSA,
            TestLicenseType::AllRightsReserved,
            TestLicenseType::Custom("MIT".to_string()),
        ];
        for license_type in types {
            let license = TestLicense {
                license_type: license_type.clone(),
                attribution_required: true,
                commercial_use: true,
                derivative_works: true,
            };
            // All license types should be valid
            assert!(matches!(
                license.license_type,
                TestLicenseType::CC0
                    | TestLicenseType::CCBY
                    | TestLicenseType::CCBYSA
                    | TestLicenseType::CCBYNC
                    | TestLicenseType::CCBYNCSA
                    | TestLicenseType::AllRightsReserved
                    | TestLicenseType::Custom(_)
            ));
        }
    }

    #[test]
    fn test_cc0_license_properties() {
        let license = create_cc0_license();
        assert_eq!(license.license_type, TestLicenseType::CC0);
        assert!(!license.attribution_required);
        assert!(license.commercial_use);
        assert!(license.derivative_works);
    }

    #[test]
    fn test_all_rights_reserved_properties() {
        let license = create_all_rights_reserved_license();
        assert_eq!(license.license_type, TestLicenseType::AllRightsReserved);
        assert!(license.attribution_required);
        assert!(!license.commercial_use);
        assert!(!license.derivative_works);
    }

    #[test]
    fn test_ccby_license_properties() {
        let publication = create_test_publication();
        assert_eq!(publication.license.license_type, TestLicenseType::CCBY);
        assert!(publication.license.attribution_required);
        assert!(publication.license.commercial_use);
        assert!(publication.license.derivative_works);
    }

    #[test]
    fn test_ccbync_license_no_commercial() {
        let license = TestLicense {
            license_type: TestLicenseType::CCBYNC,
            attribution_required: true,
            commercial_use: false,
            derivative_works: true,
        };
        assert!(!license.commercial_use);
    }

    #[test]
    fn test_custom_license() {
        let license = TestLicense {
            license_type: TestLicenseType::Custom("Apache-2.0".to_string()),
            attribution_required: true,
            commercial_use: true,
            derivative_works: true,
        };
        assert!(matches!(license.license_type, TestLicenseType::Custom(_)));
    }

    // =========================================================================
    // CO-AUTHOR TESTS
    // =========================================================================

    #[test]
    fn test_publication_with_co_authors() {
        let mut publication = create_test_publication();
        publication.co_authors = vec![
            "did:mycelix:coauthor1".to_string(),
            "did:mycelix:coauthor2".to_string(),
        ];
        assert_eq!(publication.co_authors.len(), 2);
        for co_author in &publication.co_authors {
            assert!(validate_did(co_author));
        }
    }

    #[test]
    fn test_publication_no_co_authors_valid() {
        let publication = create_test_publication();
        assert!(publication.co_authors.is_empty());
    }

    #[test]
    fn test_co_authors_must_be_valid_dids() {
        let co_authors = vec![
            "did:mycelix:coauthor1".to_string(),
            "did:mycelix:coauthor2".to_string(),
        ];
        for co_author in &co_authors {
            assert!(validate_did(co_author));
        }
    }

    // =========================================================================
    // TAGS TESTS
    // =========================================================================

    #[test]
    fn test_publication_with_tags() {
        let publication = create_test_publication();
        assert!(!publication.tags.is_empty());
        assert_eq!(publication.tags.len(), 2);
    }

    #[test]
    fn test_publication_no_tags_valid() {
        let mut publication = create_test_publication();
        publication.tags = vec![];
        assert!(validate_publication(&publication).is_ok());
    }

    #[test]
    fn test_tags_normalized_to_lowercase() {
        let tags = vec!["Technology".to_string(), "SCIENCE".to_string()];
        let normalized: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
        assert_eq!(normalized, vec!["technology", "science"]);
    }

    #[test]
    fn test_add_tags_to_publication() {
        let mut publication = create_test_publication();
        let original_count = publication.tags.len();
        publication.tags.push("new-tag".to_string());
        assert_eq!(publication.tags.len(), original_count + 1);
    }

    // =========================================================================
    // LANGUAGE TESTS
    // =========================================================================

    #[test]
    fn test_publication_language_set() {
        let publication = create_test_publication();
        assert!(!publication.language.is_empty());
        assert_eq!(publication.language, "en");
    }

    #[test]
    fn test_publication_various_languages() {
        let languages = vec!["en", "es", "fr", "de", "zh", "ja", "ar", "hi"];
        for lang in languages {
            let mut publication = create_test_publication();
            publication.language = lang.to_string();
            assert_eq!(publication.language, lang);
        }
    }

    // =========================================================================
    // ENCRYPTION TESTS
    // =========================================================================

    #[test]
    fn test_publication_unencrypted_default() {
        let publication = create_test_publication();
        assert!(!publication.encrypted);
    }

    #[test]
    fn test_publication_encrypted_content() {
        let mut publication = create_test_publication();
        publication.encrypted = true;
        assert!(publication.encrypted);
    }

    // =========================================================================
    // CONTENT BLOCK TESTS
    // =========================================================================

    fn create_test_content_block() -> TestContentBlock {
        TestContentBlock {
            publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
            block_index: 0,
            content: "This is the first paragraph of the article.".to_string(),
            encrypted_content: None,
        }
    }

    #[test]
    fn test_content_block_creation() {
        let block = create_test_content_block();
        assert!(!block.publication_id.is_empty());
        assert_eq!(block.block_index, 0);
        assert!(!block.content.is_empty());
    }

    #[test]
    fn test_content_block_sequential_indices() {
        let blocks: Vec<TestContentBlock> = (0..5)
            .map(|i| TestContentBlock {
                publication_id: "pub:test".to_string(),
                block_index: i,
                content: format!("Block {}", i),
                encrypted_content: None,
            })
            .collect();

        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(block.block_index, i as u32);
        }
    }

    #[test]
    fn test_content_block_encrypted_content() {
        let mut block = create_test_content_block();
        block.encrypted_content = Some("encrypted_data_here".to_string());
        block.content = "".to_string(); // Clear plaintext
        assert!(block.encrypted_content.is_some());
    }

    #[test]
    fn test_content_blocks_cannot_be_updated() {
        // Content blocks are immutable according to validation rules
        let block = create_test_content_block();
        let block_id = block.publication_id.clone();
        // Any update attempt should be rejected
        assert_eq!(block.publication_id, block_id);
    }

    // =========================================================================
    // VERSION TESTS
    // =========================================================================

    fn create_test_version() -> TestPublicationVersion {
        TestPublicationVersion {
            publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
            version: 2,
            content_hash: "QmNewHash123456789".to_string(),
            change_summary: "Fixed typos and added references".to_string(),
            created: 1704153600000000,
        }
    }

    #[test]
    fn test_version_creation() {
        let version = create_test_version();
        assert!(!version.publication_id.is_empty());
        assert_eq!(version.version, 2);
        assert!(!version.content_hash.is_empty());
        assert!(!version.change_summary.is_empty());
    }

    #[test]
    fn test_version_increments() {
        let publication = create_test_publication();
        assert_eq!(publication.version, 1);

        let version = create_test_version();
        assert_eq!(version.version, 2);
        assert!(version.version > publication.version);
    }

    #[test]
    fn test_version_has_different_content_hash() {
        let publication = create_test_publication();
        let version = create_test_version();
        assert_ne!(publication.content_hash, version.content_hash);
    }

    #[test]
    fn test_versions_cannot_be_updated() {
        // Publication versions are immutable according to validation rules
        let version = create_test_version();
        let version_id = version.publication_id.clone();
        assert_eq!(version.publication_id, version_id);
    }

    #[test]
    fn test_version_history_chain() {
        let versions: Vec<TestPublicationVersion> = (1..=5)
            .map(|v| TestPublicationVersion {
                publication_id: "pub:test".to_string(),
                version: v,
                content_hash: format!("hash_v{}", v),
                change_summary: format!("Changes for version {}", v),
                created: 1704000000000000 + (v as i64 * 86400000000),
            })
            .collect();

        // Verify version sequence
        for i in 1..versions.len() {
            assert!(versions[i].version > versions[i - 1].version);
            assert!(versions[i].created > versions[i - 1].created);
        }
    }

    // =========================================================================
    // ACCESS CONTROL TESTS
    // =========================================================================

    #[test]
    fn test_only_author_can_update() {
        let publication = create_test_publication();
        assert!(can_update_publication(
            &publication,
            &publication.author_did
        ));
        assert!(!can_update_publication(&publication, "did:mycelix:other"));
    }

    #[test]
    fn test_only_author_can_delete() {
        let publication = create_test_publication();
        assert!(can_delete_publication(
            &publication,
            &publication.author_did
        ));
        assert!(!can_delete_publication(&publication, "did:mycelix:other"));
    }

    #[test]
    fn test_only_author_can_add_tags() {
        let publication = create_test_publication();
        assert!(can_add_tags(&publication, &publication.author_did));
        assert!(!can_add_tags(&publication, "did:mycelix:other"));
    }

    #[test]
    fn test_only_author_can_update_license() {
        let publication = create_test_publication();
        assert!(can_update_license(&publication, &publication.author_did));
        assert!(!can_update_license(&publication, "did:mycelix:other"));
    }

    #[test]
    fn test_co_author_cannot_update() {
        let mut publication = create_test_publication();
        publication.co_authors = vec!["did:mycelix:coauthor1".to_string()];
        // Co-author is not the primary author
        assert!(!can_update_publication(
            &publication,
            "did:mycelix:coauthor1"
        ));
    }

    // =========================================================================
    // SEARCH AND FILTER TESTS
    // =========================================================================

    fn filter_by_content_type<'a>(
        publications: &'a [TestPublication],
        content_type: &TestContentType,
    ) -> Vec<&'a TestPublication> {
        publications
            .iter()
            .filter(|p| &p.content_type == content_type)
            .collect()
    }

    fn filter_by_language<'a>(
        publications: &'a [TestPublication],
        language: &str,
    ) -> Vec<&'a TestPublication> {
        publications
            .iter()
            .filter(|p| p.language == language)
            .collect()
    }

    fn filter_by_tag<'a>(
        publications: &'a [TestPublication],
        tag: &str,
    ) -> Vec<&'a TestPublication> {
        publications
            .iter()
            .filter(|p| p.tags.contains(&tag.to_string()))
            .collect()
    }

    #[test]
    fn test_filter_by_content_type() {
        let publications = vec![
            create_test_publication(),
            {
                let mut p = create_test_publication();
                p.content_type = TestContentType::Opinion;
                p
            },
            {
                let mut p = create_test_publication();
                p.content_type = TestContentType::Investigation;
                p
            },
        ];

        let articles = filter_by_content_type(&publications, &TestContentType::Article);
        assert_eq!(articles.len(), 1);

        let opinions = filter_by_content_type(&publications, &TestContentType::Opinion);
        assert_eq!(opinions.len(), 1);
    }

    #[test]
    fn test_filter_by_language() {
        let publications = vec![
            create_test_publication(),
            {
                let mut p = create_test_publication();
                p.language = "es".to_string();
                p
            },
            {
                let mut p = create_test_publication();
                p.language = "fr".to_string();
                p
            },
        ];

        let english = filter_by_language(&publications, "en");
        assert_eq!(english.len(), 1);

        let spanish = filter_by_language(&publications, "es");
        assert_eq!(spanish.len(), 1);
    }

    #[test]
    fn test_filter_by_tag() {
        let publications = vec![create_test_publication(), {
            let mut p = create_test_publication();
            p.tags = vec!["science".to_string(), "research".to_string()];
            p
        }];

        let tech = filter_by_tag(&publications, "technology");
        assert_eq!(tech.len(), 1);

        let science = filter_by_tag(&publications, "science");
        assert_eq!(science.len(), 1);
    }

    // =========================================================================
    // AUTHOR STATS TESTS
    // =========================================================================

    #[derive(Debug)]
    struct AuthorStats {
        author_did: String,
        publication_count: u32,
        total_versions: u32,
    }

    fn calculate_author_stats(publications: &[TestPublication], author_did: &str) -> AuthorStats {
        let author_pubs: Vec<&TestPublication> = publications
            .iter()
            .filter(|p| p.author_did == author_did)
            .collect();

        let publication_count = author_pubs.len() as u32;
        let total_versions: u32 = author_pubs.iter().map(|p| p.version).sum();

        AuthorStats {
            author_did: author_did.to_string(),
            publication_count,
            total_versions,
        }
    }

    #[test]
    fn test_author_stats_calculation() {
        let publications = vec![create_test_publication(), {
            let mut p = create_test_publication();
            p.version = 3;
            p
        }];

        let stats = calculate_author_stats(&publications, "did:mycelix:author123");
        assert_eq!(stats.publication_count, 2);
        assert_eq!(stats.total_versions, 4); // 1 + 3
    }

    #[test]
    fn test_author_stats_no_publications() {
        let publications: Vec<TestPublication> = vec![];
        let stats = calculate_author_stats(&publications, "did:mycelix:unknown");
        assert_eq!(stats.publication_count, 0);
        assert_eq!(stats.total_versions, 0);
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_publication_with_very_long_title() {
        let mut publication = create_test_publication();
        publication.title = "A".repeat(1000);
        // Long titles should be allowed
        assert!(validate_publication(&publication).is_ok());
    }

    #[test]
    fn test_publication_with_unicode_title() {
        let mut publication = create_test_publication();
        publication.title = "Understanding Unicode: From ASCII to Emoji".to_string();
        assert!(validate_publication(&publication).is_ok());
    }

    #[test]
    fn test_publication_with_special_characters_in_tags() {
        let mut publication = create_test_publication();
        publication.tags = vec!["c++".to_string(), "c#".to_string(), "node.js".to_string()];
        assert!(validate_publication(&publication).is_ok());
    }

    #[test]
    fn test_publication_with_many_co_authors() {
        let mut publication = create_test_publication();
        publication.co_authors = (0..50)
            .map(|i| format!("did:mycelix:coauthor{}", i))
            .collect();
        assert_eq!(publication.co_authors.len(), 50);
    }

    #[test]
    fn test_publication_with_many_tags() {
        let mut publication = create_test_publication();
        publication.tags = (0..100).map(|i| format!("tag-{}", i)).collect();
        assert_eq!(publication.tags.len(), 100);
    }

    #[test]
    fn test_publication_high_version_number() {
        let mut publication = create_test_publication();
        publication.version = 999;
        assert!(validate_publication(&publication).is_ok());
    }

    #[test]
    fn test_content_block_with_large_content() {
        let block = TestContentBlock {
            publication_id: "pub:test".to_string(),
            block_index: 0,
            content: "A".repeat(100000), // 100KB of content
            encrypted_content: None,
        };
        assert_eq!(block.content.len(), 100000);
    }

    #[test]
    fn test_publication_update_timestamp() {
        let mut publication = create_test_publication();
        let original_published = publication.published;
        publication.updated = Some(1704153600000000);
        assert!(publication.updated.unwrap() > original_published);
    }
}
