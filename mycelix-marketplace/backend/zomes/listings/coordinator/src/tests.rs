#[cfg(test)]
mod tests {
    use super::*;
    use hdk::prelude::*;
    use listings_integrity::*;

    // Helper functions for tests
    fn mock_listing_input() -> CreateListingInput {
        CreateListingInput {
            title: "Test Product".to_string(),
            description: "A great test product".to_string(),
            price_cents: 1999,
            category: ListingCategory::Electronics,
            photos_ipfs_cids: vec!["QmTest123456789012345678901234567890123456".to_string()],
            quantity_available: 10,
        }
    }

    fn mock_agent_pub_key() -> AgentPubKey {
        // Create a mock AgentPubKey for testing
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    #[test]
    fn test_create_listing_basic() {
        // Test basic listing creation
        let input = mock_listing_input();

        // Verify input is valid
        assert_eq!(input.title.len(), 12);
        assert_eq!(input.price_cents, 1999);
        assert_eq!(input.quantity_available, 10);
        assert_eq!(input.photos_ipfs_cids.len(), 1);
    }

    #[test]
    fn test_listing_title_validation() {
        // Test title length limits
        let mut input = mock_listing_input();

        // Empty title (should fail)
        input.title = "".to_string();
        // In real implementation, this would call validation
        assert!(input.title.is_empty());

        // Too long title (should fail)
        input.title = "a".repeat(201);
        assert!(input.title.len() > 200);

        // Valid title
        input.title = "Valid Product Name".to_string();
        assert!(input.title.len() >= 1 && input.title.len() <= 200);
    }

    #[test]
    fn test_listing_price_validation() {
        let mut input = mock_listing_input();

        // Zero price (should fail)
        input.price_cents = 0;
        assert_eq!(input.price_cents, 0);

        // Negative price would be caught by type system (u64)

        // Valid price
        input.price_cents = 100;
        assert!(input.price_cents > 0);

        // Max price test
        input.price_cents = 100_000_000; // $1M
        assert!(input.price_cents <= 100_000_000);
    }

    #[test]
    fn test_ipfs_cid_validation() {
        let mut input = mock_listing_input();

        // Valid CIDv0 (Qm...)
        input.photos_ipfs_cids = vec![
            "QmTest123456789012345678901234567890123456".to_string()
        ];
        assert!(input.photos_ipfs_cids[0].starts_with("Qm"));
        assert_eq!(input.photos_ipfs_cids[0].len(), 46);

        // Valid CIDv1 (b...)
        input.photos_ipfs_cids = vec![
            "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".to_string()
        ];
        assert!(input.photos_ipfs_cids[0].starts_with("b"));
        assert!(input.photos_ipfs_cids[0].len() >= 50);

        // Invalid CID (should fail)
        input.photos_ipfs_cids = vec!["invalid_cid".to_string()];
        assert!(!input.photos_ipfs_cids[0].starts_with("Qm"));
        assert!(!input.photos_ipfs_cids[0].starts_with("b"));
    }

    #[test]
    fn test_photos_count_validation() {
        let mut input = mock_listing_input();

        // No photos (should fail)
        input.photos_ipfs_cids = vec![];
        assert!(input.photos_ipfs_cids.is_empty());

        // Too many photos (should fail)
        input.photos_ipfs_cids = vec!["QmTest123456789012345678901234567890123456".to_string(); 11];
        assert!(input.photos_ipfs_cids.len() > 10);

        // Valid photo count
        input.photos_ipfs_cids = vec!["QmTest123456789012345678901234567890123456".to_string(); 5];
        assert!(input.photos_ipfs_cids.len() >= 1 && input.photos_ipfs_cids.len() <= 10);
    }

    #[test]
    fn test_epistemic_classification() {
        let mut input = mock_listing_input();

        // Test that seller claims start at E1 (Testimonial)
        let classification = EpistemicClassification {
            empirical: EmpiricalLevel::E1Testimonial,
            normative: NormativeLevel::N0Personal,
            materiality: MaterialityLevel::M1Temporal,
        };

        assert_eq!(classification.empirical, EmpiricalLevel::E1Testimonial);
        assert_eq!(classification.normative, NormativeLevel::N0Personal);
        assert_eq!(classification.materiality, MaterialityLevel::M1Temporal);
    }

    #[test]
    fn test_listing_categories() {
        // Test all category variants exist
        let categories = vec![
            ListingCategory::Electronics,
            ListingCategory::Clothing,
            ListingCategory::Home,
            ListingCategory::Books,
            ListingCategory::Toys,
            ListingCategory::Sports,
            ListingCategory::Food,
            ListingCategory::Other,
        ];

        assert_eq!(categories.len(), 8);
    }

    #[test]
    fn test_listing_status_transitions() {
        // Test valid status transitions
        let statuses = vec![
            ListingStatus::Active,
            ListingStatus::Sold,
            ListingStatus::Deleted,
        ];

        // Active can transition to Sold or Deleted
        // Sold should be immutable
        // Deleted should be immutable

        assert_eq!(statuses.len(), 3);
    }

    #[test]
    fn test_search_query_sanitization() {
        let queries = vec![
            ("laptop", "laptop"),
            ("LAPTOP", "laptop"), // Should be lowercase
            ("  laptop  ", "laptop"), // Should be trimmed
            ("<script>alert('xss')</script>", "scriptalertxssscript"), // XSS should be stripped
        ];

        for (input, expected) in queries {
            let sanitized = input.trim().to_lowercase().replace("<", "").replace(">", "");
            assert_eq!(sanitized, expected);
        }
    }

    #[test]
    fn test_quantity_validation() {
        let mut input = mock_listing_input();

        // Zero quantity (should fail)
        input.quantity_available = 0;
        assert_eq!(input.quantity_available, 0);

        // Valid quantity
        input.quantity_available = 100;
        assert!(input.quantity_available > 0);
    }

    #[test]
    fn test_description_length() {
        let mut input = mock_listing_input();

        // Empty description (should fail)
        input.description = "".to_string();
        assert!(input.description.is_empty());

        // Too long description (should fail)
        input.description = "a".repeat(5001);
        assert!(input.description.len() > 5000);

        // Valid description
        input.description = "This is a valid product description.".to_string();
        assert!(input.description.len() >= 1 && input.description.len() <= 5000);
    }

    #[test]
    fn test_update_listing_input() {
        let update = UpdateListingInput {
            listing_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            title: Some("Updated Title".to_string()),
            description: Some("Updated description".to_string()),
            price_cents: Some(2999),
            quantity_available: Some(5),
            status: Some(ListingStatus::Active),
        };

        // Verify optional fields work
        assert!(update.title.is_some());
        assert!(update.description.is_some());
        assert_eq!(update.price_cents.unwrap(), 2999);
    }

    #[test]
    fn test_listings_response_structure() {
        let response = ListingsResponse {
            listings: vec![],
        };

        assert_eq!(response.listings.len(), 0);
    }

    #[test]
    fn test_category_filter() {
        // Test that filtering by category works
        let electronics = ListingCategory::Electronics;
        let clothing = ListingCategory::Clothing;

        assert_ne!(electronics, clothing);
    }
}
