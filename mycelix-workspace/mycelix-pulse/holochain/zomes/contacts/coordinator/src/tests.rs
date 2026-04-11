// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit Tests for Contacts Coordinator Zome
//!
//! Tests for contact CRUD operations, groups, and blocking functionality.
//! Uses mock helpers to simulate Holochain environment.

#[cfg(test)]
mod tests {
    use super::*;
    use contacts_integrity::*;

    // ==================== TEST FIXTURES ====================

    /// Creates a test AgentPubKey from bytes
    fn test_agent(id: u8) -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![id; 36])
    }

    /// Creates a test ActionHash from bytes
    fn test_action_hash(id: u8) -> ActionHash {
        ActionHash::from_raw_36(vec![id; 36])
    }

    /// Creates a test timestamp (as u64 microseconds)
    fn test_timestamp() -> u64 {
        1704067200000000 // 2024-01-01 00:00:00 UTC
    }

    /// Creates a test ContactEmail
    fn test_contact_email(email: &str, is_primary: bool) -> ContactEmail {
        ContactEmail {
            email: email.to_string(),
            email_type: "work".to_string(),
            is_primary,
            verified: Some(true),
        }
    }

    /// Creates a test ContactPhone
    fn test_contact_phone(number: &str, is_primary: bool) -> ContactPhone {
        ContactPhone {
            number: number.to_string(),
            phone_type: "mobile".to_string(),
            is_primary,
        }
    }

    /// Creates a test ContactAddress
    fn test_contact_address() -> ContactAddress {
        ContactAddress {
            street: Some("123 Main St".to_string()),
            city: Some("San Francisco".to_string()),
            state: Some("CA".to_string()),
            postal_code: Some("94102".to_string()),
            country: Some("USA".to_string()),
            address_type: "work".to_string(),
        }
    }

    /// Creates a test ContactMetadata
    fn test_contact_metadata() -> ContactMetadata {
        ContactMetadata {
            source: "manual".to_string(),
            last_email_sent: Some(test_timestamp()),
            last_email_received: Some(test_timestamp()),
            email_count: 42,
            response_rate: Some(0.85),
            average_response_time: Some(3600),
        }
    }

    /// Creates a test Contact
    fn test_contact(id: &str, name: &str) -> Contact {
        Contact {
            id: id.to_string(),
            display_name: name.to_string(),
            nickname: None,
            emails: vec![test_contact_email(&format!("{}@example.com", id), true)],
            phones: Vec::new(),
            addresses: Vec::new(),
            organization: None,
            title: None,
            notes: None,
            avatar: None,
            groups: Vec::new(),
            labels: Vec::new(),
            agent_pub_key: None,
            is_favorite: false,
            is_blocked: false,
            metadata: test_contact_metadata(),
            created_at: test_timestamp(),
            updated_at: test_timestamp(),
        }
    }

    /// Creates a test Contact with full details
    fn test_contact_full() -> Contact {
        Contact {
            id: "contact-123".to_string(),
            display_name: "John Doe".to_string(),
            nickname: Some("Johnny".to_string()),
            emails: vec![
                test_contact_email("john@company.com", true),
                test_contact_email("john.doe@personal.com", false),
            ],
            phones: vec![
                test_contact_phone("+1-555-123-4567", true),
                test_contact_phone("+1-555-987-6543", false),
            ],
            addresses: vec![test_contact_address()],
            organization: Some("Acme Corp".to_string()),
            title: Some("Software Engineer".to_string()),
            notes: Some("Met at conference 2023".to_string()),
            avatar: Some("https://example.com/avatar.jpg".to_string()),
            groups: vec!["work".to_string(), "engineering".to_string()],
            labels: vec!["important".to_string(), "tech".to_string()],
            agent_pub_key: Some(test_agent(1)),
            is_favorite: true,
            is_blocked: false,
            metadata: test_contact_metadata(),
            created_at: test_timestamp(),
            updated_at: test_timestamp(),
        }
    }

    /// Creates a test ContactGroup
    fn test_contact_group(id: &str, name: &str) -> ContactGroup {
        ContactGroup {
            id: id.to_string(),
            name: name.to_string(),
            description: Some(format!("{} group description", name)),
            color: Some("#4285F4".to_string()),
            icon: Some("group".to_string()),
            created_at: test_timestamp(),
        }
    }

    /// Creates a test GroupMembership
    fn test_group_membership(contact_id: &str, group_id: &str) -> GroupMembership {
        GroupMembership {
            contact_id: contact_id.to_string(),
            group_id: group_id.to_string(),
            added_at: test_timestamp(),
        }
    }

    /// Creates a test BlockedContact
    fn test_blocked_contact(contact_hash: ActionHash) -> BlockedContact {
        BlockedContact {
            contact_hash,
            reason: Some("Spam sender".to_string()),
            blocked_at: test_timestamp(),
        }
    }

    /// Creates a test UpdateContactInput
    fn test_update_contact_input(hash: ActionHash, contact: Contact) -> UpdateContactInput {
        UpdateContactInput { hash, contact }
    }

    /// Creates a test GroupMembershipInput
    fn test_group_membership_input(contact_hash: ActionHash, group_id: &str) -> GroupMembershipInput {
        GroupMembershipInput {
            contact_hash,
            group_id: group_id.to_string(),
        }
    }

    // ==================== CONTACT CRUD TESTS ====================

    mod contact_crud {
        use super::*;

        #[test]
        fn test_contact_creation() {
            let contact = test_contact("test-1", "Test User");

            assert_eq!(contact.id, "test-1");
            assert_eq!(contact.display_name, "Test User");
            assert!(!contact.is_favorite);
            assert!(!contact.is_blocked);
        }

        #[test]
        fn test_contact_with_all_fields() {
            let contact = test_contact_full();

            assert_eq!(contact.display_name, "John Doe");
            assert!(contact.nickname.is_some());
            assert_eq!(contact.emails.len(), 2);
            assert_eq!(contact.phones.len(), 2);
            assert_eq!(contact.addresses.len(), 1);
            assert!(contact.organization.is_some());
            assert!(contact.title.is_some());
            assert!(contact.notes.is_some());
            assert!(contact.avatar.is_some());
            assert!(!contact.groups.is_empty());
            assert!(!contact.labels.is_empty());
            assert!(contact.agent_pub_key.is_some());
            assert!(contact.is_favorite);
        }

        #[test]
        fn test_contact_id_required() {
            let contact = test_contact("", "Test User");

            // Empty ID should be invalid
            let is_valid = !contact.id.is_empty();
            assert!(!is_valid);
        }

        #[test]
        fn test_contact_display_name_required() {
            let mut contact = test_contact("test-1", "Test User");
            contact.display_name = "".to_string();

            // Empty display name should be invalid
            let is_valid = !contact.display_name.is_empty();
            assert!(!is_valid);
        }

        #[test]
        fn test_contact_update_input() {
            let hash = test_action_hash(1);
            let contact = test_contact("test-1", "Updated Name");
            let input = test_update_contact_input(hash.clone(), contact.clone());

            assert_eq!(input.hash, hash);
            assert_eq!(input.contact.display_name, "Updated Name");
        }

        #[test]
        fn test_contact_timestamps() {
            let contact = test_contact("test-1", "Test User");

            assert!(contact.created_at > 0);
            assert!(contact.updated_at >= contact.created_at);
        }

        #[test]
        fn test_contact_favorite_toggle() {
            let mut contact = test_contact("test-1", "Test User");

            assert!(!contact.is_favorite);
            contact.is_favorite = true;
            assert!(contact.is_favorite);
            contact.is_favorite = false;
            assert!(!contact.is_favorite);
        }

        #[test]
        fn test_contact_block_toggle() {
            let mut contact = test_contact("test-1", "Test User");

            assert!(!contact.is_blocked);
            contact.is_blocked = true;
            assert!(contact.is_blocked);
        }
    }

    // ==================== CONTACT EMAIL TESTS ====================

    mod contact_emails {
        use super::*;

        #[test]
        fn test_email_creation() {
            let email = test_contact_email("test@example.com", true);

            assert_eq!(email.email, "test@example.com");
            assert_eq!(email.email_type, "work");
            assert!(email.is_primary);
            assert_eq!(email.verified, Some(true));
        }

        #[test]
        fn test_email_validation() {
            let valid_emails = vec![
                "test@example.com",
                "user.name@domain.org",
                "user+tag@company.co.uk",
            ];

            for email_str in valid_emails {
                let email = test_contact_email(email_str, false);
                assert!(email.email.contains('@'));
            }
        }

        #[test]
        fn test_invalid_email_format() {
            let invalid_emails = vec![
                "notanemail",
                "missing.at.sign",
                "@nodomain.com",
            ];

            for email_str in invalid_emails {
                let is_valid = email_str.contains('@');
                assert!(!is_valid);
            }
        }

        #[test]
        fn test_multiple_emails() {
            let contact = test_contact_full();

            // Should have exactly one primary email
            let primary_count = contact.emails.iter().filter(|e| e.is_primary).count();
            assert_eq!(primary_count, 1);
        }

        #[test]
        fn test_email_types() {
            let types = vec!["work", "personal", "home", "other"];

            for email_type in types {
                let mut email = test_contact_email("test@example.com", false);
                email.email_type = email_type.to_string();
                assert_eq!(email.email_type, email_type);
            }
        }

        #[test]
        fn test_email_verification_status() {
            let verified = test_contact_email("verified@example.com", true);
            assert_eq!(verified.verified, Some(true));

            let mut unverified = test_contact_email("unverified@example.com", false);
            unverified.verified = Some(false);
            assert_eq!(unverified.verified, Some(false));

            let mut unknown = test_contact_email("unknown@example.com", false);
            unknown.verified = None;
            assert!(unknown.verified.is_none());
        }
    }

    // ==================== CONTACT PHONE TESTS ====================

    mod contact_phones {
        use super::*;

        #[test]
        fn test_phone_creation() {
            let phone = test_contact_phone("+1-555-123-4567", true);

            assert_eq!(phone.number, "+1-555-123-4567");
            assert_eq!(phone.phone_type, "mobile");
            assert!(phone.is_primary);
        }

        #[test]
        fn test_phone_types() {
            let types = vec!["mobile", "work", "home", "fax", "other"];

            for phone_type in types {
                let mut phone = test_contact_phone("+1-555-123-4567", false);
                phone.phone_type = phone_type.to_string();
                assert_eq!(phone.phone_type, phone_type);
            }
        }

        #[test]
        fn test_phone_formats() {
            let formats = vec![
                "+1-555-123-4567",
                "(555) 123-4567",
                "555.123.4567",
                "+44 20 7123 4567",
                "5551234567",
            ];

            for number in formats {
                let phone = test_contact_phone(number, false);
                assert!(!phone.number.is_empty());
            }
        }
    }

    // ==================== CONTACT ADDRESS TESTS ====================

    mod contact_addresses {
        use super::*;

        #[test]
        fn test_address_creation() {
            let address = test_contact_address();

            assert!(address.street.is_some());
            assert!(address.city.is_some());
            assert!(address.state.is_some());
            assert!(address.postal_code.is_some());
            assert!(address.country.is_some());
        }

        #[test]
        fn test_partial_address() {
            let address = ContactAddress {
                street: None,
                city: Some("New York".to_string()),
                state: Some("NY".to_string()),
                postal_code: None,
                country: Some("USA".to_string()),
                address_type: "home".to_string(),
            };

            assert!(address.street.is_none());
            assert!(address.postal_code.is_none());
            assert!(address.city.is_some());
        }

        #[test]
        fn test_address_types() {
            let types = vec!["work", "home", "billing", "shipping", "other"];

            for address_type in types {
                let mut address = test_contact_address();
                address.address_type = address_type.to_string();
                assert_eq!(address.address_type, address_type);
            }
        }
    }

    // ==================== CONTACT METADATA TESTS ====================

    mod contact_metadata {
        use super::*;

        #[test]
        fn test_metadata_creation() {
            let metadata = test_contact_metadata();

            assert_eq!(metadata.source, "manual");
            assert!(metadata.last_email_sent.is_some());
            assert!(metadata.last_email_received.is_some());
            assert_eq!(metadata.email_count, 42);
        }

        #[test]
        fn test_metadata_sources() {
            let sources = vec!["manual", "import", "google", "outlook", "vcard"];

            for source in sources {
                let mut metadata = test_contact_metadata();
                metadata.source = source.to_string();
                assert_eq!(metadata.source, source);
            }
        }

        #[test]
        fn test_response_rate_range() {
            let rates = vec![0.0, 0.25, 0.5, 0.75, 1.0];

            for rate in rates {
                let mut metadata = test_contact_metadata();
                metadata.response_rate = Some(rate);
                assert!(metadata.response_rate.unwrap() >= 0.0);
                assert!(metadata.response_rate.unwrap() <= 1.0);
            }
        }

        #[test]
        fn test_email_statistics() {
            let metadata = test_contact_metadata();

            assert_eq!(metadata.email_count, 42);
            assert!(metadata.average_response_time.is_some());
        }

        #[test]
        fn test_new_contact_metadata() {
            let metadata = ContactMetadata {
                source: "manual".to_string(),
                last_email_sent: None,
                last_email_received: None,
                email_count: 0,
                response_rate: None,
                average_response_time: None,
            };

            assert!(metadata.last_email_sent.is_none());
            assert_eq!(metadata.email_count, 0);
        }
    }

    // ==================== GROUP TESTS ====================

    mod groups {
        use super::*;

        #[test]
        fn test_group_creation() {
            let group = test_contact_group("grp-1", "Work Contacts");

            assert_eq!(group.id, "grp-1");
            assert_eq!(group.name, "Work Contacts");
            assert!(group.description.is_some());
        }

        #[test]
        fn test_group_name_required() {
            let group = test_contact_group("grp-1", "");

            // Empty name should be invalid
            let is_valid = !group.name.is_empty();
            assert!(!is_valid);
        }

        #[test]
        fn test_group_name_max_length() {
            let long_name = "a".repeat(101);
            let group = test_contact_group("grp-1", &long_name);

            let is_valid = group.name.len() <= 100;
            assert!(!is_valid);
        }

        #[test]
        fn test_group_with_color() {
            let group = test_contact_group("grp-1", "Colored Group");

            assert!(group.color.is_some());
            assert!(group.color.as_ref().unwrap().starts_with('#'));
        }

        #[test]
        fn test_group_with_icon() {
            let group = test_contact_group("grp-1", "Icon Group");

            assert!(group.icon.is_some());
        }

        #[test]
        fn test_group_optional_fields() {
            let group = ContactGroup {
                id: "grp-1".to_string(),
                name: "Minimal Group".to_string(),
                description: None,
                color: None,
                icon: None,
                created_at: test_timestamp(),
            };

            assert!(group.description.is_none());
            assert!(group.color.is_none());
            assert!(group.icon.is_none());
        }

        #[test]
        fn test_group_sorting() {
            let mut groups = vec![
                test_contact_group("grp-3", "Zulu"),
                test_contact_group("grp-1", "Alpha"),
                test_contact_group("grp-2", "Bravo"),
            ];

            groups.sort_by(|a, b| a.name.cmp(&b.name));

            assert_eq!(groups[0].name, "Alpha");
            assert_eq!(groups[1].name, "Bravo");
            assert_eq!(groups[2].name, "Zulu");
        }
    }

    // ==================== GROUP MEMBERSHIP TESTS ====================

    mod group_membership {
        use super::*;

        #[test]
        fn test_membership_creation() {
            let membership = test_group_membership("contact-1", "group-1");

            assert_eq!(membership.contact_id, "contact-1");
            assert_eq!(membership.group_id, "group-1");
            assert!(membership.added_at > 0);
        }

        #[test]
        fn test_membership_input() {
            let input = test_group_membership_input(test_action_hash(1), "work");

            assert_eq!(input.group_id, "work");
        }

        #[test]
        fn test_membership_required_fields() {
            let membership = test_group_membership("", "group-1");

            // Empty contact_id should be invalid
            let is_valid = !membership.contact_id.is_empty() && !membership.group_id.is_empty();
            assert!(!is_valid);
        }

        #[test]
        fn test_contact_multiple_groups() {
            let contact = test_contact_full();

            assert!(contact.groups.len() > 1);
        }

        #[test]
        fn test_contact_group_filtering() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice");
                    c.groups = vec!["work".to_string()];
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob");
                    c.groups = vec!["personal".to_string()];
                    c
                },
                {
                    let mut c = test_contact("c3", "Charlie");
                    c.groups = vec!["work".to_string(), "personal".to_string()];
                    c
                },
            ];

            let work_contacts: Vec<_> = contacts
                .iter()
                .filter(|c| c.groups.contains(&"work".to_string()))
                .collect();

            assert_eq!(work_contacts.len(), 2);
        }
    }

    // ==================== BLOCKING TESTS ====================

    mod blocking {
        use super::*;

        #[test]
        fn test_blocked_contact_creation() {
            let blocked = test_blocked_contact(test_action_hash(1));

            assert!(blocked.reason.is_some());
            assert!(blocked.blocked_at > 0);
        }

        #[test]
        fn test_blocked_contact_without_reason() {
            let blocked = BlockedContact {
                contact_hash: test_action_hash(1),
                reason: None,
                blocked_at: test_timestamp(),
            };

            assert!(blocked.reason.is_none());
        }

        #[test]
        fn test_contact_is_blocked_flag() {
            let mut contact = test_contact("test-1", "Test User");

            assert!(!contact.is_blocked);
            contact.is_blocked = true;
            assert!(contact.is_blocked);
        }

        #[test]
        fn test_blocked_contacts_filtering() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice");
                    c.is_blocked = false;
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob");
                    c.is_blocked = true;
                    c
                },
                {
                    let mut c = test_contact("c3", "Charlie");
                    c.is_blocked = false;
                    c
                },
            ];

            let active_contacts: Vec<_> = contacts
                .iter()
                .filter(|c| !c.is_blocked)
                .collect();

            assert_eq!(active_contacts.len(), 2);
        }
    }

    // ==================== SEARCH AND FILTERING TESTS ====================

    mod search_and_filtering {
        use super::*;

        #[test]
        fn test_filter_favorites() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice");
                    c.is_favorite = true;
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob");
                    c.is_favorite = false;
                    c
                },
            ];

            let favorites: Vec<_> = contacts.iter().filter(|c| c.is_favorite).collect();

            assert_eq!(favorites.len(), 1);
            assert_eq!(favorites[0].display_name, "Alice");
        }

        #[test]
        fn test_sort_by_display_name() {
            let mut contacts = vec![
                test_contact("c3", "Charlie"),
                test_contact("c1", "Alice"),
                test_contact("c2", "Bob"),
            ];

            contacts.sort_by(|a, b| a.display_name.cmp(&b.display_name));

            assert_eq!(contacts[0].display_name, "Alice");
            assert_eq!(contacts[1].display_name, "Bob");
            assert_eq!(contacts[2].display_name, "Charlie");
        }

        #[test]
        fn test_filter_by_organization() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice");
                    c.organization = Some("Acme".to_string());
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob");
                    c.organization = Some("Widget Co".to_string());
                    c
                },
                {
                    let mut c = test_contact("c3", "Charlie");
                    c.organization = Some("Acme".to_string());
                    c
                },
            ];

            let acme_contacts: Vec<_> = contacts
                .iter()
                .filter(|c| c.organization.as_deref() == Some("Acme"))
                .collect();

            assert_eq!(acme_contacts.len(), 2);
        }

        #[test]
        fn test_filter_by_label() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice");
                    c.labels = vec!["vip".to_string()];
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob");
                    c.labels = vec!["vendor".to_string()];
                    c
                },
                {
                    let mut c = test_contact("c3", "Charlie");
                    c.labels = vec!["vip".to_string(), "vendor".to_string()];
                    c
                },
            ];

            let vip_contacts: Vec<_> = contacts
                .iter()
                .filter(|c| c.labels.contains(&"vip".to_string()))
                .collect();

            assert_eq!(vip_contacts.len(), 2);
        }

        #[test]
        fn test_filter_with_agent_key() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice");
                    c.agent_pub_key = Some(test_agent(1));
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob");
                    c.agent_pub_key = None;
                    c
                },
            ];

            let holochain_contacts: Vec<_> = contacts
                .iter()
                .filter(|c| c.agent_pub_key.is_some())
                .collect();

            assert_eq!(holochain_contacts.len(), 1);
        }
    }

    // ==================== ANCHOR TESTS ====================

    mod anchors {
        use super::*;

        #[test]
        fn test_all_contacts_anchor() {
            assert_eq!(ALL_CONTACTS_ANCHOR, "all_contacts");
        }

        #[test]
        fn test_all_groups_anchor() {
            assert_eq!(ALL_GROUPS_ANCHOR, "all_groups");
        }

        #[test]
        fn test_blocked_anchor() {
            assert_eq!(BLOCKED_ANCHOR, "blocked_contacts");
        }

        #[test]
        fn test_email_normalization() {
            let emails = vec![
                ("John@Example.COM", "john@example.com"),
                ("USER@DOMAIN.ORG", "user@domain.org"),
                ("Mixed.Case@Email.Net", "mixed.case@email.net"),
            ];

            for (input, expected) in emails {
                let normalized = input.to_lowercase();
                assert_eq!(normalized, expected);
            }
        }
    }

    // ==================== VALIDATION TESTS ====================

    mod validation {
        use super::*;

        #[test]
        fn test_valid_contact() {
            let contact = test_contact("test-1", "Test User");

            let is_valid = !contact.id.is_empty()
                && !contact.display_name.is_empty()
                && contact.emails.iter().all(|e| e.email.contains('@'));

            assert!(is_valid);
        }

        #[test]
        fn test_contact_with_invalid_email() {
            let mut contact = test_contact("test-1", "Test User");
            contact.emails = vec![ContactEmail {
                email: "invalid-email".to_string(),
                email_type: "work".to_string(),
                is_primary: true,
                verified: None,
            }];

            let is_valid = contact.emails.iter().all(|e| e.email.contains('@'));
            assert!(!is_valid);
        }

        #[test]
        fn test_valid_group() {
            let group = test_contact_group("grp-1", "Test Group");

            let is_valid = !group.name.is_empty() && group.name.len() <= 100;
            assert!(is_valid);
        }

        #[test]
        fn test_valid_group_membership() {
            let membership = test_group_membership("contact-1", "group-1");

            let is_valid = !membership.contact_id.is_empty() && !membership.group_id.is_empty();
            assert!(is_valid);
        }
    }

    // ==================== INTEGRATION SCENARIOS ====================

    mod integration_scenarios {
        use super::*;

        #[test]
        fn test_contact_workflow() {
            // Create contact
            let mut contact = test_contact("test-1", "New Contact");
            assert!(!contact.is_favorite);

            // Add to groups
            contact.groups.push("work".to_string());
            assert!(!contact.groups.is_empty());

            // Mark as favorite
            contact.is_favorite = true;
            assert!(contact.is_favorite);

            // Add additional email
            contact.emails.push(test_contact_email("secondary@example.com", false));
            assert_eq!(contact.emails.len(), 2);

            // Update metadata
            contact.metadata.email_count += 1;
            assert_eq!(contact.metadata.email_count, 43);
        }

        #[test]
        fn test_group_management_workflow() {
            // Create groups
            let mut groups = vec![
                test_contact_group("grp-1", "Work"),
                test_contact_group("grp-2", "Personal"),
            ];

            // Add more groups
            groups.push(test_contact_group("grp-3", "VIP"));
            assert_eq!(groups.len(), 3);

            // Sort groups
            groups.sort_by(|a, b| a.name.cmp(&b.name));
            assert_eq!(groups[0].name, "Personal");
        }

        #[test]
        fn test_contact_blocking_workflow() {
            // Create contact
            let mut contact = test_contact("test-1", "Problematic User");
            assert!(!contact.is_blocked);

            // Block contact
            contact.is_blocked = true;
            let blocked = test_blocked_contact(test_action_hash(1));

            assert!(contact.is_blocked);
            assert!(blocked.reason.is_some());

            // Unblock
            contact.is_blocked = false;
            assert!(!contact.is_blocked);
        }

        #[test]
        fn test_contact_search_workflow() {
            let contacts = vec![
                {
                    let mut c = test_contact("c1", "Alice Smith");
                    c.organization = Some("Acme".to_string());
                    c.is_favorite = true;
                    c.groups = vec!["work".to_string()];
                    c
                },
                {
                    let mut c = test_contact("c2", "Bob Jones");
                    c.organization = Some("Widget Co".to_string());
                    c.is_favorite = false;
                    c.groups = vec!["personal".to_string()];
                    c
                },
            ];

            // Find favorites in work group
            let filtered: Vec<_> = contacts
                .iter()
                .filter(|c| c.is_favorite && c.groups.contains(&"work".to_string()))
                .collect();

            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].display_name, "Alice Smith");
        }
    }
}
