// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit Tests for Mail Messages Coordinator Zome
//!
//! Tests for email storage, retrieval, and management functionality.
//! Uses mock helpers to simulate Holochain environment.

#[cfg(test)]
mod tests {
    use super::*;
    use mail_messages_integrity::*;

    // ==================== TEST FIXTURES ====================

    /// Creates a test AgentPubKey from bytes
    fn test_agent(id: u8) -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![id; 36])
    }

    /// Creates a test ActionHash from bytes
    fn test_action_hash(id: u8) -> ActionHash {
        ActionHash::from_raw_36(vec![id; 36])
    }

    /// Creates a test Timestamp
    fn test_timestamp(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    /// Creates a valid CryptoSuite for testing
    fn test_crypto_suite() -> CryptoSuite {
        CryptoSuite {
            key_exchange: "x25519".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "ed25519".to_string(),
        }
    }

    /// Creates a valid SendEmailInput for testing
    fn test_send_email_input(
        recipients: Vec<AgentPubKey>,
        subject: &str,
        body: &str,
    ) -> SendEmailInput {
        SendEmailInput {
            recipients,
            cc: Vec::new(),
            bcc: Vec::new(),
            encrypted_subject: subject.as_bytes().to_vec(),
            encrypted_body: body.as_bytes().to_vec(),
            encrypted_attachments: Vec::new(),
            ephemeral_pubkey: vec![1, 2, 3, 4],
            nonce: [1u8; 24],
            signature: vec![1, 2, 3, 4, 5],
            crypto_suite: test_crypto_suite(),
            message_id: format!("<test-{}>", uuid::Uuid::new_v4()),
            in_reply_to: None,
            references: Vec::new(),
            priority: EmailPriority::Normal,
            read_receipt_requested: false,
            expires_at: None,
        }
    }

    /// Creates a test EncryptedEmail
    fn test_encrypted_email(
        sender: AgentPubKey,
        recipient: AgentPubKey,
        timestamp: Timestamp,
    ) -> EncryptedEmail {
        EncryptedEmail {
            sender,
            recipient,
            encrypted_subject: b"Test Subject".to_vec(),
            encrypted_body: b"Test Body Content".to_vec(),
            encrypted_attachments: Vec::new(),
            ephemeral_pubkey: vec![1, 2, 3, 4],
            nonce: [1u8; 24],
            signature: vec![1, 2, 3, 4, 5],
            crypto_suite: test_crypto_suite(),
            message_id: "<test-123@example.com>".to_string(),
            in_reply_to: None,
            references: Vec::new(),
            timestamp,
            priority: EmailPriority::Normal,
            read_receipt_requested: false,
            expires_at: None,
        }
    }

    /// Creates a test EmailDraft
    fn test_email_draft(owner: AgentPubKey) -> EmailDraft {
        EmailDraft {
            owner,
            recipients: vec![test_agent(2)],
            cc: Vec::new(),
            bcc: Vec::new(),
            encrypted_subject: b"Draft Subject".to_vec(),
            encrypted_body: b"Draft Body".to_vec(),
            attachments: Vec::new(),
            in_reply_to: None,
            updated_at: test_timestamp(1000000),
            scheduled_for: None,
        }
    }

    /// Creates a test EmailFolder
    fn test_email_folder(owner: AgentPubKey, name: &str, is_system: bool) -> EmailFolder {
        EmailFolder {
            owner,
            encrypted_name: name.as_bytes().to_vec(),
            metadata: None,
            is_system,
            sort_order: 0,
        }
    }

    /// Creates a test EmailQuery
    fn test_email_query() -> EmailQuery {
        EmailQuery {
            folder: None,
            is_read: None,
            is_starred: None,
            is_archived: None,
            since: None,
            limit: Some(50),
            offset: Some(0),
        }
    }

    /// Creates a test EmailStateUpdate
    fn test_state_update(is_read: bool, is_starred: bool) -> EmailStateUpdate {
        EmailStateUpdate {
            is_read: Some(is_read),
            is_starred: Some(is_starred),
            is_archived: None,
            is_trashed: None,
            folders: None,
        }
    }

    /// Creates a test EncryptedAttachment
    fn test_attachment(email_hash: ActionHash, chunk_index: u32) -> EncryptedAttachment {
        EncryptedAttachment {
            email_hash,
            encrypted_filename: b"document.pdf".to_vec(),
            encrypted_mime_type: b"application/pdf".to_vec(),
            encrypted_content: vec![0u8; 1024],
            chunk_index,
            total_chunks: 1,
            content_hash: vec![1, 2, 3, 4],
            nonce: [2u8; 24],
        }
    }

    // ==================== SEND EMAIL TESTS ====================

    mod send_email {
        use super::*;

        #[test]
        fn test_send_email_input_creation() {
            let recipient = test_agent(2);
            let input = test_send_email_input(
                vec![recipient.clone()],
                "Test Subject",
                "Test Body",
            );

            assert_eq!(input.recipients.len(), 1);
            assert_eq!(input.recipients[0], recipient);
            assert!(input.cc.is_empty());
            assert!(input.bcc.is_empty());
            assert_eq!(input.priority, EmailPriority::Normal);
            assert!(!input.read_receipt_requested);
            assert!(input.expires_at.is_none());
        }

        #[test]
        fn test_send_email_input_with_cc_bcc() {
            let mut input = test_send_email_input(
                vec![test_agent(2)],
                "Subject",
                "Body",
            );
            input.cc = vec![test_agent(3)];
            input.bcc = vec![test_agent(4), test_agent(5)];

            assert_eq!(input.recipients.len(), 1);
            assert_eq!(input.cc.len(), 1);
            assert_eq!(input.bcc.len(), 2);
        }

        #[test]
        fn test_send_email_input_with_reply() {
            let mut input = test_send_email_input(
                vec![test_agent(2)],
                "Re: Original",
                "Reply body",
            );
            input.in_reply_to = Some("<original@example.com>".to_string());
            input.references = vec!["<original@example.com>".to_string()];

            assert!(input.in_reply_to.is_some());
            assert_eq!(input.references.len(), 1);
        }

        #[test]
        fn test_send_email_input_high_priority() {
            let mut input = test_send_email_input(
                vec![test_agent(2)],
                "Urgent",
                "Time sensitive",
            );
            input.priority = EmailPriority::Urgent;

            assert_eq!(input.priority, EmailPriority::Urgent);
        }

        #[test]
        fn test_send_email_input_with_expiration() {
            let mut input = test_send_email_input(
                vec![test_agent(2)],
                "Ephemeral",
                "This message will expire",
            );
            input.expires_at = Some(test_timestamp(2000000));

            assert!(input.expires_at.is_some());
        }

        #[test]
        fn test_send_email_output_structure() {
            let output = SendEmailOutput {
                email_hash: test_action_hash(1),
                delivered_to: vec![test_agent(2), test_agent(3)],
                failed_deliveries: vec![(test_agent(4), "Offline".to_string())],
            };

            assert_eq!(output.delivered_to.len(), 2);
            assert_eq!(output.failed_deliveries.len(), 1);
        }

        #[test]
        fn test_send_email_output_all_delivered() {
            let output = SendEmailOutput {
                email_hash: test_action_hash(1),
                delivered_to: vec![test_agent(2)],
                failed_deliveries: Vec::new(),
            };

            assert!(output.failed_deliveries.is_empty());
        }
    }

    // ==================== ENCRYPTED EMAIL TESTS ====================

    mod encrypted_email {
        use super::*;

        #[test]
        fn test_encrypted_email_creation() {
            let sender = test_agent(1);
            let recipient = test_agent(2);
            let timestamp = test_timestamp(1000000);

            let email = test_encrypted_email(sender.clone(), recipient.clone(), timestamp);

            assert_eq!(email.sender, sender);
            assert_eq!(email.recipient, recipient);
            assert!(!email.encrypted_body.is_empty());
            assert!(!email.signature.is_empty());
        }

        #[test]
        fn test_encrypted_email_has_valid_nonce() {
            let email = test_encrypted_email(
                test_agent(1),
                test_agent(2),
                test_timestamp(1000000),
            );

            assert_ne!(email.nonce, [0u8; 24]);
        }

        #[test]
        fn test_encrypted_email_crypto_suite() {
            let email = test_encrypted_email(
                test_agent(1),
                test_agent(2),
                test_timestamp(1000000),
            );

            assert!(!email.crypto_suite.key_exchange.is_empty());
            assert!(!email.crypto_suite.symmetric.is_empty());
            assert!(!email.crypto_suite.signature.is_empty());
        }

        #[test]
        fn test_encrypted_email_priority_levels() {
            let priorities = vec![
                EmailPriority::Low,
                EmailPriority::Normal,
                EmailPriority::High,
                EmailPriority::Urgent,
            ];

            for priority in priorities {
                let mut email = test_encrypted_email(
                    test_agent(1),
                    test_agent(2),
                    test_timestamp(1000000),
                );
                email.priority = priority.clone();
                assert_eq!(email.priority, priority);
            }
        }

        #[test]
        fn test_encrypted_email_with_attachments() {
            let mut email = test_encrypted_email(
                test_agent(1),
                test_agent(2),
                test_timestamp(1000000),
            );
            email.encrypted_attachments = vec![1, 2, 3, 4];

            assert!(!email.encrypted_attachments.is_empty());
        }

        #[test]
        fn test_encrypted_email_threading() {
            let mut email = test_encrypted_email(
                test_agent(1),
                test_agent(2),
                test_timestamp(1000000),
            );
            email.in_reply_to = Some("<parent@example.com>".to_string());
            email.references = vec![
                "<root@example.com>".to_string(),
                "<parent@example.com>".to_string(),
            ];

            assert!(email.in_reply_to.is_some());
            assert_eq!(email.references.len(), 2);
        }
    }

    // ==================== EMAIL QUERY TESTS ====================

    mod email_query {
        use super::*;

        #[test]
        fn test_default_query() {
            let query = test_email_query();

            assert!(query.folder.is_none());
            assert!(query.is_read.is_none());
            assert!(query.is_starred.is_none());
            assert!(query.is_archived.is_none());
        }

        #[test]
        fn test_query_unread() {
            let mut query = test_email_query();
            query.is_read = Some(false);

            assert_eq!(query.is_read, Some(false));
        }

        #[test]
        fn test_query_starred() {
            let mut query = test_email_query();
            query.is_starred = Some(true);

            assert_eq!(query.is_starred, Some(true));
        }

        #[test]
        fn test_query_with_folder() {
            let mut query = test_email_query();
            query.folder = Some(test_action_hash(1));

            assert!(query.folder.is_some());
        }

        #[test]
        fn test_query_pagination() {
            let mut query = test_email_query();
            query.limit = Some(20);
            query.offset = Some(40);

            assert_eq!(query.limit, Some(20));
            assert_eq!(query.offset, Some(40));
        }

        #[test]
        fn test_query_since_timestamp() {
            let mut query = test_email_query();
            query.since = Some(test_timestamp(1000000));

            assert!(query.since.is_some());
        }
    }

    // ==================== EMAIL LIST ITEM TESTS ====================

    mod email_list_item {
        use super::*;

        #[test]
        fn test_email_list_item_creation() {
            let item = EmailListItem {
                hash: test_action_hash(1),
                sender: test_agent(1),
                encrypted_subject: b"Subject".to_vec(),
                timestamp: test_timestamp(1000000),
                priority: EmailPriority::Normal,
                is_read: false,
                is_starred: false,
                has_attachments: false,
                thread_id: None,
            };

            assert!(!item.is_read);
            assert!(!item.is_starred);
            assert!(!item.has_attachments);
        }

        #[test]
        fn test_email_list_item_with_thread() {
            let item = EmailListItem {
                hash: test_action_hash(1),
                sender: test_agent(1),
                encrypted_subject: b"Re: Discussion".to_vec(),
                timestamp: test_timestamp(1000000),
                priority: EmailPriority::Normal,
                is_read: true,
                is_starred: false,
                has_attachments: false,
                thread_id: Some("<thread-123>".to_string()),
            };

            assert!(item.thread_id.is_some());
            assert!(item.is_read);
        }

        #[test]
        fn test_email_list_item_with_attachments() {
            let item = EmailListItem {
                hash: test_action_hash(1),
                sender: test_agent(1),
                encrypted_subject: b"Files attached".to_vec(),
                timestamp: test_timestamp(1000000),
                priority: EmailPriority::High,
                is_read: false,
                is_starred: true,
                has_attachments: true,
                thread_id: None,
            };

            assert!(item.has_attachments);
            assert!(item.is_starred);
            assert_eq!(item.priority, EmailPriority::High);
        }
    }

    // ==================== EMAIL STATE TESTS ====================

    mod email_state {
        use super::*;

        #[test]
        fn test_email_state_update_creation() {
            let update = test_state_update(true, false);

            assert_eq!(update.is_read, Some(true));
            assert_eq!(update.is_starred, Some(false));
            assert!(update.is_archived.is_none());
            assert!(update.is_trashed.is_none());
        }

        #[test]
        fn test_email_state_trash() {
            let update = EmailStateUpdate {
                is_read: None,
                is_starred: None,
                is_archived: None,
                is_trashed: Some(true),
                folders: None,
            };

            assert_eq!(update.is_trashed, Some(true));
        }

        #[test]
        fn test_email_state_archive() {
            let update = EmailStateUpdate {
                is_read: None,
                is_starred: None,
                is_archived: Some(true),
                is_trashed: None,
                folders: None,
            };

            assert_eq!(update.is_archived, Some(true));
        }

        #[test]
        fn test_email_state_move_to_folders() {
            let update = EmailStateUpdate {
                is_read: None,
                is_starred: None,
                is_archived: None,
                is_trashed: None,
                folders: Some(vec![test_action_hash(1), test_action_hash(2)]),
            };

            assert!(update.folders.is_some());
            assert_eq!(update.folders.as_ref().unwrap().len(), 2);
        }

        #[test]
        fn test_email_state_entry_creation() {
            let state = EmailState {
                email_hash: test_action_hash(1),
                owner: test_agent(1),
                is_read: true,
                is_starred: false,
                folders: Vec::new(),
                encrypted_labels: Vec::new(),
                snoozed_until: None,
                is_archived: false,
                is_trashed: false,
                trashed_at: None,
            };

            assert!(state.is_read);
            assert!(!state.is_starred);
            assert!(state.folders.is_empty());
        }

        #[test]
        fn test_email_state_with_snooze() {
            let state = EmailState {
                email_hash: test_action_hash(1),
                owner: test_agent(1),
                is_read: true,
                is_starred: false,
                folders: Vec::new(),
                encrypted_labels: Vec::new(),
                snoozed_until: Some(test_timestamp(2000000)),
                is_archived: false,
                is_trashed: false,
                trashed_at: None,
            };

            assert!(state.snoozed_until.is_some());
        }
    }

    // ==================== DRAFT TESTS ====================

    mod drafts {
        use super::*;

        #[test]
        fn test_draft_creation() {
            let owner = test_agent(1);
            let draft = test_email_draft(owner.clone());

            assert_eq!(draft.owner, owner);
            assert!(!draft.recipients.is_empty());
            assert!(draft.cc.is_empty());
            assert!(draft.bcc.is_empty());
        }

        #[test]
        fn test_draft_with_attachments() {
            let mut draft = test_email_draft(test_agent(1));
            draft.attachments = vec![test_action_hash(1), test_action_hash(2)];

            assert_eq!(draft.attachments.len(), 2);
        }

        #[test]
        fn test_draft_as_reply() {
            let mut draft = test_email_draft(test_agent(1));
            draft.in_reply_to = Some(test_action_hash(1));

            assert!(draft.in_reply_to.is_some());
        }

        #[test]
        fn test_scheduled_draft() {
            let mut draft = test_email_draft(test_agent(1));
            draft.scheduled_for = Some(test_timestamp(3000000));

            assert!(draft.scheduled_for.is_some());
        }

        #[test]
        fn test_draft_multiple_recipients() {
            let mut draft = test_email_draft(test_agent(1));
            draft.recipients = vec![test_agent(2), test_agent(3)];
            draft.cc = vec![test_agent(4)];
            draft.bcc = vec![test_agent(5)];

            assert_eq!(draft.recipients.len(), 2);
            assert_eq!(draft.cc.len(), 1);
            assert_eq!(draft.bcc.len(), 1);
        }
    }

    // ==================== FOLDER TESTS ====================

    mod folders {
        use super::*;

        #[test]
        fn test_folder_creation() {
            let owner = test_agent(1);
            let folder = test_email_folder(owner.clone(), "Work", false);

            assert_eq!(folder.owner, owner);
            assert!(!folder.is_system);
        }

        #[test]
        fn test_system_folder() {
            let folder = test_email_folder(test_agent(1), "Inbox", true);

            assert!(folder.is_system);
        }

        #[test]
        fn test_folder_with_metadata() {
            let mut folder = test_email_folder(test_agent(1), "Custom", false);
            folder.metadata = Some(b"{\"color\": \"blue\"}".to_vec());

            assert!(folder.metadata.is_some());
        }

        #[test]
        fn test_folder_sort_order() {
            let mut folders = vec![
                {
                    let mut f = test_email_folder(test_agent(1), "C", false);
                    f.sort_order = 3;
                    f
                },
                {
                    let mut f = test_email_folder(test_agent(1), "A", false);
                    f.sort_order = 1;
                    f
                },
                {
                    let mut f = test_email_folder(test_agent(1), "B", false);
                    f.sort_order = 2;
                    f
                },
            ];

            folders.sort_by(|a, b| a.sort_order.cmp(&b.sort_order));

            assert_eq!(String::from_utf8_lossy(&folders[0].encrypted_name), "A");
            assert_eq!(String::from_utf8_lossy(&folders[1].encrypted_name), "B");
            assert_eq!(String::from_utf8_lossy(&folders[2].encrypted_name), "C");
        }
    }

    // ==================== ATTACHMENT TESTS ====================

    mod attachments {
        use super::*;

        #[test]
        fn test_attachment_creation() {
            let email_hash = test_action_hash(1);
            let attachment = test_attachment(email_hash.clone(), 0);

            assert_eq!(attachment.email_hash, email_hash);
            assert_eq!(attachment.chunk_index, 0);
            assert_eq!(attachment.total_chunks, 1);
        }

        #[test]
        fn test_chunked_attachment() {
            let email_hash = test_action_hash(1);
            let chunks: Vec<EncryptedAttachment> = (0..5)
                .map(|i| {
                    let mut att = test_attachment(email_hash.clone(), i);
                    att.total_chunks = 5;
                    att
                })
                .collect();

            assert_eq!(chunks.len(), 5);
            for (i, chunk) in chunks.iter().enumerate() {
                assert_eq!(chunk.chunk_index, i as u32);
                assert_eq!(chunk.total_chunks, 5);
            }
        }

        #[test]
        fn test_attachment_content_hash() {
            let attachment = test_attachment(test_action_hash(1), 0);

            assert!(!attachment.content_hash.is_empty());
        }

        #[test]
        fn test_attachment_valid_nonce() {
            let attachment = test_attachment(test_action_hash(1), 0);

            assert_ne!(attachment.nonce, [0u8; 24]);
        }
    }

    // ==================== THREAD TESTS ====================

    mod threads {
        use super::*;

        #[test]
        fn test_thread_creation() {
            let thread = EmailThread {
                thread_id: "<thread-123>".to_string(),
                encrypted_subject: b"Discussion Topic".to_vec(),
                participants: vec![test_agent(1), test_agent(2)],
                message_count: 5,
                last_activity: test_timestamp(1000000),
            };

            assert_eq!(thread.participants.len(), 2);
            assert_eq!(thread.message_count, 5);
        }

        #[test]
        fn test_thread_add_participant() {
            let mut thread = EmailThread {
                thread_id: "<thread-123>".to_string(),
                encrypted_subject: b"Discussion".to_vec(),
                participants: vec![test_agent(1)],
                message_count: 1,
                last_activity: test_timestamp(1000000),
            };

            thread.participants.push(test_agent(2));
            thread.message_count += 1;

            assert_eq!(thread.participants.len(), 2);
            assert_eq!(thread.message_count, 2);
        }

        #[test]
        fn test_thread_update_activity() {
            let mut thread = EmailThread {
                thread_id: "<thread-123>".to_string(),
                encrypted_subject: b"Discussion".to_vec(),
                participants: vec![test_agent(1)],
                message_count: 1,
                last_activity: test_timestamp(1000000),
            };

            thread.last_activity = test_timestamp(2000000);
            thread.message_count += 1;

            assert_eq!(thread.last_activity.as_micros(), 2000000);
        }
    }

    // ==================== RECEIPT TESTS ====================

    mod receipts {
        use super::*;

        #[test]
        fn test_read_receipt_creation() {
            let receipt = ReadReceipt {
                email_hash: test_action_hash(1),
                reader: test_agent(2),
                read_at: test_timestamp(1000000),
                signature: vec![1, 2, 3, 4, 5],
            };

            assert!(!receipt.signature.is_empty());
        }

        #[test]
        fn test_delivery_receipt_creation() {
            let receipt = DeliveryReceipt {
                email_hash: test_action_hash(1),
                recipient: test_agent(2),
                delivered_at: test_timestamp(1000000),
                signature: vec![1, 2, 3, 4, 5],
            };

            assert!(!receipt.signature.is_empty());
        }
    }

    // ==================== SIGNAL TESTS ====================

    mod signals {
        use super::*;

        #[test]
        fn test_email_received_signal() {
            let signal = MailSignal::EmailReceived {
                email_hash: test_action_hash(1),
                sender: test_agent(1),
                encrypted_subject: b"Subject".to_vec(),
                timestamp: test_timestamp(1000000),
                priority: EmailPriority::Normal,
            };

            match signal {
                MailSignal::EmailReceived { priority, .. } => {
                    assert_eq!(priority, EmailPriority::Normal);
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_delivery_confirmed_signal() {
            let signal = MailSignal::DeliveryConfirmed {
                email_hash: test_action_hash(1),
                recipient: test_agent(2),
                delivered_at: test_timestamp(1000000),
            };

            match signal {
                MailSignal::DeliveryConfirmed { recipient, .. } => {
                    assert_eq!(recipient, test_agent(2));
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_read_receipt_signal() {
            let signal = MailSignal::ReadReceiptReceived {
                email_hash: test_action_hash(1),
                reader: test_agent(2),
                read_at: test_timestamp(1000000),
            };

            match signal {
                MailSignal::ReadReceiptReceived { reader, .. } => {
                    assert_eq!(reader, test_agent(2));
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_state_changed_signal() {
            let signal = MailSignal::EmailStateChanged {
                email_hash: test_action_hash(1),
                state: test_state_update(true, false),
            };

            match signal {
                MailSignal::EmailStateChanged { state, .. } => {
                    assert_eq!(state.is_read, Some(true));
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_thread_activity_signal() {
            let signal = MailSignal::ThreadActivity {
                thread_id: "<thread-123>".to_string(),
                email_hash: test_action_hash(1),
                sender: test_agent(1),
            };

            match signal {
                MailSignal::ThreadActivity { thread_id, .. } => {
                    assert_eq!(thread_id, "<thread-123>");
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_draft_saved_signal() {
            let signal = MailSignal::DraftSaved {
                draft_hash: test_action_hash(1),
            };

            match signal {
                MailSignal::DraftSaved { draft_hash } => {
                    assert_eq!(draft_hash, test_action_hash(1));
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_scheduled_email_sent_signal() {
            let signal = MailSignal::ScheduledEmailSent {
                email_hash: test_action_hash(1),
                recipients: vec![test_agent(2), test_agent(3)],
            };

            match signal {
                MailSignal::ScheduledEmailSent { recipients, .. } => {
                    assert_eq!(recipients.len(), 2);
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_typing_indicator_signal() {
            let signal = MailSignal::TypingIndicator {
                sender: test_agent(1),
                thread_id: Some("<thread-123>".to_string()),
            };

            match signal {
                MailSignal::TypingIndicator { thread_id, .. } => {
                    assert!(thread_id.is_some());
                }
                _ => panic!("Wrong signal type"),
            }
        }
    }

    // ==================== CRYPTO SUITE TESTS ====================

    mod crypto_suite {
        use super::*;

        #[test]
        fn test_valid_crypto_suites() {
            let valid_key_exchanges = vec!["x25519", "kyber1024", "kyber768"];
            let valid_symmetric = vec!["chacha20-poly1305", "aes-256-gcm"];
            let valid_signatures = vec!["ed25519", "dilithium3", "dilithium2"];

            for ke in valid_key_exchanges {
                for sym in &valid_symmetric {
                    for sig in &valid_signatures {
                        let suite = CryptoSuite {
                            key_exchange: ke.to_string(),
                            symmetric: sym.to_string(),
                            signature: sig.to_string(),
                        };
                        assert!(!suite.key_exchange.is_empty());
                        assert!(!suite.symmetric.is_empty());
                        assert!(!suite.signature.is_empty());
                    }
                }
            }
        }

        #[test]
        fn test_post_quantum_suite() {
            let suite = CryptoSuite {
                key_exchange: "kyber1024".to_string(),
                symmetric: "chacha20-poly1305".to_string(),
                signature: "dilithium3".to_string(),
            };

            assert_eq!(suite.key_exchange, "kyber1024");
            assert_eq!(suite.signature, "dilithium3");
        }

        #[test]
        fn test_default_crypto_suite() {
            let suite = CryptoSuite::default();

            assert!(suite.key_exchange.is_empty());
            assert!(suite.symmetric.is_empty());
            assert!(suite.signature.is_empty());
        }
    }

    // ==================== BUSINESS LOGIC TESTS ====================

    mod business_logic {
        use super::*;

        #[test]
        fn test_combine_all_recipients() {
            let input = SendEmailInput {
                recipients: vec![test_agent(2)],
                cc: vec![test_agent(3)],
                bcc: vec![test_agent(4)],
                ..test_send_email_input(vec![test_agent(2)], "Test", "Body")
            };

            let all_recipients: Vec<AgentPubKey> = input
                .recipients
                .iter()
                .chain(input.cc.iter())
                .chain(input.bcc.iter())
                .cloned()
                .collect();

            assert_eq!(all_recipients.len(), 3);
        }

        #[test]
        fn test_email_list_sorting() {
            let mut items = vec![
                EmailListItem {
                    hash: test_action_hash(1),
                    sender: test_agent(1),
                    encrypted_subject: b"Old".to_vec(),
                    timestamp: test_timestamp(1000000),
                    priority: EmailPriority::Normal,
                    is_read: false,
                    is_starred: false,
                    has_attachments: false,
                    thread_id: None,
                },
                EmailListItem {
                    hash: test_action_hash(2),
                    sender: test_agent(1),
                    encrypted_subject: b"New".to_vec(),
                    timestamp: test_timestamp(3000000),
                    priority: EmailPriority::Normal,
                    is_read: false,
                    is_starred: false,
                    has_attachments: false,
                    thread_id: None,
                },
                EmailListItem {
                    hash: test_action_hash(3),
                    sender: test_agent(1),
                    encrypted_subject: b"Middle".to_vec(),
                    timestamp: test_timestamp(2000000),
                    priority: EmailPriority::Normal,
                    is_read: false,
                    is_starred: false,
                    has_attachments: false,
                    thread_id: None,
                },
            ];

            // Sort by timestamp descending (newest first)
            items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

            assert_eq!(items[0].timestamp.as_micros(), 3000000);
            assert_eq!(items[1].timestamp.as_micros(), 2000000);
            assert_eq!(items[2].timestamp.as_micros(), 1000000);
        }

        #[test]
        fn test_filter_unread() {
            let items = vec![
                EmailListItem {
                    hash: test_action_hash(1),
                    sender: test_agent(1),
                    encrypted_subject: b"Read".to_vec(),
                    timestamp: test_timestamp(1000000),
                    priority: EmailPriority::Normal,
                    is_read: true,
                    is_starred: false,
                    has_attachments: false,
                    thread_id: None,
                },
                EmailListItem {
                    hash: test_action_hash(2),
                    sender: test_agent(1),
                    encrypted_subject: b"Unread".to_vec(),
                    timestamp: test_timestamp(2000000),
                    priority: EmailPriority::Normal,
                    is_read: false,
                    is_starred: false,
                    has_attachments: false,
                    thread_id: None,
                },
            ];

            let unread: Vec<_> = items.into_iter().filter(|i| !i.is_read).collect();

            assert_eq!(unread.len(), 1);
            assert_eq!(String::from_utf8_lossy(&unread[0].encrypted_subject), "Unread");
        }

        #[test]
        fn test_filter_starred() {
            let items = vec![
                EmailListItem {
                    hash: test_action_hash(1),
                    sender: test_agent(1),
                    encrypted_subject: b"Starred".to_vec(),
                    timestamp: test_timestamp(1000000),
                    priority: EmailPriority::Normal,
                    is_read: true,
                    is_starred: true,
                    has_attachments: false,
                    thread_id: None,
                },
                EmailListItem {
                    hash: test_action_hash(2),
                    sender: test_agent(1),
                    encrypted_subject: b"Not starred".to_vec(),
                    timestamp: test_timestamp(2000000),
                    priority: EmailPriority::Normal,
                    is_read: false,
                    is_starred: false,
                    has_attachments: false,
                    thread_id: None,
                },
            ];

            let starred: Vec<_> = items.into_iter().filter(|i| i.is_starred).collect();

            assert_eq!(starred.len(), 1);
        }

        #[test]
        fn test_pagination() {
            let items: Vec<u32> = (0..100).collect();
            let offset = 20;
            let limit = 10;

            let page: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

            assert_eq!(page.len(), 10);
            assert_eq!(page[0], 20);
            assert_eq!(page[9], 29);
        }

        #[test]
        fn test_email_expiration_check() {
            let now = test_timestamp(2000000);
            let email_expired = test_encrypted_email(test_agent(1), test_agent(2), test_timestamp(1000000));
            let mut email_with_expiry = email_expired.clone();
            email_with_expiry.expires_at = Some(test_timestamp(1500000));

            // Check if expired
            let is_expired = email_with_expiry.expires_at.map(|exp| exp < now).unwrap_or(false);

            assert!(is_expired);
        }

        #[test]
        fn test_thread_id_extraction() {
            let email = EncryptedEmail {
                references: vec!["<root@example.com>".to_string()],
                in_reply_to: Some("<parent@example.com>".to_string()),
                ..test_encrypted_email(test_agent(1), test_agent(2), test_timestamp(1000000))
            };

            let thread_id = email.references.first().cloned().or(email.in_reply_to.clone());

            assert_eq!(thread_id, Some("<root@example.com>".to_string()));
        }
    }
}
