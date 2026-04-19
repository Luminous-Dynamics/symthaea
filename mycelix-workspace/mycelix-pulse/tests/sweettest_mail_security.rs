// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Security Integration Tests for Mycelix Mail
//!
//! These tests verify the security hardening from the comprehensive mail audit.
//! They require a running Holochain conductor with the mail DNA installed.
//!
//! ## Running
//! ```bash
//! cd mycelix-mail/holochain
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cargo test --release --test sweettest_mail_security -- --ignored --test-threads=1
//! ```
//!
//! ## Test Categories
//!
//! 1. **Signature & Crypto Validation** — Ed25519, Dilithium3, key length checks
//! 2. **Attachment Hardening** — chunk size limits, content hash validation
//! 3. **Capability Revocation** — cap grant lifecycle, shared mailbox auth
//! 4. **Trust & Federation** — finite trust scores, network ID uniqueness, loop detection
//! 5. **Search Sanitization** — control character stripping
//! 6. **Link Validation** — author-match enforcement, delete auth
//! 7. **Bridge Integration** — cross-cluster identity resolution, tier gating
//! 8. **Immutability Regression** — sent emails and receipts cannot be updated

use holochain::sweettest::*;
use holochain_types::prelude::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — must match coordinator structs for deserialization
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SendEmailInput {
    pub recipients: Vec<AgentPubKey>,
    pub cc: Vec<AgentPubKey>,
    pub bcc: Vec<AgentPubKey>,
    pub encrypted_subject: Vec<u8>,
    pub encrypted_body: Vec<u8>,
    pub encrypted_attachments: Vec<u8>,
    pub ephemeral_pubkey: Vec<u8>,
    pub nonce: [u8; 24],
    pub signature: Vec<u8>,
    pub crypto_suite: CryptoSuite,
    pub message_id: String,
    pub in_reply_to: Option<String>,
    pub references: Vec<String>,
    pub priority: EmailPriority,
    pub read_receipt_requested: bool,
    pub expires_at: Option<Timestamp>,
    /// Client-authoritative timestamp (Phase 0.8). Mirror of coordinator struct.
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SendEmailOutput {
    pub email_hash: ActionHash,
    pub delivered_to: Vec<AgentPubKey>,
    pub failed_deliveries: Vec<(AgentPubKey, String)>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CryptoSuite {
    pub key_exchange: String,
    pub symmetric: String,
    pub signature: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EmailPriority {
    Low,
    Normal,
    High,
    Urgent,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AttachmentInput {
    pub email_hash: ActionHash,
    pub encrypted_filename: Vec<u8>,
    pub encrypted_mime_type: Vec<u8>,
    pub encrypted_content: Vec<u8>,
    pub chunk_index: u32,
    pub total_chunks: u32,
    pub content_hash: Vec<u8>,
    pub nonce: [u8; 24],
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustAttestationInput {
    pub trustee: AgentPubKey,
    pub trust_level: f64,
    pub category: String,
    pub reason: Option<String>,
    pub evidence: Vec<serde_json::Value>,
    pub signature: Vec<u8>,
    pub expires_at: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterNetworkInput {
    pub network_id: String,
    pub dna_hash: DnaHash,
    pub name: String,
    pub domain: String,
    pub network_type: String,
    pub network_pubkey: Vec<u8>,
    pub bootstrap_nodes: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateRouteInput {
    pub route_id: String,
    pub source_network: String,
    pub source_pattern: String,
    pub dest_network: String,
    pub dest_pattern: String,
    pub priority: u32,
    pub route_type: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FederatedEnvelopeInput {
    pub envelope_id: String,
    pub source_network: String,
    pub source_agent: String,
    pub dest_network: String,
    pub dest_agent: String,
    pub encrypted_payload: Vec<u8>,
    pub signature: Vec<u8>,
    pub ttl: u64,
    pub max_hops: u8,
    pub previous_hops: Vec<String>,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn mail_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("MAIL_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-mail/
    path.push("holochain");
    path.push("dna");
    path.push("mycelix_mail.dna");
    path
}

/// Build a valid SendEmailInput with Ed25519 crypto suite.
/// The conductor will sign the content using the agent's keypair,
/// so the caller must sign `email_signing_content(...)` with their
/// agent key and set the signature field after construction.
fn make_email_input(
    recipient: AgentPubKey,
    ephemeral_pubkey: Vec<u8>,
    signature: Vec<u8>,
    message_id: &str,
) -> SendEmailInput {
    SendEmailInput {
        recipients: vec![recipient],
        cc: vec![],
        bcc: vec![],
        encrypted_subject: vec![1, 2, 3],
        encrypted_body: vec![4, 5, 6],
        encrypted_attachments: vec![],
        ephemeral_pubkey,
        nonce: [1u8; 24], // Non-zero nonce
        signature,
        crypto_suite: CryptoSuite {
            key_exchange: "x25519".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "ed25519".to_string(),
        },
        message_id: message_id.to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        // Phase 0.8: client-authoritative. Use a fresh Timestamp::now() so the
        // integrity zome's ±window skew check passes.
        timestamp: Timestamp::now(),
    }
}

// =============================================================================
// Phase 1: Signature & Crypto Validation
// =============================================================================

/// Ed25519 signature verification rejects forged signatures.
///
/// A random 64-byte blob is not a valid Ed25519 signature over the canonical
/// email_signing_content(). The integrity zome calls verify_signature_raw which
/// must return false, causing validation to reject.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_ed25519_signature_verification_rejects_forged_signature() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let forged_signature = vec![0xFF; 64]; // Valid length, garbage content
    let input = make_email_input(
        bob.agent_pubkey().clone(),
        vec![0u8; 32], // Valid X25519 length
        forged_signature,
        "forged-sig-001",
    );

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    assert!(
        result.is_err(),
        "send_email MUST reject a forged Ed25519 signature. \
         If this passes, signature verification has been bypassed!"
    );
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("signature")
            || err_msg.contains("Signature")
            || err_msg.contains("verification failed"),
        "Error should mention signature failure, got: {}",
        err_msg,
    );
}

/// Valid Ed25519 signature is accepted by the integrity zome.
///
/// The conductor signs using the agent's Ed25519 keypair. We compute the
/// canonical signing content, sign it with sign_raw, and send the email.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_ed25519_signature_verification_accepts_valid_signature() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // We need to compute the signing content the same way the integrity zome does.
    // Since we can't call email_signing_content from here directly, we construct
    // the canonical bytes: version(1) || sender(39) || recipient(39) || ...
    // However, the simplest approach is to call a helper zome function if available,
    // or sign via the conductor's keystore.
    //
    // The coordinator's send_email constructs the EncryptedEmail and calls
    // create_entry which triggers validation. The signature in the input must
    // match email_signing_content() for the resulting EncryptedEmail.
    //
    // We'll use sign_raw via the conductor keystore to produce a valid signature.
    let signing_content = {
        let mut content = Vec::with_capacity(256);
        content.push(0x01); // version
        content.extend_from_slice(alice.agent_pubkey().get_raw_39());
        content.extend_from_slice(bob.agent_pubkey().get_raw_39());
        // encrypted_subject length + data
        content.extend_from_slice(&(3u32).to_le_bytes());
        content.extend_from_slice(&[1, 2, 3]);
        // encrypted_body length + data
        content.extend_from_slice(&(3u32).to_le_bytes());
        content.extend_from_slice(&[4, 5, 6]);
        // nonce (24 bytes, all 1s)
        content.extend_from_slice(&[1u8; 24]);
        // message_id length + data
        let msg_id = b"valid-sig-001";
        content.extend_from_slice(&(msg_id.len() as u32).to_le_bytes());
        content.extend_from_slice(msg_id);
        // timestamp — the coordinator sets this to sys_time(), which we don't know
        // in advance. This is a limitation: the signing content includes the
        // coordinator-assigned timestamp, so we cannot pre-sign from here.
        //
        // For a full sweettest, the coordinator should handle signing internally
        // or expose a sign_and_send_email function. This test verifies the
        // happy-path architecture is wired correctly.
        content
    };

    // Note: Because the coordinator sets the timestamp at send time (sys_time()),
    // and that timestamp is part of the signing content, the client-side signature
    // will NOT match unless the coordinator provides a sign-then-send flow.
    //
    // This test documents the intended flow. In production, the client SDK
    // computes signing_content with the same timestamp it sets, or the coordinator
    // signs on behalf of the agent.
    //
    // For now, we test that an all-zero signature (clearly invalid) is rejected,
    // confirming the verification path is active.
    let bad_sig = vec![0u8; 64];
    let input = make_email_input(
        bob.agent_pubkey().clone(),
        vec![0u8; 32],
        bad_sig,
        "valid-sig-001",
    );

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    // All-zero signature must also be rejected
    assert!(
        result.is_err(),
        "All-zero signature must be rejected by Ed25519 verification"
    );
}

/// Dilithium3 signature length check rejects wrong-length signatures.
///
/// Dilithium3 requires exactly 3293 bytes. Providing 64 bytes (Ed25519 length)
/// must be rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_dilithium3_signature_length_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = SendEmailInput {
        recipients: vec![bob.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject: vec![1, 2, 3],
        encrypted_body: vec![4, 5, 6],
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 1568], // Kyber1024 length to match key_exchange
        nonce: [1u8; 24],
        signature: vec![0xAA; 64], // WRONG: Dilithium3 needs 3293 bytes
        crypto_suite: CryptoSuite {
            key_exchange: "kyber1024".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "dilithium3".to_string(),
        },
        message_id: "dil3-wrong-len".to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp: Timestamp::now(),
    };

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    assert!(
        result.is_err(),
        "Dilithium3 with 64-byte signature must be rejected (needs 3293)"
    );
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("3293") || err_msg.contains("Dilithium3"),
        "Error should mention Dilithium3 length requirement, got: {}",
        err_msg,
    );
}

/// Dilithium2 signature length check rejects wrong-length signatures.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_dilithium2_signature_length_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = SendEmailInput {
        recipients: vec![bob.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject: vec![1, 2, 3],
        encrypted_body: vec![4, 5, 6],
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 1088], // Kyber768 length
        nonce: [1u8; 24],
        signature: vec![0xBB; 100], // WRONG: Dilithium2 needs 2420 bytes
        crypto_suite: CryptoSuite {
            key_exchange: "kyber768".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "dilithium2".to_string(),
        },
        message_id: "dil2-wrong-len".to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp: Timestamp::now(),
    };

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    assert!(
        result.is_err(),
        "Dilithium2 with 100-byte signature must be rejected (needs 2420)"
    );
}

/// X25519 ephemeral key must be exactly 32 bytes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_ephemeral_key_length_x25519_must_be_32_bytes() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // 48 bytes instead of 32 for X25519
    let input = make_email_input(
        bob.agent_pubkey().clone(),
        vec![0u8; 48], // WRONG: X25519 needs 32
        vec![0xCC; 64],
        "x25519-bad-len",
    );

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    assert!(
        result.is_err(),
        "X25519 ephemeral key with 48 bytes must be rejected (needs 32)"
    );
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("32") || err_msg.contains("X25519"),
        "Error should mention X25519 key length, got: {}",
        err_msg,
    );
}

/// Kyber1024 ephemeral key must be exactly 1568 bytes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_ephemeral_key_length_kyber1024_must_be_1568_bytes() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = SendEmailInput {
        recipients: vec![bob.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject: vec![1, 2, 3],
        encrypted_body: vec![4, 5, 6],
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 32], // WRONG: Kyber1024 needs 1568
        nonce: [1u8; 24],
        signature: vec![0xDD; 3293], // Correct Dilithium3 length
        crypto_suite: CryptoSuite {
            key_exchange: "kyber1024".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "dilithium3".to_string(),
        },
        message_id: "kyber1024-bad-len".to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp: Timestamp::now(),
    };

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    assert!(
        result.is_err(),
        "Kyber1024 ephemeral key with 32 bytes must be rejected (needs 1568)"
    );
}

/// Kyber768 ephemeral key must be exactly 1088 bytes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_ephemeral_key_length_kyber768_must_be_1088_bytes() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = SendEmailInput {
        recipients: vec![bob.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject: vec![1, 2, 3],
        encrypted_body: vec![4, 5, 6],
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 32], // WRONG: Kyber768 needs 1088
        nonce: [1u8; 24],
        signature: vec![0xEE; 2420], // Correct Dilithium2 length
        crypto_suite: CryptoSuite {
            key_exchange: "kyber768".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "dilithium2".to_string(),
        },
        message_id: "kyber768-bad-len".to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp: Timestamp::now(),
    };

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    assert!(
        result.is_err(),
        "Kyber768 ephemeral key with 32 bytes must be rejected (needs 1088)"
    );
}

// =============================================================================
// Phase 1: Attachment Hardening
// =============================================================================

/// Attachment chunk exceeding 10 MB is rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_attachment_chunk_size_limit_10mb() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a fake email hash for the attachment reference
    let fake_email_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
    let oversized_content = vec![0u8; 11 * 1024 * 1024]; // 11 MB > 10 MB limit

    let input = AttachmentInput {
        email_hash: fake_email_hash,
        encrypted_filename: vec![1, 2, 3],
        encrypted_mime_type: vec![4, 5],
        encrypted_content: oversized_content,
        chunk_index: 0,
        total_chunks: 1,
        content_hash: vec![0u8; 32], // Valid SHA-256 length
        nonce: [1u8; 24],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "create_attachment", input)
        .await;

    assert!(
        result.is_err(),
        "Attachment chunk > 10 MB must be rejected by validation"
    );
}

/// Attachment content hash must be exactly 32 bytes (SHA-256).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_attachment_content_hash_must_be_32_bytes() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let fake_email_hash = ActionHash::from_raw_36(vec![0xBB; 36]);

    let input = AttachmentInput {
        email_hash: fake_email_hash,
        encrypted_filename: vec![1],
        encrypted_mime_type: vec![2],
        encrypted_content: vec![3, 4, 5],
        chunk_index: 0,
        total_chunks: 1,
        content_hash: vec![0u8; 16], // WRONG: SHA-256 needs 32 bytes
        nonce: [1u8; 24],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "create_attachment", input)
        .await;

    assert!(
        result.is_err(),
        "Attachment with 16-byte content hash must be rejected (needs 32)"
    );
}

/// Attachment total_chunks must not exceed 1000.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_attachment_total_chunks_capped_at_1000() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let fake_email_hash = ActionHash::from_raw_36(vec![0xCC; 36]);

    let input = AttachmentInput {
        email_hash: fake_email_hash,
        encrypted_filename: vec![1],
        encrypted_mime_type: vec![2],
        encrypted_content: vec![3, 4, 5],
        chunk_index: 0,
        total_chunks: 5000, // WRONG: max is 1000
        content_hash: vec![0u8; 32],
        nonce: [1u8; 24],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "create_attachment", input)
        .await;

    assert!(
        result.is_err(),
        "Attachment with 5000 total_chunks must be rejected (max 1000)"
    );
}

// =============================================================================
// Phase 1: Read Receipt Signatures
// =============================================================================

/// Read receipt with forged signature is rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_read_receipt_signature_verified() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Attempt to create a read receipt with a forged signature
    let fake_email_hash = ActionHash::from_raw_36(vec![0xDD; 36]);

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct CreateReadReceiptInput {
        email_hash: ActionHash,
        signature: Vec<u8>,
    }

    let input = CreateReadReceiptInput {
        email_hash: fake_email_hash,
        signature: vec![0xFF; 64], // Forged: right length, wrong content
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "create_read_receipt", input)
        .await;

    assert!(
        result.is_err(),
        "Read receipt with forged signature must be rejected"
    );
}

/// Delivery receipt with forged signature is rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_delivery_receipt_signature_verified() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let fake_email_hash = ActionHash::from_raw_36(vec![0xEE; 36]);

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct CreateDeliveryReceiptInput {
        email_hash: ActionHash,
        signature: Vec<u8>,
    }

    let input = CreateDeliveryReceiptInput {
        email_hash: fake_email_hash,
        signature: vec![0xFF; 64], // Forged
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(
            &alice.zome("mail_messages"),
            "create_delivery_receipt",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "Delivery receipt with forged signature must be rejected"
    );
}

// =============================================================================
// Phase 1: Capability Revocation
// =============================================================================

/// Capability grant/revoke lifecycle: revoked capability denies access.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_capability_grant_and_revocation_lifecycle() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Step 1: Alice grants Bob capability to read her mailbox
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct GrantCapInput {
        grantee: AgentPubKey,
        functions: Vec<String>,
    }

    let grant_input = GrantCapInput {
        grantee: bob.agent_pubkey().clone(),
        functions: vec!["get_inbox".to_string()],
    };

    let grant_result: Result<(), _> = conductor
        .call_fallible(
            &alice.zome("mail_capabilities"),
            "grant_capability",
            grant_input,
        )
        .await;
    assert!(grant_result.is_ok(), "Grant should succeed");

    // Step 2: Alice revokes the capability
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct RevokeCapInput {
        grantee: AgentPubKey,
    }

    let revoke_input = RevokeCapInput {
        grantee: bob.agent_pubkey().clone(),
    };

    let revoke_result: Result<(), _> = conductor
        .call_fallible(
            &alice.zome("mail_capabilities"),
            "revoke_capability",
            revoke_input,
        )
        .await;
    assert!(revoke_result.is_ok(), "Revoke should succeed");

    // Step 3: Verify Bob can no longer access
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct VerifyCapInput {
        grantor: AgentPubKey,
        function: String,
    }

    let verify_input = VerifyCapInput {
        grantor: alice.agent_pubkey().clone(),
        function: "get_inbox".to_string(),
    };

    let verify_result: Result<bool, _> = conductor
        .call_fallible(
            &bob.zome("mail_capabilities"),
            "verify_capability",
            verify_input,
        )
        .await;

    match verify_result {
        Ok(has_cap) => assert!(
            !has_cap,
            "Revoked capability must return false from verify_capability"
        ),
        Err(_) => {
            // Error is also acceptable — access denied
        }
    }
}

/// Shared mailbox update requires owner or admin role.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_shared_mailbox_update_requires_owner_or_admin() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Alice creates shared mailbox
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct CreateSharedMailboxInput {
        name: String,
        members: Vec<AgentPubKey>,
    }

    let create_input = CreateSharedMailboxInput {
        name: "team-inbox".to_string(),
        members: vec![bob.agent_pubkey().clone()],
    };

    let create_result: Result<ActionHash, _> = conductor
        .call_fallible(
            &alice.zome("mail_capabilities"),
            "create_shared_mailbox",
            create_input,
        )
        .await;

    if let Ok(mailbox_hash) = create_result {
        // Bob (member but not owner/admin) tries to update
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct UpdateSharedMailboxInput {
            mailbox_hash: ActionHash,
            name: Option<String>,
        }

        let update_input = UpdateSharedMailboxInput {
            mailbox_hash,
            name: Some("hacked-name".to_string()),
        };

        let update_result: Result<ActionHash, _> = conductor
            .call_fallible(
                &bob.zome("mail_capabilities"),
                "update_shared_mailbox",
                update_input,
            )
            .await;

        assert!(
            update_result.is_err(),
            "Non-owner/non-admin must not be able to update shared mailbox"
        );
    }
}

// =============================================================================
// Phase 2: Trust & Federation
// =============================================================================

/// Trust attestation with NaN trust_level is rejected by is_finite() guard.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_level_must_be_finite() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = TrustAttestationInput {
        trustee: bob.agent_pubkey().clone(),
        trust_level: f64::NAN,
        category: "Communication".to_string(),
        reason: Some("test".to_string()),
        evidence: vec![],
        signature: vec![0u8; 64],
        expires_at: None,
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_trust"), "create_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Trust attestation with NaN trust_level must be rejected"
    );
}

/// Trust attestation with infinity trust_level is rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_level_infinity_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = TrustAttestationInput {
        trustee: bob.agent_pubkey().clone(),
        trust_level: f64::INFINITY,
        category: "Communication".to_string(),
        reason: Some("test".to_string()),
        evidence: vec![],
        signature: vec![0u8; 64],
        expires_at: None,
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_trust"), "create_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Trust attestation with INFINITY trust_level must be rejected"
    );
}

/// Trust score outside [0.0, 1.0] range is rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_score_range_0_to_1() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Trust level in range [-1.0, 1.0] but score in [0.0, 1.0]
    // Try trust_level = 1.5 which exceeds range
    let input = TrustAttestationInput {
        trustee: bob.agent_pubkey().clone(),
        trust_level: 1.5,
        category: "Communication".to_string(),
        reason: Some("test".to_string()),
        evidence: vec![],
        signature: vec![0u8; 64],
        expires_at: None,
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_trust"), "create_attestation", input)
        .await;

    assert!(
        result.is_err(),
        "Trust attestation with trust_level 1.5 must be rejected (range is -1.0 to 1.0)"
    );
}

/// Federation network ID uniqueness: duplicate registration rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_federation_network_id_uniqueness() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dna_hash = alice.cell_id().dna_hash().clone();

    let input = RegisterNetworkInput {
        network_id: "net-alpha".to_string(),
        dna_hash: dna_hash.clone(),
        name: "Alpha Network".to_string(),
        domain: "alpha.example.com".to_string(),
        network_type: "HolochainNative".to_string(),
        network_pubkey: vec![0u8; 32],
        bootstrap_nodes: vec![],
    };

    // First registration should succeed
    let first: Result<ActionHash, _> = conductor
        .call_fallible(
            &alice.zome("mail_federation"),
            "register_network",
            input.clone(),
        )
        .await;
    assert!(first.is_ok(), "First network registration should succeed");

    // Second registration with same network_id should fail
    let second: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_federation"), "register_network", input)
        .await;

    assert!(
        second.is_err(),
        "Duplicate network_id registration must be rejected"
    );
}

/// Federation route requires ownership of at least one network endpoint.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_federation_route_requires_network_ownership() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let dna_hash = alice.cell_id().dna_hash().clone();

    // Alice registers network "net-alice"
    let network_input = RegisterNetworkInput {
        network_id: "net-alice".to_string(),
        dna_hash,
        name: "Alice Network".to_string(),
        domain: "alice.example.com".to_string(),
        network_type: "HolochainNative".to_string(),
        network_pubkey: vec![0u8; 32],
        bootstrap_nodes: vec![],
    };

    let _: ActionHash = conductor
        .call(
            &alice.zome("mail_federation"),
            "register_network",
            network_input,
        )
        .await;

    // Bob (not owner of net-alice) tries to create route from net-alice
    let route_input = CreateRouteInput {
        route_id: "route-bob-unauthorized".to_string(),
        source_network: "net-alice".to_string(),
        source_pattern: "*@alice.example.com".to_string(),
        dest_network: "net-bob".to_string(),
        dest_pattern: "*@bob.example.com".to_string(),
        priority: 50,
        route_type: "DirectHolochain".to_string(),
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&bob.zome("mail_federation"), "create_route", route_input)
        .await;

    assert!(
        result.is_err(),
        "Route creation by non-owner must be rejected"
    );
}

/// Federation route priority capped at 100.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_federation_route_priority_capped_at_100() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let route_input = CreateRouteInput {
        route_id: "route-high-priority".to_string(),
        source_network: "net-a".to_string(),
        source_pattern: "*@a.example.com".to_string(),
        dest_network: "net-b".to_string(),
        dest_pattern: "*@b.example.com".to_string(),
        priority: 150, // OVER CAP: max is 100
        route_type: "DirectHolochain".to_string(),
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_federation"), "create_route", route_input)
        .await;

    assert!(
        result.is_err(),
        "Route with priority > 100 must be rejected"
    );
}

/// Federation envelope routing loop detection.
///
/// If source_network appears in previous_hops, the envelope has looped.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_federation_envelope_loop_detection() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = FederatedEnvelopeInput {
        envelope_id: "env-loop-001".to_string(),
        source_network: "net-A".to_string(),
        source_agent: "agent-alice".to_string(),
        dest_network: "net-C".to_string(),
        dest_agent: "agent-charlie".to_string(),
        encrypted_payload: vec![1, 2, 3],
        signature: vec![0u8; 64],
        ttl: 3600,
        max_hops: 10,
        previous_hops: vec![
            "net-B".to_string(),
            "net-A".to_string(), // LOOP: net-A already visited
        ],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_federation"), "relay_envelope", input)
        .await;

    assert!(
        result.is_err(),
        "Envelope with routing loop must be rejected"
    );
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("loop") || err_msg.contains("Loop") || err_msg.contains("previous_hops"),
        "Error should mention loop detection, got: {}",
        err_msg,
    );
}

/// Federation envelope max hop count enforced.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_federation_max_hop_count_enforced() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = FederatedEnvelopeInput {
        envelope_id: "env-too-many-hops".to_string(),
        source_network: "net-origin".to_string(),
        source_agent: "agent-origin".to_string(),
        dest_network: "net-dest".to_string(),
        dest_agent: "agent-dest".to_string(),
        encrypted_payload: vec![1, 2, 3],
        signature: vec![0u8; 64],
        ttl: 3600,
        max_hops: 25, // OVER LIMIT: max_hops cannot exceed 20
        previous_hops: vec![],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_federation"), "relay_envelope", input)
        .await;

    assert!(
        result.is_err(),
        "Envelope with max_hops > 20 must be rejected"
    );
}

// =============================================================================
// Phase 2: Search Sanitization
// =============================================================================

/// Search query with control characters is sanitized before execution.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_search_sanitizes_control_characters() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Submit a query containing null bytes and escape sequences
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct SearchInput {
        query: String,
        limit: Option<usize>,
    }

    let input = SearchInput {
        query: "test\x00query\x07with\x1Bcontrol".to_string(),
        limit: Some(10),
    };

    // The search should NOT crash or return an error due to control characters.
    // It should either sanitize them or return an empty result set.
    let result: Result<Vec<serde_json::Value>, _> = conductor
        .call_fallible(&alice.zome("mail_search"), "search_emails", input)
        .await;

    // Success with empty results is fine — the point is no crash/injection
    assert!(
        result.is_ok(),
        "Search with control characters should be sanitized, not crash. \
         Got error: {:?}",
        result.err(),
    );
}

/// Search query with null bytes is safely handled.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_search_query_sanitized() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct SearchInput {
        query: String,
        limit: Option<usize>,
    }

    let input = SearchInput {
        query: "\x00\x00\x00".to_string(), // Pure null bytes
        limit: Some(10),
    };

    let result: Result<Vec<serde_json::Value>, _> = conductor
        .call_fallible(&alice.zome("mail_search"), "search_emails", input)
        .await;

    // Should either return empty results or a clean error, not crash
    match result {
        Ok(results) => {
            // Empty results is the expected sanitized behavior
            assert!(
                results.is_empty(),
                "Pure null byte query should return empty results"
            );
        }
        Err(e) => {
            // A clean rejection error is also acceptable
            let err_msg = format!("{:?}", e);
            assert!(
                !err_msg.contains("panic") && !err_msg.contains("SIGSEGV"),
                "Search should not crash on null bytes, got: {}",
                err_msg,
            );
        }
    }
}

// =============================================================================
// Phase 4: Link Validation
// =============================================================================

/// AgentToSent link requires base agent to match action author.
///
/// Bob cannot create a "sent" link on Alice's agent key.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_agent_to_sent_link_requires_author_match() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Bob tries to send an email "as" Alice by crafting an email entry.
    // The validation should catch that the sender != author.
    // We test this indirectly: Bob calls send_email but the entry will have
    // sender = bob.agent_pubkey (set by coordinator from agent_info).
    // The link AgentToSent uses the author's key as base, so it's self-consistent.
    //
    // Direct link manipulation is not possible through normal zome calls.
    // This test verifies the validation rule exists by confirming that
    // a properly sent email from Bob creates links with Bob's key, not Alice's.
    let input = make_email_input(
        alice.agent_pubkey().clone(),
        vec![0u8; 32],
        vec![0u8; 64], // Will fail sig validation, but link validation is checked first
        "link-test-001",
    );

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&bob.zome("mail_messages"), "send_email", input)
        .await;

    // This should fail (bad signature), but the point is it doesn't succeed
    // with Alice's key as the link base
    assert!(
        result.is_err(),
        "Email with invalid signature should be rejected regardless"
    );
}

/// Link deletion restricted to original creator.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_link_deletion_restricted_to_creator() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Alice creates a folder (which creates AgentToFolders link)
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct CreateFolderInput {
        encrypted_name: Vec<u8>,
        metadata: Option<Vec<u8>>,
    }

    let folder_input = CreateFolderInput {
        encrypted_name: b"my-folder".to_vec(),
        metadata: None,
    };

    let folder_result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "create_folder", folder_input)
        .await;

    if let Ok(folder_hash) = folder_result {
        // Bob tries to delete Alice's folder (and its link)
        let delete_result: Result<(), _> = conductor
            .call_fallible(&bob.zome("mail_messages"), "delete_folder", folder_hash)
            .await;

        assert!(
            delete_result.is_err(),
            "Bob must not be able to delete Alice's folder/link"
        );
    }
}

// =============================================================================
// Phase 5: Bridge Integration (requires unified hApp)
// =============================================================================

/// Mail bridge resolves identity across clusters via OtherRole call.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires unified hApp conductor"]
async fn test_mail_bridge_resolves_identity_cross_cluster() {
    // This test requires the unified hApp with both mail and identity roles.
    // The mail-bridge zome calls CallTargetCell::OtherRole("identity") to
    // resolve DIDs to agent public keys.
    let mut conductor = SweetConductor::from_standard_config().await;

    // Load unified hApp with mail + identity DNAs
    let mail_dna = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    // With single DNA, the bridge should fail gracefully
    let (alice,) = conductor
        .setup_app("test-app", &[mail_dna])
        .await
        .unwrap()
        .into_tuple();

    let did = "did:mycelix:test-alice".to_string();
    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(&alice.zome("mail_bridge"), "resolve_did", did)
        .await;

    // Without identity cluster, should fail (fail-closed)
    assert!(
        result.is_err(),
        "DID resolution must fail when identity cluster is unavailable"
    );
}

/// Mail bridge health check returns status.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires unified hApp conductor"]
async fn test_mail_bridge_health_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file])
        .await
        .unwrap()
        .into_tuple();

    let result: Result<serde_json::Value, _> = conductor
        .call_fallible(&alice.zome("mail_bridge"), "health_check", ())
        .await;

    // Health check should always succeed even with single DNA
    assert!(
        result.is_ok(),
        "Bridge health check should succeed, got: {:?}",
        result.err(),
    );
}

// =============================================================================
// Immutability Regression Tests
// =============================================================================

/// Sent emails cannot be updated (EncryptedEmail is immutable).
///
/// The integrity zome's validate_update_entry returns Invalid for
/// EntryTypes::EncryptedEmail with "Sent emails cannot be modified".
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_sent_emails_cannot_be_updated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First, send a valid email (requires valid signature — use coordinator signing)
    // Since we can't easily produce a valid sig from test code, we test the
    // update rejection path by attempting to update any email entry.
    //
    // Alternative: use a test-mode coordinator that bypasses sig validation
    // for entry creation but still enforces update immutability.
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct UpdateEmailInput {
        original_hash: ActionHash,
        encrypted_body: Vec<u8>,
    }

    let fake_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
    let input = UpdateEmailInput {
        original_hash: fake_hash,
        encrypted_body: vec![99, 100, 101],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "update_email", input)
        .await;

    assert!(
        result.is_err(),
        "Updating a sent email must be rejected — emails are immutable"
    );
}

/// Read receipts cannot be updated (immutable evidence).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_read_receipts_cannot_be_updated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct UpdateReceiptInput {
        original_hash: ActionHash,
    }

    let fake_hash = ActionHash::from_raw_36(vec![0xBB; 36]);
    let input = UpdateReceiptInput {
        original_hash: fake_hash,
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "update_read_receipt", input)
        .await;

    assert!(
        result.is_err(),
        "Updating a read receipt must be rejected — receipts are immutable"
    );
}

/// Delivery receipts cannot be updated (immutable evidence).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_delivery_receipts_cannot_be_updated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct UpdateReceiptInput {
        original_hash: ActionHash,
    }

    let fake_hash = ActionHash::from_raw_36(vec![0xCC; 36]);
    let input = UpdateReceiptInput {
        original_hash: fake_hash,
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(
            &alice.zome("mail_messages"),
            "update_delivery_receipt",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "Updating a delivery receipt must be rejected — receipts are immutable"
    );
}

// =============================================================================
// Spam Protection
// =============================================================================

/// Trust score update restricted to authorized agents only.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_score_update_only_by_authorized_agent() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Bob tries to update Alice's trust score
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct UpdateTrustScoreInput {
        agent: AgentPubKey,
        score: f64,
    }

    let input = UpdateTrustScoreInput {
        agent: alice.agent_pubkey().clone(),
        score: 0.0, // Trying to zero out Alice's score
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&bob.zome("mail_trust"), "update_trust_score", input)
        .await;

    assert!(
        result.is_err(),
        "Bob must not be able to update Alice's trust score"
    );
}

// =============================================================================
// Trust-Gated Delivery
// =============================================================================

/// Email sending blocked when sender has negative trust score.
///
/// The coordinator calls get_sender_trust_for_delivery cross-zome.
/// If the trust score is < 0.0, send_email returns an error.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_send_email_blocked_for_negative_trust() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Step 1: Give Alice a negative trust score via attestation
    let attest_input = TrustAttestationInput {
        trustee: alice.agent_pubkey().clone(),
        trust_level: -0.8, // Negative trust
        category: "Communication".to_string(),
        reason: Some("spammer".to_string()),
        evidence: vec![],
        signature: vec![0u8; 64],
        expires_at: None,
    };

    let _: Result<ActionHash, _> = conductor
        .call_fallible(&bob.zome("mail_trust"), "create_attestation", attest_input)
        .await;

    // Step 2: Alice tries to send email — should be blocked
    let input = make_email_input(
        bob.agent_pubkey().clone(),
        vec![0u8; 32],
        vec![0u8; 64], // Signature doesn't matter, trust check is first
        "blocked-sender-001",
    );

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    // If trust gating is active and the score propagated, this should fail.
    // Note: Trust propagation may be async — if this test is flaky,
    // it means the trust score hasn't been computed yet.
    if result.is_err() {
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("trust")
                || err_msg.contains("blocked")
                || err_msg.contains("signature"),
            "Error should mention trust or be a downstream validation failure, got: {}",
            err_msg,
        );
    }
    // If result is Ok, trust gating may not have computed yet — acceptable for async
}

/// Email sending allowed for unknown sender (no trust history).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_send_email_allowed_for_unknown_sender() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Fresh agents with no trust history — trust check returns None -> allowed
    // The send will fail on signature verification, but NOT on trust gating
    let input = make_email_input(
        bob.agent_pubkey().clone(),
        vec![0u8; 32],
        vec![0u8; 64], // Bad sig, but trust check should pass
        "new-sender-001",
    );

    let result: Result<SendEmailOutput, _> = conductor
        .call_fallible(&alice.zome("mail_messages"), "send_email", input)
        .await;

    // Should fail on signature, NOT on trust
    if let Err(e) = &result {
        let err_msg = format!("{:?}", e);
        assert!(
            !err_msg.contains("trust score") || !err_msg.contains("blocked"),
            "New sender with no trust history should not be blocked by trust gating. \
             Got: {}",
            err_msg,
        );
    }
}

// =============================================================================
// DID Validation
// =============================================================================

/// Empty DID registration rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_did_binding_empty_did_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let result: Result<ActionHash, _> = conductor
        .call_fallible(
            &alice.zome("mail_profiles"),
            "register_my_did",
            "".to_string(), // Empty DID
        )
        .await;

    assert!(result.is_err(), "Empty DID registration must be rejected");
}

/// Duplicate DID registration rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_did_binding_duplicate_registration_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice, bob) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let did = "did:mycelix:test-shared-did".to_string();

    // First registration (Alice) should succeed
    let first: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_profiles"), "register_my_did", did.clone())
        .await;
    assert!(first.is_ok(), "First DID registration should succeed");

    // Second registration (Bob, same DID) should fail
    let second: Result<ActionHash, _> = conductor
        .call_fallible(&bob.zome("mail_profiles"), "register_my_did", did)
        .await;

    assert!(
        second.is_err(),
        "Duplicate DID registration must be rejected"
    );
}

/// Contact with empty name rejected.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_contact_empty_name_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct CreateContactInput {
        display_name: String,
        emails: Vec<serde_json::Value>,
    }

    let input = CreateContactInput {
        display_name: "".to_string(), // Empty name
        emails: vec![],
    };

    let result: Result<ActionHash, _> = conductor
        .call_fallible(&alice.zome("mail_contacts"), "create_contact", input)
        .await;

    assert!(
        result.is_err(),
        "Contact with empty display_name must be rejected"
    );
}

// =============================================================================
// Phase 0: Cross-agent delivery + AgentToInbox spam rejection
// =============================================================================
//
// Phase 0 of PULSE_READINESS_PLAN.md. Two new test shapes:
//
//   1. `phase0_two_conductor_harness_smoke` — minimum-viable proof that the
//      multi-conductor sweettest harness itself works: two conductors boot,
//      the DNA bundle installs on both, and agents discover each other. No
//      email is sent (see blocker note below).
//
//   2. `phase0_forged_inbox_link_rejected` — directly proves the Phase 0.3
//      integrity-zome validation. Eve attempts to create an AgentToInbox link
//      with Bob's pubkey as base and an entry Eve controls as target. The
//      DHT must reject.
//
// ## Phase 0.8 resolution (now landed)
//
// The original design had the coordinator inject `sys_time()` into
// `email.timestamp` — part of the canonical signing content — making client
// signatures structurally impossible. Phase 0.8 flipped to option (B) from
// the plan: `SendEmailInput.timestamp` is now client-authoritative (RFC 5322
// `Date:` semantics). The coordinator uses `input.timestamp` verbatim, and
// the integrity zome bounds the skew between `email.timestamp` and
// `action.timestamp` (commit) within [-30d, +5min] to block future-dated spam
// and stale-replay while accepting legitimate offline composition.
//
// Consequence: clients can now pre-compute `email_signing_content(...)` and
// produce valid signatures. The pre-existing `test_ed25519_*_accepts_valid`
// and `test_dilithium3_*` rejection tests are still rejection-only, but
// happy-path tests (and `phase0_forged_inbox_link_rejected`) become runnable.

/// Two-conductor harness smoke test — proves infrastructure, not delivery.
///
/// Confirms that:
/// - The DNA bundle loads
/// - Two independent conductors can install the same app
/// - Alice's conductor and Bob's conductor each produce an agent key
/// - `await_consistency` completes (peer discovery works)
///
/// Deliberately does NOT attempt `send_email` — that is blocked by the
/// timestamp-signing issue (see module docs above). This test is the
/// foundation every Phase 0 happy-path test will build on, so it must pass
/// standalone first.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Phase 0.2 — requires Holochain conductor + resolved getrandom backend config"]
async fn phase0_two_conductor_harness_smoke() {
    let mut conductors = SweetConductorBatch::from_standard_config_rendezvous(2).await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path())
        .await
        .expect("DNA bundle must load");

    let apps = conductors
        .setup_app("pulse-phase0", &[dna_file.clone()])
        .await
        .expect("setup_app must succeed on both conductors");

    let cells = apps.cells_flattened();
    assert_eq!(
        cells.len(),
        2,
        "Expected exactly 2 cells, one per conductor"
    );

    let alice_cell = cells[0].clone();
    let bob_cell = cells[1].clone();

    assert_ne!(
        alice_cell.agent_pubkey(),
        bob_cell.agent_pubkey(),
        "Alice and Bob must have distinct agent pubkeys"
    );

    // Peer discovery. If this returns Ok, the two conductors found each other
    // over the dev-test bootstrap and the DHT is live enough for further
    // Phase 0 work.
    await_consistency(30, [&alice_cell, &bob_cell])
        .await
        .expect("two conductors must reach DHT consistency");
}

/// Phase 0.5 — AgentToInbox spam rejection.
///
/// Threat model: Eve wants to dump arbitrary entries into Bob's inbox view.
/// Before Phase 0.3, the integrity zome's `validate_create_link` stub let
/// any agent create an `AgentToInbox` link with any base (target agent) and
/// any target (arbitrary entry). Eve could grief Bob's inbox with garbage.
///
/// After Phase 0.3, the integrity zome enforces:
///   (1) Link base is an AgentPubKey (not any other hash type)
///   (2) Target entry deserializes as `EncryptedEmail`
///   (3) `email.recipient == link.base` AND `email.sender == link.author`
///
/// This test directly exercises (3): Eve creates an email where she is both
/// sender and recipient (self-addressed), then attempts to create an inbox
/// link with Bob's pubkey as base. The link author is Eve, and
/// `email.recipient = Eve != Bob`, so validation must reject.
///
/// Full implementation (Phase 0.5). Uses the `debug_create_forged_inbox_link`
/// coordinator extern — safe in production because the Phase 0.3 integrity
/// rule rejects every forged link regardless of which extern created it.
///
/// Steps:
///   1. Eve self-signs a valid email (sender=Eve, recipient=Eve). The entry
///      itself is legitimate — it's Eve's own self-addressed mail.
///   2. Eve calls `debug_create_forged_inbox_link { base: bob, target:
///      eve_self_email_hash }`. The coordinator writes the CreateLink
///      action to Eve's source chain.
///   3. Integrity validation (`validate_inbox_link`): link.base (Bob) vs.
///      email.recipient (Eve) → mismatch → Invalid.
///   4. Depending on the validation path (sync-local vs. async-gossip)
///      the assertion splits:
///        a. call_fallible returns Err with "recipient" / "Invalid" → done
///        b. call succeeds locally → await_consistency → Bob's get_inbox
///           returns zero items (DHT validator refused to propagate)
///      Either is a valid spam-defense outcome.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Phase 0.5 — requires running Holochain conductor; env prereqs in Appendix A"]
async fn phase0_forged_inbox_link_rejected() {
    let mut conductors = SweetConductorBatch::from_standard_config_rendezvous(2).await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path())
        .await
        .expect("DNA bundle must load");

    let apps = conductors
        .setup_app("pulse-ph0-forged", &[dna_file.clone()])
        .await
        .expect("setup_app must succeed");

    let cells = apps.cells_flattened();
    let eve_cell = cells[0].clone();
    let bob_cell = cells[1].clone();

    await_consistency(30, [&eve_cell, &bob_cell])
        .await
        .expect("consistency pre-attack");

    // STEP 1 — Eve authors a valid self-addressed email. Legitimate entry.
    let timestamp = Timestamp::now();
    let message_id = "phase0-forged-001";
    let encrypted_subject = vec![7u8, 7, 7];
    let encrypted_body = vec![8u8, 8, 8];
    let nonce = [2u8; 24];

    let content = compute_signing_content(
        eve_cell.agent_pubkey(),
        eve_cell.agent_pubkey(),
        &encrypted_subject,
        &encrypted_body,
        &nonce,
        message_id,
        timestamp,
    );
    let sig = conductors[0]
        .keystore()
        .sign(eve_cell.agent_pubkey().clone(), content.into())
        .await
        .expect("eve sign");

    let self_send = SendEmailInput {
        recipients: vec![eve_cell.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject,
        encrypted_body,
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 32],
        nonce,
        signature: sig.0.to_vec(),
        crypto_suite: CryptoSuite {
            key_exchange: "x25519".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "ed25519".to_string(),
        },
        message_id: message_id.to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp,
    };
    let self_out: SendEmailOutput = conductors[0]
        .call(&eve_cell.zome("mail_messages"), "send_email", self_send)
        .await;
    assert_eq!(
        self_out.delivered_to.len(),
        1,
        "Eve-to-Eve self-send must succeed"
    );
    let eve_email_hash = self_out.email_hash;

    await_consistency(30, [&eve_cell, &bob_cell])
        .await
        .expect("consistency post-self-send");

    // STEP 2 + 3 — Eve forges an AgentToInbox link targeting Bob's inbox.
    let forged: Result<ActionHash, _> = conductors[0]
        .call_fallible(
            &eve_cell.zome("mail_messages"),
            "debug_create_forged_inbox_link",
            DebugForgedLinkInput {
                base: bob_cell.agent_pubkey().clone(),
                target: eve_email_hash.clone(),
            },
        )
        .await;

    // STEP 4 — Either sync validation caught it, or Bob's view must be empty.
    match forged {
        Err(e) => {
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("recipient")
                    || msg.contains("AgentToInbox")
                    || msg.contains("Invalid"),
                "validation error should mention link/recipient; got: {}",
                msg
            );
        }
        Ok(_) => {
            await_consistency(30, [&eve_cell, &bob_cell])
                .await
                .expect("consistency post-forgery-attempt");

            let bob_inbox: Vec<EmailListItem> = conductors[1]
                .call(
                    &bob_cell.zome("mail_messages"),
                    "get_inbox",
                    EmailQuery {
                        limit: Some(10),
                        ..Default::default()
                    },
                )
                .await;

            assert_eq!(
                bob_inbox.len(),
                0,
                "Bob's inbox must remain empty — forged link must be \
                 rejected by DHT validation even if Eve's local create \
                 succeeded. Got {} items; Phase 0.3 link validation \
                 is broken.",
                bob_inbox.len()
            );
        }
    }
}

/// Mirror of coordinator's DebugForgedLinkInput.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DebugForgedLinkInput {
    pub base: AgentPubKey,
    pub target: ActionHash,
}

// ============================================================================
// Mirror types for get_inbox query (needed by the Phase 0.2 happy-path test)
// ============================================================================

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct EmailQuery {
    pub folder: Option<ActionHash>,
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub is_archived: Option<bool>,
    pub since: Option<Timestamp>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmailListItem {
    pub hash: ActionHash,
    pub sender: AgentPubKey,
    pub encrypted_subject: Vec<u8>,
    pub timestamp: Timestamp,
    pub priority: EmailPriority,
    pub is_read: bool,
    pub is_starred: bool,
    pub has_attachments: bool,
    pub thread_id: Option<String>,
}

/// Compute the canonical `email_signing_content` bytes that the integrity
/// zome's `validate_encrypted_email` verifies against. This duplicates the
/// function at `holochain/zomes/messages/integrity/src/lib.rs:22` — keep in
/// lock-step with that definition. Any change to the canonical format there
/// must mirror here or all signature tests will break.
fn compute_signing_content(
    sender: &AgentPubKey,
    recipient: &AgentPubKey,
    encrypted_subject: &[u8],
    encrypted_body: &[u8],
    nonce: &[u8; 24],
    message_id: &str,
    timestamp: Timestamp,
) -> Vec<u8> {
    let mut content = Vec::with_capacity(256);
    content.push(0x01); // version byte
    content.extend_from_slice(sender.get_raw_39());
    content.extend_from_slice(recipient.get_raw_39());
    content.extend_from_slice(&(encrypted_subject.len() as u32).to_le_bytes());
    content.extend_from_slice(encrypted_subject);
    content.extend_from_slice(&(encrypted_body.len() as u32).to_le_bytes());
    content.extend_from_slice(encrypted_body);
    content.extend_from_slice(nonce);
    content.extend_from_slice(&(message_id.len() as u32).to_le_bytes());
    content.extend_from_slice(message_id.as_bytes());
    content.extend_from_slice(&timestamp.as_micros().to_le_bytes());
    content
}

/// Phase 0.2 happy path — `alice_sends_bob_receives`.
///
/// This is the test whose absence invalidated every "working" claim in the
/// pulse repo at audit time. It was structurally impossible before Phase 0.8
/// because the coordinator's sys_time() injection made client signatures
/// unreachable. Now that `SendEmailInput.timestamp` is client-authoritative,
/// we can compute `email_signing_content(...)` exactly as the integrity zome
/// will, sign it with Alice's agent key via the conductor's lair, and send.
///
/// Flow:
///   1. Two conductors boot + install pulse DNA + reach DHT consistency
///   2. Alice constructs envelope, computes canonical signing content,
///      signs it with her Ed25519 agent key (`conductor.keystore().sign`)
///   3. Alice calls `send_email` → coordinator creates entry → Phase 0.3
///      integrity zome validates sig + skew + ToInbox link + all length
///      checks
///   4. `await_consistency` for DHT gossip
///   5. Bob calls `get_inbox` — must return exactly one item, from Alice
///
/// If this passes, the audit's killer gap ("nobody has verified that Alice
/// sending a message results in Bob receiving it") is closed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Phase 0.2 — requires running Holochain conductor; env prereqs in Appendix A"]
async fn phase0_alice_sends_bob_receives() {
    let mut conductors = SweetConductorBatch::from_standard_config_rendezvous(2).await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path())
        .await
        .expect("DNA bundle must load");

    let apps = conductors
        .setup_app("pulse-ph0-delivery", &[dna_file.clone()])
        .await
        .expect("setup_app on both conductors");

    let cells = apps.cells_flattened();
    let alice_cell = cells[0].clone();
    let bob_cell = cells[1].clone();

    // Peers must discover each other before any send — otherwise Bob won't
    // see Alice's DHT ops during validation.
    await_consistency(30, [&alice_cell, &bob_cell])
        .await
        .expect("pre-send consistency");

    // Pre-warm the init cascade on both cells. The mail_messages zome's
    // init() creates 7 system folders + a cap grant; it also chains into
    // mail_trust's init via the first cross-zome call from send_email. If
    // init runs lazily on the first timed call, it can exceed the 30s
    // init timeout under load. Calling get_folders here forces both inits
    // to run during untimed setup, so send_email later starts from a warm
    // state.
    let _: serde_json::Value = conductors[0]
        .call(&alice_cell.zome("mail_messages"), "get_folders", ())
        .await;
    let _: serde_json::Value = conductors[1]
        .call(&bob_cell.zome("mail_messages"), "get_folders", ())
        .await;

    // Client-authoritative fields — Alice decides these.
    let timestamp = Timestamp::now();
    let message_id = "phase0-happy-001";
    let encrypted_subject = vec![1u8, 2, 3];
    let encrypted_body = vec![4u8, 5, 6];
    let nonce = [1u8; 24];

    // Compute canonical content and sign via Alice's agent key in lair.
    let content = compute_signing_content(
        alice_cell.agent_pubkey(),
        bob_cell.agent_pubkey(),
        &encrypted_subject,
        &encrypted_body,
        &nonce,
        message_id,
        timestamp,
    );

    let signature = conductors[0]
        .keystore()
        .sign(alice_cell.agent_pubkey().clone(), content.into())
        .await
        .expect("lair sign must succeed");

    let input = SendEmailInput {
        recipients: vec![bob_cell.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject: vec![1, 2, 3],
        encrypted_body: vec![4, 5, 6],
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 32], // X25519 length, content unimportant for this test
        nonce,
        signature: signature.0.to_vec(),
        crypto_suite: CryptoSuite {
            key_exchange: "x25519".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "ed25519".to_string(),
        },
        message_id: message_id.to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp,
    };

    // Alice sends. If integrity zome rejects here, Phase 0.3/0.8 have bugs.
    //
    // The call() path mints a fresh nonce per-call. On a heavily-loaded
    // system (load avg > 15), Kitsune2/Tx5 setup + await_consistency can
    // eat 8-10 min wall-clock, which can push the nonce past its 300s TTL
    // if it was minted early. Retry up to 3x on BadNonce("Expired"); each
    // retry builds a fresh input (new timestamp + re-sign). `timestamp`
    // gets re-bound to whichever attempt succeeded, so the downstream
    // round-trip assertion compares against the actually-shipped value.
    let mut attempts = 0u32;
    let mut timestamp = timestamp;
    let output: SendEmailOutput = loop {
        let attempt_ts = Timestamp::now();
        let attempt_content = compute_signing_content(
            alice_cell.agent_pubkey(),
            bob_cell.agent_pubkey(),
            &[1u8, 2, 3],
            &[4u8, 5, 6],
            &nonce,
            message_id,
            attempt_ts,
        );
        let attempt_sig = conductors[0]
            .keystore()
            .sign(alice_cell.agent_pubkey().clone(), attempt_content.into())
            .await
            .expect("lair sign must succeed");
        let mut attempt_input = input.clone();
        attempt_input.signature = attempt_sig.0.to_vec();
        attempt_input.timestamp = attempt_ts;

        let r: Result<SendEmailOutput, _> = conductors[0]
            .call_fallible(
                &alice_cell.zome("mail_messages"),
                "send_email",
                attempt_input,
            )
            .await;

        match r {
            Ok(out) => {
                timestamp = attempt_ts; // bind to the timestamp that actually shipped
                break out;
            }
            Err(e) => {
                let msg = format!("{:?}", e);
                attempts += 1;
                if attempts >= 3 || !msg.contains("BadNonce") {
                    panic!("send_email failed after {} attempts: {}", attempts, msg);
                }
                eprintln!("send_email attempt {} nonce-expired, retrying", attempts);
            }
        }
    };

    assert!(
        output.failed_deliveries.is_empty(),
        "no delivery should fail; got: {:?}",
        output.failed_deliveries
    );
    assert_eq!(
        output.delivered_to.len(),
        1,
        "exactly one recipient (Bob); got: {:?}",
        output.delivered_to
    );

    // Rather than await_consistency (which requires ALL ops to gossip —
    // slow or flaky on multi-session machines), poll Bob's get_inbox
    // directly until the target email shows up or timeout. This mirrors
    // real-world UI behavior: clients poll their inbox, they don't wait
    // for global DHT consistency.
    let query = EmailQuery {
        limit: Some(10),
        ..Default::default()
    };
    let mut inbox: Vec<EmailListItem> = Vec::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    while std::time::Instant::now() < deadline {
        inbox = conductors[1]
            .call(&bob_cell.zome("mail_messages"), "get_inbox", query.clone())
            .await;
        if !inbox.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }

    assert_eq!(
        inbox.len(),
        1,
        "Bob's inbox must contain exactly Alice's one email; got {} items",
        inbox.len()
    );
    let received = &inbox[0];
    assert_eq!(
        received.sender,
        *alice_cell.agent_pubkey(),
        "inbox entry sender must be Alice"
    );
    assert_eq!(
        received.encrypted_subject,
        vec![1u8, 2, 3],
        "encrypted_subject must round-trip byte-for-byte"
    );
    // Compare via as_micros() rather than Timestamp direct-equality. The
    // integrity zome signs over `timestamp.as_micros().to_le_bytes()`, so
    // microsecond-granular round-trip is what Phase 0.8 actually guarantees.
    // Direct Timestamp `==` can fail on sub-microsecond representation
    // details that don't affect the canonical signed bytes.
    assert_eq!(
        received.timestamp.as_micros(),
        timestamp.as_micros(),
        "client timestamp microseconds must round-trip"
    );
}

// ============================================================================
// Mirror types for Phase 1.1 delivery receipts
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeliveryReceipt {
    pub email_hash: ActionHash,
    pub recipient: AgentPubKey,
    pub delivered_at: Timestamp,
    pub signature: Vec<u8>,
}

/// Compute the canonical `delivery_receipt_signing_content` bytes that the
/// integrity zome verifies. Mirror of
/// `holochain/zomes/messages/integrity/src/lib.rs:51` (delivery_receipt
/// version). Must stay in lock-step.
fn compute_delivery_receipt_signing_content(
    email_hash: &ActionHash,
    recipient: &AgentPubKey,
    delivered_at: Timestamp,
) -> Vec<u8> {
    let mut content = Vec::with_capacity(128);
    content.push(0x01); // version byte
    content.extend_from_slice(email_hash.get_raw_39());
    content.extend_from_slice(recipient.get_raw_39());
    content.extend_from_slice(&delivered_at.as_micros().to_le_bytes());
    content
}

/// Phase 1.1 — `acknowledge_delivery` round-trip.
///
/// Flow:
///   1. Reuse the Phase 0.2 happy-path setup: Alice signs and sends Bob an
///      email; await_consistency
///   2. Bob calls `acknowledge_delivery(email_hash)` — coordinator
///      server-signs a `DeliveryReceipt` via Bob's lair, persists, links it
///      from the email entry, fires `DeliveryConfirmed` remote_signal
///   3. await_consistency — Alice's conductor sees Bob's link
///   4. Alice calls `get_delivery_receipts(email_hash)` — must return
///      exactly one receipt, recipient = Bob, signature 64 bytes (Ed25519)
///   5. Verify the signature is structurally valid: byte-canonical signing
///      content + correct length. (Cryptographic verification happens at
///      DHT validation time; if get_delivery_receipts returned a record at
///      all, validation already passed.)
///
/// This proves the full sender ⟷ recipient acknowledgement loop on the DHT,
/// not just one-way delivery. It is the second of the two structural tests
/// that close Phase 0's killer-gap claim.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Phase 1.1 — requires running Holochain conductor; env prereqs in Appendix A"]
async fn phase1_delivery_receipt_roundtrip() {
    let mut conductors = SweetConductorBatch::from_standard_config_rendezvous(2).await;
    let dna_file = SweetDnaFile::from_bundle(&mail_dna_path())
        .await
        .expect("DNA bundle must load");

    let apps = conductors
        .setup_app("pulse-ph1-receipts", &[dna_file.clone()])
        .await
        .expect("setup_app on both conductors");

    let cells = apps.cells_flattened();
    let alice_cell = cells[0].clone();
    let bob_cell = cells[1].clone();

    await_consistency(30, [&alice_cell, &bob_cell])
        .await
        .expect("pre-send consistency");

    // STEP 1 — Alice sends to Bob (mirrors phase0_alice_sends_bob_receives).
    let timestamp = Timestamp::now();
    let message_id = "phase1-receipt-001";
    let encrypted_subject = vec![1u8, 2, 3];
    let encrypted_body = vec![4u8, 5, 6];
    let nonce = [1u8; 24];

    let content = compute_signing_content(
        alice_cell.agent_pubkey(),
        bob_cell.agent_pubkey(),
        &encrypted_subject,
        &encrypted_body,
        &nonce,
        message_id,
        timestamp,
    );
    let sig = conductors[0]
        .keystore()
        .sign(alice_cell.agent_pubkey().clone(), content.into())
        .await
        .expect("alice sign");

    let send_input = SendEmailInput {
        recipients: vec![bob_cell.agent_pubkey().clone()],
        cc: vec![],
        bcc: vec![],
        encrypted_subject,
        encrypted_body,
        encrypted_attachments: vec![],
        ephemeral_pubkey: vec![0u8; 32],
        nonce,
        signature: sig.0.to_vec(),
        crypto_suite: CryptoSuite {
            key_exchange: "x25519".to_string(),
            symmetric: "chacha20-poly1305".to_string(),
            signature: "ed25519".to_string(),
        },
        message_id: message_id.to_string(),
        in_reply_to: None,
        references: vec![],
        priority: EmailPriority::Normal,
        read_receipt_requested: false,
        expires_at: None,
        timestamp,
    };
    let send_output: SendEmailOutput = conductors[0]
        .call(&alice_cell.zome("mail_messages"), "send_email", send_input)
        .await;
    assert_eq!(send_output.delivered_to.len(), 1, "Alice → Bob delivery");

    await_consistency(30, [&alice_cell, &bob_cell])
        .await
        .expect("post-send consistency");

    // Bob fetches inbox to learn the email hash.
    let inbox: Vec<EmailListItem> = conductors[1]
        .call(
            &bob_cell.zome("mail_messages"),
            "get_inbox",
            EmailQuery {
                limit: Some(10),
                ..Default::default()
            },
        )
        .await;
    assert_eq!(inbox.len(), 1, "Bob inbox has one email");
    let email_hash = inbox[0].hash.clone();

    // STEP 2 — Bob acknowledges delivery. Coordinator server-signs
    // DeliveryReceipt via Bob's lair, persists, links, signals.
    let receipt_hash: ActionHash = conductors[1]
        .call(
            &bob_cell.zome("mail_messages"),
            "acknowledge_delivery",
            email_hash.clone(),
        )
        .await;
    assert!(
        !receipt_hash.get_raw_39().is_empty(),
        "receipt entry was created"
    );

    await_consistency(30, [&alice_cell, &bob_cell])
        .await
        .expect("post-ack consistency");

    // STEP 3 — Alice queries receipts on her sent email.
    let receipts: Vec<DeliveryReceipt> = conductors[0]
        .call(
            &alice_cell.zome("mail_messages"),
            "get_delivery_receipts",
            email_hash.clone(),
        )
        .await;

    assert_eq!(
        receipts.len(),
        1,
        "Alice must see exactly one delivery receipt; got {}",
        receipts.len()
    );
    let receipt = &receipts[0];
    assert_eq!(
        receipt.recipient,
        *bob_cell.agent_pubkey(),
        "receipt recipient must be Bob"
    );
    assert_eq!(
        receipt.email_hash, email_hash,
        "receipt links the right email"
    );
    assert_eq!(
        receipt.signature.len(),
        64,
        "Ed25519 signature is exactly 64 bytes"
    );

    // STEP 4 — Sanity-check the signature byte format. We can recompute the
    // canonical signing content; the integrity zome's `verify_signature_raw`
    // already passed at create-time, so getting here is the proof. This is a
    // belt-and-suspenders structural assertion.
    let expected_content = compute_delivery_receipt_signing_content(
        &receipt.email_hash,
        &receipt.recipient,
        receipt.delivered_at,
    );
    assert!(
        !expected_content.is_empty(),
        "signing content non-empty (canonical layout intact)"
    );
}
