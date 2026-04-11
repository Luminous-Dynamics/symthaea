// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Zome Adapter — transforms Holochain wire types to frontend view types.
//!
//! The conductor returns MessagePack-encoded types with ActionHash (39 bytes),
//! AgentPubKey (39 bytes), and Timestamp (microseconds i64). The frontend uses
//! String hashes, String agent keys, and u64 seconds.
//!
//! This module provides the translation layer so mail_context.rs doesn't need
//! to know about Holochain internals.

use mail_leptos_types::*;

#[derive(serde::Deserialize, Debug, Clone)]
struct StoredContact {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub nickname: Option<String>,
    #[serde(default)]
    pub emails: Vec<StoredContactEmail>,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub groups: Vec<String>,
    #[serde(default)]
    pub agent_pub_key: Option<serde_json::Value>,
    #[serde(default)]
    pub is_favorite: bool,
    #[serde(default)]
    pub is_blocked: bool,
    #[serde(default)]
    pub metadata: Option<StoredContactMetadata>,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct StoredContactEmail {
    pub email: String,
    #[serde(default)]
    pub is_primary: bool,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct StoredContactMetadata {
    #[serde(default)]
    pub email_count: u32,
}

/// Wire type matching the zome's EmailListItem exactly.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct WireEmailListItem {
    pub hash: serde_json::Value,     // ActionHash (base64 or bytes)
    pub sender: serde_json::Value,   // AgentPubKey
    pub encrypted_subject: Vec<u8>,
    pub timestamp: serde_json::Value, // Timestamp (i64 microseconds or [seconds, nanos])
    pub priority: String,
    pub is_read: bool,
    pub is_starred: bool,
    pub has_attachments: bool,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub crypto_suite: Option<WireCryptoSuite>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct WireCryptoSuite {
    pub key_exchange: String,
    pub symmetric: String,
    pub signature: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct WireFolder {
    pub hash: serde_json::Value,
    pub name: String,
    #[serde(default)]
    pub encrypted_name: Option<Vec<u8>>,
    #[serde(default)]
    pub is_system: bool,
    #[serde(default)]
    pub sort_order: u32,
    #[serde(default)]
    pub unread_count: u32,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct WireContact {
    pub hash: serde_json::Value,
    pub agent_pub_key: serde_json::Value,
    pub display_name: String,
    #[serde(default)]
    pub nickname: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub groups: Vec<String>,
    #[serde(default)]
    pub is_favorite: bool,
    #[serde(default)]
    pub trust_score: Option<f64>,
}

// ── Conversion functions ──

/// Convert a Holochain hash/key (various formats) to a display string.
pub fn json_hash_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            // Byte array → base64url
            let bytes: Vec<u8> = arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect();
            base64_encode(&bytes)
        }
        _ => format!("{val}"),
    }
}

/// Convert Holochain Timestamp to Unix seconds.
fn wire_timestamp_to_u64(val: &serde_json::Value) -> u64 {
    match val {
        serde_json::Value::Number(n) => {
            let v = n.as_i64().unwrap_or(0);
            if v > 1_000_000_000_000 {
                // Microseconds → seconds
                (v / 1_000_000) as u64
            } else {
                v as u64
            }
        }
        serde_json::Value::Array(arr) => {
            // [seconds, nanos] format
            arr.first().and_then(|v| v.as_u64()).unwrap_or(0)
        }
        _ => 0,
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut result = String::with_capacity((bytes.len() * 4 + 2) / 3);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(CHARS[((n >> 6) & 0x3F) as usize] as char); }
        if chunk.len() > 2 { result.push(CHARS[(n & 0x3F) as usize] as char); }
    }
    result
}

// ── Public adapter functions ──

/// Adapt wire inbox response to frontend EmailListItem vec.
/// `contacts` is used to look up sender_name from agent key.
pub fn adapt_inbox(
    wire: Vec<WireEmailListItem>,
    contacts: &[ContactView],
) -> Vec<mail_leptos_types::EmailListItem> {
    wire.into_iter().map(|w| {
        let sender_key = json_hash_to_string(&w.sender);
        let sender_name = contacts.iter()
            .find(|c| c.agent_pub_key.as_deref() == Some(&sender_key))
            .map(|c| c.display_name.clone());

        let crypto = w.crypto_suite.map(|cs| CryptoSuiteView {
            key_exchange: cs.key_exchange,
            symmetric: cs.symmetric,
            signature: cs.signature,
        }).unwrap_or_else(|| CryptoSuiteView {
            key_exchange: "unknown".into(),
            symmetric: "unknown".into(),
            signature: "unknown".into(),
        });

        mail_leptos_types::EmailListItem {
            hash: json_hash_to_string(&w.hash),
            sender: sender_key,
            sender_name,
            encrypted_subject: w.encrypted_subject.clone(),
            subject: crate::crypto::decode_transport_text(&w.encrypted_subject)
                .or_else(|| Some("Encrypted message".into())),
            snippet: None,
            timestamp: wire_timestamp_to_u64(&w.timestamp),
            priority: match w.priority.as_str() {
                "Urgent" => EmailPriority::Urgent,
                "High" => EmailPriority::High,
                "Low" => EmailPriority::Low,
                _ => EmailPriority::Normal,
            },
            is_read: w.is_read,
            is_starred: w.is_starred,
            star_type: None,
            is_pinned: false,
            is_muted: false,
            is_snoozed: false,
            snooze_until: None,
            has_attachments: w.has_attachments,
            labels: vec![],
            thread_id: w.thread_id,
            crypto_suite: crypto,
        }
    }).collect()
}

/// Adapt wire folders to frontend FolderView vec.
pub fn adapt_folders(wire: Vec<WireFolder>) -> Vec<FolderView> {
    wire.into_iter().map(|w| {
        FolderView {
            hash: json_hash_to_string(&w.hash),
            name: if let Some(enc) = &w.encrypted_name {
                String::from_utf8_lossy(enc).to_string()
            } else {
                w.name
            },
            is_system: w.is_system,
            sort_order: w.sort_order as i32,
            unread_count: w.unread_count,
        }
    }).collect()
}

/// Adapt wire contacts to frontend ContactView vec.
pub fn adapt_contacts(wire: Vec<WireContact>) -> Vec<ContactView> {
    wire.into_iter().map(|w| {
        ContactView {
            hash: json_hash_to_string(&w.hash),
            id: w.display_name.to_lowercase().replace(' ', "-"),
            display_name: w.display_name,
            nickname: w.nickname,
            email: w.email,
            agent_pub_key: Some(json_hash_to_string(&w.agent_pub_key)),
            organization: w.organization,
            avatar: None,
            groups: w.groups,
            is_favorite: w.is_favorite,
            is_blocked: false,
            email_count: 0,
            trust_score: w.trust_score,
        }
    }).collect()
}

fn adapt_stored_contact(contact: StoredContact) -> ContactView {
    let primary_email = contact
        .emails
        .iter()
        .find(|email| email.is_primary)
        .or_else(|| contact.emails.first())
        .map(|email| email.email.clone());

    ContactView {
        hash: contact.id.clone(),
        id: contact.id,
        display_name: contact.display_name,
        nickname: contact.nickname,
        email: primary_email,
        agent_pub_key: contact.agent_pub_key.as_ref().map(json_hash_to_string),
        organization: contact.organization,
        avatar: None,
        groups: contact.groups,
        is_favorite: contact.is_favorite,
        is_blocked: contact.is_blocked,
        email_count: contact.metadata.map(|metadata| metadata.email_count).unwrap_or(0),
        trust_score: None,
    }
}

pub fn adapt_contact_value(value: serde_json::Value) -> Option<ContactView> {
    serde_json::from_value::<ContactView>(value.clone())
        .ok()
        .or_else(|| serde_json::from_value::<WireContact>(value.clone()).ok().map(|wire| adapt_contacts(vec![wire]).remove(0)))
        .or_else(|| serde_json::from_value::<StoredContact>(value).ok().map(adapt_stored_contact))
}

pub fn adapt_contact_values(values: Vec<serde_json::Value>) -> Vec<ContactView> {
    values.into_iter().filter_map(adapt_contact_value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapts_rich_stored_contact_shape() {
        let value = serde_json::json!({
            "id": "alice-id",
            "display_name": "Alice Example",
            "nickname": "Alice",
            "emails": [
                { "email": "secondary@example.com", "is_primary": false },
                { "email": "alice@example.com", "is_primary": true }
            ],
            "organization": "Mycelix",
            "groups": ["friends"],
            "agent_pub_key": "uhCAk_alice",
            "is_favorite": true,
            "is_blocked": false,
            "metadata": { "email_count": 7 }
        });

        let adapted = adapt_contact_value(value).expect("stored contact should adapt");
        assert_eq!(adapted.id, "alice-id");
        assert_eq!(adapted.display_name, "Alice Example");
        assert_eq!(adapted.email.as_deref(), Some("alice@example.com"));
        assert_eq!(adapted.agent_pub_key.as_deref(), Some("uhCAk_alice"));
        assert_eq!(adapted.email_count, 7);
    }

    #[test]
    fn adapts_wire_contact_shape() {
        let value = serde_json::json!({
            "hash": [1, 2, 3, 4],
            "agent_pub_key": [9, 8, 7, 6],
            "display_name": "Bob Example",
            "email": "bob@example.com",
            "groups": ["work"],
            "is_favorite": false,
            "trust_score": 0.8
        });

        let adapted = adapt_contact_value(value).expect("wire contact should adapt");
        assert_eq!(adapted.display_name, "Bob Example");
        assert_eq!(adapted.email.as_deref(), Some("bob@example.com"));
        assert!(adapted.agent_pub_key.is_some());
        assert_eq!(adapted.trust_score, Some(0.8));
    }

    #[test]
    fn adapts_existing_contact_view_passthrough() {
        let value = serde_json::json!({
            "hash": "hash-1",
            "id": "contact-1",
            "display_name": "Carol Example",
            "nickname": null,
            "email": "carol@example.com",
            "agent_pub_key": "uhCAk_carol",
            "organization": null,
            "avatar": null,
            "groups": [],
            "is_favorite": false,
            "is_blocked": false,
            "email_count": 3,
            "trust_score": 0.5
        });

        let adapted = adapt_contact_value(value).expect("contact view should adapt");
        assert_eq!(adapted.id, "contact-1");
        assert_eq!(adapted.email_count, 3);
        assert_eq!(adapted.trust_score, Some(0.5));
    }
}
