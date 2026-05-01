// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Three-zone privacy enforcement for Prism.
//!
//! The encoding gate is the ONLY path to the HDC encoder and DHT.
//! No code path can write to DHT without passing through this gate.
//!
//! Zones:
//! - **Private**: Never encoded, never shared. Banks, email, corporate.
//! - **Local**: HDC encoded for personal semantic memory. Never broadcast.
//! - **Public**: Encoded and offered to DHT. Requires E3+/E4 OR user consent.

use prism_common::ContentZone;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Decision from the encoding gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodingDecision {
    /// Content must not be encoded at all.
    Deny,
    /// Content may be encoded for local semantic memory only.
    LocalOnly,
    /// Content may be encoded and shared to DHT.
    EncodeAndShare,
}

/// The encoding gate — the non-bypassable enforcement point for privacy.
///
/// This is the ONLY path to HDC encoding and DHT writes.
pub fn encoding_gate(zone: ContentZone) -> EncodingDecision {
    match zone {
        ContentZone::Private => EncodingDecision::Deny,
        ContentZone::Local => EncodingDecision::LocalOnly,
        ContentZone::Public => EncodingDecision::EncodeAndShare,
    }
}

/// Persistent store of user consent decisions (per-domain).
///
/// When a domain is approved for Public sharing, it's stored here.
/// Stored as JSON on disk for persistence across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentStore {
    /// Domains the user has approved for Public zone (DHT sharing).
    approved_domains: HashSet<String>,
    /// Domains the user has explicitly marked as Private.
    denied_domains: HashSet<String>,
}

impl ConsentStore {
    pub fn new() -> Self {
        Self {
            approved_domains: HashSet::new(),
            denied_domains: HashSet::new(),
        }
    }

    /// Check if a domain has been approved for Public sharing.
    pub fn is_approved(&self, domain: &str) -> bool {
        self.approved_domains.contains(domain)
    }

    /// Check if a domain has been explicitly denied.
    pub fn is_denied(&self, domain: &str) -> bool {
        self.denied_domains.contains(domain)
    }

    /// Approve a domain for Public zone sharing.
    pub fn approve_domain(&mut self, domain: String) {
        self.denied_domains.remove(&domain);
        self.approved_domains.insert(domain);
    }

    /// Deny a domain (force Private zone).
    pub fn deny_domain(&mut self, domain: String) {
        self.approved_domains.remove(&domain);
        self.denied_domains.insert(domain);
    }

    /// Resolve zone considering user consent.
    ///
    /// If the reflex arc classified as Local, check if user has approved
    /// this domain for Public sharing.
    pub fn resolve_zone(&self, base_zone: ContentZone, domain: &str) -> ContentZone {
        // User denial always wins
        if self.is_denied(domain) {
            return ContentZone::Private;
        }

        match base_zone {
            ContentZone::Private => ContentZone::Private, // Never override Private
            ContentZone::Local => {
                if self.is_approved(domain) {
                    ContentZone::Public
                } else {
                    ContentZone::Local
                }
            }
            ContentZone::Public => ContentZone::Public,
        }
    }

    /// Number of approved domains.
    pub fn approved_count(&self) -> usize {
        self.approved_domains.len()
    }

    /// Number of denied domains.
    pub fn denied_count(&self) -> usize {
        self.denied_domains.len()
    }
}

impl Default for ConsentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_private_denies() {
        assert_eq!(encoding_gate(ContentZone::Private), EncodingDecision::Deny);
    }

    #[test]
    fn gate_local_allows_local_only() {
        assert_eq!(
            encoding_gate(ContentZone::Local),
            EncodingDecision::LocalOnly
        );
    }

    #[test]
    fn gate_public_allows_share() {
        assert_eq!(
            encoding_gate(ContentZone::Public),
            EncodingDecision::EncodeAndShare
        );
    }

    #[test]
    fn consent_approve_upgrades_local_to_public() {
        let mut store = ConsentStore::new();
        store.approve_domain("example.com".to_string());
        assert_eq!(
            store.resolve_zone(ContentZone::Local, "example.com"),
            ContentZone::Public
        );
    }

    #[test]
    fn consent_deny_downgrades_to_private() {
        let mut store = ConsentStore::new();
        store.deny_domain("evil.com".to_string());
        assert_eq!(
            store.resolve_zone(ContentZone::Local, "evil.com"),
            ContentZone::Private
        );
        assert_eq!(
            store.resolve_zone(ContentZone::Public, "evil.com"),
            ContentZone::Private
        );
    }

    #[test]
    fn consent_private_never_overridden() {
        let mut store = ConsentStore::new();
        store.approve_domain("bank.com".to_string());
        // Even with approval, Private stays Private
        assert_eq!(
            store.resolve_zone(ContentZone::Private, "bank.com"),
            ContentZone::Private
        );
    }

    #[test]
    fn consent_unapproved_stays_local() {
        let store = ConsentStore::new();
        assert_eq!(
            store.resolve_zone(ContentZone::Local, "unknown.com"),
            ContentZone::Local
        );
    }

    #[test]
    fn consent_deny_removes_approval() {
        let mut store = ConsentStore::new();
        store.approve_domain("site.com".to_string());
        assert!(store.is_approved("site.com"));
        store.deny_domain("site.com".to_string());
        assert!(!store.is_approved("site.com"));
        assert!(store.is_denied("site.com"));
    }

    #[test]
    fn subdomain_independence() {
        let mut store = ConsentStore::new();
        store.approve_domain("example.com".to_string());
        // Subdomain is independent — not automatically approved
        assert!(!store.is_approved("sub.example.com"));
        assert_eq!(
            store.resolve_zone(ContentZone::Local, "sub.example.com"),
            ContentZone::Local
        );
    }

    #[test]
    fn serialize_roundtrip() {
        let mut store = ConsentStore::new();
        store.approve_domain("wiki.org".to_string());
        store.deny_domain("evil.com".to_string());
        let json = serde_json::to_string(&store).unwrap();
        let restored: ConsentStore = serde_json::from_str(&json).unwrap();
        assert!(restored.is_approved("wiki.org"));
        assert!(restored.is_denied("evil.com"));
        assert_eq!(restored.approved_count(), 1);
        assert_eq!(restored.denied_count(), 1);
    }

    #[test]
    fn empty_domain_handled() {
        let mut store = ConsentStore::new();
        store.approve_domain(String::new());
        // Should not panic or corrupt state
        assert!(store.is_approved(""));
        assert!(!store.is_approved("example.com"));
    }
}
