// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified Mycelix identity — one First Breath, domain keys derived lazily.

pub mod master_key;

use leptos::prelude::*;
use portal_domain_trait::ConsciousnessTier;

/// Vault state across the entire portal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VaultState {
    NoVault,
    Locked,
    Unlocked,
}

/// The portal-wide identity context.
#[derive(Clone, Debug)]
pub struct PortalIdentity {
    /// DID string (e.g., "did:mycelix:uhCAk...")
    pub did: RwSignal<Option<String>>,
    /// Vault state
    pub vault: RwSignal<VaultState>,
    /// Consciousness profile combined score (0.0-1.0)
    pub consciousness_score: RwSignal<f64>,
    /// Derived tier
    pub tier: Memo<ConsciousnessTier>,
    /// Which domain keys have been derived this session
    pub active_domains: RwSignal<Vec<String>>,
}

impl PortalIdentity {
    pub fn new() -> Self {
        let consciousness_score = RwSignal::new(0.35); // Default: Participant
        let tier = Memo::new(move |_| {
            ConsciousnessTier::from_score(consciousness_score.get())
        });

        let vault = if master_key::has_stored_master() {
            VaultState::Locked
        } else {
            VaultState::NoVault
        };

        Self {
            did: RwSignal::new(None),
            vault: RwSignal::new(vault),
            consciousness_score,
            tier,
            active_domains: RwSignal::new(vec![]),
        }
    }

    /// Derive a domain-specific key from the master.
    /// Returns the 32-byte key if the vault is unlocked.
    pub fn domain_key(&self, domain_context: &[u8]) -> Option<[u8; 32]> {
        if self.vault.get() != VaultState::Unlocked {
            return None;
        }
        // In a real implementation, the master key would be held in memory
        // and domain keys derived via HKDF on demand.
        // For now, return a deterministic derivation placeholder.
        Some(master_key::derive_domain_key(domain_context))
    }
}
