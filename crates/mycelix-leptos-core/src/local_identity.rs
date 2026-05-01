// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Local agent identity — per-visitor DID stored in localStorage.
//!
//! Each visitor gets a unique `did:mycelix:<random>` identity on first visit.
//! This creates ownership of mock data without needing a conductor.
//! When a real conductor connects, the local DID can be linked to the
//! Holochain agent public key.

use leptos::prelude::*;

const IDENTITY_KEY: &str = "mycelix_local_did";

/// Generate or retrieve the local agent DID.
pub fn local_did() -> String {
    // Try to load from localStorage
    if let Some(did) = load_string(IDENTITY_KEY) {
        return did;
    }

    // Generate new DID
    let did = generate_did();
    save_string(IDENTITY_KEY, &did);
    did
}

/// Provide local identity as Leptos context.
pub fn provide_local_identity() {
    let did = local_did();
    provide_context(LocalIdentity { did });
}

/// Retrieve local identity from context.
pub fn use_local_identity() -> LocalIdentity {
    expect_context::<LocalIdentity>()
}

/// Local identity state.
#[derive(Clone, Debug)]
pub struct LocalIdentity {
    pub did: String,
}

impl LocalIdentity {
    /// Short display name (last 8 chars of DID).
    pub fn short_name(&self) -> String {
        let suffix = self.did.rsplit(':').next().unwrap_or(&self.did);
        if suffix.len() > 8 {
            format!("{}...", &suffix[..8])
        } else {
            suffix.to_string()
        }
    }
}

fn generate_did() -> String {
    // Generate 12 random bytes for a unique DID
    let mut bytes = [0u8; 12];
    for b in bytes.iter_mut() {
        *b = (js_sys::Math::random() * 256.0) as u8;
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("did:mycelix:{hex}")
}

// --- localStorage helpers ---

pub fn load_string(key: &str) -> Option<String> {
    web_sys::window()?
        .local_storage()
        .ok()??
        .get_item(key)
        .ok()?
}

pub fn save_string(key: &str, value: &str) {
    if let Some(Ok(Some(storage))) = web_sys::window().map(|w| w.local_storage()) {
        let _ = storage.set_item(key, value);
    }
}

/// Load a JSON-serializable value from localStorage.
pub fn load_json<T: serde::de::DeserializeOwned>(key: &str) -> Option<T> {
    let s = load_string(key)?;
    serde_json::from_str(&s).ok()
}

/// Save a JSON-serializable value to localStorage.
pub fn save_json<T: serde::Serialize>(key: &str, value: &T) {
    if let Ok(s) = serde_json::to_string(value) {
        save_string(key, &s);
    }
}
