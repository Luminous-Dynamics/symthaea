// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Attribution.

use leptos::prelude::*;
use attribution_leptos_types::*;

#[derive(Clone)]
pub struct AttributionCtx {
    pub dependencies: RwSignal<Vec<DependencyView>>,
    pub pledges: RwSignal<Vec<ReciprocityPledgeView>>,
    pub receipts: RwSignal<Vec<UsageReceiptView>>,
}

pub fn provide_attribution_context() {
    let state = AttributionCtx {
        dependencies: RwSignal::new(vec![
            DependencyView { id: "DEP-001".into(), name: "leptos".into(), ecosystem: DependencyEcosystem::RustCrate, maintainer_did: "did:mycelix:gbj".into(), repository_url: Some("https://github.com/leptos-rs/leptos".into()), license: Some("MIT".into()), description: "Reactive web framework for Rust".into(), version: Some("0.8.17".into()), verified: true },
            DependencyView { id: "DEP-002".into(), name: "holochain".into(), ecosystem: DependencyEcosystem::RustCrate, maintainer_did: "did:mycelix:holo".into(), repository_url: Some("https://github.com/holochain/holochain".into()), license: Some("CAL-1.0".into()), description: "Peer-to-peer application framework".into(), version: Some("0.6.0".into()), verified: true },
            DependencyView { id: "DEP-003".into(), name: "nixpkgs".into(), ecosystem: DependencyEcosystem::NixFlake, maintainer_did: "did:mycelix:nixos".into(), repository_url: Some("https://github.com/NixOS/nixpkgs".into()), license: Some("MIT".into()), description: "Nix packages collection".into(), version: None, verified: true },
            DependencyView { id: "DEP-004".into(), name: "serde".into(), ecosystem: DependencyEcosystem::RustCrate, maintainer_did: "did:mycelix:dtolnay".into(), repository_url: Some("https://github.com/serde-rs/serde".into()), license: Some("MIT/Apache-2.0".into()), description: "Serialization framework".into(), version: Some("1.0.228".into()), verified: true },
        ]),
        pledges: RwSignal::new(vec![
            ReciprocityPledgeView { id: "PL-001".into(), dependency_id: "DEP-001".into(), contributor_did: "did:mycelix:user-001".into(), pledge_type: PledgeType::DeveloperTime, amount: Some(20.0), currency: Some("hours".into()), fulfilled: false },
            ReciprocityPledgeView { id: "PL-002".into(), dependency_id: "DEP-003".into(), contributor_did: "did:mycelix:user-001".into(), pledge_type: PledgeType::Documentation, amount: None, currency: None, fulfilled: true },
        ]),
        receipts: RwSignal::new(vec![
            UsageReceiptView { id: "UR-001".into(), dependency_id: "DEP-001".into(), user_did: "did:mycelix:user-001".into(), organization: Some("Luminous Dynamics".into()), version_range: Some("0.8.x".into()), context: Some("Mycelix cluster frontends".into()) },
            UsageReceiptView { id: "UR-002".into(), dependency_id: "DEP-002".into(), user_did: "did:mycelix:user-001".into(), organization: Some("Luminous Dynamics".into()), version_range: Some("0.6.x".into()), context: Some("16 Mycelix hApps".into()) },
        ]),
    };
    provide_context(state.clone());

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[Attribution] Conductor connected — loading real data...".into());

            // Load dependencies
            if let Ok(records) = hc.call_zome_default::<(), Vec<serde_json::Value>>(
                "registry", "get_all_dependencies", &()
            ).await {
                let deps: Vec<DependencyView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !deps.is_empty() {
                    web_sys::console::log_1(&format!("[Attribution] Loaded {} dependencies", deps.len()).into());
                    state.dependencies.set(deps);
                }
            }

            // Load reciprocity pledges
            let owner = "did:mycelix:user-001".to_string();
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "reciprocity", "get_pledges_by_contributor", &owner
            ).await {
                let pledges: Vec<ReciprocityPledgeView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !pledges.is_empty() {
                    web_sys::console::log_1(&format!("[Attribution] Loaded {} pledges", pledges.len()).into());
                    state.pledges.set(pledges);
                }
            }

            // Load usage receipts
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "usage", "get_receipts_by_user", &owner
            ).await {
                let receipts: Vec<UsageReceiptView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !receipts.is_empty() {
                    web_sys::console::log_1(&format!("[Attribution] Loaded {} receipts", receipts.len()).into());
                    state.receipts.set(receipts);
                }
            }
        }
    });
}

/// Extract the entry from a Holochain Record JSON value.
/// Records have structure: { "entry": { "Present": { ...fields... } } }
fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_attribution_context() -> AttributionCtx {
    expect_context::<AttributionCtx>()
}
