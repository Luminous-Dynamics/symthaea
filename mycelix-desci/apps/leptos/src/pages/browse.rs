// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browse and search claims.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use crate::types::*;
use crate::components::claim_card::ClaimCard;

/// Demo claims for initial rendering (before API connection).
fn demo_claims() -> Vec<ClaimResponse> {
    vec![
        ClaimResponse {
            id: "lazar-gravity-a".to_string(),
            tier: EpistemicTier::E0,
            content: ClaimContent {
                dataset_hash: "none".to_string(),
                description: "Extended strong nuclear force (Gravity-A) via Element 115 produces gravity wave at 11.4 GHz".to_string(),
                category: "Modified Gravity".to_string(),
                keywords: vec!["lazar".into(), "element-115".into(), "gravity-a".into()],
                storage_ref: None,
                reproducibility_score: None,
                license: None,
            },
            creator: "Bob Lazar (1989)".to_string(),
            created_at: "1989-11-01T00:00:00Z".to_string(),
            verifications_count: 0,
            provenance_count: 0,
        },
        ClaimResponse {
            id: "arts-parts-waveguide".to_string(),
            tier: EpistemicTier::E0,
            content: ClaimContent {
                dataset_hash: "ornl-2022".to_string(),
                description: "Bi-Mg(Zn) layered metamaterial acts as THz waveguide for anti-gravity propulsion".to_string(),
                category: "Metamaterials".to_string(),
                keywords: vec!["bismuth".into(), "magnesium".into(), "waveguide".into(), "metamaterial".into()],
                storage_ref: None,
                reproducibility_score: Some(0.15),
                license: None,
            },
            creator: "Art Bell / ORNL Analysis".to_string(),
            created_at: "2022-01-15T00:00:00Z".to_string(),
            verifications_count: 1,
            provenance_count: 1,
        },
        ClaimResponse {
            id: "island-stability-z120".to_string(),
            tier: EpistemicTier::E2,
            content: ClaimContent {
                dataset_hash: "symthaea-nuclear-sweep".to_string(),
                description: "Superheavy stability island centered at Z=115-120, N=180. Element 120 (A=294) has shell correction -22.81 MeV.".to_string(),
                category: "Nuclear Physics".to_string(),
                keywords: vec!["island-of-stability".into(), "superheavy".into(), "shell-model".into()],
                storage_ref: Some("symthaea-nuclear::island_stability".to_string()),
                reproducibility_score: Some(0.85),
                license: Some("AGPL-3.0".to_string()),
            },
            creator: "Symthaea Nuclear Sweep".to_string(),
            created_at: "2026-04-04T00:00:00Z".to_string(),
            verifications_count: 0,
            provenance_count: 1,
        },
    ]
}

/// Browse claims page with search and filters.
#[component]
pub fn BrowsePage() -> impl IntoView {
    let (search_text, set_search_text) = signal(String::new());
    let (tier_filter, set_tier_filter) = signal(String::new());

    // Try API, fall back to demo data
    let api_claims = leptos::prelude::LocalResource::new(|| async {
        crate::api::query_claims(&crate::types::QueryRequest::default()).await.ok()
    });

    let claims = move || {
        api_claims
            .get()
            .flatten()
            .map(|r| r.results)
            .unwrap_or_else(demo_claims)
    };

    let filtered_claims = move || {
        let search = search_text.get().to_lowercase();
        let tier = tier_filter.get();
        let all_claims = claims();

        all_claims.into_iter()
            .filter(|c| {
                if !search.is_empty() {
                    let matches = c.content.description.to_lowercase().contains(&search)
                        || c.content.category.to_lowercase().contains(&search)
                        || c.content.keywords.iter().any(|k| k.to_lowercase().contains(&search));
                    if !matches { return false; }
                }
                if !tier.is_empty() {
                    let tier_str = format!("{:?}", c.tier);
                    if tier_str != tier { return false; }
                }
                true
            })
            .collect::<Vec<_>>()
    };

    view! {
        <div class="page-container">
            <h1 class="page-title">"Browse Claims"</h1>

            <div class="search-bar">
                <input
                    type="text"
                    placeholder="Search claims..."
                    on:input=move |ev| {
                        use wasm_bindgen::JsCast;
                        let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                        set_search_text.set(target.value());
                    }
                />
                <select on:change=move |ev| {
                    use wasm_bindgen::JsCast;
                    let target: web_sys::HtmlSelectElement = ev.target().unwrap().unchecked_into();
                    set_tier_filter.set(target.value());
                }>
                    <option value="">"All Tiers"</option>
                    <option value="E0">"E0: Unverified"</option>
                    <option value="E1">"E1: Testimonial"</option>
                    <option value="E2">"E2: Verifiable"</option>
                    <option value="E3">"E3: Reproducible"</option>
                    <option value="E4">"E4: Peer Reviewed"</option>
                </select>
            </div>

            <div class="claim-grid">
                {move || filtered_claims().into_iter().map(|claim| {
                    view! { <ClaimCard claim=claim /> }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
