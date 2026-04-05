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
        // ── Verification Successes ──
        claim("nif-ignition", EpistemicTier::E4, "NIF fusion ignition: Q ratio 1.54→4.13, eight successful shots independently verified", "Nuclear Physics", &["nif","fusion","ignition"], "LLNL/NIF", "2025-04-01", 12),
        claim("higgs-boson", EpistemicTier::E4, "Higgs boson at ~125 GeV: ATLAS + CMS independent discovery, 5.9σ significance. Nobel 2013", "Particle Physics", &["higgs","lhc","nobel"], "ATLAS+CMS (2012)", "2012-07-04", 50),
        claim("ligo-gw150914", EpistemicTier::E4, "Gravitational wave detection: binary black hole merger, SNR=24, false alarm < 1/203,000 years", "General Relativity", &["ligo","gravitational-waves","nobel"], "LIGO/Virgo (2015)", "2016-02-11", 30),
        claim("alphafold", EpistemicTier::E4, "AlphaFold2 protein structure prediction: median GDT-TS 92.4 across CASP14 targets. Nobel 2024", "Biophysics", &["alphafold","protein","deep-learning"], "DeepMind (2021)", "2021-07-15", 25),
        // ── Replication Failures ──
        claim("cold-fusion", EpistemicTier::E0, "Cold fusion: deuterium fusion at room temperature. Gamow factor ~10^-2700. Never replicated", "Nuclear Physics", &["cold-fusion","debunked"], "Pons & Fleischmann (1989)", "1989-03-23", 0),
        claim("lk99", EpistemicTier::E0, "LK-99 room-temperature superconductor: Cu₂S impurity, debunked in 3 weeks", "Condensed Matter", &["lk-99","superconductor","debunked"], "Lee & Kim (2023)", "2023-07-22", 0),
        claim("emdrive", EpistemicTier::E0, "EMDrive: all thrust was magnetic cable artifact. Conservation law violation", "Classical Mechanics", &["emdrive","debunked"], "TU Dresden (2021)", "2021-03-15", 0),
        // ── Active Science ──
        claim("nickelates", EpistemicTier::E3, "Nickelate La₃Ni₂O₇ superconductivity at ~80K, reproduced by multiple groups, SLAC ambient pressure", "Condensed Matter", &["nickelate","superconductor"], "Multiple (2023-25)", "2025-02-01", 8),
        claim("hubble-tension", EpistemicTier::E4, "Hubble tension: local H₀≈73 vs CMB H₀≈67 km/s/Mpc at 5σ. Both sides E4-verified. Unresolved", "Cosmology", &["hubble-tension","dark-energy"], "Riess+Planck", "2024-01-01", 15),
        claim("gnome", EpistemicTier::E3, "GNoME predicted 2.2M new crystal structures, 380K stable candidates. 736 independently synthesized", "Condensed Matter", &["gnome","materials-discovery"], "DeepMind (2023)", "2023-11-29", 5),
        claim("quantum-ec", EpistemicTier::E3, "Quantum error correction: <10^-6 error rate with 12 logical qubits via topological code", "Quantum Mechanics", &["quantum","error-correction"], "Microsoft+IBM (2026)", "2026-01-15", 4),
        claim("jwst-deep", EpistemicTier::E4, "JWST deep field reveals galaxies at z>13, challenging ΛCDM timeline predictions", "Cosmology", &["jwst","deep-field","high-redshift"], "NASA/ESA/CSA", "2022-07-12", 20),
    ]
}

/// Shorthand claim constructor.
fn claim(id: &str, tier: EpistemicTier, desc: &str, cat: &str, kw: &[&str], creator: &str, date: &str, verifications: usize) -> ClaimResponse {
    ClaimResponse {
        id: id.to_string(),
        tier,
        content: ClaimContent {
            dataset_hash: id.to_string(),
            description: desc.to_string(),
            category: cat.to_string(),
            keywords: kw.iter().map(|s| s.to_string()).collect(),
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        },
        creator: creator.to_string(),
        created_at: format!("{}T00:00:00Z", date),
        verifications_count: verifications,
        provenance_count: 1,
    }
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

        let mut results: Vec<_> = all_claims.into_iter()
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
            .collect();
        // Sort by tier (E4 first, then E3, E2, E1, E0) so verified science is prominent
        results.sort_by(|a, b| {
            let tier_val = |t: &crate::types::EpistemicTier| match t {
                crate::types::EpistemicTier::E4 => 4,
                crate::types::EpistemicTier::E3 => 3,
                crate::types::EpistemicTier::E2 => 2,
                crate::types::EpistemicTier::E1 => 1,
                crate::types::EpistemicTier::E0 => 0,
            };
            tier_val(&b.tier).cmp(&tier_val(&a.tier))
        });
        results
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
