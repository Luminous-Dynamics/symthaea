// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Employer Portal — Talent search and reputation-weighted verification.

use leptos::prelude::*;

#[component]
pub fn EmployerPortal() -> impl IntoView {
    let (search_query, set_search_query) = signal("2512".to_string()); // Default to SW Dev
    let (is_searching, set_is_searching) = signal(false);

    view! {
        <div class="employer-portal">
            <header class="employer-header">
                <div class="portal-brand">
                    <span class="portal-icon">"\u{1F4BC}"</span>
                    <h1>"Praxis Talent Mesh"</h1>
                </div>
                <div class="mesh-status">
                    <span class="pulse-dot"></span>
                    "Searching Local Swarm Mesh (LoRa + 802.11s)"
                </div>
            </header>

            <section class="search-hero">
                <div class="search-container">
                    <div class="search-input-wrapper">
                        <span class="search-icon">"\u{1F50D}"</span>
                        <input 
                            type="text" 
                            placeholder="Enter ESCO code (e.g. 2512 for Software Dev)" 
                            prop:value=search_query
                            on:input=move |ev| set_search_query.set(event_target_value(&ev))
                        />
                    </div>
                    <button 
                        class="btn-search"
                        on:click=move |_| {
                            set_is_searching.set(true);
                            // Simulate mesh search delay
                            wasm_bindgen_futures::spawn_local(async move {
                                gloo_timers::future::sleep(std::time::Duration::from_millis(800)).await;
                                set_is_searching.set(false);
                            });
                        }
                    >
                        {move || if is_searching.get() { "Scanning..." } else { "Scan Mesh" }}
                    </button>
                </div>
            </section>

            <main class="results-layout">
                <aside class="search-filters">
                    <h4>"Search Parameters"</h4>
                    <div class="filter-group">
                        <label>"Framework"</label>
                        <select><option>"ESCO (Europe)"</option><option>"O*NET (USA)"</option></select>
                    </div>
                    <div class="filter-group">
                        <label>"Min. MATL Trust"</label>
                        <input type="range" min="0" max="1000" step="50" />
                    </div>
                    <div class="filter-group">
                        <label>"Proof Type"</label>
                        <div class="checkbox-list">
                            <label><input type="checkbox" checked /> "Holonic Capstone"</label>
                            <label><input type="checkbox" checked /> "ZK-Mastery Proof"</label>
                            <label><input type="checkbox" /> "Peer Endorsement"</label>
                        </div>
                    </div>
                </aside>

                <div class="results-main">
                    <div class="results-header">
                        <h3>"Available Talent" <span class="count">"2 found in range"</span></h3>
                    </div>

                    <div class="results-grid">
                        <TalentCard 
                            did="did:mycelix:x7f2...89a"
                            trust=880
                            skills=vec!["Software Developer", "Systems Architect"]
                            zk_verified=true
                        />
                        <TalentCard 
                            did="did:mycelix:u9b4...22c"
                            trust=740
                            skills=vec!["Software Developer"]
                            zk_verified=true
                        />
                    </div>
                </div>
            </main>
        </div>
    }
}

#[component]
fn TalentCard(
    did: &'static str,
    trust: u16,
    skills: Vec<&'static str>,
    zk_verified: bool,
) -> impl IntoView {
    view! {
        <div class="talent-card">
            <div class="talent-header">
                <div class="talent-meta">
                    <span class="talent-did">{did}</span>
                    <div class="talent-skills">
                        {skills.into_iter().map(|s| view! { <span class="skill-tag">{s}</span> }).collect_view()}
                    </div>
                </div>
                <div class="talent-scores">
                    <div class="score-pill">
                        <span class="score-label">"Learning Rep"</span>
                        <span class="score-value">{trust}</span>
                    </div>
                    <div class="score-pill vitality">
                        <span class="score-label">"Craft Vitality"</span>
                        <span class="score-value">"92%"</span>
                    </div>
                </div>
            </div>

            <div class="talent-verification">
                {if zk_verified {
                    view! {
                        <div class="zk-badge">
                            <span class="zk-icon">"\u{1F512}"</span>
                            "ZK-Mastery Verified (Threshold > 85%)"
                        </div>
                    }.into_any()
                } else {
                    view! { <span></span> }.into_any()
                }}
            </div>

            <div class="talent-actions">
                <button class="btn-sm btn-primary">"Request Full CLR"</button>
                <button class="btn-sm btn-outline">"View Capstone Summary"</button>
            </div>
        </div>
    }
}
