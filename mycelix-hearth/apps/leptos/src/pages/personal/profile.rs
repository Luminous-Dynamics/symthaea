// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::mock_data;
use mycelix_leptos_core::use_consciousness;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let profile = mock_data::mock_profile();
    let trust = mock_data::mock_trust_credential();

    let name = profile.display_name.clone();
    let bio = profile.bio.clone().unwrap_or_default();
    let metadata: Vec<_> = profile.metadata.iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    view! {
        <div class="page profile-page">
            <h1 class="page-title">"identity vault"</h1>
            <p class="page-subtitle">"your sovereign identity"</p>

            <div class="profile-card">
                <div class="profile-avatar">
                    <div class="avatar-placeholder">{name.chars().next().unwrap_or('?').to_string()}</div>
                </div>
                <div class="profile-info">
                    <h2>{name}</h2>
                    <p class="profile-bio">{bio}</p>
                </div>
            </div>

            // Trust tier
            <section class="trust-section">
                <h2>"Trust Status"</h2>
                <div class="trust-card">
                    <div class="trust-tier">
                        <span class=format!("tier-badge {}", trust.trust_tier.css_class())>
                            {trust.trust_tier.label()}
                        </span>
                        <span class="trust-range">
                            {format!("Score range: {:.0}% - {:.0}%",
                                trust.trust_score_range.lower * 100.0,
                                trust.trust_score_range.upper * 100.0)}
                        </span>
                    </div>
                    <div class="consciousness-tier">
                        <span>"Consciousness tier: "</span>
                        <span class=move || format!("tier-badge {}", consciousness.tier.get().css_class())>
                            {move || consciousness.tier.get().label()}
                        </span>
                    </div>
                </div>
            </section>

            // Metadata
            {(!metadata.is_empty()).then(|| view! {
                <section>
                    <h2>"Details"</h2>
                    <div class="metadata-list">
                        {metadata.iter().map(|(k, v)| {
                            let key = k.clone();
                            let val = v.clone();
                            view! {
                                <div class="metadata-row">
                                    <span class="metadata-key">{key}</span>
                                    <span class="metadata-value">{val}</span>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                </section>
            })}
        </div>
    }
}
