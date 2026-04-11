// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::consciousness::use_consciousness;
use mycelix_leptos_core::SovereignRadar;
use crate::context::use_climate_context;

#[component]
pub fn ProfilePage() -> impl IntoView {
    let consciousness = use_consciousness();
    let ctx = use_climate_context();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();

    // Check for Praxis environmental credential
    let (has_env_cred, set_has_env_cred) = signal(false);
    wasm_bindgen_futures::spawn_local({
        let hc = hc.clone();
        async move {
            if !hc.is_mock() {
                let result = hc.call_zome_default::<String, bool>(
                    "bridge", "check_environmental_credential",
                    &"did:mycelix:user-001".to_string(),
                ).await.unwrap_or(false);
                set_has_env_cred.set(result);
            }
        }
    });

    view! {
        <div class="page-profile">
            <h1>"Climate Profile"</h1>

            <div class="profile-section">
                <h2>"Sovereign Profile"</h2>
                <SovereignRadar />
            </div>

            <div class="profile-section">
                <h2>"Credentials"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Environmental Credential"</span>
                        <span class="dash-value">
                            {move || if has_env_cred.get() { "Verified" } else { "Not yet earned" }}
                        </span>
                        <span class="dash-sub">
                            {move || if has_env_cred.get() {
                                "Praxis environmental science — unlocks project creation"
                            } else {
                                "Complete environmental science in Praxis to unlock"
                            }}
                        </span>
                    </div>
                </div>
            </div>

            <div class="profile-section">
                <h2>"Carbon Summary"</h2>
                <div class="dashboard-grid">
                    <div class="dash-card">
                        <span class="dash-label">"Credits Held"</span>
                        <span class="dash-value">{move || ctx.credit_summary.get().total_credits.to_string()}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Active Tonnes"</span>
                        <span class="dash-value accent">{move || format!("{:.0}", ctx.credit_summary.get().active_tonnes)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Retired Tonnes"</span>
                        <span class="dash-value">{move || format!("{:.0}", ctx.credit_summary.get().retired_tonnes)}</span>
                    </div>
                    <div class="dash-card">
                        <span class="dash-label">"Net Emissions"</span>
                        <span class="dash-value">{move || {
                            let fp: f64 = ctx.footprints.get().iter().map(|f| f.total()).sum();
                            let retired = ctx.credit_summary.get().retired_tonnes;
                            format!("{:.1} t", fp - retired)
                        }}</span>
                        <span class="dash-sub">"emissions minus retired offsets"</span>
                    </div>
                </div>
            </div>
        </div>
    }
}
