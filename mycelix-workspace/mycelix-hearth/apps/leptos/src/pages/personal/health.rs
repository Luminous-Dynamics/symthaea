// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

#[component]
pub fn HealthPage() -> impl IntoView {
    view! {
        <div class="page health-page">
            <h1 class="page-title">"health vault"</h1>
            <p class="page-subtitle">"your sovereign health records"</p>

            <div class="vault-notice">
                <p>"Health records are encrypted on your source chain."</p>
                <p>"Only you control who sees them."</p>
            </div>

            <section>
                <h2>"Records"</h2>
                <p class="text-muted">"Connect to conductor to load health records from the personal cluster."</p>
            </section>

            <section>
                <h2>"Biometrics"</h2>
                <p class="text-muted">"Biometric history will display as a timeline when connected."</p>
            </section>

            <section>
                <h2>"Consent Grants"</h2>
                <p class="text-muted">"Manage who can access your health data."</p>
            </section>
        </div>
    }
}
