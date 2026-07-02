// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

/// Disclosure scope options.
const SCOPES: &[&str] = &["hidden", "existence only", "selected fields", "full"];

#[component]
pub fn DisclosurePage() -> impl IntoView {
    // Reactive disclosure settings (mock — would be stored in personal vault)
    let (identity_scope, set_identity_scope) = signal(2usize); // selected fields
    let (health_scope, set_health_scope) = signal(1usize);     // existence only
    let (governance_scope, set_governance_scope) = signal(3usize); // full
    let (trust_scope, set_trust_scope) = signal(1usize);       // existence only (tier)

    view! {
        <div class="page disclosure-page">
            <h1 class="page-title">"disclosure settings"</h1>
            <p class="page-subtitle">"control what you share across clusters"</p>

            <div class="disclosure-info">
                <p>"your personal data lives on your source chain. other clusters can only see what you explicitly allow."</p>
            </div>

            <section>
                <h2>"cross-cluster sharing"</h2>

                <DisclosureRow
                    label="identity (profile)"
                    detail="who can see your name, pronouns, avatar"
                    scope=identity_scope
                    set_scope=set_identity_scope
                />
                <DisclosureRow
                    label="health records"
                    detail="medical data, biometrics, consent grants"
                    scope=health_scope
                    set_scope=set_health_scope
                />
                <DisclosureRow
                    label="governance credentials"
                    detail="voting eligibility, phi attestations"
                    scope=governance_scope
                    set_scope=set_governance_scope
                />
                <DisclosureRow
                    label="k-vector trust"
                    detail="trust tier, reputation range"
                    scope=trust_scope
                    set_scope=set_trust_scope
                />
            </section>
        </div>
    }
}

#[component]
fn DisclosureRow(
    label: &'static str,
    detail: &'static str,
    scope: ReadSignal<usize>,
    set_scope: WriteSignal<usize>,
) -> impl IntoView {
    view! {
        <div class="disclosure-row-interactive" role="group" aria-label=label>
            <div class="disclosure-row-info">
                <span class="disclosure-label">{label}</span>
                <span class="disclosure-detail">{detail}</span>
            </div>
            <div class="disclosure-toggles">
                {SCOPES.iter().enumerate().map(|(i, &scope_label)| {
                    let is_active = move || scope.get() == i;
                    view! {
                        <button
                            class=move || format!("disclosure-toggle {}", if is_active() { "disclosure-active" } else { "" })
                            on:click=move |_| set_scope.set(i)
                            aria-label=scope_label
                            aria-pressed=move || is_active().to_string()
                        >
                            {scope_label}
                        </button>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
