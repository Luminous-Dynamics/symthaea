// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::identity_context::use_identity;
use identity_leptos_types::*;

#[component]
pub fn MfaPage() -> impl IntoView {
    let ctx = use_identity();

    let mfa = move || ctx.mfa_state.get();
    let now_secs = move || (js_sys::Date::now() / 1000.0) as i64;

    view! {
        <div class="page page-mfa">
            <h1>"Multi-Factor Authentication"</h1>
            <p class="page-subtitle">"Your identity factors \u{2014} diversity of proof strengthens sovereignty"</p>

            {move || mfa().map(|state| {
                let level = state.assurance_level.clone();
                let next = state.factors_needed_for_next_level();
                let active = state.active_factors();
                let active_count = active.len();
                let now = now_secs();

                view! {
                    // ── Assurance Level Banner ──
                    <div class="assurance-banner">
                        <div class="assurance-current">
                            <span class="assurance-tier">{level.label()}</span>
                            <span class="assurance-strength">
                                {format!("Effective strength: {:.0}%", state.effective_strength * 100.0)}
                            </span>
                        </div>
                        {next.map(|(next_level, factors_needed)| {
                            let next_label = next_level.label().to_string();
                            view! {
                                <div class="assurance-next">
                                    <span class="next-label">{format!("Next: {next_label}")}</span>
                                    <span class="next-req">{format!("{factors_needed}+ factors from 2+ categories needed")}</span>
                                </div>
                            }
                        })}
                    </div>

                    // ── Category Coverage ──
                    <section class="mfa-section">
                        <h2>"Category Coverage"</h2>
                        <p class="section-desc">
                            {format!("{} of 5 categories covered \u{2014} diversity of proof is more valuable than quantity", state.category_count)}
                        </p>
                        <div class="category-grid">
                            {["Cryptographic", "Biometric", "Social Proof", "External Verification", "Knowledge"]
                                .iter().map(|cat| {
                                    let has = active.iter().any(|f| f.factor_type.category() == *cat);
                                    let class = if has { "category-chip active" } else { "category-chip" };
                                    view! {
                                        <span class={class}>{*cat}</span>
                                    }
                                }).collect::<Vec<_>>()}
                        </div>
                    </section>

                    // ── Factor Inventory ──
                    <section class="mfa-section">
                        <h2>{format!("Enrolled Factors ({})", active_count)}</h2>
                        <div class="factor-list">
                            {state.factors.iter().map(|factor| {
                                let strength = factor.visual_strength(now);
                                let strength_class = if strength > 70.0 { "strength-high" }
                                    else if strength > 30.0 { "strength-medium" }
                                    else { "strength-low" };
                                let icon = factor.factor_type.icon().to_string();
                                let label = factor.factor_type.label().to_string();
                                let category = factor.factor_type.category().to_string();
                                let active_label = if factor.active { "Active" } else { "Revoked" };
                                let active_class = if factor.active { "factor-active" } else { "factor-revoked" };
                                let days_since = ((now - factor.last_verified) as f64 / 86400.0).max(0.0) as u32;

                                view! {
                                    <div class={format!("factor-card {active_class}")}>
                                        <div class="factor-header">
                                            <span class="factor-icon">{icon}</span>
                                            <div class="factor-info">
                                                <span class="factor-label">{label}</span>
                                                <span class="factor-category">{category}</span>
                                            </div>
                                            <span class="factor-status">{active_label}</span>
                                        </div>
                                        <div class="factor-strength">
                                            <div class="factor-strength-bar">
                                                <div class={format!("factor-strength-fill {strength_class}")}
                                                     style=format!("width: {strength:.0}%") />
                                            </div>
                                            <div class="factor-strength-meta">
                                                <span>{format!("{strength:.0}%")}</span>
                                                <span class="factor-age">
                                                    {if days_since == 0 {
                                                        "verified today".to_string()
                                                    } else {
                                                        format!("verified {days_since}d ago")
                                                    }}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </section>

                    // ── Enroll New Factor ──
                    <section class="mfa-section">
                        <h2>"Enroll New Factor"</h2>
                        <p class="section-desc">"Add factors from different categories to increase your assurance level"</p>
                        <div class="enroll-grid">
                            {[
                                (FactorType::HardwareKey, "Hardware security key (FIDO2/WebAuthn)"),
                                (FactorType::Biometric, "Fingerprint or face recognition"),
                                (FactorType::GitcoinPassport, "External identity verification"),
                                (FactorType::RecoveryPhrase, "Mnemonic backup phrase"),
                            ].iter().map(|(ft, desc)| {
                                let icon = ft.icon().to_string();
                                let label = ft.label().to_string();
                                let desc = desc.to_string();
                                view! {
                                    <button class="enroll-card" disabled>
                                        <span class="enroll-icon">{icon}</span>
                                        <div class="enroll-info">
                                            <span class="enroll-label">{label}</span>
                                            <span class="enroll-desc">{desc}</span>
                                        </div>
                                    </button>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                        <p class="action-note">"Factor enrollment requires conductor connection"</p>
                    </section>
                }
            })}
        </div>
    }
}
