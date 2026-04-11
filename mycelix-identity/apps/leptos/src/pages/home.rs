// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::identity_context::use_identity;
use identity_leptos_types::*;
use mycelix_leptos_core::{SovereignRadar, SovereignRadarSize};

#[component]
pub fn HomePage() -> impl IntoView {
    let ctx = use_identity();

    // Derived signals
    let short_did = move || {
        ctx.did_document.get()
            .map(|d| d.short_id())
            .unwrap_or_else(|| "No DID".into())
    };

    let name = move || {
        ctx.my_name.get()
            .map(|n| n.canonical.clone())
            .unwrap_or_else(|| "unnamed".into())
    };

    let trust_tier = move || {
        ctx.reputation.get()
            .map(|r| r.trust_tier.clone())
            .unwrap_or(TrustTier::Observer)
    };

    let assurance = move || {
        ctx.mfa_state.get()
            .map(|m| m.assurance_level.clone())
            .unwrap_or(AssuranceLevel::Anonymous)
    };

    let cred_count = move || ctx.credentials_held.get().len();
    let factor_count = move || {
        ctx.mfa_state.get()
            .map(|m| m.active_factors().len())
            .unwrap_or(0)
    };

    let composite_score = move || {
        ctx.reputation.get()
            .map(|r| r.composite_score)
            .unwrap_or(0.0)
    };

    let consciousness = move || {
        ctx.reputation.get()
            .map(|r| r.consciousness_profile.clone())
            .unwrap_or(ConsciousnessProfileView {
                identity: 0.0, reputation: 0.0, community: 0.0, engagement: 0.0,
            })
    };

    view! {
        <div class="page page-home">
            // ── DID Card ──
            <div class="did-card">
                <div class="did-card-header">
                    <div class="did-avatar">
                        <span class="did-avatar-letter">
                            {move || name().chars().next().unwrap_or('?').to_uppercase().to_string()}
                        </span>
                    </div>
                    <div class="did-card-info">
                        <h1 class="did-name">{name}</h1>
                        <p class="did-id">{short_did}</p>
                    </div>
                    <div class="did-card-badges">
                        <span class="trust-badge" style=move || format!("--badge-color: {}", trust_tier().css_color())>
                            {move || trust_tier().label()}
                        </span>
                        <span class=move || format!("assurance-badge {}", assurance().css_class())>
                            {move || assurance().label()}
                        </span>
                    </div>
                </div>
            </div>

            // ── Stats Grid ──
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-value">{cred_count}</span>
                    <span class="stat-label">"Credentials"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{factor_count}</span>
                    <span class="stat-label">"MFA Factors"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{move || format!("{:.0}%", composite_score() * 100.0)}</span>
                    <span class="stat-label">"Reputation"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{move || trust_tier().label()}</span>
                    <span class="stat-label">"Trust Tier"</span>
                </div>
            </div>

            // ── Sovereign Profile ──
            <section class="sovereign-section">
                <h2>"Sovereign Profile"</h2>
                <p class="section-desc">"Your 8-dimensional civic identity across the Mycelix ecosystem"</p>
                <SovereignRadar size=SovereignRadarSize::Large />
            </section>

            // ── Domain Reputation ──
            <section class="reputation-section">
                <h2>"Domain Reputation"</h2>
                <div class="domain-scores">
                    {move || {
                        ctx.reputation.get().map(|r| {
                            r.domain_scores.iter().map(|d| {
                                let pct = (d.score * 100.0) as u32;
                                let domain = d.domain.clone();
                                view! {
                                    <div class="domain-score-row">
                                        <span class="domain-name">{domain}</span>
                                        <div class="domain-bar">
                                            <div class="domain-fill" style=format!("width: {pct}%") />
                                        </div>
                                        <span class="domain-pct">{format!("{pct}%")}</span>
                                    </div>
                                }
                            }).collect::<Vec<_>>()
                        })
                    }}
                </div>
            </section>

            // ── Quick Actions ──
            <section class="actions-section">
                <h2>"Quick Actions"</h2>
                <div class="action-grid">
                    <a href="/did" class="action-card">
                        <span class="action-icon">"\u{1F511}"</span>
                        <span class="action-label">"Manage DID"</span>
                    </a>
                    <a href="/mfa" class="action-card">
                        <span class="action-icon">"\u{1F6E1}"</span>
                        <span class="action-label">"MFA Factors"</span>
                    </a>
                    <a href="/credentials" class="action-card">
                        <span class="action-icon">"\u{1F4DC}"</span>
                        <span class="action-label">"Credentials"</span>
                    </a>
                    <a href="/recovery" class="action-card">
                        <span class="action-icon">"\u{1F91D}"</span>
                        <span class="action-label">"Recovery"</span>
                    </a>
                </div>
            </section>
        </div>
    }
}
