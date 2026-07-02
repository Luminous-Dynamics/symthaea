// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::identity_context::use_identity;
use identity_leptos_types::*;
use mycelix_leptos_core::SovereignRadar;

#[component]
pub fn TrustPage() -> impl IntoView {
    let ctx = use_identity();

    let reputation = move || ctx.reputation.get();
    let trust_creds = move || ctx.trust_credentials.get();
    let now_secs = move || (js_sys::Date::now() / 1000.0) as i64;

    view! {
        <div class="page page-trust">
            <h1>"Trust & Reputation"</h1>
            <p class="page-subtitle">"Your position in the Mycelix web of trust \u{2014} from Observer to Guardian"</p>

            // ── Tier Progression ──
            {move || reputation().map(|rep| {
                let current = rep.trust_tier.clone();
                let score = rep.composite_score;
                let tiers = [
                    TrustTier::Observer,
                    TrustTier::Basic,
                    TrustTier::Standard,
                    TrustTier::Elevated,
                    TrustTier::Guardian,
                ];

                view! {
                    <section class="tier-progression">
                        <div class="tier-banner">
                            <div class="tier-current-info">
                                <span class="tier-current-label">"Current Tier"</span>
                                <span class="tier-current-name" style=format!("color: {}", current.css_color())>
                                    {current.label()}
                                </span>
                                <span class="tier-score">{format!("Score: {:.0}%", score * 100.0)}</span>
                            </div>
                            <div class="tier-progress-bar">
                                <div class="tier-progress-fill" style=format!("width: {:.0}%", score * 100.0) />
                                // Tier markers
                                {tiers.iter().map(|t| {
                                    let pos = t.min_score() * 100.0;
                                    let is_current = *t == current;
                                    let color = t.css_color().to_string();
                                    let label = t.label().to_string();
                                    let label_title = label.clone();
                                    view! {
                                        <div class="tier-marker"
                                             style=format!("left: {pos:.0}%")
                                             title=label_title>
                                            <div class=move || if is_current { "tier-dot current" } else { "tier-dot" }
                                                 style=format!("background: {color}") />
                                            <span class="tier-marker-label">{label}</span>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>

                        // ── What each tier unlocks ──
                        <div class="tier-unlock-grid">
                            {tiers.iter().map(|t| {
                                let unlocked = score >= t.min_score();
                                let class = if unlocked { "tier-unlock-card unlocked" } else { "tier-unlock-card locked" };
                                let color = t.css_color().to_string();
                                let label = t.label().to_string();
                                let unlocks = match t {
                                    TrustTier::Observer => "View public data, browse domains",
                                    TrustTier::Basic => "Submit proposals, join discussions",
                                    TrustTier::Standard => "Vote on proposals, delegate voice",
                                    TrustTier::Elevated => "Join councils, constitutional input",
                                    TrustTier::Guardian => "Veto power, constitutional amendments",
                                };
                                view! {
                                    <div class={class}>
                                        <span class="unlock-tier" style=format!("color: {color}")>{label}</span>
                                        <span class="unlock-threshold">{format!("\u{2265} {:.0}%", t.min_score() * 100.0)}</span>
                                        <span class="unlock-desc">{unlocks}</span>
                                        <span class="unlock-status">
                                            {if unlocked { "\u{2714} Unlocked" } else { "\u{1F512} Locked" }}
                                        </span>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </section>

                    // ── Sovereign Profile Radar ──
                    <section class="trust-section">
                        <h2>"Sovereign Profile"</h2>
                        <p class="section-desc">"Your 8-dimensional civic identity"</p>
                        <SovereignRadar />
                    </section>

                    // ── Domain Reputation Breakdown ──
                    <section class="trust-section">
                        <h2>"Reputation by Domain"</h2>
                        <p class="section-desc">"Your trust score across each Mycelix cluster"</p>
                        <div class="domain-breakdown">
                            {rep.domain_scores.iter().map(|d| {
                                let pct = (d.score * 100.0) as u32;
                                let tier = TrustTier::from_score(d.score);
                                let tier_color = tier.css_color().to_string();
                                let domain = d.domain.clone();
                                view! {
                                    <div class="domain-trust-row">
                                        <span class="domain-trust-name">{domain}</span>
                                        <div class="domain-trust-bar">
                                            <div class="domain-trust-fill" style=format!("width: {pct}%; background: {tier_color}") />
                                        </div>
                                        <span class="domain-trust-tier" style=format!("color: {tier_color}")>
                                            {tier.label()}
                                        </span>
                                        <span class="domain-trust-pct">{format!("{pct}%")}</span>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </section>

                    // ── Consciousness Profile ──
                    <section class="trust-section">
                        <h2>"Consciousness Dimensions"</h2>
                        <p class="section-desc">"The four pillars that compose your trust score"</p>
                        <div class="consciousness-pillars">
                            {[
                                ("Identity", rep.consciousness_profile.identity, "Depth of self-verification (MFA, key rotation, DID maturity)"),
                                ("Reputation", rep.consciousness_profile.reputation, "Trust earned through consistent positive interactions"),
                                ("Community", rep.consciousness_profile.community, "Embeddedness in the social graph (bonds, attestations)"),
                                ("Engagement", rep.consciousness_profile.engagement, "Active participation in governance, commons, hearth"),
                            ].iter().map(|(label, value, desc)| {
                                let pct = (*value * 100.0) as u32;
                                let tier = TrustTier::from_score(*value);
                                view! {
                                    <div class="pillar-card">
                                        <div class="pillar-header">
                                            <span class="pillar-label">{*label}</span>
                                            <span class="pillar-value" style=format!("color: {}", tier.css_color())>
                                                {format!("{pct}%")}
                                            </span>
                                        </div>
                                        <div class="pillar-bar">
                                            <div class="pillar-fill" style=format!("width: {pct}%; background: {}", tier.css_color()) />
                                        </div>
                                        <p class="pillar-desc">{*desc}</p>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </section>
                }
            })}

            // ── Trust Credentials ──
            <section class="trust-section">
                <h2>"Trust Credentials"</h2>
                <p class="section-desc">"K-Vector commitment proofs issued by the network"</p>
                {move || {
                    let creds = trust_creds();
                    let now = now_secs();
                    if creds.is_empty() {
                        return view! {
                            <div class="empty-state">
                                <p>"\u{1F510} No trust credentials yet"</p>
                                <p class="empty-hint">"Trust credentials are issued as your reputation grows"</p>
                            </div>
                        }.into_any();
                    }
                    view! {
                        <div class="trust-cred-list">
                            {creds.iter().map(|tc| {
                                let tier_color = tc.trust_tier.css_color().to_string();
                                let tier_label = tc.trust_tier.label().to_string();
                                let issuer_short = if tc.issuer_did.len() > 30 {
                                    format!("{}...", &tc.issuer_did[..30])
                                } else {
                                    tc.issuer_did.clone()
                                };
                                let status = if tc.revoked { "Revoked" }
                                    else if tc.expires_at.map(|e| now > e).unwrap_or(false) { "Expired" }
                                    else { "Active" };
                                let status_class = match status {
                                    "Active" => "tc-active",
                                    "Expired" => "tc-expired",
                                    _ => "tc-revoked",
                                };

                                view! {
                                    <div class="trust-cred-card">
                                        <div class="tc-header">
                                            <span class="tc-tier" style=format!("color: {tier_color}; border-color: {tier_color}")>
                                                {tier_label}
                                            </span>
                                            <span class={format!("tc-status {status_class}")}>{status}</span>
                                        </div>
                                        <div class="tc-meta">
                                            <span class="tc-label">"Issuer"</span>
                                            <span class="tc-value">{issuer_short}</span>
                                        </div>
                                        <div class="tc-meta">
                                            <span class="tc-label">"K-Vector Commitment"</span>
                                            <span class="tc-value tc-crypto">"\u{1F510} ZKP-verified range proof"</span>
                                        </div>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }}
            </section>
        </div>
    }
}
