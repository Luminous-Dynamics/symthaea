// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Progressive Recovery — from self-recovery (Day 0) to social recovery (Hearth).
//!
//! Every user gets a self-recovery config at DID creation. As they verify
//! phone, email, passkey, or device, recovery strengthens. When they join
//! a Hearth with 3+ guardians, social recovery auto-configures and self-recovery
//! becomes a fallback.

use leptos::prelude::*;
use crate::identity_context::use_identity;
use identity_leptos_types::*;

#[component]
pub fn RecoveryPage() -> impl IntoView {
    let ctx = use_identity();

    let has_social = move || ctx.recovery_config.get().is_some();
    let social_trustees = move || {
        ctx.recovery_config.get()
            .map(|c| c.trustees.clone())
            .unwrap_or_default()
    };
    let social_threshold = move || {
        ctx.recovery_config.get()
            .map(|c| c.threshold)
            .unwrap_or(0)
    };

    // Recovery tier derived from both configs
    let recovery_tier = move || {
        if has_social() {
            RecoveryTier::SocialRecovery {
                trustee_count: social_trustees().len() as u32,
                threshold: social_threshold(),
            }
        } else {
            // No self-recovery context signal yet — show pending
            RecoveryTier::Pending
        }
    };

    view! {
        <div class="page page-recovery">
            <h1>"Recovery"</h1>
            <p class="page-subtitle">
                "Progressive protection \u{2014} your identity grows safer over time"
            </p>

            // ── Recovery Strength Indicator ──
            {move || {
                let tier = recovery_tier();
                let strength = tier.strength();
                let pct = (strength * 100.0) as u32;
                let color = tier.css_color();
                view! {
                    <div class="stat-card" style="margin-bottom: var(--space-lg);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-sm);">
                            <span style="font-weight: 600; font-size: var(--text-lg);">"Recovery Strength"</span>
                            <span style=format!("color: {}; font-weight: 700;", color)>{tier.label()}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.1); border-radius: var(--radius-pill); height: 8px; overflow: hidden;">
                            <div style=format!("width: {}%; height: 100%; background: {}; border-radius: var(--radius-pill); transition: width 0.5s;", pct, color) />
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: var(--text-xs); opacity: 0.5;">
                            <span>"Unprotected"</span>
                            <span>"Self-Recovery"</span>
                            <span>"Social Recovery"</span>
                        </div>
                    </div>
                }
            }}

            // ── Self-Recovery Section ──
            <section style="margin-bottom: var(--space-xl);">
                <h2>"Self-Recovery"</h2>
                <p style="opacity: 0.7; margin-bottom: var(--space-md);">
                    "Verify your phone, email, passkey, or device to enable self-recovery. "
                    "More anchors = shorter time lock = faster recovery."
                </p>

                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: var(--space-sm);">
                    {[
                        (VerificationAnchorType::Phone, false),
                        (VerificationAnchorType::Email, false),
                        (VerificationAnchorType::Passkey, false),
                        (VerificationAnchorType::Device, false),
                        (VerificationAnchorType::Biometric, false),
                    ].iter().map(|(anchor_type, enrolled)| {
                        let icon = anchor_type.icon();
                        let label = anchor_type.label();
                        let status = if *enrolled { "\u{2705}" } else { "\u{2795}" };
                        let opacity = if *enrolled { "1.0" } else { "0.6" };
                        view! {
                            <button
                                class="stat-card"
                                style=format!("cursor: pointer; text-align: center; opacity: {}; border: 1px solid rgba(255,255,255,0.1);", opacity)
                                disabled=*enrolled
                            >
                                <span style="font-size: 1.5rem; display: block;">{icon}</span>
                                <span style="font-size: var(--text-sm); display: block; margin-top: 4px;">{label}</span>
                                <span style="font-size: var(--text-xs); display: block; margin-top: 2px;">{status}</span>
                            </button>
                        }
                    }).collect::<Vec<_>>()}
                </div>

                <div class="stat-card" style="margin-top: var(--space-md); display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600;">"Time Lock"</span>
                        <br />
                        <span style="opacity: 0.7; font-size: var(--text-sm);">"Waiting period before self-recovery executes"</span>
                    </div>
                    <span style="font-size: var(--text-lg); font-weight: 700;">"7 days"</span>
                </div>
            </section>

            // ── Social Recovery Section ──
            <section>
                <h2>"Social Recovery"</h2>
                {move || {
                    if has_social() {
                        view! {
                            <div>
                                <div class="stat-card" style="background: rgba(68, 255, 136, 0.1); border: 1px solid rgba(68, 255, 136, 0.3); margin-bottom: var(--space-md); display: flex; align-items: center; gap: var(--space-md);">
                                    <span style="font-size: 1.5rem;">"\u{2705}"</span>
                                    <div>
                                        <strong>"Social Recovery Active"</strong>
                                        <br />
                                        <span style="opacity: 0.8;">{move || format!(
                                            "{} of {} trustees needed",
                                            social_threshold(), social_trustees().len()
                                        )}</span>
                                    </div>
                                </div>
                                <div style="display: grid; gap: var(--space-sm);">
                                    {move || social_trustees().iter().map(|did| {
                                        let short = if did.len() > 30 {
                                            format!("{}...", &did[..30])
                                        } else {
                                            did.clone()
                                        };
                                        view! {
                                            <div class="stat-card" style="display: flex; align-items: center; gap: var(--space-sm);">
                                                <span>"\u{1F6E1}\u{FE0F}"</span>
                                                <span style="font-family: var(--font-mono); font-size: var(--text-sm);">{short}</span>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div class="stat-card" style="background: rgba(255, 170, 68, 0.1); border: 1px solid rgba(255, 170, 68, 0.3);">
                                <div style="display: flex; align-items: center; gap: var(--space-md); margin-bottom: var(--space-md);">
                                    <span style="font-size: 1.5rem;">"\u{1F331}"</span>
                                    <div>
                                        <strong>"Social Recovery Available via Hearth"</strong>
                                        <br />
                                        <span style="opacity: 0.7;">"Join or create a Hearth with 3+ guardians to unlock"</span>
                                    </div>
                                </div>
                                <p style="opacity: 0.6; font-size: var(--text-sm);">
                                    "Social recovery is stronger than self-recovery because trusted humans "
                                    "independently verify your identity. When you join a Hearth, social "
                                    "recovery auto-configures and your self-recovery becomes a fallback."
                                </p>
                            </div>
                        }.into_any()
                    }
                }}
            </section>
        </div>
    }
}
