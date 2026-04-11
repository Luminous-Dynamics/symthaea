// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Social Recovery — trusted guardians help restore identity on device loss.

use leptos::prelude::*;
use crate::identity_context::use_identity;

#[component]
pub fn RecoveryPage() -> impl IntoView {
    let ctx = use_identity();

    let has_config = move || ctx.recovery_config.get().is_some();
    let trustees = move || {
        ctx.recovery_config.get()
            .map(|c| c.trustee_dids.clone())
            .unwrap_or_default()
    };
    let threshold = move || {
        ctx.recovery_config.get()
            .map(|c| c.threshold)
            .unwrap_or(0)
    };

    view! {
        <div class="page page-recovery">
            <h1>"Social Recovery"</h1>
            <p class="page-subtitle">
                "Protect your identity with trusted guardians \u{2014} no single point of failure"
            </p>

            {move || {
                if has_config() {
                    view! {
                        <div class="recovery-configured">
                            <div class="recovery-status-banner" style="background: var(--color-success-bg, #1a3a2a); padding: var(--space-md); border-radius: var(--radius-md); margin-bottom: var(--space-lg); display: flex; align-items: center; gap: var(--space-md);">
                                <span style="font-size: 1.5rem">"\u{2705}"</span>
                                <div>
                                    <strong>"Recovery Configured"</strong>
                                    <br />
                                    <span style="opacity: 0.8">{move || format!(
                                        "{} of {} trustees needed to recover",
                                        threshold(), trustees().len()
                                    )}</span>
                                </div>
                            </div>

                            <h3>"Your Trustees"</h3>
                            <div style="display: grid; gap: var(--space-sm);">
                                {move || trustees().iter().map(|did| {
                                    let short = if did.len() > 30 {
                                        format!("{}...", &did[..30])
                                    } else {
                                        did.clone()
                                    };
                                    view! {
                                        <div class="stat-card" style="display: flex; align-items: center; gap: var(--space-sm);">
                                            <span style="font-size: 1.2rem">"\u{1F6E1}\u{FE0F}"</span>
                                            <span style="font-family: var(--font-mono); font-size: var(--text-sm)">{short}</span>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                    }.into_any()
                } else {
                    view! {
                        <div class="recovery-not-configured">
                            <div style="background: var(--color-warning-bg, #3a351a); padding: var(--space-md); border-radius: var(--radius-md); margin-bottom: var(--space-lg); display: flex; align-items: center; gap: var(--space-md);">
                                <span style="font-size: 1.5rem">"\u{26A0}\u{FE0F}"</span>
                                <div>
                                    <strong>"Recovery Not Configured"</strong>
                                    <br />
                                    <span style="opacity: 0.8">"If you lose your device, you'll lose access to your identity"</span>
                                </div>
                            </div>

                            <h3>"How Social Recovery Works"</h3>
                            <div style="display: grid; gap: var(--space-md); margin: var(--space-lg) 0;">
                                <div class="stat-card" style="display: flex; gap: var(--space-md); align-items: flex-start;">
                                    <span style="background: var(--mycelix-cyan, #4ecdc4); color: #000; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">"1"</span>
                                    <div>
                                        <strong>"Choose Trustees"</strong>
                                        <p style="opacity: 0.7; margin-top: 4px;">"Select 3\u{2013}5 people you trust deeply \u{2014} family, close friends, community elders"</p>
                                    </div>
                                </div>
                                <div class="stat-card" style="display: flex; gap: var(--space-md); align-items: flex-start;">
                                    <span style="background: var(--mycelix-cyan, #4ecdc4); color: #000; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">"2"</span>
                                    <div>
                                        <strong>"Set Threshold"</strong>
                                        <p style="opacity: 0.7; margin-top: 4px;">"Decide how many must agree to restore your identity (e.g., 3 of 5)"</p>
                                    </div>
                                </div>
                                <div class="stat-card" style="display: flex; gap: var(--space-md); align-items: flex-start;">
                                    <span style="background: var(--mycelix-cyan, #4ecdc4); color: #000; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">"3"</span>
                                    <div>
                                        <strong>"Rest Easy"</strong>
                                        <p style="opacity: 0.7; margin-top: 4px;">"Your identity is protected. No single person can recover it alone."</p>
                                    </div>
                                </div>
                            </div>

                            <button class="btn btn-primary" disabled=true style="width: 100%; padding: var(--space-md);">
                                "Set Up Recovery"
                            </button>
                            <p style="text-align: center; opacity: 0.5; margin-top: var(--space-sm);">
                                "Requires Participant tier or higher"
                            </p>
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
