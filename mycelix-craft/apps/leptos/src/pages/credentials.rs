// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Credentials page — living credentials with Ebbinghaus vitality visualization.
//!
//! Shows published credentials from Praxis with decay curves, retention check
//! history, guild endorsements, and epistemic classification.
//!
//! When connected to the Holochain conductor the page merges real credential
//! data from `CraftCtx::credentials`.  Mock data is used as a fallback when
//! the conductor signal is empty.

use leptos::prelude::*;
use mycelix_leptos_core::{
    holochain_provider::use_holochain,
    toasts::{use_toasts, ToastKind},
};

use crate::context::{use_craft, PublishedCredentialView};

/// Mock credential for demonstration.
#[derive(Clone, Debug, PartialEq)]
struct CredentialView {
    title: String,
    issuer: String,
    issued_on: String,
    vitality_permille: u16,
    mastery_permille: u16,
    retention_checks: u32,
    guild_name: Option<String>,
    epistemic_code: Option<String>,
    source_dna: String,
    needs_review: bool,
}

impl From<PublishedCredentialView> for CredentialView {
    fn from(c: PublishedCredentialView) -> Self {
        Self {
            title: c.title,
            issuer: c.issuer,
            issued_on: String::new(), // conductor data omits this for now
            vitality_permille: c.vitality_permille,
            mastery_permille: c.mastery_permille,
            retention_checks: 0,
            guild_name: c.guild_name,
            epistemic_code: c.epistemic_code,
            source_dna: "praxis".into(),
            needs_review: c.needs_review,
        }
    }
}

fn mock_credentials() -> Vec<CredentialView> {
    vec![
        CredentialView {
            title: "Rust Fundamentals".into(),
            issuer: "Praxis Academy".into(),
            issued_on: "2026-03-15".into(),
            vitality_permille: 920,
            mastery_permille: 850,
            retention_checks: 3,
            guild_name: Some("Rust Developers Guild".into()),
            epistemic_code: Some("E3-N1-M2".into()),
            source_dna: "praxis".into(),
            needs_review: false,
        },
        CredentialView {
            title: "Holochain Development".into(),
            issuer: "Praxis Academy".into(),
            issued_on: "2026-02-20".into(),
            vitality_permille: 640,
            mastery_permille: 720,
            retention_checks: 1,
            guild_name: Some("Rust Developers Guild".into()),
            epistemic_code: Some("E3-N1-M2".into()),
            source_dna: "praxis".into(),
            needs_review: true,
        },
        CredentialView {
            title: "Data Science Foundations".into(),
            issuer: "Praxis Academy".into(),
            issued_on: "2026-01-10".into(),
            vitality_permille: 340,
            mastery_permille: 600,
            retention_checks: 0,
            guild_name: None,
            epistemic_code: Some("E2-N1-M1".into()),
            source_dna: "praxis".into(),
            needs_review: true,
        },
        CredentialView {
            title: "Cybersecurity Essentials".into(),
            issuer: "External Issuer".into(),
            issued_on: "2025-11-05".into(),
            vitality_permille: 180,
            mastery_permille: 500,
            retention_checks: 0,
            guild_name: None,
            epistemic_code: None,
            source_dna: "external".into(),
            needs_review: true,
        },
    ]
}

fn vitality_color(permille: u16) -> &'static str {
    match permille {
        800.. => "var(--color-success, #22c55e)",
        500..=799 => "var(--color-warning, #f59e0b)",
        200..=499 => "var(--color-error, #ef4444)",
        _ => "var(--text-secondary, #666)",
    }
}

fn vitality_label(permille: u16) -> &'static str {
    match permille {
        800.. => "Strong",
        500..=799 => "Fading",
        200..=499 => "Weak",
        _ => "Critical",
    }
}

/// Generate SVG path data for an Ebbinghaus decay curve.
///
/// Shows projected vitality from current level over `days` timespan.
/// The curve follows R(t) = current × e^(-t/S) where S depends on
/// mastery and retention check count.
fn decay_curve_svg(
    current_vitality: f32,
    mastery_permille: u16,
    retention_checks: u32,
    days: u32,
    width: u32,
    height: u32,
) -> String {
    let mastery = mastery_permille as f64 / 1000.0;
    let base = 1440.0; // 1 day in minutes
    let mastery_factor = 0.5 + mastery * 2.0;
    let review_factor = 1.5_f64.powi(retention_checks.min(10) as i32);
    let stability_minutes = base * mastery_factor * review_factor;
    let stability_days = stability_minutes / 1440.0;

    let initial = current_vitality as f64 / 1000.0;
    let steps = 60;
    let mut path = String::with_capacity(steps * 12);

    for i in 0..=steps {
        let t = (i as f64 / steps as f64) * days as f64;
        let retention = initial * (-t / stability_days).exp();
        let x = (i as f64 / steps as f64) * width as f64;
        let y = (1.0 - retention) * height as f64;

        if i == 0 {
            path.push_str(&format!("M{x:.1},{y:.1}"));
        } else {
            path.push_str(&format!(" L{x:.1},{y:.1}"));
        }
    }
    path
}

/// Days until vitality drops to the 80% review threshold.
fn days_until_review(vitality_permille: u16, mastery_permille: u16, retention_checks: u32) -> Option<f64> {
    if vitality_permille < 800 {
        return Some(0.0); // Already needs review
    }
    let mastery = mastery_permille as f64 / 1000.0;
    let stability_minutes = 1440.0 * (0.5 + mastery * 2.0) * 1.5_f64.powi(retention_checks.min(10) as i32);
    let stability_days = stability_minutes / 1440.0;
    let current = vitality_permille as f64 / 1000.0;
    let target = 0.8;
    if current <= target { return Some(0.0); }
    Some(-(target / current).ln() * stability_days)
}

#[component]
pub fn CredentialsPage() -> impl IntoView {
    let craft = use_craft();
    let hc = use_holochain();
    let toasts = use_toasts();

    // Use conductor credentials when available, else fall back to mock data.
    let credentials = Memo::new(move |_| {
        let conductor_creds = craft.credentials.get();
        if conductor_creds.is_empty() {
            mock_credentials()
        } else {
            conductor_creds.into_iter().map(CredentialView::from).collect()
        }
    });

    let needs_review_count = Memo::new(move |_| {
        credentials.get().iter().filter(|c| c.needs_review).count()
    });
    let avg_vitality = Memo::new(move |_| {
        let creds = credentials.get();
        if creds.is_empty() {
            0u32
        } else {
            creds.iter().map(|c| c.vitality_permille as u32).sum::<u32>() / creds.len() as u32
        }
    });

    // Clone hc/toasts for each closure that captures them.
    let hc_for_refresh = hc.clone();
    let toasts_for_refresh = toasts.clone();
    let hc_for_disabled = hc.clone();
    let hc_for_title = hc.clone();
    let hc_for_mock_indicator = hc.clone();
    let hc_for_connected_indicator = hc.clone();

    // Refresh handler — calls conductor for updated vitality data
    let on_refresh = move |_| {
        let hc = hc_for_refresh.clone();
        let toasts = toasts_for_refresh.clone();
        let craft = craft.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if hc.is_mock() {
                toasts.push(
                    "Conductor not connected. Showing local demo data.",
                    ToastKind::Info,
                );
                return;
            }
            match hc
                .call_zome_default::<(), Vec<PublishedCredentialView>>(
                    "craft_graph",
                    "get_credential_vitality",
                    &(),
                )
                .await
            {
                Ok(creds) => {
                    craft.credentials.set(creds);
                    toasts.success("Vitality data refreshed from conductor.");
                }
                Err(e) => {
                    toasts.error(format!("Failed to refresh vitality: {e}"));
                }
            }
        });
    };

    view! {
        <div class="page credentials-page">
            <h1>"Living Credentials"</h1>
            <p class="text-secondary">
                "Credentials decay over time via the Ebbinghaus forgetting curve. "
                "Pass retention checks to refresh vitality and prove ongoing mastery."
            </p>

            // Summary stats
            <div class="credentials-stats">
                <div class="stat-pill">
                    <span class="stat-value">{move || credentials.get().len()}</span>
                    <span class="stat-label">"Published"</span>
                </div>
                <div class="stat-pill">
                    <span class="stat-value" style=move || format!("color: {}", vitality_color(avg_vitality.get() as u16))>
                        {move || format!("{}%", avg_vitality.get() / 10)}
                    </span>
                    <span class="stat-label">"Avg Vitality"</span>
                </div>
                <div class="stat-pill">
                    <span class="stat-value" style="color: var(--color-warning, #f59e0b)">
                        {move || needs_review_count.get()}
                    </span>
                    <span class="stat-label">"Needs Review"</span>
                </div>
            </div>

            // Vitality heatmap — shows all credentials as color-coded cells
            <div class="vitality-heatmap">
                <h3>"Vitality Heatmap"</h3>
                <div class="heatmap-grid">
                    {move || credentials.get().into_iter().map(|cred| {
                        let color = vitality_color(cred.vitality_permille);
                        let pct = cred.vitality_permille as f32 / 10.0;
                        let opacity = (cred.vitality_permille as f32 / 1000.0).max(0.15);
                        view! {
                            <div class="heatmap-cell"
                                 title=format!("{}: {:.0}% vitality", cred.title, pct)
                                 style=format!(
                                     "background: {}; opacity: {:.2}",
                                     color, opacity
                                 )>
                                <span class="heatmap-label">{cred.title[..cred.title.len().min(12)].to_string()}</span>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </div>

            // Refresh vitality button
            <div class="credentials-actions">
                <button
                    class="btn-secondary"
                    on:click=on_refresh
                    disabled=move || hc_for_disabled.is_mock()
                    title=move || if hc_for_title.is_mock() { "Connect to conductor to refresh" } else { "Fetch latest vitality from conductor" }
                >
                    "Refresh Vitality"
                </button>
                {move || hc_for_mock_indicator.is_mock().then(|| view! {
                    <span class="status-indicator mock">"Local demo"</span>
                })}
                {move || (!hc_for_connected_indicator.is_mock()).then(|| view! {
                    <span class="status-indicator connected">"Connected to conductor"</span>
                })}
            </div>

            // Credential cards
            <div class="credentials-grid">
                {move || credentials.get().into_iter().map(|cred| {
                    let vitality_pct = cred.vitality_permille as f32 / 10.0;
                    let mastery_pct = cred.mastery_permille as f32 / 10.0;
                    let color = vitality_color(cred.vitality_permille);
                    let label = vitality_label(cred.vitality_permille);

                    view! {
                        <div class="credential-card">
                            <div class="credential-header">
                                <h3>{cred.title.clone()}</h3>
                                {cred.needs_review.then(|| view! {
                                    <span class="review-badge">"Needs Review"</span>
                                })}
                            </div>

                            <div class="credential-meta">
                                <span class="issuer">"Issued by "{cred.issuer.clone()}</span>
                                {(!cred.issued_on.is_empty()).then(|| view! {
                                    <span class="date">{cred.issued_on.clone()}</span>
                                })}
                            </div>

                            // Vitality gauge with decay curve
                            <div class="vitality-section">
                                <div class="vitality-header">
                                    <span>"Vitality: "
                                        <strong style=format!("color: {}", color)>
                                            {format!("{:.0}%", vitality_pct)}
                                        </strong>
                                    </span>
                                    <span class="vitality-label" style=format!("color: {}", color)>
                                        {label}
                                    </span>
                                </div>
                                <div class="vitality-bar-bg">
                                    <div class="vitality-bar"
                                        style=format!("width: {}%; background: {}", vitality_pct, color)>
                                    </div>
                                </div>

                                // SVG decay curve — projected vitality over 90 days
                                <div class="decay-curve-container">
                                    <svg viewBox="0 0 200 50" class="decay-curve-svg"
                                         preserveAspectRatio="none">
                                        // Review threshold line at 80%
                                        <line x1="0" y1="10" x2="200" y2="10"
                                              stroke="var(--color-warning, #f59e0b)"
                                              stroke-width="0.5" stroke-dasharray="4,2"
                                              opacity="0.6" />
                                        // Decay curve
                                        <path d={decay_curve_svg(
                                                cred.vitality_permille as f32,
                                                cred.mastery_permille,
                                                cred.retention_checks,
                                                90, 200, 50)}
                                              fill="none"
                                              stroke={color}
                                              stroke-width="1.5" />
                                    </svg>
                                    <div class="decay-curve-labels">
                                        <span class="decay-label-now">"Now"</span>
                                        <span class="decay-label-end">"90 days"</span>
                                    </div>
                                </div>

                                <div class="vitality-details">
                                    <span>"Mastery: "{format!("{:.0}%", mastery_pct)}</span>
                                    <span>"Checks: "{cred.retention_checks}</span>
                                    {days_until_review(
                                        cred.vitality_permille,
                                        cred.mastery_permille,
                                        cred.retention_checks,
                                    ).map(|days| {
                                        let label = if days <= 0.0 {
                                            "Review now".to_string()
                                        } else if days < 1.0 {
                                            format!("{:.0}h until review", days * 24.0)
                                        } else {
                                            format!("{:.0}d until review", days)
                                        };
                                        view! {
                                            <span class="review-countdown" title="Time until vitality drops below 80%">
                                                {label}
                                            </span>
                                        }
                                    })}
                                </div>
                            </div>

                            // Guild & epistemic info
                            <div class="credential-badges">
                                {cred.guild_name.as_ref().map(|g| view! {
                                    <span class="guild-badge">{g.clone()}</span>
                                })}
                                {cred.epistemic_code.as_ref().map(|code| view! {
                                    <span class="epistemic-badge" title="Proof of Learning classification">
                                        {code.clone()}
                                    </span>
                                })}
                                <span class="source-badge">
                                    {if cred.source_dna == "praxis" { "Praxis Verified" } else { "External" }}
                                </span>
                            </div>

                            // Actions
                            <div class="credential-actions">
                                <button class="btn-secondary btn-sm" disabled>
                                    {if cred.needs_review { "Take Retention Check" } else { "View Details" }}
                                </button>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>

            // Import section
            <div class="import-section">
                <h3>"Import Credentials"</h3>
                <p class="text-secondary">
                    "Publish credentials from Praxis or other sources to your Craft profile. "
                    "Each credential starts at 100% vitality and decays based on your mastery level."
                </p>
                <button class="btn-primary" disabled>
                    "Import from Praxis"
                </button>
            </div>
        </div>
    }
}
