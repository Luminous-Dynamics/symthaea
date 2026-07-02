// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Skill Coherence Radar — SVG radar chart showing credential vitality
//! across skill domains. Reveals synergies between high-vitality credentials.
//!
//! Uses the same radar_points() pattern as Praxis's spore_sandbox.rs.

use leptos::prelude::*;
use crate::context::{use_craft, PublishedCredentialView};

/// The 8 skill domains for the radar axes.
const DOMAINS: [&str; 8] = [
    "Programming",
    "Data Science",
    "Security",
    "Systems",
    "Teaching",
    "Research",
    "Finance",
    "Design",
];

/// Keywords that map a credential to a domain (case-insensitive).
fn domain_keywords(domain: &str) -> &'static [&'static str] {
    match domain {
        "Programming" => &["rust", "python", "javascript", "holochain", "wasm", "code", "developer", "software", "web"],
        "Data Science" => &["data", "machine learning", "statistics", "analytics", "ml", "ai"],
        "Security" => &["security", "cyber", "cryptography", "penetration", "forensic"],
        "Systems" => &["systems", "distributed", "infrastructure", "nix", "linux", "devops", "network"],
        "Teaching" => &["teaching", "education", "curriculum", "tutor", "mentor", "pedagogy"],
        "Research" => &["research", "academic", "paper", "thesis", "science", "physics"],
        "Finance" => &["finance", "accounting", "economics", "audit", "budget", "investment"],
        "Design" => &["design", "ux", "ui", "creative", "art", "visual", "graphic"],
        _ => &[],
    }
}

/// Map a credential to a domain index (0-7) based on title/guild keywords.
fn classify_credential(cred: &PublishedCredentialView) -> Option<usize> {
    let title_lower = cred.title.to_lowercase();
    let guild_lower = cred.guild_name.as_deref().unwrap_or("").to_lowercase();
    let text = format!("{} {}", title_lower, guild_lower);

    for (i, domain) in DOMAINS.iter().enumerate() {
        for kw in domain_keywords(domain) {
            if text.contains(kw) {
                return Some(i);
            }
        }
    }
    None
}

fn clamp01(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}

fn radar_points(values: &[f64], radius: f64, center: f64) -> String {
    let count = values.len().max(1) as f64;
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let angle = (i as f64 / count) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
            let r = clamp01(value) * radius;
            let x = center + r * angle.cos();
            let y = center + r * angle.sin();
            format!("{:.1},{:.1}", x, y)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Skill Coherence Radar — SVG component.
///
/// Reads credentials from CraftCtx and plots vitality across 8 domains.
/// Uses mock credentials as fallback when conductor data is empty.
#[component]
pub fn SkillRadar() -> impl IntoView {
    let craft = use_craft();

    let domain_values = Memo::new(move |_| {
        let creds = craft.credentials.get();

        // Use mock data if no conductor credentials
        let source: Vec<PublishedCredentialView> = if creds.is_empty() {
            vec![
                PublishedCredentialView {
                    credential_id: String::new(),
                    title: "Rust Fundamentals".into(),
                    issuer: String::new(),
                    vitality_permille: 920,
                    mastery_permille: 850,
                    guild_name: Some("Rust Developers".into()),
                    epistemic_code: Some("E3-N1-M2".into()),
                    needs_review: false,
                },
                PublishedCredentialView {
                    credential_id: String::new(),
                    title: "Holochain Development".into(),
                    issuer: String::new(),
                    vitality_permille: 640,
                    mastery_permille: 720,
                    guild_name: None,
                    epistemic_code: None,
                    needs_review: true,
                },
                PublishedCredentialView {
                    credential_id: String::new(),
                    title: "Data Science Foundations".into(),
                    issuer: String::new(),
                    vitality_permille: 340,
                    mastery_permille: 600,
                    guild_name: None,
                    epistemic_code: None,
                    needs_review: true,
                },
                PublishedCredentialView {
                    credential_id: String::new(),
                    title: "Cybersecurity Essentials".into(),
                    issuer: String::new(),
                    vitality_permille: 180,
                    mastery_permille: 500,
                    guild_name: None,
                    epistemic_code: None,
                    needs_review: true,
                },
            ]
        } else {
            creds
        };

        // Aggregate vitality per domain (average if multiple credentials)
        let mut domain_sums = [0.0f64; 8];
        let mut domain_counts = [0u32; 8];

        for cred in &source {
            if let Some(idx) = classify_credential(cred) {
                domain_sums[idx] += cred.vitality_permille as f64 / 1000.0;
                domain_counts[idx] += 1;
            }
        }

        let mut values = [0.0f64; 8];
        for i in 0..8 {
            if domain_counts[i] > 0 {
                values[i] = domain_sums[i] / domain_counts[i] as f64;
            }
        }
        values
    });

    let size = 200.0;
    let center = size / 2.0;
    let radius = center - 20.0;

    view! {
        <div class="skill-radar">
            <h3>"Skill Coherence"</h3>
            <svg
                viewBox=format!("0 0 {} {}", size, size)
                xmlns="http://www.w3.org/2000/svg"
                class="radar-svg"
            >
                // Background rings (25%, 50%, 75%, 100%)
                {[0.25, 0.5, 0.75, 1.0].iter().map(|&frac| {
                    let r = radius * frac;
                    view! {
                        <circle
                            cx=center cy=center r=r
                            fill="none"
                            stroke="var(--border)"
                            stroke-width="0.5"
                            opacity="0.4"
                        />
                    }
                }).collect_view()}

                // Axis lines
                {(0..8).map(|i| {
                    let angle = (i as f64 / 8.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
                    let x2 = center + radius * angle.cos();
                    let y2 = center + radius * angle.sin();
                    view! {
                        <line
                            x1=center y1=center x2=x2 y2=y2
                            stroke="var(--border)"
                            stroke-width="0.5"
                            opacity="0.3"
                        />
                    }
                }).collect_view()}

                // Data polygon
                {move || {
                    let vals = domain_values.get();
                    let points = radar_points(&vals, radius, center);
                    view! {
                        <polygon
                            points=points
                            fill="var(--primary-light)"
                            fill-opacity="0.25"
                            stroke="var(--primary-light)"
                            stroke-width="1.5"
                        />
                    }
                }}

                // Domain labels
                {DOMAINS.iter().enumerate().map(|(i, &domain)| {
                    let angle = (i as f64 / 8.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
                    let label_r = radius + 14.0;
                    let x = center + label_r * angle.cos();
                    let y = center + label_r * angle.sin();
                    let anchor = if (x - center).abs() < 5.0 { "middle" }
                        else if x > center { "start" }
                        else { "end" };
                    view! {
                        <text
                            x=x y=y
                            text-anchor=anchor
                            dominant-baseline="middle"
                            fill="var(--text-muted)"
                            font-size="7"
                            font-weight="500"
                        >
                            {domain}
                        </text>
                    }
                }).collect_view()}
            </svg>
        </div>
    }
}
