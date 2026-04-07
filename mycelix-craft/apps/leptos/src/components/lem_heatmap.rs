// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LEM Cube Heatmap — 3-panel epistemic coordinate visualization.
//!
//! Each credential's "E3-N1-M2" code is parsed into (Empirical, Normative,
//! Materiality) coordinates and plotted as colored dots across three projections:
//! E×N, E×M, N×M. Dot color = vitality, dot size = mastery.
//!
//! This tells a recruiter at a glance whether you're a "doer" (E-heavy),
//! a "governance thinker" (N-heavy), or a "systems builder" (M-heavy).

use leptos::prelude::*;
use crate::context::{use_craft, PublishedCredentialView};

/// Parse "E3-N1-M2" into (empirical, normative, materiality) as u8 values.
fn parse_epistemic_code(code: &str) -> Option<(u8, u8, u8)> {
    let parts: Vec<&str> = code.split('-').collect();
    if parts.len() != 3 { return None; }

    let e = parts[0].strip_prefix('E').and_then(|s| s.parse::<u8>().ok())?;
    let n = parts[1].strip_prefix('N').and_then(|s| s.parse::<u8>().ok())?;
    let m = parts[2].strip_prefix('M').and_then(|s| s.parse::<u8>().ok())?;
    Some((e, n, m))
}

fn vitality_color(permille: u16) -> &'static str {
    match permille {
        800.. => "#22c55e",  // green
        500..=799 => "#f59e0b",  // amber
        200..=499 => "#ef4444",  // red
        _ => "#6b7280",  // gray
    }
}

/// A single 2D projection panel (e.g., E×N).
fn render_panel(
    title: &str,
    x_label: &str,
    y_label: &str,
    points: &[(f64, f64, &str, f64)],  // (x, y, color, radius)
    x_max: f64,
    y_max: f64,
) -> impl IntoView {
    let grid_w = 100.0;
    let grid_h = 100.0;
    let margin = 20.0;
    let plot_w = grid_w - margin;
    let plot_h = grid_h - margin;

    let title = title.to_string();
    let x_label = x_label.to_string();
    let y_label = y_label.to_string();
    let points: Vec<(f64, f64, String, f64)> = points
        .iter()
        .map(|(x, y, c, r)| (*x, *y, c.to_string(), *r))
        .collect();

    view! {
        <div class="lem-panel">
            <svg viewBox=format!("0 0 {} {}", grid_w + 5.0, grid_h + 5.0) class="lem-svg">
                // Grid lines
                {(0..=4).map(|i| {
                    let x = margin + (i as f64 / 4.0) * plot_w;
                    let y = margin + (i as f64 / 4.0) * plot_h;
                    view! {
                        <line x1=x y1=margin x2=x y2=grid_h stroke="var(--border)" stroke-width="0.3" opacity="0.4" />
                        <line x1=margin y1=y x2=grid_w y2=y stroke="var(--border)" stroke-width="0.3" opacity="0.4" />
                    }
                }).collect_view()}

                // Points
                {points.iter().map(|(x, y, color, radius)| {
                    let cx = margin + (x / x_max) * plot_w;
                    let cy = grid_h - (y / y_max) * plot_h;
                    let color_fill = color.clone();
                    let color_stroke = color.clone();
                    let r = *radius;
                    view! {
                        <circle cx=cx cy=cy r=r fill=color_fill opacity="0.8" />
                        <circle cx=cx cy=cy r=r fill="none" stroke=color_stroke stroke-width="0.5" opacity="0.4" />
                    }
                }).collect_view()}

                // Axis labels
                <text x=margin + plot_w / 2.0 y=grid_h + 4.0
                    text-anchor="middle" fill="var(--text-muted)" font-size="6" font-weight="500">
                    {x_label}
                </text>
                <text x=3.0 y=margin + plot_h / 2.0
                    text-anchor="middle" fill="var(--text-muted)" font-size="6" font-weight="500"
                    transform=format!("rotate(-90, 3, {})", margin + plot_h / 2.0)>
                    {y_label}
                </text>

                // Title
                <text x=margin + plot_w / 2.0 y=10.0
                    text-anchor="middle" fill="var(--primary-light)" font-size="7" font-weight="600">
                    {title}
                </text>
            </svg>
        </div>
    }
}

/// LEM Cube Heatmap — renders 3 panels from credential epistemic codes.
#[component]
pub fn LemHeatmap() -> impl IntoView {
    let craft = use_craft();

    let panels_data = Memo::new(move |_| {
        let creds = craft.credentials.get();

        // Use mock data if empty
        let source: Vec<PublishedCredentialView> = if creds.is_empty() {
            vec![
                PublishedCredentialView {
                    credential_id: String::new(), title: "Rust".into(), issuer: String::new(),
                    vitality_permille: 920, mastery_permille: 850,
                    guild_name: None, epistemic_code: Some("E3-N1-M2".into()), needs_review: false,
                },
                PublishedCredentialView {
                    credential_id: String::new(), title: "Holochain".into(), issuer: String::new(),
                    vitality_permille: 640, mastery_permille: 720,
                    guild_name: None, epistemic_code: Some("E3-N1-M2".into()), needs_review: true,
                },
                PublishedCredentialView {
                    credential_id: String::new(), title: "Data Science".into(), issuer: String::new(),
                    vitality_permille: 340, mastery_permille: 600,
                    guild_name: None, epistemic_code: Some("E2-N1-M1".into()), needs_review: true,
                },
                PublishedCredentialView {
                    credential_id: String::new(), title: "Philosophy".into(), issuer: String::new(),
                    vitality_permille: 500, mastery_permille: 700,
                    guild_name: None, epistemic_code: Some("E1-N3-M2".into()), needs_review: false,
                },
            ]
        } else {
            creds
        };

        // Parse epistemic coordinates
        let mut en_points: Vec<(f64, f64, String, f64)> = vec![];
        let mut em_points: Vec<(f64, f64, String, f64)> = vec![];
        let mut nm_points: Vec<(f64, f64, String, f64)> = vec![];

        for cred in &source {
            if let Some(ref code) = cred.epistemic_code {
                if let Some((e, n, m)) = parse_epistemic_code(code) {
                    let color = vitality_color(cred.vitality_permille).to_string();
                    let radius = 2.0 + (cred.mastery_permille as f64 / 1000.0) * 4.0; // 2-6px
                    en_points.push((e as f64, n as f64, color.clone(), radius));
                    em_points.push((e as f64, m as f64, color.clone(), radius));
                    nm_points.push((n as f64, m as f64, color.clone(), radius));
                }
            }
        }

        (en_points, em_points, nm_points)
    });

    view! {
        <div class="lem-heatmap">
            <h3>"Epistemic Profile"</h3>
            <p class="text-secondary" style="font-size: 0.8rem; margin-bottom: 0.75rem">
                "Where your knowledge lives in the LEM space"
            </p>
            <div class="lem-panels">
                {move || {
                    let (en, em, nm) = panels_data.get();
                    let en_refs: Vec<(f64, f64, &str, f64)> = en.iter().map(|(x,y,c,r)| (*x,*y,c.as_str(),*r)).collect();
                    let em_refs: Vec<(f64, f64, &str, f64)> = em.iter().map(|(x,y,c,r)| (*x,*y,c.as_str(),*r)).collect();
                    let nm_refs: Vec<(f64, f64, &str, f64)> = nm.iter().map(|(x,y,c,r)| (*x,*y,c.as_str(),*r)).collect();
                    view! {
                        {render_panel("E \u{00d7} N", "Empirical", "Normative", &en_refs, 4.0, 3.0)}
                        {render_panel("E \u{00d7} M", "Empirical", "Material", &em_refs, 4.0, 3.0)}
                        {render_panel("N \u{00d7} M", "Normative", "Material", &nm_refs, 3.0, 3.0)}
                    }
                }}
            </div>
            <div class="lem-legend">
                <span style="color: #22c55e">"● Strong"</span>
                <span style="color: #f59e0b">"● Fading"</span>
                <span style="color: #ef4444">"● Weak"</span>
                <span style="font-size: 0.7rem; color: var(--text-muted)">"Size = mastery"</span>
            </div>
        </div>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_code() {
        assert_eq!(parse_epistemic_code("E3-N1-M2"), Some((3, 1, 2)));
        assert_eq!(parse_epistemic_code("E0-N0-M0"), Some((0, 0, 0)));
        assert_eq!(parse_epistemic_code("E4-N3-M3"), Some((4, 3, 3)));
    }

    #[test]
    fn parse_invalid_codes() {
        assert_eq!(parse_epistemic_code(""), None);
        assert_eq!(parse_epistemic_code("E3-N1"), None);
        assert_eq!(parse_epistemic_code("X3-N1-M2"), None);
    }
}
