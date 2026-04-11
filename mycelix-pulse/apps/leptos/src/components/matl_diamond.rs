// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix Hexagon — 6-axis visualization of sovereign identity coherence.
//!
//! The hexagon maps to the 6 pillars of the Mycelix ecosystem:
//! 1. Assurance — MFA escalation (cryptographic anchor, Sybil resistance)
//! 2. Network — peer attestations, Care Circles (web of trust)
//! 3. Epistemics — knowledge claim quality (E0-E4 epistemic levels)
//! 4. Vitality — Ebbinghaus curve maintenance, living credentials
//! 5. Reciprocity — TEND/SAP economic contribution
//! 6. Thermodynamics — ecological footprint and regeneration
//!
//! A complete honeycomb cell = coherent citizen.
//! A broken shard = Sybil attacker or extractor.

use leptos::prelude::*;

/// The 6 Mycelix Hexagon dimensions.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatlProfile {
    /// MFA assurance level — hardware keys, biometrics, social recovery (0.0-1.0)
    pub assurance: f64,
    /// Peer attestations and Care Circle memberships (0.0-1.0)
    pub network: f64,
    /// Epistemic quality — knowledge claims, peer challenges survived (0.0-1.0)
    pub epistemics: f64,
    /// Knowledge retention via spaced repetition, living credential vitality (0.0-1.0)
    pub vitality: f64,
    /// TEND/SAP economic contribution — net-producer vs extractor (0.0-1.0)
    pub reciprocity: f64,
    /// Ecological footprint balanced against regeneration efforts (0.0-1.0)
    pub thermodynamics: f64,
}

impl MatlProfile {
    /// All 6 values as an array (for iteration).
    pub fn values(&self) -> [f64; 6] {
        [self.assurance, self.network, self.epistemics,
         self.vitality, self.reciprocity, self.thermodynamics]
    }

    /// Axis labels.
    pub fn labels() -> [&'static str; 6] {
        ["Assurance", "Network", "Epistemics", "Vitality", "Reciprocity", "Thermo"]
    }

    /// Short labels for compact display.
    pub fn short_labels() -> [&'static str; 6] {
        ["A", "N", "Ep", "V", "R", "T"]
    }

    /// Composite score (weighted: community-heavy)
    pub fn composite(&self) -> f64 {
        let vals = self.values();
        // Equal weighting across all 6 axes
        (vals.iter().sum::<f64>() / 6.0).clamp(0.0, 1.0)
    }

    /// Check if profile looks like a Sybil (strong on few axes, weak on most)
    pub fn is_suspicious(&self) -> bool {
        let vals = self.values();
        let strong = vals.iter().filter(|&&v| v > 0.5).count();
        let weak = vals.iter().filter(|&&v| v < 0.1).count();
        strong >= 1 && weak >= 4 // One strong axis + most axes near zero
    }

    /// Hexagon area (0.0-1.0) — measures overall coherence.
    /// Full hexagon = 1.0, empty = 0.0.
    pub fn coherence(&self) -> f64 {
        let vals = self.values();
        let n = vals.len();
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += vals[i] * vals[j] * (std::f64::consts::PI / 3.0).sin();
        }
        // Normalize against max area (all 1.0)
        let max_area = n as f64 * (std::f64::consts::PI / 3.0).sin();
        (area / max_area).clamp(0.0, 1.0)
    }

    /// Compute hexagon vertex coordinates for a given center and radius.
    fn vertices(center: f64, radius: f64, values: &[f64; 6]) -> Vec<(f64, f64)> {
        values.iter().enumerate().map(|(i, &v)| {
            let angle = std::f64::consts::PI / 2.0 + (i as f64) * std::f64::consts::PI / 3.0;
            let r = v * radius;
            (center + r * angle.cos(), center - r * angle.sin())
        }).collect()
    }
}

/// Full Mycelix Hexagon visualization (for profiles, settings).
#[component]
pub fn MatlDiamond(
    #[prop(into)] profile: MatlProfile,
    #[prop(default = 140)]
    size: u32,
) -> impl IntoView {
    let s = size as f64;
    let center = s / 2.0;
    let radius = center - 20.0;
    let vals = profile.values();
    let labels = MatlProfile::labels();

    // Profile shape vertices
    let verts = MatlProfile::vertices(center, radius, &vals);
    let shape: String = verts.iter()
        .map(|(x, y)| format!("{x:.1},{y:.1}"))
        .collect::<Vec<_>>()
        .join(" ");

    // Grid hexagons at 0.25, 0.5, 0.75, 1.0
    let grids: Vec<String> = [0.25, 0.5, 0.75, 1.0].iter().map(|&level| {
        let full = [level; 6];
        let gv = MatlProfile::vertices(center, radius, &full);
        gv.iter().map(|(x, y)| format!("{x:.1},{y:.1}")).collect::<Vec<_>>().join(" ")
    }).collect();

    // Label positions (slightly outside the outer hexagon)
    let label_verts = MatlProfile::vertices(center, radius + 14.0, &[1.0; 6]);

    let suspicious = profile.is_suspicious();
    let fill = if suspicious { "rgba(239,68,68,0.12)" } else { "rgba(6,214,200,0.12)" };
    let stroke = if suspicious { "#ef4444" } else { "#06D6C8" };

    view! {
        <div class="matl-diamond" style=format!("width:{s}px;height:{s}px")>
            <svg viewBox=format!("0 0 {s} {s}") width=size height=size xmlns="http://www.w3.org/2000/svg">
                // Grid hexagons
                {grids.iter().map(|points| {
                    view! { <polygon points=points.clone() fill="none" stroke="#252838" stroke-width="0.5" /> }
                }).collect::<Vec<_>>()}
                // Axis lines from center
                {(0..6).map(|i| {
                    let angle = std::f64::consts::PI / 2.0 + (i as f64) * std::f64::consts::PI / 3.0;
                    let ex = center + radius * angle.cos();
                    let ey = center - radius * angle.sin();
                    view! { <line x1=center y1=center x2=ex y2=ey stroke="#333" stroke-width="0.5" /> }
                }).collect::<Vec<_>>()}
                // Profile shape
                <polygon points=shape fill=fill stroke=stroke stroke-width="1.5" />
                // Vertex dots
                {verts.iter().map(|(x, y)| {
                    view! { <circle cx=*x cy=*y r="2.5" fill=stroke /> }
                }).collect::<Vec<_>>()}
                // Center dot
                <circle cx=center cy=center r="1.5" fill="#555" />
                // Labels
                {label_verts.iter().enumerate().map(|(i, (x, y))| {
                    let anchor = if *x < center - 5.0 { "end" } else if *x > center + 5.0 { "start" } else { "middle" };
                    view! {
                        <text x=*x y=*y text-anchor=anchor fill="#8890a8" font-size="7" font-family="Inter,sans-serif" dominant-baseline="central">
                            {labels[i]}
                        </text>
                    }
                }).collect::<Vec<_>>()}
            </svg>
            {if suspicious {
                Some(view! { <span class="matl-warning" title="Suspicious profile — possible Sybil">"⚠"</span> })
            } else {
                None
            }}
        </div>
    }
}

/// Compact hexagonal glyph for email cards and contact lists.
#[component]
pub fn MatlGlyph(
    #[prop(into)] profile: MatlProfile,
    #[prop(default = 20)]
    size: u32,
) -> impl IntoView {
    let s = size as f64;
    let c = s / 2.0;
    let r = c - 1.0;
    let vals = profile.values();
    let verts = MatlProfile::vertices(c, r, &vals);
    let shape: String = verts.iter()
        .map(|(x, y)| format!("{x:.1},{y:.1}"))
        .collect::<Vec<_>>()
        .join(" ");

    let suspicious = profile.is_suspicious();
    let fill = if suspicious { "rgba(239,68,68,0.3)" } else { "rgba(6,214,200,0.3)" };
    let stroke = if suspicious { "#ef4444" } else { "#06D6C8" };

    let labels = MatlProfile::short_labels();
    let title = format!(
        "{}:{:.0}% {}:{:.0}% {}:{:.0}% {}:{:.0}% {}:{:.0}% {}:{:.0}%",
        labels[0], vals[0]*100.0, labels[1], vals[1]*100.0,
        labels[2], vals[2]*100.0, labels[3], vals[3]*100.0,
        labels[4], vals[4]*100.0, labels[5], vals[5]*100.0,
    );

    view! {
        <span class="matl-glyph" title=title style=format!("display:inline-block;width:{s}px;height:{s}px;vertical-align:middle")>
            <svg viewBox=format!("0 0 {s} {s}") width=size height=size xmlns="http://www.w3.org/2000/svg">
                <polygon points=shape fill=fill stroke=stroke stroke-width="1" />
            </svg>
        </span>
    }
}
