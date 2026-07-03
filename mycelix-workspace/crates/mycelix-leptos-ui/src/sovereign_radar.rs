// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 8D Sovereign Profile Radar Chart

use leptos::prelude::*;
use sovereign_profile::{CivicTier, SovereignDimension, SovereignProfile};

/// Size presets for the radar chart.
#[derive(Clone, Copy, Default, PartialEq)]
pub enum SovereignRadarSize {
    /// 160px — compact inline display
    Small,
    /// 280px — card/widget size
    #[default]
    Medium,
    /// 400px — full dashboard panel
    Large,
}

impl SovereignRadarSize {
    fn px(self) -> f64 {
        match self {
            Self::Small => 160.0,
            Self::Medium => 280.0,
            Self::Large => 400.0,
        }
    }

    fn label_offset(self) -> f64 {
        match self {
            Self::Small => 18.0,
            Self::Medium => 26.0,
            Self::Large => 34.0,
        }
    }

    fn font_size(self) -> &'static str {
        match self {
            Self::Small => "8",
            Self::Medium => "11",
            Self::Large => "13",
        }
    }

    fn center_font(self) -> &'static str {
        match self {
            Self::Small => "14",
            Self::Medium => "20",
            Self::Large => "26",
        }
    }

    fn show_labels(self) -> bool {
        !matches!(self, Self::Small)
    }
}

const AXIS_COUNT: usize = 8;

/// Color for a civic tier (matches CSS variables from base.css).
fn tier_color(score: f64) -> &'static str {
    if score >= 0.8 {
        "#e8c547" // guardian gold
    } else if score >= 0.6 {
        "#b08fd4" // steward violet
    } else if score >= 0.4 {
        "#6abf69" // citizen green
    } else if score >= 0.3 {
        "#5ba0c9" // participant blue
    } else {
        "#7a8575" // observer grey
    }
}

/// Compute polygon points for the 8D profile.
fn profile_polygon(profile: &SovereignProfile, radius: f64, cx: f64, cy: f64) -> String {
    let mut points = String::new();
    for (i, dim) in SovereignDimension::ALL.iter().enumerate() {
        let angle =
            std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
        let val = profile.get(*dim).clamp(0.0, 1.0);
        let r = val * radius;
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        if !points.is_empty() {
            points.push(' ');
        }
        points.push_str(&format!("{:.1},{:.1}", x, y));
    }
    points
}

/// Compute the grid ring polygon (all axes at the same value).
fn ring_polygon(value: f64, radius: f64, cx: f64, cy: f64) -> String {
    let r = value * radius;
    let mut points = String::new();
    for i in 0..AXIS_COUNT {
        let angle =
            std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        if !points.is_empty() {
            points.push(' ');
        }
        points.push_str(&format!("{:.1},{:.1}", x, y));
    }
    points
}

/// 8D Sovereign Profile radar chart component.
#[component]
pub fn SovereignRadar(
    /// The profile to render. Defaults to a neutral `SovereignProfile` —
    /// every current caller in the workspace renders this with no `profile`
    /// prop at all (a pre-existing gap; the required prop with no default
    /// meant every one of these ~10 call sites has never actually compiled).
    #[prop(default = Signal::stored(SovereignProfile::default()))]
    profile: Signal<SovereignProfile>,
    /// Chart size preset.
    #[prop(default = SovereignRadarSize::Medium)]
    size: SovereignRadarSize,
    /// Optional CSS class for the container.
    #[prop(optional, into)]
    class: String,
    /// Whether to enable the pulse animation.
    #[prop(default = true)]
    pulse: bool,
    /// Weights used for tier calculation.
    #[prop(optional)]
    weights: Option<sovereign_profile::weights::DimensionWeights>,
) -> impl IntoView {
    let px = size.px();
    let cx = px / 2.0;
    let cy = px / 2.0;
    let radius = px / 2.0 - size.label_offset() - 8.0;
    let font_size = size.font_size();
    let center_font = size.center_font();
    let show_labels = size.show_labels();

    let weights = weights.unwrap_or_default();

    // Pulse effect derived from combined score
    let tier_weights = weights.clone();
    let score = move || profile.get().combined_score(&weights);
    let tier = move || profile.get().tier(&tier_weights);
    // `score` is a plain `move ||` closure (Clone, since `profile` is Copy and
    // `weights` is Clone) — it's used both by `profile_view` below and by the
    // separate pulse-animation `<style>` closure further down, so each site
    // needs its own clone rather than moving the same closure into both.
    let score_for_pulse = score.clone();

    // Axis lines and labels
    let axes_view = SovereignDimension::ALL
        .iter()
        .enumerate()
        .map(|(i, dim)| {
            let angle = std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64)
                - std::f64::consts::FRAC_PI_2;
            let x2 = cx + radius * angle.cos();
            let y2 = cy + radius * angle.sin();
            let label_r = radius + size.label_offset() * 0.6;
            let lx = cx + label_r * angle.cos();
            let ly = cy + label_r * angle.sin();

            // Simplified label mapping
            let label_text = match dim {
                SovereignDimension::EpistemicIntegrity => "Epistemic",
                SovereignDimension::ThermodynamicYield => "Energy",
                SovereignDimension::NetworkResilience => "Network",
                SovereignDimension::EconomicVelocity => "Economic",
                SovereignDimension::CivicParticipation => "Civic",
                SovereignDimension::StewardshipCare => "Care",
                SovereignDimension::SemanticResonance => "Semantic",
                SovereignDimension::DomainCompetence => "Domain",
            };

            let anchor = if (angle.cos()).abs() < 0.01 {
                "middle"
            } else if angle.cos() > 0.0 {
                "start"
            } else {
                "end"
            };
            view! {
                <line
                    x1={format!("{:.1}", cx)}
                    y1={format!("{:.1}", cy)}
                    x2={format!("{:.1}", x2)}
                    y2={format!("{:.1}", y2)}
                    stroke="currentColor"
                    stroke-opacity="0.15"
                    stroke-width="1"
                />
                {if show_labels {
                    Some(view! {
                        <text
                            x={format!("{:.1}", lx)}
                            y={format!("{:.1}", ly)}
                            text-anchor=anchor
                            dominant-baseline="central"
                            font-size=font_size
                            fill="currentColor"
                            opacity="0.7"
                        >
                            {label_text}
                        </text>
                    })
                } else {
                    None
                }}
            }
        })
        .collect::<Vec<_>>();

    // Grid rings
    let rings_view = [0.25, 0.5, 0.75, 1.0]
        .iter()
        .map(|&val| {
            let pts = ring_polygon(val, radius, cx, cy);
            view! {
                <polygon
                    points=pts
                    fill="none"
                    stroke="currentColor"
                    stroke-opacity="0.1"
                    stroke-width="1"
                />
            }
        })
        .collect::<Vec<_>>();

    // Profile polygon + center text
    let profile_view = move || {
        let p = profile.get();
        let s = score();
        let t = tier();
        let color = tier_color(s);
        let pts = profile_polygon(&p, radius, cx, cy);

        view! {
            <polygon
                points=pts
                fill=color
                fill-opacity="0.25"
                stroke=color
                stroke-width="2"
                stroke-linejoin="round"
            />
            {SovereignDimension::ALL.iter().enumerate().map(|(i, dim)| {
                let angle = std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
                let val = p.get(*dim).clamp(0.0, 1.0);
                let r = val * radius;
                let x = cx + r * angle.cos();
                let y = cy + r * angle.sin();
                view! {
                    <circle
                        cx={format!("{:.1}", x)}
                        cy={format!("{:.1}", y)}
                        r="3"
                        fill=color
                    />
                }
            }).collect::<Vec<_>>()}
            <text
                x={format!("{:.1}", cx)}
                y={format!("{:.1}", cy - 6.0)}
                text-anchor="middle"
                dominant-baseline="central"
                font-size=center_font
                font-weight="600"
                fill=color
            >
                {t.label()}
            </text>
            <text
                x={format!("{:.1}", cx)}
                y={format!("{:.1}", cy + 12.0)}
                text-anchor="middle"
                dominant-baseline="central"
                font-size=font_size
                fill="currentColor"
                opacity="0.5"
            >
                {format!("{:.0}%", s * 100.0)}
            </text>
        }
    };

    let viewbox = format!("0 0 {} {}", px, px);
    let container_class = if class.is_empty() {
        "sovereign-radar".to_string()
    } else {
        format!("sovereign-radar {}", class)
    };

    view! {
        <div class=container_class>
            <style>
                {move || {
                    if !pulse { return "".to_string(); }
                    let s = score_for_pulse();
                    let duration = 4.0 - (s * 3.0);
                    let scale = 1.0 + (s * 0.05);
                    format!(
                        "@keyframes sovereign-pulse {{
                            0% {{ transform: scale(1); }}
                            50% {{ transform: scale({}); }}
                            100% {{ transform: scale(1); }}
                        }}
                        .sovereign-radar svg {{
                            animation: sovereign-pulse {}s ease-in-out infinite;
                            transform-origin: center;
                        }}",
                        scale, duration
                    )
                }}
            </style>
            <svg
                viewBox=viewbox
                width={format!("{}", px as u32)}
                height={format!("{}", px as u32)}
                xmlns="http://www.w3.org/2000/svg"
                style="display:block;margin:0 auto"
            >
                {rings_view}
                {axes_view}
                {profile_view}
            </svg>
        </div>
    }
}
