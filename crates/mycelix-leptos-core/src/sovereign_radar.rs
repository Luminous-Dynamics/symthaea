// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 8D Sovereign Profile Radar Chart
//!
//! SVG radar/spider chart that renders the agent's 8 sovereign dimensions
//! as a filled polygon on radial axes. Reactive — automatically updates
//! when the profile signal changes.
//!
//! # Usage
//!
//! ```rust,ignore
//! use mycelix_leptos_core::{SovereignRadar, SovereignRadarSize};
//!
//! #[component]
//! fn Dashboard() -> impl IntoView {
//!     view! { <SovereignRadar size=SovereignRadarSize::Large /> }
//! }
//! ```

use leptos::prelude::*;
use crate::consciousness::{
    use_consciousness, combined_score, civic_tier,
    SovereignDimension, SovereignProfile, DIMENSION_LABELS,
};

/// Size presets for the radar chart.
#[derive(Clone, Copy, Default)]
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
        let angle = std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
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
        let angle = std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
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
///
/// Renders an SVG spider/radar chart showing all 8 sovereign dimensions.
/// Reads from the consciousness context automatically.
#[component]
pub fn SovereignRadar(
    /// Chart size preset.
    #[prop(default = SovereignRadarSize::Medium)]
    size: SovereignRadarSize,
    /// Optional CSS class for the container.
    #[prop(optional)]
    class: &'static str,
) -> impl IntoView {
    let consciousness = use_consciousness();

    let px = size.px();
    let cx = px / 2.0;
    let cy = px / 2.0;
    let radius = px / 2.0 - size.label_offset() - 8.0;
    let font_size = size.font_size();
    let center_font = size.center_font();
    let show_labels = size.show_labels();

    // Grid ring values (25%, 50%, 75%, 100%)
    let rings = [0.25, 0.5, 0.75, 1.0];

    // Axis lines and labels (static — don't depend on profile)
    let axes_view = SovereignDimension::ALL.iter().enumerate().map(|(i, dim)| {
        let angle = std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
        let x2 = cx + radius * angle.cos();
        let y2 = cy + radius * angle.sin();
        let label_r = radius + size.label_offset() * 0.6;
        let lx = cx + label_r * angle.cos();
        let ly = cy + label_r * angle.sin();
        let label = &DIMENSION_LABELS[dim.index()];
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
                        {label.name_en}
                    </text>
                })
            } else {
                None
            }}
        }
    }).collect::<Vec<_>>();

    // Grid rings (static)
    let rings_view = rings.iter().map(|&val| {
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
    }).collect::<Vec<_>>();

    // Profile polygon + center text (reactive)
    let profile_view = move || {
        let profile = consciousness.profile.get();
        let score = combined_score(&profile);
        let tier = civic_tier(&profile);
        let color = tier_color(score);
        let pts = profile_polygon(&profile, radius, cx, cy);

        view! {
            <polygon
                points=pts
                fill=color
                fill-opacity="0.25"
                stroke=color
                stroke-width="2"
                stroke-linejoin="round"
            />
            // Dots at each axis vertex
            {SovereignDimension::ALL.iter().enumerate().map(|(i, dim)| {
                let angle = std::f64::consts::TAU * (i as f64) / (AXIS_COUNT as f64) - std::f64::consts::FRAC_PI_2;
                let val = profile.get(*dim).clamp(0.0, 1.0);
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
            // Center: tier label + score
            <text
                x={format!("{:.1}", cx)}
                y={format!("{:.1}", cy - 6.0)}
                text-anchor="middle"
                dominant-baseline="central"
                font-size=center_font
                font-weight="600"
                fill=color
            >
                {tier.label()}
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
                {format!("{:.0}%", score * 100.0)}
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
