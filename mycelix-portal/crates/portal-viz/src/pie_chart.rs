// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reactive SVG pie/donut chart component.

use leptos::prelude::*;

/// A segment of the pie chart.
#[derive(Clone, Debug)]
pub struct Segment {
    pub label: String,
    pub value: f64,
    pub color: String,
}

/// Reactive SVG pie chart (renders as donut by default).
///
/// Uses SVG path arcs for each segment. Labels placed outside the ring.
#[component]
pub fn PieChart(
    segments: Vec<Segment>,
    #[prop(default = 200.0)] size: f64,
    #[prop(default = 0.6)] inner_ratio: f64,
) -> impl IntoView {
    let cx = size / 2.0;
    let cy = size / 2.0;
    let outer_r = size / 2.0 - 10.0;
    let inner_r = outer_r * inner_ratio;
    let total: f64 = segments.iter().map(|s| s.value).sum::<f64>().max(0.001);

    let mut angle = -std::f64::consts::FRAC_PI_2; // start at 12 o'clock

    let arcs: Vec<_> = segments.iter().map(|seg| {
        let sweep = (seg.value / total) * std::f64::consts::TAU;
        let start_angle = angle;
        let end_angle = angle + sweep;
        angle = end_angle;

        let (sx_outer, sy_outer) = (cx + outer_r * start_angle.cos(), cy + outer_r * start_angle.sin());
        let (ex_outer, ey_outer) = (cx + outer_r * end_angle.cos(), cy + outer_r * end_angle.sin());
        let (sx_inner, sy_inner) = (cx + inner_r * end_angle.cos(), cy + inner_r * end_angle.sin());
        let (ex_inner, ey_inner) = (cx + inner_r * start_angle.cos(), cy + inner_r * start_angle.sin());

        let large_arc = if sweep > std::f64::consts::PI { 1 } else { 0 };

        let path = format!(
            "M {sx_outer:.1} {sy_outer:.1} A {outer_r:.1} {outer_r:.1} 0 {large_arc} 1 {ex_outer:.1} {ey_outer:.1} \
             L {sx_inner:.1} {sy_inner:.1} A {inner_r:.1} {inner_r:.1} 0 {large_arc} 0 {ex_inner:.1} {ey_inner:.1} Z"
        );

        // Label position at midpoint of arc, outside
        let mid_angle = start_angle + sweep / 2.0;
        let label_r = outer_r + 16.0;
        let lx = cx + label_r * mid_angle.cos();
        let ly = cy + label_r * mid_angle.sin();
        let pct = (seg.value / total * 100.0) as u32;

        (path, seg.color.clone(), seg.label.clone(), lx, ly, pct)
    }).collect();

    view! {
        <svg
            viewBox=format!("0 0 {size} {size}")
            style="width: 100%; height: auto; max-width: 200px;"
            xmlns="http://www.w3.org/2000/svg"
        >
            {arcs.iter().map(|(path, color, label, lx, ly, pct)| {
                let path = path.clone();
                let color = color.clone();
                let label = label.clone();
                let lx = *lx;
                let ly = *ly;
                let pct = *pct;
                view! {
                    <path d=path fill=color opacity="0.85" />
                    <text
                        x=lx y=ly
                        text-anchor="middle"
                        fill="rgba(126, 200, 160, 0.7)"
                        font-size="8" font-family="monospace"
                    >
                        {format!("{label} {pct}%")}
                    </text>
                }
            }).collect::<Vec<_>>()}

            // Center label
            <text
                x=cx y=cy
                text-anchor="middle" dominant-baseline="central"
                fill="rgba(232, 197, 71, 0.8)"
                font-size="14" font-family="monospace" font-weight="bold"
            >
                {format!("{:.0}", total)}
            </text>
        </svg>
    }
}
