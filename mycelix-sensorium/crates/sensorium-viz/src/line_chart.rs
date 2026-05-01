// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reactive SVG line chart component.

use leptos::prelude::*;

/// A data series in the line chart.
#[derive(Clone, Debug)]
pub struct Series {
    pub label: String,
    pub color: String,
    pub data: Vec<f64>,
}

/// Reactive SVG line chart with multiple series.
///
/// Renders polylines for each series, y-axis grid lines, and a legend.
/// X-axis is evenly spaced across data points.
#[component]
pub fn LineChart(
    series: Vec<Series>,
    #[prop(default = 400.0)] width: f64,
    #[prop(default = 200.0)] height: f64,
    #[prop(optional)] x_labels: Option<Vec<String>>,
) -> impl IntoView {
    let padding = 40.0;
    let chart_w = width - 2.0 * padding;
    let chart_h = height - 2.0 * padding;

    let all_max = series.iter()
        .flat_map(|s| s.data.iter())
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0);

    let all_min = series.iter()
        .flat_map(|s| s.data.iter())
        .copied()
        .fold(f64::INFINITY, f64::min)
        .min(0.0);

    let range = (all_max - all_min).max(0.001);

    view! {
        <svg
            viewBox=format!("0 0 {width} {height}")
            style="width: 100%; height: auto;"
            xmlns="http://www.w3.org/2000/svg"
        >
            // Grid lines (3 horizontal)
            {(0..=3).map(|i| {
                let y = padding + (i as f64 / 3.0) * chart_h;
                let val = all_max - (i as f64 / 3.0) * range;
                view! {
                    <line
                        x1=padding y1=y x2={width - padding} y2=y
                        stroke="rgba(126, 200, 160, 0.1)" stroke-width="0.5"
                    />
                    <text
                        x={padding - 4.0} y={y + 3.0}
                        text-anchor="end"
                        fill="rgba(126, 200, 160, 0.5)"
                        font-size="8" font-family="monospace"
                    >
                        {format!("{:.0}", val)}
                    </text>
                }
            }).collect::<Vec<_>>()}

            // Series lines
            {series.iter().map(|s| {
                let n = s.data.len().max(1);
                let points: String = s.data.iter().enumerate().map(|(i, &v)| {
                    let x = padding + (i as f64 / (n - 1).max(1) as f64) * chart_w;
                    let y = padding + ((all_max - v) / range) * chart_h;
                    format!("{:.1},{:.1}", x, y)
                }).collect::<Vec<_>>().join(" ");
                let color = s.color.clone();

                view! {
                    <polyline
                        points=points
                        fill="none"
                        stroke=color
                        stroke-width="1.5"
                        opacity="0.85"
                    />
                }
            }).collect::<Vec<_>>()}

            // Legend (top-right)
            {series.iter().enumerate().map(|(i, s)| {
                let y = 12.0 + i as f64 * 14.0;
                let color = s.color.clone();
                let label = s.label.clone();
                view! {
                    <rect x={width - padding - 80.0} y={y - 4.0} width="8" height="8" fill=color />
                    <text
                        x={width - padding - 68.0} y={y + 4.0}
                        fill="rgba(126, 200, 160, 0.7)"
                        font-size="9" font-family="monospace"
                    >
                        {label}
                    </text>
                }
            }).collect::<Vec<_>>()}

            // X-axis labels
            {x_labels.unwrap_or_default().iter().enumerate().map(|(i, label)| {
                let n = series.first().map(|s| s.data.len()).unwrap_or(1).max(1);
                let x = padding + (i as f64 / (n - 1).max(1) as f64) * chart_w;
                let label = label.clone();
                view! {
                    <text
                        x=x y={height - padding + 14.0}
                        text-anchor="middle"
                        fill="rgba(126, 200, 160, 0.5)"
                        font-size="8" font-family="monospace"
                    >
                        {label}
                    </text>
                }
            }).collect::<Vec<_>>()}
        </svg>
    }
}
