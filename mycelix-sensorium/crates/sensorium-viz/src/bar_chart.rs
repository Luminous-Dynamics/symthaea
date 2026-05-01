// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reactive SVG bar chart component.

use leptos::prelude::*;

/// A single bar in the chart.
#[derive(Clone, Debug)]
pub struct Bar {
    pub label: String,
    pub value: f64,
    pub color: String,
}

/// Reactive SVG bar chart.
///
/// Renders vertical bars with labels below, value labels above,
/// and a configurable height. Bars scale to the maximum value.
///
/// # Example
/// ```rust,no_run
/// use sensorium_viz::BarChart;
/// use sensorium_viz::bar_chart::Bar;
///
/// let data = vec![
///     Bar { label: "Rust".into(), value: 85.0, color: "#7ec8a0".into() },
///     Bar { label: "TS".into(), value: 60.0, color: "#e8c547".into() },
/// ];
/// // view! { <BarChart data=data width=400.0 height=200.0 /> }
/// ```
#[component]
pub fn BarChart(
    data: Vec<Bar>,
    #[prop(default = 400.0)] width: f64,
    #[prop(default = 200.0)] height: f64,
) -> impl IntoView {
    let padding = 40.0;
    let chart_w = width - 2.0 * padding;
    let chart_h = height - 2.0 * padding;
    let max_val = data.iter().map(|b| b.value).fold(0.0_f64, f64::max).max(1.0);
    let n = data.len().max(1);
    let bar_w = (chart_w / n as f64) * 0.7;
    let gap = (chart_w / n as f64) * 0.3;

    view! {
        <svg
            viewBox=format!("0 0 {width} {height}")
            style="width: 100%; height: auto;"
            xmlns="http://www.w3.org/2000/svg"
        >
            // Axis
            <line
                x1=padding y1={height - padding}
                x2={width - padding} y2={height - padding}
                stroke="rgba(126, 200, 160, 0.3)" stroke-width="1"
            />

            // Bars
            {data.iter().enumerate().map(|(i, bar)| {
                let x = padding + i as f64 * (bar_w + gap) + gap / 2.0;
                let bar_h = (bar.value / max_val) * chart_h;
                let y = height - padding - bar_h;
                let color = bar.color.clone();
                let label = bar.label.clone();
                let value = bar.value;

                view! {
                    <rect x=x y=y width=bar_w height=bar_h fill=color rx="3" opacity="0.85" />
                    // Value label above bar
                    <text
                        x={x + bar_w / 2.0} y={y - 4.0}
                        text-anchor="middle"
                        fill="rgba(126, 200, 160, 0.8)"
                        font-size="10" font-family="monospace"
                    >
                        {format!("{:.0}", value)}
                    </text>
                    // Category label below
                    <text
                        x={x + bar_w / 2.0} y={height - padding + 14.0}
                        text-anchor="middle"
                        fill="rgba(126, 200, 160, 0.6)"
                        font-size="9" font-family="monospace"
                    >
                        {label}
                    </text>
                }
            }).collect::<Vec<_>>()}
        </svg>
    }
}
