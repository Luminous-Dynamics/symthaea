// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Progress bar component.
//!
//! A horizontal progress bar with optional label and configurable color.

use leptos::prelude::*;

/// A horizontal progress bar.
///
/// The `value` prop should be between 0.0 and 1.0. Values outside this
/// range are clamped. The optional `label` is displayed centered over the
/// bar, and `color` overrides the default fill color.
///
/// # Props
///
/// * `value` — Progress fraction, 0.0 to 1.0.
/// * `label` — Optional text displayed over the bar.
/// * `color` — Optional CSS color for the fill (default: `"#2563eb"`).
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::ProgressBar;
/// // view! {
/// //     <ProgressBar value=0.65 label=Some("65%".to_string()) />
/// // }
/// ```
#[component]
pub fn ProgressBar(
    value: f64,
    #[prop(optional)] label: Option<String>,
    #[prop(optional)] color: Option<String>,
) -> impl IntoView {
    let clamped = value.clamp(0.0, 1.0);
    let pct = clamped * 100.0;
    let fill_color = color.unwrap_or_else(|| "#2563eb".to_string());

    view! {
        <div
            style="position: relative; width: 100%; height: 20px; \
                   background-color: #e5e7eb; border-radius: 9999px; \
                   overflow: hidden; font-family: system-ui, sans-serif;"
            role="progressbar"
            aria-valuenow=pct
            aria-valuemin=0.0
            aria-valuemax=100.0
        >
            <div
                style=format!(
                    "height: 100%; width: {pct:.1}%; background-color: {fill_color}; \
                     border-radius: 9999px; transition: width 0.3s ease;"
                )
            ></div>
            {label.map(|text| {
                view! {
                    <span
                        style="position: absolute; inset: 0; display: flex; \
                               align-items: center; justify-content: center; \
                               font-size: 0.7rem; font-weight: 600; color: #1f2937;"
                    >
                        {text}
                    </span>
                }
            })}
        </div>
    }
}
