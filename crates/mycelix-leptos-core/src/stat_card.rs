// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Statistic card component.
//!
//! Displays a labeled value with optional subtitle and icon, suitable for
//! dashboards and summary views.

use leptos::prelude::*;

/// A card displaying a labeled statistic.
///
/// Renders a bordered card with a large value, a label above it, and
/// optional subtitle and icon. Intended for dashboard KPI displays.
///
/// # Props
///
/// * `label` — Descriptive label (e.g. "Active Proposals").
/// * `value` — The primary value to display (e.g. "42" or "$1.2M").
/// * `subtitle` — Optional secondary text below the value.
/// * `icon` — Optional icon/emoji displayed alongside the label.
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::StatCard;
/// // view! {
/// //     <StatCard
/// //         label="Active Nodes"
/// //         value="128".to_string()
/// //         subtitle=Some("across 12 clusters".to_string())
/// //     />
/// // }
/// ```
#[component]
pub fn StatCard(
    label: &'static str,
    value: String,
    #[prop(optional)] subtitle: Option<String>,
    #[prop(optional)] icon: Option<&'static str>,
) -> impl IntoView {
    view! {
        <div
            style="padding: 16px 20px; border-radius: 8px; \
                   border: 1px solid #e5e7eb; background-color: white; \
                   font-family: system-ui, sans-serif; min-width: 140px;"
        >
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                {icon.map(|i| view! { <span style="font-size: 1rem;">{i}</span> })}
                <span style="font-size: 0.75rem; font-weight: 500; color: #6b7280; \
                             text-transform: uppercase; letter-spacing: 0.05em;">
                    {label}
                </span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #111827; line-height: 1.2;">
                {value}
            </div>
            {subtitle.map(|sub| {
                view! {
                    <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 4px;">
                        {sub}
                    </div>
                }
            })}
        </div>
    }
}
