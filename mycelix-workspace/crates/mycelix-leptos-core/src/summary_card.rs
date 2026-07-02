// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic summary/status card primitives for Mycelix frontends.
//!
//! These components intentionally avoid domain-specific semantics. Product shells
//! like Sensorium can map their own availability, attention, and launch models
//! into these generic shapes.

use leptos::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryStatusBadge {
    pub label: String,
    pub class_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryMetricItem {
    pub id: String,
    pub label: String,
    pub value: String,
    pub hint: Option<String>,
    pub tone_class: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryAttentionItem {
    pub id: String,
    pub label: String,
    pub detail: String,
    pub level_label: String,
    pub level_class: Option<String>,
    pub accent_color: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryActionItem {
    pub id: String,
    pub label: String,
    pub href: Option<String>,
    pub primary: bool,
    pub disabled: bool,
}

/// Generic summary card with metrics, attention items, and launch actions.
#[component]
pub fn SummaryCard(
    #[prop(into)] title: String,
    status_badge: SummaryStatusBadge,
    #[prop(into)] status_line: String,
    metrics: Vec<SummaryMetricItem>,
    attention: Vec<SummaryAttentionItem>,
    actions: Vec<SummaryActionItem>,
    footer_note: Option<String>,
) -> impl IntoView {
    let title_for_meta = title.clone();
    let status_label = status_badge.label.clone();
    let status_class = status_badge.class_name.clone().unwrap_or_default();
    let metrics_for_list = metrics.clone();
    let attention_view = if attention.is_empty() {
        None
    } else {
        let attention_for_list = attention.clone();
        Some(view! {
            <div class="mycelix-summary-attention" style="display: flex; flex-direction: column; gap: 0.6rem;">
                <For
                    each=move || attention_for_list.clone()
                    key=|item| item.id.clone()
                    children=move |item| view! { <SummaryAttentionCard item /> }
                />
            </div>
        })
    };
    let actions_view = if actions.is_empty() {
        None
    } else {
        let actions_for_list = actions.clone();
        Some(view! {
            <div class="mycelix-summary-actions hero-actions">
                <For
                    each=move || actions_for_list.clone()
                    key=|action| action.id.clone()
                    children=move |action| view! { <SummaryAction action /> }
                />
            </div>
        })
    };

    view! {
        <div class="mycelix-summary-card" style="display: flex; flex-direction: column; gap: 0.85rem;">
            <div class="mycelix-summary-meta" style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span
                    class="mycelix-summary-title"
                    style="font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;"
                >
                    {title_for_meta}
                </span>
                <span class=move || format!("status-pill {}", status_class)>
                    {status_label}
                </span>
            </div>

            <p
                class="mycelix-summary-status"
                style="font-size: 0.9rem; line-height: 1.5; color: var(--text-primary); margin: 0;"
            >
                {status_line}
            </p>

            <div
                class="mycelix-summary-metrics"
                style="display: flex; gap: 2rem; flex-wrap: wrap;"
            >
                <For
                    each=move || metrics_for_list.clone()
                    key=|metric| metric.id.clone()
                    children=move |metric| view! { <SummaryMetric metric /> }
                />
            </div>

            {attention_view}

            {actions_view}

            {footer_note.map(|note| view! {
                <p
                    class="mycelix-summary-footer"
                    style="font-size: 0.75rem; color: var(--text-muted); margin: 0;"
                >
                    {note}
                </p>
            })}
        </div>
    }
}

#[component]
fn SummaryMetric(metric: SummaryMetricItem) -> impl IntoView {
    let tone_class = metric.tone_class.clone().unwrap_or_default();
    view! {
        <div class=format!("mycelix-summary-metric domain-stat {}", tone_class)>
            <span class="stat-value">{metric.value}</span>
            <span class="stat-label">{metric.label}</span>
            {metric.hint.map(|hint| view! {
                <span class="stat-label" style="opacity: 0.8;">{hint}</span>
            })}
        </div>
    }
}

#[component]
fn SummaryAttentionCard(item: SummaryAttentionItem) -> impl IntoView {
    let accent = item
        .accent_color
        .clone()
        .unwrap_or_else(|| "var(--border)".into());
    let level_class = item.level_class.clone().unwrap_or_default();
    view! {
        <div
            class="mycelix-summary-attention-card thought-card"
            style=format!("border-left: 3px solid {accent}; margin-bottom: 0.5rem;")
        >
            <div class="thought-meta">
                <span class=format!("thought-type {}", level_class)>{item.level_label}</span>
                <span class="thought-domain">{item.label}</span>
            </div>
            <p style="font-size: 0.78rem; color: var(--text-secondary); margin: 0;">
                {item.detail}
            </p>
        </div>
    }
}

#[component]
fn SummaryAction(action: SummaryActionItem) -> impl IntoView {
    let class_name = if action.primary {
        "btn btn-primary"
    } else {
        "btn"
    };
    if let Some(href) = action.href {
        view! {
            <a class=class_name href=href target="_blank" rel="noopener">
                {action.label}
            </a>
        }
        .into_any()
    } else {
        view! {
            <button class=class_name disabled=action.disabled>
                {action.label}
            </button>
        }
        .into_any()
    }
}
