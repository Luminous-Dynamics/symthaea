// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic recent-activity feed primitives.

use leptos::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivityFeedItem {
    pub id: String,
    pub domain_label: String,
    pub description: String,
    pub emphasis_class: Option<String>,
}

#[component]
pub fn ActivityFeed(items: Vec<ActivityFeedItem>) -> impl IntoView {
    let items_for_view = items.clone();
    if items.is_empty() {
        view! {
            <div class="activity-feed activity-feed-empty">
                <p class="activity-feed-empty-copy">"No recent activity yet."</p>
            </div>
        }
        .into_any()
    } else {
        view! {
            <div class="activity-feed">
                <For
                    each=move || items_for_view.clone()
                    key=|item| item.id.clone()
                    children=move |item| view! { <ActivityFeedCard item /> }
                />
            </div>
        }
        .into_any()
    }
}

#[component]
fn ActivityFeedCard(item: ActivityFeedItem) -> impl IntoView {
    let emphasis = item.emphasis_class.unwrap_or_default();
    view! {
        <div class=format!("activity-feed-card {}", emphasis)>
            <span class="activity-feed-dot" />
            <div class="activity-feed-body">
                <span class="activity-feed-text">{item.description}</span>
                <span class="activity-feed-domain">{item.domain_label}</span>
            </div>
        </div>
    }
}
