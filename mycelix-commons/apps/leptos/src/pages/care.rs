// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::contexts::commons_context::use_commons;

#[component]
pub fn CarePage() -> impl IntoView {
    let commons = use_commons();

    view! {
        <div class="care-page" data-page="care" role="main">
            <h1 class="page-title">"Mutual Care Network"</h1>
            <p class="page-subtitle">"care circles, reciprocity, and direct support within your community"</p>

            <section class="care-stats" data-section="care-stats">
                <div class="stat-card">
                    <span class="stat-value">{move || commons.care_circles.get().len()}</span>
                    <span class="stat-label">"Care Circles"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">
                        {move || commons.care_circles.get().iter().map(|circle| circle.member_count).sum::<u32>()}
                    </span>
                    <span class="stat-label">"Circle Memberships"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">
                        {move || commons.care_circles.get().iter().filter(|circle| circle.active).count()}
                    </span>
                    <span class="stat-label">"Active Circles"</span>
                </div>
            </section>

            <section class="care-circles-section" data-section="care-circles">
                <h2 class="section-title">"Care Circles"</h2>
                <div class="category-grid" role="list">
                    {move || commons.care_circles.get().into_iter().map(|circle| {
                        let circle_type = circle.circle_type.label().to_string();
                        let circle_type_attr = circle_type.clone();
                        let active = if circle.active { "active" } else { "inactive" };
                        view! {
                            <div class="category-card" data-circle-hash=circle.hash.clone() data-circle-type=circle_type_attr role="listitem">
                                <strong>{circle.name}</strong>
                                <span>{circle_type}</span>
                                <span>{format!("{} members", circle.member_count)}</span>
                                <span>{active}</span>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>

            <section class="recent-activity">
                <h2 class="section-title">"Care Posture"</h2>
                <div class="activity-list">
                    {move || commons.care_circles.get().into_iter().map(|circle| {
                        let circle_type = circle.circle_type.label().to_string();
                        view! {
                            <div class="activity-item" data-circle-hash=circle.hash.clone() data-circle-active=circle.active.to_string()>
                                <span class="activity-type">{circle_type}</span>
                                <span>{circle.description}</span>
                                <span class=if circle.active { "activity-status fulfilled" } else { "activity-status pending" }>
                                    {if circle.active { "Active" } else { "Inactive" }}
                                </span>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>
        </div>
    }
}
