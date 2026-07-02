// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};

#[component]
pub fn MilestonesPage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page milestones-page">
            <h1 class="page-title">"milestones"</h1>
            <p class="page-subtitle">"life events and liminal transitions"</p>

            {move || {
                let members = hearth.members.get();
                let milestones = hearth.milestones.get();

                if milestones.is_empty() {
                    view! { <div class="empty-state">"no markers on the trail yet. every journey begins somewhere."</div> }.into_any()
                } else {
                    view! {
                        <div class="timeline">
                            {milestones.iter().map(|m| {
                                let who = member_name(&members, &m.member);
                                let mtype = format!("{:?}", m.milestone_type);
                                let desc = m.description.clone();
                                view! {
                                    <div class="milestone-card">
                                        <div class="milestone-marker"></div>
                                        <div class="milestone-content">
                                            <div class="milestone-header">
                                                <span class="milestone-who">{who}</span>
                                                <span class="milestone-type">{mtype}</span>
                                            </div>
                                            <p>{desc}</p>
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
