// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::hearth_actions;
use crate::components::TierGate;
use hearth_leptos_types::*;
use personal_leptos_types::TrustTier;

#[component]
pub fn CarePage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page care-page">
            <h1 class="page-title">"care board"</h1>
            <p class="page-subtitle">"schedules, swaps, and shared meals"</p>

            <TierGate min_tier=TrustTier::Standard action_label="manage care schedules">
                <button class="action-btn">"+ New Care Task"</button>
            </TierGate>

            // Active tasks
            {move || {
                let members = hearth.members.get();
                let schedules = hearth.care_schedules.get();
                let active: Vec<_> = schedules.iter()
                    .filter(|c| c.status == CareScheduleStatus::Active)
                    .cloned().collect();
                let completed: Vec<_> = schedules.iter()
                    .filter(|c| c.status == CareScheduleStatus::Completed)
                    .cloned().collect();

                view! {
                    <section>
                        <h2>{format!("Active ({})", active.len())}</h2>
                        {if active.is_empty() {
                            view! {
                                <div class="empty-state">"nothing to tend right now. the hearth hums quietly."</div>
                            }.into_any()
                        } else {
                            view! {
                                <div>
                                    {active.iter().map(|c| {
                                        let assigned = member_name(&members, &c.assigned_to);
                                        let care_label = c.care_type.label().to_string();
                                        let title = c.title.clone();
                                        let desc = c.description.clone();
                                        let recurrence = format!("{:?}", c.recurrence);
                                        let hash = c.hash.clone();
                                        view! {
                                            <div class="care-card">
                                                <div class="care-header">
                                                    <span class="care-type-badge">{care_label}</span>
                                                    <span class="care-title">{title}</span>
                                                </div>
                                                <p class="care-desc">{desc}</p>
                                                <div class="care-footer">
                                                    <span class="care-assigned">{assigned}</span>
                                                    <span class="care-recurrence">{recurrence}</span>
                                                    <button
                                                        class="complete-btn"
                                                        on:click={
                                                            let hash = hash.clone();
                                                            move |_| hearth_actions::complete_care_task(hash.clone())
                                                        }
                                                    >
                                                        "{2714} Done"
                                                    </button>
                                                </div>
                                            </div>
                                        }
                                    }).collect_view()}
                                </div>
                            }.into_any()
                        }}
                    </section>

                    {(!completed.is_empty()).then(|| view! {
                        <section>
                            <h2>{format!("Completed ({})", completed.len())}</h2>
                            {completed.iter().map(|c| {
                                let title = c.title.clone();
                                let care_label = c.care_type.label().to_string();
                                view! {
                                    <div class="care-card care-completed">
                                        <span class="care-type-badge">{care_label}</span>
                                        <span class="care-title">{title}</span>
                                        <span class="care-done-mark">"{2714}"</span>
                                    </div>
                                }
                            }).collect_view()}
                        </section>
                    })}
                }
            }}
        </div>
    }
}
