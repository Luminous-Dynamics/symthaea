// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::hearth_actions;
use hearth_leptos_types::*;

#[component]
pub fn RhythmsPage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page rhythms-page">
            <h1 class="page-title">"rhythms & presence"</h1>
            <p class="page-subtitle">"daily routines and who is home"</p>

            // Presence strip
            <section>
                <h2>"Right Now"</h2>
                <div class="presence-row">
                    {move || {
                        let members = hearth.members.get();
                        hearth.presence.get().iter().map(|p| {
                            let name = member_name(&members, &p.agent);
                            let status = p.status.label().to_string();
                            let css = match p.status {
                                PresenceStatusType::Home => "presence-home",
                                PresenceStatusType::Sleeping => "presence-sleeping",
                                _ => "presence-away",
                            };
                            view! {
                                <div class=format!("presence-chip {css}")>
                                    <span class="presence-name">{name}</span>
                                    <span class="presence-status">{status}</span>
                                </div>
                            }
                        }).collect_view()
                    }}
                </div>
            </section>

            // My presence toggle
            <section>
                <h2>"Set Your Status"</h2>
                <div class="presence-toggle-row">
                    {[
                        ("Home", PresenceStatusType::Home),
                        ("Away", PresenceStatusType::Away),
                        ("Working", PresenceStatusType::Working),
                        ("Sleeping", PresenceStatusType::Sleeping),
                        ("Do Not Disturb", PresenceStatusType::DoNotDisturb),
                    ].into_iter().map(|(label, status)| {
                        let my_status = move || {
                            let my_agent = hearth.my_agent.get();
                            hearth.presence.get().iter()
                                .find(|p| p.agent == my_agent)
                                .map(|p| p.status.clone())
                                .unwrap_or(PresenceStatusType::Away)
                        };
                        let is_active = move || my_status() == status;
                        view! {
                            <button
                                class=move || format!("presence-btn {}", if is_active() { "presence-btn-active" } else { "" })
                                on:click=move |_| hearth_actions::change_presence(status.clone())
                            >
                                {label}
                            </button>
                        }
                    }).collect_view()}
                </div>
            </section>

            // Rhythms
            <section>
                <h2>"Family Rhythms"</h2>
                {move || {
                    let members = hearth.members.get();
                    hearth.rhythms.get().iter().map(|r| {
                        let name = r.name.clone();
                        let desc = r.description.clone();
                        let rtype = format!("{:?}", r.rhythm_type);
                        let participants: Vec<_> = r.participants.iter()
                            .map(|a| member_name(&members, a))
                            .collect();
                        view! {
                            <div class="rhythm-card">
                                <div class="rhythm-header">
                                    <h3>{name}</h3>
                                    <span class="rhythm-type">{rtype}</span>
                                </div>
                                <p class="rhythm-desc">{desc}</p>
                                <p class="rhythm-participants">{participants.join(", ")}</p>
                            </div>
                        }
                    }).collect_view()
                }}
            </section>
        </div>
    }
}
