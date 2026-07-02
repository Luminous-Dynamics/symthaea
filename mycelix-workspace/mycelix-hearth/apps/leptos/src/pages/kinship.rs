// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::hearth_actions;
use crate::components::TierGate;
use crate::visualization::KinshipCanvas;
use hearth_leptos_types::*;
use personal_leptos_types::TrustTier;

#[component]
pub fn KinshipPage() -> impl IntoView {
    let hearth = use_hearth();

    // Invite form state
    let (show_invite, set_show_invite) = signal(false);
    let (invite_name, set_invite_name) = signal(String::new());
    let (invite_role, set_invite_role) = signal("Adult".to_string());

    let submit_invite = move |_| {
        let name = invite_name.get();
        if name.is_empty() { return; }
        let role = match invite_role.get().as_str() {
            "Elder" => MemberRole::Elder,
            "Youth" => MemberRole::Youth,
            "Guest" => MemberRole::Guest,
            _ => MemberRole::Adult,
        };
        hearth_actions::invite_member(name, role);
        set_invite_name.set(String::new());
        set_show_invite.set(false);
    };

    view! {
        <div class="page kinship-page">
            <h1 class="page-title">"bonds"</h1>
            <p class="page-subtitle">"the living web of your relationships"</p>

            // Kinship web visualization
            <div class="viz-container">
                <KinshipCanvas />
            </div>

            // Invite member (gated to Standard+)
            <TierGate min_tier=TrustTier::Standard action_label="invite members">
                <button
                    class="action-btn"
                    on:click=move |_| set_show_invite.set(!show_invite.get())
                >
                    {move || if show_invite.get() { "Cancel" } else { "+ Invite Member" }}
                </button>
            </TierGate>

            // Invite form
            {move || show_invite.get().then(|| view! {
                <div class="form-card">
                    <div class="form-row">
                        <label>"Name"</label>
                        <input
                            type="text"
                            placeholder="Display name"
                            prop:value=move || invite_name.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                if let Some(input) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlInputElement>().cloned()) {
                                    set_invite_name.set(input.value());
                                }
                            }
                        />
                    </div>
                    <div class="form-row">
                        <label>"Role"</label>
                        <select on:change=move |ev| {
                            use wasm_bindgen::JsCast;
                            if let Some(sel) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlSelectElement>().cloned()) {
                                set_invite_role.set(sel.value());
                            }
                        }>
                            <option value="Adult">"Adult"</option>
                            <option value="Elder">"Elder"</option>
                            <option value="Youth">"Youth"</option>
                            <option value="Guest">"Guest"</option>
                        </select>
                    </div>
                    <button class="action-btn" on:click=submit_invite>"Send Invitation"</button>
                </div>
            })}

            // Bonds first — the most important interactive element
            <section>
                <h2>"bond health"</h2>
                {move || {
                    let members = hearth.members.get();
                    hearth.bonds.get().iter().map(|b| {
                        let name_a = member_name(&members, &b.member_a);
                        let name_b = member_name(&members, &b.member_b);
                        let bond_label = b.bond_type.label().to_string();
                        let strength = b.strength_bp;
                        let pct = (strength as f64 / BOND_MAX as f64 * 100.0) as u32;
                        let health_class = if strength >= 7000 {
                            "bond-bar-healthy"
                        } else if strength >= 4000 {
                            "bond-bar-moderate"
                        } else {
                            "bond-bar-neglected"
                        };
                        let hash = b.hash.clone();
                        let at_max = strength >= BOND_MAX;
                        view! {
                            <div class="bond-detail-card">
                                <div class="bond-names">
                                    <span>{name_a}</span>
                                    <span class="bond-type-label">{bond_label}</span>
                                    <span>{name_b}</span>
                                </div>
                                <div class="bond-bar-row">
                                    <div class="bond-bar-container">
                                        <div
                                            class=format!("bond-bar {health_class}")
                                            style=format!("width: {pct}%")
                                        ></div>
                                    </div>
                                    <span class="bond-pct">{format!("{pct}%")}</span>
                                    <button
                                        class="tend-btn"
                                        disabled=at_max
                                        on:click={
                                            let hash = hash.clone();
                                            move |_| hearth_actions::tend_bond(hash.clone())
                                        }
                                    >
                                        {if at_max { "Full" } else { "Tend" }}
                                    </button>
                                </div>
                            </div>
                        }
                    }).collect_view()
                }}
            </section>

            // Members below bonds
            <section>
                <h2>"family"</h2>
                {move || hearth.members.get().iter().map(|m| {
                    let name = m.display_name.clone();
                    let role_label = m.role.label().to_string();
                    let is_guardian = m.role.is_guardian();
                    let status = if m.status == MembershipStatus::Invited { " (invited)" } else { "" };
                    view! {
                        <div class="member-card" role="listitem">
                            <div class="member-info">
                                <span class="member-name">{name}{status}</span>
                                <span class="member-role">{role_label}</span>
                            </div>
                            {is_guardian.then(|| view! {
                                <span class="guardian-badge">"guardian"</span>
                            })}
                        </div>
                    }
                }).collect_view()}
            </section>
        </div>
    }
}
