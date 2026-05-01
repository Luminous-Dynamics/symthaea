// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hearth domain — excellent: kinship bonds, gratitude, stories, milestones.

use domain_hearth::types::*;
use leptos::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum HearthView {
    Kinship,
    Gratitude,
    Stories,
    Milestones,
}

#[derive(Clone)]
struct HearthState {
    bonds: RwSignal<Vec<KinshipBond>>,
    gratitude: RwSignal<Vec<GratitudeNote>>,
    stories: RwSignal<Vec<Story>>,
    milestones: RwSignal<Vec<Milestone>>,
    show_gratitude_form: RwSignal<bool>,
}

impl HearthState {
    fn new() -> Self {
        Self {
            bonds: RwSignal::new(vec![
                KinshipBond { id: "kb-1".into(), from_did: "did:me".into(), to_did: "did:elena".into(), bond_type: BondType::Partner, strength: 0.95, created_at: 1700000000 },
                KinshipBond { id: "kb-2".into(), from_did: "did:me".into(), to_did: "did:ama".into(), bond_type: BondType::Mentor, strength: 0.8, created_at: 1705000000 },
                KinshipBond { id: "kb-3".into(), from_did: "did:me".into(), to_did: "did:marcus".into(), bond_type: BondType::Companion, strength: 0.75, created_at: 1708000000 },
                KinshipBond { id: "kb-4".into(), from_did: "did:me".into(), to_did: "did:kai".into(), bond_type: BondType::Child, strength: 1.0, created_at: 1690000000 },
            ]),
            gratitude: RwSignal::new(vec![
                GratitudeNote { id: "g-1".into(), from_did: "did:me".into(), to_did: "did:marcus".into(), message: "Thank you for watching Kai while I was at the council meeting. Your presence gave me peace.".into(), tags: vec!["care".into(), "trust".into()], created_at: 1711900800 },
                GratitudeNote { id: "g-2".into(), from_did: "did:elena".into(), to_did: "did:me".into(), message: "The solar garden proposal was beautiful. You spoke for all of us.".into(), tags: vec!["governance".into()], created_at: 1711814400 },
                GratitudeNote { id: "g-3".into(), from_did: "did:ama".into(), to_did: "did:community".into(), message: "The food distribution yesterday was nourishing. We are fed.".into(), tags: vec!["food".into(), "community".into()], created_at: 1711728000 },
            ]),
            stories: RwSignal::new(vec![
                Story { id: "s-1".into(), author_did: "did:ama".into(), title: "The Day the Lights Came On".into(), content: "When the solar panels first activated on Block 7, the children ran outside to watch...".into(), tags: vec!["energy".into(), "community".into()], visibility: Visibility::Community, created_at: 1711641600 },
                Story { id: "s-2".into(), author_did: "did:me".into(), title: "Kai's First Word".into(), content: "It was 'agua'. Of course it was agua.".into(), tags: vec!["family".into()], visibility: Visibility::Family, created_at: 1711555200 },
            ]),
            milestones: RwSignal::new(vec![
                Milestone { id: "m-1".into(), person_did: "did:kai".into(), title: "Kai starts pre-school".into(), description: "First day at the community learning center".into(), milestone_type: MilestoneType::School, date: 1712505600 },
                Milestone { id: "m-2".into(), person_did: "did:me".into(), title: "1 year in the Commons".into(), description: "Anniversary of joining the Mycelix community".into(), milestone_type: MilestoneType::Custom, date: 1711900800 },
            ]),
            show_gratitude_form: RwSignal::new(false),
        }
    }
}

fn bond_type_label(b: &BondType) -> &'static str {
    match b {
        BondType::Parent => "Parent",
        BondType::Child => "Child",
        BondType::Sibling => "Sibling",
        BondType::Partner => "Partner",
        BondType::Chosen => "Chosen",
        BondType::Mentor => "Mentor",
        BondType::Elder => "Elder",
        BondType::Companion => "Companion",
    }
}
fn bond_color(b: &BondType) -> &'static str {
    match b {
        BondType::Partner => "#ec4899",
        BondType::Child => "#f59e0b",
        BondType::Parent => "#8b5cf6",
        BondType::Mentor => "#06b6d4",
        BondType::Companion => "#22c55e",
        _ => "#DB2777",
    }
}

#[component]
pub fn HearthOverview() -> impl IntoView {
    let (active_view, set_active_view) = signal(HearthView::Kinship);
    let state = HearthState::new();
    provide_context(state.clone());

    view! {
        <div class="hearth-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == HearthView::Kinship { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HearthView::Kinship)>"Kinship"</button>
                <button class=move || if active_view.get() == HearthView::Gratitude { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HearthView::Gratitude)>"Gratitude"</button>
                <button class=move || if active_view.get() == HearthView::Stories { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HearthView::Stories)>"Stories"</button>
                <button class=move || if active_view.get() == HearthView::Milestones { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HearthView::Milestones)>"Milestones"</button>
            </div>

            <Show when=move || active_view.get() == HearthView::Kinship>
                <div class="thought-list">
                    {move || { state.bonds.get().iter().map(|b| {
                        let label = bond_type_label(&b.bond_type);
                        let color = bond_color(&b.bond_type);
                        let other = b.to_did.clone();
                        let strength = b.strength;
                        let display_name = other[4..other.len().min(20)].to_string();
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {color};")>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {color}")>{label}</span>
                                    <span class="thought-confidence">{format!("strength: {:.0}%", strength * 100.0)}</span>
                                </div>
                                <p class="thought-content" style="font-weight: 600;">{display_name}</p>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            <Show when=move || active_view.get() == HearthView::Gratitude>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-size: 0.7rem; color: var(--text-muted)">{move || format!("{} notes", state.gratitude.get().len())}</span>
                    <button class="domain-nav-btn" on:click=move |_| state.show_gratitude_form.update(|v| *v = !*v)>
                        {move || if state.show_gratitude_form.get() { "- Close" } else { "+ Send Gratitude" }}
                    </button>
                </div>
                <div class="thought-form" style:display=move || if state.show_gratitude_form.get() { "flex" } else { "none" }>
                    <GratitudeForm />
                </div>
                <div class="thought-list">
                    {move || { state.gratitude.get().iter().map(|g| {
                        let from = g.from_did.clone();
                        let to = g.to_did.clone();
                        let msg = g.message.clone();
                        let tags = g.tags.clone();
                        let is_mine = from == "did:me";
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {};", if is_mine { "#DB2777" } else { "#F472B6" })>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {}", if is_mine { "#DB2777" } else { "#F472B6" })>
                                        {if is_mine { "SENT" } else { "RECEIVED" }}
                                    </span>
                                    <span class="thought-domain">{if is_mine { format!("\u{2192} {}", &to[4..to.len().min(16)]) } else { format!("\u{2190} {}", &from[4..from.len().min(16)]) }}</span>
                                </div>
                                <p class="thought-content" style="font-style: italic;">"\u{201c}"{msg}"\u{201d}"</p>
                                <div class="thought-tags">
                                    {tags.iter().map(|t| { let t = t.clone(); view! { <span class="thought-tag">{t}</span> } }).collect::<Vec<_>>()}
                                </div>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            <Show when=move || active_view.get() == HearthView::Stories>
                <div class="thought-list">
                    {move || { state.stories.get().iter().map(|s| {
                        let title = s.title.clone();
                        let content = s.content.clone();
                        let vis = format!("{:?}", s.visibility);
                        let tags = s.tags.clone();
                        view! {
                            <div class="thought-card">
                                <div class="thought-meta">
                                    <span class="thought-type" style="color: #DB2777">{vis}</span>
                                </div>
                                <h3 style="font-size: 0.95rem; font-weight: 600; margin-bottom: 0.3rem;">{title}</h3>
                                <p style="font-size: 0.85rem; line-height: 1.6; color: var(--text-secondary); font-style: italic;">{content}</p>
                                <div class="thought-tags" style="margin-top: 0.3rem;">
                                    {tags.iter().map(|t| { let t = t.clone(); view! { <span class="thought-tag">{t}</span> } }).collect::<Vec<_>>()}
                                </div>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            <Show when=move || active_view.get() == HearthView::Milestones>
                <div class="thought-list">
                    {move || { state.milestones.get().iter().map(|m| {
                        let title = m.title.clone();
                        let desc = m.description.clone();
                        let mtype = format!("{:?}", m.milestone_type);
                        view! {
                            <div class="thought-card" style="border-left: 3px solid #F472B6;">
                                <div class="thought-meta">
                                    <span class="thought-type" style="color: #F472B6">{mtype}</span>
                                </div>
                                <h3 style="font-size: 0.95rem; font-weight: 600;">{title}</h3>
                                <p style="font-size: 0.8rem; color: var(--text-secondary)">{desc}</p>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>
        </div>
    }
}

#[component]
fn GratitudeForm() -> impl IntoView {
    let state = use_context::<HearthState>().expect("HearthState");
    let (to, set_to) = signal(String::new());
    let (message, set_message) = signal(String::new());

    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        let t = to.get();
        let m = message.get();
        if m.trim().is_empty() {
            return;
        }
        state.gratitude.update(|notes| {
            notes.insert(
                0,
                GratitudeNote {
                    id: format!("g-{}", notes.len() + 4),
                    from_did: "did:me".into(),
                    to_did: if t.is_empty() {
                        "did:community".into()
                    } else {
                        format!("did:{t}")
                    },
                    message: m.trim().to_string(),
                    tags: vec![],
                    created_at: 0,
                },
            );
        });
        set_to.set(String::new());
        set_message.set(String::new());
        state.show_gratitude_form.set(false);
    };

    view! {
        <form on:submit=on_submit style="display: flex; flex-direction: column; gap: 0.5rem; width: 100%;">
            <input type="text" class="form-input" placeholder="To (name or leave empty for community)..."
                prop:value=move || to.get()
                on:input=move |ev| set_to.set(event_target_value(&ev)) />
            <textarea class="form-textarea" placeholder="What are you grateful for?..." rows="3"
                prop:value=move || message.get()
                on:input=move |ev| set_message.set(event_target_value(&ev)) />
            <button type="submit" class="form-submit">"Send Gratitude"</button>
        </form>
    }
}
