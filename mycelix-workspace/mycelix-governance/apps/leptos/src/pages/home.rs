// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civic home: a living overview of the commons' health.
//!
//! Not stat cards. A breathing organism that shows:
//! - How many proposals are growing (discussion)
//! - How many are deciding (voting)
//! - Whether the commons is at rest (homeostasis)
//! - The constitutional heartbeat

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;
use mycelix_leptos_core::{
    AvailabilityState, AvailabilityStateKind, FreshnessBadge, FreshnessLevel, use_consciousness,
};
use crate::components::ProposalCard;
use governance_leptos_types::*;

#[component]
pub fn HomePage() -> impl IntoView {
    let gov = use_governance();
    let consciousness = use_consciousness();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();

    let active_proposals = Memo::new(move |_| {
        gov.proposals.get()
            .into_iter()
            .filter(|p| p.status.is_active())
            .collect::<Vec<_>>()
    });

    let growing = Memo::new(move |_| {
        gov.proposals.get()
            .iter()
            .filter(|p| p.status == ProposalStatus::Draft)
            .count()
    });

    let deciding = Memo::new(move |_| {
        gov.proposals.get()
            .iter()
            .filter(|p| p.status == ProposalStatus::Active)
            .count()
    });

    view! {
        <div class="home-page">
            <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                {move || {
                    gov.charter.get().map(|charter| {
                        view! { <FreshnessBadge level=freshness_from_millis(charter.adopted) detail=format!("Charter {}", format_relative_millis(charter.adopted)) /> }
                    }).unwrap_or_else(|| view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="No charter timestamp yet" /> })
                }}
            </div>

            {move || {
                if hc.is_mock() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Mock
                            title="Mock Civic Posture"
                            description="Governance is showing simulated proposals and constitutional state while live conductor-backed civic data continues wiring in."
                            action={None}
                        />
                    }.into_any()
                } else if gov.proposals.get().is_empty() && gov.charter.get().is_none() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Empty
                            title="Civic Surface Empty"
                            description="Connected, but governance has not returned proposals or charter posture yet."
                            action={None}
                        />
                    }.into_any()
                } else {
                    view! { <></> }.into_any()
                }
            }}

            <section class="civic-breath" aria-label="commons health">
                <div class="breath-state">
                    {move || {
                        let g = growing.get();
                        let d = deciding.get();
                        if g == 0 && d == 0 {
                            view! {
                                <p class="breath-message homeostatic">
                                    "the commons is at rest"
                                </p>
                            }.into_any()
                        } else {
                            view! {
                                <div class="breath-activity">
                                    {(g > 0).then(|| view! {
                                        <p class="breath-growing">
                                            {format!("{g} proposal{} growing", if g == 1 { "" } else { "s" })}
                                        </p>
                                    })}
                                    {(d > 0).then(|| view! {
                                        <p class="breath-deciding">
                                            {format!("{d} decision{} in tension", if d == 1 { "" } else { "s" })}
                                        </p>
                                    })}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
                <div class="breath-tier">
                    <span class="tier-label">"your voice carries as "</span>
                    <span class=move || {
                        format!("tier-badge tier-{}", consciousness.tier.get().css_class())
                    }>
                        {move || consciousness.tier.get().label()}
                    </span>
                </div>
            </section>

            <section class="active-proposals" aria-label="active proposals">
                <h2 class="section-title">"Living proposals"</h2>
                <div class="proposal-grid">
                    {move || {
                        let proposals = active_proposals.get();
                        if proposals.is_empty() {
                            view! {
                                <p class="empty-state">"no proposals are growing right now"</p>
                            }.into_any()
                        } else {
                            proposals.into_iter().map(|p| {
                                view! { <ProposalCard proposal=p /> }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>

            {move || {
                gov.charter.get().map(|charter| view! {
                    <section class="charter-pulse" aria-label="constitutional heartbeat">
                        <h2 class="section-title">"Constitutional heartbeat"</h2>
                        <blockquote class="charter-preamble">
                            {charter.preamble}
                        </blockquote>
                        <div class="charter-rights">
                            {charter.rights.into_iter().map(|right| {
                                view! { <span class="right-badge">{right}</span> }
                            }).collect_view()}
                        </div>
                    </section>
                })
            }}
        </div>
    }
}

fn freshness_from_millis(timestamp_millis: i64) -> FreshnessLevel {
    let now_millis = js_sys::Date::now() as i64;
    let age_minutes = now_millis.saturating_sub(timestamp_millis) / 60_000;
    if age_minutes <= 5 {
        FreshnessLevel::Fresh
    } else if age_minutes <= 60 {
        FreshnessLevel::Aging
    } else {
        FreshnessLevel::Stale
    }
}

fn format_relative_millis(timestamp_millis: i64) -> String {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(timestamp_millis as f64));
    date.to_locale_string("en-US", &wasm_bindgen::JsValue::UNDEFINED)
        .as_string()
        .unwrap_or_else(|| "recently".into())
}
