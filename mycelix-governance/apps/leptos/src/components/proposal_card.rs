// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Proposal card: a living organism, not a table row.
//!
//! Visual treatment changes based on lifecycle phase:
//! - Discussion: warm orange pulse (growing)
//! - Voting: tense violet contraction (deciding)
//! - Timelock: cool blue breath (gestating)
//! - Executed: green release (completed)
//! - Rejected: grey dimming (composted)

use leptos::prelude::*;
use leptos_router::components::A;
use governance_leptos_types::*;

#[component]
pub fn ProposalCard(proposal: ProposalView) -> impl IntoView {
    let phase = proposal.status.phase();
    let status_label = proposal.status.label();
    let type_label = proposal.proposal_type.label();
    let id = proposal.id.clone();
    let id_for_label = id.clone();
    let title = proposal.title.clone();
    let description = proposal.description.clone();
    let author = proposal.author.clone();
    let href = format!("/proposals/{}", proposal.hash);

    // Time context
    let age_hours = ((js_sys::Date::now() * 1000.0) as i64 - proposal.created) / 3_600_000_000;
    let age_display = if age_hours < 24 {
        format!("{age_hours}h ago")
    } else {
        format!("{}d ago", age_hours / 24)
    };

    view! {
        <A href=href
            attr:class=format!("proposal-card phase-{phase}")
            attr:aria-label=format!("proposal {id_for_label}")
            attr:data-proposal-id=id_for_label.clone()
            attr:data-proposal-hash=proposal.hash
            attr:data-status=status_label.to_string()
            attr:data-phase=phase.to_string()
            attr:data-type=type_label.to_string()
            attr:role="article"
        >
            <div class="proposal-header">
                <span class="proposal-id">{id}</span>
                <span class=format!("proposal-type {}", proposal.proposal_type.css_class())>
                    {type_label}
                </span>
                <span class="proposal-status">{status_label}</span>
            </div>
            <h3 class="proposal-title">{title}</h3>
            <p class="proposal-description">{
                if description.len() > 200 {
                    format!("{}…", &description[..200])
                } else {
                    description
                }
            }</p>
            <div class="proposal-meta">
                <span class="proposal-author">{format!("by {}", author.split(':').last().unwrap_or(&author))}</span>
                <span class="proposal-age">{age_display}</span>
            </div>
            <div class=format!("proposal-pulse phase-{phase}-pulse")></div>
        </A>
    }
}
