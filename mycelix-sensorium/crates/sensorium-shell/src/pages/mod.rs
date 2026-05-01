// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain page components — rendered when the user zooms into a domain.

pub mod admin;
pub mod catalog;
pub mod commons;
pub mod finance;
pub mod governance;
pub mod health;
pub mod hearth;
pub mod knowledge;
pub mod lucid;
pub mod mail;
pub mod praxis;
pub mod sovereignty;

use leptos::prelude::*;
use mycelix_leptos_core::{AvailabilityState, AvailabilityStateKind};

/// Route domain_id to the appropriate page component.
#[component]
pub fn DomainContent(domain_id: String) -> impl IntoView {
    match domain_id.as_str() {
        "governance" => view! { <governance::GovernanceOverview /> }.into_any(),
        "lucid" => view! { <lucid::LucidOverview /> }.into_any(),
        "commons" => view! { <commons::CommonsOverview /> }.into_any(),
        "finance" => view! { <finance::FinanceOverview /> }.into_any(),
        "health" => view! { <health::HealthOverview /> }.into_any(),
        "praxis" => view! { <praxis::PraxisOverview /> }.into_any(),
        "hearth" => view! { <hearth::HearthOverview /> }.into_any(),
        "knowledge" => view! { <knowledge::KnowledgeOverview /> }.into_any(),
        "mail" => view! { <mail::MailOverview /> }.into_any(),
        "admin" => view! { <admin::AdminOverview /> }.into_any(),
        // Meta-views accessible via admin or direct orbital node
        "catalog" => view! { <catalog::CatalogOverview /> }.into_any(),
        "sovereignty" => view! { <sovereignty::SovereigntyDashboard /> }.into_any(),
        _ => view! {
            <AvailabilityState
                kind=AvailabilityStateKind::Unavailable
                title="Domain Unavailable"
                description="This domain is not currently wired into the Sensorium shell. Connect the domain module or standalone app to make it available here."
                action={None}
            />
        }
        .into_any(),
    }
}
