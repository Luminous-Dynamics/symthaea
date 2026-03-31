// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Participatory budgeting: quadratic allocation cycles.
//! AI-interactable: data-cycle-id, data-phase, data-project-id.

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;
use governance_leptos_types::*;

#[component]
pub fn BudgetingPage() -> impl IntoView {
    let gov = use_governance();

    view! {
        <div class="budgeting-page" data-page="budgeting" role="main">
            <h1 class="page-title">"Participatory Budgeting"</h1>
            <p class="page-subtitle">"allocate the commons treasury through quadratic voice credits"</p>

            <div class="budget-cycles" data-section="budget-cycles">
                {move || {
                    let cycles = gov.budget_cycles.get();
                    if cycles.is_empty() {
                        view! {
                            <div class="empty-state-card">
                                <p class="empty-state">"no budget cycles are active"</p>
                                <p class="empty-state-sub">
                                    "budget cycles are created by the Finance Council "
                                    "to allocate treasury funds to community projects"
                                </p>
                            </div>
                        }.into_any()
                    } else {
                        cycles.into_iter().map(|cycle| {
                            let phase_label = cycle.phase.label().to_string();
                            let phase_label2 = phase_label.clone();
                            let budget_sap = cycle.total_budget as f64 / 1_000_000.0;
                            view! {
                                <div
                                    class="budget-cycle-card"
                                    data-cycle-id=cycle.id.clone()
                                    data-phase=format!("{:?}", cycle.phase)
                                    data-budget=cycle.total_budget.to_string()
                                >
                                    <div class="cycle-header">
                                        <h2 class="cycle-name">{cycle.name}</h2>
                                        <span class="cycle-phase-badge">{phase_label}</span>
                                    </div>
                                    <div class="cycle-budget" data-metric="total-budget">
                                        {format!("{budget_sap:.0} SAP to allocate")}
                                    </div>
                                    <div class="cycle-phase-info">
                                        {format!("phase: {phase_label2}")}
                                    </div>
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}
