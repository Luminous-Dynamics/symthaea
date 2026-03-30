// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::use_hearth;

#[component]
pub fn EmergencyPage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page emergency-page">
            <h1 class="page-title">"emergency center"</h1>
            <p class="page-subtitle">"safety plans, alerts, and check-ins"</p>

            {move || {
                let alerts = hearth.emergency_alerts.get();
                if alerts.is_empty() {
                    view! {
                        <div class="safe-banner">
                            <span class="safe-icon">"{2714}"</span>
                            <span>"No active alerts. The hearth is safe."</span>
                        </div>
                    }.into_any()
                } else {
                    view! {
                        <div class="alert-list">
                            {alerts.iter().map(|a| {
                                let msg = a.message.clone();
                                let sev = a.severity.css_class().to_string();
                                view! {
                                    <div class=format!("alert-card {sev}")>
                                        <p>{msg}</p>
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
