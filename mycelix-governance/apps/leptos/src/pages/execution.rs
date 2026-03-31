// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Execution page: timelocks, guardian vetoes, override challenges.
//!
//! A timelock is a held breath — the proposal gestates before release.
//! Guardian veto is a red membrane; overrides require 67% supermajority.

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;
use governance_leptos_types::*;

#[component]
pub fn ExecutionPage() -> impl IntoView {
    let gov = use_governance();

    view! {
        <div class="execution-page" data-page="execution" role="main">
            <h1 class="page-title">"Execution Queue"</h1>
            <p class="page-subtitle">"proposals gestating before release into the commons"</p>

            <div class="timelock-list" data-section="timelocks" role="list">
                {move || {
                    let timelocks = gov.timelocks.get();
                    if timelocks.is_empty() {
                        view! {
                            <div class="empty-state-card">
                                <p class="empty-state">"no proposals are gestating"</p>
                                <p class="empty-state-sub">
                                    "approved proposals enter a timelock period before execution. "
                                    "guardians may veto during this window."
                                </p>
                            </div>
                        }.into_any()
                    } else {
                        timelocks.into_iter().map(|tl| {
                            let status_label = tl.status.label().to_string();
                            let status_label2 = status_label.clone();
                            let pid = tl.proposal_id.clone();
                            let pid2 = pid.clone();
                            let is_vetoed = tl.status == TimelockStatus::Vetoed;
                            view! {
                                <div
                                    class=format!("timelock-card timelock-{}", if is_vetoed { "vetoed" } else { "pending" })
                                    data-timelock-hash=tl.hash.clone()
                                    data-proposal-id=pid.clone()
                                    data-status=status_label
                                    data-duration-hours=tl.duration_hours.to_string()
                                    role="listitem"
                                >
                                    <div class="timelock-header">
                                        <span class="timelock-proposal">{pid2}</span>
                                        <span class="timelock-status">{status_label2}</span>
                                    </div>
                                    <div class="timelock-duration">
                                        {format!("{}h gestation period", tl.duration_hours)}
                                    </div>
                                    {is_vetoed.then(|| view! {
                                        <div class="veto-membrane" data-component="veto-challenge">
                                            <p class="veto-notice">
                                                "a guardian has challenged this proposal. "
                                                "67% supermajority required to override."
                                            </p>
                                        </div>
                                    })}
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}
