// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! SAP payments page: stored value with demurrage (2%/yr).
//! AI-interactable: data-balance, data-demurrage-pending, data-payment-id.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;

#[component]
pub fn PaymentsPage() -> impl IntoView {
    let fin = use_finance();

    view! {
        <div class="payments-page" data-page="payments" role="main">
            <h1 class="page-title">"SAP"</h1>
            <p class="page-subtitle">"stored value that composts back to the commons"</p>

            // SAP Balance
            <section class="sap-balance-section" aria-label="SAP balance" data-section="sap-balance">
                {move || {
                    let bal = fin.sap_balance.get();
                    view! {
                        <div
                            class="sap-balance-card"
                            data-balance=bal.balance.to_string()
                            data-display-balance=format!("{:.2}", bal.display_balance())
                            data-demurrage-pending=bal.demurrage_pending.to_string()
                        >
                            <div class="sap-main">
                                <span class="sap-amount" data-metric="balance">
                                    {format!("{:.2} SAP", bal.display_balance())}
                                </span>
                            </div>
                            {(bal.demurrage_pending > 0).then(|| {
                                let pending_sap = bal.demurrage_pending as f64 / 1_000_000.0;
                                view! {
                                    <div class="demurrage-notice" data-metric="demurrage">
                                        {format!("{pending_sap:.2} SAP composting back to the commons (2%/yr demurrage)")}
                                    </div>
                                }
                            })}
                        </div>
                    }
                }}
            </section>

            // Payment history
            <section class="payment-history-section" aria-label="payment history" data-section="payment-history">
                <h2 class="section-title">"Payment history"</h2>
                <div class="payment-list" role="list">
                    {move || {
                        let payments = fin.sap_payments.get();
                        if payments.is_empty() {
                            view! {
                                <p class="empty-state">"no payments recorded"</p>
                            }.into_any()
                        } else {
                            payments.into_iter().map(|p| {
                                let is_outgoing = p.from_did.contains("mock-citizen");
                                let direction = if is_outgoing { "sent" } else { "received" };
                                let other = if is_outgoing { &p.to_did } else { &p.from_did };
                                let other_short = other.split(':').last().unwrap_or(other).to_string();
                                let amount_sap = p.amount as f64 / 1_000_000.0;
                                view! {
                                    <div
                                        class=format!("payment-card payment-{direction}")
                                        data-payment-id=p.id.clone()
                                        data-direction=direction
                                        data-amount=p.amount.to_string()
                                        data-status=format!("{:?}", p.status)
                                        role="listitem"
                                    >
                                        <div class="payment-header">
                                            <span class="payment-direction">
                                                {format!("{direction} {amount_sap:.2} SAP")}
                                            </span>
                                            <span class="payment-to">
                                                {format!("{} {other_short}",
                                                    if is_outgoing { "to" } else { "from" })}
                                            </span>
                                            <span class="payment-status">{p.status.label()}</span>
                                        </div>
                                        {p.memo.map(|m| view! {
                                            <p class="payment-memo">{m}</p>
                                        })}
                                    </div>
                                }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>
        </div>
    }
}
