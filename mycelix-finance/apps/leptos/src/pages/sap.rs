// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn SapPage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-sap">
            <h1>"SAP Wallet"</h1>
            <p class="subtitle">"Transferable currency with 2%/yr demurrage."</p>

            <div class="sap-balance-display">
                <span class="sap-amount">{move || format!("{:.2} SAP", ctx.sap_balance.get().display_balance())}</span>
                <span class="sap-demurrage">{move || format!("Demurrage pending: {:.4} SAP", ctx.sap_balance.get().demurrage_pending as f64 / 1_000_000.0)}</span>
            </div>

            <h2>"Payment History"</h2>
            <div class="payment-list">
                {move || ctx.sap_payments.get().iter().map(|p| {
                    let amount = p.amount as f64 / 1_000_000.0;
                    let memo = p.memo.clone().unwrap_or_default();
                    let status = p.status.label();
                    let to = p.to_did.clone();
                    view! {
                        <div class="payment-card">
                            <div class="payment-header">
                                <span class="payment-amount">{format!("{amount:.2} SAP")}</span>
                                <span class="badge">{status}</span>
                            </div>
                            <span class="payment-to">{format!("To: {to}")}</span>
                            {(!memo.is_empty()).then(|| view! { <span class="payment-memo">{memo}</span> })}
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
