// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Finance action dispatch: mock branch (instant local mutation) vs
//! real branch (spawn_local zome call).

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use finance_leptos_types::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, ToastKind};
use crate::context::use_finance_context;

/// Record a TEND mutual-credit exchange (hours of care).
pub fn record_tend_exchange(
    receiver_did: String,
    hours: f32,
    description: String,
) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("TND-{:04}", ctx.tend_exchanges.get_untracked().len() + 1);
        let now = (js_sys::Date::now() / 1000.0) as i64;
        ctx.tend_exchanges.update(|exchanges| {
            exchanges.push(TendExchangeView {
                hash: String::new(),
                id: id.clone(),
                provider_did: "did:mycelix:user-001".into(),
                receiver_did: receiver_did.clone(),
                hours,
                service_description: description.clone(),
                service_category: ServiceCategory::GeneralAssistance,
                status: ExchangeStatus::Confirmed,
                created: now,
            });
        });
        ctx.tend_balance.update(|b| {
            b.balance += hours as i32;
            b.total_provided += hours;
            b.exchange_count += 1;
            b.last_activity = now;
        });
        toasts.push(
            format!("{hours} TEND recorded — care given to the commons"),
            ToastKind::Custom("finance".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                receiver_did: String,
                hours: f32,
                description: String,
            }
            let input = Input { receiver_did, hours, description };
            match hc.call_zome_default::<_, serde_json::Value>(
                "tend", "record_exchange", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("{hours} TEND recorded — care given to the commons"),
                    ToastKind::Custom("finance".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("record TEND exchange failed: {e}").into());
                    toasts.push("TEND exchange could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}

/// Send a SAP payment to another member.
pub fn send_sap_payment(to_did: String, amount: u64, memo: String) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("SAP-{:04}", ctx.sap_payments.get_untracked().len() + 1);
        let now = (js_sys::Date::now() / 1000.0) as i64;
        ctx.sap_payments.update(|payments| {
            payments.push(SapPaymentView {
                hash: String::new(),
                id: id.clone(),
                from_did: "did:mycelix:user-001".into(),
                to_did: to_did.clone(),
                amount,
                fee: 0,
                memo: if memo.is_empty() { None } else { Some(memo.clone()) },
                status: PaymentStatus::Completed,
                created: now,
            });
        });
        ctx.sap_balance.update(|b| {
            b.balance = b.balance.saturating_sub(amount);
        });
        let display = amount as f64 / 1_000_000.0;
        toasts.push(
            format!("{display:.2} SAP sent to {to_did}"),
            ToastKind::Custom("finance".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                to_did: String,
                amount: u64,
                memo: String,
            }
            let input = Input { to_did: to_did.clone(), amount, memo };
            match hc.call_zome_default::<_, serde_json::Value>(
                "sap", "send_payment", &input,
            ).await {
                Ok(_) => {
                    let display = amount as f64 / 1_000_000.0;
                    toasts.push(
                        format!("{display:.2} SAP sent to {to_did}"),
                        ToastKind::Custom("finance".into()),
                    );
                }
                Err(e) => {
                    web_sys::console::log_1(&format!("send SAP payment failed: {e}").into());
                    toasts.push("SAP payment could not be sent", ToastKind::Error);
                }
            }
        });
    }
}

/// Recognize a community member's contribution (affects MYCEL score).
pub fn recognize_member(recipient_did: String, contribution_type: ContributionType) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let now = (js_sys::Date::now() / 1000.0) as i64;
        ctx.recognitions.update(|recs| {
            recs.push(RecognitionEventView {
                hash: String::new(),
                recognizer_did: "did:mycelix:user-001".into(),
                recipient_did: recipient_did.clone(),
                contribution_type: contribution_type.clone(),
                weight: 1.0,
                cycle_id: format!("cycle-{}", now / 86400),
                created: now,
            });
        });
        toasts.push(
            format!("{} contribution recognized for {recipient_did}", contribution_type.label()),
            ToastKind::Custom("finance".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                recipient_did: String,
                contribution_type: ContributionType,
            }
            let input = Input { recipient_did: recipient_did.clone(), contribution_type: contribution_type.clone() };
            match hc.call_zome_default::<_, serde_json::Value>(
                "recognition", "recognize_member", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("{} contribution recognized for {recipient_did}", contribution_type.label()),
                    ToastKind::Custom("finance".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("recognize member failed: {e}").into());
                    toasts.push("Recognition could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}

/// Stake SAP tokens for governance weight.
pub fn stake_sap(amount: u64) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("STK-{:04}", ctx.stakes.get_untracked().len() + 1);
        let now = (js_sys::Date::now() / 1000.0) as i64;
        ctx.stakes.update(|stakes| {
            stakes.push(StakeView {
                hash: String::new(),
                id: id.clone(),
                staker_did: "did:mycelix:user-001".into(),
                sap_amount: amount,
                mycel_score: 0.0,
                stake_weight: 0.0,
                status: StakeStatus::Active,
                created: now,
                unbonding_until: None,
            });
        });
        ctx.sap_balance.update(|b| {
            b.balance = b.balance.saturating_sub(amount);
        });
        let display = amount as f64 / 1_000_000.0;
        toasts.push(
            format!("{display:.2} SAP staked for governance"),
            ToastKind::Custom("finance".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input { amount: u64 }
            let input = Input { amount };
            match hc.call_zome_default::<_, serde_json::Value>(
                "staking", "stake", &input,
            ).await {
                Ok(_) => {
                    let display = amount as f64 / 1_000_000.0;
                    toasts.push(
                        format!("{display:.2} SAP staked for governance"),
                        ToastKind::Custom("finance".into()),
                    );
                }
                Err(e) => {
                    web_sys::console::log_1(&format!("stake SAP failed: {e}").into());
                    toasts.push("Staking could not be completed", ToastKind::Error);
                }
            }
        });
    }
}
