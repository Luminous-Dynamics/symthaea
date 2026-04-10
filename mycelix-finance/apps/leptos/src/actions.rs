// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Finance action dispatch: mock branch (instant local mutation) vs
//! real branch (spawn_local zome call).

use finance_leptos_types::*;
use finance_wire_types as wire;
use leptos::prelude::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_local_identity, use_toasts, ToastKind};
use wasm_bindgen_futures::spawn_local;

use crate::adapters::{map_exchange, map_payment, map_recognition, map_stake};
use crate::context::use_finance_context;
use crate::finance_config::use_finance_runtime_config;

/// Record a TEND mutual-credit exchange (hours of care).
pub fn record_tend_exchange(receiver_did: String, hours: f32, description: String) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let config_store = use_finance_runtime_config();
    let local_identity = use_local_identity();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("TND-{:04}", ctx.tend_exchanges.get_untracked().len() + 1);
        let now = current_unix_seconds();
        ctx.tend_exchanges.update(|exchanges| {
            exchanges.push(TendExchangeView {
                hash: String::new(),
                id: id.clone(),
                provider_did: local_identity.did.clone(),
                receiver_did: receiver_did.clone(),
                hours,
                service_description: description.clone(),
                service_category: ServiceCategory::GeneralAssistance,
                status: ExchangeStatus::Confirmed,
                created: now,
            });
        });
        ctx.tend_balance.update(|balance| {
            balance.balance += hours as i32;
            balance.total_provided += hours;
            balance.exchange_count += 1;
            balance.last_activity = now;
        });
        toasts.push(
            format!("{hours} TEND recorded - care given to the commons"),
            ToastKind::Custom("finance".into()),
        );
        return;
    }

    let Some(dao_did) = config_store.get_untracked().dao_did.clone() else {
        toasts.push(
            "Recording TEND exchanges requires a configured or discovered finance DAO DID",
            ToastKind::Error,
        );
        return;
    };

    let hc = hc.clone();
    spawn_local(async move {
        let input = wire::RecordExchangeInput {
            receiver_did,
            hours,
            service_description: description,
            service_category: ServiceCategory::GeneralAssistance,
            cultural_alias: None,
            dao_did,
            service_date: None,
        };

        match hc
            .call_zome_default::<_, wire::ExchangeRecord>("tend", "record_exchange", &input)
            .await
        {
            Ok(exchange) => {
                ctx.tend_exchanges.update(|exchanges| {
                    exchanges.insert(0, map_exchange(exchange));
                });
                toasts.push(
                    format!("{hours} TEND proposed - awaiting confirmation"),
                    ToastKind::Custom("finance".into()),
                );
            }
            Err(error) => {
                web_sys::console::log_1(&format!("record TEND exchange failed: {error}").into());
                toasts.push("TEND exchange could not be recorded", ToastKind::Error);
            }
        }
    });
}

/// Send a SAP payment to another member.
pub fn send_sap_payment(to_did: String, amount: u64, memo: String) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let local_identity = use_local_identity();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("SAP-{:04}", ctx.sap_payments.get_untracked().len() + 1);
        let now = current_unix_seconds();
        ctx.sap_payments.update(|payments| {
            payments.push(SapPaymentView {
                hash: String::new(),
                id: id.clone(),
                from_did: local_identity.did.clone(),
                to_did: to_did.clone(),
                amount,
                fee: 0,
                memo: if memo.is_empty() {
                    None
                } else {
                    Some(memo.clone())
                },
                status: PaymentStatus::Completed,
                created: now,
            });
        });
        ctx.sap_balance.update(|balance| {
            balance.balance = balance.balance.saturating_sub(amount);
        });
        let display = amount as f64 / 1_000_000.0;
        toasts.push(
            format!("{display:.2} SAP sent to {to_did}"),
            ToastKind::Custom("finance".into()),
        );
        return;
    }

    let Some(from_did) = hc.connected_agent_did() else {
        toasts.push(
            "SAP payment requires a connected conductor identity",
            ToastKind::Error,
        );
        return;
    };

    let hc = hc.clone();
    spawn_local(async move {
        let input = wire::SendPaymentInput {
            from_did,
            to_did: to_did.clone(),
            amount,
            currency: "SAP".to_string(),
            payment_type: wire::PaymentType::Direct,
            memo: if memo.is_empty() { None } else { Some(memo) },
        };

        match hc
            .call_zome_default::<_, serde_json::Value>("payments", "send_payment", &input)
            .await
        {
            Ok(record) => {
                if let Some(payment) = extract_entry::<wire::Payment>(&record) {
                    let payment_view = map_payment(payment.clone());
                    let total_debit = payment.amount.saturating_add(payment.fee);
                    ctx.sap_payments.update(|payments| {
                        payments.insert(0, payment_view);
                    });
                    ctx.sap_balance.update(|balance| {
                        balance.balance = balance.balance.saturating_sub(total_debit);
                    });
                }
                let display = amount as f64 / 1_000_000.0;
                toasts.push(
                    format!("{display:.2} SAP sent to {to_did}"),
                    ToastKind::Custom("finance".into()),
                );
            }
            Err(error) => {
                web_sys::console::log_1(&format!("send SAP payment failed: {error}").into());
                toasts.push("SAP payment could not be sent", ToastKind::Error);
            }
        }
    });
}

/// Recognize a community member's contribution (affects MYCEL score).
pub fn recognize_member(recipient_did: String, contribution_type: ContributionType) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let local_identity = use_local_identity();
    let toasts = use_toasts();

    if hc.is_mock() {
        let now = current_unix_seconds();
        ctx.recognitions.update(|recognitions| {
            recognitions.push(RecognitionEventView {
                hash: String::new(),
                recognizer_did: local_identity.did.clone(),
                recipient_did: recipient_did.clone(),
                contribution_type: contribution_type.clone(),
                weight: 1.0,
                cycle_id: current_cycle_id(),
                created: now,
            });
        });
        toasts.push(
            format!(
                "{} contribution recognized for {recipient_did}",
                contribution_type.label()
            ),
            ToastKind::Custom("finance".into()),
        );
        return;
    }

    let hc = hc.clone();
    spawn_local(async move {
        let input = wire::RecognizeMemberInput {
            recipient_did: recipient_did.clone(),
            contribution_type: contribution_type.clone(),
            cycle_id: current_cycle_id(),
        };

        match hc
            .call_zome_default::<_, serde_json::Value>("recognition", "recognize_member", &input)
            .await
        {
            Ok(record) => {
                if let Some(event) = extract_entry::<wire::RecognitionEvent>(&record) {
                    ctx.recognitions.update(|recognitions| {
                        recognitions.insert(0, map_recognition(event));
                    });
                }
                toasts.push(
                    format!(
                        "{} contribution recognized for {recipient_did}",
                        contribution_type.label()
                    ),
                    ToastKind::Custom("finance".into()),
                );
            }
            Err(error) => {
                web_sys::console::log_1(&format!("recognize member failed: {error}").into());
                toasts.push("Recognition could not be recorded", ToastKind::Error);
            }
        }
    });
}

/// Stake SAP tokens for governance weight.
pub fn stake_sap(amount: u64) {
    let hc = use_holochain();
    let ctx = use_finance_context();
    let local_identity = use_local_identity();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("STK-{:04}", ctx.stakes.get_untracked().len() + 1);
        let now = current_unix_seconds();
        ctx.stakes.update(|stakes| {
            stakes.push(StakeView {
                hash: String::new(),
                id: id.clone(),
                staker_did: local_identity.did.clone(),
                sap_amount: amount,
                mycel_score: 0.0,
                stake_weight: 1.0,
                status: StakeStatus::Active,
                created: now,
                unbonding_until: None,
            });
        });
        ctx.sap_balance.update(|balance| {
            balance.balance = balance.balance.saturating_sub(amount);
        });
        let display = amount as f64 / 1_000_000.0;
        toasts.push(
            format!("{display:.2} SAP staked for governance"),
            ToastKind::Custom("finance".into()),
        );
        return;
    }

    let Some(staker_did) = hc.connected_agent_did() else {
        toasts.push(
            "Staking requires a connected conductor identity",
            ToastKind::Error,
        );
        return;
    };

    let hc = hc.clone();
    spawn_local(async move {
        let input = wire::CreateStakeInput {
            staker_did,
            sap_amount: amount,
        };

        match hc
            .call_zome_default::<_, serde_json::Value>("staking", "create_stake", &input)
            .await
        {
            Ok(record) => {
                if let Some(stake) = extract_entry::<wire::CollateralStake>(&record) {
                    ctx.stakes.update(|stakes| {
                        stakes.insert(0, map_stake(stake));
                    });
                }
                let display = amount as f64 / 1_000_000.0;
                toasts.push(
                    format!("{display:.2} SAP staked for governance"),
                    ToastKind::Custom("finance".into()),
                );
            }
            Err(error) => {
                web_sys::console::log_1(&format!("stake SAP failed: {error}").into());
                toasts.push("Staking could not be completed", ToastKind::Error);
            }
        }
    });
}

fn current_unix_seconds() -> i64 {
    (js_sys::Date::now() / 1000.0) as i64
}

fn current_cycle_id() -> String {
    let date = js_sys::Date::new_0();
    format!(
        "{:04}-{:02}",
        date.get_utc_full_year(),
        date.get_utc_month() + 1
    )
}

fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}
