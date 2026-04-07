// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Finance.

use leptos::prelude::*;
use finance_leptos_types::*;
use mycelix_mock_data::finance as mock;

#[derive(Clone)]
pub struct FinanceCtx {
    pub tend_balance: RwSignal<TendBalanceView>,
    pub sap_balance: RwSignal<SapBalanceView>,
    pub mycel_score: RwSignal<MycelScoreView>,
    pub tend_exchanges: RwSignal<Vec<TendExchangeView>>,
    pub sap_payments: RwSignal<Vec<SapPaymentView>>,
    pub treasury: RwSignal<Option<TreasuryView>>,
    pub stakes: RwSignal<Vec<StakeView>>,
    pub oracle_state: RwSignal<OracleStateView>,
    pub recognitions: RwSignal<Vec<RecognitionEventView>>,
}

pub fn provide_finance_context() {
    let state = FinanceCtx {
        tend_balance: RwSignal::new(mock::tend_balance()),
        sap_balance: RwSignal::new(mock::sap_balance()),
        mycel_score: RwSignal::new(mock::mycel_score()),
        tend_exchanges: RwSignal::new(mock::tend_exchanges()),
        sap_payments: RwSignal::new(mock::sap_payments()),
        treasury: RwSignal::new(Some(mock::treasury())),
        stakes: RwSignal::new(mock::stakes()),
        oracle_state: RwSignal::new(mock::oracle_state()),
        recognitions: RwSignal::new(mock::recognitions()),
    };
    provide_context(state.clone());

    // Attempt real data load from conductor
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            web_sys::console::log_1(&"[Finance] Conductor connected — loading real data...".into());

            let owner = "did:mycelix:user-001".to_string();

            // Load TEND balance
            if let Ok(balance) = hc.call_zome_default::<String, TendBalanceView>(
                "tend", "get_balance", &owner
            ).await {
                state.tend_balance.set(balance);
            }

            // Load SAP balance
            if let Ok(balance) = hc.call_zome_default::<String, SapBalanceView>(
                "sap", "get_balance", &owner
            ).await {
                state.sap_balance.set(balance);
            }

            // Load MYCEL score
            if let Ok(score) = hc.call_zome_default::<String, MycelScoreView>(
                "recognition", "get_mycel_score", &owner
            ).await {
                state.mycel_score.set(score);
            }

            // Load TEND exchanges
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "tend", "get_exchanges_by_agent", &owner
            ).await {
                let exchanges: Vec<TendExchangeView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !exchanges.is_empty() {
                    web_sys::console::log_1(&format!("[Finance] Loaded {} TEND exchanges", exchanges.len()).into());
                    state.tend_exchanges.set(exchanges);
                }
            }

            // Load SAP payments
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "sap", "get_payments_by_agent", &owner
            ).await {
                let payments: Vec<SapPaymentView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !payments.is_empty() {
                    web_sys::console::log_1(&format!("[Finance] Loaded {} SAP payments", payments.len()).into());
                    state.sap_payments.set(payments);
                }
            }

            // Load treasury
            if let Ok(treasury) = hc.call_zome_default::<(), TreasuryView>(
                "treasury", "get_treasury", &()
            ).await {
                state.treasury.set(Some(treasury));
            }

            // Load stakes
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "staking", "get_stakes_by_agent", &owner
            ).await {
                let stakes: Vec<StakeView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !stakes.is_empty() {
                    web_sys::console::log_1(&format!("[Finance] Loaded {} stakes", stakes.len()).into());
                    state.stakes.set(stakes);
                }
            }

            // Load oracle state
            if let Ok(oracle) = hc.call_zome_default::<(), OracleStateView>(
                "tend", "get_oracle_state", &()
            ).await {
                state.oracle_state.set(oracle);
            }

            // Load recognitions
            if let Ok(records) = hc.call_zome_default::<String, Vec<serde_json::Value>>(
                "recognition", "get_recognitions_by_agent", &owner
            ).await {
                let recognitions: Vec<RecognitionEventView> = records.iter()
                    .filter_map(|r| extract_entry(r))
                    .collect();
                if !recognitions.is_empty() {
                    web_sys::console::log_1(&format!("[Finance] Loaded {} recognitions", recognitions.len()).into());
                    state.recognitions.set(recognitions);
                }
            }
        }
    });
}

/// Extract the entry from a Holochain Record JSON value.
/// Records have structure: { "entry": { "Present": { ...fields... } } }
fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_finance_context() -> FinanceCtx {
    expect_context::<FinanceCtx>()
}
