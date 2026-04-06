// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Finance context: TEND/SAP/MYCEL balances, exchanges, treasury, staking, oracle.
//!
//! Mock-first, same pattern as governance_context.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use finance_leptos_types::*;
use crate::mock_data;
use mycelix_leptos_core::holochain_provider::use_holochain;

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
    pub marketplace_listings: RwSignal<Vec<ServiceListingView>>,
    pub marketplace_requests: RwSignal<Vec<ServiceRequestView>>,
    pub recognitions_received: RwSignal<Vec<RecognitionEventView>>,
}

pub fn provide_finance_context() -> FinanceCtx {
    let ctx = FinanceCtx {
        tend_balance: RwSignal::new(mock_data::mock_tend_balance()),
        sap_balance: RwSignal::new(mock_data::mock_sap_balance()),
        mycel_score: RwSignal::new(mock_data::mock_mycel_score()),
        tend_exchanges: RwSignal::new(mock_data::mock_tend_exchanges()),
        sap_payments: RwSignal::new(mock_data::mock_sap_payments()),
        treasury: RwSignal::new(Some(mock_data::mock_treasury())),
        stakes: RwSignal::new(Vec::new()),
        oracle_state: RwSignal::new(mock_data::mock_oracle_state()),
        marketplace_listings: RwSignal::new(mock_data::mock_listings()),
        marketplace_requests: RwSignal::new(mock_data::mock_requests()),
        recognitions_received: RwSignal::new(mock_data::mock_recognitions()),
    };

    provide_context(ctx.clone());

    let ctx_for_load = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4000).await;
        try_load_real_data(ctx_for_load).await;
    });

    ctx
}

async fn try_load_real_data(ctx: FinanceCtx) {
    let hc = use_holochain();
    if hc.is_mock() {
        return;
    }

    web_sys::console::log_1(&"[Civic] Loading real finance data from conductor…".into());

    // Load TEND balance
    #[derive(serde::Serialize)]
    struct GetBalanceInput { member_did: String, dao_did: String }

    match hc.call_zome::<_, serde_json::Value>(
        "finance", "tend", "get_balance",
        &GetBalanceInput {
            member_did: "self".into(),
            dao_did: "default".into(),
        },
    ).await {
        Ok(record) => {
            if let Ok(balance) = serde_json::from_value::<TendBalanceView>(record) {
                ctx.tend_balance.set(balance);
                web_sys::console::log_1(&"[Civic] Loaded TEND balance from conductor".into());
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Civic] Could not load TEND balance: {e}").into());
        }
    }

    // Load oracle state
    match hc.call_zome::<(), serde_json::Value>("finance", "tend", "get_oracle_state", &()).await {
        Ok(record) => {
            if let Ok(oracle) = serde_json::from_value::<OracleStateView>(record) {
                ctx.oracle_state.set(oracle);
                web_sys::console::log_1(&"[Civic] Loaded oracle state from conductor".into());
            }
        }
        Err(e) => {
            web_sys::console::log_1(&format!("[Civic] Could not load oracle: {e}").into());
        }
    }
}

pub fn use_finance() -> FinanceCtx {
    expect_context::<FinanceCtx>()
}
