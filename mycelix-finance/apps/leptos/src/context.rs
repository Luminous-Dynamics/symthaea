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

    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if !hc.is_mock() {
            let _ = &state;
        }
    });
}

pub fn use_finance_context() -> FinanceCtx {
    expect_context::<FinanceCtx>()
}
