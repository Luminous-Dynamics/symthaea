// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Finance.

use finance_leptos_types::*;
use finance_wire_types as wire;
use leptos::prelude::*;
use mycelix_leptos_core::{holochain_provider::use_holochain, use_local_identity};
use mycelix_mock_data::finance as mock;

use crate::adapters::{
    map_commons_pool, map_exchange, map_mycel_score, map_oracle_state, map_payment,
    map_recognition, map_sap_balance, map_stake, map_tend_balance, map_treasury,
};
use crate::finance_config::{merge_runtime_discovery, use_finance_runtime_config};

const LOAD_LIMIT: usize = 100;

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
    let identity = use_local_identity();
    let config_store = use_finance_runtime_config();
    let initial_config = config_store.get_untracked();
    let default_dao_did = initial_config
        .dao_did
        .clone()
        .unwrap_or_else(|| mock::tend_balance().dao_did);
    let state = FinanceCtx {
        tend_balance: RwSignal::new(mock_tend_balance(&identity.did, &default_dao_did)),
        sap_balance: RwSignal::new(mock_sap_balance(&identity.did)),
        mycel_score: RwSignal::new(mock_mycel_score(&identity.did)),
        tend_exchanges: RwSignal::new(mock::tend_exchanges()),
        sap_payments: RwSignal::new(mock::sap_payments()),
        treasury: RwSignal::new(Some(mock::treasury())),
        stakes: RwSignal::new(mock_stakes(&identity.did)),
        oracle_state: RwSignal::new(mock::oracle_state()),
        recognitions: RwSignal::new(mock::recognitions()),
    };
    provide_context(state.clone());

    let hc = use_holochain();
    let fallback_member_did = identity.did.clone();

    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(4_000).await;
        if hc.is_mock() {
            return;
        }

        web_sys::console::log_1(&"[Finance] Conductor connected - loading real data".into());

        let member_did = hc
            .connected_agent_did()
            .unwrap_or_else(|| fallback_member_did.clone());

        load_member_scoped_views(&hc, &state, &member_did).await;

        let effective_config = load_runtime_discovery(&hc, &config_store, &member_did).await;

        if let Some(dao_did) = effective_config.dao_did.clone() {
            load_dao_scoped_views(&hc, &state, &member_did, &dao_did).await;
            load_treasury_view(&hc, &state, &member_did, &effective_config, Some(dao_did)).await;
        } else {
            web_sys::console::warn_1(
                &"[Finance] No DAO DID configured or discovered; skipping DAO-scoped TEND and commons queries"
                    .into(),
            );
            load_treasury_view(&hc, &state, &member_did, &effective_config, None).await;
        }
    });
}

async fn load_runtime_discovery(
    hc: &mycelix_leptos_core::HolochainCtx,
    config_store: &crate::finance_config::FinanceRuntimeConfigStore,
    member_did: &str,
) -> crate::finance_config::FinanceRuntimeConfig {
    let configured = config_store.get_untracked();
    if let Ok(discovered) = hc
        .call_zome_default::<_, wire::FinanceRuntimeDiscovery>(
            "finance_bridge",
            "discover_runtime_context",
            &(),
        )
        .await
    {
        if discovered.member_did == member_did {
            let merged = merge_runtime_discovery(&configured, &discovered);
            config_store.set(merged.clone());
            return merged;
        }
    }

    configured
}

async fn load_member_scoped_views(
    hc: &mycelix_leptos_core::HolochainCtx,
    state: &FinanceCtx,
    member_did: &str,
) {
    if let Ok(balance) = hc
        .call_zome_default::<_, wire::SapBalanceResponse>(
            "payments",
            "get_sap_balance",
            &member_did.to_string(),
        )
        .await
    {
        state.sap_balance.set(map_sap_balance(balance));
    }

    if let Ok(score) = hc
        .call_zome_default::<_, wire::MemberMycelState>(
            "recognition",
            "get_mycel_score",
            &member_did.to_string(),
        )
        .await
    {
        state.mycel_score.set(map_mycel_score(score));
    }

    if let Ok(records) = hc
        .call_zome_default::<_, Vec<serde_json::Value>>(
            "payments",
            "get_payment_history",
            &wire::GetPaymentHistoryInput {
                did: member_did.to_string(),
                limit: Some(LOAD_LIMIT),
            },
        )
        .await
    {
        let payments: Vec<SapPaymentView> = records
            .iter()
            .filter_map(extract_entry::<wire::Payment>)
            .map(map_payment)
            .collect();
        if !payments.is_empty() {
            state.sap_payments.set(payments);
        }
    }

    if let Ok(records) = hc
        .call_zome_default::<_, Vec<serde_json::Value>>(
            "staking",
            "get_staker_stakes",
            &wire::PaginatedDidInput {
                did: member_did.to_string(),
                limit: Some(LOAD_LIMIT),
            },
        )
        .await
    {
        let stakes: Vec<StakeView> = records
            .iter()
            .filter_map(extract_entry::<wire::CollateralStake>)
            .map(map_stake)
            .collect();
        if !stakes.is_empty() {
            state.stakes.set(stakes);
        }
    }

    if let Ok(events) = hc
        .call_zome_default::<_, Vec<wire::RecognitionEvent>>(
            "recognition",
            "get_recognition_received",
            &wire::GetRecognitionsInput {
                member_did: member_did.to_string(),
                cycle_id: None,
                limit: Some(LOAD_LIMIT),
            },
        )
        .await
    {
        state
            .recognitions
            .set(events.into_iter().map(map_recognition).collect());
    }
}

async fn load_dao_scoped_views(
    hc: &mycelix_leptos_core::HolochainCtx,
    state: &FinanceCtx,
    member_did: &str,
    dao_did: &str,
) {
    if let Ok(balance) = hc
        .call_zome_default::<_, wire::BalanceInfo>(
            "tend",
            "get_balance",
            &wire::GetBalanceInput {
                member_did: member_did.to_string(),
                dao_did: dao_did.to_string(),
            },
        )
        .await
    {
        state
            .tend_balance
            .set(map_tend_balance(balance, current_unix_seconds()));
    }

    if let Ok(exchanges) = hc
        .call_zome_default::<_, Vec<wire::ExchangeRecord>>(
            "tend",
            "get_my_exchanges",
            &wire::PaginatedDaoInput {
                dao_did: dao_did.to_string(),
                limit: Some(LOAD_LIMIT),
            },
        )
        .await
    {
        state
            .tend_exchanges
            .set(exchanges.into_iter().map(map_exchange).collect());
    }

    if let Ok(oracle) = hc
        .call_zome_default::<_, wire::OracleStateResponse>("tend", "get_oracle_state", &())
        .await
    {
        state
            .oracle_state
            .set(map_oracle_state(oracle, current_unix_seconds()));
    }
}

async fn load_treasury_view(
    hc: &mycelix_leptos_core::HolochainCtx,
    state: &FinanceCtx,
    member_did: &str,
    config: &crate::finance_config::FinanceRuntimeConfig,
    dao_did: Option<String>,
) {
    if let Some(pool_id) = resolve_commons_pool_id(hc, config, dao_did.as_deref()).await {
        if let Ok(Some(record)) = hc
            .call_zome_default::<_, Option<serde_json::Value>>(
                "treasury",
                "get_commons_pool",
                &pool_id,
            )
            .await
        {
            if let Some(pool) = extract_entry::<wire::CommonsPool>(&record) {
                state.treasury.set(Some(map_commons_pool(pool)));
                return;
            }
        }
    }

    if let Some(treasury_id) = resolve_treasury_id(hc, config, member_did).await {
        if let Ok(Some(record)) = hc
            .call_zome_default::<_, Option<serde_json::Value>>(
                "treasury",
                "get_treasury",
                &treasury_id,
            )
            .await
        {
            if let Some(treasury) = extract_entry::<wire::Treasury>(&record) {
                state.treasury.set(Some(map_treasury(treasury)));
            }
        }
    }
}

async fn resolve_commons_pool_id(
    hc: &mycelix_leptos_core::HolochainCtx,
    config: &crate::finance_config::FinanceRuntimeConfig,
    dao_did: Option<&str>,
) -> Option<String> {
    if let Some(pool_id) = config.commons_pool_id.clone() {
        return Some(pool_id);
    }

    let dao_did = dao_did?;
    let record = hc
        .call_zome_default::<_, Option<serde_json::Value>>(
            "treasury",
            "get_dao_commons_pool",
            &dao_did.to_string(),
        )
        .await
        .ok()??;
    extract_entry::<wire::CommonsPool>(&record).map(|pool| pool.id)
}

async fn resolve_treasury_id(
    hc: &mycelix_leptos_core::HolochainCtx,
    config: &crate::finance_config::FinanceRuntimeConfig,
    member_did: &str,
) -> Option<String> {
    if let Some(treasury_id) = config.treasury_id.clone() {
        return Some(treasury_id);
    }

    let records = hc
        .call_zome_default::<_, Vec<serde_json::Value>>(
            "treasury",
            "get_manager_treasuries",
            &wire::ListInput {
                id: member_did.to_string(),
                limit: Some(1),
            },
        )
        .await
        .ok()?;
    records
        .iter()
        .find_map(extract_entry::<wire::Treasury>)
        .map(|treasury| treasury.id)
}

fn current_unix_seconds() -> i64 {
    ((js_sys::Date::now() / 1000.0).round()) as i64
}

fn mock_tend_balance(member_did: &str, dao_did: &str) -> TendBalanceView {
    let mut balance = mock::tend_balance();
    balance.member_did = member_did.to_string();
    balance.dao_did = dao_did.to_string();
    balance
}

fn mock_sap_balance(member_did: &str) -> SapBalanceView {
    let mut balance = mock::sap_balance();
    balance.member_did = member_did.to_string();
    balance
}

fn mock_mycel_score(member_did: &str) -> MycelScoreView {
    let mut score = mock::mycel_score();
    score.member_did = member_did.to_string();
    score
}

fn mock_stakes(member_did: &str) -> Vec<StakeView> {
    let mut stakes = mock::stakes();
    for stake in &mut stakes {
        stake.staker_did = member_did.to_string();
    }
    stakes
}

/// Extract the entry from a Holochain Record JSON value.
fn extract_entry<T: serde::de::DeserializeOwned>(record: &serde_json::Value) -> Option<T> {
    let entry = record.get("entry")?.get("Present")?;
    serde_json::from_value(entry.clone()).ok()
}

pub fn use_finance_context() -> FinanceCtx {
    expect_context::<FinanceCtx>()
}
