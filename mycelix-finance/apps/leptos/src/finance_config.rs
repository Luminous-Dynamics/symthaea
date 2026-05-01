// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finance runtime configuration — discovered from conductor or local defaults.

use leptos::prelude::*;
use finance_wire_types::FinanceRuntimeDiscovery;

/// Runtime configuration for the finance cluster.
#[derive(Debug, Clone)]
pub struct FinanceRuntimeConfig {
    pub member_did: String,
    pub dao_did: Option<String>,
    pub treasury_id: Option<String>,
    pub commons_pool_id: Option<String>,
}

impl Default for FinanceRuntimeConfig {
    fn default() -> Self {
        Self {
            member_did: String::new(),
            dao_did: None,
            treasury_id: None,
            commons_pool_id: None,
        }
    }
}

/// Merge runtime discovery from conductor with local config.
pub fn merge_runtime_discovery(
    discovery: &FinanceRuntimeDiscovery,
    local: &FinanceRuntimeConfig,
) -> FinanceRuntimeConfig {
    FinanceRuntimeConfig {
        member_did: if discovery.member_did.is_empty() {
            local.member_did.clone()
        } else {
            discovery.member_did.clone()
        },
        dao_did: discovery
            .default_dao_did
            .clone()
            .or_else(|| local.dao_did.clone()),
        treasury_id: discovery
            .treasury_id
            .clone()
            .or_else(|| local.treasury_id.clone()),
        commons_pool_id: discovery
            .commons_pool_id
            .clone()
            .or_else(|| local.commons_pool_id.clone()),
    }
}

/// Store for runtime config (Leptos context).
#[derive(Clone)]
pub struct FinanceRuntimeConfigStore {
    pub config: ReadSignal<FinanceRuntimeConfig>,
    pub set_config: WriteSignal<FinanceRuntimeConfig>,
}

/// Provide the finance runtime config context.
pub fn provide_finance_runtime_config() -> FinanceRuntimeConfigStore {
    let (config, set_config) = signal(FinanceRuntimeConfig::default());
    let store = FinanceRuntimeConfigStore { config, set_config };
    provide_context(store.clone());
    store
}

/// Retrieve finance runtime config from context.
pub fn use_finance_runtime_config() -> FinanceRuntimeConfigStore {
    expect_context::<FinanceRuntimeConfigStore>()
}
