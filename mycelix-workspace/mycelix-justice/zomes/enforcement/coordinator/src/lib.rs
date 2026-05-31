// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Enforcement Coordinator Zome
use enforcement_integrity::*;
use hdk::prelude::*;

#[hdk_extern]
pub fn create_enforcement_order(input: CreateEnforcementInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let order = EnforcementOrder {
        id: format!("enforcement:{}:{}", input.judgment_id, now.as_micros()),
        judgment_id: input.judgment_id.clone(),
        target: input.target.clone(),
        enforcement_type: input.enforcement_type,
        status: EnforcementStatus::Pending,
        issued: now,
        executed: None,
    };
    let action_hash = create_entry(&EntryTypes::EnforcementOrder(order))?;
    create_link(
        hash_entry(&input.judgment_id)?,
        action_hash.clone(),
        LinkTypes::JudgmentToEnforcement,
        (),
    )?;
    create_link(
        hash_entry(&input.target)?,
        action_hash.clone(),
        LinkTypes::TargetToEnforcement,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateEnforcementInput {
    pub judgment_id: String,
    pub target: String,
    pub enforcement_type: EnforcementType,
}

#[hdk_extern]
pub fn execute_enforcement(order_id: String) -> ExternResult<Record> {
    // In production: execute cross-hApp enforcement actions
    // For now: find and update the order status
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::EnforcementOrder,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(order) = record
            .entry()
            .to_app_option::<EnforcementOrder>()
            .ok()
            .flatten()
        {
            if order.id == order_id {
                let now = sys_time()?;
                let updated = EnforcementOrder {
                    status: EnforcementStatus::Executed,
                    executed: Some(now),
                    ..order
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::EnforcementOrder(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Order not found".into())))
}

/// Get an enforcement order by ID
#[hdk_extern]
pub fn get_enforcement_order(order_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::EnforcementOrder,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(order) = record
            .entry()
            .to_app_option::<EnforcementOrder>()
            .ok()
            .flatten()
        {
            if order.id == order_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all enforcement orders by status
#[hdk_extern]
pub fn get_orders_by_status(status: EnforcementStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::EnforcementOrder,
        )?))
        .include_entries(true);

    let mut orders = Vec::new();
    for record in query(filter)? {
        if let Some(order) = record
            .entry()
            .to_app_option::<EnforcementOrder>()
            .ok()
            .flatten()
        {
            if order.status == status {
                orders.push(record);
            }
        }
    }
    Ok(orders)
}

/// Get all enforcement orders targeting a specific party
#[hdk_extern]
pub fn get_orders_for_target(target: String) -> ExternResult<Vec<Record>> {
    let mut orders = Vec::new();
    for link in get_links(
        GetLinksInputBuilder::try_new(hash_entry(&target)?, LinkTypes::TargetToEnforcement)?
            .build(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            orders.push(record);
        }
    }
    Ok(orders)
}

/// Appeal an enforcement order
#[hdk_extern]
pub fn appeal_enforcement(input: AppealEnforcementInput) -> ExternResult<Record> {
    let appeal = EnforcementAppeal {
        id: format!("appeal:{}:{}", input.order_id, sys_time()?.as_micros()),
        order_id: input.order_id.clone(),
        appellant: input.appellant.clone(),
        reason: input.reason,
        filed: sys_time()?,
        status: AppealStatus::Pending,
    };
    let action_hash = create_entry(&EntryTypes::EnforcementAppeal(appeal))?;
    create_link(
        hash_entry(&input.order_id)?,
        action_hash.clone(),
        LinkTypes::OrderToAppeal,
        (),
    )?;

    // Update original order status to Appealed
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::EnforcementOrder,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(order) = record
            .entry()
            .to_app_option::<EnforcementOrder>()
            .ok()
            .flatten()
        {
            if order.id == input.order_id {
                let updated = EnforcementOrder {
                    status: EnforcementStatus::Appealed,
                    ..order
                };
                update_entry(
                    record.action_address().clone(),
                    &EntryTypes::EnforcementOrder(updated),
                )?;
                break;
            }
        }
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AppealEnforcementInput {
    pub order_id: String,
    pub appellant: String,
    pub reason: String,
}

/// Suspend an enforcement order (by authorized party)
#[hdk_extern]
pub fn suspend_enforcement(order_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::EnforcementOrder,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(order) = record
            .entry()
            .to_app_option::<EnforcementOrder>()
            .ok()
            .flatten()
        {
            if order.id == order_id {
                let updated = EnforcementOrder {
                    status: EnforcementStatus::Suspended,
                    ..order
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::EnforcementOrder(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Order not found".into())))
}

/// Get appeals for an enforcement order
#[hdk_extern]
pub fn get_order_appeals(order_id: String) -> ExternResult<Vec<Record>> {
    let mut appeals = Vec::new();
    for link in get_links(
        GetLinksInputBuilder::try_new(hash_entry(&order_id)?, LinkTypes::OrderToAppeal)?.build(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            appeals.push(record);
        }
    }
    Ok(appeals)
}
