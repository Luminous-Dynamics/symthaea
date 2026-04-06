// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use commons_leptos_types::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, ToastKind};
use crate::contexts::commons_context::use_commons;

#[derive(Clone)]
pub struct CommonsActions;

pub fn provide_commons_actions() { provide_context(CommonsActions); }

pub fn post_need(title: String, description: String, category: NeedCategory, urgency: Urgency) {
    let hc = use_holochain();
    let commons = use_commons();
    let toasts = use_toasts();
    if hc.is_mock() {
        let id = format!("need-{:03}", commons.needs.get_untracked().len() + 1);
        commons.needs.update(|needs| {
            needs.push(NeedView {
                hash: id.clone(), id: id.clone(), title, description, category, urgency,
                requester_did: commons.my_agent_did.get_untracked(),
                status: NeedStatus::Open,
                created: (js_sys::Date::now() * 1000.0) as i64,
            });
        });
        toasts.push("your need has been shared with the commons", ToastKind::Custom("commons".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct CreateNeedInput { title: String, description: String, category: String, urgency: String }
            let input = CreateNeedInput { title, description, category: format!("{:?}", category), urgency: format!("{:?}", urgency) };
            match hc.call_zome::<_, serde_json::Value>("commons_care", "mutualaid-needs", "create_need", &input).await {
                Ok(_) => toasts.push("your need has been shared with the commons", ToastKind::Custom("commons".into())),
                Err(e) => { web_sys::console::log_1(&format!("create need failed: {e}").into()); }
            }
        });
    }
}

pub fn post_offer(title: String, description: String, category: NeedCategory) {
    let hc = use_holochain();
    let commons = use_commons();
    let toasts = use_toasts();
    if hc.is_mock() {
        let id = format!("offer-{:03}", commons.offers.get_untracked().len() + 1);
        commons.offers.update(|offers| {
            offers.push(OfferView {
                hash: id.clone(), id: id.clone(), title, description, category,
                offerer_did: commons.my_agent_did.get_untracked(),
                created: (js_sys::Date::now() * 1000.0) as i64,
            });
        });
        toasts.push("your offer has been planted in the commons", ToastKind::Custom("commons".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct CreateOfferInput { title: String, description: String, category: String }
            let input = CreateOfferInput { title, description, category: format!("{:?}", category) };
            match hc.call_zome::<_, serde_json::Value>("commons_care", "mutualaid-needs", "create_offer", &input).await {
                Ok(_) => toasts.push("your offer has been planted", ToastKind::Custom("commons".into())),
                Err(e) => { web_sys::console::log_1(&format!("create offer failed: {e}").into()); }
            }
        });
    }
}
