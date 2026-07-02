// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IndexedDB-backed curriculum graph cache.
//!
//! Stores subject chunks in the browser's IndexedDB for offline access.
//! The curriculum graph is built incrementally: a core graph is always
//! available (embedded in WASM), and additional subjects are fetched
//! and cached on demand.
//!
//! # Architecture
//!
//! ```text
//! WASM (always available)     IndexedDB (cached)        Network (on demand)
//! ┌─────────────────────┐    ┌──────────────────┐     ┌─────────────────┐
//! │ Core graph (403     │    │ Previously loaded │     │ /curriculum/    │
//! │ nodes: AI, Climate, │ +  │ subject chunks    │  +  │ chunks/{slug}   │
//! │ CS, Math partial)   │    │ (persisted)       │     │ .json           │
//! └─────────────────────┘    └──────────────────┘     └─────────────────┘
//! ```

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbObjectStore, IdbRequest, IdbTransaction, IdbTransactionMode};
use std::cell::RefCell;
use std::rc::Rc;

const DB_NAME: &str = "praxis_curriculum";
const DB_VERSION: u32 = 1;
const STORE_NAME: &str = "subject_chunks";

/// Open (or create) the IndexedDB for curriculum caching.
pub async fn open_db() -> Result<IdbDatabase, String> {
    let window = web_sys::window().ok_or("no window")?;
    let idb_factory = window
        .indexed_db()
        .map_err(|_| "indexedDB not available")?
        .ok_or("indexedDB is None")?;

    let open_req = idb_factory
        .open_with_u32(DB_NAME, DB_VERSION)
        .map_err(|e| format!("open error: {:?}", e))?;

    // Handle upgrade (first time or version bump)
    let open_req_clone = open_req.clone();
    let on_upgrade = Closure::once(move |_event: web_sys::Event| {
        if let Ok(result) = open_req_clone.result() {
            let db: IdbDatabase = result.unchecked_into();
            if !db.object_store_names().contains(STORE_NAME) {
                let _ = db.create_object_store(STORE_NAME);
            }
        }
    });
    open_req.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
    on_upgrade.forget(); // Must keep alive

    // Wait for open to complete
    let result = JsFuture::from(idb_request_to_promise(&open_req))
        .await
        .map_err(|e| format!("open failed: {:?}", e))?;

    Ok(result.unchecked_into())
}

/// Save a subject chunk JSON string to IndexedDB.
pub async fn save_chunk(db: &IdbDatabase, slug: &str, json: &str) -> Result<(), String> {
    let tx = db
        .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
        .map_err(|e| format!("tx error: {:?}", e))?;
    let store = tx
        .object_store(STORE_NAME)
        .map_err(|e| format!("store error: {:?}", e))?;

    let key = JsValue::from_str(slug);
    let value = JsValue::from_str(json);
    store
        .put_with_key(&value, &key)
        .map_err(|e| format!("put error: {:?}", e))?;

    // Wait for transaction to complete
    let _ = JsFuture::from(idb_transaction_to_promise(&tx)).await;
    Ok(())
}

/// Load a cached subject chunk from IndexedDB.
pub async fn load_chunk(db: &IdbDatabase, slug: &str) -> Result<Option<String>, String> {
    let tx = db
        .transaction_with_str(STORE_NAME)
        .map_err(|e| format!("tx error: {:?}", e))?;
    let store = tx
        .object_store(STORE_NAME)
        .map_err(|e| format!("store error: {:?}", e))?;

    let key = JsValue::from_str(slug);
    let req = store
        .get(&key)
        .map_err(|e| format!("get error: {:?}", e))?;

    let result = JsFuture::from(idb_request_to_promise(&req))
        .await
        .map_err(|e| format!("get failed: {:?}", e))?;

    if result.is_undefined() || result.is_null() {
        Ok(None)
    } else {
        Ok(result.as_string())
    }
}

/// List all cached subject slugs.
pub async fn list_cached(db: &IdbDatabase) -> Result<Vec<String>, String> {
    let tx = db
        .transaction_with_str(STORE_NAME)
        .map_err(|e| format!("tx error: {:?}", e))?;
    let store = tx
        .object_store(STORE_NAME)
        .map_err(|e| format!("store error: {:?}", e))?;

    let req = store
        .get_all_keys()
        .map_err(|e| format!("keys error: {:?}", e))?;

    let result = JsFuture::from(idb_request_to_promise(&req))
        .await
        .map_err(|e| format!("keys failed: {:?}", e))?;

    let array: js_sys::Array = result.unchecked_into();
    let keys: Vec<String> = array
        .iter()
        .filter_map(|v| v.as_string())
        .collect();

    Ok(keys)
}

/// Fetch a subject chunk from the network.
pub async fn fetch_chunk(slug: &str) -> Result<String, String> {
    let url = format!("/curriculum/chunks/{}.json", slug);
    let window = web_sys::window().ok_or("no window")?;
    let resp = JsFuture::from(window.fetch_with_str(&url))
        .await
        .map_err(|e| format!("fetch error: {:?}", e))?;

    let resp: web_sys::Response = resp.unchecked_into();
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let text = JsFuture::from(
        resp.text().map_err(|_| "text() failed".to_string())?,
    )
    .await
    .map_err(|e| format!("text error: {:?}", e))?;

    text.as_string().ok_or("not a string".to_string())
}

/// Fetch a chunk, cache it in IndexedDB, return the JSON string.
pub async fn fetch_and_cache(slug: &str) -> Result<String, String> {
    let json = fetch_chunk(slug).await?;

    // Cache in background (don't block on IndexedDB write)
    let slug_owned = slug.to_string();
    let json_clone = json.clone();
    wasm_bindgen_futures::spawn_local(async move {
        if let Ok(db) = open_db().await {
            let _ = save_chunk(&db, &slug_owned, &json_clone).await;
        }
    });

    Ok(json)
}

/// Load a chunk: try IndexedDB first, then network, cache on success.
pub async fn load_or_fetch(slug: &str) -> Result<String, String> {
    // Try IndexedDB cache first
    if let Ok(db) = open_db().await {
        if let Ok(Some(cached)) = load_chunk(&db, slug).await {
            return Ok(cached);
        }
    }

    // Fall back to network
    fetch_and_cache(slug).await
}

// ============================================================
// Helpers: convert IDB requests to promises
// ============================================================

fn idb_request_to_promise(req: &IdbRequest) -> js_sys::Promise {
    let req = req.clone();
    let cb = Rc::new(RefCell::new(None::<(Closure<dyn FnMut(web_sys::Event)>, Closure<dyn FnMut(web_sys::Event)>)>));

    let promise = js_sys::Promise::new(&mut {
        let req = req.clone();
        let cb = cb.clone();
        move |resolve, reject| {
            let req_inner = req.clone();
            let on_success = Closure::wrap(Box::new(move |_: web_sys::Event| {
                let result = req_inner.result().unwrap_or(JsValue::UNDEFINED);
                let _ = resolve.call1(&JsValue::NULL, &result);
            }) as Box<dyn FnMut(web_sys::Event)>);

            let on_error = Closure::wrap(Box::new(move |_: web_sys::Event| {
                let _ = reject.call1(&JsValue::NULL, &JsValue::from_str("IDB request failed"));
            }) as Box<dyn FnMut(web_sys::Event)>);

            req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
            req.set_onerror(Some(on_error.as_ref().unchecked_ref()));

            *cb.borrow_mut() = Some((on_success, on_error));
        }
    });

    // Leak the closures (they're one-shot IDB callbacks)
    if let Some((s, e)) = cb.borrow_mut().take() {
        s.forget();
        e.forget();
    }

    promise
}

fn idb_transaction_to_promise(tx: &IdbTransaction) -> js_sys::Promise {
    let tx = tx.clone();
    js_sys::Promise::new(&mut |resolve, reject| {
        let resolve_clone = resolve.clone();
        let reject_clone = reject.clone();

        let on_complete = Closure::once(move |_: web_sys::Event| {
            let _ = resolve_clone.call0(&JsValue::NULL);
        });

        let on_error = Closure::once(move |_: web_sys::Event| {
            let _ = reject_clone.call1(&JsValue::NULL, &JsValue::from_str("IDB transaction failed"));
        });

        tx.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
        tx.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        on_complete.forget();
        on_error.forget();
    })
}
