// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, Worker};

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// Wrapper that implements Send + Sync for single-threaded WASM contexts.
/// SAFETY: WASM targets are inherently single-threaded; there is no
/// concurrent access to the inner value.
struct SendWrapper<T>(T);

unsafe impl<T> Send for SendWrapper<T> {}
unsafe impl<T> Sync for SendWrapper<T> {}

impl<T> SendWrapper<T> {
    fn inner(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::Deref for SendWrapper<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// A resolver wraps a JS Function that cannot be sent across threads.
/// In WASM (single-threaded) we wrap it in SendWrapper for compiler satisfaction.
struct ResolverFn(SendWrapper<js_sys::Function>);

// SAFETY: WASM is single-threaded.
unsafe impl Send for ResolverFn {}

impl ResolverFn {
    fn new(f: js_sys::Function) -> Self {
        Self(SendWrapper(f))
    }

    fn call(&self, arg: &JsValue) {
        let _ = self.0.call1(&JsValue::NULL, arg);
    }
}

/// Broadcast handler callback (for fire-and-forget messages from the worker).
struct BroadcastHandler(SendWrapper<Box<dyn Fn(JsValue)>>);

// SAFETY: WASM is single-threaded.
unsafe impl Send for BroadcastHandler {}
unsafe impl Sync for BroadcastHandler {}

/// Bridge to the SporeEngine Web Worker.
///
/// Messages use a correlation-ID protocol:
///   send:  `{ id, action, ...params }`
///   recv:  `{ id, type: "response"|"error", result|error }`
///
/// Fire-and-forget messages (no `id`) are routed to the broadcast handler.
#[derive(Clone)]
pub struct EngineWorker {
    worker: Arc<SendWrapper<Option<Worker>>>,
    pending: Arc<Mutex<HashMap<u32, ResolverFn>>>,
    broadcast: Arc<Mutex<Option<BroadcastHandler>>>,
}

impl EngineWorker {
    pub fn new() -> Self {
        let pending: Arc<Mutex<HashMap<u32, ResolverFn>>> = Arc::new(Mutex::new(HashMap::new()));
        let broadcast: Arc<Mutex<Option<BroadcastHandler>>> = Arc::new(Mutex::new(None));

        // Attempt to create the worker as an ES module (spore-worker.js uses `import`).
        let opts = web_sys::WorkerOptions::new();
        opts.set_type(web_sys::WorkerType::Module);
        let worker = Worker::new_with_options("./assets/spore-worker.js", &opts).ok();

        if let Some(ref w) = worker {
            let pending_clone = pending.clone();
            let broadcast_clone = broadcast.clone();
            let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
                let data = e.data();

                // Extract correlation ID
                let id = js_sys::Reflect::get(&data, &"id".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as u32);

                if let Some(id) = id {
                    // Correlated response — resolve the pending promise
                    let mut map = pending_clone.lock().unwrap();
                    if let Some(resolver) = map.remove(&id) {
                        let msg_type = js_sys::Reflect::get(&data, &"type".into())
                            .ok()
                            .and_then(|v| v.as_string())
                            .unwrap_or_default();

                        if msg_type == "error" {
                            let err = js_sys::Reflect::get(&data, &"error".into())
                                .unwrap_or(JsValue::from_str("unknown worker error"));
                            let obj = js_sys::Object::new();
                            let _ = js_sys::Reflect::set(&obj, &"_error".into(), &err);
                            resolver.call(&obj.into());
                        } else {
                            let result = js_sys::Reflect::get(&data, &"result".into())
                                .unwrap_or(JsValue::NULL);
                            resolver.call(&result);
                        }
                    }
                } else {
                    // Fire-and-forget broadcast (cycle updates, battery_progress, etc.)
                    let handler = broadcast_clone.lock().unwrap();
                    if let Some(ref h) = *handler {
                        (h.0.0)(data);
                    }
                }
            }) as Box<dyn FnMut(MessageEvent)>);

            w.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            onmessage.forget(); // Leak — lives for app lifetime

            // Error handler — catches WASM traps and worker crashes
            let onerror = Closure::wrap(Box::new(move |e: web_sys::ErrorEvent| {
                log::error!(
                    "Worker error: {} ({}:{})",
                    e.message(),
                    e.filename(),
                    e.lineno()
                );
            }) as Box<dyn FnMut(web_sys::ErrorEvent)>);
            w.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onerror.forget();
        } else {
            log::warn!("SporeEngine worker not available — running in UI-only mode");
        }

        Self {
            worker: Arc::new(SendWrapper(worker)),
            pending,
            broadcast,
        }
    }

    /// Set the broadcast handler for fire-and-forget messages from the worker.
    /// Called once after AppState is available.
    pub fn set_broadcast_handler(&self, handler: impl Fn(JsValue) + 'static) {
        let mut guard = self.broadcast.lock().unwrap();
        *guard = Some(BroadcastHandler(SendWrapper(Box::new(handler))));
    }

    /// Returns `true` if the underlying Web Worker was created successfully.
    pub fn is_available(&self) -> bool {
        self.worker.inner().is_some()
    }

    /// Send a message to the worker and get a `js_sys::Promise` for the
    /// response.  The promise resolves with the `result` field from the
    /// worker's response message.
    pub fn send(&self, action: &str, params: &JsValue) -> js_sys::Promise {
        let worker = match self.worker.inner() {
            Some(w) => w,
            None => {
                return js_sys::Promise::reject(&JsValue::from_str(
                    "SporeEngine worker not available",
                ));
            }
        };

        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        // Build message: { id, action, ...params }
        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&msg, &"id".into(), &JsValue::from(id));
        let _ = js_sys::Reflect::set(&msg, &"action".into(), &JsValue::from(action));

        // Spread params into msg
        if params.is_object() && !params.is_null() && !params.is_undefined() {
            let obj_ref: &js_sys::Object = params.unchecked_ref();
            let keys = js_sys::Object::keys(obj_ref);
            for i in 0..keys.length() {
                let key = keys.get(i);
                if let Ok(val) = js_sys::Reflect::get(params, &key) {
                    let _ = js_sys::Reflect::set(&msg, &key, &val);
                }
            }
        }

        // Create a JS Promise whose resolver is stored in `pending`.
        let pending = self.pending.clone();
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let mut map = pending.lock().unwrap();
            map.insert(id, ResolverFn::new(resolve));
        });

        if let Err(e) = worker.post_message(&msg) {
            log::error!("Failed to post message to worker: {:?}", e);
        }

        promise
    }

    /// Convenience: send an action with no extra params.
    pub fn send_simple(&self, action: &str) -> js_sys::Promise {
        self.send(action, &JsValue::NULL)
    }
}
