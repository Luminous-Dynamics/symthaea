use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
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

/// Bridge to the SporeEngine Web Worker.
///
/// Messages use a correlation-ID protocol:
///   send:  `{ id, action, ...params }`
///   recv:  `{ id, type: "response"|"error", result|error }`
#[derive(Clone)]
pub struct EngineWorker {
    worker: Arc<SendWrapper<Option<Worker>>>,
    pending: Arc<Mutex<HashMap<u32, ResolverFn>>>,
}

impl EngineWorker {
    pub fn new() -> Self {
        let pending: Arc<Mutex<HashMap<u32, ResolverFn>>> = Arc::new(Mutex::new(HashMap::new()));

        // Attempt to create the worker; gracefully degrade if the
        // spore-worker.js asset is not present (e.g. during dev builds).
        let worker = Worker::new("./assets/spore-worker.js").ok();

        if let Some(ref w) = worker {
            // Set up the onmessage handler to resolve pending promises.
            let pending_clone = pending.clone();
            let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
                let data = e.data();

                // Extract correlation ID
                let id = js_sys::Reflect::get(&data, &"id".into())
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as u32);

                if let Some(id) = id {
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
                }
                // Messages without an id (e.g. cycle broadcasts, battery_progress)
                // are fire-and-forget and intentionally ignored here.
            }) as Box<dyn FnMut(MessageEvent)>);

            w.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            onmessage.forget(); // Leak — lives for app lifetime
        } else {
            log::warn!("SporeEngine worker not available — running in UI-only mode");
        }

        Self {
            worker: Arc::new(SendWrapper(worker)),
            pending,
        }
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
