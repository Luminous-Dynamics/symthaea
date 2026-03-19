use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, Worker};

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// Callback invoked when the worker responds to a request.
type Resolver = Box<dyn FnOnce(JsValue) + 'static>;

/// Bridge to the SporeEngine Web Worker.
///
/// Messages use a correlation-ID protocol:
///   send:  `{ id, action, ...params }`
///   recv:  `{ id, type: "response"|"error", result|error }`
///
/// Note: `web_sys::Worker` is `!Send + !Sync`, but WASM is single-threaded
/// so wrapping in `SendWrapper` is safe and satisfies Leptos context bounds.
#[derive(Clone)]
pub struct EngineWorker {
    worker: Arc<SendWrapper<Option<Worker>>>,
    pending: Arc<Mutex<HashMap<u32, Resolver>>>,
}

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

impl EngineWorker {
    pub fn new() -> Self {
        let pending: Arc<Mutex<HashMap<u32, Resolver>>> = Arc::new(Mutex::new(HashMap::new()));

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
                            // Wrap error in a JsValue with an `_error` key so callers
                            // can distinguish errors from successful results.
                            let obj = js_sys::Object::new();
                            let _ =
                                js_sys::Reflect::set(&obj, &"_error".into(), &err);
                            resolver(obj.into());
                        } else {
                            let result = js_sys::Reflect::get(&data, &"result".into())
                                .unwrap_or(JsValue::NULL);
                            resolver(result);
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
                // Return a rejected promise when no worker is available.
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
            if let Ok(keys) = js_sys::Object::keys(params.unchecked_ref())
                .dyn_into::<js_sys::Array>()
            {
                for i in 0..keys.length() {
                    let key = keys.get(i);
                    if let Ok(val) = js_sys::Reflect::get(params, &key) {
                        let _ = js_sys::Reflect::set(&msg, &key, &val);
                    }
                }
            }
        }

        // Create a JS Promise whose resolver is stored in `pending`.
        let pending = self.pending.clone();
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let mut map = pending.lock().unwrap();
            map.insert(
                id,
                Box::new(move |result: JsValue| {
                    let _ = resolve.call1(&JsValue::NULL, &result);
                }),
            );
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
