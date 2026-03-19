use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use web_sys::Worker;

/// Bridge to the SporeEngine Web Worker.
///
/// For now this is a placeholder that creates the worker handle.
/// The full async message-passing implementation (request/response with
/// correlation IDs) will be wired in a follow-up pass once the UI
/// compiles and renders.
///
/// Note: `web_sys::Worker` is `!Send + !Sync`, so we store it behind
/// a `SendWrapper` to satisfy Leptos's context requirements. This is
/// safe because WASM is single-threaded.
#[derive(Clone)]
pub struct EngineWorker {
    #[allow(dead_code)]
    worker: Arc<SendWrapper<Option<Worker>>>,
    #[allow(dead_code)]
    next_id: Arc<Mutex<u32>>,
    #[allow(dead_code)]
    pending: Arc<Mutex<HashMap<u32, ()>>>,
}

/// Wrapper that implements Send + Sync for single-threaded WASM contexts.
/// SAFETY: WASM targets are inherently single-threaded; there is no
/// concurrent access to the inner value.
struct SendWrapper<T>(T);

unsafe impl<T> Send for SendWrapper<T> {}
unsafe impl<T> Sync for SendWrapper<T> {}

impl EngineWorker {
    pub fn new() -> Self {
        // Attempt to create the worker; gracefully degrade if the
        // spore-worker.js asset is not present (e.g. during dev builds).
        let worker = Worker::new("./assets/spore-worker.js").ok();

        if worker.is_none() {
            log::warn!("SporeEngine worker not available — running in UI-only mode");
        }

        Self {
            worker: Arc::new(SendWrapper(worker)),
            next_id: Arc::new(Mutex::new(0)),
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
