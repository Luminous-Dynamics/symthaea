/// Register the service worker for asset caching.
pub fn register() {
    if let Some(window) = web_sys::window() {
        if let Ok(sw) = window.navigator().service_worker() {
            let _ = sw.register("./assets/sw.js");
            log::info!("Service worker registered");
        }
    }
}
