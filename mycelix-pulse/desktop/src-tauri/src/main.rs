// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tauri v2 desktop shell for Mycelix Pulse.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use serde::Serialize;
use tauri::{
    menu::{Menu, MenuItem, Submenu},
    AppHandle, Manager, WebviewWindow,
};

// ==================== COMMANDS ====================

#[derive(Serialize)]
struct DesktopRuntimeInfo<'a> {
    product_name: &'a str,
    frontend: &'a str,
    bundle_identifier: &'a str,
    shell: &'a str,
}

#[tauri::command]
fn desktop_runtime_info() -> DesktopRuntimeInfo<'static> {
    DesktopRuntimeInfo {
        product_name: "Mycelix Pulse",
        frontend: "apps/leptos",
        bundle_identifier: "com.mycelix.mail",
        shell: "tauri-v2",
    }
}

#[tauri::command]
fn open_route(app: AppHandle, path: String) -> Result<(), String> {
    let window = app.get_webview_window("main")
        .ok_or_else(|| "main window unavailable".to_string())?;
    navigate_spa(&window, &path)?;
    focus_main_window(&app);
    Ok(())
}

#[tauri::command]
fn focus_window(app: AppHandle) {
    focus_main_window(&app);
}

#[tauri::command]
fn conductor_status() -> Result<String, String> {
    match std::net::TcpStream::connect_timeout(
        &"127.0.0.1:8888".parse().unwrap(),
        std::time::Duration::from_secs(2),
    ) {
        Ok(_) => Ok("connected".to_string()),
        Err(_) => Ok("disconnected".to_string()),
    }
}

// ==================== HELPERS ====================

fn focus_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.set_zoom(1.0);
        normalize_linux_webview(&window);
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[cfg(target_os = "linux")]
fn normalize_linux_webview(window: &WebviewWindow) {
    let _ = window.with_webview(|webview| {
        use webkit2gtk::{SettingsExt, WebViewExt};
        let inner = webview.inner();
        if let Some(settings) = inner.settings() {
            settings.set_default_font_size(16);
            settings.set_default_monospace_font_size(13);
        }
        inner.set_zoom_level(1.0);
    });
}

#[cfg(not(target_os = "linux"))]
fn normalize_linux_webview(_window: &WebviewWindow) {}

fn navigate_spa(window: &WebviewWindow, path: &str) -> Result<(), String> {
    let normalized = if path.starts_with('/') { path.to_string() } else { format!("/{path}") };
    let script = format!(
        "window.history.pushState({{}}, '', {}); window.dispatchEvent(new PopStateEvent('popstate'));",
        serde_json::to_string(&normalized).map_err(|e| e.to_string())?
    );
    window.eval(&script).map_err(|e| e.to_string())
}

// ==================== PAGE LOAD JS ====================

const NORMALIZE_PAGE_RENDERING_JS: &str = r#"
(() => {
  try {
    window.__MYCELIX_DESKTOP = true;

    // Desktop-specific CSS — fixes webkit2gtk viewport bug
    // webkit2gtk miscalculates 'inset: 0' on fixed-position elements.
    // Override all fixed overlays to use explicit 100vw/100vh dimensions.
    const fix = document.createElement('style');
    fix.id = '__mycelix_desktop_fix';
    fix.textContent = `
      .bottom-nav { display: none !important; }
      .navbar { display: none !important; }
      .app-layout { height: 100vh !important; width: 100vw !important; }
      /* Force all fixed overlays to full viewport */
      .welcome-overlay, .conductor-overlay, .cmd-palette-container,
      .keyboard-overlay, .pwa-overlay, .tour-overlay {
        position: fixed !important;
        left: 0 !important; top: 0 !important;
        width: 100vw !important; height: 100vh !important;
        right: auto !important; bottom: auto !important;
      }
      /* Ensure modals inside overlays are properly sized */
      .welcome-modal {
        width: 520px !important;
        max-width: 90vw !important;
        min-width: 320px !important;
      }
      /* Force sidebar to exact width */
      .sidebar {
        width: 240px !important; min-width: 240px !important;
        max-width: 240px !important; flex: 0 0 240px !important;
      }
      .main-content {
        flex: 1 1 0% !important; min-width: 0 !important;
      }
    `;
    document.head.appendChild(fix);
  } catch (_) {}
})();
"#;

// ==================== MENU ====================

fn build_menu(app: &AppHandle) -> Result<Menu<tauri::Wry>, tauri::Error> {
    let file_menu = Submenu::with_items(app, "File", true, &[
        &MenuItem::with_id(app, "compose", "New Message", true, Some("CmdOrCtrl+N"))?,
        &MenuItem::with_id(app, "quit", "Quit", true, Some("CmdOrCtrl+Q"))?,
    ])?;
    let view_menu = Submenu::with_items(app, "View", true, &[
        &MenuItem::with_id(app, "inbox", "Inbox", true, Some("CmdOrCtrl+1"))?,
        &MenuItem::with_id(app, "sent", "Sent", true, Some("CmdOrCtrl+2"))?,
        &MenuItem::with_id(app, "drafts", "Drafts", true, Some("CmdOrCtrl+3"))?,
        &MenuItem::with_id(app, "settings", "Settings", true, Some("CmdOrCtrl+,"))?,
    ])?;
    let help_menu = Submenu::with_items(app, "Help", true, &[
        &MenuItem::with_id(app, "about", "About Mycelix Pulse", true, None::<&str>)?,
    ])?;
    Menu::with_items(app, &[&file_menu, &view_menu, &help_menu])
}

// ==================== MAIN ====================

fn main() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_deep_link::init())
        .invoke_handler(tauri::generate_handler![
            desktop_runtime_info,
            open_route,
            focus_window,
            conductor_status,
        ])
        .on_page_load(|webview, _payload| {
            let _ = webview.set_zoom(1.0);
            let _ = webview.eval(NORMALIZE_PAGE_RENDERING_JS);
        })
        .setup(|app| {
            // Menu bar
            if let Ok(menu) = build_menu(&app.handle().clone()) {
                let _ = app.set_menu(menu);
            }

            // WebKitGTK on Wayland: webview doesn't fill window on initial load.
            // Fix: force pixel widths, override viewport, and toggle window size.
            if let Some(window) = app.get_webview_window("main") {
                let w = window.clone();
                std::thread::spawn(move || {
                    // Get the actual window pixel size
                    let win_width = w.inner_size().map(|s| s.width).unwrap_or(1280);

                    std::thread::sleep(std::time::Duration::from_millis(500));
                    // Force viewport and all containers to actual window width
                    let _ = w.eval(&format!(
                        "document.querySelector('meta[name=viewport]')?.setAttribute('content','width={win_width},initial-scale=1');\
                         document.documentElement.style.width='{win_width}px';\
                         document.body.style.width='{win_width}px';\
                         document.body.style.margin='0';\
                         document.body.style.overflow='hidden auto';\
                         window.dispatchEvent(new Event('resize'));"
                    ));
                    // Toggle size to force WebKitGTK relayout
                    if let Ok(size) = w.inner_size() {
                        let _ = w.set_size(tauri::Size::Physical(tauri::PhysicalSize::new(
                            size.width.saturating_sub(1), size.height,
                        )));
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        let _ = w.set_size(tauri::Size::Physical(tauri::PhysicalSize::new(
                            size.width, size.height,
                        )));
                    }
                    // Repeat after WASM loads
                    std::thread::sleep(std::time::Duration::from_millis(3000));
                    let _ = w.eval(&format!(
                        "document.querySelector('meta[name=viewport]')?.setAttribute('content','width={win_width},initial-scale=1');\
                         document.documentElement.style.width='{win_width}px';\
                         document.body.style.width='{win_width}px';\
                         document.querySelectorAll('.welcome-overlay,.welcome-modal').forEach(e=>e.style.width='{win_width}px');\
                         window.dispatchEvent(new Event('resize'));"
                    ));
                    if let Ok(size) = w.inner_size() {
                        let _ = w.set_size(tauri::Size::Physical(tauri::PhysicalSize::new(
                            size.width.saturating_sub(1), size.height,
                        )));
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        let _ = w.set_size(tauri::Size::Physical(tauri::PhysicalSize::new(
                            size.width, size.height,
                        )));
                    }
                });
            }

            focus_main_window(&app.handle().clone());
            Ok(())
        })
        .on_menu_event(|app, event| {
            match event.id().as_ref() {
                "compose" => { if let Some(w) = app.get_webview_window("main") { let _ = navigate_spa(&w, "/compose"); } }
                "inbox" => { if let Some(w) = app.get_webview_window("main") { let _ = navigate_spa(&w, "/"); } }
                "sent" => { if let Some(w) = app.get_webview_window("main") { let _ = navigate_spa(&w, "/sent"); } }
                "drafts" => { if let Some(w) = app.get_webview_window("main") { let _ = navigate_spa(&w, "/drafts"); } }
                "settings" => { if let Some(w) = app.get_webview_window("main") { let _ = navigate_spa(&w, "/settings"); } }
                "about" => { if let Some(w) = app.get_webview_window("main") { let _ = w.eval("alert('Mycelix Pulse v0.1.0\\nDecentralized encrypted communication on Holochain')"); } }
                "quit" => { app.exit(0); }
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            eprintln!("FATAL: Mycelix Pulse failed: {e}");
            std::process::exit(1);
        });
}
