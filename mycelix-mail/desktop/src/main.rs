//! Mycelix Mail Desktop Application
//!
//! Tauri-based desktop application providing native features:
//! - System tray integration
//! - Native notifications
//! - Offline support
//! - Deep OS integration

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use tauri::{
    CustomMenuItem, Manager, SystemTray, SystemTrayEvent, SystemTrayMenu,
    SystemTrayMenuItem, Window, WindowEvent,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// State Management
// ============================================================================

#[derive(Default)]
struct AppState {
    unread_count: Mutex<u32>,
    is_connected: Mutex<bool>,
    last_sync: Mutex<Option<String>>,
}

// ============================================================================
// Commands
// ============================================================================

#[derive(Serialize, Deserialize)]
struct EmailNotification {
    id: String,
    from: String,
    subject: String,
    preview: String,
    timestamp: String,
}

#[derive(Serialize, Deserialize)]
struct SyncStatus {
    is_syncing: bool,
    last_sync: Option<String>,
    pending_changes: u32,
}

/// Update unread count and badge
#[tauri::command]
fn update_unread_count(count: u32, state: tauri::State<AppState>, app: tauri::AppHandle) {
    *state.unread_count.lock().unwrap() = count;

    // Update dock badge on macOS
    #[cfg(target_os = "macos")]
    {
        if count > 0 {
            app.tray_handle().set_title(&count.to_string()).ok();
        } else {
            app.tray_handle().set_title("").ok();
        }
    }

    // Update tray tooltip
    let tooltip = if count > 0 {
        format!("Mycelix Mail - {} unread", count)
    } else {
        "Mycelix Mail".to_string()
    };
    app.tray_handle().set_tooltip(&tooltip).ok();
}

/// Show native notification for new email
#[tauri::command]
fn show_email_notification(notification: EmailNotification, app: tauri::AppHandle) {
    tauri::api::notification::Notification::new(&app.config().tauri.bundle.identifier)
        .title(&format!("New email from {}", notification.from))
        .body(&notification.subject)
        .show()
        .ok();
}

/// Check if running in background
#[tauri::command]
fn is_background_mode(window: Window) -> bool {
    !window.is_visible().unwrap_or(true)
}

/// Get sync status
#[tauri::command]
fn get_sync_status(state: tauri::State<AppState>) -> SyncStatus {
    SyncStatus {
        is_syncing: false,
        last_sync: state.last_sync.lock().unwrap().clone(),
        pending_changes: 0,
    }
}

/// Update connection status
#[tauri::command]
fn set_connection_status(connected: bool, state: tauri::State<AppState>) {
    *state.is_connected.lock().unwrap() = connected;
}

/// Trigger background sync
#[tauri::command]
async fn trigger_sync(state: tauri::State<'_, AppState>) -> Result<(), String> {
    // Update last sync time
    let now = chrono::Utc::now().to_rfc3339();
    *state.last_sync.lock().unwrap() = Some(now);
    Ok(())
}

/// Get platform info
#[tauri::command]
fn get_platform_info() -> serde_json::Value {
    serde_json::json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "family": std::env::consts::FAMILY,
    })
}

/// Open compose window
#[tauri::command]
fn open_compose_window(app: tauri::AppHandle, to: Option<String>, subject: Option<String>) {
    let url = format!(
        "/compose?to={}&subject={}",
        to.unwrap_or_default(),
        subject.unwrap_or_default()
    );

    if let Some(window) = app.get_window("main") {
        window.eval(&format!("window.location.href = '{}'", url)).ok();
        window.show().ok();
        window.set_focus().ok();
    }
}

/// Set window always on top
#[tauri::command]
fn set_always_on_top(window: Window, always_on_top: bool) {
    window.set_always_on_top(always_on_top).ok();
}

/// Minimize to tray
#[tauri::command]
fn minimize_to_tray(window: Window) {
    window.hide().ok();
}

// ============================================================================
// System Tray
// ============================================================================

fn create_system_tray() -> SystemTray {
    let compose = CustomMenuItem::new("compose", "Compose New Email");
    let inbox = CustomMenuItem::new("inbox", "Go to Inbox");
    let sync = CustomMenuItem::new("sync", "Sync Now");
    let separator = SystemTrayMenuItem::Separator;
    let settings = CustomMenuItem::new("settings", "Settings");
    let quit = CustomMenuItem::new("quit", "Quit Mycelix Mail");

    let menu = SystemTrayMenu::new()
        .add_item(compose)
        .add_item(inbox)
        .add_native_item(separator.clone())
        .add_item(sync)
        .add_native_item(separator)
        .add_item(settings)
        .add_item(quit);

    SystemTray::new().with_menu(menu)
}

fn handle_system_tray_event(app: &tauri::AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick { .. } => {
            if let Some(window) = app.get_window("main") {
                if window.is_visible().unwrap_or(false) {
                    window.hide().ok();
                } else {
                    window.show().ok();
                    window.set_focus().ok();
                }
            }
        }
        SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
            "compose" => {
                open_compose_window(app.clone(), None, None);
            }
            "inbox" => {
                if let Some(window) = app.get_window("main") {
                    window.eval("window.location.href = '/inbox'").ok();
                    window.show().ok();
                    window.set_focus().ok();
                }
            }
            "sync" => {
                if let Some(window) = app.get_window("main") {
                    window.emit("sync-requested", ()).ok();
                }
            }
            "settings" => {
                if let Some(window) = app.get_window("main") {
                    window.eval("window.location.href = '/settings'").ok();
                    window.show().ok();
                    window.set_focus().ok();
                }
            }
            "quit" => {
                std::process::exit(0);
            }
            _ => {}
        },
        _ => {}
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .system_tray(create_system_tray())
        .on_system_tray_event(handle_system_tray_event)
        .on_window_event(|event| {
            if let WindowEvent::CloseRequested { api, .. } = event.event() {
                // Minimize to tray instead of closing
                event.window().hide().ok();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            update_unread_count,
            show_email_notification,
            is_background_mode,
            get_sync_status,
            set_connection_status,
            trigger_sync,
            get_platform_info,
            open_compose_window,
            set_always_on_top,
            minimize_to_tray,
        ])
        .setup(|app| {
            // Register global shortcuts
            #[cfg(desktop)]
            {
                use tauri::GlobalShortcutManager;

                let app_handle = app.handle();

                // Global compose shortcut
                app.global_shortcut_manager()
                    .register("CmdOrCtrl+Shift+M", move || {
                        open_compose_window(app_handle.clone(), None, None);
                    })
                    .ok();
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_info() {
        let info = get_platform_info();
        assert!(info.get("os").is_some());
        assert!(info.get("arch").is_some());
    }
}
