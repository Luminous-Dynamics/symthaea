//! Mycelix Mail Desktop Application
//!
//! A cross-platform desktop email client built with Tauri.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Manager, State, SystemTray, SystemTrayEvent,
    CustomMenuItem, SystemTrayMenu, SystemTrayMenuItem,
};

mod commands;
mod keychain;
mod notifications;
mod settings;

// Application state
struct AppState {
    api_base_url: Mutex<String>,
    access_token: Mutex<Option<String>>,
    unread_count: Mutex<u32>,
}

fn main() {
    env_logger::init();

    let tray_menu = SystemTrayMenu::new()
        .add_item(CustomMenuItem::new("open", "Open Mycelix Mail"))
        .add_item(CustomMenuItem::new("compose", "New Message"))
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("check", "Check for Mail"))
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("settings", "Settings"))
        .add_item(CustomMenuItem::new("quit", "Quit"));

    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .manage(AppState {
            api_base_url: Mutex::new("https://api.mycelix.mail".to_string()),
            access_token: Mutex::new(None),
            unread_count: Mutex::new(0),
        })
        .system_tray(system_tray)
        .on_system_tray_event(handle_tray_event)
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            // Handle deep links when app is already running
            if let Some(url) = argv.get(1) {
                handle_deep_link(app, url);
            }
            // Focus the main window
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
            }
        }))
        .invoke_handler(tauri::generate_handler![
            commands::login,
            commands::logout,
            commands::get_emails,
            commands::get_email,
            commands::send_email,
            commands::delete_email,
            commands::move_email,
            commands::search_emails,
            commands::get_folders,
            commands::get_contacts,
            commands::check_for_updates,
            commands::get_settings,
            commands::save_settings,
            commands::set_badge_count,
            commands::show_notification,
            commands::store_credentials,
            commands::get_stored_credentials,
            commands::clear_credentials,
        ])
        .setup(|app| {
            // Load stored credentials on startup
            if let Ok(Some(token)) = keychain::get_access_token() {
                let state: State<AppState> = app.state();
                *state.access_token.lock().unwrap() = Some(token);
            }

            // Start background sync
            let handle = app.handle().clone();
            tokio::spawn(async move {
                background_sync(handle).await;
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn handle_tray_event(app: &AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick { .. } => {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
        SystemTrayEvent::MenuItemClick { id, .. } => {
            match id.as_str() {
                "open" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
                "compose" => {
                    let _ = app.emit("navigate", "/compose");
                }
                "check" => {
                    let _ = app.emit("check-mail", ());
                }
                "settings" => {
                    let _ = app.emit("navigate", "/settings");
                }
                "quit" => {
                    app.exit(0);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn handle_deep_link(app: &AppHandle, url: &str) {
    // Handle mailto: and mycelix: deep links
    if url.starts_with("mailto:") {
        let recipient = url.trim_start_matches("mailto:");
        let _ = app.emit("compose-to", recipient);
    } else if url.starts_with("mycelix://") {
        let path = url.trim_start_matches("mycelix://");
        let _ = app.emit("navigate", path);
    }
}

async fn background_sync(app: AppHandle) {
    use std::time::Duration;
    use tokio::time::interval;

    let mut interval = interval(Duration::from_secs(60)); // Check every minute

    loop {
        interval.tick().await;

        let state: State<AppState> = app.state();
        let token = state.access_token.lock().unwrap().clone();

        if let Some(token) = token {
            // Check for new mail
            if let Ok(count) = check_unread_count(&state.api_base_url.lock().unwrap(), &token).await {
                let old_count = *state.unread_count.lock().unwrap();

                if count > old_count {
                    // Show notification for new mail
                    let new_count = count - old_count;
                    let _ = notifications::show_new_mail_notification(&app, new_count);
                }

                *state.unread_count.lock().unwrap() = count;

                // Update badge
                let _ = update_badge(&app, count);
            }
        }
    }
}

async fn check_unread_count(api_url: &str, token: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let response = client
        .get(format!("{}/api/v1/emails/unread/count", api_url))
        .bearer_auth(token)
        .send()
        .await?
        .json::<UnreadCountResponse>()
        .await?;

    Ok(response.count)
}

#[derive(Deserialize)]
struct UnreadCountResponse {
    count: u32,
}

fn update_badge(app: &AppHandle, count: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(target_os = "macos")]
    {
        use cocoa::appkit::NSApplication;
        use cocoa::base::nil;
        use cocoa::foundation::NSString;

        unsafe {
            let app = NSApplication::sharedApplication(nil);
            if count > 0 {
                let badge = NSString::alloc(nil).init_str(&count.to_string());
                app.setDockBadge_(badge);
            } else {
                app.setDockBadge_(nil);
            }
        }
    }

    // Update tray icon tooltip
    if let Some(tray) = app.tray_by_id("main") {
        let tooltip = if count > 0 {
            format!("Mycelix Mail - {} unread", count)
        } else {
            "Mycelix Mail".to_string()
        };
        let _ = tray.set_tooltip(Some(&tooltip));
    }

    Ok(())
}

// Command module implementations
mod commands {
    use super::*;
    use tauri::command;

    #[derive(Serialize, Deserialize)]
    pub struct Email {
        pub id: String,
        pub from: String,
        pub to: Vec<String>,
        pub subject: String,
        pub body: String,
        pub date: String,
        pub is_read: bool,
        pub is_starred: bool,
        pub folder: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct LoginRequest {
        pub email: String,
        pub password: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct LoginResponse {
        pub access_token: String,
        pub refresh_token: String,
        pub user: UserInfo,
    }

    #[derive(Serialize, Deserialize)]
    pub struct UserInfo {
        pub id: String,
        pub email: String,
        pub name: String,
    }

    #[command]
    pub async fn login(
        state: State<'_, AppState>,
        email: String,
        password: String,
    ) -> Result<LoginResponse, String> {
        let client = reqwest::Client::new();
        let api_url = state.api_base_url.lock().unwrap().clone();

        let response = client
            .post(format!("{}/api/v1/auth/login", api_url))
            .json(&LoginRequest { email, password })
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<LoginResponse>()
            .await
            .map_err(|e| e.to_string())?;

        // Store token
        *state.access_token.lock().unwrap() = Some(response.access_token.clone());

        // Store in keychain
        let _ = keychain::store_access_token(&response.access_token);

        Ok(response)
    }

    #[command]
    pub async fn logout(state: State<'_, AppState>) -> Result<(), String> {
        *state.access_token.lock().unwrap() = None;
        let _ = keychain::clear_access_token();
        Ok(())
    }

    #[command]
    pub async fn get_emails(
        state: State<'_, AppState>,
        folder: String,
        page: u32,
        limit: u32,
    ) -> Result<Vec<Email>, String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/v1/emails", api_url))
            .query(&[
                ("folder", folder),
                ("page", page.to_string()),
                ("limit", limit.to_string()),
            ])
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<Vec<Email>>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(response)
    }

    #[command]
    pub async fn get_email(
        state: State<'_, AppState>,
        id: String,
    ) -> Result<Email, String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/v1/emails/{}", api_url, id))
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<Email>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(response)
    }

    #[command]
    pub async fn send_email(
        state: State<'_, AppState>,
        to: Vec<String>,
        subject: String,
        body: String,
    ) -> Result<Email, String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        #[derive(Serialize)]
        struct SendEmailRequest {
            to: Vec<String>,
            subject: String,
            body: String,
        }

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/api/v1/emails/send", api_url))
            .bearer_auth(token)
            .json(&SendEmailRequest { to, subject, body })
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<Email>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(response)
    }

    #[command]
    pub async fn delete_email(
        state: State<'_, AppState>,
        id: String,
    ) -> Result<(), String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        let client = reqwest::Client::new();
        client
            .delete(format!("{}/api/v1/emails/{}", api_url, id))
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    #[command]
    pub async fn move_email(
        state: State<'_, AppState>,
        id: String,
        folder: String,
    ) -> Result<(), String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        #[derive(Serialize)]
        struct MoveRequest { folder: String }

        let client = reqwest::Client::new();
        client
            .patch(format!("{}/api/v1/emails/{}/move", api_url, id))
            .bearer_auth(token)
            .json(&MoveRequest { folder })
            .send()
            .await
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    #[command]
    pub async fn search_emails(
        state: State<'_, AppState>,
        query: String,
    ) -> Result<Vec<Email>, String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/v1/emails/search", api_url))
            .query(&[("q", query)])
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<Vec<Email>>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(response)
    }

    #[command]
    pub async fn get_folders(state: State<'_, AppState>) -> Result<Vec<String>, String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/v1/folders", api_url))
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<Vec<String>>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(response)
    }

    #[command]
    pub async fn get_contacts(state: State<'_, AppState>) -> Result<Vec<String>, String> {
        let token = state.access_token.lock().unwrap().clone()
            .ok_or("Not authenticated")?;
        let api_url = state.api_base_url.lock().unwrap().clone();

        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/api/v1/contacts", api_url))
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json::<Vec<String>>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(response)
    }

    #[command]
    pub async fn check_for_updates() -> Result<bool, String> {
        // Updates handled by tauri-plugin-updater
        Ok(false)
    }

    #[command]
    pub fn get_settings() -> Result<settings::AppSettings, String> {
        settings::load_settings().map_err(|e| e.to_string())
    }

    #[command]
    pub fn save_settings(settings: settings::AppSettings) -> Result<(), String> {
        settings::save_settings(&settings).map_err(|e| e.to_string())
    }

    #[command]
    pub fn set_badge_count(app: AppHandle, count: u32) -> Result<(), String> {
        super::update_badge(&app, count).map_err(|e| e.to_string())
    }

    #[command]
    pub fn show_notification(
        app: AppHandle,
        title: String,
        body: String,
    ) -> Result<(), String> {
        notifications::show_notification(&app, &title, &body).map_err(|e| e.to_string())
    }

    #[command]
    pub fn store_credentials(key: String, value: String) -> Result<(), String> {
        keychain::store_credential(&key, &value).map_err(|e| e.to_string())
    }

    #[command]
    pub fn get_stored_credentials(key: String) -> Result<Option<String>, String> {
        keychain::get_credential(&key).map_err(|e| e.to_string())
    }

    #[command]
    pub fn clear_credentials(key: String) -> Result<(), String> {
        keychain::clear_credential(&key).map_err(|e| e.to_string())
    }
}

mod keychain {
    use keyring::Entry;

    const SERVICE_NAME: &str = "mycelix-mail";

    pub fn store_access_token(token: &str) -> Result<(), Box<dyn std::error::Error>> {
        store_credential("access_token", token)
    }

    pub fn get_access_token() -> Result<Option<String>, Box<dyn std::error::Error>> {
        get_credential("access_token")
    }

    pub fn clear_access_token() -> Result<(), Box<dyn std::error::Error>> {
        clear_credential("access_token")
    }

    pub fn store_credential(key: &str, value: &str) -> Result<(), Box<dyn std::error::Error>> {
        let entry = Entry::new(SERVICE_NAME, key)?;
        entry.set_password(value)?;
        Ok(())
    }

    pub fn get_credential(key: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let entry = Entry::new(SERVICE_NAME, key)?;
        match entry.get_password() {
            Ok(password) => Ok(Some(password)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(Box::new(e)),
        }
    }

    pub fn clear_credential(key: &str) -> Result<(), Box<dyn std::error::Error>> {
        let entry = Entry::new(SERVICE_NAME, key)?;
        let _ = entry.delete_credential();
        Ok(())
    }
}

mod notifications {
    use tauri::AppHandle;

    pub fn show_new_mail_notification(
        app: &AppHandle,
        count: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let title = "New Mail";
        let body = if count == 1 {
            "You have 1 new message".to_string()
        } else {
            format!("You have {} new messages", count)
        };

        show_notification(app, title, &body)
    }

    pub fn show_notification(
        app: &AppHandle,
        title: &str,
        body: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        app.notification()
            .builder()
            .title(title)
            .body(body)
            .show()?;

        Ok(())
    }
}

mod settings {
    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::path::PathBuf;

    #[derive(Serialize, Deserialize, Clone)]
    pub struct AppSettings {
        pub theme: String,
        pub notifications_enabled: bool,
        pub sound_enabled: bool,
        pub sync_interval_minutes: u32,
        pub default_account_id: Option<String>,
        pub compact_view: bool,
        pub show_preview: bool,
        pub start_minimized: bool,
        pub minimize_to_tray: bool,
    }

    impl Default for AppSettings {
        fn default() -> Self {
            Self {
                theme: "system".to_string(),
                notifications_enabled: true,
                sound_enabled: true,
                sync_interval_minutes: 5,
                default_account_id: None,
                compact_view: false,
                show_preview: true,
                start_minimized: false,
                minimize_to_tray: true,
            }
        }
    }

    fn settings_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mycelix-mail")
            .join("settings.json")
    }

    pub fn load_settings() -> Result<AppSettings, Box<dyn std::error::Error>> {
        let path = settings_path();

        if path.exists() {
            let content = fs::read_to_string(&path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(AppSettings::default())
        }
    }

    pub fn save_settings(settings: &AppSettings) -> Result<(), Box<dyn std::error::Error>> {
        let path = settings_path();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(settings)?;
        fs::write(path, content)?;

        Ok(())
    }
}
