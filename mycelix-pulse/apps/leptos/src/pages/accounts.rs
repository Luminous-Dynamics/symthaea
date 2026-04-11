// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! External email accounts page — IMAP/SMTP configuration for legacy email.
//!
//! Supports Gmail, Outlook, Yahoo, iCloud, ProtonMail Bridge, Fastmail,
//! AOL, Zoho, GMX, Mail.de, and custom IMAP/SMTP servers.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use crate::toasts::use_toasts;

const STORAGE_KEY: &str = "mycelix_mail_external_accounts";

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExternalAccount {
    pub id: String,
    pub label: String,
    pub email: String,
    pub provider: Provider,
    pub imap_host: String,
    pub imap_port: u16,
    pub imap_tls: bool,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub smtp_tls: bool,
    pub username: String,
    /// Not stored in localStorage in production — would use secure keystore
    pub password: String,
    pub enabled: bool,
    pub last_sync: Option<u64>,
    pub sync_interval_secs: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Provider {
    Gmail,
    Outlook,
    Yahoo,
    ICloud,
    ProtonMail,
    Fastmail,
    Aol,
    Zoho,
    Gmx,
    MailDe,
    Custom,
}

impl Provider {
    pub const ALL: &[Provider] = &[
        Self::Gmail, Self::Outlook, Self::Yahoo, Self::ICloud,
        Self::ProtonMail, Self::Fastmail, Self::Aol, Self::Zoho,
        Self::Gmx, Self::MailDe, Self::Custom,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Gmail => "Gmail",
            Self::Outlook => "Outlook / Hotmail / Live",
            Self::Yahoo => "Yahoo Mail",
            Self::ICloud => "iCloud Mail",
            Self::ProtonMail => "ProtonMail (Bridge)",
            Self::Fastmail => "Fastmail",
            Self::Aol => "AOL Mail",
            Self::Zoho => "Zoho Mail",
            Self::Gmx => "GMX Mail",
            Self::MailDe => "Mail.de",
            Self::Custom => "Custom IMAP/SMTP",
        }
    }

    pub fn supports_oauth(&self) -> bool {
        matches!(self, Self::Gmail | Self::Outlook)
    }

    pub fn oauth_url(&self) -> Option<&'static str> {
        match self {
            Self::Gmail => Some("https://accounts.google.com/o/oauth2/v2/auth"),
            Self::Outlook => Some("https://login.microsoftonline.com/common/oauth2/v2/authorize"),
            _ => None,
        }
    }

    pub fn defaults(&self) -> ProviderDefaults {
        match self {
            Self::Gmail => ProviderDefaults {
                imap_host: "imap.gmail.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.gmail.com", smtp_port: 587, smtp_tls: true,
                note: "Sign in with Google — no app password needed. OAuth2 handles authentication securely.",
            },
            Self::Outlook => ProviderDefaults {
                imap_host: "outlook.office365.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.office365.com", smtp_port: 587, smtp_tls: true,
                note: "Sign in with Microsoft — no app password needed. OAuth2 handles authentication securely.",
            },
            Self::Yahoo => ProviderDefaults {
                imap_host: "imap.mail.yahoo.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.mail.yahoo.com", smtp_port: 587, smtp_tls: true,
                note: "Generate App Password at login.yahoo.com/account/security",
            },
            Self::ICloud => ProviderDefaults {
                imap_host: "imap.mail.me.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.mail.me.com", smtp_port: 587, smtp_tls: true,
                note: "Generate App-Specific Password at appleid.apple.com/account/manage",
            },
            Self::ProtonMail => ProviderDefaults {
                imap_host: "127.0.0.1", imap_port: 1143, imap_tls: false,
                smtp_host: "127.0.0.1", smtp_port: 1025, smtp_tls: false,
                note: "Requires ProtonMail Bridge running locally. Download from proton.me/mail/bridge",
            },
            Self::Fastmail => ProviderDefaults {
                imap_host: "imap.fastmail.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.fastmail.com", smtp_port: 587, smtp_tls: true,
                note: "Generate App Password at fastmail.com/settings/security/tokens",
            },
            Self::Aol => ProviderDefaults {
                imap_host: "imap.aol.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.aol.com", smtp_port: 587, smtp_tls: true,
                note: "Generate App Password in AOL account security settings",
            },
            Self::Zoho => ProviderDefaults {
                imap_host: "imap.zoho.com", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.zoho.com", smtp_port: 587, smtp_tls: true,
                note: "Enable IMAP in Zoho Mail settings. Use App-Specific Password if 2FA enabled",
            },
            Self::Gmx => ProviderDefaults {
                imap_host: "imap.gmx.com", imap_port: 993, imap_tls: true,
                smtp_host: "mail.gmx.com", smtp_port: 587, smtp_tls: true,
                note: "Enable IMAP/SMTP in GMX settings under E-Mail > POP3 & IMAP",
            },
            Self::MailDe => ProviderDefaults {
                imap_host: "imap.mail.de", imap_port: 993, imap_tls: true,
                smtp_host: "smtp.mail.de", smtp_port: 465, smtp_tls: true,
                note: "Enable IMAP in Mail.de account settings",
            },
            Self::Custom => ProviderDefaults {
                imap_host: "", imap_port: 993, imap_tls: true,
                smtp_host: "", smtp_port: 587, smtp_tls: true,
                note: "Enter your mail server details manually",
            },
        }
    }
}

pub struct ProviderDefaults {
    pub imap_host: &'static str,
    pub imap_port: u16,
    pub imap_tls: bool,
    pub smtp_host: &'static str,
    pub smtp_port: u16,
    pub smtp_tls: bool,
    pub note: &'static str,
}

fn load_accounts() -> Vec<ExternalAccount> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

fn save_accounts(accounts: &[ExternalAccount]) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        if let Ok(json) = serde_json::to_string(accounts) {
            let _ = storage.set_item(STORAGE_KEY, &json);
        }
    }
}

fn input_value(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

fn select_value(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

#[component]
pub fn AccountsPage() -> impl IntoView {
    let toasts = use_toasts();
    let accounts = RwSignal::new(load_accounts());
    let adding = RwSignal::new(false);

    // OAuth callback: check for ?code= in URL
    {
        let toasts_oauth_cb = toasts.clone();
        if let Some(search) = web_sys::window().and_then(|w| w.location().search().ok()) {
            if search.contains("code=") {
                // Extract authorization code
                let code = search.split("code=").nth(1)
                    .and_then(|s| s.split('&').next())
                    .unwrap_or("");
                if !code.is_empty() {
                    web_sys::console::log_1(&format!("[OAuth] Received auth code: {}...", &code[..code.len().min(10)]).into());
                    toasts_oauth_cb.push("OAuth authorization received! Token exchange would happen here.", "success");
                    // In production: exchange code for access_token via token endpoint
                    // Then save the token in the account config
                    // Clear the URL params
                    let _ = web_sys::window().and_then(|w| {
                        w.history().ok().and_then(|h| {
                            h.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some("/accounts")).ok()
                        })
                    });
                }
            }
        }
    }

    // New account form state
    let provider = RwSignal::new(Provider::Gmail);
    let email = RwSignal::new(String::new());
    let username = RwSignal::new(String::new());
    let password = RwSignal::new(String::new());
    let imap_host = RwSignal::new(String::new());
    let imap_port = RwSignal::new(993u16);
    let smtp_host = RwSignal::new(String::new());
    let smtp_port = RwSignal::new(587u16);
    let show_advanced = RwSignal::new(false);

    // Apply provider defaults when provider changes
    let apply_defaults = move || {
        let defaults = provider.get_untracked().defaults();
        imap_host.set(defaults.imap_host.to_string());
        imap_port.set(defaults.imap_port);
        smtp_host.set(defaults.smtp_host.to_string());
        smtp_port.set(defaults.smtp_port);
    };

    let toasts_save = toasts.clone();
    let on_save = move |_| {
        let p = provider.get_untracked();
        let defaults = p.defaults();
        let e = email.get_untracked();
        let u = username.get_untracked();
        let pw = password.get_untracked();

        if e.is_empty() || pw.is_empty() {
            toasts_save.push("Email and password are required", "error");
            return;
        }

        let account = ExternalAccount {
            id: format!("ext-{}", js_sys::Date::now() as u64),
            label: p.label().to_string(),
            email: e.clone(),
            provider: p,
            imap_host: if imap_host.get_untracked().is_empty() { defaults.imap_host.to_string() } else { imap_host.get_untracked() },
            imap_port: imap_port.get_untracked(),
            imap_tls: defaults.imap_tls,
            smtp_host: if smtp_host.get_untracked().is_empty() { defaults.smtp_host.to_string() } else { smtp_host.get_untracked() },
            smtp_port: smtp_port.get_untracked(),
            smtp_tls: defaults.smtp_tls,
            username: if u.is_empty() { e.clone() } else { u },
            password: pw,
            enabled: true,
            last_sync: None,
            sync_interval_secs: 300,
        };

        accounts.update(|a| a.push(account));
        save_accounts(&accounts.get_untracked());

        // Reset form
        email.set(String::new());
        username.set(String::new());
        password.set(String::new());
        adding.set(false);
        toasts_save.push(format!("Added {} account", e), "success");
    };

    let toasts_del = toasts.clone();
    let _on_delete = move |id: String| {
        accounts.update(|a| a.retain(|acc| acc.id != id));
        save_accounts(&accounts.get_untracked());
        toasts_del.push("Account removed", "info");
    };

    let toasts_toggle = toasts.clone();
    let toasts_oauth = toasts.clone();
    let _on_toggle = move |id: String| {
        accounts.update(|a| {
            if let Some(acc) = a.iter_mut().find(|acc| acc.id == id) {
                acc.enabled = !acc.enabled;
            }
        });
        save_accounts(&accounts.get_untracked());
        toasts_toggle.push("Account updated", "info");
    };

    let provider_note = move || provider.get().defaults().note;

    view! {
        <div class="page page-accounts">
            <div class="page-header">
                <h1>"External Accounts"</h1>
                <button class="btn btn-primary" on:click=move |_| { adding.set(true); apply_defaults(); }>
                    "+ Add Account"
                </button>
            </div>

            <p class="page-desc">
                "Connect your existing email accounts (Gmail, Outlook, Yahoo, etc.) to receive "
                "and send legacy email alongside your Holochain-encrypted messages."
            </p>

            // Existing accounts
            <div class="accounts-list">
                {move || accounts.get().iter().map(|acc| {
                    let id = acc.id.clone();
                    let id2 = acc.id.clone();
                    let label = acc.label.clone();
                    let email_display = acc.email.clone();
                    let enabled = acc.enabled;
                    let imap = format!("{}:{}", acc.imap_host, acc.imap_port);
                    let smtp = format!("{}:{}", acc.smtp_host, acc.smtp_port);
                    let last = acc.last_sync.map(|_| "Recently").unwrap_or("Never");

                    // Each card gets its own toasts clone
                    let t1 = toasts.clone();
                    let t2 = toasts.clone();

                    view! {
                        <div class=if enabled { "account-card" } else { "account-card disabled" }>
                            <div class="account-header">
                                <div class="account-info">
                                    <span class="account-provider">{label}</span>
                                    <span class="account-email">{email_display}</span>
                                </div>
                                <div class="account-actions">
                                    <button class="btn btn-sm btn-secondary"
                                        on:click=move |_| {
                                            accounts.update(|a| {
                                                if let Some(acc) = a.iter_mut().find(|a| a.id == id2) { acc.enabled = !acc.enabled; }
                                            });
                                            save_accounts(&accounts.get_untracked());
                                            t1.push("Account updated", "info");
                                        }>
                                        {if enabled { "Disable" } else { "Enable" }}
                                    </button>
                                    <button class="btn btn-sm btn-secondary danger"
                                        on:click=move |_| {
                                            accounts.update(|a| a.retain(|a| a.id != id));
                                            save_accounts(&accounts.get_untracked());
                                            t2.push("Account removed", "info");
                                        }>
                                        "Remove"
                                    </button>
                                </div>
                            </div>
                            <div class="account-details">
                                <span class="detail">"IMAP: "{imap}</span>
                                <span class="detail">"SMTP: "{smtp}</span>
                                <span class="detail">"Last sync: "{last}</span>
                                <span class=if enabled { "status-badge active" } else { "status-badge" }>
                                    {if enabled { "Active" } else { "Disabled" }}
                                </span>
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>()}

                <div class="empty-state" style=move || if accounts.get().is_empty() { "" } else { "display:none" }>
                    <span class="empty-icon">"\u{1F4EC}"</span>
                    <p class="empty-title">"No external accounts"</p>
                    <p class="empty-desc">"Add your Gmail, Outlook, or other email accounts to send and receive legacy email."</p>
                </div>
            </div>

            // Add account form
            <div class="add-account-form" style=move || if adding.get() { "" } else { "display:none" }>
                    <h2>"Add External Account"</h2>

                    <div class="form-field">
                        <label>"Provider"</label>
                        <select on:change=move |ev| {
                            let val = select_value(&ev);
                            let p = match val.as_str() {
                                "Gmail" => Provider::Gmail,
                                "Outlook" => Provider::Outlook,
                                "Yahoo" => Provider::Yahoo,
                                "ICloud" => Provider::ICloud,
                                "ProtonMail" => Provider::ProtonMail,
                                "Fastmail" => Provider::Fastmail,
                                "Aol" => Provider::Aol,
                                "Zoho" => Provider::Zoho,
                                "Gmx" => Provider::Gmx,
                                "MailDe" => Provider::MailDe,
                                _ => Provider::Custom,
                            };
                            provider.set(p);
                            apply_defaults();
                        }>
                            {Provider::ALL.iter().map(|p| {
                                let label = p.label();
                                let value = format!("{:?}", p);
                                view! { <option value=value>{label}</option> }
                            }).collect::<Vec<_>>()}
                        </select>
                    </div>

                    <div class="provider-note">
                        <span class="note-icon">"\u{2139}"</span>
                        <span>{provider_note}</span>
                    </div>

                    <div class="form-field">
                        <label>"Email Address"</label>
                        <input type="email" placeholder="you@gmail.com"
                            prop:value=move || email.get()
                            on:input=move |ev| email.set(input_value(&ev)) />
                    </div>

                    // OAuth sign-in for supported providers
                    <div class="oauth-section" style=move || if provider.get().supports_oauth() { "" } else { "display:none" }>
                        <button class="btn btn-primary oauth-btn" on:click=move |_| {
                            let p = provider.get_untracked();
                            if let Some(auth_url) = p.oauth_url() {
                                let client_id = match p {
                                    Provider::Gmail => "mycelix-pulse-mail.apps.googleusercontent.com",
                                    Provider::Outlook => "mycelix-pulse-mail",
                                    _ => "",
                                };
                                // OAuth callback goes to the bridge's auth server, not the frontend
                                let redirect = "http://localhost:8118/callback";
                                let scope = match p {
                                    Provider::Gmail => "https://mail.google.com/",
                                    Provider::Outlook => "https://outlook.office365.com/IMAP.AccessAsUser.All%20https://outlook.office365.com/SMTP.Send",
                                    _ => "",
                                };
                                let url = format!(
                                    "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&access_type=offline&prompt=consent",
                                    auth_url, client_id, redirect, scope
                                );
                                let _ = web_sys::window().unwrap().open_with_url_and_target(&url, "_blank");
                            }
                            toasts_oauth.push("Opening sign-in window...", "info");
                        }>
                            {move || match provider.get() {
                                Provider::Gmail => "\u{1F310} Sign in with Google",
                                Provider::Outlook => "\u{1F310} Sign in with Microsoft",
                                _ => "Sign in",
                            }}
                        </button>
                        <p class="oauth-hint">"No password needed — OAuth2 handles authentication securely."</p>
                    </div>

                    // Password field for non-OAuth providers
                    <div class="form-field" style=move || if provider.get().supports_oauth() { "display:none" } else { "" }>
                        <label>"Password / App Password"</label>
                        <input type="password" placeholder="App-specific password"
                            prop:value=move || password.get()
                            on:input=move |ev| password.set(input_value(&ev)) />
                    </div>

                    <button class="toggle-advanced" on:click=move |_| show_advanced.update(|v| *v = !*v)>
                        {move || if show_advanced.get() { "\u{25BC} Hide server settings" } else { "\u{25B6} Show server settings" }}
                    </button>

                    <div class="advanced-settings" style=move || if show_advanced.get() { "" } else { "display:none" }>
                            <div class="form-row">
                                <div class="form-field">
                                    <label>"IMAP Host"</label>
                                    <input type="text"
                                        prop:value=move || imap_host.get()
                                        on:input=move |ev| imap_host.set(input_value(&ev)) />
                                </div>
                                <div class="form-field narrow">
                                    <label>"Port"</label>
                                    <input type="number"
                                        prop:value=move || imap_port.get().to_string()
                                        on:input=move |ev| {
                                            if let Ok(p) = input_value(&ev).parse() { imap_port.set(p); }
                                        } />
                                </div>
                            </div>
                            <div class="form-row">
                                <div class="form-field">
                                    <label>"SMTP Host"</label>
                                    <input type="text"
                                        prop:value=move || smtp_host.get()
                                        on:input=move |ev| smtp_host.set(input_value(&ev)) />
                                </div>
                                <div class="form-field narrow">
                                    <label>"Port"</label>
                                    <input type="number"
                                        prop:value=move || smtp_port.get().to_string()
                                        on:input=move |ev| {
                                            if let Ok(p) = input_value(&ev).parse() { smtp_port.set(p); }
                                        } />
                                </div>
                            </div>
                            <div class="form-field">
                                <label>"Username (if different from email)"</label>
                                <input type="text" placeholder="Optional"
                                    prop:value=move || username.get()
                                    on:input=move |ev| username.set(input_value(&ev)) />
                            </div>
                    </div>

                    <div class="form-actions">
                        <button class="btn btn-secondary" on:click=move |_| adding.set(false)>"Cancel"</button>
                        <button class="btn btn-primary" on:click=on_save>"Add Account"</button>
                    </div>
            </div>
        </div>
    }
}
