// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::ev::SubmitEvent;
use leptos::prelude::*;
use leptos_router::components::{Route, Router, Routes, A};
use leptos_router::hooks::use_location;
use leptos_router::path;
use wasm_bindgen_futures::spawn_local;

use mycelix_leptos_core::consciousness::refresh_consciousness_from_conductor;
use mycelix_leptos_core::{
    init_consciousness_ui, provide_consciousness_context, provide_homeostasis_context,
    provide_local_identity, provide_theme_context, provide_thermodynamic_context,
    provide_toast_context, use_toasts, ActivityFeed, ActivityFeedItem, AppShell, AvailabilityState,
    AvailabilityStateKind, ConnectStrategy, EmptyState, FreshnessBadge, FreshnessLevel,
    HolochainProviderAuto, HolochainProviderConfig, NavLink, NavTab, ToastContainer, ToastKind,
};
use personal_leptos_types::{ConsentGrantView, CredentialType, StoredCredentialView};

use crate::context::{
    provide_cultural_context, provide_personal_context, refresh_health_state,
    refresh_identity_state, refresh_preferences_state, use_cultural, use_personal,
};
use crate::telemetry::ConstellationTelemetry;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PersonalTheme {
    Vault,
    Dawn,
}

impl mycelix_leptos_core::AppTheme for PersonalTheme {
    fn label(&self) -> &'static str {
        match self {
            Self::Vault => "vault",
            Self::Dawn => "dawn",
        }
    }

    fn all() -> &'static [Self] {
        &[Self::Vault, Self::Dawn]
    }

    fn next(&self) -> Self {
        match self {
            Self::Vault => Self::Dawn,
            Self::Dawn => Self::Vault,
        }
    }

    fn is_light(&self) -> bool {
        matches!(self, Self::Dawn)
    }
}

#[component]
pub fn App() -> impl IntoView {
    let config = HolochainProviderConfig {
        app_id: "mycelix-unified".into(),
        default_role: Some("personal".into()),
        log_prefix: "[Personal]",
        connect_strategy: ConnectStrategy::WebSocket,
        status_labels: None,
    };

    view! {
        <HolochainProviderAuto config=config>
            <AppInner />
        </HolochainProviderAuto>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    provide_theme_context("personal-theme", PersonalTheme::Vault);
    provide_thermodynamic_context();
    let consciousness = provide_consciousness_context();
    provide_toast_context();
    provide_homeostasis_context(1, "--personal-homeostasis");
    provide_local_identity();
    provide_personal_context();
    provide_cultural_context();
    init_consciousness_ui();

    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    refresh_consciousness_from_conductor(&consciousness, &hc);

    let nav_links = vec![
        NavLink {
            href: "/",
            label: "Vault",
            icon: Some("◉"),
        },
        NavLink {
            href: "/identity",
            label: "Identity",
            icon: Some("ID"),
        },
        NavLink {
            href: "/wallet",
            label: "Wallet",
            icon: Some("VC"),
        },
        NavLink {
            href: "/health",
            label: "Health",
            icon: Some("HX"),
        },
        NavLink {
            href: "/preferences",
            label: "Preferences",
            icon: Some("PX"),
        },
        NavLink {
            href: "/activity",
            label: "Activity",
            icon: Some("AX"),
        },
        NavLink {
            href: "/profile",
            label: "Profile",
            icon: Some("ME"),
        },
        // Constellation Group (Satellite hApps)
        NavLink {
            href: "/civic",
            label: "Civic",
            icon: Some("⚔"),
        },
        NavLink {
            href: "/knowledge",
            label: "Knowledge",
            icon: Some("📖"),
        },
        NavLink {
            href: "/finance",
            label: "Finance",
            icon: Some("💰"),
        },
    ];

    let mobile_tabs = vec![
        NavTab {
            href: "/",
            icon: "◉",
            label: "Vault",
        },
        NavTab {
            href: "/wallet",
            icon: "VC",
            label: "Wallet",
        },
        NavTab {
            href: "/health",
            icon: "HX",
            label: "Health",
        },
        NavTab {
            href: "/preferences",
            icon: "PX",
            label: "Prefs",
        },
    ];

    view! {
        <Router>
            <AppShell
                brand_name="Personal"
                brand_icon="◉"
                nav_links=nav_links
                mobile_tabs=mobile_tabs
            >
                <Routes fallback=|| view! {
                    <EmptyState icon="?" title="Vault route not found" />
                }>
                    <Route path=path!("/") view=VaultPage />
                    <Route path=path!("/identity") view=IdentityPage />
                    <Route path=path!("/wallet") view=WalletPage />
                    <Route path=path!("/health") view=HealthPage />
                    <Route path=path!("/preferences") view=PreferencesPage />
                    <Route path=path!("/activity") view=ActivityPage />
                    <Route path=path!("/profile") view=IdentityPage />
                    <Route path=path!("/unlock") view=UnlockPage />
                    // Constellation Satellite Routes
                    <Route path=path!("/civic") view=|| view! { <EmbeddedSatellite name="Civic" port=5174 /> } />
                    <Route path=path!("/knowledge") view=|| view! { <EmbeddedSatellite name="Knowledge" port=5175 /> } />
                    <Route path=path!("/finance") view=|| view! { <EmbeddedSatellite name="Finance" port=5176 /> } />
                </Routes>
            </AppShell>
            <ToastContainer />
        </Router>
    }
}

#[component]
fn CulturalReskinSelector() -> impl IntoView {
    let cultural = use_cultural();
    let symbols = cultural.symbols;

    let set_context = move |id: &str| {
        let new_symbols = match id {
            "indigenous" => SymbolRegistry {
                hearth_alias: "WELL".into(),
                mycel_alias: "SPARK".into(),
                genesis_alias: "The Sun-Rise".into(),
                orientation: "Indigenous Stewardship metaphors".into(),
            },
            "community" => SymbolRegistry {
                hearth_alias: "CAMPFIRE".into(),
                mycel_alias: "EMBER".into(),
                genesis_alias: "The Ignition".into(),
                orientation: "Urban Mutual-Aid metaphors".into(),
            },
            _ => SymbolRegistry::default(),
        };
        symbols.set(new_symbols);
    };

    view! {
        <div style="display: flex; gap: 0.8rem; margin-bottom: 2rem; padding: 0.8rem; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid var(--md-divider);">
            <span style="font-size: 0.75rem; color: var(--md-fg-muted); align-self: center; margin-right: 0.5rem;">
                "CULTURAL HUD:"
            </span>
            <button class="btn-vault" on:click=move |_| set_context("canonical")> "Canonical" </button>
            <button class="btn-vault" on:click=move |_| set_context("indigenous")> "Indigenous" </button>
            <button class="btn-vault" on:click=move |_| set_context("community")> "Community" </button>
        </div>
    }
}

#[component]
fn VaultPage() -> impl IntoView {
    let ctx = use_personal();
    let cultural = use_cultural();
    let symbols = cultural.symbols;

    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    let active_consents = Memo::new(move |_| {
        ctx.consents
            .get()
            .into_iter()
            .filter(|grant| grant.active)
            .count()
    });
    let revoked_credentials = Memo::new(move |_| {
        ctx.credentials
            .get()
            .into_iter()
            .filter(|cred| cred.revoked)
            .count()
    });
    let latest_activity_at = Memo::new(move |_| {
        ctx.activity
            .get()
            .into_iter()
            .map(|item| item.created_at)
            .max()
    });
    let activity_feed = move || {
        ctx.activity
            .get()
            .into_iter()
            .map(|item| ActivityFeedItem {
                id: item.id,
                domain_label: item.domain,
                description: format!("{}: {}", item.title, item.detail),
                emphasis_class: item.success.map(|ok| {
                    if ok {
                        "activity-feed-success".to_string()
                    } else {
                        "activity-feed-warning".to_string()
                    }
                }),
            })
            .collect::<Vec<_>>()
    };

    view! {
        <div class="vault-page">
            <PageHeader
                eyebrow="Sovereign Vault"
                title=move || format!("Private posture, proof posture, and {} posture in one place.", symbols.get().hearth_alias.to_lowercase())
                summary=move || format!("{}. {}", ctx.status_note.get(), symbols.get().orientation)
            />

            <CulturalReskinSelector />

            <div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                {move || {
                    latest_activity_at.get().map(|timestamp| {
                        let level = freshness_from_micros(timestamp);
                        let detail = format!("Activity {}", format_relative_micros(timestamp));
                        view! { <FreshnessBadge level detail /> }.into_any()
                    }).unwrap_or_else(|| {
                        view! { <FreshnessBadge level=FreshnessLevel::Unknown detail="No live activity yet" /> }.into_any()
                    })
                }}
            </div>

            {move || {
                if ctx.loading.get() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Degraded
                            title="Vault Sync In Progress"
                            description="Personal is establishing posture from the conductor and reconciling typed view endpoints."
                            action={None}
                        />
                    }.into_any()
                } else if hc.is_mock() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Mock
                            title="Mock Vault Posture"
                            description="This Personal shell is running without a live conductor. Typed vault flows are visible, but the current records are illustrative."
                            action={None}
                        />
                    }.into_any()
                } else if !ctx.live_sync_ready.get() {
                    view! {
                        <AvailabilityState
                            kind=AvailabilityStateKind::Empty
                            title="Live Vault, Empty Summary"
                            description="Connected to a live conductor, but Personal view endpoints have not returned records yet."
                            action={Some(view! {
                                <A href="/identity" attr:class="btn btn-primary">"Review Identity"</A>
                            }.into_any())}
                        />
                    }.into_any()
                } else {
                    view! { <></> }.into_any()
                }
            }}

            <div class="hero-strip">
                <div class="hero-panel">
                    <span class="hero-kicker">"Vault state"</span>
                    <h2>"Unlocked architecture, guarded disclosure"</h2>
                    <p>
                        "This shell is now live as the canonical Personal frontend scaffold. "
                        "It is intentionally typed and stable while the zome adapters are normalized."
                    </p>
                    <div class="hero-actions">
                        <A href="/wallet" attr:class="btn btn-primary">"Open wallet"</A>
                        <A href="/preferences" attr:class="btn">"Review preferences"</A>
                    </div>
                </div>
                <div class="hero-panel hero-panel-accent">
                    <span class="hero-kicker">"Next infrastructure step"</span>
                    <p>
                        "Expose frontend-facing Personal view endpoints so this app can replace scaffolded vault models with live conductor data."
                    </p>
                </div>
            </div>

            <ConstellationTelemetry />

            <div class="stats-grid">
                <VaultStat label="Credentials" value=move || ctx.credentials.get().len().to_string() />
                <VaultStat label=move || format!("Active {} consents", symbols.get().hearth_alias.to_lowercase()) value=move || active_consents.get().to_string() />
                <VaultStat label="Health records" value=move || ctx.health_record_count.get().to_string() />
                <VaultStat label=move || format!("{} status", symbols.get().mycel_alias) value=move || "Resonant".to_string() />
            </div>

            <div class="vault-columns">
                <section class="vault-card">
                    <SectionTitle title="Recent activity" />
                    <ActivityFeed items=activity_feed() />
                </section>

                <section class="vault-card">
                    <SectionTitle title="Launch paths" />
                    <div class="link-stack">
                        <a class="launch-link" href="/identity">"Identity posture"</a>
                        <a class="launch-link" href="/wallet">"Credential wallet"</a>
                        <a class="launch-link" href="/health">"Health summary"</a>
                        <a class="launch-link" href="/preferences">"Sharing controls"</a>
                    </div>
                </section>
            </div>
        </div>
    }
}

fn freshness_from_micros(timestamp_micros: i64) -> FreshnessLevel {
    let now_micros = (js_sys::Date::now() * 1000.0) as i64;
    let age_minutes = now_micros.saturating_sub(timestamp_micros) / 60_000_000;
    if age_minutes <= 5 {
        FreshnessLevel::Fresh
    } else if age_minutes <= 60 {
        FreshnessLevel::Aging
    } else {
        FreshnessLevel::Stale
    }
}

fn format_relative_micros(timestamp_micros: i64) -> String {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(
        (timestamp_micros / 1000) as f64,
    ));
    date.to_locale_string("en-US", &wasm_bindgen::JsValue::UNDEFINED)
        .as_string()
        .unwrap_or_else(|| "recently".into())
}

#[component]
fn IdentityPage() -> impl IntoView {
    let ctx = use_personal();
    let location = use_location();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    let toasts = use_toasts();
    let ctx_for_save = ctx.clone();

    let save_profile = move |ev: SubmitEvent| {
        ev.prevent_default();
        let draft = ctx_for_save.draft_profile.get();
        let previous = ctx_for_save.profile.get();
        ctx_for_save.profile.set(draft.clone());

        let hc = hc.clone();
        let toasts = toasts.clone();
        let ctx = ctx_for_save.clone();
        spawn_local(async move {
            match hc
                .call_zome_default::<_, serde_json::Value>(
                    "identity_vault",
                    "set_profile_view",
                    &draft,
                )
                .await
            {
                Ok(_) => {
                    refresh_identity_state(ctx.clone(), hc.clone()).await;
                    toasts.push("Profile saved to Personal vault", ToastKind::Success);
                }
                Err(err) => {
                    ctx.profile.set(previous.clone());
                    ctx.draft_profile.set(previous);
                    toasts.push(format!("Profile save failed: {err}"), ToastKind::Error);
                }
            }
        });
    };

    view! {
        <div class="stack-page">
            <PageHeader
                eyebrow="Identity Vault"
                title="Profile posture and key posture"
                summary={
                    if location.pathname.get().contains("/profile") {
                        "Profile route aliases into the identity vault until deeper Personal profile pages are split."
                            .to_string()
                    } else {
                        "Profile editing is scaffolded locally; live mutation will plug into Personal view adapters once exposed."
                            .to_string()
                    }
                }
            />

            <div class="two-up">
                <section class="vault-card">
                    <SectionTitle title="Profile" />
                    <form on:submit=save_profile class="profile-form">
                        <label class="field-block">
                            <span>"Display name"</span>
                            <input
                                class="form-input"
                                prop:value=move || ctx.draft_profile.get().display_name
                                on:input=move |ev| {
                                    let value = event_target_value(&ev);
                                    ctx.draft_profile.update(|profile| profile.display_name = value);
                                }
                            />
                        </label>
                        <label class="field-block">
                            <span>"Bio"</span>
                            <textarea
                                class="form-textarea"
                                prop:value=move || ctx.draft_profile.get().bio.unwrap_or_default()
                                on:input=move |ev| {
                                    let value = event_target_value(&ev);
                                    ctx.draft_profile.update(|profile| {
                                        profile.bio = if value.trim().is_empty() { None } else { Some(value) };
                                    });
                                }
                            />
                        </label>
                        <div class="form-actions">
                            <button class="btn btn-primary" type="submit">"Apply local draft"</button>
                            <a class="btn" href="/wallet">"Open wallet"</a>
                        </div>
                    </form>
                </section>

                <section class="vault-card">
                    <SectionTitle title="Key posture" />
                    <div class="key-list">
                        <For
                            each=move || ctx.keys.get()
                            key=|key| format!("{}-{}", key.label, key.purpose)
                            children=move |key| view! { <KeyCard key_data=key /> }
                        />
                    </div>
                </section>
            </div>
        </div>
    }
}

#[component]
fn WalletPage() -> impl IntoView {
    let ctx = use_personal();

    view! {
        <div class="stack-page">
            <PageHeader
                eyebrow="Credential Wallet"
                title="Portable proof inventory"
                summary="Stored credentials, proof posture, and trust-bearing materials live here.".to_string()
            />

            <section class="vault-card">
                <SectionTitle title="Credentials" />
                <div class="credential-grid">
                    <For
                        each=move || ctx.credentials.get()
                        key=|cred| cred.hash.clone()
                        children=move |cred| view! { <CredentialCard credential=cred /> }
                    />
                </div>
            </section>
        </div>
    }
}

#[component]
fn HealthPage() -> impl IntoView {
    let ctx = use_personal();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    let toasts = use_toasts();
    let consent_grantee = RwSignal::new(String::new());
    let consent_types = RwSignal::new("allergy, medication".to_string());
    let ctx_for_consent = ctx.clone();

    let create_consent = move |ev: SubmitEvent| {
        ev.prevent_default();
        let grantee = consent_grantee.get();
        let record_types: Vec<String> = consent_types
            .get()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect();

        let input = personal_leptos_types::ConsentGrantInputView {
            grantee,
            record_types,
            expires_at: None,
            active: true,
        };

        let hc = hc.clone();
        let toasts = toasts.clone();
        let ctx = ctx_for_consent.clone();
        let consent_grantee_signal = consent_grantee;
        let consent_types_signal = consent_types;
        spawn_local(async move {
            match hc
                .call_zome_default::<_, serde_json::Value>(
                    "health_vault",
                    "grant_consent_view",
                    &input,
                )
                .await
            {
                Ok(_) => {
                    refresh_health_state(ctx.clone(), hc.clone()).await;
                    consent_grantee_signal.set(String::new());
                    consent_types_signal.set("allergy, medication".to_string());
                    toasts.push("Consent grant created", ToastKind::Success);
                }
                Err(err) => toasts.push(format!("Consent grant failed: {err}"), ToastKind::Error),
            }
        });
    };

    view! {
        <div class="stack-page">
            <PageHeader
                eyebrow="Health Vault"
                title="Private summary without replacing the Health domain app"
                summary="Personal owns posture and disclosure. Health owns deeper medical workflows.".to_string()
            />

            <div class="two-up">
                <section class="vault-card">
                    <SectionTitle title="Recent biometrics" />
                    <div class="biometric-list">
                        <For
                            each=move || ctx.biometrics.get()
                            key=|item| item.hash.clone()
                            children=move |item| view! {
                                <div class="metric-row">
                                    <span class="metric-name">{item.metric_type}</span>
                                    <span class="metric-value">{format!("{} {}", item.value, item.unit)}</span>
                                </div>
                            }
                        />
                    </div>
                </section>

                <section class="vault-card">
                    <SectionTitle title="Consent grants" />
                    <form class="profile-form" on:submit=create_consent>
                        <label class="field-block">
                            <span>"Grantee key (base64 raw39)"</span>
                            <input
                                class="form-input"
                                prop:value=move || consent_grantee.get()
                                on:input=move |ev| consent_grantee.set(event_target_value(&ev))
                            />
                        </label>
                        <label class="field-block">
                            <span>"Record types"</span>
                            <input
                                class="form-input"
                                prop:value=move || consent_types.get()
                                on:input=move |ev| consent_types.set(event_target_value(&ev))
                            />
                        </label>
                        <div class="form-actions">
                            <button class="btn btn-primary" type="submit">"Create Consent"</button>
                        </div>
                    </form>
                    <div class="consent-list">
                        <For
                            each=move || ctx.consents.get()
                            key=|grant| grant.hash.clone()
                            children=move |grant| view! { <ConsentCard grant=grant /> }
                        />
                    </div>
                </section>
            </div>
        </div>
    }
}

#[component]
fn PreferencesPage() -> impl IntoView {
    let ctx = use_personal();

    view! {
        <div class="stack-page">
            <PageHeader
                eyebrow="Preferences"
                title="Sharing posture and disclosure policy"
                summary=ctx.status_note.get_untracked()
            />

            <div class="two-up">
                <section class="vault-card">
                    <SectionTitle title="Domain posture" />
                    <div class="preference-list">
                        <For
                            each=move || ctx.preferences.get()
                            key=|pref| format!("{}-{}", pref.source_cluster, pref.target_cluster)
                            children=move |pref| view! { <PreferenceCard pref=pref /> }
                        />
                    </div>
                </section>

                <section class="vault-card">
                    <SectionTitle title="Change log" />
                    <div class="preference-list">
                        <For
                            each=move || ctx.preference_log.get()
                            key=|log| format!("{}-{}-{}", log.source_cluster, log.target_cluster, log.changed_at)
                            children=move |log| view! {
                                <div class="mini-card">
                                    <div class="mini-card-header">
                                        <strong>{format!("{} -> {}", log.source_cluster, log.target_cluster)}</strong>
                                        <span class="status-pill" class:status-pill-active=log.now_allowed>
                                            {if log.now_allowed { "Allowed" } else { "Blocked" }}
                                        </span>
                                    </div>
                                    <p class="mini-card-meta">
                                        {format!("Was allowed: {}", if log.was_allowed { "yes" } else { "no" })}
                                    </p>
                                </div>
                            }
                        />
                    </div>
                </section>
            </div>
        </div>
    }
}

#[component]
fn ActivityPage() -> impl IntoView {
    let ctx = use_personal();

    view! {
        <div class="stack-page">
            <PageHeader
                eyebrow="Activity"
                title="What left the vault, and why"
                summary="This page will eventually reflect bridge queries, events, and disclosures directly.".to_string()
            />
            <section class="vault-card">
                <SectionTitle title="Disclosure and handoff log" />
                <ul class="activity-list">
                    <For
                        each=move || ctx.activity.get()
                        key=|entry| entry.id.clone()
                        children=move |entry| view! { <ActivityItemCard item=entry wide=true /> }
                    />
                </ul>
            </section>
        </div>
    }
}

#[component]
fn UnlockPage() -> impl IntoView {
    view! {
        <div class="stack-page">
            <PageHeader
                eyebrow="Unlock"
                title="Vault entry surface"
                summary="Biometric and passphrase unlock belongs here once Personal runtime security flows are wired.".to_string()
            />
            <section class="vault-card narrow-card">
                <p class="supporting-copy">
                    "The route now exists so portal and mobile wrappers have a stable unlock target. "
                    "Secure unlock implementation should follow after the conductor-facing Personal view layer."
                </p>
            </section>
        </div>
    }
}

#[component]
fn PageHeader(eyebrow: &'static str, title: &'static str, summary: String) -> impl IntoView {
    view! {
        <header class="page-header">
            <span class="page-eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <p>{summary}</p>
        </header>
    }
}

#[component]
fn SectionTitle(title: &'static str) -> impl IntoView {
    view! { <h2 class="section-title">{title}</h2> }
}

#[component]
fn VaultStat<F>(label: &'static str, value: F) -> impl IntoView
where
    F: Fn() -> String + 'static,
{
    view! {
        <div class="vault-stat">
            <span class="vault-stat-label">{label}</span>
            <strong class="vault-stat-value">{value()}</strong>
        </div>
    }
}

#[component]
fn KeyCard(key_data: personal_leptos_types::MasterKeyView) -> impl IntoView {
    view! {
        <div class="mini-card">
            <div class="mini-card-header">
                <strong>{key_data.label}</strong>
                <span class="status-pill" class:status-pill-active=key_data.active>
                    {if key_data.active { "Active" } else { "Inactive" }}
                </span>
            </div>
            <p class="mini-card-meta">{format!("Purpose: {}", key_data.purpose)}</p>
            <code class="hash-line">{key_data.public_key_hex}</code>
        </div>
    }
}

#[component]
fn CredentialCard(credential: StoredCredentialView) -> impl IntoView {
    let kind_label = credential.credential_type.label().to_string();
    let tone = match credential.credential_type {
        CredentialType::Identity => "tone-identity",
        CredentialType::Health => "tone-health",
        CredentialType::FederatedLearning => "tone-learning",
        CredentialType::Governance => "tone-governance",
        CredentialType::Domain(_) => "tone-domain",
    };

    view! {
        <article class=format!("credential-card {}", tone)>
            <div class="credential-topline">
                <span class="credential-kind">{kind_label}</span>
                <span class="status-pill" class:status-pill-active=!credential.revoked>
                    {if credential.revoked { "Revoked" } else { "Active" }}
                </span>
            </div>
            <strong class="credential-issuer">{credential.issuer}</strong>
            <p class="supporting-copy">
                {match credential.expires_at {
                    Some(_) => "Portable credential with an explicit expiry window.",
                    None => "Portable credential with no current expiry recorded.",
                }}
            </p>
            <code class="hash-line">{credential.hash}</code>
        </article>
    }
}

#[component]
fn ConsentCard(grant: ConsentGrantView) -> impl IntoView {
    view! {
        <article class="mini-card">
            <div class="mini-card-header">
                <strong>{grant.grantee}</strong>
                <span class="status-pill" class:status-pill-active=grant.active>
                    {if grant.active { "Active" } else { "Inactive" }}
                </span>
            </div>
            <p class="mini-card-meta">{format!("Types: {}", grant.record_types.join(", "))}</p>
        </article>
    }
}

#[component]
fn PreferenceCard(pref: personal_leptos_types::DataSharingPreferenceView) -> impl IntoView {
    let ctx = use_personal();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();
    let toasts = use_toasts();
    let local_pref = RwSignal::new(pref);
    let blocked_zomes_text = RwSignal::new(local_pref.get_untracked().blocked_zomes.join(", "));
    let toggle_ctx = ctx.clone();
    let toggle_hc = hc.clone();
    let toggle_toasts = toasts.clone();
    let save_ctx = ctx.clone();
    let save_hc = hc.clone();
    let save_toasts = toasts.clone();

    let toggle = move |_| {
        let previous = local_pref.get();
        let mut next = local_pref.get();
        next.allowed = !next.allowed;
        local_pref.set(next.clone());
        blocked_zomes_text.set(next.blocked_zomes.join(", "));

        let hc = toggle_hc.clone();
        let toasts = toggle_toasts.clone();
        let ctx = toggle_ctx.clone();
        let local_pref_signal = local_pref;
        let blocked_zomes_signal = blocked_zomes_text;
        spawn_local(async move {
            match hc
                .call_zome_default::<_, String>("data_preferences", "set_preference_view", &next)
                .await
            {
                Ok(_) => {
                    refresh_preferences_state(ctx.clone(), hc.clone()).await;
                    let state = if next.allowed { "allowed" } else { "blocked" };
                    toasts.push(
                        format!(
                            "{} -> {} now {}",
                            next.source_cluster, next.target_cluster, state
                        ),
                        ToastKind::Success,
                    );
                }
                Err(err) => {
                    local_pref_signal.set(previous.clone());
                    blocked_zomes_signal.set(previous.blocked_zomes.join(", "));
                    toasts.push(format!("Preference update failed: {err}"), ToastKind::Error);
                }
            }
        });
    };

    let save_details = move |_| {
        let previous = local_pref.get();
        let mut next = local_pref.get();
        next.blocked_zomes = blocked_zomes_text
            .get()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect();
        local_pref.set(next.clone());

        let hc = save_hc.clone();
        let toasts = save_toasts.clone();
        let ctx = save_ctx.clone();
        let local_pref_signal = local_pref;
        let blocked_zomes_signal = blocked_zomes_text;
        spawn_local(async move {
            match hc
                .call_zome_default::<_, String>("data_preferences", "set_preference_view", &next)
                .await
            {
                Ok(_) => {
                    refresh_preferences_state(ctx.clone(), hc.clone()).await;
                    let state = if next.allowed { "allowed" } else { "blocked" };
                    toasts.push(
                        format!(
                            "{} -> {} now {}",
                            next.source_cluster, next.target_cluster, state
                        ),
                        ToastKind::Success,
                    );
                }
                Err(err) => {
                    local_pref_signal.set(previous.clone());
                    blocked_zomes_signal.set(previous.blocked_zomes.join(", "));
                    toasts.push(format!("Preference update failed: {err}"), ToastKind::Error);
                }
            }
        });
    };

    view! {
        <article class="mini-card">
            <div class="mini-card-header">
                <strong>{move || format!(
                    "{} -> {}",
                    local_pref.get().source_cluster,
                    local_pref.get().target_cluster
                )}</strong>
                <span class="status-pill" class:status-pill-active=move || local_pref.get().allowed>
                    {move || if local_pref.get().allowed { "Allowed" } else { "Blocked" }}
                </span>
            </div>
            <label class="field-block preference-field">
                <span>"Reason"</span>
                <textarea
                    class="form-textarea preference-textarea"
                    prop:value=move || local_pref.get().reason
                    on:input=move |ev| {
                        let value = event_target_value(&ev);
                        local_pref.update(|pref| pref.reason = value);
                    }
                />
            </label>
            <label class="field-block preference-field">
                <span>"Blocked zomes"</span>
                <input
                    class="form-input"
                    prop:value=move || blocked_zomes_text.get()
                    placeholder="lab_result, claims, records"
                    on:input=move |ev| {
                        blocked_zomes_text.set(event_target_value(&ev));
                    }
                />
            </label>
            <div class="form-actions">
                <button class="btn" on:click=toggle>
                    {move || if local_pref.get().allowed { "Block Flow" } else { "Allow Flow" }}
                </button>
                <button class="btn btn-primary" on:click=save_details>
                    "Save Details"
                </button>
            </div>
        </article>
    }
}

#[component]
fn EmbeddedSatellite(name: &'static str, port: u16) -> impl IntoView {
    let url = format!("http://localhost:{}", port);

    view! {
        <div class="embedded-satellite-container">
            <div class="embedded-header">
                <span class="embedded-title">{format!("{} Satellite", name)}</span>
                <span class="telemetry-badge">"STANDALONE"</span>
            </div>
            <iframe
                src=url
                title=name
                class="embedded-satellite-frame"
                style="width: 100%; height: calc(100vh - 120px); border: none; background: var(--bg-surface);"
            ></iframe>
        </div>
    }
}

#[component]
fn ActivityItemCard(item: personal_leptos_types::ActivityItemView, wide: bool) -> impl IntoView {
    let class_name = if wide {
        "activity-item activity-item-wide"
    } else {
        "activity-item"
    };

    view! {
        <li class=class_name>
            <div class="mini-card-header">
                <strong>{format!("{} · {}", item.domain, item.title)}</strong>
                {item.success.map(|ok| {
                    view! {
                        <span class="status-pill" class:status-pill-active=ok>
                            {if ok { "Success" } else { "Blocked" }}
                        </span>
                    }
                })}
            </div>
            <p class="mini-card-meta">{item.detail}</p>
        </li>
    }
}
