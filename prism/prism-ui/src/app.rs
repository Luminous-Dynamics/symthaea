// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Root application component with state providers.

use leptos::prelude::LocalStorage;
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::components::search_bar::SearchBar;
use crate::components::security_badge::SecurityBadge;
use crate::components::theme_switcher::ThemeSwitcher;
use crate::holochain::DhtStatusBadge;
use crate::pages::content_router::ContentRouter;
use crate::state::{BrowserState, PageView};
use prism_reflex::ReflexArc;
use prism_search::SearchEngine;
use symthaea_spore::engine::SporeEngine;

/// Fetch a precomputed index file, returning None on any failure.
async fn fetch_index(path: &str) -> Option<SearchEngine> {
    let resp = gloo_net::http::Request::get(path).send().await.ok()?;
    if !resp.ok() {
        return None;
    }
    let bytes = resp.binary().await.ok()?;
    log::info!("Loaded index from {} ({} bytes)", path, bytes.len());
    Some(SearchEngine::from_precomputed(&bytes))
}

/// Two-phase index loading:
/// 1. Fetch core index (436 KB, ~189 curated claims) — instant
/// 2. Fetch full index (22 MB, ~10K claims) in background — merges when ready
async fn load_core_engine() -> SearchEngine {
    if let Some(engine) = fetch_index("/static/prism-index-core.bin").await {
        return engine;
    }
    log::info!("Core index unavailable, computing from curated claims");
    SearchEngine::with_core_claims()
}

const PRISM_ICON_MINI: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="24" height="24"><defs><linearGradient id="pg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#2DD4BF" stop-opacity="0.3"/><stop offset="100%" stop-color="#050507" stop-opacity="0.6"/></linearGradient></defs><polygon points="256,80 400,380 112,380" fill="url(#pg)" stroke="#2DD4BF" stroke-width="12"/><line x1="256" y1="80" x2="256" y2="380" stroke="#2DD4BF" stroke-width="8" opacity="0.9"/><circle cx="256" cy="460" r="16" fill="#2DD4BF" opacity="0.8"/></svg>"##;

const GEAR_ICON: &str = r##"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>"##;

const NAV_BACK: &str = r##"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>"##;
const NAV_FORWARD: &str = r##"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>"##;
const NAV_REFRESH: &str = r##"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>"##;
const NAV_HOME: &str = r##"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>"##;

#[component]
pub fn App() -> impl IntoView {
    let (engine_ready, set_engine_ready) = signal(false);
    let reflex = StoredValue::new(ReflexArc::new());
    let state = BrowserState::new();

    provide_context(state.clone());
    provide_context(reflex);

    // Initialize Symthaea Spore consciousness kernel
    let spore_engine = SporeEngine::new(Default::default());
    log::info!("Spore consciousness kernel initialized (cycle 0)");
    let spore: StoredValue<Option<SporeEngine>, LocalStorage> =
        StoredValue::new_local(Some(spore_engine));
    provide_context(spore);

    // Two-phase index loading:
    // Phase 1: Fetch core index (436 KB) — app becomes usable in <1s
    // Phase 2: Fetch full index (22 MB) in background — merges silently
    let engine_cell: StoredValue<Option<SearchEngine>> = StoredValue::new(None);
    provide_context(engine_cell);

    wasm_bindgen_futures::spawn_local(async move {
        // Phase 1: core index (instant)
        let core = load_core_engine().await;
        let core_count = core.claim_count();
        engine_cell.set_value(Some(core));
        set_engine_ready.set(true);
        log::info!("Phase 1: {} core claims ready", core_count);

        // Phase 2: full index (background, deferred)
        // The 23MB download + merge blocks the main thread and causes UI lag.
        // Defer to idle callback so the search bar remains responsive.
        wasm_bindgen_futures::spawn_local(async move {
            // Brief yield to let the UI settle before starting the big download
            gloo_timers::future::TimeoutFuture::new(500).await;

            if let Some(full) = fetch_index("/static/prism-index.bin").await {
                engine_cell.update_value(|opt| {
                    if let Some(engine) = opt {
                        engine.merge(full);
                        log::info!("Phase 2: merged to {} total claims", engine.claim_count());
                    }
                });
            }
        });
    });

    // Load persisted settings on mount
    Effect::new(move |_| {
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(el) = doc.document_element() {
                if let Some(theme) = crate::persistence::load::<String>("theme") {
                    let _ = el.set_attribute("data-theme", &theme);
                }
                if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
                    let style = html_el.style();
                    if let Some(fs) = crate::persistence::load::<f64>("font-size") {
                        let _ = style.set_property("--font-size", &format!("{}px", fs));
                    }
                    if let Some(lh) = crate::persistence::load::<f64>("line-height") {
                        let _ = style.set_property("--line-height", &format!("{}", lh));
                    }
                    if let Some(cw) = crate::persistence::load::<f64>("content-width") {
                        let _ = style.set_property("--content-width", &format!("{}px", cw));
                    }
                }
            }
        }
    });

    let state_gear = state.clone();
    let open_settings = move |_| {
        use leptos::prelude::Set;
        state_gear
            .set_current_url
            .set("prism://settings".to_string());
        state_gear.set_page_title.set("Settings".to_string());
        state_gear.set_view.set(PageView::Settings);
    };

    let loading_display = move || if engine_ready.get() { "none" } else { "flex" };
    let app_display = move || if engine_ready.get() { "block" } else { "none" };

    view! {
        <div class="loading-screen" style:display=loading_display>
            <img src="/static/prism-loading.jpg" alt="Prism" class="loading-prism" width="400" height="218" />
            <span class="brand">"Prism"</span>
            <p>"Loading epistemic index\u{2026}"</p>
        </div>

        <div class="app-root" style:display=app_display>
            <div class="chrome">
                <div class="chrome-top">
                    <span class="brand-icon" inner_html=PRISM_ICON_MINI></span>
                    <span class="brand">"Prism"</span>

                    // Navigation buttons
                    {
                        let s_back_cls = state.clone();
                        let s_back_click = state.clone();
                        let s_fwd_cls = state.clone();
                        let s_fwd_click = state.clone();
                        let s_home = state.clone();
                        let s_bookmarks = state.clone();
                        view! {
                            <div class="nav-buttons">
                                <button
                                    class=move || if s_back_cls.can_go_back() { "nav-btn" } else { "nav-btn disabled" }
                                    title="Back" attr:aria-label="Back"
                                    on:click=move |_| {
                                        if let Some(url) = s_back_click.go_back() {
                                            crate::engine::navigate_history(&url);
                                        }
                                    }
                                    inner_html=NAV_BACK
                                />
                                <button
                                    class=move || if s_fwd_cls.can_go_forward() { "nav-btn" } else { "nav-btn disabled" }
                                    title="Forward" attr:aria-label="Forward"
                                    on:click=move |_| {
                                        if let Some(url) = s_fwd_click.go_forward() {
                                            crate::engine::navigate_history(&url);
                                        }
                                    }
                                    inner_html=NAV_FORWARD
                                />
                                <button class="nav-btn" title="Refresh" attr:aria-label="Refresh page"
                                    on:click=move |_| {
                                        if let Some(window) = web_sys::window() {
                                            let _ = window.location().reload();
                                        }
                                    }
                                    inner_html=NAV_REFRESH
                                />
                                <button class="nav-btn" title="Home" attr:aria-label="Home"
                                    on:click=move |_| {
                                        s_home.set_view.set(PageView::Welcome);
                                        s_home.set_current_url.set("prism://welcome".to_string());
                                        s_home.set_page_title.set("Prism".to_string());
                                        s_home.push_history("prism://welcome", "Prism", &PageView::Welcome);
                                    }
                                    inner_html=NAV_HOME
                                />
                                <button class="nav-btn" title="Bookmarks" attr:aria-label="Bookmarks"
                                    on:click=move |_| {
                                        let view = PageView::Bookmarks;
                                        s_bookmarks.set_view.set(view.clone());
                                        s_bookmarks.set_current_url.set("prism://bookmarks".to_string());
                                        s_bookmarks.set_page_title.set("Bookmarks".to_string());
                                        s_bookmarks.push_history("prism://bookmarks", "Bookmarks", &view);
                                    }
                                >{"\u{2605}"}</button>
                            </div>
                        }
                    }

                    <div style="flex:1"></div>
                    <ConsciousnessBadge />
                    <DhtStatusBadge />
                    <ThemeSwitcher />
                    <button class="gear-btn" on:click=open_settings title="Settings" attr:aria-label="Settings" inner_html=GEAR_ICON></button>
                    <SecurityBadge />
                </div>
                <TabBar />
                <div class="search-row">
                    <SearchBar />
                    <RenderModeToggle />
                    <BookmarkButton />
                </div>
            </div>

            <div class="content-viewport">
                <ContentRouter />
            </div>
        </div>
    }
}

/// Displays the Spore consciousness level (Psi) as a small badge in the chrome.
#[component]
fn ConsciousnessBadge() -> impl IntoView {
    let state = expect_context::<BrowserState>();

    let psi_display = move || {
        let psi = state.consciousness.get();
        if psi > 0.001 {
            format!("\u{03A8} {:.0}%", psi * 100.0)
        } else {
            "\u{03A8} \u{2014}".to_string()
        }
    };

    let psi_class = move || {
        let psi = state.consciousness.get();
        if psi >= 0.6 {
            "consciousness-badge high"
        } else if psi >= 0.3 {
            "consciousness-badge mid"
        } else if psi > 0.001 {
            "consciousness-badge low"
        } else {
            "consciousness-badge dormant"
        }
    };

    let title = move || {
        let psi = state.consciousness.get();
        let conf = state.epistemic_confidence.get();
        format!(
            "Consciousness: {:.1}% | Confidence: {:.0}%",
            psi * 100.0,
            conf * 100.0
        )
    };

    view! {
        <span class=psi_class title=title attr:aria-label="Consciousness level">
            {psi_display}
        </span>
    }
}

const TAB_PLUS: &str = "+";
const TAB_CLOSE: &str = "\u{00D7}";

/// Tab bar — shows open tabs with close buttons and a new-tab button.
#[component]
fn TabBar() -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let s_new = state.clone();

    view! {
        <div class="tab-bar">
            <For
                each=move || state.tabs.get()
                key=|tab| tab.id
                children=move |tab| {
                    let id = tab.id;
                    let title = tab.title.clone();
                    let s_switch = expect_context::<BrowserState>();
                    let s_close = expect_context::<BrowserState>();
                    let s_active = expect_context::<BrowserState>();
                    let active_class = move || {
                        if s_active.active_tab.get() == id { "tab active" } else { "tab" }
                    };
                    view! {
                        <div class=active_class>
                            <span class="tab-title"
                                on:click=move |_| { s_switch.switch_tab(id); }
                            >{title}</span>
                            <span class="tab-close"
                                on:click=move |_| { s_close.close_tab(id); }
                            >{TAB_CLOSE}</span>
                        </div>
                    }
                }
            />
            <button class="tab-new" title="New tab" attr:aria-label="New tab"
                on:click=move |_| { s_new.new_tab(); }
            >{TAB_PLUS}</button>
        </div>
    }
}

const BOOKMARK_STAR: &str = "\u{2606}";
const BOOKMARK_STAR_FILLED: &str = "\u{2605}";

/// Bookmark button — toggles bookmark for the current page.
/// Toggle between Reader mode (sanitized, overlay works) and Full Page mode (real site).
#[component]
fn RenderModeToggle() -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let s_click = state.clone();

    let label = move || match state.render_mode.get() {
        crate::state::RenderMode::Reader => "Reader",
        crate::state::RenderMode::FullPage => "Full",
    };
    let title = move || match state.render_mode.get() {
        crate::state::RenderMode::Reader => "Reader mode: safe, with epistemic overlay",
        crate::state::RenderMode::FullPage => "Full page: real JS/CSS, no overlay",
    };
    let class = move || match state.render_mode.get() {
        crate::state::RenderMode::Reader => "render-mode-btn reader",
        crate::state::RenderMode::FullPage => "render-mode-btn fullpage",
    };

    view! {
        <button class=class title=title attr:aria-label="Toggle render mode"
            on:click=move |_| {
                use crate::state::RenderMode;
                let next = match s_click.render_mode.get() {
                    RenderMode::Reader => RenderMode::FullPage,
                    RenderMode::FullPage => RenderMode::Reader,
                };
                s_click.set_render_mode.set(next);
                crate::persistence::save("render-mode", &next);
            }
        >{label}</button>
    }
}

#[component]
fn BookmarkButton() -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let s_icon = state.clone();
    let s_class = state.clone();
    let s_click = state.clone();

    let icon = move || {
        if s_icon.is_bookmarked() {
            BOOKMARK_STAR_FILLED
        } else {
            BOOKMARK_STAR
        }
    };
    let class = move || {
        if s_class.is_bookmarked() {
            "bookmark-btn active"
        } else {
            "bookmark-btn"
        }
    };

    view! {
        <button class=class title="Bookmark this page" attr:aria-label="Bookmark"
            on:click=move |_| {
                if s_click.is_bookmarked() {
                    let url = s_click.current_url.get_untracked();
                    s_click.remove_bookmark(&url);
                } else {
                    s_click.add_bookmark();
                }
            }
        >{icon}</button>
    }
}
