// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Root application component with state providers.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::components::search_bar::SearchBar;
use crate::components::security_badge::SecurityBadge;
use crate::components::theme_switcher::ThemeSwitcher;
use crate::holochain::DhtStatusBadge;
use crate::pages::content_router::ContentRouter;
use crate::state::{BrowserState, PageView};
use plexus_search::SearchEngine;
use plexus_reflex::ReflexArc;

const PRISM_ICON_MINI: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="24" height="24"><defs><linearGradient id="pg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#2DD4BF" stop-opacity="0.3"/><stop offset="100%" stop-color="#050507" stop-opacity="0.6"/></linearGradient></defs><polygon points="256,80 400,380 112,380" fill="url(#pg)" stroke="#2DD4BF" stroke-width="12"/><line x1="256" y1="80" x2="256" y2="380" stroke="#2DD4BF" stroke-width="8" opacity="0.9"/><circle cx="256" cy="460" r="16" fill="#2DD4BF" opacity="0.8"/></svg>"##;

const GEAR_ICON: &str = r##"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>"##;

#[component]
pub fn App() -> impl IntoView {
    let search_engine = StoredValue::new(SearchEngine::with_seed_claims());
    let reflex = StoredValue::new(ReflexArc::new());
    let state = BrowserState::new();

    provide_context(state.clone());
    provide_context(search_engine);
    provide_context(reflex);

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
        state_gear.set_current_url.set("prism://settings".to_string());
        state_gear.set_page_title.set("Settings".to_string());
        state_gear.set_view.set(PageView::Settings);
    };

    view! {
        <div class="chrome">
            <div class="chrome-top">
                <span class="brand-icon" inner_html=PRISM_ICON_MINI></span>
                <span class="brand">"Prism"</span>
                <span class="brand-sub">"by Symthaea"</span>
                <div style="flex:1"></div>
                <DhtStatusBadge />
                <ThemeSwitcher />
                <div class="gear-btn" on:click=open_settings title="Settings" inner_html=GEAR_ICON></div>
                <SecurityBadge />
            </div>
            <div class="search-row">
                <SearchBar />
            </div>
        </div>

        <div class="content-viewport">
            <ContentRouter />
        </div>
    }
}
