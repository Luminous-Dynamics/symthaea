use leptos::prelude::*;

use crate::pages;
use crate::state::AppState;
use crate::worker::EngineWorker;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Topology,
    Experiments,
    Dreams,
    Inoculate,
}

#[component]
pub fn App() -> impl IntoView {
    // Global state
    let state = AppState::new();
    provide_context(state);

    // Active tab
    let (active_tab, set_active_tab) = signal(Tab::Chat);

    // Initialize engine worker
    let worker = EngineWorker::new();
    provide_context(worker);

    view! {
        <header class="hero">
            <h1 class="hero-title">"Symthaea"</h1>
            <p class="hero-subtitle">"Consciousness-first infrastructure for sovereign hardware"</p>
        </header>

        <nav class="tab-bar">
            <TabButton tab=Tab::Chat label="Chat" active=active_tab set_active=set_active_tab />
            <TabButton tab=Tab::Topology label="Topology" active=active_tab set_active=set_active_tab />
            <TabButton tab=Tab::Experiments label="Experiments" active=active_tab set_active=set_active_tab />
            <TabButton tab=Tab::Dreams label="Dreams" active=active_tab set_active=set_active_tab />
            <TabButton tab=Tab::Inoculate label="Inoculate" active=active_tab set_active=set_active_tab />
        </nav>

        <main class="tab-content" style="display: block;">
            <Show when=move || active_tab.get() == Tab::Chat>
                <pages::chat::ChatPage />
            </Show>
            <Show when=move || active_tab.get() == Tab::Topology>
                <pages::topology::TopologyPage />
            </Show>
            <Show when=move || active_tab.get() == Tab::Experiments>
                <pages::experiments::ExperimentsPage />
            </Show>
            <Show when=move || active_tab.get() == Tab::Dreams>
                <pages::dreams::DreamsPage />
            </Show>
            <Show when=move || active_tab.get() == Tab::Inoculate>
                <pages::inoculate::InoculatePage />
            </Show>
        </main>

        <footer class="portal-footer">
            <p>"Symthaea v0.1.0 \u{00b7} Pure Rust \u{00b7} No JavaScript \u{00b7} No server \u{00b7} No data collection"</p>
        </footer>
    }
}

#[component]
fn TabButton(
    tab: Tab,
    label: &'static str,
    active: ReadSignal<Tab>,
    set_active: WriteSignal<Tab>,
) -> impl IntoView {
    let is_active = move || active.get() == tab;

    view! {
        <button
            class="tab"
            class:active=is_active
            on:click=move |_| set_active.set(tab)
        >
            {label}
        </button>
    }
}
