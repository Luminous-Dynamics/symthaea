// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mail app shell — provider stack + routing.

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes},
    path,
};

use crate::holochain::HolochainProvider;
use crate::mail_context::provide_mail_context;
use crate::toasts::{provide_toast_context, ToastContainer};
use crate::keyboard::provide_keyboard_context;
use crate::notifications::provide_notification_context;
use crate::theme::provide_theme_context;
use crate::offline::provide_offline_context;
use crate::preferences::provide_preferences_context;
use crate::pages::*;
use crate::holochain::ConnectionStatus;
use crate::components::{Nav, Sidebar, BottomNav, KeyboardHelp, CommandPalette, ThemePanel, WelcomeModal, ConductorSetup, OnboardingTour, ProfileSetup, PwaInstallPrompt};

#[component]
pub fn App() -> impl IntoView {
    view! {
        <HolochainProvider>
            <AppInner />
        </HolochainProvider>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    provide_mail_context();
    provide_toast_context();
    provide_preferences_context();
    provide_theme_context();
    provide_offline_context();
    provide_keyboard_context();
    provide_notification_context();
    crate::karma::provide_karma_context();

    // Favicon unread badge — updates tab title and favicon dynamically
    {
        let mail_fav = crate::mail_context::use_mail();
        Effect::new(move |_| {
            let unread = mail_fav.inbox.get().iter().filter(|e| !e.is_read).count();
            // Update document title
            if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                let title = if unread > 0 {
                    format!("({unread}) Mycelix Pulse")
                } else {
                    "Mycelix Pulse".into()
                };
                doc.set_title(&title);

                // Generate favicon with badge via canvas
                if unread > 0 {
                    let badge_label = if unread > 9 { "9+".into() } else { unread.to_string() };
                    let _ = js_sys::eval(&format!(
                        "((label)=>{{const c=document.createElement('canvas');c.width=32;c.height=32;const x=c.getContext('2d');\
                        x.font='20px sans-serif';x.textAlign='center';x.fillText('\\u2709',16,22);\
                        x.fillStyle='#ef4444';x.beginPath();x.arc(24,8,8,0,Math.PI*2);x.fill();\
                        x.fillStyle='#fff';x.font='bold 10px sans-serif';x.fillText(label,24,12);\
                        let l=document.querySelector('link[rel=icon]');if(!l){{l=document.createElement('link');l.rel='icon';document.head.appendChild(l)}}\
                        l.href=c.toDataURL()}})('{badge_label}')"
                    ));
                } else {
                    // Reset to default favicon
                    let _ = js_sys::eval(
                        "let l=document.querySelector('link[rel=icon]');if(l)l.href=\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>\\u2709</text></svg>\""
                    );
                }
            }
        });
    }

    let show_theme_panel = RwSignal::new(false);
    provide_context(show_theme_panel);

    view! {
        <Router>
            <a href="#main-content" class="skip-link">"Skip to main content"</a>
            <OfflineBanner />
            <Nav />
            <div class="app-layout">
                <Sidebar />
                <main id="main-content" class="main-content">
                    <Routes fallback=|| view! { <div class="page"><h1>"404 — Page not found"</h1></div> }>
                        <Route path=path!("/") view=InboxPage />
                        <Route path=path!("/compose") view=ComposePage />
                        <Route path=path!("/read/:id") view=ReadPage />
                        <Route path=path!("/contacts") view=ContactsPage />
                        <Route path=path!("/search") view=SearchPage />
                        <Route path=path!("/settings") view=SettingsPage />
                        <Route path=path!("/drafts") view=DraftsPage />
                        <Route path=path!("/accounts") view=AccountsPage />
                        <Route path=path!("/calendar") view=CalendarPage />
                        <Route path=path!("/chat") view=ChatPage />
                        <Route path=path!("/meet") view=MeetPage />
                    </Routes>
                </main>
            </div>
            <BottomNav />
            <KeyboardHelp />
            <CommandPalette />
            <ThemePanel show=show_theme_panel />
            <WelcomeModal />
            <ConductorSetup />
            <ProfileSetup />
            <PwaInstallPrompt />
            <OnboardingTour />
            <ToastContainer />
        </Router>
    }
}

/// Network/conductor status banner.
#[component]
fn OfflineBanner() -> impl IntoView {
    let hc = crate::holochain::use_holochain();
    let offline = crate::offline::use_offline();

    let is_online = RwSignal::new(
        web_sys::window().map(|w| w.navigator().on_line()).unwrap_or(true)
    );

    // Listen for online/offline events
    Effect::new(move |_| {
        let _ = js_sys::eval("
            window.__mycelix_online_handler = () => window.__mycelix_online = navigator.onLine;
            window.addEventListener('online', window.__mycelix_online_handler);
            window.addEventListener('offline', window.__mycelix_online_handler);
        ");
    });

    // Poll navigator.onLine (events don't trigger Leptos reactivity)
    let check_online = move || {
        web_sys::window().map(|w| w.navigator().on_line()).unwrap_or(true)
    };

    view! {
        // Network offline
        <div class="offline-banner network-offline"
             style=move || if check_online() { "display:none" } else { "" }>
            "\u{1F4F5} You're offline — changes will sync when reconnected"
            {move || {
                let q = offline.queue_size.get();
                (q > 0).then(|| view! { <span class="offline-queue-count">{format!(" ({q} queued)")}</span> })
            }}
        </div>
        // Conductor disconnected (but network is up)
        <div class="offline-banner conductor-offline"
             style=move || {
                 if !check_online() { return "display:none"; }
                 match hc.status.get() {
                     ConnectionStatus::Disconnected => "",
                     _ => "display:none",
                 }
             }>
            "\u{26A0} Conductor disconnected — reconnecting..."
        </div>
    }
}
