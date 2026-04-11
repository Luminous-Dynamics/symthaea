// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Notification sounds + favicon badge (#5).

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use crate::mail_context::use_mail;

#[derive(Clone, Copy)]
pub struct NotificationState {
    pub tab_visible: RwSignal<bool>,
    pub sound_enabled: RwSignal<bool>,
}

pub fn provide_notification_context() {
    let state = NotificationState {
        tab_visible: RwSignal::new(true),
        sound_enabled: RwSignal::new(true),
    };
    provide_context(state);

    // Track tab visibility
    let visible = state.tab_visible;
    let closure = Closure::<dyn Fn()>::new(move || {
        let doc = web_sys::window().unwrap().document().unwrap();
        let hidden = js_sys::Reflect::get(&doc, &JsValue::from_str("hidden"))
            .unwrap_or(JsValue::TRUE)
            .as_bool()
            .unwrap_or(true);
        visible.set(!hidden);
    });
    let doc = web_sys::window().unwrap().document().unwrap();
    let _ = doc.add_event_listener_with_callback(
        "visibilitychange",
        closure.as_ref().unchecked_ref(),
    );
    closure.forget();

    // Reactive title update based on unread count
    let mail = use_mail();
    Effect::new(move |_| {
        let unread = mail.inbox.get().iter().filter(|e| !e.is_read).count();
        let doc = web_sys::window().unwrap().document().unwrap();
        if unread > 0 {
            doc.set_title(&format!("Mycelix Pulse ({unread})"));
        } else {
            doc.set_title("Mycelix Pulse");
        }
    });
}

pub fn use_notifications() -> NotificationState {
    expect_context::<NotificationState>()
}

/// Request notification permission on first user interaction.
pub fn request_notification_permission() {
    let _ = js_sys::eval(
        "if('Notification' in window && Notification.permission==='default'){Notification.requestPermission()}"
    );
}

/// Show a desktop notification for a new email.
pub fn show_desktop_notification(sender: &str, subject: &str) {
    let sender = sender.replace('\'', "\\'").replace('\\', "\\\\");
    let subject = subject.replace('\'', "\\'").replace('\\', "\\\\");
    let js = format!(
        "if('Notification' in window && Notification.permission==='granted'){{new Notification('{}',{{body:'{}',icon:'/icons/icon-96.png',tag:'mycelix-pulse'}})}}",
        sender, subject
    );
    let _ = js_sys::eval(&js);
}

/// Play a short notification beep using Web Audio API.
pub fn play_notification_sound() {
    let state = use_notifications();
    if !state.sound_enabled.get_untracked() { return; }
    if state.tab_visible.get_untracked() { return; }

    // Use eval for simple Web Audio beep
    let _ = js_sys::eval(
        "try{const c=new(window.AudioContext||window.webkitAudioContext)();const o=c.createOscillator();const g=c.createGain();o.type='sine';o.frequency.value=880;g.gain.value=0.08;o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+0.1)}catch(e){}"
    );
}
