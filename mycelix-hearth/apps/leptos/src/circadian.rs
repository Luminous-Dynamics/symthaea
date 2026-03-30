// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Circadian awareness: the hearth knows what time of day it is.
//!
//! Morning = warmer, brighter tones. Evening = cooler, dimmer.
//! Night = deep warmth, minimal. The CSS shifts subtly with the sun.

use leptos::prelude::*;
use gloo_timers::callback::Interval;

/// Circadian phase based on local browser time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CircadianPhase {
    /// 5am-9am: golden morning warmth
    Dawn,
    /// 9am-5pm: bright, clear, full energy
    Day,
    /// 5pm-9pm: cooling, amber deepens
    Dusk,
    /// 9pm-5am: deep warmth, minimal, restful
    Night,
}

impl CircadianPhase {
    pub fn from_hour(hour: u32) -> Self {
        match hour {
            5..=8 => Self::Dawn,
            9..=16 => Self::Day,
            17..=20 => Self::Dusk,
            _ => Self::Night,
        }
    }

    /// Warmth multiplier (applied to --consciousness-warmth).
    pub fn warmth(&self) -> f64 {
        match self {
            Self::Dawn => 1.15,
            Self::Day => 1.0,
            Self::Dusk => 1.1,
            Self::Night => 0.85,
        }
    }

    /// Brightness multiplier (applied to text opacity).
    pub fn brightness(&self) -> f64 {
        match self {
            Self::Dawn => 0.95,
            Self::Day => 1.0,
            Self::Dusk => 0.9,
            Self::Night => 0.8,
        }
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct CircadianState {
    pub phase: ReadSignal<CircadianPhase>,
    pub hour: ReadSignal<u32>,
}

/// Initialize circadian awareness. Updates CSS vars every 60 seconds.
pub fn provide_circadian_context() -> CircadianState {
    let current_hour = get_local_hour();
    let (hour, set_hour) = signal(current_hour);
    let (phase, set_phase) = signal(CircadianPhase::from_hour(current_hour));

    // Update every 60 seconds
    let _interval = Interval::new(60_000, move || {
        let h = get_local_hour();
        set_hour.set(h);
        set_phase.set(CircadianPhase::from_hour(h));
    });
    std::mem::forget(_interval);

    // Apply CSS vars
    Effect::new(move |_| {
        let p = phase.get();
        set_css_var("--circadian-warmth", &format!("{:.3}", p.warmth()));
        set_css_var("--circadian-brightness", &format!("{:.3}", p.brightness()));
    });

    let state = CircadianState { phase, hour };
    provide_context(state.clone());
    state
}

fn get_local_hour() -> u32 {
    js_sys::Date::new_0().get_hours()
}

fn set_css_var(name: &str, value: &str) {
    use wasm_bindgen::JsCast;
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(root) = document.document_element() {
                if let Some(el) = root.dyn_ref::<web_sys::HtmlElement>() {
                    let _ = el.style().set_property(name, value);
                }
            }
        }
    }
}

#[allow(dead_code)]
pub fn use_circadian() -> CircadianState {
    expect_context::<CircadianState>()
}
