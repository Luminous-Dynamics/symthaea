// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-coupled UI — the app breathes with the student.
//!
//! Reads cognitive state signals from the adaptivity engine, applies
//! exponential moving average smoothing, and writes CSS custom properties
//! on `<html>` to create subliminal environmental adaptation.
//!
//! Changes happen over 4 seconds — the student feels the environment
//! becoming comfortable without identifying the exact moment of change.

use leptos::prelude::*;
use crate::adaptivity_provider::use_adaptivity;

/// EMA-smoothed consciousness values.
struct SmoothedState {
    warmth: f64,
    density: f64,
    chrome_opacity: f64,
    animation_speed: f64,
    accent_saturation: f64,
}

impl Default for SmoothedState {
    fn default() -> Self {
        Self {
            warmth: 0.0,
            density: 1.0,
            chrome_opacity: 1.0,
            animation_speed: 1.0,
            accent_saturation: 1.0,
        }
    }
}

impl SmoothedState {
    /// EMA update with alpha = 0.05 (slow, subliminal)
    fn update(&mut self, target: &TargetState) {
        const ALPHA: f64 = 0.05;
        self.warmth += ALPHA * (target.warmth - self.warmth);
        self.density += ALPHA * (target.density - self.density);
        self.chrome_opacity += ALPHA * (target.chrome_opacity - self.chrome_opacity);
        self.animation_speed += ALPHA * (target.animation_speed - self.animation_speed);
        self.accent_saturation += ALPHA * (target.accent_saturation - self.accent_saturation);
    }

    fn apply_to_dom(&self) {
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(el) = doc.document_element() {
                if let Some(style) = el.dyn_ref::<web_sys::HtmlElement>().map(|e| e.style()) {
                    let _ = style.set_property("--consciousness-warmth", &format!("{:.3}", self.warmth));
                    let _ = style.set_property("--consciousness-density", &format!("{:.3}", self.density));
                    let _ = style.set_property("--consciousness-chrome-opacity", &format!("{:.3}", self.chrome_opacity));
                    let _ = style.set_property("--consciousness-animation-speed", &format!("{:.3}", self.animation_speed));
                    let _ = style.set_property("--consciousness-accent-saturation", &format!("{:.3}", self.accent_saturation));
                }
            }
        }
    }
}

struct TargetState {
    warmth: f64,
    density: f64,
    chrome_opacity: f64,
    animation_speed: f64,
    accent_saturation: f64,
}

impl TargetState {
    fn from_cognitive_state(focus: f32, stress: f32, motivation: f32) -> Self {
        Self {
            // Warmth: stress increases warmth, high focus cools
            warmth: (stress as f64 * 0.7 - focus as f64 * 0.2).clamp(-1.0, 1.0),
            // Density: high focus = dense (compact UI), low focus = spacious
            density: 0.7 + focus as f64 * 0.3,
            // Chrome opacity: high focus = fade chrome, low = show everything
            chrome_opacity: 0.4 + (1.0 - focus as f64) * 0.6,
            // Animation speed: high motivation = snappy, low = calm
            animation_speed: 0.6 + motivation as f64 * 0.8,
            // Accent saturation: high motivation = vivid, low = muted
            accent_saturation: 0.5 + motivation as f64 * 0.5,
        }
    }
}

use wasm_bindgen::JsCast;

/// Initialize the consciousness-to-UI bridge.
/// Call once after AdaptivityProvider is set up.
pub fn init_consciousness_ui() {
    let adaptivity = use_adaptivity();
    let cognitive_state = adaptivity.cognitive_state;

    // Store the smoothed state in a cell for the interval closure
    let smoothed = std::cell::RefCell::new(SmoothedState::default());

    // Update at 4Hz (250ms intervals)
    let interval = gloo_timers::callback::Interval::new(250, move || {
        let cog = cognitive_state.get_untracked();
        let target = TargetState::from_cognitive_state(cog.focus, cog.stress, cog.motivation);

        let mut state = smoothed.borrow_mut();
        state.update(&target);
        state.apply_to_dom();
    });

    // Leak the interval handle (same pattern as consciousness.rs and learning_engine.rs)
    std::mem::forget(interval);
}
