// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Thermodynamic UI coupling: device energy + network health -> CSS variables.
//!
//! The UI is physically constrained by its substrate. Low battery dims
//! visuals and slows animations. The organism enters torpor to conserve
//! energy. No error popups — just graceful degradation.
//!
//! # Browser Compatibility
//!
//! The Battery Status API (`navigator.getBattery()`) is a fingerprinting
//! vector. It works on Chrome/Android but is disabled in Safari and Firefox.
//! We catch the Promise rejection gracefully and default to full energy.

use crate::util::set_css_var;
use gloo_timers::callback::Interval;
use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

/// Torpor threshold: below this battery level (15%), the UI enters torpor.
const TORPOR_BATTERY_THRESHOLD: f64 = 0.15;

/// Polling interval for battery status (milliseconds).
const BATTERY_POLL_MS: u32 = 10_000;

/// Reactive state for thermodynamic coupling.
#[derive(Clone)]
pub struct ThermodynamicState {
    /// Device battery level 0.0-1.0 (1.0 if API unavailable).
    pub device_energy: ReadSignal<f64>,
    /// Network health 0.0-1.0 (1.0 = healthy, 0.0 = unreachable).
    pub network_health: ReadSignal<f64>,
    /// Collective Phi (MCE) 0.0-1.0 (global consciousness level).
    pub collective_phi: ReadSignal<f64>,
    /// Torpor level 0.0-1.0 (0.0 = fully active, 1.0 = deep torpor).
    pub torpor_level: ReadSignal<f64>,
    /// Whether the Battery API is available.
    pub battery_available: ReadSignal<bool>,
}

/// Initialize thermodynamic coupling and provide it as Leptos context.
pub fn provide_thermodynamic_context() -> ThermodynamicState {
    let (device_energy, set_device_energy) = signal(1.0_f64);
    let (network_health, _set_network_health) = signal(1.0_f64);
    let (collective_phi, _set_collective_phi) = signal(0.5_f64); // Baseline
    let (torpor_level, set_torpor_level) = signal(0.0_f64);
    let (battery_available, set_battery_available) = signal(false);

    spawn_local(async move {
        match try_get_battery().await {
            Some(battery) => {
                set_battery_available.set(true);
                let level = battery.level();
                set_device_energy.set(level);
                update_torpor(level, &set_torpor_level);

                let battery_clone = battery.clone();
                let _interval = Interval::new(BATTERY_POLL_MS, move || {
                    let level = battery_clone.level();
                    set_device_energy.set(level);
                    update_torpor(level, &set_torpor_level);
                });
                std::mem::forget(_interval);
            }
            None => {
                web_sys::console::log_1(
                    &"[Thermo] Battery API unavailable. Defaulting to full energy.".into(),
                );
                set_battery_available.set(false);
            }
        }
    });

    let state = ThermodynamicState {
        device_energy,
        network_health,
        collective_phi,
        torpor_level,
        battery_available,
    };

    provide_context(state.clone());

    // Wire CSS custom properties
    Effect::new(move |_| {
        set_css_var("--device-energy", &format!("{:.3}", device_energy.get()));
        set_css_var("--network-health", &format!("{:.3}", network_health.get()));
        set_css_var("--collective-phi", &format!("{:.3}", collective_phi.get()));
        set_css_var("--torpor-level", &format!("{:.3}", torpor_level.get()));
    });

    state
}

fn update_torpor(battery_level: f64, set_torpor: &WriteSignal<f64>) {
    if battery_level < TORPOR_BATTERY_THRESHOLD {
        let torpor = 1.0 - (battery_level / TORPOR_BATTERY_THRESHOLD);
        set_torpor.set(torpor.clamp(0.0, 1.0));
    } else {
        set_torpor.set(0.0);
    }
}

async fn try_get_battery() -> Option<web_sys::BatteryManager> {
    let window = web_sys::window()?;
    let navigator = window.navigator();
    let get_battery = js_sys::Reflect::get(&navigator, &JsValue::from_str("getBattery")).ok()?;
    if get_battery.is_undefined() || get_battery.is_null() {
        return None;
    }
    let func: js_sys::Function = get_battery.dyn_into().ok()?;
    let promise: js_sys::Promise = func.call0(&navigator).ok()?.dyn_into().ok()?;
    let result = wasm_bindgen_futures::JsFuture::from(promise).await.ok()?;
    result.dyn_into().ok()
}

pub fn use_thermodynamic() -> ThermodynamicState {
    expect_context::<ThermodynamicState>()
}
