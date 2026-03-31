// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use gloo_timers::callback::Interval;

#[derive(Clone)]
pub struct ThermodynamicState {
    pub device_energy: ReadSignal<f64>,
    pub network_health: ReadSignal<f64>,
    pub torpor_level: ReadSignal<f64>,
}

pub fn provide_thermodynamic_context() -> ThermodynamicState {
    let (device_energy, set_device_energy) = signal(1.0_f64);
    let (network_health, _) = signal(1.0_f64);
    let (torpor_level, set_torpor_level) = signal(0.0_f64);
    spawn_local(async move {
        if let Some(battery) = try_get_battery().await {
            let level = battery.level();
            set_device_energy.set(level);
            if level < 0.15 { set_torpor_level.set(1.0 - level / 0.15); }
            let b = battery.clone();
            let _i = Interval::new(10_000, move || {
                let l = b.level();
                set_device_energy.set(l);
                set_torpor_level.set(if l < 0.15 { 1.0 - l / 0.15 } else { 0.0 });
            });
            std::mem::forget(_i);
        }
    });
    let state = ThermodynamicState { device_energy, network_health, torpor_level };
    provide_context(state.clone());
    Effect::new(move |_| {
        set_css("--device-energy", &format!("{:.3}", device_energy.get()));
        set_css("--torpor-level", &format!("{:.3}", torpor_level.get()));
    });
    state
}

pub fn use_thermodynamic() -> ThermodynamicState { expect_context::<ThermodynamicState>() }

fn set_css(name: &str, value: &str) {
    use wasm_bindgen::JsCast;
    if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.document_element()).and_then(|r| r.dyn_into::<web_sys::HtmlElement>().ok()) {
        let _ = el.style().set_property(name, value);
    }
}

async fn try_get_battery() -> Option<web_sys::BatteryManager> {
    let nav = web_sys::window()?.navigator();
    let f: js_sys::Function = js_sys::Reflect::get(&nav, &JsValue::from_str("getBattery")).ok()?.dyn_into().ok()?;
    let p: js_sys::Promise = f.call0(&nav).ok()?.dyn_into().ok()?;
    wasm_bindgen_futures::JsFuture::from(p).await.ok()?.dyn_into().ok()
}
