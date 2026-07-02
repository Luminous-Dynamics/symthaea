// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Carbon footprint calculator — interactive, shareable.
//!
//! Converts everyday activities (electricity, driving, flights) into
//! Scope 1/2/3 emissions. South Africa emission factors.

use leptos::prelude::*;
use crate::actions;

// Grid emission factors by country (kg CO2/kWh, 2024 IEA estimates)
fn grid_factor(country: &str) -> f64 {
    match country {
        "ZA" => 0.95,  // South Africa (coal-heavy)
        "IN" => 0.71,  // India
        "CN" => 0.58,  // China
        "US" => 0.39,  // United States
        "DE" => 0.35,  // Germany
        "GB" => 0.21,  // United Kingdom
        "BR" => 0.07,  // Brazil (hydro-heavy)
        "FR" => 0.06,  // France (nuclear)
        "NO" => 0.02,  // Norway (hydro)
        "WORLD" => 0.49, // Global average
        _ => 0.49,
    }
}

fn avg_per_capita(country: &str) -> f64 {
    match country {
        "US" => 14.4, "AU" => 15.0, "CA" => 14.2, "DE" => 8.1,
        "GB" => 5.2, "CN" => 8.0, "IN" => 1.9, "BR" => 2.2,
        "ZA" => 7.5, "FR" => 4.5, "NO" => 7.5,
        "WORLD" => 4.7,
        _ => 4.7,
    }
}

const PETROL_FACTOR: f64 = 2.31;       // kg CO2/litre (universal)
const AVG_FUEL_CONSUMPTION: f64 = 8.0; // litres/100km (global avg)
const FLIGHT_FACTOR: f64 = 0.255;      // kg CO2/km (economy)

#[component]
pub fn CalculatorPage() -> impl IntoView {
    let (country, set_country) = signal("WORLD".to_string());
    let (electricity_kwh, set_electricity) = signal(String::new());
    let (driving_km, set_driving) = signal(String::new());
    let (flights_km, set_flights) = signal(String::new());
    let (gas_kg, set_gas) = signal(String::new());
    let (calculated, set_calculated) = signal(false);

    // Parse URL hash on load: #country=US&kwh=4800&km=15000&flights=5000&gas=0
    if let Some(hash) = web_sys::window()
        .and_then(|w| w.location().hash().ok())
        .filter(|h| h.len() > 1)
    {
        let params: std::collections::HashMap<String, String> = hash[1..]
            .split('&')
            .filter_map(|kv| {
                let mut parts = kv.splitn(2, '=');
                Some((parts.next()?.to_string(), parts.next()?.to_string()))
            })
            .collect();
        if let Some(v) = params.get("country") { set_country.set(v.clone()); }
        if let Some(v) = params.get("kwh") { set_electricity.set(v.clone()); }
        if let Some(v) = params.get("km") { set_driving.set(v.clone()); }
        if let Some(v) = params.get("flights") { set_flights.set(v.clone()); }
        if let Some(v) = params.get("gas") { set_gas.set(v.clone()); }
        if !params.is_empty() { set_calculated.set(true); }
    }

    let scope1 = move || {
        let driving: f64 = driving_km.get().parse().unwrap_or(0.0);
        let gas: f64 = gas_kg.get().parse().unwrap_or(0.0);
        let driving_litres = driving / 100.0 * AVG_FUEL_CONSUMPTION;
        (driving_litres * PETROL_FACTOR + gas * 2.75) / 1000.0 // tonnes
    };

    let scope2 = move || {
        let kwh: f64 = electricity_kwh.get().parse().unwrap_or(0.0);
        kwh * grid_factor(&country.get()) / 1000.0 // tonnes
    };

    let scope3 = move || {
        let flights: f64 = flights_km.get().parse().unwrap_or(0.0);
        flights * FLIGHT_FACTOR / 1000.0 // tonnes
    };

    let total = move || scope1() + scope2() + scope3();

    let on_calculate = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        set_calculated.set(true);
    };

    let on_save = move |_| {
        let s1 = scope1();
        let s2 = scope2();
        let s3 = scope3();
        if s1 + s2 + s3 > 0.0 {
            actions::record_footprint(s1, s2, s3, format!("Calculator ({} grid)", country.get_untracked()));
        }
    };

    view! {
        <div class="page-calculator">
            <section class="hero">
                <h1>"Carbon Calculator"</h1>
                <p class="hero-subtitle">"Estimate your annual carbon footprint in 30 seconds."</p>
            </section>

            <form class="calculator-form" on:submit=on_calculate>
                <div class="calc-field">
                    <label>"Your country"</label>
                    <select class="form-select"
                        prop:value=move || country.get()
                        on:change=move |ev| { set_country.set(event_target_value(&ev)); set_calculated.set(false); }
                    >
                        <option value="WORLD">"Global average"</option>
                        <option value="ZA">"South Africa"</option>
                        <option value="IN">"India"</option>
                        <option value="CN">"China"</option>
                        <option value="US">"United States"</option>
                        <option value="GB">"United Kingdom"</option>
                        <option value="DE">"Germany"</option>
                        <option value="BR">"Brazil"</option>
                        <option value="FR">"France"</option>
                        <option value="AU">"Australia"</option>
                        <option value="NO">"Norway"</option>
                    </select>
                    <span class="calc-hint">{move || format!("Grid factor: {} kg CO2/kWh", grid_factor(&country.get()))}</span>
                </div>

                <div class="calc-field">
                    <label>"Electricity (kWh/year)"</label>
                    <input class="form-input" type="number" step="100" min="0" placeholder="4800 (SA avg household)"
                        prop:value=move || electricity_kwh.get()
                        on:input=move |ev| { set_electricity.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"Check your utility bill \u{2014} monthly kWh \u{00D7} 12"</span>
                </div>

                <div class="calc-field">
                    <label>"Driving (km/year)"</label>
                    <input class="form-input" type="number" step="500" min="0" placeholder="15000 (SA avg)"
                        prop:value=move || driving_km.get()
                        on:input=move |ev| { set_driving.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"Odometer: this year minus last year"</span>
                </div>

                <div class="calc-field">
                    <label>"Flights (km/year)"</label>
                    <input class="form-input" type="number" step="500" min="0" placeholder="0"
                        prop:value=move || flights_km.get()
                        on:input=move |ev| { set_flights.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"NYC\u{2192}LON = 5,570 km, SYD\u{2192}SIN = 6,290 km"</span>
                </div>

                <div class="calc-field">
                    <label>"Gas/LPG (kg/year)"</label>
                    <input class="form-input" type="number" step="5" min="0" placeholder="0"
                        prop:value=move || gas_kg.get()
                        on:input=move |ev| { set_gas.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"9kg cylinder = 9, two per year = 18"</span>
                </div>

                <button type="submit" class="btn btn-primary">"Calculate"</button>
            </form>

            {move || calculated.get().then(|| {
                let s1 = scope1();
                let s2 = scope2();
                let s3 = scope3();
                let t = total();
                let c = country.get();
                let avg = avg_per_capita(&c);
                let pct = if avg > 0.0 { t / avg * 100.0 } else { 0.0 };
                let country_label = if c == "WORLD" { "global".to_string() } else { c.clone() };

                view! {
                    <div class="calc-results">
                        <div class="calc-total">
                            <span class="calc-total-value">{format!("{t:.1}")}</span>
                            <span class="calc-total-unit">"tonnes CO2e/year"</span>
                        </div>

                        <div class="calc-comparison">
                            {if pct < 80.0 {
                                format!("Below {country_label} average ({avg:.1}t) \u{2014} {pct:.0}%")
                            } else if pct < 120.0 {
                                format!("Near {country_label} average ({avg:.1}t)")
                            } else {
                                format!("Above {country_label} average ({avg:.1}t) \u{2014} {pct:.0}%")
                            }}
                        </div>

                        <div class="calc-breakdown">
                            <div class="scope-bar">
                                {(s1 > 0.0).then(|| view! {
                                    <div class="scope-fill scope1" style=format!("width: {}%", s1 / t * 100.0)></div>
                                })}
                                {(s2 > 0.0).then(|| view! {
                                    <div class="scope-fill scope2" style=format!("width: {}%", s2 / t * 100.0)></div>
                                })}
                                {(s3 > 0.0).then(|| view! {
                                    <div class="scope-fill scope3" style=format!("width: {}%", s3 / t * 100.0)></div>
                                })}
                            </div>
                            <div class="scope-legend">
                                <span class="scope-item"><span class="dot scope1-dot"></span>{format!("Scope 1: {s1:.2}t (driving, gas)")}</span>
                                <span class="scope-item"><span class="dot scope2-dot"></span>{format!("Scope 2: {s2:.2}t (electricity)")}</span>
                                <span class="scope-item"><span class="dot scope3-dot"></span>{format!("Scope 3: {s3:.2}t (flights)")}</span>
                            </div>
                        </div>

                        <div class="calc-actions">
                            <button class="btn btn-primary" on:click=on_save>
                                "Save to my profile"
                            </button>
                            <button class="btn btn-ghost" on:click=move |_| {
                                // Build shareable URL
                                if let Some(window) = web_sys::window() {
                                    let base = window.location().origin().unwrap_or_default();
                                    let hash = format!(
                                        "#country={}&kwh={}&km={}&flights={}&gas={}",
                                        country.get_untracked(),
                                        electricity_kwh.get_untracked(),
                                        driving_km.get_untracked(),
                                        flights_km.get_untracked(),
                                        gas_kg.get_untracked(),
                                    );
                                    let url = format!("{base}/calculator{hash}");
                                    // Copy to clipboard via JS eval
                                    let _ = js_sys::eval(&format!(
                                        "navigator.clipboard.writeText('{url}')"
                                    ));
                                    let toasts = mycelix_leptos_core::use_toasts();
                                    toasts.push("Link copied \u{2014} share your footprint!", mycelix_leptos_core::ToastKind::Success);
                                }
                            }>
                                "Share your footprint"
                            </button>
                        </div>
                    </div>
                }
            })}
        </div>
    }
}
