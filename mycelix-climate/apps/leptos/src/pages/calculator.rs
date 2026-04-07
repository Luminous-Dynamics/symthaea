// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Carbon footprint calculator — interactive, shareable.
//!
//! Converts everyday activities (electricity, driving, flights) into
//! Scope 1/2/3 emissions. South Africa emission factors.

use leptos::prelude::*;
use crate::actions;

// SA emission factors (2024 estimates)
const ELECTRICITY_FACTOR: f64 = 0.95;  // kg CO2/kWh (Eskom grid)
const PETROL_FACTOR: f64 = 2.31;       // kg CO2/litre
const DIESEL_FACTOR: f64 = 2.68;       // kg CO2/litre
const AVG_FUEL_CONSUMPTION: f64 = 8.0; // litres/100km (SA average)
const FLIGHT_FACTOR: f64 = 0.255;      // kg CO2/km (economy)

#[component]
pub fn CalculatorPage() -> impl IntoView {
    let (electricity_kwh, set_electricity) = signal(String::new());
    let (driving_km, set_driving) = signal(String::new());
    let (flights_km, set_flights) = signal(String::new());
    let (gas_kg, set_gas) = signal(String::new());
    let (calculated, set_calculated) = signal(false);

    let scope1 = move || {
        let driving: f64 = driving_km.get().parse().unwrap_or(0.0);
        let gas: f64 = gas_kg.get().parse().unwrap_or(0.0);
        let driving_litres = driving / 100.0 * AVG_FUEL_CONSUMPTION;
        (driving_litres * PETROL_FACTOR + gas * 2.75) / 1000.0 // tonnes
    };

    let scope2 = move || {
        let kwh: f64 = electricity_kwh.get().parse().unwrap_or(0.0);
        kwh * ELECTRICITY_FACTOR / 1000.0 // tonnes
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
            actions::record_footprint(s1, s2, s3, "Calculator (SA factors)".into());
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
                    <label>"Electricity (kWh/year)"</label>
                    <input class="form-input" type="number" step="100" min="0" placeholder="4800 (SA avg household)"
                        prop:value=move || electricity_kwh.get()
                        on:input=move |ev| { set_electricity.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"Check your Eskom bill — monthly kWh \u{00D7} 12"</span>
                </div>

                <div class="calc-field">
                    <label>"Driving (km/year)"</label>
                    <input class="form-input" type="number" step="500" min="0" placeholder="15000 (SA avg)"
                        prop:value=move || driving_km.get()
                        on:input=move |ev| { set_driving.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"Your car odometer: this year minus last year"</span>
                </div>

                <div class="calc-field">
                    <label>"Flights (km/year)"</label>
                    <input class="form-input" type="number" step="500" min="0" placeholder="0"
                        prop:value=move || flights_km.get()
                        on:input=move |ev| { set_flights.set(event_target_value(&ev)); set_calculated.set(false); }
                    />
                    <span class="calc-hint">"JNB\u{2192}CPT = 1,270 km (one way)"</span>
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
                let sa_avg = 7.5; // tonnes CO2/person/year SA average
                let pct = if sa_avg > 0.0 { t / sa_avg * 100.0 } else { 0.0 };

                view! {
                    <div class="calc-results">
                        <div class="calc-total">
                            <span class="calc-total-value">{format!("{t:.1}")}</span>
                            <span class="calc-total-unit">"tonnes CO2e/year"</span>
                        </div>

                        <div class="calc-comparison">
                            {if pct < 80.0 {
                                format!("Below SA average ({sa_avg:.1}t) \u{2014} {pct:.0}% of average")
                            } else if pct < 120.0 {
                                format!("Near SA average ({sa_avg:.1}t)")
                            } else {
                                format!("Above SA average ({sa_avg:.1}t) \u{2014} {pct:.0}% of average")
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
                        </div>
                    </div>
                }
            })}
        </div>
    }
}
