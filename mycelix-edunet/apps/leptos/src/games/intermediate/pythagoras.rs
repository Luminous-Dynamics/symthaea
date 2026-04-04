// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pythagorean Theorem Explorer — Gr8. Visual proof with interactive right triangle.

use leptos::prelude::*;

#[component]
pub fn PythagorasExplorer(node_id: String) -> impl IntoView {
    let (side_a, set_a) = signal(3.0f64);
    let (side_b, set_b) = signal(4.0f64);

    let hypotenuse = move || (side_a.get().powi(2) + side_b.get().powi(2)).sqrt();
    let a_sq = move || side_a.get().powi(2);
    let b_sq = move || side_b.get().powi(2);
    let c_sq = move || hypotenuse().powi(2);

    let scale = 30.0; // pixels per unit

    view! {
        <div class="game-container pythagoras-game">
            <h3>"Pythagorean Theorem Explorer"</h3>
            <p class="game-instructions">"Adjust the sides to see a\u{00b2} + b\u{00b2} = c\u{00b2} in action!"</p>

            <svg viewBox="0 0 500 350" class="game-svg" aria-label="Interactive Pythagorean theorem visualization">
                // Triangle
                {move || {
                    let a = side_a.get();
                    let b = side_b.get();
                    let c = hypotenuse();
                    let ox = 50.0;
                    let oy = 280.0;
                    let ax = ox + a * scale;
                    let by = oy - b * scale;

                    view! {
                        // Right triangle
                        <polygon
                            points=format!("{},{} {},{} {},{}", ox, oy, ax, oy, ox, by)
                            fill="var(--primary, #7c3aed)" opacity="0.15"
                            stroke="var(--primary)" stroke-width="2"
                        />

                        // Right angle marker
                        <rect x=ox y=oy-15.0 width="15" height="15"
                            fill="none" stroke="var(--text-secondary)" stroke-width="1" />

                        // Side labels
                        <text x=(ox+ax)/2.0 y=oy+20.0 text-anchor="middle" font-size="14" fill="var(--info, #3b82f6)" font-weight="bold">
                            {format!("a = {:.1}", a)}
                        </text>
                        <text x=ox-20.0 y=(oy+by)/2.0 text-anchor="middle" font-size="14" fill="var(--success, #22c55e)" font-weight="bold"
                            transform=format!("rotate(-90, {}, {})", ox-20.0, (oy+by)/2.0)>
                            {format!("b = {:.1}", b)}
                        </text>
                        <text x=(ox+ax)/2.0+10.0 y=(oy+by)/2.0-5.0 text-anchor="middle" font-size="14" fill="var(--error, #ef4444)" font-weight="bold">
                            {format!("c = {:.2}", c)}
                        </text>

                        // Square on side a (blue)
                        <rect x=ox y=oy width=a*scale height=a*scale
                            fill="var(--info)" opacity="0.2" stroke="var(--info)" stroke-width="1"
                            transform=format!("translate(0, 0)")
                        />
                        <text x=ox+a*scale/2.0 y=oy+a*scale/2.0+5.0 text-anchor="middle" font-size="12" fill="var(--info)">
                            {format!("a\u{00b2}={:.1}", a_sq())}
                        </text>

                        // Square on side b (green) — drawn to the left
                        <rect x=ox-b*scale y=by width=b*scale height=b*scale
                            fill="var(--success)" opacity="0.2" stroke="var(--success)" stroke-width="1"
                        />
                        <text x=ox-b*scale/2.0 y=by+b*scale/2.0+5.0 text-anchor="middle" font-size="12" fill="var(--success)">
                            {format!("b\u{00b2}={:.1}", b_sq())}
                        </text>
                    }
                }}
            </svg>

            // Equation display
            <div style="text-align: center; font-size: 1.4rem; font-family: monospace; margin: 0.5rem 0; font-weight: 700">
                {move || format!("{:.1}\u{00b2} + {:.1}\u{00b2} = {:.1} + {:.1} = {:.1}", side_a.get(), side_b.get(), a_sq(), b_sq(), c_sq())}
            </div>
            <div style="text-align: center; font-size: 1.2rem; color: var(--error)">
                {move || format!("c = \u{221a}{:.1} = {:.2}", c_sq(), hypotenuse())}
            </div>

            // Sliders
            <div class="game-controls" style="display: flex; gap: 2rem; justify-content: center; margin-top: 1rem">
                <div>
                    <label style="color: var(--info)">"Side a: " {move || format!("{:.1}", side_a.get())}</label>
                    <input type="range" min="1" max="10" step="0.5"
                        prop:value=move || side_a.get().to_string()
                        on:input=move |ev| {
                            use wasm_bindgen::JsCast;
                            let val: f64 = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse().unwrap_or(3.0);
                            set_a.set(val);
                        }
                    />
                </div>
                <div>
                    <label style="color: var(--success)">"Side b: " {move || format!("{:.1}", side_b.get())}</label>
                    <input type="range" min="1" max="10" step="0.5"
                        prop:value=move || side_b.get().to_string()
                        on:input=move |ev| {
                            use wasm_bindgen::JsCast;
                            let val: f64 = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value().parse().unwrap_or(4.0);
                            set_b.set(val);
                        }
                    />
                </div>
            </div>

            // Famous triples
            <div style="text-align: center; margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-secondary)">
                "Famous Pythagorean triples: "
                <button class="btn-secondary" style="font-size: 0.8rem; padding: 2px 8px" on:click=move |_| { set_a.set(3.0); set_b.set(4.0); }>"3-4-5"</button>
                " "
                <button class="btn-secondary" style="font-size: 0.8rem; padding: 2px 8px" on:click=move |_| { set_a.set(5.0); set_b.set(12.0); }>"5-12-13"</button>
                " "
                <button class="btn-secondary" style="font-size: 0.8rem; padding: 2px 8px" on:click=move |_| { set_a.set(8.0); set_b.set(6.0); }>"6-8-10"</button>
            </div>
        </div>
    }
}
