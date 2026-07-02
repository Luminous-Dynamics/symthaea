// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Equation Explorer — interactive physics equation browser with structural search.
//!
//! Students browse the 208-equation catalog, explore structural similarity,
//! and challenge themselves to identify equation domains from skeletons.
//!
//! Uses the symthaea-physics-catalog WASM crate for client-side HDC search.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

/// Equation data for display.
#[derive(Clone)]
struct EquationInfo {
    name: String,
    domain: String,
    latex: String,
}

/// All equations from the catalog (simplified — in production, import from WASM crate).
fn all_equations() -> Vec<EquationInfo> {
    // Representative subset for the game
    vec![
        EquationInfo { name: "Newton Second Law".into(), domain: "Classical Mechanics".into(), latex: r"F = ma".into() },
        EquationInfo { name: "Mass-Energy Equivalence".into(), domain: "General Relativity".into(), latex: r"E = mc^2".into() },
        EquationInfo { name: "Schrodinger Equation".into(), domain: "Quantum Mechanics".into(), latex: r"i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi".into() },
        EquationInfo { name: "Maxwell Gauss Law".into(), domain: "Electromagnetism".into(), latex: r"\nabla \cdot E = \rho/\varepsilon_0".into() },
        EquationInfo { name: "Shannon Entropy".into(), domain: "Information Theory".into(), latex: r"H(X) = -\sum p(x)\log p(x)".into() },
        EquationInfo { name: "Navier-Stokes".into(), domain: "Fluid Dynamics".into(), latex: r"\rho(\partial_t v + v\cdot\nabla v) = -\nabla p + \mu\nabla^2 v".into() },
        EquationInfo { name: "Boltzmann Entropy".into(), domain: "Statistical Mechanics".into(), latex: r"S = k_B \ln W".into() },
        EquationInfo { name: "Hodgkin-Huxley".into(), domain: "Biophysics".into(), latex: r"C\frac{dV}{dt} = -g_{Na}m^3h(V-E_{Na}) - g_Kn^4(V-E_K) + I".into() },
        EquationInfo { name: "BCS Gap Equation".into(), domain: "Condensed Matter".into(), latex: r"\Delta = 2\hbar\omega_D e^{-1/(N(0)V)}".into() },
        EquationInfo { name: "Yukawa Potential".into(), domain: "Nuclear Physics".into(), latex: r"V(r) = -g^2 e^{-\mu r}/(4\pi r)".into() },
        EquationInfo { name: "Friedmann Equation".into(), domain: "Cosmology".into(), latex: r"H^2 = \frac{8\pi G\rho}{3}".into() },
        EquationInfo { name: "Ideal Gas Law".into(), domain: "Thermodynamics".into(), latex: r"PV = nRT".into() },
        EquationInfo { name: "Chandrasekhar Limit".into(), domain: "Astrophysics".into(), latex: r"M_{Ch} \approx 1.4 M_\odot".into() },
        EquationInfo { name: "Debye Length".into(), domain: "Plasma Physics".into(), latex: r"\lambda_D = \sqrt{\varepsilon_0 kT / (ne^2)}".into() },
        EquationInfo { name: "Snell Law".into(), domain: "Optics".into(), latex: r"n_1\sin\theta_1 = n_2\sin\theta_2".into() },
        EquationInfo { name: "Euler Identity".into(), domain: "Mathematics".into(), latex: r"e^{i\pi} + 1 = 0".into() },
        EquationInfo { name: "Doppler Effect".into(), domain: "Acoustics".into(), latex: r"f' = f\frac{v \pm v_{obs}}{v \mp v_{src}}".into() },
        EquationInfo { name: "Drake Equation".into(), domain: "Astrophysics".into(), latex: r"N = R_* \times f_p \times n_e \times f_l \times f_i \times f_c \times L".into() },
    ]
}

/// Domain challenge: guess which domain an equation belongs to.
fn domain_challenge_options() -> Vec<&'static str> {
    vec![
        "Classical Mechanics", "Electromagnetism", "Quantum Mechanics",
        "General Relativity", "Thermodynamics", "Fluid Dynamics",
        "Information Theory", "Nuclear Physics", "Condensed Matter",
        "Biophysics", "Cosmology", "Astrophysics", "Optics",
        "Plasma Physics", "Mathematics", "Acoustics",
    ]
}

/// Equation Explorer component.
#[component]
pub fn EquationExplorer(node_id: String) -> impl IntoView {
    let equations = RwSignal::new(all_equations());
    let (selected_idx, set_selected_idx) = signal(0usize);
    let (search_text, set_search_text) = signal(String::new());
    let (mode, set_mode) = signal("browse".to_string());
    let (challenge_idx, set_challenge_idx) = signal(0usize);
    let (score, set_score) = signal(0usize);
    let (total, set_total) = signal(0usize);

    let filtered = move || {
        let eqs = equations.get();
        let search = search_text.get().to_lowercase();
        if search.is_empty() {
            eqs
        } else {
            eqs.into_iter()
                .filter(|e| e.name.to_lowercase().contains(&search) || e.domain.to_lowercase().contains(&search))
                .collect()
        }
    };

    let challenge_eq = move || {
        let eqs = equations.get();
        let idx = challenge_idx.get() % eqs.len().max(1);
        eqs.get(idx).cloned()
    };

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 1rem; color: var(--accent-color, #6366f1);">"Physics Equation Explorer"</h2>

            // Mode selector
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                <button
                    style=move || if mode.get() == "browse" { "padding: 0.5rem 1rem; background: #6366f1; color: white; border: none; border-radius: 0.375rem; cursor: pointer;" } else { "padding: 0.5rem 1rem; background: #1f2937; color: #94a3b8; border: 1px solid #374151; border-radius: 0.375rem; cursor: pointer;" }
                    on:click=move |_| set_mode.set("browse".into())
                >"Browse Equations"</button>
                <button
                    style=move || if mode.get() == "challenge" { "padding: 0.5rem 1rem; background: #10b981; color: white; border: none; border-radius: 0.375rem; cursor: pointer;" } else { "padding: 0.5rem 1rem; background: #1f2937; color: #94a3b8; border: 1px solid #374151; border-radius: 0.375rem; cursor: pointer;" }
                    on:click=move |_| { set_mode.set("challenge".into()); set_challenge_idx.set(0); set_score.set(0); set_total.set(0); }
                >"Domain Challenge"</button>
            </div>

            {move || {
                if mode.get() == "challenge" {
                    // Challenge mode: guess the domain
                    let eq = challenge_eq();
                    view! {
                        <div style="background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.75rem; padding: 1.5rem;">
                            <div style="text-align: center; margin-bottom: 1rem;">
                                <span style="font-size: 0.875rem; color: #94a3b8;">"Score: "</span>
                                <span style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{move || score.get()}</span>
                                <span style="font-size: 0.875rem; color: #94a3b8;">" / " {move || total.get()}</span>
                            </div>

                            {eq.map(|e| {
                                let correct_domain = e.domain.clone();
                                view! {
                                    <div style="text-align: center; margin-bottom: 1.5rem;">
                                        <h3 style="font-size: 1.25rem; margin-bottom: 0.5rem;">{e.name.clone()}</h3>
                                        <code style="font-size: 1rem; color: #10b981;">{e.latex.clone()}</code>
                                    </div>
                                    <p style="text-align: center; font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">"Which domain does this equation belong to?"</p>
                                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem;">
                                        {domain_challenge_options().into_iter().map({
                                            let correct = correct_domain.clone();
                                            move |opt| {
                                                let is_correct = opt == correct.as_str();
                                                view! {
                                                    <button
                                                        style="padding: 0.5rem; background: #1f2937; color: #e2e8f0; border: 1px solid #374151; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem;"
                                                        on:click=move |_| {
                                                            set_total.update(|t| *t += 1);
                                                            if is_correct {
                                                                set_score.update(|s| *s += 1);
                                                            }
                                                            set_challenge_idx.update(|i| *i += 1);
                                                        }
                                                    >{opt}</button>
                                                }
                                            }
                                        }).collect::<Vec<_>>()}
                                    </div>
                                }
                            })}
                        </div>
                    }.into_any()
                } else {
                    // Browse mode
                    view! {
                        <div>
                            <input
                                type="text"
                                placeholder="Search equations..."
                                style="width: 100%; padding: 0.5rem 0.75rem; background: #1f2937; border: 1px solid #374151; border-radius: 0.375rem; color: #e2e8f0; font-size: 0.875rem; margin-bottom: 1rem;"
                                on:input=move |ev| {
                                    let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                                    set_search_text.set(target.value());
                                }
                            />
                            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.5rem;">
                                {move || filtered().into_iter().enumerate().map(|(i, eq)| {
                                    view! {
                                        <div
                                            style="padding: 0.75rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; cursor: pointer;"
                                            on:click=move |_| set_selected_idx.set(i)
                                        >
                                            <div style="font-weight: 600; font-size: 0.875rem; margin-bottom: 0.25rem;">{eq.name}</div>
                                            <div style="font-size: 0.75rem; color: #6366f1;">{eq.domain}</div>
                                            <code style="font-size: 0.7rem; color: #94a3b8; display: block; margin-top: 0.25rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                                {eq.latex}
                                            </code>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
