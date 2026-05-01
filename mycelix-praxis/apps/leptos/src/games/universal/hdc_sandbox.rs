// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! HDC Logic Sandbox — Interactive lesson for Hyperdimensional Computing.
//! Teaches XOR binding, Superposition, and Permutation as the "physics of thought."

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

/// A simplified 256-bit HD vector for visual interaction.
/// In production, Symthaea uses 16,384 bits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcVector {
    pub bits: Vec<bool>,
}

impl HdcVector {
    pub fn random() -> Self {
        let mut bits = Vec::with_capacity(256);
        for _ in 0..256 {
            bits.push(rand::random());
        }
        Self { bits }
    }

    /// Binding (XOR)
    pub fn bind(&self, other: &Self) -> Self {
        let bits = self.bits.iter().zip(other.bits.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        Self { bits }
    }

    /// Superposition (Consensus Sum / Majority)
    pub fn superpose(vectors: &[&Self]) -> Self {
        let mut bits = Vec::with_capacity(256);
        for i in 0..256 {
            let mut set_count = 0;
            for v in vectors {
                if v.bits[i] { set_count += 1; }
            }
            // Simple majority rule for binary HDC
            bits.push(set_count > (vectors.len() / 2));
        }
        Self { bits }
    }

    /// Permutation (Cyclic Shift)
    pub fn permute(&self, shift: usize) -> Self {
        let mut bits = self.bits.clone();
        bits.rotate_left(shift % 256);
        Self { bits }
    }
}

#[component]
pub fn HdcSandbox() -> impl IntoView {
    // Lesson State
    let (v_role, set_v_role) = signal(HdcVector::random());
    let (v_subject, set_v_role_subject) = signal(HdcVector::random());
    let (v_memory, set_v_memory) = signal::<Option<HdcVector>>(None);
    
    // Step-by-step guidance
    let (current_step, set_step) = signal(1);

    view! {
        <div class="hdc-sandbox">
            <header class="sandbox-header">
                <h3>"HDC Logic: The Physics of Meaning"</h3>
                <div class="step-indicator">"Step " {current_step} " of 3"</div>
            </header>

            <div class="sandbox-layout">
                // 1. Visualizer Panel
                <div class="visualizer-panel">
                    <div class="vector-display">
                        <label>"Role Vector (e.g. Teacher)"</label>
                        <HdcVisualizer vector=v_role />
                        <button class="btn-refresh" on:click=move |_| set_v_role.set(HdcVector::random())>"\u{21BB} New Seed"</button>
                    </div>

                    <div class="vector-display">
                        <label>"Subject Vector (e.g. Math)"</label>
                        <HdcVisualizer vector=v_subject />
                        <button class="btn-refresh" on:click=move |_| set_v_role_subject.set(HdcVector::random())>"\u{21BB} New Seed"</button>
                    </div>

                    {move || v_memory.get().map(|v| {
                        view! {
                            <div class="vector-display memory">
                                <label>"Memory Vector (Result)"</label>
                                <HdcVisualizer vector=Signal::derive(move || v.clone()) />
                            </div>
                        }
                    })}
                </div>

                // 2. Control Panel (The Lesson)
                <div class="lesson-controls">
                    {move || match current_step.get() {
                        1 => view! {
                            <div class="lesson-step">
                                <h4>"\u{2297} Part 1: XOR Binding"</h4>
                                <p>"In HDC, we 'bind' a variable to a value using XOR. This creates a new, nearly orthogonal vector that represents the concept 'Teacher of Math'."</p>
                                <button class="btn-action" on:click=move |_| {
                                    let bound = v_role.get().bind(&v_subject.get());
                                    set_v_memory.set(Some(bound));
                                    set_step.set(2);
                                }>"Bind Role to Subject"</button>
                            </div>
                        }.into_any(),
                        2 => view! {
                            <div class="lesson-step">
                                <h4>"\u{2295} Part 2: Superposition"</h4>
                                <p>"We can store multiple facts in the same vector by 'superposing' them. This is like a holographic memory where everything is stored everywhere."</p>
                                <button class="btn-action" on:click=move |_| {
                                    // Add a third random vector to simulate bundle
                                    let v3 = HdcVector::random();
                                    let bundle = HdcVector::superpose(&[&v_memory.get().unwrap(), &v3]);
                                    set_v_memory.set(Some(bundle));
                                    set_step.set(3);
                                }>"Superpose additional facts"</button>
                            </div>
                        }.into_any(),
                        3 => view! {
                            <div class="lesson-step challenge">
                                <h4>"\u{1F3AF} The Unbinding Challenge"</h4>
                                <p>"You have a memory vector. Can you extract the 'Subject' by unbinding the 'Role'?"</p>
                                <button class="btn-action success" on:click=move |_| {
                                    // SUCCESS: Generate Audit Artifact
                                    set_step.set(4);
                                }>"Extract Subject"</button>
                            </div>
                        }.into_any(),
                        _ => view! {
                            <div class="lesson-complete">
                                <h4>"\u{2705} Mastery Proven"</h4>
                                <p>"You have successfully manipulated high-dimensional meaning."</p>
                                
                                <div class="deep-dive">
                                    <h5>"Extra: 16k Deep Dive"</h5>
                                    <p>"See how 16,384 dimensions handle noise."</p>
                                    <crate::games::universal::hdc_holograph::HdcHolograph />
                                </div>

                                <button class="btn-action" style="margin-top: 1rem" on:click=move |_| set_step.set(1)>"Reset Sandbox"</button>
                            </div>
                        }.into_any()
                    }}
                </div>
            </div>
        </div>
    }
}

/// Renders a 256-bit vector as a 16x16 grid of "holographic" pixels.
#[component]
fn HdcVisualizer(vector: Signal<HdcVector>) -> impl IntoView {
    view! {
        <div class="vector-grid">
            {move || {
                let v = vector.get();
                v.bits.iter().map(|&bit| {
                    let color = if bit { "var(--primary)" } else { "var(--surface-low)" };
                    view! {
                        <div class="vector-pixel" style=format!("background-color: {}", color)></div>
                    }
                }).collect_view()
            }}
        </div>
    }
}
