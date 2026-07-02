// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verification Gateway — Employer Portal for Sovereign Credentials.

use leptos::prelude::*;
use crate::curriculum::{curriculum_graph, CurriculumNode};

#[component]
pub fn VerificationPage() -> impl IntoView {
    let (input_hash, set_input_hash) = signal("".to_string());
    let (is_verifying, set_is_verifying) = signal(false);
    let (verification_result, set_verification_result) = signal(None);

    let verify_hash = move |_| {
        set_is_verifying.set(true);
        let hash = input_hash.get();
        
        // SIMULATED: Verification logic against DHT/Source Chain
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_millis(1200)).await;
            
            if hash.len() >= 8 {
                set_verification_result.set(Some(true));
            } else {
                set_verification_result.set(Some(false));
            }
            set_is_verifying.set(false);
        });
    };

    view! {
        <div class="verify-page">
            <header class="verify-header">
                <h2>"Verification Gateway"</h2>
                <p>"Verifying Sovereign Proof-of-Learning via Mycelix Praxis."</p>
            </header>

            <div class="verify-container">
                <div class="verify-input-box">
                    <label>"Enter Mastery DID or Achievement Hash:"</label>
                    <input 
                        type="text" 
                        placeholder="e.g. did:mycelix:praxis-alpha-student or 8f2b3c..."
                        prop:value=input_hash
                        on:input=move |ev| set_input_hash.set(event_target_value(&ev))
                    />
                    <button 
                        class="btn-primary" 
                        style="width: 100%; margin-top: 1rem"
                        on:click=verify_hash
                        disabled=move || is_verifying.get()
                    >
                        {move || if is_verifying.get() { "Consulting Mycelial Ledger..." } else { "Verify Credentials" }}
                    </button>
                </div>

                <div class="verify-result-area">
                    {move || match (is_verifying.get(), verification_result.get()) {
                        (true, _) => view! { <div class="loading-spinner"></div> }.into_any(),
                        (false, Some(true)) => view! { <VerificationSuccess /> }.into_any(),
                        (false, Some(false)) => view! { <div class="verify-fail">"No record found for this hash."</div> }.into_any(),
                        _ => view! { <div class="verify-placeholder">"Paste a hash to verify legacy alignment."</div> }.into_any(),
                    }}
                </div>
            </div>

            <section class="verify-footer">
                <p>"All credentials verified via CLR 2.0 (Comprehensive Learner Record) standards."</p>
                <div class="standard-badges">
                    <span class="badge">"Holochain Verifiable"</span>
                    <span class="badge">"W3C DID Compliant"</span>
                    <span class="badge">"AGPL-3.0 Licensed"</span>
                </div>
            </section>
        </div>
    }
}

#[component]
fn VerificationSuccess() -> impl IntoView {
    view! {
        <div class="verify-success">
            <div class="success-icon">"\u{2705}"</div>
            <h3>"Credential Verified"</h3>
            <div class="success-details">
                <div class="detail-row">
                    <span class="label">"Status:"</span>
                    <span class="value" style="color: var(--success)">"ACTIVE & VERIFIED"</span>
                </div>
                <div class="detail-row">
                    <span class="label">"Issuer:"</span>
                    <span class="value">"Mycelix Praxis Protocol"</span>
                </div>
                <div class="detail-row">
                    <span class="label">"Compliance:"</span>
                    <span class="value">"CLR 2.0 / ZK-Proof Ready"</span>
                </div>
            </div>

            <div class="verified-badges" style="margin-top: 1.5rem">
                <h5>"Verified Global Alignments:"</h5>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap">
                    <span class="badge-v">"AWS Certified Solutions Architect (92%)"</span>
                    <span class="badge-v">"CompTIA Security+ (100%)"</span>
                    <span class="badge-v">"TEFL Educator (88%)"</span>
                </div>
            </div>
        </div>
    }
}
