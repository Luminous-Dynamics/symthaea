// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! About page — protocol docs and LEM Cube explanation.

use leptos::prelude::*;

#[component]
pub fn AboutPage() -> impl IntoView {
    view! {
        <div class="page-container">
            <h1 class="page-title">"About Mycelix DeSci"</h1>

            <div class="glass-panel" style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Epistemic Charter v2.0 — LEM Cube"</h2>
                <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.8;">
                    "Every claim in the Mycelix DeSci protocol is classified along three orthogonal axes:"
                </p>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                        <h3 style="color: var(--accent-indigo); margin-bottom: 0.5rem;">"E-Axis: Empirical Verifiability"</h3>
                        <ul style="font-size: 0.8rem; color: var(--text-secondary); line-height: 2; list-style: none; padding: 0;">
                            <li><span class="lem-badge e0">"E0"</span>" Unverifiable belief or opinion"</li>
                            <li><span class="lem-badge e1">"E1"</span>" Personal attestation or witness"</li>
                            <li><span class="lem-badge e2">"E2"</span>" Expert/audit verification"</li>
                            <li><span class="lem-badge e3">"E3"</span>" Cryptographic or ZK proof"</li>
                            <li><span class="lem-badge e4">"E4"</span>" Publicly reproducible"</li>
                        </ul>
                    </div>

                    <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                        <h3 style="color: var(--accent-indigo); margin-bottom: 0.5rem;">"N-Axis: Normative Authority"</h3>
                        <ul style="font-size: 0.8rem; color: var(--text-secondary); line-height: 2; list-style: none; padding: 0;">
                            <li>"N0: Personal (self only)"</li>
                            <li>"N1: Communal (local DAO)"</li>
                            <li>"N2: Network (global consensus)"</li>
                            <li>"N3: Axiomatic (mathematical proof)"</li>
                        </ul>
                    </div>

                    <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                        <h3 style="color: var(--accent-indigo); margin-bottom: 0.5rem;">"M-Axis: Materiality"</h3>
                        <ul style="font-size: 0.8rem; color: var(--text-secondary); line-height: 2; list-style: none; padding: 0;">
                            <li>"M0: Ephemeral (momentary)"</li>
                            <li>"M1: Temporal (session-scoped)"</li>
                            <li>"M2: Persistent (recorded)"</li>
                            <li>"M3: Foundational (permanent)"</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="glass-panel" style="margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">"Physics Discovery Bridge"</h2>
                <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.8;">
                    "The Symthaea physics engine encodes 93 landmark equations as 16,384-dimensional "
                    "hypervectors using Hyperdimensional Computing (HDC). When a new claim is submitted, "
                    "the discovery bridge searches for structural analogs using skeleton encoding — "
                    "stripping all variable names to reveal pure mathematical structure. "
                    "Two equations with identical skeletons are structurally isomorphic, even if they "
                    "come from different physics domains."
                </p>
            </div>

            <div class="glass-panel">
                <h2 style="font-size: 1.25rem; margin-bottom: 0.5rem;">"Protocol"</h2>
                <p style="font-size: 0.875rem; color: var(--text-secondary);">
                    "Mycelix DeSci Protocol v0.4.0 | Symthaea Physics Bridge v0.1.0 | "
                    "93 equations | 16 physics domains | HDC 16,384D"
                </p>
            </div>
        </div>
    }
}
