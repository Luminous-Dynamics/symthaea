// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::identity_context::use_identity;

#[component]
pub fn DidPage() -> impl IntoView {
    let ctx = use_identity();

    let did = move || ctx.did_document.get();
    let name = move || ctx.my_name.get();

    view! {
        <div class="page page-did">
            <h1>"DID Document"</h1>
            <p class="page-subtitle">"Your decentralized identifier \u{2014} the cryptographic root of your sovereign identity"</p>

            {move || did().map(|doc| {
                let active_class = if doc.active { "did-status-active" } else { "did-status-inactive" };
                view! {
                    // ── DID Header ──
                    <div class="did-doc-card">
                        <div class="did-doc-header">
                            <div class="did-doc-id-row">
                                <span class="did-doc-label">"DID"</span>
                                <code class="did-doc-id">{doc.id.clone()}</code>
                            </div>
                            <span class={format!("did-status-badge {active_class}")}>
                                {if doc.active { "Active" } else { "Deactivated" }}
                            </span>
                        </div>

                        <div class="did-doc-meta">
                            <div class="meta-item">
                                <span class="meta-label">"Controller"</span>
                                <code class="meta-value">{doc.controller.clone()}</code>
                            </div>
                            <div class="meta-item">
                                <span class="meta-label">"Version"</span>
                                <span class="meta-value">{format!("v{}", doc.version)}</span>
                            </div>
                            {move || name().map(|n| view! {
                                <div class="meta-item">
                                    <span class="meta-label">"Mesh Name"</span>
                                    <span class="meta-value meta-name">{n.canonical}</span>
                                </div>
                            })}
                        </div>
                    </div>

                    // ── Verification Methods ──
                    <section class="did-section">
                        <h2>"Verification Methods"</h2>
                        <p class="section-desc">"Cryptographic keys used for authentication and signing"</p>
                        <div class="method-list">
                            {doc.verification_methods.iter().map(|method| {
                                let type_name = method.type_name.clone();
                                let key_preview = if method.public_key_multibase.len() > 20 {
                                    format!("{}...{}", &method.public_key_multibase[..12], &method.public_key_multibase[method.public_key_multibase.len()-8..])
                                } else {
                                    method.public_key_multibase.clone()
                                };
                                let is_encryption = type_name.contains("X25519") || type_name.contains("KeyAgreement");
                                view! {
                                    <div class="method-card">
                                        <div class="method-icon">
                                            {if is_encryption { "\u{1F510}" } else { "\u{1F511}" }}
                                        </div>
                                        <div class="method-info">
                                            <span class="method-type">{type_name}</span>
                                            <code class="method-key">{key_preview}</code>
                                        </div>
                                        <span class="method-purpose">
                                            {if is_encryption { "Encryption" } else { "Signing" }}
                                        </span>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </section>

                    // ── Service Endpoints ──
                    <section class="did-section">
                        <h2>"Service Endpoints"</h2>
                        <p class="section-desc">"Discovery endpoints for cross-cluster communication"</p>
                        {if doc.services.is_empty() {
                            view! {
                                <div class="empty-state">
                                    <p>"No service endpoints registered"</p>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <div class="service-list">
                                    {doc.services.iter().map(|svc| {
                                        view! {
                                            <div class="service-card">
                                                <span class="service-type">{svc.type_name.clone()}</span>
                                                <code class="service-endpoint">{svc.endpoint.clone()}</code>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }}
                    </section>

                    // ── Key Actions ──
                    <section class="did-section">
                        <h2>"Key Management"</h2>
                        <div class="key-actions">
                            <button class="btn" disabled>
                                "\u{1F504} Rotate Signing Key"
                            </button>
                            <button class="btn" disabled>
                                "\u{1F510} Rotate Encryption Key"
                            </button>
                            <button class="btn btn-danger" disabled>
                                "\u{26A0} Deactivate DID"
                            </button>
                        </div>
                        <p class="action-note">"Key rotation and DID deactivation require conductor connection"</p>
                    </section>
                }
            })}
        </div>
    }
}
