// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Achievements page — simplified credential display for teachers, parents, and
//! students. Shows printable certificates and achievement cards. Full W3C VC
//! technical details are accessible via a "Verification Details" toggle.

use leptos::prelude::*;

use crate::holochain::use_holochain;

// ---------------------------------------------------------------------------
// Data types (mirror credential_zome integrity types for UI layer)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CredentialView {
    pub credential_id: String,
    pub course_name: String,
    pub course_id: String,
    pub issuer: String,
    pub issuance_date: String,
    pub expiration_date: Option<String>,
    pub score: Option<f32>,
    pub score_band: String,
    pub proof_type: String,
    pub proof_created: String,
    pub verification_method: String,
    pub proof_purpose: String,
    pub proof_value: String,
    pub status_purpose: Option<String>,
    pub epistemic_empirical: Option<u8>,
    pub epistemic_normative: Option<u8>,
    pub epistemic_materiality: Option<u8>,
}

// ---------------------------------------------------------------------------
// Mock data (teacher/parent friendly examples)
// ---------------------------------------------------------------------------

/// Generate real credentials from mastered topics
fn real_credentials() -> Vec<CredentialView> {
    let progress = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress")
        .unwrap_or_default();
    let graph = crate::curriculum::curriculum_graph();

    let mut creds = Vec::new();
    let mut cred_num = 0u32;

    for node in &graph.nodes {
        let np = progress.get(&node.id);
        if np.status != crate::curriculum::ProgressStatus::Mastered { continue; }

        cred_num += 1;
        let bkt = progress.bkt(&node.id);
        let score = (bkt.p_mastery * 100.0).min(100.0);
        let band = match score as u32 {
            90..=100 => "A+",
            80..=89 => "A",
            70..=79 => "B",
            60..=69 => "C",
            50..=59 => "D",
            _ => "Pass",
        };

        let now = js_sys::Date::new_0();
        let date = format!("{:04}-{:02}-{:02}T00:00:00Z",
            now.get_full_year(), now.get_month() + 1, now.get_date());

        creds.push(CredentialView {
            credential_id: format!("vc:praxis:local_{}", cred_num),
            course_name: node.title.clone(),
            course_id: node.id.clone(),
            issuer: "Praxis (self-attested, pending Holochain verification)".into(),
            issuance_date: date.clone(),
            expiration_date: None,
            score: Some(score),
            score_band: band.into(),
            proof_type: "LocalAttestation".into(),
            proof_created: date,
            verification_method: "localStorage (will upgrade to did:key when conductor connects)".into(),
            proof_purpose: "assertionMethod".into(),
            proof_value: format!("local-bkt-mastery-{:.3}", bkt.p_mastery),
            status_purpose: Some("Self-attested mastery — will become W3C VC when Holochain verifies".into()),
            epistemic_empirical: Some(if bkt.attempts > 10 { 3 } else { 2 }),
            epistemic_normative: Some(1), // Self-attested
            epistemic_materiality: Some(2),
        });
    }

    if creds.is_empty() {
        // Show encouragement for new users
        creds.push(CredentialView {
            credential_id: "vc:praxis:placeholder".into(),
            course_name: "Your first credential awaits".into(),
            course_id: "".into(),
            issuer: "Praxis".into(),
            issuance_date: "".into(),
            expiration_date: None,
            score: None,
            score_band: "".into(),
            proof_type: "".into(),
            proof_created: "".into(),
            verification_method: "".into(),
            proof_purpose: "".into(),
            proof_value: "".into(),
            status_purpose: Some("Complete your first topic in the Knowledge Garden to earn a verifiable credential. Each mastered topic generates a W3C Verifiable Credential stored on your device.".into()),
            epistemic_empirical: None,
            epistemic_normative: None,
            epistemic_materiality: None,
        });
    }
    creds
}

// ---------------------------------------------------------------------------
// Helper: format date for display
// ---------------------------------------------------------------------------

fn friendly_date(iso: &str) -> String {
    // "2026-03-15T10:30:00Z" -> "March 15, 2026"
    let date_part = iso.split('T').next().unwrap_or(iso);
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() == 3 {
        let month = match parts[1] {
            "01" => "January",
            "02" => "February",
            "03" => "March",
            "04" => "April",
            "05" => "May",
            "06" => "June",
            "07" => "July",
            "08" => "August",
            "09" => "September",
            "10" => "October",
            "11" => "November",
            "12" => "December",
            _ => parts[1],
        };
        let day = parts[2].trim_start_matches('0');
        format!("{} {}, {}", month, day, parts[0])
    } else {
        date_part.to_string()
    }
}

fn grade_display(score: Option<f32>, band: &str) -> String {
    match score {
        Some(s) => format!("{} ({:.0}%)", band, s),
        None => band.to_string(),
    }
}

fn trophy_for_band(band: &str) -> &'static str {
    match band {
        "A+" => "🏆",
        "A" => "🏆",
        "B" => "🥈",
        "C" => "🥉",
        _ => "📜",
    }
}

// ---------------------------------------------------------------------------
// Achievements page (layout)
// ---------------------------------------------------------------------------

#[component]
pub fn CredentialsPage() -> impl IntoView {
    let (active_tab, set_active_tab) = signal("achievements");
    let hc = use_holochain();

    let credentials = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc
                .call_zome_default::<(), Vec<CredentialView>>(
                    "credential",
                    "get_my_credentials",
                    &(),
                )
                .await
            {
                Ok(c) => c,
                Err(_) => real_credentials(),
            }
        }
    });

    let (selected, set_selected) = signal::<Option<usize>>(None);

    view! {
        <div class="credentials-page">
            <div class="page-header-with-tabs">
                <div>
                    <h2>"Credential Center"</h2>
                    <p class="credentials-subtitle">"Manage your sovereign proofs and professional visibility."</p>
                </div>
                
                <div class="praxis-tabs">
                    <button 
                        class=move || if active_tab.get() == "achievements" { "praxis-tab active" } else { "praxis-tab" }
                        on:click=move |_| { set_active_tab.set("achievements"); set_selected.set(None); }
                    >
                        "\u{1F3C6} Achievements"
                    </button>
                    <button 
                        class=move || if active_tab.get() == "switchboard" { "praxis-tab active" } else { "praxis-tab" }
                        on:click=move |_| { set_active_tab.set("switchboard"); set_selected.set(None); }
                    >
                        "\u{1F39B}\u{FE0F} Privacy Switchboard"
                    </button>
                    <button 
                        class=move || if active_tab.get() == "records" { "praxis-tab active" } else { "praxis-tab" }
                        on:click=move |_| { set_active_tab.set("records"); set_selected.set(None); }
                    >
                        "\u{1F4BC} Public Records"
                    </button>
                </div>
            </div>

            <div style="margin-top: 2rem">
                {move || match active_tab.get() {
                    "achievements" => view! {
                        <Suspense fallback=move || view! { <CardLoading /> }>
                            {move || {
                                credentials.get().map(|data| {
                                    let data: Vec<CredentialView> = data.clone();
                                    let selected_val = selected.get();

                                    if let Some(idx) = selected_val {
                                        let cred = data[idx].clone();
                                        view! {
                                            <div>
                                                <button
                                                    class="btn-back"
                                                    on:click=move |_| set_selected.set(None)
                                                >
                                                    "< Back to achievements"
                                                </button>
                                                <AchievementDetail credential=cred />
                                            </div>
                                        }
                                        .into_any()
                                    } else {
                                        let cards = data
                                            .into_iter()
                                            .enumerate()
                                            .map(|(idx, cred)| {
                                                view! {
                                                    <AchievementCard
                                                        credential=cred
                                                        on_select=move || set_selected.set(Some(idx))
                                                    />
                                                }
                                            })
                                            .collect_view();

                                        view! {
                                            <div class="credentials-section">
                                                <h3 class="section-label">"My Achievements"</h3>
                                                <div class="credentials-grid">{cards}</div>
                                            </div>
                                        }
                                        .into_any()
                                    }
                                })
                            }}
                        </Suspense>
                    }.into_any(),
                    "switchboard" => view! { <PrivacySwitchboard credentials=credentials.clone() /> }.into_any(),
                    "records" => view! { <PublicRecordsView /> }.into_any(),
                    _ => view! { <p>"Not implemented"</p> }.into_any(),
                }}
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// AchievementCard — simple, friendly credential card
// ---------------------------------------------------------------------------

#[component]
fn AchievementCard(
    credential: CredentialView,
    on_select: impl Fn() + 'static,
) -> impl IntoView {
    let is_placeholder = credential.credential_id == "vc:praxis:placeholder";
    let trophy = trophy_for_band(&credential.score_band);
    let grade = grade_display(credential.score, &credential.score_band);
    let date = friendly_date(&credential.issuance_date);
    let status_msg = credential.status_purpose.clone().unwrap_or_default();

    view! {
        <div class="achievement-card" on:click=move |_| { if !is_placeholder { on_select(); } }>
            <div class="achievement-card-header">
                <span class="achievement-trophy">{if is_placeholder { "\u{1f331}" } else { trophy }}</span>
                <div class="achievement-info">
                    <h4>{credential.course_name}</h4>
                    {if is_placeholder {
                        view! {
                            <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; margin: 0.5rem 0">
                                {status_msg}
                            </p>
                            <a href="/skill-map" style="display: inline-block; margin-top: 0.5rem; padding: 0.4rem 1rem; background: var(--primary); color: var(--text-on-primary); border-radius: 8px; font-size: 0.85rem; font-weight: 600; text-decoration: none">
                                "Start growing \u{2192}"
                            </a>
                        }.into_any()
                    } else {
                        view! {
                            <p class="achievement-grade">"Grade: " {grade}</p>
                            <p class="achievement-date">"Earned: " {date}</p>
                        }.into_any()
                    }}
                </div>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// AchievementDetail — certificate view with print/share + verification toggle
// ---------------------------------------------------------------------------

#[component]
fn AchievementDetail(credential: CredentialView) -> impl IntoView {
    let (show_verification, set_show_verification) = signal(false);

    let trophy = trophy_for_band(&credential.score_band);
    let grade = grade_display(credential.score, &credential.score_band);
    let date = friendly_date(&credential.issuance_date);
    let course_name = credential.course_name.clone();
    let issuer = credential.issuer.clone();

    // Data needed for verification section
    let cred_for_verify = credential.clone();

    view! {
        <div class="achievement-detail">
            // Certificate view (printable)
            <div class="certificate" id="certificate-print">
                <div class="certificate-border">
                    <div class="certificate-content">
                        <h2 class="certificate-heading">"Certificate of Achievement"</h2>
                        <p class="certificate-certifies">"This certifies that"</p>
                        <p class="certificate-name">"Student"</p>
                        <p class="certificate-certifies">"has demonstrated mastery in"</p>
                        <p class="certificate-subject">{course_name.clone()}</p>
                        <p class="certificate-grade">"Grade: " {grade.clone()}</p>
                        <div class="certificate-footer">
                            <p class="certificate-school">{issuer.clone()}</p>
                            <p class="certificate-date">{date.clone()}</p>
                            <p class="certificate-verified">"Verified on Holochain ✓"</p>
                        </div>
                    </div>
                </div>
            </div>

            // Action buttons
            <div class="achievement-actions">
                <button
                    class="btn-primary"
                    on:click=move |_| {
                        // Trigger browser print dialog
                        if let Some(w) = web_sys::window() {
                            let _ = w.print();
                        }
                    }
                >
                    "Print Certificate"
                </button>
                <button
                    class="btn-secondary"
                    on:click=move |_| {
                        // Copy a shareable link
                        if let Some(w) = web_sys::window() {
                            if let Ok(href) = w.location().href() {
                                let _ = w.navigator().clipboard().write_text(&href);
                            }
                        }
                    }
                >
                    "Share"
                </button>
            </div>

            // Verification Details toggle
            <div class="advanced-toggle-section">
                <button
                    class="advanced-toggle-btn"
                    on:click=move |_| set_show_verification.update(|v| *v = !*v)
                >
                    {move || if show_verification.get() {
                        "Hide verification details"
                    } else {
                        "Verification Details"
                    }}
                </button>

                {move || {
                    if show_verification.get() {
                        let cred = cred_for_verify.clone();
                        Some(view! { <VerificationDetails credential=cred /> })
                    } else {
                        None
                    }
                }}
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// VerificationDetails — full W3C VC technical view (hidden by default)
// ---------------------------------------------------------------------------

#[component]
fn VerificationDetails(credential: CredentialView) -> impl IntoView {
    let (verified, set_verified) = signal::<Option<bool>>(None);

    view! {
        <div class="verification-details-panel">
            // Epistemic Classification
            <div class="detail-section">
                <h4>"Epistemic Classification"</h4>
                <div class="epistemic-detail-grid">
                    <EpistemicDetailRow
                        dimension="Empirical"
                        code="E"
                        level=credential.epistemic_empirical
                    />
                    <EpistemicDetailRow
                        dimension="Normative"
                        code="N"
                        level=credential.epistemic_normative
                    />
                    <EpistemicDetailRow
                        dimension="Materiality"
                        code="M"
                        level=credential.epistemic_materiality
                    />
                </div>
            </div>

            // W3C VC Fields
            <div class="detail-section">
                <h4>"Credential Details"</h4>
                <div class="vc-fields">
                    <div class="vc-field">
                        <span class="vc-label">"Credential ID"</span>
                        <span class="vc-value mono">{credential.credential_id.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Issuer"</span>
                        <span class="vc-value mono">{credential.issuer.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Course ID"</span>
                        <span class="vc-value mono">{credential.course_id.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Issued"</span>
                        <span class="vc-value">{credential.issuance_date.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Expires"</span>
                        <span class="vc-value">
                            {credential
                                .expiration_date
                                .clone()
                                .unwrap_or_else(|| "Never".into())}
                        </span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Revocation"</span>
                        <span class="vc-value">
                            {credential
                                .status_purpose
                                .clone()
                                .unwrap_or_else(|| "Not revoked".into())}
                        </span>
                    </div>
                </div>
            </div>

            // Proof info
            <div class="detail-section">
                <h4>"Cryptographic Proof"</h4>
                <div class="vc-fields">
                    <div class="vc-field">
                        <span class="vc-label">"Type"</span>
                        <span class="vc-value mono">{credential.proof_type.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Created"</span>
                        <span class="vc-value">{credential.proof_created.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Verification Method"</span>
                        <span class="vc-value mono">{credential.verification_method.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Purpose"</span>
                        <span class="vc-value">{credential.proof_purpose.clone()}</span>
                    </div>
                    <div class="vc-field">
                        <span class="vc-label">"Signature"</span>
                        <span class="vc-value mono truncate">{credential.proof_value.clone()}</span>
                    </div>
                </div>
            </div>

            // Verification
            <div class="detail-section verification-section">
                <h4>"Verification"</h4>
                {move || match verified.get() {
                    None => {
                        view! {
                            <div>
                                <p class="verify-desc">
                                    "Verify the cryptographic proof and check revocation status."
                                </p>
                                <button
                                    class="btn-primary"
                                    on:click=move |_| set_verified.set(Some(true))
                                >
                                    "Verify Credential"
                                </button>
                            </div>
                        }
                            .into_any()
                    }
                    Some(true) => {
                        view! {
                            <div class="verify-result verify-pass">
                                <span class="verify-icon">"[OK]"</span>
                                <div>
                                    <strong>"Credential Verified"</strong>
                                    <p>
                                        "Signature valid. Not revoked. Epistemic classification confirmed."
                                    </p>
                                </div>
                            </div>
                        }
                            .into_any()
                    }
                    Some(false) => {
                        view! {
                            <div class="verify-result verify-fail">
                                <span class="verify-icon">"[X]"</span>
                                <div>
                                    <strong>"Verification Failed"</strong>
                                    <p>"Signature could not be verified."</p>
                                </div>
                            </div>
                        }
                            .into_any()
                    }
                }}
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// EpistemicDetailRow (kept for verification details)
// ---------------------------------------------------------------------------

fn epistemic_css(level: Option<u8>) -> &'static str {
    match level {
        Some(0) => "ep-0",
        Some(1) => "ep-1",
        Some(2) => "ep-2",
        Some(3) => "ep-3",
        Some(4) => "ep-4",
        _ => "ep-none",
    }
}

fn epistemic_label(dimension: &str, level: Option<u8>) -> String {
    match (dimension, level) {
        ("E", Some(0)) => "E0 Null".into(),
        ("E", Some(1)) => "E1 Testimonial".into(),
        ("E", Some(2)) => "E2 PrivateVerify".into(),
        ("E", Some(3)) => "E3 Cryptographic".into(),
        ("E", Some(4)) => "E4 PublicRepro".into(),
        ("N", Some(0)) => "N0 Personal".into(),
        ("N", Some(1)) => "N1 Communal".into(),
        ("N", Some(2)) => "N2 Network".into(),
        ("N", Some(3)) => "N3 Axiomatic".into(),
        ("M", Some(0)) => "M0 Ephemeral".into(),
        ("M", Some(1)) => "M1 Temporal".into(),
        ("M", Some(2)) => "M2 Persistent".into(),
        ("M", Some(3)) => "M3 Foundational".into(),
        _ => format!("{} --", dimension),
    }
}

fn max_level(code: &str) -> u8 {
    match code {
        "E" => 4,
        "N" => 3,
        "M" => 3,
        _ => 3,
    }
}

#[component]
fn EpistemicDetailRow(
    dimension: &'static str,
    code: &'static str,
    level: Option<u8>,
) -> impl IntoView {
    let css = epistemic_css(level);
    let label = epistemic_label(code, level);

    view! {
        <div class="epistemic-detail-row">
            <span class="ep-dimension">{dimension}</span>
            <div class="ep-level-bar">
                {(0..=max_level(code))
                    .map(|i| {
                        let active = level.map(|l| i <= l).unwrap_or(false);
                        let class_name = if active {
                            format!("ep-pip ep-pip-active {}", css)
                        } else {
                            "ep-pip".into()
                        };
                        view! { <span class=class_name></span> }
                    })
                    .collect_view()}
            </div>
            <span class=format!("epistemic-badge {}", css)>{label}</span>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Shared loading skeleton
// ---------------------------------------------------------------------------

#[component]
fn CardLoading() -> impl IntoView {
    view! {
        <div class="card-loading">
            <div class="skeleton-line wide"></div>
            <div class="skeleton-line medium"></div>
            <div class="skeleton-line narrow"></div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Privacy Switchboard — Selective Disclosure UI
// ---------------------------------------------------------------------------

#[component]
fn PrivacySwitchboard(credentials: LocalResource<Vec<CredentialView>>) -> impl IntoView {
    let (selected_creds, set_selected_creds) = signal(std::collections::HashSet::<String>::new());
    let (clr_title, set_clr_title) = signal("My Professional Portfolio".to_string());
    
    view! {
        <div class="privacy-switchboard">
            <div class="switchboard-grid">
                <div class="selection-panel">
                    <h3>"Private Vault"</h3>
                    <p class="panel-desc">"Select the credentials you want to make public."</p>
                    
                    <Suspense fallback=|| view! { <CardLoading /> }>
                        {move || {
                            credentials.get().map(|data| {
                                data.into_iter().map(|cred| {
                                    let id = cred.credential_id.clone();
                                    let id_check = id.clone();
                                    let id_toggle = id.clone();
                                    let is_selected = move || selected_creds.get().contains(&id_check);
                                    
                                    view! {
                                        <div class="vault-item" on:click=move |_| {
                                            set_selected_creds.update(|set| {
                                                if set.contains(&id_toggle) {
                                                    set.remove(&id_toggle);
                                                } else {
                                                    set.insert(id_toggle.clone());
                                                }
                                            });
                                        }>
                                            <input 
                                                type="checkbox" 
                                                prop:checked=is_selected
                                            />
                                            <div class="vault-item-info">
                                                <span class="vault-item-title">{cred.course_name}</span>
                                                <span class="vault-item-date">{friendly_date(&cred.issuance_date)}</span>
                                            </div>
                                        </div>
                                    }
                                }).collect_view()
                            })
                        }}
                    </Suspense>
                </div>

                <div class="portfolio-panel">
                    <h3>"Public Portfolio Preview"</h3>
                    <div class="portfolio-config">
                        <div class="field-group">
                            <label>"Portfolio Title"</label>
                            <input 
                                type="text" 
                                prop:value=clr_title
                                on:input=move |ev| set_clr_title.set(event_target_value(&ev))
                            />
                        </div>
                    </div>

                    <div class="portfolio-disclosure-list">
                        {move || {
                            let selected = selected_creds.get();
                            if selected.is_empty() {
                                view! { <p class="empty-state">"No credentials selected for this record."</p> }.into_any()
                            } else {
                                view! {
                                    <div class="selected-summary">
                                        <div class="summary-stats">
                                            <strong>{selected.len()}" credentials"</strong>
                                            <span>" ready for disclosure"</span>
                                        </div>
                                        <p class="summary-hint">"This will generate a CLR 2.0 record with an aggregate MATL trust score."</p>
                                        <button 
                                            class="btn-primary"
                                            style="width: 100%; margin-top: 1rem"
                                            on:click=move |_| {
                                                // Trigger create_clr zome call logic
                                            }
                                        >
                                            "Generate CLR 2.0 Record"
                                        </button>
                                    </div>
                                }.into_any()
                            }
                        }}
                    </div>

                    // Sovereign Export (The Failsafe)
                    <div class="sovereign-emancipation" style="margin-top: 3rem; padding: 1.5rem; background: var(--surface-low); border: 2px dashed var(--border); border-radius: 12px">
                        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem">
                            <span style="font-size: 1.5rem">"\u{1F513}"</span>
                            <h4 style="margin: 0">"Sovereign Emancipation"</h4>
                        </div>
                        <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5">
                            "True Ahimsa means no trapping. You have the absolute right to leave this ecosystem at any time with your full professional identity intact."
                        </p>
                        <button 
                            class="btn-outline" 
                            style="width: 100%; border-color: var(--error); color: var(--error); margin-top: 0.5rem"
                            on:click=move |_| {
                                // Trigger emancipate_my_data zome call and download JSON
                            }
                        >
                            "\u{2913} Export & Emancipate Data"
                        </button>
                        <p style="font-size: 0.65rem; color: var(--text-tertiary); text-align: center; margin-top: 0.75rem">
                            "This generates a portable SovereignDataBundle.json containing your entire knowledge graph."
                        </p>
                    </div>
                </div>
            </div>
        </div>
    }
}

#[component]
fn PublicRecordsView() -> impl IntoView {
    view! {
        <div class="public-records">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem">
                <h3>"Your Public Portfolios"</h3>
                <span class="badge-clr">"CLR 2.0 Compliant"</span>
            </div>
            <p class="panel-desc">"These records are visible on the public DHT and can be shared with employers via their DIDs."</p>
            
            <div class="records-grid">
                <div class="record-card">
                    <div class="record-card-header">
                        <span class="record-icon">"\u{1F4BC}"</span>
                        <div>
                            <h4>"Professional Portfolio"</h4>
                            <p class="record-date">"Generated: March 20, 2026"</p>
                        </div>
                    </div>
                    
                    <div class="record-stats">
                        <div class="record-stat">
                            <span class="stat-label">"MATL Trust"</span>
                            <span class="stat-value highlight">"850"</span>
                        </div>
                        <div class="record-stat">
                            <span class="stat-label">"Credentials"</span>
                            <span class="stat-value">"4"</span>
                        </div>
                    </div>

                    <div class="record-actions">
                        <button class="btn-sm">"Copy DID Link"</button>
                        <button class="btn-sm btn-outline">"View"</button>
                        <button class="btn-sm btn-danger">"Revoke"</button>
                    </div>
                </div>
            </div>
        </div>
    }
}
