// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Health domain — excellent: vault, records, consent, privacy budget.

use domain_health::types::*;
use leptos::prelude::*;
use mycelix_leptos_core::{AvailabilityState, AvailabilityStateKind};
use sensorium_viz::{
    bar_chart::Bar, line_chart::Series, pie_chart::Segment, BarChart, LineChart, PieChart,
};

// ── State Store ──

#[derive(Clone, Copy, PartialEq)]
enum HealthView {
    Home,
    Records,
    Consent,
    Privacy,
    Metabolism,
}

#[derive(Clone)]
struct HealthState {
    vault_unlocked: RwSignal<bool>,
    records: RwSignal<Vec<HealthRecord>>,
    consents: RwSignal<Vec<Consent>>,
    privacy_budget: RwSignal<PrivacyBudget>,
    dividends: RwSignal<Vec<DataDividend>>,
}

impl HealthState {
    fn new() -> Self {
        Self {
            vault_unlocked: RwSignal::new(false),
            records: RwSignal::new(mock_records()),
            consents: RwSignal::new(mock_consents()),
            privacy_budget: RwSignal::new(PrivacyBudget {
                patient_did: "did:me".into(),
                epsilon_total: 10.0,
                epsilon_spent: 7.6,
                delta: 1e-5,
                fl_contributions: 23,
                last_contribution_at: Some(1711900800),
            }),
            dividends: RwSignal::new(vec![
                DataDividend {
                    id: "dd-1".into(),
                    patient_did: "did:me".into(),
                    cohort_id: "diabetes-2026".into(),
                    cohort_name: "Diabetes Prevention Study".into(),
                    amount: 42,
                    currency: "SAP".into(),
                    earned_at: 1711900800,
                    claimed: true,
                },
                DataDividend {
                    id: "dd-2".into(),
                    patient_did: "did:me".into(),
                    cohort_id: "cardiac-ml".into(),
                    cohort_name: "Cardiac ML Cohort".into(),
                    amount: 86,
                    currency: "SAP".into(),
                    earned_at: 1711814400,
                    claimed: false,
                },
            ]),
        }
    }
}

fn mock_records() -> Vec<HealthRecord> {
    vec![
        HealthRecord {
            id: "hr-1".into(),
            patient_did: "did:me".into(),
            record_type: RecordType::LabResult,
            encrypted_payload: vec![0; 128],
            metadata: RecordMetadata {
                provider_did: Some("did:lab".into()),
                facility: Some("Community Health Center".into()),
                date_of_service: 1711900800,
                sensitivity: SensitivityLevel::Normal,
            },
            created_at: 1711900800,
        },
        HealthRecord {
            id: "hr-2".into(),
            patient_did: "did:me".into(),
            record_type: RecordType::Vitals,
            encrypted_payload: vec![0; 64],
            metadata: RecordMetadata {
                provider_did: Some("did:nurse".into()),
                facility: Some("Home Visit".into()),
                date_of_service: 1711814400,
                sensitivity: SensitivityLevel::Normal,
            },
            created_at: 1711814400,
        },
        HealthRecord {
            id: "hr-3".into(),
            patient_did: "did:me".into(),
            record_type: RecordType::Medication,
            encrypted_payload: vec![0; 96],
            metadata: RecordMetadata {
                provider_did: Some("did:pharm".into()),
                facility: Some("Community Pharmacy".into()),
                date_of_service: 1711728000,
                sensitivity: SensitivityLevel::Normal,
            },
            created_at: 1711728000,
        },
        HealthRecord {
            id: "hr-4".into(),
            patient_did: "did:me".into(),
            record_type: RecordType::LabResult,
            encrypted_payload: vec![0; 256],
            metadata: RecordMetadata {
                provider_did: Some("did:lab".into()),
                facility: None,
                date_of_service: 1711641600,
                sensitivity: SensitivityLevel::MentalHealth,
            },
            created_at: 1711641600,
        },
    ]
}

fn mock_consents() -> Vec<Consent> {
    vec![
        Consent {
            id: "c-1".into(),
            patient_did: "did:me".into(),
            recipient_did: "did:dr-chen".into(),
            categories: vec![ConsentCategory::LabResults, ConsentCategory::Vitals],
            purpose: "Primary care management".into(),
            granted_at: 1710000000,
            expires_at: Some(1741536000),
            revoked_at: None,
            no_further_disclosure: false,
        },
        Consent {
            id: "c-2".into(),
            patient_did: "did:me".into(),
            recipient_did: "did:research-u".into(),
            categories: vec![ConsentCategory::LabResults],
            purpose: "Diabetes prevention study (FL only)".into(),
            granted_at: 1711000000,
            expires_at: Some(1742536000),
            revoked_at: None,
            no_further_disclosure: true,
        },
        Consent {
            id: "c-3".into(),
            patient_did: "did:me".into(),
            recipient_did: "did:old-clinic".into(),
            categories: vec![ConsentCategory::All],
            purpose: "Former provider".into(),
            granted_at: 1700000000,
            expires_at: None,
            revoked_at: Some(1710000000),
            no_further_disclosure: false,
        },
    ]
}

fn record_type_color(t: &RecordType) -> &'static str {
    match t {
        RecordType::LabResult => "#0D7377",
        RecordType::Vitals => "#06D6C8",
        RecordType::Medication => "#22c55e",
        RecordType::Immunization => "#60a5fa",
        RecordType::Encounter => "#8b5cf6",
        RecordType::Imaging => "#f59e0b",
        RecordType::Procedure => "#ec4899",
    }
}
fn record_type_label(t: &RecordType) -> &'static str {
    match t {
        RecordType::LabResult => "Lab",
        RecordType::Vitals => "Vitals",
        RecordType::Medication => "Meds",
        RecordType::Immunization => "Immun",
        RecordType::Encounter => "Visit",
        RecordType::Imaging => "Imaging",
        RecordType::Procedure => "Proc",
    }
}

// ── Main ──

#[component]
pub fn HealthOverview() -> impl IntoView {
    let (active_view, set_active_view) = signal(HealthView::Home);
    let state = HealthState::new();
    provide_context(state.clone());

    view! {
        <div class="health-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == HealthView::Home { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HealthView::Home)>"Homeostasis"</button>
                <button class=move || if active_view.get() == HealthView::Records { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HealthView::Records)>"Records"</button>
                <button class=move || if active_view.get() == HealthView::Consent { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HealthView::Consent)>"Consent"</button>
                <button class=move || if active_view.get() == HealthView::Privacy { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HealthView::Privacy)>"Privacy"</button>
                <button class=move || if active_view.get() == HealthView::Metabolism { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(HealthView::Metabolism)>"Metabolism"</button>
            </div>

            // Vault gate
            <Show when=move || !state.vault_unlocked.get() && active_view.get() != HealthView::Home>
                <AvailabilityState
                    kind=AvailabilityStateKind::Locked
                    title="Vault Locked"
                    description="Unlock your health vault to view records, consent state, and private health posture."
                    action={Some(view! {
                        <button class="btn btn-primary" on:click=move |_| state.vault_unlocked.set(true)>
                            "Unlock Vault"
                        </button>
                    }.into_any())}
                />
            </Show>

            // Home
            <Show when=move || active_view.get() == HealthView::Home>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #0D7377">"RECORDS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.records.get().len().to_string()}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Encrypted in vault"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #06D6C8">"CONSENTS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.consents.get().iter().filter(|c| c.revoked_at.is_none()).count().to_string()}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Active"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #f59e0b">"PRIVACY \u{03b5}"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{:.1}", state.privacy_budget.get().epsilon_total - state.privacy_budget.get().epsilon_spent)}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">{move || format!("of {:.1} remaining", state.privacy_budget.get().epsilon_total)}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"DIVIDENDS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{} SAP", state.dividends.get().iter().map(|d| d.amount).sum::<u64>())}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"From FL research"</p>
                    </div>
                </div>
                <h3 class="section-title">"Record Types"</h3>
                <PieChart segments=vec![
                    Segment { label: "Lab".into(), value: 18.0, color: "#0D7377".into() },
                    Segment { label: "Vitals".into(), value: 12.0, color: "#06D6C8".into() },
                    Segment { label: "Meds".into(), value: 8.0, color: "#22c55e".into() },
                    Segment { label: "Imaging".into(), value: 5.0, color: "#60a5fa".into() },
                ] />
            </Show>

            // Records (vault-gated)
            <Show when=move || active_view.get() == HealthView::Records && state.vault_unlocked.get()>
                <div class="thought-list">
                    {move || { state.records.get().iter().map(|r| {
                        let rtype = record_type_label(&r.record_type);
                        let rcolor = record_type_color(&r.record_type);
                        let facility = r.metadata.facility.clone().unwrap_or("Unknown".into());
                        let sensitive = r.metadata.sensitivity != SensitivityLevel::Normal;
                        let size = r.encrypted_payload.len();
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {rcolor};")>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {rcolor}")>{rtype}</span>
                                    {sensitive.then(|| view! { <span class="my-vote-badge" style="background: rgba(239,68,68,0.1); color: #ef4444">"SENSITIVE"</span> })}
                                    <span class="thought-domain">{facility}</span>
                                </div>
                                <p style="font-size: 0.8rem; color: var(--text-secondary)">
                                    {format!("Encrypted ({size} bytes) \u{00b7} ID: {}", r.id)}
                                </p>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            // Consent management
            <Show when=move || active_view.get() == HealthView::Consent && state.vault_unlocked.get()>
                <div class="thought-list">
                    {move || { state.consents.get().iter().map(|c| {
                        let active = c.revoked_at.is_none();
                        let recipient = c.recipient_did.clone();
                        let purpose = c.purpose.clone();
                        let categories: String = c.categories.iter().map(|cat| format!("{:?}", cat)).collect::<Vec<_>>().join(", ");
                        let nfd = c.no_further_disclosure;
                        let cid = c.id.clone();
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {};", if active { "#22c55e" } else { "#ef4444" })>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {}", if active { "#22c55e" } else { "#ef4444" })>
                                        {if active { "ACTIVE" } else { "REVOKED" }}
                                    </span>
                                    {nfd.then(|| view! { <span class="my-vote-badge" style="background: rgba(245,158,11,0.1); color: #f59e0b">"NO FURTHER DISCLOSURE"</span> })}
                                </div>
                                <p style="font-size: 0.85rem; font-weight: 600;">{purpose}</p>
                                <p style="font-size: 0.75rem; color: var(--text-muted)">{format!("To: {recipient}")}</p>
                                <p style="font-size: 0.7rem; color: var(--text-muted)">{format!("Categories: {categories}")}</p>
                                {active.then(|| {
                                    let cid = cid.clone();
                                    view! {
                                        <button class="domain-nav-btn" style="margin-top: 0.5rem; color: #ef4444; border-color: #ef4444;"
                                            on:click=move |_| {
                                                state.consents.update(|consents| {
                                                    if let Some(c) = consents.iter_mut().find(|c| c.id == cid) {
                                                        c.revoked_at = Some(1711900800);
                                                    }
                                                });
                                            }>"Revoke"</button>
                                    }
                                })}
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            // Privacy budget
            <Show when=move || active_view.get() == HealthView::Privacy && state.vault_unlocked.get()>
                <div class="thought-card" style="margin-bottom: 1rem;">
                    <h3 class="section-title">"Differential Privacy Budget (\u{03b5})"</h3>
                    <div style="position: relative; height: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; overflow: hidden; margin-bottom: 0.5rem;">
                        <div style=move || format!("height: 100%; width: {:.0}%; background: linear-gradient(90deg, #22c55e, #f59e0b); border-radius: 10px; transition: width 0.3s;", (state.privacy_budget.get().epsilon_spent / state.privacy_budget.get().epsilon_total) * 100.0) />
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted);">
                        <span>{move || format!("\u{03b5} spent: {:.1}", state.privacy_budget.get().epsilon_spent)}</span>
                        <span>{move || format!("\u{03b5} remaining: {:.1}", state.privacy_budget.get().epsilon_total - state.privacy_budget.get().epsilon_spent)}</span>
                    </div>
                    <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                        {move || format!("{} FL contributions made", state.privacy_budget.get().fl_contributions)}
                    </p>
                </div>
                <h3 class="section-title">"Budget Consumption Over Time"</h3>
                <LineChart series=vec![
                    Series { label: "\u{03b5} spent".into(), color: "#f59e0b".into(), data: vec![0.0, 0.8, 1.5, 2.1, 3.2, 4.8, 6.1, 7.6] },
                ] width=350.0 height=140.0 />
            </Show>

            // Metabolism (data dividends)
            <Show when=move || active_view.get() == HealthView::Metabolism && state.vault_unlocked.get()>
                <div class="thought-list">
                    {move || { state.dividends.get().iter().map(|d| {
                        let name = d.cohort_name.clone();
                        let amount = d.amount;
                        let claimed = d.claimed;
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {};", if claimed { "#22c55e" } else { "#f59e0b" })>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {}", if claimed { "#22c55e" } else { "#f59e0b" })>
                                        {if claimed { "CLAIMED" } else { "UNCLAIMED" }}
                                    </span>
                                    <span class="thought-confidence">{format!("{amount} SAP")}</span>
                                </div>
                                <p class="thought-content">{name}</p>
                                {(!claimed).then(|| view! {
                                    <button class="form-submit" style="margin-top: 0.5rem;">"Claim Dividend"</button>
                                })}
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
                <h3 class="section-title" style="margin-top: 1rem">"Earnings by Cohort"</h3>
                <BarChart data=vec![
                    Bar { label: "Diabetes".into(), value: 42.0, color: "#0D7377".into() },
                    Bar { label: "Cardiac".into(), value: 86.0, color: "#06D6C8".into() },
                ] width=280.0 height=140.0 />
            </Show>
        </div>
    }
}
