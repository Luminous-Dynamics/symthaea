// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finance domain — excellent: SAP balance, TEND flows, staking, recognition.

use domain_finance::types::*;
use leptos::prelude::*;
use sensorium_viz::{bar_chart::Bar, line_chart::Series, BarChart, LineChart};

#[derive(Clone, Copy, PartialEq)]
enum FinView {
    Overview,
    Payments,
    Staking,
    Recognition,
}

#[derive(Clone)]
struct FinState {
    balance: RwSignal<SapBalance>,
    stake: RwSignal<Option<CollateralStake>>,
    mycel: RwSignal<MemberMycelState>,
    payments: RwSignal<Vec<Payment>>,
}

impl FinState {
    fn new() -> Self {
        Self {
            balance: RwSignal::new(SapBalance {
                member_did: "did:me".into(),
                balance: 4_280_000,
                last_demurrage_at: 1711900800,
            }),
            stake: RwSignal::new(Some(CollateralStake {
                id: "s-1".into(),
                staker_did: "did:me".into(),
                sap_amount: 1_200_000,
                mycel_score: 0.72,
                stake_weight: 1.72,
                staked_at: 1710000000,
                unbonding_until: None,
                status: StakeStatus::Active,
                pending_rewards: 45_000,
            })),
            mycel: RwSignal::new(MemberMycelState {
                member_did: "did:me".into(),
                mycel_score: 0.72,
                participation: 0.85,
                recognition: 0.65,
                validation: 0.70,
                longevity: 0.68,
                active_months: 14,
                is_apprentice: false,
            }),
            payments: RwSignal::new(vec![
                Payment {
                    id: "p-1".into(),
                    from_did: "did:me".into(),
                    to_did: "did:elena".into(),
                    amount: 150_000,
                    fee: 1500,
                    currency: "SAP".into(),
                    payment_type: PaymentType::Direct,
                    status: TransferStatus::Completed,
                    memo: Some("Care coordination".into()),
                    created: 1711900800,
                    completed: Some(1711900900),
                },
                Payment {
                    id: "p-2".into(),
                    from_did: "did:commune".into(),
                    to_did: "did:me".into(),
                    amount: 320_000,
                    fee: 0,
                    currency: "SAP".into(),
                    payment_type: PaymentType::CommonsContribution,
                    status: TransferStatus::Completed,
                    memo: Some("Solar garden yield".into()),
                    created: 1711814400,
                    completed: Some(1711814500),
                },
                Payment {
                    id: "p-3".into(),
                    from_did: "did:me".into(),
                    to_did: "did:treasury".into(),
                    amount: 50_000,
                    fee: 500,
                    currency: "SAP".into(),
                    payment_type: PaymentType::TreasuryContribution,
                    status: TransferStatus::Completed,
                    memo: None,
                    created: 1711728000,
                    completed: Some(1711728100),
                },
            ]),
        }
    }
}

#[component]
pub fn FinanceOverview() -> impl IntoView {
    let (active_view, set_active_view) = signal(FinView::Overview);
    let state = FinState::new();
    provide_context(state.clone());

    view! {
        <div class="finance-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == FinView::Overview { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(FinView::Overview)>"Overview"</button>
                <button class=move || if active_view.get() == FinView::Payments { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(FinView::Payments)>"Payments"</button>
                <button class=move || if active_view.get() == FinView::Staking { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(FinView::Staking)>"Staking"</button>
                <button class=move || if active_view.get() == FinView::Recognition { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(FinView::Recognition)>"Recognition"</button>
            </div>

            <Show when=move || active_view.get() == FinView::Overview>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #D97706">"SAP BALANCE"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{:.0}", state.balance.get().balance as f64 / 1_000_000.0)}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"SAP (after demurrage)"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #FBBF24">"MYCEL"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{:.2}", state.mycel.get().mycel_score)}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Reputation score"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"STAKED"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.stake.get().map(|s| format!("{:.0}", s.sap_amount as f64 / 1_000_000.0)).unwrap_or("0".into())}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">{move || state.stake.get().map(|s| format!("Weight: {:.2}x", s.stake_weight)).unwrap_or_default()}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #8b5cf6">"REWARDS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.stake.get().map(|s| format!("{:.0}", s.pending_rewards as f64 / 1_000_000.0)).unwrap_or("0".into())}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Pending (SAP)"</p>
                    </div>
                </div>
                <h3 class="section-title">"TEND Flow (8 weeks)"</h3>
                <LineChart series=vec![
                    Series { label: "Inflow".into(), color: "#22c55e".into(), data: vec![1.2, 1.85, 2.1, 1.75, 2.3, 1.95, 2.45, 2.2] },
                    Series { label: "Outflow".into(), color: "#f59e0b".into(), data: vec![0.8, 0.95, 1.1, 1.4, 1.05, 1.3, 1.15, 1.45] },
                ] width=400.0 height=180.0 />
            </Show>

            <Show when=move || active_view.get() == FinView::Payments>
                <div class="thought-list">
                    {move || { state.payments.get().iter().map(|p| {
                        let is_incoming = p.to_did == "did:me";
                        let amount = p.amount as f64 / 1_000_000.0;
                        let memo = p.memo.clone().unwrap_or_default();
                        let ptype = format!("{:?}", p.payment_type);
                        let other = if is_incoming { p.from_did.clone() } else { p.to_did.clone() };
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {};", if is_incoming { "#22c55e" } else { "#f59e0b" })>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {}", if is_incoming { "#22c55e" } else { "#f59e0b" })>
                                        {if is_incoming { "RECEIVED" } else { "SENT" }}
                                    </span>
                                    <span class="thought-confidence">{format!("{}{:.2} SAP", if is_incoming { "+" } else { "-" }, amount)}</span>
                                    <span class="thought-domain">{ptype}</span>
                                </div>
                                {(!memo.is_empty()).then(|| view! { <p class="thought-content">{memo}</p> })}
                                <p style="font-size: 0.7rem; color: var(--text-muted)">{format!("{} {}", if is_incoming { "From" } else { "To" }, &other[..20.min(other.len())])}</p>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            <Show when=move || active_view.get() == FinView::Staking>
                {move || state.stake.get().map(|s| {
                    view! {
                        <div class="thought-card" style="border-left: 3px solid #22c55e;">
                            <div class="thought-meta">
                                <span class="thought-type" style=format!("color: {}", match s.status { StakeStatus::Active => "#22c55e", StakeStatus::Unbonding => "#f59e0b", _ => "#ef4444" })>{format!("{:?}", s.status)}</span>
                                <span class="thought-confidence">{format!("{:.0} SAP staked", s.sap_amount as f64 / 1_000_000.0)}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.5rem; text-align: center;">
                                <div><div style="font-size: 1.2rem; font-weight: 700; color: #FBBF24">{format!("{:.2}", s.mycel_score)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"MYCEL"</div></div>
                                <div><div style="font-size: 1.2rem; font-weight: 700; color: #22c55e">{format!("{:.2}x", s.stake_weight)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Weight"</div></div>
                                <div><div style="font-size: 1.2rem; font-weight: 700; color: #8b5cf6">{format!("{:.3}", s.pending_rewards as f64 / 1_000_000.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Rewards"</div></div>
                            </div>
                            <button class="form-submit" style="margin-top: 0.75rem;">"Claim Rewards"</button>
                        </div>
                    }
                })}
            </Show>

            <Show when=move || active_view.get() == FinView::Recognition>
                <div class="thought-card" style="margin-bottom: 1rem;">
                    <h3 class="section-title">"MYCEL Score Breakdown"</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; text-align: center;">
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #2563EB">{move || format!("{:.0}%", state.mycel.get().participation * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Participation (40%)"</div></div>
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #ec4899">{move || format!("{:.0}%", state.mycel.get().recognition * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Recognition (20%)"</div></div>
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #22c55e">{move || format!("{:.0}%", state.mycel.get().validation * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Validation (20%)"</div></div>
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #06b6d4">{move || format!("{:.0}%", state.mycel.get().longevity * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Longevity (20%)"</div></div>
                    </div>
                    <p style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem;">
                        {move || format!("{} active months", state.mycel.get().active_months)}
                    </p>
                </div>
                <h3 class="section-title">"Recognition This Cycle"</h3>
                <BarChart data=vec![
                    Bar { label: "Care".into(), value: 35.0, color: "#ec4899".into() },
                    Bar { label: "Teaching".into(), value: 28.0, color: "#2563EB".into() },
                    Bar { label: "Governance".into(), value: 18.0, color: "#7C3AED".into() },
                    Bar { label: "Infra".into(), value: 12.0, color: "#059669".into() },
                ] width=350.0 height=160.0 />
            </Show>
        </div>
    }
}
