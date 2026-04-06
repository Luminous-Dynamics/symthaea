// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civic action dispatch: mock branch (instant local mutation) vs
//! real branch (spawn_local zome call).

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use governance_leptos_types::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, use_consciousness, ToastKind};
use crate::contexts::governance_context::use_governance;

/// Marker struct for action dispatch context.
#[derive(Clone)]
pub struct CivicActions;

pub fn provide_civic_actions() {
    provide_context(CivicActions);
}

/// Cast a phi-weighted vote on a proposal.
pub fn cast_vote(proposal_id: String, choice: VoteChoice, reasoning: Option<String>) {
    let hc = use_holochain();
    let gov = use_governance();
    let consciousness = use_consciousness();
    let toasts = use_toasts();

    if hc.is_mock() {
        let my_did = gov.my_agent_did.get_untracked();
        let phi = consciousness.profile.get_untracked().combined_score();
        gov.my_votes.update(|votes| {
            votes.retain(|v| v.proposal_id != proposal_id);
            votes.push(PhiVoteView {
                hash: format!("vote_{}", votes.len()),
                proposal_id: proposal_id.clone(),
                voter_did: my_did,
                tier: ProposalTier::Basic,
                choice,
                effective_weight: phi,
                phi_score: phi,
                reasoning,
                delegated: false,
                created: now_micros(),
            });
        });
        toasts.push("your voice has been heard", ToastKind::Custom("governance".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct CastPhiVoteInput {
                proposal_id: String,
                voter_did: String,
                tier: String,
                choice: String,
                reason: Option<String>,
            }

            let input = CastPhiVoteInput {
                proposal_id,
                voter_did: gov.my_agent_did.get_untracked(),
                tier: "Basic".to_string(),
                choice: match choice {
                    VoteChoice::For => "For".to_string(),
                    VoteChoice::Against => "Against".to_string(),
                    VoteChoice::Abstain => "Abstain".to_string(),
                },
                reason: reasoning,
            };

            match hc.call_zome::<_, serde_json::Value>(
                "governance", "voting", "cast_phi_weighted_vote", &input,
            ).await {
                Ok(_) => toasts.push("your voice has been heard", ToastKind::Custom("governance".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("cast vote failed: {e}").into());
                    toasts.push("your voice could not reach the commons", ToastKind::Error);
                }
            }
        });
    }
}

/// Submit a new proposal.
pub fn create_proposal(title: String, description: String, proposal_type: ProposalType) {
    let hc = use_holochain();
    let gov = use_governance();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("MIP-{:04}", gov.proposals.get_untracked().len() + 1);
        gov.proposals.update(|proposals| {
            proposals.push(ProposalView {
                hash: format!("proposal_{}", proposals.len()),
                id: id.clone(),
                title,
                description,
                proposal_type,
                author: gov.my_agent_did.get_untracked(),
                status: ProposalStatus::Draft,
                actions: "[]".to_string(),
                discussion_url: None,
                voting_starts: now_micros() + 86_400_000_000,
                voting_ends: now_micros() + 604_800_000_000,
                created: now_micros(),
                updated: now_micros(),
                version: 1,
            });
        });
        toasts.push(format!("{id} has begun to grow"), ToastKind::Custom("governance".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct CreateProposalInput {
                title: String,
                description: String,
                proposal_type: String,
            }

            let input = CreateProposalInput {
                title,
                description,
                proposal_type: match proposal_type {
                    ProposalType::Standard => "Standard",
                    ProposalType::Emergency => "Emergency",
                    ProposalType::Constitutional => "Constitutional",
                    ProposalType::Parameter => "Parameter",
                    ProposalType::Funding => "Funding",
                }.to_string(),
            };

            match hc.call_zome::<_, serde_json::Value>(
                "governance", "proposals", "create_proposal", &input,
            ).await {
                Ok(_) => toasts.push("a new proposal has begun to grow", ToastKind::Custom("governance".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("create proposal failed: {e}").into());
                    toasts.push("the proposal could not take root", ToastKind::Error);
                }
            }
        });
    }
}

/// Add a discussion contribution to a proposal.
pub fn add_contribution(proposal_id: String, content: String, stance: Option<ContributionStance>) {
    let hc = use_holochain();
    let toasts = use_toasts();

    if hc.is_mock() {
        toasts.push("your perspective has joined the discussion", ToastKind::Custom("governance".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct AddContributionInput {
                proposal_id: String,
                content: String,
                stance: Option<String>,
            }

            let input = AddContributionInput {
                proposal_id,
                content,
                stance: stance.map(|s| match s {
                    ContributionStance::Support => "Support",
                    ContributionStance::Oppose => "Oppose",
                    ContributionStance::Concern => "Concern",
                    ContributionStance::Question => "Question",
                    ContributionStance::Amendment => "Amendment",
                }.to_string()),
            };

            match hc.call_zome::<_, serde_json::Value>(
                "governance", "proposals", "add_contribution", &input,
            ).await {
                Ok(_) => toasts.push("your perspective has joined the discussion", ToastKind::Custom("governance".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("add contribution failed: {e}").into());
                }
            }
        });
    }
}

// ============================================================================
// Finance Actions
// ============================================================================

/// Record a TEND exchange (care given).
pub fn record_tend_exchange(receiver_did: String, hours: f32, description: String) {
    let hc = use_holochain();
    let toasts = use_toasts();

    if hc.is_mock() {
        toasts.push(
            format!("recorded {hours:.1}h of care — thank you"),
            ToastKind::Custom("finance".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct RecordExchangeInput {
                provider_did: String,
                receiver_did: String,
                hours: f32,
                service_description: String,
                service_category: String,
                dao_did: String,
            }

            let input = RecordExchangeInput {
                provider_did: "self".into(),
                receiver_did,
                hours,
                service_description: description,
                service_category: "GeneralAssistance".into(),
                dao_did: "default".into(),
            };

            match hc.call_zome::<_, serde_json::Value>(
                "finance", "tend", "record_exchange", &input,
            ).await {
                Ok(_) => toasts.push("care recorded \u{2014} the community is richer", ToastKind::Custom("finance".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("record exchange failed: {e}").into());
                    toasts.push("the exchange could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}

/// Recognize a community member's contribution (builds their MYCEL).
pub fn recognize_member(recipient_did: String, contribution_type: String) {
    let hc = use_holochain();
    let toasts = use_toasts();

    if hc.is_mock() {
        let short = recipient_did.split(':').last()
            .unwrap_or(&recipient_did).to_string();
        toasts.push(
            format!("{short}'s contribution has been recognized"),
            ToastKind::Custom("finance".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct RecognizeMemberInput {
                recipient_did: String,
                contribution_type: String,
                cycle_id: String,
            }

            let cycle_id = {
                let now = js_sys::Date::new_0();
                format!("{}-{:02}", now.get_full_year(), now.get_month() + 1)
            };

            let input = RecognizeMemberInput {
                recipient_did,
                contribution_type,
                cycle_id,
            };

            match hc.call_zome::<_, serde_json::Value>(
                "finance", "recognition", "recognize_member", &input,
            ).await {
                Ok(_) => toasts.push("recognition has been given", ToastKind::Custom("finance".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("recognize member failed: {e}").into());
                    toasts.push("recognition could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}

fn now_micros() -> i64 {
    (js_sys::Date::now() * 1000.0) as i64
}
