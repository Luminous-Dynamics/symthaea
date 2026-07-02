// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Knowledge action dispatch: mock branch (instant local mutation) vs
//! real branch (spawn_local zome call).

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use knowledge_leptos_types::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, ToastKind};
use crate::context::use_knowledge_context;

/// Submit a new claim to the knowledge commons.
pub fn submit_claim(
    content: String,
    claim_type: ClaimType,
    tags: Vec<String>,
    sources: Vec<String>,
) {
    let hc = use_holochain();
    let ctx = use_knowledge_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("CLM-{:04}", ctx.claims.get_untracked().len() + 1);
        ctx.claims.update(|claims| {
            claims.push(ClaimView {
                id: id.clone(),
                content: content.clone(),
                classification: EpistemicClassificationView {
                    empirical: EmpiricalLevel::E0,
                    normative: NormativeLevel::N0,
                    materiality: MaterialityLevel::M0,
                    confidence: 0.0,
                },
                author: "did:mycelix:user-001".into(),
                sources: sources.clone(),
                tags: tags.clone(),
                claim_type: claim_type.clone(),
                created: (js_sys::Date::now() / 1000.0) as i64,
                version: 1,
            });
        });
        ctx.graph_stats.update(|s| {
            s.relationship_count += 1;
        });
        toasts.push("Claim submitted to the commons", ToastKind::Success);
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                content: String,
                claim_type: ClaimType,
                tags: Vec<String>,
                sources: Vec<String>,
            }
            let input = Input { content, claim_type, tags, sources };
            match hc.call_zome_default::<_, serde_json::Value>(
                "claims", "submit_claim", &input,
            ).await {
                Ok(_) => toasts.push("Claim submitted to the commons", ToastKind::Success),
                Err(e) => {
                    web_sys::console::log_1(&format!("submit claim failed: {e}").into());
                    toasts.push("Claim could not be submitted", ToastKind::Error);
                }
            }
        });
    }
}

/// Raise a challenge against an existing claim.
pub fn challenge_claim(claim_id: String, reason: String) {
    let hc = use_holochain();
    let ctx = use_knowledge_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        // Push a fact-check result as a visible challenge artefact
        ctx.fact_checks.update(|fcs| {
            fcs.push(FactCheckResultView {
                id: format!("CHK-{:04}", fcs.len() + 1),
                statement: format!("Challenge to {claim_id}: {reason}"),
                verdict: FactCheckVerdict::InsufficientEvidence,
                verdict_confidence: 0.0,
                credibility_score: 0.5,
                evidence_count: 0,
            });
        });
        toasts.push("Challenge raised", ToastKind::Custom("knowledge".into()));
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                claim_id: String,
                reason: String,
            }
            let input = Input { claim_id, reason };
            match hc.call_zome_default::<_, serde_json::Value>(
                "claims", "challenge_claim", &input,
            ).await {
                Ok(_) => toasts.push("Challenge raised", ToastKind::Custom("knowledge".into())),
                Err(e) => {
                    web_sys::console::log_1(&format!("challenge claim failed: {e}").into());
                    toasts.push("Challenge could not be raised", ToastKind::Error);
                }
            }
        });
    }
}

/// Run a fact check on a given statement.
pub fn run_fact_check(statement: String) {
    let hc = use_holochain();
    let ctx = use_knowledge_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        ctx.fact_checks.update(|fcs| {
            fcs.push(FactCheckResultView {
                id: format!("FC-{:04}", fcs.len() + 1),
                statement: statement.clone(),
                verdict: FactCheckVerdict::MostlyTrue,
                verdict_confidence: 0.72,
                credibility_score: 0.68,
                evidence_count: 3,
            });
        });
        toasts.push("Fact check completed", ToastKind::Success);
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input { statement: String }
            let input = Input { statement };
            match hc.call_zome_default::<_, serde_json::Value>(
                "factcheck", "fact_check", &input,
            ).await {
                Ok(_) => toasts.push("Fact check completed", ToastKind::Success),
                Err(e) => {
                    web_sys::console::log_1(&format!("fact check failed: {e}").into());
                    toasts.push("Fact check could not be completed", ToastKind::Error);
                }
            }
        });
    }
}
